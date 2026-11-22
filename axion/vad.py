"""Lightweight, NumPy-only voice activity detection for AXION.

The STT pipeline calls :func:`extract_speech` on the 16 kHz mono float32
buffer produced by ``whisper.load_audio``, after Feature 02 noise reduction
and before Feature 01 normalization.  Detection is frame based (25 ms frames
with a 10 ms hop), using frame energy, an adaptive noise floor, and a
hysteresis threshold.  Detected speech is returned as one contiguous span
from the first to the last speech frame, with configurable pre-roll and
post-roll, so internal pauses and word boundaries are preserved.

The feature is opt-in (``AUDIO_VAD_ENABLED`` is false, so
``AXION_AUDIO_VAD=1`` must be set explicitly).  A second, lower boundary
threshold examines activity just outside the confident span and expands it,
so quiet leading/trailing words are not removed.  Detection never discards
audio on uncertainty: disabled, silent, very short, noise-only, no-speech,
speech-dominant, short-only, and low-contrast recordings are returned
unchanged.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from . import config as axion_config
from .audio_normalization import validate_audio


_BOOL_TRUE = {"1", "true", "yes", "on", "y", "enabled"}
_BOOL_FALSE = {"0", "false", "no", "off", "n", "disabled", ""}


@dataclass(frozen=True)
class VadConfig:
    """Tunable limits for frame-based voice activity detection."""

    enabled: bool = axion_config.AUDIO_VAD_ENABLED
    sample_rate: int = axion_config.AUDIO_VAD_SAMPLE_RATE
    frame_ms: float = axion_config.AUDIO_VAD_FRAME_MS
    hop_ms: float = axion_config.AUDIO_VAD_HOP_MS
    noise_percentile: float = axion_config.AUDIO_VAD_NOISE_PERCENTILE
    speech_margin_db: float = axion_config.AUDIO_VAD_SPEECH_MARGIN_DB
    hysteresis_db: float = axion_config.AUDIO_VAD_HYSTERESIS_DB
    min_speech_dbfs: float = axion_config.AUDIO_VAD_MIN_SPEECH_DBFS
    min_speech_ms: float = axion_config.AUDIO_VAD_MIN_SPEECH_MS
    min_silence_ms: float = axion_config.AUDIO_VAD_MIN_SILENCE_MS
    pre_roll_ms: float = axion_config.AUDIO_VAD_PRE_ROLL_MS
    post_roll_ms: float = axion_config.AUDIO_VAD_POST_ROLL_MS
    min_total_speech_ms: float = axion_config.AUDIO_VAD_MIN_TOTAL_SPEECH_MS
    min_trim_ms: float = axion_config.AUDIO_VAD_MIN_TRIM_MS
    boundary_percentile: float = axion_config.AUDIO_VAD_BOUNDARY_PERCENTILE
    boundary_margin_db: float = axion_config.AUDIO_VAD_BOUNDARY_MARGIN_DB
    boundary_min_dbfs: float = axion_config.AUDIO_VAD_BOUNDARY_MIN_DBFS
    boundary_min_activity_ms: float = (
        axion_config.AUDIO_VAD_BOUNDARY_MIN_ACTIVITY_MS
    )
    boundary_link_ms: float = axion_config.AUDIO_VAD_BOUNDARY_LINK_MS
    min_confident_speech_ms: float = (
        axion_config.AUDIO_VAD_MIN_CONFIDENT_SPEECH_MS
    )
    min_contrast_db: float = axion_config.AUDIO_VAD_MIN_CONTRAST_DB
    max_speech_ratio: float = axion_config.AUDIO_VAD_MAX_SPEECH_RATIO


@dataclass(frozen=True)
class SpeechSegment:
    """One detected speech interval in the original (untrimmed) audio."""

    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class VadResult:
    """Detected speech, extraction span, and the metrics behind the decision."""

    audio: np.ndarray
    status: str
    usable: bool
    applied: bool
    boundary_expanded: bool = False
    segments: tuple[SpeechSegment, ...] = ()
    noise_floor_dbfs: float = -math.inf
    threshold_dbfs: float = -math.inf
    speech_ratio: float = 0.0
    frames: int = 0
    speech_frames: int = 0
    original_samples: int = 0
    output_samples: int = 0
    start_sample: int = 0
    end_sample: int = 0
    sample_rate: int = axion_config.AUDIO_VAD_SAMPLE_RATE
    original_duration_seconds: float = 0.0
    output_duration_seconds: float = 0.0


@dataclass(frozen=True)
class _Analysis:
    """Internal result of frame analysis before span extraction."""

    status: str
    segments: tuple[SpeechSegment, ...]
    span_start: int
    span_end: int
    noise_floor_dbfs: float
    threshold_dbfs: float
    speech_ratio: float
    frames: int
    speech_frames: int
    boundary_expanded: bool = False


def _number(value, default: float) -> float:
    """Return a finite float, or ``default`` for missing/invalid input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _warn(message: str) -> None:
    """Report a configuration problem without interrupting transcription."""
    print(f"AXION config warning: {message}", file=sys.stderr, flush=True)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Parse a boolean environment variable, reporting unrecognised values."""
    if name not in env:
        return default
    raw = env.get(name)
    text = str(raw).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    _warn(
        f"{name}={raw!r} is not a valid boolean; expected one of "
        "1/0, true/false, yes/no, on/off, y/n, enabled/disabled. "
        f"Ignoring it and using the default ({'1' if default else '0'})."
    )
    return default


def _env_number(
    env: Mapping[str, str],
    name: str,
    default: float,
    low: float,
    high: float,
) -> float:
    """Parse a bounded numeric environment variable, reporting bad values."""
    if name not in env:
        return default
    raw = env.get(name)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value):
        _warn(
            f"{name}={raw!r} is not a valid number; "
            f"ignoring it and using the default ({default})."
        )
        return default
    if not low <= value <= high:
        _warn(
            f"{name}={raw!r} is outside [{low}, {high}]; "
            f"ignoring it and using the default ({default})."
        )
        return default
    return value


def config_from_env(
    environ: Mapping[str, str] | None = None,
) -> VadConfig:
    """Build a configuration from environment variables with safe defaults."""
    env = os.environ if environ is None else environ

    return VadConfig(
        enabled=_env_bool(
            env, "AXION_AUDIO_VAD", axion_config.AUDIO_VAD_ENABLED
        ),
        sample_rate=axion_config.AUDIO_VAD_SAMPLE_RATE,
        frame_ms=axion_config.AUDIO_VAD_FRAME_MS,
        hop_ms=axion_config.AUDIO_VAD_HOP_MS,
        noise_percentile=_env_number(
            env,
            "AXION_AUDIO_VAD_NOISE_PERCENTILE",
            axion_config.AUDIO_VAD_NOISE_PERCENTILE,
            5.0,
            50.0,
        ),
        speech_margin_db=_env_number(
            env,
            "AXION_AUDIO_VAD_MARGIN_DB",
            axion_config.AUDIO_VAD_SPEECH_MARGIN_DB,
            0.0,
            30.0,
        ),
        hysteresis_db=axion_config.AUDIO_VAD_HYSTERESIS_DB,
        min_speech_dbfs=_env_number(
            env,
            "AXION_AUDIO_VAD_MIN_SPEECH_DBFS",
            axion_config.AUDIO_VAD_MIN_SPEECH_DBFS,
            -100.0,
            -20.0,
        ),
        min_speech_ms=_env_number(
            env,
            "AXION_AUDIO_VAD_MIN_SPEECH_MS",
            axion_config.AUDIO_VAD_MIN_SPEECH_MS,
            0.0,
            2000.0,
        ),
        min_silence_ms=_env_number(
            env,
            "AXION_AUDIO_VAD_MIN_SILENCE_MS",
            axion_config.AUDIO_VAD_MIN_SILENCE_MS,
            0.0,
            5000.0,
        ),
        pre_roll_ms=_env_number(
            env,
            "AXION_AUDIO_VAD_PRE_ROLL_MS",
            axion_config.AUDIO_VAD_PRE_ROLL_MS,
            0.0,
            1000.0,
        ),
        post_roll_ms=_env_number(
            env,
            "AXION_AUDIO_VAD_POST_ROLL_MS",
            axion_config.AUDIO_VAD_POST_ROLL_MS,
            0.0,
            1000.0,
        ),
        min_total_speech_ms=axion_config.AUDIO_VAD_MIN_TOTAL_SPEECH_MS,
        min_trim_ms=axion_config.AUDIO_VAD_MIN_TRIM_MS,
    )


def _mono_array(audio) -> np.ndarray | None:
    """Return a 1-D float view/array, or ``None`` for unsupported input."""
    if audio is None:
        return None
    try:
        array = np.asarray(audio)
    except (TypeError, ValueError):
        return None
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        return None
    if array.dtype.kind != "f":
        return None
    return array


def _unchanged(
    array: np.ndarray,
    status: str,
    usable: bool = True,
    rate: int = axion_config.AUDIO_VAD_SAMPLE_RATE,
) -> VadResult:
    """Return the input untouched with duration metadata."""
    count = int(array.size)
    duration = count / rate if rate > 0 else 0.0
    return VadResult(
        audio=array,
        status=status,
        usable=usable,
        applied=False,
        original_samples=count,
        output_samples=count,
        end_sample=count,
        sample_rate=rate,
        original_duration_seconds=duration,
        output_duration_seconds=duration,
    )


def _median_filter(values: np.ndarray, width: int = 3) -> np.ndarray:
    """Centered median filter with edge padding."""
    if values.size == 0 or width <= 1:
        return values
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _frame_levels(
    samples: np.ndarray,
    frame: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame start/end indices and frame energy in dBFS.

    Only real samples are averaged, so no zero padding lowers the energy of
    the final frame.  Timestamps are derived directly from these indices.
    """
    count = int(samples.size)
    starts = np.arange(0, count, hop, dtype=np.int64)
    ends = np.minimum(starts + frame, count)
    lengths = np.maximum(ends - starts, 1).astype(np.float64)

    squares = np.square(samples.astype(np.float64, copy=False))
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squares, dtype=np.float64))
    )
    mean_square = (cumulative[ends] - cumulative[starts]) / lengths
    levels = 10.0 * np.log10(np.maximum(mean_square, 1e-20))
    return starts, ends, levels


def _speech_runs(levels: np.ndarray, threshold: float, hysteresis: float):
    """Two-state onset/release classification with hysteresis."""
    is_speech = np.zeros(levels.size, dtype=bool)
    active = False
    release = threshold - hysteresis
    for index, level in enumerate(levels):
        if active:
            if level < release:
                active = False
        elif level >= threshold:
            active = True
        is_speech[index] = active

    padded = np.concatenate(([False], is_speech, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return is_speech, [
        (int(edges[index]), int(edges[index + 1]))
        for index in range(0, edges.size, 2)
    ]


def _passthrough_view(audio):
    """Return the original input as an ndarray, or an empty buffer."""
    if audio is None:
        return np.array([], dtype=np.float32)
    try:
        return np.asarray(audio)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float32)


def _sample_runs(
    levels: np.ndarray,
    threshold: float,
    frame: int,
    hop: int,
    count: int,
):
    """Return activity runs in sample coordinates for a given threshold."""
    _, runs = _speech_runs(levels, threshold, 0.0)
    result = []
    for first, last in runs:
        start = int(first * hop)
        end = int(min(count, (last - 1) * hop + frame))
        if end > start:
            result.append((start, end))
    return result


def _merge_sample_runs(runs, gap_samples: int):
    """Merge activity runs separated by less than ``gap_samples``."""
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] < gap_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _expand_span_for_activity(
    span_start: int,
    span_end: int,
    runs,
    link_samples: int,
    pre_samples: int,
    post_samples: int,
    count: int,
):
    """Extend a span over credible activity linked to its boundaries.

    Only runs that overlap or lie within ``link_samples`` of the current
    boundary are followed, so a quiet word next to the utterance is included
    while distant noise is not chained into the span.
    """
    leading = span_start
    for _ in range(len(runs) + 1):
        candidates = [
            (start, end)
            for start, end in runs
            if start < leading and end >= leading - link_samples
        ]
        if not candidates:
            break
        new_start = min(start for start, _ in candidates)
        if new_start >= leading:
            break
        leading = new_start

    trailing = span_end
    for _ in range(len(runs) + 1):
        candidates = [
            (start, end)
            for start, end in runs
            if end > trailing and start <= trailing + link_samples
        ]
        if not candidates:
            break
        new_end = max(end for _, end in candidates)
        if new_end <= trailing:
            break
        trailing = new_end

    expanded_start = (
        max(0, leading - pre_samples) if leading < span_start else span_start
    )
    expanded_end = (
        min(count, trailing + post_samples) if trailing > span_end else span_end
    )
    return (
        expanded_start,
        expanded_end,
        expanded_start != span_start or expanded_end != span_end,
    )


def _analyze(samples: np.ndarray, rate: int, config: VadConfig) -> _Analysis:
    """Classify frames and build padded extraction spans."""
    count = int(samples.size)

    frame_ms = _clamp(_number(config.frame_ms, 25.0), 10.0, 50.0)
    hop_ms = _clamp(_number(config.hop_ms, 10.0), 5.0, frame_ms)
    frame = max(1, int(round(rate * frame_ms / 1000.0)))
    hop = max(1, int(round(rate * hop_ms / 1000.0)))
    hop = min(hop, frame)

    min_speech_samples = max(
        0, int(round(_clamp(_number(config.min_speech_ms, 150.0), 0.0, 2000.0) * rate / 1000.0))
    )
    min_silence_samples = max(
        0, int(round(_clamp(_number(config.min_silence_ms, 300.0), 0.0, 5000.0) * rate / 1000.0))
    )
    min_total_samples = max(
        0,
        int(round(
            _clamp(_number(config.min_total_speech_ms, 200.0), 0.0, 5000.0)
            * rate / 1000.0
        )),
    )
    pre_samples = max(
        0, int(round(_clamp(_number(config.pre_roll_ms, 200.0), 0.0, 1000.0) * rate / 1000.0))
    )
    post_samples = max(
        0, int(round(_clamp(_number(config.post_roll_ms, 250.0), 0.0, 1000.0) * rate / 1000.0))
    )
    boundary_min_samples = max(
        0,
        int(round(
            _clamp(_number(config.boundary_min_activity_ms, 80.0), 0.0, 1000.0)
            * rate / 1000.0
        )),
    )
    boundary_link_samples = max(
        0,
        int(round(
            _clamp(_number(config.boundary_link_ms, 500.0), 0.0, 5000.0)
            * rate / 1000.0
        )),
    )
    min_confident_samples = max(
        0,
        int(round(
            _clamp(_number(config.min_confident_speech_ms, 300.0), 0.0, 5000.0)
            * rate / 1000.0
        )),
    )

    empty = _Analysis(
        status="too_short",
        segments=(),
        span_start=0,
        span_end=count,
        noise_floor_dbfs=-math.inf,
        threshold_dbfs=-math.inf,
        speech_ratio=0.0,
        frames=0,
        speech_frames=0,
    )
    if count == 0 or count < max(frame, min_total_samples):
        return empty

    percentile = _clamp(_number(config.noise_percentile, 20.0), 5.0, 50.0)
    margin = _clamp(_number(config.speech_margin_db, 8.0), 0.0, 30.0)
    hysteresis = _clamp(_number(config.hysteresis_db, 3.0), 0.0, 20.0)
    minimum = _clamp(_number(config.min_speech_dbfs, -60.0), -100.0, -20.0)
    boundary_percentile = _clamp(
        _number(config.boundary_percentile, 5.0), 1.0, 20.0
    )
    boundary_margin = _clamp(
        _number(config.boundary_margin_db, 3.0), 0.0, 20.0
    )
    boundary_minimum = _clamp(
        _number(config.boundary_min_dbfs, -70.0), -100.0, -20.0
    )
    min_contrast = _clamp(
        _number(config.min_contrast_db, 6.0), 0.0, 40.0
    )

    starts, ends, levels = _frame_levels(samples, frame, hop)
    smoothed = _median_filter(levels, 3)
    noise_floor = float(np.percentile(smoothed, percentile))
    threshold = max(noise_floor + margin, minimum)
    boundary_floor = float(np.percentile(smoothed, boundary_percentile))
    boundary_threshold = max(
        boundary_floor + boundary_margin, boundary_minimum
    )
    # The boundary detector must be strictly more sensitive than the main
    # detector, even if the caller configures unusual margins.
    boundary_threshold = min(boundary_threshold, threshold - 0.5)
    high_level = float(np.percentile(smoothed, 95.0))
    contrast_db = high_level - boundary_floor

    is_speech, runs = _speech_runs(smoothed, threshold, hysteresis)
    speech_frames = int(np.count_nonzero(is_speech))

    raw_segments = []
    for first, last in runs:
        start = int(first * hop)
        end = int(min(count, (last - 1) * hop + frame))
        if end > start:
            raw_segments.append((start, end))

    # Bridge short gaps first so an intra-phrase pause is never split, then
    # drop isolated clicks shorter than the minimum speech duration.
    merged = []
    for start, end in raw_segments:
        if merged and start - merged[-1][1] < min_silence_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    kept = [
        (start, end)
        for start, end in merged
        if end - start >= min_speech_samples
    ]
    speech_samples = sum(end - start for start, end in kept)
    dominance = speech_samples / count if count else 0.0
    max_ratio = _clamp(_number(config.max_speech_ratio, 0.70), 0.0, 1.0)
    longest_segment = max((end - start for start, end in kept), default=0)
    # If the main detector flags many frames as speech but the fraction is
    # very low (e.g. a sparse tonal burst or narrowband noise), treat this
    # as insufficient confidence rather than real speech.
    speech_fraction = speech_frames / max(int(smoothed.size), 1)

    if speech_frames == 0:
        status = "no_speech"
    elif not kept or speech_samples < min_total_samples:
        status = "too_short"
    elif longest_segment < min_confident_samples:
        # Isolated short bursts are more likely to be transients than words;
        # passing through preserves any genuinely short command.
        status = "low_confidence"
    elif dominance > max_ratio:
        # When speech fills most of the buffer the adaptive floor may have
        # been estimated from speech itself, so quiet word edges could be
        # mistaken for silence.  Keep the original audio rather than risk a
        # partial trim.
        status = "speech_dominant"
    elif contrast_db < min_contrast:
        # Little separation between quiet and active frames: the adaptive
        # floor cannot be trusted to distinguish speech from noise.
        status = "low_contrast"
    elif speech_fraction < 0.10:
        # Very few frames above the main threshold strongly suggest the
        # detector is picking up sparse tonal/narrowband activity rather than
        # real speech.  Passthrough is the safe default.
        status = "low_confidence"
    else:
        status = "detected"

    segments = tuple(
        SpeechSegment(
            start_sample=start,
            end_sample=end,
            start_seconds=start / rate,
            end_seconds=end / rate,
            duration_seconds=(end - start) / rate,
        )
        for start, end in kept
    )

    if segments:
        span_start = max(0, segments[0].start_sample - pre_samples)
        span_end = min(count, segments[-1].end_sample + post_samples)
    else:
        span_start, span_end = 0, count

    boundary_expanded = False
    if status == "detected" and segments:
        boundary_runs = _sample_runs(
            smoothed, boundary_threshold, frame, hop, count
        )
        boundary_runs = _merge_sample_runs(
            boundary_runs, min_silence_samples
        )
        boundary_runs = [
            (start, end)
            for start, end in boundary_runs
            if end - start >= boundary_min_samples
        ]
        span_start, span_end, boundary_expanded = _expand_span_for_activity(
            span_start,
            span_end,
            boundary_runs,
            boundary_link_samples,
            pre_samples,
            post_samples,
            count,
        )

    return _Analysis(
        status=status,
        segments=segments,
        span_start=span_start,
        span_end=span_end,
        noise_floor_dbfs=noise_floor,
        threshold_dbfs=threshold,
        speech_ratio=speech_frames / max(int(smoothed.size), 1),
        frames=int(smoothed.size),
        speech_frames=speech_frames,
        boundary_expanded=boundary_expanded,
    )


def detect_speech(
    audio,
    sample_rate: int | float | None = None,
    vad_config: VadConfig | None = None,
) -> VadResult:
    """Detect speech intervals without trimming the audio.

    ``result.audio`` is always the original buffer.  ``result.segments`` holds
    the detected intervals in original sample and second coordinates;
    ``result.start_sample`` / ``result.end_sample`` describe the contiguous
    span that :func:`extract_speech` would keep (speech plus roll).
    """
    config = vad_config or config_from_env()
    validation = validate_audio(audio, sample_rate=sample_rate)
    array = _mono_array(audio)

    # The validated/expected rate always wins; a manually constructed
    # VadConfig cannot desynchronise frame sizes from the audio.
    rate = int(
        _number(
            validation.sample_rate,
            _number(
                config.sample_rate, axion_config.AUDIO_VAD_SAMPLE_RATE
            ),
        )
    )
    if rate <= 0:
        rate = axion_config.AUDIO_VAD_SAMPLE_RATE

    if array is None:
        return _unchanged(
            _passthrough_view(audio),
            validation.status,
            usable=False,
            rate=rate,
        )
    if not validation.usable:
        return _unchanged(
            array, validation.status, usable=False, rate=rate
        )
    if not config.enabled:
        return _unchanged(array, "disabled", rate=rate)

    if validation.status in {"silent", "very_low_volume"}:
        return _unchanged(array, validation.status, rate=rate)

    samples = np.asarray(array, dtype=np.float32)
    analysis = _analyze(samples, rate, config)

    if analysis.status != "detected":
        return replace(
            _unchanged(array, analysis.status, rate=rate),
            segments=analysis.segments,
            noise_floor_dbfs=analysis.noise_floor_dbfs,
            threshold_dbfs=analysis.threshold_dbfs,
            speech_ratio=analysis.speech_ratio,
            frames=analysis.frames,
            speech_frames=analysis.speech_frames,
            start_sample=analysis.span_start,
            end_sample=analysis.span_end,
            boundary_expanded=analysis.boundary_expanded,
        )

    duration = samples.size / rate
    return VadResult(
        audio=array,
        status="detected",
        usable=True,
        applied=False,
        boundary_expanded=analysis.boundary_expanded,
        segments=analysis.segments,
        noise_floor_dbfs=analysis.noise_floor_dbfs,
        threshold_dbfs=analysis.threshold_dbfs,
        speech_ratio=analysis.speech_ratio,
        frames=analysis.frames,
        speech_frames=analysis.speech_frames,
        original_samples=int(samples.size),
        output_samples=int(samples.size),
        start_sample=analysis.span_start,
        end_sample=analysis.span_end,
        sample_rate=rate,
        original_duration_seconds=duration,
        output_duration_seconds=duration,
    )


def extract_speech(
    audio,
    sample_rate: int | float | None = None,
    vad_config: VadConfig | None = None,
) -> VadResult:
    """Detect speech and keep one contiguous span from first to last speech.

    Internal pauses are preserved because only the leading and trailing
    non-speech regions are removed.  When the span removes less than
    ``min_trim_ms``, when no speech is found, or when detection is uncertain,
    the original audio is returned unchanged.
    """
    config = vad_config or config_from_env()
    result = detect_speech(
        audio,
        sample_rate=sample_rate,
        vad_config=config,
    )
    if result.status != "detected":
        return result

    rate = result.sample_rate
    min_trim_samples = max(
        0,
        int(round(
            _clamp(_number(config.min_trim_ms, 300.0), 0.0, 5000.0)
            * rate / 1000.0
        )),
    )

    span_start = int(result.start_sample)
    span_end = int(result.end_sample)
    removed = result.original_samples - (span_end - span_start)
    if span_end <= span_start or removed < min_trim_samples:
        return replace(
            result,
            status="already_tight",
            start_sample=0,
            end_sample=result.original_samples,
        )

    trimmed = np.array(
        result.audio[span_start:span_end],
        dtype=np.float32,
        copy=True,
    )
    return replace(
        result,
        audio=trimmed,
        status="trimmed",
        applied=True,
        output_samples=int(trimmed.size),
        start_sample=span_start,
        end_sample=span_end,
        output_duration_seconds=trimmed.size / rate,
    )
