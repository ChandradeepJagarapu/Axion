"""Lightweight, NumPy-only noise reduction for AXION.

The STT pipeline calls :func:`reduce_noise` on the exact 16 kHz mono float32
array produced by ``whisper.load_audio``, immediately before Feature 01
normalization.  Processing is CPU-only, deterministic, and never modifies the
caller's array.

The algorithm is STFT spectral gating: a stationary noise magnitude spectrum
is estimated from the quietest *noise-like* (spectrally flat) frames, a soft
attenuation mask is applied, and the signal is reconstructed by overlap-add.
It targets steady background noise (fans, air conditioning, hum, road noise)
while leaving high-SNR speech bins essentially untouched.

The feature is opt-in (``AUDIO_NOISE_REDUCTION_ENABLED`` is false, so
``AXION_AUDIO_DENOISE=1`` must be set explicitly).  When processing is
disabled, the audio is silent or malformed, no noise-like frames can be
found, the noise estimate is too close to the speech level, or the audio is
otherwise too clean, the original audio is returned unchanged.  A local
energy limiter additionally guarantees that reconstruction only attenuates:
no output sample is ever larger than the corresponding input sample.
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
class NoiseReductionConfig:
    """Tunable limits for pre-Whisper noise reduction."""

    enabled: bool = axion_config.AUDIO_NOISE_REDUCTION_ENABLED
    sample_rate: int = axion_config.AUDIO_DENOISE_SAMPLE_RATE
    frame_samples: int = axion_config.AUDIO_DENOISE_FRAME_SAMPLES
    hop_samples: int = axion_config.AUDIO_DENOISE_HOP_SAMPLES
    noise_percentile: float = axion_config.AUDIO_DENOISE_NOISE_PERCENTILE
    over_subtraction: float = axion_config.AUDIO_DENOISE_OVER_SUBTRACTION
    spectral_floor_db: float = axion_config.AUDIO_DENOISE_SPECTRAL_FLOOR_DB
    freq_smoothing_bins: int = axion_config.AUDIO_DENOISE_FREQ_SMOOTHING_BINS
    time_smoothing_frames: int = axion_config.AUDIO_DENOISE_TIME_SMOOTHING_FRAMES
    min_noise_dbfs: float = axion_config.AUDIO_DENOISE_MIN_NOISE_DBFS
    min_contrast_db: float = axion_config.AUDIO_DENOISE_MIN_CONTRAST_DB
    min_noise_flatness: float = axion_config.AUDIO_DENOISE_MIN_NOISE_FLATNESS
    min_snr_db: float = axion_config.AUDIO_DENOISE_MIN_SNR_DB
    highpass_hz: float = axion_config.AUDIO_DENOISE_HIGHPASS_HZ


@dataclass(frozen=True)
class NoiseReductionResult:
    """Denoised audio plus the metrics used to decide what to do."""

    audio: np.ndarray
    status: str
    usable: bool
    applied: bool
    input_peak: float = 0.0
    input_rms: float = 0.0
    output_peak: float = 0.0
    output_rms: float = 0.0
    estimated_noise_dbfs: float = -math.inf
    contrast_db: float = -math.inf
    speech_to_noise_db: float = -math.inf
    frames: int = 0
    noise_frames: int = 0


def _dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS, safely handling silence."""
    if value <= 0.0:
        return -math.inf
    return 20.0 * math.log10(value)


def _number(value, default):
    """Return a finite float, or ``default`` for missing/invalid input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _warn(message: str) -> None:
    """Report a configuration problem without interrupting transcription."""
    print(f"AXION config warning: {message}", file=sys.stderr, flush=True)


def _env_number(
    env: Mapping[str, str],
    name: str,
    default: float,
    low: float,
    high: float,
) -> float:
    if name not in env:
        return default
    value = _number(env.get(name), default)
    if not low <= value <= high:
        return default
    return value


def _env_bool(
    env: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
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


def config_from_env(
    environ: Mapping[str, str] | None = None,
) -> NoiseReductionConfig:
    """Build a configuration from environment variables with safe defaults."""
    env = os.environ if environ is None else environ

    enabled = _env_bool(
        env,
        "AXION_AUDIO_DENOISE",
        axion_config.AUDIO_NOISE_REDUCTION_ENABLED,
    )
    spectral_floor_db = _env_number(
        env, "AXION_AUDIO_DENOISE_FLOOR_DB",
        axion_config.AUDIO_DENOISE_SPECTRAL_FLOOR_DB, -40.0, 0.0,
    )
    over_subtraction = _env_number(
        env, "AXION_AUDIO_DENOISE_OVER_SUBTRACTION",
        axion_config.AUDIO_DENOISE_OVER_SUBTRACTION, 0.5, 4.0,
    )
    noise_percentile = _env_number(
        env, "AXION_AUDIO_DENOISE_NOISE_PERCENTILE",
        axion_config.AUDIO_DENOISE_NOISE_PERCENTILE, 5.0, 40.0,
    )
    min_noise_dbfs = _env_number(
        env, "AXION_AUDIO_DENOISE_MIN_NOISE_DBFS",
        axion_config.AUDIO_DENOISE_MIN_NOISE_DBFS, -80.0, -30.0,
    )
    min_contrast_db = _env_number(
        env, "AXION_AUDIO_DENOISE_MIN_CONTRAST_DB",
        axion_config.AUDIO_DENOISE_MIN_CONTRAST_DB, 0.0, 20.0,
    )
    min_noise_flatness = _env_number(
        env, "AXION_AUDIO_DENOISE_MIN_FLATNESS",
        axion_config.AUDIO_DENOISE_MIN_NOISE_FLATNESS, 0.02, 0.9,
    )
    min_snr_db = _env_number(
        env, "AXION_AUDIO_DENOISE_MIN_SNR_DB",
        axion_config.AUDIO_DENOISE_MIN_SNR_DB, 0.0, 40.0,
    )
    highpass_hz = _env_number(
        env, "AXION_AUDIO_DENOISE_HIGHPASS_HZ",
        axion_config.AUDIO_DENOISE_HIGHPASS_HZ, 0.0, 300.0,
    )

    return NoiseReductionConfig(
        enabled=enabled,
        sample_rate=axion_config.AUDIO_DENOISE_SAMPLE_RATE,
        frame_samples=axion_config.AUDIO_DENOISE_FRAME_SAMPLES,
        hop_samples=axion_config.AUDIO_DENOISE_HOP_SAMPLES,
        noise_percentile=noise_percentile,
        over_subtraction=over_subtraction,
        spectral_floor_db=spectral_floor_db,
        freq_smoothing_bins=axion_config.AUDIO_DENOISE_FREQ_SMOOTHING_BINS,
        time_smoothing_frames=axion_config.AUDIO_DENOISE_TIME_SMOOTHING_FRAMES,
        min_noise_dbfs=min_noise_dbfs,
        min_contrast_db=min_contrast_db,
        min_noise_flatness=min_noise_flatness,
        min_snr_db=min_snr_db,
        highpass_hz=highpass_hz,
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


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    work = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(work))))


def _unchanged(
    array: np.ndarray,
    status: str,
    validation,
    usable: bool = True,
) -> NoiseReductionResult:
    return NoiseReductionResult(
        audio=array,
        status=status,
        usable=usable,
        applied=False,
        input_peak=validation.peak,
        input_rms=validation.rms,
        output_peak=validation.peak,
        output_rms=validation.rms,
    )


def _smooth(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Centered moving average along one axis, using edge padding."""
    width = int(width)
    if width <= 1:
        return values
    if width % 2 == 0:
        width += 1
    pad = width // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    padded = np.pad(values, pad_width, mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis,
        padded,
    )


def _padded_frames(samples: np.ndarray, frame: int, hop: int):
    """Pad a signal by half a frame and return (padded, frames, pad)."""
    pad = frame // 2
    mode = "reflect" if samples.size > pad else "constant"
    padded = np.pad(samples, (pad, pad), mode=mode)
    if padded.size < frame:
        padded = np.pad(padded, (0, frame - padded.size))
    frames = np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    )
    return padded, frames, pad


def _frame_rms(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))


def _spectral_flatness(power: np.ndarray) -> np.ndarray:
    """Geometric mean divided by arithmetic mean, ignoring the DC bin.

    Steady broadband noise is flat (values around 0.3-0.6 for a Hann-windowed
    periodogram); voiced speech is peaky (values far below 0.1).  The DC bin
    is excluded because microphone offset would otherwise dominate it.
    """
    band = np.maximum(power[:, 1:], 1e-20)
    geometric = np.exp(np.mean(np.log(band), axis=1))
    arithmetic = np.mean(band, axis=1)
    return geometric / np.maximum(arithmetic, 1e-20)


def _limit_local_growth(
    output: np.ndarray,
    reference: np.ndarray,
    frame: int,
    hop: int,
) -> np.ndarray:
    """Prevent spectral gating from amplifying the signal.

    The frame-energy ceiling removes gross growth introduced by rapidly
    changing masks.  The final sample ceiling enforces the stronger
    invariant that gating only ever attenuates: no output sample may exceed
    the corresponding input sample.  That removes pre-echo and local
    amplification in pauses without touching frames where the mask is near
    unity, and it cannot make clean audio louder.
    """
    if output.size == 0:
        return output

    _, out_frames, pad = _padded_frames(output, frame, hop)
    _, ref_frames, _ = _padded_frames(reference, frame, hop)
    count = min(out_frames.shape[0], ref_frames.shape[0])
    if count == 0:
        return output

    out_rms = _frame_rms(out_frames[:count])
    ref_rms = _frame_rms(ref_frames[:count])
    limit = np.minimum(1.0, ref_rms / np.maximum(out_rms, 1e-12))

    centers = np.arange(count, dtype=np.float64) * hop - pad + frame / 2.0
    positions = np.arange(output.size, dtype=np.float64)
    envelope = np.interp(positions, centers, limit, left=limit[0], right=limit[-1])
    limited = output.astype(np.float64) * envelope

    reference_abs = np.abs(reference.astype(np.float64))
    limited = np.sign(limited) * np.minimum(np.abs(limited), reference_abs)
    return limited.astype(np.float32)


def reduce_noise(
    audio,
    sample_rate: int | float | None = None,
    noise_config: NoiseReductionConfig | None = None,
) -> NoiseReductionResult:
    """Reduce steady background noise in a 16 kHz mono buffer.

    The original array is returned unchanged whenever processing is
    disabled, unsafe, unnecessary, or when no sufficiently noise-like frames
    exist to estimate a trustworthy noise profile.
    """
    cfg = noise_config or config_from_env()

    validation = validate_audio(audio, sample_rate=sample_rate)
    array = _mono_array(audio)

    if array is None:
        return NoiseReductionResult(
            audio=np.array([], dtype=np.float32),
            status=validation.status,
            usable=False,
            applied=False,
        )
    if not validation.usable:
        return _unchanged(array, validation.status, validation, usable=False)
    if not cfg.enabled:
        return _unchanged(array, "disabled", validation)
    if validation.status in {"silent", "very_low_volume"}:
        return _unchanged(array, validation.status, validation)

    frame = int(_number(cfg.frame_samples, 512))
    frame = max(16, min(frame, 16384))
    hop = int(_number(cfg.hop_samples, 128))
    # Hann windows overlap-add correctly at <= 50 % hop; anything larger
    # leaves gaps or double-counts samples in the reconstruction envelope.
    hop = max(1, min(hop, frame // 2))
    rate = int(_number(cfg.sample_rate, axion_config.AUDIO_DENOISE_SAMPLE_RATE))
    if rate <= 0:
        rate = axion_config.AUDIO_DENOISE_SAMPLE_RATE

    percentile = _number(cfg.noise_percentile, 15.0)
    percentile = min(max(percentile, 1.0), 50.0)
    over_subtraction = _number(cfg.over_subtraction, 2.5)
    over_subtraction = min(max(over_subtraction, 0.1), 10.0)
    floor_db = _number(cfg.spectral_floor_db, -18.0)
    floor_db = min(max(floor_db, -60.0), 0.0)
    spectral_floor = 10.0 ** (floor_db / 20.0)
    min_noise_dbfs = _number(cfg.min_noise_dbfs, -60.0)
    min_noise_dbfs = min(max(min_noise_dbfs, -100.0), 0.0)
    min_contrast_db = _number(cfg.min_contrast_db, 6.0)
    min_contrast_db = min(max(min_contrast_db, 0.0), 40.0)
    min_flatness = _number(cfg.min_noise_flatness, 0.20)
    min_flatness = min(max(min_flatness, 0.0), 1.0)
    min_snr_db = _number(cfg.min_snr_db, 10.0)
    min_snr_db = min(max(min_snr_db, 0.0), 40.0)
    highpass_hz = _number(cfg.highpass_hz, 60.0)
    highpass_hz = min(max(highpass_hz, 0.0), rate / 2.0)
    freq_smoothing = int(_number(cfg.freq_smoothing_bins, 3))
    freq_smoothing = max(1, min(freq_smoothing, 31))
    time_smoothing = int(_number(cfg.time_smoothing_frames, 3))
    time_smoothing = max(1, min(time_smoothing, 31))

    samples = np.asarray(array, dtype=np.float32)
    total = samples.size
    padded, frames, pad = _padded_frames(samples, frame, hop)

    window = np.hanning(frame + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum)
    power = np.square(magnitude.astype(np.float64))

    frame_rms = _frame_rms(frames)
    active = frame_rms > 10.0 ** (min_noise_dbfs / 20.0)
    if not np.any(active):
        return _unchanged(array, "low_noise", validation)

    loud_rms = float(np.percentile(frame_rms, 90.0))
    quiet_rms = float(np.percentile(frame_rms, percentile))
    contrast_db = min(_dbfs(loud_rms) - _dbfs(quiet_rms), 99.0)
    if contrast_db < min_contrast_db:
        return replace(
            _unchanged(array, "no_noise_profile", validation),
            contrast_db=contrast_db,
            frames=int(frames.shape[0]),
        )

    # A candidate noise frame must be quiet and spectrally flat.  Speech,
    # even quiet or continuous speech, is strongly harmonic and fails the
    # flatness test, so it cannot become the noise profile.
    flatness = _spectral_flatness(power)
    threshold = float(np.percentile(frame_rms[active], percentile))
    noise_like = active & (frame_rms <= threshold) & (flatness >= min_flatness)
    noise_frame_count = int(np.count_nonzero(noise_like))
    required_noise_frames = max(3, int(frames.shape[0] * 0.02))
    if noise_frame_count < required_noise_frames:
        return replace(
            _unchanged(array, "no_noise_profile", validation),
            contrast_db=contrast_db,
            frames=int(frames.shape[0]),
            noise_frames=noise_frame_count,
        )

    noise_rms = _rms(frames[noise_like])
    estimated_noise_dbfs = _dbfs(noise_rms)
    if estimated_noise_dbfs < min_noise_dbfs:
        return replace(
            _unchanged(array, "low_noise", validation),
            estimated_noise_dbfs=estimated_noise_dbfs,
            contrast_db=contrast_db,
            frames=int(frames.shape[0]),
            noise_frames=noise_frame_count,
        )

    # If the estimated "noise" is nearly as loud as the loudest speech, it is
    # speech rather than a noise floor.  Gating in that regime suppresses the
    # signal instead of the noise, so leave the audio untouched.
    speech_to_noise_db = _dbfs(loud_rms) - estimated_noise_dbfs
    if speech_to_noise_db < min_snr_db:
        return replace(
            _unchanged(array, "low_snr", validation),
            estimated_noise_dbfs=estimated_noise_dbfs,
            contrast_db=contrast_db,
            speech_to_noise_db=speech_to_noise_db,
            frames=int(frames.shape[0]),
            noise_frames=noise_frame_count,
        )

    # Be gentler when the margin above the SNR threshold is small.
    alpha_scale = min(
        1.0,
        0.6 + 0.4 * max(0.0, speech_to_noise_db - min_snr_db) / 12.0,
    )
    effective_over_subtraction = over_subtraction * alpha_scale

    noise_magnitude = np.median(magnitude[noise_like], axis=0)
    noise_power = np.maximum(
        np.square(noise_magnitude.astype(np.float64)),
        1e-20,
    )

    gain = 1.0 - effective_over_subtraction * noise_power / (power + 1e-20)
    gain = np.clip(gain, spectral_floor, 1.0)
    gain = _smooth(gain, freq_smoothing, axis=1)
    gain = _smooth(gain, time_smoothing, axis=0)

    if highpass_hz > 0.0:
        frequencies = np.fft.rfftfreq(frame, d=1.0 / rate)
        ramp = np.clip(frequencies / highpass_hz, 0.0, 1.0)
        gain *= ramp[None, :]

    gain = np.clip(gain, 0.0, 1.0).astype(np.float32)

    cleaned_frames = (
        np.fft.irfft(spectrum * gain, n=frame, axis=1).astype(np.float32)
        * window
    )
    output = np.zeros(padded.size, dtype=np.float32)
    envelope = np.zeros(padded.size, dtype=np.float32)
    window_squared = window * window
    for index, cleaned in enumerate(cleaned_frames):
        start = index * hop
        output[start:start + frame] += cleaned
        envelope[start:start + frame] += window_squared

    valid = envelope > 1e-8
    output[valid] /= envelope[valid]
    output = output[pad:pad + total]
    if output.size < total:
        output = np.pad(output, (0, total - output.size))
    output = output[:total]

    output = _limit_local_growth(output, samples, frame, hop)
    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    return NoiseReductionResult(
        audio=output,
        status="denoised",
        usable=True,
        applied=True,
        input_peak=validation.peak,
        input_rms=validation.rms,
        output_peak=float(np.max(np.abs(output))) if output.size else 0.0,
        output_rms=_rms(output),
        estimated_noise_dbfs=estimated_noise_dbfs,
        contrast_db=contrast_db,
        speech_to_noise_db=speech_to_noise_db,
        frames=int(frames.shape[0]),
        noise_frames=noise_frame_count,
    )
