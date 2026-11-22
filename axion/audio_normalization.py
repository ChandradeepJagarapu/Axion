"""Lightweight audio validation and RMS normalization for AXION.

The STT pipeline calls :func:`normalize_audio` on the exact 16 kHz mono
float32 array produced by ``whisper.load_audio``, immediately before
``model.transcribe``.  Processing is NumPy-only, CPU-only, and in-place
safe: the caller's array is never modified, and the same array object is
returned unchanged when normalization is disabled or unsafe.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from . import config as axion_config


@dataclass(frozen=True)
class AudioNormalizationConfig:
    """Tunable limits for pre-Whisper audio normalization."""

    enabled: bool = axion_config.AUDIO_NORMALIZATION_ENABLED
    target_rms_dbfs: float = axion_config.AUDIO_TARGET_RMS_DBFS
    max_gain_db: float = axion_config.AUDIO_MAX_GAIN_DB
    max_attenuation_db: float = axion_config.AUDIO_MAX_ATTENUATION_DB
    peak_limit: float = axion_config.AUDIO_PEAK_LIMIT
    silence_peak_threshold: float = axion_config.AUDIO_SILENCE_PEAK_THRESHOLD
    min_rms_for_boost: float = axion_config.AUDIO_MIN_RMS_FOR_BOOST
    target_tolerance_db: float = axion_config.AUDIO_TARGET_TOLERANCE_DB
    sample_rate: int = axion_config.AUDIO_SAMPLE_RATE
    # Legacy ``AXION_AUDIO_GAIN`` multiplier.  ``None`` selects automatic
    # RMS normalization; a positive value applies fixed gain with peak
    # protection instead.
    fixed_gain: float | None = None


@dataclass(frozen=True)
class AudioValidation:
    """Result of checking one audio buffer without modifying it."""

    usable: bool
    status: str
    samples: int = 0
    sample_rate: int | None = None
    peak: float = 0.0
    rms: float = 0.0
    peak_dbfs: float = -math.inf
    rms_dbfs: float = -math.inf
    non_finite_samples: int = 0
    clipped_fraction: float = 0.0


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized audio plus the metrics used to decide what to do."""

    audio: np.ndarray
    status: str
    usable: bool
    applied: bool
    gain: float
    gain_db: float
    input_peak: float
    input_rms: float
    output_peak: float
    output_rms: float
    validation: AudioValidation


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


def _env_number(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    if name not in env:
        return default
    value = _number(env.get(name), default)
    if not low <= value <= high:
        return default
    return value


def _env_bool(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def config_from_env(environ: Mapping[str, str] | None = None) -> AudioNormalizationConfig:
    """Build a configuration from environment variables with safe defaults."""
    env = os.environ if environ is None else environ
    normalize_raw = env.get("AXION_AUDIO_NORMALIZE")
    enabled = _env_bool(normalize_raw, axion_config.AUDIO_NORMALIZATION_ENABLED)

    target_rms_dbfs = _env_number(
        env, "AXION_AUDIO_TARGET_DBFS",
        axion_config.AUDIO_TARGET_RMS_DBFS, -60.0, -1.0,
    )
    max_gain_db = _env_number(
        env, "AXION_AUDIO_MAX_GAIN_DB",
        axion_config.AUDIO_MAX_GAIN_DB, 0.0, 60.0,
    )
    max_attenuation_db = _env_number(
        env, "AXION_AUDIO_MAX_ATTENUATION_DB",
        axion_config.AUDIO_MAX_ATTENUATION_DB, -60.0, 0.0,
    )
    peak_limit = _env_number(
        env, "AXION_AUDIO_PEAK_LIMIT",
        axion_config.AUDIO_PEAK_LIMIT, 0.05, 1.0,
    )

    fixed_gain = None
    legacy_gain = env.get("AXION_AUDIO_GAIN")
    if legacy_gain not in (None, ""):
        value = _number(legacy_gain, None)
        if value is not None:
            if value <= 0.0:
                enabled = False
            else:
                # Preserve the legacy variable, but never allow it to push
                # the configured maximum gain (and never clip).
                fixed_gain = min(value, 10.0 ** (max_gain_db / 20.0))
                if normalize_raw is None:
                    enabled = True

    return AudioNormalizationConfig(
        enabled=enabled,
        target_rms_dbfs=target_rms_dbfs,
        max_gain_db=max_gain_db,
        max_attenuation_db=max_attenuation_db,
        peak_limit=peak_limit,
        fixed_gain=fixed_gain,
    )


def _mono_array(audio) -> np.ndarray | None:
    """Return a 1-D numeric view/array, or ``None`` for unsupported input."""
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
    # The STT pipeline feeds Whisper-decoded float32 samples.  Rejecting
    # integer PCM here avoids silently treating 16-bit counts as amplitudes.
    if array.dtype.kind != "f":
        return None
    return array


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    work = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(work))))


def validate_audio(
    audio,
    sample_rate: int | float | None = None,
    normalization_config: AudioNormalizationConfig | None = None,
) -> AudioValidation:
    """Inspect audio for empty, silent, loud, malformed, or non-finite data.

    This function never raises for ordinary invalid input and never
    modifies the buffer.
    """
    cfg = normalization_config or AudioNormalizationConfig()
    expected_rate = int(cfg.sample_rate)
    rate = expected_rate if sample_rate is None else sample_rate

    if isinstance(rate, bool) or not isinstance(rate, (int, float, np.integer, np.floating)):
        return AudioValidation(False, "invalid_sample_rate")
    if not math.isfinite(float(rate)) or float(rate) != float(expected_rate):
        return AudioValidation(False, "invalid_sample_rate", sample_rate=None)

    if audio is None:
        return AudioValidation(False, "empty", sample_rate=expected_rate)

    try:
        array = np.asarray(audio)
    except (TypeError, ValueError):
        return AudioValidation(False, "invalid_dtype", sample_rate=expected_rate)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        return AudioValidation(False, "invalid_shape", sample_rate=expected_rate)
    if array.dtype.kind != "f":
        return AudioValidation(False, "invalid_dtype", sample_rate=expected_rate)
    if array.size == 0:
        return AudioValidation(False, "empty", sample_rate=expected_rate)

    non_finite = 0
    if array.dtype.kind == "f":
        non_finite = int(array.size - np.count_nonzero(np.isfinite(array)))
        if non_finite:
            return AudioValidation(
                False,
                "non_finite",
                samples=int(array.size),
                sample_rate=expected_rate,
                non_finite_samples=non_finite,
            )

    work = array.astype(np.float64, copy=False)
    absolute = np.abs(work)
    peak = float(np.max(absolute))
    rms = _rms(work)
    clipped_fraction = float(np.mean(absolute >= 0.999))

    if peak <= cfg.silence_peak_threshold:
        status = "silent"
    elif rms < cfg.min_rms_for_boost:
        status = "very_low_volume"
    elif peak >= 0.99 or clipped_fraction >= 0.01:
        status = "very_loud"
    else:
        status = "ok"

    return AudioValidation(
        True,
        status,
        samples=int(array.size),
        sample_rate=expected_rate,
        peak=peak,
        rms=rms,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        non_finite_samples=0,
        clipped_fraction=clipped_fraction,
    )


def _unchanged(
    array: np.ndarray,
    status: str,
    validation: AudioValidation,
    usable: bool = True,
) -> NormalizationResult:
    return NormalizationResult(
        audio=array,
        status=status,
        usable=usable,
        applied=False,
        gain=1.0,
        gain_db=0.0,
        input_peak=validation.peak,
        input_rms=validation.rms,
        output_peak=validation.peak,
        output_rms=validation.rms,
        validation=validation,
    )


def normalize_audio(
    audio,
    sample_rate: int | float | None = None,
    normalization_config: AudioNormalizationConfig | None = None,
) -> NormalizationResult:
    """Normalize a 16 kHz mono buffer toward the configured RMS level.

    The gain is limited by ``max_gain_db`` and ``max_attenuation_db`` and is
    additionally capped so the output peak never exceeds ``peak_limit``.
    Silent, extremely quiet, empty, malformed, or non-finite recordings are
    returned unchanged so the caller can decide whether to transcribe.
    """
    cfg = normalization_config or config_from_env()
    validation = validate_audio(audio, sample_rate=sample_rate, normalization_config=cfg)
    array = _mono_array(audio)

    if array is None:
        return NormalizationResult(
            audio=np.array([], dtype=np.float32),
            status=validation.status,
            usable=False,
            applied=False,
            gain=1.0,
            gain_db=0.0,
            input_peak=0.0,
            input_rms=0.0,
            output_peak=0.0,
            output_rms=0.0,
            validation=validation,
        )
    if not validation.usable:
        return _unchanged(array, validation.status, validation, usable=False)
    if not cfg.enabled:
        return _unchanged(array, "disabled", validation)
    if validation.status in {"silent", "very_low_volume"}:
        return _unchanged(array, validation.status, validation)

    target_rms = 10.0 ** (cfg.target_rms_dbfs / 20.0)
    if cfg.fixed_gain is not None:
        gain = float(cfg.fixed_gain)
        status = "fixed_gain"
    else:
        status = "normalized"
        if validation.rms <= 0.0:
            return _unchanged(array, "silent", validation)
        if abs(validation.rms_dbfs - cfg.target_rms_dbfs) <= cfg.target_tolerance_db:
            return _unchanged(array, "already_normalized", validation)
        gain = target_rms / validation.rms
        gain = min(gain, 10.0 ** (cfg.max_gain_db / 20.0))
        gain = max(gain, 10.0 ** (cfg.max_attenuation_db / 20.0))

    # Peak protection: reduce gain rather than hard-clipping the waveform.
    if validation.peak > 0.0:
        gain = min(gain, cfg.peak_limit / validation.peak)

    if not math.isfinite(gain) or gain <= 0.0:
        return _unchanged(array, "unsafe_gain", validation)
    if abs(gain - 1.0) < 1e-4:
        return _unchanged(array, "already_normalized", validation)

    output = np.multiply(array, np.float32(gain)).astype(np.float32, copy=False)
    output_peak = float(np.max(np.abs(output)))
    if output_peak > cfg.peak_limit:
        # Floating-point safety only; rescaling preserves waveform shape.
        correction = np.float32(cfg.peak_limit / output_peak)
        output = output * correction
        gain *= float(correction)
        output_peak = float(np.max(np.abs(output)))

    output_rms = _rms(output)
    return NormalizationResult(
        audio=output,
        status=status,
        usable=True,
        applied=True,
        gain=gain,
        gain_db=_dbfs(gain),
        input_peak=validation.peak,
        input_rms=validation.rms,
        output_peak=output_peak,
        output_rms=output_rms,
        validation=validation,
    )
