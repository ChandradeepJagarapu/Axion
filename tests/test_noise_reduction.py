"""Unit and integration tests for AXION pre-Whisper noise reduction."""

import contextlib
import dataclasses
import io
import os
import unittest
from unittest.mock import patch

import numpy as np

from axion.audio_normalization import normalize_audio
from axion.noise_reduction import (
    NoiseReductionConfig,
    config_from_env,
    reduce_noise,
)
from axion.stt import SpeechToText

SAMPLE_RATE = 16000

# Noise reduction is opt-in; most algorithm tests opt in explicitly.
ENABLED = NoiseReductionConfig(enabled=True)


def tone(amplitude, duration=1.0, frequency=440.0):
    """Generate a deterministic mono 16 kHz test tone."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def rms(samples):
    work = np.asarray(samples, dtype=np.float64)
    if work.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(work))))


def dbfs(value):
    if value <= 0.0:
        return -float("inf")
    return 20.0 * np.log10(value)


def block_rms(samples, block=320):
    """RMS per 20 ms block, dropping a trailing partial block."""
    work = np.asarray(samples, dtype=np.float64)
    count = work.size // block
    return np.sqrt(np.mean(work[: count * block].reshape(count, block) ** 2, axis=1))


def quiet_block_gain_db(reference, output, quiet_mask, block=320):
    """Gain per fully-quiet 20 ms block, in dB."""
    count = reference.size // block
    blocks = quiet_mask[: count * block].reshape(count, block).all(axis=1)
    before = block_rms(reference, block)
    after = block_rms(output, block)
    with np.errstate(divide="ignore"):
        gain = 20.0 * np.log10(
            np.maximum(after, 1e-12) / np.maximum(before, 1e-12)
        )
    return gain[blocks]


def speech_like(duration=3.0):
    """Build deterministic speech-like audio and its voiced-frame mask."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    envelope = np.zeros_like(times)
    spans = [(0.10, 0.60), (0.80, 1.30), (1.60, 2.10), (2.30, 2.80)]

    for start, end in spans:
        selected = (times >= start) & (times < end)
        envelope[selected] = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * 3.0 * (times[selected] - start)
        )

    fundamental = 140.0 + 20.0 * np.sin(2.0 * np.pi * 0.7 * times)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / SAMPLE_RATE
    harmonics = sum(
        (1.0 / harmonic) * np.sin(harmonic * phase)
        for harmonic in range(1, 12)
    )
    signal = envelope * 0.15 * harmonics

    rng = np.random.default_rng(1)
    for start, end in [(0.55, 0.70), (1.25, 1.40), (2.05, 2.20), (2.75, 2.90)]:
        selected = (times >= start) & (times < end)
        signal[selected] += 0.03 * rng.standard_normal(int(selected.sum()))

    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak * 0.5

    speech_mask = np.zeros_like(times, dtype=bool)
    for start, end in [(0.10, 0.70), (0.80, 1.40), (1.60, 2.20), (2.30, 2.90)]:
        speech_mask |= (times >= start) & (times < end)

    return signal.astype(np.float32), speech_mask


def clean_speech(duration=3.0):
    """Speech-like audio with digital silence and no embedded noise bursts."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    envelope = np.zeros_like(times)
    for start, end in [(0.10, 0.60), (0.80, 1.30), (1.60, 2.10), (2.30, 2.80)]:
        selected = (times >= start) & (times < end)
        envelope[selected] = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * 3.0 * (times[selected] - start)
        )

    fundamental = 140.0 + 20.0 * np.sin(2.0 * np.pi * 0.7 * times)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / SAMPLE_RATE
    harmonics = sum(
        (1.0 / harmonic) * np.sin(harmonic * phase)
        for harmonic in range(1, 12)
    )
    signal = envelope * 0.15 * harmonics
    peak = float(np.max(np.abs(signal)))
    return (signal / peak * 0.5).astype(np.float32)


def continuous_speech(duration=3.0):
    """Speech-like audio whose envelope never reaches digital silence."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    envelope = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 1.3 * times))
    fundamental = 140.0 + 20.0 * np.sin(2.0 * np.pi * 0.7 * times)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / SAMPLE_RATE
    harmonics = sum(
        (1.0 / harmonic) * np.sin(harmonic * phase)
        for harmonic in range(1, 12)
    )
    signal = envelope * 0.15 * harmonics
    peak = float(np.max(np.abs(signal)))
    return (signal / peak * 0.5).astype(np.float32)


def noisy_speech(noise_amplitude=0.02, duration=3.0, seed=7):
    clean, speech_mask = speech_like(duration)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(clean.size) * noise_amplitude).astype(np.float32)
    return clean, (clean + noise).astype(np.float32), speech_mask


class NoiseReductionTests(unittest.TestCase):
    def test_steady_noise_is_reduced_without_removing_speech(self):
        clean, noisy, speech_mask = noisy_speech()
        quiet_mask = ~speech_mask
        result = reduce_noise(noisy, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertTrue(result.applied)
        self.assertEqual(result.status, "denoised")
        self.assertEqual(result.audio.shape, noisy.shape)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result.audio)))
        self.assertGreater(result.noise_frames, 0)

        noise_reduction_db = dbfs(rms(noisy[quiet_mask])) - dbfs(
            rms(result.audio[quiet_mask])
        )
        speech_change_db = dbfs(rms(result.audio[speech_mask])) - dbfs(
            rms(clean[speech_mask])
        )

        self.assertGreater(noise_reduction_db, 3.0)
        self.assertGreater(speech_change_db, -2.0)
        self.assertGreater(
            np.corrcoef(result.audio, clean)[0, 1],
            np.corrcoef(noisy, clean)[0, 1],
        )
        self.assertLessEqual(
            result.output_peak,
            result.input_peak * 1.05 + 1e-6,
        )
        self.assertLessEqual(
            result.output_rms,
            result.input_rms * 1.05 + 1e-9,
        )

    def test_noise_profile_is_estimated_from_noise_like_frames(self):
        _, noisy, _ = noisy_speech(noise_amplitude=0.02)
        result = reduce_noise(noisy, noise_config=ENABLED)

        self.assertTrue(result.applied)
        self.assertGreater(result.noise_frames, 0)
        # White noise at amplitude 0.02 has a true level of about -34 dBFS.
        self.assertLess(abs(result.estimated_noise_dbfs - dbfs(0.02)), 3.0)
        self.assertGreater(result.speech_to_noise_db, 10.0)

    def test_continuous_speech_with_noise_is_not_over_suppressed(self):
        clean = continuous_speech()
        rng = np.random.default_rng(9)
        noise = (rng.standard_normal(clean.size) * 0.02).astype(np.float32)
        noisy = (clean + noise).astype(np.float32)

        result = reduce_noise(noisy, noise_config=ENABLED)

        if result.applied:
            change_db = dbfs(rms(result.audio)) - dbfs(rms(clean))
            self.assertGreater(
                change_db,
                -2.0,
                msg=f"continuous speech was attenuated by {change_db:.2f} dB",
            )
            self.assertGreater(np.corrcoef(result.audio, clean)[0, 1], 0.97)
        else:
            self.assertIn(
                result.status,
                {"no_noise_profile", "low_snr"},
                msg=f"unexpected skip reason: {result.status}",
            )
            self.assertEqual(result.noise_frames, 0)

    def test_clean_speech_with_digital_silence_is_not_denoised(self):
        audio = clean_speech()

        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertIn(result.status, {"no_noise_profile", "low_snr", "low_noise"})
        self.assertEqual(result.noise_frames, 0)
        np.testing.assert_array_equal(result.audio, audio)

    def test_sub_threshold_noise_on_clean_speech_is_not_denoised(self):
        audio = clean_speech()
        rng = np.random.default_rng(7)
        noisy = (
            audio + (rng.standard_normal(audio.size) * 0.0005).astype(np.float32)
        ).astype(np.float32)

        result = reduce_noise(noisy, noise_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertIn(result.status, {"no_noise_profile", "low_noise", "low_snr"})
        np.testing.assert_array_equal(result.audio, noisy)

    def test_pause_regions_are_not_amplified(self):
        clean, noisy, speech_mask = noisy_speech()
        quiet_mask = ~speech_mask
        result = reduce_noise(noisy, noise_config=ENABLED)

        self.assertTrue(result.applied)

        gains = quiet_block_gain_db(noisy, result.audio, quiet_mask)
        self.assertGreater(gains.size, 0)
        self.assertLessEqual(
            float(np.max(gains)),
            1.0,
            msg="pre-echo raised a quiet 20 ms block above the input level",
        )

        amplified = np.abs(result.audio) > np.abs(noisy) + 1e-9
        self.assertFalse(
            bool(np.any(amplified)),
            msg="gating amplified samples instead of only attenuating",
        )

        # Speech must also not be silently amplified by the limiter path.
        self.assertLessEqual(
            float(np.max(np.abs(result.audio))),
            float(np.max(np.abs(noisy))) + 1e-9,
        )

    def test_guards_bypassed_still_never_amplify_digital_silence(self):
        audio = clean_speech()
        forced = dataclasses.replace(
            ENABLED,
            min_contrast_db=0.0,
            min_noise_flatness=0.0,
            min_snr_db=0.0,
        )

        result = reduce_noise(audio, noise_config=forced)

        self.assertTrue(result.applied)
        silence = audio == 0.0
        self.assertTrue(np.any(silence))
        self.assertEqual(float(np.max(np.abs(result.audio[silence]))), 0.0)
        self.assertFalse(bool(np.any(np.abs(result.audio) > np.abs(audio) + 1e-9)))

    def test_unsafe_hop_is_clamped_for_overlap_add(self):
        config = NoiseReductionConfig(
            enabled=True,
            frame_samples=512,
            hop_samples=512,
            spectral_floor_db=0.0,
            over_subtraction=0.1,
            min_noise_dbfs=-100.0,
            min_contrast_db=0.0,
            min_noise_flatness=0.0,
            min_snr_db=0.0,
            highpass_hz=0.0,
        )
        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32)

        result = reduce_noise(audio, noise_config=config)

        self.assertEqual(result.status, "denoised")
        self.assertLessEqual(
            float(np.max(np.abs(result.audio - audio))),
            1e-4,
            msg="hop above half a frame corrupted the overlap-add identity",
        )

    def test_pure_tone_is_left_unchanged(self):
        audio = tone(0.05, duration=1.0)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "no_noise_profile")
        np.testing.assert_array_equal(result.audio, audio)

    def test_noise_only_audio_is_left_unchanged(self):
        rng = np.random.default_rng(3)
        audio = (rng.standard_normal(SAMPLE_RATE) * 0.02).astype(np.float32)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "no_noise_profile")
        np.testing.assert_array_equal(result.audio, audio)

    def test_silent_audio_is_preserved_unchanged(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "silent")
        np.testing.assert_array_equal(result.audio, audio)

    def test_low_level_noise_is_left_unchanged(self):
        rng = np.random.default_rng(11)
        audio = (rng.standard_normal(SAMPLE_RATE) * 0.0005).astype(np.float32)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "low_noise")
        np.testing.assert_array_equal(result.audio, audio)

    def test_extremely_low_volume_is_not_processed(self):
        audio = tone(0.00001, duration=1.0)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "very_low_volume")
        np.testing.assert_array_equal(result.audio, audio)

    def test_empty_and_invalid_audio_are_rejected_gracefully(self):
        cases = [
            (None, "empty", False),
            (np.array([], dtype=np.float32), "empty", False),
            (
                np.array([0.0, np.nan, 0.1], dtype=np.float32),
                "non_finite",
                False,
            ),
            (
                np.array([0.0, np.inf, 0.1], dtype=np.float32),
                "non_finite",
                False,
            ),
            (np.zeros((100, 2), dtype=np.float32), "invalid_shape", False),
            (np.array([1, 2, 3], dtype=np.int16), "invalid_dtype", False),
        ]

        for audio, expected_status, expected_usable in cases:
            with self.subTest(status=expected_status):
                result = reduce_noise(audio, noise_config=ENABLED)
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.usable, expected_usable)
                self.assertFalse(result.applied)
                self.assertIsInstance(result.audio, np.ndarray)

    def test_invalid_sample_rate_is_rejected_gracefully(self):
        audio = tone(0.2)
        result = reduce_noise(audio, sample_rate=44100, noise_config=ENABLED)

        self.assertFalse(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "invalid_sample_rate")

    def test_disabled_configuration_returns_input_unchanged(self):
        _, noisy, _ = noisy_speech()
        result = reduce_noise(
            noisy,
            noise_config=NoiseReductionConfig(enabled=False),
        )

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "disabled")
        np.testing.assert_array_equal(result.audio, noisy)

    def test_default_configuration_is_disabled(self):
        _, noisy, _ = noisy_speech()
        result = reduce_noise(noisy)

        self.assertFalse(result.applied)
        self.assertEqual(result.status, "disabled")
        np.testing.assert_array_equal(result.audio, noisy)

    def test_input_is_not_modified_in_place(self):
        _, noisy, _ = noisy_speech()
        original = noisy.copy()
        reduce_noise(noisy, noise_config=ENABLED)
        np.testing.assert_array_equal(noisy, original)

    def test_mono_compatible_two_dimensional_audio_is_accepted(self):
        audio = tone(0.2, duration=0.5).reshape(-1, 1)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.size, audio.size)

    def test_short_audio_is_handled_without_crashing(self):
        rng = np.random.default_rng(5)
        audio = (rng.standard_normal(100) * 0.02).astype(np.float32)
        result = reduce_noise(audio, noise_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertEqual(result.audio.size, audio.size)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result.audio)))


class NoiseReductionConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = config_from_env({})

        self.assertFalse(config.enabled)
        self.assertEqual(config.sample_rate, SAMPLE_RATE)
        self.assertEqual(config.frame_samples, 512)
        self.assertEqual(config.hop_samples, 128)
        self.assertEqual(config.noise_percentile, 15.0)
        self.assertEqual(config.over_subtraction, 2.5)
        self.assertEqual(config.spectral_floor_db, -18.0)
        self.assertEqual(config.min_noise_dbfs, -60.0)
        self.assertEqual(config.min_contrast_db, 6.0)
        self.assertEqual(config.min_noise_flatness, 0.20)
        self.assertEqual(config.min_snr_db, 10.0)
        self.assertEqual(config.highpass_hz, 60.0)

    def test_environment_enables_and_disables_the_feature(self):
        for value in ["1", "true", "TRUE", "yes", "on", "y", "enabled", " true "]:
            with self.subTest(value=value):
                self.assertTrue(
                    config_from_env({"AXION_AUDIO_DENOISE": value}).enabled
                )
        for value in ["0", "false", "no", "off", "n", "disabled", ""]:
            with self.subTest(value=value):
                self.assertFalse(
                    config_from_env({"AXION_AUDIO_DENOISE": value}).enabled
                )

    def test_invalid_boolean_value_is_reported_and_falls_back(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            config = config_from_env({"AXION_AUDIO_DENOISE": "2"})

        self.assertFalse(config.enabled)
        message = stderr.getvalue()
        self.assertIn("AXION_AUDIO_DENOISE", message)
        self.assertIn("not a valid boolean", message)

    def test_environment_overrides(self):
        config = config_from_env({
            "AXION_AUDIO_DENOISE": "0",
            "AXION_AUDIO_DENOISE_FLOOR_DB": "-24",
            "AXION_AUDIO_DENOISE_OVER_SUBTRACTION": "3.5",
            "AXION_AUDIO_DENOISE_NOISE_PERCENTILE": "20",
            "AXION_AUDIO_DENOISE_MIN_NOISE_DBFS": "-70",
            "AXION_AUDIO_DENOISE_MIN_CONTRAST_DB": "4",
            "AXION_AUDIO_DENOISE_MIN_FLATNESS": "0.35",
            "AXION_AUDIO_DENOISE_MIN_SNR_DB": "15",
            "AXION_AUDIO_DENOISE_HIGHPASS_HZ": "0",
        })

        self.assertFalse(config.enabled)
        self.assertEqual(config.spectral_floor_db, -24.0)
        self.assertEqual(config.over_subtraction, 3.5)
        self.assertEqual(config.noise_percentile, 20.0)
        self.assertEqual(config.min_noise_dbfs, -70.0)
        self.assertEqual(config.min_contrast_db, 4.0)
        self.assertEqual(config.min_noise_flatness, 0.35)
        self.assertEqual(config.min_snr_db, 15.0)
        self.assertEqual(config.highpass_hz, 0.0)

    def test_invalid_or_out_of_range_environment_values_fall_back(self):
        config = config_from_env({
            "AXION_AUDIO_DENOISE_FLOOR_DB": "not-a-number",
            "AXION_AUDIO_DENOISE_OVER_SUBTRACTION": "99",
            "AXION_AUDIO_DENOISE_NOISE_PERCENTILE": "0",
            "AXION_AUDIO_DENOISE_MIN_NOISE_DBFS": "5",
            "AXION_AUDIO_DENOISE_MIN_CONTRAST_DB": "-1",
            "AXION_AUDIO_DENOISE_MIN_FLATNESS": "2",
            "AXION_AUDIO_DENOISE_MIN_SNR_DB": "-5",
            "AXION_AUDIO_DENOISE_HIGHPASS_HZ": "999",
        })

        self.assertEqual(config.spectral_floor_db, -18.0)
        self.assertEqual(config.over_subtraction, 2.5)
        self.assertEqual(config.noise_percentile, 15.0)
        self.assertEqual(config.min_noise_dbfs, -60.0)
        self.assertEqual(config.min_contrast_db, 6.0)
        self.assertEqual(config.min_noise_flatness, 0.20)
        self.assertEqual(config.min_snr_db, 10.0)
        self.assertEqual(config.highpass_hz, 60.0)


class FakeModel:
    def __init__(self, text="hello axion"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((np.array(audio, copy=True), options))
        return {
            "text": self.text,
            "segments": [
                {"text": self.text, "no_speech_prob": 0.05},
            ],
        }


class SpeechToTextIntegrationTests(unittest.TestCase):
    def _stt_with_fake_model(self, text="hello axion"):
        stt = SpeechToText.__new__(SpeechToText)
        stt.model = FakeModel(text=text)
        return stt

    def test_transcribe_runs_denoise_before_normalization(self):
        clean, noisy, _ = noisy_speech(duration=2.0)
        stt = self._stt_with_fake_model()
        calls = []

        def ordered_reduce(audio, **kwargs):
            calls.append("denoise")
            return reduce_noise(audio, **kwargs)

        def ordered_normalize(audio, **kwargs):
            calls.append("normalize")
            return normalize_audio(audio, **kwargs)

        with patch("axion.stt.reduce_noise", side_effect=ordered_reduce), \
                patch("axion.stt.normalize_audio", side_effect=ordered_normalize), \
                patch("axion.stt.whisper.load_audio", return_value=noisy.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "1",
                    "AXION_AUDIO_NORMALIZE": "1",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(calls, ["denoise", "normalize"])
        self.assertEqual(len(stt.model.calls), 1)

        received, options = stt.model.calls[0]
        self.assertEqual(received.shape, clean.shape)
        self.assertLessEqual(float(np.max(np.abs(received))), 0.95 + 1e-6)
        self.assertGreater(
            np.corrcoef(received, clean)[0, 1],
            np.corrcoef(noisy, clean)[0, 1],
        )
        self.assertEqual(options["language"], "en")
        self.assertFalse(options["fp16"])
        self.assertEqual(options["temperature"], 0)

    def test_transcribe_with_denoise_disabled_still_normalizes(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "0",
                    "AXION_AUDIO_NORMALIZE": "1",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        self.assertGreater(rms(received), rms(quiet) * 5.0)

    def test_transcribe_without_denoise_setting_leaves_audio_untouched(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_NORMALIZE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, quiet)

    def test_transcribe_with_denoise_disabled_leaves_audio_untouched(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "0",
                    "AXION_AUDIO_NORMALIZE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, quiet)

    def test_transcribe_denoises_when_normalization_is_disabled(self):
        clean, noisy, _ = noisy_speech(duration=2.0)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=noisy.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "1",
                    "AXION_AUDIO_NORMALIZE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        self.assertEqual(received.shape, noisy.shape)
        self.assertGreater(
            np.corrcoef(received, clean)[0, 1],
            np.corrcoef(noisy, clean)[0, 1],
        )

    def test_transcribe_falls_back_to_original_audio_if_denoise_fails(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch(
            "axion.stt.reduce_noise",
            side_effect=RuntimeError("simulated denoise failure"),
        ), patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "1",
                    "AXION_AUDIO_NORMALIZE": "1",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(len(stt.model.calls), 1)
        received, _ = stt.model.calls[0]
        self.assertEqual(received.shape, quiet.shape)
        self.assertGreater(rms(received), rms(quiet) * 5.0)


if __name__ == "__main__":
    unittest.main()
