"""Unit tests for AXION audio validation and pre-Whisper normalization."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from axion.audio_normalization import (
    AudioNormalizationConfig,
    config_from_env,
    normalize_audio,
    validate_audio,
)
from axion.stt import SpeechToText

SAMPLE_RATE = 16000


def tone(amplitude, duration=1.0, frequency=440.0):
    """Generate a deterministic mono 16 kHz test tone."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def rms(samples):
    work = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(work))))


class AudioNormalizationTests(unittest.TestCase):
    def test_normal_audio_is_normalized_without_clipping(self):
        audio = tone(0.30)
        result = normalize_audio(audio)

        self.assertTrue(result.usable)
        self.assertTrue(result.applied)
        self.assertEqual(result.status, "normalized")
        self.assertEqual(result.audio.shape, audio.shape)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(result.audio))), 0.95 + 1e-6)
        self.assertFalse(np.array_equal(result.audio, audio))
        self.assertTrue(np.all(np.isfinite(result.audio)))

    def test_quiet_audio_is_boosted(self):
        audio = tone(0.005)
        result = normalize_audio(audio)

        self.assertTrue(result.applied)
        self.assertGreater(result.output_rms, result.input_rms * 5.0)
        self.assertLessEqual(result.output_peak, 0.95 + 1e-6)
        self.assertGreater(result.gain, 1.0)
        self.assertLessEqual(result.gain, 10.0 + 1e-6)  # +20 dB default cap

    def test_loud_audio_is_attenuated(self):
        audio = tone(0.999)
        validation = validate_audio(audio)
        result = normalize_audio(audio)

        self.assertEqual(validation.status, "very_loud")
        self.assertTrue(result.applied)
        self.assertLess(result.output_peak, result.input_peak)
        self.assertLessEqual(result.output_peak, 0.95 + 1e-6)
        self.assertGreater(result.output_rms, 0.01)

    def test_silent_audio_is_preserved_unchanged(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = normalize_audio(audio)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "silent")
        np.testing.assert_array_equal(result.audio, audio)

    def test_extremely_low_volume_is_not_boosted(self):
        audio = tone(0.00001)
        result = normalize_audio(audio)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "very_low_volume")
        np.testing.assert_array_equal(result.audio, audio)

    def test_empty_audio_is_rejected_gracefully(self):
        empty = np.array([], dtype=np.float32)
        result = normalize_audio(empty)

        self.assertFalse(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.audio.size, 0)

        none_result = normalize_audio(None)
        self.assertFalse(none_result.usable)
        self.assertEqual(none_result.status, "empty")

    def test_invalid_audio_is_rejected_gracefully(self):
        cases = [
            ("non_finite", np.array([0.0, np.nan, 0.1], dtype=np.float32)),
            ("non_finite", np.array([0.0, np.inf, 0.1], dtype=np.float32)),
            ("invalid_shape", np.zeros((100, 2), dtype=np.float32)),
            ("invalid_dtype", np.array(["not", "audio"], dtype=object)),
        ]
        for expected_status, audio in cases:
            with self.subTest(expected_status=expected_status):
                result = normalize_audio(audio)
                self.assertFalse(result.usable)
                self.assertFalse(result.applied)
                self.assertEqual(result.status, expected_status)
                self.assertIsInstance(result.audio, np.ndarray)

    def test_invalid_sample_rate_is_rejected_gracefully(self):
        audio = tone(0.2)
        result = normalize_audio(audio, sample_rate=44100)
        self.assertFalse(result.usable)
        self.assertEqual(result.status, "invalid_sample_rate")

        for bad_rate in (0, -16000, float("nan"), "16000"):
            with self.subTest(rate=bad_rate):
                validation = validate_audio(audio, sample_rate=bad_rate)
                self.assertFalse(validation.usable)
                self.assertEqual(validation.status, "invalid_sample_rate")

    def test_clipping_prevention_caps_peak_instead_of_hard_clipping(self):
        audio = tone(0.5)
        config = AudioNormalizationConfig(
            target_rms_dbfs=-3.0,
            max_gain_db=30.0,
            peak_limit=0.95,
        )
        result = normalize_audio(audio, normalization_config=config)

        self.assertTrue(result.applied)
        self.assertLessEqual(result.output_peak, 0.95 + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(result.audio))), 0.95 + 1e-6)
        # Rescaling preserves the waveform shape; it must not create a flat
        # clipped region of samples sitting exactly at full scale.
        self.assertLess(float(np.mean(np.abs(result.audio) >= 0.95)), 0.01)

    def test_output_sample_rate_and_shape_are_preserved(self):
        audio = tone(0.2, duration=0.5)
        validation = validate_audio(audio)
        result = normalize_audio(audio)

        self.assertEqual(validation.sample_rate, SAMPLE_RATE)
        self.assertEqual(validation.samples, audio.size)
        self.assertEqual(result.audio.shape, audio.shape)
        self.assertEqual(result.audio.size, audio.size)
        self.assertEqual(result.audio.dtype, np.float32)

    def test_normalization_can_be_disabled(self):
        audio = tone(0.2)
        config = AudioNormalizationConfig(enabled=False)
        result = normalize_audio(audio, normalization_config=config)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "disabled")
        np.testing.assert_array_equal(result.audio, audio)

    def test_mono_compatible_two_dimensional_audio_is_accepted(self):
        audio = tone(0.2, duration=0.25).reshape(-1, 1)
        result = normalize_audio(audio)

        self.assertTrue(result.usable)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.size, audio.size)

    def test_legacy_fixed_gain_is_peak_protected(self):
        audio = tone(0.3)
        config = AudioNormalizationConfig(fixed_gain=10.0)
        result = normalize_audio(audio, normalization_config=config)

        self.assertTrue(result.applied)
        self.assertEqual(result.status, "fixed_gain")
        self.assertLessEqual(result.output_peak, 0.95 + 1e-6)


class AudioNormalizationConfigTests(unittest.TestCase):
    def test_defaults_are_enabled(self):
        config = AudioNormalizationConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.sample_rate, SAMPLE_RATE)

    def test_environment_can_disable_normalization(self):
        self.assertFalse(config_from_env({"AXION_AUDIO_NORMALIZE": "0"}).enabled)
        self.assertTrue(config_from_env({"AXION_AUDIO_NORMALIZE": "1"}).enabled)

    def test_environment_values_are_parsed_and_clamped(self):
        config = config_from_env({
            "AXION_AUDIO_TARGET_DBFS": "-15",
            "AXION_AUDIO_MAX_GAIN_DB": "12",
            "AXION_AUDIO_MAX_ATTENUATION_DB": "-6",
            "AXION_AUDIO_PEAK_LIMIT": "0.9",
        })
        self.assertEqual(config.target_rms_dbfs, -15.0)
        self.assertEqual(config.max_gain_db, 12.0)
        self.assertEqual(config.max_attenuation_db, -6.0)
        self.assertEqual(config.peak_limit, 0.9)

        fallback = config_from_env({
            "AXION_AUDIO_TARGET_DBFS": "not-a-number",
            "AXION_AUDIO_PEAK_LIMIT": "2.0",
        })
        self.assertEqual(fallback.target_rms_dbfs, -20.0)
        self.assertEqual(fallback.peak_limit, 0.95)

    def test_legacy_gain_environment_variable(self):
        self.assertFalse(config_from_env({"AXION_AUDIO_GAIN": "0"}).enabled)
        config = config_from_env({"AXION_AUDIO_GAIN": "1.5"})
        self.assertTrue(config.enabled)
        self.assertAlmostEqual(config.fixed_gain, 1.5)


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

    def test_transcribe_normalizes_audio_before_whisper(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {"AXION_AUDIO_NORMALIZE": "1"}, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(len(stt.model.calls), 1)
        received, options = stt.model.calls[0]
        self.assertEqual(received.shape, quiet.shape)
        self.assertLessEqual(float(np.max(np.abs(received))), 0.95 + 1e-6)
        self.assertGreater(rms(received), rms(quiet) * 5.0)
        self.assertEqual(options["language"], "en")
        self.assertFalse(options["fp16"])
        self.assertEqual(options["temperature"], 0)

    def test_transcribe_preserves_audio_when_normalization_disabled(self):
        quiet = tone(0.005, duration=0.5)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=quiet.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {"AXION_AUDIO_NORMALIZE": "0"}, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, quiet)

    def test_transcribe_skips_whisper_for_invalid_audio(self):
        invalid = np.array([0.0, np.nan, 0.1], dtype=np.float32)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=invalid), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {"AXION_AUDIO_NORMALIZE": "1"}, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "")
        self.assertEqual(stt.model.calls, [])


if __name__ == "__main__":
    unittest.main()
