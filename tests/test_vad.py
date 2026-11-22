"""Unit and integration tests for AXION voice activity detection."""

import contextlib
import dataclasses
import io
import os
import unittest
from unittest.mock import patch

import numpy as np

from axion.audio_normalization import config_from_env as normalization_config
from axion.audio_normalization import normalize_audio
from axion.noise_reduction import config_from_env as noise_config
from axion.noise_reduction import reduce_noise
from axion.stt import SpeechToText
from axion.vad import (
    VadConfig,
    config_from_env,
    detect_speech,
    extract_speech,
)

SAMPLE_RATE = 16000

# VAD is opt-in; algorithm tests opt in explicitly.
ENABLED = VadConfig(enabled=True)

# A harmonic word 6 dB above the -46 dBFS test noise: audible, but below the
# main 8 dB speech margin.  Used by the boundary-protection regressions.
QUIET_WORD_AMP = 0.005 * 10 ** (6.0 / 20.0) * 1.75 / 0.81


def tone(amplitude, duration=1.0, frequency=200.0):
    """Generate a deterministic mono 16 kHz test tone."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def rms(samples):
    work = np.asarray(samples, dtype=np.float64)
    if work.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(work))))


def speech_burst(duration=3.0, spans=((0.6, 1.4),), amplitude=0.2):
    """Build harmonic speech-like bursts separated by digital silence."""
    times = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros(times.size, dtype=np.float64)

    for start, end in spans:
        selected = (times >= start) & (times < end)
        local = times[selected] - start
        span = max(end - start, 1e-6)
        # Flat top with a 10 ms raised-cosine attack/release: realistic word
        # edges without the long fade that would hide speech from any
        # energy-based detector.
        fade = min(0.010, span / 4.0)
        attack = np.clip(local / fade, 0.0, 1.0)
        release = np.clip((span - local) / fade, 0.0, 1.0)
        envelope = np.minimum(attack, release)
        envelope = np.sin(0.5 * np.pi * envelope) ** 2
        phase = 2.0 * np.pi * 180.0 * local
        wave = (
            np.sin(phase)
            + 0.5 * np.sin(2.0 * phase)
            + 0.25 * np.sin(3.0 * phase)
        )
        signal[selected] = amplitude * envelope * wave / 1.75

    return signal.astype(np.float32)


def white_noise(amplitude, duration=3.0, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SAMPLE_RATE * duration)) * amplitude).astype(
        np.float32
    )


class VadConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = config_from_env({})

        self.assertFalse(config.enabled)
        self.assertEqual(config.sample_rate, SAMPLE_RATE)
        self.assertEqual(config.frame_ms, 25.0)
        self.assertEqual(config.hop_ms, 10.0)
        self.assertEqual(config.noise_percentile, 20.0)
        self.assertEqual(config.speech_margin_db, 8.0)
        self.assertEqual(config.hysteresis_db, 3.0)
        self.assertEqual(config.min_speech_dbfs, -60.0)
        self.assertEqual(config.min_speech_ms, 150.0)
        self.assertEqual(config.min_silence_ms, 300.0)
        self.assertEqual(config.pre_roll_ms, 200.0)
        self.assertEqual(config.post_roll_ms, 250.0)
        self.assertEqual(config.min_total_speech_ms, 200.0)
        self.assertEqual(config.min_trim_ms, 300.0)
        self.assertEqual(config.boundary_percentile, 5.0)
        self.assertEqual(config.boundary_margin_db, 3.0)
        self.assertEqual(config.boundary_min_dbfs, -70.0)
        self.assertEqual(config.boundary_min_activity_ms, 80.0)
        self.assertEqual(config.boundary_link_ms, 500.0)
        self.assertEqual(config.min_confident_speech_ms, 300.0)
        self.assertEqual(config.min_contrast_db, 6.0)
        self.assertEqual(config.max_speech_ratio, 0.70)

    def test_environment_enables_and_disables_the_feature(self):
        for value in ["1", "true", "TRUE", "yes", "on", "y", "enabled", " true "]:
            with self.subTest(value=value):
                self.assertTrue(
                    config_from_env({"AXION_AUDIO_VAD": value}).enabled
                )
        for value in ["0", "false", "no", "off", "n", "disabled", ""]:
            with self.subTest(value=value):
                self.assertFalse(
                    config_from_env({"AXION_AUDIO_VAD": value}).enabled
                )

    def test_invalid_boolean_value_is_reported_and_falls_back(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            config = config_from_env({"AXION_AUDIO_VAD": "maybe"})

        self.assertFalse(config.enabled)
        message = stderr.getvalue()
        self.assertIn("AXION_AUDIO_VAD", message)
        self.assertIn("not a valid boolean", message)

    def test_environment_overrides(self):
        config = config_from_env({
            "AXION_AUDIO_VAD": "1",
            "AXION_AUDIO_VAD_NOISE_PERCENTILE": "30",
            "AXION_AUDIO_VAD_MARGIN_DB": "12",
            "AXION_AUDIO_VAD_MIN_SPEECH_DBFS": "-50",
            "AXION_AUDIO_VAD_MIN_SPEECH_MS": "250",
            "AXION_AUDIO_VAD_MIN_SILENCE_MS": "500",
            "AXION_AUDIO_VAD_PRE_ROLL_MS": "300",
            "AXION_AUDIO_VAD_POST_ROLL_MS": "350",
        })

        self.assertTrue(config.enabled)
        self.assertEqual(config.noise_percentile, 30.0)
        self.assertEqual(config.speech_margin_db, 12.0)
        self.assertEqual(config.min_speech_dbfs, -50.0)
        self.assertEqual(config.min_speech_ms, 250.0)
        self.assertEqual(config.min_silence_ms, 500.0)
        self.assertEqual(config.pre_roll_ms, 300.0)
        self.assertEqual(config.post_roll_ms, 350.0)

    def test_invalid_numeric_values_are_reported_and_fall_back(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            config = config_from_env({
                "AXION_AUDIO_VAD_MARGIN_DB": "abc",
                "AXION_AUDIO_VAD_MIN_SPEECH_MS": "-5",
                "AXION_AUDIO_VAD_MIN_SILENCE_MS": "nan",
            })

        self.assertEqual(config.speech_margin_db, 8.0)
        self.assertEqual(config.min_speech_ms, 150.0)
        self.assertEqual(config.min_silence_ms, 300.0)
        message = stderr.getvalue()
        self.assertIn("AXION_AUDIO_VAD_MARGIN_DB", message)
        self.assertIn("not a valid number", message)
        self.assertIn("outside", message)

    def test_vad_setting_is_independent_of_other_audio_features(self):
        self.assertTrue(config_from_env({"AXION_AUDIO_VAD": "1"}).enabled)
        self.assertFalse(noise_config({"AXION_AUDIO_VAD": "1"}).enabled)
        self.assertTrue(normalization_config({"AXION_AUDIO_VAD": "1"}).enabled)

        self.assertFalse(config_from_env({"AXION_AUDIO_DENOISE": "1"}).enabled)
        self.assertFalse(config_from_env({"AXION_AUDIO_NORMALIZE": "0"}).enabled)
        self.assertFalse(
            config_from_env({
                "AXION_AUDIO_VAD": "0",
                "AXION_AUDIO_DENOISE": "1",
                "AXION_AUDIO_NORMALIZE": "1",
            }).enabled
        )


class VadValidationTests(unittest.TestCase):
    def test_invalid_audio_is_rejected_gracefully(self):
        cases = [
            (None, "empty", False),
            (np.array([], dtype=np.float32), "empty", False),
            (np.array([0.0, np.nan, 0.1], dtype=np.float32), "non_finite", False),
            (np.array([0.0, np.inf, 0.1], dtype=np.float32), "non_finite", False),
            (np.zeros((100, 2), dtype=np.float32), "invalid_shape", False),
            (np.array([1, 2, 3], dtype=np.int16), "invalid_dtype", False),
            (np.float32(0.5), "invalid_shape", False),
            ("not audio", "invalid_shape", False),
        ]

        for audio, expected_status, expected_usable in cases:
            with self.subTest(status=expected_status):
                result = detect_speech(audio, vad_config=ENABLED)
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.usable, expected_usable)
                self.assertFalse(result.applied)
                self.assertIsInstance(result.audio, np.ndarray)
                if audio is None:
                    self.assertEqual(result.audio.size, 0)
                else:
                    # Malformed input is returned untouched, not replaced by
                    # an unrelated empty buffer.
                    np.testing.assert_array_equal(
                        result.audio, np.asarray(audio)
                    )

    def test_invalid_sample_rate_is_rejected_gracefully(self):
        result = detect_speech(
            tone(0.2), sample_rate=44100, vad_config=ENABLED
        )

        self.assertFalse(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "invalid_sample_rate")

    def test_silent_audio_is_preserved_unchanged(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "silent")
        np.testing.assert_array_equal(result.audio, audio)

    def test_short_audio_is_preserved_unchanged(self):
        audio = tone(0.2, duration=0.05)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "too_short")
        np.testing.assert_array_equal(result.audio, audio)

    def test_two_dimensional_mono_audio_is_accepted(self):
        audio = speech_burst().reshape(-1, 1)
        result = detect_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.size, audio.size)

    def test_float64_audio_is_accepted(self):
        audio = speech_burst().astype(np.float64)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertTrue(result.applied)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result.audio)))

    def test_config_sample_rate_cannot_override_validated_rate(self):
        audio = speech_burst(3.0, [(1.0, 1.8)], amplitude=0.2)
        result = extract_speech(
            audio,
            sample_rate=16000,
            vad_config=VadConfig(enabled=True, sample_rate=8000),
        )

        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.status, "trimmed")
        self.assertEqual(len(result.segments), 1)
        self.assertAlmostEqual(
            result.segments[0].start_seconds, 1.0, delta=0.08
        )

    def test_input_is_not_modified_in_place(self):
        audio = speech_burst()
        original = audio.copy()

        extract_speech(audio, vad_config=ENABLED)

        np.testing.assert_array_equal(audio, original)

    def test_default_configuration_is_disabled(self):
        audio = speech_burst()
        with patch.dict(os.environ, {}, clear=True):
            result = extract_speech(audio)

        self.assertFalse(result.usable is False)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "disabled")
        np.testing.assert_array_equal(result.audio, audio)

    def test_explicitly_disabled_configuration_returns_input(self):
        audio = speech_burst()
        result = extract_speech(
            audio, vad_config=VadConfig(enabled=False)
        )

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "disabled")
        np.testing.assert_array_equal(result.audio, audio)


class VadDetectionTests(unittest.TestCase):
    def test_single_burst_is_trimmed_with_pre_and_post_roll(self):
        audio = speech_burst(3.0, [(0.6, 1.4)], amplitude=0.2)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertTrue(result.applied)
        self.assertEqual(result.status, "trimmed")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result.audio)))

        segment = result.segments[0]
        self.assertAlmostEqual(
            segment.start_seconds * SAMPLE_RATE, segment.start_sample, places=6
        )
        self.assertAlmostEqual(
            segment.end_seconds * SAMPLE_RATE, segment.end_sample, places=6
        )
        self.assertGreaterEqual(segment.start_seconds, 0.55)
        self.assertLessEqual(segment.start_seconds, 0.65)
        self.assertGreaterEqual(segment.end_seconds, 1.35)
        self.assertLessEqual(segment.end_seconds, 1.45)

        # Boundaries must include the whole word, not cut its onset or tail.
        self.assertLessEqual(result.start_sample, int(0.60 * SAMPLE_RATE))
        self.assertGreaterEqual(result.end_sample, int(1.40 * SAMPLE_RATE))
        self.assertLess(result.start_sample, result.end_sample)
        np.testing.assert_array_equal(
            result.audio, audio[result.start_sample:result.end_sample]
        )
        self.assertEqual(result.output_samples, result.audio.size)
        self.assertLess(result.output_samples, audio.size)

    def test_two_bursts_keep_both_and_preserve_the_gap(self):
        audio = speech_burst(3.0, [(0.4, 0.9), (2.0, 2.5)])
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(result.status, "trimmed")
        self.assertEqual(len(result.segments), 2)
        self.assertLessEqual(result.start_sample, int(0.4 * SAMPLE_RATE))
        self.assertGreaterEqual(result.end_sample, int(2.5 * SAMPLE_RATE))
        self.assertGreaterEqual(
            result.output_samples, int((2.5 - 0.4) * SAMPLE_RATE)
        )
        np.testing.assert_array_equal(
            result.audio, audio[result.start_sample:result.end_sample]
        )

    def test_short_gap_between_bursts_is_bridged(self):
        audio = speech_burst(3.0, [(0.6, 1.0), (1.15, 1.55)])
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(result.status, "trimmed")
        self.assertEqual(len(result.segments), 1)
        self.assertLessEqual(result.start_sample, int(0.6 * SAMPLE_RATE))
        self.assertGreaterEqual(result.end_sample, int(1.55 * SAMPLE_RATE))

    def test_very_short_burst_is_never_discarded(self):
        audio = speech_burst(3.0, [(1.0, 1.08)])
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertIn(result.status, {"too_short", "no_speech"})
        np.testing.assert_array_equal(result.audio, audio)

    def test_speech_touching_the_edges_is_not_trimmed(self):
        audio = speech_burst(3.0, [(0.0, 3.0)])
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertIn(
            result.status,
            {"already_tight", "no_speech", "speech_dominant"},
        )
        np.testing.assert_array_equal(result.audio, audio)

    def test_quiet_speech_over_noise_is_detected(self):
        audio = (
            speech_burst(3.0, [(0.7, 1.5)], amplitude=0.008)
            + white_noise(0.0005, 3.0, seed=11)
        ).astype(np.float32)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(result.status, "trimmed")
        self.assertLessEqual(result.start_sample, int(0.7 * SAMPLE_RATE))
        self.assertGreaterEqual(result.end_sample, int(1.5 * SAMPLE_RATE))
        self.assertLess(result.noise_floor_dbfs, -55.0)
        self.assertGreaterEqual(result.threshold_dbfs, -60.0)
        self.assertGreater(rms(result.audio), 0.0)

    def test_continuous_speech_is_not_over_trimmed(self):
        audio = speech_burst(3.0, [(0.15, 2.85)])
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertIn(
            result.status,
            {"already_tight", "trimmed", "no_speech", "speech_dominant"},
        )
        if result.applied:
            self.assertLessEqual(result.start_sample, int(0.15 * SAMPLE_RATE))
            self.assertGreaterEqual(result.end_sample, int(2.85 * SAMPLE_RATE))
            np.testing.assert_array_equal(
                result.audio, audio[result.start_sample:result.end_sample]
            )
        else:
            np.testing.assert_array_equal(result.audio, audio)

    def test_speech_dominant_audio_is_not_trimmed(self):
        # Speech fills more than 70 % of the buffer, so the adaptive floor may
        # be estimated from speech itself; passthrough is safer than trimming
        # quiet word edges.
        audio = (
            speech_burst(3.0, [(0.35, 2.65)], amplitude=0.2)
            + white_noise(0.001, 3.0, seed=3)
        ).astype(np.float32)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(result.status, "speech_dominant")
        self.assertFalse(result.applied)
        self.assertEqual(len(result.segments), 1)
        np.testing.assert_array_equal(result.audio, audio)

    def test_noisy_speech_is_detected_without_clipping_speech(self):
        audio = (
            speech_burst(3.0, [(0.7, 1.6)], amplitude=0.2)
            + white_noise(0.02, 3.0, seed=13)
        ).astype(np.float32)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(result.status, "trimmed")
        self.assertLessEqual(result.start_sample, int(0.7 * SAMPLE_RATE))
        self.assertGreaterEqual(result.end_sample, int(1.6 * SAMPLE_RATE))
        self.assertGreater(result.speech_frames, 0)
        self.assertGreater(result.speech_ratio, 0.0)

    def test_noise_only_audio_is_not_trimmed(self):
        audio = white_noise(0.02, 3.0, seed=17)
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "no_speech")
        np.testing.assert_array_equal(result.audio, audio)

    def test_zero_roll_configuration_uses_exact_segment_bounds(self):
        audio = speech_burst(3.0, [(0.6, 1.4)])
        config = dataclasses.replace(
            ENABLED, pre_roll_ms=0.0, post_roll_ms=0.0
        )
        result = extract_speech(audio, vad_config=config)

        self.assertEqual(result.status, "trimmed")
        self.assertEqual(result.start_sample, result.segments[0].start_sample)
        self.assertEqual(result.end_sample, result.segments[-1].end_sample)

    def test_trim_is_skipped_when_removal_is_below_min_trim(self):
        audio = speech_burst(3.0, [(0.6, 1.4)])
        config = dataclasses.replace(ENABLED, min_trim_ms=5000.0)
        result = extract_speech(audio, vad_config=config)

        self.assertEqual(result.status, "already_tight")
        self.assertFalse(result.applied)
        np.testing.assert_array_equal(result.audio, audio)

    def test_detection_keeps_the_audio_and_reports_segments(self):
        audio = speech_burst(3.0, [(0.6, 1.4)])
        result = detect_speech(audio, vad_config=ENABLED)

        self.assertTrue(result.usable)
        self.assertFalse(result.applied)
        self.assertEqual(result.status, "detected")
        self.assertEqual(len(result.segments), 1)
        np.testing.assert_array_equal(result.audio, audio)
        self.assertGreater(result.frames, 0)
        self.assertGreater(result.speech_frames, 0)
        self.assertGreater(result.speech_ratio, 0.0)
        self.assertLess(result.speech_ratio, 1.0)

    def test_segment_timestamps_are_monotonic_and_in_range(self):
        audio = speech_burst(3.0, [(0.4, 0.9), (2.0, 2.5)])
        result = extract_speech(audio, vad_config=ENABLED)

        previous_end = 0
        for segment in result.segments:
            self.assertGreaterEqual(segment.start_sample, previous_end)
            self.assertGreater(segment.end_sample, segment.start_sample)
            self.assertLessEqual(segment.end_sample, audio.size)
            self.assertAlmostEqual(
                segment.start_seconds,
                segment.start_sample / SAMPLE_RATE,
                places=6,
            )
            self.assertAlmostEqual(
                segment.duration_seconds,
                (segment.end_sample - segment.start_sample) / SAMPLE_RATE,
                places=6,
            )
            previous_end = segment.end_sample

    def test_detection_is_deterministic(self):
        audio = speech_burst(3.0, [(0.6, 1.4)])
        first = extract_speech(audio, vad_config=ENABLED)
        second = extract_speech(audio, vad_config=ENABLED)

        self.assertEqual(first.status, second.status)
        self.assertEqual(first.segments, second.segments)
        np.testing.assert_array_equal(first.audio, second.audio)


class VadBoundaryProtectionTests(unittest.TestCase):
    """Regressions for quiet words at the edges of a louder utterance."""

    def _mixed(self, duration, loud_span, quiet_span, seed=31):
        audio = white_noise(0.005, duration, seed=seed)
        audio = audio + speech_burst(
            duration, [quiet_span], amplitude=QUIET_WORD_AMP
        )
        audio = audio + speech_burst(
            duration, [loud_span], amplitude=0.2
        )
        return audio.astype(np.float32)

    def test_quiet_leading_word_survives_pauses_up_to_600ms(self):
        for gap in (0.1, 0.2, 0.3, 0.6):
            with self.subTest(gap=gap):
                quiet = (0.50, 0.75)
                loud = (0.75 + gap, 1.35 + gap)
                audio = self._mixed(3.0, loud, quiet, seed=31)
                result = extract_speech(audio, vad_config=ENABLED)

                self.assertEqual(result.status, "trimmed")
                self.assertTrue(result.boundary_expanded)
                self.assertLessEqual(
                    result.start_sample, int(quiet[0] * SAMPLE_RATE)
                )
                self.assertGreaterEqual(
                    result.end_sample, int(loud[1] * SAMPLE_RATE)
                )
                np.testing.assert_array_equal(
                    result.audio,
                    audio[result.start_sample:result.end_sample],
                )

    def test_quiet_trailing_word_survives_pauses_up_to_600ms(self):
        for gap in (0.1, 0.3, 0.6):
            with self.subTest(gap=gap):
                loud = (0.50, 1.20)
                quiet = (1.20 + gap, 1.80 + gap)
                audio = self._mixed(3.0, loud, quiet, seed=32)
                result = extract_speech(audio, vad_config=ENABLED)

                self.assertEqual(result.status, "trimmed")
                self.assertTrue(result.boundary_expanded)
                self.assertLessEqual(
                    result.start_sample, int(loud[0] * SAMPLE_RATE)
                )
                self.assertGreaterEqual(
                    result.end_sample, int(quiet[1] * SAMPLE_RATE)
                )
                np.testing.assert_array_equal(
                    result.audio,
                    audio[result.start_sample:result.end_sample],
                )

    def test_dominance_guard_covers_quiet_edges_at_90_percent_occupancy(self):
        audio = np.zeros(int(SAMPLE_RATE * 4.0), dtype=np.float32)
        audio += speech_burst(4.0, [(0.2, 0.8)], amplitude=0.01)
        audio += speech_burst(4.0, [(0.8, 3.4)], amplitude=0.2)
        audio += speech_burst(4.0, [(3.4, 3.8)], amplitude=0.01)

        detected = detect_speech(audio, vad_config=ENABLED)
        extracted = extract_speech(audio, vad_config=ENABLED)

        self.assertTrue(detected.boundary_expanded)
        self.assertLessEqual(detected.start_sample, int(0.2 * SAMPLE_RATE))
        self.assertGreaterEqual(
            detected.end_sample, int(3.8 * SAMPLE_RATE)
        )
        self.assertFalse(extracted.applied)
        np.testing.assert_array_equal(extracted.audio, audio)

    def test_200ms_tonal_bursts_do_not_trigger_trimming(self):
        audio = white_noise(0.0005, 4.0, seed=41)
        times = np.arange(audio.size) / SAMPLE_RATE
        for start in (1.0, 1.8, 2.6):
            selected = (times >= start) & (times < start + 0.2)
            local = times[selected] - start
            audio[selected] += (
                0.1
                * np.sin(np.pi * local / 0.2) ** 2
                * np.sin(2.0 * np.pi * 900.0 * local)
            ).astype(np.float32)

        result = extract_speech(audio, vad_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertIn(
            result.status, {"low_confidence", "no_speech", "too_short"}
        )
        np.testing.assert_array_equal(result.audio, audio)

    def test_short_transients_do_not_trigger_trimming(self):
        audio = white_noise(0.0005, 4.0, seed=42)
        times = np.arange(audio.size) / SAMPLE_RATE
        for start in np.arange(0.4, 3.6, 0.5):
            selected = (times >= start) & (times < start + 0.04)
            local = times[selected] - start
            audio[selected] += (
                0.2
                * np.exp(-local / 0.005)
                * np.sin(2.0 * np.pi * 2000.0 * local)
            ).astype(np.float32)

        result = extract_speech(audio, vad_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertIn(result.status, {"no_speech", "too_short"})
        np.testing.assert_array_equal(result.audio, audio)

    def test_isolated_short_command_passes_through_unchanged(self):
        audio = white_noise(0.0005, 3.0, seed=43)
        audio = audio + speech_burst(
            3.0, [(1.0, 1.25)], amplitude=0.2
        )
        result = extract_speech(audio, vad_config=ENABLED)

        self.assertFalse(result.applied)
        self.assertEqual(result.status, "low_confidence")
        np.testing.assert_array_equal(result.audio, audio)


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


def noisy_speech_fixture():
    return (
        speech_burst(3.0, [(0.6, 1.8)], amplitude=0.2)
        + white_noise(0.02, 3.0, seed=7)
    ).astype(np.float32)


class SpeechToTextVadIntegrationTests(unittest.TestCase):
    def _stt_with_fake_model(self, text="hello axion"):
        stt = SpeechToText.__new__(SpeechToText)
        stt.model = FakeModel(text=text)
        return stt

    def test_transcribe_runs_denoise_vad_normalize_in_order(self):
        audio = noisy_speech_fixture()
        stt = self._stt_with_fake_model()
        calls = []
        lengths = {}

        def ordered_reduce(samples, **kwargs):
            calls.append("denoise")
            result = reduce_noise(samples, **kwargs)
            lengths["denoise"] = result.audio.size
            return result

        def ordered_vad(samples, **kwargs):
            calls.append("vad")
            lengths["vad_in"] = samples.size
            result = extract_speech(samples, **kwargs)
            lengths["vad_out"] = result.audio.size
            return result

        def ordered_normalize(samples, **kwargs):
            calls.append("normalize")
            return normalize_audio(samples, **kwargs)

        with patch("axion.stt.reduce_noise", side_effect=ordered_reduce), \
                patch("axion.stt.extract_speech", side_effect=ordered_vad), \
                patch("axion.stt.normalize_audio", side_effect=ordered_normalize), \
                patch("axion.stt.whisper.load_audio", return_value=audio.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_DENOISE": "1",
                    "AXION_AUDIO_VAD": "1",
                    "AXION_AUDIO_NORMALIZE": "1",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(calls, ["denoise", "vad", "normalize"])
        self.assertEqual(lengths["vad_in"], lengths["denoise"])
        self.assertLess(lengths["vad_out"], audio.size)
        self.assertEqual(len(stt.model.calls), 1)

        received, options = stt.model.calls[0]
        self.assertEqual(received.size, lengths["vad_out"])
        self.assertLessEqual(float(np.max(np.abs(received))), 0.95 + 1e-6)
        self.assertEqual(options["language"], "en")
        self.assertFalse(options["fp16"])
        self.assertEqual(options["temperature"], 0)
        self.assertEqual(options["no_speech_threshold"], 0.60)
        self.assertFalse(options["condition_on_previous_text"])
        self.assertEqual(options["compression_ratio_threshold"], 2.4)
        self.assertEqual(options["logprob_threshold"], -1.0)

    def test_vad_disabled_by_default_leaves_audio_untouched(self):
        audio = noisy_speech_fixture()
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=audio.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_NORMALIZE": "0",
                    "AXION_AUDIO_DENOISE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, audio)

    def test_vad_enabled_trims_before_whisper(self):
        audio = noisy_speech_fixture()
        expected = extract_speech(
            audio, sample_rate=16000, vad_config=ENABLED
        ).audio
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=audio.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_VAD": "1",
                    "AXION_AUDIO_NORMALIZE": "0",
                    "AXION_AUDIO_DENOISE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        received, _ = stt.model.calls[0]
        self.assertLess(received.size, audio.size)
        np.testing.assert_array_equal(received, expected)

    def test_vad_no_speech_policy_passes_audio_through(self):
        audio = white_noise(0.02, 3.0, seed=23)
        stt = self._stt_with_fake_model()

        with patch("axion.stt.whisper.load_audio", return_value=audio.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_VAD": "1",
                    "AXION_AUDIO_NORMALIZE": "0",
                    "AXION_AUDIO_DENOISE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(len(stt.model.calls), 1)
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, audio)

    def test_vad_failure_falls_back_to_original_audio(self):
        audio = speech_burst(3.0, [(0.6, 1.8)])
        stt = self._stt_with_fake_model()

        with patch(
            "axion.stt.extract_speech",
            side_effect=RuntimeError("simulated VAD failure"),
        ), patch("axion.stt.whisper.load_audio", return_value=audio.copy()), \
                patch("axion.stt.os.path.exists", return_value=True), \
                patch.dict(os.environ, {
                    "AXION_AUDIO_VAD": "1",
                    "AXION_AUDIO_NORMALIZE": "0",
                    "AXION_AUDIO_DENOISE": "0",
                }, clear=True):
            text = stt.transcribe("/tmp/fake-axion.wav")

        self.assertEqual(text, "hello axion")
        self.assertEqual(len(stt.model.calls), 1)
        received, _ = stt.model.calls[0]
        np.testing.assert_array_equal(received, audio)


if __name__ == "__main__":
    unittest.main()


