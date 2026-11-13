import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from tests.diagnose_wake_audio import decoded_diagnostics, pcm_diagnostics


class AudioDiagnosticTests(unittest.TestCase):
    def test_pcm_diagnostics_report_format_levels_and_full_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            samples = np.array([0, 1000, -2000, 32767], dtype="<i2")
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(samples.tobytes())
            info = pcm_diagnostics(path)
        self.assertEqual(info["sample_rate"], 48000)
        self.assertAlmostEqual(info["duration_seconds"], 4 / 48000)
        self.assertAlmostEqual(info["peak_normalized"], 1.0)
        self.assertGreater(info["near_full_scale_fraction"], 0)

    def test_decoded_diagnostics_report_whisper_audio_levels(self):
        info = decoded_diagnostics(np.array([0.0, 0.01, -0.02], dtype=np.float32))
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(info["samples"], 3)
        self.assertAlmostEqual(info["peak"], 0.02)
        self.assertEqual(info["non_finite_samples"], 0)


if __name__ == "__main__":
    unittest.main()
