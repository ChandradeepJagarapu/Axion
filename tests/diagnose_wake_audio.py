"""Record and compare the exact AXION Whisper settings with alternative decoding.

Run from the project root with ``python tests/diagnose_wake_audio.py``.
The captured WAV is retained at ``axion_whisper_diagnostic.wav``.
"""

import argparse
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import whisper

from axion.stt import MODEL_NAME, SpeechToText


def pcm_diagnostics(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        limit = 32767
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128)
        limit = 127
    else:
        samples = np.array([], dtype=np.float64)
        limit = 0
    if channels > 1 and samples.size:
        samples = samples.reshape(-1, channels)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    near_full = float(np.mean(np.abs(samples) >= limit * 0.999)) if samples.size and limit else 0.0
    dc = float(np.mean(samples)) if samples.size else 0.0
    return {
        "channels": channels, "sample_width_bytes": width, "sample_rate": rate,
        "frames": frames, "duration_seconds": frames / rate if rate else 0,
        "peak_pcm": peak, "peak_normalized": peak / limit if limit else 0,
        "rms_pcm": rms, "rms_normalized": rms / limit if limit else 0,
        "crest_factor": peak / rms if rms else 0,
        "near_full_scale_fraction": near_full, "dc_offset_pcm": dc,
        "size_bytes": Path(path).stat().st_size,
    }


def decoded_diagnostics(audio):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    return {
        "samples": int(audio.size), "duration_seconds": audio.size / 16000,
        "sample_rate": 16000, "channels": 1, "dtype": str(audio.dtype),
        "peak": peak, "rms": rms,
        "clipped_fraction_at_0.999": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0,
        "dc_offset": float(np.mean(audio)) if audio.size else 0.0,
        "non_finite_samples": int(np.size(audio) - np.count_nonzero(np.isfinite(audio))),
    }


def print_result(label, model, audio, options):
    print(f"\n===== {label} =====")
    print(f"Options: {json.dumps(options, sort_keys=True)}")
    result = model.transcribe(audio, **options)
    print(f"Complete raw transcription: {result.get('text', '')!r}")
    print("Segments:")
    for segment in result.get("segments", []):
        print(json.dumps({key: segment.get(key) for key in (
            "id", "start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio"
        )}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=8, choices=range(5, 11),
                        help="capture duration (5 to 10 seconds; default: 8)")
    args = parser.parse_args()

    print("Speak naturally during the recording: say ‘Hey AXION’ OR any normal sentence, "
          "for example ‘Please tell me what the weather will be like tomorrow.’")
    input("Press Enter when ready to start recording... ")
    stt = SpeechToText()
    audio_file = stt.record(seconds=args.seconds)
    if not audio_file:
        print(f"Recording failed: {stt.last_record_status}")
        return 1

    source = Path(audio_file)
    retained = Path.cwd() / "axion_whisper_diagnostic.wav"
    shutil.copy2(source, retained)
    print(f"\nCaptured WAV retained at: {retained}")
    try:
        print("PCM WAV diagnostics:")
        print(json.dumps(pcm_diagnostics(source), indent=2))
        audio = whisper.load_audio(str(source))
        print("Whisper-decoded/resampled audio diagnostics:")
        print(json.dumps(decoded_diagnostics(audio), indent=2))
        common = {"language": "en", "fp16": False}
        current = {
            **common, "condition_on_previous_text": False, "no_speech_threshold": 0.60,
            "temperature": 0, "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
        }
        alternatives = [
            ("Current AXION settings", current),
            ("Beam search; thresholds disabled", {
                **common, "condition_on_previous_text": False, "temperature": 0,
                "beam_size": 5, "no_speech_threshold": None,
                "compression_ratio_threshold": None, "logprob_threshold": None,
            }),
            ("Beam search; previous-text conditioning enabled", {
                **common, "condition_on_previous_text": True, "temperature": 0,
                "beam_size": 5, "no_speech_threshold": 0.60,
                "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
            }),
        ]
        for label, options in alternatives:
            print_result(label, stt.model, audio, options)
        print("\nInterpretation: clipping flags indicate samples at digital full scale. "
              "Low peak/RMS alone does not prove a fault; review the retained WAV for "
              "noise, distortion, dropouts, and whether speech is clearly audible.")
        print(f"Model: {MODEL_NAME} English. All three runs used the same decoded samples.")
        return 0
    finally:
        source.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
