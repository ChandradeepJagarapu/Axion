import os
import array
import hashlib
import math
import subprocess
import tempfile
import threading
import wave

import whisper

from .audio_normalization import normalize_audio


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_NAME = "base"

POWERSHELL = (
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/"
    "powershell.exe"
)

RECORDER = (
    "C:\\AXION-voice-test\\record.py"
)

WINDOWS_AUDIO = (
    "C:\\Users\\Public\\axion_recording.wav"
)

WINDOWS_AUDIO_WSL = (
    "/mnt/c/Users/Public/"
    "axion_recording.wav"
)

STOP_FILE = (
    "/mnt/c/Users/Public/"
    "axion_stop_recording"
)


def _wav_diagnostics(path):
    """Read WAV metadata and PCM level metrics without printing audio."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        audio_bytes = wav.readframes(frames)

    samples = array.array("h", audio_bytes) if sample_width == 2 else array.array("b", audio_bytes)
    peak = max((abs(sample) for sample in samples), default=0)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) if samples else 0.0
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration": frames / sample_rate if sample_rate else 0.0,
        "size": os.path.getsize(path),
        "peak": peak,
        "rms": rms,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as audio:
        for chunk in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SpeechToText:

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self):

        print(
            f"Loading Whisper ({MODEL_NAME}, English)..."
    )

        self.model = whisper.load_model(
            MODEL_NAME
    )

        print(
            "Whisper ready."
    )

        self.recorder_process = None

        self.process_lock = (
            threading.Lock()
    )

        self.listen_lock = threading.Lock()
        self.last_record_status = None

    # =========================================================
    # RECORD
    # =========================================================

    def record(self, seconds=15):

        self.last_record_status = "recording"

        audio_file = os.path.join(
            tempfile.gettempdir(),
            "axion_recording.wav",
    )

        print(
            "\nStarting microphone...",
            flush=True,
    )

        # Delete both sides first so a failed run cannot reuse an earlier WAV.
        try:
            for old_audio in (WINDOWS_AUDIO_WSL, audio_file):
                if os.path.exists(old_audio):
                    os.remove(old_audio)
        except OSError as exc:
            print(f"AXION microphone error: cannot remove old recording: {exc}", flush=True)
            self.last_record_status = "error"
            return ""

        # -----------------------------------------------------
        # Remove old stop signal
        # -----------------------------------------------------

        try:

            if os.path.exists(
                STOP_FILE
        ):

                os.remove(
                    STOP_FILE
            )

        except OSError:

            pass

        # -----------------------------------------------------
        # Windows recorder command
        # -----------------------------------------------------

        mic_device = os.environ.get("AXION_MIC_DEVICE", "")
        if mic_device:
            command = (
                f'$env:AXION_MIC_DEVICE="{mic_device}"; python "{RECORDER}" '
                f'"{WINDOWS_AUDIO}" '
                f'{int(seconds)}'
        )
        else:
            command = (
                f'python "{RECORDER}" '
                f'"{WINDOWS_AUDIO}" '
                f'{int(seconds)}'
        )

        try:

            process = subprocess.Popen(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-Command",
                    command,
                ],

                # IMPORTANT:
                # Windows recorder must NOT
                # consume AXION's keyboard input.
                stdin=subprocess.DEVNULL,

                # Ensure environment variables set in WSL/
                # the parent process (e.g. AXION_MIC_DEVICE)
                # are transmitted to the Windows recorder.
                env=os.environ,
        )

            with self.process_lock:

                self.recorder_process = (
                    process
            )

            # -------------------------------------------------
            # Wait for recording to finish.
            # -------------------------------------------------

            try:
                process.wait(timeout=max(20, int(seconds) + 18))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print("AXION microphone error: Windows recorder timed out.", flush=True)
                self.last_record_status = "error"
                return ""

            if process.returncode != 0:
                print(
                    f"AXION microphone error: Windows recorder exited with code {process.returncode}.",
                    flush=True,
            )
                self.last_record_status = "error"
                return ""

        except Exception as exc:

            print(
                f"AXION microphone error: "
                f"{exc}",
                flush=True,
        )

            self.last_record_status = "error"
            return ""

        finally:

            with self.process_lock:

                self.recorder_process = None

        # -----------------------------------------------------
        # Validate the completed WAV before handing it to Whisper.
        # -----------------------------------------------------

        if not os.path.exists(
            WINDOWS_AUDIO_WSL
    ):

            print(
                "AXION microphone error: "
                "recording file was not created.",
                flush=True,
        )

            self.last_record_status = "error"
            return ""

        try:
            source_info = _wav_diagnostics(WINDOWS_AUDIO_WSL)

            if (
                not source_info["frames"]
                or source_info["channels"] < 1
                or source_info["sample_width"] != 2
                or source_info["sample_rate"] < 8000
        ):
                print("AXION microphone error: recording WAV is empty or unsupported.", flush=True)
                self.last_record_status = "invalid_audio"
                return ""

            if source_info["rms"] < 2.0 and source_info["peak"] < 32:
                print("AXION microphone error: recording is nearly silent; Whisper was not called.", flush=True)
                self.last_record_status = "silent"
                return ""
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            print(f"AXION microphone error: invalid or incomplete WAV: {exc}", flush=True)
            self.last_record_status = "invalid_audio"
            return ""

        print(
            "AXION audio capture: fresh WAV; "
            f"duration={source_info['duration']:.2f}s, "
            f"rate={source_info['sample_rate']}Hz, "
            f"channels={source_info['channels']}, "
            f"format=PCM-{source_info['sample_width'] * 8}-bit, "
            f"size={source_info['size']} bytes, "
            f"peak={source_info['peak']}, rms={source_info['rms']:.2f}.",
            flush=True,
    )

        # -----------------------------------------------------
        # Copy WAV to WSL
        # -----------------------------------------------------

        try:

            with open(
                WINDOWS_AUDIO_WSL,
                "rb",
        ) as source:

                with open(
                    audio_file,
                    "wb",
            ) as destination:

                    destination.write(
                        source.read()
                )

        except OSError as exc:

            print(
                f"AXION audio copy error: "
                f"{exc}",
                flush=True,
        )

            self.last_record_status = "error"
            return ""

        try:
            copied_info = _wav_diagnostics(audio_file)
            if (
                copied_info["size"] != source_info["size"]
                or _sha256(audio_file) != _sha256(WINDOWS_AUDIO_WSL)
        ):
                print("AXION audio copy error: WSL copy differs from the latest Windows WAV.", flush=True)
                self.last_record_status = "copy_error"
                return ""
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            print(f"AXION audio copy error: copied WAV is invalid: {exc}", flush=True)
            self.last_record_status = "copy_error"
            return ""

        print(
            f"AXION Whisper input: {audio_file} "
            f"(fresh copy verified; {copied_info['size']} bytes).",
            flush=True,
    )

        self.last_record_status = "ok"
        return audio_file

    # =========================================================
    # STOP RECORDING
    # =========================================================

    def stop_recording(self):

        """
        Signal Windows record.py to stop
        recording immediately.
        """

        try:

            with open(
                STOP_FILE,
                "w",
                encoding="utf-8",
        ) as file:

                file.write(
                    "stop"
            )

        except OSError as exc:

            print(
                f"AXION stop signal error: "
                f"{exc}",
                flush=True,
        )

    # =========================================================
    # TRANSCRIBE
    # =========================================================

    def transcribe(
        self,
        audio_file,
    ):

        if (
            not audio_file
            or not os.path.exists(
                audio_file
            )
        ):

            print("AXION transcription skipped: audio file is missing.", flush=True)

            return ""

        try:
            # Use and measure the exact 16 kHz mono float audio Whisper gets
            # after its ffmpeg WAV decoding and resampling step.
            audio = whisper.load_audio(audio_file)
        except Exception as exc:
            print(f"AXION audio decode error: {type(exc).__name__}: {exc}", flush=True)
            return ""

        # Normalize immediately before Whisper sees the decoded audio.
        # ``audio`` is the 16 kHz mono float32 array returned by load_audio.
        normalization = normalize_audio(audio, sample_rate=16000)
        if not normalization.usable:
            print(
                f"AXION audio normalization: unusable audio "
                f"({normalization.status}); Whisper was not called.",
                flush=True,
            )
            return ""

        audio = normalization.audio

        if normalization.applied:
            print(
                f"AXION audio normalization: {normalization.status}, "
                f"gain={normalization.gain:.3f}x "
                f"({normalization.gain_db:+.1f} dB), "
                f"peak={normalization.input_peak:.5f} -> "
                f"{normalization.output_peak:.5f}, "
                f"rms={normalization.input_rms:.5f} -> "
                f"{normalization.output_rms:.5f}.",
                flush=True,
            )
        elif normalization.status in {"silent", "very_low_volume"}:
            print(
                f"AXION audio normalization: skipped "
                f"({normalization.status}).",
                flush=True,
            )

        converted_peak = normalization.output_peak
        converted_rms = normalization.output_rms
        print(
            f"AXION Whisper audio: path={audio_file}, "
            f"duration={len(audio) / 16000:.2f}s, rate=16000Hz, channels=1, "
            f"format=float32, peak={converted_peak:.5f}, rms={converted_rms:.5f}.",
            flush=True,
        )
        if not len(audio) or converted_peak < 0.0001:
            print("AXION audio decode error: converted audio is empty or nearly silent.", flush=True)
            return ""

        try:
            result = self.model.transcribe(
                audio,
                language="en",
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.60,
                temperature=0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
            )

        except Exception as exc:

            print(
                f"AXION transcription error: "
                f"{exc}",
                flush=True,
            )

            return ""

        raw_text = str(result.get("text", "")).strip()
        segments = result.get("segments", [])
        print(
            f"AXION Whisper raw result: text_chars={len(raw_text)}, "
            f"segments={len(segments)}.",
            flush=True,
        )

        # -----------------------------------------------------
        # Filter Whisper segments
        # -----------------------------------------------------

        valid_segments = []
        rejected_segments = 0

        for segment in segments:

            text = segment.get(
                "text",
                "",
            ).strip()

            no_speech_prob = (
                segment.get(
                    "no_speech_prob",
                    0.0,
                )
            )

            if not text:
                rejected_segments += 1
                continue

            if no_speech_prob > 0.60:
                rejected_segments += 1
                continue

            valid_segments.append(
                text
            )

        text = " ".join(
            valid_segments
        ).strip()

        if raw_text and not text:
            print(
                "AXION transcription cleanup discarded Whisper's text; "
                "using the model result.",
                flush=True,
            )
            text = raw_text
        elif rejected_segments:
            print(
                f"AXION transcription cleanup skipped {rejected_segments} "
                "low-confidence/empty segments.",
                flush=True,
            )

        if not raw_text and not text:
            print("AXION Whisper returned no text.", flush=True)

        # -----------------------------------------------------
        # Very short transcription
        # -----------------------------------------------------

        if len(text) < 2:

            print("AXION transcription cleanup rejected text shorter than two characters.", flush=True)

            return ""

        # -----------------------------------------------------
        # Common Whisper hallucinations
        # -----------------------------------------------------

        hallucinations = {
            "thank you",
            "thanks for watching",
            "subscribe",
            "you",
            "bye",
            "goodbye",
            "okay",
            "ok",
        }

        normalized = (
            text.lower()
            .strip(" .!?")
        )

        if normalized in hallucinations:

            print("AXION transcription cleanup rejected a known short hallucination.", flush=True)

            return ""

        return text

    # =========================================================
    # LISTEN
    # =========================================================

    def listen(self, seconds=15):

        # Never allow two workers to own the microphone at once.
        if not self.listen_lock.acquire(blocking=False):
            return ""

        try:
            return self._listen_once(seconds)
        finally:
            self.listen_lock.release()

    def _listen_once(self, seconds):

        audio_file = self.record(seconds=seconds)

        if not audio_file:

            return ""

        try:

            return self.transcribe(
                audio_file
        )

        finally:

            if os.path.exists(
                audio_file
        ):

                try:

                    os.remove(
                        audio_file
                )

                except OSError:

                    pass
