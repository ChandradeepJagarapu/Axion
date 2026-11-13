import os
import queue
import subprocess
import tempfile
import threading
import time
import wave

from piper import PiperVoice


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_PATH = os.path.expanduser(
    "~/AXION/voices/en_US-lessac-high.onnx"
)

WINDOWS_POWERSHELL = (
    "/mnt/c/Windows/System32/WindowsPowerShell/"
    "v1.0/powershell.exe"
)

WINDOWS_WAV = (
    "/mnt/c/Users/Public/axion_tts.wav"
)

WINDOWS_WAV_PATH = (
    r"C:\Users\Public\axion_tts.wav"
)


# =============================================================
# TEXT TO SPEECH
# =============================================================

class TextToSpeech:

    def __init__(self):

        # -----------------------------------------------------
        # Check model
        # -----------------------------------------------------

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Piper voice model not found: {MODEL_PATH}"
            )

        # -----------------------------------------------------
        # Load Piper once
        # -----------------------------------------------------

        print(
            "AXION: Loading voice model...",
            flush=True,
        )

        self.voice = PiperVoice.load(
            MODEL_PATH
        )

        print(
            "AXION: Voice model loaded.",
            flush=True,
        )

        # -----------------------------------------------------
        # Stream state
        # -----------------------------------------------------

        self.buffer = ""

        self.buffer_lock = threading.Lock()

        # -----------------------------------------------------
        # Queues
        # -----------------------------------------------------

        self.sentence_queue = queue.Queue()

        self.audio_queue = queue.Queue()

        # -----------------------------------------------------
        # Stop state
        # -----------------------------------------------------

        self.stop_event = threading.Event()

        # -----------------------------------------------------
        # Playback state
        # -----------------------------------------------------

        self.playback_process = None

        self.playback_lock = threading.Lock()

        self.audio_playback_lock = threading.Lock()

        # -----------------------------------------------------
        # Stream state
        # -----------------------------------------------------

        self.stream_lock = threading.Lock()

        self.stream_active = False

        # -----------------------------------------------------
        # Generator worker
        # -----------------------------------------------------

        self.generator_thread = threading.Thread(
            target=self._generator_worker,
            daemon=True,
            name="AXION-TTS-Generator",
        )

        # -----------------------------------------------------
        # Player worker
        # -----------------------------------------------------

        self.player_thread = threading.Thread(
            target=self._player_worker,
            daemon=True,
            name="AXION-TTS-Player",
        )

        self.generator_thread.start()

        self.player_thread.start()

    # =========================================================
    # START STREAM
    # =========================================================

    def start_stream(self) -> None:

        # -----------------------------------------------------
        # Make sure old state is cleared.
        # -----------------------------------------------------

        self.stop_event.clear()

        with self.buffer_lock:

            self.buffer = ""

        with self.stream_lock:

            self.stream_active = True

        # -----------------------------------------------------
        # Remove stale queued items.
        # -----------------------------------------------------

        self._clear_sentence_queue()

        self._clear_audio_queue()

    # =========================================================
    # GENERATE AUDIO
    # =========================================================

    def _generate_audio(
        self,
        text: str,
    ) -> str:

        text = text.strip()

        if not text:

            raise ValueError(
                "Cannot generate audio from empty text."
            )

        linux_wav = tempfile.mktemp(
            prefix="axion_tts_",
            suffix=".wav",
        )

        try:

            with wave.open(
                linux_wav,
                "wb",
            ) as wav_file:

                self.voice.synthesize_wav(
                    text,
                    wav_file,
                )

            return linux_wav

        except Exception:

            if os.path.exists(linux_wav):

                os.remove(
                    linux_wav
                )

            raise

    # =========================================================
    # GENERATOR WORKER
    # =========================================================

    def _generator_worker(self):

        while True:

            sentence = self.sentence_queue.get()

            # -------------------------------------------------
            # Shutdown signal
            # -------------------------------------------------

            if sentence is None:

                self.sentence_queue.task_done()

                break

            try:

                if self.stop_event.is_set():

                    continue

                sentence = sentence.strip()

                if not sentence:

                    continue

                # -------------------------------------------------
                # Generate WAV
                # -------------------------------------------------

                wav_file = self._generate_audio(
                    sentence
                )

                # -------------------------------------------------
                # Queue audio
                # -------------------------------------------------

                if not self.stop_event.is_set():

                    self.audio_queue.put(
                        wav_file
                    )

                else:

                    if os.path.exists(wav_file):

                        os.remove(
                            wav_file
                        )

            except Exception as error:

                if not self.stop_event.is_set():

                    print(
                        "\nAXION TTS generation error: "
                        f"{error}\n",
                        flush=True,
                    )

            finally:

                self.sentence_queue.task_done()

    # =========================================================
    # PLAY AUDIO
    # =========================================================

    def _play_audio(
        self,
        linux_wav: str,
    ) -> None:

        with self.audio_playback_lock:

            self._play_audio_unlocked(
                linux_wav
            )

    # =========================================================
    # PLAY AUDIO - INTERNAL
    # =========================================================

    def _play_audio_unlocked(
        self,
        linux_wav: str,
    ) -> None:

        if self.stop_event.is_set():

            if os.path.exists(linux_wav):

                os.remove(
                    linux_wav
                )

            return

        try:

            # -------------------------------------------------
            # Copy WAV to Windows
            # -------------------------------------------------

            with open(
                linux_wav,
                "rb",
            ) as source:

                with open(
                    WINDOWS_WAV,
                    "wb",
                ) as destination:

                    while True:

                        chunk = source.read(
                            1024 * 64
                        )

                        if not chunk:

                            break

                        destination.write(
                            chunk
                        )

            # -------------------------------------------------
            # Check stop before starting playback
            # -------------------------------------------------

            if self.stop_event.is_set():

                return

            # -------------------------------------------------
            # PowerShell SoundPlayer
            # -------------------------------------------------

            command = (
                "$player = New-Object "
                "Media.SoundPlayer "
                f"'{WINDOWS_WAV_PATH}'; "
                "$player.PlaySync()"
            )

            process = subprocess.Popen(
                [
                    WINDOWS_POWERSHELL,
                    "-NoProfile",
                    "-Command",
                    command,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )

            # -------------------------------------------------
            # Store current playback process
            # -------------------------------------------------

            with self.playback_lock:

                self.playback_process = process

            try:

                # -------------------------------------------------
                # Wait for playback to finish. A timeout is only a
                # polling interval; it must not release the playback
                # lock while SoundPlayer is still using WINDOWS_WAV.
                # -------------------------------------------------

                while process.poll() is None:

                    if self.stop_event.is_set():

                        try:

                            process.terminate()

                        except Exception:

                            pass

                        try:

                            process.wait(timeout=0.5)

                        except subprocess.TimeoutExpired:

                            try:

                                process.kill()

                            except Exception:

                                pass

                            process.wait()

                        break

                    try:

                        process.wait(timeout=0.05)

                    except subprocess.TimeoutExpired:

                        # Still playing; keep ownership of the shared
                        # Windows WAV and playback lock.
                        continue

            finally:

                with self.playback_lock:

                    if (
                        self.playback_process
                        is process
                    ):

                        self.playback_process = None

        finally:

            # -------------------------------------------------
            # Remove Linux temporary WAV
            # -------------------------------------------------

            if os.path.exists(linux_wav):

                try:

                    os.remove(
                        linux_wav
                    )

                except Exception:

                    pass

    # =========================================================
    # PLAYER WORKER
    # =========================================================

    def _player_worker(self):

        while True:

            wav_file = self.audio_queue.get()

            # -------------------------------------------------
            # Shutdown signal
            # -------------------------------------------------

            if wav_file is None:

                self.audio_queue.task_done()

                break

            try:

                if not self.stop_event.is_set():

                    self._play_audio(
                        wav_file
                    )

                else:

                    if os.path.exists(wav_file):

                        os.remove(
                            wav_file
                        )

            except Exception as error:

                if not self.stop_event.is_set():

                    print(
                        "\nAXION TTS playback error: "
                        f"{error}\n",
                        flush=True,
                    )

            finally:

                self.audio_queue.task_done()

    # =========================================================
    # NORMAL SPEAK
    # =========================================================

    def speak(
        self,
        text: str,
    ) -> None:

        text = text.strip()

        if not text:
            return

        if self.stop_event.is_set():
            return

        # Generate audio first.
        wav_file = self._generate_audio(text)

        if self.stop_event.is_set():

            if os.path.exists(wav_file):

                os.remove(
                    wav_file
                )

            return

        # Send it through the same single playback queue
        # used by streaming TTS.
        self.audio_queue.put(
            wav_file
        )

        # Wait until this audio has finished playing.
        self.audio_queue.join()

    # =========================================================
    # STREAM SPEAK
    # =========================================================

    def speak_stream(
        self,
        text: str,
    ) -> None:

        if not text:

            return

        if self.stop_event.is_set():

            return

        # -----------------------------------------------------
        # Add new model chunk to buffer.
        # -----------------------------------------------------

        with self.buffer_lock:

            self.buffer += text

            while True:

                boundary = self._find_sentence_boundary(
                    self.buffer
                )

                if boundary == -1:

                    # -------------------------------------------------
                    # Don't speak tiny fragments.
                    # -------------------------------------------------

                    if len(self.buffer) > 180:

                        boundary = self._find_soft_boundary(
                            self.buffer
                        )

                    else:

                        break

                if boundary == -1:

                    break

                sentence = (
                    self.buffer[
                        :boundary
                    ].strip()
                )

                self.buffer = (
                    self.buffer[
                        boundary:
                    ]
                )

                if not sentence:

                    continue

                if self.stop_event.is_set():

                    break

                self.sentence_queue.put(
                    sentence
                )

    # =========================================================
    # FIND SENTENCE BOUNDARY
    # =========================================================

    def _find_sentence_boundary(
        self,
        text: str,
    ) -> int:

        # -----------------------------------------------------
        # We wait for real sentence endings.
        # -----------------------------------------------------

        markers = (
            ".",
            "!",
            "?",
        )

        positions = []

        for marker in markers:

            position = text.find(
                marker
            )

            if position != -1:

                positions.append(
                    position + 1
                )

        if not positions:

            return -1

        return min(
            positions
        )

    # =========================================================
    # FIND SOFT BOUNDARY
    # =========================================================

    def _find_soft_boundary(
        self,
        text: str,
    ) -> int:

        # -----------------------------------------------------
        # If the model produces a very long sentence,
        # split at a natural comma/space rather than
        # generating one huge audio file.
        # -----------------------------------------------------

        target = 140

        comma_position = text.find(
            ",",
            target // 2,
        )

        if comma_position != -1:

            return comma_position + 1

        space_position = text.find(
            " ",
            target,
        )

        if space_position != -1:

            return space_position + 1

        return -1

    # =========================================================
    # FINISH STREAM
    # =========================================================

    def finish_stream(self) -> None:

        if self.stop_event.is_set():

            with self.buffer_lock:

                self.buffer = ""

            with self.stream_lock:

                self.stream_active = False

            return

        # -----------------------------------------------------
        # Send remaining text.
        # -----------------------------------------------------

        with self.buffer_lock:

            remaining = self.buffer.strip()

            self.buffer = ""

        if remaining:

            self.sentence_queue.put(
                remaining
            )

        with self.stream_lock:

            self.stream_active = False

    # =========================================================
    # WAIT UNTIL FINISHED
    # =========================================================

    def wait_until_finished(self) -> None:

        # -----------------------------------------------------
        # Wait until every sentence has been converted
        # to audio.
        # -----------------------------------------------------

        self.sentence_queue.join()

        # -----------------------------------------------------
        # Wait until every generated WAV has played.
        # -----------------------------------------------------

        self.audio_queue.join()

        # -----------------------------------------------------
        # Wait for current playback process if necessary.
        # -----------------------------------------------------

        while True:

            with self.playback_lock:

                process = self.playback_process

            if process is None:

                break

            if process.poll() is not None:

                break

            time.sleep(0.02)

    # =========================================================
    # CLEAR SENTENCE QUEUE
    # =========================================================

    def _clear_sentence_queue(self):

        while True:

            try:

                item = (
                    self.sentence_queue.get_nowait()
                )

            except queue.Empty:

                break

            try:

                if item is not None:

                    pass

            finally:

                self.sentence_queue.task_done()

    # =========================================================
    # CLEAR AUDIO QUEUE
    # =========================================================

    def _clear_audio_queue(self):

        while True:

            try:

                wav_file = (
                    self.audio_queue.get_nowait()
                )

            except queue.Empty:

                break

            try:

                if (
                    wav_file
                    and os.path.exists(wav_file)
                ):

                    try:

                        os.remove(
                            wav_file
                        )

                    except Exception:

                        pass

            finally:

                self.audio_queue.task_done()

    # =========================================================
    # STOP
    # =========================================================

    def stop(self) -> None:

        # -----------------------------------------------------
        # Set stop immediately.
        # -----------------------------------------------------

        self.stop_event.set()

        # -----------------------------------------------------
        # Clear text buffer.
        # -----------------------------------------------------

        with self.buffer_lock:

            self.buffer = ""

        # -----------------------------------------------------
        # Clear waiting sentences.
        # -----------------------------------------------------

        self._clear_sentence_queue()

        # -----------------------------------------------------
        # Clear generated audio.
        # -----------------------------------------------------

        self._clear_audio_queue()

        # -----------------------------------------------------
        # Stop currently playing PowerShell process.
        # -----------------------------------------------------

        with self.playback_lock:

            process = self.playback_process

        if process is not None:

            try:

                process.terminate()

            except Exception:

                pass

            try:

                process.kill()

            except Exception:

                pass

    # =========================================================
    # SHUTDOWN
    # =========================================================

    def shutdown(self) -> None:

        # -----------------------------------------------------
        # Stop current playback.
        # -----------------------------------------------------

        self.stop()

        # -----------------------------------------------------
        # Stop generator worker.
        # -----------------------------------------------------

        try:

            self.sentence_queue.put(
                None
            )

        except Exception:

            pass

        # -----------------------------------------------------
        # Stop player worker.
        # -----------------------------------------------------

        try:

            self.audio_queue.put(
                None
            )

        except Exception:

            pass
