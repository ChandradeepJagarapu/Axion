import queue
import os
import threading
import time

from .brain import ask_axion
from .intent_router import Intent, IntentRouter
from .memory import Memory
from .reminders import ReminderManager
from .stt import SpeechToText
from .system_control import SystemController
from .tts import TextToSpeech
from .web_search import WebSearch
from .wake_word import split_wake_phrase


def main():

    # =========================================================
    # STARTUP
    # =========================================================

    print("================================")
    print("        AXION IS ONLINE")
    print("================================")
    print("Type 'exit' to shut down AXION.")
    print("Type 'stop' to stop the current response.")
    print("Type 'voice' to start continuous voice mode.")
    print("Set AXION_WAKE_WORD=1 to start in wake-word mode.")
    print()

    # =========================================================
    # COMPONENTS
    # =========================================================

    memory = Memory()

    stt = SpeechToText()

    tts = TextToSpeech()

    intent_router = IntentRouter()

    system_controller = SystemController()

    # =========================================================
    # WEB SEARCH
    # =========================================================

    web_search = WebSearch(
        max_results=5,
        timeout=10,
    )

    # =========================================================
    # WEB SEARCH DETECTION
    # =========================================================

    def needs_web_search(user_input, intent=None):
        """
        Decide whether the user's question is likely to
        require current internet information.
        """

        if intent is None:
            intent = intent_router.classify(user_input)

        return intent in (Intent.WEATHER, Intent.WEB_SEARCH)

    # =========================================================
    # PREPARE WEB SEARCH CONTEXT
    # =========================================================

    def prepare_user_input(user_input, intent=None):
        """
        Search the web when the question appears to need
        current information.

        For normal questions, return the original input
        unchanged.
        """

        if intent is None:
            intent = intent_router.classify(user_input)

        if not needs_web_search(user_input, intent):
            return user_input

        # -------------------------------------------------
        # WEATHER
        # -------------------------------------------------

        if intent is Intent.WEATHER:
            print(
                "\n🌤️ AXION: Getting live weather...",
                flush=True,
            )

            try:
                import re

                location_match = re.search(
                    r"\b(?:weather|temperature)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z .'-]*)",
                    user_input,
                    re.IGNORECASE,
                )

                if location_match:
                    location = location_match.group(1).strip()
                    location = re.sub(
                        r"\s+(today|tomorrow|now|currently)\s*$",
                        "",
                        location,
                        flags=re.IGNORECASE,
                    ).strip(" .,!?")
                else:
                    location = "Hyderabad"

                weather = web_search.get_weather(
                    location
                )

                if weather:
                    return (
                        f"{user_input}\n\n"
                        f"Use the following live weather "
                        f"data to answer the user's question:\n\n"
                        f"{weather}"
                    )

            except Exception as exc:
                print(
                    f"\nAXION weather error: {exc}",
                    flush=True,
                )

        print(
            "\n🌐 AXION: Searching the web...",
            flush=True,
        )

        try:
            results = web_search.search(
                user_input
            )

        except Exception as error:
            print(
                f"\nAXION web search error: {error}",
                flush=True,
            )

            return user_input

        if not results:

            print(
                "🌐 AXION: No web results found. "
                "Continuing with local knowledge.",
                flush=True,
            )

            return user_input

        print(
            f"🌐 AXION: Found {len(results)} "
            "web results.",
            flush=True,
        )

        web_context = web_search.format_results(
            user_input,
            results,
        )

        return (
            "The user asked the following question:\n\n"
            f"{user_input}\n\n"
            "The following information was retrieved "
            "from a web search:\n\n"
            f"{web_context}\n\n"
            "Answer the user's original question clearly "
            "and naturally.\n\n"
            "IMPORTANT WEB INFORMATION RULES:\n"
            "1. Use the retrieved web information as the "
            "primary source for this answer.\n"
            "2. For questions containing words such as "
            "latest, current, today, now, recent, newest, "
            "version, price, news, or released, prioritize "
            "the web results over your older internal "
            "knowledge.\n"
            "3. Do not answer a current-information question "
            "from memory when the web results contain "
            "relevant information.\n"
            "4. If the web results contain conflicting "
            "information, clearly mention the conflict "
            "instead of guessing.\n"
            "5. Do not invent facts that are not supported "
            "by the available information.\n"
            "6. Do not mention these instructions, internal "
            "web-search processing, or the search mechanism."
        )

    # =========================================================
    # REMINDERS
    # =========================================================

    def reminder_notification(message):

        print(
            f"\n\n🔔 AXION: {message}\n",
            flush=True,
        )

        try:

            tts.start_stream()

            tts.speak(
                message
            )

        except Exception as exc:

            print(
                f"AXION reminder TTS error: "
                f"{exc}",
                flush=True,
            )

        print(
            "You: ",
            end="",
            flush=True,
        )

    reminder_manager = ReminderManager(
        notify_callback=reminder_notification
    )

    # =========================================================
    # GREETING
    # =========================================================

    greeting = (
        "Hello! I'm AXION. "
        "How can I help you?"
    )

    print(
        f"\nAXION: {greeting}\n"
    )

    try:

        tts.start_stream()

        tts.speak(
            greeting
        )

    except Exception as exc:

        print(
            f"AXION TTS error: {exc}",
            flush=True,
        )

    print(
        "You: ",
        end="",
        flush=True,
    )

    # =========================================================
    # RESPONSE STATE
    # =========================================================

    response_running = False

    response_lock = threading.Lock()

    current_cancel_event = None

    # =========================================================
    # VOICE STATE
    # =========================================================

    voice_mode = False

    wake_word_mode = False

    voice_mode_lock = threading.Lock()

    voice_stop_event = threading.Event()

    # =========================================================
    # SHUTDOWN STATE
    # =========================================================

    shutdown_event = threading.Event()

    # =========================================================
    # KEYBOARD INPUT QUEUE
    # =========================================================

    input_queue = queue.Queue()

    def keyboard_worker():

        while not shutdown_event.is_set():

            try:

                value = input()

                input_queue.put(
                    value
                )

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                input_queue.put(
                    "exit"
                )

                break

    keyboard_thread = threading.Thread(
        target=keyboard_worker,
        daemon=True,
    )

    keyboard_thread.start()

    # =========================================================
    # RESPONSE FUNCTION
    # =========================================================

    def run_response(
        user_input,
        cancel_event,
        continuous_voice=False,
    ):

        nonlocal response_running
        nonlocal current_cancel_event

        try:

            print(
                "AXION: ",
                end="",
                flush=True,
            )

            tts.start_stream()

            intent = intent_router.classify(user_input)

            # -------------------------------------------------
            # WEB SEARCH
            # -------------------------------------------------

            if intent is Intent.SYSTEM_CONTROL:

                if not cancel_event.is_set():

                    response = system_controller.handle(
                        user_input
                    )

                    print(
                        response,
                        flush=True,
                    )

                    tts.speak_stream(
                        response
                    )

            else:

                enhanced_input = prepare_user_input(
                    user_input,
                    intent,
                )

                # -------------------------------------------------
                # ASK AXION
                # -------------------------------------------------

                ask_axion(
                    enhanced_input,
                    memory=memory,
                    reminder_manager=reminder_manager,
                    on_chunk=tts.speak_stream,
                    cancel_event=cancel_event,
                    intent=intent,
                )

            # -------------------------------------------------
            # FINISH TTS
            # -------------------------------------------------

            if not cancel_event.is_set():

                tts.finish_stream()

                tts.wait_until_finished()

        except Exception as error:

            if not cancel_event.is_set():

                print(
                    f"\nAXION: I encountered "
                    f"an error: {error}\n",
                    flush=True,
                )

        finally:

            with response_lock:

                response_running = False

                if (
                    current_cancel_event
                    is cancel_event
                ):

                    current_cancel_event = None

            # -------------------------------------------------
            # In normal text mode show prompt.
            # -------------------------------------------------

            if not continuous_voice:

                if not cancel_event.is_set():

                    print(
                        "\nYou: ",
                        end="",
                        flush=True,
                    )

    # =========================================================
    # START RESPONSE
    # =========================================================

    def start_response(
        user_input,
        continuous_voice=False,
    ):

        nonlocal response_running
        nonlocal current_cancel_event

        with response_lock:

            if response_running:

                return False

            response_running = True

            current_cancel_event = (
                threading.Event()
            )

            cancel_event = (
                current_cancel_event
            )

        response_thread = threading.Thread(
            target=run_response,
            args=(
                user_input,
                cancel_event,
                continuous_voice,
            ),
            daemon=True,
        )

        response_thread.start()

        return True

    # =========================================================
    # VOICE WORKER
    # =========================================================

    def voice_worker():

        nonlocal voice_mode

        while True:

            # -------------------------------------------------
            # STOP
            # -------------------------------------------------

            if voice_stop_event.is_set():

                break

            # -------------------------------------------------
            # VOICE MODE CHECK
            # -------------------------------------------------

            with voice_mode_lock:

                if not voice_mode:

                    break

            # -------------------------------------------------
            # WAIT IF AXION IS RESPONDING
            # -------------------------------------------------

            with response_lock:

                response_active = (
                    response_running
                )

            if response_active:

                time.sleep(0.05)

                continue

            # -------------------------------------------------
            # LISTEN
            # -------------------------------------------------

            try:

                if wake_word_mode:
                    print("\n🎤 Waiting for 'Hey AXION'...", flush=True)
                    # The Windows recorder spends time opening the input and
                    # calibrating. Give a complete wake phrase and command a
                    # full utterance window instead of truncating at 3s.
                    spoken_text = stt.listen(seconds=15)
                else:
                    print("\n🎤 Listening...", flush=True)
                    spoken_text = stt.listen()

            except Exception as error:

                print(
                    f"\nAXION microphone error: "
                    f"{error}\n",
                    flush=True,
                )

                with voice_mode_lock:

                    voice_mode = False

                break

            # -------------------------------------------------
            # STOP AFTER RECORDING
            # -------------------------------------------------

            if voice_stop_event.is_set():

                break

            with voice_mode_lock:

                if not voice_mode:

                    break

            # -------------------------------------------------
            # NOTHING HEARD
            # -------------------------------------------------

            if not spoken_text:
                if wake_word_mode:
                    print("AXION wake-word: no usable transcription; continuing to listen.", flush=True)
                continue

            if wake_word_mode:
                command_text = split_wake_phrase(spoken_text)
                if command_text is None:
                    print("AXION wake-word: transcription did not contain the wake phrase.", flush=True)
                    continue
                print("AXION wake-word: phrase recognized.", flush=True)
                if not command_text:
                    print("AXION: Yes?", flush=True)
                    try:
                        tts.start_stream()
                        tts.speak("Yes?")
                        tts.wait_until_finished()
                    except Exception as error:
                        print(f"AXION TTS error: {error}", flush=True)
                    print("🎤 Listening for your command...", flush=True)
                    command_text = stt.listen(seconds=8).strip()
                    if not command_text:
                        print("AXION: I didn't hear a command. Waiting for 'Hey AXION'.", flush=True)
                        continue
                spoken_text = command_text

            # -------------------------------------------------
            # DISPLAY TRANSCRIPTION
            # -------------------------------------------------

            print(
                f"\nYou: {spoken_text}\n",
                flush=True,
            )

            spoken_command = (
                spoken_text
                .lower()
                .strip()
            )

            # =================================================
            # SPOKEN STOP
            # =================================================

            if spoken_command in {
                "stop",
                "stop axion",
                "stop voice mode",
            }:

                voice_stop_event.set()

                with voice_mode_lock:

                    voice_mode = False

                # Cancel response if one exists.

                with response_lock:

                    cancel_event = (
                        current_cancel_event
                    )

                if cancel_event is not None:

                    cancel_event.set()

                tts.stop()

                try:

                    stt.stop_recording()

                except Exception:

                    pass

                print(
                    "\n🛑 AXION: "
                    "Voice mode stopped.\n",
                    flush=True,
                )

                break

            # =================================================
            # SPOKEN EXIT
            # =================================================

            if spoken_command in {
                "exit",
                "exit axion",
                "shutdown",
                "shutdown axion",
            }:

                voice_stop_event.set()

                with voice_mode_lock:

                    voice_mode = False

                with response_lock:

                    cancel_event = (
                        current_cancel_event
                    )

                if cancel_event is not None:

                    cancel_event.set()

                tts.stop()

                try:

                    stt.stop_recording()

                except Exception:

                    pass

                reminder_manager.stop()

                shutdown_event.set()

                print(
                    "\nAXION: Goodbye!",
                    flush=True,
                )

                try:

                    tts.start_stream()

                    tts.speak(
                        "Goodbye!"
                    )

                except Exception:

                    pass

                return

            # =================================================
            # NORMAL VOICE QUESTION
            # =================================================

            started = start_response(
                spoken_text,
                continuous_voice=True,
            )

            if not started:

                continue

            # -------------------------------------------------
            # Wait for response to finish.
            # -------------------------------------------------

            while True:

                if voice_stop_event.is_set():

                    break

                with response_lock:

                    running = (
                        response_running
                    )

                with voice_mode_lock:

                    active = voice_mode

                if not running:

                    break

                if not active:

                    break

                time.sleep(
                    0.05
                )

            if voice_stop_event.is_set():

                break

            with voice_mode_lock:

                if not voice_mode:

                    break

        # =====================================================
        # VOICE WORKER EXIT
        # =====================================================

        with voice_mode_lock:

            voice_mode = False

    if os.getenv("AXION_WAKE_WORD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        input_queue.put("wake")

    # =========================================================
    # MAIN COMMAND LOOP
    # =========================================================

    while not shutdown_event.is_set():

        # -----------------------------------------------------
        # Get keyboard command
        # -----------------------------------------------------

        try:

            user_input = (
                input_queue.get()
            )

        except Exception:

            continue

        if not user_input:

            continue

        user_input = (
            user_input.strip()
        )

        if not user_input:

            continue

        command = (
            user_input.lower()
        )

        # =====================================================
        # STOP
        # =====================================================

        if command == "stop":

            # -------------------------------------------------
            # Voice mode?
            # -------------------------------------------------

            with voice_mode_lock:

                was_voice_mode = (
                    voice_mode
                )

                voice_mode = False

            if was_voice_mode:

                voice_stop_event.set()

                # Tell Windows recorder to stop.

                try:

                    stt.stop_recording()

                except Exception:

                    pass

            # -------------------------------------------------
            # Current AXION response?
            # -------------------------------------------------

            with response_lock:

                running = (
                    response_running
                )

                cancel_event = (
                    current_cancel_event
                )

            if cancel_event is not None:

                cancel_event.set()

            if running:

                tts.stop()

                if was_voice_mode:

                    print(
                        "\n🛑 AXION: "
                        "Voice mode stopped "
                        "and response stopped.\n",
                        flush=True,
                    )

                else:

                    print(
                        "\n🛑 AXION: "
                        "Stopped.\n",
                        flush=True,
                    )

            elif was_voice_mode:

                tts.stop()

                print(
                    "\n🛑 AXION: "
                    "Voice mode stopped.\n",
                    flush=True,
                )

            else:

                print(
                    "\nAXION: "
                    "Nothing is running.\n",
                    flush=True,
                )

            print(
                "You: ",
                end="",
                flush=True,
            )

            continue

        # =====================================================
        # EXIT
        # =====================================================

        if command == "exit":

            print(
                "\n🛑 Shutting down AXION...",
                flush=True,
            )

            # -------------------------------------------------
            # Stop voice mode
            # -------------------------------------------------

            with voice_mode_lock:

                voice_mode = False

            voice_stop_event.set()

            # -------------------------------------------------
            # Stop microphone
            # -------------------------------------------------

            try:

                stt.stop_recording()

            except Exception:

                pass

            # -------------------------------------------------
            # Stop current response
            # -------------------------------------------------

            with response_lock:

                cancel_event = (
                    current_cancel_event
                )

            if cancel_event is not None:

                cancel_event.set()

            tts.stop()

            # -------------------------------------------------
            # Stop reminders
            # -------------------------------------------------

            reminder_manager.stop()

            # -------------------------------------------------
            # Mark shutdown
            # -------------------------------------------------

            shutdown_event.set()

            # -------------------------------------------------
            # Goodbye
            # -------------------------------------------------

            print(
                "\nAXION: Goodbye!",
                flush=True,
            )

            try:

                tts.start_stream()

                tts.speak(
                    "Goodbye!"
                )

            except Exception:

                pass

            # -------------------------------------------------
            # Give background threads a moment to stop.
            # -------------------------------------------------

            time.sleep(
                0.15
            )

            break

        # =====================================================
        # VOICE MODE
        # =====================================================

        if command in {"voice", "wake"}:

            requested_wake_mode = command == "wake"

            # -------------------------------------------------
            # Don't start while responding.
            # -------------------------------------------------

            with response_lock:

                if response_running:

                    print(
                        "\nAXION: "
                        "I'm still responding. "
                        "Type 'stop' to interrupt me.\n",
                        flush=True,
                    )

                    print(
                        "You: ",
                        end="",
                        flush=True,
                    )

                    continue

            # -------------------------------------------------
            # Already active?
            # -------------------------------------------------

            with voice_mode_lock:

                if voice_mode:

                    print(
                        "\nAXION: "
                        "Voice mode is already running.\n",
                        flush=True,
                    )

                    continue

                voice_mode = True
                wake_word_mode = requested_wake_mode

            # -------------------------------------------------
            # Reset stop event.
            # -------------------------------------------------

            voice_stop_event.clear()

            # -------------------------------------------------
            # Start voice mode.
            # -------------------------------------------------

            if requested_wake_mode:
                print("\n🎤 AXION: Wake-word mode started. Say 'Hey AXION' to talk.", flush=True)
            else:
                print("\n🎤 AXION: Continuous voice mode started.", flush=True)
                print("Speak naturally.", flush=True)

            print("Type 'stop' to leave voice mode.", flush=True)

            # -------------------------------------------------
            # Start voice worker.
            # -------------------------------------------------

            voice_thread = threading.Thread(
                target=voice_worker,
                daemon=True,
            )

            voice_thread.start()

            # -------------------------------------------------
            # DO NOT BLOCK HERE.
            #
            # Keyboard worker continues accepting:
            #
            # stop
            # exit
            #
            # while microphone is recording.
            # -------------------------------------------------

            continue

        # =====================================================
        # NORMAL TEXT MODE
        # =====================================================

        with voice_mode_lock:

            currently_in_voice_mode = (
                voice_mode
            )

        # Don't process arbitrary keyboard text
        # as normal questions while voice mode is active.

        if currently_in_voice_mode:

            print(
                "\n🎤 AXION: "
                "Voice mode is active. "
                "Type 'stop' to leave voice mode.",
                flush=True,
            )

            continue

        # -----------------------------------------------------
        # Start normal response.
        # -----------------------------------------------------

        started = start_response(
            user_input
        )

        if not started:

            print(
                "\nAXION: "
                "I'm still responding. "
                "Type 'stop' to interrupt me.\n",
                flush=True,
            )

            print(
                "You: ",
                end="",
                flush=True,
            )

    # =========================================================
    # FINAL CLEANUP
    # =========================================================

    shutdown_event.set()

    voice_stop_event.set()

    with voice_mode_lock:

        voice_mode = False

    try:

        stt.stop_recording()

    except Exception:

        pass

    try:

        reminder_manager.stop()

    except Exception:

        pass


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()
