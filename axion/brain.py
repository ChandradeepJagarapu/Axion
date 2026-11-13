import re
from datetime import datetime, timedelta
from typing import Callable, Optional

import ollama

from .config import MODEL_NAME
from .intent_router import Intent, IntentRouter


# =============================================================
# AXION SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = """
You are AXION, a personal AI assistant.

IDENTITY:
- Your name is AXION.
- You are the AI assistant.
- Chandradeep is the developer and user of AXION.
- Chandradeep is NOT AXION.
- Never say that your name is Chandradeep.
- Never describe Chandradeep's personal identity as your own identity.

IMPORTANT IDENTITY QUESTION RULES:
- If the user asks "Who are you?", answer about AXION.
- If the user asks "Tell me about yourself", answer about AXION.
- If the user asks "Introduce yourself", answer about AXION.
- If the user asks "What are you?", answer about AXION.
- If the user asks "Who developed you?", answer that Chandradeep developed AXION.
- If the user asks "Who am I?", answer about Chandradeep using saved memory.
- If the user asks "What is my name?", answer with the user's saved name.
- Never answer "My name is Chandradeep."
- Never say "I am Chandradeep."
- Never treat information about Chandradeep as information about yourself.

ABOUT AXION:
- AXION is a personal AI assistant developed by Chandradeep.
- AXION can help with conversations, coding, reminders, memory, and voice interaction.
- When describing yourself, speak in the first person as AXION.
- Example: "I'm AXION, your personal AI assistant."
- Example: "I was developed by Chandradeep."

PERSONAL MEMORY:
- Saved memory is reliable user-provided information.
- Use it when relevant.
- Never invent personal information.
- Never claim you have no memory when relevant memory exists.
- When asked what you remember, summarize the saved information.
- When asked about AXION, use the saved AXION/project information.
- Remember that saved information about Chandradeep describes the user, not AXION.

RESPONSE STYLE:
- Be concise and natural by default.
- For normal questions, answer in 1-3 short sentences.
- Do not add unnecessary explanations, examples, or repetition.
- Give longer answers only when the user asks for detail.

ACTIONS:
- Memory and reminders are handled by the application.
- Never claim an action was completed unless the application actually performed it.
"""


# =============================================================
# DATE / TIME
# =============================================================

def get_local_datetime() -> str:
    return datetime.now().strftime(
        "%A, %d %B %Y, %I:%M %p"
    )


def is_datetime_question(message: str) -> bool:
    text = message.lower().strip()

    phrases = [
        "what time is it",
        "what's the time",
        "whats the time",
        "current time",
        "time now",
        "what is the time",
        "what day is it",
        "what's today's date",
        "whats today's date",
        "what is today's date",
        "today's date",
        "todays date",
        "current date",
        "date today",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


# =============================================================
# EXPLICIT MEMORY
# =============================================================

def extract_explicit_memory(message: str):
    patterns = [
        r"^remember that (.+)$",
        r"^remember (.+)$",
        r"^don't forget that (.+)$",
        r"^dont forget that (.+)$",
        r"^save that (.+)$",
        r"^save (.+)$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            message.strip(),
            flags=re.IGNORECASE,
        )

        if match:
            fact = match.group(1).strip()

            if fact:
                return fact

    return None


# =============================================================
# AUTOMATIC MEMORY
# =============================================================

def extract_automatic_memory(message: str):
    text = message.strip()

    text = text.strip("\"'")

    lower = text.lower()

    if "?" in text:
        return None

    question_starts = [
        "what ",
        "who ",
        "where ",
        "when ",
        "why ",
        "how ",
        "do you ",
        "can you ",
        "could you ",
        "would you ",
        "is ",
        "are ",
        "am i ",
    ]

    if any(
        lower.startswith(start)
        for start in question_starts
    ):
        return None

    command_starts = [
        "remind me",
        "tell me",
        "explain",
        "show me",
        "give me",
        "help me",
    ]

    if any(
        lower.startswith(start)
        for start in command_starts
    ):
        return None

    # ---------------------------------------------------------
    # Name
    # ---------------------------------------------------------

    match = re.match(
        r"my name is (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "name",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Internship
    # ---------------------------------------------------------

    internship_patterns = [
        r"i(?:'m| am) currently doing an internship",
        r"i(?:'m| am) doing an internship",
        r"i(?:'m| am) currently in an internship",
        r"i(?:'m| am) an intern",
    ]

    for pattern in internship_patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return (
                "internship",
                "currently doing an internship",
            )

    # ---------------------------------------------------------
    # Developer of AXION
    # ---------------------------------------------------------

    developer_patterns = [
        r"i(?:'m| am) the developer of axion",
        r"axion is .*developed by me",
        r"i developed axion",
        r"i am developing axion",
        r"i(?:'m| am) developing axion",
    ]

    for pattern in developer_patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return (
                "role",
                "developer of AXION",
            )

    # ---------------------------------------------------------
    # AXION project
    # ---------------------------------------------------------

    project_patterns = [
        r"axion is an ai project",
        r"axion is an ai project that i",
        r"axion is a project that i",
        r"i(?:'m| am) working on axion",
        r"i work on axion",
        r"i(?:'m| am) currently working on axion",
    ]

    for pattern in project_patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return (
                "project",
                "AXION",
            )

    # ---------------------------------------------------------
    # Favorite language
    # ---------------------------------------------------------

    match = re.search(
        r"my (?:favorite|favourite)"
        r" (?:programming )?language is (?:now )?(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "favorite_language",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Favorite color
    # ---------------------------------------------------------

    match = re.search(
        r"my (?:favorite|favourite) color is (?:now )?(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "favorite_color",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Favorite food
    # ---------------------------------------------------------

    match = re.search(
        r"my (?:favorite|favourite) food is (?:now )?(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "favorite_food",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Location
    # ---------------------------------------------------------

    match = re.search(
        r"i live in (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "location",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Hometown
    # ---------------------------------------------------------

    match = re.search(
        r"i(?:'m| am) from (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "hometown",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Learning
    # ---------------------------------------------------------

    match = re.search(
        r"i(?:'m| am) learning (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "learning",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Current work
    # ---------------------------------------------------------

    match = re.search(
        r"i(?:'m| am) working on (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "current_work",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Likes
    # ---------------------------------------------------------

    match = re.search(
        r"i like (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "likes",
            match.group(1).strip(),
        )

    # ---------------------------------------------------------
    # Dislikes
    # ---------------------------------------------------------

    match = re.search(
        r"i don't like (.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            "dislikes",
            match.group(1).strip(),
        )

    return None


# =============================================================
# MEMORY KEY
# =============================================================

def create_memory_key(fact: str) -> str:
    lower = fact.lower()

    if re.search(r"\bi don't like\b", lower):
        return "dislikes"

    if re.search(r"\bi like\b", lower):
        return "likes"

    if "my name is" in lower:
        return "name"

    if "internship" in lower:
        return "internship"

    if (
        "developer of axion" in lower
        or "developed axion" in lower
    ):
        return "role"

    if "working on axion" in lower:
        return "project"

    if "favorite language" in lower:
        return "favorite_language"

    if "favourite language" in lower:
        return "favorite_language"

    if "favorite color" in lower:
        return "favorite_color"

    if "favourite color" in lower:
        return "favorite_color"

    if "favorite food" in lower:
        return "favorite_food"

    if "favourite food" in lower:
        return "favorite_food"

    if "live in" in lower:
        return "location"

    if "from" in lower:
        return "hometown"

    words = re.findall(
        r"[a-zA-Z0-9]+",
        fact.lower(),
    )

    if words:
        return "_".join(words[:4])

    return "memory"


# =============================================================
# REMINDER
# =============================================================

def extract_reminder_time(message: str):
    """
    Extract reminder time from natural language.

    Supported:
      - remind me in 5 minutes
      - remind me in 30 seconds
      - remind me in 2 hours
      - remind me in 1 day
      - remind me at 12:41 PM
      - remind me at 18:30
      - remind me tomorrow at 9 AM
      - remind me today at 12:41 PM
    """

    message_clean = message.strip()

    # =========================================================
    # RELATIVE TIME
    # =========================================================

    relative_match = re.search(
        r"\bin\s+"
        r"(\d+(?:\.\d+)?)\s*"
        r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
        message_clean,
        flags=re.IGNORECASE,
    )

    if relative_match:
        amount = float(relative_match.group(1))
        unit = relative_match.group(2).lower()

        if unit.startswith(("second", "sec")):
            seconds = amount

        elif unit.startswith(("minute", "min")):
            seconds = amount * 60

        elif unit.startswith(("hour", "hr")):
            seconds = amount * 60 * 60

        elif unit.startswith("day"):
            seconds = amount * 24 * 60 * 60

        else:
            return None

        now = datetime.now()
        remind_at = now + timedelta(seconds=seconds)

        reminder_message = re.sub(
            r"^remind me",
            "",
            message_clean,
            flags=re.IGNORECASE,
        )

        reminder_message = re.sub(
            r"\bin\s+"
            r"\d+(?:\.\d+)?\s*"
            r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
            "",
            reminder_message,
            flags=re.IGNORECASE,
        )

        reminder_message = re.sub(
            r"\s+",
            " ",
            reminder_message,
        ).strip()

        if reminder_message.lower().startswith("to "):
            reminder_message = reminder_message[3:].strip()

        if not reminder_message:
            reminder_message = "the reminder you requested"

        return (
            reminder_message,
            remind_at,
        )

    # =========================================================
    # ABSOLUTE CLOCK TIME
    # =========================================================

    time_match = re.search(
        r"\b("
        r"(?:[01]?\d|2[0-3])"
        r"(?::[0-5]\d)?"
        r"\s*(?:AM|PM|am|pm)"
        r"|"
        r"(?:[01]?\d|2[0-3]):[0-5]\d"
        r")\b",
        message_clean,
    )

    if not time_match:
        return None

    time_text = time_match.group(1).strip()

    try:
        if re.search(
            r"(AM|PM)",
            time_text,
            flags=re.IGNORECASE,
        ):
            parsed_time = datetime.strptime(
                time_text.upper().replace(" ", ""),
                "%I:%M%p"
                if ":" in time_text
                else "%I%p",
            ).time()

        else:
            parsed_time = datetime.strptime(
                time_text,
                "%H:%M",
            ).time()

    except ValueError:
        return None

    now = datetime.now()
    reminder_date = now.date()

    tomorrow = re.search(
        r"\btomorrow\b",
        message_clean,
        flags=re.IGNORECASE,
    )

    if tomorrow:
        reminder_date += timedelta(days=1)

    remind_at = datetime.combine(
        reminder_date,
        parsed_time,
    )

    if (
        remind_at <= now
        and not tomorrow
    ):
        reminder_date += timedelta(days=1)

        remind_at = datetime.combine(
            reminder_date,
            parsed_time,
        )

    # =========================================================
    # CLEAN REMINDER MESSAGE
    # =========================================================

    reminder_message = re.sub(
        r"^remind me",
        "",
        message_clean,
        flags=re.IGNORECASE,
    )

    reminder_message = re.sub(
        r"\bat\s+"
        + re.escape(time_text)
        + r"\b",
        "",
        reminder_message,
        flags=re.IGNORECASE,
    )

    reminder_message = re.sub(
        r"\btoday\b",
        "",
        reminder_message,
        flags=re.IGNORECASE,
    )

    reminder_message = re.sub(
        r"\btomorrow\b",
        "",
        reminder_message,
        flags=re.IGNORECASE,
    )

    reminder_message = re.sub(
        r"\s+",
        " ",
        reminder_message,
    ).strip()

    if reminder_message.lower().startswith("to "):
        reminder_message = reminder_message[3:].strip()

    if not reminder_message:
        reminder_message = "the reminder you requested"

    return (
        reminder_message,
        remind_at,
    )


# =============================================================
# CLEAN TEXT FOR TTS
# =============================================================

def clean_response_for_tts(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"\.{2,}",
        ".",
        text,
    )

    return text


# =============================================================
# MEMORY TEXT
# =============================================================

def build_memory_text(memory):
    if not memory:
        return "(No personal information is currently saved.)"

    lines = []

    for key, value in memory.items():
        lines.append(
            f"- {key}: {value}"
        )

    return "\n".join(lines)


_MEMORY_FIELD_ALIASES = {
    "favorite_color": ("favorite color", "favourite color", "color", "colour"),
    "favorite_language": ("favorite language", "favourite language", "programming language"),
    "favorite_food": ("favorite food", "favourite food"),
    "name": ("my name", "name"),
    "location": ("where i live", "location", "live in"),
    "hometown": ("hometown", "where i'm from", "where i am from"),
    "likes": ("things i like", "what i like"),
    "dislikes": ("things i dislike", "what i dislike"),
}


def _memory_key_mentioned(text: str, memories: dict) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9_ ]+", " ", text.lower())
    if re.search(r"\bwhat do i like\b|\bi like\b", normalized):
        return "likes"
    if re.search(r"\bwhat do i dislike\b|\bi don't like\b", text.lower()):
        return "dislikes"
    for key, aliases in _MEMORY_FIELD_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return key

    for key in memories:
        label = key.replace("_", " ").lower()
        if label and label in normalized:
            return key
    return None


def _memory_value_for_fact(key: str, fact: str) -> str:
    if key == "likes":
        match = re.search(r"\bi like\s+(.+)", fact, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .!?\"'")
    if key == "dislikes":
        match = re.search(r"\bi don't like\s+(.+)", fact, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .!?\"'")
    aliases = _MEMORY_FIELD_ALIASES.get(key, ())
    field_pattern = "|".join(re.escape(alias) for alias in aliases)
    if field_pattern:
        match = re.search(
            rf"(?:my\s+)?(?:{field_pattern})\s+(?:is|=)\s+(?:now\s+)?(.+)",
            fact,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .!?\"'")
    return fact.strip()


def _memory_forget_target(message: str, memories: dict) -> Optional[str]:
    key = _memory_key_mentioned(message, memories)
    if key in memories:
        return key

    normalized = re.sub(r"[^a-z0-9_ ]+", " ", message.lower())
    if key and key not in ("likes", "dislikes"):
        if any(alias in normalized for alias in _MEMORY_FIELD_ALIASES.get(key, ())):
            return key

    # Handle "I don't like X anymore" and "forget that I like X" by
    # matching the stated value against the existing likes/dislikes facts.
    value_match = re.search(
        r"(?:don't\s+like|do\s+not\s+like|like)\s+(.+?)(?:\s+anymore)?[.!?]*$",
        message,
        flags=re.IGNORECASE,
    )
    if value_match:
        value = value_match.group(1).strip(" .!?\"'").lower()
        matches = [
            stored_key
            for stored_key, stored_value in memories.items()
            if stored_key in ("likes", "dislikes")
            and value
            and value in str(stored_value).lower()
        ]
        if len(matches) == 1:
            return matches[0]
    return None


# =============================================================
# RESPONSE LENGTH
# =============================================================

def wants_detailed_answer(message: str) -> bool:
    text = message.lower()

    detailed_phrases = [
        "in detail",
        "detailed",
        "explain fully",
        "explain everything",
        "give me a detailed",
        "long answer",
        "deep explanation",
        "step by step",
        "step-by-step",
        "elaborate",
        "comprehensive",
    ]

    return any(
        phrase in text
        for phrase in detailed_phrases
    )


# =============================================================
# MAIN BRAIN
# =============================================================

def ask_axion(
    message: str,
    memory=None,
    reminder_manager=None,
    on_chunk: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    intent: Optional[Intent] = None,
) -> str:

    # ---------------------------------------------------------
    # Cancellation helper
    # ---------------------------------------------------------

    def cancelled() -> bool:
        return (
            cancel_event is not None
            and cancel_event.is_set()
        )

    if cancelled():
        return ""

    if intent is None:
        intent = IntentRouter().classify(message)

    def respond(response: str) -> str:
        response = clean_response_for_tts(response)
        if not cancelled():
            print(response)
            if on_chunk:
                on_chunk(response)
        return response

    try:
        personal_memory = memory.get_all() if memory is not None else {}
    except Exception:
        personal_memory = {}

    if intent is Intent.MEMORY_LIST:
        if not personal_memory:
            return respond("I don't have any memories saved yet.")
        lines = [
            f"- {key.replace('_', ' ').title()}: "
            f"{_memory_value_for_fact(key, str(value))}"
            for key, value in personal_memory.items()
        ]
        return respond("Here is what I remember:\n" + "\n".join(lines))

    if intent is Intent.MEMORY_FORGET:
        if re.search(
            r"\bforget\s+(?:everything|all)(?:\s+you\s+remember(?:\s+about\s+me)?)?\b",
            message,
            flags=re.IGNORECASE,
        ):
            if memory is None:
                return respond("I can't access saved memory right now.")
            memory.clear()
            return respond("I've forgotten everything I had saved about you.")

        key = _memory_forget_target(message, personal_memory)
        if key is None:
            return respond("Which specific memory would you like me to forget?")
        if key not in personal_memory:
            label = key.replace("_", " ")
            return respond(f"I don't have your {label} saved.")
        memory.delete(key)
        return respond(f"I've forgotten your {key.replace('_', ' ')}.")

    if intent is Intent.MEMORY_UPDATE:
        key = _memory_key_mentioned(message, personal_memory)
        match = re.search(
            r"\b(?:change|update)\s+(?:my\s+)?(.+?)\s+to\s+(.+?)[.!?]*$",
            message,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\b(my\s+(?:favorite|favourite)\s+.+?)\s+is\s+now\s+(.+?)[.!?]*$",
                message,
                flags=re.IGNORECASE,
            )
        if key is None and match:
            key = _memory_key_mentioned(match.group(1), personal_memory)
        if key is None or match is None:
            return respond("What memory should I update, and what should its new value be?")
        value = match.group(2).strip(" .!?\"'")
        if not value:
            return respond("What should the new value be?")
        if memory is None:
            return respond("I can't access saved memory right now.")
        memory.update(key, value)
        return respond(
            f"I've updated your {key.replace('_', ' ')} to {value}."
        )

    if intent is Intent.MEMORY_RECALL:
        key = _memory_key_mentioned(message, personal_memory)
        if key is not None:
            if key not in personal_memory:
                return respond(f"I don't have your {key.replace('_', ' ')} saved.")
            value = _memory_value_for_fact(
                key,
                str(personal_memory[key]),
            )
            return respond(f"Your {key.replace('_', ' ')} is {value}.")

    # ---------------------------------------------------------
    # Explicit memory
    # ---------------------------------------------------------

    explicit_fact = extract_explicit_memory(
        message
    )

    if explicit_fact and memory is not None:

        key = create_memory_key(
            explicit_fact
        )

        memory.set(
            key,
            _memory_value_for_fact(key, explicit_fact),
        )

        response = (
            f"Got it. I'll remember that "
            f"{explicit_fact}"
        )

        response = clean_response_for_tts(
            response
        )

        if not cancelled():

            print(response)

            if on_chunk:
                on_chunk(response)

        return response

    # ---------------------------------------------------------
    # Reminder
    # ---------------------------------------------------------

    if (
        re.search(
            r"\bremind me\b",
            message,
            flags=re.IGNORECASE,
        )
        and reminder_manager is not None
    ):

        reminder_data = extract_reminder_time(
            message
        )

        if reminder_data:

            reminder_message, remind_at = (
                reminder_data
            )

            reminder_manager.add_reminder(
                reminder_message,
                remind_at,
            )

            formatted_time = (
                remind_at.strftime(
                    "%I:%M %p on %A, %d %B"
                )
            )

            response = (
                f"Done. I'll remind you to "
                f"{reminder_message} at "
                f"{formatted_time}."
            )

            response = clean_response_for_tts(
                response
            )

            if not cancelled():

                print(response)

                if on_chunk:
                    on_chunk(response)

            return response

    # ---------------------------------------------------------
    # Automatic memory
    # ---------------------------------------------------------

    automatic_memory = (
        extract_automatic_memory(message)
    )

    if automatic_memory and memory is not None:

        key, value = automatic_memory

        memory.set(
            key,
            value,
        )

        original_statement = (
            message.strip().strip("\"'")
        )

        response = (
            f"I'll remember that "
            f"{original_statement}."
        )

        response = clean_response_for_tts(
            response
        )

        if not cancelled():

            print(response)

            if on_chunk:
                on_chunk(response)

        return response

    # ---------------------------------------------------------
    # Date / time
    # ---------------------------------------------------------

    if is_datetime_question(message):

        response = (
            "The current local date and time is "
            f"{get_local_datetime()}."
        )

        if not cancelled():

            print(response)

            if on_chunk:
                on_chunk(response)

        return response

    # ---------------------------------------------------------
    # Load memory
    # ---------------------------------------------------------

    personal_memory = {}

    if memory is not None:

        try:
            personal_memory = memory.get_all()

        except Exception:
            personal_memory = {}

    memory_text = build_memory_text(
        personal_memory
    )

    # ---------------------------------------------------------
    # Active reminders
    # ---------------------------------------------------------

    reminders_text = "(No active reminders.)"

    if reminder_manager is not None:

        try:

            active_reminders = (
                reminder_manager.get_active_reminders()
            )

            if active_reminders:

                reminder_lines = []

                for reminder in active_reminders:

                    try:

                        remind_at = datetime.fromisoformat(
                            reminder["remind_at"]
                        )

                        formatted = remind_at.strftime(
                            "%A, %d %B %Y, %I:%M %p"
                        )

                    except Exception:

                        formatted = reminder.get(
                            "remind_at",
                            "unknown time",
                        )

                    reminder_lines.append(
                        f"- {reminder.get('message', 'Reminder')} "
                        f"at {formatted}"
                    )

                reminders_text = "\n".join(
                    reminder_lines
                )

        except Exception:
            pass

    # ---------------------------------------------------------
    # Ollama prompt
    # ---------------------------------------------------------

    current_datetime = get_local_datetime()

    system_message = (
        SYSTEM_PROMPT
        + "\n\nCURRENT LOCAL DATE/TIME:\n"
        + current_datetime
        + "\n\nPERSONAL MEMORY:\n"
        + memory_text
        + "\n\nACTIVE REMINDERS:\n"
        + reminders_text
    )

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    # ---------------------------------------------------------
    # Ollama generation settings
    # ---------------------------------------------------------

    if wants_detailed_answer(message):

        num_predict = 256

    else:

        num_predict = 128

    full_response = ""

    # ---------------------------------------------------------
    # TTS buffer
    #
    # Ollama may return very small chunks.
    # We collect them here and only send meaningful
    # sentence-sized text to TTS.
    # ---------------------------------------------------------

    tts_buffer = ""

    stream = None

    try:

        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.3,
                "num_predict": num_predict,
            },
        )

        for chunk in stream:

            # -------------------------------------------------
            # Cancellation
            # -------------------------------------------------

            if cancelled():

                try:

                    close_method = getattr(
                        stream,
                        "close",
                        None,
                    )

                    if callable(close_method):

                        close_method()

                except Exception:

                    pass

                return full_response.strip()

            # -------------------------------------------------
            # Get content
            # -------------------------------------------------

            content = (
                chunk
                .get("message", {})
                .get("content", "")
            )

            if not content:

                continue

            # -------------------------------------------------
            # Save complete response
            # -------------------------------------------------

            full_response += content

            # -------------------------------------------------
            # Display response immediately.
            #
            # This keeps terminal streaming fast.
            # -------------------------------------------------

            print(
                content,
                end="",
                flush=True,
            )

            # -------------------------------------------------
            # Add to TTS buffer.
            # -------------------------------------------------

            tts_buffer += content

            # -------------------------------------------------
            # Send complete sentences to TTS.
            # -------------------------------------------------

            while True:

                sentence_end = -1

                for marker in (
                    ".",
                    "!",
                    "?",
                ):

                    position = tts_buffer.find(
                        marker
                    )

                    if position != -1:

                        if (
                            sentence_end == -1
                            or position < sentence_end
                        ):

                            sentence_end = position

                # -------------------------------------------------
                # No complete sentence yet.
                # -------------------------------------------------

                if sentence_end == -1:

                    break

                sentence = (
                    tts_buffer[
                        :sentence_end + 1
                    ].strip()
                )

                tts_buffer = (
                    tts_buffer[
                        sentence_end + 1:
                    ]
                )

                if not sentence:

                    continue

                if cancelled():

                    return full_response.strip()

                if on_chunk:

                    on_chunk(
                        sentence
                    )

        # ---------------------------------------------------------
        # Flush remaining TTS text.
        # ---------------------------------------------------------

        if not cancelled():

            remaining_tts = tts_buffer.strip()

            if remaining_tts:

                if on_chunk:

                    on_chunk(
                        remaining_tts
                    )

        # ---------------------------------------------------------
        # Cancellation check
        # ---------------------------------------------------------

        if cancelled():

            return full_response.strip()

        print()

        return clean_response_for_tts(
            full_response
        )

    except Exception as error:

        if cancelled():

            return ""

        raise error
