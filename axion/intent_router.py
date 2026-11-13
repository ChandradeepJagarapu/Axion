import re
from enum import Enum


class Intent(str, Enum):
    CHAT = "chat"
    WEATHER = "weather"
    WEB_SEARCH = "web_search"
    REMINDER = "reminder"
    MEMORY_SAVE = "memory_save"
    MEMORY_RECALL = "memory_recall"
    MEMORY_FORGET = "memory_forget"
    MEMORY_LIST = "memory_list"
    MEMORY_UPDATE = "memory_update"
    DATE_TIME = "date_time"
    SYSTEM_CONTROL = "system_control"


class IntentRouter:
    """Classify user requests using local, deterministic text rules."""

    _date_time_phrases = (
        "what time is it",
        "what's the time",
        "whats the time",
        "what is the time",
        "current time",
        "time now",
        "what day is it",
        "what's today's date",
        "whats today's date",
        "what is today's date",
        "today's date",
        "todays date",
        "current date",
        "date today",
    )

    _web_search_phrases = (
        "latest", "current", "today", "now", "recent", "news",
        "update", "updates", "price", "prices", "stock", "stocks",
        "score", "scores", "result", "results", "released", "release",
        "version", "versions", "trending", "search", "look up", "lookup",
        "internet", "online", "website", "who is", "what happened",
        "what's happening", "happening now", "this week", "this month",
        "this year",
    )

    @staticmethod
    def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
        return any(
            re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
            for phrase in phrases
        )

    def classify(self, user_input: str) -> Intent:
        text = re.sub(r"\s+", " ", (user_input or "").strip().lower())
        if not text:
            return Intent.CHAT

        # Specific personal actions take priority over broad current-info words.
        if re.search(r"\bremind\s+me\b", text):
            return Intent.REMINDER

        if self._contains_phrase(text, (
            "list my memories", "show my memories", "list memories",
            "show memories", "what have you remembered",
        )):
            return Intent.MEMORY_LIST

        if re.search(r"\bdon't\s+forget\b|\bdont\s+forget\b", text):
            return Intent.MEMORY_SAVE

        if re.search(
            r"\b(forget|remove|delete)\b", text
        ) or re.search(r"\bi\s+don't\s+like\s+.+\s+anymore\b", text):
            return Intent.MEMORY_FORGET

        if re.search(r"\b(change|update)\b", text) or re.search(
            r"\bmy\s+(?:favorite|favourite)\b.+\bis\s+now\b", text
        ):
            return Intent.MEMORY_UPDATE

        if self._contains_phrase(text, (
            "what do you remember", "what do you know about me",
            "what did i tell you", "do you remember",
        )):
            return Intent.MEMORY_RECALL

        if re.search(r"\bwhat\s+(?:is|are)\s+my\b", text) or re.search(
            r"\bwhat\s+(?:color|colour|language|food)\s+do\s+i\s+(?:like|prefer)\b",
            text,
        ):
            return Intent.MEMORY_RECALL

        if re.search(r"\b(remember|save)\b", text) or re.search(
            r"\bdon't\s+forget\b", text
        ):
            return Intent.MEMORY_SAVE

        if re.search(
            r"\b(cpu|processor|ram|memory usage|disk usage|system status)\b",
            text,
        ) or re.search(r"\bhow much (?:system )?memory\b", text):
            return Intent.SYSTEM_CONTROL

        if re.search(
            r"^(?:(?:please|could you|can you|would you)\s+)?"
            r"(?:axion\s+)?(?:open(?:\s+up)?|launch|start)\b",
            text,
        ):
            return Intent.SYSTEM_CONTROL

        if re.search(r"\b(weather|temperature)\b", text):
            return Intent.WEATHER

        if self._contains_phrase(text, self._date_time_phrases):
            return Intent.DATE_TIME

        if self._contains_phrase(text, self._web_search_phrases):
            return Intent.WEB_SEARCH

        return Intent.CHAT
