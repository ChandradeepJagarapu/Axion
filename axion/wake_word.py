"""Small, deterministic helpers for AXION's optional wake-word mode."""

import os
import re
from difflib import SequenceMatcher

# Words that are common English words (or close variants) and should NOT
# trigger wake-word detection via the _is_similar_wake_word fallback,
# even though they pass the length + ratio check.  This prevents false
# activations from words like 'axiom' or 'action' appearing in normal
# speech while still catching genuine Whisper errors such as 'Exion'.
_BLACKLIST = {"action", "axiom"}


_WAKE_PHRASE = re.compile(
    r"^(?:hey|hello)\b[\s,.:;!?—–\-\"'“”‘’]+(?:axion|action)\b",
    re.IGNORECASE,
)
_LEADING_PUNCTUATION = " \t\r\n\"'“”‘’([{<,.:;!?—–-"
_PHRASE_SEPARATORS = " \t,.:;!?—–-…"


def _normalized_transcription(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lstrip(_LEADING_PUNCTUATION)


def _is_similar_wake_word(normalized):
    """Check if a single-word normalized transcription is a Whisper error
    of 'hey AXION' or 'hey Action' (e.g. 'Exion' -> 'axion').

    The length‑+‑ratio check still applies.  The _BLACKLIST guard below
    prevents false wakeword activation from standalone English words such
    as 'action' or 'axiom' that happen to match the length + ratio check."""
    words = normalized.split()
    if len(words) != 1:
        return False
    word = words[0]
    # Strip trailing punctuation so 'Exion!' compares as 'Exion'
    word_stripped = word.rstrip("!.,;:?\"'")
    # Skip words that are common English words; those should be handled
    # by the regex pattern or rejected entirely, not by a fallback that
    # could activate on ordinary vocabulary.
    if word_stripped.lower() in _BLACKLIST:
        return False
    for target in ["axion", "action"]:
        target_lower = target.lower()
        # Same length and good edit ratio catches Whisper errors like Exion↔axion
        # while rejecting plurals (axions→axion, diff length) and derivatives
        # (actionable→action, diff length).
        if len(word_stripped) == len(target_lower):
            ratio = SequenceMatcher(
                None, word_stripped.lower(), target_lower
            ).ratio()
            if ratio >= 0.8:
                return True
    return False


def split_wake_phrase(text, debug=None):
    """Return the command after a leading supported wake phrase, else None.

    Set ``AXION_WAKE_DEBUG=1`` to print the normalized transcription and match
    result. Transcribed words are never logged unless that flag is enabled.
    """
    normalized = _normalized_transcription(text)
    if debug is None:
        debug = os.getenv("AXION_WAKE_DEBUG", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

    match = _WAKE_PHRASE.match(normalized) if normalized else None
    command = None
    if match:
        remainder = normalized[match.end():].lstrip(_PHRASE_SEPARATORS)
        # Punctuation after a wake-only utterance is part of the phrase, but
        # keep the command's own ending punctuation untouched.
        command = remainder if re.search(r"\w", remainder) else ""

    # FIX: If the regex didn't match, check for Whisper errors where
    # 'Hey AXION' or 'Hey Action' is transcribed as a single similar word
    # (e.g. 'Exion' instead of 'Hey AXION'). If the normalized transcription
    # is a single word that is edit-close to 'axion' or 'action', treat it
    # as a wake-word match.
    if not match and normalized and _is_similar_wake_word(normalized):
        match = True
        command = ""

    if debug:
        print(
            f"AXION wake debug: transcription={str(text)!r}; "
            f"normalized={normalized!r}; "
            f"phrase_matched={match is not None}.",
            flush=True,
        )

    return command
