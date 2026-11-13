# Wake-Word Fix Summary

## Issue
Whisper transcribes "Hey AXION" as "Exion", preventing the wake-word detector from triggering.

## Root Cause
The `split_wake_phrase()` function in `axion/wake_word.py` uses a strict regex requiring the transcription to start with "hey" + "axion"/"action". Whisper errors like "Exion" bypass this entirely.

## Fix Applied
Added a fallback check in `split_wake_phrase()`: if the regex doesn't match, check if the normalized transcription is a single word with the same length and edit-ratio ≥ 0.8 to "axion" or "action". This catches Whisper errors while rejecting plurals/derivatives.

## File Modified
- `axion/wake_word.py` — added `_is_similar_wake_word()` helper and fallback logic

## Test Results
- All 11 existing unit tests pass
- "Exion" → wake word matched (fixes the reported issue)
- "Hey AXION", "hey AXION", "HEY AXION!" → still work correctly
- No false positives: "axions", "actionable", "Hey AXIOM", "Hey Actions" correctly rejected
- "random text" correctly rejected

## Verification
```
$ python3 tests/check_wake_phrase.py "Exion"
AXION wake debug: transcription='Exion'; normalized='Exion'; phrase_matched=True.
Result: wake phrase matched; command=''.
```