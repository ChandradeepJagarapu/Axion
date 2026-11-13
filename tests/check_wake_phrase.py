"""Check a prior Whisper transcription without recording another sample."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axion.wake_word import split_wake_phrase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcription", help="Text returned by Whisper")
    args = parser.parse_args()

    command = split_wake_phrase(args.transcription, debug=True)
    if command is None:
        print("Result: wake phrase not matched.")
        return 1

    print(f"Result: wake phrase matched; command={command!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
