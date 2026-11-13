import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from axion.wake_word import split_wake_phrase


class WakePhraseTests(unittest.TestCase):
    def test_wake_phrase_only(self):
        self.assertEqual(split_wake_phrase("Hey AXION"), "")

    def test_wake_hello_phrase_only(self):
        self.assertEqual(split_wake_phrase("Hello AXION"), "")

    def test_wake_phrase_with_action(self):
        self.assertEqual(split_wake_phrase("Hey Action"), "")

    def test_wake_hello_action(self):
        self.assertEqual(split_wake_phrase("Hello Action"), "")

    def test_wake_only_with_terminal_punctuation(self):
        self.assertEqual(split_wake_phrase("  HEY, AXION!!!  "), "")

    def test_wake_only_with_terminal_punctuation_hello(self):
        self.assertEqual(split_wake_phrase("  HELLO, AXION!!!  "), "")

    def test_supported_transcription_variants(self):
        variants = {
            "Hey AXION": "",
            "Hello AXION": "",
            "HEY, AXION!": "",
            "HELLO, AXION!": "",
            "Hey Action": "",
            "Hello Action": "",
            "hey   action, open YouTube": "open YouTube",
        }
        for text, expected in variants.items():
            with self.subTest(text=text):
                self.assertEqual(split_wake_phrase(text), expected)

    def test_wake_phrase_and_command_preserves_command_punctuation(self):
        self.assertEqual(
            split_wake_phrase("Hey AXION, open YouTube."),
            "open YouTube.",
        )

    def test_wake_hello_phrase_and_command_preserves_command_punctuation(self):
        self.assertEqual(
            split_wake_phrase("Hello AXION, set a reminder."),
            "set a reminder.",
        )

    def test_punctuation_and_extra_whitespace_variant(self):
        self.assertEqual(
            split_wake_phrase(" \tHey:   AXION — what time is it?  "),
            "what time is it?",
        )

    def test_punctuation_and_extra_whitespace_hello_variant(self):
        self.assertEqual(
            split_wake_phrase(" \tHello:   AXION — what time is it?  "),
            "what time is it?",
        )

    def test_comma_between_name_and_command(self):
        self.assertEqual(
            split_wake_phrase("hey, AXION, open YouTube"),
            "open YouTube",
        )

    def test_comma_between_name_and_command_hello(self):
        self.assertEqual(
            split_wake_phrase("hello, AXION, open YouTube"),
            "open YouTube",
        )

    def test_leading_harmless_punctuation(self):
        self.assertEqual(
            split_wake_phrase("— Hey AXION; set a reminder"),
            "set a reminder",
        )

    def test_leading_harmless_punctuation_hello(self):
        self.assertEqual(
            split_wake_phrase("— Hello AXION; set a reminder"),
            "set a reminder",
        )

    def test_wake_phrase_must_start_transcription(self):
        self.assertIsNone(split_wake_phrase("I said hey AXION, open YouTube"))

    def test_wake_hello_phrase_must_start_transcription(self):
        self.assertIsNone(split_wake_phrase("I said hello AXION, open YouTube"))

    def test_unrelated_similar_phrases_do_not_match(self):
        for text in (
            "Hey there, AXION, open YouTube",
            "Hi AXION, open YouTube",
            "Hey AXIOM, open YouTube",
            "Hey Actions, open YouTube",
            "Action, open YouTube",
            "Hey there, Action, open YouTube",
            "Hello there, Action, open YouTube",
        ):
            with self.subTest(text=text):
                self.assertIsNone(split_wake_phrase(text))

    def test_empty_input(self):
        self.assertIsNone(split_wake_phrase(""))
        self.assertIsNone(split_wake_phrase(None))

    def test_debug_transcription_is_opt_in(self):
        output = StringIO()
        with patch.dict(os.environ, {"AXION_WAKE_DEBUG": "0"}):
            with redirect_stdout(output):
                split_wake_phrase("  Hey, AXION, open YouTube  ")
        self.assertEqual(output.getvalue(), "")

        output = StringIO()
        with patch.dict(os.environ, {"AXION_WAKE_DEBUG": "1"}):
            with redirect_stdout(output):
                split_wake_phrase("  Hey, AXION, open YouTube  ")
        self.assertIn("normalized='Hey, AXION, open YouTube'", output.getvalue())
        self.assertIn("transcription='  Hey, AXION, open YouTube  '", output.getvalue())
        self.assertIn("phrase_matched=True", output.getvalue())


if __name__ == "__main__":
    unittest.main()
