"""Synthetic tests for cvqa_prompt.py: the agnostic prompt is unchanged, the country comes from the row (stdlib)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvqa_prompt import build_cvqa_open_ended_prompt, country_of, location_sentence  # noqa: E402


class CvqaPromptTests(unittest.TestCase):
    def test_the_location_agnostic_prompt_is_byte_identical_to_the_old_one(self):
        """Every cell scored before 2026-10-03 used this exact string."""
        self.assertEqual(build_cvqa_open_ended_prompt("কোন দেবতা?"), "Question: কোন দেবতা?")

    def test_an_empty_country_is_the_agnostic_prompt(self):
        """--blind-question leaves an empty question; an empty country must not
        produce a half-written location sentence."""
        self.assertEqual(build_cvqa_open_ended_prompt("q", ""), "Question: q")
        self.assertEqual(build_cvqa_open_ended_prompt("q", None), "Question: q")

    def test_the_location_aware_prompt_states_the_country_before_the_question(self):
        self.assertEqual(
            build_cvqa_open_ended_prompt("q", "India"),
            "This picture is from India.\nQuestion: q",
        )

    def test_underscores_in_country_names_are_spelled_out(self):
        """The inventory writes Sri_Lanka; a prompt should not."""
        self.assertEqual(location_sentence("Sri_Lanka"), "This picture is from Sri Lanka.")

    def test_the_country_comes_from_the_subset_field(self):
        self.assertEqual(country_of({"id": "7_0", "subset": "('Bengali', 'India')"}), "India")
        self.assertEqual(country_of({"id": "7_0", "subset": "('Sinhala', 'Sri_Lanka')"}), "Sri_Lanka")

    def test_a_row_without_a_subset_fails_rather_than_guessing(self):
        """Bengali's CVQA subset is India, not Bangladesh, so a language code is
        not a country and must never be used as one."""
        with self.assertRaises(ValueError):
            country_of({"id": "7_0", "source_language": "bn"})

    def test_a_malformed_subset_fails(self):
        for bad in ("Bengali", "('Bengali',)", "('Bengali', '')", "[1, 2]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                country_of({"id": "7_0", "subset": bad})


if __name__ == "__main__":
    unittest.main()
