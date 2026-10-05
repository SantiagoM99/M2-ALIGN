"""Synthetic tests for cvqa_prompt.py: the agnostic prompt is unchanged, the country comes from the row (stdlib)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json  # noqa: E402
import tempfile  # noqa: E402

from cvqa_prompt import (  # noqa: E402
    AMBIGUOUS,
    build_cvqa_open_ended_prompt,
    country_lookup,
    country_of,
    location_sentence,
)


def inventory(units):
    """A minimal committed-inventory shape: units with an nllb tag and subsets."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"units": units}, handle)
    handle.close()
    return handle.name


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

    def test_the_legacy_placeholder_is_not_a_subset(self):
        """The legacy evaluation copy stores subset='legacy'; jobs 22448653 and
        22448654 died on it, which is the behaviour we want without a lookup."""
        with self.assertRaises(ValueError):
            country_of({"id": "7_0", "subset": "legacy"})

    def test_the_inventory_supplies_the_country_for_legacy_rows(self):
        path = inventory([{"nllb": "ben_Beng", "subsets": [{"subset": "('Bengali', 'India')"}]}])
        lookup = country_lookup(path)
        self.assertEqual(lookup["ben_Beng"], "India")
        self.assertEqual(country_of({"id": "7_0", "subset": "legacy"}, lookup, "ben_Beng"), "India")

    def test_the_rows_own_subset_wins_over_the_inventory(self):
        """An S1 panel row knows its own country; the fallback must not override it."""
        path = inventory([{"nllb": "ben_Beng", "subsets": [{"subset": "('Bengali', 'India')"}]}])
        row = {"id": "7_0", "subset": "('Bengali', 'Bangladesh')"}
        self.assertEqual(country_of(row, country_lookup(path), "ben_Beng"), "Bangladesh")

    def test_a_language_spanning_two_countries_is_refused(self):
        """CVQA's Chinese is China and Singapore; picking one would be a guess."""
        path = inventory([
            {"nllb": "zho_Hans", "subsets": [{"subset": "('Chinese', 'China')"},
                                             {"subset": "('Chinese', 'Singapore')"}]},
        ])
        lookup = country_lookup(path)
        self.assertIs(lookup["zho_Hans"], AMBIGUOUS)
        with self.assertRaises(ValueError) as caught:
            country_of({"id": "7_0", "subset": "legacy"}, lookup, "zho_Hans")
        self.assertIn("S1 panel", str(caught.exception))

    def test_an_inventory_with_no_usable_subset_fails_closed(self):
        with self.assertRaises(ValueError):
            country_lookup(inventory([{"nllb": "ben_Beng", "subsets": []}]))

    def test_the_real_inventory_resolves_the_five_target_languages(self):
        """Sourced from the committed record, not from a table anyone typed."""
        lookup = country_lookup(Path(__file__).resolve().parent.parent / "audits" / "cvqa_inventory.json")
        self.assertEqual(
            {t: lookup[t] for t in ("ben_Beng", "gle_Latn", "jav_Latn", "khk_Cyrl", "sin_Sinh")},
            {"ben_Beng": "India", "gle_Latn": "Ireland", "jav_Latn": "Indonesia",
             "khk_Cyrl": "Mongolia", "sin_Sinh": "Sri_Lanka"},
        )

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
