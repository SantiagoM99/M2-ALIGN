"""Synthetic tests for build_stage3_english.py: the schema her trainer reads (stdlib)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_stage3_english as b  # noqa: E402

# Her Stage3/train.py reads exactly these from each row.
TRAINER_READS = ("vg_image_id", "query", "answer", "nllb_lang_tag")


class EnglishRowTests(unittest.TestCase):
    def test_the_converted_row_carries_what_the_trainer_reads(self):
        rows = b.convert([{"question": "Is there a bread?", "answer": "no",
                           "vg_image_id": "2375767"}])
        self.assertEqual(len(rows), 1)
        for key in TRAINER_READS:
            self.assertIn(key, rows[0])
        self.assertEqual(rows[0]["query"], "Is there a bread?")
        self.assertEqual(rows[0]["nllb_lang_tag"], "eng_Latn")
        self.assertEqual(rows[0]["source_language"], "English")

    def test_the_row_count_is_preserved(self):
        source = [{"question": f"q{i}", "answer": "yes", "vg_image_id": str(i)}
                  for i in range(100)]
        self.assertEqual(len(b.convert(source)), 100)

    def test_ids_are_deterministic_and_distinct(self):
        source = [{"question": "q", "answer": "a", "vg_image_id": "1"},
                  {"question": "q", "answer": "a", "vg_image_id": "2"}]
        first = b.convert(source)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in b.convert(source)])
        self.assertNotEqual(first[0]["id"], first[1]["id"])

    def test_two_identical_rows_collide_and_fail_closed(self):
        """A duplicated row would make the id ambiguous, so say so."""
        source = [{"question": "q", "answer": "a", "vg_image_id": "1"}] * 2
        with self.assertRaisesRegex(SystemExit, "collide on id"):
            b.convert(source)

    def test_a_missing_field_names_itself(self):
        for row in ({"answer": "a", "vg_image_id": "1"},
                    {"question": "q", "vg_image_id": "1"},
                    {"question": "q", "answer": "a"},
                    {"question": "  ", "answer": "a", "vg_image_id": "1"}):
            with self.subTest(row=row), self.assertRaises(SystemExit):
                b.convert([row])


if __name__ == "__main__":
    unittest.main()
