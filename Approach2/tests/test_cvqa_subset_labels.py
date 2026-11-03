"""The country map must key on our item ids and keep the two Chinese subsets apart (stdlib)."""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cvqa_subset_labels as labels  # noqa: E402


def page(rows, total):
    return {"num_rows_total": total, "rows": [{"row": r} for r in rows]}


class SubsetLabelTests(unittest.TestCase):
    def fetch_from(self, pages):
        """A stub for the rows API: one page per call, in order."""
        calls = iter(pages)

        def fetch(_url):
            return next(calls, {"rows": []})

        return fetch

    def test_chinese_keeps_its_two_countries_apart(self):
        rows = [
            {"ID": "1_0", "Subset": "('Chinese', 'China')"},
            {"ID": "1_1", "Subset": "('Chinese', 'Singapore')"},
            {"ID": "2_0", "Subset": "('Bengali', 'India')"},
        ]
        out, counts = labels.collect(self.fetch_from([page(rows, 3)]))
        self.assertEqual(out["zh"], {"1_0": "China", "1_1": "Singapore"})
        self.assertEqual(out["bn"], {"2_0": "India"})
        self.assertEqual(counts["rows_read"], 3)

    def test_languages_we_do_not_score_are_skipped_and_counted(self):
        rows = [
            {"ID": "1_0", "Subset": "('Spanish', 'Mexico')"},
            {"ID": "2_0", "Subset": "('Korean', 'South Korea')"},
        ]
        out, counts = labels.collect(self.fetch_from([page(rows, 2)]))
        self.assertEqual(sorted(out), ["ko"])
        self.assertEqual(counts["other_language"], 1)

    def test_a_malformed_id_is_counted_not_guessed(self):
        rows = [
            {"ID": "not-an-id", "Subset": "('Bengali', 'India')"},
            {"ID": "3_2", "Subset": "('Bengali', 'India')"},
        ]
        out, counts = labels.collect(self.fetch_from([page(rows, 2)]))
        self.assertEqual(sorted(out["bn"]), ["3_2"])
        self.assertEqual(counts["bad_id"], 1)

    def test_paging_continues_until_the_total_is_reached(self):
        first = page([{"ID": f"{i}_0", "Subset": "('Irish', 'Ireland')"} for i in range(2)], 4)
        second = page([{"ID": f"{i}_0", "Subset": "('Irish', 'Ireland')"} for i in range(2, 4)], 4)
        out, counts = labels.collect(self.fetch_from([first, second]))
        self.assertEqual(len(out["ga"]), 4)
        self.assertEqual(counts["rows_read"], 4)

    def test_no_scored_language_at_all_fails_closed(self):
        rows = [{"ID": "1_0", "Subset": "('Spanish', 'Mexico')"}]
        with self.assertRaises(SystemExit):
            labels.collect(self.fetch_from([page(rows, 1)]))


if __name__ == "__main__":
    unittest.main()
