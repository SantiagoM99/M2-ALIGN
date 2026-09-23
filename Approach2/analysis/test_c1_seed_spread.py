"""Synthetic tests for c1_seed_spread.py: contrast arithmetic, spread, pairing guards (stdlib)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_seed_spread import analyse  # noqa: E402
from test_block_c import N, fixture  # noqa: E402


class SeedSpreadTests(unittest.TestCase):
    def test_the_report_labels_itself_post_hoc(self):
        r = analyse(fixture(), {"14": fixture(cvqa_n={"C1": 12})}, B=60)
        self.assertIn("post-hoc", r["status"])
        self.assertIn("no gate input", r["status"])
        self.assertIn("excludes training variance", r["inference"])

    def test_contrast_and_spread_are_exact(self):
        reference = fixture()
        r = analyse(reference, {"14": fixture(cvqa_n={"C1": 12}), "15": fixture(cvqa_n={"C1": 28})}, B=120)
        self.assertEqual(r["across_seeds"]["grounding"]["seeds"], ["13", "14", "15"])
        self.assertAlmostEqual(r["per_seed"]["13"]["grounding"]["estimate"], 0.0)
        self.assertAlmostEqual(r["per_seed"]["14"]["grounding"]["estimate"], 100 * 8 / N)
        self.assertAlmostEqual(r["per_seed"]["15"]["grounding"]["estimate"], -100 * 8 / N)
        across = r["across_seeds"]["grounding"]
        self.assertAlmostEqual(across["mean"], 0.0)
        self.assertAlmostEqual(across["range"][0], -100 * 8 / N)
        self.assertAlmostEqual(across["range"][1], 100 * 8 / N)
        self.assertGreater(across["sd"], 0)

    def test_utility_endpoint_is_reported_too(self):
        r = analyse(fixture(), {"14": fixture(cvqa_n={"C1": 12})}, B=120)
        self.assertAlmostEqual(r["per_seed"]["14"]["U"]["estimate"], 100 * 8 / N)

    def test_a_replicate_on_different_shuffle_maps_is_refused(self):
        replicate = fixture(cvqa_n={"C1": 12})
        replicate[("cvqa", "jv", "C1", "correct")]["manifest"]["shuffle_map_hashes"] = {"0": "different"}
        with self.assertRaisesRegex(SystemExit, "shuffle_map_hashes differs"):
            analyse(fixture(), {"14": replicate}, B=60)

    def test_a_replicate_on_a_different_item_set_is_refused(self):
        replicate = fixture(cvqa_n={"C1": 12})
        for condition in ("correct", "shuffled0", "shuffled1", "shuffled2", "gray"):
            replicate[("cvqa", "mn", "C1", condition)]["rows"].pop("3")
        with self.assertRaisesRegex(SystemExit, "different item set"):
            analyse(fixture(), {"14": replicate}, B=60)

    def test_a_replicate_missing_a_cell_is_refused(self):
        replicate = fixture(cvqa_n={"C1": 12})
        del replicate[("cvqa", "ga", "C1", "shuffled1")]
        with self.assertRaisesRegex(SystemExit, "missing the cvqa/ga/shuffled1 cell"):
            analyse(fixture(), {"14": replicate}, B=60)

    def test_two_seeds_give_a_two_point_spread(self):
        r = analyse(fixture(), {"14": fixture(cvqa_n={"C1": 12})}, B=60)
        across = r["across_seeds"]["grounding"]
        self.assertEqual(across["seeds"], ["13", "14"])
        self.assertAlmostEqual(across["mean"], 100 * 4 / N)
        self.assertAlmostEqual(across["range"], [0.0, 100 * 8 / N])
        self.assertIsNotNone(across["sd"])


if __name__ == "__main__":
    unittest.main()
