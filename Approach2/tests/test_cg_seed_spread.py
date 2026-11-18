"""The across-seed reader refuses reports that are not of the same contrast."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import cg_seed_spread as spread  # noqa: E402


def report(path: Path, est: float, items=1783, name="cg-minus-ctrl", langs=None) -> Path:
    langs = langs if langs is not None else {"bn": est - 1, "ru": est + 1}
    path.write_text(json.dumps({
        "contrasts": {name: {
            "full": {"estimate": est, "ci95": [est - 2, est + 2], "items": items},
            "per_language_full": langs,
        }}
    }))
    return path


class TestSpread(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def three(self, **kw):
        return {
            "42": spread.read_report(report(self.d / "a.json", 13.46, **kw)),
            "13": spread.read_report(report(self.d / "b.json", 12.39, **kw)),
            "7": spread.read_report(report(self.d / "c.json", 13.63, **kw)),
        }

    def test_mean_sd_and_range(self):
        out = spread.spread(self.three())
        self.assertAlmostEqual(out["mean"], 13.16, places=2)
        self.assertAlmostEqual(out["sd"], 0.672, places=2)
        self.assertAlmostEqual(out["range"], 1.24, places=2)
        self.assertTrue(out["all_same_sign"])

    def test_scope_travels_with_the_number(self):
        """A bare SD invites being read as a population CI, so the caveat ships."""
        out = spread.spread(self.three())
        self.assertIn("stage-3", out["scope"])
        self.assertIn("not a population CI", out["scope"].replace("is not", "not"))

    def test_two_seeds_are_refused(self):
        two = {k: v for k, v in list(self.three().items())[:2]}
        with self.assertRaises(SystemExit):
            spread.spread(two)

    def test_per_language_sign_replication_is_reported(self):
        out = spread.spread(self.three())
        self.assertTrue(out["per_language"]["ru"]["replicates_sign"])
        self.assertEqual(sorted(out["per_language"]), ["bn", "ru"])

    def test_a_language_missing_from_one_seed_is_dropped(self):
        seeds = self.three()
        del seeds["7"]["per_language"]["bn"]
        out = spread.spread(seeds)
        self.assertEqual(sorted(out["per_language"]), ["ru"])

    def test_a_sign_flip_is_visible(self):
        seeds = self.three()
        seeds["7"]["per_language"]["bn"] = -5.0
        out = spread.spread(seeds)
        self.assertFalse(out["per_language"]["bn"]["replicates_sign"])

    def test_a_subset_is_read_from_its_own_arm(self):
        p = self.d / "sub.json"
        p.write_text(json.dumps({"contrasts": {"cg-minus-ctrl": {
            "full": {"estimate": 12.30, "ci95": [10.6, 14.1], "items": 2943},
            "per_language_full": {"bn": 11.0},
            "subsets": {
                "clean": {"estimate": 11.16, "ci95": [9.3, 13.0], "items": 2626},
                "overlap": {"estimate": 21.77, "ci95": [16.0, 27.5], "items": 317},
            },
        }}}))
        clean = spread.read_report(p, "clean")
        self.assertAlmostEqual(clean["estimate"], 11.16)
        self.assertEqual(clean["items"], 2626)
        self.assertIn("[clean]", clean["contrast"])
        # The per-language table belongs to the full arm, so a subset carries none.
        self.assertEqual(clean["per_language"], {})
        self.assertAlmostEqual(spread.read_report(p)["estimate"], 12.30)

    def test_a_missing_subset_is_fatal_not_a_fallback(self):
        """Reading the full arm as if it were the clean one is the error the
        audit exists to prevent, so it must never happen silently."""
        p = report(self.d / "nosub.json", 12.30)
        with self.assertRaises(SystemExit) as cm:
            spread.read_report(p, "clean")
        self.assertIn("--subset-labels", str(cm.exception))

    def test_a_report_with_two_contrasts_is_refused(self):
        p = self.d / "two.json"
        p.write_text(json.dumps({"contrasts": {
            "a": {"full": {"estimate": 1.0, "items": 10}, "per_language_full": {}},
            "b": {"full": {"estimate": 2.0, "items": 10}, "per_language_full": {}},
        }}))
        with self.assertRaises(SystemExit):
            spread.read_report(p)

    def test_a_report_without_an_estimate_is_refused(self):
        p = self.d / "empty.json"
        p.write_text(json.dumps({"contrasts": {"a": {"full": {}}}}))
        with self.assertRaises(SystemExit):
            spread.read_report(p)


if __name__ == "__main__":
    unittest.main()
