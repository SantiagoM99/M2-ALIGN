"""Synthetic tests for arch_compare.py: clustering, closed-form contrasts, McNemar, fail-closed (stdlib)."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_compare import analyse, load_arm, mcnemar_exact, read_image_map  # noqa: E402

N_IMAGES = 20
QUESTIONS_PER_IMAGE = 3
N = N_IMAGES * QUESTIONS_PER_IMAGE


def cvqa_ids():
    return [f"img{i}_{q}" for i in range(N_IMAGES) for q in range(QUESTIONS_PER_IMAGE)]


def write_arm(root, name, benchmark, lang, n_full, n_blind, ids=None, image_ids=False):
    """Item k is correct iff k < n, so every accuracy and contrast is known exactly."""
    ids = ids or cvqa_ids()
    for condition, marker, n in (("full", "", n_full), ("blind", "_BLIND", n_blind)):
        path = Path(root) / f"{name}_{benchmark}_{lang}{marker}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for k, item in enumerate(ids):
                row = {"id": item, "correct": k < n}
                if image_ids:
                    row["image_id"] = item.rsplit("_", 1)[0]
                handle.write(json.dumps(row) + "\n")


class ArchCompareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def arms(self, spec, benchmark="cvqa", lang="jv", **kwargs):
        for name, (n_full, n_blind) in spec.items():
            write_arm(self.root, name, benchmark, lang, n_full, n_blind, **kwargs)
        return {name: load_arm(str(Path(self.root) / (name + "_{b}_{L}{blind}.jsonl")), benchmark, [lang])
                for name in spec}

    def test_cvqa_clusters_come_from_the_id_prefix(self):
        arms = self.arms({"a": (30, 10), "b": (30, 10)})
        r = analyse(arms, "cvqa", ["jv"], {}, B=50)
        self.assertEqual(r["pooled"]["a"]["full"]["items"], N)
        self.assertEqual(r["pooled"]["a"]["full"]["image_clusters"], N_IMAGES)

    def test_xgqa_without_an_image_map_fails_closed(self):
        ids = [str(1000 + k) for k in range(N)]
        arms = self.arms({"a": (30, 10), "b": (20, 10)}, benchmark="xgqa", ids=ids)
        with self.assertRaisesRegex(SystemExit, "--image-map"):
            analyse(arms, "xgqa", ["jv"], {}, B=50)

    def test_xgqa_with_an_image_map_clusters_by_image(self):
        arms = self.arms({"a": (30, 10), "b": (20, 10)}, benchmark="xgqa", image_ids=True)
        image_map = read_image_map(Path(self.root) / "a_xgqa_jv.jsonl")
        r = analyse(arms, "xgqa", ["jv"], image_map, B=50)
        self.assertEqual(r["pooled"]["a"]["full"]["image_clusters"], N_IMAGES)

    def test_contrast_and_dv_are_exact(self):
        arms = self.arms({"a": (36, 12), "b": (24, 12)})
        r = analyse(arms, "cvqa", ["jv"], {}, B=100)
        self.assertAlmostEqual(r["per_language"]["a"]["jv"]["full"], 100 * 36 / N)
        self.assertAlmostEqual(r["pooled"]["a"]["dv"]["estimate"], 100 * 24 / N)
        c = r["contrasts"]["a-minus-b"]
        self.assertAlmostEqual(c["full"]["estimate"], 100 * 12 / N)
        self.assertAlmostEqual(c["dv"]["estimate"], 100 * 12 / N)

    def test_mcnemar_counts_only_discordant_items(self):
        arms = self.arms({"a": (36, 12), "b": (24, 12)})
        m = analyse(arms, "cvqa", ["jv"], {}, B=50)["contrasts"]["a-minus-b"]["mcnemar"]
        self.assertEqual((m["a_only"], m["b_only"]), (12, 0))
        self.assertAlmostEqual(m["p_two_sided"], mcnemar_exact(12, 0))
        self.assertLess(m["p_two_sided"], 0.001)

    def test_mcnemar_exact_is_two_sided_and_symmetric(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(1, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(8, 2), mcnemar_exact(2, 8))
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 / 2 ** 10)

    def test_items_missing_from_one_arm_are_dropped_from_all(self):
        arms = self.arms({"a": (30, 10), "b": (30, 10)})
        del arms["b"]["jv"]["full"]["img0_0"]
        r = analyse(arms, "cvqa", ["jv"], {}, B=50)
        self.assertEqual(r["pooled"]["a"]["full"]["items"], N - 1)
        self.assertEqual(r["pooled"]["b"]["full"]["items"], N - 1)

    def test_no_shared_items_fails_closed(self):
        arms = self.arms({"a": (30, 10), "b": (30, 10)})
        arms["b"]["jv"]["full"] = {"other": 1}
        with self.assertRaisesRegex(SystemExit, "share no scored items"):
            analyse(arms, "cvqa", ["jv"], {}, B=50)

    def test_the_image_map_accepts_her_vg_image_id_field(self):
        """Her xGQA rows name the Visual Genome image `vg_image_id`; ours `image_id`."""
        path = Path(self.root) / "hers.jsonl"
        path.write_text('{"id": "7", "vg_image_id": "img3", "correct": true}\n', encoding="utf-8")
        self.assertEqual(read_image_map(path), {"7": "img3"})

    def test_an_image_map_with_neither_field_fails_closed(self):
        path = Path(self.root) / "plain.jsonl"
        path.write_text('{"id": "7", "correct": true}\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            read_image_map(path)

    def test_a_full_only_arm_reports_accuracy_and_no_dv(self):
        """A collaborator's system with no gray canvas still gets a paired contrast."""
        arms = self.arms({"ours": (30, 10)})
        write_arm(self.root, "theirs", "cvqa", "jv", 24, 0)
        Path(self.root, "theirs_cvqa_jv_BLIND.jsonl").unlink()
        arms["theirs"] = load_arm(str(Path(self.root) / "theirs_{b}_{L}.jsonl"),
                                  "cvqa", ["jv"], full_only=True)
        report = analyse(arms, "cvqa", ["jv"], {}, B=50)
        self.assertAlmostEqual(report["pooled"]["theirs"]["full"]["estimate"], 100 * 24 / N)
        self.assertNotIn("dv", report["pooled"]["theirs"])
        self.assertIsNone(report["per_language"]["theirs"]["jv"]["dv"])
        contrast = report["contrasts"]["ours-minus-theirs"]
        self.assertAlmostEqual(contrast["full"]["estimate"], 100 * (30 - 24) / N)
        self.assertNotIn("dv", contrast)
        self.assertEqual(contrast["mcnemar"]["ours_only"], 6)
        self.assertEqual(contrast["mcnemar"]["theirs_only"], 0)

    def test_the_other_arm_keeps_its_dv(self):
        """One arm without a blind mode must not cost the others their dV."""
        arms = self.arms({"ours": (30, 10), "backbone": (20, 5)})
        write_arm(self.root, "theirs", "cvqa", "jv", 24, 0)
        arms["theirs"] = load_arm(str(Path(self.root) / "theirs_{b}_{L}.jsonl"),
                                  "cvqa", ["jv"], full_only=True)
        report = analyse(arms, "cvqa", ["jv"], {}, B=50)
        self.assertAlmostEqual(report["pooled"]["ours"]["dv"]["estimate"], 100 * (30 - 10) / N)
        self.assertIn("dv", report["contrasts"]["backbone-minus-ours"])

    def test_a_missing_blind_file_without_the_flag_fails_closed(self):
        write_arm(self.root, "theirs", "cvqa", "jv", 24, 0)
        Path(self.root, "theirs_cvqa_jv_BLIND.jsonl").unlink()
        with self.assertRaises(SystemExit):
            load_arm(str(Path(self.root) / "theirs_{b}_{L}{blind}.jsonl"), "cvqa", ["jv"])

    def test_arm_order_does_not_change_the_report(self):
        arms = self.arms({"a": (36, 12), "b": (24, 12)})
        reversed_arms = {name: arms[name] for name in reversed(list(arms))}
        self.assertEqual(analyse(arms, "cvqa", ["jv"], {}, B=80),
                         analyse(reversed_arms, "cvqa", ["jv"], {}, B=80))



class SubsetSplitTests(unittest.TestCase):
    """A gain carried by one subset reads differently from the same pooled gain."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        ids = cvqa_ids()
        # b answers the first 30 items, a the first 10, so every one of the 20
        # items b gains lies in ids[10:30].
        write_arm(self.root, "a", "cvqa", "jv", 10, 0, ids=ids)
        write_arm(self.root, "b", "cvqa", "jv", 30, 0, ids=ids)
        self.arms = {
            "a": load_arm(f"{self.root}/a_{{b}}_{{L}}{{blind}}.jsonl", "cvqa", ["jv"]),
            "b": load_arm(f"{self.root}/b_{{b}}_{{L}}{{blind}}.jsonl", "cvqa", ["jv"]),
        }
        self.ids = ids

    def report(self, labels):
        return analyse(self.arms, "cvqa", ["jv"], {}, B=200, subset_labels=labels)

    def test_without_labels_no_subsets_appear(self):
        report = analyse(self.arms, "cvqa", ["jv"], {}, B=200)
        self.assertNotIn("subsets", report["contrasts"]["a-minus-b"])

    def test_the_whole_gain_sits_in_the_labelled_subset(self):
        labels = {"jv": {i: ("gain" if k in range(10, 30) else "rest")
                         for k, i in enumerate(self.ids)}}
        subsets = self.report(labels)["contrasts"]["a-minus-b"]["subsets"]
        # a minus b, so the sign is negative where b wins.
        self.assertAlmostEqual(subsets["gain"]["estimate"], -100.0, places=6)
        self.assertAlmostEqual(subsets["rest"]["estimate"], 0.0, places=6)
        self.assertEqual(subsets["gain"]["items"], 20)
        self.assertEqual(subsets["rest"]["items"], len(self.ids) - 20)

    def test_labels_from_another_item_universe_fail_closed(self):
        """A join that covers almost nothing would report an empty split, not a wrong one."""
        labels = {"jv": {self.ids[0]: "only"}}
        with self.assertRaisesRegex(SystemExit, "different item universe"):
            self.report(labels)

    def test_partial_but_sufficient_coverage_is_reported(self):
        """Above the threshold the split stands, and says how many items it used."""
        labels = {"jv": {i: "kept" for i in self.ids[: int(len(self.ids) * 0.8)]}}
        subsets = self.report(labels)["contrasts"]["a-minus-b"]["subsets"]
        self.assertEqual(sorted(subsets), ["kept"])
        self.assertEqual(subsets["kept"]["items"], int(len(self.ids) * 0.8))

if __name__ == "__main__":
    unittest.main()
