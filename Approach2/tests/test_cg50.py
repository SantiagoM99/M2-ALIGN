"""CG50's reader on a synthetic 60-cell grid: the two readings stay apart (stdlib + PIL).

Written from the two readings as DESIGN recorded them on 2026-10-08, against
data built so each one's answer is known in advance: CG50 is made to gain only on
the items the reference answers wrongly (`other`), and to tie on the ones where
the image and the choices already suffice (`shortcut`). A reader that mixed the
subsets, or that let reading 2 stand in for reading 1, would not produce this.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))
from s1_contract import SPEC_SHA, atomic_json, digest, item_identity  # noqa: E402
from s1_plan import build  # noqa: E402
from eval_matrix import read_plan  # noqa: E402
from eval_runtime import condition, prepare, make_manifest, evaluate  # noqa: E402
from block_a import load_cells  # noqa: E402
import cg50  # noqa: E402

ITEMS = 16                      # 8 image clusters, two questions each
NECESSARY = {"0", "1", "2", "3"}
SHORTCUT = {"4", "5", "6", "7"}
OTHER = {str(i) for i in range(8, ITEMS)}
# The items CG50 is given and C1 is not: odd ids inside `other`, where C1 is wrong.
BOOST = {i for i in OTHER if int(i) % 2}


class Runtime:
    """C1 answers even ids; CG50 also answers the boosted `other` ids."""

    def predict(self, a, p):
        c = condition(a)
        for row in p["rows"]:
            i = str(row["id"])
            correct = c == "correct" and (
                int(i) % 2 == 0 or (a.arm == "CG50" and i in BOOST)
            )
            r = {**row, **item_identity(row, a.benchmark), "correct": correct, "condition": c}
            r["assigned_image_id"] = (
                p["shuffled"]["items"][i]["assigned_image_id"]
                if p["shuffled"]
                else r["image_id"]
                if c == "correct"
                else None
            )
            if a.benchmark == "cvqa":
                r.update(pred_index=0 if correct else 1, scores=[-1.0, -2.0])
            else:
                r["pred"] = "yes" if correct else "no"
            yield r


def reference_files(root):
    """A reference whose question decides NECESSARY, suffices on SHORTCUT, fails on OTHER."""
    with_q, blind = [], []
    for i in range(ITEMS):
        i = str(i)
        right = i in NECESSARY or i in SHORTCUT
        with_q.append({"id": i, "correct": right})
        blind.append({"id": i, "correct": i in SHORTCUT})
    paths = []
    for name, rows in (("qwen_with.jsonl", with_q), ("qwen_blind.jsonl", blind)):
        p = root / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(p)
    return paths


class CG50Tests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self._tmp = tempfile.TemporaryDirectory()
        self.root = root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        images = root / "images"
        images.mkdir()
        for i in range(ITEMS // 2):
            Image.new("RGB", (4, 4), (i * 15, 0, 0)).save(images / f"{i}.jpg")
        panels = {"cvqa": {}, "xgqa": {}}
        for b, targets in (("cvqa", cg50.CVQA_TARGETS), ("xgqa", ("bn",))):
            for t in targets:
                rows = [
                    {
                        "id": str(i),
                        "image_id": str(i // 2),
                        "subset": "s",
                        "query": "q",
                        "english_query": "english",
                        "choices": ["yes", "no"],
                        "answer_index": 0,
                        "answer": "yes",
                        "nllb_lang_tag": "eng_Latn",
                    }
                    for i in range(ITEMS)
                ]
                path = root / f"{b}_{t}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                panels[b][t] = {"data": str(path), "images": str(images)}
        atomic_json(root / "panels.json", panels)
        for name in ("s1_C1_seed13", "cg50_seed13"):
            ck = root / "outputs" / name / "mapping" / "pytorch_model.bin"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(b"fixture")
            atomic_json(root / "outputs" / name / "complete.json", {"epochs_completed": 2})
        atomic_json(root / "gate.json", {"spec_sha": SPEC_SHA, "gates": {"G0": True}})

        a = SimpleNamespace(
            panels=str(root / "panels.json"),
            checkpoints=str(root / "outputs"),
            block="C",
            arms=["C1"],
            arm_seed=13,
            extra_arm=["CG50=cg50_seed13"],
            maps_dir=str(root / "maps"),
            results=str(root / "results"),
            output=str(root / "plan.json"),
            block_a_report=str(root / "gate.json"),
        )
        with patch("s1_plan.ROOT", root):
            build(a)
        plan, cells = read_plan(Path(a.output))
        self.cell_count = len(cells)
        sub = {
            "spec_sha": SPEC_SHA,
            "code_sha": "fixture",
            "plan": plan,
            "plan_sha256": digest(plan),
            "manifests": {},
        }
        for cell, args in cells:
            prep = prepare(args)
            with patch(
                "eval_runtime.model_records",
                return_value={"llm": {"revision": "fixture"}, "text": None, "vision": None},
            ):
                m = make_manifest(
                    args, prep, {"spec_sha": SPEC_SHA, "code_sha": "fixture", "dirty": False}
                )
            sub["manifests"][cell["id"]] = m
            evaluate(args, Runtime(), prep, m)
        atomic_json(root / "submission.json", sub)
        self.submission = root / "submission.json"
        self.cells = load_cells(self.submission, block="C")
        self.with_q, self.blind = reference_files(root)

    def labels(self):
        return cg50.labels_for_panel(self.cells, self.with_q, self.blind)

    def test_the_grid_is_c1s_grid_with_one_arm_added(self):
        self.assertEqual(self.cell_count, 60)
        self.assertEqual(set(self.cells), cg50.required_cells())

    def test_the_reference_split_is_recovered(self):
        self.assertEqual(
            {k: sum(1 for v in self.labels().values() if v == k) for k in cg50.SUBSETS},
            {"necessary": 4, "shortcut": 4, "other": 8},
        )

    def test_reading_one_finds_the_gain_where_it_was_planted(self):
        report = cg50.analyse(self.cells, self.labels(), B=200)
        bn = report["readings"]["1_bn_country_matched"]
        self.assertGreater(bn["subsets"]["other"]["U"]["lb5"], 0)
        self.assertEqual(bn["subsets"]["shortcut"]["U"]["estimate"], 0.0)
        self.assertGreater(bn["other_minus_shortcut_U"]["estimate"], 0)
        self.assertTrue(bn["passes"])

    def test_reading_two_is_computed_separately_and_is_not_reading_one(self):
        report = cg50.analyse(self.cells, self.labels(), B=200)
        two = report["readings"]["2_zero_shot_transfer"]
        self.assertEqual(two["panel"], "cvqa jv/mn/ga pooled")
        # Shuffles and gray are wrong for both arms here, so grounding == U.
        self.assertAlmostEqual(
            two["grounding_gain"]["estimate"], two["utility_gain"]["estimate"], places=6
        )
        self.assertGreater(two["grounding_gain"]["lb5"], 0)
        self.assertLess(two["source_retention_xgqa_bn"]["ub95"], cg50.DELTA)
        self.assertTrue(two["passes"])

    def test_a_gain_only_in_shortcut_fails_reading_one(self):
        """The rule that refuted the location-aware diagnostic, enforced here."""
        swap = {"other": "shortcut", "shortcut": "other"}
        labels = {i: swap.get(v, v) for i, v in self.labels().items()}
        report = cg50.analyse(self.cells, labels, B=200)
        self.assertFalse(report["readings"]["1_bn_country_matched"]["passes"])

    def test_a_gain_only_on_overlapping_answers_does_not_survive(self):
        """The whole point of the split: memorised answers cannot carry the claim."""
        overlap = {i: ("overlap" if i in BOOST else "clean") for i in self.labels()}
        split = cg50.analyse(self.cells, self.labels(), B=200, overlap=overlap)[
            "readings"
        ]["1b_answer_set_overlap"]
        self.assertEqual(split["clean"]["U"]["estimate"], 0.0)
        self.assertGreater(split["overlap"]["U"]["estimate"], 0)
        self.assertFalse(split["survives_the_answer_set_objection"])

    def test_a_gain_on_unseen_answers_survives(self):
        overlap = {i: ("clean" if i in BOOST else "overlap") for i in self.labels()}
        split = cg50.analyse(self.cells, self.labels(), B=200, overlap=overlap)[
            "readings"
        ]["1b_answer_set_overlap"]
        self.assertGreater(split["clean"]["U"]["lb5"], 0)
        self.assertTrue(split["survives_the_answer_set_objection"])

    def test_an_unknown_overlap_label_is_refused(self):
        bad = {i: "maybe" for i in self.labels()}
        with self.assertRaisesRegex(ValueError, "overlap/clean"):
            cg50.analyse(self.cells, self.labels(), B=20, overlap=bad)

    def test_an_incomplete_grid_is_refused(self):
        cells = dict(self.cells)
        cells.pop(("cvqa", "bn", "CG50", "gray"))
        with self.assertRaisesRegex(ValueError, "complete 60-cell grid"):
            cg50.analyse(cells, self.labels(), B=20)

    def test_a_reference_from_another_universe_is_refused(self):
        other = self.root / "elsewhere.jsonl"
        other.write_text(json.dumps({"id": "999", "correct": True}) + "\n")
        with self.assertRaisesRegex(ValueError, "not the same item universe"):
            cg50.labels_for_panel(self.cells, other, other)


if __name__ == "__main__":
    unittest.main()
