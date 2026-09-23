"""Synthetic Block C tests: grid, lineage, contrast arithmetic, G4, G5, G1-T readout and order (stdlib)."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_c import ARMS, POSTHOC_CONTRASTS, analyse, required_cells, validate_grid  # noqa: E402

N = 40


def fixture(cvqa_n=None, xgqa_n=None):
    """Correct-image hits: item i is right iff i < n(arm); shuffles and gray always wrong.

    With shuffles and gray wrong, utility, grounding and gray sensitivity all
    equal the correct-image hit rate, so every contrast is known in closed form.
    """
    cvqa_n = {arm: 20 for arm in ARMS} | (cvqa_n or {})
    xgqa_n = {arm: 20 for arm in ARMS} | (xgqa_n or {})
    cells = {}
    for key in sorted(required_cells()):
        b, t, arm, c = key
        n = (cvqa_n if b == "cvqa" else xgqa_n)[arm]
        text_dir, vision_dir, no_text = ARMS[arm]
        rows = {
            str(i): {
                "id": str(i), "image_id": str(i // 2), "subset": "s", "query": "q",
                "choices": ["a", "b"], "answer_index": 0, "answer": "a",
                "correct": c == "correct" and i < n,
            }
            for i in range(N)
        }
        checkpoints = {"mapping_vis": {"path": f"/o/{vision_dir}/mapping/pytorch_model.bin", "sha256": vision_dir}}
        checkpoints["mapping_txt"] = {"path": f"/o/{text_dir}/mapping/pytorch_model.bin", "sha256": text_dir}
        cells[key] = {
            "rows": rows,
            "manifest": {
                "vis_layers": "9,18,-1",
                "arguments": {"no_text_branch": no_text, "prompt_mode": "question", "question_field": "query"},
                "checkpoints": checkpoints,
                "data": b + t, "id_universe_sha256": "u", "image_sha256": "i",
                "shuffle_map_hashes": {"0": "a"}, "decoding": {}, "environment": {"python": "3.11.5"},
            },
        }
    return cells


class BlockCTests(unittest.TestCase):
    def test_grid_is_150_cells(self):
        self.assertEqual(len(required_cells()), 150)
        validate_grid(fixture())

    def test_grid_fails_closed(self):
        cells = fixture()
        del cells[next(iter(sorted(cells)))]
        with self.assertRaisesRegex(ValueError, "complete 150-cell grid"):
            validate_grid(cells)
        cells = fixture()
        cells[("cvqa", "jv", "C2", "correct")]["manifest"]["checkpoints"]["mapping_txt"]["path"] = \
            "/o/stage1/mapping/pytorch_model.bin"
        with self.assertRaisesRegex(ValueError, "mapping_txt is not loaded from s1_C2_seed13"):
            validate_grid(cells)
        cells = fixture()
        cells[("xgqa", "bn", "C5", "gray")]["manifest"]["arguments"]["no_text_branch"] = False
        with self.assertRaisesRegex(ValueError, "text-branch flag"):
            validate_grid(cells)
        cells = fixture()
        cells[("cvqa", "mn", "C4", "shuffled2")]["rows"].pop("3")
        with self.assertRaisesRegex(ValueError, "unpaired"):
            validate_grid(cells)

    def test_null_pilot(self):
        r = analyse(fixture(), B=200)
        for name in ("P5", "P6", "P7", "D8"):
            self.assertEqual(r["panels"]["primary"]["contrasts"][name]["grounding"]["estimate"], 0.0)
        g = r["gates"]
        self.assertFalse(g["G4_freeze_text_helps_transfer"])
        self.assertTrue(g["G5_source_task_retained"])
        self.assertEqual(g["G1T_pilot_readout"], "dispensable in training (single-seed pilot)")
        self.assertFalse(g["G1T_decidable"])

    def test_freezing_text_helps(self):
        r = analyse(fixture(cvqa_n={"C2": 36}), B=300)
        p5 = r["panels"]["primary"]["contrasts"]["P5"]
        self.assertAlmostEqual(p5["grounding"]["estimate"], 100 * 16 / N)
        self.assertTrue(r["gates"]["G4_freeze_text_helps_transfer"])
        self.assertTrue(r["gates"]["P5_predicted_positive_on_grounding"])

    def test_source_task_loss_fails_g5(self):
        r = analyse(fixture(xgqa_n={"C2": 8}), B=300)
        self.assertFalse(r["gates"]["G5_source_task_retained"])
        self.assertAlmostEqual(r["gates"]["G5_bound"]["UB95(xGQA-bn U(C1)-U(C2))"] > 1.0, True)

    def test_p7_arithmetic(self):
        r = analyse(fixture(cvqa_n={"C4": 28, "C2": 24}), B=200)
        # (C4 - C3) - (C2 - C1) = (28 - 20) - (24 - 20) = 4 items of 40
        self.assertAlmostEqual(r["panels"]["primary"]["contrasts"]["P7"]["grounding"]["estimate"], 100 * 4 / N)

    def test_vision_only_arm_materially_worse(self):
        r = analyse(fixture(cvqa_n={"C5": 4}), B=300)
        d8 = r["panels"]["primary"]["contrasts"]["D8"]["grounding"]
        self.assertAlmostEqual(d8["estimate"], 100 * 16 / N)
        self.assertEqual(r["gates"]["G1T_pilot_readout"], "needed in training (single-seed pilot)")

    def test_posthoc_contrasts_are_absent_unless_asked_for(self):
        cells = fixture(cvqa_n={"C4": 32})
        frozen = analyse(cells, B=120)
        self.assertNotIn("posthoc_contrasts", frozen)
        for panel in frozen["panels"].values():
            self.assertEqual(set(panel["contrasts"]) & set(POSTHOC_CONTRASTS), set())
        with_posthoc = analyse(cells, B=120, posthoc=True)
        self.assertEqual(with_posthoc["posthoc_contrasts"], ["X_C4_minus_C1"])
        # the frozen contrasts must be unchanged by asking for the extra one
        for name in ("P5", "P6", "P7", "D8"):
            self.assertEqual(with_posthoc["panels"]["primary"]["contrasts"][name],
                             frozen["panels"]["primary"]["contrasts"][name])
        x = with_posthoc["panels"]["primary"]["contrasts"]["X_C4_minus_C1"]["grounding"]
        self.assertAlmostEqual(x["estimate"], 100 * 12 / N)

    def test_insertion_order_does_not_change_the_report(self):
        cells = fixture(cvqa_n={"C2": 26, "C5": 14})
        keys = list(cells)
        random.Random(5).shuffle(keys)
        self.assertEqual(analyse(cells, B=150), analyse({k: cells[k] for k in keys}, B=150))


if __name__ == "__main__":
    unittest.main()
