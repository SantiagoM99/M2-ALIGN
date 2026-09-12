"""Synthetic Block B tests: grid, lineage, pairing, P3/P4 arithmetic, G2 and order (stdlib)."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_b import (  # noqa: E402
    TEXT, VISION, analyse, arm, required_cells, validate_grid,
)

N_ITEMS = 40


def fixture(text_effect=0, vision_effect=0, matched_effect=0, nox_drop=0):
    """Correct-image hits: item i is right iff i < n(cell); shuffles and gray are always wrong.

    n = 20 + text_effect for T3_id text, + vision_effect for V3_id vision, and
    + matched_effect for the two matched stage-3 corners. So D_T, D_V and P4
    are known in closed form: text_effect, vision_effect and matched_effect
    items out of 40.
    """
    cells = {}
    for key in sorted(required_cells()):
        b, t, a, c = key
        parts = a.split("__")
        text, vision, nox = parts[0], parts[1], len(parts) == 3
        n = 20 + text_effect * (text == "T3_id") + vision_effect * (vision == "V3_id")
        if (text, vision) in (("T3_bn", "V3_bn"), ("T3_id", "V3_id")):
            n += matched_effect
        if nox:
            n -= nox_drop
        rows = {
            str(i): {
                "id": str(i), "image_id": str(i // 2), "subset": "s", "query": "q",
                "choices": ["a", "b"], "answer_index": 0, "answer": "a",
                "correct": c == "correct" and i < n,
            }
            for i in range(N_ITEMS)
        }
        cells[key] = {
            "rows": rows,
            "manifest": {
                "vis_layers": "9,18,-1",
                "arguments": {"no_text_branch": nox, "prompt_mode": "question", "question_field": "query"},
                "checkpoints": {
                    "mapping_txt": {"path": f"/o/{TEXT[text]}/mapping/pytorch_model.bin", "sha256": TEXT[text]},
                    "mapping_vis": {"path": f"/o/{VISION[vision]}/mapping/pytorch_model.bin", "sha256": VISION[vision]},
                },
                "data": b + t, "id_universe_sha256": "u", "image_sha256": "i",
                "shuffle_map_hashes": {"0": "a"}, "decoding": {}, "environment": {"python": "3.11.5"},
                "frozen_models": {"llm": {"revision": "r"}},
            },
        }
    return cells


class BlockBTests(unittest.TestCase):
    def test_grid_is_376_cells(self):
        self.assertEqual(len(required_cells()), 376)
        validate_grid(fixture())

    def test_grid_fails_closed(self):
        cells = fixture()
        del cells[next(iter(sorted(cells)))]
        with self.assertRaisesRegex(ValueError, "complete 376-cell grid"):
            validate_grid(cells)

        cells = fixture()
        key = ("cvqa", "jv", arm("T3_id", "V2"), "correct")
        cells[key]["manifest"]["checkpoints"]["mapping_txt"]["path"] = "/o/stage3_bn_dcl/mapping/pytorch_model.bin"
        with self.assertRaisesRegex(ValueError, "mapping_txt is not loaded from stage3_id_v4"):
            validate_grid(cells)

        cells = fixture()
        cells[("cvqa", "mn", arm("T1_bn", "V3_id"), "shuffled1")]["rows"].pop("7")
        with self.assertRaisesRegex(ValueError, "unpaired"):
            validate_grid(cells)

        cells = fixture()
        cells[("cvqa", "ga", arm("T1_bn", "V2", True), "correct")]["manifest"]["arguments"]["no_text_branch"] = False
        with self.assertRaisesRegex(ValueError, "text-branch flag"):
            validate_grid(cells)

    def test_null_factorial_is_exactly_zero_and_passes_nothing(self):
        report = analyse(fixture(), B=200)
        contrasts = report["panels"]["primary"]["contrasts"]
        for name in ("D_T", "D_V", "D_T-D_V", "P4"):
            for ep in ("U", "grounding"):
                self.assertEqual(contrasts[name][ep]["estimate"], 0.0, (name, ep))
        self.assertFalse(report["gates"]["G2_follows_text_branch"])
        self.assertFalse(report["gates"]["exploratory_follows_vision_branch"])

    def test_text_localisation_passes_g2(self):
        report = analyse(fixture(text_effect=16), B=300)
        g = report["panels"]["primary"]["contrasts"]
        self.assertAlmostEqual(g["D_T"]["grounding"]["estimate"], 100 * 16 / N_ITEMS)
        self.assertAlmostEqual(g["D_V"]["grounding"]["estimate"], 0.0)
        self.assertTrue(report["gates"]["G2_follows_text_branch"])
        self.assertFalse(report["gates"]["exploratory_follows_vision_branch"])

    def test_vision_localisation_fails_g2_and_reads_as_vision(self):
        report = analyse(fixture(vision_effect=16), B=300)
        g = report["panels"]["primary"]["contrasts"]
        self.assertAlmostEqual(g["D_V"]["grounding"]["estimate"], 100 * 16 / N_ITEMS)
        self.assertAlmostEqual(g["D_T-D_V"]["grounding"]["estimate"], -100 * 16 / N_ITEMS)
        self.assertFalse(report["gates"]["G2_follows_text_branch"])
        self.assertTrue(report["gates"]["exploratory_follows_vision_branch"])

    def test_p4_sign_and_independence_from_p3(self):
        report = analyse(fixture(matched_effect=16), B=200)
        g = report["panels"]["primary"]["contrasts"]
        self.assertAlmostEqual(g["P4"]["grounding"]["estimate"], 100 * 16 / N_ITEMS)
        self.assertAlmostEqual(g["D_T"]["grounding"]["estimate"], 0.0)
        self.assertAlmostEqual(g["D_V"]["grounding"]["estimate"], 0.0)

    def test_no_text_control_sign(self):
        report = analyse(fixture(nox_drop=8), B=200)
        control = report["controls"]["no_text_branch_primary"]["T3_bn__V3_bn noX-minus-withX"]
        self.assertAlmostEqual(control["U"]["estimate"], -100 * 8 / N_ITEMS)

    def test_insertion_order_does_not_change_the_report(self):
        cells = fixture(text_effect=8, vision_effect=4)
        keys = list(cells)
        random.Random(3).shuffle(keys)
        reordered = {k: cells[k] for k in keys}
        self.assertEqual(analyse(cells, B=150), analyse(reordered, B=150))


if __name__ == "__main__":
    unittest.main()
