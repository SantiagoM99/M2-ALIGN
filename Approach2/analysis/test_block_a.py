"""Synthetic Block A gate, pairing, order and fail-closed tests (stdlib)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_a import CONDITIONS, analyse, validate_grid


def fixture(ground=True, branch_used=False):
    cells = {}
    for b, targets, arms in [
        ("cvqa", ["jv", "mn", "ga", "si"], ["A1", "A2", "A3", "A4"]),
        ("xgqa", ["bn", "de", "ko"], ["A1", "A2", "A3"]),
        ("xgqa", ["en"], ["A4"]),
    ]:
        for t in targets:
            for a in arms:
                for c in CONDITIONS:
                    value = (c == "correct" or not ground) and not (
                        branch_used and a == "A2"
                    )
                    rows = {
                        str(i): {
                            "id": str(i),
                            "image_id": str(i // 2),
                            "subset": "s",
                            "query": "q",
                            "choices": ["a", "b"],
                            "answer_index": 0,
                            "answer": "a",
                            "correct": bool(value),
                        }
                        for i in range(8)
                    }
                    ck = {
                        "path": "/outputs/stage3_bn_dcl/mapping/pytorch_model.bin",
                        "sha256": "same",
                    }
                    m = {
                        "source": "bn",
                        "vis_layers": "9,18,-1",
                        "checkpoints": {"mapping_txt": ck, "mapping_vis": ck},
                        "arguments": {
                            "no_text_branch": a in ("A2", "A4"),
                            "prompt_mode": "instruction" if a == "A3" else "question",
                            "question_field": "english_query"
                            if a == "A4" and b == "cvqa"
                            else "query",
                        },
                        "data": f"{b}-{t}",
                        "id_universe_sha256": "same",
                        "image_sha256": "same",
                        "shuffle_map_hashes": "same",
                        "decoding": {},
                    }
                    cells[(b, t, a, c)] = {"manifest": m, "rows": rows}
    return cells


class BlockATests(unittest.TestCase):
    def test_identical_arms_dispense_and_joint_xgqa(self):
        r = analyse(fixture(), B=30)
        self.assertTrue(r["gates"]["G0"])
        self.assertEqual(r["gates"]["G1-I"], "dispensable")
        for ep in ("U", "grounding"):
            self.assertEqual(
                r["panels"]["primary"]["contrasts"]["A2-A1"][ep]["ci95"], [0, 0]
            )
        self.assertEqual(
            r["panels"]["xgqa-secondary"]["arms"]["A1"]["U"]["image_clusters"], 4
        )

    def test_used_and_g0_stop(self):
        self.assertEqual(
            analyse(fixture(branch_used=True), B=30)["gates"]["G1-I"], "used"
        )
        g = analyse(fixture(ground=False), B=30)["gates"]
        self.assertFalse(g["G0"])
        self.assertFalse(g["launch_B_C"])
        self.assertTrue(g["G1-I"].startswith("not evaluated"))

    def test_order_invariance(self):
        f = fixture()
        g = dict(reversed(list(f.items())))
        for cell in g.values():
            cell["rows"] = dict(reversed(list(cell["rows"].items())))
        self.assertEqual(analyse(f, B=30), analyse(g, B=30))

    def test_missing_cells_items_and_changed_config_abort(self):
        f = fixture()
        f.pop(next(iter(f)))
        with self.assertRaises(ValueError):
            validate_grid(f)
        f = fixture()
        f[("cvqa", "jv", "A2", "correct")]["rows"].pop("0")
        with self.assertRaises(ValueError):
            validate_grid(f)
        f = fixture()
        f[("cvqa", "jv", "A2", "correct")]["manifest"]["decoding"] = {"different": True}
        with self.assertRaises(ValueError):
            validate_grid(f)
        f = fixture()
        f[("xgqa", "ko", "A1", "correct")]["manifest"]["image_sha256"] = "different"
        with self.assertRaises(ValueError):
            validate_grid(f)


if __name__ == "__main__":
    unittest.main()
