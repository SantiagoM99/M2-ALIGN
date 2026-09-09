import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))
from s1_contract import (
    SPEC_SHA,
    atomic_json,
    digest,
    item_identity,
    read_json,
    file_sha,
)
from s1_plan import build
from eval_matrix import read_plan
from eval_runtime import condition, prepare, make_manifest, evaluate
from block_a import load_cells, analyse


class FakeRuntime:
    def predict(self, a, p):
        c = condition(a)
        for row in p["rows"]:
            r = {
                **row,
                **item_identity(row, a.benchmark),
                "correct": c == "correct",
                "condition": c,
            }
            r["assigned_image_id"] = (
                p["shuffled"]["items"][str(row["id"])]["assigned_image_id"]
                if p["shuffled"]
                else r["image_id"]
                if c == "correct"
                else None
            )
            if a.benchmark == "cvqa":
                r.update(pred_index=0 if r["correct"] else 1, scores=[-1.0, -2.0])
            else:
                r["pred"] = "yes" if r["correct"] else "no"
            yield r


class EndToEndTests(unittest.TestCase):
    def test_full_A_plan_manifests_predictions_analysis_and_tampering(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images = root / "images"
            images.mkdir()
            for i in range(4):
                Image.new("RGB", (4, 4), (i * 30, 0, 0)).save(images / f"{i}.jpg")
            panels = {}
            for b, ts in [
                ("xgqa", ["bn", "de", "ko", "en"]),
                ("cvqa", ["jv", "mn", "ga", "si"]),
            ]:
                panels[b] = {}
                for t in ts:
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
                        for i in range(8)
                    ]
                    p = root / f"{b}_{t}.jsonl"
                    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
                    panels[b][t] = {"data": str(p), "images": str(images)}
            atomic_json(root / "panels.json", panels)
            a = SimpleNamespace(
                panels=str(root / "panels.json"),
                checkpoints=str(root / "outputs"),
                block="A",
                maps_dir=str(root / "maps"),
                results=str(root / "results"),
                output=str(root / "plan.json"),
                block_a_report=None,
            )
            ck = root / "outputs/stage3_bn_dcl/mapping/pytorch_model.bin"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(b"fixture")
            with patch("s1_plan.ROOT", root):
                build(a)
            plan, cells = read_plan(a.output)
            self.assertEqual(len(cells), 156)
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
                    return_value={
                        "llm": {"revision": "fixture"},
                        "text": None,
                        "vision": None,
                    },
                ):
                    m = make_manifest(
                        args,
                        prep,
                        {"spec_sha": SPEC_SHA, "code_sha": "fixture", "dirty": False},
                    )
                sub["manifests"][cell["id"]] = m
                evaluate(args, FakeRuntime(), prep, m)
            atomic_json(root / "submission.json", sub)
            report = analyse(load_cells(root / "submission.json"), B=20)
            self.assertTrue(report["gates"]["G0"])
            self.assertEqual(report["gates"]["G1-I"], "dispensable")
            c, args = next(
                (c, args) for c, args in cells if condition(args) == "shuffled0"
            )
            path = Path(args.output_path)
            rr = path.read_text().splitlines()
            r = json.loads(rr[0])
            r["assigned_image_id"] = r["image_id"]
            rr[0] = json.dumps(r)
            path.write_text("\n".join(rr) + "\n")
            with self.assertRaisesRegex(ValueError, "completion hash"):
                load_cells(root / "submission.json")
            marker = read_json(str(path) + ".complete.json")
            marker["predictions_sha256"] = file_sha(path)
            atomic_json(str(path) + ".complete.json", marker)
            with self.assertRaisesRegex(ValueError, "assigned image"):
                load_cells(root / "submission.json")
            # B has 360 CVQA + 16 xGQA cells and twenty mandatory parity cells.
            panels["cvqa"]["bn"] = panels["cvqa"]["jv"]
            panels["xgqa"]["id"] = panels["xgqa"]["bn"]
            atomic_json(root / "panels.json", panels)
            atomic_json(
                root / "gate.json", {"spec_sha": SPEC_SHA, "gates": {"G0": True}}
            )
            a.block = "B"
            a.output = str(root / "planB.json")
            a.legacy_results = str(root / "legacy")
            a.block_a_report = str(root / "gate.json")
            with patch("s1_plan.ROOT", root):
                build(a)
            bp, bc = read_plan(a.output)
            self.assertEqual(len(bc), 376)
            self.assertEqual(len(bp["parity"]), 20)
            bp["parity"].pop()
            atomic_json(a.output, bp)
            with self.assertRaisesRegex(ValueError, "parity"):
                read_plan(a.output)


if __name__ == "__main__":
    unittest.main()
