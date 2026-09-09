import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_contract import atomic_json, digest, file_sha, git_state, result_complete
from eval_runtime import Runtime
from eval_matrix import compare_predictions


class MatrixTests(unittest.TestCase):
    def test_pristine_cells_and_only_requested_branch(self):
        import torch
        from types import SimpleNamespace
        from model import Mapping

        with tempfile.TemporaryDirectory() as d:
            txt, vis = Mapping(3, 4), Mapping(3, 4)

            class Model:
                mapping_txt = txt
                mapping_vis = vis

                def eval(self):
                    return self

            runtime = Runtime.__new__(Runtime)
            runtime.model = Model()
            runtime.pristine = {
                k: copy.deepcopy(getattr(runtime.model, k).state_dict())
                for k in ("mapping_txt", "mapping_vis")
            }
            text = copy.deepcopy(txt.state_dict())
            vision = copy.deepcopy(vis.state_dict())
            text["gate"].fill_(2)
            vision["gate"].fill_(3)
            # Opposite branch payloads are deliberately invalid: must never be loaded.
            tp = Path(d) / "text.pt"
            vp = Path(d) / "vision.pt"
            torch.save(
                {"mapping_txt": text, "mapping_vis": {"wrong": torch.ones(1)}}, tp
            )
            torch.save(
                {"mapping_vis": vision, "mapping_txt": {"wrong": torch.ones(1)}}, vp
            )
            a = SimpleNamespace(
                txt_ckpt=str(tp),
                vis_ckpt=str(vp),
                ckpt=None,
                extra_ckpt=None,
                no_text_branch=False,
                no_image=False,
                max_gen_len=8,
            )
            runtime.reset(a)
            self.assertEqual(txt.gate.item(), 2)
            self.assertEqual(vis.gate.item(), 3)
            del text["gate"]
            torch.save({"mapping_txt": text}, tp)
            runtime.reset(a)
            self.assertEqual(txt.gate.item(), 1)
            self.assertEqual(vis.gate.item(), 3)
            a.no_text_branch = True
            runtime.reset(a)
            self.assertTrue(
                torch.equal(txt.gate, runtime.pristine["mapping_txt"]["gate"])
            )

    def test_prediction_parity_checks_gold_and_universe(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a"
            b = Path(d) / "b"
            row = {
                "id": "x",
                "query": "q",
                "choices": ["a", "b"],
                "answer_index": 0,
                "pred_index": 0,
                "correct": True,
            }
            a.write_text(json.dumps(row) + "\n")
            b.write_text(json.dumps(row) + "\n")
            compare_predictions(a, b)
            row["pred_index"] = 1
            b.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                compare_predictions(a, b)
            row["id"] = "different"
            b.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                compare_predictions(a, b)

    def test_guard_dirty_and_ancestry(self):
        with tempfile.TemporaryDirectory() as d:

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", d, *args], text=True, stderr=subprocess.DEVNULL
                ).strip()

            git("init")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            p = Path(d) / "a"
            p.write_text("freeze")
            git("add", "a")
            git("commit", "-m", "freeze")
            freeze = git("rev-parse", "HEAD")
            with patch("s1_contract.SPEC_SHA", freeze):
                self.assertFalse(git_state(d)["dirty"])
                p.write_text("dirty")
                with self.assertRaisesRegex(ValueError, "clean"):
                    git_state(d)
                git("checkout", "--", "a")
                with self.assertRaisesRegex(ValueError, "code SHA"):
                    git_state(d, expected_code="bad")
                git("checkout", "--orphan", "unrelated")
                git("add", "a")
                git("commit", "-m", "unrelated")
                with self.assertRaisesRegex(ValueError, "freeze"):
                    git_state(d)

    def test_completion_requires_all_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out"
            p.write_text("{}\n")
            m = {"spec_sha": "test"}
            self.assertFalse(result_complete(p, m))
            atomic_json(str(p) + ".summary.json", {"scored": 1})
            atomic_json(
                str(p) + ".complete.json",
                {
                    "manifest_sha256": digest(m),
                    "predictions_sha256": file_sha(p),
                    "summary_sha256": file_sha(str(p) + ".summary.json"),
                },
            )
            self.assertTrue(result_complete(p, m))
            p.write_text("partial")
            with self.assertRaises(ValueError):
                result_complete(p, m)


if __name__ == "__main__":
    unittest.main()
