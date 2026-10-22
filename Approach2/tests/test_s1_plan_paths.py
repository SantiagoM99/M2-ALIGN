"""A plan must freeze every path it hands a cell, including the CVQA inventory (stdlib + PIL).

Job 24806466 refused all 60 CG50 cells on
`s1_C_cvqa_jv_C1_correct: changed after submission: arguments`. The cause was
not the arm: `--cvqa-inventory` defaults to a path derived from
`eval_runtime.__file__`, the submission is prepared in the clone and the
allocation runs from a worktree of the same commit, so the two namespaces
disagreed on a string while pointing at identical bytes. The manifest freezes
`vars(a)` whole and has no notion of a path that may move, which is the right
default -- so the plan has to pass the path rather than let it be inferred.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_contract import atomic_json  # noqa: E402
from eval_matrix import read_plan  # noqa: E402
from s1_plan import build  # noqa: E402


def block_a_plan(root: Path) -> Path:
    """The smallest real plan that contains CVQA cells."""
    from PIL import Image

    images = root / "images"
    images.mkdir()
    for i in range(4):
        Image.new("RGB", (4, 4), (i * 30, 0, 0)).save(images / f"{i}.jpg")
    panels = {}
    for b, targets in [("xgqa", ["bn", "de", "ko", "en"]), ("cvqa", ["jv", "mn", "ga", "si"])]:
        panels[b] = {}
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
                for i in range(8)
            ]
            path = root / f"{b}_{t}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            panels[b][t] = {"data": str(path), "images": str(images)}
    atomic_json(root / "panels.json", panels)
    checkpoint = root / "outputs/stage3_bn_dcl/mapping/pytorch_model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    a = SimpleNamespace(
        panels=str(root / "panels.json"),
        checkpoints=str(root / "outputs"),
        block="A",
        maps_dir=str(root / "maps"),
        results=str(root / "results"),
        output=str(root / "plan.json"),
        block_a_report=None,
    )
    with patch("s1_plan.ROOT", root):
        build(a)
    return Path(a.output)


class PlanPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.plan_path = block_a_plan(self.root)

    def test_every_cvqa_cell_carries_the_inventory_path(self):
        plan = json.loads(self.plan_path.read_text())
        cvqa = [c for c in plan["cells"] if c["benchmark"] == "cvqa"]
        self.assertTrue(cvqa)
        for cell in cvqa:
            with self.subTest(cell=cell["id"]):
                self.assertIn("--cvqa-inventory", cell["argv"])
                value = cell["argv"][cell["argv"].index("--cvqa-inventory") + 1]
                self.assertTrue(Path(value).is_absolute())
                # resolve() on the planner side, and /var is a symlink on macOS:
                # compare the physical spelling, which is the whole point here.
                self.assertTrue(value.startswith(str(self.root.resolve())), value)

    def test_xgqa_cells_do_not_carry_it(self):
        """Its parser has no such flag; passing it anyway would fail to parse."""
        plan = json.loads(self.plan_path.read_text())
        for cell in plan["cells"]:
            if cell["benchmark"] == "xgqa":
                self.assertNotIn("--cvqa-inventory", cell["argv"])

    def test_the_namespace_survives_a_different_working_directory(self):
        """The invariant the manifest comparison actually depends on."""
        here = os.getcwd()
        _, first = read_plan(self.plan_path)
        try:
            os.chdir(self.root)
            _, second = read_plan(self.plan_path)
        finally:
            os.chdir(here)
        for (_, a), (_, b) in zip(first, second):
            self.assertEqual(vars(a), vars(b))


if __name__ == "__main__":
    unittest.main()
