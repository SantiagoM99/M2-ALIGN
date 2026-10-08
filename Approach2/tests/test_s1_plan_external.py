"""Synthetic tests for s1_plan's external arms: the frozen contract cannot be borrowed (stdlib)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_plan import FROZEN_C_ARMS, resolve_external_arms  # noqa: E402


class ExternalArmTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def finished(self, name):
        (self.root / name).mkdir(parents=True)
        (self.root / name / "complete.json").write_text("{}", encoding="utf-8")

    def test_a_finished_checkpoint_resolves(self):
        self.finished("cg50_seed13")
        self.assertEqual(
            resolve_external_arms(["CG50=cg50_seed13"], self.root),
            [{"arm": "CG50", "checkpoint_dir": "cg50_seed13"}],
        )

    def test_a_frozen_arm_name_is_refused(self):
        """Borrowing C1's name would put a non-S1 arm inside the frozen contract."""
        for name in FROZEN_C_ARMS:
            self.finished(f"dir_{name}")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "frozen S1 arm"):
                resolve_external_arms([f"{name}=dir_{name}"], self.root)

    def test_an_unfinished_checkpoint_is_refused(self):
        """The best checkpoint appears after epoch 1, so its existence is not completion."""
        (self.root / "cg50_seed13").mkdir()
        with self.assertRaisesRegex(ValueError, "has not finished training"):
            resolve_external_arms(["CG50=cg50_seed13"], self.root)

    def test_a_repeated_name_is_refused(self):
        self.finished("a")
        self.finished("b")
        with self.assertRaisesRegex(ValueError, "given twice"):
            resolve_external_arms(["X=a", "X=b"], self.root)

    def test_a_name_already_taken_by_this_plan_is_refused(self):
        self.finished("a")
        with self.assertRaisesRegex(ValueError, "given twice"):
            resolve_external_arms(["X=a"], self.root, taken={"X"})

    def test_a_malformed_spec_is_refused(self):
        for spec in ("CG50", "=dir", "CG50=", ""):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                resolve_external_arms([spec], self.root)

    def test_no_specs_is_no_arms(self):
        self.assertEqual(resolve_external_arms(None, self.root), [])
        self.assertEqual(resolve_external_arms([], self.root), [])


if __name__ == "__main__":
    unittest.main()
