"""Synthetic tests for image_leakage.py: content equality, prefixes, fail-closed (stdlib)."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_leakage import collisions, index  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "image_leakage.py"


class ImageLeakageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.train = self.root / "train"
        self.evaluation = self.root / "eval"
        self.train.mkdir()
        self.evaluation.mkdir()

    def write(self, directory, name, content):
        (directory / name).write_bytes(content)

    def test_the_same_bytes_under_different_names_collide(self):
        self.write(self.train, "cg_a.jpg", b"photo")
        self.write(self.evaluation, "12345.jpg", b"photo")
        found = collisions(index(self.train), index(self.evaluation))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["train"], ["cg_a.jpg"])
        self.assertEqual(found[0]["eval"], ["12345.jpg"])

    def test_different_bytes_do_not_collide(self):
        self.write(self.train, "cg_a.jpg", b"photo one")
        self.write(self.evaluation, "12345.jpg", b"photo two")
        self.assertEqual(collisions(index(self.train), index(self.evaluation)), [])

    def test_the_prefix_selects_only_the_built_images(self):
        """CG images share a directory with GQA's, so the prefix is what separates them."""
        self.write(self.train, "cg_a.jpg", b"cultural")
        self.write(self.train, "99.jpg", b"gqa image")
        self.write(self.evaluation, "12345.jpg", b"gqa image")
        self.assertEqual(collisions(index(self.train, "cg_"), index(self.evaluation)), [])
        self.assertEqual(len(collisions(index(self.train), index(self.evaluation))), 1)

    def test_a_missing_directory_fails_closed(self):
        with self.assertRaises(SystemExit):
            index(self.root / "absent")

    def test_a_prefix_matching_nothing_fails_closed(self):
        """Silently checking zero images would pass the gate without testing anything."""
        self.write(self.train, "99.jpg", b"gqa image")
        with self.assertRaises(SystemExit):
            index(self.train, "cg_")

    def test_the_cli_exits_non_zero_on_a_leak(self):
        self.write(self.train, "cg_a.jpg", b"photo")
        self.write(self.evaluation, "12345.jpg", b"photo")
        out = self.root / "report.json"
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--train-dir", str(self.train),
             "--eval-dir", str(self.evaluation), "--output", str(out)],
            capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("LEAK", run.stdout)
        self.assertEqual(len(json.loads(out.read_text())["collisions"]), 1)

    def test_the_cli_exits_zero_and_reports_when_clean(self):
        self.write(self.train, "cg_a.jpg", b"cultural")
        self.write(self.evaluation, "12345.jpg", b"gqa image")
        out = self.root / "report.json"
        run = subprocess.run(
            [sys.executable, str(SCRIPT), "--train-dir", str(self.train),
             "--eval-dir", str(self.evaluation), "--output", str(out)],
            capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("no file-level overlap", run.stdout)
        self.assertEqual(json.loads(out.read_text())["collisions"], [])


if __name__ == "__main__":
    unittest.main()
