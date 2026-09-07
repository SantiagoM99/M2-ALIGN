"""Adversarial contract tests for the stdlib-only S1 analysis helpers.

Unlike ``test_invariance.py``, these tests use tiny synthetic fixtures and do
not require the real evaluation data. Run from any directory:

    python3 Approach2/analysis/test_fail_closed.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _boot import boot_stratified, load_items  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def expect_exit(label: str, needle: str, fn) -> None:
    try:
        fn()
    except SystemExit as exc:
        if needle not in str(exc):
            raise AssertionError(f"{label}: expected {needle!r}, got {exc!r}") from exc
    else:
        raise AssertionError(f"{label}: expected SystemExit")


def run_fails(cwd: Path, script: str, args: list[str], needle: str) -> None:
    p = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    msg = p.stdout + p.stderr
    assert p.returncode != 0, f"{script} unexpectedly succeeded: {msg}"
    assert needle in msg, f"{script}: expected {needle!r}, got {msg!r}"


def make_x1_arm_set(root: Path, item_id: str = "10_0") -> None:
    for tag in ("cvqa_jv_zsid", "cvqa_jv_zsid_BLIND", "cvqa_jv_zsbn", "cvqa_jv_zsbn_BLIND"):
        write_jsonl(root / f"eval_{tag}.jsonl", [{"id": item_id, "correct": True}])


def make_e3_target(root: Path, lang: str, item_id: str) -> None:
    for tag in (f"xgqa_{lang}_v4", f"xgqa_{lang}_BLIND_v4", f"xgqa_{lang}_zsbn", f"xgqa_{lang}_zsbn_BLIND"):
        write_jsonl(root / f"eval_{tag}.jsonl", [{"id": item_id, "correct": True}])


def main() -> None:
    # Recover the draws of stratum 'a' after adding a lexically earlier,
    # constant stratum. A shared sequential RNG would fail this equality.
    variable = {"a": {"1": [0], "2": [1]}}
    with_constant = {"0": {"1": [0], "2": [0]}, **variable}
    solo = boot_stratified(variable, 80, 7, "rng-isolation")
    pooled = boot_stratified(with_constant, 80, 7, "rng-isolation")
    assert solo == [2 * value for value in pooled], "stratum RNG changed when another stratum was added"
    reversed_clusters = {"a": {"2": [1], "1": [0]}}
    assert solo == boot_stratified(reversed_clusters, 80, 7, "rng-isolation")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_cwd = Path.cwd()
        os.chdir(root)
        try:
            write_jsonl(root / "eval_bad.jsonl", [{"id": "x"}])
            expect_exit("missing correct", "must be a JSON boolean", lambda: load_items("bad"))
            write_jsonl(root / "eval_bad.jsonl", [{"id": "x", "correct": "false"}])
            expect_exit("non-bool correct", "must be a JSON boolean", lambda: load_items("bad"))
            write_jsonl(root / "eval_bad.jsonl", [{"id": "x", "correct": True}, {"id": "x", "correct": False}])
            expect_exit("duplicate id", "duplicate id", lambda: load_items("bad"))
            write_jsonl(root / "eval_bad.jsonl", [{"id": None, "correct": True}])
            expect_exit("empty id", "missing or empty id", lambda: load_items("bad"))
            write_jsonl(root / "eval_bad.jsonl", [])
            expect_exit("empty file", "empty result file", lambda: load_items("bad"))
        finally:
            os.chdir(old_cwd)

        run_fails(root, "x1_did.py", ["--targets", "jv", "--pool", "jv", "jv"], "repeated target in --pool")
        run_fails(root, "x1_did.py", ["--targets", "jv", "--pool", "ga"], "--pool members not in --targets")
        make_x1_arm_set(root, "malformed")
        run_fails(root, "x1_did.py", ["--targets", "jv", "--pool", "jv", "--boot", "2"], "cannot cluster")
        make_x1_arm_set(root)
        write_jsonl(root / "eval_cvqa_jv_zsbn_BLIND.jsonl", [{"id": "10_1", "correct": True}])
        run_fails(root, "x1_did.py", ["--targets", "jv", "--pool", "jv", "--boot", "2"], "do not share identical item sets")

        make_e3_target(root, "de", "1")
        duplicate_map = root / "duplicate_map.jsonl"
        write_jsonl(duplicate_map, [{"question_id": "1", "image_id": "a"}, {"question_id": "1", "image_id": "b"}])
        run_fails(root, "e3_noninferiority.py", ["--targets", "de", "--cluster-map", str(duplicate_map), "--boot", "2"], "duplicate question_id")

        make_e3_target(root, "ru", "2")
        complete_map = root / "map.jsonl"
        write_jsonl(complete_map, [{"question_id": "1", "image_id": "a"}, {"question_id": "2", "image_id": "b"}])
        run_fails(root, "e3_noninferiority.py", ["--targets", "de", "ru", "--cluster-map", str(complete_map), "--boot", "2"], "id universe differs")

        write_jsonl(complete_map, [{"question_id": "2", "image_id": "b"}])
        run_fails(root, "e3_noninferiority.py", ["--targets", "de", "--cluster-map", str(complete_map), "--boot", "2"], "ids have no image")

    print("fail-closed: OK (RNG isolation, schema, duplicate/pool/id/arm/map/universe guards)")


if __name__ == "__main__":
    main()
