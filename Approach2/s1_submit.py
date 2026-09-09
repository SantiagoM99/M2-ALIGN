"""Validate and freeze inputs before sbatch; defaults to prepare-only (no GPU)."""

import argparse
from pathlib import Path
import subprocess

from eval_matrix import read_plan, validate_submission
from eval_runtime import prepare, make_manifest
from s1_contract import ROOT, SPEC_SHA, atomic_json, digest, file_sha, git_state


def build_submission(plan_path):
    state = git_state()
    plan, cells = read_plan(plan_path)
    # All analysis/instrument components must exist before submission.
    for rel in (
        "Approach2/analysis/block_a.py",
        "Approach2/shuffle/generate.py",
        "Approach2/tests/test_s1_inputs.py",
        "Approach2/tests/test_s1_matrix.py",
        "Approach2/analysis/test_block_a.py",
    ):
        if not (ROOT / rel).is_file():
            raise ValueError(f"missing required implementation {rel}")
    manifests = {}
    grid = {}
    image_cache = {}
    for cell, a in cells:
        registry = Path(a.shuffle_registry).resolve()
        try:
            rel = registry.relative_to(ROOT)
        except ValueError:
            raise ValueError(
                "shuffle registry must live under tracked Approach2/shuffle"
            )
        if not str(rel).startswith("Approach2/shuffle/"):
            raise ValueError("registry must be under Approach2/shuffle")
        subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(rel)],
            check=True,
            capture_output=True,
        )
        prepared = prepare(a, image_cache)
        m = make_manifest(a, prepared, state)
        manifests[cell["id"]] = m
        key = (a.benchmark, a.target, a.arm, m["condition"])
        if key in grid:
            raise ValueError("duplicate target/arm/condition")
        grid[key] = {"manifest": m, "rows": {str(r["id"]): r for r in prepared["rows"]}}
    if plan["block"] == "A":
        from analysis.block_a import validate_grid

        validate_grid(grid)
    git_state(expected_code=state["code_sha"])
    return {
        **state,
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": digest(plan),
        "manifests": manifests,
        "parity_hashes": {
            p["cell_id"]: file_sha(p["expected_path"]) for p in plan.get("parity", [])
        },
        "block_a_report_sha256": file_sha(plan["block_a_report"])
        if plan.get("block_a_report")
        else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan")
    p.add_argument("--submission", required=True)
    p.add_argument(
        "--resume", action="store_true", help="Resubmit the same immutable submission"
    )
    p.add_argument(
        "--dependency", help="Optional SLURM dependency, e.g. afterany:12345"
    )
    p.add_argument(
        "--submit", action="store_true", help="Actually call sbatch after validation"
    )
    a = p.parse_args()
    target = Path(a.submission).resolve()
    if a.resume:
        submission = validate_submission(target)
    else:
        if target.exists():
            raise ValueError("submission already exists; use --resume")
        if not a.plan:
            p.error("--plan is required unless --resume is used")
        submission = build_submission(a.plan)
        atomic_json(target, submission)
    if a.submit:
        git_state(expected_code=submission["code_sha"])
        command = ["sbatch", "--parsable"]
        if a.dependency:
            command.extend(["--dependency", a.dependency])
        out = subprocess.check_output(
            [*command, str(ROOT / "Approach2/job-scripts/s1_eval.sh"), str(target)],
            text=True,
        ).strip()
        atomic_json(
            str(target) + f".receipt.{out.split(';')[0]}.json",
            {
                "slurm_job_id": out,
                "submission_sha256": file_sha(target),
                "spec_sha": SPEC_SHA,
                "code_sha": submission["code_sha"],
            },
        )
        print(out)
    else:
        print(f"Prepared {target}; no job submitted")


if __name__ == "__main__":
    main()
