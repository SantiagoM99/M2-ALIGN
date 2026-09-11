"""Execute a submitted S1 matrix with shared towers and pristine branch swaps."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_runtime import Runtime, evaluate, make_manifest, parser, prepare
from s1_contract import (
    SPEC_SHA,
    atomic_json,
    digest,
    file_sha,
    git_state,
    read_json,
    read_rows,
    result_complete,
)


def read_plan(path):
    plan = read_json(path)
    if plan.get("spec_sha") != SPEC_SHA or plan.get("block") not in ("A", "B"):
        raise ValueError("plan requires exact S1 spec SHA and block A or B")
    cells = plan.get("cells", [])
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("empty matrix or duplicate cell id")
    result = []
    for cell in cells:
        if cell["benchmark"] not in ("cvqa", "xgqa"):
            raise ValueError("unsupported benchmark")
        a = parser(cell["benchmark"]).parse_args(cell["argv"])
        if not a.s1 or a.experiment_id != cell["id"]:
            raise ValueError("every matrix cell must be S1 with matching experiment id")
        result.append((cell, a))
    if len({a.output_path for _, a in result}) != len(result):
        raise ValueError("cells must have distinct output paths")
    tower_keys = (
        "mt_path",
        "vis_path",
        "llm_path",
        "vis_layers",
        "max_vis_tokens",
        "local_files_only",
    )
    for _, a in result[1:]:
        if any(getattr(a, k) != getattr(result[0][1], k) for k in tower_keys):
            raise ValueError("shared runtime requires identical tower configuration")
    parity = plan.get("parity", [])
    known = {c["id"]: a for c, a in result}
    if any(p["cell_id"] not in known for p in parity):
        raise ValueError("parity references unknown cell")
    if plan["block"] == "B":
        if not plan.get("block_a_report"):
            raise ValueError("Block B requires a Block A G0 report")
        gate = read_json(plan["block_a_report"])
        if (
            gate.get("spec_sha") != SPEC_SHA
            or gate.get("gates", {}).get("G0") is not True
        ):
            raise ValueError("Block B requires G0 to pass under the frozen S1 spec")
        targets = {a.target for _, a in result if a.benchmark == "cvqa"}
        # bn/bn is only referenceable on CVQA-bn (stage3_bn_dcl results exist
        # there alone); id/id on every other target (stage3_id_v4 `zsid` files).
        required = {
            (s, t, cond)
            for s in ("bn", "id")
            for t in targets
            for cond in ("correct", "gray")
            if (s == "bn") == (t == "bn")
        }
        actual = set()
        for p in parity:
            a = known[p["cell_id"]]
            from eval_runtime import condition
            from s1_contract import branch_paths

            paths = branch_paths(a)
            if (
                a.benchmark != "cvqa"
                or a.no_text_branch
                or a.prompt_mode != "question"
                or paths["mapping_txt"] != paths["mapping_vis"]
            ):
                raise ValueError("parity must test a full matched checkpoint")
            expected_name = "stage3_bn_dcl" if p["donor"] == "bn" else "stage3_id_v4"
            if expected_name not in Path(paths["mapping_txt"]).parts:
                raise ValueError("parity donor does not match checkpoint lineage")
            actual.add((p["donor"], a.target, condition(a)))
        if actual != required:
            raise ValueError(
                "Block B requires bn/bn parity on CVQA-bn and id/id parity on every other "
                "CVQA target, correct and gray"
            )
    return plan, result


PARITY_CHECKPOINT = {"bn": "stage3_bn_dcl", "id": "stage3_id_v4"}


def check_parity_lineage(plan):
    """Every legacy reference must have been produced by the checkpoint the cell evaluates."""
    for p in plan.get("parity", []):
        summary_path = p["expected_path"] + ".summary.json"
        if not Path(p["expected_path"]).is_file() or not Path(summary_path).is_file():
            raise ValueError(f"parity reference or its summary is missing: {p['expected_path']}")
        ckpt = str(read_json(summary_path).get("ckpt", ""))
        expected = PARITY_CHECKPOINT[p["donor"]]
        if expected not in Path(ckpt).parts:
            raise ValueError(
                f"parity reference {p['expected_path']} was produced by {ckpt or 'an unknown checkpoint'}, "
                f"not {expected}"
            )


def compare_predictions(actual, expected):
    a = {str(r["id"]): r for r in read_rows(actual)}
    e = {str(r["id"]): r for r in read_rows(expected)}
    if a.keys() != e.keys():
        raise ValueError("parity item universe mismatch")
    for i in a:
        fields = (
            ("query", "choices", "answer_index", "pred_index", "correct")
            if "choices" in e[i]
            else ("query", "answer", "pred", "correct")
        )
        if any(a[i].get(k) != e[i].get(k) for k in fields):
            raise ValueError(f"prediction parity mismatch on item {i}")


def validate_submission(path):
    submission = read_json(path)
    git_state(expected_code=submission["code_sha"])
    if submission["spec_sha"] != SPEC_SHA:
        raise ValueError("wrong spec SHA")
    if digest(submission["plan"]) != submission["plan_sha256"]:
        raise ValueError("submitted plan was altered")
    return submission


def run(submission_path):
    submission = validate_submission(submission_path)
    # Parse the exact submitted plan, not a mutable source file.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plan.json"
        atomic_json(p, submission["plan"])
        plan, cells = read_plan(p)
    prepared = {}
    image_cache = {}
    manifests = {}
    for cell, a in cells:
        prep = prepare(a, image_cache)
        m = make_manifest(
            a,
            prep,
            {"spec_sha": SPEC_SHA, "code_sha": submission["code_sha"], "dirty": False},
        )
        frozen = submission["manifests"][cell["id"]]
        changed = sorted(
            k
            for k in set(m) | set(frozen)
            if k != "slurm_job_id" and m.get(k) != frozen.get(k)
        )
        if changed:
            raise ValueError(
                f"{cell['id']}: changed after submission: {', '.join(changed)}"
            )
        old = Path(a.output_path + ".manifest.json")
        if old.exists():
            previous = read_json(old)
            if {k: v for k, v in previous.items() if k != "slurm_job_id"} != {
                k: v for k, v in m.items() if k != "slurm_job_id"
            }:
                raise ValueError("existing result belongs to another experiment")
            if result_complete(a.output_path, previous):
                m = previous
        prepared[cell["id"]] = prep
        manifests[cell["id"]] = m
    check_parity_lineage(plan)
    for p in plan.get("parity", []):
        if file_sha(p["expected_path"]) != submission["parity_hashes"][p["cell_id"]]:
            raise ValueError("legacy parity reference changed after submission")
    if (
        plan.get("block_a_report")
        and file_sha(plan["block_a_report"]) != submission["block_a_report_sha256"]
    ):
        raise ValueError("Block A gate report changed after submission")
    runtime = None
    lookup = {c["id"]: (c, a) for c, a in cells}

    def execute(cid):
        nonlocal runtime
        _, a = lookup[cid]
        m = manifests[cid]
        if not result_complete(a.output_path, m) and runtime is None:
            runtime = Runtime(
                cells[0][1],
                use_text=any(not x.no_text_branch for _, x in cells),
                use_vision=any(not x.no_image for _, x in cells),
            )
        evaluate(a, runtime, prepared[cid], m)

    # No crossed matrix cell can run before every matched parity cell passes.
    for p in plan.get("parity", []):
        execute(p["cell_id"])
        compare_predictions(lookup[p["cell_id"]][1].output_path, p["expected_path"])
    for cell, a in cells:
        execute(cell["id"])
    atomic_json(
        str(submission_path) + ".complete.json",
        {
            "spec_sha": SPEC_SHA,
            "code_sha": submission["code_sha"],
            "submission_sha256": file_sha(submission_path),
            "cells": list(lookup),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True)
    run(p.parse_args().submission)
