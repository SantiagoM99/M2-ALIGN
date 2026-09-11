"""Build `block_d.py`'s input from a finished S1 Block D submission (stdlib).

Two things have to come together before the D-donor analysis can run, and they
come from different places:

* the **damage** of each donor, its predictor, which is read from that donor's
  OWN stage-1 and stage-3 pair-alignment files as
  ``mean over T of [margin_stage1_S(S->T) - margin_stage3_S(S->T)]`` on the
  centered margin, positive meaning deterioration. It is an input, fixed and
  committed before any outcome is read;
* the **outcome** of each (donor, target) cell, built from the Block D matrix:
  per item, ``ground`` is the correct-image score minus the mean of the three
  shuffled-image scores, and ``utility`` is the correct-image score.

Every cell is authenticated the way Block A's are, by `block_a.load_cells`:
manifest, completion hashes, item universe, regenerated shuffle maps and the
assigned image of every row. Item identities keep their subset and image, which
is what lets the analysis resample images jointly across donors within a
target/subset stratum.

    python3 Approach2/analysis/block_d_input.py \\
      --submission Approach2/outputs/s1_D.submission.json \\
      --alignment-dir Approach2/results \\
      --output Approach2/outputs/s1_D.input.json
    python3 Approach2/analysis/block_d.py --from-json Approach2/outputs/s1_D.input.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from block_a import load_cells  # noqa: E402
from block_d import SCHEMA_VERSION  # noqa: E402
from s1_contract import atomic_json, read_json  # noqa: E402

# Each donor's own pre-VQA and post-VQA text mapping. Bengali's stage 1 is
# `stage1` and its stage 3 is the dcl run, the checkpoint every transfer number
# so far came from; the other six are the v4 rounds. Must stay identical to the
# donor table in s1_plan.py.
DONORS = {
    "bn": ("stage1", "stage3_bn_dcl"),
    "id": ("stage1_id", "stage3_id_v4"),
    "ru": ("stage1_ru", "stage3_ru_v4"),
    "zh": ("stage1_zh", "stage3_zh_v4"),
    "de": ("stage1_de", "stage3_de_v4"),
    "pt": ("stage1_pt", "stage3_pt_v4"),
    "ko": ("stage1_ko", "stage3_ko_v4"),
}
TARGETS = ("jv", "mn", "ga", "si")
SHUFFLE_CONDITIONS = ("shuffled0", "shuffled1", "shuffled2")
NLLB = {
    "bn": "ben_Beng", "id": "ind_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "de": "deu_Latn", "pt": "por_Latn", "ko": "kor_Hang",
}


def fail(message: str):
    raise SystemExit(f"block_d_input: {message}")


def centered_margin(path: Path, source: str, target: str) -> float:
    report = read_json(path)
    matrix = report.get("matrix", {}).get("centered")
    if not isinstance(matrix, dict):
        fail(f"{path}: no centered matrix")
    row = matrix.get(source)
    if not isinstance(row, dict) or target not in row:
        fail(f"{path}: no centered margin for {source} -> {target}")
    value = row[target].get("margin")
    if not isinstance(value, (int, float)):
        fail(f"{path}: margin for {source} -> {target} is not a number")
    return float(value)


def damage(alignment_dir: Path, donor: str, targets) -> dict:
    """Alignment lost between the donor's own stage 1 and stage 3, per target."""
    stage1, stage3 = DONORS[donor]
    paths = {
        "stage1": alignment_dir / f"pairalign_{stage1}.json",
        "stage3": alignment_dir / f"pairalign_{stage3}.json",
    }
    for stage, path in paths.items():
        if not path.is_file():
            fail(
                f"{path} is missing: score {donor}'s {stage} checkpoint first with "
                f"CKPTS='{stage1 if stage == 'stage1' else stage3}' "
                "bash Approach2/job-scripts/pair_alignment.sh"
            )
    per_target = {
        t: centered_margin(paths["stage1"], donor, t)
        - centered_margin(paths["stage3"], donor, t)
        for t in targets
    }
    return {
        "value": sum(per_target.values()) / len(per_target),
        "per_target": per_target,
        "stage1_label": stage1,
        "stage3_label": stage3,
    }


def endpoints(cells, donor: str, target: str) -> dict:
    """Per item: grounding as correct minus the mean of the three shuffles."""
    arm = "D_" + donor
    try:
        correct = cells[("cvqa", target, arm, "correct")]
    except KeyError:
        fail(f"no correct-image cell for donor {donor} on {target}")
    shuffles = []
    for condition in SHUFFLE_CONDITIONS:
        try:
            shuffles.append(cells[("cvqa", target, arm, condition)])
        except KeyError:
            fail(f"no {condition} cell for donor {donor} on {target}")
    out: dict = {}
    for item_id, row in correct["rows"].items():
        values = []
        for cell in shuffles:
            other = cell["rows"].get(item_id)
            if other is None:
                fail(f"{donor}/{target}: item {item_id} is missing from a shuffled cell")
            values.append(float(other["correct"]))
        subset = out.setdefault(str(row["subset"]), {})
        image = subset.setdefault(str(row["image_id"]), {})
        image[item_id] = {
            "ground": float(row["correct"]) - sum(values) / len(values),
            "utility": float(row["correct"]),
        }
    return out


def build(args) -> dict:
    cells = load_cells(args.submission, args.results_dir, block="D")
    donors = sorted({arm[2:] for _, _, arm, _ in cells if arm.startswith("D_")})
    if not donors or any(d not in DONORS for d in donors):
        fail(f"unexpected donor set {donors}")
    targets = sorted({t for _, t, _, _ in cells})
    if not targets or any(t not in TARGETS for t in targets):
        fail(f"unexpected target set {targets}")
    alignment_dir = Path(args.alignment_dir)
    damages = {d: damage(alignment_dir, d, targets) for d in donors}
    spec = {
        "schema_version": SCHEMA_VERSION,
        "donors": {d: NLLB[d] for d in donors},
        "damage": {d: damages[d]["value"] for d in donors},
        "delta_u": args.delta_u,
        "cells": {d: {t: endpoints(cells, d, t) for t in targets} for d in donors},
        "provenance": {
            "submission": str(Path(args.submission).resolve()),
            "spec_sha": next(iter(cells.values()))["manifest"]["spec_sha"],
            "code_sha": next(iter(cells.values()))["manifest"]["code_sha"],
            "targets": targets,
            "panel": "P_current (exploratory): the four current transfer targets",
            "damage_detail": damages,
            "endpoint": "ground = correct - mean(shuffled 0,1,2); utility = correct",
        },
    }
    return spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True)
    p.add_argument("--results-dir", default=None, help="override the cluster output directory")
    p.add_argument("--alignment-dir", default="Approach2/results")
    p.add_argument("--delta-u", type=float, default=1.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    spec = build(args)
    atomic_json(args.output, spec)
    print(
        f"{len(spec['donors'])} donors x {len(spec['provenance']['targets'])} targets -> {args.output}\n"
        + "\n".join(
            f"  damage {d}: {v:+.5f}" for d, v in sorted(spec["damage"].items(), key=lambda kv: kv[1])
        )
    )
    print(json.dumps({"least_damaged": min(spec["damage"], key=spec["damage"].get)}))


if __name__ == "__main__":
    main()
