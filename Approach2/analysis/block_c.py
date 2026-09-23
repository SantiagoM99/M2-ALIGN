"""S1 Block C: freeze-factorial pilot, P5 to P8, G4, G5 and the G1-T readout (stdlib).

Run only after all 150 cells finish. Every contrast is a fixed combination of
cell endpoints taken item by item on identical items, pooled micro over
jv/mn/ga, with one paired image-cluster bootstrap stratified by target and
subset; xGQA-bn is its own stratum.

The frozen text writes P5 to P7 as candidate minus reference and D8 and the
utility half of G4 as reference minus candidate. Each is reported exactly as
written. Where a non-inferiority region is needed, the bound is read from the
written statistic, and the equivalence with the candidate-minus-reference
convention is exact because the percentile bootstrap maps the 95th percentile
of X to minus the 5th percentile of -X on the same resamples:
UB95(U(C1) - U(C2)) < delta  <=>  LB5(U(C2) - U(C1)) > -delta.

Under Option 1, G1-T needs three paired seeds of C5 and is not decidable here;
the single-seed D8 readout is reported and labelled as a pilot.

    python Approach2/analysis/block_c.py \\
      --submission Approach2/outputs/s1_C.submission.json \\
      --output Approach2/audits/s1_C_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_contract import SPEC_SHA, atomic_json, file_sha  # noqa: E402
from _boot import boot_stratified, pct  # noqa: E402
from block_a import load_cells  # noqa: E402

DELTA = 1.0
# arm -> (text checkpoint directory, vision checkpoint directory, text branch absent)
ARMS = {
    "C1": ("s1_C1_seed13", "s1_C1_seed13", False),
    "C2": ("s1_C2_seed13", "s1_C2_seed13", False),
    "C3": ("s1_C3_seed13", "s1_C3_seed13", False),
    "C4": ("stage1", "stage2_dc_llava", False),
    "C5": ("s1_C5_seed13", "s1_C5_seed13", True),
}
CVQA_TARGETS = ("jv", "mn", "ga", "si", "bn")
PRIMARY = ("jv", "mn", "ga")
CONDITIONS = ("correct", "shuffled0", "shuffled1", "shuffled2", "gray")

CONTRASTS = {
    "P5": {"C2": 1.0, "C1": -1.0},
    "P6": {"C3": 1.0, "C1": -1.0},
    "P7": {"C4": 1.0, "C3": -1.0, "C2": -1.0, "C1": 1.0},
    "D8": {"C1": 1.0, "C5": -1.0},
}

# Declared post-hoc on 2026-09-23, after the frozen contrasts were read, because
# the arm table was being quoted as "C4 beats C1 on the targets" and a
# difference of two separately-bounded levels is not a tested difference. Kept
# apart from CONTRASTS so the pre-registered report stays byte-identical, and
# carried under keys prefixed X_ so no reader mistakes it for a gate input.
POSTHOC_CONTRASTS = {
    "X_C4_minus_C1": {"C4": 1.0, "C1": -1.0},
}


def required_cells():
    req = set()
    for arm in ARMS:
        for t in CVQA_TARGETS:
            req.update(("cvqa", t, arm, c) for c in CONDITIONS)
        req.update(("xgqa", "bn", arm, c) for c in CONDITIONS)
    return req


def checkpoint_name(record):
    return Path(record["path"]).parts[-3]


def validate_grid(cells):
    required = required_cells()
    if set(cells) != required:
        raise ValueError(
            f"Block C requires its complete {len(required)}-cell grid: missing "
            f"{len(required - set(cells))}, extra {len(set(cells) - required)}"
        )
    refs = {}
    for (b, t, arm, c), cell in sorted(cells.items()):
        m, rows = cell["manifest"], cell["rows"]
        args = m["arguments"]
        text_dir, vision_dir, no_text = ARMS[arm]
        if m["vis_layers"] != "9,18,-1":
            raise ValueError("Block C requires dense vision layers 9,18,-1")
        if args["no_text_branch"] != no_text:
            raise ValueError(f"{arm}/{c}: text-branch flag does not match the arm")
        if args["prompt_mode"] != "question" or args["question_field"] != "query":
            raise ValueError(f"{arm}/{c}: Block C prompts carry the native question")
        vis = m["checkpoints"].get("mapping_vis")
        if not vis or checkpoint_name(vis) != vision_dir:
            raise ValueError(f"{arm}/{c}: mapping_vis is not loaded from {vision_dir}")
        if not no_text:
            txt = m["checkpoints"].get("mapping_txt")
            if not txt or checkpoint_name(txt) != text_dir:
                raise ValueError(f"{arm}/{c}: mapping_txt is not loaded from {text_dir}")
        ref = refs.setdefault((b, t), cell)
        if rows.keys() != ref["rows"].keys():
            raise ValueError("unpaired item ids across cells of one target")
        for field in ("data", "id_universe_sha256", "image_sha256", "shuffle_map_hashes", "decoding"):
            if m[field] != ref["manifest"][field]:
                raise ValueError(f"{field} differs across cells of one target")
        if m.get("environment") != ref["manifest"].get("environment"):
            raise ValueError("runtime environment differs across cells")
        fields = ("query", "image_id", "subset", "choices", "answer_index") if b == "cvqa" else (
            "query", "image_id", "subset", "answer")
        for i, r in rows.items():
            if any(r.get(k) != ref["rows"][i].get(k) for k in fields):
                raise ValueError("item content differs across cells of one target")


def endpoints(cells, b, t, arm):
    ref = cells[(b, t, arm, "correct")]["rows"]
    shuffles = [cells[(b, t, arm, f"shuffled{k}")]["rows"] for k in range(3)]
    gray = cells[(b, t, arm, "gray")]["rows"]
    out = {"U": {}, "grounding": {}, "gray": {}}
    for i in sorted(ref):
        u = int(ref[i]["correct"])
        out["U"][i] = u
        out["grounding"][i] = u - sum(int(s[i]["correct"]) for s in shuffles) / 3
        out["gray"][i] = u - int(gray[i]["correct"])
    return out


def interval(cells, b, values, B, seed, label):
    strata, macros = {}, []
    for t in sorted(values):
        vs = values[t]
        rows = cells[(b, t, "C1", "correct")]["rows"]
        macros.append(100 * sum(vs.values()) / len(vs))
        for i in sorted(vs):
            r = rows[i]
            stratum = f"{t}:{r['subset']}" if b == "cvqa" else f"xgqa:{t}"
            strata.setdefault(stratum, {}).setdefault(str(r["image_id"]), []).append(vs[i])
    samples = boot_stratified(strata, B, seed, label)
    flat = [v for t in values for v in values[t].values()]
    return {
        "estimate": 100 * sum(flat) / len(flat),
        "macro_estimate": sum(macros) / len(macros),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "lb5": pct(samples, 0.05),
        "ub95": pct(samples, 0.95),
        "items": len(flat),
        "image_clusters": sum(len(s) for s in strata.values()),
    }


def panel(cells, b, targets, B, seed, name, with_arms, contrasts=CONTRASTS):
    eps = {arm: {t: endpoints(cells, b, t, arm) for t in targets} for arm in ARMS}
    out = {"benchmark": b, "targets": list(targets), "arms": {}, "contrasts": {}}
    if with_arms:
        for arm in ARMS:
            out["arms"][arm] = {
                ep: interval(cells, b, {t: eps[arm][t][ep] for t in targets}, B, seed, f"C:{name}:{arm}:{ep}")
                for ep in ("U", "grounding", "gray")
            }
    for cname, weights in contrasts.items():
        out["contrasts"][cname] = {}
        for ep in ("U", "grounding"):
            values = {
                t: {i: sum(w * eps[arm][t][ep][i] for arm, w in weights.items()) for i in eps["C1"][t][ep]}
                for t in targets
            }
            out["contrasts"][cname][ep] = interval(cells, b, values, B, seed, f"C:{name}:{cname}:{ep}")
    return out


def analyse(cells, B=4000, seed=0, posthoc=False):
    if B <= 0:
        raise ValueError("bootstrap count must be positive")
    validate_grid(cells)
    report = {
        "spec_sha": SPEC_SHA,
        "boot": B,
        "seed": seed,
        "difference": "P5-P7 candidate minus reference; D8 reference (C1) minus candidate (C5); percentage points",
        "inference": "single training seed per arm: conditional on these trajectories, no training variance",
        "panels": {},
    }
    contrasts = {**CONTRASTS, **POSTHOC_CONTRASTS} if posthoc else CONTRASTS
    if posthoc:
        report["posthoc_contrasts"] = sorted(POSTHOC_CONTRASTS)
    report["panels"]["primary"] = panel(cells, "cvqa", PRIMARY, B, seed, "primary", True, contrasts)
    for t in PRIMARY:
        report["panels"][f"cvqa-{t}"] = panel(cells, "cvqa", (t,), B, seed, f"cvqa-{t}", False, contrasts)
    report["panels"]["si-secondary"] = panel(cells, "cvqa", ("si",), B, seed, "si", False, contrasts)
    report["panels"]["bn-control"] = panel(cells, "cvqa", ("bn",), B, seed, "bn-control", True, contrasts)
    report["panels"]["xgqa-bn"] = panel(cells, "xgqa", ("bn",), B, seed, "xgqa-bn", True, contrasts)

    p = report["panels"]["primary"]["contrasts"]
    x = report["panels"]["xgqa-bn"]["contrasts"]
    g4_grounding = p["P5"]["grounding"]["lb5"] > 0
    g4_utility = p["P5"]["U"]["lb5"] > -DELTA          # UB95(U(C1) - U(C2)) < delta
    g5 = x["P5"]["U"]["lb5"] > -DELTA                   # xGQA-bn U(C2) non-inferior to U(C1)
    d8u, d8g = p["D8"]["U"], p["D8"]["grounding"]
    if d8u["ub95"] < DELTA and d8g["ub95"] < DELTA:
        g1t = "dispensable in training (single-seed pilot)"
    elif d8g["lb5"] > DELTA:
        g1t = "needed in training (single-seed pilot)"
    else:
        g1t = "unresolved (single-seed pilot)"
    report["gates"] = {
        "P5_predicted_positive_on_grounding": p["P5"]["grounding"]["estimate"] > 0,
        "G4_freeze_text_helps_transfer": g4_grounding and g4_utility,
        "G4_bounds": {"LB95(P5 grounding)": p["P5"]["grounding"]["lb5"],
                      "UB95(U(C1)-U(C2))": -p["P5"]["U"]["lb5"]},
        "G5_source_task_retained": g5,
        "G5_bound": {"UB95(xGQA-bn U(C1)-U(C2))": -x["P5"]["U"]["lb5"]},
        "G1T_pilot_readout": g1t,
        "G1T_decidable": False,
        "G1T_note": "Option 1 defers the three paired C5 seeds that P8 requires",
        "P6_ci95_grounding": p["P6"]["grounding"]["ci95"],
        "P7_ci95_grounding": p["P7"]["grounding"]["ci95"],
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--results-dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posthoc", action="store_true",
                    help="also report POSTHOC_CONTRASTS; write to a separate output file")
    a = ap.parse_args()
    report = analyse(load_cells(a.submission, a.results_dir, block="C"), a.boot, a.seed, a.posthoc)
    report["submission_sha256"] = file_sha(a.submission)
    atomic_json(a.output, report)
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
