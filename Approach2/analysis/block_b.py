"""S1 Block B: branch factorial, P3 functional localisation, P4 co-adaptation, G2 (stdlib).

Run only after all 376 cells finish. Every contrast is a fixed linear
combination of cell endpoints taken item by item on identical items, so one
paired image-cluster bootstrap carries it, stratified by target and subset.

P3 and P4 are reported exactly as their frozen formulas define them. G2 fixes
their sign ("LB95(D_T) > 0"), so the candidate-minus-reference convention of
the non-inferiority paragraph does not apply here: that paragraph governs
contrasts judged against the delta margin, and Block B has none.

    python Approach2/analysis/block_b.py \\
      --submission Approach2/outputs/s1_B.submission.json \\
      --output Approach2/audits/s1_B_analysis.json
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

TEXT = {
    "T1_bn": "stage1",
    "T1_id": "stage1_id",
    "T3_bn": "stage3_bn_dcl",
    "T3_id": "stage3_id_v4",
}
VISION = {
    "V2": "stage2_dc_llava",
    "V3_bn": "stage3_bn_dcl",
    "V3_id": "stage3_id_v4",
}
CVQA_TARGETS = ("jv", "mn", "ga", "si", "bn")
PRIMARY = ("jv", "mn", "ga")
MAIN_CONDITIONS = ("correct", "shuffled0", "shuffled1", "shuffled2", "gray")
NOX_CELLS = (("T3_bn", "V3_bn"), ("T3_id", "V3_id"), ("T1_bn", "V2"))
NOX_CONDITIONS = ("correct", "shuffled0", "shuffled1", "shuffled2")
XGQA_TARGETS = ("bn", "id")
XGQA_CORNERS = (("T3_bn", "V3_bn"), ("T3_bn", "V3_id"), ("T3_id", "V3_bn"), ("T3_id", "V3_id"))
XGQA_CONDITIONS = ("correct", "gray")

# P3 and P4 as frozen in DESIGN.md S1, written as coefficients on G(text, vision).
D_T = {("T3_id", "V2"): 1.0, ("T1_id", "V2"): -1.0, ("T3_bn", "V2"): -1.0, ("T1_bn", "V2"): 1.0}
D_V = {("T1_bn", "V3_id"): 0.5, ("T1_bn", "V3_bn"): -0.5, ("T1_id", "V3_id"): 0.5, ("T1_id", "V3_bn"): -0.5}
P4 = {("T3_bn", "V3_bn"): 0.5, ("T3_id", "V3_id"): 0.5, ("T3_bn", "V3_id"): -0.5, ("T3_id", "V3_bn"): -0.5}


def difference(first, second):
    out = dict(first)
    for key, weight in second.items():
        out[key] = out.get(key, 0.0) - weight
    return {k: w for k, w in out.items() if w != 0.0}


CONTRASTS = {"D_T": D_T, "D_V": D_V, "D_T-D_V": difference(D_T, D_V), "P4": P4}


def arm(text, vision, nox=False):
    return f"{text}__{vision}" + ("__noX" if nox else "")


def required_cells():
    required = set()
    for t in CVQA_TARGETS:
        for text in TEXT:
            for vision in VISION:
                required.update(("cvqa", t, arm(text, vision), c) for c in MAIN_CONDITIONS)
        for text, vision in NOX_CELLS:
            required.update(("cvqa", t, arm(text, vision, True), c) for c in NOX_CONDITIONS)
    for t in XGQA_TARGETS:
        for text, vision in XGQA_CORNERS:
            required.update(("xgqa", t, arm(text, vision), c) for c in XGQA_CONDITIONS)
    return required


def checkpoint_name(record):
    return Path(record["path"]).parts[-3]


def validate_grid(cells):
    required = required_cells()
    if set(cells) != required:
        raise ValueError(
            f"Block B requires its complete {len(required)}-cell grid: missing "
            f"{len(required - set(cells))}, extra {len(set(cells) - required)}"
        )
    refs = {}
    for (b, t, a, c), cell in sorted(cells.items()):
        m, rows = cell["manifest"], cell["rows"]
        args = m["arguments"]
        parts = a.split("__")
        text, vision, nox = parts[0], parts[1], len(parts) == 3
        if m["vis_layers"] != "9,18,-1":
            raise ValueError("Block B requires dense vision layers 9,18,-1")
        if args["no_text_branch"] != nox:
            raise ValueError(f"{a}/{c}: text-branch flag does not match the arm")
        if args["prompt_mode"] != "question" or args["question_field"] != "query":
            raise ValueError(f"{a}/{c}: Block B prompts carry the native question")
        for branch, expected in (("mapping_txt", TEXT[text]), ("mapping_vis", VISION[vision])):
            record = m["checkpoints"].get(branch)
            if not record or checkpoint_name(record) != expected:
                raise ValueError(f"{a}/{c}: {branch} is not loaded from {expected}")
        ref = refs.setdefault((b, t), cell)
        if rows.keys() != ref["rows"].keys():
            raise ValueError("unpaired item ids across cells of one target")
        for field in ("data", "id_universe_sha256", "image_sha256", "shuffle_map_hashes", "decoding"):
            if m[field] != ref["manifest"][field]:
                raise ValueError(f"{field} differs across cells of one target")
        if m.get("environment") != ref["manifest"].get("environment"):
            raise ValueError("runtime environment differs across cells")
        for branch, record in m.get("frozen_models", {}).items():
            other = ref["manifest"].get("frozen_models", {}).get(branch)
            if record is not None and other is not None and record != other:
                raise ValueError("frozen model revision differs across cells")
        fields = ("query", "image_id", "subset", "choices", "answer_index") if b == "cvqa" else (
            "query", "image_id", "subset", "answer")
        for i, r in rows.items():
            if any(r.get(k) != ref["rows"][i].get(k) for k in fields):
                raise ValueError("item content differs across cells of one target")


def endpoints(cells, b, t, a):
    ref = cells[(b, t, a, "correct")]["rows"]
    shuffles = [cells.get((b, t, a, f"shuffled{k}")) for k in range(3)]
    gray = cells.get((b, t, a, "gray"))
    out = {"U": {}}
    if all(shuffles):
        out["grounding"] = {}
    if gray:
        out["gray"] = {}
    for i in sorted(ref):
        u = int(ref[i]["correct"])
        out["U"][i] = u
        if all(shuffles):
            out["grounding"][i] = u - sum(int(s["rows"][i]["correct"]) for s in shuffles) / 3
        if gray:
            out["gray"][i] = u - int(gray["rows"][i]["correct"])
    return out


def interval(cells, b, values, B, seed, label):
    """Item-micro estimate and paired image-cluster bootstrap over target:subset strata."""
    strata, macros = {}, []
    for t in sorted(values):
        vs = values[t]
        rows = cells[(b, t, arm("T1_bn", "V2") if b == "cvqa" else arm("T3_bn", "V3_bn"), "correct")]["rows"]
        macros.append(100 * sum(vs.values()) / len(vs))
        for i in sorted(vs):
            r = rows[i]
            stratum = f"{t}:{r['subset']}" if b == "cvqa" else t
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


def combine(eps, targets, endpoint, weights):
    out = {}
    for t in targets:
        items = eps[next(iter(weights))][t][endpoint].keys()
        out[t] = {i: sum(w * eps[key][t][endpoint][i] for key, w in weights.items()) for i in items}
    return out


def cvqa_panel(cells, targets, B, seed, name, with_arms):
    eps = {
        (text, vision): {t: endpoints(cells, "cvqa", t, arm(text, vision)) for t in targets}
        for text in TEXT for vision in VISION
    }
    panel = {"benchmark": "cvqa", "targets": list(targets), "arms": {}, "contrasts": {}}
    if with_arms:
        for key in eps:
            panel["arms"][arm(*key)] = {
                ep: interval(cells, "cvqa", {t: eps[key][t][ep] for t in targets}, B, seed,
                             f"B:{name}:{arm(*key)}:{ep}")
                for ep in ("U", "grounding")
            }
    for cname, weights in CONTRASTS.items():
        panel["contrasts"][cname] = {
            ep: interval(cells, "cvqa", combine(eps, targets, ep, weights), B, seed, f"B:{name}:{cname}:{ep}")
            for ep in ("U", "grounding")
        }
    return panel


def analyse(cells, B=4000, seed=0):
    if B <= 0:
        raise ValueError("bootstrap count must be positive")
    validate_grid(cells)
    report = {
        "spec_sha": SPEC_SHA,
        "boot": B,
        "seed": seed,
        "difference": "P3 and P4 exactly as their frozen formulas; percentage points",
        "inference": "conditional on the evaluated checkpoints; no training variance",
        "panels": {},
        "controls": {},
    }
    report["panels"]["primary"] = cvqa_panel(cells, PRIMARY, B, seed, "primary", with_arms=True)
    for t in PRIMARY:
        report["panels"][f"cvqa-{t}"] = cvqa_panel(cells, (t,), B, seed, f"cvqa-{t}", with_arms=False)
    report["panels"]["si-secondary"] = cvqa_panel(cells, ("si",), B, seed, "si", with_arms=False)
    report["panels"]["bn-control"] = cvqa_panel(cells, ("bn",), B, seed, "bn-control", with_arms=True)

    # No-X_f controls, descriptive: the same cell with the text branch removed.
    nox = {}
    for text, vision in NOX_CELLS:
        values = {}
        for ep in ("U", "grounding"):
            per = {}
            for t in PRIMARY:
                with_x = endpoints(cells, "cvqa", t, arm(text, vision))[ep]
                without = endpoints(cells, "cvqa", t, arm(text, vision, True))[ep]
                per[t] = {i: without[i] - with_x[i] for i in with_x}
            values[ep] = interval(cells, "cvqa", per, B, seed, f"B:noX:{text}__{vision}:{ep}")
        nox[arm(text, vision) + " noX-minus-withX"] = values
    report["controls"]["no_text_branch_primary"] = nox

    # xGQA task positive controls, descriptive: utility and gray-canvas sensitivity.
    xgqa = {}
    for t in XGQA_TARGETS:
        for text, vision in XGQA_CORNERS:
            eps = endpoints(cells, "xgqa", t, arm(text, vision))
            xgqa[f"{t}:{arm(text, vision)}"] = {
                ep: interval(cells, "xgqa", {t: eps[ep]}, B, seed, f"B:xgqa:{t}:{arm(text, vision)}:{ep}")
                for ep in ("U", "gray")
            }
    report["controls"]["xgqa_task_positive"] = xgqa

    primary = report["panels"]["primary"]["contrasts"]
    g = {k: primary[k]["grounding"] for k in ("D_T", "D_V", "D_T-D_V")}
    report["gates"] = {
        "G2_follows_text_branch": g["D_T"]["lb5"] > 0 and g["D_T-D_V"]["lb5"] > 0,
        "G2_bounds": {"LB95(D_T)": g["D_T"]["lb5"], "LB95(D_T-D_V)": g["D_T-D_V"]["lb5"]},
        # Declared 2026-09-12 in DESIGN.md before this analysis was run, as an
        # exploratory reading symmetric to G2. It is not a frozen criterion.
        "exploratory_follows_vision_branch": g["D_V"]["lb5"] > 0 and g["D_T-D_V"]["ub95"] < 0,
        "exploratory_bounds": {"LB95(D_V)": g["D_V"]["lb5"], "UB95(D_T-D_V)": g["D_T-D_V"]["ub95"]},
        "P4_co_adaptation_ci95": primary["P4"]["grounding"]["ci95"],
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True)
    p.add_argument("--results-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    report = analyse(load_cells(a.submission, a.results_dir, block="B"), a.boot, a.seed)
    report["submission_sha256"] = file_sha(a.submission)
    atomic_json(a.output, report)
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
