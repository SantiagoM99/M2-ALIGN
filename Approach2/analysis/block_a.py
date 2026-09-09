"""S1 Block A: authenticated per-item U, grounding, P1/P2 and G0/G1-I (stdlib).

Run only after all 156 cells finish. No item is silently dropped; shuffled
replicates are averaged within item. xGQA translations share image resamples.
All differences are candidate minus reference, in percentage points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_contract import (
    SPEC_SHA,
    atomic_json,
    digest,
    file_sha,
    read_json,
    read_rows,
    shuffle_maps,
    universe,
)
from _boot import boot_stratified, pct, region

CONDITIONS = ("correct", "shuffled0", "shuffled1", "shuffled2", "gray", "no-image")


def load_cells(submission_path, results_dir=None):
    sub = read_json(submission_path)
    if (
        sub.get("spec_sha") != SPEC_SHA
        or sub["plan"]["block"] != "A"
        or digest(sub["plan"]) != sub["plan_sha256"]
    ):
        raise ValueError("not an authenticated S1 Block A submission")
    cells = {}
    for cid, expected in sub["manifests"].items():
        original = Path(expected["arguments"]["output_path"])
        path = Path(results_dir) / original.name if results_dir else original
        m = read_json(str(path) + ".manifest.json")
        if {k: v for k, v in m.items() if k != "slurm_job_id"} != {
            k: v for k, v in expected.items() if k != "slurm_job_id"
        }:
            raise ValueError(f"{cid}: manifest differs from submitted experiment")
        if m["spec_sha"] != SPEC_SHA or m["dirty"] or m["code_sha"] != sub["code_sha"]:
            raise ValueError("invalid spec/code/dirty provenance")
        marker = read_json(str(path) + ".complete.json")
        summary = read_json(str(path) + ".summary.json")
        if (
            marker["manifest_sha256"] != digest(m)
            or marker["predictions_sha256"] != file_sha(path)
            or marker["summary_sha256"] != file_sha(str(path) + ".summary.json")
        ):
            raise ValueError(f"{cid}: completion hash mismatch")
        if any(summary.get(k) != v for k, v in m.items()):
            raise ValueError("summary provenance differs from manifest")
        rows = read_rows(path)
        if summary["skipped"] != 0 or summary["scored"] != len(rows):
            raise ValueError("incomplete scored universe")
        if any(type(r.get("correct")) is not bool for r in rows):
            raise ValueError("correct must be a JSON boolean")
        if summary["correct"] != sum(r["correct"] for r in rows) or summary[
            "accuracy"
        ] != sum(r["correct"] for r in rows) / len(rows):
            raise ValueError("summary accuracy mismatch")
        identities = universe(rows, m["benchmark"])
        if (
            digest(identities) != m["id_universe_sha256"]
            or m["split_sha256"] != m["id_universe_sha256"]
        ):
            raise ValueError("item/image universe differs from manifest")
        maps = shuffle_maps(rows, m["benchmark"])
        if m["shuffle_map_hashes"] != {str(x["seed"]): x["sha256"] for x in maps}:
            raise ValueError("shuffle map hashes differ from regenerated maps")
        cond = m["condition"]
        for r in rows:
            expected_image = (
                maps[int(cond[-1])]["items"][str(r["id"])]["assigned_image_id"]
                if cond.startswith("shuffled")
                else str(r["image_id"])
                if cond == "correct"
                else None
            )
            if (
                r.get("condition") != cond
                or r.get("assigned_image_id") != expected_image
            ):
                raise ValueError("incorrect assigned image or condition")
        key = (m["benchmark"], m["target"], m["arm"], cond)
        if key in cells:
            raise ValueError("duplicate arm/target/condition")
        cells[key] = {"manifest": m, "rows": {str(r["id"]): r for r in rows}}
    return cells


def validate_grid(cells):
    required = {
        (b, t, a, c)
        for b, targets, arms in [
            ("cvqa", ("jv", "mn", "ga", "si"), ("A1", "A2", "A3", "A4")),
            ("xgqa", ("bn", "de", "ko"), ("A1", "A2", "A3")),
            ("xgqa", ("en",), ("A4",)),
        ]
        for t in targets
        for a in arms
        for c in CONDITIONS
    }
    if set(cells) != required:
        raise ValueError(
            f"Block A requires its complete 156-cell grid: missing {len(required - set(cells))}, extra {len(set(cells) - required)}"
        )
    refs = {}
    for (b, t, a, c), cell in sorted(cells.items()):
        m = cell["manifest"]
        rows = cell["rows"]
        args = m["arguments"]
        if m["source"] != "bn" or m["vis_layers"] != "9,18,-1":
            raise ValueError(
                "Block A requires Bengali dcl source and dense vision layers"
            )
        if not m["checkpoints"]["mapping_txt"] or not m["checkpoints"]["mapping_vis"]:
            raise ValueError("Block A requires checkpoint provenance for both branches")
        if (
            Path(m["checkpoints"]["mapping_txt"]["path"]).parts[-3] != "stage3_bn_dcl"
            or m["checkpoints"]["mapping_txt"] != m["checkpoints"]["mapping_vis"]
        ):
            raise ValueError("Block A must use the same stage3_bn_dcl checkpoint")
        if (
            args["no_text_branch"] != (a in ("A2", "A4"))
            or args["prompt_mode"] != ("instruction" if a == "A3" else "question")
            or args["question_field"]
            != ("english_query" if a == "A4" and b == "cvqa" else "query")
        ):
            raise ValueError("arm flags do not implement A1/A2/A3/A4")
        group = (b, t)
        ref = refs.setdefault(group, cell)
        if rows.keys() != ref["rows"].keys():
            raise ValueError("unpaired item ids across conditions/arms")
        for field in (
            "data",
            "id_universe_sha256",
            "image_sha256",
            "shuffle_map_hashes",
            "checkpoints",
            "decoding",
        ):
            if m[field] != ref["manifest"][field]:
                raise ValueError(f"{field} differs across conditions/arms")
        if m.get("environment") != ref["manifest"].get("environment"):
            raise ValueError("runtime environment differs across conditions/arms")
        for branch, record in m.get("frozen_models", {}).items():
            reference = ref["manifest"].get("frozen_models", {}).get(branch)
            if record is not None and reference is not None and record != reference:
                raise ValueError("frozen model revision differs across conditions/arms")
        for i, r in rows.items():
            fields = (
                ("query", "image_id", "subset", "choices", "answer_index")
                if b == "cvqa"
                else ("query", "image_id", "subset", "answer")
            )
            if any(r.get(k) != ref["rows"][i].get(k) for k in fields):
                raise ValueError("item content differs across arms")
    base = refs[("xgqa", "bn")]
    for t in ("de", "ko", "en"):
        ref = refs[("xgqa", t)]
        for f in ("id_universe_sha256", "image_sha256", "shuffle_map_hashes"):
            if ref["manifest"][f] != base["manifest"][f]:
                raise ValueError("xGQA translations must share images, ids and maps")
        for i, r in ref["rows"].items():
            if r["answer"] != base["rows"][i]["answer"]:
                raise ValueError("xGQA translation gold labels differ")


def endpoints(cells, b, t, a):
    ref = cells[(b, t, a, "correct")]["rows"]
    out = {k: {} for k in ("U", "grounding", "gray", "none")}
    for i in sorted(ref):
        u = int(ref[i]["correct"])
        out["U"][i] = u
        out["grounding"][i] = (
            u
            - sum(
                cells[(b, t, a, f"shuffled{k}")]["rows"][i]["correct"] for k in range(3)
            )
            / 3
        )
        out["gray"][i] = u - int(cells[(b, t, a, "gray")]["rows"][i]["correct"])
        out["none"][i] = u - int(cells[(b, t, a, "no-image")]["rows"][i]["correct"])
    return out


def interval(cells, b, targets, values, B, seed, label):
    strata = {}
    macros = []
    for t in sorted(targets):
        vs = values[t]
        rows = cells[(b, t, "A1", "correct")]["rows"]
        macros.append(100 * sum(vs.values()) / len(vs))
        for i in sorted(vs):
            r = rows[i]
            s = "joint-translations" if b == "xgqa" else t + ":" + r["subset"]
            strata.setdefault(s, {}).setdefault(r["image_id"], []).append(vs[i])
    samples = boot_stratified(strata, B, seed, label)
    vals = [v for t in targets for v in values[t].values()]
    lb, ub = pct(samples, 0.05), pct(samples, 0.95)
    return {
        "estimate": 100 * sum(vals) / len(vals),
        "macro_estimate": sum(macros) / len(macros),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "lb5": lb,
        "ub95": ub,
        "region": region(lb, ub, 1.0),
        "items": len(vals),
        "image_clusters": sum(len(s) for s in strata.values()),
    }


def analyse(cells, B=4000, seed=0):
    if B <= 0:
        raise ValueError("bootstrap count must be positive")
    validate_grid(cells)
    report = {
        "spec_sha": SPEC_SHA,
        "boot": B,
        "seed": seed,
        "difference": "candidate minus reference; percentage points",
        "inference": "conditional on the evaluated checkpoint; no training variance",
        "panels": {},
    }
    panels = [
        ("primary", "cvqa", ["jv", "mn", "ga"]),
        ("xgqa-secondary", "xgqa", ["bn", "de", "ko"]),
        ("si-secondary", "cvqa", ["si"]),
    ]
    panels += [
        (f"{b}-{t}", b, [t])
        for b, ts in [("xgqa", ["bn", "de", "ko"]), ("cvqa", ["jv", "mn", "ga"])]
        for t in ts
    ]
    for name, b, ts in panels:
        panel = {
            "benchmark": b,
            "targets": ts,
            "family": "intervention-primary" if name == "primary" else "secondary",
            "arms": {},
            "contrasts": {},
        }
        available = ("A1", "A2", "A3", "A4") if b == "cvqa" else ("A1", "A2", "A3")
        eps = {a: {t: endpoints(cells, b, t, a) for t in ts} for a in available}
        for a in available:
            panel["arms"][a] = {
                ep: interval(
                    cells,
                    b,
                    ts,
                    {t: eps[a][t][ep] for t in ts},
                    B,
                    seed,
                    f"{b}:{name}:{a}:{ep}",
                )
                for ep in ("U", "grounding", "gray", "none")
            }
        for a in ("A2", "A3"):
            contrast = {}
            for ep in ("U", "grounding"):
                values = {
                    t: {i: v - eps["A1"][t][ep][i] for i, v in eps[a][t][ep].items()}
                    for t in ts
                }
                contrast[ep] = interval(
                    cells, b, ts, values, B, seed, f"{b}:{name}:{a}-A1:{ep}"
                )
            panel["contrasts"][a + "-A1"] = contrast
        report["panels"][name] = panel
    # xGQA A4 is run once; its U and sensitivity are an English descriptive reference.
    en = endpoints(cells, "xgqa", "en", "A4")
    report["english_reference"] = {
        ep: 100 * sum(v.values()) / len(v) for ep, v in en.items()
    }
    primary = report["panels"]["primary"]
    g0 = primary["arms"]["A1"]["grounding"]["lb5"] > 0
    contrast = primary["contrasts"]["A2-A1"]
    g1 = (
        "dispensable"
        if all(contrast[e]["region"] == "non-inferior" for e in ("U", "grounding"))
        else "used"
        if contrast["grounding"]["region"] == "mat. inferior"
        else "inconclusive"
    )
    a3 = primary["contrasts"]["A3-A1"]
    report["gates"] = {
        "G0": g0,
        "G1-I": g1 if g0 else "not evaluated: G0 failed",
        "A3_carries_question": all(
            a3[e]["region"] == "non-inferior" for e in ("U", "grounding")
        )
        if g0
        else None,
        "G1-T": "not evaluated; Option 1 defers replicated C5",
        "launch_B_C": g0,
        "preservation_loss": "future work under Option 1",
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
    report = analyse(load_cells(a.submission, a.results_dir), a.boot, a.seed)
    report["submission_sha256"] = file_sha(a.submission)
    atomic_json(a.output, report)
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
