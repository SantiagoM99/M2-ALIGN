"""Training variance of Block C's C4 − C1 contrast across C1 seeds (stdlib).

`block_c.py` reads one seed per arm, so every bound it reports is conditional on
those four trajectories. The post-hoc C4 − C1 contrast (DESIGN 2026-09-23) is
the number the joint paper leans on, and a single seed cannot certify it, so C1
is retrained at further seeds and this reports the contrast once per seed plus
the mean, SD and range across them — the form CLAUDE.md requires when seeds are
few and fixed.

C4 receives no training, so it contributes the same cells to every seed's
contrast; only C1 moves. Pairing is checked rather than assumed: the item
universe, the assigned images and the regenerated shuffle maps must be
bit-identical between the reference submission and each replicate, because a
replicate scored against different shuffled images is not a replicate of this
contrast at all.

    python3 Approach2/analysis/c1_seed_spread.py \\
      --reference Approach2/outputs/s1_C.submission.json \\
      --replicate 14=Approach2/outputs/s1_C1_seed14.eval.submission.json \\
      --replicate 15=Approach2/outputs/s1_C1_seed15.eval.submission.json \\
      --output Approach2/audits/s1_C_seed_spread.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from block_a import load_cells  # noqa: E402
from block_c import CONDITIONS, PRIMARY, endpoints, interval  # noqa: E402
from s1_contract import SPEC_SHA, atomic_json, file_sha  # noqa: E402

PAIRING_FIELDS = ("data", "id_universe_sha256", "image_sha256", "shuffle_map_hashes")


def fail(message: str):
    raise SystemExit(f"c1_seed_spread: {message}")


def require_arm(cells, arm: str, targets, benchmark="cvqa"):
    for t in targets:
        for c in CONDITIONS:
            if (benchmark, t, arm, c) not in cells:
                fail(f"{arm} is missing the {benchmark}/{t}/{c} cell")


def require_pairing(reference, replicate, targets, benchmark="cvqa"):
    """A replicate must sit on the same items, images and shuffle maps."""
    for t in targets:
        ref = reference[(benchmark, t, "C1", "correct")]
        rep = replicate[(benchmark, t, "C1", "correct")]
        for field in PAIRING_FIELDS:
            if ref["manifest"][field] != rep["manifest"][field]:
                fail(f"{benchmark}/{t}: {field} differs between the reference and the replicate")
        if ref["rows"].keys() != rep["rows"].keys():
            fail(f"{benchmark}/{t}: the replicate scored a different item set")


def contrast(c4_cells, c1_cells, targets, B, seed, label):
    """C4 − C1 per item, on the items both arms scored, pooled micro."""
    out = {}
    for ep in ("U", "grounding"):
        values = {}
        for t in targets:
            four = endpoints(c4_cells, "cvqa", t, "C4")[ep]
            one = endpoints(c1_cells, "cvqa", t, "C1")[ep]
            if four.keys() != one.keys():
                fail(f"cvqa/{t}: C4 and C1 scored different items")
            values[t] = {i: four[i] - one[i] for i in four}
        out[ep] = interval(c1_cells, "cvqa", values, B, seed, f"seedspread:{label}:{ep}")
    return out


def analyse(reference, replicates, B=4000, seed=0, targets=PRIMARY) -> dict:
    require_arm(reference, "C4", targets)
    require_arm(reference, "C1", targets)
    report = {
        "spec_sha": SPEC_SHA,
        "boot": B,
        "seed": seed,
        "targets": list(targets),
        "difference": "C4 (untrained composition) minus C1 (stage-3 trained), percentage points",
        "status": "post-hoc and exploratory, as declared in DESIGN 2026-09-23; no gate input",
        "per_seed": {},
    }
    report["per_seed"]["13"] = contrast(reference, reference, targets, B, seed, "13")
    for label, cells in sorted(replicates.items()):
        require_arm(cells, "C1", targets)
        require_pairing(reference, cells, targets)
        report["per_seed"][label] = contrast(reference, cells, targets, B, seed, label)

    report["across_seeds"] = {}
    for ep in ("U", "grounding"):
        points = [report["per_seed"][s][ep]["estimate"] for s in sorted(report["per_seed"])]
        report["across_seeds"][ep] = {
            "seeds": sorted(report["per_seed"]),
            "estimates": points,
            "mean": sum(points) / len(points),
            "sd": statistics.stdev(points) if len(points) > 1 else None,
            "range": [min(points), max(points)] if points else None,
        }
    report["inference"] = (
        "each per-seed interval resamples image clusters and excludes training variance; "
        "the spread across seeds is the only training-variance evidence here, and with "
        f"{len(report['per_seed'])} fixed seeds it is a range, not a population SD"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, help="the full Block C submission (C4 and C1 seed 13)")
    p.add_argument("--replicate", action="append", default=[], metavar="SEED=PATH")
    p.add_argument("--results-dir")
    p.add_argument("--output", required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    replicates = {}
    for spec in a.replicate:
        if "=" not in spec:
            fail(f"--replicate {spec!r} is not SEED=PATH")
        label, path = spec.split("=", 1)
        replicates[label] = load_cells(path, a.results_dir, block="C")
    if not replicates:
        fail("at least one --replicate is needed; without one this adds nothing to block_c.py")

    report = analyse(load_cells(a.reference, a.results_dir, block="C"), replicates, a.boot, a.seed)
    report["submissions"] = {"reference": file_sha(a.reference),
                             **{s: file_sha(spec.split("=", 1)[1])
                                for s, spec in zip(sorted(replicates), sorted(a.replicate))}}
    atomic_json(a.output, report)
    for ep in ("grounding", "U"):
        across = report["across_seeds"][ep]
        per = "  ".join(
            f"s{s} {report['per_seed'][s][ep]['estimate']:+.2f}" for s in across["seeds"]
        )
        sd = "n/a" if across["sd"] is None else f"{across['sd']:.2f}"
        print(f"{ep:9s} {per}   mean {across['mean']:+.2f}  SD {sd}  "
              f"range [{across['range'][0]:+.2f}, {across['range'][1]:+.2f}]")


if __name__ == "__main__":
    main()
