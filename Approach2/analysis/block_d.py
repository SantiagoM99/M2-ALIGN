"""S1 Block D, D-donor analysis (stdlib): damage → donor effect, exact test, regret.

Frozen procedure (DESIGN.md S1 v2.1.3, Block D and G3):
  * donor effect α_s from the additive model m_st = μ + α_s + β_t on the
    S × T table of cell means of the endpoint (macro over targets,
    sum-to-zero; closed form for a complete table);
  * damage_s = mean over targets T of [margin_stage1_s(s→T) − margin_stage3_s(s→T)],
    centered margin from each donor's OWN stage-1 and stage-3 pair-alignment
    files; positive = deterioration;
  * Spearman(damage, α) with midranks, exact permutation of the |S| labels,
    one-sided in the direction "more damage → lower α";
  * top-donor regret = α_max − α_(argmin damage); ties in damage broken by
    NLLB code (alphabetical), ties in α toward the worse donor; UB95 from an
    image-cluster bootstrap stratified by target that refits α inside every
    resample;
  * guardrails: pooled endpoint of the selected donor, LB95 > 0.

Inputs: either real result files (not yet produced: Δ_ground needs the
shuffled condition) or --from-json with the same structure, which is what
test_block_d.py uses. Shape of the JSON:
  {"donors": {"<s>": "<nllb code>"}, "damage": {"<s>": float},
   "cells": {"<s>": {"<t>": {"<image>": [per-item values ...]}}}}

    python3 Approach2/analysis/block_d.py --from-json spec.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import pct, rng_for  # noqa: E402


def midranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return float("nan") if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y) -> float:
    return pearson(midranks(x), midranks(y))


def exact_perm_p_one_sided(x, y) -> float:
    """P(rho_perm <= rho_obs) under exchangeability: one-sided for a NEGATIVE
    association (more damage → lower α)."""
    obs = spearman(x, y)
    n = len(y)
    hits = total = 0
    for p in itertools.permutations(range(n)):
        total += 1
        if spearman(x, [y[i] for i in p]) <= obs + 1e-12:
            hits += 1
    return hits / total


def alpha_from_cells(cells: dict[str, dict[str, float]], donors: list[str], targets: list[str]) -> dict[str, float]:
    """Closed-form α_s for a complete S×T table of cell means (sum-to-zero)."""
    mu = sum(cells[s][t] for s in donors for t in targets) / (len(donors) * len(targets))
    return {s: sum(cells[s][t] for t in targets) / len(targets) - mu for s in donors}


def cell_means(data, donors, targets) -> dict[str, dict[str, float]]:
    out = {}
    for s in donors:
        out[s] = {}
        for t in targets:
            vals = [v for img in data[s][t].values() for v in img]
            out[s][t] = 100.0 * sum(vals) / len(vals)
    return out


def select_donor(damage: dict[str, float], code: dict[str, str]) -> str:
    return sorted(damage, key=lambda s: (damage[s], code[s]))[0]


def regret(alpha: dict[str, float], selected: str) -> float:
    return max(alpha.values()) - alpha[selected]


def analyse(spec: dict, B: int = 2000, seed: int = 0) -> dict:
    donors = sorted(spec["donors"])
    code = spec["donors"]
    targets = sorted({t for s in donors for t in spec["cells"][s]})
    for s in donors:
        missing = [t for t in targets if t not in spec["cells"][s]]
        if missing:
            raise SystemExit(f"donor {s} lacks targets {missing}; the additive closed form needs a complete table")
    damage = {s: float(spec["damage"][s]) for s in donors}
    cm = cell_means(spec["cells"], donors, targets)
    alpha = alpha_from_cells(cm, donors, targets)
    x = [damage[s] for s in donors]
    y = [alpha[s] for s in donors]
    rho = spearman(x, y)
    p = exact_perm_p_one_sided(x, y)
    sel = select_donor(damage, code)
    reg = regret(alpha, sel)
    # bootstrap: resample images within each (donor, target) cell, refit α, recompute regret and the selected donor's pooled endpoint
    regs, pooled = [], []
    for b in range(B):
        cells_b = {}
        for s in donors:
            cells_b[s] = {}
            for t in targets:
                imgs = sorted(spec["cells"][s][t])
                rnd = rng_for(seed, "block_d", s, t)
                rnd = rng_for(seed, "block_d", s, t, str(b))
                C = len(imgs)
                draw = [imgs[rnd.randrange(C)] for _ in range(C)]
                vals = [v for k in draw for v in spec["cells"][s][t][k]]
                cells_b[s][t] = 100.0 * sum(vals) / len(vals)
        a_b = alpha_from_cells(cells_b, donors, targets)
        regs.append(regret(a_b, sel))
        pooled.append(sum(cells_b[sel][t] for t in targets) / len(targets))
    return {
        "donors": donors, "targets": targets, "damage": damage, "alpha": alpha,
        "spearman": rho, "exact_one_sided_p": p, "selected_donor": sel,
        "regret": reg, "regret_UB95": pct(regs, .95),
        "selected_pooled_endpoint": sum(cm[sel][t] for t in targets) / len(targets),
        "selected_pooled_LB95": pct(pooled, .05),
        "verdict": {
            "spearman_p_lt_0.10": p < 0.10,
            "regret_UB95_le_1.0": pct(regs, .95) <= 1.0,
            "selected_LB95_gt_0": pct(pooled, .05) > 0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = analyse(json.load(open(a.from_json)), a.boot, a.seed)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
