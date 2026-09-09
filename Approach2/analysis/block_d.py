"""S1 Block D, D-donor analysis (stdlib): damage → donor effect, exact test, regret, G3.

Frozen procedure (DESIGN.md S1 v2.1.3, Block D-donor and G3):
  * donor effect α_s from the additive model m_st = μ + α_s + β_t on the
    S × T table of cell estimates (macro over targets, sum-to-zero; closed
    form for a complete table). Each (s, t) cell is the item-micro mean of
    the endpoint over every item of target unit t, all subsets pooled;
  * two endpoints per item: Δ_ground (correct − shuffled) gives α^G and
    utility (correct-image accuracy) gives α^U;
  * damage_s = mean over targets of [margin_stage1_s − margin_stage3_s] from
    the donor's OWN pair-alignment files; positive = deterioration. It is an
    input here: it is computed and committed before any outcome is read;
  * Spearman(damage, α^G) with midranks, exact permutation of the |S| labels,
    one-sided in the direction "more damage → lower α". A constant damage or
    α vector has no rank correlation: the result is UNDEFINED, never p = 0;
  * selection: least damage = best donor, damage ties broken by NLLB code;
    regret = max_s α_s − α_selected on both endpoints;
  * paired cluster bootstrap: in replicate b ONE draw of images with
    replacement per (target, subset) stratum is applied to every donor, the
    cell table is recomputed from the drawn items, α is refitted and both
    regrets plus the selected donor's pooled Δ_ground are recomputed.
    Identical donors therefore give a regret distribution that is exactly 0;
  * the selected donor's pooled Δ_ground follows the general item-micro rule
    (every item of every target weighs 1), unlike α, which is the declared
    macro-over-target exception;
  * G3 = all four: p < 0.10, UB95(regret^G) ≤ 1.0, LB95(pooled Δ_ground of
    the selected donor) > 0, UB95(regret^U) ≤ δ_U (default 1.0). If the
    correlation is undefined the verdict is "undefined", not a pass.

Input (schema_version 2), --from-json; the item universe must be identical
for every donor, otherwise the analysis aborts:
  {"schema_version": 2,
   "donors": {"<s>": "<nllb code>"},
   "damage": {"<s>": float},
   "delta_u": 1.0,                       # optional, utility-regret margin
   "cells": {"<s>": {"<target>": {"<subset>": {"<image>":
             {"<question>": {"ground": -1|0|1, "utility": 0|1}}}}}}}

The production adapter that builds this JSON from `eval_*` prediction files
and `pair_alignment` outputs does not exist yet: Δ_ground needs the shuffled
condition, which no evaluator produces at this HEAD (post-freeze work).

    python3 Approach2/analysis/block_d.py --from-json spec.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import pct, rng_for  # noqa: E402

SCHEMA_VERSION = 2
ENDPOINTS = ("ground", "utility")
P_THRESHOLD = 0.10
DELTA_G = 1.0


def fail(message: str) -> None:
    raise SystemExit(f"block_d: {message}")


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


def pearson(x, y) -> float | None:
    """Pearson r, or None when either vector is constant (r is undefined)."""
    n = len(x)
    if n != len(y) or n < 2:
        fail("correlation needs two vectors of equal length >= 2")
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y) -> float | None:
    return pearson(midranks(x), midranks(y))


def exact_perm_p_one_sided(x, y) -> float | None:
    """P(rho_perm <= rho_obs) under exchangeability: one-sided for a NEGATIVE
    association (more damage → lower α). None when rho_obs is undefined."""
    obs = spearman(x, y)
    if obs is None:
        return None
    rx = midranks(x)
    n = len(y)
    hits = total = 0
    for p in itertools.permutations(range(n)):
        total += 1
        rho = pearson(rx, midranks([y[i] for i in p]))
        if rho is not None and rho <= obs + 1e-12:
            hits += 1
    return hits / total


def alpha_from_cells(cells: dict[str, dict[str, float]], donors: list[str], targets: list[str]) -> dict[str, float]:
    """Closed-form α_s for a complete S×T table of cell means (sum-to-zero)."""
    mu = sum(cells[s][t] for s in donors for t in targets) / (len(donors) * len(targets))
    return {s: sum(cells[s][t] for t in targets) / len(targets) - mu for s in donors}


def select_donor(damage: dict[str, float], code: dict[str, str]) -> str:
    return sorted(damage, key=lambda s: (damage[s], code[s]))[0]


def regret(alpha: dict[str, float], selected: str) -> float:
    return max(alpha.values()) - alpha[selected]


def _finite(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{where}: value must be a finite number, got {value!r}")
    return float(value)


def validate(spec: dict) -> tuple[list[str], dict[str, str], dict[str, float], list[str], dict, float]:
    """-> donors (sorted), codes, damage, targets (sorted), universe, delta_u.

    universe[target][subset][image] = sorted question ids; identical for every
    donor by construction (the analysis aborts on any difference).
    """
    if int(spec.get("schema_version", 0)) != SCHEMA_VERSION:
        fail(f"schema_version must be {SCHEMA_VERSION} (items need subset, image and question ids)")
    codes = spec.get("donors")
    if not isinstance(codes, dict) or len(codes) < 2:
        fail("at least two donors are required")
    donors = sorted(codes)
    for s in donors:
        if not isinstance(codes[s], str) or not codes[s]:
            fail(f"donor {s}: NLLB code must be a non-empty string")
    raw_damage = spec.get("damage")
    if not isinstance(raw_damage, dict) or sorted(raw_damage) != donors:
        fail("damage must list exactly the donors")
    damage = {s: _finite(raw_damage[s], f"damage[{s}]") for s in donors}
    delta_u = _finite(spec.get("delta_u", DELTA_G), "delta_u")
    if delta_u < 0:
        fail("delta_u must be non-negative")
    cells = spec.get("cells")
    if not isinstance(cells, dict) or sorted(cells) != donors:
        fail("cells must list exactly the donors")
    targets = sorted(cells[donors[0]])
    if not targets:
        fail("at least one target unit is required")

    universe: dict = {}
    for s in donors:
        if sorted(cells[s]) != targets:
            missing = sorted(set(targets) - set(cells[s]))
            extra = sorted(set(cells[s]) - set(targets))
            fail(f"donor {s} lacks targets {missing} / has extra targets {extra}; the additive closed form needs a complete table")
        for t in targets:
            subsets = cells[s][t]
            if not isinstance(subsets, dict) or not subsets:
                fail(f"cell ({s}, {t}) has no subsets")
            u_t = universe.setdefault(t, {})
            if s != donors[0] and sorted(subsets) != sorted(u_t):
                fail(f"donor {s}, target {t}: subsets {sorted(subsets)} differ from {sorted(u_t)}")
            for subset, images in subsets.items():
                if not isinstance(images, dict) or not images:
                    fail(f"({s}, {t}, {subset}) has no images")
                u_img = u_t.setdefault(subset, {})
                if s != donors[0] and sorted(images) != sorted(u_img):
                    fail(f"donor {s}, target {t}, subset {subset}: image universe differs from donor {donors[0]}")
                for image, questions in images.items():
                    if not isinstance(questions, dict) or not questions:
                        fail(f"({s}, {t}, {subset}, {image}) has no questions")
                    qids = sorted(questions)
                    if s == donors[0]:
                        u_img[image] = qids
                    elif qids != u_img[image]:
                        fail(f"donor {s}, target {t}, subset {subset}, image {image}: question ids differ from donor {donors[0]}")
                    for q, values in questions.items():
                        if not isinstance(values, dict) or sorted(values) != sorted(ENDPOINTS):
                            fail(f"({s}, {t}, {subset}, {image}, {q}): expected keys {ENDPOINTS}")
                        for e in ENDPOINTS:
                            _finite(values[e], f"({s}, {t}, {subset}, {image}, {q}).{e}")
    return donors, codes, damage, targets, universe, delta_u


def image_sums(spec: dict, donors: list[str], targets: list[str], universe: dict) -> dict:
    """sums[t][subset][image][s] = (sum_ground, sum_utility, n_items)."""
    out: dict = {}
    for t in targets:
        out[t] = {}
        for subset, images in universe[t].items():
            out[t][subset] = {}
            for image, qids in images.items():
                per_donor = {}
                for s in donors:
                    qs = spec["cells"][s][t][subset][image]
                    per_donor[s] = (
                        sum(float(qs[q]["ground"]) for q in qids),
                        sum(float(qs[q]["utility"]) for q in qids),
                        len(qids),
                    )
                out[t][subset][image] = per_donor
    return out


def table_from_draw(sums: dict, donors: list[str], targets: list[str], draw: dict, selected: str) -> tuple[dict, dict, float]:
    """Cell tables (points) for both endpoints and the selected donor's item-micro pooled Δ_ground.

    draw[t][subset] = list of image ids (with multiplicity); the SAME list is
    used for every donor, which is what makes the bootstrap paired.
    """
    cells_g = {s: {} for s in donors}
    cells_u = {s: {} for s in donors}
    sel_sum = 0.0
    sel_n = 0
    for t in targets:
        acc = {s: [0.0, 0.0, 0] for s in donors}
        for subset, images in draw[t].items():
            stratum = sums[t][subset]
            for image in images:
                per_donor = stratum[image]
                for s in donors:
                    g, u, n = per_donor[s]
                    a = acc[s]
                    a[0] += g
                    a[1] += u
                    a[2] += n
        for s in donors:
            g, u, n = acc[s]
            if n == 0:
                fail(f"cell ({s}, {t}) has no items")
            cells_g[s][t] = 100.0 * g / n
            cells_u[s][t] = 100.0 * u / n
        sel_sum += acc[selected][0]
        sel_n += acc[selected][2]
    return cells_g, cells_u, 100.0 * sel_sum / sel_n


def analyse(spec: dict, B: int = 2000, seed: int = 0) -> dict:
    if B <= 0:
        fail("B must be positive")
    donors, codes, damage, targets, universe, delta_u = validate(spec)
    sums = image_sums(spec, donors, targets, universe)
    sel = select_donor(damage, codes)

    full_draw = {t: {subset: sorted(images) for subset, images in universe[t].items()} for t in targets}
    cells_g, cells_u, pooled_sel = table_from_draw(sums, donors, targets, full_draw, sel)
    alpha_g = alpha_from_cells(cells_g, donors, targets)
    alpha_u = alpha_from_cells(cells_u, donors, targets)
    x = [damage[s] for s in donors]
    y = [alpha_g[s] for s in donors]
    rho = spearman(x, y)
    p = exact_perm_p_one_sided(x, y)
    reg_g = regret(alpha_g, sel)
    reg_u = regret(alpha_u, sel)

    # Paired cluster bootstrap: one image draw per (target, subset) stratum,
    # shared by all donors; the RNG key excludes the donor on purpose.
    regs_g, regs_u, pooled = [], [], []
    for b in range(B):
        draw = {}
        for t in targets:
            draw[t] = {}
            for subset in sorted(universe[t]):
                images = sorted(universe[t][subset])
                rnd = rng_for(seed, "block_d", t, subset, str(b))
                C = len(images)
                draw[t][subset] = [images[rnd.randrange(C)] for _ in range(C)]
        cg_b, cu_b, pooled_b = table_from_draw(sums, donors, targets, draw, sel)
        regs_g.append(regret(alpha_from_cells(cg_b, donors, targets), sel))
        regs_u.append(regret(alpha_from_cells(cu_b, donors, targets), sel))
        pooled.append(pooled_b)

    reg_g_ub = pct(regs_g, .95)
    reg_u_ub = pct(regs_u, .95)
    pooled_lb = pct(pooled, .05)
    conditions = {
        "spearman_p_lt_0.10": None if p is None else p < P_THRESHOLD,
        "regret_UB95_le_1.0": reg_g_ub <= DELTA_G,
        "selected_LB95_gt_0": pooled_lb > 0,
        "utility_regret_UB95_le_delta_U": reg_u_ub <= delta_u,
    }
    if any(v is None for v in conditions.values()):
        g3 = "undefined"
    else:
        g3 = "pass" if all(conditions.values()) else "fail"
    n_items = sum(len(q) for t in targets for imgs in universe[t].values() for q in imgs.values())
    n_images = sum(len(imgs) for t in targets for imgs in universe[t].values())
    n_strata = sum(len(universe[t]) for t in targets)
    return {
        "schema_version": SCHEMA_VERSION,
        "donors": donors, "targets": targets, "damage": damage,
        "items": n_items, "images": n_images, "strata": n_strata,
        "alpha": alpha_g, "alpha_utility": alpha_u,
        "spearman": rho, "exact_one_sided_p": p,
        "correlation_defined": rho is not None,
        "selected_donor": sel,
        "regret": reg_g, "regret_UB95": reg_g_ub,
        "utility_regret": reg_u, "utility_regret_UB95": reg_u_ub, "delta_u": delta_u,
        "selected_pooled_endpoint": pooled_sel,
        "selected_pooled_LB95": pooled_lb,
        "bootstrap": {"B": B, "seed": seed, "paired": True, "strata": "target x subset", "cluster": "image"},
        "verdict": conditions,
        "g3": g3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.from_json, encoding="utf-8") as handle:
        spec = json.load(handle)
    res = analyse(spec, a.boot, a.seed)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
