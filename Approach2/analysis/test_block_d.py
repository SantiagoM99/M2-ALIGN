"""Synthetic tests for block_d.py (schema 2).

Closure criteria (review 2026-09-08, accepted by Santiago the same day):
  * identical donors → regret and its UB95 exactly 0 on both endpoints;
  * constant ranks → undefined correlation, no significant evidence, G3 undefined;
  * incompatible item/image universes → abort;
  * G3 with its four conditions, joint resampling per target/subset, and
    item-micro pooling of the selected donor's Δ_ground.
Plus the original checks: α closed form, exact p = 1/5040, midranks,
end-to-end recovery, damage tie rule, insertion-order invariance, and the
incomplete-table abort.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_d import (  # noqa: E402
    SCHEMA_VERSION,
    alpha_from_cells,
    analyse,
    exact_perm_p_one_sided,
    spearman,
)


def item(ground: float, utility: float) -> dict:
    return {"ground": ground, "utility": utility}


def make_spec(seed: int, true_alpha: dict[str, float], damage: dict[str, float],
              n_img: int = 40, per_img: int = 3) -> dict:
    """Four targets, one of them with two subsets; Δ_ground per item in {-1, 0, 1}."""
    rnd = random.Random(seed)
    targets = {"t1": ["a"], "t2": ["a", "b"], "t3": ["a"], "t4": ["a"]}
    beta = {"t1": 2.0, "t2": -1.0, "t3": 0.5, "t4": -1.5}
    # one shared item universe, then donor-specific outcomes on it
    universe = {t: {sub: {f"{t}_{sub}_img{k}": [f"{t}_{sub}_img{k}_{j}" for j in range(per_img)]
                         for k in range(n_img)} for sub in subs} for t, subs in targets.items()}
    cells = {}
    for s, a in true_alpha.items():
        cells[s] = {}
        for t, subs in universe.items():
            cells[s][t] = {}
            for sub, images in subs.items():
                p_ground = 0.30 + (a + beta[t]) / 100.0
                cells[s][t][sub] = {
                    img: {q: item(1.0 if rnd.random() < p_ground else (-1.0 if rnd.random() < 0.15 else 0.0),
                                  1.0 if rnd.random() < 0.5 + a / 100.0 else 0.0)
                          for q in qids}
                    for img, qids in images.items()
                }
    return {"schema_version": SCHEMA_VERSION, "donors": {s: f"{s}_Latn" for s in true_alpha},
            "damage": damage, "cells": cells}


def expect_exit(needle: str, fn) -> None:
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), f"expected abort mentioning {needle!r}, got {exc}"
        return
    raise AssertionError(f"expected abort mentioning {needle!r}")


def reorder(d: dict) -> dict:
    return dict(reversed(list(d.items())))


def main() -> None:
    donors = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]

    # 1. α closed form on exact cell means: truth up to a constant
    cm = {s: {t: 10.0 + (i - 3) * 2.0 + b for t, b in zip("abcd", [1, 2, 3, 4])} for i, s in enumerate(donors)}
    al = alpha_from_cells(cm, donors, list("abcd"))
    assert all(abs((al[s] - al["s4"]) - (i - 3) * 2.0) < 1e-9 for i, s in enumerate(donors)), "alpha closed form wrong"

    # 2. exact permutation p and rank handling
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [7, 6, 5, 4, 3, 2, 1]
    assert abs(exact_perm_p_one_sided(x, y) - 1 / 5040) < 1e-12, "exact p for rho=-1 must be 1/5040"
    assert abs(spearman([1, 2, 2, 4], [1, 2, 2, 4]) - 1.0) < 1e-12, "midranks broke a perfect tie correlation"
    assert spearman([1] * 7, list(range(7))) is None, "constant x must give an undefined correlation"
    assert spearman(list(range(7)), [2.5] * 7) is None, "constant y must give an undefined correlation"
    assert exact_perm_p_one_sided([1] * 7, list(range(7))) is None, "constant ranks must not return p = 0"
    assert exact_perm_p_one_sided(list(range(7)), [0.0] * 7) is None, "constant α must not return p = 0"

    # 3. end to end: more damage → lower α; the least-damaged donor has small regret; four conditions reported
    true_alpha = {s: 6.0 - i * 2.0 for i, s in enumerate(donors)}   # s1 best ... s7 worst
    damage = {s: 0.01 * i for i, s in enumerate(donors)}             # s1 least damaged
    spec = make_spec(11, true_alpha, damage)
    res = analyse(spec, B=300, seed=1)
    assert res["selected_donor"] == "s1", res["selected_donor"]
    assert res["correlation_defined"] and res["exact_one_sided_p"] < 0.10, res["exact_one_sided_p"]
    assert res["regret"] < 3.0, res["regret"]
    assert sorted(res["verdict"]) == sorted([
        "spearman_p_lt_0.10", "regret_UB95_le_1.0", "selected_LB95_gt_0", "utility_regret_UB95_le_delta_U",
    ]), res["verdict"]
    assert res["g3"] in {"pass", "fail"} and all(v is not None for v in res["verdict"].values())
    assert res["strata"] == 5 and res["images"] == 200 and res["items"] == 600, (res["strata"], res["images"], res["items"])

    # 4. damage tie broken by NLLB code (alphabetical), for selection only
    spec2 = copy.deepcopy(spec)
    spec2["damage"]["s2"] = spec2["damage"]["s1"]
    spec2["donors"]["s2"] = "aaa_Latn"
    assert analyse(spec2, B=50, seed=1)["selected_donor"] == "s2", "damage tie must break by NLLB code"

    # 5. invariance to donor / target / subset / image insertion order
    spec3 = {
        "schema_version": SCHEMA_VERSION,
        "donors": reorder(spec["donors"]),
        "damage": reorder(spec["damage"]),
        "cells": {s: {t: {sub: reorder(imgs) for sub, imgs in reorder(subs).items()}
                      for t, subs in reorder(spec["cells"][s]).items()}
                  for s in reversed(list(spec["cells"]))},
    }
    r3 = analyse(spec3, B=300, seed=1)
    assert r3["regret_UB95"] == res["regret_UB95"] and r3["alpha"] == res["alpha"], "order changed the result"
    assert r3["selected_pooled_LB95"] == res["selected_pooled_LB95"] and r3["utility_regret_UB95"] == res["utility_regret_UB95"]

    # 6. incomplete table and incompatible universes must abort
    spec4 = copy.deepcopy(spec)
    del spec4["cells"]["s3"]["t2"]
    expect_exit("complete table", lambda: analyse(spec4, B=10))
    spec5 = copy.deepcopy(spec)
    spec5["cells"]["s3"]["t1"]["a"]["extra_img"] = {"extra_q": item(1.0, 1.0)}
    expect_exit("image universe differs", lambda: analyse(spec5, B=10))
    spec6 = copy.deepcopy(spec)
    first_img = sorted(spec6["cells"]["s5"]["t3"]["a"])[0]
    spec6["cells"]["s5"]["t3"]["a"][first_img]["ghost_question"] = item(0.0, 0.0)
    expect_exit("question ids differ", lambda: analyse(spec6, B=10))
    spec7 = copy.deepcopy(spec)
    spec7["cells"]["s2"]["t2"]["c"] = spec7["cells"]["s2"]["t2"]["b"]
    expect_exit("subsets", lambda: analyse(spec7, B=10))
    spec8 = copy.deepcopy(spec)
    spec8["damage"]["s4"] = float("nan")
    expect_exit("finite", lambda: analyse(spec8, B=10))
    spec9 = copy.deepcopy(spec)
    spec9["cells"]["s1"]["t1"]["a"][first_img.replace("t3", "t1")]["t1_a_img0_0"]["ground"] = float("inf")
    expect_exit("finite", lambda: analyse(spec9, B=10))
    spec10 = copy.deepcopy(spec)
    spec10["schema_version"] = 1
    expect_exit("schema_version", lambda: analyse(spec10, B=10))

    # 7. identical donors → regret and UB95 exactly zero on both endpoints (paired draws)
    #    This is the review's reproducer: 40 images with alternating outcomes, 400 resamples, seed 0.
    images = {str(i): {f"{i}_0": item(float(i % 2), float(i % 2))} for i in range(40)}
    same = {"schema_version": SCHEMA_VERSION, "donors": {s: f"{s}_Latn" for s in donors},
            "damage": {s: float(i) for i, s in enumerate(donors)},
            "cells": {s: {"t": {"all": copy.deepcopy(images)}} for s in donors}}
    r_same = analyse(same, B=400, seed=0)
    assert r_same["regret"] == 0.0 and r_same["regret_UB95"] == 0.0, (r_same["regret"], r_same["regret_UB95"])
    assert r_same["utility_regret"] == 0.0 and r_same["utility_regret_UB95"] == 0.0
    # identical donors also mean a constant α: the correlation is undefined, not significant
    assert r_same["spearman"] is None and r_same["exact_one_sided_p"] is None
    assert r_same["verdict"]["spearman_p_lt_0.10"] is None and r_same["g3"] == "undefined", r_same["g3"]

    # 8. constant ranks with a non-constant α: constant damage → undefined, never a pass
    spec11 = copy.deepcopy(spec)
    spec11["damage"] = {s: 0.5 for s in donors}
    r11 = analyse(spec11, B=20, seed=1)
    assert r11["exact_one_sided_p"] is None and r11["g3"] == "undefined", (r11["exact_one_sided_p"], r11["g3"])
    #    constant positive outcomes everywhere (the review's third case): no donor association, no pass
    const = copy.deepcopy(same)
    for s in donors:
        const["cells"][s]["t"] = {"all": {str(i): {f"{i}_0": item(1.0, 1.0)} for i in range(8)}}
    r_const = analyse(const, B=20, seed=0)
    assert r_const["g3"] == "undefined" and r_const["verdict"]["spearman_p_lt_0.10"] is None
    assert r_const["selected_pooled_LB95"] == 100.0  # the pooled guardrail alone is fine; G3 still is not a pass

    # 9. the selected donor's pooled Δ_ground is item-micro, α stays macro over targets
    two = {"schema_version": SCHEMA_VERSION, "donors": {"a": "a_Latn", "b": "b_Latn"},
           "damage": {"a": 0.0, "b": 1.0}, "cells": {}}
    big = {f"big{i}": {f"big{i}_q": item(1.0, 1.0)} for i in range(90)}        # 90 items, all +1
    small = {f"small{i}": {f"small{i}_q": item(-1.0, 0.0)} for i in range(10)}  # 10 items, all -1
    for s in ("a", "b"):
        two["cells"][s] = {"tb": {"x": copy.deepcopy(big)}, "ts": {"x": copy.deepcopy(small)}}
    r_two = analyse(two, B=30, seed=3)
    assert abs(r_two["selected_pooled_endpoint"] - 80.0) < 1e-9, r_two["selected_pooled_endpoint"]  # micro: (90-10)/100
    assert r_two["alpha"]["a"] == r_two["alpha"]["b"] == 0.0                                            # macro: equal donors
    assert r_two["selected_pooled_LB95"] > 0 and r_two["regret_UB95"] == 0.0

    # 10. joint resampling: with donors that differ by a constant shift on every item, α_b − α_b' is
    #     that constant in every replicate, so the regret distribution is degenerate at the point value
    shift = copy.deepcopy(same)
    for s_i, s in enumerate(donors):
        for img, qs in shift["cells"][s]["t"]["all"].items():
            for q in qs:
                qs[q]["ground"] = float(int(img) % 2) + 0.01 * s_i
                qs[q]["utility"] = float(int(img) % 2)
    r_shift = analyse(shift, B=200, seed=0)
    assert abs(r_shift["regret"] - r_shift["regret_UB95"]) < 1e-9, (r_shift["regret"], r_shift["regret_UB95"])
    assert abs(r_shift["regret"] - 6.0) < 1e-9, r_shift["regret"]   # 0.06 × 100 points between s1 and s7

    # 11. δ_U is honoured
    spec12 = copy.deepcopy(spec)
    spec12["delta_u"] = 1000.0
    assert analyse(spec12, B=20, seed=1)["verdict"]["utility_regret_UB95_le_delta_U"] is True
    spec13 = copy.deepcopy(spec)
    spec13["delta_u"] = -1.0
    expect_exit("non-negative", lambda: analyse(spec13, B=10))

    print("block_d: OK (alpha closed form, exact p=1/5040, midranks, undefined correlations, end-to-end with four "
          "conditions, tie rule, order invariance, incomplete/incompatible/non-finite aborts, identical donors -> "
          "regret 0 exactly, constant ranks -> undefined, micro pooling, joint resampling, delta_U)")


if __name__ == "__main__":
    main()
