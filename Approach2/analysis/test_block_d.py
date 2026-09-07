"""Synthetic tests for block_d.py: α recovery, exact permutation p, regret, ties, invariance."""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_d import analyse, exact_perm_p_one_sided, spearman, alpha_from_cells  # noqa: E402


def make_spec(seed: int, true_alpha: dict[str, float], damage: dict[str, float], n_img: int = 60, per_img: int = 3):
    rnd = random.Random(seed)
    targets = ["t1", "t2", "t3", "t4"]
    beta = {"t1": 2.0, "t2": -1.0, "t3": 0.5, "t4": -1.5}
    cells = {}
    for s, a in true_alpha.items():
        cells[s] = {}
        for t in targets:
            p = 0.30 + (a + beta[t]) / 100.0
            cells[s][t] = {f"img{k}": [1 if rnd.random() < p else 0 for _ in range(per_img)] for k in range(n_img)}
    return {"donors": {s: f"{s}_Latn" for s in true_alpha}, "damage": damage, "cells": cells}


def main() -> None:
    donors = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    # 1. α recovery on exact cell means (no noise): closed form must return the truth up to a constant
    cm = {s: {t: 10.0 + (i - 3) * 2.0 + b for t, b in zip("abcd", [1, 2, 3, 4])} for i, s in enumerate(donors)}
    al = alpha_from_cells(cm, donors, list("abcd"))
    assert all(abs((al[s] - al["s4"]) - (i - 3) * 2.0) < 1e-9 for i, s in enumerate(donors)), "alpha closed form wrong"
    # 2. exact permutation p: perfectly monotone negative association over 7 → 1/5040
    x = [1, 2, 3, 4, 5, 6, 7]; y = [7, 6, 5, 4, 3, 2, 1]
    assert abs(exact_perm_p_one_sided(x, y) - 1 / 5040) < 1e-12, "exact p for rho=-1 must be 1/5040"
    assert abs(spearman([1, 2, 2, 4], [1, 2, 2, 4]) - 1.0) < 1e-12, "midranks broke a perfect tie correlation"
    # 3. end to end: more damage → lower α, with noise; the best-by-damage donor should have small regret
    true_alpha = {s: 6.0 - i * 2.0 for i, s in enumerate(donors)}   # s1 best ... s7 worst
    damage = {s: 0.01 * i for i, s in enumerate(donors)}             # s1 least damaged
    spec = make_spec(11, true_alpha, damage)
    res = analyse(spec, B=300, seed=1)
    assert res["selected_donor"] == "s1", res["selected_donor"]
    assert res["exact_one_sided_p"] < 0.10, res["exact_one_sided_p"]
    assert res["regret"] < 3.0, res["regret"]
    # 4. tie in damage broken by code (alphabetical); tie in α not needed for selection
    spec2 = json.loads(json.dumps(spec)); spec2["damage"]["s2"] = spec2["damage"]["s1"]
    spec2["donors"]["s2"] = "aaa_Latn"
    assert analyse(spec2, B=50, seed=1)["selected_donor"] == "s2", "damage tie must break by NLLB code"
    # 5. invariance to donor/target insertion order
    spec3 = {"donors": dict(reversed(list(spec["donors"].items()))), "damage": spec["damage"],
             "cells": {s: dict(reversed(list(spec["cells"][s].items()))) for s in reversed(list(spec["cells"]))}}
    r3 = analyse(spec3, B=300, seed=1)
    assert r3["regret_UB95"] == res["regret_UB95"] and r3["alpha"] == res["alpha"], "order changed the result"
    # 6. incomplete table must abort
    spec4 = json.loads(json.dumps(spec)); del spec4["cells"]["s3"]["t2"]
    try:
        analyse(spec4, B=10); raise AssertionError("incomplete table accepted")
    except SystemExit:
        pass
    print("block_d: OK (alpha closed form, exact p=1/5040, midranks, end-to-end, tie rule, order invariance, incomplete-table abort)")


if __name__ == "__main__":
    main()
