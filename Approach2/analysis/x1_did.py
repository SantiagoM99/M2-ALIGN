"""X1 as a paired difference-in-differences on identical CVQA items.

The 2026-09-06 record reported "id significant (p=0.00085), bn not
(p=0.36)" and a McNemar on full accuracy (p=0.051). Neither is the
estimand. Per target and pooled over a named group:

    DiD = ΔV_A − ΔV_B,   ΔV = correct − gray  (Δ_gray), per item.

CVQA ids are "<image_id>_<k>" with up to three questions per image. Primary
interval: image-cluster bootstrap, stratified by target for the pooled
estimate, one RNG per (estimand, target) derived from --seed, canonical
ordering, so the pooled interval is invariant to the order of --targets
and to extra targets outside --pool (tested by analysis/test_invariance.py).
Fail-closed: aborts on invalid result schemas, duplicate ids, unequal item sets across the four
arms, ids without the "<image>_<k>" structure, repeated targets, or a
--pool member absent from --targets. Two-sided 95% CIs. Retention ratios
are not computed.

Run from Approach2/results:
    python3 ../analysis/x1_did.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import boot_stratified, load_items, pct  # noqa: E402

ID_RE = re.compile(r"^(?P<image>\d+)_(?P<k>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="id")
    ap.add_argument("--b", default="bn")
    ap.add_argument("--bench", default="cvqa")
    ap.add_argument("--targets", nargs="+", default=["jv", "mn", "ga", "si"])
    ap.add_argument("--pool", nargs="+", default=["jv", "mn", "ga"])
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if len(set(a.targets)) != len(a.targets):
        raise SystemExit("repeated target in --targets")
    if len(set(a.pool)) != len(a.pool):
        raise SystemExit("repeated target in --pool")
    if not set(a.pool) <= set(a.targets):
        raise SystemExit(f"--pool members not in --targets: {sorted(set(a.pool) - set(a.targets))}")
    if a.boot <= 0:
        raise SystemExit("--boot must be positive")
    est = f"x1:{a.bench}:{a.a}-{a.b}"
    print(f"X1 DiD = ΔV_{a.a} − ΔV_{a.b} on identical {a.bench} items | two-sided 95% CIs | "
          f"B={a.boot} seed={a.seed} | RNG per (estimand, target), canonical order")
    print(f"{'target':8s}{'items':>6}{'images':>8}{'DiD':>8}{'image-cluster CI':>20}{'item CI (ref.)':>18}")
    strata: dict[str, dict[str, list[int]]] = {}
    for L in sorted(a.targets):
        af, ab = load_items(f"{a.bench}_{L}_zs{a.a}"), load_items(f"{a.bench}_{L}_zs{a.a}_BLIND")
        bf, bb = load_items(f"{a.bench}_{L}_zs{a.b}"), load_items(f"{a.bench}_{L}_zs{a.b}_BLIND")
        sets = [set(af), set(ab), set(bf), set(bb)]
        if any(x != sets[0] for x in sets[1:]):
            diff = set.union(*sets) - set.intersection(*sets)
            raise SystemExit(f"{L}: the four arms do not share identical item sets ({len(diff)} ids differ)")
        cl: dict[str, list[int]] = {}
        items: dict[str, list[int]] = {}
        for i in sorted(sets[0]):
            m = ID_RE.match(i)
            if not m:
                raise SystemExit(f"{L}: id {i!r} lacks the '<image>_<k>' structure; cannot cluster")
            v = (af[i] - ab[i]) - (bf[i] - bb[i])
            cl.setdefault(m.group("image"), []).append(v)
            items[i] = [v]
        n = len(items)
        point = 100 * sum(sum(v) for v in cl.values()) / n
        cb = boot_stratified({L: cl}, a.boot, a.seed, est + ":image")
        ib = boot_stratified({L: items}, a.boot, a.seed, est + ":item")
        print(f"{L:8s}{n:6d}{len(cl):8d}{point:+8.2f}     [{pct(cb, .025):+.2f}, {pct(cb, .975):+.2f}]"
              f"     [{pct(ib, .025):+.2f}, {pct(ib, .975):+.2f}]")
        if L in a.pool:
            strata[L] = cl
    n = sum(len(v) for s in strata.values() for v in s.values())
    point = 100 * sum(sum(v) for s in strata.values() for v in s.values()) / n
    cb = boot_stratified(strata, a.boot, a.seed, est + ":image")
    print(f"pooled {' '.join(sorted(a.pool))}: items={n} images={sum(len(s) for s in strata.values())}  "
          f"DiD={point:+.2f}  stratified image-cluster 95% CI [{pct(cb, .025):+.2f}, {pct(cb, .975):+.2f}]")
    print("Three targets, not 935 independent linguistic observations; per-target rows are the honest unit.")


if __name__ == "__main__":
    main()
