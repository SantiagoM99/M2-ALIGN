"""E3 against the target-supervised reference, with non-inferiority regions.

The 2026-09-06 record compared each zero-shot target against Bengali's own
supervised ΔV. The conventional reference is the target's own supervised
checkpoint (stage3_<L>_v4 on L). Per target, on identical items:

    D = ΔV_zs − ΔV_sup,   ΔV = correct − gray  (Δ_gray)

Bootstraps: by item (reference only) and by image cluster (primary; xGQA has
398 images behind 12,578 questions, mapped through the GQA test-dev file).
One RNG per (estimand, target), canonical ordering: a target's interval
never depends on which other targets are requested (analysis/test_invariance.py).
Pooled: the same image is resampled jointly across all its translations,
which requires every translation to carry the identical id universe.
Fail-closed: aborts on an invalid result schema, missing map, duplicate ids, unmapped ids, unequal
item sets across the four arms of a target, or unequal universes across
targets. Prints two-sided 95% CIs (2.5/97.5) for tables and one-sided 95%
bounds (5/95) for the three regions:
  non-inferior if the 5th percentile of D > −δ; materially inferior if the
  95th percentile < −δ; inconclusive otherwise.
Margins given after the data were seen are sensitivity analyses.

Run from Approach2/results:
    python3 ../analysis/e3_noninferiority.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import boot_stratified, load_items, pct, region  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bn")
    ap.add_argument("--targets", nargs="+", default=["de", "ru", "zh", "pt", "id", "ko"])
    ap.add_argument("--round", default="v4")
    ap.add_argument("--bench", default="xgqa")
    ap.add_argument("--deltas", nargs="+", type=float, default=[1.0, 1.75])
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cluster-map", default="../../gqa_testdev_questions.jsonl")
    a = ap.parse_args()
    if len(set(a.targets)) != len(a.targets):
        raise SystemExit("repeated target in --targets")
    if a.boot <= 0:
        raise SystemExit("--boot must be positive")
    if any(delta <= 0 for delta in a.deltas):
        raise SystemExit("every --deltas value must be positive")
    if not os.path.exists(a.cluster_map):
        raise SystemExit(f"{a.cluster_map} not found: the image-cluster bootstrap is primary and "
                         "cannot be replaced by an item bootstrap silently")
    q2img: dict[str, str] = {}
    with open(a.cluster_map) as f:
        for lineno, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{a.cluster_map}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(r, dict) or "question_id" not in r or "image_id" not in r:
                raise SystemExit(f"{a.cluster_map}:{lineno}: expected question_id and image_id")
            q = str(r["question_id"])
            image = str(r["image_id"])
            if not q.strip() or not image.strip():
                raise SystemExit(f"{a.cluster_map}:{lineno}: empty question_id or image_id")
            if q in q2img:
                raise SystemExit(f"{a.cluster_map}: duplicate question_id {q}")
            q2img[q] = image
    est = f"e3:{a.bench}:{a.source}-vs-{a.round}"
    print(f"E3 vs target-supervised | {a.bench} | source={a.source} round={a.round} | "
          f"D = ΔV_zs − ΔV_sup | B={a.boot} seed={a.seed} | RNG per (estimand, target), canonical order")
    print(f"{'lang':5s}{'sup U / ΔV':>13}{'zs U / ΔV':>13}{'ret%':>7}{'D':>7}"
          f"{'item 95% CI':>17}{'image 95% CI':>17}{'image 1-sided[5,95]':>22}"
          + "".join(f"{'δ=' + str(dl):>15}" for dl in a.deltas))
    means = [0.0] * 4
    universe: set[str] | None = None
    per_image_all: dict[str, list[int]] = {}
    for L in sorted(a.targets):
        sf, sb = load_items(f"{a.bench}_{L}_{a.round}"), load_items(f"{a.bench}_{L}_BLIND_{a.round}")
        zf, zb = load_items(f"{a.bench}_{L}_zs{a.source}"), load_items(f"{a.bench}_{L}_zs{a.source}_BLIND")
        sets = [set(sf), set(sb), set(zf), set(zb)]
        if any(x != sets[0] for x in sets[1:]):
            diff = set.union(*sets) - set.intersection(*sets)
            raise SystemExit(f"{L}: the four arms do not share identical item sets ({len(diff)} ids differ)")
        if universe is None:
            universe = sets[0]
        elif sets[0] != universe:
            raise SystemExit(f"{L}: id universe differs from the other translations "
                             f"({len(sets[0] ^ universe)} ids); the joint pooled bootstrap needs one universe")
        missing = [i for i in sets[0] if i not in q2img]
        if missing:
            raise SystemExit(f"{L}: {len(missing)} ids have no image in {a.cluster_map}")
        ids = sorted(sets[0])
        d = {i: (zf[i] - zb[i]) - (sf[i] - sb[i]) for i in ids}
        cl: dict[str, list[int]] = {}
        for i in ids:
            cl.setdefault(q2img[i], []).append(d[i])
            per_image_all.setdefault(q2img[i], []).append(d[i])
        n = len(ids)
        point = 100 * sum(d.values()) / n
        ib = boot_stratified({L: {i: [d[i]] for i in ids}}, a.boot, a.seed, est + ":item")
        cb = boot_stratified({L: cl}, a.boot, a.seed, est + ":image")
        m = lambda v: 100 * sum(v) / len(v)
        su, sdv = m([sf[i] for i in ids]), m([sf[i] - sb[i] for i in ids])
        zu, zdv = m([zf[i] for i in ids]), m([zf[i] - zb[i] for i in ids])
        for k, v in enumerate((su, sdv, zu, zdv)):
            means[k] += v / len(a.targets)
        lb5, ub95 = pct(cb, .05), pct(cb, .95)
        retention = f"{100 * zdv / sdv:7.1f}" if sdv else f"{'NA':>7}"
        print(f"{L:5s}{su:6.2f}/{sdv:5.2f} {zu:6.2f}/{zdv:5.2f}{retention}{point:+7.2f}"
              f"  [{pct(ib, .025):+.2f},{pct(ib, .975):+.2f}]  [{pct(cb, .025):+.2f},{pct(cb, .975):+.2f}]"
              f"       [{lb5:+.2f},{ub95:+.2f}]"
              + "".join(f"{region(lb5, ub95, dl):>15}" for dl in a.deltas)
              + f"   n={n} images={len(cl)}")
    mean_retention = f"{100 * means[3] / means[1]:7.1f}" if means[1] else f"{'NA':>7}"
    print(f"mean  {means[0]:6.2f}/{means[1]:5.2f} {means[2]:6.2f}/{means[3]:5.2f}"
          f"{mean_retention}{means[3] - means[1]:+7.2f}")
    pooled_label = est + ":image-joint:" + "+".join(sorted(a.targets))
    cb = boot_stratified({"all": per_image_all}, a.boot, a.seed, pooled_label)
    n_all = sum(len(v) for v in per_image_all.values())
    d_all = 100 * sum(sum(v) for v in per_image_all.values()) / n_all
    print(f"pooled over {len(a.targets)} targets, images resampled jointly across translations: "
          f"D={d_all:+.2f}  two-sided 95% CI [{pct(cb, .025):+.2f}, {pct(cb, .975):+.2f}]  "
          f"one-sided [5,95] [{pct(cb, .05):+.2f}, {pct(cb, .95):+.2f}]  (clusters={len(per_image_all)})")
    print("Regions are read on the image-cluster one-sided bounds. Margins chosen after seeing the data are sensitivity analyses.")


if __name__ == "__main__":
    main()
