"""Power of the S1 Block A contrast A1 − A2 on the real cluster structure (stdlib).

What is simulated
-----------------
Per-item paired differences D_i = d + ε_i between the current system (A1) and
a hypothetical ablation (A2), for true mean differences d in --deltas, on two
panels with their real items and image clusters:
  * xGQA pooled bn/de/ko: 12,578 questions on 398 images per language;
  * CVQA pooled jv/mn/ga: 935 items on 401 images.
The noise ε_i is not parametric: it is the observed, centred per-item paired
difference between two real arms on the same items, so its variance and its
within-image correlation are the data's own. A new realisation is generated
by flipping the sign of every image's ε values at once (cluster-level
Rademacher perturbation), which preserves within-image dependence.

Assumptions, stated
-------------------
1. Δ_ground has never been measured; its noise is proxied by Δ_gray's
   (correct − gray) per-item paired differences. Δ_ground noise is expected
   to be at least as large (shuffled images vary), so power is optimistic.
2. Comparators: xGQA uses bn zero-shot vs the target-supervised checkpoint;
   CVQA uses bn-source vs id-source zero-shot. Both are real arms on
   identical items.
3. Regions use the same stratified image-cluster bootstrap as
   e3_noninferiority.py, one-sided 5/95 bounds, δ = 1.0.
4. B and S are small by design (cluster-level sums make each bootstrap
   cheap); rerun with --boot 2000 --sims 400 for the paper's appendix.

Run from Approach2/results:
    python3 ../analysis/power_sim.py --out ../audits/power_sim.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import load_items, pct, region, rng_for  # noqa: E402


def cluster_sums(strata: dict[str, dict[str, list[float]]]):
    """-> {stratum: (keys, sums, counts)} with canonical order."""
    out = {}
    for s in sorted(strata):
        keys = sorted(strata[s])
        out[s] = (keys, [sum(strata[s][k]) for k in keys], [len(strata[s][k]) for k in keys])
    return out


def boot_bounds(cs, B: int, seed: int, label: str):
    tot = [0.0] * B
    cnt = [0] * B
    for s, (keys, sums, counts) in cs.items():
        C = len(keys)
        rnd = rng_for(seed, label, s)
        for b in range(B):
            for _ in range(C):
                j = rnd.randrange(C)
                tot[b] += sums[j]
                cnt[b] += counts[j]
    v = [100.0 * t / c for t, c in zip(tot, cnt)]
    return pct(v, .05), pct(v, .95), pct(v, .025), pct(v, .975)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.0, -0.5, -1.0, -2.0, -3.0])
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cluster-map", default="../../gqa_testdev_questions.jsonl")
    ap.add_argument("--out", default="../audits/power_sim.json")
    a = ap.parse_args()
    q2img = {}
    with open(a.cluster_map) as f:
        for line in f:
            r = json.loads(line)
            q2img[str(r["question_id"])] = r["image_id"]

    panels = {}
    # xGQA: A1 = bn checkpoint on L (zsbn; bn itself is supervised v4), comparator = target-supervised v4
    strata_g, strata_u = {}, {}
    for L in ["bn", "de", "ko"]:
        a1f = load_items(f"xgqa_{L}_v4") if L == "bn" else load_items(f"xgqa_{L}_zsbn")
        a1b = load_items(f"xgqa_{L}_BLIND_v4") if L == "bn" else load_items(f"xgqa_{L}_zsbn_BLIND")
        cf, cb = load_items(f"xgqa_{L}_v4"), load_items(f"xgqa_{L}_BLIND_v4")
        if L == "bn":
            cf, cb = load_items("xgqa_bn_v3"), load_items("xgqa_bn_BLIND_v3")  # same-language different-recipe arm
        g, u = {}, {}
        for i in a1f:
            g.setdefault(q2img[i], []).append((a1f[i] - a1b[i]) - (cf[i] - cb[i]))
            u.setdefault(q2img[i], []).append(a1f[i] - cf[i])
        strata_g[L], strata_u[L] = g, u
    panels["xGQA bn/de/ko"] = {"grounding(Δ_gray proxy)": strata_g, "utility": strata_u}
    strata_g, strata_u = {}, {}
    for L in ["jv", "mn", "ga"]:
        a1f, a1b = load_items(f"cvqa_{L}_zsbn"), load_items(f"cvqa_{L}_zsbn_BLIND")
        cf, cb = load_items(f"cvqa_{L}_zsid"), load_items(f"cvqa_{L}_zsid_BLIND")
        g, u = {}, {}
        for i in a1f:
            img = re.sub(r"_\d+$", "", i)
            g.setdefault(img, []).append((a1f[i] - a1b[i]) - (cf[i] - cb[i]))
            u.setdefault(img, []).append(a1f[i] - cf[i])
        strata_g[L], strata_u[L] = g, u
    panels["CVQA jv/mn/ga"] = {"grounding(Δ_gray proxy)": strata_g, "utility": strata_u}

    results = {}
    desc = {}
    print(f"power_sim | δ={a.margin} | B={a.boot} S={a.sims} seed={a.seed} | regions on one-sided image-cluster bounds")
    print("assumed quantities per panel/endpoint (from the real arms): baseline = mean of the A1 endpoint; "
          "discordance = share of items whose paired difference is non-zero; ICC = intra-image correlation of ε (ANOVA estimator)")
    for pname, eps in panels.items():
        for ename, strata in eps.items():
            vals = [v for st in strata.values() for cl in st.values() for v in cl]
            n = len(vals); mean = sum(vals) / n
            disc = sum(1 for v in vals if v != 0) / n
            # one-way ANOVA ICC over image clusters
            groups = [cl for st in strata.values() for cl in st.values()]
            k = len(groups); n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
            gm = [sum(g) / len(g) for g in groups]
            msb = sum(len(g) * (m - mean) ** 2 for g, m in zip(groups, gm)) / (k - 1)
            msw = sum((v - m) ** 2 for g, m in zip(groups, gm) for v in g) / (n - k)
            icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
            desc.setdefault(pname, {})[ename] = {"items": n, "images": k, "mean_paired_diff": 100 * mean,
                                                 "discordance": disc, "icc": icc}
            print(f"  {pname:16s}{ename:24s} items={n:5d} images={k:4d} mean(ε+d0)={100*mean:+.2f} "
                  f"discordance={disc:.2f} ICC={icc:.3f}")
    print(f"{'panel':16s}{'endpoint':24s}{'d':>6}{'P(NI)':>8}{'P(inc)':>8}{'P(MI)':>8}{'half-width':>12}")
    for pname, eps in panels.items():
        for ename, strata in eps.items():
            n = sum(len(v) for s in strata.values() for v in s.values())
            # centre ε within the whole panel
            mean = sum(sum(v) for s in strata.values() for v in s.values()) / n
            for d in a.deltas:
                rnd = rng_for(a.seed, "power", pname, ename, str(d))
                counts = {"non-inferior": 0, "inconclusive": 0, "mat. inferior": 0}
                widths = []
                for sidx in range(a.sims):
                    sim = {}
                    for s, cl in strata.items():
                        sim[s] = {}
                        for k, vals in cl.items():
                            flip = 1 if rnd.random() < 0.5 else -1
                            sim[s][k] = [d / 100.0 + flip * (v - mean) for v in vals]
                    lb5, ub95, lo, hi = boot_bounds(cluster_sums(sim), a.boot, a.seed + sidx, f"{pname}:{ename}:{d}")
                    counts[region(lb5, ub95, a.margin)] += 1
                    widths.append((hi - lo) / 2)
                row = {k: v / a.sims for k, v in counts.items()}
                row["half_width"] = sum(widths) / len(widths)
                results.setdefault(pname, {}).setdefault(ename, {})[str(d)] = row
                print(f"{pname:16s}{ename:24s}{d:6.1f}{row['non-inferior']:8.2f}{row['inconclusive']:8.2f}"
                      f"{row['mat. inferior']:8.2f}{row['half_width']:12.2f}")
    # G3 attainable power: seven donors, true α = -slope*rank(damage) + noise, exact one-sided permutation p < 0.10
    import itertools
    from block_d import spearman  # noqa: E402
    g3 = {}
    print("G3 power grid: 7 donors, α_conf = −1.0·rank(damage) + N(0, σ) points, exact one-sided permutation test at 0.10")
    perms = list(itertools.permutations(range(7)))
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        rnd = rng_for(a.seed, "g3", str(sigma))
        hits = 0
        S3 = 80
        for _ in range(S3):
            x = list(range(7))
            y = [-1.0 * r + rnd.gauss(0, sigma) for r in x]
            obs = spearman(x, y)
            p = sum(1 for pp in perms if spearman(x, [y[i] for i in pp]) <= obs + 1e-12) / len(perms)
            hits += p < 0.10
        g3[str(sigma)] = hits / S3
        print(f"  σ={sigma:.1f}: P(p<0.10) = {hits / S3:.2f}")
    json.dump({"margin": a.margin, "boot": a.boot, "sims": a.sims, "seed": a.seed, "descriptives": desc, "g3_power": g3,
               "assumptions": __doc__.split("Assumptions, stated")[1].strip(), "results": results},
              open(a.out, "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
