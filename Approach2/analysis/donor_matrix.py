"""Donor quality: which language should you train the multimodal bridge in?

This is the project's question in its current form. E3 established that the
visual pathway is language-blind by construction, so all cross-lingual
variation lives in the text bridge; X1 established that the variation is a
property of the source-target PAIR, not of the target. What remains is the
practical half of the original question -- "can you tell before you build
it?" -- which now reads: can a label-free measure say in advance which
donor language will transfer to which target?

Two tables, and one test that joins them:

  1. TRANSFER   dV and retention for every measured (source, target) pair,
                from source_ablation.sh output, plus per-source donor quality
                averaged over targets.
  2. ALIGNMENT  pairwise FLORES retrieval@1 from pair_alignment.py, if it has
                run.
  3. The pre-registered test: does alignment[src][tgt] predict retention for
     the same pair? n is the number of measured cells, which is far larger
     than the 9 languages X2 had to work with.

Run from Approach2/results:
    python3 ../analysis/donor_matrix.py
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os


def load(tag: str) -> dict | None:
    p = f"eval_{tag}.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            out[r.get("id", r.get("question_id", k))] = 1 if r.get("correct") else 0
    return out


def dv(full: dict, blind: dict) -> tuple[float, int]:
    ids = [i for i in full if i in blind]
    return 100.0 * sum(full[i] - blind[i] for i in ids) / len(ids), len(ids)


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def perm_p(x, y, limit=20000):
    obs = abs(spearman(x, y))
    if len(y) <= 8:
        perms = list(itertools.permutations(range(len(y))))
    else:
        import random
        rnd = random.Random(0)
        idx = list(range(len(y)))
        perms = []
        for _ in range(limit):
            rnd.shuffle(idx)
            perms.append(tuple(idx))
    hits = sum(1 for p in perms if abs(spearman(x, [y[i] for i in p])) >= obs - 1e-12)
    return hits / len(perms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cvqa")
    # Alignment must be read from the SOURCE's own checkpoint: transferring from
    # S runs S's mapping, so S's prefix space is the one T has to land in.
    # Reading every row out of one file (the original default) measures a
    # different bridge for every source but bn, and inflates the apparent
    # correlation with target difficulty. --align is now only the fallback for
    # sources whose own pairalign_stage3_<S>_v4.json has not been scored.
    ap.add_argument("--align", default="pairalign_stage3_bn_dcl.json",
                    help="fallback only; per-source files are preferred")
    ap.add_argument("--metric", default="retrieval@1")
    ap.add_argument("--space", default="centered", choices=("centered", "raw"))
    args = ap.parse_args()
    B = args.bench

    # Discover every measured (source, target) pair from the filenames.
    pairs = {}
    for path in glob.glob(f"eval_{B}_*_zs*.jsonl"):
        base = os.path.basename(path)[:-6]
        if base.endswith("_BLIND"):
            continue
        body = base[len(f"eval_{B}_"):]
        tgt, _, src = body.partition("_zs")
        if not src:
            continue
        f, b = load(f"{B}_{tgt}_zs{src}"), load(f"{B}_{tgt}_zs{src}_BLIND")
        if not (f and b):
            continue
        pairs[(src, tgt)] = dv(f, b)[0]

    sup = {}
    for (_, tgt) in pairs:
        if tgt in sup:
            continue
        f, b = load(f"{B}_{tgt}_v4"), load(f"{B}_{tgt}_BLIND_v4")
        if f and b:
            sup[tgt] = dv(f, b)[0]

    sources = sorted({s for s, _ in pairs})
    targets = sorted({t for _, t in pairs})

    # Donor quality is only comparable over a COMMON set of targets. Averaging
    # each source over whatever it happens to have been run on rewards the
    # source that was run on the easy targets: Bengali covers nine including
    # pt/ru/si, the others cover only the three hard ones, and the naive mean
    # would rank Bengali first and invert X1's finding.
    common = sorted(t for t in targets
                    if all((s, t) in pairs for s in sources) and sup.get(t))
    print(f"=== {B.upper()} transfer dV, matrix[source][target] "
          f"({len(pairs)} measured pairs) ===\n")
    print(f"{'src':<7}" + "".join(f"{t:>8}" for t in targets) + f"{'donor':>9}")
    for s in sources:
        cells = []
        for t in targets:
            cells.append(f"{pairs[(s, t)]:+.2f}" if (s, t) in pairs else "--")
        rets = [100.0 * pairs[(s, t)] / sup[t] for t in common]
        donor = f"{sum(rets) / len(rets):.0f}%" if rets else "--"
        print(f"{s:<7}" + "".join(f"{c:>8}" for c in cells) + f"{donor:>9}")
    print(f"{'SUPERV':<7}" + "".join(
        f"{sup[t]:>+8.2f}" if t in sup else f"{'--':>8}" for t in targets))
    print(f"\n'donor' = mean retention over the {len(common)} target(s) every source was")
    print(f"run on ({' '.join(common) if common else 'none yet'}), which is the only")
    print("comparable basis. It is the number this project should be selecting on,")
    print("and it was never measured before X1: every transfer result until then")
    print("used Bengali by default.")
    if common and len(common) < len(targets):
        print(f"NOTE: {len(targets) - len(common)} target(s) are missing for at least one")
        print("source and are shown but excluded from the donor score. Finish the")
        print("matrix to widen it.")

    if not os.path.exists(args.align):
        found = sorted(glob.glob("pairalign_*.json"))
        print(f"\nNo {args.align} yet — pairwise alignment (X2b) has not run.")
        print(f"Available: {found or '(none)'}")
        return

    def align_for(src: str):
        """The pairwise matrix from src's OWN checkpoint, else the fallback."""
        own = (f"pairalign_stage3_{src}_v4.json", f"pairalign_stage3_{src}_dcl.json")
        for cand in own + (args.align,):
            if os.path.exists(cand):
                return json.load(open(cand))["matrix"][args.space], cand not in own
        return None, True

    xs, ys, labels, borrowed = [], [], [], []
    for (s, t), d in sorted(pairs.items()):
        mat, is_fallback = align_for(s)
        if mat is None or s not in mat or t not in mat[s] or not sup.get(t):
            continue
        xs.append(mat[s][t][args.metric])
        ys.append(100.0 * d / sup[t])
        labels.append(f"{s}->{t}" + ("*" if is_fallback else ""))
        if is_fallback:
            borrowed.append(s)
    print(f"\n=== Does pairwise alignment predict pairwise transfer? "
          f"({args.space} {args.metric}) ===\n")
    print(f"{'pair':<10}{'alignment':>11}{'retention%':>12}")
    for lab, x, y in sorted(zip(labels, xs, ys), key=lambda r: -r[1]):
        print(f"{lab:<10}{x:>11.3f}{y:>12.1f}")
    if borrowed:
        print(f"\n* {len(set(borrowed))} source(s) have no own-checkpoint alignment "
              f"({' '.join(sorted(set(borrowed)))}) and fall back to {args.align}.")
        print("  Those rows measure a different bridge than the one that transfers —")
        print("  score them with pair_alignment.sh before reading their cells.")
    if len(xs) >= 5:
        print(f"\nn={len(xs)} pairs  Pearson r={pearson(xs, ys):+.3f}  "
              f"Spearman rho={spearman(xs, ys):+.3f}  permutation p={perm_p(xs, ys):.4f}")
        # The global correlation is dominated by target difficulty: easy targets
        # have both high alignment and high retention. The question that matters
        # is whether alignment picks the SOURCE for a fixed target.
        by_t: dict[str, list] = {}
        for lab, x, y in zip(labels, xs, ys):
            by_t.setdefault(lab.split("->")[1].rstrip("*"), []).append((x, y))
        within = [spearman([a for a, _ in v], [b for _, b in v])
                  for v in by_t.values() if len(v) >= 3]
        if within:
            print(f"mean WITHIN-target Spearman = {sum(within) / len(within):+.3f} "
                  f"over {len(within)} targets  <-- this is the one that matters")
        print("\nPre-registered (DESIGN.md X2): this must rank id above bn and ru as a")
        print("donor for jv/mn/ga, and reproduce zh high for ga / low for mn. A")
        print("correlation that misses that dissociation is not the mechanism.")
    else:
        print(f"\nOnly {len(xs)} pairs overlap — run more of the donor matrix first.")


if __name__ == "__main__":
    main()
