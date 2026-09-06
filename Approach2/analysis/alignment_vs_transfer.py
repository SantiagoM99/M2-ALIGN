"""Does text-bridge alignment predict zero-shot multimodal transfer?

The pre-registered discriminating prediction (DESIGN.md, "Sufficiency is not
alignment", 2026-09-06):

    the alignment score must correlate with transfer retention and must NOT
    correlate with % of frozen-LLM ceiling. If it correlates with both, it is
    measuring general bridge quality, the alignment mechanism is unsupported,
    and the finding degrades from mechanism to description.

This script evaluates exactly that, and prints the verdict it was told to
print before the data existed. It reports Spearman alongside Pearson because
n is 9-11 languages and one outlier can manufacture a Pearson r, and it
attaches an exact permutation p-value because at that n the usual asymptotic
p-values are not trustworthy.

Run from Approach2/results:
    python3 ../analysis/alignment_vs_transfer.py [--label stage3_bn_dcl]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os

BENCHES = ("mgsm", "msvamp")


def acc(tag: str) -> float | None:
    path = f"eval_{tag}.jsonl.summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return 100.0 * json.load(f)["accuracy"]


def dv(full_tag: str, blind_tag: str) -> float | None:
    """Full-image minus blind accuracy.

    The two arms do not share a naming convention: the supervised round files
    are eval_cvqa_<L>_BLIND_v4 (evaluate_all.sh puts BLIND before the round)
    while the zero-shot ones are eval_cvqa_<L>_zsbn_BLIND. Both tags are
    therefore passed explicitly rather than derived.
    """
    f, b = acc(full_tag), acc(blind_tag)
    if f is None or b is None:
        return None
    return f - b


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def perm_p(x: list[float], y: list[float], stat=spearman, limit: int = 200000) -> float:
    """Two-sided exact permutation p-value, or a deterministic subsample of it."""
    obs = abs(stat(x, y))
    perms = list(itertools.permutations(range(len(y))))
    if len(perms) > limit:
        step = len(perms) // limit
        perms = perms[::step]
    hits = sum(1 for p in perms if abs(stat(x, [y[i] for i in p])) >= obs - 1e-12)
    return hits / len(perms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="stage3_bn_dcl",
                    help="which alignment_<label>.json to test (default: the bridge that transfers)")
    ap.add_argument("--metric", default="margin", choices=("margin", "retrieval@1", "retrieval@5"))
    ap.add_argument("--reference", default="bridge", choices=("bridge", "llm"))
    args = ap.parse_args()

    path = f"alignment_{args.label}.json"
    if not os.path.exists(path):
        found = sorted(glob.glob("alignment_*.json"))
        print(f"No {path}. Available: {found or '(none — X2 has not run yet)'}")
        return
    with open(path) as f:
        align = json.load(f)["languages"]

    rows = []
    for iso, r in align.items():
        score = r[args.reference][args.metric]

        # Dependent variable 1: zero-shot multimodal transfer retention.
        zs = dv(f"cvqa_{iso}_zsbn", f"cvqa_{iso}_zsbn_BLIND")
        sup = dv(f"cvqa_{iso}_v4", f"cvqa_{iso}_BLIND_v4")
        retention = 100.0 * zs / sup if (zs is not None and sup and sup > 0) else None

        # Dependent variable 2 (the one it must NOT track): text sufficiency.
        pct = []
        for b in BENCHES:
            suffix = "dcl" if iso == "bn" else "v4"
            a, c = acc(f"{b}_{iso}_stage3_{suffix}"), acc(f"{b}_{iso}_gemma")
            if a is not None and c:
                pct.append(100.0 * a / c)
        ceiling_pct = sum(pct) / len(pct) if pct else None
        rows.append((iso, score, retention, ceiling_pct))

    print(f"=== alignment ({args.reference} {args.metric}) from {path} ===\n")
    print(f"{'lang':<6}{'alignment':>11}{'transfer ret%':>15}{'% of ceiling':>14}")
    for iso, s, ret, pct in sorted(rows, key=lambda r: -(r[1] or 0)):
        f2 = lambda v, w: (f"{v:>{w}.1f}" if v is not None else f"{'--':>{w}}")
        print(f"{iso:<6}{s:>11.4f}{f2(ret, 15)}{f2(pct, 14)}")

    print()
    for name, idx in (("transfer retention (MUST correlate)", 2),
                      ("% of ceiling (must NOT correlate)", 3)):
        pairs = [(r[1], r[idx]) for r in rows if r[idx] is not None]
        if len(pairs) < 4:
            print(f"{name}: only {len(pairs)} languages — not enough to test")
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        rp, rs = pearson(x, y), spearman(x, y)
        p = perm_p(x, y)
        print(f"{name}: n={len(pairs)}  Pearson r={rp:+.3f}  Spearman rho={rs:+.3f}  "
              f"permutation p={p:.4f}")

    # The verdict this script was told to print before the data existed.
    tr = [(r[1], r[2]) for r in rows if r[2] is not None]
    ce = [(r[1], r[3]) for r in rows if r[3] is not None]
    print()
    if len(tr) >= 4 and len(ce) >= 4:
        p_tr = perm_p([a for a, _ in tr], [b for _, b in tr])
        p_ce = perm_p([a for a, _ in ce], [b for _, b in ce])
        if p_tr < 0.05 and p_ce >= 0.05:
            print("VERDICT: prediction HELD — alignment tracks transfer and not sufficiency.")
            print("         The mechanism claim is supported.")
        elif p_tr < 0.05 and p_ce < 0.05:
            print("VERDICT: prediction FAILED — alignment tracks both. It is measuring")
            print("         general bridge quality; report this as description, not mechanism.")
        elif p_tr >= 0.05:
            print("VERDICT: prediction FAILED — alignment does not track transfer at all.")
            print("         The alignment hypothesis is not supported by this instrument.")
    print()
    print("Caveat that applies to every line above: CVQA retention is noisy")
    print("(n=200-412/language, +/-2.8 on dV) and disagrees with xGQA for zh, ko and id.")
    print("Only the pooled jv/mn/ga group result is solid. Treat a per-language")
    print("regression over these points as suggestive, never as the headline.")


if __name__ == "__main__":
    main()
