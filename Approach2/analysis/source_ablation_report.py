"""X1 — does the transfer failure belong to the target language or the source?

Reads the source x target matrix produced by job-scripts/source_ablation.sh
and answers the discriminator it was built for:

    H_source  jv/mn/ga fail because BENGALI is a bad source for them.
              Predicts: a closer source (id->jv) transfers where bn does not.
    H_align   jv/mn/ga fail because their own text bridge is misaligned.
              Predicts: they fail from every source alike.

The readable statistic is the pooled paired McNemar over the failing group,
not per-language dV: CVQA is n=200-412 per language with a +/-2.8 noise floor
on dV, and E2's group finding (n=935, p=0.36 flat) is the only thing at this
scale that held up.

Run from Approach2/results:
    python3 ../analysis/source_ablation_report.py
"""
from __future__ import annotations

import argparse
import json
import math
import os

FAIL_GROUP = ["jv", "mn", "ga"]
CONTROL = ["si"]


def load(tag: str) -> dict | None:
    path = f"eval_{tag}.jsonl"
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            out[r.get("id", r.get("question_id", k))] = 1 if r.get("correct") else 0
    return out


def mcnemar(a: dict, b: dict) -> tuple[int, int, float, int]:
    """Exact two-sided McNemar of a vs b over shared ids. Returns (a_only, b_only, p, n)."""
    ids = [i for i in a if i in b]
    n01 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    n10 = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0, len(ids)
    k = min(n01, n10)
    lp = lambda j: (math.lgamma(n + 1) - math.lgamma(j + 1)
                    - math.lgamma(n - j + 1) - n * math.log(2))
    return n01, n10, min(1.0, 2 * sum(math.exp(lp(j)) for j in range(k + 1))), len(ids)


def dv(full: dict, blind: dict) -> tuple[float, int]:
    ids = [i for i in full if i in blind]
    return 100.0 * sum(full[i] - blind[i] for i in ids) / len(ids), len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="bn id ru zh")
    ap.add_argument("--targets", default="jv mn ga si")
    ap.add_argument("--bench", default="cvqa")
    args = ap.parse_args()
    sources, targets, B = args.sources.split(), args.targets.split(), args.bench

    # Supervised reference per target: its own stage3_<L>_v4 arm.
    sup = {}
    for t in targets:
        f, b = load(f"{B}_{t}_v4"), load(f"{B}_{t}_BLIND_v4")
        if f and b:
            sup[t] = dv(f, b)[0]

    print(f"=== {B.upper()}: dV by source x target (supervised reference in the last row) ===\n")
    print(f"{'source':<9}" + "".join(f"{t:>10}" for t in targets))
    for s in sources:
        cells = []
        for t in targets:
            if t == s:
                cells.append("(src)")
                continue
            f, b = load(f"{B}_{t}_zs{s}"), load(f"{B}_{t}_zs{s}_BLIND")
            cells.append(f"{dv(f, b)[0]:+.2f}" if (f and b) else "--")
        print(f"{s:<9}" + "".join(f"{c:>10}" for c in cells))
    print(f"{'SUPERVISED':<9}" + "".join(
        f"{sup[t]:>+10.2f}" if t in sup else f"{'--':>10}" for t in targets))

    # Pooled paired test per source over the failing group — the readable one.
    for label, group in (("failing group " + "/".join(FAIL_GROUP), FAIL_GROUP),
                         ("control " + "/".join(CONTROL), CONTROL)):
        print(f"\n=== Pooled paired McNemar, {label}: image vs blind, per source ===")
        print(f"{'source':<9}{'n':>7}{'image-only':>12}{'blind-only':>12}{'dV':>8}{'p':>11}")
        for s in sources:
            a_tot = b_tot = n_tot = 0
            gain = 0.0
            for t in group:
                if t == s:
                    continue
                f, b = load(f"{B}_{t}_zs{s}"), load(f"{B}_{t}_zs{s}_BLIND")
                if not (f and b):
                    continue
                n01, n10, _, n = mcnemar(f, b)
                a_tot += n01
                b_tot += n10
                n_tot += n
                gain += sum(f[i] - b[i] for i in f if i in b)
            if n_tot == 0:
                print(f"{s:<9}{'-- not run --':>50}")
                continue
            n = a_tot + b_tot
            k = min(a_tot, b_tot)
            lp = lambda j: (math.lgamma(n + 1) - math.lgamma(j + 1)
                            - math.lgamma(n - j + 1) - n * math.log(2))
            p = min(1.0, 2 * sum(math.exp(lp(j)) for j in range(k + 1))) if n else 1.0
            print(f"{s:<9}{n_tot:>7}{a_tot:>12}{b_tot:>12}{100 * gain / n_tot:>8.2f}{p:>11.2g}")

    print("""
How to read it
  Any source lifting the failing group to a significant pooled p  -> H_source.
  The fix is source selection or multi-source stage 3, and the failure is a
  property of the PAIR, not of jv/mn/ga.

  Every source flat on the failing group while the control stays significant
  -> H_align. The failure is intrinsic to those languages' text bridges, and
  X2 (alignment scoring) plus X3 (D12 joint stage 1) are the way forward.

  The control going flat too means the harness broke, not the languages.
  Check the checkpoint paths before reading anything else here.""")


if __name__ == "__main__":
    main()
