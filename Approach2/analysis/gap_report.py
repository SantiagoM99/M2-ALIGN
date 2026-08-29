"""Gap report: the metric MindMerger is actually judged on.

Average accuracy hides the thing this project is for. MindMerger's claim is
that injecting a multilingual encoder lets weak languages borrow from strong
ones, so success is measured as *gap closure*, not as a mean. This reports
three gaps between two rounds:

  1. resource gap   low-resource minus higher-resource, on full accuracy and
                    on dV (full - blind, the part that is genuinely visual)
  2. cultural gap   CVQA minus xGQA for the same languages. xGQA is GQA's
                    Western-sourced images with translated questions; CVQA is
                    locally-sourced images and culture-specific questions, so
                    the drop between them is what cultural grounding costs.
  3. paired test    pooled McNemar over the low-resource group, because at
                    n=286 per language nothing else is readable (DESIGN.md
                    D10 puts the per-language CVQA noise floor at +-2.8)

Usage (from Approach2/results):
    python3 ../analysis/gap_report.py <base_tag> <new_tag>
    python3 ../analysis/gap_report.py "" v3        # round B vs v3
    python3 ../analysis/gap_report.py v3 v4        # control vs LLaVA arm

An empty tag means the untagged round. Missing files are reported, never
silently averaged over.
"""
from __future__ import annotations

import json
import math
import os
import sys

LRL = ["bn", "jv", "mn", "si", "ga"]
HRL = ["ru", "zh", "pt", "id", "ko"]
XGQA_LANGS = ["bn", "de", "ru", "zh", "pt", "id", "ko"]
CVQA_LANGS = LRL + HRL


def suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def acc(bench: str, lang: str, tag: str, blind: bool = False) -> float | None:
    p = f"eval_{bench}_{lang}{'_BLIND' if blind else ''}{suffix(tag)}.jsonl.summary.json"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)["accuracy"] * 100
    except Exception:
        return None


def per_item(bench: str, lang: str, tag: str) -> dict | None:
    p = f"eval_{bench}_{lang}{suffix(tag)}.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get("id", r.get("question_id"))
            if k is not None:
                out[k] = bool(r.get("correct"))
    return out


def group(bench: str, langs: list[str], tag: str):
    """(mean full, mean dV, langs used, langs missing)"""
    full, dv, used, missing = [], [], [], []
    for L in langs:
        a, b = acc(bench, L, tag), acc(bench, L, tag, blind=True)
        if a is None or b is None:
            missing.append(L)
            continue
        full.append(a); dv.append(a - b); used.append(L)
    if not used:
        return None, None, used, missing
    return sum(full) / len(full), sum(dv) / len(dv), used, missing


def mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return float("nan")
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    return math.erfc(z / math.sqrt(2))


def pooled_test(bench: str, langs: list[str], base: str, new: str) -> str:
    b = c = n = 0
    seen = 0
    for L in langs:
        x, y = per_item(bench, L, base), per_item(bench, L, new)
        if x is None or y is None:
            continue
        seen += 1
        ks = set(x) & set(y)
        n += len(ks)
        b += sum(1 for k in ks if x[k] and not y[k])
        c += sum(1 for k in ks if y[k] and not x[k])
    if seen == 0:
        return "  (no per-item files — harvest eval_*.jsonl to enable this)"
    p = mcnemar(b, c)
    arrow = "new better" if c > b else "base better"
    return (f"  pooled over {seen} langs: n={n}  base-only={b}  new-only={c}  "
            f"p={p:.2e}  ({arrow})")


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else ""
    new = sys.argv[2] if len(sys.argv) > 2 else "v3"
    bl, nl = base or "(untagged)", new or "(untagged)"
    print(f"base = {bl}    new = {nl}\n")

    print("=== 1. RESOURCE GAP (CVQA — the only benchmark covering all 5 LRLs) ===")
    print(f"{'group':18} {'full base':>10} {'full new':>9} {'dV base':>8} {'dV new':>7}")
    res = {}
    for label, langs in (("low-resource", LRL), ("higher-resource", HRL)):
        fb, vb, used, miss = group("cvqa", langs, base)
        fn, vn, used2, miss2 = group("cvqa", langs, new)
        res[label] = (fb, vb, fn, vn)
        if fb is None or fn is None:
            print(f"{label:18} incomplete (missing {miss or miss2})")
            continue
        print(f"{label:18} {fb:10.2f} {fn:9.2f} {vb:8.2f} {vn:7.2f}"
              + (f"   [missing {miss+miss2}]" if miss or miss2 else ""))
    l, h = res.get("low-resource"), res.get("higher-resource")
    if l and h and None not in l and None not in h:
        print(f"\n  gap LRL-HRL, full : {l[0]-h[0]:+.2f}  ->  {l[2]-h[2]:+.2f}"
              f"   ({(l[2]-h[2])-(l[0]-h[0]):+.2f})")
        print(f"  gap LRL-HRL, dV   : {l[1]-h[1]:+.2f}  ->  {l[3]-h[3]:+.2f}"
              f"   ({(l[3]-h[3])-(l[1]-h[1]):+.2f})")
        print("  (negative = low-resource behind; closing the gap means moving toward 0)")

    print("\n=== 2. CULTURAL GAP (CVQA - xGQA, languages present in both) ===")
    shared = [L for L in CVQA_LANGS if L in XGQA_LANGS]
    print(f"{'lang':6} {'xGQA base':>10} {'CVQA base':>10} {'gap':>7} | "
          f"{'xGQA new':>9} {'CVQA new':>9} {'gap':>7}")
    gb, gn = [], []
    for L in shared:
        xb, cb = acc("xgqa", L, base), acc("cvqa", L, base)
        xn, cn = acc("xgqa", L, new), acc("cvqa", L, new)
        if None in (xb, cb, xn, cn):
            print(f"{L:6} incomplete")
            continue
        gb.append(cb - xb); gn.append(cn - xn)
        print(f"{L:6} {xb:10.2f} {cb:10.2f} {cb-xb:+7.2f} | "
              f"{xn:9.2f} {cn:9.2f} {cn-xn:+7.2f}")
    if gb and gn:
        mb, mn = sum(gb)/len(gb), sum(gn)/len(gn)
        print(f"{'avg':6} {'':10} {'':10} {mb:+7.2f} | {'':9} {'':9} {mn:+7.2f}"
              f"    ({mn-mb:+.2f})")

    print("\n=== 3. POOLED PAIRED TEST, low-resource group ===")
    print("CVQA:"); print(pooled_test("cvqa", LRL, base, new))
    print("xGQA (bn only among LRLs):"); print(pooled_test("xgqa", ["bn"], base, new))

    print("\n=== 4. REASONING (bn is the only low-resource language covered) ===")
    print(f"{'lang':6} {'MGSM base':>10} {'MGSM new':>9} | {'MSVAMP base':>12} {'MSVAMP new':>11}")
    for L in ["bn", "de", "ru", "zh"]:
        def t(bench, tag):
            p = f"eval_{bench}_{L}_stage3{suffix(tag)}.jsonl.summary.json"
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)["accuracy"] * 100
            except Exception:
                return None
        mb, mn2 = t("mgsm", base), t("mgsm", new)
        sb, sn = t("msvamp", base), t("msvamp", new)
        fmt = lambda v, w: (f"{v:{w}.1f}" if v is not None else " " * (w - 3) + "n/a")
        print(f"{L:6} {fmt(mb,10)} {fmt(mn2,9)} | {fmt(sb,12)} {fmt(sn,11)}")


if __name__ == "__main__":
    main()
