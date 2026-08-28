"""Generation-health diagnostics for the text benchmarks (MGSM / MSVAMP).

An accuracy delta between two runs can come from better reasoning or merely
from fewer broken generations (empty outputs, repetition loops, answers the
extractor cannot parse). This separates the two: it reports the degeneration
rates side by side and re-scores accuracy over the clean subset only.

Usage:
    python3 Approach2/analysis/text_gen_health.py <run.jsonl> [<run.jsonl> ...]

Rows are those written by evaluate_text.py:
    {"id", "question", "answer", "pred_text", "pred_extracted", "correct"}
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter


def looped(text: str, min_repeats: int = 3) -> bool:
    """True if some non-trivial line is emitted min_repeats+ times."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 10]
    if not lines:
        return False
    return Counter(lines).most_common(1)[0][1] >= min_repeats


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    lens = [len(r.get("pred_text") or "") for r in rows]
    empty = [r for r, L in zip(rows, lens) if L < 5]
    noext = [r for r in rows if r.get("pred_extracted") in (None, "")]
    loops = [r for r in rows if looped(r.get("pred_text") or "")]
    bad_ids = {id(r) for r in empty} | {id(r) for r in noext} | {id(r) for r in loops}
    clean = [r for r in rows if id(r) not in bad_ids]
    acc = sum(bool(r.get("correct")) for r in rows) / n
    acc_clean = (sum(bool(r.get("correct")) for r in clean) / len(clean)) if clean else float("nan")
    acc_loop = (sum(bool(r.get("correct")) for r in loops) / len(loops)) if loops else float("nan")

    print(f"{path.split('/')[-1]}")
    print(f"  n {n} | acc {acc:6.3f} | median chars {st.median(lens):6.1f}")
    print(f"  empty {len(empty):4d} ({len(empty)/n:5.1%}) | "
          f"no-extract {len(noext):4d} ({len(noext)/n:5.1%}) | "
          f"loops {len(loops):4d} ({len(loops)/n:5.1%})")
    print(f"  clean subset: n {len(clean)} acc {acc_clean:6.3f} | "
          f"acc within loops {acc_loop:6.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
        print()
