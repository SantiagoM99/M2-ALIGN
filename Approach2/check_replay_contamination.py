"""Is any MGSM test item present in the replay training pool?

MGSM's 250 problems are GSM8K *test*; the replay pool must be train-only.
GSM8K train is safe by construction (disjoint splits). MetaMathQA is not
safe by construction: it is built by *rephrasing* GSM8K and MATH training
problems, so if a test item ever leaked into its source, exact matching
would miss it — the leak would be a paraphrase.

Two detectors, cheap and complementary:
  - numeric signature: the multiset of numbers in a problem survives
    rephrasing almost perfectly, so a shared signature is a strong flag
  - token Jaccard over content words, on the candidates that signature
    matching turns up

Neither is proof of absence; both are enough to catch a real leak.

    python check_replay_contamination.py --pool metamath --sample 100000
    python check_replay_contamination.py --pool gsm8k-train      # sanity check
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

STOP = {"the","a","an","of","in","on","at","to","for","and","or","is","are",
        "was","were","how","many","much","does","do","did","if","then","she",
        "he","they","it","her","his","their","each","has","have","had","with"}


def numbers(text: str) -> tuple:
    return tuple(sorted(re.findall(r"\d+(?:\.\d+)?", text)))


def content(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay-pool contamination check.")
    ap.add_argument("--mgsm-en", default="evaluation/MGSM_en.jsonl",
                    help="English MGSM (fetch_text_benchmarks.py --langs en).")
    ap.add_argument("--pool", default="metamath",
                    choices=["metamath", "gsm8k-train", "gsm8k-test"])
    ap.add_argument("--sample", type=int, default=0, help="0 = whole pool.")
    ap.add_argument("--min-numbers", type=int, default=2,
                    help="Ignore signatures with fewer numbers (too common).")
    ap.add_argument("--jaccard", type=float, default=0.5)
    args = ap.parse_args()

    from datasets import load_dataset

    test = [json.loads(l)["question"] for l in open(args.mgsm_en, encoding="utf-8")]
    print(f"MGSM English test items: {len(test)}")

    if args.pool == "metamath":
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        pool = [r["query"] for r in ds]
    else:
        split = "train" if args.pool == "gsm8k-train" else "test"
        ds = load_dataset("openai/gsm8k", "main", split=split)
        pool = [r["question"] for r in ds]
    if args.sample and args.sample < len(pool):
        import random
        pool = random.Random(0).sample(pool, args.sample)
    print(f"pool = {args.pool}: {len(pool)} items\n")

    index = defaultdict(list)
    for q in pool:
        sig = numbers(q)
        if len(sig) >= args.min_numbers:
            index[sig].append(q)

    sig_hits = 0
    confirmed = []
    for t in test:
        sig = numbers(t)
        if len(sig) < args.min_numbers:
            continue
        cands = index.get(sig, [])
        if not cands:
            continue
        sig_hits += 1
        ct = content(t)
        best = max(cands, key=lambda c: jaccard(ct, content(c)))
        j = jaccard(ct, content(best))
        if j >= args.jaccard:
            confirmed.append((j, t, best))

    print(f"numeric-signature collisions: {sig_hits}/{len(test)}")
    print(f"confirmed overlaps (jaccard >= {args.jaccard}): {len(confirmed)}\n")
    for j, t, c in sorted(confirmed, reverse=True)[:5]:
        print(f"--- jaccard {j:.2f}")
        print(f"  MGSM test : {t[:180]}")
        print(f"  pool item : {c[:180]}\n")
    if not confirmed:
        print("No overlap detected. A signature collision without a token match is "
              "two different problems that happen to use the same numbers.")


if __name__ == "__main__":
    main()
