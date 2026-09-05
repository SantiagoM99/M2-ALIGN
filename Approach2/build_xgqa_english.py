"""Build the English arm of xGQA, paired item-for-item with our language files.

xGQA is GQA test-dev translated into 7 languages, and our per-language files
keep GQA's question `id`. So the English questions join on that id exactly —
no ordering assumption, no image-level ambiguity, and the resulting file
covers the SAME items as the target-language files. That pairing is what
makes an "English → target drop" comparable to the 38 points reported by
Pfeiffer et al. (Findings of ACL 2022).

The gold answer is taken from OUR file, not from GQA, so every language is
scored against identical labels; disagreements with GQA's own answer are
counted and reported rather than silently resolved.

    python3 Approach2/build_xgqa_english.py \\
      --reference $DT/Stage3/data/xgqa/bn.jsonl \\
      --questions gqa_testdev_questions.jsonl \\
      --output    $DT/Stage3/data/xgqa/en.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="English xGQA arm, paired by question id.")
    ap.add_argument("--reference", required=True,
                    help="Any per-language xGQA file — defines the item set.")
    ap.add_argument("--questions", required=True,
                    help="gqa_testdev_questions.jsonl from fetch_gqa_english.py.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ref = load_jsonl(args.reference)
    qs = {r["question_id"]: r for r in load_jsonl(args.questions)}
    print(f"reference items: {len(ref)} | English questions available: {len(qs)}")

    rows, missing, gold_mismatch = [], 0, 0
    for r in ref:
        qid = str(r.get("id", "")).strip()
        src = qs.get(qid)
        if not src or not src.get("question"):
            missing += 1
            continue
        if src.get("answer") and src["answer"].strip().lower() != str(r["answer"]).strip().lower():
            gold_mismatch += 1
        rows.append({
            "id": qid,
            "vg_image_id": r["vg_image_id"],
            "query": src["question"],
            "answer": r["answer"],          # our gold, so every language is scored alike
            "source_language": "en",
            "nllb_lang_tag": "eng_Latn",    # read first by evaluate_vqa.row_nllb_tag
            "source_dataset": "xgqa_testdev",
        })

    cov = len(rows) / len(ref) if ref else 0
    print(f"matched {len(rows)}/{len(ref)} ({cov:.1%}) | missing {missing} | "
          f"gold disagreements with GQA {gold_mismatch}")
    if cov < 0.95:
        print("WARNING: coverage below 95%. The English arm would not be paired with "
              "the target-language files, and a drop computed against it mixes an "
              "item-set difference into the language difference.")

    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(f"wrote {len(rows)} -> {args.output}")
    if rows:
        print("  sample:", json.dumps(rows[0], ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
