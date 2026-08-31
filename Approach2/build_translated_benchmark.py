"""NLLB-translate a text benchmark into languages the original does not cover.

MGSM and MSVAMP intersect our 11 languages in only bn/de/ru/zh — one
low-resource language. But stage-3 checkpoints exist for all eleven, so the
four remaining LRLs (jv, mn, si, ga) are missing nothing but test data, and
both benchmarks have NUMERIC gold answers: translating the question cannot
corrupt the label.

This is a machine-translated evaluation set and must be reported as one. Its
defence is the metric: DESIGN.md D10/D11 score each language as a percentage
of its own frozen-LLM ceiling, and the ceiling is measured on these same
translated items, so translation artifacts sit in both numerator and
denominator and largely cancel. It is the same NLLB model already used for
the training-side replay (build_math_replay.py), so no new dependency and no
new methodological compromise.

Source files come from fetch_text_benchmarks.py --langs en.

    python build_translated_benchmark.py \\
      --input ../evaluation/MGSM_en.jsonl --nllb-tag jav_Latn \\
      --output ../evaluation/MGSM_jv.jsonl

Run it on a GPU node (or a login node for a quick one): ~250-1000 short
sentences per benchmark, minutes.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate a benchmark with NLLB.")
    ap.add_argument("--input", required=True, help="JSONL with question/answer (English).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--nllb-tag", required=True, help="Target tag, e.g. jav_Latn.")
    ap.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--src-tag", default="eng_Latn")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output):
        print(f"{args.output} exists — skipping")
        return

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} rows from {args.input} → {args.nllb_tag}")

    mt_path = args.mt_path
    if os.path.isdir(mt_path):
        subs = [d for d in os.listdir(mt_path) if os.path.isdir(os.path.join(mt_path, d))]
        if subs:
            mt_path = os.path.join(mt_path, subs[0])

    tok = AutoTokenizer.from_pretrained(mt_path, local_files_only=args.local_files_only)
    tok.src_lang = args.src_tag
    model = AutoModelForSeq2SeqLM.from_pretrained(
        mt_path, local_files_only=args.local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # transformers moved this between versions; try both.
    try:
        forced_bos = tok.convert_tokens_to_ids(args.nllb_tag)
        if forced_bos is None or forced_bos == tok.unk_token_id:
            raise KeyError
    except Exception:
        forced_bos = tok.lang_code_to_id[args.nllb_tag]

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size), desc=f"→{args.nllb_tag}"):
            batch = rows[i: i + args.batch_size]
            enc = tok([r["question"] for r in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=args.max_len).to(device)
            gen = model.generate(**enc, forced_bos_token_id=forced_bos,
                                 max_length=args.max_len, num_beams=args.num_beams)
            for r, q in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                # The gold is numeric: carried over untouched, so MT cannot
                # corrupt the label, only the question.
                out.append({"question": q.strip(), "answer": str(r["answer"]).strip(),
                            "question_en": r["question"]})

    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(f"wrote {len(out)} → {args.output}")
    print(f"  sample: {json.dumps(out[0], ensure_ascii=False)[:220]}")


if __name__ == "__main__":
    main()
