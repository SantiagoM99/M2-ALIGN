"""Build math-CoT replay data for Stage 3: GSM8K or MetaMathQA → NLLB-translated questions.

Each output row pairs a *target-language* math question (translated from
GSM8K with the same frozen NLLB used by the pipeline) with the original
English chain-of-thought solution:

    {"query": "<question in target lang>", "target": "<English CoT ... The answer is N.>",
     "nllb_lang_tag": "<tag>", "task": "math"}

Mixed into Stage 3 via train_stage3_vqa.py --replay-data, this keeps the
text mapping covering math-question territory during VQA training — the
measured cause of the MGSM/MSVAMP collapse is precisely that the prefix
never saw such inputs (MindMerger's stage-2 augmentation / IJCNLP-2025
IFL's text-only injection, applied to this pipeline).

GSM8K solutions are cleaned: calculator annotations ``<<...>>`` removed and
the final ``#### N`` rewritten as ``The answer is N.`` so training targets
match evaluate_text.py's answer-extraction patterns.

The dataset must already be in the HF cache (compute nodes are offline):
    HF_HOME=$SCRATCH/huggingface python -c \\
        "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"

Usage
-----
    python build_math_replay.py --nllb-tag ben_Beng \\
        --output ../Stage1/data/math_replay_bn.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

import torch
from tqdm import tqdm


def clean_solution(answer: str) -> str:
    """Strip calculator annotations; rewrite '#### N' as 'The answer is N.'"""
    text = re.sub(r"<<[^>]*>>", "", answer)
    m = re.search(r"####\s*(.+)\s*$", text)
    if m:
        final = m.group(1).strip()
        text = text[: m.start()].rstrip()
        text = f"{text}\nThe answer is {final}."
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K → NLLB math replay data.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nllb-tag", type=str, default="ben_Beng",
                        help="Target language for the questions (FLORES-200 tag).")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--n", type=int, default=0,
                        help="Max problems (0 = all of the pool).")
    parser.add_argument("--source", choices=["gsm8k", "metamath"], default="gsm8k",
                        help="Replay pool. 'metamath' = meta-math/MetaMathQA, the "
                             "source MindMerger's scale, MERLIN and Approach 1 all "
                             "use (30k/language). NOT a generalization argument: "
                             "60.8%% of its rows are GSM_* rephrasings of GSM8K "
                             "train, the same split MGSM's test items come from. "
                             "Adopt it for volume and phrasing diversity only.")
    parser.add_argument("--metamath-types", type=str, default="GSM_",
                        help="Comma-separated prefixes of MetaMathQA's `type` to "
                             "keep. Default GSM_ (grade-school word problems). "
                             "MATH_ exists and shares no source with either "
                             "evaluation, but 7 of 8 sampled MATH_ rows carry "
                             "LaTeX, which NLLB destroys — see DESIGN.md.")
    parser.add_argument("--seed", type=int, default=13,
                        help="Sampling seed; fixed so larger --n is a superset.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--no-translate", action="store_true",
                        help="Keep questions in English (tag eng_Latn) — ablation arm.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    if args.source == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        rows = [{"question": r["question"], "answer": clean_solution(r["answer"])} for r in ds]
        label = "GSM8K train"
    else:
        keep = tuple(t.strip() for t in args.metamath_types.split(",") if t.strip())
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        rows = [
            # `response` already ends in "The answer is: N", which
            # extract_math_answer parses (its regex accepts "is:").
            {"question": r["query"].strip(), "answer": r["response"].strip()}
            for r in ds
            if str(r.get("type", "")).startswith(keep)
            and r.get("query", "").strip() and r.get("response", "").strip()
        ]
        label = f"MetaMathQA[{'|'.join(keep)}]"
    if args.n > 0 and args.n < len(rows):
        rows = random.Random(args.seed).sample(rows, args.n)
    print(f"{label}: {len(rows)} problems")

    if args.no_translate:
        out_rows = [
            {"id": i, "query": r["question"], "target": r["answer"],
             "nllb_lang_tag": "eng_Latn", "task": "math"}
            for i, r in enumerate(rows)
        ]
    else:
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = NllbTokenizer.from_pretrained(args.mt_path)
        tokenizer.src_lang = "eng_Latn"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.mt_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        forced_bos = tokenizer.convert_tokens_to_ids(args.nllb_tag)

        translated: list[str] = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(rows), args.batch_size), desc=f"translate→{args.nllb_tag}"):
                batch = [r["question"] for r in rows[i: i + args.batch_size]]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_src_len).to(device)
                gen = model.generate(**enc, forced_bos_token_id=forced_bos,
                                     max_new_tokens=args.max_src_len, num_beams=1)
                translated.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

        out_rows = [
            {"id": i, "query": q, "target": r["answer"],
             "nllb_lang_tag": args.nllb_tag, "task": "math"}
            for i, (r, q) in enumerate(zip(rows, translated))
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
