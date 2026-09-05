"""Approach 2 text-reasoning evaluation on MGSM / MSVAMP (Bengali et al.).

Measures whether the modular architecture preserves the frozen LLM's
reasoning ability — the counterpart to Approach 1's reported collapse on
these benchmarks after VQA specialization.

Three model variants:
    --no-mapping          Plain frozen Gemma (upper bound / sanity control:
                          the mappings cannot have damaged this by design).
    --ckpt stage1/...     Gemma + text mapping trained on translation only.
    --ckpt stage3/...     Gemma + text mapping after joint VQA training —
                          the row directly comparable to Approach 1's table.

Protocol mirrors Stage3/evaluate_text.py exactly (prompt, answer
extraction, float comparison) so numbers are comparable across approaches.

Input JSONL fields: ``question``, ``answer`` (as shared in evaluation/).

Usage
-----
    python evaluate_text.py \\
        --data-path ../evaluation/MGSM.jsonl --benchmark mgsm \\
        --ckpt outputs/stage3/mapping/pytorch_model.bin \\
        --output-path outputs/stage3/eval_mgsm_bn.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, NllbTokenizer

from common import (
    llm_input_features,
    load_jsonl,
    load_mapping_checkpoint,
    mt_input_features,
    setup_logging,
)
from model import DualEncoderMerger


# ─── Protocol functions — identical to Stage3/evaluate_text.py ──────────────

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


def extract_math_answer(text: str) -> str | None:
    m = re.search(r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def math_correct(pred_text: str, target: str) -> bool:
    pred = extract_math_answer(pred_text)
    if pred is None:
        return False
    try:
        return float(pred) == float(target.replace(",", ""))
    except ValueError:
        return pred.strip() == target.strip()


# ─── Prompt formatting (Gemma chat template, no system role) ────────────────

def format_math_chat(tokenizer_llm, question: str, use_chat_template: bool) -> str:
    user_prompt = build_math_prompt(question)
    if not use_chat_template or getattr(tokenizer_llm, "chat_template", None) is None:
        return user_prompt
    text = tokenizer_llm.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    bos = getattr(tokenizer_llm, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 2 MGSM/MSVAMP evaluation.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="JSONL with `question` and `answer` fields.")
    parser.add_argument("--benchmark", type=str, default="mgsm",
                        help="Label recorded in the summary (mgsm / msvamp).")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Mapping checkpoint (stage 1 or stage 3). Required unless --no-mapping.")
    parser.add_argument("--no-mapping", action="store_true",
                        help="Evaluate the plain frozen LLM without the NLLB branch.")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--nllb-tag", type=str, default="ben_Beng",
                        help="NLLB language tag for the questions.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--max-mt-seq-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-gen-len", type=int, default=512,
                        help="Room for chain-of-thought before the final answer.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if not args.no_mapping and not args.ckpt:
        parser.error("--ckpt is required unless --no-mapping is set.")

    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), "a2_eval_text"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_jsonl(args.data_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info("Loaded %d rows from %s (benchmark=%s, no_mapping=%s)",
                len(rows), args.data_path, args.benchmark, args.no_mapping)

    tokenizer_llm = AutoTokenizer.from_pretrained(
        args.llm_path, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    if args.no_mapping:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, torch_dtype=torch.bfloat16, local_files_only=args.local_files_only
        ).to(device).eval()
        tokenizer_mt = None
        model = None
    else:
        tokenizer_mt = NllbTokenizer.from_pretrained(
            args.mt_path, local_files_only=args.local_files_only)
        model = DualEncoderMerger(
            mt_path=args.mt_path,
            vis_path=None,
            llm_path=args.llm_path,
            max_gen_len=args.max_gen_len,
            llm_bos_token_id=tokenizer_llm.bos_token_id,
            llm_pad_token_id=tokenizer_llm.pad_token_id,
            use_text_branch=True,
            use_vision_branch=False,
            local_files_only=args.local_files_only,
        ).to(device)
        load_mapping_checkpoint(args.ckpt, model, logger)
        model.eval()
        llm = None

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    n_correct, n_scored = 0, 0

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(tqdm(rows, desc=f"eval-{args.benchmark}")):
            prompt = format_math_chat(
                tokenizer_llm, row["question"], use_chat_template=not args.no_chat_template
            )

            if args.no_mapping:
                tokenizer_llm.add_bos_token = True
                tokenizer_llm.add_eos_token = False
                enc = tokenizer_llm(prompt, return_tensors="pt", truncation=True,
                                    max_length=args.max_seq_len).to(device)
                with torch.inference_mode():
                    out_ids = llm.generate(
                        **enc, max_new_tokens=args.max_gen_len,
                        pad_token_id=tokenizer_llm.pad_token_id, do_sample=False,
                    )
                pred_text = tokenizer_llm.decode(
                    out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            else:
                input_ids_mt, mask_mt = mt_input_features(
                    [row["question"]], [args.nllb_tag], tokenizer_mt, args.max_mt_seq_len, device
                )
                input_ids_prompt, mask_prompt = llm_input_features(
                    [prompt], tokenizer_llm, args.max_seq_len,
                    add_bos=False, add_eos=False, device=device,
                )
                pred_text = model.generate(
                    tokenizer_llm,
                    input_ids_mt=input_ids_mt,
                    attention_mask_mt=mask_mt,
                    input_ids_prompt=input_ids_prompt,
                    mask_prompt=mask_prompt,
                )[0].strip()

            ok = math_correct(pred_text, str(row["answer"]))
            n_correct += int(ok)
            n_scored += 1
            fout.write(json.dumps({
                "id": idx,
                "question": row["question"],
                "answer": str(row["answer"]),
                "pred_text": pred_text,
                "pred_extracted": extract_math_answer(pred_text),
                "correct": ok,
            }, ensure_ascii=False) + "\n")

            if args.log_every > 0 and idx % args.log_every == 0:
                logger.info("idx=%d acc=%.4f extracted=%r target=%r ok=%s",
                            idx, n_correct / max(1, n_scored),
                            extract_math_answer(pred_text), row["answer"], ok)

    accuracy = n_correct / max(1, n_scored)
    summary = {
        "benchmark": args.benchmark,
        "scored": n_scored, "correct": n_correct, "accuracy": accuracy,
        "data_path": args.data_path,
        "ckpt": args.ckpt, "no_mapping": args.no_mapping,
        "llm_path": args.llm_path, "mt_path": args.mt_path,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(args.output_path + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("DONE | %s", json.dumps(summary))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
