"""Approach 2 evaluation on CVQA (multiple-choice, scored open-ended via likelihood).

Mirrors Stage3/evaluate.py's `evaluate_cvqa_open_ended` protocol on the
`parallel` branch: the model is prompted with the question only (no visible
options), and each answer choice is scored by its length-normalized
log-likelihood as a continuation of that prompt — the choice with the
highest mean per-token log-probability wins, and its index is compared
against the gold ``answer_index``. `DualEncoderMerger.forward` already
returns exactly that quantity (mean CE over the label tokens, batch of 1),
so each choice costs one forward pass (~4 per question).

Input JSONL from Stage3/load_evaluation_data.py --benchmark cvqa:
    ``id``, ``query`` (native-language question), ``choices`` (English),
    ``answer_index``, ``source_language``.
Images live at ``<images_dir>/<id>.jpg``.

Usage
-----
    python evaluate_cvqa.py \\
        --data-path ../Stage3/data/cvqa/mn.jsonl \\
        --images-dir ../Stage3/data/cvqa/images \\
        --ckpt ./outputs/stage3/mapping/pytorch_model.bin \\
        --output-path ./outputs/stage3/eval_cvqa_mn.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, NllbTokenizer

from common import (
    _VQA_SYSTEM,
    llm_input_features,
    load_jsonl,
    load_mapping_checkpoint,
    mt_input_features,
    setup_logging,
)
from evaluate_vqa import row_nllb_tag
from model import DualEncoderMerger


def build_cvqa_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_cvqa_open_ended_prompt`:
    question only, no visible options."""
    return f"Question: {question}"


def format_cvqa_chat(tokenizer_llm, question: str, use_chat_template: bool) -> str:
    """Chat-wrap the CVQA prompt the same way common.format_chat_prompt does
    for open-ended VQA (system folded into the user turn, leading BOS
    stripped) — only the task prompt differs."""
    user_prompt = build_cvqa_open_ended_prompt(question)
    if not use_chat_template or getattr(tokenizer_llm, "chat_template", None) is None:
        return user_prompt
    messages = [{"role": "user", "content": f"{_VQA_SYSTEM}\n\n{user_prompt}"}]
    text = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    bos = getattr(tokenizer_llm, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 2 CVQA likelihood evaluation.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="CVQA JSONL from Stage3/load_evaluation_data.py --benchmark cvqa.")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory with <id>.jpg per row.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Mapping checkpoint (Stage 3, or Stage 1+2 combined).")
    parser.add_argument("--extra-ckpt", type=str, default=None,
                        help="Optional second checkpoint (e.g. --ckpt stage1 --extra-ckpt stage2).")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--vis-path", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--blind", action="store_true",
                        help="Replace every image with a neutral gray canvas — the "
                             "language-prior baseline, as in evaluate_vqa.py.")
    parser.add_argument("--max-vis-tokens", type=int, default=0)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-choice-len", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), "a2_eval_cvqa"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_jsonl(args.data_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info("Loaded %d CVQA rows from %s (blind=%s)", len(rows), args.data_path, args.blind)

    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    image_processor = AutoImageProcessor.from_pretrained(args.vis_path)

    model = DualEncoderMerger(
        mt_path=args.mt_path,
        vis_path=args.vis_path,
        llm_path=args.llm_path,
        max_gen_len=args.max_choice_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_text_branch=True,
        use_vision_branch=True,
        max_vis_tokens=args.max_vis_tokens,
        local_files_only=args.local_files_only,
    ).to(device)
    load_mapping_checkpoint(args.ckpt, model, logger)
    if args.extra_ckpt:
        load_mapping_checkpoint(args.extra_ckpt, model, logger)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    n_correct, n_scored, n_skipped = 0, 0, 0

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(tqdm(rows, desc="eval-cvqa")):
            if args.blind:
                image = Image.new("RGB", (384, 384), (128, 128, 128))
            else:
                image_path = os.path.join(args.images_dir, f"{row['id']}.jpg")
                if not os.path.exists(image_path):
                    n_skipped += 1
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    n_skipped += 1
                    continue

            pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
            input_ids_mt, mask_mt = mt_input_features(
                [row["query"]], [row_nllb_tag(row)], tokenizer_mt, args.max_mt_seq_len, device
            )
            prompt = format_cvqa_chat(
                tokenizer_llm, row["query"], use_chat_template=not args.no_chat_template
            )
            input_ids_prompt, mask_prompt = llm_input_features(
                [prompt], tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
            )

            scores: list[float] = []
            for choice in row["choices"]:
                if not str(choice).strip():
                    scores.append(float("-inf"))
                    continue
                labels, mask_label = llm_input_features(
                    [str(choice)], tokenizer_llm, args.max_choice_len,
                    add_bos=False, add_eos=False, device=device,
                )
                with torch.inference_mode():
                    loss = model(
                        labels=labels, mask_label=mask_label,
                        input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
                        pixel_values=pixel_values,
                        input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
                    )
                scores.append(-loss.item())

            pred_index = max(range(len(scores)), key=scores.__getitem__)
            ok = pred_index == int(row["answer_index"])
            n_correct += int(ok)
            n_scored += 1
            fout.write(json.dumps({
                "id": row["id"],
                "query": row["query"],
                "choices": row["choices"],
                "answer_index": int(row["answer_index"]),
                "pred_index": pred_index,
                "scores": scores,
                "correct": ok,
            }, ensure_ascii=False) + "\n")

            if args.log_every > 0 and idx % args.log_every == 0:
                logger.info(
                    "idx=%d acc=%.4f pred=%r gold=%r ok=%s",
                    idx, n_correct / max(1, n_scored),
                    row["choices"][pred_index], row["choices"][int(row["answer_index"])], ok,
                )

    accuracy = n_correct / max(1, n_scored)
    logger.info(
        "DONE | scored=%d skipped=%d correct=%d accuracy=%.4f → %s",
        n_scored, n_skipped, n_correct, accuracy, args.output_path,
    )
    summary = {
        "benchmark": "cvqa",
        "scored": n_scored, "skipped": n_skipped,
        "correct": n_correct, "accuracy": accuracy,
        "data_path": args.data_path, "ckpt": args.ckpt,
        "llm_path": args.llm_path, "vis_path": args.vis_path, "mt_path": args.mt_path,
        "blind": args.blind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(args.output_path + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
