"""Approach 2 evaluation on open-ended multilingual VQA benchmarks (e.g. xGQA).

Consumes the same eval JSONL produced by Stage3/load_vqa_eval_data.py
(fields: ``query``, ``answer``, ``nllb_lang_tag``, and either
``vg_image_id`` for local images or ``image_url`` for cached downloads) and
scores normalized exact match, mirroring Stage3/evaluate_vqa.py so numbers
are directly comparable between approaches.

Usage
-----
    python evaluate_vqa.py \\
        --data-path ../Stage3/data/stage3b_eval/xgqa/bn.jsonl \\
        --images-dir ../Stage3/data/gqa/images \\
        --ckpt ./outputs/stage3/mapping/pytorch_model.bin \\
        --mt-path facebook/nllb-200-distilled-600M \\
        --vis-path google/siglip2-so400m-patch14-384 \\
        --llm-path google/gemma-2-9b-it \\
        --output-path ./outputs/stage3/eval_xgqa_bn.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, NllbTokenizer

from common import (
    format_chat_prompt,
    llm_input_features,
    load_jsonl,
    load_mapping_checkpoint,
    mt_input_features,
    setup_logging,
)
from model import DualEncoderMerger


# Fallback for eval files that carry an ISO code in `source_language`
# instead of an `nllb_lang_tag` field (e.g. the shared xGQA testdev JSONL).
ISO_TO_NLLB = {
    "bn": "ben_Beng", "sw": "swh_Latn", "yo": "yor_Latn", "wo": "wol_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "en": "eng_Latn", "zh": "zho_Hans",
    "ko": "kor_Hang", "ru": "rus_Cyrl", "pt": "por_Latn", "id": "ind_Latn",
    "ga": "gle_Latn", "jv": "jav_Latn", "mn": "khk_Cyrl", "si": "sin_Sinh",
}


def row_nllb_tag(row: dict) -> str:
    """Return the row's NLLB tag, deriving it from `source_language` if absent."""
    tag = row.get("nllb_lang_tag")
    if tag:
        return tag
    lang = row.get("source_language", "")
    if lang in ISO_TO_NLLB:
        return ISO_TO_NLLB[lang]
    raise KeyError(f"Row has neither nllb_lang_tag nor a known source_language: {lang!r}")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (VQA-style)."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def open_ended_correct(pred_text: str, target: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(target)


def resolve_image(row: dict, images_dir: str | None, url_cache_dir: str | None) -> Image.Image | None:
    """Load a row's image from local id (``vg_image_id``) or URL cache."""
    image_id = row.get("vg_image_id")
    if image_id is not None and images_dir:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    return None
    url = row.get("image_url")
    if url and url_cache_dir:
        cache_path = os.path.join(url_cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg")
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach 2 open-ended VQA evaluation.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Eval JSONL from Stage3/load_vqa_eval_data.py.")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Local images dir for vg_image_id benchmarks (xGQA).")
    parser.add_argument("--image-cache-dir", type=str, default=None,
                        help="sha1(url).jpg cache dir for image_url benchmarks.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Mapping checkpoint (Stage 3, or Stage 1+2 combined).")
    parser.add_argument("--extra-ckpt", type=str, default=None,
                        help="Optional second checkpoint (e.g. --ckpt stage1 --extra-ckpt stage2 "
                             "to evaluate the zero-shot composition without Stage 3).")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Where to write per-example predictions JSONL.")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--vis-path", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--blind", action="store_true",
                        help="Replace every image with a neutral gray canvas — measures the "
                             "language-prior baseline (how much accuracy needs no vision at all).")
    parser.add_argument("--max-vis-tokens", type=int, default=0)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-gen-len", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N rows (0 = all).")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), "a2_eval_vqa"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_jsonl(args.data_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info("Loaded %d eval rows from %s", len(rows), args.data_path)

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
        max_gen_len=args.max_gen_len,
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
        for idx, row in enumerate(tqdm(rows, desc="eval")):
            if args.blind:
                image = Image.new("RGB", (384, 384), (128, 128, 128))
            else:
                image = resolve_image(row, args.images_dir, args.image_cache_dir)
            if image is None:
                n_skipped += 1
                continue

            pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
            input_ids_mt, mask_mt = mt_input_features(
                [row["query"]], [row_nllb_tag(row)], tokenizer_mt, args.max_mt_seq_len, device
            )
            prompt = format_chat_prompt(
                tokenizer_llm, row["query"], use_chat_template=not args.no_chat_template
            )
            input_ids_prompt, mask_prompt = llm_input_features(
                [prompt], tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
            )

            pred = model.generate(
                tokenizer_llm,
                input_ids_mt=input_ids_mt,
                attention_mask_mt=mask_mt,
                pixel_values=pixel_values,
                input_ids_prompt=input_ids_prompt,
                mask_prompt=mask_prompt,
            )[0].strip()

            ok = open_ended_correct(pred, row["answer"])
            n_correct += int(ok)
            n_scored += 1
            fout.write(json.dumps({
                "id": row.get("id", idx),
                "query": row["query"],
                "answer": row["answer"],
                "pred": pred,
                "correct": ok,
            }, ensure_ascii=False) + "\n")

            if args.log_every > 0 and idx % args.log_every == 0:
                logger.info(
                    "idx=%d acc=%.4f question=%r pred=%r target=%r ok=%s",
                    idx, n_correct / max(1, n_scored), row["query"][:80], pred, row["answer"], ok,
                )

    accuracy = n_correct / max(1, n_scored)
    logger.info(
        "DONE | scored=%d skipped=%d correct=%d accuracy=%.4f → %s",
        n_scored, n_skipped, n_correct, accuracy, args.output_path,
    )
    summary = {
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
