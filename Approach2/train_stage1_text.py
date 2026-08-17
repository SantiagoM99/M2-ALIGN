"""Approach 2, Stage 1: train the NLLB → Gemma text mapping on translation pairs.

Reads the same ``{Language}_to_English.jsonl`` files produced by
Stage1/load_text.py (fields: ``source``, ``target``) and trains only
``mapping_txt`` to reconstruct the English target from the mapped NLLB
encoding of the source:

    ``[BOS] + X_f + [b_txt]  →  target (English)``

The vision branch is disabled; NLLB encoder and Gemma 2 are frozen.

Usage
-----
    python train_stage1_text.py \\
        --data-dir ../Stage1/data \\
        --languages Bengali,French \\
        --output-dir ./outputs/stage1 \\
        --mt-path facebook/nllb-200-distilled-600M \\
        --llm-path google/gemma-2-9b-it
"""
from __future__ import annotations

import argparse
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from common import (
    init_wandb_or_disable,
    llm_input_features,
    load_jsonl,
    load_mapping_checkpoint,
    load_training_state,
    mt_input_features,
    save_mapping_checkpoint,
    save_training_state,
    set_seed,
    setup_logging,
)
from model import DualEncoderMerger

try:
    import wandb
except ImportError:
    wandb = None

# Human-readable language name → NLLB tag, matching Stage1/train.py.
LANGS_MAP_NLLB = {
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
    "Bengali": "ben_Beng",
    "German": "deu_Latn",
    "Russian": "rus_Cyrl",
    "Mandarin": "zho_Hans",
    "Chinese": "zho_Hans",
    "Korean": "kor_Hang",
    "Portuguese": "por_Latn",
    "Indonesian": "ind_Latn",
    "Irish": "gle_Latn",
    "Javanese": "jav_Latn",
    "Mongolian": "khk_Cyrl",
    "Sinhala": "sin_Sinh",
}


class TranslationDataset(Dataset):
    """Translation pairs with per-row NLLB language tags."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def read_translation_pairs(data_dir: str, languages: list[str], train_num: int) -> list[dict]:
    """Load ``{Language}_to_English.jsonl`` files into tagged rows."""
    rows: list[dict] = []
    for language in languages:
        path = os.path.join(data_dir, f"{language}_to_English.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"NLLB file not found: {path}")
        if language not in LANGS_MAP_NLLB:
            raise KeyError(f"No NLLB tag for language {language}; extend LANGS_MAP_NLLB.")
        for pair in load_jsonl(path)[:train_num]:
            rows.append({
                "source": pair["source"],
                "target": pair["target"],
                "nllb_lang_tag": LANGS_MAP_NLLB[language],
            })
    random.shuffle(rows)
    return rows


def collate_pairs(batch: list[dict]) -> dict:
    return {
        "sources": [x["source"] for x in batch],
        "targets": [x["target"] for x in batch],
        "nllb_lang_tags": [x["nllb_lang_tag"] for x in batch],
    }


def run_validation(model, val_loader, tokenizer_mt, tokenizer_llm, args, device) -> float:
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids_mt, mask_mt = mt_input_features(
                batch["sources"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device
            )
            labels, mask_label = llm_input_features(
                batch["targets"], tokenizer_llm, args.max_gen_len,
                add_bos=False, add_eos=True, device=device,
            )
            loss = model(
                labels=labels, mask_label=mask_label,
                input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
            )
            val_loss += loss.item()
            val_steps += 1
    return val_loss / max(1, val_steps)


def main(args, logger) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_stage1_text.py requires CUDA.")

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    rows = read_translation_pairs(args.data_dir, languages, args.train_num)
    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d languages=%s", len(train_rows), len(val_rows), languages)

    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

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

    if args.init_checkpoint:
        load_mapping_checkpoint(args.init_checkpoint, model, logger)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    train_loader = DataLoader(
        TranslationDataset(train_rows), batch_size=args.train_batch_size,
        shuffle=True, collate_fn=collate_pairs, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TranslationDataset(val_rows), batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=collate_pairs, num_workers=args.num_workers,
    )

    use_wandb = init_wandb_or_disable(args, {
        "stage": "approach2_stage1_text",
        "mt_path": args.mt_path,
        "llm_path": args.llm_path,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "grad_accum": args.grad_accum,
        "languages": languages,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    })

    best_val = float("inf")
    global_step = 0
    start_epoch = 0
    resume_skip_batches = 0
    if args.resume_from_checkpoint:
        start_epoch, resume_skip_batches, global_step, best_val = load_training_state(
            args.resume_from_checkpoint, model, optimizer, device
        )
        logger.info(
            "Resumed from %s: epoch=%d step_in_epoch=%d global_step=%d best_val=%.4f",
            args.resume_from_checkpoint, start_epoch, resume_skip_batches, global_step, best_val,
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)
        skip_batches = resume_skip_batches if epoch == start_epoch else 0

        for batch_idx, batch in enumerate(pbar):
            if batch_idx < skip_batches:
                continue

            input_ids_mt, mask_mt = mt_input_features(
                batch["sources"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device
            )
            labels, mask_label = llm_input_features(
                batch["targets"], tokenizer_llm, args.max_gen_len,
                add_bos=False, add_eos=True, device=device,
            )

            loss = model(
                labels=labels, mask_label=mask_label,
                input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
            )
            (loss / args.grad_accum).backward()
            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{running / steps:.4f}")
            if use_wandb:
                wandb.log({"train/loss": running / steps, "train/global_step": global_step})

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                state_path = os.path.join(args.output_dir, "training_state.pt")
                save_training_state(
                    state_path, model, optimizer, epoch, batch_idx + 1, global_step, best_val
                )
                logger.info("Saved training state (global_step=%d) → %s", global_step, state_path)

        val_loss = run_validation(model, val_loader, tokenizer_mt, tokenizer_llm, args, device)
        val_ppl = math.exp(min(20.0, val_loss))
        logger.info("Epoch %d | val_loss=%.4f | val_ppl=%.4f", epoch, val_loss, val_ppl)
        if use_wandb:
            wandb.log({"eval/loss": val_loss, "eval/ppl": val_ppl, "train/global_step": global_step})

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.output_dir, "mapping", "pytorch_model.bin")
            save_mapping_checkpoint(ckpt_path, model, global_step, val_loss)
            logger.info("Saved best checkpoint (val_loss=%.4f) → %s", val_loss, ckpt_path)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approach 2 Stage 1: NLLB → Gemma text mapping.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with {Language}_to_English.jsonl files (Stage1/load_text.py).")
    parser.add_argument("--languages", type=str, default="Bengali",
                        help="Comma-separated human-readable language names.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Optional mapping checkpoint to warm-start mapping_txt from "
                             "(Approach 2 or legacy Approach 1 format).")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train-num", type=int, default=100000,
                        help="Max pairs per language.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-mt-seq-len", type=int, default=512)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="m2align-approach2")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
                           "a2_stage1_text")
    main(args, logger)
