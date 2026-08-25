"""Approach 2, Stage 3: joint VQA training with both branches active.

Reads the same VQA JSONL produced by Stage3/load_vqa_data.py (fields:
``vg_image_id``, ``query``, ``answer``, ``source_language``,
``nllb_lang_tag``; images at ``{images_dir}/{vg_image_id}.jpg``) and trains
both mappings on:

    ``[BOS] + X_f + [b_txt] + V_f + [b_vis] + T  →  answer``

where ``X_f`` is the mapped NLLB encoding of the source-language question,
``V_f`` the mapped SigLIP encoding of the image, and ``T`` the LLM token
embedding of the task-formatted question (the MindMerger "collaboration"
term). Warm-starts ``mapping_txt`` from Stage 1 and ``mapping_vis`` from
Stage 2; either can be frozen for ablations.

Usage
-----
    python train_stage3_vqa.py \\
        --data-path ../Stage3/data/stage3b/bengali.jsonl \\
        --images-dir ../Stage3/data/gqa/images \\
        --output-dir ./outputs/stage3 \\
        --stage1-ckpt ./outputs/stage1/mapping/pytorch_model.bin \\
        --stage2-ckpt ./outputs/stage2/mapping/pytorch_model.bin \\
        --mt-path facebook/nllb-200-distilled-600M \\
        --vis-path google/siglip2-so400m-patch14-384 \\
        --llm-path google/gemma-2-9b-it
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import random
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, NllbTokenizer

from common import (
    format_chat_prompt,
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


class VQADataset(Dataset):
    """VQA rows with images stored as ``{images_dir}/{vg_image_id}.jpg``."""

    def __init__(self, rows: list[dict], images_dir: str) -> None:
        self.rows = rows
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(row["vg_image_id"])
        if image is None:
            return None
        return {
            "image": image,
            "query": row["query"],
            "answer": row["answer"],
            "nllb_lang_tag": row["nllb_lang_tag"],
        }

    def _load_image(self, image_id) -> Image.Image | None:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    return None
        return None


def collate_vqa(batch: list[dict | None], image_processor) -> dict | None:
    valid = [x for x in batch if x is not None]
    if not valid:
        return None
    pixel_values = image_processor(
        images=[x["image"] for x in valid], return_tensors="pt"
    )["pixel_values"]
    return {
        "pixel_values": pixel_values,
        "queries": [x["query"] for x in valid],
        "answers": [x["answer"] for x in valid],
        "nllb_lang_tags": [x["nllb_lang_tag"] for x in valid],
    }


class TextReplayDataset(Dataset):
    """Text-only replay rows mixed into VQA training (no image).

    Accepts stage-1 translation rows (``source``/``target``) and math-CoT
    rows from build_math_replay.py (``query``/``target``/``task=math``).
    Purpose: the mappings keep seeing non-VQA territory during stage 3, so
    the prefix can't specialize into corrupting everything that isn't a
    short-answer VQA question (the measured MGSM/MSVAMP collapse).
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {
            "query": r.get("query") or r.get("source"),
            "target": r["target"],
            "nllb_lang_tag": r["nllb_lang_tag"],
            "task": r.get("task", "translation"),
        }


def collate_replay(batch: list[dict]) -> dict:
    return {
        "queries": [x["query"] for x in batch],
        "targets": [x["target"] for x in batch],
        "nllb_lang_tags": [x["nllb_lang_tag"] for x in batch],
        "tasks": [x["task"] for x in batch],
    }


def build_replay_inputs(batch, tokenizer_mt, tokenizer_llm, args, device):
    """Tokenise a text-only replay batch. Math rows get the exact eval-time
    math prompt (evaluate_text.format_math_chat); translation rows get no
    prompt, matching stage 1's format."""
    from evaluate_text import format_math_chat

    input_ids_mt, mask_mt = mt_input_features(
        batch["queries"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device
    )
    prompts = [
        format_math_chat(tokenizer_llm, q, use_chat_template=not args.no_chat_template)
        if task == "math" else ""
        for q, task in zip(batch["queries"], batch["tasks"])
    ]
    input_ids_prompt, mask_prompt = llm_input_features(
        prompts, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
    )
    labels, mask_label = llm_input_features(
        batch["targets"], tokenizer_llm, args.replay_max_gen_len,
        add_bos=False, add_eos=True, device=device,
    )
    return {
        "input_ids_mt": input_ids_mt,
        "attention_mask_mt": mask_mt,
        "input_ids_prompt": input_ids_prompt,
        "mask_prompt": mask_prompt,
        "labels": labels,
        "mask_label": mask_label,
    }


def build_batch_inputs(batch, tokenizer_mt, tokenizer_llm, args, device):
    """Tokenise one collated batch into model inputs."""
    input_ids_mt, mask_mt = mt_input_features(
        batch["queries"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device
    )
    prompts = [
        format_chat_prompt(tokenizer_llm, q, use_chat_template=not args.no_chat_template)
        for q in batch["queries"]
    ]
    input_ids_prompt, mask_prompt = llm_input_features(
        prompts, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
    )
    labels, mask_label = llm_input_features(
        batch["answers"], tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=True, device=device
    )
    return {
        "input_ids_mt": input_ids_mt,
        "attention_mask_mt": mask_mt,
        "pixel_values": batch["pixel_values"].to(device),
        "input_ids_prompt": input_ids_prompt,
        "mask_prompt": mask_prompt,
        "labels": labels,
        "mask_label": mask_label,
    }


def run_validation(model, val_loader, tokenizer_mt, tokenizer_llm, args, device) -> float:
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            inputs = build_batch_inputs(batch, tokenizer_mt, tokenizer_llm, args, device)
            val_loss += model(**inputs).item()
            val_steps += 1
    return val_loss / max(1, val_steps)


def main(args, logger) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_stage3_vqa.py requires CUDA.")

    rows = load_jsonl(args.data_path)
    random.shuffle(rows)
    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d", len(train_rows), len(val_rows))

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

    if args.stage1_ckpt:
        load_mapping_checkpoint(args.stage1_ckpt, model, logger)
    if args.stage2_ckpt:
        load_mapping_checkpoint(args.stage2_ckpt, model, logger)

    if args.freeze_text_mapping:
        for p in model.mapping_txt.parameters():
            p.requires_grad = False
        logger.info("mapping_txt frozen.")
    if args.freeze_vision_mapping:
        for p in model.mapping_vis.parameters():
            p.requires_grad = False
        logger.info("mapping_vis frozen.")

    if args.zero_init_gate:
        with torch.no_grad():
            model.mapping_txt.gate.zero_()
            model.mapping_vis.gate.zero_()
        logger.info("Mapping gates zero-initialized — the prefix starts inert.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    _collate = partial(collate_vqa, image_processor=image_processor)
    train_loader = DataLoader(
        VQADataset(train_rows, args.images_dir), batch_size=args.train_batch_size,
        shuffle=True, collate_fn=_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        VQADataset(val_rows, args.images_dir), batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=_collate, num_workers=args.num_workers,
    )

    replay_iter = None
    if args.replay_data:
        replay_rows: list[dict] = []
        for path in [p.strip() for p in args.replay_data.split(",") if p.strip()]:
            file_rows = load_jsonl(path)
            if 0 < args.replay_max_rows_per_file < len(file_rows):
                file_rows = random.sample(file_rows, args.replay_max_rows_per_file)
            for r in file_rows:
                r.setdefault("nllb_lang_tag", args.replay_default_tag)
            replay_rows.extend(file_rows)
            logger.info("Replay: %d rows from %s", len(file_rows), path)
        replay_loader = DataLoader(
            TextReplayDataset(replay_rows),
            batch_size=args.replay_batch_size or args.train_batch_size,
            shuffle=True, collate_fn=collate_replay, num_workers=2,
        )
        replay_iter = itertools.cycle(replay_loader)
        logger.info("Replay enabled: %d rows total, 1 replay batch per %d VQA batches.",
                    len(replay_rows), args.replay_every)

    use_wandb = init_wandb_or_disable(args, {
        "stage": "approach2_stage3_vqa",
        "mt_path": args.mt_path,
        "vis_path": args.vis_path,
        "llm_path": args.llm_path,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "grad_accum": args.grad_accum,
        "max_vis_tokens": args.max_vis_tokens,
        "freeze_text_mapping": args.freeze_text_mapping,
        "freeze_vision_mapping": args.freeze_vision_mapping,
        "replay_data": args.replay_data,
        "replay_every": args.replay_every,
        "zero_init_gate": args.zero_init_gate,
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
        replay_running, replay_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)
        skip_batches = resume_skip_batches if epoch == start_epoch else 0

        for batch_idx, batch in enumerate(pbar):
            if batch_idx < skip_batches:
                continue
            if batch is None:
                continue

            inputs = build_batch_inputs(batch, tokenizer_mt, tokenizer_llm, args, device)
            loss = model(**inputs)
            (loss / args.grad_accum).backward()

            if replay_iter is not None and args.replay_every > 0 and steps % args.replay_every == 0:
                rbatch = next(replay_iter)
                rinputs = build_replay_inputs(rbatch, tokenizer_mt, tokenizer_llm, args, device)
                rloss = model(**rinputs)
                (rloss / args.grad_accum).backward()
                replay_running += rloss.item()
                replay_steps += 1

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            steps += 1
            global_step += 1
            postfix = {"loss": f"{running / steps:.4f}"}
            if replay_steps:
                postfix["replay"] = f"{replay_running / replay_steps:.4f}"
            pbar.set_postfix(**postfix)
            if use_wandb:
                log = {"train/loss": running / steps, "train/global_step": global_step}
                if replay_steps:
                    log["train/replay_loss"] = replay_running / replay_steps
                wandb.log(log)

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
    parser = argparse.ArgumentParser(description="Approach 2 Stage 3: joint VQA training.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="VQA JSONL from Stage3/load_vqa_data.py.")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory with {vg_image_id}.jpg images.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Stage 1 text-mapping checkpoint to warm-start from.")
    parser.add_argument("--stage2-ckpt", type=str, default=None,
                        help="Stage 2 vision-mapping checkpoint to warm-start from.")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--vis-path", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Use the bare task prompt for T (for non-instruction-tuned LLMs).")
    parser.add_argument("--replay-data", type=str, default=None,
                        help="Comma-separated text-only JSONL files (stage-1 translation "
                             "rows and/or build_math_replay.py math rows) mixed into "
                             "training so the prefix keeps covering non-VQA territory.")
    parser.add_argument("--replay-every", type=int, default=3,
                        help="One replay batch per N VQA batches (with --replay-data).")
    parser.add_argument("--replay-batch-size", type=int, default=0,
                        help="0 = same as --train-batch-size.")
    parser.add_argument("--replay-max-gen-len", type=int, default=512,
                        help="Label budget for replay targets (CoT solutions are long).")
    parser.add_argument("--replay-max-rows-per-file", type=int, default=10000,
                        help="Random subsample per replay file, so a 100k translation "
                             "file doesn't drown the math data (0 = no cap).")
    parser.add_argument("--replay-default-tag", type=str, default="ben_Beng",
                        help="NLLB tag for replay rows that lack one (stage-1 files).")
    parser.add_argument("--zero-init-gate", action="store_true",
                        help="Reset both mapping gates to 0 after warm-start: the prefix "
                             "starts inert and must earn its influence during training.")
    parser.add_argument("--freeze-text-mapping", action="store_true")
    parser.add_argument("--freeze-vision-mapping", action="store_true")
    parser.add_argument("--max-vis-tokens", type=int, default=0,
                        help="τ_k: keep only the first k visual tokens (0 = keep all).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Max token length for the LLM-side prompt T.")
    parser.add_argument("--max-gen-len", type=int, default=64,
                        help="Max token length for the (short) VQA answer.")
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
                           "a2_stage3_vqa")
    main(args, logger)
