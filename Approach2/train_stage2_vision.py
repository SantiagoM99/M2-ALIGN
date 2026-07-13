"""Approach 2, Stage 2: train the SigLIP → Gemma vision mapping on WIT captions.

Reads the same ``wit_pairs.jsonl`` + image cache produced by
Stage2/load_image.py and trains only ``mapping_vis`` to caption the image in
English:

    ``[BOS] + V_f + [b_vis]  →  target_caption (English)``

This is the LLaVA-style alignment stage, but through a *frozen* LLM: the
mapping MLP alone must carry SigLIP features into Gemma's embedding space.
The text branch is disabled — unlike Approach 1's Stage 2, the multilingual
caption is NOT part of the input, so the vision mapping cannot lean on the
text side and is forced to ground the image itself.

Usage
-----
    python train_stage2_vision.py \\
        --data-path ../Stage2/data/wit_pairs.jsonl \\
        --image-cache-dir ../Stage2/data/image_cache \\
        --output-dir ./outputs/stage2 \\
        --vis-path google/siglip2-so400m-patch14-384 \\
        --llm-path google/gemma-2-9b-it
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from common import (
    init_wandb_or_disable,
    llm_input_features,
    load_jsonl,
    load_training_state,
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


class WITImageDataset(Dataset):
    """WIT JSONL pairs with pre-downloaded images keyed by ``sha1(url).jpg``.

    Same cache layout as Stage2/train.py, so both approaches share one
    image cache.
    """

    def __init__(self, rows: list[dict], cache_dir: str) -> None:
        self.rows = rows
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(row["image_url"])
        if image is None:
            return None
        return {"image": image, "target_caption": row["target_caption"]}

    def _load_image(self, url: str) -> Image.Image | None:
        cache_key = hashlib.sha1(url.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, cache_key + ".jpg")
        if not os.path.exists(cache_path):
            return None
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            return None


def collate_wit(batch: list[dict | None], image_processor) -> dict | None:
    """Collate a batch, processing images with the SigLIP image processor.

    SigLIP resizes every image to a fixed resolution, so ``pixel_values`` is
    a dense ``[B, 3, H, W]`` tensor with a constant visual token count — no
    grid bookkeeping or visual padding needed.
    """
    valid = [x for x in batch if x is not None]
    if not valid:
        return None
    pixel_values = image_processor(
        images=[x["image"] for x in valid], return_tensors="pt"
    )["pixel_values"]
    return {
        "pixel_values": pixel_values,
        "target_captions": [x["target_caption"] for x in valid],
    }


def run_validation(model, val_loader, tokenizer_llm, args, device) -> float:
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            labels, mask_label = llm_input_features(
                batch["target_captions"], tokenizer_llm, args.max_gen_len,
                add_bos=False, add_eos=True, device=device,
            )
            loss = model(
                labels=labels, mask_label=mask_label,
                pixel_values=batch["pixel_values"].to(device),
            )
            val_loss += loss.item()
            val_steps += 1
    return val_loss / max(1, val_steps)


def main(args, logger) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_stage2_vision.py requires CUDA.")

    rows = load_jsonl(args.data_path)
    random.shuffle(rows)
    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d", len(train_rows), len(val_rows))

    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    image_processor = AutoImageProcessor.from_pretrained(args.vis_path)

    model = DualEncoderMerger(
        mt_path=None,
        vis_path=args.vis_path,
        llm_path=args.llm_path,
        max_gen_len=args.max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_text_branch=False,
        use_vision_branch=True,
        max_vis_tokens=args.max_vis_tokens,
        local_files_only=args.local_files_only,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    _collate = partial(collate_wit, image_processor=image_processor)
    train_loader = DataLoader(
        WITImageDataset(train_rows, args.image_cache_dir),
        batch_size=args.train_batch_size, shuffle=True,
        collate_fn=_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        WITImageDataset(val_rows, args.image_cache_dir),
        batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers,
    )

    use_wandb = init_wandb_or_disable(args, {
        "stage": "approach2_stage2_vision",
        "vis_path": args.vis_path,
        "llm_path": args.llm_path,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "grad_accum": args.grad_accum,
        "max_vis_tokens": args.max_vis_tokens,
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
            if batch is None:
                continue

            labels, mask_label = llm_input_features(
                batch["target_captions"], tokenizer_llm, args.max_gen_len,
                add_bos=False, add_eos=True, device=device,
            )
            loss = model(
                labels=labels, mask_label=mask_label,
                pixel_values=batch["pixel_values"].to(device),
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

        val_loss = run_validation(model, val_loader, tokenizer_llm, args, device)
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
    parser = argparse.ArgumentParser(description="Approach 2 Stage 2: SigLIP → Gemma vision mapping.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to wit_pairs.jsonl from Stage2/load_image.py.")
    parser.add_argument("--image-cache-dir", type=str, required=True,
                        help="Image cache directory populated by Stage2/load_image.py.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vis-path", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-vis-tokens", type=int, default=0,
                        help="τ_k: keep only the first k visual tokens (0 = keep all; "
                             "siglip2-so400m-patch14-384 yields 729).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.05)
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
                           "a2_stage2_vision")
    main(args, logger)
