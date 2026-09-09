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
import math
import os
import random
import signal
import sys
from pathlib import Path
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
    mt_input_features,
    save_mapping_checkpoint,
    set_seed,
    setup_logging,
)
from model import DualEncoderMerger
from s1_contract import atomic_json, digest, file_record, git_state, read_json
from training_resume import (
    epoch_batches,
    split_rows,
    save_snapshot,
    restore_snapshot,
    completed,
    mark_completed,
)

try:
    import wandb
except ImportError:
    wandb = None


class VQADataset(Dataset):
    """VQA rows with images stored as ``{images_dir}/{vg_image_id}.jpg``."""

    def __init__(
        self, rows: list[dict], images_dir: str, load_images: bool = True
    ) -> None:
        self.rows = rows
        self.images_dir = images_dir
        # --no-vision keeps the same rows and the same task, but there is no
        # vision branch to feed, so decoding ~34k JPEGs per epoch is pure waste.
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = None
        if self.load_images:
            image = self._load_image(row["vg_image_id"])
            if image is None:
                raise ValueError(
                    f"Missing or corrupt training image {row['vg_image_id']}"
                )
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
    pixel_values = None
    if image_processor is not None and valid[0]["image"] is not None:
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

    input_ids_mt = mask_mt = None
    if not args.no_text_branch:
        input_ids_mt, mask_mt = mt_input_features(
            batch["queries"],
            batch["nllb_lang_tags"],
            tokenizer_mt,
            args.max_mt_seq_len,
            device,
        )
    prompts = [
        format_math_chat(tokenizer_llm, q, use_chat_template=not args.no_chat_template)
        if task == "math"
        else ""
        for q, task in zip(batch["queries"], batch["tasks"])
    ]
    input_ids_prompt, mask_prompt = llm_input_features(
        prompts,
        tokenizer_llm,
        args.max_seq_len,
        add_bos=False,
        add_eos=False,
        device=device,
    )
    labels, mask_label = llm_input_features(
        batch["targets"],
        tokenizer_llm,
        args.replay_max_gen_len,
        add_bos=False,
        add_eos=True,
        device=device,
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
    input_ids_mt = mask_mt = None
    if not args.no_text_branch:
        input_ids_mt, mask_mt = mt_input_features(
            batch["queries"],
            batch["nllb_lang_tags"],
            tokenizer_mt,
            args.max_mt_seq_len,
            device,
        )
    prompts = [
        format_chat_prompt(
            tokenizer_llm, q, use_chat_template=not args.no_chat_template
        )
        for q in batch["queries"]
    ]
    input_ids_prompt, mask_prompt = llm_input_features(
        prompts,
        tokenizer_llm,
        args.max_seq_len,
        add_bos=False,
        add_eos=False,
        device=device,
    )
    labels, mask_label = llm_input_features(
        batch["answers"],
        tokenizer_llm,
        args.max_gen_len,
        add_bos=False,
        add_eos=True,
        device=device,
    )
    inputs = {
        "input_ids_mt": input_ids_mt,
        "attention_mask_mt": mask_mt,
        "input_ids_prompt": input_ids_prompt,
        "mask_prompt": mask_prompt,
        "labels": labels,
        "mask_label": mask_label,
    }
    # Omitted entirely under --no-vision: the model raises if it is handed
    # pixel_values with no vision branch to consume them.
    if batch.get("pixel_values") is not None:
        inputs["pixel_values"] = batch["pixel_values"].to(device)
    return inputs


def run_validation(
    model, val_loader, tokenizer_mt, tokenizer_llm, args, device
) -> float:
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            inputs = build_batch_inputs(
                batch, tokenizer_mt, tokenizer_llm, args, device
            )
            val_loss += model(**inputs).item()
            val_steps += 1
    return val_loss / max(1, val_steps)


def training_configuration(args):
    import transformers

    ignored = {
        "resume_from_checkpoint",
        "check_complete",
        "use_wandb",
        "wandb_mode",
        "wandb_project",
        "wandb_run_name",
        "save_steps",
    }
    arguments = {k: v for k, v in vars(args).items() if k not in ignored}
    files = {
        k: file_record(getattr(args, k))
        for k in ("data_path", "stage1_ckpt", "stage2_ckpt")
        if getattr(args, k)
    }
    if args.replay_data:
        files["replay"] = [
            file_record(p.strip()) for p in args.replay_data.split(",") if p.strip()
        ]
    # This implementation fingerprint allows harmless docs/launcher changes
    # while refusing resume under changed optimization or model code.
    code = {
        name: file_record(Path(__file__).parent / name)["sha256"]
        for name in (
            "train_stage3_vqa.py",
            "training_resume.py",
            "model.py",
            "common.py",
        )
    }
    return {
        "arguments": arguments,
        "files": files,
        "implementation": code,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
    }


def training_device():
    if not torch.cuda.is_available():
        raise RuntimeError("train_stage3_vqa.py requires CUDA")
    return torch.device("cuda")


def main(args, logger) -> None:
    if args.grad_accum <= 0 or args.replay_every <= 0:
        raise ValueError("gradient accumulation and replay interval must be positive")
    if args.no_text_branch and args.replay_data:
        raise ValueError("text replay requires a trainable text branch")
    if args.replay_data and args.freeze_text_mapping:
        raise ValueError("replay has no trainable text path")
    if args.zero_init_gate and (args.freeze_text_mapping or args.freeze_vision_mapping):
        raise ValueError("cannot change a frozen gate")
    if args.s1 and (
        args.epochs != 2
        or args.vis_layers != "9,18,-1"
        or args.replay_data
        or args.no_vision
    ):
        raise ValueError("S1 C requires two epochs, dense layers, vision and no replay")
    config = training_configuration(args)
    if args.check_complete:
        sys.exit(0 if completed(args.output_dir, config) else 3)
    if completed(args.output_dir, config):
        logger.info("Complete, configuration and artifact hashes verified.")
        return
    state_path = Path(args.output_dir) / "training_state.pt"
    if not args.resume_from_checkpoint and state_path.exists():
        args.resume_from_checkpoint = str(state_path)
    if (
        Path(args.output_dir) / "mapping/pytorch_model.bin"
    ).exists() and not args.resume_from_checkpoint:
        raise ValueError(
            "best checkpoint exists without resumable state/completion marker; use a new output directory"
        )
    device = training_device()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(args.seed)
    rows = load_jsonl(args.data_path)
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed, by_image=args.s1)
    split_hash = digest({"train": train_rows, "validation": val_rows})
    config["split_sha256"] = split_hash
    # Keep completion checks independent of data loading, while still including
    # the deterministic split in the run manifest and each snapshot.
    run_path = Path(args.output_dir) / "training_run.json"
    provenance = git_state(require_clean=args.s1)
    run = {
        "configuration": config,
        **provenance,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "order_hashes": {
            str(e): digest(
                epoch_batches(len(train_rows), args.train_batch_size, args.seed, e)
            )
            for e in range(args.epochs)
        },
    }
    if run_path.exists():
        old = read_json(run_path)
        if old["configuration"] != config:
            raise ValueError("training run configuration changed")
    else:
        atomic_json(run_path, run)
    tokenizer_mt = (
        NllbTokenizer.from_pretrained(
            args.mt_path, local_files_only=args.local_files_only
        )
        if not args.no_text_branch
        else None
    )
    tokenizer_llm = AutoTokenizer.from_pretrained(
        args.llm_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    processor = (
        AutoImageProcessor.from_pretrained(
            args.vis_path, local_files_only=args.local_files_only
        )
        if not args.no_vision
        else None
    )
    model = DualEncoderMerger(
        args.mt_path,
        args.vis_path,
        args.llm_path,
        args.max_gen_len,
        tokenizer_llm.bos_token_id,
        tokenizer_llm.pad_token_id,
        use_text_branch=not args.no_text_branch,
        use_vision_branch=not args.no_vision,
        max_vis_tokens=args.max_vis_tokens,
        vis_layers=args.vis_layers,
        local_files_only=args.local_files_only,
    ).to(device)
    from common import load_branch_checkpoint

    if args.stage1_ckpt and model.mapping_txt is not None:
        load_branch_checkpoint(args.stage1_ckpt, model.mapping_txt, "mapping_txt")
    if args.stage2_ckpt and model.mapping_vis is not None:
        load_branch_checkpoint(args.stage2_ckpt, model.mapping_vis, "mapping_vis")
    for name, freeze in [
        ("mapping_txt", args.freeze_text_mapping),
        ("mapping_vis", args.freeze_vision_mapping),
    ]:
        module = getattr(model, name)
        if module is None:
            if freeze:
                raise ValueError(f"cannot freeze absent {name}")
            continue
        if args.zero_init_gate:
            with torch.no_grad():
                module.gate.zero_()
        for param in module.parameters():
            param.requires_grad = not freeze
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("both mappings frozen: evaluate C4 without training")
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    dataset = VQADataset(train_rows, args.images_dir, load_images=not args.no_vision)
    collator = partial(collate_vqa, image_processor=processor)
    val_loader = DataLoader(
        VQADataset(val_rows, args.images_dir, load_images=not args.no_vision),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed + 100),
    )
    replay_rows = []
    if args.replay_data:
        rng = random.Random(args.seed)
        for path in [x.strip() for x in args.replay_data.split(",") if x.strip()]:
            rr = load_jsonl(path)
            if 0 < args.replay_max_rows_per_file < len(rr):
                rr = rng.sample(rr, args.replay_max_rows_per_file)
            for r in rr:
                r.setdefault("nllb_lang_tag", args.replay_default_tag)
            replay_rows.extend(rr)
    replay_dataset = TextReplayDataset(replay_rows) if replay_rows else None
    progress = {
        "epoch": 0,
        "next_batch": 0,
        "global_step": 0,
        "optimizer_steps": 0,
        "best_val": None,
        "replay_cursor": 0,
    }
    if args.resume_from_checkpoint:
        progress = restore_snapshot(
            args.resume_from_checkpoint, model, optimizer, config
        )
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    use_wandb = init_wandb_or_disable(args, config)
    last_saved = progress["global_step"]
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(progress["epoch"], args.epochs):
        model.train()
        if args.s1:
            for tower in (model.model_llm, model.model_mt, model.encoder_vis):
                if tower is not None:
                    tower.eval()
        schedule = epoch_batches(
            len(train_rows), args.train_batch_size, args.seed, epoch
        )
        order_hash = digest(schedule)
        start = progress["next_batch"] if epoch == progress["epoch"] else 0
        if start and progress.get("order_sha256") != order_hash:
            raise ValueError("resumed sampler order differs")
        # Slicing the original batch schedule avoids generating a new random
        # permutation and throwing away its prefix on resume. Worker RNG is
        # independent of the model's dropout RNG; preprocessing is deterministic.
        loader = DataLoader(
            dataset,
            batch_sampler=schedule[start:],
            collate_fn=collator,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed + epoch + 1000),
        )
        accumulated = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"epoch={epoch}"), start):
            if batch is None:
                raise ValueError("empty training batch")
            loss = model(
                **build_batch_inputs(batch, tokenizer_mt, tokenizer_llm, args, device)
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss / args.grad_accum).backward()
            if replay_dataset is not None and batch_idx % args.replay_every == 0:
                replay_bs = args.replay_batch_size or args.train_batch_size
                per_cycle = (len(replay_dataset) + replay_bs - 1) // replay_bs
                cycle, offset = divmod(progress["replay_cursor"], per_cycle)
                indices = epoch_batches(
                    len(replay_dataset), replay_bs, args.seed, cycle, "replay"
                )[offset]
                rb = collate_replay([replay_dataset[i] for i in indices])
                rloss = model(
                    **build_replay_inputs(rb, tokenizer_mt, tokenizer_llm, args, device)
                )
                if not torch.isfinite(rloss):
                    raise ValueError("non-finite replay loss")
                (rloss / args.grad_accum).backward()
                progress["replay_cursor"] += 1
            accumulated += 1
            progress.update(
                epoch=epoch,
                next_batch=batch_idx + 1,
                global_step=progress["global_step"] + 1,
                order_sha256=order_hash,
            )
            boundary = accumulated == args.grad_accum or batch_idx + 1 == len(schedule)
            if boundary:
                if accumulated != args.grad_accum:
                    for param in params:
                        if param.grad is not None:
                            param.grad.mul_(args.grad_accum / accumulated)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                progress["optimizer_steps"] += 1
                if stop["requested"] or (
                    args.save_steps > 0
                    and progress["global_step"] - last_saved >= args.save_steps
                ):
                    save_snapshot(state_path, model, optimizer, progress, config)
                    last_saved = progress["global_step"]
                if stop["requested"]:
                    logger.info(
                        "Saved exact optimizer-boundary state; resume required."
                    )
                    sys.exit(75)
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/global_step": progress["global_step"],
                    }
                )
        val_loss = run_validation(
            model, val_loader, tokenizer_mt, tokenizer_llm, args, device
        )
        if not math.isfinite(val_loss):
            raise ValueError("non-finite validation loss")
        logger.info(
            "Epoch %d | val_loss=%.4f | val_ppl=%.4f",
            epoch,
            val_loss,
            math.exp(min(20, val_loss)),
        )
        if progress["best_val"] is None or val_loss < progress["best_val"]:
            progress["best_val"] = val_loss
            best = Path(args.output_dir) / "mapping/pytorch_model.bin"
            save_mapping_checkpoint(
                str(best) + ".tmp", model, progress["global_step"], val_loss
            )
            os.replace(str(best) + ".tmp", best)
        progress.update(epoch=epoch + 1, next_batch=0)
        save_snapshot(state_path, model, optimizer, progress, config)
    # Completion uses the external configuration, whose data hash determines
    # the split. The full split hash remains in snapshots and training_run.json.
    completion_config = {k: v for k, v in config.items() if k != "split_sha256"}
    mark_completed(args.output_dir, completion_config, args.epochs)
    if use_wandb:
        wandb.finish()


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Approach 2 Stage 3: joint VQA training."
    )
    parser.add_argument(
        "--s1",
        action="store_true",
        help="Corrected Block C recipe: frozen towers eval, split by image.",
    )
    parser.add_argument(
        "--no-text-branch", action="store_true", help="C5: no NLLB or text mapping."
    )
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="Exit 0 if verified complete, 3 if incomplete; no model load.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="VQA JSONL from Stage3/load_vqa_data.py.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory with {vg_image_id}.jpg images.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--stage1-ckpt",
        type=str,
        default=None,
        help="Stage 1 text-mapping checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable the vision branch: train the SAME joint stage "
        "(same VQA data, same replay, same epochs) with no "
        "visual prefix at all. This is the control for "
        "'what does adding the vision bridge cost the text "
        "bridge?' — without it, D11's law is compatible with "
        "both 'vision costs and better vision costs less' and "
        "'vision helps and better vision helps more'. Training "
        "is blind: the VQA answers must come from the question "
        "alone. See DESIGN.md H1.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        type=str,
        default=None,
        help="Stage 2 vision-mapping checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--mt-path", type=str, default="facebook/nllb-200-distilled-600M"
    )
    parser.add_argument(
        "--vis-path", type=str, default="google/siglip2-so400m-patch14-384"
    )
    parser.add_argument("--llm-path", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use the bare task prompt for T (for non-instruction-tuned LLMs).",
    )
    parser.add_argument(
        "--replay-data",
        type=str,
        default=None,
        help="Comma-separated text-only JSONL files (stage-1 translation "
        "rows and/or build_math_replay.py math rows) mixed into "
        "training so the prefix keeps covering non-VQA territory.",
    )
    parser.add_argument(
        "--replay-every",
        type=int,
        default=3,
        help="One replay batch per N VQA batches (with --replay-data).",
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=0,
        help="0 = same as --train-batch-size.",
    )
    parser.add_argument(
        "--replay-max-gen-len",
        type=int,
        default=512,
        help="Label budget for replay targets (CoT solutions are long).",
    )
    parser.add_argument(
        "--replay-max-rows-per-file",
        type=int,
        default=10000,
        help="Random subsample per replay file, so a 100k translation "
        "file doesn't drown the math data (0 = no cap).",
    )
    parser.add_argument(
        "--replay-default-tag",
        type=str,
        default="ben_Beng",
        help="NLLB tag for replay rows that lack one (stage-1 files).",
    )
    parser.add_argument(
        "--zero-init-gate",
        action="store_true",
        help="Reset both mapping gates to 0 after warm-start: the prefix "
        "starts inert and must earn its influence during training.",
    )
    parser.add_argument("--freeze-text-mapping", action="store_true")
    parser.add_argument("--freeze-vision-mapping", action="store_true")
    parser.add_argument(
        "--max-vis-tokens",
        type=int,
        default=0,
        help="τ_k: keep only the first k visual tokens (0 = keep all).",
    )
    parser.add_argument(
        "--vis-layers",
        type=str,
        default="",
        help="DenseConnector: comma-separated SigLIP hidden-state indices "
        "channel-concatenated per patch (e.g. '9,18,-1'; -1 = "
        "post-layernorm final layer). Empty = final layer only. "
        "MUST match the --stage2-ckpt's setting.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max token length for the LLM-side prompt T.",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=64,
        help="Max token length for the (short) VQA answer.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="m2align-approach2")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="auto")
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
        "a2_stage3_vqa",
    )
    main(args, logger)
