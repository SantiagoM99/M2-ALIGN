"""Stage: align the SigLIP2 expert to Qwen3-VL's embedding space on captions.

This is the only training the merged architecture needs that neither approach can
donate: the text mapping warm-starts from Maryam's Stage 2 checkpoint, Qwen's own
tower is already aligned, and what does not exist anywhere is a SigLIP2 dense
mapping that speaks Qwen. It is trained the way `train_stage2_vision.py` trains
ours for Gemma — an English caption conditioned on the image — with one change
that decides whether it learns anything at all.

**The native pathway gets a gray canvas.** If Qwen sees the real image through its
own tower it can caption from that alone and the new mapping receives no gradient.
Removing the image is not an option in early mode, where the injection needs the
native visual positions to exist, so a gray canvas supplies positions without
content and the new stream has to carry the image. The same canvas is used in both
modes, so the two runs differ in exactly one thing.

The processor is pinned to one resolution (`--pixels`), because early injection
resamples onto Qwen's merged grid and a batch of differently sized images would
need a different resample per example.

    python Approach2/merged/align_vision.py --mode prefix \\
      --data-path $LL/llava_pretrain.jsonl --image-cache-dir $LL/image_cache \\
      --output-dir Approach2/outputs/merged_align_prefix --local-files-only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge import MergedVLM  # noqa: E402
from model import DenseVisionExpert, SIGLIP_LAYERS  # noqa: E402

CAPTION_PROMPT = "Describe this image in one sentence."
GRAY = (128, 128, 128)
IGNORE = -100


def fail(message: str):
    raise SystemExit(f"align_vision: {message}")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        fail(f"{path} is empty")
    return rows


def split_rows(rows: list[dict], val_fraction: float, seed: int):
    """A seeded split, so a resumed run trains and validates on the same halves."""
    shuffled = sorted(rows, key=lambda r: r["image_url"])
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_fraction))
    return shuffled[cut:], shuffled[:cut]


def build_example(processor, caption: str, gray, n_placeholders: int,
                  placeholder: str, max_caption_tokens: int) -> dict:
    """One training example: a prompt Qwen embeds itself, then the caption.

    The caption is appended to the prompt's ids rather than run through the chat
    template a second time, so the label mask is exactly the appended span and no
    template token is ever scored.
    """
    reserved = placeholder * n_placeholders
    messages = [{"role": "user", "content": [
        {"type": "image", "image": gray},
        {"type": "text", "text": reserved + CAPTION_PROMPT},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    answer = processor.tokenizer(
        caption, add_special_tokens=False, return_tensors="pt",
        truncation=True, max_length=max_caption_tokens,
    )["input_ids"]
    input_ids = torch.cat([prompt["input_ids"], answer], dim=1)[0]
    labels = torch.cat([
        torch.full_like(prompt["input_ids"][0], IGNORE), answer[0],
    ])
    example = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in prompt:
            example[key] = prompt[key]
    return example


def pad_batch(examples: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest example; padded labels are never scored."""
    width = max(e["input_ids"].shape[0] for e in examples)
    out: dict = {}
    for key, filler in (("input_ids", pad_id), ("labels", IGNORE), ("attention_mask", 0)):
        rows = []
        for e in examples:
            row = e[key]
            if row.shape[0] < width:
                row = torch.cat([row, torch.full((width - row.shape[0],), filler,
                                                 dtype=row.dtype)])
            rows.append(row)
        out[key] = torch.stack(rows)
    for key in ("pixel_values", "image_grid_thw"):
        if key in examples[0]:
            out[key] = torch.cat([e[key] for e in examples], dim=0)
    return out


class Captions(Dataset):
    def __init__(self, rows, cache_dir: Path, siglip_processor, side: int):
        self.rows = rows
        self.cache_dir = cache_dir
        self.siglip_processor = siglip_processor
        self.side = side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import hashlib

        from PIL import Image

        row = self.rows[index]
        name = hashlib.sha1(row["image_url"].encode()).hexdigest() + ".jpg"
        path = self.cache_dir / name
        if not path.is_file():
            return None
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except Exception:
            return None
        pixels = self.siglip_processor(images=image, return_tensors="pt")["pixel_values"]
        return {"siglip_pixel_values": pixels, "target_caption": row["target_caption"]}


def make_collate(processor, gray_size, n_placeholders, placeholder, max_caption_tokens):
    from PIL import Image

    # The processor is pinned to one resolution, so this canvas is resized to it;
    # its own size only has to be square and large enough not to be upscaled far.
    gray = Image.new("RGB", (gray_size, gray_size), GRAY)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def collate(items):
        items = [i for i in items if i is not None]
        if not items:
            return None
        examples = [
            build_example(processor, i["target_caption"], gray, n_placeholders,
                          placeholder, max_caption_tokens)
            for i in items
        ]
        batch = pad_batch(examples, pad_id)
        batch["siglip_pixel_values"] = torch.cat([i["siglip_pixel_values"] for i in items], dim=0)
        return batch

    return collate


def load_models(a):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, SiglipVisionModel

    processor = AutoProcessor.from_pretrained(
        a.llm_path, local_files_only=a.local_files_only,
        min_pixels=a.pixels, max_pixels=a.pixels,
    )
    config = AutoConfig.from_pretrained(a.llm_path, local_files_only=a.local_files_only)
    llm = Qwen3VLForConditionalGeneration.from_pretrained(
        a.llm_path, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, local_files_only=a.local_files_only,
    )
    vision = SiglipVisionModel.from_pretrained(
        a.vis_path, torch_dtype=torch.bfloat16, local_files_only=a.local_files_only,
    )
    text_config = getattr(config, "text_config", config)
    vision_config = getattr(config, "vision_config", config)
    merge = getattr(vision_config, "spatial_merge_size", None) or getattr(config, "spatial_merge_size")
    expert = DenseVisionExpert(
        vision.config.hidden_size, text_config.hidden_size, a.mode,
        n_layers=len(SIGLIP_LAYERS), prefix_side=a.prefix_side,
    )
    placeholder_id = processor.tokenizer.convert_tokens_to_ids(a.placeholder)
    if placeholder_id is None or placeholder_id < 0:
        fail(f"{a.placeholder!r} is not a single token in this tokenizer")
    siglip_side = vision.config.image_size // vision.config.patch_size
    model = MergedVLM(
        llm, vision, expert,
        siglip_side=siglip_side,
        image_token_id=getattr(config, "image_token_id", None)
        or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        spatial_merge_size=merge, placeholder_token_id=placeholder_id,
    )
    return model, processor


def run_epoch(model, loader, optimizer, device, grad_accum, train: bool, log_every: int):
    model.expert.train(train)
    total, seen = 0.0, 0
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        pixels = batch.pop("siglip_pixel_values").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(pixels, **batch)
            loss = out.loss
        if train:
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total += float(loss) * batch["input_ids"].shape[0]
        seen += batch["input_ids"].shape[0]
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    if not seen:
        fail("no usable batches; check the image cache")
    return total / seen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["prefix", "early"], required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--image-cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm-path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--vis-path", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--pixels", type=int, default=448 * 448,
                   help="min_pixels and max_pixels both, so the merged grid is constant")
    p.add_argument("--prefix-side", type=int, default=12)
    p.add_argument("--placeholder", default="<|vision_pad|>")
    p.add_argument("--gray-size", type=int, default=448,
                   help="side of the gray canvas handed to Qwen's own tower")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-caption-tokens", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rows = read_rows(Path(a.data_path))
    if a.limit:
        rows = rows[: a.limit]
    train_rows, val_rows = split_rows(rows, a.val_fraction, a.seed)
    print(f"{len(train_rows)} train / {len(val_rows)} val captions")

    model, processor = load_models(a)
    from transformers import AutoImageProcessor

    siglip_processor = AutoImageProcessor.from_pretrained(
        a.vis_path, local_files_only=a.local_files_only
    )
    device = next(model.expert.parameters()).device
    if torch.cuda.is_available():
        model.expert.to("cuda")
        device = torch.device("cuda")

    n_placeholders = a.prefix_side * a.prefix_side + 1 if a.mode == "prefix" else 0
    collate = make_collate(processor, a.gray_size, n_placeholders, a.placeholder,
                           a.max_caption_tokens)
    print(f"mode {a.mode}, {n_placeholders} reserved tokens, "
          f"SigLIP2 grid {model.siglip_side}x{model.siglip_side}")

    loaders = {
        "train": DataLoader(Captions(train_rows, Path(a.image_cache_dir), siglip_processor, 384),
                            batch_size=a.train_batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(Captions(val_rows, Path(a.image_cache_dir), siglip_processor, 384),
                          batch_size=a.eval_batch_size, shuffle=False, collate_fn=collate),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.expert.parameters() if p.requires_grad], lr=a.lr
    )
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(a.epochs):
        train_loss = run_epoch(model, loaders["train"], optimizer, device, a.grad_accum,
                               True, a.log_every)
        val_loss = run_epoch(model, loaders["val"], optimizer, device, a.grad_accum,
                             False, 0)
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
              f"ppl {math.exp(min(val_loss, 20)):.2f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"epoch": epoch, "val_loss": val_loss, "mode": a.mode,
                        "prefix_side": a.prefix_side,
                        "model_state_dict": model.trainable_state_dict()},
                       out_dir / "expert.bin")
            print(f"  saved {out_dir / 'expert.bin'}", flush=True)
    (out_dir / "complete.json").write_text(
        json.dumps({"mode": a.mode, "epochs": a.epochs, "best_val_loss": best,
                    "pixels": a.pixels, "prefix_side": a.prefix_side}, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
