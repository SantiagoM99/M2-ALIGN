"""Check the two Qwen3-VL assumptions the merged architecture rests on.

`model.py` isolates everything Qwen-specific in `native_visual_positions` and
`native_grid`, and both encode an assumption about how this version of
transformers lays out a visual turn:

1. the processor returns `image_grid_thw` in pre-merge patch units, so the merged
   grid is (h // spatial_merge_size, w // spatial_merge_size);
2. the tokenizer expands exactly one image-pad id per merged visual token, so
   counting that id gives the same number the grid predicts.

If either is false, early injection would add features at the wrong positions and
still train to a plausible loss, which is the failure this file exists to prevent.
It also reports the decoder's hidden size and the attribute path to its layer
list, both of which the trainer needs.

Run on a GPU-less login node; it loads the processor and the config, never the
weights:

    python Approach2/merged/verify_qwen_api.py
"""
from __future__ import annotations

import argparse
import sys

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=MODEL)
    p.add_argument("--size", type=int, nargs=2, default=[448, 448], metavar=("H", "W"))
    a = p.parse_args()

    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(a.model_id, local_files_only=True)
    config = AutoConfig.from_pretrained(a.model_id, local_files_only=True)

    image = Image.new("RGB", (a.size[1], a.size[0]), (128, 128, 128))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "what is this?"},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    print("processor returned:", sorted(inputs.keys()))

    if "image_grid_thw" not in inputs:
        sys.exit("FAIL: no image_grid_thw; native_grid() has nothing to read")

    vision = getattr(config, "vision_config", config)
    merge = getattr(vision, "spatial_merge_size", None) or getattr(config, "spatial_merge_size", None)
    if merge is None:
        sys.exit("FAIL: no spatial_merge_size in the config; native_grid() cannot merge")

    thw = inputs["image_grid_thw"]
    t, h, w = (int(v) for v in thw.reshape(-1)[:3])
    predicted = (h // merge) * (w // merge)

    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    counted = int((inputs["input_ids"] == image_token_id).sum())

    text_config = getattr(config, "text_config", config)
    hidden = getattr(text_config, "hidden_size", None)

    print(f"image_grid_thw = (t={t}, h={h}, w={w}), spatial_merge_size = {merge}")
    print(f"merged grid    = {h // merge} x {w // merge} = {predicted} tokens")
    print(f"image_token_id = {image_token_id}, counted in input_ids = {counted}")
    print(f"decoder hidden size = {hidden}")
    print(f"num decoder layers  = {getattr(text_config, 'num_hidden_layers', '?')}")

    if counted != predicted:
        sys.exit(
            f"FAIL: {counted} image-pad tokens but the grid predicts {predicted}. "
            "Early injection would add features at the wrong positions; fix "
            "native_grid()/native_visual_positions() before training."
        )
    print("OK: both assumptions hold for this transformers version")


if __name__ == "__main__":
    main()
