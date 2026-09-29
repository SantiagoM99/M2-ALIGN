"""How many visual tokens does Qwen3-VL actually spend on a CVQA image?

Arm 2 of the culture-adapter design raises `max_pixels` to test whether visual
token density explains our CVQA advantage. That only makes sense if there is
headroom, and nobody has measured what the default already spends: CVQA images are
user photographs, often much larger than the 448 square the API check used, so the
processor may already be running them at high resolution. This reports the merged
token count per image at the default and at each requested cap, so arm 2's value is
chosen from data instead of guessed — and it needs no GPU, so it runs on a login
node in minutes.

    python Approach2/merged/visual_tokens.py --images-dir $DT/Stage3/data/cvqa/images
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path


def summarise(name: str, counts: list[int]) -> None:
    counts = sorted(counts)
    q = lambda f: counts[min(len(counts) - 1, int(f * len(counts)))]  # noqa: E731
    print(f"{name:>22s}: median {statistics.median(counts):7.0f}  "
          f"p10 {q(0.10):6d}  p90 {q(0.90):6d}  max {counts[-1]:6d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--caps", type=int, nargs="+",
                   default=[200704, 602112, 1204224],  # 448^2, ~768^2, ~1097^2
                   help="max_pixels values to compare against the default")
    a = p.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    paths = sorted(Path(a.images_dir).glob("*.jpg"))[: a.sample]
    if not paths:
        raise SystemExit(f"no .jpg under {a.images_dir}")
    images = []
    for path in paths:
        try:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        except Exception:
            continue
    print(f"{len(images)} images from {a.images_dir}")
    print(f"source pixels: median {statistics.median(i.width * i.height for i in images):.0f}")

    for cap in [None, *a.caps]:
        kwargs = {} if cap is None else {"max_pixels": cap, "min_pixels": cap}
        processor = AutoProcessor.from_pretrained(a.model_id, local_files_only=True, **kwargs)
        merge = processor.image_processor.merge_size
        counts = []
        for image in images:
            grid = processor.image_processor(images=image, return_tensors="pt")["image_grid_thw"]
            _, h, w = (int(v) for v in grid.reshape(-1)[:3])
            counts.append((h // merge) * (w // merge))
        summarise("default" if cap is None else f"max_pixels={cap}", counts)

    print("\nOur SigLIP2 path spends 729 unmerged tokens at 384 square, for comparison.")


if __name__ == "__main__":
    main()
