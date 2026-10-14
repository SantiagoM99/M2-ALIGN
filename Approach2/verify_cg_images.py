"""Verify an existing CulturalGround pool by the trainer's own criterion.

This arm has cost two GPU jobs to images that pass every cheap check and fail
`image.load()` (DESIGN 2026-10-09): `Image.open` succeeds on a truncated JPEG and
the trainer's `convert("RGB")` is what triggers the load that raises, three hours
into training. The builder now decodes what it extracts, but a pool built before
that fix has the same row count as one built after it, so neither the line count
nor the leakage audit can tell them apart. This re-runs the consumer's criterion
over a pool that already exists, without rebuilding it.

Run on a login node with the venv active:

    python Approach2/verify_cg_images.py $DT/Stage3/data/gqa/images \
      --rows $DT/Stage3/data/bn_cg50.jsonl

Exit status is 1 if any image fails, so a launcher can gate on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def decodes(path: Path) -> str:
    """Return "" when the trainer could open this file, else why it could not."""
    try:
        with Image.open(path) as image:
            image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - the criterion is "whatever the trainer sees"
        return f"{type(exc).__name__}: {exc}"
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images_dir")
    p.add_argument("--prefix", default="cg_", help="only check files under this namespace")
    p.add_argument("--rows", help="a stage-3 jsonl; reports how many of its rows a failure kills")
    p.add_argument("--delete", action="store_true",
                   help="remove the files that fail (the cg_ namespace is ours)")
    args = p.parse_args()

    images = sorted(Path(args.images_dir).glob(f"{args.prefix}*"))
    if not images:
        raise SystemExit(f"no {args.prefix}* images under {args.images_dir}")

    broken: dict[str, str] = {}
    for path in images:
        why = decodes(path)
        if why:
            broken[path.name] = why

    print(f"checked {len(images)} {args.prefix}* images: {len(broken)} fail to decode")
    for name, why in list(broken.items())[:10]:
        print(f"  {name}  {why}")
    if len(broken) > 10:
        print(f"  ... and {len(broken) - 10} more")

    if args.rows:
        # Rows carry `vg_image_id` with no extension and the trainer appends one
        # (`train_stage3_vqa.py:106`), so a row is matched by stem, not filename.
        broken_stems = {Path(name).stem for name in broken}
        hit = 0
        total = 0
        with open(args.rows, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                if str(json.loads(line).get("vg_image_id", "")) in broken_stems:
                    hit += 1
        print(f"{hit} of {total} training rows reference an image that fails")

    if broken and args.delete:
        for name in broken:
            (Path(args.images_dir) / name).unlink()
        print(f"deleted {len(broken)} files; rebuild the pool so its rows match")

    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
