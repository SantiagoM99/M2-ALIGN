"""Build Stage-2 caption pairs from LLaVA-Pretrain (BLIP-LAION-CC-SBU 558k).

Produces ``{"image_url", "target_caption"}`` rows drop-in compatible with
train_stage2_vision.py, extracting images straight out of ``images.zip``
into a ``sha1(url).jpg`` cache. No full unzip: 558k files twice over would
eat Lustre's ~1M-inode scratch quota, so only the sampled images are ever
materialized. English captions only — Approach 2's stage 2 needs no
bilingual pairs (see DESIGN.md D4), which is why this 558k set is usable
as-is.

The sample is a deterministic seeded shuffle prefix: raising ``--sample``
later re-extracts the SAME first N plus new ones, so the cache and jsonl
grow incrementally instead of resampling from scratch. Already-cached
images are skipped, so re-runs are cheap.

Download first (login node — compute nodes are offline):
    from huggingface_hub import hf_hub_download
    hf_hub_download("liuhaotian/LLaVA-Pretrain", <file>, repo_type="dataset",
                    local_dir=...)
    # files: blip_laion_cc_sbu_558k_meta.json, images.zip (~25GB)

Usage
-----
    python build_llava_pretrain.py \\
        --meta  $LL/blip_laion_cc_sbu_558k_meta.json \\
        --zip   $LL/images.zip \\
        --output    $LL/llava_pairs.jsonl \\
        --cache-dir $LL/image_cache \\
        --sample 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import zipfile

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaVA-Pretrain → Stage-2 pairs.")
    parser.add_argument("--meta", type=str, required=True,
                        help="blip_laion_cc_sbu_558k_meta.json (id/image/blip_caption/url).")
    parser.add_argument("--zip", dest="zip_path", type=str, required=True,
                        help="images.zip from the same HF dataset repo.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pairs JSONL (rewritten each run).")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="sha1(url).jpg image cache to populate.")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Rows to keep after the seeded shuffle (0 = all 558k — "
                             "mind the scratch inode quota).")
    parser.add_argument("--seed", type=int, default=13,
                        help="Shuffle seed. Keep it fixed so larger --sample values "
                             "are supersets of smaller ones.")
    args = parser.parse_args()

    with open(args.meta, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"meta rows: {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    if args.sample > 0:
        rows = rows[: args.sample]

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    zf = zipfile.ZipFile(args.zip_path)
    names = set(zf.namelist())

    n_ok = n_skip = n_cached = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="extract"):
            caption = (r.get("blip_caption") or "").strip()
            member = r.get("image")
            url = r.get("url") or member
            if not caption or not member or not url:
                n_skip += 1
                continue
            if member not in names:
                if f"images/{member}" in names:
                    member = f"images/{member}"
                else:
                    n_skip += 1
                    continue
            cache_path = os.path.join(
                args.cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg"
            )
            if os.path.exists(cache_path):
                n_cached += 1
            else:
                try:
                    data = zf.read(member)
                except Exception:
                    n_skip += 1
                    continue
                tmp = cache_path + ".tmp"
                with open(tmp, "wb") as g:
                    g.write(data)
                os.replace(tmp, cache_path)
            out.write(json.dumps(
                {"image_url": url, "target_caption": caption}, ensure_ascii=False
            ) + "\n")
            n_ok += 1

    print(f"wrote {n_ok} pairs → {args.output} "
          f"(skipped {n_skip}, already cached {n_cached})")


if __name__ == "__main__":
    main()
