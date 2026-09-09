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

import hashlib
import os
import re
import string

from PIL import Image


# Fallback for eval files that carry an ISO code in `source_language`
# instead of an `nllb_lang_tag` field (e.g. the shared xGQA testdev JSONL).
ISO_TO_NLLB = {
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "wo": "wol_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "id": "ind_Latn",
    "ga": "gle_Latn",
    "jv": "jav_Latn",
    "mn": "khk_Cyrl",
    "si": "sin_Sinh",
}


def row_nllb_tag(row: dict) -> str:
    """Return the row's NLLB tag, deriving it from `source_language` if absent."""
    tag = row.get("nllb_lang_tag")
    if tag:
        return tag
    lang = row.get("source_language", "")
    if lang in ISO_TO_NLLB:
        return ISO_TO_NLLB[lang]
    raise KeyError(
        f"Row has neither nllb_lang_tag nor a known source_language: {lang!r}"
    )


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (VQA-style)."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def open_ended_correct(pred_text: str, target: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(target)


def resolve_image(
    row: dict, images_dir: str | None, url_cache_dir: str | None
) -> Image.Image | None:
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
        cache_path = os.path.join(
            url_cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg"
        )
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                return None
    return None


def main() -> None:
    from eval_runtime import main as run

    run("xgqa")


if __name__ == "__main__":
    main()
