"""Build stage-3 VQA rows from WorldCuisines, and mix them with the GQA rows.

Why this dataset. Block C measured that stage-3 supervision on NLLB-translated
GQA buys grounding on the benchmark that shares its shape (+12.39 on xGQA-bn)
and nothing measurable on culture-specific targets (+1.32 [−1.13, +3.93]);
Maryam's a1 reproduces the same split on a frozen VLM (+6.28 xGQA, flat CVQA).
If the supervision's *distribution* is what confines the gain, culturally
grounded supervision should move CVQA. WorldCuisines is the only multilingual,
culturally grounded VQA set with a training split (1M rows, 30 languages);
MaXM, the other candidate, is test-only.

Two rules this file exists to enforce.

**The target languages are never trained on.** WorldCuisines ships native
Javanese and Sinhala, and jv/mn/ga/si are this project's zero-shot targets: a
single Javanese training row would end the zero-shot claim for Javanese and make
every transfer number incomparable with the rest of the record. ``--lang``
refuses them outright. Bengali is the default because it is the donor the
current stage 3 already uses, which leaves the image and question distribution
as the only thing that changes.

**The item count stays fixed.** ``--mix-with`` replaces a fraction of the GQA
rows rather than adding to them, so the contrast measures the distribution and
not the volume — the same discipline D11 used for stage-2 data.

Run on a login node; compute nodes have no network.

    python build_worldcuisines.py --lang bn --sample 15000 \\
      --images-dir $DT/Stage3/data/worldcuisines/images \\
      --output $DT/Stage3/data/worldcuisines_bn.jsonl

    python build_worldcuisines.py --lang bn --mix-with $DT/Stage3/data/bn.jsonl \\
      --fraction 0.5 --images-dir $DT/Stage3/data/worldcuisines/images \\
      --output $DT/Stage3/data/bn_wc50.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

NLLB = {
    "bn": ("Bengali", "ben_Beng"),
    "en": ("English", "eng_Latn"),
    "id_formal": ("Indonesian", "ind_Latn"),
    "ru_formal": ("Russian", "rus_Cyrl"),
    "zh": ("Chinese", "zho_Hans"),
}
# This project's zero-shot transfer targets. WorldCuisines has native data for
# jv and si; using it would make those languages supervised.
FORBIDDEN = ("jv", "mn", "ga", "si", "su")
REPO = "worldcuisines/vqa"
SOURCE_DATASET = "worldcuisines_train_task1"
# Wikimedia's User-Agent policy wants a descriptive agent that identifies the
# client and where to read about it. A repository URL says both without sending
# anyone's personal contact details to a third party.
USER_AGENT = "M2-ALIGN/1.0 (academic research; +https://github.com/SantiagoM99/M2-ALIGN)"
# Training images only, and the vision tower resizes to 384 anyway, so the
# originals are downscaled on the way into the cache: full-resolution Commons
# files would cost several GB of scratch for no signal.
MAX_SIDE = 768


def fail(message: str):
    raise SystemExit(f"build_worldcuisines: {message}")


def donor_tag(lang: str) -> tuple[str, str]:
    """The (language, NLLB tag) of a legitimate donor, or abort."""
    if any(lang.startswith(t) for t in FORBIDDEN):
        fail(f"{lang} is a zero-shot transfer target: training on it ends the zero-shot "
             "claim for that language. Use a donor language instead")
    if lang not in NLLB:
        fail(f"no NLLB tag recorded for {lang}; add it to NLLB first")
    return NLLB[lang]


def stable_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def download_split(lang: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, f"hf_prompt/train_task1/{lang}_train_task1.jsonl",
                                repo_type="dataset"))


def select(rows, sample: int, per_image: int, seed: int):
    """Deterministic prefix of a seeded shuffle, capped per dish.

    The prefix is stable under a larger --sample, so raising it keeps every row
    already built and every image already cached instead of resampling. The cap
    matters more here than in GQA: WorldCuisines asks many questions about the
    same dish, and without it a handful of foods would carry the whole sample.
    """
    rows = sorted(rows, key=lambda r: r["qa_id"])
    random.Random(seed).shuffle(rows)
    taken, per_food = [], {}
    for row in rows:
        food = row["food_id"]
        if per_food.get(food, 0) >= per_image:
            continue
        per_food[food] = per_food.get(food, 0) + 1
        taken.append(row)
        if len(taken) >= sample:
            break
    return taken


def original_upload_url(url: str) -> str:
    """Rewrite a Wikimedia thumbnail URL to the original file.

    WorldCuisines stores thumbnails at arbitrary widths (`1279px-`), and
    Wikimedia now answers those with HTTP 400 and a pointer to its list of
    allowed sizes. The original file is always served, so `/thumb/<a>/<ab>/
    <name>/NNNpx-<name>` collapses to `/<a>/<ab>/<name>`.
    """
    url = url.split("?", 1)[0]
    if "/thumb/" not in url:
        return url
    head, _, _tail = url.rpartition("/")
    return head.replace("/thumb/", "/", 1)


def fetch(url: str, throttle: float, attempts: int = 4):
    """GET with a descriptive agent, a pause between calls and backoff on 429.

    Without both, Wikimedia rate-limits the run within a few hundred images.
    """
    import requests

    for attempt in range(attempts):
        time.sleep(throttle)
        response = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
        if response.status_code == 429:
            time.sleep(throttle + 2 ** attempt)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError(f"rate-limited after {attempts} attempts")


def cache_image(url: str, name: str, images_dir: Path, throttle: float = 0.2) -> bool:
    from PIL import Image

    path = images_dir / f"{name}.jpg"
    if path.exists():
        return True
    try:
        body = fetch(original_upload_url(url), throttle)
        image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:  # a dead URL is a skipped row, never a failed build
        print(f"  skip {name}: {exc}", file=sys.stderr)
        return False
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(path, format="JPEG", quality=95)
    return True


def build_rows(selected, lang: str, images_dir: Path, throttle: float = 0.2) -> list[dict]:
    language, tag = donor_tag(lang)
    out, skipped = [], 0
    for row in selected:
        question = (row.get("open_ended_prompt") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        name = f"wc_{row['food_id']}"
        if not cache_image(row["image_url"], name, images_dir, throttle):
            skipped += 1
            continue
        out.append({
            "id": stable_id(SOURCE_DATASET, lang, row["qa_id"]),
            "vg_image_id": name,
            "query": question,
            "answer": answer,
            "source_language": language,
            "nllb_lang_tag": tag,
            "source_dataset": SOURCE_DATASET,
        })
    if skipped:
        print(f"{skipped} rows skipped for an unusable image", file=sys.stderr)
    return out


def mix(cultural: list[dict], gqa_path: Path, fraction: float, seed: int) -> list[dict]:
    """Replace `fraction` of the GQA rows, holding the total row count fixed."""
    gqa = list(read_jsonl(gqa_path))
    total = len(gqa)
    want_cultural = round(fraction * total)
    if want_cultural > len(cultural):
        fail(f"need {want_cultural} cultural rows for fraction {fraction} of {total}, "
             f"built {len(cultural)}: raise --sample")
    rng = random.Random(seed)
    keep = sorted(gqa, key=lambda r: r["id"])
    rng.shuffle(keep)
    mixed = keep[: total - want_cultural] + cultural[:want_cultural]
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", default="bn", help=f"donor language; one of {sorted(NLLB)}")
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--per-image", type=int, default=4, help="cap on questions per dish")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mix-with", help="GQA-style stage-3 jsonl whose row count is preserved")
    p.add_argument("--fraction", type=float, default=0.5,
                   help="share of --mix-with's rows replaced by cultural rows")
    p.add_argument("--throttle", type=float, default=0.2,
                   help="seconds between image requests; Wikimedia rate-limits without it")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()

    donor_tag(a.lang)
    if not 0.0 < a.fraction <= 1.0:
        fail("--fraction must be in (0, 1]")

    images_dir = Path(a.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    split = download_split(a.lang)
    selected = select(list(read_jsonl(split)), a.sample, a.per_image, a.seed)
    print(f"{split.name}: {len(selected)} rows selected; caching images into {images_dir}")
    cultural = build_rows(selected, a.lang, images_dir, a.throttle)
    print(f"{len(cultural)} rows with a usable image")

    rows = mix(cultural, Path(a.mix_with), a.fraction, a.seed) if a.mix_with else cultural
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
    print(f"wrote {len(rows)} rows to {out}")
    print(json.dumps({"rows_by_source": counts,
                      "distinct_dishes": len({r['vg_image_id'] for r in rows
                                              if r['source_dataset'] == SOURCE_DATASET})}, indent=1))


if __name__ == "__main__":
    main()
