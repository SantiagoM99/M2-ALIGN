"""Build stage-3 VQA rows from CulturalGround, and mix them with the GQA rows.

CulturalGround (Nyandwi et al., EMNLP 2025; `neulab/CulturalGround`) is 22M
open-ended culturally grounded VQA pairs over 42 countries and 39 languages,
built from Wikidata entities with Wikimedia Commons images, and it supersedes
WorldCuisines for this intervention on three counts: it is broad-domain rather
than food-only, so a null result no longer confounds "the distribution is not the
cause" with "one domain is too narrow"; it ships images as per-country archives,
so there is no scraping, no thumbnail-width rejection and no rate limiting; and
it is Apache-2.0.

What this intervention may and may not claim. That culturally grounded
supervision helps culture-specific VQA is **their** result, at 22M pairs with
full instruction tuning. Ours is the dose-and-mechanism question their scale
leaves open: whether a small, fixed budget of such supervision — replacing an
equal number of translated-GQA rows, so the item count never changes — moves
culture-specific transfer in a stack whose only trainable part is a 58M
connector.

Two rules carried over from `build_worldcuisines.py`, for the same reasons:
`--lang` refuses this project's zero-shot targets, because one training row in a
target language ends the zero-shot claim for it; and `--mix-with` replaces rather
than adds, so the contrast measures distribution and not volume.

Discover the schema before building, because the per-country files are not
documented field by field:

    python build_culturalground.py --probe --country bangladesh

Then build and mix:

    python build_culturalground.py --country bangladesh --lang bn \\
      --sample 20000 --images-dir $DT/Stage3/data/gqa/images \\
      --mix-with $DT/Stage3/data/bn.jsonl --fraction 0.5 \\
      --output $DT/Stage3/data/bn_cg50.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import tarfile
from pathlib import Path

REPO = "neulab/CulturalGround"
OPEN_ENDED = "CulturalGround-Recipes/CulturalGround-Refined-OE/{country}_refined.jsonl"
IMAGES = "CultureGroundImages/{country}.tar.gz"
SOURCE_DATASET = "culturalground_refined_oe"

NLLB = {
    "bn": ("Bengali", "ben_Beng"),
    "en": ("English", "eng_Latn"),
    "id": ("Indonesian", "ind_Latn"),
    "ru": ("Russian", "rus_Cyrl"),
    "zh": ("Chinese", "zho_Hans"),
    "pt": ("Portuguese", "por_Latn"),
    "ko": ("Korean", "kor_Hang"),
    "de": ("German", "deu_Latn"),
}
# This project's zero-shot transfer targets; training on one ends its claim.
FORBIDDEN = ("jv", "mn", "ga", "si", "su")

# Filled in from --probe output. Each value is the list of row keys that may hold
# the field, tried in order, so the builder fails loudly naming the real keys
# rather than silently emitting empty questions.
# Confirmed against `--probe --country india` (717,590 rows): the questions are
# `reformulated_question` (natural phrasing) with `original_question` as the
# Wikidata-derived fallback; the short answer is `label`, the long ones being
# `reformulated_answer`/`original_answer`; and the image filename lives in
# `media`, because `image` holds the string "None" on these rows.
FIELDS = {
    "question": ["reformulated_question", "original_question", "question", "text"],
    "answer": ["label", "answer", "response"],
    "image": ["image", "media", "image_path", "file_name"],
    "language": ["language", "lang"],
    "identifier": ["id", "qa_id", "uid"],
    # The answer belongs to an (entity, property, question type), not to a row,
    # which is what lets the English answer be joined onto a Bengali question.
    "property": ["property_id"],
    "question_type": ["question_type"],
}

# These rows spell a missing value as the string "None", so `row[key] not in
# (None, "")` is not enough and a bare str() would emit the word into a prompt.
MISSING = {"none", "null", "nan", ""}


def fail(message: str):
    raise SystemExit(f"build_culturalground: {message}")


def donor_tag(lang: str) -> tuple[str, str]:
    if any(lang.startswith(t) for t in FORBIDDEN):
        fail(f"{lang} is a zero-shot transfer target: training on it ends the zero-shot "
             "claim for that language. Use a donor language instead")
    if lang not in NLLB:
        fail(f"no NLLB tag recorded for {lang}; add it to NLLB first")
    return NLLB[lang]


def pick(row: dict, field: str):
    """The first candidate key this row actually carries."""
    for key in FIELDS[field]:
        if key in row and row[key] is not None and str(row[key]).strip().lower() not in MISSING:
            return row[key]
    return None


def stable_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def download(path_template: str, country: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, path_template.format(country=country), repo_type="dataset"))


def probe(country: str, lang: str = "bn") -> None:
    rows = list(read_jsonl(download(OPEN_ENDED, country)))
    print(f"{country}: {len(rows)} rows")
    print(f"keys: {sorted(rows[0])}")
    for field in FIELDS:
        found = [k for k in FIELDS[field] if k in rows[0]]
        print(f"  {field:11s} -> {found or 'NONE OF ' + str(FIELDS[field])}")
    langs = collections.Counter(str(pick(r, "language")) for r in rows)
    print(f"languages: {langs.most_common(10)}")

    # What the build actually depends on, counted before anyone spends an hour
    # on it: a question, a resolvable English answer and an image filename.
    wanted = {lang.lower(), NLLB[lang][0].lower()} if lang in NLLB else {lang.lower()}
    mine = [r for r in rows
            if any(str(pick(r, "language") or "").lower().startswith(w) for w in wanted)]
    answers = english_answers(rows)
    usable = sum(1 for r in mine if pick(r, "question") and pick(r, "image")
                 and answers.get(answer_key(r)))
    print(f"{lang}: {len(mine)} rows, {usable} with question + image + English answer "
          f"({len(answers)} English answers joinable)")
    for field in ("question", "answer", "image"):
        sample = [str(pick(r, field)) for r in mine[:200] if pick(r, field)][:2]
        print(f"  {field:9s} sample: {sample}")
    print("first row of this language (truncated):")
    if mine:
        print(json.dumps({k: str(v)[:120] for k, v in mine[0].items()},
                         ensure_ascii=False, indent=1))


def select(rows, lang: str, sample: int, per_image: int, seed: int):
    """Deterministic prefix of a seeded shuffle, capped per image.

    The prefix is stable under a larger --sample so raising it keeps every row
    already built. The cap matters because these rows are generated per cultural
    entity, so one entity can carry many questions.
    """
    # CulturalGround's `language` may hold the code or the name, so accept both
    # rather than spending a round trip on the cluster to find out which.
    wanted = {lang.lower(), NLLB[lang][0].lower()}
    kept = [r for r in rows
            if any(str(pick(r, "language") or "").lower().startswith(w) for w in wanted)]
    if not kept:
        seen = collections.Counter(str(pick(r, "language")) for r in rows)
        fail(f"no rows matching {sorted(wanted)}; this file carries {seen.most_common(6)}")
    kept.sort(key=lambda r: str(pick(r, "identifier")))
    random.Random(seed).shuffle(kept)
    taken, per_entity = [], collections.Counter()
    for row in kept:
        image = str(pick(row, "image"))
        if per_entity[image] >= per_image:
            continue
        per_entity[image] += 1
        taken.append(row)
        if len(taken) >= sample:
            break
    return taken


def extract_images(country: str, wanted: set[str], images_dir: Path) -> set[str]:
    """Extract only the wanted members, reading each with extractfile().

    Never extractall(): an archive member can carry an absolute path or `..` and
    write outside the destination. The names are also flattened to `cg_<basename>`
    so they cannot collide with GQA's numeric ids in the shared images directory.
    """
    archive = download(IMAGES, country)
    written = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            target = images_dir / f"cg_{base}"
            if not target.exists():
                target.write_bytes(handle.read())
            written.add(base)
    return written


def answer_key(row) -> tuple:
    """What an answer is about: the entity, the property, the question type.

    Joining on the entity alone would mix a property answer ("its capital") with
    an entity answer ("Narendra Modi") whenever one entity carries both.
    """
    return (str(pick(row, "identifier")), str(pick(row, "property")),
            str(pick(row, "question_type")))


def english_answers(rows) -> dict:
    """(entity, property, type) -> English answer, when it is unambiguous.

    CulturalGround's `label` is in the row's own language, but our stage-3 rows
    are native question with **English** answer — that is the format CVQA is
    scored in and the one the translated-GQA rows use. Taking the Bengali label
    would change the output language as well as the knowledge, and the language
    change would dominate. India ships 100,191 English rows against 48,562
    Bengali ones, and the entity id is shared, so the English answer is a join
    rather than a translation.
    """
    by_key: dict = {}
    for row in rows:
        if str(pick(row, "language") or "").lower()[:2] != "en":
            continue
        answer = str(pick(row, "answer") or "").strip()
        if not answer:
            continue
        by_key.setdefault(answer_key(row), set()).add(answer)
    return {k: v.pop() for k, v in by_key.items() if len(v) == 1}


def build_rows(selected, lang: str, country: str, images_dir: Path,
               answers: dict | None = None) -> list[dict]:
    language, tag = donor_tag(lang)
    wanted = {Path(str(pick(r, "image"))).name for r in selected}
    have = extract_images(country, wanted, images_dir)
    missing = len(wanted - have)
    if missing:
        print(f"{missing} of {len(wanted)} images were not in the archive", file=sys.stderr)
    out = []
    unresolved = 0
    for row in selected:
        question = str(pick(row, "question") or "").strip()
        if answers is None:
            answer = str(pick(row, "answer") or "").strip()
        else:
            answer = answers.get(answer_key(row), "")
            if not answer:
                unresolved += 1
        base = Path(str(pick(row, "image") or "")).name
        if not question or not answer or base not in have:
            continue
        out.append({
            "id": stable_id(SOURCE_DATASET, country, lang, pick(row, "identifier"),
                            pick(row, "property"), pick(row, "question_type")),
            "vg_image_id": f"cg_{Path(base).stem}",
            "query": question,
            "answer": answer,
            "source_language": language,
            "nllb_lang_tag": tag,
            "source_dataset": SOURCE_DATASET,
        })
    if unresolved:
        print(f"{unresolved} of {len(selected)} rows had no unambiguous English answer",
              file=sys.stderr)
    return out


def mix(cultural: list[dict], gqa_path: Path, fraction: float, seed: int) -> list[dict]:
    """Replace `fraction` of the GQA rows, holding the total row count fixed."""
    gqa = list(read_jsonl(gqa_path))
    total = len(gqa)
    want = round(fraction * total)
    if want > len(cultural):
        fail(f"need {want} cultural rows for fraction {fraction} of {total}, built "
             f"{len(cultural)}: raise --sample or add another --country")
    rng = random.Random(seed)
    keep = sorted(gqa, key=lambda r: r["id"])
    rng.shuffle(keep)
    mixed = keep[: total - want] + cultural[:want]
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", required=True, help="one of CulturalGround's country files")
    p.add_argument("--probe", action="store_true", help="print the schema and languages, build nothing")
    p.add_argument("--native-answers", action="store_true",
                   help="keep CulturalGround's own-language answers instead of joining "
                        "the English ones; changes the output language as well as the "
                        "knowledge, so it is an ablation and not the arm")
    p.add_argument("--lang", default="bn", help=f"donor language; one of {sorted(NLLB)}")
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--per-image", type=int, default=4)
    p.add_argument("--images-dir")
    p.add_argument("--output")
    p.add_argument("--mix-with")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()

    if a.probe:
        probe(a.country, a.lang)
        return
    for required in ("images_dir", "output"):
        if not getattr(a, required):
            fail(f"--{required.replace('_', '-')} is required unless --probe")
    donor_tag(a.lang)
    if not 0.0 < a.fraction <= 1.0:
        fail("--fraction must be in (0, 1]")

    images_dir = Path(a.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(download(OPEN_ENDED, a.country)))
    selected = select(rows, a.lang, a.sample, a.per_image, a.seed)
    print(f"{a.country}: {len(selected)} rows selected in {a.lang}")
    answers = None if a.native_answers else english_answers(rows)
    if answers is not None:
        print(f"{len(answers)} unambiguous English answers available to join")
    cultural = build_rows(selected, a.lang, a.country, images_dir, answers)
    print(f"{len(cultural)} rows with a usable image")

    final = mix(cultural, Path(a.mix_with), a.fraction, a.seed) if a.mix_with else cultural
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_source = collections.Counter(r["source_dataset"] for r in final)
    print(f"wrote {len(final)} rows to {out}")
    print(json.dumps({"rows_by_source": dict(by_source),
                      "distinct_images": len({r["vg_image_id"] for r in final
                                              if r["source_dataset"] == SOURCE_DATASET})}, indent=1))


if __name__ == "__main__":
    main()
