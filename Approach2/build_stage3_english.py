"""Put the untranslated English GQA rows into the per-language stage-3 schema.

Her pooled stage 3 includes English — she verified against MindMerger's
`read_datasets.py` that the augmentation stage does, while the mapping stage does
not — and reads it from `Stage3/data/en.jsonl` in the same schema as every
translated language. Our data mirror has the same 30,000 rows under
`english.jsonl`, but in the *source* schema it was translated from:
`{question, answer, vg_image_id}`, with no `query` and no `nllb_lang_tag`. Her
trainer reads `row["query"]` and `row["nllb_lang_tag"]`, so the file cannot be
pooled as it stands.

This is a rename and a tag, not a transformation: the rows are the untranslated
originals of our own `bn.jsonl` (both start at vg_image_id 2375767, "Is there a
bread to the right of the soup?"), so the English pool is exactly the English
side of what the other eleven languages were translated from.

    python3 Approach2/build_stage3_english.py \\
      --input $DT/Stage3/data/english.jsonl \\
      --output $DT/Stage3/data/en.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_DATASET = "gqa_balanced_train_english"
NLLB_TAG = "eng_Latn"


def fail(message: str):
    raise SystemExit(f"build_stage3_english: {message}")


def convert(rows) -> list[dict]:
    out = []
    for lineno, row in enumerate(rows, 1):
        missing = [k for k in ("question", "answer", "vg_image_id")
                   if not str(row.get(k, "")).strip()]
        if missing:
            fail(f"line {lineno}: missing {', '.join(missing)}")
        question = str(row["question"]).strip()
        out.append({
            "id": hashlib.sha1(
                f"{SOURCE_DATASET}|{row['vg_image_id']}|{question}".encode()
            ).hexdigest(),
            "vg_image_id": str(row["vg_image_id"]),
            "query": question,
            "answer": str(row["answer"]).strip(),
            "source_language": "English",
            "nllb_lang_tag": NLLB_TAG,
            "source_dataset": SOURCE_DATASET,
        })
    ids = {r["id"] for r in out}
    if len(ids) != len(out):
        fail(f"{len(out) - len(ids)} rows collide on id; the hash inputs are not unique")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    source = Path(a.input)
    if not source.is_file():
        fail(f"missing {source}")
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    converted = convert(rows)

    out = Path(a.output)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted)
    if out.is_file() and out.read_text(encoding="utf-8") == body:
        print(f"{out} already holds these {len(converted)} rows")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {len(converted)} rows to {out} (nllb_lang_tag={NLLB_TAG})")


if __name__ == "__main__":
    main()
