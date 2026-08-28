"""Download MGSM in the languages this project already covers.

MGSM and MSVAMP overlap our 11 languages in exactly bn/de/ru/zh, so the
reasoning-retention finding (DESIGN.md D11) can be measured in four
languages instead of one. Writes ``evaluation/MGSM_<lang>.jsonl`` with the
schema evaluate_text.py expects (``question``, ``answer``), which is what
evaluate_all.sh looks for.

Run on a LOGIN node — compute nodes have no network.

    module load StdEnv/2023 python/3.11.5 && source $SCRATCH/venvs/m2-align/bin/activate
    HF_HOME=$SCRATCH/huggingface python Approach2/fetch_text_benchmarks.py --langs de ru zh

Existing files are left alone. Nothing here goes into git: evaluation/ is
gitignored (benchmark data stays out of the repo).
"""
from __future__ import annotations

import argparse
import json
import os

MGSM_REPO = "juletxara/mgsm"
# juletxara/mgsm config names are the language codes themselves.
SUPPORTED = {"bn", "de", "ru", "zh", "en", "es", "fr", "ja", "sw", "te", "th"}


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch MGSM for extra languages.")
    ap.add_argument("--langs", nargs="+", default=["de", "ru", "zh"])
    ap.add_argument("--out-dir", default="evaluation")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    for lang in args.langs:
        if lang not in SUPPORTED:
            print(f"{lang}: MGSM does not cover it — skipping")
            continue
        out = os.path.join(args.out_dir, f"MGSM_{lang}.jsonl")
        if os.path.exists(out):
            print(f"{lang}: {out} exists — skipping")
            continue
        ds = load_dataset(MGSM_REPO, lang, split=args.split)
        rows = []
        for r in ds:
            # `answer` in this repo is the CoT string; the numeric gold is
            # `answer_number`, and evaluate_text.py compares against a number.
            num = r.get("answer_number")
            if num is None:
                continue
            rows.append({"question": r["question"], "answer": str(num)})
        write_jsonl(out, rows)
        print(f"{lang}: wrote {len(rows)} rows -> {out}")

    ref = os.path.join(args.out_dir, "MGSM.jsonl")
    if os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
            print("\nschema of the existing bn file, for comparison:")
            print(" ", f.readline().strip()[:300])


if __name__ == "__main__":
    main()
