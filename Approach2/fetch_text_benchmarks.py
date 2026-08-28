"""Fetch MGSM / MSVAMP in the languages this project already covers.

MGSM and MSVAMP overlap our 11 languages in exactly bn/de/ru/zh, so the
reasoning-retention finding (DESIGN.md D11) can be measured in four
languages instead of one. Writes ``<out-dir>/<BENCH>_<lang>.jsonl`` with the
two fields evaluate_text.py reads — ``question`` and ``answer`` — which is
what evaluate_all.sh looks for.

STANDARD LIBRARY ONLY (urllib + json): no datasets, no huggingface_hub, no
packaging. So it runs anywhere with network — a laptop, or an Alliance LOGIN
node. Compute nodes have no network; if the login node is blocked too, run
this locally and rsync the resulting files across.

Sources and the field mapping each needs:
  MGSM    juletxara/mgsm, per-language parquet, read through the HF
          datasets-server rows API (100 rows/page).  question, answer_number
          (`answer` is null outside English — it holds the English CoT).
  MSVAMP  Mathoctopus/MSVAMP, test_<Language>.json, actually JSON *lines*.
          m_query is the localized question (`query` is English), response
          is the numeric gold.

Answers are kept verbatim; evaluate_text.py's math_correct compares as
floats, so "8.0" and "8" already match.

    python3 Approach2/fetch_text_benchmarks.py --langs de ru zh --out-dir evaluation

Generate bn too and diff it against the existing evaluation/MGSM.jsonl to
confirm protocol parity before trusting the new languages. Nothing here
goes into git: evaluation/ is gitignored (benchmark data stays out).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

MGSM_REPO = "juletxara/mgsm"
MSVAMP_URL = ("https://huggingface.co/datasets/Mathoctopus/MSVAMP/resolve/main/"
              "test_{name}.json")
ROWS_API = "https://datasets-server.huggingface.co/rows"

# MSVAMP names its files by the English language name; MGSM uses codes.
MSVAMP_NAME = {"bn": "Bengali", "de": "German", "ru": "Russian", "zh": "Chinese",
               "en": "English", "es": "Spanish", "fr": "French", "ja": "Japanese",
               "sw": "Swahili", "th": "Thai"}
MGSM_LANGS = {"bn", "de", "ru", "zh", "en", "es", "fr", "ja", "sw", "te", "th"}


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "m2-align/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_mgsm(lang: str) -> list[dict]:
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": MGSM_REPO, "config": lang,
                                    "split": "test", "offset": offset, "length": 100})
        page = json.loads(get(f"{ROWS_API}?{q}"))
        batch = page.get("rows", [])
        if not batch:
            break
        for item in batch:
            r = item["row"]
            num = r.get("answer_number")
            if num is None or not r.get("question"):
                continue
            rows.append({"question": r["question"].strip(), "answer": str(num)})
        offset += len(batch)
        if offset >= page.get("num_rows_total", 0):
            break
    return rows


def fetch_msvamp(lang: str) -> list[dict]:
    raw = get(MSVAMP_URL.format(name=MSVAMP_NAME[lang])).decode("utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:            # the release is JSON lines
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    rows = []
    for r in data:
        q = (r.get("m_query") or r.get("query") or "").strip()
        a = r.get("response")
        if q and a not in (None, ""):
            rows.append({"question": q, "answer": str(a).strip()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch MGSM/MSVAMP for extra languages.")
    ap.add_argument("--langs", nargs="+", default=["de", "ru", "zh"])
    ap.add_argument("--benchmarks", nargs="+", default=["mgsm", "msvamp"],
                    choices=["mgsm", "msvamp"])
    ap.add_argument("--out-dir", default="evaluation")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for bench in args.benchmarks:
        name = "MGSM" if bench == "mgsm" else "MSVAMP"
        for lang in args.langs:
            if bench == "mgsm" and lang not in MGSM_LANGS:
                print(f"  {name} {lang}: not covered — skipping"); continue
            if bench == "msvamp" and lang not in MSVAMP_NAME:
                print(f"  {name} {lang}: not covered — skipping"); continue
            out = os.path.join(args.out_dir, f"{name}_{lang}.jsonl")
            if os.path.exists(out) and not args.overwrite:
                print(f"  {name} {lang}: exists — skipping"); continue
            try:
                rows = fetch_mgsm(lang) if bench == "mgsm" else fetch_msvamp(lang)
            except Exception as e:
                print(f"  {name} {lang}: FAILED — {type(e).__name__}: {e}"); continue
            if not rows:
                print(f"  {name} {lang}: 0 rows — inspect the source"); continue
            tmp = out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, out)
            print(f"  {name} {lang}: {len(rows)} rows -> {out}")
            print(f"      {json.dumps(rows[0], ensure_ascii=False)[:160]}")


if __name__ == "__main__":
    main()
