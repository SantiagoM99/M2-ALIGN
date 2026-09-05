"""Fetch GQA test-dev questions in English, keyed by question id.

xGQA is GQA's test-dev split translated into 7 languages, and our per-language
files keep GQA's `id`, so the English source joins on that id exactly — no
ordering assumptions. This writes {"question_id", "question", "answer"} rows;
`build_xgqa_english.py` turns them into an evaluation file matching our schema.

The English arm is the denominator for every "drop of X points" claim against
xGQA's reported 38 (Pfeiffer et al., Findings of ACL 2022).

Standard library only, so it runs on a login node or a laptop. Paginates the
HF datasets-server (100 rows per request), which returns image URLs rather
than bytes, so the download stays small.

    python3 Approach2/fetch_gqa_english.py --out gqa_testdev_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "theblackcat102/gqa-testdev-balanced"
ROWS = "https://datasets-server.huggingface.co/rows"


def get(url: str, timeout: int = 60, retries: int = 6) -> dict:
    """GET with backoff. The rows API throttles at a few thousand rows, so a
    long paginated pull needs to wait rather than give up."""
    delay = 5.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "m2-align/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            print(f"\n  HTTP {e.code}; waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def fetch_split(split: str, out: dict[str, dict] | None = None) -> dict[str, dict]:
    if out is None:
        out = {}
    offset, total = 0, None
    while True:
        q = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
                                    "split": split, "offset": offset, "length": 100})
        page = get(f"{ROWS}?{q}")
        batch = page.get("rows", [])
        if not batch:
            break
        total = page.get("num_rows_total", total)
        for item in batch:
            r = item["row"]
            qid = str(r.get("question_id", "")).strip()
            if qid:
                out[qid] = {"question": (r.get("question") or "").strip(),
                            "answer": (r.get("answer") or "").strip(),
                            "image_id": (r.get("image_id") or "").strip()}
        offset += len(batch)
        print(f"  {split}: {offset}/{total}", end="\r", flush=True)
        if total and offset >= total:
            break
    print(f"  {split}: {len(out)} questions{' ' * 20}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GQA test-dev English questions by id.")
    ap.add_argument("--out", default="gqa_testdev_questions.jsonl")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    if os.path.exists(args.out):   # resume a partial pull
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                merged[r["question_id"]] = {k: v for k, v in r.items() if k != "question_id"}
        print(f"resuming with {len(merged)} already fetched")
    for split in args.splits:
        try:
            fetch_split(split, merged)
        except Exception as e:
            print(f"\n  {split}: stopped — {type(e).__name__}: {e} (keeping {len(merged)})")
    if not merged:
        raise SystemExit("nothing fetched")

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for qid, r in merged.items():
            f.write(json.dumps({"question_id": qid, **r}, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"wrote {len(merged)} questions -> {args.out}")


if __name__ == "__main__":
    main()
