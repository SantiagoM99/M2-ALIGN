"""Fetch FLORES-200 dev as a multi-way parallel file over our eleven languages.

Why FLORES and not the stage-1 data
-----------------------------------
X2 failed partly because it asked a target-only question. X1 showed transfer
is a property of the source-target PAIR, so the measure has to be pairwise:
how close is language T's prefix to language S's? That needs the SAME
sentences in both languages, and the stage-1 files cannot provide it — each
`<Language>_to_English.jsonl` is a different L-to-English corpus, so two
languages share no sentence. FLORES-200 is multi-way parallel by
construction: line i of every language file is the same sentence.

It is also the right choice specifically here: FLORES is NLLB's own
evaluation set and its files are named by NLLB language code, which is the
tag this pipeline already threads through every row.

Source
------
The Hugging Face mirrors (`facebook/flores`, `openlanguagedata/flores_plus`)
are gated and answer 401 without a token, and tokens do not belong in repo
scripts. This pulls the canonical tarball from the NLLB release instead,
which needs no authentication:

    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz   (~25 MB)

STANDARD LIBRARY ONLY (urllib + tarfile + json): runs on an Alliance LOGIN
node. Compute nodes have no network. Members are read with `extractfile()`
rather than `extractall()`, so a hostile path in the archive cannot escape
the output directory.

    python3 Approach2/fetch_flores.py --out-dir evaluation

Writes `<out-dir>/flores_dev.jsonl`, one row per sentence:
    {"idx": 0, "ben_Beng": "...", "deu_Latn": "...", ...}
Nothing here goes into git: evaluation/ is gitignored (benchmark data stays
out of the repository).
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import urllib.error
import urllib.request

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# The eleven languages this project covers, as NLLB tags — the same map used
# by evaluate_vqa.py:ISO_TO_NLLB and train_stage1_text.py:LANGS_MAP_NLLB.
TAGS = {
    "bn": "ben_Beng", "de": "deu_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn",
    "mn": "khk_Cyrl", "si": "sin_Sinh", "ga": "gle_Latn", "en": "eng_Latn",
}


def download(url: str, dest: str, retries: int = 6) -> None:
    """Download with exponential backoff. Skips a file that is already there."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"already downloaded: {dest} ({os.path.getsize(dest)} bytes)")
        return
    delay = 5
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "m2-align/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"downloaded {dest} ({os.path.getsize(dest)} bytes)")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries:
                raise
            print(f"  attempt {attempt} failed ({e}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="evaluation")
    ap.add_argument("--split", default="dev", choices=("dev", "devtest"))
    ap.add_argument("--tarball", default="", help="use an already-downloaded tarball")
    ap.add_argument("--langs", default="", help="ISO codes; default = all eleven + en")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tarball = args.tarball or os.path.join(args.out_dir, "flores200_dataset.tar.gz")
    download(URL, tarball)

    wanted = set(args.langs.split()) if args.langs.strip() else set(TAGS)
    tags = {iso: tag for iso, tag in TAGS.items() if iso in wanted}

    # Map tag -> the member whose basename is "<tag>.<split>", wherever it sits.
    by_tag: dict[str, list[str]] = {}
    with tarfile.open(tarball, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        for iso, tag in tags.items():
            hit = next((n for n in members
                        if os.path.basename(n) == f"{tag}.{args.split}"), None)
            if hit is None:
                print(f"!! {iso} ({tag}): no {tag}.{args.split} in the archive — skipped")
                continue
            data = tf.extractfile(members[hit])
            if data is None:
                print(f"!! {iso} ({tag}): unreadable member — skipped")
                continue
            by_tag[tag] = data.read().decode("utf-8").splitlines()
            print(f"{iso:>3} ({tag}): {len(by_tag[tag])} lines")

    if not by_tag:
        raise SystemExit("nothing extracted — check the archive layout")

    lengths = {len(v) for v in by_tag.values()}
    if len(lengths) != 1:
        # FLORES is line-aligned; unequal lengths mean the alignment assumption
        # this whole measurement rests on is broken, so refuse rather than
        # silently truncate to the shortest.
        raise SystemExit(f"line counts disagree across languages: {lengths}. "
                         "FLORES must be line-aligned; refusing to write.")
    n = lengths.pop()

    out = os.path.join(args.out_dir, f"flores_{args.split}.jsonl")
    kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {"idx": i}
            row.update({tag: by_tag[tag][i].strip() for tag in by_tag})
            if any(not v for k, v in row.items() if k != "idx"):
                continue          # a blank line in any language breaks the pairing
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"\nwrote {out}: {kept}/{n} rows x {len(by_tag)} languages")
    if kept < n:
        print(f"  ({n - kept} rows dropped for a blank line in at least one language)")


if __name__ == "__main__":
    main()
