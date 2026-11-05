#!/bin/bash
# How complete is the local CVQA image cache, per language?
#
# Her `load_cvqa` takes each row's image from the dataset's own embedded `image`
# feature and saves it to <image-cache-dir>/<id>.jpg -- deliberately, because
# `Image Source` is a sentinel string or a page URL for most rows. So a missing
# file is not a missing download: the evaluator reaches for the dataset, and on
# a compute node with no network that hangs instead of failing. Jobs 25535463
# and 25535464 died at the 12 h limit with ko producing no rows in 11 hours.
#
#   bash Approach2/job-scripts/cvqa_cache_report.sh
#
# Env: LANGS, DT
set -uo pipefail

DT="${DT:-/scratch/santimn/datatransfer}"
LANGS="${LANGS:-bn zh id pt ru ko jv mn ga si}"
D="$DT/Stage3/data/cvqa"

printf '%-4s %7s %7s %7s  %s\n' lang rows cached missing status
for L in $LANGS; do
  f="$D/$L.jsonl"
  [ -s "$f" ] || { printf '%-4s %7s\n' "$L" "NO PANEL"; continue; }
  python - "$L" "$f" "$D/images" <<'PY'
import json, os, sys
lang, path, images = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()]
have = sum(any(os.path.exists(os.path.join(images, f"{i}{e}"))
               for e in (".jpg", ".jpeg", ".png", ".webp")) for i in ids)
miss = len(ids) - have
print(f"{lang:<4} {len(ids):7d} {have:7d} {miss:7d}  "
      f"{'complete' if miss == 0 else '*** WILL HANG ***'}")
PY
done
echo
echo "A language with missing images cannot be evaluated on a compute node."
echo "Populate its cache on a LOGIN node first, then resubmit: a1_eval.sh skips"
echo "non-empty outputs, so it fills the gaps."
