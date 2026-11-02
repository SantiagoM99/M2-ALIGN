#!/bin/bash
# Everything that has to hold before the pooled cultural arm is trained.
# One short line, on a login node with the venv active.
#
# Four checks, each of which has cost this project a job or a claim before:
#   1. composition  -- 15,000 cultural + 15,000 translated GQA per donor, total
#                      fixed, so the contrast measures distribution not volume
#   2. targets      -- jv/mn/ga/si carry NO cultural rows, or the zero-shot
#                      claim is over for them
#   3. decode       -- every cg_ image opens by the trainer's own criterion
#                      (three GPU jobs died on this)
#   4. leakage      -- no training image is an evaluation image, by SHA-256,
#                      now over six countries rather than India alone
#
#   bash Approach2/job-scripts/cg_verify_pools.sh
#
# Env: DONORS, TARGETS, DT
set -uo pipefail

[ -d Approach2 ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first"; exit 1; }
DT="${DT:-/scratch/santimn/datatransfer}"
DONORS="${DONORS:-bn zh id pt ru ko}"
TARGETS="${TARGETS:-jv mn si ga}"
IMAGES="$DT/Stage3/data/gqa/images"
bad=0

echo "=== 1. composition ==="
for L in $DONORS; do
  f="$DT/Stage3/data/${L}_cg50.jsonl"
  if [ ! -s "$f" ]; then echo "MISSING $f"; bad=1; continue; fi
  python - "$L" "$f" <<'PY'
import collections, json, sys
lang, path = sys.argv[1], sys.argv[2]
counts = collections.Counter()
images = set()
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            counts[row.get("source_dataset", "?")] += 1
            images.add(str(row.get("vg_image_id", "")))
total = sum(counts.values())
cultural = counts.get("culturalground_refined_oe", 0)
gqa = counts.get("gqa_balanced_train_translated", 0)
ok = total == 30000 and cultural == 15000 and gqa == 15000
print(f"  {lang}: {total} rows, {cultural} cultural, {gqa} gqa, "
      f"{len(images)} distinct images  {'OK' if ok else '*** OFF ***'}")
sys.exit(0 if ok else 1)
PY
  [ $? -eq 0 ] || bad=1
done

echo
echo "=== 2. zero-shot targets carry no cultural rows ==="
for L in $TARGETS; do
  if [ -e "$DT/Stage3/data/${L}_cg50.jsonl" ]; then
    echo "  *** $L has a cultural pool; training on it ends its zero-shot claim"
    bad=1
  else
    echo "  $L: clean"
  fi
done

echo
echo "=== 3. every cg_ image decodes ==="
python Approach2/verify_cg_images.py "$IMAGES" || bad=1

echo
echo "=== 4. no training image is an evaluation image ==="
python Approach2/analysis/image_leakage.py \
  --train-dir "$IMAGES" --train-prefix cg_ \
  --eval-dir "$DT/Stage3/data/cvqa/images" \
  --output Approach2/audits/cg_leakage_six_donors.json || bad=1

echo
if [ "$bad" = 0 ]; then
  echo "ALL CHECKS PASSED -- safe to train"
else
  echo "SOMETHING FAILED -- do not train until it is explained"
fi
exit $bad
