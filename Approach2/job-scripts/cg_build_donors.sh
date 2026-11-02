#!/bin/bash
# Build one culturally grounded stage-3 pool per donor language, each from its
# own CVQA country. Login node: this downloads from CulturalGround and decodes
# every image it extracts, and compute nodes have no network.
#
# Countries come from audits/cvqa_inventory.json, matched to each language's own
# CVQA subset, and the spellings are the ones cg_probe_donors.sh resolved:
#   bn india | zh china | id indonesia | pt brazil | ru russia | ko south_korea
# Singapore is deliberately absent: it is the held-out culture for Chinese.
# German has no CVQA unit, so it keeps its translated-GQA rows.
#
# Idempotent: a language whose mixed file already exists is skipped, which is
# how bn survives untouched -- its file is the one CG50 trained on and
# rebuilding it would change the sha every manifest of that arm carries.
#
#   bash Approach2/job-scripts/cg_build_donors.sh
#   PAIRS="ko:south_korea" FRACTION=0.4 bash Approach2/job-scripts/cg_build_donors.sh
#
# Env: PAIRS, FRACTION, SAMPLE, PER_IMAGE, SEED, DT
set -uo pipefail

[ -d Approach2 ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first"
  exit 1
}
DT="${DT:-/scratch/santimn/datatransfer}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
PAIRS="${PAIRS:-bn:india zh:china id:indonesia pt:brazil ru:russia ko:south_korea}"
FRACTION="${FRACTION:-0.5}"
SAMPLE="${SAMPLE:-20000}"
PER_IMAGE="${PER_IMAGE:-4}"
SEED="${SEED:-13}"
IMAGES="$DT/Stage3/data/gqa/images"

built=""; skipped=""; failed=""
for pair in $PAIRS; do
  lang="${pair%%:*}"; country="${pair##*:}"
  out="$DT/Stage3/data/${lang}_cg50.jsonl"
  base="$DT/Stage3/data/${lang}.jsonl"
  echo "=== $lang from $country ==="
  if [ -s "$out" ]; then
    echo "--- skip: $out exists ($(wc -l < "$out") rows)"
    skipped="$skipped $lang"; continue
  fi
  [ -f "$base" ] || { echo "ERROR: no $base to mix into"; failed="$failed $lang"; continue; }
  if python Approach2/build_culturalground.py \
      --country "$country" --lang "$lang" \
      --sample "$SAMPLE" --per-image "$PER_IMAGE" --seed "$SEED" \
      --images-dir "$IMAGES" \
      --mix-with "$base" --fraction "$FRACTION" \
      --output "$out"; then
    echo "--- built $out ($(wc -l < "$out") rows)"
    built="$built $lang"
  else
    # mix() fails closed when the cultural pool cannot cover the fraction, and
    # says how many rows it got: lower FRACTION for that language and record the
    # per-language dose rather than letting the total drift.
    rm -f "$out"
    echo "--- FAILED $lang; see the message above for the shortfall"
    failed="$failed $lang"
  fi
  echo
done
echo "================================"
echo "built:  $built"
echo "skipped:$skipped"
echo "failed: $failed"
echo
echo "Then audit the images once, over the whole cg_ namespace:"
echo "  python Approach2/verify_cg_images.py $IMAGES"
