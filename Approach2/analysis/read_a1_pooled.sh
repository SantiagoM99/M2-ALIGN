#!/bin/bash
# Read the registered a1-vs-v4 contrast on CVQA (2026-10-02 prediction) for a
# pooled a1 arm. A launcher, not logic: the arch_compare invocation is five
# template arguments long and the terminal splits it on paste, which is how
# job 22343575 died.
#
#   bash Approach2/analysis/read_a1_pooled.sh                 # tag a1_pooled_matched
#   bash Approach2/analysis/read_a1_pooled.sh a1_pooled_replay_matched
#
# Env: LANGS (default her six), BENCH (cvqa|xgqa), OUT.
set -uo pipefail

TAG="${1:-a1_pooled_matched}"
BENCH="${BENCH:-cvqa}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../results" || exit 1

if [ "$BENCH" = xgqa ]; then
  LANGS="${LANGS:-bn de ru zh pt id ko}"
  # xGQA ids do not carry their image, and 12,578 questions sit on 398 images;
  # her files name it vg_image_id, which arch_compare reads as the image map.
  MAP=(--image-map "eval_xgqa_bn_${TAG}.jsonl")
else
  LANGS="${LANGS:-jv mn ga si bn id}"
  MAP=()
fi
OUT="${OUT:-$HERE/../audits/arch_compare_${BENCH}_v4_${TAG}.json}"

python3 "$HERE/arch_compare.py" \
  --arm "v4=eval_{b}_{L}{blind}_v4.jsonl" \
  --arm "a1=eval_{b}_{L}_${TAG}.jsonl" \
  --full-only a1 \
  --benchmark "$BENCH" \
  --langs $LANGS \
  ${MAP[@]+"${MAP[@]}"} \
  --output "$OUT"
