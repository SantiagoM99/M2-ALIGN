#!/bin/bash
# Does CG50's training set already carry CVQA's answers? One short line.
#
# Runs on a login node: the training file lives under $DT and the panels are
# named by the submission itself, so the audit measures exactly what was scored.
#
#   bash Approach2/analysis/read_cg_overlap.sh
#
# Env: DT, TRAINING, SUB, OUT
set -uo pipefail

[ -d Approach2/analysis ] || { echo "run this from the repo root"; exit 1; }
DT="${DT:-/scratch/santimn/datatransfer}"
# Every pool the arm trained on. Exposure for any panel is their union, not the
# panel language's own file: Javanese shares its country with the Indonesian pool.
POOLS="${POOLS:-bn zh id pt ru ko}"
# The pooled arm has no S1 submission, and its scored result files already carry
# id, choices and answer_index -- so the audit reads the universe that was
# actually scored instead of a panel that should match it. TARGETS empty falls
# back to SUB, which is how the per-language arm was audited.
TARGETS="${TARGETS:-bn zh id pt ru ko jv mn ga si}"
ARM_TAG="${ARM_TAG:-a1_pooled_cg_matched}"
SUB="${SUB:-Approach2/outputs/s1_CG50.eval.submission.json}"
OUT="${OUT:-Approach2/audits/cg_entity_overlap_pooled.json}"
# Per-item labels, so read_cg50.sh can split the gain into overlapping and
# clean items: the share bounds the contribution, the split measures it.
LABELS="${LABELS:-Approach2/audits/cg_overlap_labels_pooled.json}"

TRAIN_ARGS=()
for L in $POOLS; do
  f="$DT/Stage3/data/${L}_cg50.jsonl"
  [ -s "$f" ] || { echo "ERROR: missing $f"; exit 1; }
  TRAIN_ARGS+=(--training "$f")
done

PANEL_ARGS=()
if [ -n "$TARGETS" ]; then
  for L in $TARGETS; do
    f="Approach2/results/eval_cvqa_${L}_${ARM_TAG}.jsonl"
    [ -s "$f" ] || { echo "ERROR: missing scored panel $f"; exit 1; }
    PANEL_ARGS+=(--panel "$L=$f")
  done
else
  PANEL_ARGS+=(--submission "$SUB")
fi

python3 Approach2/analysis/cg_entity_overlap.py \
  "${TRAIN_ARGS[@]}" \
  "${PANEL_ARGS[@]}" \
  --output "$OUT" \
  --emit-labels "$LABELS"
