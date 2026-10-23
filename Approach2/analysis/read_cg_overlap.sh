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
TRAINING="${TRAINING:-$DT/Stage3/data/bn_cg50.jsonl}"
SUB="${SUB:-Approach2/outputs/s1_CG50.eval.submission.json}"
OUT="${OUT:-Approach2/audits/cg_entity_overlap.json}"

python3 Approach2/analysis/cg_entity_overlap.py \
  --training "$TRAINING" \
  --submission "$SUB" \
  --output "$OUT"
