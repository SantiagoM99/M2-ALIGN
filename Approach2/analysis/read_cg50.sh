#!/bin/bash
# Read CG50 against C1. One short line: the invocation is five paths long and
# the terminal splits it on paste.
#
# Runs where the cells are -- the login node, against the submission, stdlib
# only, no GPU. Writes the report harvest then commits.
#
#   bash Approach2/analysis/read_cg50.sh
#
# Env: SUB, WITH_Q, QBLIND, OUT, BOOT
set -uo pipefail

[ -d Approach2/analysis ] || { echo "run this from the repo root"; exit 1; }
A2=Approach2
SUB="${SUB:-$A2/outputs/s1_CG50.eval.submission.json}"
WITH_Q="${WITH_Q:-$A2/results/qwen_cvqa_bn.jsonl}"
QBLIND="${QBLIND:-$A2/results/qwen_cvqa_bn_QBLIND.jsonl}"
OUT="${OUT:-$A2/audits/s1_CG50_analysis.json}"

python3 "$A2/analysis/cg50.py" \
  --submission "$SUB" \
  --reference-with "$WITH_Q" \
  --reference-blind "$QBLIND" \
  --output "$OUT" \
  ${BOOT:+--boot "$BOOT"}
