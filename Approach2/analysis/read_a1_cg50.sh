#!/bin/bash
# CG50 on a1 against the a1 control, split by answer overlap. One short line.
#
# The control is the per-language a1-bn arm already scored at 588 tokens, so the
# only difference between the arms is --data-path. Both are read full-only: her
# evaluator has no blind arm, so no grounding statement travels with these.
#
# The split matters for comparability, not decoration: our own +13.28 is the
# clean-item figure, and an all-items contrast is a different quantity.
#
#   bash Approach2/analysis/read_a1_cg50.sh
#
# Env: TAG, LANGS, LABELS, OUT
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../results" || exit 1
TAG="${TAG:-a1_bn_cg50_matched}"
LANGS="${LANGS:-bn jv mn ga si}"
LABELS="${LABELS:-$HERE/../audits/cg_overlap_labels.json}"
OUT="${OUT:-$HERE/../audits/arch_compare_cvqa_a1cg50_vs_a1.json}"

python3 "$HERE/arch_compare.py" \
  --arm "a1=eval_{b}_{L}_matched.jsonl" \
  --arm "a1cg=eval_{b}_{L}_${TAG}.jsonl" \
  --full-only a1 --full-only a1cg \
  --benchmark cvqa \
  --langs $LANGS \
  ${LABELS:+--subset-labels "$LABELS"} \
  --output "$OUT"
