#!/bin/bash
# Read the pooled cultural arm against its control: the six matched panels, the
# four zero-shot targets, and the Chinese China/Singapore hold-out.
#
# Three readings, registered 2026-11-02 before the arm was built:
#   1. replication  -- a clean-item gain with LB5 > 0 on at least four of the
#                      six matched panels; refuted if only bn moves
#   2. culture      -- the China subset gains more than the held-out Singapore
#                      subset, which the arm never trained on
#   3. cost         -- xGQA-bn retention against delta = 1.0
#
#   bash Approach2/analysis/read_cg_pooled.sh
#
# Env: TAG, CTRL, LANGS, COUNTRY_LABELS, OVERLAP_LABELS
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
A="$HERE/.."
cd "$A/results" || exit 1
TAG="${TAG:-a1_pooled_cg_matched}"
CTRL="${CTRL:-a1_pooled_matched}"
LANGS="${LANGS:-bn zh id pt ru ko}"
COUNTRY_LABELS="${COUNTRY_LABELS:-$A/audits/cvqa_subset_labels.json}"
OVERLAP_LABELS="${OVERLAP_LABELS:-$A/audits/cg_overlap_labels.json}"

run () {  # run <name> <langs> <labels-or-empty>
  local name="$1" langs="$2" labels="$3"
  echo "=== $name ==="
  python3 "$HERE/arch_compare.py" \
    --arm "ctrl=eval_{b}_{L}_${CTRL}.jsonl" \
    --arm "cg=eval_{b}_{L}_${TAG}.jsonl" \
    --full-only ctrl --full-only cg \
    --benchmark cvqa --langs $langs \
    ${labels:+--subset-labels "$labels"} \
    --output "$A/audits/arch_compare_cvqa_cgpooled_${name}.json"
  echo
}

# Reading 1, by answer overlap, per matched panel and pooled over the six.
run matched_overlap "$LANGS" "$OVERLAP_LABELS"
for L in $LANGS; do run "matched_$L" "$L" ""; done
# Reading 2: the Chinese panel split by its own two countries.
run zh_by_country zh "$COUNTRY_LABELS"
# Transfer, for the same arm, on languages no cultural row touched.
run transfer "jv mn ga si" ""
