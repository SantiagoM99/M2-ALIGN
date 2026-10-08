#!/bin/bash
# Plan, submit and queue the 30-cell evaluation of the CG50 arm.
#
# CG50 is not an S1 arm, but its registered reading is a contrast against C1, so
# it has to be scored on C1's grid: the same six panels (CVQA jv/mn/ga/si/bn and
# xGQA-bn), the same five conditions (correct, three seeded shuffles, gray) and
# above all the SAME shuffle maps. The evaluator's own seed stays 13 inside
# s1_plan.py for that reason -- it generates the maps, and a contrast scored
# against regenerated maps is no longer paired.
#
# This mirrors c1_eval.sh, which does the same job for a C1 seed replicate; the
# only difference is --extra-arm, which adds a checkpoint trained outside the
# frozen contract without touching the five arms inside it.
#
# Run from the repo root on a LOGIN node, with the modules and venv active:
#   bash Approach2/job-scripts/cg50_eval.sh
#
# Env: ARM (default CG50), CKPT_DIR (default cg50_seed13), DT, GATE, TIME
set -uo pipefail

ARM="${ARM:-CG50}"
CKPT_DIR="${CKPT_DIR:-cg50_seed13}"
DT="${DT:-/scratch/santimn/datatransfer}"
A2="Approach2"
O="$PWD/$A2/outputs"
GATE="${GATE:-$A2/audits/s1_A_analysis.json}"
TIME="${TIME:-12:00:00}"
PLAN="evaluation/s1_${ARM}.plan.json"
SUB="$O/s1_${ARM}.eval.submission.json"

[ -d "$A2" ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first (activate only, install nothing)"
  exit 1
}
[ -f "$GATE" ] || { echo "no Block A report at $GATE"; exit 1; }
[ -f "$O/$CKPT_DIR/complete.json" ] || {
  echo "$CKPT_DIR has not finished training: $O/$CKPT_DIR/complete.json is missing."
  echo "The best checkpoint appears after epoch 1, so its existence is not completion."
  exit 1
}
# The arm is only interpretable if its training data was the mixed file; a run
# fed the unmixed rows would be a C1 replicate under another name.
[ -f "$DT/Stage3/data/bn_cg50.jsonl" ] || {
  echo "no $DT/Stage3/data/bn_cg50.jsonl: build it before reading the arm"; exit 1; }
[ -f "$A2/audits/cg_leakage.json" ] || {
  echo "no $A2/audits/cg_leakage.json: the leakage check is required before the run"; exit 1; }

export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== planning $ARM from $CKPT_DIR ==="
python "$A2/s1_plan.py" --block C --arms C1 --extra-arm "$ARM=$CKPT_DIR" \
  --panels evaluation/s1_panels.json \
  --checkpoints "$O" \
  --results "$O/s1_${ARM}_eval" \
  --block-a-report "$GATE" \
  --output "$PLAN" || exit 1

# Planning regenerates the shuffle-map hash registries, and s1_submit.py freezes
# the code sha of a clean tree, so a registry change has to be committed before
# the submission is frozen. Planning is deterministic: re-running after the
# commit produces the same plan.
if ! git diff --quiet || ! git diff --cached --quiet || \
   [ -n "$(git ls-files --others --exclude-standard "$A2/shuffle" 2>/dev/null)" ]; then
  echo
  echo "The tree is not clean. Commit the shuffle-map registries, then re-run this script:"
  echo "  git add $A2/shuffle && git commit -m 'shuffle: registries for $ARM eval' && git push"
  exit 2
fi

echo "=== submitting $ARM ==="
python "$A2/s1_submit.py" --plan "$PLAN" --submission "$SUB" || exit 1
sbatch --time="$TIME" "$A2/job-scripts/s1_eval.sh" "$SUB"
