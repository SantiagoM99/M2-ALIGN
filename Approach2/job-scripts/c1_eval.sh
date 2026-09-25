#!/bin/bash
# Plan, submit and queue the 30-cell evaluation of one C1 seed replicate.
#
# A replicate needs only C1's own cells: C4 is untrained and already evaluated in
# the 150-cell Block C grid, so `--arms C1 --arm-seed <seed>` is 30 cells and
# about 3 h instead of 15. The evaluator's own seed stays 13 inside s1_plan.py,
# because it generates the shuffled-image maps and the replicate must reuse the
# maps the reference cells were scored with; analysis/c1_seed_spread.py refuses a
# replicate whose map hashes differ, so a mistake here fails loudly later.
#
# Run from the repo root on a LOGIN node, with the modules and venv active:
#   SEED=14 bash Approach2/job-scripts/c1_eval.sh
#
# Env: SEED (required), DT (required), GATE, TIME
set -uo pipefail

SEED="${SEED:?set SEED, e.g. SEED=14}"
DT="${DT:?set DT=/scratch/santimn/datatransfer}"
A2="Approach2"
O="$PWD/$A2/outputs"
GATE="${GATE:-$A2/audits/s1_A_analysis.json}"
TIME="${TIME:-12:00:00}"
PLAN="evaluation/s1_C1_seed${SEED}.plan.json"
SUB="$O/s1_C1_seed${SEED}.eval.submission.json"

[ -d "$A2" ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first (activate only, install nothing)"
  exit 1
}
[ -f "$GATE" ] || { echo "no Block A report at $GATE"; exit 1; }
[ -f "$O/s1_C1_seed${SEED}/complete.json" ] || {
  echo "seed $SEED has not finished training: $O/s1_C1_seed${SEED}/complete.json is missing."
  echo "The best checkpoint appears after epoch 1, so its existence is not completion."
  exit 1
}
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== planning seed $SEED ==="
python "$A2/s1_plan.py" --block C --arms C1 --arm-seed "$SEED" \
  --panels evaluation/s1_panels.json \
  --checkpoints "$O" \
  --results "$O/s1_C1_seed${SEED}_eval" \
  --block-a-report "$GATE" \
  --output "$PLAN" || exit 1

# Planning regenerates the shuffle-map hash registries, and s1_submit.py freezes
# the code sha of a clean tree. If planning changed a registry, that change has
# to be committed before the submission is frozen, or the allocation would run
# against a tree that no commit describes. Planning is deterministic, so
# re-running this script after committing produces the same plan.
if ! git diff --quiet || ! git diff --cached --quiet || \
   [ -n "$(git ls-files --others --exclude-standard "$A2/shuffle" 2>/dev/null)" ]; then
  echo
  echo "The tree is not clean. Commit the shuffle-map registries, then re-run this script:"
  echo "  git add $A2/shuffle && git commit -m 'shuffle: registries for C1 seed $SEED eval' && git push"
  exit 2
fi

echo "=== submitting seed $SEED ==="
python "$A2/s1_submit.py" --plan "$PLAN" --submission "$SUB" || exit 1
sbatch --time="$TIME" "$A2/job-scripts/s1_eval.sh" "$SUB"
