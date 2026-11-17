#!/bin/bash
# Two more seeds of the pooled cultural arm, so its gain stops resting on one
# training run.
#
# Registered before submission (DESIGN 2026-11-17). Every cultural number in the
# project comes from a single checkpoint, and CLAUDE.md requires three seeds for
# a recipe claim -- which is what "swap half the rows for cultural ones" is. The
# seed 42 arm is already trained and scored; this adds 13 and 7.
#
# **Stages 1 and 2 are shared and are NOT retrained.** Her stage-3 trainer is
# the only one that takes --seed, and reusing one stage-2 checkpoint is the
# scope the claim needs: the spread this measures is the intervention's, not the
# whole pipeline's. Say so wherever the SD is reported -- a cluster CI over
# these three is conditional on that shared warm start.
#
# Prediction: the three-seed mean stays above the control by a margin whose
# lower bound clears zero, and the SD across seeds is small next to the +7.26
# point estimate. Refutation: an SD comparable to the effect, which would mean
# the single-seed number was a draw from a wide distribution and no cultural
# claim in the paper survives without many more runs.
#
#   bash Approach2/job-scripts/a1_cg_pooled_seeds.sh
#
# Env: SEEDS, DONORS, MATCHED_PIXELS, DT
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
DT="${DT:-/scratch/santimn/datatransfer}"
O="$PWD/Approach2/outputs"
SEEDS="${SEEDS:-13 7}"
DONORS="${DONORS:-bn zh id pt ru ko}"
MATCHED_PIXELS="${MATCHED_PIXELS:-602112}"
ALL="bn zh id pt ru ko jv mn ga si"

[ -f "$O/a1_pooled_stage2/pytorch_model.bin" ] || {
  echo "ERROR: no pooled stage 2 at $O/a1_pooled_stage2;"
  echo "  every seed warm-starts from the same one, by design."
  exit 1
}
for L in $DONORS; do
  f="$DT/Stage3/data/${L}_cg50.jsonl"
  [ -s "$f" ] || { echo "ERROR: missing donor pool $f"; exit 1; }
done

for S in $SEEDS; do
  OUT="$O/a1_pooled_cg_s${S}_stage3"
  if [ -f "$OUT/pytorch_model.bin" ]; then
    echo "seed $S: checkpoint already present, skipping training -> $OUT"
    TRAIN=""
  else
    TRAIN=$(SEED="$S" POOLED=1 ARM_TAG=cg CULTURAL_LANGS="$DONORS" STAGE=3 \
      sbatch --job-name="a1cgs$S" --time=24:00:00 \
      Approach2/job-scripts/a1_train.sh | awk '{print $NF}')
    [ -n "$TRAIN" ] || { echo "seed $S: training did not submit"; exit 1; }
    echo "seed $S stage 3: $TRAIN  (1 GPU, 24 h) -> $OUT"
  fi

  DEP=()
  [ -n "$TRAIN" ] && DEP=(--dependency=afterok:"$TRAIN")
  EV=$(CKPT="$OUT/pytorch_model.bin" BENCHMARK=cvqa LANGS="$ALL" \
    VISUAL_PIXELS="$MATCHED_PIXELS" TAG="a1_pooled_cg_s${S}_matched" \
    sbatch --job-name="a1cgs${S}ev" ${DEP[@]+"${DEP[@]}"} --time=12:00:00 \
    Approach2/job-scripts/a1_eval.sh | awk '{print $NF}')
  echo "seed $S eval:    ${EV:-DID NOT SUBMIT}  (tag a1_pooled_cg_s${S}_matched)"
done

echo
echo "Seed 42 is the arm already scored as a1_pooled_cg_matched."
echo "Read each seed against the control with arch_compare.py --arm ctrl=... --arm cg=...;"
echo "the across-seed mean/SD/range still needs its own reader in analysis/."
echo "Watch with: squeue -u \$USER"
