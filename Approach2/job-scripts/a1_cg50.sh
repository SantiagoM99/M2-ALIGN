#!/bin/bash
# CG50 on a1: the same data swap, on the other architecture, in one line.
#
# Registered before submission (DESIGN 2026-10-25). Our stack gained +13.28
# [+6.72, +20.00] on CVQA-bn items whose answer the arm had never seen, with
# grounding up +11.46. In a1 the vision path is Qwen's own frozen tower and the
# only trainable mapping carries the question, so the prediction is a gain
# materially smaller than that -- and a gain as large as ours would refute the
# reading that the effect lives in the trainable vision connector.
#
# Per-language bn, not pooled: replacing half of one file in an eleven-language
# pool changes ~1/24 of the rows, which is too diluted to measure. The control
# is the per-language a1-bn arm already trained and already scored at 588 tokens
# (CVQA-bn 37.76), and `--data-path` is the only difference between them.
#
# Stages 1 and 2 are shared with that control and are NOT retrained.
#
#   bash Approach2/job-scripts/a1_cg50.sh
#
# Env: DATA_PATH, ARM_TAG, LANGS, VISUAL_PIXELS, DT
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
DT="${DT:-/scratch/santimn/datatransfer}"
ARM_TAG="${ARM_TAG:-cg50}"
DATA_PATH="${DATA_PATH:-$DT/Stage3/data/bn_cg50.jsonl}"
LANGS="${LANGS:-bn jv mn ga si}"
VISUAL_PIXELS="${VISUAL_PIXELS:-602112}"
OUTDIR="$PWD/Approach2/outputs/a1_bn_${ARM_TAG}_stage3"

[ -f "$DATA_PATH" ] || { echo "ERROR: no $DATA_PATH"; exit 1; }
[ -f "$PWD/Approach2/outputs/a1_bn_stage2/pytorch_model.bin" ] || {
  echo "ERROR: no per-language stage 2 at Approach2/outputs/a1_bn_stage2;"
  echo "  this arm warm-starts from the control's own stage 2."
  exit 1
}

TRAIN=$(DATA_PATH="$DATA_PATH" ARM_TAG="$ARM_TAG" A1_LANG=bn STAGE=3 \
  sbatch --job-name="a1bn$ARM_TAG" --time=12:00:00 \
  Approach2/job-scripts/a1_train.sh | awk '{print $NF}')
[ -n "$TRAIN" ] || { echo "the training job did not submit"; exit 1; }
echo "stage 3 ($ARM_TAG): $TRAIN  (1 GPU, 12 h) -> $OUTDIR"

EVAL=$(CKPT="$OUTDIR/pytorch_model.bin" BENCHMARK=cvqa LANGS="$LANGS" \
  VISUAL_PIXELS="$VISUAL_PIXELS" TAG="a1_bn_${ARM_TAG}_matched" \
  sbatch --job-name="a1bn${ARM_TAG}ev" --dependency=afterok:"$TRAIN" --time=12:00:00 \
  Approach2/job-scripts/a1_eval.sh | awk '{print $NF}')
echo "eval: $EVAL  (after $TRAIN, langs: $LANGS)"
echo
echo "Control, already scored: eval_cvqa_<L>_matched.jsonl"
echo "Watch with: squeue -u \$USER"
