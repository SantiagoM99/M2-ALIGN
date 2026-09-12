#!/bin/bash
# D12 control: the INDEPENDENT arm retrained with the current stage-3 trainer.
#
# The pooled arm (round vj) was trained on 2026-09-11 with the trainer rewritten
# on 09-09, under transformers 5.x. The v4 checkpoints it was meant to be
# compared with predate both, so v4 against vj is not a single variable
# (DESIGN.md 2026-09-12). This round retrains exactly those per-language
# checkpoints, each from its own stage1_<L>, with the same stage 2, recipe,
# replay and epochs that launch_joint.sh gave vj. v4r and vj then differ in one
# thing only: whether the text mapping is shared across languages.
#
#   [1] a2_s3_all_v4r   stage 3 per language from its own stage1_<L>
#   [2] a2_ev_all_v4r   CVQA, xGQA, MGSM/MSVAMP under EVAL_TAG=tf5, after [1] succeeds
#
# The vj arm needs the same tagged evaluation (ROUND=vj EVAL_TAG=tf5 through
# evaluate_all.sh); the comparison is then
#   cd Approach2/results && python3 ../analysis/gap_report.py v4r_tf5 vj_tf5
#
# Usage (repo root, Rorqual):
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_v4r.sh
# Idempotent: re-running submits only what is missing.

set -euo pipefail

DT="${DT:?Set DT, e.g. DT=/scratch/santimn/datatransfer}"
ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root."; exit 1; }
mkdir -p "$A2/logs"

# sbatch runs with --export=ALL. An exported STAGE1_CKPT left in the shell from
# launching the joint round would pin the shared mapping for every language and
# silently turn this control into a second copy of vj.
unset STAGE1_CKPT

ROUND="${ROUND:-v4r}"
LANGS="${LANGS:-jv mn ga de ru zh}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"
S3_EPOCHS="${S3_EPOCHS:-2}"
S3_TIME="${S3_TIME:-48:00:00}"
EV_TIME="${EV_TIME:-12:00:00}"
EVAL_TAG="${EVAL_TAG:-tf5}"
S2_CKPT="${S2_CKPT:-$A2/outputs/stage2_dc_llava/mapping/pytorch_model.bin}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"

echo "=== Sanity checks ==="
missing=0
[ -f "$S2_CKPT" ] || { echo "MISSING: stage-2 checkpoint $S2_CKPT"; missing=1; }
[ -d "$GQA_IMAGES" ] || { echo "MISSING: GQA images $GQA_IMAGES"; missing=1; }
for L in $LANGS; do
  s1="$A2/outputs/stage1_$L/mapping/pytorch_model.bin"
  [ "$L" = bn ] && s1="$A2/outputs/stage1/mapping/pytorch_model.bin"
  # Every language is required: the registered D12 prediction is differential,
  # so a verdict read on a subset would be a selection of the controls.
  [ -f "$s1" ] || { echo "MISSING: own stage-1 mapping for $L ($s1)"; missing=1; }
  case "$L" in
    bn|de|ru|zh|pt|id|ko|jv|mn|si|ga) ;;
    *) echo "UNKNOWN language code $L"; missing=1 ;;
  esac
done
[ "$missing" = 0 ] || exit 1
echo "ROUND=$ROUND LANGS=$LANGS S3_EPOCHS=$S3_EPOCHS EVAL_TAG=$EVAL_TAG"
echo "stage-2 checkpoint: $S2_CKPT"

queued () { squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -qx "$1"; }

echo "=== [1] Stage 3, own stage-1 mapping per language ==="
S3_JOB=""
done_all=1
for L in $LANGS; do
  [ -f "$A2/outputs/stage3_${L}_${ROUND}/complete.json" ] || done_all=0
done
if [ "$done_all" = 1 ]; then
  echo "  all $ROUND checkpoints carry a completion marker -> skipping"
elif queued "a2_s3_all_$ROUND"; then
  echo "  a2_s3_all_$ROUND already queued or running -> not resubmitting"; exit 0
else
  S3_JOB=$(DT="$DT" ROUND="$ROUND" LANGS="$LANGS" STAGE2_CKPT="$S2_CKPT" \
           VIS_LAYERS="$VIS_LAYERS" REPLAY=1 S3_EPOCHS="$S3_EPOCHS" GQA_IMAGES="$GQA_IMAGES" \
           sbatch --parsable --export=ALL --job-name="a2_s3_all_$ROUND" --time="$S3_TIME" \
           "$JS/train_stage3_all.sh")
  S3_JOB="${S3_JOB%%;*}"
  echo "  a2_s3_all_$ROUND: submitted job $S3_JOB"
fi

echo "=== [2] Evaluations under EVAL_TAG=$EVAL_TAG ==="
if queued "a2_ev_all_$ROUND"; then
  echo "  a2_ev_all_$ROUND already queued or running -> not resubmitting"; exit 0
fi
DEP=()
# afterok: the verdict needs all six languages, and train_stage3_all.sh exits
# non-zero when any of them fails, so evaluations never run on a partial round.
[ -n "$S3_JOB" ] && DEP=(--dependency="afterok:$S3_JOB")
EV_JOB=$(DT="$DT" ROUND="$ROUND" EVAL_TAG="$EVAL_TAG" VIS_LAYERS="$VIS_LAYERS" \
         GQA_IMAGES="$GQA_IMAGES" BENCHES="cvqa xgqa" \
         XGQA_LANGS="de ru zh" CVQA_LANGS="jv mn ga ru zh" TEXT_LANGS="de ru zh" \
         sbatch --parsable --export=ALL --job-name="a2_ev_all_$ROUND" --time="$EV_TIME" \
         ${DEP[@]+"${DEP[@]}"} "$JS/evaluate_all.sh")
echo "  a2_ev_all_$ROUND: submitted job ${EV_JOB%%;*}${S3_JOB:+ (afterok:$S3_JOB)}"
echo
echo "=== Done. Also run ROUND=vj EVAL_TAG=$EVAL_TAG through evaluate_all.sh if it has not run. ==="
