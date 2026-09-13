#!/bin/bash
#SBATCH --job-name=a2_bisect_trainer
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a2_bisect_trainer_%j.log

# Old trainer, current environment (DESIGN.md 2026-09-13).
#
# Checkpoints trained by the stage-3 trainer rewritten on 09-09 lose about 23
# MGSM and 10 MSVAMP points of reasoning, and re-evaluating stage3_bn_dcl with
# the current code reproduces it, so the loss is in training. Two things changed
# at once for training: the trainer code, and the environment it runs in
# (torch 2.13, transformers 5.13.1). This job separates them. It trains Bengali
# stage 3 with the GSM8K replay using the trainer as it was at TRAINER_SHA
# (default 287bae9, the last commit before the rewrite), run from a detached
# worktree of that commit so its model.py and common.py come with it, in the
# current venv, with exactly the arguments of the stage3_bn_gsm8k control. It
# then evaluates MGSM and MSVAMP with the CURRENT evaluation code, which
# reproduced dcl (job 21009320).
#   near dcl   (62.0 MGSM / 64.5 MSVAMP): the rewrite causes the loss
#   near gsm8k (39.2 MGSM / 54.4 MSVAMP): the training environment causes it
#
# Env: DT (required), TRAINER_SHA (287bae9).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
TRAINER_SHA="${TRAINER_SHA:-287bae9}"
FULL_SHA=$(git -C "$PROJECT_ROOT" rev-parse --verify "$TRAINER_SHA^{commit}") || { echo "ERROR: unknown commit $TRAINER_SHA"; exit 1; }
TAG="gsm8k_old${FULL_SHA:0:7}"
S3_OUT="$A2/outputs/stage3_bn_$TAG"
WT_ROOT="${S1_WORKTREE_ROOT:-$SCRATCH/s1_worktrees}"
WT="$WT_ROOT/trainer_$FULL_SHA"

MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
S2_CKPT="$A2/outputs/stage2_dc_llava/mapping/pytorch_model.bin"
MATH_REPLAY="$A2/data/math_replay_bn.jsonl"
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"
GQA_IMAGES="$DT/Stage3/data/gqa/images"
DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
[ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"

echo "=== Job info ==="; date; hostname
echo "TRAINER_SHA=$FULL_SHA TAG=$TAG"
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

for f in "$STAGE1_CKPT" "$S2_CKPT" "$MATH_REPLAY" "$TRANS_REPLAY" "$DATA"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

mkdir -p "$WT_ROOT"
git -C "$PROJECT_ROOT" worktree prune
flock "$WT_ROOT/.lock" bash -c '
  [ -e "$1/Approach2/train_stage3_vqa.py" ] || git -C "$2" worktree add --detach "$1" "$3"
' _ "$WT" "$PROJECT_ROOT" "$FULL_SHA"
[ -e "$WT/Approach2/train_stage3_vqa.py" ] || { echo "ERROR: no worktree at $WT"; exit 1; }
echo "training code from $WT"

echo "=== [1] Stage 3 bn, GSM8K replay, trainer at $FULL_SHA ==="
# The pre-rewrite trainer has no completion marker, and its best checkpoint is
# written after the first improving epoch, so this job marks completion itself.
if [ -f "$S3_OUT/trained.ok" ]; then
  echo "training complete -> skipping"
else
  RESUME_ARGS=()
  [ -f "$S3_OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$S3_OUT/training_state.pt")
  ( cd "$WT/Approach2" && python -u train_stage3_vqa.py \
      --data-path "$DATA" --images-dir "$GQA_IMAGES" --output-dir "$S3_OUT" \
      --stage1-ckpt "$STAGE1_CKPT" --stage2-ckpt "$S2_CKPT" \
      --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
      --vis-layers 9,18,-1 \
      --replay-data "$MATH_REPLAY,$TRANS_REPLAY" --replay-every 3 \
      --epochs 2 --lr 2e-5 \
      --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 \
      --max-gen-len 64 --save-steps 200 \
      --use-wandb --wandb-mode offline --wandb-project m2-align \
      --wandb-run-name "a2-stage3-bn-$TAG" --local-files-only \
      ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ) || { echo "ERROR: training failed"; exit 1; }
  [ -f "$S3_OUT/mapping/pytorch_model.bin" ] || { echo "ERROR: no checkpoint written"; exit 1; }
  touch "$S3_OUT/trained.ok"
fi

echo "=== [2] MGSM and MSVAMP with the current evaluation code ==="
CKPT="$S3_OUT/mapping/pytorch_model.bin"
cd "$A2"
FAILED=0
for spec in "mgsm:$PROJECT_ROOT/evaluation/MGSM.jsonl" "msvamp:$PROJECT_ROOT/evaluation/MSVAMP.jsonl"; do
  bench="${spec%%:*}"; data="${spec#*:}"
  out="$S3_OUT/eval_${bench}_bn_stage3_${TAG}.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $bench (summary exists)"; continue; }
  python -u evaluate_text.py --data-path "$data" --benchmark "$bench" \
    --output-path "$out" --ckpt "$CKPT" --nllb-tag ben_Beng \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || { echo "### $bench FAILED"; FAILED=1; }
done

cp "$S3_OUT"/eval_*_bn_stage3_"$TAG".jsonl "$A2/results/" 2>/dev/null || true
cp "$S3_OUT"/eval_*_bn_stage3_"$TAG".jsonl.summary.json "$A2/results/" 2>/dev/null || true
grep -h "val_loss" "$S3_OUT"/*.log 2>/dev/null | tail -4 || true
echo "Harvested into Approach2/results; review and commit by hand."
git -C "$PROJECT_ROOT" status --short Approach2/results | head
echo "=== Done === $(date)"
exit "$FAILED"
