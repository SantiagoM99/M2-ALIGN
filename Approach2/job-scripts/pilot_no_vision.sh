#!/bin/bash
#SBATCH --job-name=a2_no_vision
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_no_vision_%j.log

# H1 control — what does the vision bridge cost the text bridge?
#
# D11 shows that better visual alignment gives better cross-lingual reasoning.
# That is compatible with two opposite readings: "vision costs, and better
# vision costs less" or "vision helps, and better vision helps more". They
# imply different papers, and we have never run the arm that separates them.
#
# This trains the identical stage 3 — same VQA data, same replay, same epochs,
# same lr — with the vision branch switched off. Everything else matches
# stage3_bn_v4, so the only variable is the presence of the visual prefix.
#
# Env: DT (required), S3_EPOCHS (2), REPLAY_EVERY (3).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
S3_EPOCHS="${S3_EPOCHS:-2}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
MATH_REPLAY="$A2/data/math_replay_bn.jsonl"
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"
TAG="novis"
S3_OUT="$A2/outputs/stage3_bn_$TAG"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do [ -d "$d" ] && { MT_PATH="$d"; break; }; done
fi

echo "=== Job info ==="; date; hostname; nvidia-smi || true

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"

export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for f in "$STAGE1_CKPT" "$MATH_REPLAY" "$TRANS_REPLAY"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

cd "$A2"

echo "=== Stage 3 bn, vision branch OFF ==="
if [ -f "$S3_OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> skipping"
else
  RESUME_ARGS=()
  [ -f "$S3_OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$S3_OUT/training_state.pt")
  DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
  [ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"
  python -u train_stage3_vqa.py \
    --data-path "$DATA" --images-dir "$GQA_IMAGES" --output-dir "$S3_OUT" \
    --stage1-ckpt "$STAGE1_CKPT" --no-vision \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" \
    --replay-data "$MATH_REPLAY,$TRANS_REPLAY" --replay-every "$REPLAY_EVERY" \
    --epochs "$S3_EPOCHS" --lr 2e-5 \
    --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 \
    --max-gen-len 64 --save-steps 200 \
    --use-wandb --wandb-mode offline --wandb-project m2-align \
    --wandb-run-name "a2-stage3-bn-$TAG" --local-files-only \
    "${RESUME_ARGS[@]}" || exit 1
fi
CKPT="$S3_OUT/mapping/pytorch_model.bin"

echo "=== Text benchmarks (the point of this arm) ==="
for spec in "mgsm:$PROJECT_ROOT/evaluation/MGSM.jsonl" "msvamp:$PROJECT_ROOT/evaluation/MSVAMP.jsonl"; do
  b="${spec%%:*}"; d="${spec#*:}"
  out="$S3_OUT/eval_${b}_bn_stage3_$TAG.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $b"; continue; }
  python -u evaluate_text.py --data-path "$d" --benchmark "$b" \
    --output-path "$out" --ckpt "$CKPT" --nllb-tag ben_Beng \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || echo "### $b FAILED"
done

echo "=== Reference: stage3_bn_v4 has MGSM 62.0, MSVAMP 64.5 ==="
for s in "$S3_OUT"/eval_*.summary.json; do echo "$s"; cat "$s"; echo; done

mkdir -p "$A2/results"
cp "$S3_OUT"/eval_*.summary.json "$A2/results/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: H1 control, stage 3 without the vision branch (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
