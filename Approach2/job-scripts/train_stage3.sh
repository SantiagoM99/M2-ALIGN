#!/bin/bash
#SBATCH --job-name=a2_stage3_vqa
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_stage3_%j.log

# Approach 2, Stage 3: joint VQA training (both mappings).
# Reuses Stage3's VQA data (Stage3/load_vqa_data.py + GQA images).
# Submit from the repo root: sbatch Approach2/job-scripts/train_stage3.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Stage3/data/stage3b/bengali.jsonl}"
IMAGES_DIR="${IMAGES_DIR:-$PROJECT_ROOT/Stage3/data/gqa/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$A2/outputs/stage3}"
STAGE1_CKPT="${STAGE1_CKPT:-$A2/outputs/stage1/mapping/pytorch_model.bin}"
STAGE2_CKPT="${STAGE2_CKPT:-$A2/outputs/stage2/mapping/pytorch_model.bin}"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi || true

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/21.0.0

echo "=== Activate virtual environment ==="
source "$SCRATCH/venvs/m2-align/bin/activate"
python -V

echo "=== Hugging Face cache/offline config ==="
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -f "$PROJECT_ROOT/.tokens" ]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.tokens"
fi

for f in "$STAGE1_CKPT" "$STAGE2_CKPT" "$DATA_PATH"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f"
    exit 1
  fi
done

RESUME_ARGS=()
TRAINING_STATE="$OUTPUT_DIR/training_state.pt"
if [ -f "$TRAINING_STATE" ]; then
  echo "Found existing training_state.pt — resuming: $TRAINING_STATE"
  RESUME_ARGS=(--resume-from-checkpoint "$TRAINING_STATE")
fi

cd "$A2"
python -u train_stage3_vqa.py \
  --data-path   "$DATA_PATH" \
  --images-dir  "$IMAGES_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --stage1-ckpt "$STAGE1_CKPT" \
  --stage2-ckpt "$STAGE2_CKPT" \
  --mt-path     "$MT_PATH" \
  --vis-path    "$VIS_PATH" \
  --llm-path    "$LLM_PATH" \
  --epochs      1 \
  --lr          2e-5 \
  --train-batch-size 2 \
  --eval-batch-size  2 \
  --grad-accum  16 \
  --max-gen-len 64 \
  --save-steps  200 \
  --use-wandb \
  --wandb-mode     offline \
  --wandb-project  m2-align \
  --wandb-run-name "${RUN_NAME:-a2-stage3-vqa}" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
APP_LOG=$(ls -t "$A2"/logs/a2_stage3_vqa_*.log 2>/dev/null | head -1)
if [ -n "$APP_LOG" ]; then
  grep -E "Epoch [0-9]+ \| val_loss" "$APP_LOG" \
    > "$RESULTS_DIR/stage3_curve_${SLURM_JOB_ID:-manual}.txt" || true
fi
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage3 curve (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done ==="
date
