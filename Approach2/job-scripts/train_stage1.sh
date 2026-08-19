#!/bin/bash
#SBATCH --job-name=a2_stage1_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_stage1_%j.log

# Approach 2, Stage 1: NLLB -> Gemma text mapping on translation pairs.
# Submit from the repo root: sbatch Approach2/job-scripts/train_stage1.sh
# (run `mkdir -p Approach2/logs` once before the first submission)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/Stage1/data}"
LANGUAGES="${LANGUAGES:-Bengali}"
OUTPUT_DIR="${OUTPUT_DIR:-$A2/outputs/stage1}"

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

RESUME_ARGS=()
TRAINING_STATE="$OUTPUT_DIR/training_state.pt"
if [ -f "$TRAINING_STATE" ]; then
  echo "Found existing training_state.pt — resuming: $TRAINING_STATE"
  RESUME_ARGS=(--resume-from-checkpoint "$TRAINING_STATE")
fi

cd "$A2"
python -u train_stage1_text.py \
  --data-dir   "$DATA_DIR" \
  --languages  "$LANGUAGES" \
  --output-dir "$OUTPUT_DIR" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --train-num  100000 \
  --epochs     "${S1_EPOCHS:-3}" \
  --lr         "${S1_LR:-2e-5}" \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode     offline \
  --wandb-project  m2-align \
  --wandb-run-name "${RUN_NAME:-a2-stage1-text-${LANGUAGES}}" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
APP_LOG=$(ls -t "$A2"/logs/a2_stage1_text_*.log 2>/dev/null | head -1)
if [ -n "$APP_LOG" ]; then
  grep -E "Epoch [0-9]+ \| val_loss" "$APP_LOG" \
    > "$RESULTS_DIR/stage1_curve_${SLURM_JOB_ID:-manual}.txt" || true
fi
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage1 curve (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done ==="
date
