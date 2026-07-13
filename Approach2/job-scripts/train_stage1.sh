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
MT_PATH="${MT_PATH:-$SCRATCH/huggingface/nllb-200-distilled-600M-full}"
DATA_DIR="$PROJECT_ROOT/Stage1/data"
OUTPUT_DIR="$A2/outputs/stage1"

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
module load arrow/18.1.0

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
  --languages  Bengali \
  --output-dir "$OUTPUT_DIR" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --train-num  100000 \
  --epochs     3 \
  --lr         2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode     offline \
  --wandb-project  m2-align \
  --wandb-run-name "a2-stage1-text" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
date
