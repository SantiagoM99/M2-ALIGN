#!/bin/bash
#SBATCH --job-name=a2_stage2_vision
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_stage2_%j.log

# Approach 2, Stage 2: SigLIP -> Gemma vision mapping on WIT captions.
# Reuses Stage2's wit_pairs.jsonl and image cache (Stage2/load_image.py).
# Submit from the repo root: sbatch Approach2/job-scripts/train_stage2.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
DATA_PATH="$PROJECT_ROOT/Stage2/data/wit_pairs.jsonl"
IMAGE_CACHE_DIR="$PROJECT_ROOT/Stage2/data/image_cache"
OUTPUT_DIR="$A2/outputs/stage2"

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

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Data file not found: $DATA_PATH"
  echo "       Run: python Stage2/load_image.py --languages bn --n-per-language 100000 --output-dir Stage2/data"
  exit 1
fi

RESUME_ARGS=()
TRAINING_STATE="$OUTPUT_DIR/training_state.pt"
if [ -f "$TRAINING_STATE" ]; then
  echo "Found existing training_state.pt — resuming: $TRAINING_STATE"
  RESUME_ARGS=(--resume-from-checkpoint "$TRAINING_STATE")
fi

cd "$A2"
python -u train_stage2_vision.py \
  --data-path       "$DATA_PATH" \
  --image-cache-dir "$IMAGE_CACHE_DIR" \
  --output-dir      "$OUTPUT_DIR" \
  --vis-path        "$VIS_PATH" \
  --llm-path        "$LLM_PATH" \
  --epochs     1 \
  --lr         2e-5 \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --max-gen-len 512 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode     offline \
  --wandb-project  m2-align \
  --wandb-run-name "a2-stage2-vision" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
date
