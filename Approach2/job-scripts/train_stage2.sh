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

# Approach 2, Stage 2: SigLIP -> Gemma vision mapping on English captions.
# DATA_PATH / IMAGE_CACHE_DIR accept comma-separated lists, so WIT and CC3M
# can be combined (cc3m_pairs.jsonl uses the shared cc3m/image_cache):
#   DATA_PATH=$DT/Stage2/data/bn/cc3m_pairs.jsonl,$DT/Stage2/data/bn/wit_pairs.jsonl \
#   IMAGE_CACHE_DIR=$DT/Stage2/data/cc3m/image_cache,$DT/Stage2/data/bn/image_cache \
#   OUTPUT_DIR=$PWD/Approach2/outputs/stage2_cc3m \
#     sbatch --export=ALL Approach2/job-scripts/train_stage2.sh
# Submit from the repo root: sbatch Approach2/job-scripts/train_stage2.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Stage2/data/wit_pairs.jsonl}"
IMAGE_CACHE_DIR="${IMAGE_CACHE_DIR:-$PROJECT_ROOT/Stage2/data/image_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$A2/outputs/stage2}"

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

IFS=',' read -ra _DATA_FILES <<< "$DATA_PATH"
for f in "${_DATA_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Data file not found: $f"
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
python -u train_stage2_vision.py \
  --data-path       "$DATA_PATH" \
  --image-cache-dir "$IMAGE_CACHE_DIR" \
  --output-dir      "$OUTPUT_DIR" \
  --vis-path        "$VIS_PATH" \
  --llm-path        "$LLM_PATH" \
  --epochs     "${S2_EPOCHS:-1}" \
  --lr         "${S2_LR:-2e-5}" \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum "${S2_GRAD_ACCUM:-8}" \
  --max-gen-len 512 \
  --save-steps 200 \
  --use-wandb \
  --wandb-mode     offline \
  --wandb-project  m2-align \
  --wandb-run-name "${RUN_NAME:-a2-stage2-vision}" \
  --local-files-only \
  "${RESUME_ARGS[@]}"

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
APP_LOG=$(ls -t "$A2"/logs/a2_stage2_vision_*.log 2>/dev/null | head -1)
if [ -n "$APP_LOG" ]; then
  grep -E "Epoch [0-9]+ \| val_loss" "$APP_LOG" \
    > "$RESULTS_DIR/stage2_curve_${SLURM_JOB_ID:-manual}.txt" || true
fi
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage2 curve (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done ==="
date
