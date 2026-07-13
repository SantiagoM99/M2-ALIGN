#!/bin/bash
#SBATCH --job-name=a2_eval_vqa
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_eval_%j.log

# Approach 2 evaluation on xGQA (Bengali by default).
# Submit from the repo root: sbatch Approach2/job-scripts/evaluate_vqa.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-$SCRATCH/huggingface/nllb-200-distilled-600M-full}"
EVAL_LANG="${EVAL_LANG:-bn}"
DATA_PATH="$PROJECT_ROOT/Stage3/data/stage3b_eval/xgqa/$EVAL_LANG.jsonl"
IMAGES_DIR="$PROJECT_ROOT/Stage3/data/gqa/images"
CKPT="$A2/outputs/stage3/mapping/pytorch_model.bin"
OUTPUT_PATH="$A2/outputs/stage3/eval_xgqa_${EVAL_LANG}.jsonl"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
nvidia-smi || true

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0

source "$SCRATCH/venvs/m2-align/bin/activate"

export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$A2"
python -u evaluate_vqa.py \
  --data-path   "$DATA_PATH" \
  --images-dir  "$IMAGES_DIR" \
  --ckpt        "$CKPT" \
  --output-path "$OUTPUT_PATH" \
  --mt-path     "$MT_PATH" \
  --vis-path    "$VIS_PATH" \
  --llm-path    "$LLM_PATH" \
  --local-files-only

echo "=== Done ==="
date
