#!/bin/bash
#SBATCH --job-name=a2_eval_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_eval_text_%j.log

# Approach 2 text-reasoning evaluation: MGSM + MSVAMP (Bengali), three
# variants each — plain Gemma (control), stage-1 mapping, stage-3 mapping.
# The stage-3 row is the one directly comparable to Approach 1's table.
#
# Expects the shared eval files at:  $PROJECT_ROOT/evaluation/{MGSM,MSVAMP}.jsonl
# Submit from the repo root: sbatch Approach2/job-scripts/evaluate_text.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_ROOT/evaluation}"
STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
STAGE3_CKPT="$A2/outputs/stage3/mapping/pytorch_model.bin"
OUT="$A2/outputs/text_eval"

echo "=== Job info ==="
date; hostname
nvidia-smi || true

echo "=== Load modules ==="
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

for f in "$EVAL_DIR/MGSM.jsonl" "$EVAL_DIR/MSVAMP.jsonl" "$STAGE1_CKPT" "$STAGE3_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f"
    exit 1
  fi
done

mkdir -p "$OUT"
cd "$A2"

run_eval () {
  local bench="$1" data="$2" tag="$3"; shift 3
  echo "=== $bench / $tag ==="
  python -u evaluate_text.py \
    --data-path   "$data" \
    --benchmark   "$bench" \
    --output-path "$OUT/eval_${bench}_bn_${tag}.jsonl" \
    --mt-path     "$MT_PATH" \
    --llm-path    "$LLM_PATH" \
    --local-files-only \
    "$@"
}

# MGSM (250 q) first — fast signal; then MSVAMP (1000 q).
run_eval mgsm   "$EVAL_DIR/MGSM.jsonl"   gemma  --no-mapping
run_eval mgsm   "$EVAL_DIR/MGSM.jsonl"   stage1 --ckpt "$STAGE1_CKPT"
run_eval mgsm   "$EVAL_DIR/MGSM.jsonl"   stage3 --ckpt "$STAGE3_CKPT"
run_eval msvamp "$EVAL_DIR/MSVAMP.jsonl" gemma  --no-mapping
run_eval msvamp "$EVAL_DIR/MSVAMP.jsonl" stage1 --ckpt "$STAGE1_CKPT"
run_eval msvamp "$EVAL_DIR/MSVAMP.jsonl" stage3 --ckpt "$STAGE3_CKPT"

echo "=== Summary table ==="
for s in "$OUT"/*.summary.json; do
  echo "$s"; cat "$s"; echo
done

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
cp "$OUT"/*.summary.json "$RESULTS_DIR/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: MGSM/MSVAMP text evals (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done ==="
date
