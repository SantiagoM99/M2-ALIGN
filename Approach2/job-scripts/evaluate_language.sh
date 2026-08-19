#!/bin/bash
#SBATCH --job-name=a2_eval_lang
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_eval_lang_%j.log

# Batched per-language evaluation: xGQA full+blind and CVQA full+blind run
# sequentially inside ONE GPU allocation, instead of 2-4 separate jobs.
# Cuts scheduler/model-load overhead and is kinder to the shared account's
# fairshare than a fan-out of tiny jobs.
#
# Env (submit with --export=ALL):
#   EVAL_LANG   (required)  ISO code, e.g. de
#   CKPT        (required)  stage-3 mapping checkpoint for this language
#   OUT_DIR     (required)  where eval_*.jsonl(+.summary.json) are written
#   XGQA_DATA / XGQA_IMAGES   optional — xGQA evals skipped if unset/missing
#   CVQA_DATA / CVQA_IMAGES   optional — CVQA evals skipped if unset/missing
# Evals whose .summary.json already exists are skipped individually.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
EVAL_LANG="${EVAL_LANG:?set EVAL_LANG}"
CKPT="${CKPT:?set CKPT}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "EVAL_LANG=$EVAL_LANG CKPT=$CKPT"
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

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$OUT_DIR"
cd "$A2"

run_one () {  # <kind:xgqa|cvqa> <blind:0|1> <data> <images>
  local kind="$1" blind="$2" data="$3" images="$4"
  local suffix="" script="evaluate_vqa.py" args=()
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  [ "$kind" = cvqa ] && script="evaluate_cvqa.py"
  local out="$OUT_DIR/eval_${kind}_${EVAL_LANG}${suffix}.jsonl"
  if [ -z "$data" ] || [ ! -f "$data" ]; then
    echo "--- skip $kind blind=$blind (no data file)"; return
  fi
  if [ -f "$out.summary.json" ]; then
    echo "--- skip $kind blind=$blind (summary exists)"; return
  fi
  echo "=== $kind blind=$blind → $out ==="
  python -u "$script" \
    --data-path   "$data" \
    --images-dir  "$images" \
    --ckpt        "$CKPT" \
    --output-path "$out" \
    --mt-path     "$MT_PATH" \
    --vis-path    "$VIS_PATH" \
    --llm-path    "$LLM_PATH" \
    --local-files-only \
    "${args[@]}"
}

run_one xgqa 0 "${XGQA_DATA:-}" "${XGQA_IMAGES:-}"
run_one xgqa 1 "${XGQA_DATA:-}" "${XGQA_IMAGES:-}"
run_one cvqa 0 "${CVQA_DATA:-}" "${CVQA_IMAGES:-}"
run_one cvqa 1 "${CVQA_DATA:-}" "${CVQA_IMAGES:-}"

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
cp "$OUT_DIR"/eval_*'.jsonl' "$RESULTS_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/eval_*.summary.json "$RESULTS_DIR/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: $EVAL_LANG evals (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done ==="
date
