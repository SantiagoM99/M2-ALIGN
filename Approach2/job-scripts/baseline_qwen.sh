#!/bin/bash
#SBATCH --job-name=a2_qwen_base
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_qwen_base_%j.log

# Zero-shot Qwen3-VL-8B baseline (Maryam's Baseline protocol, ported to
# Approach2/baseline_qwen/evaluate.py) run packed: full + BLIND (gray image)
# per language, per benchmark, in one GPU allocation. Produces the per-item
# predictions the upstream script doesn't write, completing the
# {Qwen3-VL, A2} x {full, blind} 2x2.
#
# Env:
#   DT     (required) datatransfer root
#   BENCH  xgqa | cvqa | all   (default all; xGQA is the long half — split
#          into BENCH=xgqa and BENCH=cvqa jobs if queue times matter)
#   LANGS  override language list for the chosen benchmark
#   QWEN_VENV  venv with recent transformers (default $SCRATCH/venvs/qwen)
#
# One-time setup on a login node (tmux):
#   module load StdEnv/2023 python/3.11.5
#   python -m venv $SCRATCH/venvs/qwen && source $SCRATCH/venvs/qwen/bin/activate
#   pip install --upgrade pip && pip install torch transformers accelerate pillow tqdm requests
#   HF_HOME=$SCRATCH/huggingface hf download Qwen/Qwen3-VL-8B-Instruct

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
BENCH="${BENCH:-all}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_DIR="$A2/outputs/baseline_qwen"

GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

XGQA_LANGS="${LANGS:-bn de ru zh pt id ko}"
CVQA_LANGS="${LANGS:-bn ru zh pt id ko jv mn si ga}"

echo "=== Job info ==="
date; hostname
echo "BENCH=$BENCH MODEL_ID=$MODEL_ID"
nvidia-smi || true

echo "=== Load modules ==="
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2

source "${QWEN_VENV:-$SCRATCH/venvs/qwen}/bin/activate"

export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT_DIR"
cd "$A2/baseline_qwen"
FAILED=()

run_one () {  # <bench> <lang> <blind:0|1>
  local bench="$1" L="$2" blind="$3" suffix="" args=() data
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  local out="$OUT_DIR/qwen_${bench}_${L}${suffix}.jsonl"
  if [ "$bench" = xgqa ]; then
    data="$DT/Stage3/data/xgqa/$L.jsonl"
    args+=(--images-dir "$GQA_IMAGES")
  else
    data="$DT/Stage3/data/cvqa/$L.jsonl"
    args+=(--image-cache-dir "$DT/Stage3/data/cvqa/images")
  fi
  if [ ! -f "$data" ]; then echo "--- skip $bench $L blind=$blind (no data)"; return; fi
  if [ -f "$out.summary.json" ]; then echo "--- skip $bench $L blind=$blind (summary exists)"; return; fi
  echo "=== qwen $bench $L blind=$blind === $(date)"
  if ! python -u evaluate.py \
      --benchmark   "$bench" \
      --lang        "$L" \
      --eval-data   "$data" \
      --model-id    "$MODEL_ID" \
      --output-path "$out" \
      --local-files-only \
      "${args[@]}"; then
    echo "### qwen $bench $L blind=$blind FAILED — continuing"
    FAILED+=("$bench:$L:$blind")
  fi
}

if [ "$BENCH" = cvqa ] || [ "$BENCH" = all ]; then
  for L in $CVQA_LANGS; do
    run_one cvqa "$L" 0
    run_one cvqa "$L" 1
  done
fi
if [ "$BENCH" = xgqa ] || [ "$BENCH" = all ]; then
  for L in $XGQA_LANGS; do
    run_one xgqa "$L" 0
    run_one xgqa "$L" 1
  done
fi

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
cp "$OUT_DIR"/qwen_*.jsonl "$RESULTS_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/qwen_*.summary.json "$RESULTS_DIR/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: qwen baseline $BENCH (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
