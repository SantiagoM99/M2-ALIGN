#!/bin/bash
#SBATCH --job-name=a2_ev_all
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_ev_all_%j.log

# ALL evaluations in ONE GPU allocation (packed mode): per language, xGQA
# full+blind and CVQA full+blind, then the Bengali MGSM/MSVAMP text eval
# (stage-3 variant only — the plain-Gemma and stage-1 rows don't change
# between rounds and were already measured). Each eval is skipped when its
# .summary.json already exists, so re-running only fills gaps.
#
# Env: DT (required), ROUND (optional round tag), VIS_LAYERS (default ""
#      — MUST match what the round's checkpoints were trained with).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
R="${ROUND:+_$ROUND}"
VIS_LAYERS="${VIS_LAYERS:-}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

XGQA_LANGS="bn de ru zh pt id ko"
CVQA_LANGS="bn ru zh pt id ko jv mn si ga"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "ROUND=${ROUND:-}"
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

cd "$A2"
FAILED=()

run_one () {  # <kind:xgqa|cvqa> <lang> <blind:0|1>
  local kind="$1" L="$2" blind="$3"
  local ckpt="$A2/outputs/stage3_$L$R/mapping/pytorch_model.bin"
  local outdir="$A2/outputs/stage3_$L$R"
  local suffix="" script="evaluate_vqa.py" args=() data images
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  if [ "$kind" = cvqa ]; then
    script="evaluate_cvqa.py"
    data="$DT/Stage3/data/cvqa/$L.jsonl"
    images="$DT/Stage3/data/cvqa/images"
  else
    data="$DT/Stage3/data/xgqa/$L.jsonl"
    images="$GQA_IMAGES"
  fi
  local out="$outdir/eval_${kind}_${L}${suffix}.jsonl"
  if [ ! -f "$ckpt" ]; then echo "--- skip $kind $L blind=$blind (no stage-3 ckpt)"; return; fi
  if [ ! -f "$data" ]; then echo "--- skip $kind $L blind=$blind (no data)"; return; fi
  if [ -f "$out.summary.json" ]; then echo "--- skip $kind $L blind=$blind (summary exists)"; return; fi
  echo "=== $kind $L blind=$blind === $(date)"
  if ! python -u "$script" \
      --data-path   "$data" \
      --images-dir  "$images" \
      --ckpt        "$ckpt" \
      --output-path "$out" \
      --mt-path     "$MT_PATH" \
      --vis-path    "$VIS_PATH" \
      --llm-path    "$LLM_PATH" \
      --vis-layers  "$VIS_LAYERS" \
      --local-files-only \
      "${args[@]}"; then
    echo "### $kind $L blind=$blind FAILED — continuing"
    FAILED+=("$kind:$L:$blind")
  fi
}

for L in $XGQA_LANGS; do
  run_one xgqa "$L" 0
  run_one xgqa "$L" 1
done
for L in $CVQA_LANGS; do
  run_one cvqa "$L" 0
  run_one cvqa "$L" 1
done

echo "=== Text evals MGSM/MSVAMP (bn, stage-3 variant) ==="
TXT_OUT="$A2/outputs/text_eval_bn_v2$R"
S3_BN="$A2/outputs/stage3_bn$R/mapping/pytorch_model.bin"
if [ -f "$PROJECT_ROOT/evaluation/MGSM.jsonl" ] && [ -f "$S3_BN" ]; then
  mkdir -p "$TXT_OUT"
  for bench in mgsm msvamp; do
    case "$bench" in
      mgsm)   data="$PROJECT_ROOT/evaluation/MGSM.jsonl" ;;
      msvamp) data="$PROJECT_ROOT/evaluation/MSVAMP.jsonl" ;;
    esac
    out="$TXT_OUT/eval_${bench}_bn_stage3.jsonl"
    if [ -f "$out.summary.json" ]; then echo "--- skip $bench (summary exists)"; continue; fi
    if ! python -u evaluate_text.py \
        --data-path   "$data" \
        --benchmark   "$bench" \
        --output-path "$out" \
        --ckpt        "$S3_BN" \
        --mt-path     "$MT_PATH" \
        --llm-path    "$LLM_PATH" \
        --local-files-only; then
      echo "### text $bench FAILED — continuing"
      FAILED+=("text:$bench")
    fi
  done
else
  echo "(text evals skipped: evaluation/ files or bn stage-3 ckpt missing)"
fi

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
# Round tag goes into the copied filename — rounds A (no tag) and B (_v2)
# would otherwise overwrite each other's summaries in results/.
harvest_dir () {
  local dir="$1" f base
  for f in "$dir"/eval_*.summary.json; do
    [ -f "$f" ] || continue
    base="$(basename "${f%.jsonl.summary.json}")"
    cp "$f" "$RESULTS_DIR/${base}${R}.jsonl.summary.json" 2>/dev/null || true
  done
  # Per-item xGQA predictions too — category breakdowns + McNemar
  # (analysis/xgqa_category_breakdown.py) need them.
  for f in "$dir"/eval_xgqa_*.jsonl; do
    [ -f "$f" ] || continue
    base="$(basename "${f%.jsonl}")"
    cp "$f" "$RESULTS_DIR/${base}${R}.jsonl" 2>/dev/null || true
  done
}
for L in $XGQA_LANGS $CVQA_LANGS; do
  harvest_dir "$A2/outputs/stage3_$L$R"
done
harvest_dir "$TXT_OUT"
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: packed evals$R (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED evals: ${FAILED[*]}"
  exit 1
fi
