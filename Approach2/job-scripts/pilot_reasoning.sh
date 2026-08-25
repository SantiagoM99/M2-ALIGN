#!/bin/bash
#SBATCH --job-name=a2_pilot_reason
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_pilot_reason_%j.log

# Reasoning-preservation pilot (Bengali), packed in one GPU allocation:
#   [1] build GSM8K→bn math replay data (NLLB) if missing
#   [2] retrain stage 3 bn with text replay (translation + math) mixed in
#       (optionally GATE=1 → --zero-init-gate, separate output dir)
#   [3] MGSM + MSVAMP with the new checkpoint (stage3 variant only)
#   [4] xGQA bn full + blind — regression check (must stay ≈41 / ≈30)
#
# Success = MGSM way above 9.2 with xGQA intact.
#
# One-time on a login node (compute nodes are offline):
#   HF_HOME=$SCRATCH/huggingface python -c \
#     "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
#
# Env: DT (required), GATE=0|1 (default 0), REPLAY_EVERY (default 3),
#      STAGE2_CKPT (default: the round-B stage2_cc3m_v2 checkpoint).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
GATE="${GATE:-0}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
STAGE2_CKPT="${STAGE2_CKPT:-$A2/outputs/stage2_cc3m_v2/mapping/pytorch_model.bin}"
MATH_REPLAY="$A2/data/math_replay_bn.jsonl"
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"

TAG="replay"; GATE_ARGS=()
if [ "$GATE" = 1 ]; then TAG="replay_gate"; GATE_ARGS=(--zero-init-gate); fi
OUT="$A2/outputs/stage3_bn_$TAG"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "TAG=$TAG REPLAY_EVERY=$REPLAY_EVERY STAGE2_CKPT=$STAGE2_CKPT"
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

for f in "$STAGE1_CKPT" "$STAGE2_CKPT" "$TRANS_REPLAY"; do
  [ -f "$f" ] || { echo "ERROR: required file not found: $f"; exit 1; }
done

cd "$A2"

echo "=== [1] Math replay data ==="
if [ -f "$MATH_REPLAY" ]; then
  echo "exists: $MATH_REPLAY"
else
  python -u build_math_replay.py \
    --output "$MATH_REPLAY" --nllb-tag ben_Beng \
    --mt-path "$MT_PATH" --local-files-only || exit 1
fi

echo "=== [2] Stage 3 bn + text replay ($TAG) ==="
if [ -f "$OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> skipping training"
else
  RESUME_ARGS=()
  [ -f "$OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$OUT/training_state.pt")
  DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
  [ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"
  python -u train_stage3_vqa.py \
    --data-path   "$DATA" \
    --images-dir  "$GQA_IMAGES" \
    --output-dir  "$OUT" \
    --stage1-ckpt "$STAGE1_CKPT" \
    --stage2-ckpt "$STAGE2_CKPT" \
    --mt-path     "$MT_PATH" \
    --vis-path    "$VIS_PATH" \
    --llm-path    "$LLM_PATH" \
    --replay-data "$MATH_REPLAY,$TRANS_REPLAY" \
    --replay-every "$REPLAY_EVERY" \
    --epochs 1 --lr 2e-5 \
    --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 \
    --max-gen-len 64 --save-steps 200 \
    --use-wandb --wandb-mode offline --wandb-project m2-align \
    --wandb-run-name "a2-stage3-bn-$TAG" \
    --local-files-only \
    "${GATE_ARGS[@]}" "${RESUME_ARGS[@]}" || exit 1
fi
CKPT="$OUT/mapping/pytorch_model.bin"

run_text () {  # <bench> <data>
  local bench="$1" data="$2" out="$OUT/eval_${bench}_bn_stage3_$TAG.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $bench (summary exists)"; return; }
  python -u evaluate_text.py \
    --data-path "$data" --benchmark "$bench" \
    --output-path "$out" --ckpt "$CKPT" \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || echo "### $bench FAILED"
}

run_vqa () {  # <blind:0|1>
  local blind="$1" suffix="" args=()
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  local out="$OUT/eval_xgqa_bn_${TAG}${suffix}.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip xgqa blind=$blind (summary exists)"; return; }
  python -u evaluate_vqa.py \
    --data-path "$DT/Stage3/data/xgqa/bn.jsonl" --images-dir "$GQA_IMAGES" \
    --ckpt "$CKPT" --output-path "$out" \
    --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
    --local-files-only "${args[@]}" || echo "### xgqa blind=$blind FAILED"
}

echo "=== [3] MGSM + MSVAMP ==="
run_text mgsm   "$PROJECT_ROOT/evaluation/MGSM.jsonl"
run_text msvamp "$PROJECT_ROOT/evaluation/MSVAMP.jsonl"

echo "=== [4] xGQA bn full + blind (regression check) ==="
run_vqa 0
run_vqa 1

echo "=== Summary ==="
for s in "$OUT"/eval_*.summary.json; do echo "$s"; cat "$s"; echo; done

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
cp "$OUT"/eval_*.summary.json "$RESULTS_DIR/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: reasoning pilot bn ($TAG, job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
