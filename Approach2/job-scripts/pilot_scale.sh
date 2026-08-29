#!/bin/bash
#SBATCH --job-name=a2_pilot_scale
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_pilot_scale_%j.log
#
# Stage-2 data scale-up pilot (Bengali), packed in one GPU allocation:
#   [1] stage 2 on CC3M + WIT + LLaVA-Pretrain (~111k pairs vs 11.5k), same
#       DenseConnector setting as D9 — the ONLY variable vs stage2_dc is the
#       data. Fewer epochs (2, not 10) to keep sample-epochs comparable:
#       11.5k x 10 = 115k, 111k x 2 = 222k, and D4's best val_ppl came at
#       epoch 4 of 10 (~46k), so 2 epochs is already past that point.
#   [2] stage 3 bn: identical to stage3_bn_dc_e2 (2 ep, replay, same lr) —
#       the D9b-accepted recipe, so the delta measures what LLaVA adds ON TOP
#       of everything already accepted
#   [3] xGQA bn full+blind, CVQA bn full+blind, MGSM, MSVAMP
#
# Baseline to beat (stage3_bn_dc_e2, DESIGN.md D9b): xGQA 46.34/30.98,
# CVQA 41.61/32.52, MGSM 35.6, MSVAMP 54.2.
#
# Prereq: build_llava_pretrain.py already run on a login node.
# Env: DT (required), LLAVA_DIR (default $DT/Stage2/data/llava),
#      VIS_LAYERS (default "9,18,-1"), S2_EPOCHS (2), S3_EPOCHS (2),
#      REPLAY_EVERY (3).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
LLAVA_DIR="${LLAVA_DIR:-$DT/Stage2/data/llava}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"
S2_EPOCHS="${S2_EPOCHS:-2}"
S3_EPOCHS="${S3_EPOCHS:-2}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
S2_OUT="$A2/outputs/stage2_dc_llava"
MATH_REPLAY="$A2/data/math_replay_bn.jsonl"
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"

TAG="dcl"
[ "$S3_EPOCHS" != 2 ] && TAG="dcl_e$S3_EPOCHS"
S3_OUT="$A2/outputs/stage3_bn_$TAG"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "TAG=$TAG VIS_LAYERS=$VIS_LAYERS S2_EPOCHS=$S2_EPOCHS S3_EPOCHS=$S3_EPOCHS LLAVA_DIR=$LLAVA_DIR"
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

for f in "$STAGE1_CKPT" "$TRANS_REPLAY" "$MATH_REPLAY" \
         "$LLAVA_DIR/llava_pairs.jsonl"; do
  [ -f "$f" ] || { echo "ERROR: required file not found: $f"; exit 1; }
done
[ -d "$LLAVA_DIR/image_cache" ] || { echo "ERROR: no $LLAVA_DIR/image_cache"; exit 1; }

# Same assembly as pilot_dense.sh (CC3M + every language's WIT), plus LLaVA.
S2_DATA="$DT/Stage2/data/bn/cc3m_pairs.jsonl"
S2_CACHES="$DT/Stage2/data/cc3m/image_cache"
for wp in "$DT"/Stage2/data/*/wit_pairs.jsonl; do
  [ -f "$wp" ] || continue
  d="$(dirname "$wp")"
  [ -d "$d/image_cache" ] || continue
  S2_DATA="$S2_DATA,$wp"
  S2_CACHES="$S2_CACHES,$d/image_cache"
done
S2_DATA="$S2_DATA,$LLAVA_DIR/llava_pairs.jsonl"
S2_CACHES="$S2_CACHES,$LLAVA_DIR/image_cache"
echo "stage-2 pairs: $(cat $(echo "$S2_DATA" | tr ',' ' ') 2>/dev/null | wc -l)"

cd "$A2"

echo "=== [1] Stage 2, scaled data (vis_layers=$VIS_LAYERS) ==="
if [ -f "$S2_OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> skipping stage 2"
else
  RESUME_ARGS=()
  [ -f "$S2_OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$S2_OUT/training_state.pt")
  python -u train_stage2_vision.py \
    --data-path       "$S2_DATA" \
    --image-cache-dir "$S2_CACHES" \
    --output-dir      "$S2_OUT" \
    --vis-path        "$VIS_PATH" \
    --llm-path        "$LLM_PATH" \
    --vis-layers      "$VIS_LAYERS" \
    --epochs "$S2_EPOCHS" --lr 1e-4 \
    --train-batch-size 4 --eval-batch-size 4 --grad-accum 2 \
    --max-gen-len 512 --save-steps 500 \
    --use-wandb --wandb-mode offline --wandb-project m2-align \
    --wandb-run-name "a2-stage2-dc-llava" \
    --local-files-only \
    "${RESUME_ARGS[@]}" || exit 1
fi
S2_CKPT="$S2_OUT/mapping/pytorch_model.bin"

echo "=== [2] Stage 3 bn + replay on scaled stage 2 ($TAG) ==="
if [ -f "$S3_OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> skipping stage 3"
else
  RESUME_ARGS=()
  [ -f "$S3_OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$S3_OUT/training_state.pt")
  DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
  [ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"
  python -u train_stage3_vqa.py \
    --data-path   "$DATA" \
    --images-dir  "$GQA_IMAGES" \
    --output-dir  "$S3_OUT" \
    --stage1-ckpt "$STAGE1_CKPT" \
    --stage2-ckpt "$S2_CKPT" \
    --mt-path     "$MT_PATH" \
    --vis-path    "$VIS_PATH" \
    --llm-path    "$LLM_PATH" \
    --vis-layers  "$VIS_LAYERS" \
    --replay-data "$MATH_REPLAY,$TRANS_REPLAY" \
    --replay-every "$REPLAY_EVERY" \
    --epochs "$S3_EPOCHS" --lr 2e-5 \
    --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 \
    --max-gen-len 64 --save-steps 200 \
    --use-wandb --wandb-mode offline --wandb-project m2-align \
    --wandb-run-name "a2-stage3-bn-$TAG" \
    --local-files-only \
    "${RESUME_ARGS[@]}" || exit 1
fi
CKPT="$S3_OUT/mapping/pytorch_model.bin"

run_text () {  # <bench> <data>
  local bench="$1" data="$2"
  local out="$S3_OUT/eval_${bench}_bn_stage3_$TAG.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $bench (summary exists)"; return; }
  python -u evaluate_text.py \
    --data-path "$data" --benchmark "$bench" \
    --output-path "$out" --ckpt "$CKPT" \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || echo "### $bench FAILED"
}

run_vqa () {  # <kind:xgqa|cvqa> <blind:0|1>
  local kind="$1" blind="$2"
  local suffix="" args=() script="evaluate_vqa.py" data images
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  if [ "$kind" = cvqa ]; then
    script="evaluate_cvqa.py"
    data="$DT/Stage3/data/cvqa/bn.jsonl"
    images="$DT/Stage3/data/cvqa/images"
  else
    data="$DT/Stage3/data/xgqa/bn.jsonl"
    images="$GQA_IMAGES"
  fi
  local out="$S3_OUT/eval_${kind}_bn_${TAG}${suffix}.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $kind blind=$blind (summary exists)"; return; }
  python -u "$script" \
    --data-path "$data" --images-dir "$images" \
    --ckpt "$CKPT" --output-path "$out" \
    --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" \
    --local-files-only "${args[@]}" || echo "### $kind blind=$blind FAILED"
}

echo "=== [3] Evaluations ==="
run_vqa xgqa 0
run_vqa xgqa 1
run_vqa cvqa 0
run_vqa cvqa 1
run_text mgsm   "$PROJECT_ROOT/evaluation/MGSM.jsonl"
run_text msvamp "$PROJECT_ROOT/evaluation/MSVAMP.jsonl"

echo "=== Summary ==="
for s in "$S3_OUT"/eval_*.summary.json; do echo "$s"; cat "$s"; echo; done

echo "=== Harvest results into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
cp "$S3_OUT"/eval_*.summary.json "$RESULTS_DIR/" 2>/dev/null || true
cp "$S3_OUT"/eval_xgqa_*.jsonl "$S3_OUT"/eval_cvqa_*.jsonl "$RESULTS_DIR/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage-2 scale pilot bn ($TAG, job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
