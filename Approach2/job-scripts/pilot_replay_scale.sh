#!/bin/bash
#SBATCH --job-name=a2_pilot_replay
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_pilot_replay_%j.log

# D13 — replay volume, one language, single variable.
#
# The replay pool has not changed since D6: 7,473 GSM8K train questions.
# MindMerger's scale, MERLIN and Approach 1 all use 30k per language from
# MetaMathQA (Stage3/load_text.py:162), so our 4x-smaller pool is a confound
# in any comparison against them. It is also the one data axis we have never
# scaled, and the two largest gains in this project (D4, D11) both came from
# scaling data.
#
# NOT a generalization argument: 60.8% of MetaMathQA is GSM_* rephrasings of
# GSM8K train, the same split MGSM's test items come from (DESIGN.md).
# Justified by volume and phrasing diversity, and by matching the baselines.
#
# Single variable vs stage3_bn_v4: the replay pool. Same stage-2 checkpoint,
# same epochs, same lr, same replay-every.
#
# Env: DT (required), N_REPLAY (30000), S3_EPOCHS (2), REPLAY_EVERY (3).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
N_REPLAY="${N_REPLAY:-30000}"
S3_EPOCHS="${S3_EPOCHS:-2}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
S2_CKPT="$A2/outputs/stage2_dc_llava/mapping/pytorch_model.bin"
MATH_REPLAY="$A2/data/math_replay_bn_metamath.jsonl"
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"
TAG="mm${N_REPLAY}"
S3_OUT="$A2/outputs/stage3_bn_$TAG"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="; date; hostname
echo "TAG=$TAG N_REPLAY=$N_REPLAY S3_EPOCHS=$S3_EPOCHS"
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

for f in "$STAGE1_CKPT" "$S2_CKPT" "$TRANS_REPLAY"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

cd "$A2"

echo "=== [1] Build the MetaMathQA replay pool ==="
if [ -f "$MATH_REPLAY" ]; then
  echo "exists -> skipping ($(wc -l < "$MATH_REPLAY") rows)"
else
  python -u build_math_replay.py \
    --source metamath --metamath-types GSM_ \
    --n "$N_REPLAY" --nllb-tag ben_Beng \
    --mt-path "$MT_PATH" --output "$MATH_REPLAY" \
    --local-files-only || exit 1
fi

echo "=== [2] Stage 3 bn with the scaled replay ==="
if [ -f "$S3_OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> skipping"
else
  RESUME_ARGS=()
  [ -f "$S3_OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$S3_OUT/training_state.pt")
  DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
  [ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"
  python -u train_stage3_vqa.py \
    --data-path "$DATA" --images-dir "$GQA_IMAGES" --output-dir "$S3_OUT" \
    --stage1-ckpt "$STAGE1_CKPT" --stage2-ckpt "$S2_CKPT" \
    --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" \
    --replay-data "$MATH_REPLAY,$TRANS_REPLAY" --replay-every "$REPLAY_EVERY" \
    --epochs "$S3_EPOCHS" --lr 2e-5 \
    --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 \
    --max-gen-len 64 --save-steps 200 \
    --use-wandb --wandb-mode offline --wandb-project m2-align \
    --wandb-run-name "a2-stage3-bn-$TAG" --local-files-only \
    "${RESUME_ARGS[@]}" || exit 1
fi
CKPT="$S3_OUT/mapping/pytorch_model.bin"

echo "=== [3] Evaluations ==="
for spec in "mgsm:$PROJECT_ROOT/evaluation/MGSM.jsonl" "msvamp:$PROJECT_ROOT/evaluation/MSVAMP.jsonl"; do
  b="${spec%%:*}"; d="${spec#*:}"
  out="$S3_OUT/eval_${b}_bn_stage3_$TAG.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $b"; continue; }
  python -u evaluate_text.py --data-path "$d" --benchmark "$b" \
    --output-path "$out" --ckpt "$CKPT" --nllb-tag ben_Beng \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || echo "### $b FAILED"
done
for blind in 0 1; do
  suffix=""; args=()
  [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
  out="$S3_OUT/eval_xgqa_bn_${TAG}${suffix}.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip xgqa blind=$blind"; continue; }
  python -u evaluate_vqa.py --data-path "$DT/Stage3/data/xgqa/bn.jsonl" \
    --images-dir "$GQA_IMAGES" --ckpt "$CKPT" --output-path "$out" \
    --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" --local-files-only "${args[@]}" \
    || echo "### xgqa blind=$blind FAILED"
done

echo "=== Summary (baseline stage3_bn_v4: MGSM 62.0, MSVAMP 64.5, xGQA 47.66/30.83) ==="
for s in "$S3_OUT"/eval_*.summary.json; do echo "$s"; cat "$s"; echo; done

echo "=== Harvest ==="
mkdir -p "$A2/results"
cp "$S3_OUT"/eval_*.summary.json "$A2/results/" 2>/dev/null || true
cp "$S3_OUT"/eval_xgqa_*.jsonl "$A2/results/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: replay-scale pilot bn ($TAG, job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
