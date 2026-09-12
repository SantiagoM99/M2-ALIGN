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
# Env: DT (required), REPLAY_SOURCE (metamath), N_REPLAY (30000), S3_EPOCHS (2),
#      REPLAY_EVERY (3).
#
# REPLAY_SOURCE=gsm8k is D13's control (2026-09-12): the D6/v4 GSM8K pool that
# stage3_bn_dcl trained with, rerun under the current trainer and transformers
# 5.x. The MetaMathQA run and the dcl checkpoint differ in pool, trainer and
# environment at once; the control shares trainer and environment with the
# MetaMathQA run, so metamath minus gsm8k is the pool alone. Note that the
# trainer caps every replay file at --replay-max-rows-per-file (10,000), so
# N_REPLAY=30000 trains on a 10,000-row sample; GSM8K's 7,473 are all kept.

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
REPLAY_SOURCE="${REPLAY_SOURCE:-metamath}"
case "$REPLAY_SOURCE" in
  metamath) MATH_REPLAY="$A2/data/math_replay_bn_metamath.jsonl"; TAG="mm${N_REPLAY}" ;;
  gsm8k)    MATH_REPLAY="$A2/data/math_replay_bn.jsonl";          TAG="gsm8k" ;;
  *) echo "ERROR: REPLAY_SOURCE must be metamath or gsm8k"; exit 1 ;;
esac
TRANS_REPLAY="$DT/Stage1/data/Bengali_to_English.jsonl"
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

echo "=== [1] Replay pool: $REPLAY_SOURCE ==="
if [ -f "$MATH_REPLAY" ]; then
  echo "exists -> skipping ($(wc -l < "$MATH_REPLAY") rows)"
else
  BUILD_ARGS=(--source "$REPLAY_SOURCE")
  [ "$REPLAY_SOURCE" = metamath ] && BUILD_ARGS+=(--metamath-types GSM_ --n "$N_REPLAY")
  python -u build_math_replay.py "${BUILD_ARGS[@]}" \
    --nllb-tag ben_Beng --mt-path "$MT_PATH" --output "$MATH_REPLAY" \
    --local-files-only || exit 1
fi

echo "=== [2] Stage 3 bn with the $REPLAY_SOURCE replay pool ==="
DATA="$DT/Stage3/data/stage3b/bengali.jsonl"
[ -f "$DATA" ] || DATA="$DT/Stage3/data/bn.jsonl"
TRAIN_ARGS=(
  --data-path "$DATA" --images-dir "$GQA_IMAGES" --output-dir "$S3_OUT"
  --stage1-ckpt "$STAGE1_CKPT" --stage2-ckpt "$S2_CKPT"
  --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH"
  --vis-layers "$VIS_LAYERS"
  --replay-data "$MATH_REPLAY,$TRANS_REPLAY" --replay-every "$REPLAY_EVERY"
  --epochs "$S3_EPOCHS" --lr 2e-5
  --train-batch-size 2 --eval-batch-size 2 --grad-accum 16
  --max-gen-len 64 --save-steps 200
  --use-wandb --wandb-mode offline --wandb-project m2-align
  --wandb-run-name "a2-stage3-bn-$TAG" --local-files-only
)
# Done means the trainer's completion marker, not the best checkpoint: that file
# is written after the first improving epoch, so a run preempted in epoch 2
# would be skipped as finished. The trainer resumes its own snapshot.
python -u train_stage3_vqa.py "${TRAIN_ARGS[@]}" --check-complete
case $? in
  0) echo "training complete -> skipping" ;;
  3) python -u train_stage3_vqa.py "${TRAIN_ARGS[@]}" || exit 1 ;;
  *) echo "ERROR: completion check failed for $S3_OUT"; exit 1 ;;
esac
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
cp "$S3_OUT"/eval_*.jsonl "$A2/results/" 2>/dev/null || true   # per-item files: needed for the paired tests
cp "$S3_OUT"/eval_xgqa_*.jsonl "$A2/results/" 2>/dev/null || true
cd "$PROJECT_ROOT"
# No automatic commit: a harvest that moves HEAD under a prepared S1 submission
# breaks it. Commit by hand from a login node.
echo "Harvested into Approach2/results; review and commit by hand."
git -C "$PROJECT_ROOT" status --short Approach2/results | head -20
echo "=== Done === $(date)"
