#!/bin/bash
#SBATCH --job-name=a2_s3_all
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_s3_all_%j.log

# All stage-3 joint VQA trainings in ONE GPU allocation, one language after
# the other (packed mode). Skips languages whose checkpoint exists or whose
# prerequisites (stage-1/stage-2 ckpts, stage3b data) are missing; a failed
# language is reported and the loop continues.
#
# Env: DT (required), ROUND (optional round tag, e.g. v2),
#      STAGE2_CKPT (required), LANGS (default: bn + 10),
#      S3_EPOCHS / S3_LR overrides.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
R="${ROUND:+_$ROUND}"
STAGE2_CKPT="${STAGE2_CKPT:?set STAGE2_CKPT}"
LANGS="${LANGS:-bn de ru zh pt id ko jv mn si ga}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

declare -A NAME=( [bn]=Bengali [de]=German [ru]=Russian [zh]=Chinese
                  [pt]=Portuguese [id]=Indonesian [ko]=Korean [jv]=Javanese
                  [mn]=Mongolian [si]=Sinhalese [ga]=Irish )

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "ROUND=${ROUND:-} STAGE2_CKPT=$STAGE2_CKPT"
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

if [ ! -f "$STAGE2_CKPT" ]; then
  echo "ERROR: stage-2 checkpoint not found: $STAGE2_CKPT"
  exit 1
fi

cd "$A2"
FAILED=()
for L in $LANGS; do
  name="${NAME[$L]:-}"
  [ -n "$name" ] || { echo "### $L: unknown language code, skipping"; continue; }
  OUT="$A2/outputs/stage3_$L$R"
  if [ -f "$OUT/mapping/pytorch_model.bin" ]; then
    echo "### stage3 $L: checkpoint exists -> done"
    continue
  fi
  DATA="$DT/Stage3/data/stage3b/$(echo "$name" | tr '[:upper:]' '[:lower:]').jsonl"
  [ -f "$DATA" ] || DATA="$DT/Stage3/data/$L.jsonl"
  if [ ! -f "$DATA" ]; then
    echo "### stage3 $L: no stage3b data, skipping"
    continue
  fi
  if [ "$L" = bn ]; then
    S1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
  else
    S1_CKPT="$A2/outputs/stage1_$L/mapping/pytorch_model.bin"
  fi
  if [ ! -f "$S1_CKPT" ]; then
    echo "### stage3 $L: stage-1 checkpoint missing ($S1_CKPT), skipping"
    continue
  fi
  RESUME_ARGS=()
  [ -f "$OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$OUT/training_state.pt")
  echo "=== stage3 $L ($name) === $(date)"
  if ! python -u train_stage3_vqa.py \
      --data-path   "$DATA" \
      --images-dir  "$GQA_IMAGES" \
      --output-dir  "$OUT" \
      --stage1-ckpt "$S1_CKPT" \
      --stage2-ckpt "$STAGE2_CKPT" \
      --mt-path     "$MT_PATH" \
      --vis-path    "$VIS_PATH" \
      --llm-path    "$LLM_PATH" \
      --epochs      "${S3_EPOCHS:-1}" \
      --lr          "${S3_LR:-2e-5}" \
      --train-batch-size 2 \
      --eval-batch-size  2 \
      --grad-accum  16 \
      --max-gen-len 64 \
      --save-steps  200 \
      --use-wandb --wandb-mode offline --wandb-project m2-align \
      --wandb-run-name "a2-stage3-$L$R" \
      --local-files-only \
      "${RESUME_ARGS[@]}"; then
    echo "### stage3 $L FAILED — continuing with next language"
    FAILED+=("$L")
  fi
done

echo "=== Harvest curves into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
grep -h -E "Epoch [0-9]+ \| val_loss" "$A2"/logs/a2_stage3_vqa_*.log 2>/dev/null \
  > "$RESULTS_DIR/stage3_all_curves${R}_${SLURM_JOB_ID:-manual}.txt" || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage3 packed curves$R (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED languages: ${FAILED[*]}"
  exit 1
fi
