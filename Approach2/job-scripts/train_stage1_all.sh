#!/bin/bash
#SBATCH --job-name=a2_s1_all
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=28:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_s1_all_%j.log

# All stage-1 text mappings in ONE GPU allocation, one language after the
# other (packed mode: hold a single GPU and run as much as possible on it).
# A language whose checkpoint already exists is skipped; a language that
# fails is reported and the loop continues (the packed stage-3 job checks
# per-language prerequisites itself, so one failure doesn't block the rest).
#
# Env: DT (required), LANGS (default: the 10 non-Bengali languages),
#      S1_EPOCHS / S1_LR overrides.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
LANGS="${LANGS:-de ru zh pt id ko jv mn si ga}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"

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
for L in $LANGS; do
  name="${NAME[$L]:-}"
  [ -n "$name" ] || { echo "### $L: unknown language code, skipping"; continue; }
  OUT="$A2/outputs/stage1_$L"
  if [ -f "$OUT/mapping/pytorch_model.bin" ]; then
    echo "### stage1 $L: checkpoint exists -> done"
    continue
  fi
  if [ ! -f "$DT/Stage1/data/${name}_to_English.jsonl" ]; then
    echo "### stage1 $L: no data (${name}_to_English.jsonl), skipping"
    continue
  fi
  RESUME_ARGS=()
  [ -f "$OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$OUT/training_state.pt")
  echo "=== stage1 $L ($name) === $(date)"
  if ! python -u train_stage1_text.py \
      --data-dir   "$DT/Stage1/data" \
      --languages  "$name" \
      --output-dir "$OUT" \
      --mt-path    "$MT_PATH" \
      --llm-path   "$LLM_PATH" \
      --train-num  100000 \
      --epochs     "${S1_EPOCHS:-3}" \
      --lr         "${S1_LR:-2e-5}" \
      --train-batch-size 4 \
      --eval-batch-size  4 \
      --grad-accum 8 \
      --save-steps 200 \
      --use-wandb --wandb-mode offline --wandb-project m2-align \
      --wandb-run-name "a2-stage1-$L" \
      --local-files-only \
      "${RESUME_ARGS[@]}"; then
    echo "### stage1 $L FAILED — continuing with next language"
    FAILED+=("$L")
  fi
done

echo "=== Harvest curves into git ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
grep -h -E "Epoch [0-9]+ \| val_loss" "$A2"/logs/a2_stage1_text_*.log 2>/dev/null \
  > "$RESULTS_DIR/stage1_all_curves_${SLURM_JOB_ID:-manual}.txt" || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: stage1 packed curves (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED languages: ${FAILED[*]}"
  exit 1
fi
