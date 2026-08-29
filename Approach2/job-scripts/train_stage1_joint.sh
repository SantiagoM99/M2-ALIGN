#!/bin/bash
#SBATCH --job-name=a2_s1_joint
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_s1_joint_%j.log

# ONE text mapping trained on all 11 languages shuffled together, instead of
# 11 separate per-language mappings.
#
# This is how MindMerger trains its mapping stage — `read_lego`/`read_nllb`
# in Stage1/tools/read_datasets.py accumulate every language into one list
# and shuffle it — and it is the mechanism by which a low-resource language
# borrows structure from a high-resource one: they share the parameters. We
# diverged from that without testing it (train_stage1_all.sh passes
# `--languages "$name"`, one language per run), and DESIGN.md D10 shows our
# low-resource languages falling further behind on vision extraction, so the
# divergence is worth pricing.
#
# TRAIN_NUM is per language. The default 30000 x 11 = 330k samples is about
# one per-language run's budget (100k x 3 epochs), so the joint mapping runs
# under a handicap: less data per language than the baseline it is compared
# against. If it still wins on the low-resource group, the full-data version
# wins by more.
#
# Env: DT (required), TRAIN_NUM (30000), S1_EPOCHS (1), S1_LR (2e-5),
#      LANGS (all 11), OUT (Approach2/outputs/stage1_joint).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
LANGS="${LANGS:-bn de ru zh pt id ko jv mn si ga}"
TRAIN_NUM="${TRAIN_NUM:-30000}"
S1_EPOCHS="${S1_EPOCHS:-1}"
S1_LR="${S1_LR:-2e-5}"
OUT="${OUT:-$A2/outputs/stage1_joint}"

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
echo "LANGS=$LANGS TRAIN_NUM=$TRAIN_NUM S1_EPOCHS=$S1_EPOCHS OUT=$OUT"
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

# Only languages whose translation file is actually present; a missing one
# would otherwise abort the whole joint run rather than a single language.
NAMES=""
for L in $LANGS; do
  n="${NAME[$L]:-}"
  [ -n "$n" ] || { echo "### $L: unknown code, skipping"; continue; }
  if [ ! -f "$DT/Stage1/data/${n}_to_English.jsonl" ]; then
    echo "### $L ($n): no translation data, skipping"
    continue
  fi
  NAMES="${NAMES:+$NAMES,}$n"
done
[ -n "$NAMES" ] || { echo "ERROR: no usable languages"; exit 1; }
echo "joint languages: $NAMES"

cd "$A2"

if [ -f "$OUT/mapping/pytorch_model.bin" ]; then
  echo "checkpoint exists -> nothing to do"
  exit 0
fi

RESUME_ARGS=()
[ -f "$OUT/training_state.pt" ] && RESUME_ARGS=(--resume-from-checkpoint "$OUT/training_state.pt")

echo "=== stage1 joint === $(date)"
python -u train_stage1_text.py \
  --data-dir   "$DT/Stage1/data" \
  --languages  "$NAMES" \
  --output-dir "$OUT" \
  --mt-path    "$MT_PATH" \
  --llm-path   "$LLM_PATH" \
  --train-num  "$TRAIN_NUM" \
  --epochs     "$S1_EPOCHS" \
  --lr         "$S1_LR" \
  --train-batch-size 4 \
  --eval-batch-size  4 \
  --grad-accum 8 \
  --save-steps 200 \
  --use-wandb --wandb-mode offline --wandb-project m2-align \
  --wandb-run-name "a2-stage1-joint" \
  --local-files-only \
  "${RESUME_ARGS[@]}" || exit 1

echo "=== Done === $(date)"
