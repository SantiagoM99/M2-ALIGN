#!/bin/bash
#SBATCH --job-name=a2_align
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_align_%j.log

# X2 — score the text bridge's cross-lingual ALIGNMENT, per language.
#
# No generation, no benchmark, no labels: NLLB encoder forward passes plus a
# Gemma embedding lookup over held-out parallel sentences. Minutes per
# checkpoint, which is why this asks for the 3h partition rather than 12h.
#
# Checkpoints scored, and what each is for:
#   stage3_bn_dcl   the bridge that actually does the transfer in E2/E3.
#                   THIS is the one the mechanism claim is about.
#   stage1          the Bengali-only mapping before any VQA training — says
#                   whether stage 3 improved or damaged alignment.
#   stage3_<L>_v4   each language's own supervised bridge, for contrast.
#   stage1_joint    D12's shared multilingual mapping, when it exists.
#
# That last row is the point of running this before X3 finishes: if the joint
# mapping does not raise the alignment margin for jv/mn/ga, D12 will not fix
# their transfer either, and we learn that without training eleven stage 3s.
#
# Env: DT (required), N (1000 held-out pairs per language), CKPTS (override).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
N="${N:-1000}"
LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
S1_DIR="${S1_DIR:-$DT/Stage1/data}"
[ -d "$S1_DIR" ] || S1_DIR="$PROJECT_ROOT/Stage1/data"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="; date; hostname
echo "N=$N per language, stage-1 data: $S1_DIR"
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
mkdir -p results

if [ -z "${CKPTS:-}" ]; then
  CKPTS="stage3_bn_dcl stage1 stage1_joint"
  for L in de ru zh pt id ko jv mn si ga; do CKPTS="$CKPTS stage3_${L}_v4"; done
fi

FAILED=()
for name in $CKPTS; do
  ckpt="$A2/outputs/$name/mapping/pytorch_model.bin"
  out="$A2/results/alignment_${name}.json"
  [ -f "$ckpt" ] || { echo "--- skip $name (no checkpoint)"; continue; }
  [ -f "$out" ]  && { echo "--- skip $name (already scored)"; continue; }
  echo "=== scoring $name === $(date)"
  python -u alignment_score.py \
    --ckpt "$ckpt" --label "$name" \
    --stage1-dir "$S1_DIR" --output "$out" \
    --n "$N" --mt-path "$MT_PATH" --llm-path "$LLM_PATH" \
    --local-files-only || FAILED+=("$name")
done

echo
echo "=== Read the result with: ==="
echo "  cd Approach2/results && python3 ../analysis/alignment_vs_transfer.py"

cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: X2 text-bridge alignment scores (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; } || exit 0
