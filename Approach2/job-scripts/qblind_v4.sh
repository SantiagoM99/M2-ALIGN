#!/bin/bash
#SBATCH --job-name=a2_qblind
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a2_qblind_%j.log
#
# Does a system TRAINED to use the question also do better without it?
#
# On the zero-shot VLM, removing CVQA's native question raised accuracy by +1.50
# and dV by +4.01 in all ten languages (DESIGN 2026-09-26). If a trained system
# behaves the same way, the benchmark is not exercising the pathway either
# approach exists to improve, and every CVQA claim in the paper has to say so.
#
# The arm is the Bengali donor checkpoint evaluated zero-shot on the transfer
# targets, which is the setting the paper argues about. Both question conditions
# run in this one allocation on the same day and code, because the with-question
# numbers on record are from August and CVQA's real-image cells drift across that
# boundary; pairing today against today avoids the question entirely.
#
#   DT=/scratch/santimn/datatransfer sbatch Approach2/job-scripts/qblind_v4.sh
#
# Env: DT (required), LANGS, CKPT, TAG
set -uo pipefail

ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$ROOT/Approach2"
DT="${DT:?set DT}"
LANGS="${LANGS:-jv mn ga si}"
CKPT="${CKPT:-$A2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin}"
TAG="${TAG:-0926}"
OUT_DIR="$A2/outputs/qblind"
RESULTS_DIR="${RESULTS_DIR:-$A2/results}"

echo "=== Job info ==="; date; hostname; nvidia-smi || true

module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

[ -f "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }
mkdir -p "$OUT_DIR"
cd "$A2"
RAN=0
FAILED=()

run_cell () {  # <lang> <question:wq|qb> <image:correct|gray>
  # bash 3.2 (where this gets dry-run) treats "${EMPTY[@]}" as unbound under
  # `set -u`, so the array is expanded only when it has elements.
  local L="$1" q="$2" img="$3" name
  local args=()
  name="eval_cvqa_${L}_zsbn"
  [ "$img" = gray ] && { name="${name}_BLIND"; args+=(--blind); }
  [ "$q" = qb ] && args+=(--blind-question)
  local out="$OUT_DIR/${name}_${q}_${TAG}.jsonl"
  if [ -f "$out.summary.json" ]; then echo "--- skip $(basename "$out") (exists)"; return; fi
  echo "=== $L question=$q image=$img -> $(basename "$out") === $(date)"
  RAN=$((RAN + 1))
  if ! python -u evaluate_cvqa.py \
      --data-path "$DT/Stage3/data/cvqa/$L.jsonl" \
      --images-dir "$DT/Stage3/data/cvqa/images" \
      --ckpt "$CKPT" \
      --vis-layers "9,18,-1" \
      --output-path "$out" \
      --local-files-only \
      ${args[@]+"${args[@]}"}; then
    echo "### $L $q $img FAILED — continuing"
    FAILED+=("$L:$q:$img")
  fi
}

for L in $LANGS; do
  for q in wq qb; do
    run_cell "$L" "$q" correct
    run_cell "$L" "$q" gray
  done
done

mkdir -p "$RESULTS_DIR"
cp "$OUT_DIR"/eval_cvqa_*_"$TAG".jsonl "$RESULTS_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/eval_cvqa_*_"$TAG".jsonl.summary.json "$RESULTS_DIR/" 2>/dev/null || true
echo "=== Done === $(date): $RAN cells evaluated"
[ "$RAN" -gt 0 ] || { echo "ERROR: nothing ran; every cell already had a summary"; exit 1; }
echo "Harvest with: bash Approach2/job-scripts/harvest.sh"
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED: ${FAILED[*]}"; exit 1; }
