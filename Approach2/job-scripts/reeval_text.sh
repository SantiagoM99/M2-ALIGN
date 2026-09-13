#!/bin/bash
#SBATCH --job-name=a2_reeval_text
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a2_reeval_text_%j.log

# Re-evaluate an existing stage-3 checkpoint on MGSM and MSVAMP with the
# CURRENT code and environment, without retraining (2026-09-13).
#
# The GSM8K replay control lost 22.8 MGSM and 10.1 MSVAMP points against
# stage3_bn_dcl while xGQA reproduced exactly. That difference mixes training
# with evaluation: dcl's 62.0 was measured before model.generate changed on
# 09-09, and nothing so far shows that long chain-of-thought generation
# reproduces across that change (the gray cells test choice scoring, xGQA tests
# 1-3 token answers). Re-evaluating dcl here separates the two: near 62 means
# the loss is in training; near the control means it is in evaluation and no
# checkpoint trained after 09-09 is implicated.
#
# Env: CKPT_NAME (stage3_bn_dcl), TAG (dcl_tf5), NLLB_TAG (ben_Beng).
# Outputs go to outputs/<CKPT_NAME>/eval_tf5 and are copied to results as
# eval_<bench>_bn_stage3_<TAG>.jsonl, never over the historical files.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
CKPT_NAME="${CKPT_NAME:-stage3_bn_dcl}"
TAG="${TAG:-dcl_tf5}"
NLLB_TAG="${NLLB_TAG:-ben_Beng}"
CKPT="$A2/outputs/$CKPT_NAME/mapping/pytorch_model.bin"
OUT="$A2/outputs/$CKPT_NAME/eval_tf5"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"

echo "=== Job info ==="; date; hostname
echo "CKPT=$CKPT TAG=$TAG"
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

[ -f "$CKPT" ] || { echo "ERROR: missing $CKPT"; exit 1; }
mkdir -p "$OUT"
cd "$A2"
FAILED=0
for spec in "mgsm:$PROJECT_ROOT/evaluation/MGSM.jsonl" "msvamp:$PROJECT_ROOT/evaluation/MSVAMP.jsonl"; do
  bench="${spec%%:*}"; data="${spec#*:}"
  out="$OUT/eval_${bench}_bn_stage3_${TAG}.jsonl"
  [ -f "$out.summary.json" ] && { echo "--- skip $bench (summary exists)"; continue; }
  [ -f "$data" ] || { echo "ERROR: missing $data"; FAILED=1; continue; }
  echo "=== $bench === $(date)"
  python -u evaluate_text.py --data-path "$data" --benchmark "$bench" \
    --output-path "$out" --ckpt "$CKPT" --nllb-tag "$NLLB_TAG" \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || { echo "### $bench FAILED"; FAILED=1; }
done

cp "$OUT"/eval_*_bn_stage3_"$TAG".jsonl "$A2/results/" 2>/dev/null || true
cp "$OUT"/eval_*_bn_stage3_"$TAG".jsonl.summary.json "$A2/results/" 2>/dev/null || true
echo "Harvested into Approach2/results; review and commit by hand."
git -C "$PROJECT_ROOT" status --short Approach2/results | head
echo "=== Done === $(date)"
exit "$FAILED"
