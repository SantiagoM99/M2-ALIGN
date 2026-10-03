#!/bin/bash
#SBATCH --job-name=a1_eval
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a1_eval_%j.log
#
# Evaluate a1 with HER evaluator, on OUR data, writing per-item results.
#
# This is what makes a1 a row in our tables rather than a number quoted from her
# message: same CVQA image copy, same item ids, same environment, so the
# comparison against v4 is paired and the question-necessity split (which needs
# two systems that are not the reference) becomes usable.
#
# **VISUAL_PIXELS matters more than it looks.** Her `Stage3/evaluate.py` defaults
# to `--visual-pixels 65536` (256x256) and her launcher does not override it, so
# her whole evaluation — baseline and a1 — runs at roughly 64 merged visual
# tokens: about a twenty-second of the processor's own default (~1400 on CVQA
# images, measured 10-01) and a eleventh of our 729. Our density sweep says that
# costs around a point, and her reported CVQA baseline sits ~1.7 below our
# measurement of the same model on the same languages, which is consistent. So
# this launcher runs her default first, for comparability with her numbers, and
# a matched-resolution point second, for comparability with ours.
#
#   bash/sbatch, from the repo root:
#   TAG=her_default sbatch Approach2/job-scripts/a1_eval.sh
#   VISUAL_PIXELS=602112 TAG=matched sbatch Approach2/job-scripts/a1_eval.sh
#
# The matched arm also needs a longer prompt budget; the launcher derives one
# from the resolution rather than leaving her 512 to truncate the image tokens.
#
# Env: BENCHMARK (cvqa|xgqa), LANGS, CKPT, VISUAL_PIXELS, MAX_LLM_SEQ_LEN, TAG,
#      A1_ROOT, DT
set -uo pipefail

ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
A2="$ROOT/Approach2"
DT="${DT:-/scratch/santimn/datatransfer}"
A1_ROOT="${A1_ROOT:-$SCRATCH/a1}"
BENCHMARK="${BENCHMARK:-cvqa}"
LANGS="${LANGS:-jv mn ga si bn}"
CKPT="${CKPT:-$A2/outputs/a1_bn_stage3/pytorch_model.bin}"
VISUAL_PIXELS="${VISUAL_PIXELS:-65536}"
# Raising the resolution without raising this truncates the prompt. Her
# `--max-llm-seq-len` defaults to 512 and her tokenizer truncates from the LEFT,
# so at 588 visual tokens the truncation eats part of the image placeholder run
# and the processor raises "Mismatch in `image` token count ... ids=[492] and
# text=[588]" on every item (job 22300403, 10-02). The merge ratio is 32x32 px
# per merged token, so the prompt needs the visual tokens plus room for the
# question and the choices.
# Nominal merged tokens. The real count varies with aspect ratio even at a fixed
# pixel budget (551-630 observed at 602112), so the budget below carries slack.
VISUAL_TOKENS=$(( VISUAL_PIXELS / 1024 ))
MAX_LLM_SEQ_LEN="${MAX_LLM_SEQ_LEN:-$(( VISUAL_TOKENS + 256 > 512 ? VISUAL_TOKENS + 256 : 512 ))}"
TAG="${TAG:-a1}"
MT="${MT:-facebook/nllb-200-distilled-600M}"
LLM="${LLM:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_DIR="$A2/outputs/a1_eval"
RESULTS_DIR="${RESULTS_DIR:-$A2/results}"

echo "=== Job info ==="; date; hostname; nvidia-smi || true

[ -f "$A1_ROOT/Stage3/evaluate.py" ] || {
  echo "ERROR: no worktree of her branch at $A1_ROOT."
  echo "  git fetch upstream && git worktree add $A1_ROOT upstream/parallel"
  exit 1
}
A1_SHA=$(git -C "$A1_ROOT" rev-parse HEAD)
[ -f "$CKPT" ] || { echo "ERROR: no a1 checkpoint at $CKPT"; exit 1; }

module --force purge
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/13.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT_DIR"
echo "her code $A1_SHA | ckpt $CKPT | visual_pixels $VISUAL_PIXELS (~$VISUAL_TOKENS tokens)"
echo "max_llm_seq_len $MAX_LLM_SEQ_LEN | tag $TAG"
[ "$MAX_LLM_SEQ_LEN" -gt "$VISUAL_TOKENS" ] || {
  echo "ERROR: max_llm_seq_len $MAX_LLM_SEQ_LEN does not even fit $VISUAL_TOKENS visual tokens"
  exit 1
}
cd "$A1_ROOT"
RAN=0
FAILED=()

for L in $LANGS; do
  if [ "$BENCHMARK" = xgqa ]; then
    DATA="$DT/Stage3/data/xgqa/$L.jsonl"
    IMAGE_ARGS=(--images-dir "$DT/Stage3/data/gqa/images")
  else
    DATA="$DT/Stage3/data/cvqa/$L.jsonl"
    IMAGE_ARGS=(--image-cache-dir "$DT/Stage3/data/cvqa/images")
  fi
  OUT="$OUT_DIR/eval_${BENCHMARK}_${L}_${TAG}.jsonl"
  if [ ! -f "$DATA" ]; then echo "--- skip $L (no $DATA)"; continue; fi
  # Only a non-empty output counts as done. Her evaluator opens --results-jsonl
  # before scoring, so a cell that crashes leaves a zero-byte file behind, and
  # treating that as finished made the rerun skip all five cells and exit 1
  # (job 22343574).
  if [ -s "$OUT" ]; then echo "--- skip $L ($(basename "$OUT") exists)"; continue; fi
  [ -e "$OUT" ] && { echo "--- $(basename "$OUT") is empty from an earlier failure; redoing"; rm -f "$OUT"; }
  echo "=== a1 $BENCHMARK $L -> $(basename "$OUT") === $(date)"
  RAN=$((RAN + 1))
  if ! python -u Stage3/evaluate.py \
      --benchmark "$BENCHMARK" \
      --lang "$L" \
      --eval-data "$DATA" \
      --mapping-ckpt "$CKPT" \
      --llm-path "$LLM" \
      --mt-path "$MT" \
      --visual-pixels "$VISUAL_PIXELS" \
      --max-llm-seq-len "$MAX_LLM_SEQ_LEN" \
      --results-jsonl "$OUT" \
      --local-files-only \
      "${IMAGE_ARGS[@]}"; then
    echo "### a1 $BENCHMARK $L FAILED — continuing"
    # Leave no stub behind: the next run must see this cell as undone.
    [ -s "$OUT" ] || rm -f "$OUT"
    FAILED+=("$L")
  fi
done

mkdir -p "$RESULTS_DIR"
cp "$OUT_DIR"/eval_*_"$TAG".jsonl "$RESULTS_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/eval_*_"$TAG".jsonl.summary.json "$RESULTS_DIR/" 2>/dev/null || true
echo "=== Done === $(date): $RAN evaluated, her_code=$A1_SHA"
[ "$RAN" -gt 0 ] || { echo "ERROR: nothing ran; every output already existed"; exit 1; }
echo "Harvest with: bash Approach2/job-scripts/harvest.sh"
[ ${#FAILED[@]} -eq 0 ] || { echo "FAILED: ${FAILED[*]}"; exit 1; }
