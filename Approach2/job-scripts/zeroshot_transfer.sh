#!/bin/bash
#SBATCH --job-name=a2_zeroshot
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_zeroshot_%j.log

# Zero-shot cross-lingual multimodal transfer.
#
# The question this architecture uniquely affords: can a language get VQA
# with NO multimodal data of its own? Language and modality enter through
# separate bridges here — the vision mapping is trained on ENGLISH captions
# and knows nothing about languages; the text mapping is trained on bitext
# and knows nothing about images. mBLIP needs multimodal data machine-
# translated into 95 languages; Qwen3-VL got it in pretraining. For the long
# tail neither is available.
#
# So: take a stage-3 checkpoint trained on VQA in ONE language and evaluate
# it on another language's VQA. The evaluator reads the NLLB tag from each
# data row (`row_nllb_tag`), so nothing else has to change.
#
# Three arms, two of which already exist:
#   zero-shot   this job          — never saw VQA in the target language
#   supervised  stage3_<L>_<ROUND> — the upper reference, already computed
#   blind       this job, --blind  — the language prior with no pixels
# The metric is not accuracy: it is what fraction of the supervised arm's
# dV (full - blind) the zero-shot arm retains.
#
# Our own D11 measurement predicts this may fail: joint stage 3 entangles the
# two mappings (update-direction cosine 0.93 same-recipe vs 0.51 across
# stage-2 checkpoints). A null result is evidence for that entanglement.
#
# Env: DT (required), SRC (bn), ROUND (v4), LANGS ("jv mn si ga"),
#      BENCHES ("cvqa"), VIS_LAYERS ("9,18,-1" — must match the checkpoint).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
SRC="${SRC:-bn}"
ROUND="${ROUND:-v4}"
LANGS="${LANGS:-jv mn si ga}"
BENCHES="${BENCHES:-cvqa}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

CKPT="$A2/outputs/stage3_${SRC}_${ROUND}/mapping/pytorch_model.bin"
OUT="$A2/outputs/zeroshot_${SRC}_${ROUND}"
TAG="zs${SRC}"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="; date; hostname
echo "SRC=$SRC ROUND=$ROUND LANGS=$LANGS BENCHES=$BENCHES"
echo "checkpoint: $CKPT"
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

[ -f "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }
mkdir -p "$OUT"
cd "$A2"

FAILED=()
for bench in $BENCHES; do
  for L in $LANGS; do
    [ "$L" = "$SRC" ] && { echo "--- skip $L (it is the source)"; continue; }
    if [ "$bench" = cvqa ]; then
      script="evaluate_cvqa.py"; data="$DT/Stage3/data/cvqa/$L.jsonl"; images="$DT/Stage3/data/cvqa/images"
    else
      script="evaluate_vqa.py";  data="$DT/Stage3/data/xgqa/$L.jsonl"; images="$GQA_IMAGES"
    fi
    [ -f "$data" ] || { echo "--- skip $bench $L (no data)"; continue; }
    for blind in 0 1; do
      suffix=""; args=()
      [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
      out="$OUT/eval_${bench}_${L}_${TAG}${suffix}.jsonl"
      [ -f "$out.summary.json" ] && { echo "--- skip $bench $L blind=$blind"; continue; }
      echo "=== $bench $L blind=$blind (zero-shot from $SRC) === $(date)"
      python -u "$script" \
        --data-path "$data" --images-dir "$images" \
        --ckpt "$CKPT" --output-path "$out" \
        --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
        --vis-layers "$VIS_LAYERS" --local-files-only "${args[@]}" \
        || FAILED+=("$bench:$L:$blind")
    done
  done
done

echo "=== Zero-shot from $SRC — dV is the number that matters ==="
python - "$OUT" "$TAG" "$LANGS" <<'PY'
import json, os, sys
out, tag, langs = sys.argv[1], sys.argv[2], sys.argv[3].split()
def acc(p):
    try:
        with open(p) as f: return json.load(f)["accuracy"] * 100
    except Exception: return None
for bench in ("cvqa", "xgqa"):
    rows = []
    for L in langs:
        f = acc(f"{out}/eval_{bench}_{L}_{tag}.jsonl.summary.json")
        b = acc(f"{out}/eval_{bench}_{L}_{tag}_BLIND.jsonl.summary.json")
        if f is not None and b is not None:
            rows.append((L, f, b))
    if not rows: continue
    print(f"\n{bench.upper()} zero-shot")
    print(f"{'lang':6} {'full':>7} {'blind':>7} {'dV':>7}")
    for L, f, b in rows:
        print(f"{L:6} {f:7.2f} {b:7.2f} {f-b:+7.2f}")
PY

echo "=== Harvest ==="
mkdir -p "$A2/results"
cp "$OUT"/eval_*.summary.json "$A2/results/" 2>/dev/null || true
cp "$OUT"/eval_*.jsonl "$A2/results/" 2>/dev/null || true
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: zero-shot cross-lingual multimodal transfer from $SRC (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; } || exit 0
