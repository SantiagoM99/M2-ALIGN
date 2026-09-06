#!/bin/bash
#SBATCH --job-name=a2_srcabl
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_srcabl_%j.log

# X1 — Does the transfer failure belong to the TARGET or to the SOURCE?
#
# E2 found zero-shot multimodal transfer from Bengali works for ru/zh/pt/id/si
# and is completely flat for jv/mn/ga (pooled n=935, p=0.36). Two hypotheses
# survive, and they prescribe different papers:
#
#   H_source  jv/mn/ga fail because BENGALI is a bad source for them —
#             the failure is a property of the source-target pair. Then the
#             fix is source selection, or training from several sources.
#   H_align   jv/mn/ga fail because their own text bridge does not land where
#             the visual prefix expects it, whatever the source. Then the fix
#             has to change how their text mapping is trained (D12).
#
# Discriminator: transfer into the same three targets from sources that are
# typologically or genealogically closer. Javanese from Indonesian is the
# sharpest single cell — both Austronesian, and Indonesian retains 99% on
# xGQA, so the source bridge is known to be healthy.
#
#   H_source predicts: id->jv works, bn->jv does not.
#   H_align  predicts: jv fails from every source alike.
#
# This costs NO training. All eleven stage3_<L>_v4 checkpoints already exist;
# this is evaluation only. Sinhala is included as a positive control: it
# retained 86% from Bengali, so it must keep transferring from other sources
# too — if si also collapses under a new source, the harness is at fault
# rather than the languages.
#
# Outputs go to the same outputs/zeroshot_<SRC>_<ROUND> that
# zeroshot_transfer.sh uses, so SRC=bn is free: those summaries already exist
# and every one of them is skipped.
#
# Env: DT (required), SOURCES ("bn id ru zh"), TARGETS ("jv mn ga si"),
#      ROUND (v4), BENCH (cvqa), VIS_LAYERS ("9,18,-1" — must match the ckpt).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT}"
SOURCES="${SOURCES:-bn id ru zh}"
TARGETS="${TARGETS:-jv mn ga si}"
ROUND="${ROUND:-v4}"
BENCH="${BENCH:-cvqa}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
GQA_IMAGES="${GQA_IMAGES:-$DT/Stage3/data/gqa/images}"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$PROJECT_ROOT/Stage3/data/gqa/images"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="; date; hostname
echo "SOURCES=$SOURCES TARGETS=$TARGETS ROUND=$ROUND BENCH=$BENCH"
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

if [ "$BENCH" = cvqa ]; then
  SCRIPT="evaluate_cvqa.py"; DATA_DIR="$DT/Stage3/data/cvqa"; IMAGES="$DT/Stage3/data/cvqa/images"
else
  SCRIPT="evaluate_vqa.py";  DATA_DIR="$DT/Stage3/data/xgqa";  IMAGES="$GQA_IMAGES"
fi

FAILED=()
for SRC in $SOURCES; do
  CKPT="$A2/outputs/stage3_${SRC}_${ROUND}/mapping/pytorch_model.bin"
  # Bengali's v4 run is named stage3_bn_dcl (D11 pilot), not stage3_bn_v4.
  [ -f "$CKPT" ] || CKPT="$A2/outputs/stage3_${SRC}_dcl/mapping/pytorch_model.bin"
  if [ ! -f "$CKPT" ]; then echo "!!! no checkpoint for source $SRC -> skipping source"; continue; fi
  OUT="$A2/outputs/zeroshot_${SRC}_${ROUND}"
  mkdir -p "$OUT"
  for L in $TARGETS; do
    [ "$L" = "$SRC" ] && { echo "--- skip $L (it is the source)"; continue; }
    data="$DATA_DIR/$L.jsonl"
    [ -f "$data" ] || { echo "--- skip $BENCH $L (no data)"; continue; }
    for blind in 0 1; do
      suffix=""; args=()
      [ "$blind" = 1 ] && { suffix="_BLIND"; args+=(--blind); }
      out="$OUT/eval_${BENCH}_${L}_zs${SRC}${suffix}.jsonl"
      [ -f "$out.summary.json" ] && { echo "--- skip $BENCH $L blind=$blind (from $SRC, exists)"; continue; }
      echo "=== $BENCH $L blind=$blind (zero-shot from $SRC) === $(date)"
      python -u "$SCRIPT" \
        --data-path "$data" --images-dir "$IMAGES" \
        --ckpt "$CKPT" --output-path "$out" \
        --mt-path "$MT_PATH" --vis-path "$VIS_PATH" --llm-path "$LLM_PATH" \
        --vis-layers "$VIS_LAYERS" --local-files-only "${args[@]}" \
        || FAILED+=("$SRC:$L:$blind")
    done
  done
done

echo "=== Harvest ==="
mkdir -p "$A2/results"
for SRC in $SOURCES; do
  OUT="$A2/outputs/zeroshot_${SRC}_${ROUND}"
  [ -d "$OUT" ] || continue
  cp "$OUT"/eval_*.summary.json "$A2/results/" 2>/dev/null || true
  cp "$OUT"/eval_*.jsonl "$A2/results/" 2>/dev/null || true
done

echo
echo "=== Source x target dV matrix ($BENCH) ==="
python - "$A2/results" "$SOURCES" "$TARGETS" "$BENCH" <<'PY'
import json, os, sys
res, sources, targets, bench = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4]
def acc(p):
    try:
        with open(p) as f: return json.load(f)["accuracy"] * 100
    except Exception: return None
hdr = "src\\tgt"
print(f"{hdr:<9}" + "".join(f"{t:>9}" for t in targets))
for s in sources:
    cells = []
    for t in targets:
        if t == s: cells.append("  (src)"); continue
        f = acc(f"{res}/eval_{bench}_{t}_zs{s}.jsonl.summary.json")
        b = acc(f"{res}/eval_{bench}_{t}_zs{s}_BLIND.jsonl.summary.json")
        cells.append(f"{f-b:+.2f}" if (f is not None and b is not None) else "--")
    print(f"{s:<9}" + "".join(f"{c:>9}" for c in cells))
print("\ndV = full - blind. The supervised reference for each target is in")
print("results/eval_%s_<tgt>_v4 vs _BLIND_v4. Read this with:" % bench)
print("  cd Approach2/results && python3 ../analysis/source_ablation_report.py")
PY

cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: X1 source ablation, ${BENCH} transfer from {${SOURCES// /,}} (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; } || exit 0
