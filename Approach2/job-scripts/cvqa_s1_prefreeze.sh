#!/bin/bash
#SBATCH --job-name=a2_cvqa_s1_pre
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_cvqa_s1_pre_%j.log

# Close the two remaining S1 pre-freeze checks in one offline job:
#
#   1. rebuild Bengali from the CVQA parquet and compare the exact canonical
#      id -> native-query map with the JSONL used by existing cluster runs;
#   2. build the full Spanish unit from embedded parquet images (the largest
#      confirmatory unit; its cold extraction is what is timed) and time two
#      correct-image evaluations of a NON-panel unit with stage3_bn_dcl: the
#      full unit and a --limit run, so the per-item cost and the model-load
#      overhead are separated. Both timed runs are preceded by one untimed
#      --limit warm-up run: the first load in an allocation reads ~20 GB of
#      weights from Lustre (356 s in job 20443726) while later loads come from
#      the page cache (12 s), and the two-point subtraction only holds when
#      both timed runs share the same load cost. The warm-up's own wall time
#      is recorded as the cold first-load overhead per allocation.
#      The timing unit defaults to Japanese (203 items,
#      eligible but over the cap) on purpose: S1 Block D forbids evaluating any
#      donor on a confirmatory-panel unit before the damage ranking is
#      committed, and the builder refuses a panel unit here.
#
# The parquet must already be on the shared filesystem: compute nodes have no
# network. On a login node, once (≈ 4.9 GB):
#
#   hf download afaji/cvqa --repo-type dataset \
#     --include 'data/test-*.parquet' --local-dir $SCRATCH/cvqa
#   # (`huggingface-cli` is deprecated and no longer works in huggingface_hub
#   # >= 1.x, which the venv has); or, if `hf` is not on PATH inside the venv:
#   python -c "from huggingface_hub import snapshot_download as d; \
#     d('afaji/cvqa', repo_type='dataset', allow_patterns=['data/test-*.parquet'], \
#       local_dir='$SCRATCH/cvqa')"
#
# The repository holds exactly data/test-00000-of-00010.parquet ... -00009
# (verified against the Hub API on 2026-09-07). Submit from a CLEAN,
# committed checkout at the repo root:
#
#   DT=/scratch/santimn/datatransfer \
#   CVQA_PARQUET="$SCRATCH/cvqa/data/test-*.parquet" \
#     sbatch --export=ALL Approach2/job-scripts/cvqa_s1_prefreeze.sh
#
# The scratch output is keyed by the code commit and is resumable.  The
# versioned report is only copied into the checkout after both checks pass.
# Optional env: CURRENT_JSONL, CKPT, OUT_ROOT, REPORT, EXTRACT_UNIT (Spanish),
# EVAL_UNIT (Japanese), LIMIT_N (40), CHECK_UNIT, CHECK_CODE, MT_PATH, VIS_PATH,
# LLM_PATH, VIS_LAYERS.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
DT="${DT:?set DT to the datatransfer root}"
CVQA_PARQUET="${CVQA_PARQUET:?set CVQA_PARQUET to local parquet file(s), directory, or glob}"

EXTRACT_UNIT="${EXTRACT_UNIT:-Spanish}"
EVAL_UNIT="${EVAL_UNIT:-Japanese}"
LIMIT_N="${LIMIT_N:-40}"
CHECK_UNIT="${CHECK_UNIT:-Bengali}"
CHECK_CODE="${CHECK_CODE:-bn}"
CURRENT_JSONL="${CURRENT_JSONL:-$DT/Stage3/data/cvqa/$CHECK_CODE.jsonl}"

REPORT="${REPORT:-$A2/audits/cvqa_s1_prefreeze.json}"
INVENTORY="${INVENTORY:-$A2/audits/cvqa_inventory.json}"

CKPT="${CKPT:-$A2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin}"
LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
VIS_PATH="${VIS_PATH:-google/siglip2-so400m-patch14-384}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  echo "ERROR: PROJECT_ROOT is not the M2-ALIGN checkout: $PROJECT_ROOT"
  exit 1
fi
# The report this job writes into the checkout is the one file allowed to be
# dirty, so a completed job can be re-run for its resumable checks.
DIRTY=$(git -C "$PROJECT_ROOT" status --porcelain | grep -v 'Approach2/audits/cvqa_s1_prefreeze.json' || true)
if [ -n "$DIRTY" ]; then
  echo "ERROR: pre-freeze evidence must be produced from a clean committed tree."
  echo "$DIRTY"
  exit 1
fi
CODE_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
OUT_ROOT="${OUT_ROOT:-$DT/Stage3/data/cvqa_s1_prefreeze_$CODE_HEAD}"
DATA_OUT="$OUT_ROOT/data"
IMAGES_OUT="$OUT_ROOT/images"
PILOT_OUT_DIR="$OUT_ROOT/pilot"
WORK_REPORT="$OUT_ROOT/cvqa_s1_prefreeze.work.json"
# A report is the completion marker for the build.  If an earlier allocation
# died mid-extraction, leave its partial files untouched and start cold in a
# fresh attempt directory; a completed build remains resumable across jobs.
if [ ! -f "$WORK_REPORT" ] && { [ -d "$DATA_OUT" ] || [ -d "$IMAGES_OUT" ]; }; then
  OUT_ROOT="${OUT_ROOT}_retry_${SLURM_JOB_ID:-manual}"
  DATA_OUT="$OUT_ROOT/data"
  IMAGES_OUT="$OUT_ROOT/images"
  PILOT_OUT_DIR="$OUT_ROOT/pilot"
  WORK_REPORT="$OUT_ROOT/cvqa_s1_prefreeze.work.json"
fi
for path in "$CURRENT_JSONL" "$INVENTORY" "$CKPT"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: required file not found: $path"
    exit 1
  fi
done

if [ -d "$MT_PATH" ]; then
  for directory in "$MT_PATH"/*; do
    if [ -d "$directory" ]; then MT_PATH="$directory"; break; fi
  done
fi

echo "=== S1 CVQA pre-freeze job ==="
date --utc
hostname
echo "HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "parquet=$CVQA_PARQUET"
echo "current query reference=$CURRENT_JSONL"
echo "output root=$OUT_ROOT"
echo "report=$REPORT"
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

mkdir -p "$DATA_OUT" "$IMAGES_OUT" "$PILOT_OUT_DIR" "$(dirname "$REPORT")"
cd "$PROJECT_ROOT"

echo "=== Build Spanish and audit rebuilt Bengali queries ==="
if [ -f "$WORK_REPORT" ]; then
  echo "Reusing completed data build: $WORK_REPORT"
else
  python -u Approach2/build_cvqa_s1.py build \
    --parquet "$CVQA_PARQUET" \
    --inventory "$INVENTORY" \
    --unit "$EXTRACT_UNIT" \
    --unit "$EVAL_UNIT" \
    --unit "$CHECK_UNIT" \
    --extract-images-for "$EXTRACT_UNIT" \
    --extract-images-for "$EVAL_UNIT" \
    --compare-existing "$CHECK_UNIT=$CURRENT_JSONL" \
    --output-dir "$DATA_OUT" \
    --images-dir "$IMAGES_OUT" \
    --report "$WORK_REPORT" \
    --repo-root "$PROJECT_ROOT"
fi
python - "$WORK_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
comparisons = report.get("comparisons")
if not comparisons or any(not row.get("passed") for row in comparisons):
    raise SystemExit("ERROR: rebuilt-panel query comparison did not pass")
if report.get("pilot") is not None:
    print("Existing work report already includes a pilot; provenance will be revalidated.")
PY

PILOT_SLUG=$(printf '%s' "$EVAL_UNIT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')
PILOT_DATA="$DATA_OUT/$PILOT_SLUG.jsonl"
PILOT_OUTPUT="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct.jsonl"
TIME_FILE="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct.seconds"
LIM_OUTPUT="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct_limit${LIMIT_N}.jsonl"
LIM_TIME_FILE="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct_limit${LIMIT_N}.seconds"

if [ ! -f "$PILOT_DATA" ]; then
  echo "ERROR: builder did not produce $PILOT_DATA"
  exit 1
fi

WARMUP_OUTPUT="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct_warmup_limit${LIMIT_N}.jsonl"
WARMUP_TIME_FILE="$PILOT_OUT_DIR/eval_cvqa_${PILOT_SLUG}_zsbn_correct_warmup_limit${LIMIT_N}.seconds"

NEED_FULL=1
NEED_LIM=1
if [ -f "$PILOT_OUTPUT" ] && [ -f "$PILOT_OUTPUT.summary.json" ] && [ -f "$TIME_FILE" ]; then NEED_FULL=0; fi
if [ -f "$LIM_OUTPUT.summary.json" ] && [ -f "$LIM_TIME_FILE" ]; then NEED_LIM=0; fi

# Any timed run executed in this allocation needs the weights in the page
# cache first, otherwise the first run pays a cold Lustre read that the second
# run does not, and the two-point subtraction returns a negative load cost.
if [ "$NEED_FULL" = 1 ] || [ "$NEED_LIM" = 1 ]; then
  echo "=== Untimed warm-up: --limit $LIMIT_N run to bring the weights into the page cache ==="
  WARMUP_TMP="$WARMUP_OUTPUT.tmp.${SLURM_JOB_ID:-manual}"
  WARMUP_START=$(date +%s)
  cd "$A2"
  python -u evaluate_cvqa.py \
    --data-path "$PILOT_DATA" \
    --images-dir "$IMAGES_OUT" \
    --ckpt "$CKPT" \
    --output-path "$WARMUP_TMP" \
    --mt-path "$MT_PATH" \
    --vis-path "$VIS_PATH" \
    --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" \
    --limit "$LIMIT_N" \
    --local-files-only
  WARMUP_END=$(date +%s)
  mv "$WARMUP_TMP" "$WARMUP_OUTPUT"
  mv "$WARMUP_TMP.summary.json" "$WARMUP_OUTPUT.summary.json"
  printf '%s\n' "$((WARMUP_END - WARMUP_START))" > "$WARMUP_TIME_FILE"
fi

echo "=== Timed correct-image evaluations: $EVAL_UNIT from Bengali (full, then --limit $LIMIT_N) ==="
if [ "$NEED_FULL" = 0 ]; then
  echo "Reusing completed timed evaluation: $PILOT_OUTPUT"
  EVAL_SECONDS=$(tr -d '[:space:]' < "$TIME_FILE")
else
  PILOT_TMP="$PILOT_OUTPUT.tmp.${SLURM_JOB_ID:-manual}"
  EVAL_START=$(date +%s)
  cd "$A2"
  python -u evaluate_cvqa.py \
    --data-path "$PILOT_DATA" \
    --images-dir "$IMAGES_OUT" \
    --ckpt "$CKPT" \
    --output-path "$PILOT_TMP" \
    --mt-path "$MT_PATH" \
    --vis-path "$VIS_PATH" \
    --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" \
    --local-files-only
  EVAL_END=$(date +%s)
  EVAL_SECONDS=$((EVAL_END - EVAL_START))
  mv "$PILOT_TMP" "$PILOT_OUTPUT"
  mv "$PILOT_TMP.summary.json" "$PILOT_OUTPUT.summary.json"
  printf '%s\n' "$EVAL_SECONDS" > "$TIME_FILE"
fi
if [ "$NEED_LIM" = 0 ]; then
  echo "Reusing completed limited evaluation: $LIM_OUTPUT"
  LIM_SECONDS=$(tr -d '[:space:]' < "$LIM_TIME_FILE")
else
  LIM_TMP="$LIM_OUTPUT.tmp.${SLURM_JOB_ID:-manual}"
  LIM_START=$(date +%s)
  cd "$A2"
  python -u evaluate_cvqa.py \
    --data-path "$PILOT_DATA" \
    --images-dir "$IMAGES_OUT" \
    --ckpt "$CKPT" \
    --output-path "$LIM_TMP" \
    --mt-path "$MT_PATH" \
    --vis-path "$VIS_PATH" \
    --llm-path "$LLM_PATH" \
    --vis-layers "$VIS_LAYERS" \
    --limit "$LIMIT_N" \
    --local-files-only
  LIM_END=$(date +%s)
  LIM_SECONDS=$((LIM_END - LIM_START))
  mv "$LIM_TMP" "$LIM_OUTPUT"
  mv "$LIM_TMP.summary.json" "$LIM_OUTPUT.summary.json"
  printf '%s\n' "$LIM_SECONDS" > "$LIM_TIME_FILE"
fi

cd "$PROJECT_ROOT"
WARMUP_ARGS=()
if [ -f "$WARMUP_OUTPUT.summary.json" ] && [ -f "$WARMUP_TIME_FILE" ]; then
  WARMUP_ARGS=(--warmup-eval-summary "$WARMUP_OUTPUT.summary.json"
               --warmup-elapsed-seconds "$(tr -d '[:space:]' < "$WARMUP_TIME_FILE")")
fi
python -u Approach2/build_cvqa_s1.py attach-pilot \
  --report "$WORK_REPORT" \
  --repo-root "$PROJECT_ROOT" \
  --eval-summary "$PILOT_OUTPUT.summary.json" \
  --elapsed-seconds "$EVAL_SECONDS" \
  --limited-eval-summary "$LIM_OUTPUT.summary.json" \
  --limited-elapsed-seconds "$LIM_SECONDS" \
  ${WARMUP_ARGS[@]+"${WARMUP_ARGS[@]}"} \
  --unit "$EVAL_UNIT" \
  --extraction-unit "$EXTRACT_UNIT" \
  --donor bn \
  --condition correct \
  --checkpoint "$CKPT" \
  --vis-layers "$VIS_LAYERS" \
  --donors 7 \
  --conditions 5
REPORT_TMP="$REPORT.tmp.${SLURM_JOB_ID:-manual}"
cp "$WORK_REPORT" "$REPORT_TMP"
mv "$REPORT_TMP" "$REPORT"

echo "=== Freeze evidence ready for review ==="
python - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
comparison = report["comparisons"][0]
pilot = report["pilot"]
print("query audit:", "PASS" if comparison["passed"] else "FAIL")
print("query sha256:", comparison["rebuilt_canonical_query_sha256"])
print("timing unit:", pilot["unit"], "| extraction unit:", pilot["extraction_unit"])
print("full-run seconds:", pilot["elapsed_seconds"], "| per item:", pilot["seconds_per_item"], "| model load (warm):", pilot["load_seconds"])
warmup = pilot.get("warmup_run")
print("cold first load per allocation:", warmup["cold_load_seconds"] if warmup else "not measured")
print("estimated donor confirmation GPU-h:", pilot["estimate"]["donor_confirmation_gpu_hours"])
print("report:", sys.argv[1])
PY

echo "Commit the completed report with the S1 freeze; do not launch S1 before that commit."
git -C "$PROJECT_ROOT" status --short
date --utc
