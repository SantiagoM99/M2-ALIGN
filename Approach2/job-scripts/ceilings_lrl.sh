#!/bin/bash
#SBATCH --job-name=a2_ceil_lrl
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_ceil_lrl_%j.log

# The measurement the whole approach rests on.
#
# Gemma-2-9b-it alone scores 74.8 on Bengali MGSM; our best bridged system
# scores 62.0. So in Bengali the bridge still SUBTRACTS — everything D6/D9/D11
# bought is recovery of damage the prefix itself causes, not gain. The prefix
# can only pay for itself in a language the frozen LLM cannot read on its own.
#
# jv, mn, si and ga are the four candidates and none has ever had its ceiling
# measured. If any of them has a LOW ceiling, that is where the bridge has
# positive value and the approach is justified on text. If all four are high,
# we learn it now and reframe toward the multimodal side, where Gemma alone
# scores nothing because it cannot see.
#
# Packed into one GPU allocation:
#   [1] NLLB-translate MGSM/MSVAMP from English into the four languages
#       (numeric gold answers are carried over untouched, so MT cannot
#       corrupt the label — see build_translated_benchmark.py)
#   [2] frozen-Gemma ceiling for each: --no-mapping, no checkpoint, no training
#
# Prereq (login node, needs network):
#   python3 Approach2/fetch_text_benchmarks.py --langs en --out-dir evaluation
#
# Env: LANGS (default "jv mn si ga"), EVAL_DIR.
# Idempotent: existing translations and existing summaries are skipped.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
EVAL_DIR="${EVAL_DIR:-$PROJECT_ROOT/evaluation}"
LANGS="${LANGS:-jv mn si ga}"
OUT_BASE="${OUT_BASE:-$A2/outputs}"

LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"

declare -A TAG=( [jv]=jav_Latn [mn]=khk_Cyrl [si]=sin_Sinh [ga]=gle_Latn
                 [bn]=ben_Beng [sw]=swh_Latn )

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="
date; hostname
echo "LANGS=$LANGS EVAL_DIR=$EVAL_DIR"
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

for B in MGSM MSVAMP; do
  [ -f "$EVAL_DIR/${B}_en.jsonl" ] || {
    echo "ERROR: missing $EVAL_DIR/${B}_en.jsonl"
    echo "Run on a LOGIN node first (needs network):"
    echo "  python3 Approach2/fetch_text_benchmarks.py --langs en --out-dir evaluation"
    exit 1
  }
done

cd "$A2"

echo "=== [1] Translate the benchmarks ==="
for L in $LANGS; do
  t="${TAG[$L]:-}"
  [ -n "$t" ] || { echo "### $L: no NLLB tag, skipping"; continue; }
  for B in MGSM MSVAMP; do
    out="$EVAL_DIR/${B}_${L}.jsonl"
    if [ -f "$out" ]; then echo "--- $B $L exists"; continue; fi
    echo "=== translate $B -> $L ($t) === $(date)"
    python -u build_translated_benchmark.py \
      --input "$EVAL_DIR/${B}_en.jsonl" \
      --output "$out" \
      --nllb-tag "$t" \
      --mt-path "$MT_PATH" \
      --local-files-only || echo "### $B $L translation FAILED"
  done
done

echo "=== [2] Frozen-LLM ceilings (no mapping) ==="
FAILED=()
for L in $LANGS; do
  OUT="$OUT_BASE/text_eval_$L"
  mkdir -p "$OUT"
  for bench in mgsm msvamp; do
    case "$bench" in
      mgsm)   data="$EVAL_DIR/MGSM_${L}.jsonl" ;;
      msvamp) data="$EVAL_DIR/MSVAMP_${L}.jsonl" ;;
    esac
    [ -f "$data" ] || { echo "--- skip $bench $L (no data)"; continue; }
    out="$OUT/eval_${bench}_${L}_gemma.jsonl"
    [ -f "$out.summary.json" ] && { echo "--- skip $bench $L (summary exists)"; continue; }
    echo "=== ceiling $bench $L === $(date)"
    python -u evaluate_text.py \
      --data-path "$data" \
      --benchmark "$bench" \
      --output-path "$out" \
      --no-mapping \
      --mt-path "$MT_PATH" \
      --llm-path "$LLM_PATH" \
      --local-files-only || FAILED+=("$bench:$L")
  done
done

echo "=== Summary ==="
for L in $LANGS; do
  for s in "$OUT_BASE/text_eval_$L"/eval_*_gemma.jsonl.summary.json; do
    [ -f "$s" ] || continue
    printf '%-46s ' "$(basename "$s")"
    python - "$s" <<'PY'
import json,sys
print(f"{json.load(open(sys.argv[1]))['accuracy']*100:.1f}")
PY
  done
done

echo "=== Harvest ==="
RESULTS_DIR="$A2/results"
mkdir -p "$RESULTS_DIR"
for L in $LANGS; do
  cp "$OUT_BASE/text_eval_$L"/eval_*_gemma.jsonl.summary.json "$RESULTS_DIR/" 2>/dev/null || true
done
cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: frozen-LLM ceilings for $LANGS (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."

echo "=== Done === $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
