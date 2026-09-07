#!/bin/bash
#SBATCH --job-name=a2_pairalign
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=Approach2/logs/a2_pairalign_%j.log

# X2b — pairwise cross-lingual alignment on FLORES-200.
#
# X2 measured a TARGET-only quantity and correlated it against retention from
# a single source. X1 then showed transfer is a property of the source-target
# PAIR, so that design could not have worked. This measures the pair: how
# close is target T's prefix to source S's prefix, on the same sentence.
#
# Pre-registered (DESIGN.md, X2): the pair score must rank id above bn and ru
# as a donor for jv/mn/ga, and must reproduce the one clean dissociation X1
# produced — zh high for ga (87% retention) and low for mn (0%). If it cannot
# reproduce that, the alignment mechanism is abandoned rather than
# re-instrumented a third time.
#
# REQUIRES evaluation/flores_dev.jsonl. Compute nodes have no network, so
# build it first on a LOGIN node:
#     python3 Approach2/fetch_flores.py --out-dir evaluation
#
# Env: FLORES (evaluation/flores_dev.jsonl), CKPTS (which mappings to score).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
A2="$PROJECT_ROOT/Approach2"
FLORES="${FLORES:-$PROJECT_ROOT/evaluation/flores_dev.jsonl}"
LLM_PATH="${LLM_PATH:-google/gemma-2-9b-it}"
MT_PATH="${MT_PATH:-facebook/nllb-200-distilled-600M}"

if [ -d "$MT_PATH" ]; then
  for d in "$MT_PATH"/*; do
    if [ -d "$d" ]; then MT_PATH="$d"; break; fi
  done
fi

echo "=== Job info ==="; date; hostname
if [ ! -f "$FLORES" ]; then
  echo "ERROR: no FLORES file at $FLORES"
  echo "Build it on a LOGIN node: python3 Approach2/fetch_flores.py --out-dir evaluation"
  exit 1
fi
echo "FLORES: $FLORES ($(wc -l < "$FLORES") sentences)"
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

# stage3_bn_dcl is the bridge that produced every transfer number so far, so
# it is the one the mechanism claim is about. stage1_joint prices D12 before
# eleven stage-3 runs. The rest are for contrast.
CKPTS="${CKPTS:-stage3_bn_dcl stage1_joint stage1 stage3_id_v4}"

FAILED=()
for name in $CKPTS; do
  ckpt="$A2/outputs/$name/mapping/pytorch_model.bin"
  out="$A2/results/pairalign_${name}.json"
  [ -f "$ckpt" ] || { echo "--- skip $name (no checkpoint)"; continue; }
  [ -f "$out" ]  && { echo "--- skip $name (already scored)"; continue; }
  echo "=== scoring $name === $(date)"
  python -u pair_alignment.py \
    --ckpt "$ckpt" --label "$name" --flores "$FLORES" --output "$out" \
    --mt-path "$MT_PATH" --llm-path "$LLM_PATH" --local-files-only \
    || FAILED+=("$name")
done

echo
echo "=== Read it with: cd Approach2/results && python3 ../analysis/donor_matrix.py ==="

cd "$PROJECT_ROOT"
git add Approach2/results 2>/dev/null || true
git commit -m "results: X2b pairwise FLORES alignment (job ${SLURM_JOB_ID:-manual})" Approach2/results \
  || echo "No new results to commit."
echo "=== Done === $(date)"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; } || exit 0
