#!/bin/bash
#SBATCH --job-name=a1_replay_pool
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a1_replay_pool_%j.log
#
# Build the text replay pool her pooled+replay arm consumes, with her own
# `Stage3/load_replay_data.py` run unedited from the worktree. Her
# `load_replay_data.sh` cannot be used as it stands: it writes to
# /scratch/tajm and reads an NLLB snapshot under her home, so this is her
# script's argument list with our paths, the same relationship a1_train.sh
# has to her trainers.
#
# Job 23145191 died in two seconds on `missing .../replay/translation_bn.jsonl`:
# the replay arm has a prerequisite the pooled arm does not, and nothing built it.
#
# **Four languages, not her eleven.** Her builder writes a file per pooled
# language, but `train_pooled_replay.sh` reads only REPLAY_LANGS, which is
# `bn de ru zh` in her launcher and in ours, plus math_en. Building the seven
# the trainer never opens would spend NLLB on rows no arm sees; the recipe the
# checkpoint is trained on is identical either way. Set LANGS to build more.
#
# **NLLB-3.3B, and no silent fallback.** Her math questions are translated with
# the 3.3B model, the one that translated her Stage 3 VQA data, not the 600M our
# own pipeline runs. Substituting the smaller model would change her recipe
# without saying so, which is the error the 10-02 entry exists to prevent, so a
# missing snapshot is a failure here rather than a fallback.
#
#   sbatch Approach2/job-scripts/a1_replay_pool.sh
#   N_MATH=2000 sbatch Approach2/job-scripts/a1_replay_pool.sh   (a cheaper pilot)
#
# Env: LANGS, N_TRANSLATION, N_MATH, INCLUDE_ENGLISH, OVERWRITE, MT, DT, A1_ROOT, S1_DATA
set -uo pipefail

ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DT="${DT:-/scratch/santimn/datatransfer}"
A1_ROOT="${A1_ROOT:-$SCRATCH/a1}"
LANGS="${LANGS:-bn,ru,de,zh}"
N_TRANSLATION="${N_TRANSLATION:-5000}"
N_MATH="${N_MATH:-5000}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"
OVERWRITE="${OVERWRITE:-0}"
S1_DATA="${S1_DATA:-$DT/Stage1/data}"
OUT_DIR="$DT/Stage3/data/replay"

echo "=== Job info ==="; date; hostname; nvidia-smi || true

[ -f "$A1_ROOT/Stage3/load_replay_data.py" ] || {
  echo "ERROR: no worktree of her branch at $A1_ROOT."
  echo "  git fetch upstream && git worktree add $A1_ROOT upstream/parallel"
  exit 1
}
A1_SHA=$(git -C "$A1_ROOT" rev-parse HEAD)

module --force purge
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/13.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# The snapshot hash differs per download, so discover it rather than pin it.
MT="${MT:-$(ls -d "$HF_HOME"/hub/models--facebook--nllb-200-3.3B/snapshots/*/ 2>/dev/null | head -1)}"
[ -n "$MT" ] && [ -d "$MT" ] || {
  echo "ERROR: no NLLB-200-3.3B snapshot under $HF_HOME/hub."
  echo "  Her replay pool is translated with 3.3B, not the 600M we use elsewhere."
  echo "  On a LOGIN node (compute nodes have no network):"
  echo "    HF_HUB_OFFLINE=0 huggingface-cli download facebook/nllb-200-3.3B"
  echo "  Or pass MT=/path/to/snapshot if it lives somewhere else."
  exit 1
}

# Her builder names Stage 1 files {Language}_to_English.jsonl; check the ones
# this run needs before spending a GPU hour on the first translation batch.
# A case, not an associative array: bash 3.2 has no `declare -A` and the repo
# keeps its launchers runnable on the laptop too.
stage1_name () {
  case "$1" in
    bn) echo Bengali ;;    ru) echo Russian ;;   de) echo German ;;
    zh) echo Chinese ;;    pt) echo Portuguese ;; id) echo Indonesian ;;
    ko) echo Korean ;;     jv) echo Javanese ;;  mn) echo Mongolian ;;
    si) echo Sinhalese ;;  ga) echo Irish ;;     *) echo "" ;;
  esac
}
MISSING=()
for code in ${LANGS//,/ }; do
  name="$(stage1_name "$code")"
  [ -n "$name" ] || { echo "ERROR: $code is not one of her pooled languages"; exit 1; }
  [ -f "$S1_DATA/${name}_to_English.jsonl" ] || MISSING+=("${name}_to_English.jsonl")
done
[ ${#MISSING[@]} -eq 0 ] || {
  echo "ERROR: $S1_DATA is missing ${MISSING[*]}"
  echo "  Those are Stage 1's own mapping rows, the translation half of the pool."
  echo "  Set S1_DATA to wherever this mirror keeps them."
  exit 1
}

mkdir -p "$OUT_DIR"
echo "her code $A1_SHA | langs $LANGS | n_translation $N_TRANSLATION | n_math $N_MATH"
echo "mt $MT"
echo "out $OUT_DIR"

EXTRA=(--local-files-only)
[ "$INCLUDE_ENGLISH" = 1 ] && EXTRA+=(--include-english-math)
[ "$OVERWRITE" = 1 ] && EXTRA+=(--overwrite)

cd "$A1_ROOT"
python -u Stage3/load_replay_data.py \
  --langs "$LANGS" \
  --stage1-dir "$S1_DATA" \
  --output-dir "$OUT_DIR" \
  --mt-path "$MT" \
  --n-translation "$N_TRANSLATION" \
  --n-math "$N_MATH" \
  "${EXTRA[@]}"
STATUS=$?

wc -l "$OUT_DIR"/*.jsonl || true
echo "=== Done === $(date) status=$STATUS her_code=$A1_SHA"
exit $STATUS
