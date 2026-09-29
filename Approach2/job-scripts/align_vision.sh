#!/bin/bash
#SBATCH --job-name=a2_align_vis
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a2_align_vis_%j.log
#
# Align the SigLIP2 dense expert to Qwen3-VL's embedding space on captions
# (DESIGN 2026-09-26, 2026-09-29). The only training the merged architecture needs
# that neither approach can donate, and the one schedule risk it carries: if this
# does not converge, neither injection mode can.
#
# Qwen's own tower sees a gray canvas throughout, so the caption can only be
# produced through the new mapping. The processor is pinned to one resolution
# because early injection resamples onto Qwen's merged grid.
#
#   SMOKE=1 DT=... sbatch Approach2/job-scripts/align_vision.sh   # minutes
#   MODE=prefix DT=... sbatch Approach2/job-scripts/align_vision.sh
#   MODE=early  DT=... sbatch Approach2/job-scripts/align_vision.sh
#
# Env: DT (required), MODE (prefix|early, default prefix), SMOKE, LIMIT, EPOCHS,
#      PIXELS, PREFIX_SIDE, LLAVA_DIR, OUTPUT_DIR
set -uo pipefail

ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DT="${DT:?set DT}"
MODE="${MODE:-prefix}"
LLAVA_DIR="${LLAVA_DIR:-$DT/Stage2/data/llava}"
LIMIT="${LIMIT:-40000}"
EPOCHS="${EPOCHS:-1}"
PIXELS="${PIXELS:-200704}"          # 448*448, so the merged grid is 14x14
PREFIX_SIDE="${PREFIX_SIDE:-12}"
LOG_EVERY=50

if [ "${SMOKE:-0}" = 1 ]; then
  LIMIT=200
  LOG_EVERY=1
  echo "SMOKE: 200 captions, one epoch — this only proves nothing crashes"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/Approach2/outputs/merged_align_${MODE}${SMOKE:+_smoke}}"

echo "=== Job info ==="; date; hostname; nvidia-smi || true

module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

DATA="$LLAVA_DIR/llava_pairs.jsonl"
CACHE="$LLAVA_DIR/image_cache"
[ -f "$DATA" ] || { echo "ERROR: no $DATA (run build_llava_pretrain.py on a login node)"; exit 1; }
[ -d "$CACHE" ] || { echo "ERROR: no $CACHE"; exit 1; }

# Same reason as the other S1 launchers: a `git pull` in the main clone must not
# change what an already-queued job executes, so only committed state runs.
CODE_SHA=$(git -C "$ROOT" rev-parse HEAD)
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet \
  || echo "WARNING: the tree was dirty at launch; only committed state runs"
WORKTREE="${S1_WORKTREE_ROOT:-$SCRATCH/s1_worktrees}/$CODE_SHA"
mkdir -p "$(dirname "$WORKTREE")"
git -C "$ROOT" worktree prune
flock "$(dirname "$WORKTREE")/.lock" bash -c '
  [ -e "$1/Approach2/merged/align_vision.py" ] || git -C "$2" worktree add --detach "$1" "$3"
' _ "$WORKTREE" "$ROOT" "$CODE_SHA"
echo "running from $WORKTREE (code $CODE_SHA)"
cd "$WORKTREE"

echo "=== aligning: mode=$MODE limit=$LIMIT epochs=$EPOCHS pixels=$PIXELS ==="
srun python -u Approach2/merged/align_vision.py \
  --mode "$MODE" \
  --data-path "$DATA" \
  --image-cache-dir "$CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --pixels "$PIXELS" \
  --prefix-side "$PREFIX_SIDE" \
  --epochs "$EPOCHS" \
  --limit "$LIMIT" \
  --log-every "$LOG_EVERY" \
  --local-files-only
STATUS=$?

echo "=== Done === $(date) status=$STATUS"
[ -f "$OUTPUT_DIR/complete.json" ] && cat "$OUTPUT_DIR/complete.json"
exit $STATUS
