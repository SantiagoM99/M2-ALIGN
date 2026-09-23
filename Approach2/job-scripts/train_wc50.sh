#!/bin/bash
#SBATCH --job-name=a2_wc50
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a2_wc50_%j.log
#
# Intervention WC50 (DESIGN 2026-09-23): C1's recipe with half the
# NLLB-translated GQA rows replaced by culturally grounded WorldCuisines rows in
# the same donor language, the row count held fixed. Every flag below is copied
# from what s1_train_submit.py passes for arm C1, so --data-path is the only
# difference; change one of them and the contrast stops being a single variable.
#
# WC50 is deliberately NOT an S1 arm: the frozen Block C contract stays untouched
# and this runs outside its provenance machinery. The control is C1 itself.
#
# Both row sources must resolve against one --images-dir, because the trainer
# takes a single directory. build_worldcuisines.py therefore caches its images
# into the GQA directory under a `wc_` prefix, which cannot collide with GQA's
# numeric ids and is undone with `rm $IMAGES/wc_*.jpg`.
#
#   DT=/scratch/santimn/datatransfer sbatch Approach2/job-scripts/train_wc50.sh
#
# Env: DT (required), DATA_PATH, IMAGES_DIR, OUTPUT_DIR, SEED
set -euo pipefail
ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DT="${DT:?set DT}"
SEED="${SEED:-13}"
DATA_PATH="${DATA_PATH:-$DT/Stage3/data/bn_wc50.jsonl}"
IMAGES_DIR="${IMAGES_DIR:-$DT/Stage3/data/gqa/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/Approach2/outputs/wc50_seed$SEED}"

module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Same reason as train_stage3_s1.sh: a `git pull` in the main clone must not
# change what an already-queued job executes. Only committed state runs, and the
# log records which commit it was.
CODE_SHA=$(git -C "$ROOT" rev-parse HEAD)
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet \
  || echo "WARNING: the tree was dirty at launch; only committed state runs"
WORKTREE="${S1_WORKTREE_ROOT:-$SCRATCH/s1_worktrees}/$CODE_SHA"
mkdir -p "$(dirname "$WORKTREE")"
git -C "$ROOT" worktree prune
flock "$(dirname "$WORKTREE")/.lock" bash -c '
  [ -e "$1/Approach2/train_stage3_vqa.py" ] || git -C "$2" worktree add --detach "$1" "$3"
' _ "$WORKTREE" "$ROOT" "$CODE_SHA"
echo "running from $WORKTREE (code $CODE_SHA)"

for f in "$DATA_PATH" "$IMAGES_DIR"; do
  [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
python - "$DATA_PATH" <<'PY'
import collections, json, sys
counts = collections.Counter()
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            counts[json.loads(line)["source_dataset"]] += 1
print("training rows by source:", dict(counts))
if len(counts) < 2:
    raise SystemExit("WC50 needs both sources in one file; build it with --mix-with")
PY

cd "$WORKTREE"
srun python -u Approach2/train_stage3_vqa.py \
  --s1 \
  --data-path "$DATA_PATH" \
  --images-dir "$IMAGES_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --stage1-ckpt "$ROOT/Approach2/outputs/stage1/mapping/pytorch_model.bin" \
  --stage2-ckpt "$ROOT/Approach2/outputs/stage2_dc_llava/mapping/pytorch_model.bin" \
  --vis-layers "9,18,-1" \
  --epochs 2 \
  --seed "$SEED" \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --grad-accum 16 \
  --lr 2e-5 \
  --save-steps 200 \
  --local-files-only
