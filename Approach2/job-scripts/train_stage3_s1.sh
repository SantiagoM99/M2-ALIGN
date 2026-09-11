#!/bin/bash
#SBATCH --job-name=s1_stage3
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/s1_stage3_%j.log
# Submit through s1_train_submit.py; re-use the same immutable submission to resume.
set -euo pipefail
ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
SUBMISSION="${1:?use s1_train_submit.py before sbatch}"
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Run the submitted code from an immutable checkout of its own commit, so a
# `git pull` in the main clone cannot change what a queued job will execute.
# Jobs 20655462 and 20659537 died on exactly that, and the guard was right:
# without this, the only way to keep a queued S1 job alive is to stop pulling.
# Outputs are unaffected: every path in a submission is absolute and still
# points at the main clone.
CODE_SHA=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['code_sha'])" "$SUBMISSION")
WORKTREE_ROOT="${S1_WORKTREE_ROOT:-$SCRATCH/s1_worktrees}"
WORKTREE="$WORKTREE_ROOT/$CODE_SHA"
mkdir -p "$WORKTREE_ROOT"
git -C "$ROOT" worktree prune
flock "$WORKTREE_ROOT/.lock" bash -c '
  [ -e "$1/Approach2/eval_matrix.py" ] || git -C "$2" worktree add --detach "$1" "$3"
' _ "$WORKTREE" "$ROOT" "$CODE_SHA"
[ -e "$WORKTREE/Approach2/eval_matrix.py" ] || { echo "ERROR: no worktree at $WORKTREE"; exit 1; }
echo "running from $WORKTREE (code $CODE_SHA)"
cd "$WORKTREE"
srun python -u Approach2/s1_train_submit.py --execute-submission "$SUBMISSION"
