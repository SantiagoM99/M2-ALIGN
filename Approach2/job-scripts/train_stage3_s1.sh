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
cd "$ROOT"
srun python -u Approach2/s1_train_submit.py --execute-submission "$SUBMISSION"
