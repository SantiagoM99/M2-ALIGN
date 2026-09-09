#!/bin/bash
#SBATCH --job-name=s1_eval
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/s1_eval_%j.log
set -euo pipefail
ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
SUBMISSION="${1:?use s1_submit.py to prepare the immutable submission}"
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
python -u Approach2/eval_matrix.py --submission "$SUBMISSION"
