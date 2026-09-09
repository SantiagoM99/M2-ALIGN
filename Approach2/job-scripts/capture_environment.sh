#!/bin/bash
# Run directly on a login node (not sbatch).
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python Approach2/capture_environment.py --output "${1:-Approach2/audits/rorqual_environment.json}"
