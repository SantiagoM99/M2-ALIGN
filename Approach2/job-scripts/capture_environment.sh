#!/bin/bash
# Run directly on a login node (not sbatch).
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
# Not $SCRATCH: Alliance purges it by file age, which deleted torch's own
# modules between two jobs of one chain on 2026-11-19 while pip's metadata
# stayed intact -- `pip check` reported nothing and the first sign was an
# ImportError nine hours into the chain. VENV overrides the location.
source "${VENV:-$HOME/venvs/m2-align}/bin/activate"
# Fail in the first seconds rather than after a stage completes: a purged venv
# imports nothing, and the queue time is the expensive part.
python -c "import torch" 2>/dev/null || {
  echo "ERROR: ${VIRTUAL_ENV:-no venv} cannot import torch."
  echo "  Rebuild:  pip install --no-index -r Approach2/job-scripts/requirements.txt"
  echo "  Or point at another one: VENV=/path/to/venv sbatch ..."
  exit 1
}
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python Approach2/capture_environment.py --output "${1:-Approach2/audits/rorqual_environment.json}"
