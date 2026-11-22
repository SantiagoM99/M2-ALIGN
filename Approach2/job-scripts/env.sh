# The login-node environment, in one line: source Approach2/job-scripts/env.sh
#
# Anything that downloads (build_culturalground.py, fetch_*) needs huggingface_hub
# from the venv, and the venv needs the modules loaded first. Stdlib scripts run
# without it, which is why the failure always arrives one command later than the
# mistake.
module load StdEnv/2023 python/3.11.5 arrow/21.0.0
# Not $SCRATCH: Alliance purges it by file age, which deleted torch's own
# modules between two jobs of one chain on 2026-11-19 while pip's metadata
# stayed intact -- `pip check` reported nothing and the first sign was an
# ImportError nine hours into the chain. VENV overrides the location.
source "${VENV:-$HOME/venvs/m2-align}/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
# torch joins the check because a purge takes modules out of an installed
# package while pip's metadata still looks healthy, so nothing else notices.
# This file is sourced, so it reports and returns; the batch scripts hard-fail.
python3 -c "import huggingface_hub, PIL, torch" 2>/dev/null \
  && echo "ready: $(python3 -V 2>&1), venv $(basename "$VIRTUAL_ENV")" \
  || echo "WARNING: venv active but huggingface_hub, PIL or torch is missing;" \
          "rebuild with pip install --no-index -r Approach2/job-scripts/requirements.txt"
