# The login-node environment, in one line: source Approach2/job-scripts/env.sh
#
# Anything that downloads (build_culturalground.py, fetch_*) needs huggingface_hub
# from the venv, and the venv needs the modules loaded first. Stdlib scripts run
# without it, which is why the failure always arrives one command later than the
# mistake.
module load StdEnv/2023 python/3.11.5 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
python3 -c "import huggingface_hub, PIL" 2>/dev/null \
  && echo "ready: $(python3 -V 2>&1), venv $(basename "$VIRTUAL_ENV")" \
  || echo "WARNING: venv active but huggingface_hub or PIL missing"
