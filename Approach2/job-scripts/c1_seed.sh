#!/bin/bash
# Prepare and submit one C1 seed replicate of Block C (DESIGN 2026-09-23).
#
# C1 is the only arm that needs replication: C4 is untrained, so the post-hoc
# C4 - C1 contrast moves only with C1's trajectory. Read the seeds afterwards
# with analysis/c1_seed_spread.py.
#
# Run from the repo root on a LOGIN node, with the modules and venv active:
#   SEED=14 bash Approach2/job-scripts/c1_seed.sh
#   for S in 14 15; do SEED=$S bash Approach2/job-scripts/c1_seed.sh; done
#
# Env:
#   SEED  (required) training seed of the replicate
#   DT    (required) datatransfer root
#   REFERENCE  submission whose frozen environment this must match
#              (default Approach2/outputs/s1_C1.submission.json)
#   ALLOW_PILLOW_DRIFT=1  proceed although pillow differs from the reference.
#              Only with a recorded reason: pillow decodes every image, and the
#              contrast is not paired across two decoding paths.
set -uo pipefail

SEED="${SEED:?set SEED, e.g. SEED=14}"
DT="${DT:?set DT=/scratch/santimn/datatransfer}"
A2="Approach2"
OUT="$PWD/$A2/outputs"
REFERENCE="${REFERENCE:-$A2/outputs/s1_C1.submission.json}"
SUB="$OUT/s1_C1_seed${SEED}.submission.json"
# The committed Block A report, not the run-time copy under outputs/: outputs is
# gitignored, so a fresh clone has only the audit, and that is the artifact whose
# G0 pass and spec SHA the submission authenticates against.
GATE="${GATE:-$A2/audits/s1_A_analysis.json}"
[ -f "$GATE" ] || { echo "no Block A report at $GATE; set GATE=<path>"; exit 1; }

[ -d "$A2" ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first (activate only, install nothing)"
  exit 1
}

# The reference submission is the authority on the environment a replicate has
# to share. Pillow was silently downgraded once between the seed-13 submission
# and 09-23, which changes image decoding and would make the contrast unpaired.
python - "$REFERENCE" <<'PY' || exit 1
import json, sys
sys.path.insert(0, "Approach2")
from eval_runtime import environment_record
import os

ref_path = sys.argv[1]
try:
    with open(ref_path) as f:
        ref = json.load(f)
except OSError as exc:
    raise SystemExit(f"cannot read the reference submission {ref_path}: {exc}")
frozen = (ref.get("environment") or ref.get("manifest", {}).get("environment") or {}).get("packages", {})
here = environment_record()["packages"]
drift = {k: (frozen.get(k), here.get(k)) for k in sorted(set(frozen) | set(here))
         if frozen.get(k) != here.get(k)}
for name, (was, now) in drift.items():
    print(f"environment drift: {name} was {was} in {ref_path}, is {now} here")
if drift and not os.environ.get("ALLOW_PILLOW_DRIFT"):
    raise SystemExit(
        "refusing: a seed replicate must share the reference's environment, because pillow "
        "decodes the images and the C4 - C1 contrast is paired. Restore the reference version, "
        "or set ALLOW_PILLOW_DRIFT=1 and record why in DESIGN.md"
    )
print("environment matches the reference submission" if not drift else "proceeding with recorded drift")
PY

if [ -f "$OUT/s1_C1_seed${SEED}/complete.json" ]; then
  echo "seed $SEED already finished training; nothing to submit"
  exit 0
fi

echo "=== preparing seed $SEED ==="
python "$A2/s1_train_submit.py" --arm C1 --seed "$SEED" \
  --data-root "$DT" \
  --checkpoints "$OUT" \
  --output-dir "$OUT/s1_C1_seed${SEED}" \
  --block-a-report "$GATE" \
  --submission "$SUB" || exit 1

echo "=== submitting seed $SEED ==="
python "$A2/s1_train_submit.py" --resume --submit --submission "$SUB"
