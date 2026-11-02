#!/bin/bash
# Which donor countries does CulturalGround actually carry, and under what
# spelling? One short line, on a LOGIN node (this needs the dataset, not a GPU).
#
# The pooled cultural arm needs one country per donor language, matched to that
# language's own CVQA subset (audits/cvqa_inventory.json):
#   bn India | zh China (+Singapore held out) | id Indonesia
#   ko South Korea | pt Brazil | ru Russia        (de has no CVQA unit)
#
# Spelling is not guessable: this arm has already lost three GPU jobs to
# properties of this dataset that nobody checked first, so the probe runs before
# anything is built.
#
#   bash Approach2/job-scripts/cg_probe_donors.sh
#
# Env: COUNTRIES, LANGS_FOR (reported, not used by the probe)
set -uo pipefail

[ -d Approach2 ] || { echo "run this from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || {
  echo "no venv active: source \$SCRATCH/venvs/m2-align/bin/activate first"
  exit 1
}
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

# country:lang pairs, because --probe counts the rows of one language inside one
# country file and that count is the dose. Several spellings for the multi-word
# ones: only the dataset knows which it ships.
PAIRS="${PAIRS:-india:bn china:zh indonesia:id brazil:pt russia:ru south_korea:ko southkorea:ko korea:ko}"

ok=""; missing=""
for pair in $PAIRS; do
  country="${pair%%:*}"; lang="${pair##*:}"
  echo "=== $country / $lang ==="
  if python Approach2/build_culturalground.py --probe --country "$country" --lang "$lang" 2>&1 | tail -18; then
    ok="$ok $country:$lang"
  else
    missing="$missing $country:$lang"
  fi
  echo
done
echo "================================"
echo "resolved:$ok"
echo "absent:  $missing"
echo
echo "The usable count per pair is the dose, not the 20000 the build asks for."
echo "Singapore is deliberately absent: it is the held-out culture for Chinese,"
echo "read on CVQA and never trained on."
