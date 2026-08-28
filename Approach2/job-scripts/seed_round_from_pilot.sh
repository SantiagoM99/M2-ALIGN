#!/bin/bash
# Seed a round's Bengali slot from a finished bn pilot.
#
# A pilot (pilot_dense.sh / pilot_scale.sh) writes outputs/stage3_bn_<TAG>
# with tagged eval filenames; a round (train_stage3_all.sh + evaluate_all.sh)
# expects outputs/stage3_bn_<ROUND> with untagged ones, and puts the text
# evals in outputs/text_eval_bn_v2_<ROUND>. When the pilot's recipe IS the
# round's recipe, copying across saves bn's stage 3 plus its four VQA and two
# text evals (~5-6 h of GPU); both round jobs skip whatever already has a
# checkpoint or a .summary.json.
#
# Usage (repo root):  SRC=dcl ROUND=v4 bash Approach2/job-scripts/seed_round_from_pilot.sh
#
# Idempotent: never overwrites an existing destination file.

set -euo pipefail

SRC="${SRC:?set SRC to the pilot tag, e.g. SRC=dcl for outputs/stage3_bn_dcl}"
ROUND="${ROUND:?set ROUND, e.g. ROUND=v4}"
L="${LANG_CODE:-bn}"

A2="$PWD/Approach2"
[ -d "$A2/job-scripts" ] || { echo "Run this from the repo root."; exit 1; }
S="$A2/outputs/stage3_${L}_${SRC}"
D="$A2/outputs/stage3_${L}_${ROUND}"

[ -f "$S/mapping/pytorch_model.bin" ] || { echo "ERROR: no $S/mapping/pytorch_model.bin"; exit 1; }

copy () {  # <src> <dst>
  if [ -f "$1" ] && [ ! -f "$2" ]; then
    cp "$1" "$2"
    echo "  + ${2#$A2/outputs/}"
  fi
}

if [ -f "$D/mapping/pytorch_model.bin" ]; then
  echo "  = $D/mapping already present"
else
  mkdir -p "$D/mapping"
  cp "$S/mapping/"* "$D/mapping/"
  echo "  + stage3_${L}_${ROUND}/mapping (checkpoint)"
fi

for k in xgqa cvqa; do
  for suf in "" "_BLIND"; do
    for ext in "" ".summary.json"; do
      copy "$S/eval_${k}_${L}_${SRC}${suf}.jsonl$ext" "$D/eval_${k}_${L}${suf}.jsonl$ext"
    done
  done
done

T="$A2/outputs/text_eval_bn_v2_${ROUND}"
if [ "$L" = bn ]; then
  mkdir -p "$T"
  for b in mgsm msvamp; do
    for ext in "" ".summary.json"; do
      copy "$S/eval_${b}_bn_stage3_${SRC}.jsonl$ext" "$T/eval_${b}_bn_stage3.jsonl$ext"
    done
  done
fi

echo "Seeded $D — the $ROUND jobs will skip $L."
