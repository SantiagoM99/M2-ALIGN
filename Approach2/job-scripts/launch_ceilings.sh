#!/bin/bash
# Frozen-LLM ceiling (Gemma-2-9b-it, no mapping) on MGSM/MSVAMP for the
# languages where we now have stage-3 scores: de, ru, zh. Bengali already has
# one (74.8 / 69.6).
#
# Without these, a stage-3 score cannot be read: v3 puts Russian at MGSM 75.6,
# which is above the *Bengali* ceiling, so we cannot tell whether the mapping
# is at ceiling in Russian or still leaving points on the table (DESIGN.md D10).
#
# One chained job per language, 12h each, 1 GPU at a time. Each is ~40 min of
# real work (250 + 1000 items, no mapping), so this is cheap.
#
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_ceilings.sh
#
# Idempotent: an eval with a .summary.json is skipped inside the job.

set -euo pipefail

ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root."; exit 1; }
mkdir -p "$A2/logs"

declare -A TAG=( [de]=deu_Latn [ru]=rus_Cyrl [zh]=zho_Hans [bn]=ben_Beng )
LANGS="${LANGS:-de ru zh}"

dep=""
for L in $LANGS; do
  MG="$ROOT/evaluation/MGSM_$L.jsonl"
  MS="$ROOT/evaluation/MSVAMP_$L.jsonl"
  if [ ! -f "$MG" ] || [ ! -f "$MS" ]; then
    echo "  $L: missing evaluation/{MGSM,MSVAMP}_$L.jsonl — skipping"
    continue
  fi
  id=$(EVAL_LANG="$L" EVAL_NLLB_TAG="${TAG[$L]}" VARIANTS=gemma \
       MGSM_PATH="$MG" MSVAMP_PATH="$MS" OUT="$A2/outputs/text_eval_$L" \
       sbatch --parsable --export=ALL --job-name="a2_ceil_$L" --time=12:00:00 \
              ${dep:+--dependency="$dep"} "$JS/evaluate_text.sh")
  id="${id%%;*}"
  echo "  $L: submitted job $id${dep:+ ($dep)}"
  dep="afterany:$id"
done
echo "Done. squeue -u \$USER to monitor."
