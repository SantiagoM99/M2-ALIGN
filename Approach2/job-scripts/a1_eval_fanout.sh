#!/bin/bash
# One evaluation job per language instead of one job for all of them.
#
# Her evaluator reloads NLLB and the VLM for every language and spends a long
# setup phase before the scoring loop: in job 25535464 the mapping for `ru`
# loaded at 09:56:55 and the first item was scored at 11:29:41 -- an hour and a
# half outside the progress bar -- after which 200 items took 12 minutes. Ten
# languages in series spend that cost ten times and hit the 12 h wall; jobs
# 25535463 and 25535464 both died there with most cells unwritten.
#
# Fanning out also isolates `ko`, which in that job never reached its first item
# in eleven hours while its images were all cached. Alone, it either finishes or
# hangs in a log of its own.
#
# a1_eval.sh skips non-empty outputs, so cells already written are not redone.
#
#   CKPT=... TAG=... bash Approach2/job-scripts/a1_eval_fanout.sh
#   LANGS="ko" TIME=24:00:00 bash Approach2/job-scripts/a1_eval_fanout.sh
#
# Env: CKPT (required), TAG (required), LANGS, BENCHMARK, VISUAL_PIXELS, TIME
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
: "${CKPT:?set CKPT to the mapping checkpoint}"
: "${TAG:?set TAG, the suffix every output file carries}"
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }
LANGS="${LANGS:-bn zh id pt ru ko jv mn ga si}"
BENCHMARK="${BENCHMARK:-cvqa}"
VISUAL_PIXELS="${VISUAL_PIXELS:-602112}"
TIME="${TIME:-6:00:00}"

for L in $LANGS; do
  out="Approach2/outputs/a1_eval/eval_${BENCHMARK}_${L}_${TAG}.jsonl"
  if [ -s "$out" ]; then
    echo "skip $L ($(wc -l < "$out") rows already)"
    continue
  fi
  id=$(CKPT="$CKPT" BENCHMARK="$BENCHMARK" LANGS="$L" \
       VISUAL_PIXELS="$VISUAL_PIXELS" TAG="$TAG" \
       sbatch --job-name="ev_${L}_${TAG}" --time="$TIME" \
       Approach2/job-scripts/a1_eval.sh | awk '{print $NF}')
  echo "$L: ${id:-DID NOT SUBMIT}"
done
echo
echo "Watch with: squeue -u \$USER"
