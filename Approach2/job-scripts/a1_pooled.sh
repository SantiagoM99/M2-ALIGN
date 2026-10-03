#!/bin/bash
# Submit the pooled a1 chain: one short line, because the long sbatch line is
# what broke job 22343575 -- the terminal split it on paste, POOLED=1 never
# reached the job, and a 4-GPU node spent 51 minutes retraining Bengali.
#
# Stage 1 is the only stage that needs a whole node (~1.07M rows x 3 epochs;
# her own launcher asks for 4 GPUs for the same reason), and the mode and the
# resources are now set in the same place so they cannot disagree.
#
# Run from the repo root on a LOGIN node:
#   bash Approach2/job-scripts/a1_pooled.sh
#   REPLAY=1 bash Approach2/job-scripts/a1_pooled.sh   (adds her replay arm at stage 3)
#
# Env: REPLAY (also submit the stage-3 replay arm), DT, A1_ROOT, POOL_LANGS
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
TRAIN=Approach2/job-scripts/a1_train.sh
REPLAY="${REPLAY:-0}"

# REPLAY is passed per submission, never inherited: sbatch exports the whole
# environment, so a REPLAY=1 in the caller's shell would otherwise turn the
# plain stage-3 job into a second replay arm writing the same checkpoint.
submit () {  # submit <stage> <replay 0|1> <dependency-or-empty> <extra sbatch args...>
  local stage="$1" replay="$2" dep="$3"; shift 3
  local args=(--job-name="a1p$stage")
  [ "$replay" = 1 ] && args[0]="--job-name=a1p${stage}r"
  [ -n "$dep" ] && args+=(--dependency=afterok:"$dep")
  POOLED=1 REPLAY="$replay" STAGE="$stage" sbatch "${args[@]}" "$@" "$TRAIN" | awk '{print $NF}'
}

S1=$(submit 1 0 "" --gres=gpu:4 --cpus-per-task=48 --mem=490G --time=2-00:00:00)
[ -n "$S1" ] || { echo "stage 1 did not submit"; exit 1; }
echo "stage 1: $S1  (4 GPUs, 2 days)"

S2=$(submit 2 0 "$S1" --time=24:00:00)
echo "stage 2: $S2  (after $S1)"

S3=$(submit 3 0 "$S2" --time=24:00:00)
echo "stage 3: $S3  (after $S2)"

if [ "$REPLAY" = 1 ]; then
  R3=$(submit 3 1 "$S2" --time=48:00:00)
  echo "stage 3 replay: $R3  (after $S2, in parallel with $S3)"
fi
echo
echo "Watch with: squeue -u \$USER"
