#!/bin/bash
# Submit her pooled+replay arm as the two-job chain it actually is: build the
# text replay pool, then train stage 3 on it. One short line, for the same
# reason a1_pooled.sh exists -- the long sbatch line is what broke job 22343575
# when the terminal split it on paste.
#
# The pool is a prerequisite only this arm has, and nothing built it, which is
# how job 23145191 died in two seconds. Stages 1 and 2 are shared with the
# no-replay pooled arm and are NOT retrained: the chain forks at stage 3, and
# writes to a1_pooled_replay_stage3, so the checkpoint read on 10-13 is untouched.
#
# Run from the repo root on a LOGIN node:
#   bash Approach2/job-scripts/a1_replay.sh
#   N_MATH=2000 bash Approach2/job-scripts/a1_replay.sh    (cheaper pool pilot)
#
# Env: anything a1_replay_pool.sh or a1_train.sh reads (N_MATH, LANGS, MT, DT...)
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }

POOL=$(sbatch --job-name=a1rpool Approach2/job-scripts/a1_replay_pool.sh | awk '{print $NF}')
[ -n "$POOL" ] || { echo "the pool job did not submit"; exit 1; }
echo "replay pool: $POOL  (1 GPU, 12 h)"

# REPLAY is passed per submission rather than inherited, for the reason spelled
# out in a1_pooled.sh: sbatch exports the caller's whole environment.
TRAIN=$(POOLED=1 REPLAY=1 STAGE=3 sbatch --job-name=a1p3r \
  --dependency=afterok:"$POOL" --time=48:00:00 \
  Approach2/job-scripts/a1_train.sh | awk '{print $NF}')
echo "stage 3 replay: $TRAIN  (after $POOL)"
echo
echo "Watch with: squeue -u \$USER"
echo "Then evaluate: CKPT=\$PWD/Approach2/outputs/a1_pooled_replay_stage3/pytorch_model.bin"
