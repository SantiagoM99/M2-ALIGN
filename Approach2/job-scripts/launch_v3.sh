#!/bin/bash
# Round v3 launcher: the full 11-language round with the two ACCEPTED levers
# (DESIGN.md D6 + D9):
#   - DenseConnector stage 2: REUSES outputs/stage2_dc from the D9 pilot
#     (shared across languages — nothing to retrain)
#   - text replay in every stage 3 (math replay built in-job per language)
# Stage-1 checkpoints are reused from round B. So the whole round is TWO
# chained jobs, one GPU at a time:
#
#   [1] a2_s3_all_v3   11 stage-3 trainings + per-language math replay builds
#   [2] a2_ev_all_v3   every eval: xGQA/CVQA full+blind, bn MGSM/MSVAMP
#
# Usage (from the repo root on Rorqual; set S3_EPOCHS per the e2 verdict):
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_v3.sh
#   ROUND=v4 DT=... bash Approach2/job-scripts/launch_v3.sh   # after D11
#
# Incremental and safe to re-run: queued jobs are reused, existing
# checkpoints/summaries are skipped inside each packed job.

set -euo pipefail

DT="${DT:?Set DT to the datatransfer root, e.g. DT=/scratch/santimn/datatransfer}"
ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root (Approach2/job-scripts not found)."; exit 1; }
mkdir -p "$A2/logs"

ROUND="${ROUND:-v3}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"
S3_EPOCHS="${S3_EPOCHS:-2}"
# Stage-2 checkpoint: the LLaVA-scaled one (DESIGN.md D11) when it exists,
# otherwise the D9 pilot's. Override with S2_CKPT=... to pin one.
if [ -z "${S2_CKPT:-}" ]; then
  S2_CKPT="$A2/outputs/stage2_dc_llava/mapping/pytorch_model.bin"
  [ -f "$S2_CKPT" ] || S2_CKPT="$A2/outputs/stage2_dc/mapping/pytorch_model.bin"
fi
GQA_IMAGES="$DT/Stage3/data/gqa/images"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$ROOT/Stage3/data/gqa/images"

# Short reservations, chained. Alliance partitions by walltime (3/12/24/72/
# 168 h): a 46 h request competes for the 72 h pool, which is a fraction of
# the nodes a 12 h request can use, so one long job waits far longer than
# several short ones. Each link depends on the previous with afterany, so
# only ONE runs at a time — the GPU footprint is unchanged, only the
# reservation shrinks. Safe because both scripts are incremental: a language
# with a checkpoint is skipped, an interrupted one resumes from
# training_state.pt (--save-steps 200), and an eval with a .summary.json is
# skipped. A link that finds nothing left to do exits in seconds.
S3_TIME="${S3_TIME:-12:00:00}"
EV_TIME="${EV_TIME:-12:00:00}"
S3_CHAIN="${S3_CHAIN:-5}"   # 5 x 12 h covers ~35 h of stage-3 training
EV_CHAIN="${EV_CHAIN:-3}"

queued_id () { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -1; }

# Any link of a chain still queued/running?
queued_prefix () {
  squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk -v p="$1" '$2 ~ "^" p {print $1; exit}'
}

submit () {  # <job-name> <time> <dependency-or-empty> <script> — env from caller
  local name="$1" time="$2" dep="$3" script="$4"
  local id
  id=$(queued_id "$name")
  if [ -n "$id" ]; then
    echo "  $name: already queued/running -> reusing job $id" >&2
    printf '%s' "$id"
    return
  fi
  if [ -n "$dep" ]; then
    id=$(sbatch --parsable --export=ALL --job-name="$name" --time="$time" --dependency="$dep" "$script")
  else
    id=$(sbatch --parsable --export=ALL --job-name="$name" --time="$time" "$script")
  fi
  id="${id%%;*}"
  echo "  $name: submitted job $id${dep:+ ($dep)}" >&2
  printf '%s' "$id"
}

echo "=== Sanity checks ==="
hard=0
[ -f "$S2_CKPT" ] || { echo "MISSING: DC stage-2 checkpoint ($S2_CKPT) — run pilot_dense.sh / pilot_scale.sh first"; hard=1; }
echo "stage-2 checkpoint: $S2_CKPT"
[ -d "$GQA_IMAGES" ] || { echo "MISSING: GQA images dir"; hard=1; }
[ "$hard" = 0 ] || exit 1
for L in bn de ru zh pt id ko jv mn si ga; do
  if [ "$L" = bn ]; then s1="$A2/outputs/stage1/mapping/pytorch_model.bin"
  else s1="$A2/outputs/stage1_$L/mapping/pytorch_model.bin"; fi
  [ -f "$s1" ] || echo "WARNING: no stage-1 ckpt for $L — that language will be skipped"
done
ls "$SCRATCH/huggingface/datasets" 2>/dev/null | grep -qi gsm8k || \
  echo "WARNING: GSM8K not found in the HF datasets cache — math replay builds
  will fail on the offline compute node. Pre-download on this login node:
  module load StdEnv/2023 python/3.11.5 && source \$SCRATCH/venvs/m2-align/bin/activate &&
  HF_HOME=\$SCRATCH/huggingface python -c \"from datasets import load_dataset; load_dataset('openai/gsm8k','main')\""
n_math=$(ls "$A2"/data/math_replay_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "OK. Math replay files present: $n_math/11 (missing ones are built in-job)."
echo "ROUND=$ROUND S3_EPOCHS=$S3_EPOCHS VIS_LAYERS=$VIS_LAYERS"
echo "chains: stage3 ${S3_CHAIN}x${S3_TIME}, evals ${EV_CHAIN}x${EV_TIME} (1 GPU at a time)"

# Submit `links` copies of one script, each waiting on the previous.
# Echoes the last job id (what a downstream stage should depend on).
submit_chain () {  # <name-prefix> <links> <time> <dep-or-empty> <script>
  local prefix="$1" links="$2" time="$3" dep="$4" script="$5" i id=""
  for i in $(seq 1 "$links"); do
    id=$(submit "${prefix}_$i" "$time" "$dep" "$script")
    dep="afterany:$id"
  done
  printf '%s' "$id"
}

echo "=== [1] Stage 3 (all languages, chained ${S3_CHAIN}x${S3_TIME}) ==="
S3_JOB=""
s3_missing=0
for L in bn de ru zh pt id ko jv mn si ga; do
  [ -f "$A2/outputs/stage3_${L}_${ROUND}/mapping/pytorch_model.bin" ] || s3_missing=1
done
if [ -n "$(queued_prefix "a2_s3_all_$ROUND")" ] || [ "$s3_missing" = 1 ]; then
  S3_JOB=$(DT="$DT" ROUND="$ROUND" STAGE2_CKPT="$S2_CKPT" VIS_LAYERS="$VIS_LAYERS" \
           REPLAY=1 S3_EPOCHS="$S3_EPOCHS" GQA_IMAGES="$GQA_IMAGES" \
           submit_chain "a2_s3_all_$ROUND" "$S3_CHAIN" "$S3_TIME" "" "$JS/train_stage3_all.sh")
else
  echo "  stage3: all 11 $ROUND checkpoints exist -> done"
fi

echo "=== [2] Evals (everything, chained ${EV_CHAIN}x${EV_TIME}) ==="
dep=""
[ -n "$S3_JOB" ] && dep="afterany:$S3_JOB"
DT="$DT" ROUND="$ROUND" VIS_LAYERS="$VIS_LAYERS" GQA_IMAGES="$GQA_IMAGES" \
  submit_chain "a2_ev_all_$ROUND" "$EV_CHAIN" "$EV_TIME" "$dep" "$JS/evaluate_all.sh" > /dev/null

echo
echo "=== Done: $((S3_CHAIN+EV_CHAIN)) chained jobs, 1 GPU at a time. squeue -u \$USER to monitor ==="
echo "Re-run the same command any time — it only fills what's missing."
