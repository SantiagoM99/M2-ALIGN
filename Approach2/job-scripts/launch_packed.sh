#!/bin/bash
# Packed launcher: the whole multi-language round as FOUR chained SLURM jobs
# (at most ~2 GPUs in use at any moment):
#
#   [1] a2_s2_cc3m   stage 2, CC3M + every language's WIT       (~1 GPU)
#   [2] a2_s1_all    the 10 stage-1 trainings, sequential       (~1 GPU)
#   [3] a2_s3_all    the 11 stage-3 trainings, sequential  (waits on 1+2)
#   [4] a2_ev_all    every evaluation, sequential          (waits on 3)
#
# Usage (from the repo root on Rorqual):
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_packed.sh
#   # round B with better stage-2 hyperparameters:
#   export S2_EPOCHS=10 S2_LR=1e-4 S2_GRAD_ACCUM=2
#   ROUND=v2 DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_packed.sh
#
# Incremental and safe to re-run: queued jobs are reused, existing
# checkpoints/summaries are skipped inside each packed job, and stages 3/4
# are chained with afterany + per-language prerequisite checks, so one
# failed language doesn't block the rest.

set -euo pipefail

DT="${DT:?Set DT to the datatransfer root, e.g. DT=/scratch/santimn/datatransfer}"
ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root (Approach2/job-scripts not found)."; exit 1; }
mkdir -p "$A2/logs"

R="${ROUND:+_$ROUND}"
S2_OUT="$A2/outputs/stage2_cc3m$R"
BN_STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"
GQA_IMAGES="$DT/Stage3/data/gqa/images"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$ROOT/Stage3/data/gqa/images"

queued_id () { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -1; }

submit () {  # <job-name> <dependency-or-empty> <script> — env from caller
  local name="$1" dep="$2" script="$3" id
  id=$(queued_id "$name")
  if [ -n "$id" ]; then
    echo "  $name: already queued/running -> reusing job $id" >&2
    printf '%s' "$id"
    return
  fi
  if [ -n "$dep" ]; then
    id=$(sbatch --parsable --export=ALL --job-name="$name" --dependency="$dep" "$script")
  else
    id=$(sbatch --parsable --export=ALL --job-name="$name" "$script")
  fi
  id="${id%%;*}"
  echo "  $name: submitted job $id${dep:+ ($dep)}" >&2
  printf '%s' "$id"
}

echo "=== Sanity checks (shared inputs) ==="
hard=0
[ -f "$DT/Stage2/data/bn/cc3m_pairs.jsonl" ] || { echo "MISSING: $DT/Stage2/data/bn/cc3m_pairs.jsonl"; hard=1; }
[ -d "$DT/Stage2/data/cc3m/image_cache" ]    || { echo "MISSING: $DT/Stage2/data/cc3m/image_cache"; hard=1; }
[ -d "$GQA_IMAGES" ]                         || { echo "MISSING: GQA images dir"; hard=1; }
[ -f "$BN_STAGE1_CKPT" ]                     || { echo "MISSING: $BN_STAGE1_CKPT"; hard=1; }
[ "$hard" = 0 ] || exit 1
echo "OK."

echo "=== [1] Stage 2 ==="
S2_JOB=""
if [ -n "$(queued_id "a2_s2_cc3m$R")" ] || [ ! -f "$S2_OUT/mapping/pytorch_model.bin" ]; then
  S2_DATA="$DT/Stage2/data/bn/cc3m_pairs.jsonl"
  S2_CACHES="$DT/Stage2/data/cc3m/image_cache"
  for wp in "$DT"/Stage2/data/*/wit_pairs.jsonl; do
    [ -f "$wp" ] || continue
    d="$(dirname "$wp")"
    [ -d "$d/image_cache" ] || continue
    S2_DATA="$S2_DATA,$wp"
    S2_CACHES="$S2_CACHES,$d/image_cache"
  done
  S2_JOB=$(DATA_PATH="$S2_DATA" IMAGE_CACHE_DIR="$S2_CACHES" OUTPUT_DIR="$S2_OUT" \
           RUN_NAME="a2-stage2-cc3m$R" submit "a2_s2_cc3m$R" "" "$JS/train_stage2.sh")
else
  echo "  stage2: checkpoint exists ($S2_OUT) -> done"
fi

echo "=== [2] Stage 1 (all languages, one job) ==="
S1_JOB=""
s1_missing=0
for L in de ru zh pt id ko jv mn si ga; do
  [ -f "$A2/outputs/stage1_$L/mapping/pytorch_model.bin" ] || s1_missing=1
done
if [ -n "$(queued_id a2_s1_all)" ] || [ "$s1_missing" = 1 ]; then
  S1_JOB=$(DT="$DT" submit a2_s1_all "" "$JS/train_stage1_all.sh")
else
  echo "  stage1: all 10 checkpoints exist -> done"
fi

echo "=== [3] Stage 3 (all languages, one job) ==="
deps=()
[ -n "$S2_JOB" ] && deps+=("$S2_JOB")
[ -n "$S1_JOB" ] && deps+=("$S1_JOB")
dep=""
[ ${#deps[@]} -gt 0 ] && dep="afterany:$(IFS=:; echo "${deps[*]}")"
s3_missing=0
for L in bn de ru zh pt id ko jv mn si ga; do
  [ -f "$A2/outputs/stage3_$L$R/mapping/pytorch_model.bin" ] || s3_missing=1
done
S3_JOB=""
if [ -n "$(queued_id "a2_s3_all$R")" ] || [ "$s3_missing" = 1 ]; then
  S3_JOB=$(DT="$DT" STAGE2_CKPT="$S2_OUT/mapping/pytorch_model.bin" GQA_IMAGES="$GQA_IMAGES" \
           submit "a2_s3_all$R" "$dep" "$JS/train_stage3_all.sh")
else
  echo "  stage3: all 11 checkpoints exist -> done"
fi

echo "=== [4] Evals (everything, one job) ==="
dep=""
[ -n "$S3_JOB" ] && dep="afterany:$S3_JOB"
DT="$DT" GQA_IMAGES="$GQA_IMAGES" submit "a2_ev_all$R" "$dep" "$JS/evaluate_all.sh" > /dev/null

echo
echo "=== Done: at most 4 jobs, ~2 GPUs concurrent. squeue -u \$USER to monitor ==="
echo "Re-run the same command any time — it only fills what's missing."
