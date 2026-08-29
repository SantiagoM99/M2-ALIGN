#!/bin/bash
# D12 pilot — ONE shared multilingual text mapping vs eleven per-language ones.
#
#   [1] a2_s1_joint_*   stage 1 on all 11 languages shuffled together
#   [2] a2_s3_all_vj_*  stage 3 per language, warm-started from that shared
#                       mapping + the D11 stage-2 checkpoint
#   [3] a2_ev_all_vj_*  CVQA first (all 5 LRLs, cultural, cheap), then the
#                       text benchmarks, then xGQA
#
# Single variable versus round v4: where the text mapping comes from.
# Everything else — stage 2, stage-3 recipe, replay, epochs — is identical,
# so the delta is attributable to sharing parameters across languages.
#
# The metric is NOT the average. Judge it with
#   cd Approach2/results && python3 ../analysis/gap_report.py v4 vj
# on the LRL-HRL gap and the CVQA-xGQA cultural gap. A round that lifts the
# mean while widening either gap is a failure for this project's purpose.
#
# Usage (repo root, Rorqual):
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_joint.sh
#
# Chained 12h links, one GPU at a time (see the SLURM notes in launch_v3.sh).
# Idempotent: re-run any time, it only fills what is missing.

set -euo pipefail

DT="${DT:?Set DT, e.g. DT=/scratch/santimn/datatransfer}"
ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root."; exit 1; }
mkdir -p "$A2/logs"

ROUND="${ROUND:-vj}"
VIS_LAYERS="${VIS_LAYERS:-9,18,-1}"
S3_EPOCHS="${S3_EPOCHS:-2}"
TRAIN_NUM="${TRAIN_NUM:-30000}"
S1_EPOCHS="${S1_EPOCHS:-1}"
S1_JOINT_OUT="$A2/outputs/stage1_joint"
S1_JOINT_CKPT="$S1_JOINT_OUT/mapping/pytorch_model.bin"
S3_TIME="${S3_TIME:-12:00:00}"
EV_TIME="${EV_TIME:-12:00:00}"
S1_CHAIN="${S1_CHAIN:-2}"
S3_CHAIN="${S3_CHAIN:-4}"
EV_CHAIN="${EV_CHAIN:-3}"

# Stage 2: the D11 LLaVA-scaled checkpoint when present, else the D9 one.
if [ -z "${S2_CKPT:-}" ]; then
  S2_CKPT="$A2/outputs/stage2_dc_llava/mapping/pytorch_model.bin"
  [ -f "$S2_CKPT" ] || S2_CKPT="$A2/outputs/stage2_dc/mapping/pytorch_model.bin"
fi
GQA_IMAGES="$DT/Stage3/data/gqa/images"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$ROOT/Stage3/data/gqa/images"

queued_prefix () {
  squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk -v p="$1" '$2 ~ "^" p {print $1; exit}'
}

submit () {  # <name> <time> <dep-or-empty> <script>
  local name="$1" time="$2" dep="$3" script="$4" id
  if [ -n "$dep" ]; then
    id=$(sbatch --parsable --export=ALL --job-name="$name" --time="$time" --dependency="$dep" "$script")
  else
    id=$(sbatch --parsable --export=ALL --job-name="$name" --time="$time" "$script")
  fi
  id="${id%%;*}"
  echo "  $name: submitted job $id${dep:+ ($dep)}" >&2
  printf '%s' "$id"
}

submit_chain () {  # <prefix> <links> <time> <dep-or-empty> <script>
  local prefix="$1" links="$2" time="$3" dep="$4" script="$5" i id=""
  for i in $(seq 1 "$links"); do
    id=$(submit "${prefix}_$i" "$time" "$dep" "$script")
    dep="afterany:$id"
  done
  printf '%s' "$id"
}

echo "=== Sanity checks ==="
hard=0
[ -f "$S2_CKPT" ] || { echo "MISSING: stage-2 checkpoint ($S2_CKPT)"; hard=1; }
[ -d "$GQA_IMAGES" ] || { echo "MISSING: GQA images dir"; hard=1; }
[ "$hard" = 0 ] || exit 1
echo "stage-2 checkpoint: $S2_CKPT"
n_data=0
for n in Bengali German Russian Chinese Portuguese Indonesian Korean Javanese Mongolian Sinhalese Irish; do
  [ -f "$DT/Stage1/data/${n}_to_English.jsonl" ] && n_data=$((n_data+1))
done
echo "stage-1 translation files present: $n_data/11 (missing ones drop out of the joint mix)"
[ "$n_data" -ge 2 ] || { echo "ERROR: joint training needs at least 2 languages"; exit 1; }
echo "ROUND=$ROUND TRAIN_NUM=$TRAIN_NUM/lang S1_EPOCHS=$S1_EPOCHS S3_EPOCHS=$S3_EPOCHS"

echo "=== [1] Joint stage 1 ==="
S1_JOB=""
if [ -f "$S1_JOINT_CKPT" ]; then
  echo "  joint mapping exists -> skipping"
elif [ -n "$(queued_prefix "a2_s1_joint")" ]; then
  echo "  already queued/running -> not resubmitting"
  S1_JOB=""
else
  S1_JOB=$(DT="$DT" TRAIN_NUM="$TRAIN_NUM" S1_EPOCHS="$S1_EPOCHS" OUT="$S1_JOINT_OUT" \
           submit_chain "a2_s1_joint" "$S1_CHAIN" 12:00:00 "" "$JS/train_stage1_joint.sh")
fi

echo "=== [2] Stage 3, all languages, from the shared mapping ==="
dep=""
[ -n "$S1_JOB" ] && dep="afterany:$S1_JOB"
S3_JOB=""
s3_missing=0
for L in bn de ru zh pt id ko jv mn si ga; do
  [ -f "$A2/outputs/stage3_${L}_${ROUND}/mapping/pytorch_model.bin" ] || s3_missing=1
done
if [ -n "$(queued_prefix "a2_s3_all_$ROUND")" ] || [ "$s3_missing" = 1 ]; then
  S3_JOB=$(DT="$DT" ROUND="$ROUND" STAGE2_CKPT="$S2_CKPT" STAGE1_CKPT="$S1_JOINT_CKPT" \
           VIS_LAYERS="$VIS_LAYERS" REPLAY=1 S3_EPOCHS="$S3_EPOCHS" GQA_IMAGES="$GQA_IMAGES" \
           submit_chain "a2_s3_all_$ROUND" "$S3_CHAIN" "$S3_TIME" "$dep" "$JS/train_stage3_all.sh")
else
  echo "  all 11 $ROUND checkpoints exist -> done"
fi

echo "=== [3] Evals — CVQA first (the gap metrics), xGQA last ==="
dep=""
[ -n "$S3_JOB" ] && dep="afterany:$S3_JOB"
DT="$DT" ROUND="$ROUND" VIS_LAYERS="$VIS_LAYERS" GQA_IMAGES="$GQA_IMAGES" \
  BENCHES="cvqa xgqa" \
  submit_chain "a2_ev_all_$ROUND" "$EV_CHAIN" "$EV_TIME" "$dep" "$JS/evaluate_all.sh" > /dev/null

echo
echo "=== Done. Judge it with: cd Approach2/results && python3 ../analysis/gap_report.py v4 $ROUND ==="
