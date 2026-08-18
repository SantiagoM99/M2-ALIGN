#!/bin/bash
# Launch the full multi-language Approach 2 round as one dependency-chained
# SLURM pipeline:
#
#   stage2 (CC3M+WIT, shared) ──┐
#   stage1 per language ────────┴→ stage3 per language → xGQA full+blind
#                                                      → CVQA full+blind
#                                                      → MGSM/MSVAMP (bn, optional)
#
# Usage (from the repo root on Rorqual, AFTER git pull):
#   DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_all.sh
#
# Notes:
# - Verifies every input file first and aborts with a list of what's
#   missing (edit the language arrays below to drop a language instead).
# - If a parent job fails, its dependents die as DependencyNeverSatisfied;
#   fix the cause and re-run this script — completed stages resume/skip
#   via each job's own training_state.pt logic, and you can comment out
#   the blocks that already finished.
# - Bengali reuses its existing stage-1 checkpoint (translation training is
#   unaffected by the prompt change); its stage 3 IS retrained (new prompt
#   + new stage-2 checkpoint) into outputs/stage3_bn.

set -euo pipefail

DT="${DT:?Set DT to the datatransfer root, e.g. DT=/scratch/santimn/datatransfer}"
ROOT="$PWD"
A2="$ROOT/Approach2"
JS="$A2/job-scripts"
[ -d "$JS" ] || { echo "Run this from the repo root (Approach2/job-scripts not found)."; exit 1; }
mkdir -p "$A2/logs"

# xGQA languages (stage1 + stage3 + xGQA eval). bn is handled separately.
XGQA_LANGS=(de ru zh pt id ko)
# CVQA-only languages (stage1 + stage3 + CVQA eval; no xGQA coverage).
CVQA_ONLY_LANGS=(jv mn si ga)
# CVQA coverage set (no de — CVQA has no German subset).
CVQA_LANGS=(bn ru zh pt id ko jv mn si ga)

declare -A NAME=( [bn]=Bengali [de]=German [ru]=Russian [zh]=Chinese
                  [pt]=Portuguese [id]=Indonesian [ko]=Korean [jv]=Javanese
                  [mn]=Mongolian [si]=Sinhalese [ga]=Irish )

GQA_IMAGES="$DT/Stage3/data/gqa/images"
[ -d "$GQA_IMAGES" ] || GQA_IMAGES="$ROOT/Stage3/data/gqa/images"

BN_STAGE1_CKPT="$A2/outputs/stage1/mapping/pytorch_model.bin"

echo "=== Sanity checks ==="
missing=0
require_f () { [ -f "$1" ] || { echo "MISSING file: $1"; missing=1; }; }
require_d () { [ -d "$1" ] || { echo "MISSING dir:  $1"; missing=1; }; }

require_f "$DT/Stage2/data/bn/cc3m_pairs.jsonl"
require_d "$DT/Stage2/data/cc3m/image_cache"
require_d "$GQA_IMAGES"
require_f "$BN_STAGE1_CKPT"
for L in "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  require_f "$DT/Stage1/data/${NAME[$L]}_to_English.jsonl"
done
for L in bn "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  require_f "$DT/Stage3/data/stage3b/${NAME[$L],,}.jsonl"
done
for L in bn "${XGQA_LANGS[@]}"; do
  require_f "$DT/Stage3/data/xgqa/$L.jsonl"
done
for L in "${CVQA_LANGS[@]}"; do
  require_f "$DT/Stage3/data/cvqa/$L.jsonl"
done
[ "$missing" = 0 ] || { echo; echo "Fix the missing inputs (or trim the language arrays) and re-run."; exit 1; }
echo "All inputs present."

submit () {  # submit <job-name> <dependency-or-empty> <script> — env comes from caller
  local name="$1" dep="$2" script="$3" id
  if [ -n "$dep" ]; then
    id=$(sbatch --parsable --export=ALL --job-name="$name" --dependency="$dep" "$script")
  else
    id=$(sbatch --parsable --export=ALL --job-name="$name" "$script")
  fi
  id="${id%%;*}"
  echo "  $name -> job $id" >&2
  printf '%s' "$id"
}

echo "=== Stage 2 (shared: CC3M + WIT-bn, English captions) ==="
S2_DATA="$DT/Stage2/data/bn/cc3m_pairs.jsonl"
S2_CACHES="$DT/Stage2/data/cc3m/image_cache"
if [ -f "$DT/Stage2/data/bn/wit_pairs.jsonl" ] && [ -d "$DT/Stage2/data/bn/image_cache" ]; then
  S2_DATA="$S2_DATA,$DT/Stage2/data/bn/wit_pairs.jsonl"
  S2_CACHES="$S2_CACHES,$DT/Stage2/data/bn/image_cache"
fi
S2_OUT="$A2/outputs/stage2_cc3m"
S2=$(DATA_PATH="$S2_DATA" IMAGE_CACHE_DIR="$S2_CACHES" OUTPUT_DIR="$S2_OUT" \
     RUN_NAME="a2-stage2-cc3m" submit a2_s2_cc3m "" "$JS/train_stage2.sh")

echo "=== Stage 1 per language ==="
declare -A S1
for L in "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  S1[$L]=$(LANGUAGES="${NAME[$L]}" DATA_DIR="$DT/Stage1/data" \
           OUTPUT_DIR="$A2/outputs/stage1_$L" \
           submit "a2_s1_$L" "" "$JS/train_stage1.sh")
done

echo "=== Stage 3 per language (waits on stage 2 + its stage 1) ==="
declare -A S3
for L in bn "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  if [ "$L" = bn ]; then
    dep="afterok:$S2"
    s1_ckpt="$BN_STAGE1_CKPT"
  else
    dep="afterok:$S2:${S1[$L]}"
    s1_ckpt="$A2/outputs/stage1_$L/mapping/pytorch_model.bin"
  fi
  S3[$L]=$(DATA_PATH="$DT/Stage3/data/stage3b/${NAME[$L],,}.jsonl" \
           IMAGES_DIR="$GQA_IMAGES" \
           STAGE1_CKPT="$s1_ckpt" \
           STAGE2_CKPT="$S2_OUT/mapping/pytorch_model.bin" \
           OUTPUT_DIR="$A2/outputs/stage3_$L" \
           RUN_NAME="a2-stage3-$L" \
           submit "a2_s3_$L" "$dep" "$JS/train_stage3.sh")
done

echo "=== xGQA evals: full + blind per language (wait on stage 3) ==="
for L in bn "${XGQA_LANGS[@]}"; do
  for B in 0 1; do
    suffix=""; [ "$B" = 1 ] && suffix="_b"
    BLIND="$B" EVAL_LANG="$L" \
    DATA_PATH="$DT/Stage3/data/xgqa/$L.jsonl" \
    IMAGES_DIR="$GQA_IMAGES" \
    CKPT="$A2/outputs/stage3_$L/mapping/pytorch_model.bin" \
    submit "a2_evx_${L}${suffix}" "afterok:${S3[$L]}" "$JS/evaluate_vqa.sh" > /dev/null
  done
done

echo "=== CVQA evals: full + blind per language (wait on stage 3) ==="
for L in "${CVQA_LANGS[@]}"; do
  for B in 0 1; do
    suffix=""; [ "$B" = 1 ] && suffix="_b"
    BLIND="$B" EVAL_LANG="$L" \
    DATA_PATH="$DT/Stage3/data/cvqa/$L.jsonl" \
    IMAGES_DIR="$DT/Stage3/data/cvqa/images" \
    CKPT="$A2/outputs/stage3_$L/mapping/pytorch_model.bin" \
    submit "a2_evc_${L}${suffix}" "afterok:${S3[$L]}" "$JS/evaluate_cvqa.sh" > /dev/null
  done
done

if [ -f "$ROOT/evaluation/MGSM.jsonl" ] && [ -f "$ROOT/evaluation/MSVAMP.jsonl" ]; then
  echo "=== Text evals MGSM/MSVAMP (bn, new stage-3 checkpoint) ==="
  STAGE1_CKPT="$BN_STAGE1_CKPT" \
  STAGE3_CKPT="$A2/outputs/stage3_bn/mapping/pytorch_model.bin" \
  OUT="$A2/outputs/text_eval_bn_v2" \
  submit a2_evt_bn "afterok:${S3[bn]}" "$JS/evaluate_text.sh" > /dev/null
else
  echo "(skipping MGSM/MSVAMP: evaluation/{MGSM,MSVAMP}.jsonl not found in repo root)"
fi

echo
echo "=== All submitted. Monitor with: squeue -u \$USER ==="
echo "If a parent fails, dependents show DependencyNeverSatisfied — fix and re-run."
