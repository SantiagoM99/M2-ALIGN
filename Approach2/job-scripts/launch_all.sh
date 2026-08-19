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
# Safe to re-run at any time — it is incremental:
#   - a step whose job is already queued/running is reused, not resubmitted;
#   - a step whose output already exists on disk is treated as done and
#     skipped (delete the output dir/file to force a rerun);
#   - a language whose input data is missing is skipped with a warning
#     (only globally-shared inputs abort the launch), so the same command
#     picks up stage 3 + evals later, once stage3b data lands in $DT.
#
# Bengali reuses its existing stage-1 checkpoint (translation training is
# unaffected by the prompt change); its stage 3 IS retrained (new prompt +
# new stage-2 checkpoint) into outputs/stage3_bn.

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
S2_OUT="$A2/outputs/stage2_cc3m"

contains () { local x="$1"; shift; local e; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }
queued_id () { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -1; }

# submit <job-name> <dependency-or-empty> <script> — env comes from the caller.
# Prints the job id; reuses an already-queued job with the same name.
submit () {
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

echo "=== Sanity checks (shared inputs — these abort) ==="
hard=0
[ -f "$DT/Stage2/data/bn/cc3m_pairs.jsonl" ] || { echo "MISSING (required): $DT/Stage2/data/bn/cc3m_pairs.jsonl"; hard=1; }
[ -d "$DT/Stage2/data/cc3m/image_cache" ]    || { echo "MISSING (required): $DT/Stage2/data/cc3m/image_cache"; hard=1; }
[ -d "$GQA_IMAGES" ]                         || { echo "MISSING (required): GQA images dir ($GQA_IMAGES)"; hard=1; }
[ -f "$BN_STAGE1_CKPT" ]                     || { echo "MISSING (required): $BN_STAGE1_CKPT"; hard=1; }
[ "$hard" = 0 ] || exit 1
echo "Shared inputs OK."

echo "=== Stage 2 (shared: CC3M + WIT-bn, English captions) ==="
S2_JOB=""
if [ -n "$(queued_id a2_s2_cc3m)" ] || [ ! -f "$S2_OUT/mapping/pytorch_model.bin" ]; then
  S2_DATA="$DT/Stage2/data/bn/cc3m_pairs.jsonl"
  S2_CACHES="$DT/Stage2/data/cc3m/image_cache"
  if [ -f "$DT/Stage2/data/bn/wit_pairs.jsonl" ] && [ -d "$DT/Stage2/data/bn/image_cache" ]; then
    S2_DATA="$S2_DATA,$DT/Stage2/data/bn/wit_pairs.jsonl"
    S2_CACHES="$S2_CACHES,$DT/Stage2/data/bn/image_cache"
  fi
  S2_JOB=$(DATA_PATH="$S2_DATA" IMAGE_CACHE_DIR="$S2_CACHES" OUTPUT_DIR="$S2_OUT" \
           RUN_NAME="a2-stage2-cc3m" submit a2_s2_cc3m "" "$JS/train_stage2.sh")
else
  echo "  stage2: checkpoint exists ($S2_OUT) -> done (delete the dir to retrain)"
fi

echo "=== Stage 1 per language ==="
declare -A S1_JOB
for L in "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  if [ ! -f "$DT/Stage1/data/${NAME[$L]}_to_English.jsonl" ]; then
    echo "  stage1 $L: SKIPPED — no ${NAME[$L]}_to_English.jsonl in \$DT/Stage1/data"
    continue
  fi
  if [ -z "$(queued_id "a2_s1_$L")" ] && [ -f "$A2/outputs/stage1_$L/mapping/pytorch_model.bin" ]; then
    echo "  stage1 $L: checkpoint exists -> done"
    S1_JOB[$L]=""
    continue
  fi
  S1_JOB[$L]=$(LANGUAGES="${NAME[$L]}" DATA_DIR="$DT/Stage1/data" \
               OUTPUT_DIR="$A2/outputs/stage1_$L" \
               submit "a2_s1_$L" "" "$JS/train_stage1.sh")
done

echo "=== Stage 3 per language (waits on stage 2 + its stage 1) ==="
declare -A S3_JOB
S3_READY=()
for L in bn "${XGQA_LANGS[@]}" "${CVQA_ONLY_LANGS[@]}"; do
  # Two known layouts: the repo default (stage3b/<language>.jsonl) and what
  # Maryam's load_translated_data.sh actually writes (Stage3/data/<iso>.jsonl).
  s3b="$DT/Stage3/data/stage3b/${NAME[$L],,}.jsonl"
  [ -f "$s3b" ] || s3b="$DT/Stage3/data/$L.jsonl"
  if [ ! -f "$s3b" ]; then
    echo "  stage3 $L: SKIPPED — no stage3b data found for $L (re-run this script when it lands)"
    continue
  fi
  if [ "$L" = bn ]; then
    s1_ckpt="$BN_STAGE1_CKPT"; s1_dep=""
  else
    [ -n "${S1_JOB[$L]+x}" ] || { echo "  stage3 $L: SKIPPED — its stage 1 was skipped"; continue; }
    s1_ckpt="$A2/outputs/stage1_$L/mapping/pytorch_model.bin"; s1_dep="${S1_JOB[$L]}"
  fi
  if [ -z "$(queued_id "a2_s3_$L")" ] && [ -f "$A2/outputs/stage3_$L/mapping/pytorch_model.bin" ]; then
    echo "  stage3 $L: checkpoint exists -> done"
    S3_JOB[$L]=""
    S3_READY+=("$L")
    continue
  fi
  deps=()
  [ -n "$S2_JOB" ] && deps+=("$S2_JOB")
  [ -n "$s1_dep" ] && deps+=("$s1_dep")
  dep=""
  [ ${#deps[@]} -gt 0 ] && dep="afterok:$(IFS=:; echo "${deps[*]}")"
  S3_JOB[$L]=$(DATA_PATH="$s3b" IMAGES_DIR="$GQA_IMAGES" \
               STAGE1_CKPT="$s1_ckpt" \
               STAGE2_CKPT="$S2_OUT/mapping/pytorch_model.bin" \
               OUTPUT_DIR="$A2/outputs/stage3_$L" \
               RUN_NAME="a2-stage3-$L" \
               submit "a2_s3_$L" "$dep" "$JS/train_stage3.sh")
  S3_READY+=("$L")
done

# submit_eval <kind:xgqa|cvqa> <lang> <blind:0|1>
submit_eval () {
  local kind="$1" L="$2" B="$3" suffix="" jobsuffix="" data images name out
  # Plain if — a $( [ ... ] && echo ) here returns exit 1 when B=0, and a
  # bare assignment takes the command substitution's status, killing the
  # script under set -e (this silently ate every eval submission once).
  if [ "$B" = 1 ]; then suffix="_BLIND"; jobsuffix="_b"; fi
  out="$A2/outputs/stage3_$L/eval_${kind}_${L}${suffix}.jsonl"
  name="a2_ev_${kind:0:1}_${L}${jobsuffix}"
  if [ "$kind" = xgqa ]; then
    data="$DT/Stage3/data/xgqa/$L.jsonl"; images="$GQA_IMAGES"
  else
    data="$DT/Stage3/data/cvqa/$L.jsonl"; images="$DT/Stage3/data/cvqa/images"
  fi
  [ -f "$data" ] || { echo "  $name: SKIPPED — $data not found"; return; }
  if [ -z "$(queued_id "$name")" ] && [ -f "$out.summary.json" ]; then
    echo "  $name: summary exists -> done"
    return
  fi
  local dep=""
  [ -n "${S3_JOB[$L]:-}" ] && dep="afterok:${S3_JOB[$L]}"
  local script="$JS/evaluate_vqa.sh"
  [ "$kind" = cvqa ] && script="$JS/evaluate_cvqa.sh"
  BLIND="$B" EVAL_LANG="$L" DATA_PATH="$data" IMAGES_DIR="$images" \
  CKPT="$A2/outputs/stage3_$L/mapping/pytorch_model.bin" \
  OUTPUT_PATH="$out" \
  submit "$name" "$dep" "$script" > /dev/null
}

echo "=== Evals: full + blind per language (wait on stage 3) ==="
for L in "${S3_READY[@]}"; do
  if [ "$L" = bn ] || contains "$L" "${XGQA_LANGS[@]}"; then
    submit_eval xgqa "$L" 0
    submit_eval xgqa "$L" 1
  fi
  if contains "$L" "${CVQA_LANGS[@]}"; then
    submit_eval cvqa "$L" 0
    submit_eval cvqa "$L" 1
  fi
done

if contains bn "${S3_READY[@]}" && [ -f "$ROOT/evaluation/MGSM.jsonl" ] && [ -f "$ROOT/evaluation/MSVAMP.jsonl" ]; then
  echo "=== Text evals MGSM/MSVAMP (bn, new stage-3 checkpoint) ==="
  TXT_OUT="$A2/outputs/text_eval_bn_v2"
  if [ -z "$(queued_id a2_evt_bn)" ] && [ -f "$TXT_OUT/eval_msvamp_bn_stage3.jsonl.summary.json" ]; then
    echo "  a2_evt_bn: summaries exist -> done"
  else
    dep=""
    [ -n "${S3_JOB[bn]:-}" ] && dep="afterok:${S3_JOB[bn]}"
    STAGE1_CKPT="$BN_STAGE1_CKPT" \
    STAGE3_CKPT="$A2/outputs/stage3_bn/mapping/pytorch_model.bin" \
    OUT="$TXT_OUT" \
    submit a2_evt_bn "$dep" "$JS/evaluate_text.sh" > /dev/null
  fi
else
  echo "(MGSM/MSVAMP: skipped — bn stage 3 not ready or evaluation/{MGSM,MSVAMP}.jsonl missing)"
fi

echo
echo "=== Done. Monitor with: squeue -u \$USER ==="
echo "Re-run this same command any time — it only adds what's missing."
