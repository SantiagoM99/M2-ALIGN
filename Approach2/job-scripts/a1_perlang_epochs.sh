#!/bin/bash
# Remove the epoch confound from the per-language against pooled contrast.
#
# Registered before submission (DESIGN 2026-11-18, and named as required work in
# the 11-03 entry). The +2.77 that the two-regimes frame rests on compares a
# per-language arm trained with ONE stage-1 epoch against a pooled arm trained
# with three. The confound runs against the winner on ga/jv/mn, which is why the
# direction survived it, but bn and si are the cells where pooled wins and the
# confound favours pooled -- so those are exactly the cells a retrain can move.
#
# Three stages, chained, because stage 2 warm-starts from stage 1 and stage 3
# from stage 2: changing the stage-1 epoch count invalidates both. RUN_TAG=e3
# keeps every output beside the existing arm instead of overwriting it, so the
# old contrast stays reproducible.
#
# Stage 1 for one language at three epochs is ~13.5 h (her measurement, quoted
# in a1_train.sh), so it asks for 24 h; the other two take the 12 h partition.
#
# Prediction: the per-language arm improves, and the question is whether it
# improves enough to flip bn or si. If the pooled lead on those two survives
# three matched epochs, the regime trade-off is real and the frame holds as
# written. If both flip, the frame becomes "per-language wins everywhere" and
# the paper's regime story needs rewriting, not just renumbering.
#
# Refutation of the frame: per-language ahead on every one of the five panels.
# That would mean pooling never helps and the trade-off does not exist.
#
#   bash Approach2/job-scripts/a1_perlang_epochs.sh
#
# Env: A1_LANG, RUN_TAG, EPOCHS, LANGS, MATCHED_PIXELS, DT
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
DT="${DT:-/scratch/santimn/datatransfer}"
A1_LANG="${A1_LANG:-bn}"
RUN_TAG="${RUN_TAG:-e3}"
EPOCHS="${EPOCHS:-3}"
LANGS="${LANGS:-bn jv mn ga si}"
MATCHED_PIXELS="${MATCHED_PIXELS:-602112}"
SLUG="${A1_LANG}_${RUN_TAG}"
O="$PWD/Approach2/outputs"

for f in "$DT/Stage1/data/Bengali_to_English.jsonl" "$DT/Stage3/data/${A1_LANG}.jsonl"; do
  [ -s "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

submit () {  # submit <stage> <hours> [dependency]
  local stage="$1" hours="$2" dep="${3:-}" args=()
  [ -n "$dep" ] && args=(--dependency=afterok:"$dep")
  STAGE="$stage" A1_LANG="$A1_LANG" RUN_TAG="$RUN_TAG" EPOCHS="$EPOCHS" \
    sbatch --job-name="a1${A1_LANG}${RUN_TAG}s${stage}" --time="${hours}:00:00" \
    ${args[@]+"${args[@]}"} Approach2/job-scripts/a1_train.sh | awk '{print $NF}'
}

S1=$(submit 1 24);  [ -n "$S1" ] || { echo "stage 1 did not submit"; exit 1; }
echo "stage 1 ($EPOCHS epochs): $S1  (24 h) -> $O/a1_${SLUG}_stage1"
S2=$(submit 2 12 "$S1"); echo "stage 2: ${S2:-DID NOT SUBMIT}  (after $S1)"
S3=$(submit 3 12 "$S2"); echo "stage 3: ${S3:-DID NOT SUBMIT}  (after $S2)"

EV=$(CKPT="$O/a1_${SLUG}_stage3/pytorch_model.bin" BENCHMARK=cvqa LANGS="$LANGS" \
  VISUAL_PIXELS="$MATCHED_PIXELS" TAG="a1_${SLUG}_matched" \
  sbatch --job-name="a1${A1_LANG}${RUN_TAG}ev" --dependency=afterok:"$S3" --time=12:00:00 \
  Approach2/job-scripts/a1_eval.sh | awk '{print $NF}')
echo "eval:    ${EV:-DID NOT SUBMIT}  (tag a1_${SLUG}_matched, langs: $LANGS)"
echo
echo "Controls, both already scored: a1_${A1_LANG}_matched (1 epoch) and"
echo "a1_pooled_matched (3 epochs). The new arm matches the pooled one on epochs."
echo "Watch with: squeue -u \$USER"
