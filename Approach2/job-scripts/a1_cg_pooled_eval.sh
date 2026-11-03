#!/bin/bash
# Evaluate the pooled cultural arm and fill the control's missing panels.
#
# The six matched panels (bn zh id pt ru ko), the four zero-shot targets
# (jv mn ga si) and xGQA-bn for cost, all in one line.
#
# **The control is incomplete and that is the easy mistake here.** The plain
# pooled arm was only ever scored on jv/mn/ga/si/bn/id, so zh/pt/ru/ko have no
# reference. Those four are submitted for the control too, under its existing
# TAG: a1_eval.sh skips non-empty outputs, so it fills the gaps and rewrites
# nothing.
#
#   bash Approach2/job-scripts/a1_cg_pooled_eval.sh
#
# Env: CG_CKPT, CTRL_CKPT, MATCHED_PIXELS, DT
set -uo pipefail

[ -d Approach2/job-scripts ] || { echo "run this from the repo root"; exit 1; }
O="$PWD/Approach2/outputs"
CG_CKPT="${CG_CKPT:-$O/a1_pooled_cg_stage3/pytorch_model.bin}"
CTRL_CKPT="${CTRL_CKPT:-$O/a1_pooled_stage3/pytorch_model.bin}"
MATCHED_PIXELS="${MATCHED_PIXELS:-602112}"
ALL="bn zh id pt ru ko jv mn ga si"
MISSING="zh pt ru ko"

for f in "$CG_CKPT" "$CTRL_CKPT"; do
  [ -f "$f" ] || { echo "ERROR: no checkpoint at $f"; exit 1; }
done
# No complete.json check: that marker is written by OUR stage-3 trainer, not by
# hers, so no a1 arm has one and the warning fired on every healthy run.
# Completion for an a1 arm is the `=== Done === ... status=0` line in its log,
# and her trainer saves pytorch_model.bin per improving epoch rather than at the
# end -- so the file existing is not completion either.
LOG_DONE=$(grep -lh "=== Done ===.*arm=pooled_cg status=0" Approach2/logs/a1_train_*.log 2>/dev/null | tail -1)
[ -n "$LOG_DONE" ] || echo "WARNING: no log line says arm=pooled_cg finished with status=0"

submit () {  # submit <name> <ckpt> <benchmark> <pixels> <tag> <langs>
  local name="$1" ckpt="$2" bench="$3" px="$4" tag="$5"; shift 5
  local id
  id=$(CKPT="$ckpt" BENCHMARK="$bench" VISUAL_PIXELS="$px" LANGS="$*" TAG="$tag" \
    sbatch --job-name="$name" --time=12:00:00 \
    Approach2/job-scripts/a1_eval.sh | awk '{print $NF}')
  echo "$name: ${id:-DID NOT SUBMIT}  ($bench, $px px, tag $tag) -> $*"
}

submit cgpool_cvqa  "$CG_CKPT"   cvqa "$MATCHED_PIXELS" a1_pooled_cg_matched      $ALL
submit ctrl_fill    "$CTRL_CKPT" cvqa "$MATCHED_PIXELS" a1_pooled_matched         $MISSING
submit cgpool_xgqa  "$CG_CKPT"   xgqa 65536             a1_pooled_cg_her_default  bn
echo
echo "Watch with: squeue -u \$USER"
echo "The Chinese China/Singapore split needs a per-item subset map; it is not"
echo "in the legacy rows (DESIGN 2026-10-05) and is built separately."
