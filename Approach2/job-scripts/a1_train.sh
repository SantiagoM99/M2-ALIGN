#!/bin/bash
#SBATCH --job-name=a1_train
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/a1_train_s%x_%j.log
#
# Train Approach 1 here, one stage per submission, from Maryam's own code.
#
# Why train it ourselves when a checkpoint is one file transfer away: every
# comparison then lives in one environment with one trainer. This project has
# lost weeks to environment boundaries (DESIGN 09-15), and our own rules forbid
# pairing numbers across them, so an a1 trained here makes every contrast
# internally valid without an asterisk. The cost is GPU time, not data: the
# NLLB sentence pairs, the WIT and CC3M captions and the translated GQA are all
# already under $DT.
#
# Her code is run from a WORKTREE of upstream/parallel, never merged into this
# branch: our Stage1-3 are an older lineage and a merge would conflict across
# her whole pipeline. The worktree also pins which commit of hers trained this.
#   git fetch upstream && git worktree add $SCRATCH/a1 upstream/parallel
#
#   STAGE=1 A1_LANG=bn DT=... sbatch Approach2/job-scripts/a1_train.sh
#   STAGE=2 A1_LANG=bn DT=... sbatch Approach2/job-scripts/a1_train.sh
#   STAGE=3 A1_LANG=bn DT=... sbatch Approach2/job-scripts/a1_train.sh
#
# POOLED=1 reproduces her current system instead of her per-language track, and
# mirrors her own train_pooled.sh at each stage (11 languages for stages 1-2,
# plus English for stage 3, 3 epochs everywhere). Stage 1 pooled is ~11x the
# data, so it needs the resources her launcher asks for, which only the sbatch
# line can set:
#
#   POOLED=1 STAGE=1 sbatch --gres=gpu:4 --cpus-per-task=48 --mem=490G \
#     --time=2-00:00:00 Approach2/job-scripts/a1_train.sh
#   POOLED=1 STAGE=2 sbatch Approach2/job-scripts/a1_train.sh
#   POOLED=1 STAGE=3 sbatch Approach2/job-scripts/a1_train.sh
#
# REPLAY=1 (stage 3 only) runs her train_replay.py arm instead, which is her
# best CVQA result (+1.63 over pooled, her message 10-01) and therefore the
# arm a comparison against "her current numbers" actually means.
#
# Env: STAGE (1|2|3), A1_LANG (default bn), POOLED, POOL_LANGS, INCLUDE_ENGLISH,
#      REPLAY, REPLAY_LANGS, REPLAY_EVERY, DT, A1_ROOT, EPOCHS
set -uo pipefail

ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:?set STAGE=1, 2 or 3}"
# Never LANG: that is the shell's locale variable, which SLURM re-exports into
# the job, so ${LANG:-bn} resolves to en_US.UTF-8 and the language lookup dies.
A1_LANG="${A1_LANG:-bn}"
# POOLED=1 trains her current architecture: one shared mapping over every
# language, which is what replaced her per-language runs ("one shared checkpoint
# over all 11 languages, replacing the per-language train.sh runs", her
# train_pooled.sh). A per-language a1 is her superseded track, so a comparison
# against her current system has to be pooled.
POOLED="${POOLED:-0}"
POOL_LANGS="${POOL_LANGS:-bn ru de zh pt id ko jv mn si ga}"
INCLUDE_ENGLISH="${INCLUDE_ENGLISH:-1}"
REPLAY="${REPLAY:-0}"
REPLAY_LANGS="${REPLAY_LANGS:-bn de ru zh}"
REPLAY_EVERY="${REPLAY_EVERY:-3}"
DT="${DT:-/scratch/santimn/datatransfer}"
A1_ROOT="${A1_ROOT:-$SCRATCH/a1}"
OUT="${OUT:-$ROOT/Approach2/outputs}"
LLM="${LLM:-Qwen/Qwen3-VL-8B-Instruct}"
MT="${MT:-facebook/nllb-200-distilled-600M}"

# Her Stage 1 reads the NLLB pairs by full language name, the way its loader
# wrote them (`<Language>_to_English.jsonl`).
declare -A FULL_NAME=(
  [bn]=Bengali [ru]=Russian [de]=German [zh]=Chinese [pt]=Portuguese
  [id]=Indonesian [ko]=Korean [jv]=Javanese [mn]=Mongolian [si]=Sinhalese [ga]=Irish
)
# One deepspeed rendezvous port per language, so two languages can train at once.
declare -A PORT=(
  [bn]=50010 [ru]=50011 [de]=50012 [zh]=50013 [pt]=50017
  [id]=50018 [ko]=50019 [jv]=50020 [mn]=50021 [si]=50022 [ga]=50023
)

echo "=== Job info ==="; date; hostname; nvidia-smi || true

[ -d "$A1_ROOT/Stage3" ] || {
  echo "ERROR: no worktree of her branch at $A1_ROOT."
  echo "  git fetch upstream && git worktree add $A1_ROOT upstream/parallel"
  exit 1
}
A1_SHA=$(git -C "$A1_ROOT" rev-parse HEAD)
echo "her code: $A1_ROOT at $A1_SHA"
echo "DT=$DT"

# Her modules, not ours: her Stage 1 lets deepspeed JIT-compile CPUAdam, and
# that refuses to build unless the loaded CUDA matches the one torch was built
# against. Ours loads cudacore/.12.2.2 for the Gemma pipeline; torch here is
# built against 13.2, and `cuda/13.2` is exactly what her own launcher loads.
module --force purge
module load StdEnv/2023 python/3.11.5 gcc/12.3 cuda/13.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SLUG="$A1_LANG"
[ "$POOLED" = 1 ] && SLUG="pooled"
S1_OUT="$OUT/a1_${SLUG}_stage1"
S2_OUT="$OUT/a1_${SLUG}_stage2"
# The replay arm shares stages 1 and 2 with the pooled arm and only forks at
# stage 3, exactly as her launchers do, so it must not overwrite pooled's
# stage-3 checkpoint or its pooled data directory.
S3_SLUG="$SLUG"
[ "$POOLED" = 1 ] && [ "$REPLAY" = 1 ] && S3_SLUG="pooled_replay"
S3_OUT="$OUT/a1_${S3_SLUG}_stage3"
POOL_DATA="$OUT/a1_${S3_SLUG}_stage3_data"
echo "arm: $S3_SLUG (POOLED=$POOLED REPLAY=$REPLAY)"

require () { [ -e "$1" ] || { echo "ERROR: missing $1"; exit 1; }; }

# A whole node is only ever requested for the pooled run, so a job holding four
# GPUs with POOLED=0 means the flag was lost between the shell and sbatch --
# which is what happened to job 22343575, where a mangled command line spent 51
# minutes of a 4-GPU node retraining Bengali.
if [ "$POOLED" != 1 ] && [ "$(nvidia-smi --list-gpus 2>/dev/null | wc -l)" -ge 4 ]; then
  echo "ERROR: $(nvidia-smi --list-gpus | wc -l) GPUs allocated but POOLED=0."
  echo "  The pooled flag did not reach the job. Submit with Approach2/job-scripts/a1_pooled.sh,"
  echo "  which sets the mode and the resources together."
  exit 1
fi

cd "$A1_ROOT"
case "$STAGE" in
  1)
    if [ "$POOLED" = 1 ]; then
      # Her Stage 1 takes a comma-separated list and reads the union; her note
      # says the mapping stage stays English-free, so English is not added here
      # even when INCLUDE_ENGLISH=1 (that flag is about stage 3).
      NAME=""
      for L in $POOL_LANGS; do
        full="${FULL_NAME[$L]:?no full language name recorded for $L}"
        require "$DT/Stage1/data/${full}_to_English.jsonl"
        NAME="${NAME:+$NAME,}$full"
      done
      echo "stage 1 pooled over: $NAME"
      # Her launcher asks for a whole node because the pool is ~1.07M rows x 3
      # epochs: she measured 13.5 h for one language on one A100-40, so ~150 h
      # on one GPU — past the 7-day limit with no margin. train.py derives
      # grad-accum from train_batch_size 24 / n_gpus, so 4 GPUs is the same
      # effective batch as 1, just 4x faster.
      GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
      [ "$GPUS" -ge 4 ] || echo "WARNING: $GPUS GPU(s) visible; her pooled stage 1 uses 4. Expect ~150 h and resubmit with --gres=gpu:4 --cpus-per-task=48 --mem=490G --time=2-00:00:00"
    else
      NAME="${FULL_NAME[$A1_LANG]:?no full language name recorded for $A1_LANG}"
      require "$DT/Stage1/data/${NAME}_to_English.jsonl"
    fi
    python -c "import deepspeed" 2>/dev/null || {
      echo "ERROR: stage 1 needs deepspeed, which this venv does not have."
      echo "  pip install --no-index deepspeed"
      echo "Or skip stage 1: her Stage2 takes --stage1-mapping-ckpt as optional,"
      echo "but then the text bridge never gets its text-only alignment, which is"
      echo "the core of what a1 claims — say so in the paper if you go that way."
      exit 1
    }
    # Her per-language runs are 1 epoch here; her pooled run is 3, matching
    # MindMerger's mapping stage.
    S1_EPOCHS="${EPOCHS:-1}"
    [ "$POOLED" = 1 ] && S1_EPOCHS="${EPOCHS:-3}"
    mkdir -p "$S1_OUT"
    # Her Stage 1 through our wrapper, which turns off DeepSpeed's CPU offload:
    # that default forces DeepSpeedCPUAdam, whose AVX-512 kernel will not build
    # against this toolchain's -march=x86-64-v3 (job 22218332). Her code itself is
    # untouched; the wrapper patches the one default and calls her main.
    # The pooled run is one job over every language, so it needs a port of its
    # own rather than any one language's.
    PORT_FOR="${PORT[$A1_LANG]}"
    [ "$POOLED" = 1 ] && PORT_FOR=50009
    A1_ROOT="$A1_ROOT" deepspeed --master_port "$PORT_FOR" \
      "$ROOT/Approach2/merged/a1_stage1_no_offload.py" --deepspeed \
      --llm_path "$LLM" --mt_path "$MT" \
      --save_name "a1-$SLUG" --output_dir "$S1_OUT" \
      --stage_name mapping --task nllb_corpus --augmentation False \
      --nllb_data_dir "$DT/Stage1/data" --nllb_languages "$NAME" \
      --train_num 100000 --val_size 3000 \
      --train_batch_size 24 --train_micro_batch_size_per_gpu 1 \
      --epoch_num "$S1_EPOCHS" --max_seq_len 256 --max_gen_len 256 \
      --eval_batch_size 2
    ;;
  2)
    DATA=(); CACHES=()
    S2_LANGS="$A1_LANG"
    [ "$POOLED" = 1 ] && S2_LANGS="$POOL_LANGS"
    for L in $S2_LANGS; do
      WIT="$DT/Stage2/data/$L/wit_pairs.jsonl"
      CC3M="$DT/Stage2/data/$L/cc3m_pairs.jsonl"
      if [ -f "$WIT" ]; then
        DATA+=("$WIT"); CACHES+=("$DT/Stage2/data/$L/image_cache")
      elif [ "$POOLED" = 1 ]; then
        # Dropping a language silently would make this a different recipe from
        # the checkpoint we are claiming to reproduce.
        echo "ERROR: pooled stage 2 is missing $WIT. Set POOL_LANGS explicitly to pool fewer languages."
        exit 1
      else
        echo "NOTE: no $WIT"
      fi
      if [ -f "$CC3M" ]; then
        DATA+=("$CC3M"); CACHES+=("$DT/Stage2/data/cc3m/image_cache")
      fi
    done
    [ ${#DATA[@]} -gt 0 ] || { echo "ERROR: no stage-2 caption files found"; exit 1; }
    echo "stage 2 over ${#DATA[@]} caption files"
    for c in "${CACHES[@]}"; do require "$c"; done
    S1_CKPT=$(ls "$S1_OUT"/*.bin "$S1_OUT"/**/*.bin 2>/dev/null | head -1)
    INIT=()
    if [ -n "$S1_CKPT" ]; then
      INIT=(--stage1-mapping-ckpt "$S1_CKPT"); echo "warm start: $S1_CKPT"
    elif [ "$POOLED" = 1 ]; then
      # In a chained pooled run this must never be a warning: the job would
      # train a different recipe from the one being reproduced, succeed, and
      # hand stage 3 a checkpoint nobody can describe.
      echo "ERROR: no stage-1 checkpoint under $S1_OUT, but POOLED=1 warm-starts from it."
      echo "  Check that STAGE=1 wrote it before letting the dependency run."
      exit 1
    else
      echo "WARNING: no stage-1 checkpoint under $S1_OUT; stage 2 starts cold."
      echo "That is a different recipe from hers and has to be reported as such."
    fi
    mkdir -p "$S2_OUT"
    python -u Stage2/train.py \
      --data-path "${DATA[@]}" --image-cache-dir "${CACHES[@]}" \
      --output-dir "$S2_OUT" ${INIT[@]+"${INIT[@]}"} \
      --mt-path "$MT" --llm-path "$LLM" \
      --epochs "${EPOCHS:-3}" --lr 2e-5 \
      --train-batch-size 4 --eval-batch-size 4 --grad-accum 8 \
      --max-mt-seq-len 512 --max-gen-len 512 --save-steps 200 \
      --local-files-only
    ;;
  3)
    require "$DT/Stage3/data/gqa/images"
    if [ "$POOLED" = 1 ]; then
      # Her Stage3/train.py globs *.jsonl in --data-dir and shuffles the union,
      # so pooling is a directory of links rather than a concatenated file.
      rm -rf "$POOL_DATA"; mkdir -p "$POOL_DATA"
      POOLED_IN=""
      for L in $POOL_LANGS; do
        f="$DT/Stage3/data/$L.jsonl"
        require "$f"
        ln -sf "$f" "$POOL_DATA/$L.jsonl"; POOLED_IN="$POOLED_IN $L"
      done
      # English is in her pool by default: she verified against MindMerger's
      # read_datasets.py that the augmentation stage includes it (the mapping
      # stage does not). Pooling without it is a different recipe, so it has to
      # be asked for rather than fallen into.
      if [ "$INCLUDE_ENGLISH" = 1 ]; then
        require "$DT/Stage3/data/en.jsonl"
        ln -sf "$DT/Stage3/data/en.jsonl" "$POOL_DATA/en.jsonl"; POOLED_IN="$POOLED_IN en"
      else
        echo "INCLUDE_ENGLISH=0 — the no-English ablation, not her recipe"
      fi
      echo "stage 3 pooled over:$POOLED_IN"
      S3_DATA="$POOL_DATA"
    else
      require "$DT/Stage3/data/$A1_LANG.jsonl"
      S3_DATA="$DT/Stage3/data/$A1_LANG.jsonl"
    fi
    TRAINER=Stage3/train.py
    REPLAY_ARGS=()
    if [ "$REPLAY" = 1 ]; then
      [ -f Stage3/train_replay.py ] || { echo "ERROR: $A1_SHA has no Stage3/train_replay.py"; exit 1; }
      FILES=()
      for L in $REPLAY_LANGS; do
        FILES+=("$DT/Stage3/data/replay/translation_$L.jsonl" "$DT/Stage3/data/replay/math_$L.jsonl")
      done
      [ "$INCLUDE_ENGLISH" = 1 ] && FILES+=("$DT/Stage3/data/replay/math_en.jsonl")
      for f in "${FILES[@]}"; do require "$f"; done
      TRAINER=Stage3/train_replay.py
      REPLAY_ARGS=(--replay-data "$(IFS=,; echo "${FILES[*]}")"
                   --replay-every "$REPLAY_EVERY"
                   --replay-max-gen-len 512 --replay-max-query-len 1024
                   --gradient-checkpointing)
      echo "replay over ${#FILES[@]} files in: $REPLAY_LANGS, every $REPLAY_EVERY VQA batches"
    fi
    S2_CKPT=$(ls "$S2_OUT"/pytorch_model.bin "$S2_OUT"/*.bin 2>/dev/null | head -1)
    [ -n "$S2_CKPT" ] || { echo "ERROR: no stage-2 checkpoint under $S2_OUT; run STAGE=2 first"; exit 1; }
    mkdir -p "$S3_OUT"
    python -u "$TRAINER" \
      --data-dir "$S3_DATA" \
      --images-dir "$DT/Stage3/data/gqa/images" \
      --output-dir "$S3_OUT" --init-mapping-ckpt "$S2_CKPT" \
      --mt-path "$MT" --llm-path "$LLM" \
      --epochs "${EPOCHS:-3}" --lr 2e-5 \
      --train-batch-size 4 --eval-batch-size 4 --grad-accum 8 \
      --max-mt-seq-len 256 --max-seq-len 256 --max-gen-len 16 \
      ${REPLAY_ARGS[@]+"${REPLAY_ARGS[@]}"} \
      --local-files-only
    ;;
  *) echo "ERROR: STAGE must be 1, 2 or 3"; exit 1 ;;
esac
STATUS=$?
echo "=== Done === $(date) stage=$STAGE arm=$S3_SLUG status=$STATUS her_code=$A1_SHA"
exit $STATUS
