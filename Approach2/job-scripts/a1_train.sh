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
# Env: STAGE (1|2|3), A1_LANG (default bn), DT (required), A1_ROOT, EPOCHS
set -uo pipefail

ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:?set STAGE=1, 2 or 3}"
# Never LANG: that is the shell's locale variable, which SLURM re-exports into
# the job, so ${LANG:-bn} resolves to en_US.UTF-8 and the language lookup dies.
A1_LANG="${A1_LANG:-bn}"
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

S1_OUT="$OUT/a1_${A1_LANG}_stage1"
S2_OUT="$OUT/a1_${A1_LANG}_stage2"
S3_OUT="$OUT/a1_${A1_LANG}_stage3"

require () { [ -e "$1" ] || { echo "ERROR: missing $1"; exit 1; }; }

cd "$A1_ROOT"
case "$STAGE" in
  1)
    NAME="${FULL_NAME[$A1_LANG]:?no full language name recorded for $A1_LANG}"
    require "$DT/Stage1/data/${NAME}_to_English.jsonl"
    python -c "import deepspeed" 2>/dev/null || {
      echo "ERROR: stage 1 needs deepspeed, which this venv does not have."
      echo "  pip install --no-index deepspeed"
      echo "Or skip stage 1: her Stage2 takes --stage1-mapping-ckpt as optional,"
      echo "but then the text bridge never gets its text-only alignment, which is"
      echo "the core of what a1 claims — say so in the paper if you go that way."
      exit 1
    }
    mkdir -p "$S1_OUT"
    # Her Stage 1 through our wrapper, which turns off DeepSpeed's CPU offload:
    # that default forces DeepSpeedCPUAdam, whose AVX-512 kernel will not build
    # against this toolchain's -march=x86-64-v3 (job 22218332). Her code itself is
    # untouched; the wrapper patches the one default and calls her main.
    A1_ROOT="$A1_ROOT" deepspeed --master_port "${PORT[$A1_LANG]}" \
      "$ROOT/Approach2/merged/a1_stage1_no_offload.py" --deepspeed \
      --llm_path "$LLM" --mt_path "$MT" \
      --save_name "a1-$A1_LANG" --output_dir "$S1_OUT" \
      --stage_name mapping --task nllb_corpus --augmentation False \
      --nllb_data_dir "$DT/Stage1/data" --nllb_languages "$NAME" \
      --train_num 100000 --val_size 3000 \
      --train_batch_size 24 --train_micro_batch_size_per_gpu 1 \
      --epoch_num "${EPOCHS:-1}" --max_seq_len 256 --max_gen_len 256 \
      --eval_batch_size 2
    ;;
  2)
    WIT="$DT/Stage2/data/$A1_LANG/wit_pairs.jsonl"
    CC3M="$DT/Stage2/data/$A1_LANG/cc3m_pairs.jsonl"
    require "$WIT"
    DATA=("$WIT"); CACHES=("$DT/Stage2/data/$A1_LANG/image_cache")
    if [ -f "$CC3M" ]; then
      DATA+=("$CC3M"); CACHES+=("$DT/Stage2/data/cc3m/image_cache")
    else
      echo "NOTE: no $CC3M — training stage 2 on WIT only"
    fi
    for c in "${CACHES[@]}"; do require "$c"; done
    S1_CKPT=$(ls "$S1_OUT"/*.bin "$S1_OUT"/**/*.bin 2>/dev/null | head -1)
    INIT=()
    if [ -n "$S1_CKPT" ]; then
      INIT=(--stage1-mapping-ckpt "$S1_CKPT"); echo "warm start: $S1_CKPT"
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
    require "$DT/Stage3/data/$A1_LANG.jsonl"
    require "$DT/Stage3/data/gqa/images"
    S2_CKPT=$(ls "$S2_OUT"/pytorch_model.bin "$S2_OUT"/*.bin 2>/dev/null | head -1)
    [ -n "$S2_CKPT" ] || { echo "ERROR: no stage-2 checkpoint under $S2_OUT; run STAGE=2 first"; exit 1; }
    mkdir -p "$S3_OUT"
    python -u Stage3/train.py \
      --data-dir "$DT/Stage3/data/$A1_LANG.jsonl" \
      --images-dir "$DT/Stage3/data/gqa/images" \
      --output-dir "$S3_OUT" --init-mapping-ckpt "$S2_CKPT" \
      --mt-path "$MT" --llm-path "$LLM" \
      --epochs "${EPOCHS:-3}" --lr 2e-5 \
      --train-batch-size 4 --eval-batch-size 4 --grad-accum 8 \
      --max-mt-seq-len 256 --max-seq-len 256 --max-gen-len 16 \
      --local-files-only
    ;;
  *) echo "ERROR: STAGE must be 1, 2 or 3"; exit 1 ;;
esac
STATUS=$?
echo "=== Done === $(date) stage=$STAGE lang=$A1_LANG status=$STATUS her_code=$A1_SHA"
exit $STATUS
