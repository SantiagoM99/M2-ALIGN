#!/bin/bash
#SBATCH --job-name=s1_smoke
#SBATCH --account=def-annielee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --signal=USR1@120
#SBATCH --gres=gpu:1
#SBATCH --output=Approach2/logs/s1_smoke_%j.log
# Smoke test of the S1 stage-3 trainer on a real H100: 120 Bengali rows, the
# exact C1 arguments (two epochs, dense layers, deterministic algorithms), in
# a throwaway output directory. It exists only to prove that
# torch.use_deterministic_algorithms(True) has a kernel for every op in the
# Gemma 2 / SigLIP 2 / NLLB path before the real C pilot is submitted.
# Not an experiment: nothing here is read as a result.
set -euo pipefail
ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DT="${DT:-/scratch/santimn/datatransfer}"
module --force purge
module load StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0
source "$SCRATCH/venvs/m2-align/bin/activate"
export HF_HOME="$SCRATCH/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
SMOKE="$ROOT/Approach2/outputs/smoke_stage3_s1_${SLURM_JOB_ID}"
mkdir -p "$SMOKE"
head -n 120 "$DT/Stage3/data/bn.jsonl" > "$SMOKE/rows.jsonl"
srun python -u Approach2/train_stage3_vqa.py --s1 \
  --data-path "$SMOKE/rows.jsonl" \
  --images-dir "$DT/Stage3/data/gqa/images" \
  --output-dir "$SMOKE/run" \
  --stage1-ckpt Approach2/outputs/stage1/mapping/pytorch_model.bin \
  --stage2-ckpt Approach2/outputs/stage2_dc_llava/mapping/pytorch_model.bin \
  --vis-layers 9,18,-1 --epochs 2 --seed 13 \
  --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 --lr 2e-5 --save-steps 200 \
  --local-files-only
python Approach2/train_stage3_vqa.py --s1 --data-path "$SMOKE/rows.jsonl" \
  --images-dir "$DT/Stage3/data/gqa/images" --output-dir "$SMOKE/run" \
  --stage1-ckpt Approach2/outputs/stage1/mapping/pytorch_model.bin \
  --stage2-ckpt Approach2/outputs/stage2_dc_llava/mapping/pytorch_model.bin \
  --vis-layers 9,18,-1 --epochs 2 --seed 13 \
  --train-batch-size 2 --eval-batch-size 2 --grad-accum 16 --lr 2e-5 --save-steps 200 \
  --local-files-only --check-complete && echo "SMOKE OK: deterministic training path completes on this GPU"
