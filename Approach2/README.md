# Approach 2: frozen multilingual + vision encoders → frozen text-only LLM

Composes a multilingual VLM out of three frozen parts, training only two
lightweight mapping MLPs:

```
NLLB-200 encoder (frozen)  ──► mapping_txt ──┐
                                             ├──►  [BOS] X_f [b_txt] V_f [b_vis] T  ──► Gemma 2 (frozen)
SigLIP 2 vision tower (frozen) ► mapping_vis ┘                                  ▲
                                                     Gemma embedding of prompt ─┘
```

This contrasts with Approach 1 (Stage1-3 in this repo), which injects the
multilingual encoder into Qwen3-VL — a model that already sees. Here the
question is whether modality *and* multilinguality can be composed post-hoc
onto any text-only LLM. See MindMerger (NeurIPS 2024), MERLIN (EACL 2026)
for the text-side lineage; the vision branch follows the LLaVA projector
recipe but through a frozen LLM.

## Files

| File | Purpose |
|---|---|
| `model.py` | `DualEncoderMerger` — one class serves all stages via branch toggles |
| `common.py` | Shared tokenisation / checkpoint / W&B / prompt helpers |
| `train_stage1_text.py` | Text mapping: NLLB → Gemma on translation pairs |
| `train_stage2_vision.py` | Vision mapping: SigLIP → Gemma on WIT captioning |
| `train_stage3_vqa.py` | Joint VQA training (both mappings, warm-started) |
| `evaluate_vqa.py` | Open-ended VQA eval (xGQA etc.), same metric as Stage3 |

## Data reuse (nothing new to download)

All three stages consume the data Approach 1 already produces:

- Stage 1: `Stage1/data/{Language}_to_English.jsonl` (from `Stage1/load_text.py`)
- Stage 2: `Stage2/data/wit_pairs.jsonl` + `Stage2/data/image_cache/` (from `Stage2/load_image.py`)
- Stage 3: `Stage3/data/stage3b/*.jsonl` + GQA images (from `Stage3/load_vqa_data.py`)
- Eval:    `Stage3/data/stage3b_eval/xgqa/*.jsonl` (from `Stage3/load_vqa_eval_data.py`)

## Running (Narval)

```bash
mkdir -p Approach2/logs   # once
sbatch Approach2/job-scripts/train_stage1.sh
sbatch Approach2/job-scripts/train_stage2.sh   # independent of stage 1 — can run in parallel
sbatch Approach2/job-scripts/train_stage3.sh   # needs both checkpoints
sbatch Approach2/job-scripts/evaluate_vqa.sh
```

Models (pre-download to `$SCRATCH/huggingface` on a login node, compute
nodes are offline):

- LLM: `google/gemma-2-9b-it` (hidden 3584 — same as Qwen3-VL-8B, so
  Approach 1 mapping checkpoints are dimensionally compatible for
  warm-start experiments, `--init-checkpoint`)
- Vision: `google/siglip2-so400m-patch14-384` (729 visual tokens; needs
  `transformers >= 4.49`)
- MT: `facebook/nllb-200-distilled-600M`

## Design notes / deviations from the sketch

- Instead of a single `<sep>` (eq. 7 of the sketch), each branch keeps its
  own learnable boundary token (`b_txt`, `b_vis`), matching how the repo's
  `Mapping` class already works and giving each training stage a
  self-consistent prefix (`Stage1 = [BOS] X_f b_txt`, `Stage2 = [BOS] V_f
  b_vis`, `Stage3 = concatenation`). Strict superset of the sketch.
- τ_k (`--max-vis-tokens`) slices the *first* k visual tokens as in eq. (5).
  For a raster-scan patch order this keeps the top of the image; if τ_k
  ablations matter later, 2D pooling would be a fairer selector.
- The mappings train in fp32 while the frozen towers run in bf16 (outputs
  cast at the branch boundary), mirroring Stage 3's `llm_dtype` pattern.
- Gemma 2's chat template rejects a `system` role and injects its own
  `<bos>`; `common.format_chat_prompt` folds the system text into the user
  turn and strips the leading BOS (the model adds its own BOS embedding).

## Known risk (expected, part of the story)

A frozen LLM + MLP-only vision alignment is the weak configuration in the
literature (BLIP-2 needed a Q-Former + 129M pairs; LLaVA unfreezes the LLM).
If Stage 2/3 plateau, the planned mitigation is LoRA on the Gemma body
during Stage 3 (keeping the parameter-efficiency story), which is also what
MERLIN's stage 2 does on the text side.
