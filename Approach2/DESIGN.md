# Approach 2 — Design Log

Running record of every design decision, why it was made, and the measured
effect of each change. One entry per decision, chronological. Numbers cite the
summary JSONs in `Approach2/results/`; "ΔV" = full-image accuracy minus blind
(gray-image) accuracy, i.e. how much the model actually extracts from pixels.

**Architecture in one line**: frozen NLLB-200-600M (text) + frozen
SigLIP2-so400m-384 (vision) → two trainable MLP mappings (~40M params) →
frozen Gemma-2-9b-it. Prefix: `[BOS] + X_f + b_txt + V_f + b_vis + T`.
Curriculum: stage 1 text-only, stage 2 vision-only, stage 3 joint VQA
warm-started from both.

---

## Decision log

### D1 — Initial pipeline (2026-07-15)
Encoder-injection paradigm (MindMerger/LangBridge) extended with a second,
visual mapping into a text-only LLM. Everything frozen except the mappings, so
any capability change is attributable to the 40M trainable params.
**Result**: xGQA-bn 32.0.

### D2 — Blind (gray-image) control (2026-07-21)
Replace every image with a 384×384 gray canvas at eval; the same trained
stack minus image content. Decomposes accuracy into language prior vs vision.
**Finding**: xGQA-bn blind 30.7 vs full 32.0 → ΔV only +1.3; the model was
nearly vision-blind. Diagnosis: stage-2 data-limited (745 usable WIT pairs).
Also discovered the reasoning collapse: MGSM-bn 74.8 (raw Gemma) → ~12 with
the mapping prefix. Mechanism verified in samples: fluent English CoT over
corrupted premises → the OOD prefix actively interferes (frozen LLM rules out
weight updates as cause).

### D3 — Protocol parity with Approach 1 (2026-08-17)
Adopted Maryam's exact prompt (`"...Answer with a single word or short
phrase, in English."`), her CVQA answer-choice log-likelihood protocol, and
verified her answer normalization shifts our accuracy by 0.00pp (full) /
≤0.25pp (blind). **Effect**: no accuracy claim; makes every cross-approach
comparison legitimate. Pre-/post-prompt-change numbers are NOT comparable.

### D4 — Stage-2 data scaling, 745 → ~11.5k pairs (2026-08-19)
CC3M translated set (English captions; only one copy needed — our stage 2
requires no bilingual pairs, a structural advantage over mBLIP-style
approaches) + all languages' WIT pairs. Plus recipe fix v2: 10 epochs,
lr 1e-4, grad-accum 2 (was 1 epoch lr 2e-5 → val_ppl 48.8; v2 → 11.7@ep4).
**Result**: ΔV +1.3 → **+11.0** (7-lang xGQA avg blind 32.4 → full 43.3).
The single most effective change so far. Key asymmetry: the same extra data
did nothing for Approach 1 (Qwen already sees; our mapping IS the vision
channel).

### D5 — Multi-language round B (2026-08-19)
11 languages, per-language stage 1/3, shared stage 2. **Scoreboard**: xGQA
(7 langs) 43.3 full / 32.4 blind; CVQA (10 langs) 42.6 / 33.9 — beats
Approach 1 (M2RB 38.81) in 8/10 languages and the zero-shot Qwen3-VL
baseline (40.75). Category breakdown: vision gains concentrate in
object/other (+18.5 ΔV); spatial is flat (−0.6) — captions teach content,
not geometry.

### D6 — Reasoning replay in stage 3 (2026-08-25) — ACCEPTED
Text-only replay batches (GSM8K questions NLLB-translated to bn with
eval-exact math prompts, + stage-1 translation rows) interleaved every 3
VQA steps (IJCNLP'25 IFL / MindMerger stage-2, applied to our pipeline).
**Result**: MGSM 9.2 → **37.6**, MSVAMP 36.2 → **49.8**, xGQA untouched
(41.2 → 41.3 full, 29.9 → 31.0 blind). ~14% wall-clock overhead. Confirms
the D2 interference diagnosis causally. Now part of the standard recipe.

### D7 — Zero-init prefix gate (2026-08-25) — REJECTED (as trained)
LLaMA-Adapter-style learnable scalar gate on each mapping, initialized to 0
in stage 3 so the prefix must earn influence. **Result**: best reasoning yet
(MGSM 54.4, MSVAMP 56.5) but the vision channel never re-opened in 1 epoch:
xGQA collapsed to 19.4 full / 10.1 blind. Defines the trade-off frontier;
possible future work with a gradual gate ramp. The `gate` parameter remains
in `Mapping` (init 1.0 = identity, old checkpoints unaffected).

### D8 — Qwen3-VL zero-shot 2×2 baseline (2026-08-26)
Ported Maryam's `Baseline/evaluate.py` verbatim + `--blind` + per-item
output. **Protocol validation**: our port reproduces her numbers to 0.04pp
(xGQA 53.00 vs 52.96; CVQA 40.75 vs 40.65).

| avg | xGQA full | xGQA blind | ΔV | CVQA full | CVQA blind | ΔV |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B zero-shot | 53.0 | 23.2 | +29.8 | 40.75 | 33.78 | +6.97 |
| A2 (frozen Gemma2 + 40M) | 43.3 | 32.4 | +11.0 | **42.56** | 33.89 | **+8.67** |

- xGQA: A2's −9.7 gap = −18.8 vision extraction + +9.2 trained format prior
  (both McNemar-significant, n=88k paired items).
- CVQA: blind priors statistically identical (p=0.87) → A2's +1.8 win
  (8/10 langs) is genuinely visual; pooled McNemar p=0.135 (consistent
  trend, not significant at n=2943).

Category breakdown of the xGQA gap (7 langs pooled, by gold-answer type):

| category | n | Qwen ΔV | A2 ΔV | share of the 9.7pp gap |
|---|---|---|---|---|
| object/other | 44k | +30.8 | +18.5 | 12% (full acc: 36.6 vs 34.2 — near parity) |
| yes/no | 32k | +22.0 | +3.2 | **54%** |
| spatial | 5k | +62.5 | −0.6 | 17% |
| color | 5k | +38.2 | +7.8 | 12% |
| material | 2k | +26.6 | +4.3 | 4% |

**Diagnosis**: the mapping transmits *what* is in the image (object naming
near-matches an 8B VLM) but not structure — verification (yes/no), spatial
layout, fine attributes. Consistent with caption-only stage-2 supervision:
captions carry content words, never negative evidence or precise geometry.
Not a token-budget issue: all 729 SigLIP2 patches pass through, in raster
order.

### D9 — DenseConnector multi-layer vision features (2026-08-26) — ACCEPTED
DCI variant: channel-concatenate SigLIP2 hidden states from layers 9, 18 and
the post-layernorm final layer per patch before the vision mapping
(`--vis-layers "9,18,-1"`; mapping input 1152→3456). Motivated by D8:
shallow/mid ViT layers carry the low-level attribute and layout information
(color, texture, position) that the final contrastive-aligned layer
discards — exactly the failing categories. ~15-line change in
`model.py::_encode_vision`; default `""` keeps old behavior so existing
checkpoints load unchanged. Checkpoints are only interchangeable under the
same `--vis-layers`; stage 3 and all evals must pass the value the stage-2
checkpoint was trained with.

**Ablation discipline**: the pilot (`job-scripts/pilot_dense.sh`) keeps
stage 3 at 1 epoch with replay — identical to D6's `stage3_bn_replay` in
every respect except the connector, so the delta is attributable. The
stage-3-epochs lever (targets yes/no, learned from GQA itself, not from
captions) is tested separately afterwards by re-running with `S3_EPOCHS=2`
(reuses the stage-2 DC checkpoint).

**Result** (bn, job 19664268, ~10h wall-clock):

| bench | baseline | + DenseConnector | Δ |
|---|---|---|---|
| xGQA full | 41.26 | **44.23** | **+2.97** |
| xGQA blind | 31.01 | 30.93 | −0.08 |
| xGQA ΔV | +10.25 | **+13.30** | +3.05 |
| CVQA full | 39.16† | 39.16 | ±0.00 |
| CVQA blind | 32.52† | 28.32 | −4.20 |
| CVQA ΔV | +6.64† | **+10.84** | +4.20 |
| MGSM | 37.6 | 32.4 | −5.2 (n=250, ~1.7σ) |
| MSVAMP | 49.8 | 49.4 | −0.4 (noise) |

† CVQA baseline is stage3_bn_v2 — the replay pilot didn't run CVQA; D6
showed replay leaves VQA intact, so the comparison stands with that caveat.

**Reading**: the +2.97 xGQA gain is entirely visual — blind is unchanged, so
ΔV rises 1:1 with full accuracy. First genuine vision-extraction gain since
D4 and the best xGQA-bn to date. Unpaired z ≈ 4.8 at n=12,578 (paired
McNemar pending per-item harvest). CVQA tells the same story differently:
identical full accuracy but −4.2 blind — the model answers the same
questions leaning less on prior and more on pixels. MSVAMP flat; MGSM −5.2
is ~1.7σ at n=250 — watch whether the e2 arm recovers it (more replay
steps). Paired McNemar on the full-image runs: b/c = 1161/1534,
p = 6.7e-13. **ACCEPTED into the standard recipe.**

**Category attribution** (bn, paired per-item,
`analysis/xgqa_category_breakdown.py`) — every category moved in the
predicted direction:

| category | n | Δ full | ΔV: replay → DC | McNemar p |
|---|---|---|---|---|
| color | 758 | +5.80 | +3.8 → **+16.0** | 3e-4 |
| yes/no | 4525 | +4.38 | +2.5 → +5.8 | 9e-8 |
| spatial | 713 | +3.37 | **−0.3 → +3.7** | 0.10 |
| material | 309 | +2.27 | +0.7 → +4.5 | 0.35 |
| object/other | 6273 | +1.59 | +18.3 → +19.9 | 9e-4 |

Color is the standout (ΔV ×4, exactly the shallow-layer signal the final
contrastive layer discards) and spatial turns positive for the first time.
The two small-n categories (spatial, material) are trends, not yet
significant. yes/no contributes the largest share of the total gain by
volume but its ΔV (+5.8) is still far from Qwen's (+22) — the D8 diagnosis
stands: verification needs training signal (stage-3 epochs / existence-QA),
not just better features.

---

## Improvement queue (evidence-ranked, 2026-08-26)

1. ~~DenseConnector~~ → D9 ACCEPTED (+2.97 xGQA full, all of it ΔV).
2. Stage-3 epochs ≥2 (yes/no verification is learned in stage 3; currently
   1 epoch) — free, `S3_EPOCHS=2` reuses the stage-2 DC checkpoint (~4h,
   only stage 3 + evals rerun). NEXT.
3. Stage-2 scale-up to LLaVA-Pretrain-558k (English captions only — our
   structural advantage; D4 showed the derivative: 745→11.5k ≈ +10 ΔV).
   Needs a ~30GB login-node download.
4. Honeybee C-Abstractor (2D-aware abstraction) — targets spatial (ΔV −0.6).
5. M3IT-style instruction diversity in stages 2/3, incl. existence-QA
   synthesized from captions — targets yes/no (negative evidence).
6. LwF-KL do-no-harm loss — reasoning-preservation novelty candidate,
   complementary to D6.
7. Gate ramp-up schedule — revisit D7's frontier.
