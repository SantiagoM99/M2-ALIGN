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

### D9b — Stage-3 epochs 1 → 2 (2026-08-27) — ACCEPTED
The epochs lever isolated in D9's ablation plan: a second epoch of joint VQA
training on the same `stage2_dc` checkpoint (`S3_EPOCHS=2`, job 19737101).
It also doubles the number of replay steps.

| bench | DC 1 ep | DC 2 ep | Δ |
|---|---|---|---|
| xGQA full | 44.23 | **46.34** | **+2.11** |
| xGQA blind | 30.93 | 30.98 | +0.05 |
| xGQA ΔV | +13.30 | **+15.36** | +2.06 |
| CVQA full | 39.16 | **41.61** | +2.45 |
| CVQA blind | 28.32 | 32.52 | +4.20 |
| MGSM | 32.4 | 35.6 | +3.2 |
| MSVAMP | 49.4 | **54.2** | +4.8 |

Paired McNemar on xGQA full: b/c = 613/879, p = 5.7e-12. Wins on every
benchmark with nothing traded away — blind is flat again, so the xGQA gain
is visual. MSVAMP 54.2 is the best figure measured anywhere in this project
(beats D6's replay-only 49.8) and MGSM recovers most of the D9 dip, which
is what doubling the replay steps should do.

Category attribution vs DC at 1 epoch:

| category | n | Δ full | ΔV: 1ep → 2ep | McNemar p |
|---|---|---|---|---|
| material | 309 | **+6.15** | +4.5 → +11.0 | 1e-3 |
| color | 758 | +2.77 | +16.0 → +18.7 | 0.011 |
| yes/no | 4525 | +2.72 | +5.8 → **+8.9** | 2e-6 |
| object/other | 6273 | +1.79 | +19.9 → +21.2 | 4e-6 |
| spatial | 713 | −1.26 | +3.7 → +3.1 | 0.44 |

The D8 prediction holds from the other side: yes/no barely responded to
better features (D9: +2.5 ΔV) but responds to training signal (+3.1 ΔV
here), because verification is learned from GQA, not from captions.
Spatial is now the only category that moves for neither lever — it stays
the open problem (queue item 4). **v3 runs with `S3_EPOCHS=2`.**

### D10 — Round v3: replay + DenseConnector, all 11 languages (2026-08-27)
Composition of every accepted lever into a full round: per-language stage 3
warm-started from round-B stage 1 + the shared D9 `stage2_dc` (nothing
retrained below stage 3), with text replay (D6) in every language — each
language's GSM8K math replay is NLLB-built in-job on first use
(`train_stage3_all.sh` gained `REPLAY`/`VIS_LAYERS`/per-language NLLB
tags; `--replay-default-tag` matters: translation rows carry no tag and
default to ben_Beng otherwise). `evaluate_all.sh` gained `VIS_LAYERS` and
now harvests per-item xGQA predictions for paired analyses. Launcher
`launch_v3.sh`: two chained jobs (train → eval), one GPU at a time.
`S3_EPOCHS=2` per D9b. Outputs
`stage3_<lang>_v3`, results harvested as `*_v3`. **Result**: _pending._

### D11 — Stage-2 scale-up to LLaVA-Pretrain (2026-08-28) — ACCEPTED
D4's derivative is the strongest evidence we have (745 → 11.5k pairs bought
+9.7 ΔV), and the 558k BLIP-LAION-CC-SBU captions are English-only, which
this architecture accepts as-is. `build_llava_pretrain.py` streams a seeded
sample straight out of `images.zip` into the `sha1(url).jpg` cache (no full
unzip: 558k files would threaten the scratch inode quota; a larger
`--sample` later re-extracts the same prefix plus new rows). Default sample
100k → ~111k stage-2 pairs with CC3M+WIT.

**Ablation discipline**: `job-scripts/pilot_scale.sh` keeps the D9 connector
and a stage 3 identical to `stage3_bn_dc_e2` (the D9b recipe), so stage-2
*data* is the only variable and the delta measures what LLaVA adds on top of
everything already accepted. Epochs drop 10 → 2 to hold sample-epochs comparable (11.5k×10 =
115k vs 111k×2 = 222k) — D4's best val_ppl landed at epoch 4 of 10 (~46k),
so 2 epochs is already well past that point.

**Result** (job 19754253, `stage3_bn_dcl`): the largest single-lever gain
measured in this project, and it lands on *reasoning*, not on VQA.

| bench | stage2_dc (11.5k) | + LLaVA (~111k) | Δ |
|---|---|---|---|
| xGQA full | 46.34 | **47.66** | +1.32 |
| xGQA blind | 30.98 | 30.83 | −0.15 |
| xGQA ΔV | +15.36 | **+16.83** | +1.47 |
| CVQA full | 41.61 | 39.16 | −2.45 |
| CVQA blind | 32.52 | 30.42 | −2.10 |
| CVQA ΔV | +9.09 | +8.74 | −0.35 |
| MGSM | 35.6 | **62.0** | **+26.4** |
| MSVAMP | 54.2 | **64.5** | **+10.3** |

Paired McNemar on xGQA full: b/c = 1146/1312, p = 8.1e-4; blind flat again,
so the whole VQA gain is visual. CVQA moves 7 items on n=286 (≈1σ) with ΔV
unchanged — noise, not a regression.

**The reasoning result is the finding.** Stage-2 data is vision-caption data
that never touches the text mapping, yet it nearly doubles MGSM. Placed
against the frozen-LLM ceiling (Gemma-2-9b-it answering the same items in
English: MGSM 74.8, MSVAMP 69.6), the pipeline now recovers **83%** of the
MGSM ceiling and **93%** of MSVAMP, from 9.2/36.2 before replay.

| variant | MGSM | MSVAMP | xGQA full |
|---|---|---|---|
| stage 3, no replay (D2) | 9.2 | 36.2 | 41.2 |
| + replay (D6) | 37.6 | 49.8 | 41.3 |
| + zero-init gate (D7) | 54.4 | 56.5 | 19.4 ✗ |
| + DC + 2 ep (D9/D9b) | 35.6 | 54.2 | 46.34 |
| **+ LLaVA stage 2 (D11)** | **62.0** | **64.5** | **47.66** |
| frozen Gemma-2-9b-it, English | 74.8 | 69.6 | — |

Mechanism (hypothesis): stage 3 trains both mappings jointly into one frozen
LLM. A poorly aligned vision mapping puts a large gradient on the shared
input space, and the text mapping drifts off its stage-1 solution to
compensate — which is exactly the D2 interference, only sourced upstream.
Better visual alignment removes the pressure, so the text path survives.
This **moves the D7 frontier instead of trading along it**: D7 bought
MGSM 54.4 by closing the vision channel (xGQA 19.4); D11 beats that MGSM
*and* posts the best xGQA measured. Reasoning retention in joint training is
therefore partly a function of the *other* modality's alignment quality —
a claim worth its own ablation in the paper.

**Generation-health verification** (`analysis/text_gen_health.py`): an
accuracy jump this large could come from fewer broken generations rather
than better reasoning. It does not — the opposite holds. `dcl` degenerates
*more* (MGSM repetition loops 1.2% → 4.4%, unparseable 0% → 1.2%), and the
gap **widens** once degenerate rows are dropped:

| run | acc | loops | no-extract | acc, clean subset |
|---|---|---|---|---|
| MGSM dc_e2 | 35.6 | 1.2% | 0.0% | 36.2 |
| MGSM **dcl** | 62.0 | 4.4% | 1.2% | **65.3** (+29.1) |
| MSVAMP dc_e2 | 54.2 | 1.0% | 1.4% | 55.3 |
| MSVAMP **dcl** | 64.5 | 3.1% | 3.9% | **68.4** (+13.1) |

Loops score 9% on MGSM against 65% for everything else, so they are almost
pure loss — a repetition penalty at decoding is ~2-3 points of untapped
headroom, deliberately left alone to preserve the D8 protocol parity
(future work / separately-marked row).

Category attribution vs `stage3_bn_dc_e2` (paired, n = 12,578):

| category | n | Δ full | ΔV: dc_e2 → dcl | McNemar p |
|---|---|---|---|---|
| material | 309 | **+7.77** | +11.0 → **+19.4** | 1.3e-3 |
| color | 758 | **+6.46** | +18.7 → +21.8 | 6.7e-5 |
| spatial | 713 | +2.66 | +3.1 → +2.1 | 0.25 |
| object/other | 6273 | +0.81 | +21.2 → +22.3 | 0.092 |
| yes/no | 4525 | +0.51 | +8.9 → +10.6 | 0.48 |

Fine attributes (material, color) take the whole gain — more captions teach
more attribute vocabulary. Spatial's *full* accuracy rises but its ΔV falls:
the gain there is prior, not sight. Both open categories from D8/D9b stand
unmoved by data volume, which sharpens the queue: **spatial needs
architecture (C-Abstractor), yes/no needs task supervision (existence-QA)**;
neither is a data-quantity problem.

---

## Improvement queue (evidence-ranked, 2026-08-26)

1. ~~DenseConnector~~ → D9 ACCEPTED (+2.97 xGQA full, all of it ΔV).
2. ~~Stage-3 epochs ≥2~~ → D9b ACCEPTED (+2.11 xGQA, +4.8 MSVAMP).
3. ~~Stage-2 scale-up to LLaVA-Pretrain-558k~~ → D11 ACCEPTED (+1.32 xGQA,
   **+26.4 MGSM**, +10.3 MSVAMP). Follow-up: raise `--sample` 100k → 300k,
   the derivative has not flattened.
4. Honeybee C-Abstractor (2D-aware abstraction) — targets spatial (ΔV −0.6).
5. M3IT-style instruction diversity in stages 2/3, incl. existence-QA
   synthesized from captions — targets yes/no (negative evidence).
6. LwF-KL do-no-harm loss — reasoning-preservation novelty candidate,
   complementary to D6.
7. Gate ramp-up schedule — revisit D7's frontier.
