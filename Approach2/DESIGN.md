# Approach 2 — Design Log

Running record of every design decision, why it was made, and the measured
effect of each change. One entry per decision, chronological. Numbers cite the
summary JSONs in `Approach2/results/`; "ΔV" = full-image accuracy minus blind
(gray-image) accuracy. **Renamed Δ_gray on 2026-09-07**: it measures sensitivity
to a natural image versus a gray canvas, not instance-specific grounding, which
is Δ_ground = correct − shuffled (S1 v2). Every "ΔV" below is Δ_gray.

**Architecture in one line**: frozen NLLB-200-600M (text) + frozen
SigLIP2-so400m-384 (vision) → two trainable MLP mappings →
frozen Gemma-2-9b-it. Prefix: `[BOS] + X_f + b_txt + V_f + b_vis + T`.
Curriculum: stage 1 text-only, stage 2 vision-only, stage 3 joint VQA
warm-started from both.

Trainable parameter count depends on `--vis-layers`: `mapping_txt` is 9.4M
throughout, `mapping_vis` is 10.9M last-layer-only and **48.7M** under the
DenseConnector DCI setting `"9,18,-1"` (input 1152x3 = 3456). So **58.1M
total for v3/v4**, 20.4M before D9. The header previously said "~40M", which
matched neither configuration.

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
`stage3_<lang>_v3`, results harvested as `*_v3`.

Amended 2026-08-28 after D11: `launch_v3.sh` now takes `ROUND` and
`S2_CKPT` (defaulting to the LLaVA-scaled `stage2_dc_llava` when present),
and `seed_round_from_pilot.sh` transplants a finished bn pilot into a
round's bn slot (~5-6 h of GPU saved). `evaluate_all.sh` runs the text
benchmarks for every language MGSM/MSVAMP covers among ours — **bn, de, ru,
zh** — with the right NLLB tag, so D11's reasoning claim stops resting on
one language and n=250.

Benchmark provenance: `fetch_text_benchmarks.py` (stdlib only, so it runs on
a login node or a laptop) pulls MGSM from `juletxara/mgsm` and MSVAMP from
`Mathoctopus/MSVAMP` — the same two releases Approach 1 uses — with the same
field mapping as her `Stage1/tools/read_datasets.py` (MSVAMP `m_query` /
`response`; MGSM question / numeric answer), and `--from-mindmerger` converts
her local copies directly when they are reachable. Validated against the
Bengali files behind every number above: MGSM identical, MSVAMP identical in
all 1000 golds and in all 1000 questions after whitespace normalization (76
rows carry a leading space in the July file). The bn evals keep using the
original `MGSM.jsonl` / `MSVAMP.jsonl` so continuity is exact.

**Result** (jobs 19753766 + 19853876). The levers generalize: all 7 xGQA
languages gain 3.5-5.5 points with the blind score flat, so the round-level
gain is entirely visual.

| | round B | v3 | Δ |
|---|---|---|---|
| xGQA full (7 langs) | 43.34 | **47.73** | +4.39 |
| xGQA blind | 32.37 | 32.29 | −0.08 |
| xGQA ΔV | +10.96 | **+15.43** | +4.47 |
| CVQA full (10 langs) | 42.56 | **44.02** | +1.46 |
| CVQA blind | 33.89 | 31.44 | −2.45 |
| CVQA ΔV | +8.67 | **+12.58** | +3.91 |

Pooled paired McNemar on xGQA, n = 88,046 shared items: b/c = 8155/12023,
z = 27.2, p = 3.5e-163. Against the D8 references, xGQA is now 47.73 vs
Qwen3-VL zero-shot 53.0 (gap 9.7 → 5.3) and Maryam's M2RB 55.59, while CVQA
44.02 beats both (Qwen 40.75, M2RB 38.81) on a *lower* blind prior.

**Replication / noise floor.** `stage3_bn_v3` retrains the D9b recipe from
scratch, so comparing it against the `dc_e2` pilot measures run-to-run
variance directly: xGQA −0.42, xGQA blind −0.16, MGSM −0.40, MSVAMP +0.30 —
but **CVQA −2.80**. xGQA and the text benchmarks are stable to ~0.4 points,
which every accepted lever clears by 3-5x; CVQA at n=286 has a noise floor
of ±2.8 points, so no CVQA delta below ~3 points means anything, D11's −2.45
included. Every CVQA claim in this document should be read against that bar.

**Reasoning is not uniformly broken — Bengali is the outlier.** First
measurement outside bn:

| lang | MGSM | MSVAMP |
|---|---|---|
| bn | 35.2 | 54.5 |
| de | 62.4 | 77.3 |
| ru | **75.6** | 78.1 |
| zh | 69.2 | 78.5 |

de/ru/zh land 27-40 points above Bengali on MGSM, and their MSVAMP scores
(77-78) exceed the *Bengali* frozen-LLM ceiling of 69.6 outright.

Read against the project's actual target — low-resource languages — this is
the levers landing on the intended population, not a weakened claim. MGSM
and MSVAMP intersect our 11 languages in bn, de, ru, zh: one low-resource
language and three high-resource ones. The collapse appears in the former
and not the latter, and D6/D9b/D11 recover it there while leaving the
high-resource languages untouched, which is what a targeted fix looks like.
The honest limit is n=1: Bengali is our only low-resource observation on
these benchmarks, so "low-resource" rather than "Bengali" is a hypothesis
the reasoning data cannot yet separate. The four remaining LRLs (jv, mn, si,
ga) have no reasoning benchmark at all.
**Per-language ceilings (measured 2026-08-29, `launch_ceilings.sh`).** Frozen
Gemma-2-9b-it answering each language's own questions with no mapping:

| lang | MGSM ceiling | v3 | % of ceiling | MSVAMP ceiling | v3 | % of ceiling |
|---|---|---|---|---|---|---|
| **bn** | 74.8 | 35.2 | **47.1%** | 69.6 | 54.5 | **78.3%** |
| de | 70.8 | 62.4 | 88.1% | 80.7 | 77.3 | 95.8% |
| ru | 76.8 | 75.6 | **98.4%** | 77.9 | 78.1 | **100.3%** |
| zh | 72.0 | 69.2 | 96.1% | 80.9 | 78.5 | 97.0% |

The bridge already works for de/ru/zh — Russian is at ceiling on both
benchmarks. The system's entire deficit is Bengali's.

**This is the project's thesis, measured against the right denominator.**
Each language against its own ceiling, not against English and not against a
mean. D11's arm on Bengali:

| Bengali | deficit vs its own ceiling |
|---|---|
| v3 (control stage 2, 11.5k pairs) | MGSM −39.6, MSVAMP −15.1 |
| dcl (LLaVA stage 2, ~111k pairs) | MGSM −12.8, MSVAMP −5.1 |

Visual grounding removes **68% of the MGSM deficit and 66% of the MSVAMP
deficit**. Two benchmarks, independent item sets, the same fraction.

It also kills the obvious confound. Gemma's Bengali MGSM ceiling (74.8) is
the *highest* of the four, above German (70.8) and Chinese (72.0); all four
sit in a 70-77 band, so the spread is noise at n=250, but that is the point —
the frozen LLM reads Bengali grade-school maths perfectly well. The 40-point
deficit was never the language's nor the LLM's, it was the bridge's, and
scaling English caption data recovered two thirds of it.

**Same-task limitation, and what answers it.** The replay trains on GSM8K
train and MGSM *is* GSM8K test, so MGSM alone cannot support a
generalization claim (Santiago, 2026-09-02). Three findings settle what to
do about it:

- MetaMathQA does not fix it. Its `type` field shows 240k/395k rows
  (**60.8%**) are `GSM_*` — rephrasings of GSM8K train. It dilutes the
  overlap, it does not remove it.
- The `MATH_*` subset (155k, 39.2%) would remove it entirely — MATH train
  shares no source with either evaluation — but it is unusable here: 7 of 8
  sampled `MATH_AnsAug` queries carry LaTeX (`$\dbinom{14}{11}$`,
  `$10101_3$`), and this pipeline must NLLB-translate the question. The
  translation would be noise.
- **MSVAMP already is the out-of-source evaluation.** It derives from SVAMP,
  which shares no origin with GSM8K, and the D11 law holds there
  independently (slope 0.73 against 0.71 on MGSM, disjoint items). Present it
  as the out-of-distribution arm, not as a second math benchmark.

Decision: move the replay to MetaMathQA's `GSM_*` rows for **volume and
phrasing diversity** — 30k per language, MindMerger's scale and the same
source Approach 1 uses (`Stage3/load_text.py:162`) — and justify it that way,
never as a generalization argument. **Stated limitation**: every reasoning
benchmark here is a math word problem. XNLI and X-CSQA would not help, since
neither covers jv/mn/si/ga.

**No contamination.** The D6 replay is built from GSM8K's *train* split
(`build_math_replay.py`: `load_dataset("openai/gsm8k", "main", split="train")`,
7,473 problems); MGSM's 250 items come from GSM8K's *test* split. Disjoint.
MSVAMP derives from SVAMP and shares no origin with GSM8K at all — which is
why the two benchmarks agreeing on the recovery slope (0.71 vs 0.73) matters:
one shares a source with the replay data and the other cannot.

**Prediction for v4, recorded before it lands**: de/ru/zh should move very
little (≤3 points — they have 1-8 points of headroom and Russian has none),
while Bengali holds the pilot's 62.0/64.5. If de/ru/zh improve substantially
under the LLaVA arm, the effect is not low-resource-specific and this section
needs rewriting.

**Where the LRLs can be read: CVQA.** It is the only benchmark covering all
five (bn, jv, mn, si, ga). Splitting the round B → v3 delta by resource
level:

| group | CVQA full B → v3 | ΔV B → v3 |
|---|---|---|
| low-resource (bn jv mn si ga) | 38.66 → 40.37 (+1.71) | +7.97 → +10.51 (**+2.54**) |
| higher-resource (ru zh pt id ko) | 46.46 → 47.66 (+1.20) | +9.37 → +14.64 (**+5.27**) |

The vision-extraction gap between the groups **widens**, −1.40 → −4.13: the
accepted levers buy roughly twice as much ΔV for the languages that were
already ahead. At n=286 per language (noise floor ±2.8, so ±1.3 on a
5-language mean) the between-group difference is about 1σ — a signal, not a
result. But it is the signal that matters most for an LRL-focused paper, and
averaging cannot settle it. It needs a pooled paired test over the
low-resource group (n ≈ 1430), which is why the harvest now keeps per-item
CVQA predictions.

If it holds, it is an argument for a lever aimed at LRLs specifically rather
than more of the same: every accepted lever so far (multi-layer features,
more epochs, more English captions) improves the *vision* side, which is
language-independent by construction, so its benefit can only reach a
language through a text mapping that is already good enough to carry it.

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

Mechanism: **the first hypothesis was tested and refuted.** It held that a
poorly aligned vision mapping puts large gradients on the shared input space,
so the text mapping drifts off its stage-1 solution to compensate, and better
visual alignment removes that pressure. `analysis/mapping_drift.py` states
the prediction (the dcl arm should drift less) and measures it:

| arm | ‖ΔW‖ / ‖W₁‖ on `mapping_txt` | `end_boundary` |
|---|---|---|
| dc_e2 | 6.17% | 16.19% |
| dcl | **6.44%** | **23.86%** |

dcl drifts marginally *more*, not less, so the prediction fails and the
drift-magnitude story is dead.

`stage3_bn_v3` retrains the dc_e2 recipe, which supplies the missing noise
floor for free, and it separates the two quantities cleanly:

| pair | cosine of update direction | drift |
|---|---|---|
| dc_e2 vs **v3** (same recipe) | **+0.928** | 6.17 / 6.30 |
| dc_e2 vs dcl (different stage 2) | **+0.515** | 6.17 / 6.44 |
| v3 vs dcl (different stage 2) | **+0.514** | 6.30 / 6.44 |

Magnitude carries no signal: same-recipe runs differ by 0.13 points against
0.27 between arms, the same order. Direction does. Retraining reproduces the
update direction at 0.93, while changing the stage-2 checkpoint drops it to
0.51 — against both runs, 0.5147 and 0.5140, far outside the run-to-run
spread. The falsification condition was named before the measurement ("if
same-recipe runs also sit near 0.51, this is noise") and did not trigger.

**Revised claim, and it is stronger than the one it replaces**: the stage-2
checkpoint *determines where joint training takes the text mapping*. English
caption data, which never touches the text path, reroutes it to a
substantially different solution — far more than retraining does. The two
mappings are not independent; they are coupled through the frozen LLM's
shared input space, which is the coupling D11's effect requires in order to
exist at all. The parameter separating the arms most is `end_boundary`, the
embedding marking the seam between injected prefix and native text. The gate
sits at ~1.009 in all three (identity, no differential).

Still open, and post-hoc, is *which* destination is better and why. Weight
geometry cannot answer that — two 2-layer MLPs can sit 6% apart and compute
nearly the same map — so the instrument is functional:
`analysis/mapping_function_drift.py` encodes benchmark questions with NLLB,
pushes them through each arm's mapping and compares the produced prefixes.

**D11's result stands regardless**; what does not stand is the explanation
for it.
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

The frozen-LLM ceiling measured the same way (`text_eval_bn_v2_v2`, Gemma
answering in English with no mapping): MGSM 74.8 raw / **75.0 clean**,
MSVAMP 69.6 raw / **74.0 clean** — the extractor costs the ceiling 5.9% of
MSVAMP rows, more than it costs us. Comparing like with like, D11 recovers
**87% of the MGSM ceiling and 92% of MSVAMP** (raw-vs-raw gives 83%/93%, so
the claim is robust to how degenerate rows are handled). Not parity, but the
remaining gap is 9.7 points on MGSM where it was 65.6 before replay.

One qualitative difference worth a sentence in the paper: the frozen LLM
writes a median 499 characters of chain-of-thought per MGSM item, the mapped
pipeline 227. It reasons correctly in half the tokens — or stops too early,
which the repetition/length axis above could disambiguate.

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

### D12 — Joint multilingual stage-1 mapping (2026-08-29) — PENDING

**Why this and not another vision lever.** D10 measured two gaps, and the
accepted levers move both the wrong way:

| gap | round B | v3 |
|---|---|---|
| resource, CVQA full (LRL − HRL) | −7.80 | −7.29 |
| resource, CVQA ΔV (LRL − HRL) | −1.40 | **−4.13** |
| cultural (CVQA − xGQA, 6 shared langs) | +2.25 | **−1.18** |

Every accepted lever so far — multi-layer features, more stage-3 epochs,
more English captions — improves the *vision* side, which is
language-independent by construction. Its benefit can only reach a language
through a text mapping good enough to carry it, which predicts exactly what
we see: high-resource languages convert the better visual channel into
accuracy and low-resource ones do not. The cultural gap has a second, blunter
cause: CC3M, WIT and LLaVA-Pretrain are Western English web imagery, so
scaling stage 2 optimizes the distribution xGQA is drawn from.

Judged by MindMerger's own standard this is a failure so far. Its thesis is
that a frozen LLM already reasons and what non-English inputs lack is
*understanding*, supplied by a multilingual encoder — so success is gap
closure, not a higher mean.

**The divergence, and who shares it.** MindMerger's mapping stage trains ONE
mapping over nine languages at once. In the upstream repo,
`MindMerger/run_training.py:35` selects
`['Bengali','Thai','Swahili','Japanese','Chinese','German','French','Russian','Spanish']`
for the math task and hands them to `read_lego`
(`mindmerger_tools/read_datasets.py:36`), which accumulates every language
into a single `dataset_train`. Approach 2 trains eleven separate mappings
(`train_stage1_all.sh` passes `--languages "$name"`, one per run).

Approach 1 does the same as us: its stage-1 job invokes
`--nllb_languages Bengali` (`Stage1/job-scripts/train.sh:89`), one language
per run. So the two approaches are configured alike — the cross-approach
comparison is apples to apples — but **both deviate from the baseline they
cite**, and neither has tested the joint setting. Sharing parameters across
languages is precisely the mechanism by which a low-resource language borrows
structure from a high-resource one. A flat result closes the question; not
running it leaves a reviewer's objection open. If it wins, the fix is one
argument and it applies to Approach 1 too.

**Design.** `launch_joint.sh`: one stage-1 mapping over all 11 languages
(`train_stage1_joint.sh`), then the standard per-language stage 3 warm-started
from it, then evals. Single variable vs v4 — the origin of the text mapping;
stage 2, recipe, replay and epochs are unchanged (`train_stage3_all.sh` gained
`STAGE1_CKPT` to pin a shared mapping, `evaluate_all.sh` gained `BENCHES` to
order the benchmarks). `TRAIN_NUM=30000` per language x 11 ≈ 330k samples,
about one per-language run's budget (100k x 3 epochs), so the joint mapping
is handicapped on per-language data: winning under that handicap is the
stronger result.

**Metric**: `analysis/gap_report.py v4 vj` — the LRL−HRL gap and the
CVQA−xGQA cultural gap. A round that lifts the mean while widening either
gap does not count as progress here. **Result**: _pending._

---

## Positioning (literature sweep, 2026-09-01)

11 searches over arXiv and the ACL Anthology, 10 abstracts read at source.
Not a substitute for a related-work pass on a draft, but enough to fix the
claim.

**Occupied — do not claim these.**

| work | setup | why it does not cover us |
|---|---|---|
| MERLIN (arXiv 2509.08105, EACL'26) | NLLB-600M + Gemma-2-9b, text only, DoRA in the decoder | no vision; LLM not frozen. Reports MGSM 76.2 / MSVAMP 79.2 with **our exact components** |
| LLINK, "Languages are Modalities" (2510.27254) | frozen decoder, contrastive projector + soft slots | text only, one bridge. **The phrase is taken** |
| MindMerger, LangBridge, SOLAR (2606.26466) | text encoder → frozen LLM | one bridge |
| mBLIP (ACL'24 ALVR) | frozen multilingual LLM + vision bridge | requires MT'd multimodal data in 95 languages; no text-only eval; multilinguality lives in the LLM, not in a bridge |
| X-Fusion (ICCV'25) | frozen LLM + vision, dual tower, "preserving language capabilities" | monolingual, no multilingual bridge |
| VFA (2608.26155) | task-vector merging | fine-tunes; measures the text→vision direction |
| AlignVLM (2502.01341) | Align connector: convex combination of LLM vocab embeddings | **trains the full LLM and reports no text-only benchmark** |
| Cai et al. (2505.19616) | owns the term *modality interference* | spurious signals at inference, fixed by fine-tuning |
| TowerVision (2510.21849) | multilingual VLM design study | fine-tunes; no text-only reasoning eval |
| Puranegedara et al. (2508.09091) | fuses all intermediate layers of the text encoder into the LLM | kills the "text-side DenseConnector" idea outright |

**What survives.** *Superseded 2026-09-07*: all four claims once listed
here were retracted or narrowed (see "Corrections to the record", items
5–7): "no instance of two bridges into one frozen LLM" → X-LLM exists;
"degradation with zero weight updates" → connector-mediated interference,
the 58M mapping parameters do move; the D11 recovery law → removed; "no
translated multimodal data" → "no target-language multimodal supervision".
What actually survives is the claim chain in `SCIENCE.md` §2.

**Consequences** *(2026-09-07: the AlignVLM connector is out of scope before
the deadline, and the CVQA − xGQA difference is a cross-benchmark gap, not a
causal cultural measurement).*

- Terminology: cite Cai et al. and LLINK, and find our own words for the
  phenomenon.
- **Required experiment**: the AlignVLM connector under our frozen setting.
  A reviewer will say the interference is an artifact of a plain MLP
  projection. Constraining the prefix to the convex hull of Gemma's
  embeddings either leaves the interference standing (the law is robust) or
  removes it (we found the fix). Note the cost: Gemma's 256k vocabulary makes
  `P_vocab` large per patch, and logits memory already dominates here.
- The cultural gap is promoted. The multilingual-VLM survey (2509.22123, 33
  models / 23 benchmarks) names "language neutrality vs cultural awareness"
  as the field's central tension, and TowerVision works that line. Our
  measurement — scaling Western web captions widens the CVQA−xGQA gap, with a
  blind control — lands directly in it.

### The English arm of xGQA (2026-09-05)

Every "drop of X points from English" claim needs an English arm on the SAME
items, and `$DT/Stage3/data/xgqa/` had none. The HF mirror `floschne/xgqa`
carries an `en` split but only 9,666 rows against our 12,578, so a comparison
against it would fold an item-set difference into the language difference.

Our per-language rows keep GQA's question `id`, so the join is exact:
`fetch_gqa_english.py` pulls GQA test-dev questions keyed by `question_id`
from `theblackcat102/gqa-testdev-balanced` (dev 10,062 + test 2,516 = 12,578,
the exact size of our files), and `build_xgqa_english.py` emits the English
file. Verified: **matched 12,578/12,578 (100%), 0 missing, 0 gold
disagreements with GQA**. The gold is taken from our file so every language is
scored against identical labels.

Rows carry `nllb_lang_tag: eng_Latn`, which `evaluate_vqa.row_nllb_tag` reads
before falling back to `source_language` (`en` is not in `ISO_TO_NLLB`).

**Caveat to state when comparing against Pfeiffer et al.'s 38 points**: this
arm is `stage3_bn_v4` reading English, not a system *trained* in English. It
is the right denominator for a drop measured inside our system; it is not
their baseline, which was English-trained.

### E2 — Zero-shot cross-lingual multimodal transfer (2026-09-05) — WORKS, AND FAILS INFORMATIVELY

`stage3_bn_v4`, trained on Bengali VQA only, evaluated on nine other CVQA
languages. None was ever seen with an image. Threshold declared before the
run: retain ≥50% of the supervised arm's ΔV.

| | supervised ΔV | zero-shot ΔV | retention |
|---|---|---|---|
| average over 9 | +10.45 | **+6.96** | **67%** |

Per-language retention runs 12% to 218%, which at n=286 is mostly noise — the
Russian 218% reflects an anomalously low *supervised* ΔV (+5.50), not a
zero-shot system beating its teacher. The pooled paired test is the readable
statistic:

| group | n | blind-only | image-only | p |
|---|---|---|---|---|
| all 9 | 2657 | 182 | **353** | **2.0e-13** |
| ru zh pt id si | 1432 | 81 | 223 | 6.1e-16 |
| **jv mn ga** | 935 | 81 | 94 | **0.36** |
| ko | 290 | 20 | 36 | 0.045 |

**The pixels genuinely contribute** in a language the model never saw with an
image — a positive answer to the question xGQA left open (Pfeiffer et al.
report a 38-point drop and "latent multilingual multimodal misalignment").

**And it fails completely for jv, mn and ga**, where ΔV is +0.9 to +1.9 and
the paired test is flat. None of the obvious explanations survives:

- Not typology: Sinhala is Indo-Aryan like Bengali and retains 86%, but
  Russian — Slavic — retains more.
- Not script: Mongolian is Cyrillic like Russian; one fails, the other is the
  best transfer in the set.
- Not the frozen LLM's competence in the language: Sinhala's MGSM ceiling is
  37.6 and it transfers well; Javanese's is 47.2, higher, and it transfers at
  13%.
- Not the blind prior: jv/mn/ga sit at 34.3-34.6 blind, the same band as
  si (33.3), pt (32.8) and ru (34.0).

What remains untested is **the quality of each language's text mapping**,
which this project has never measured. `analysis/` has no instrument for it;
the sibling ALIGNFREEZE work (`scripts/2027_eacl/sufficiency_scoring.py`)
scores exactly that, label-free, as retrieval@1 over held-out parallel
sentences — and MERLIN uses Retrieval@5 for its layer-wise analysis, so the
measure is already established in this lineage.

**Next**: retrieval@1 per language, correlated against transfer retention. If
it predicts, the paper gains a mechanism *and* a cheap way to say in advance
which languages will transfer — without any multimodal data or evaluation in
the target language.

### E3 — Zero-shot multimodal transfer on xGQA (2026-09-06) — TRANSFER IS ESSENTIALLY FREE

`stage3_bn_v4`, trained on Bengali VQA only, evaluated on seven xGQA languages
it never saw with an image. n=12,578 per language, and the item sets are
parallel — the same GQA questions translated — so image content is held
constant and language is the only variable. (E2 could not do this: CVQA varies
language *and* culturally-specific images at n=286.)

| lang | full | blind | ΔV | ΔV retention | McNemar p |
|---|---|---|---|---|---|
| **bn*** | 47.66 | 30.83 | **+16.83** | (reference) | 2.0e-263 |
| en | 50.86 | 33.40 | +17.46 | 104% | 3.3e-269 |
| de | 49.24 | 31.64 | +17.59 | 105% | 3.9e-300 |
| ru | 48.39 | 31.11 | +17.28 | 103% | 2.0e-265 |
| zh | 48.88 | 32.11 | +16.77 | 100% | 4.9e-254 |
| pt | 48.65 | 31.89 | +16.76 | 100% | 1.6e-258 |
| id | 48.00 | 31.36 | +16.65 | 99% | 9.1e-269 |
| ko | 47.33 | 32.01 | +15.32 | 91% | 1.0e-220 |

\* in-language and supervised; every other row is zero-shot from Bengali.

Mean ΔV over the six unseen languages is **+16.73 against the supervised
+16.83 — 99.4% retention**, and ΔV spans only 15.3–17.6 across all seven.
A paired bootstrap over the 12,578 shared question ids (2,000 resamples) puts
six of seven **statistically indistinguishable from the in-language supervised
model**; only Korean is lower, by 1.51 points [−2.46, −0.56], and it still
retains 91%.

**Absolute accuracy goes up, not down**: source 47.66 → 48.42 mean over the six
unseen, **+0.75 points**. Pfeiffer et al. report roughly −38 for zero-shot
transfer on this benchmark.

**Do not report that as a 38-point win.** The protocols differ in ways that
favour us and must be stated: they transfer from English and fine-tune a
multimodal transformer; we transfer from Bengali into a frozen LLM that
already speaks all seven target languages, with NLLB doing the language work.

**The mechanism is architectural, and it is checkable in the code rather than
inferred from the numbers.** `model.py:325` — `mapping_vis` consumes only
`encoder_vis(pixel_values)`. No language input, no cross-attention with the
question; the visual prefix is computed independently of the question text.
So the visual pathway *cannot* be language-specific, and cross-lingual visual
transfer costs nothing by construction.

That is the honest claim, and it is a better one than "we transfer better":
**we do not transfer better, we decoupled the two axes so that there is
almost nothing left to transfer.** The residual variation across languages is
therefore attributable to the text bridge alone — which is what E2 and E4
independently point at.

**What the trainable parameters actually saw.** Worth stating precisely,
because it makes the result stronger than "never seen with an image".
`stage3_bn_v4`'s two mappings (58.1M params) were trained on, in full:
stage 1 Bengali→English text (`train_stage1.sh:25`, `LANGUAGES=Bengali`;
`outputs/stage1/` is the Bengali-only mapping), stage 2 English LLaVA-Pretrain
captions, and stage 3 Bengali VQA + Bengali maths replay + the
`Bengali_to_English` translation replay (`pilot_dense.sh:46-47`). **de, ru, zh,
pt, id and ko appear in no training signal at any stage, in any modality.**
The only components that have ever seen them are frozen NLLB and frozen Gemma.

**English is not a clean zero-shot cell** and should be reported separately or
dropped: it is the target side of the stage-1 translation data and the whole
of stage 2. Its 104% is unsurprising. The six-language mean of 99.4% excludes
it.

**Scope limit.** All six clean languages are mid-to-high resource and all are
well served by NLLB. jv/mn/ga — the E2 failures — are not in xGQA at all. E3
says nothing about them, and the 99.4% must never be quoted as a
low-resource result.

---

### E4 — D11 tested out of sample on four LRLs (2026-09-06) — THE LAW DOES NOT HOLD

D11's recovery law was fitted on bn/de/ru/zh: refitting through the origin on
those 8 cells gives gain = **0.677 × deficit**, r = **+0.989**. jv/si/ga/mn
were never used to fit it and now have both v3 and v4 arms.

| lang | bench | ceiling | v3 | v4 | deficit | gain | predicted | residual |
|---|---|---|---|---|---|---|---|---|
| jv | MGSM | 47.2 | 42.4 | 40.4 | 4.8 | −2.0 | +3.2 | −5.2 |
| jv | MSVAMP | 53.6 | 52.4 | 45.9 | 1.2 | **−6.5** | +0.8 | −7.3 |
| si | MGSM | 37.6 | 34.8 | 32.4 | 2.8 | −2.4 | +1.9 | −4.3 |
| si | MSVAMP | 35.0 | 38.7 | 39.7 | −3.7 | +1.0 | −2.5 | +3.5 |
| ga | MGSM | 43.2 | 33.6 | 33.2 | **9.6** | −0.4 | +6.5 | −6.9 |
| ga | MSVAMP | 61.4 | 56.9 | 57.3 | 4.5 | +0.4 | +3.0 | −2.6 |
| mn | MGSM | 17.2 | 16.4 | 14.4 | 0.8 | −2.0 | +0.5 | −2.5 |
| mn | MSVAMP | 27.0 | 25.7 | 26.0 | 1.3 | +0.3 | +0.9 | −0.6 |

Mean predicted gain **+1.8**, mean observed **−1.4**; out-of-sample RMSE
**4.65** against a ±0.4 noise floor. Pooled paired McNemar over the four LRLs
has v4 **significantly worse** than v3 (v4-only 323, v3-only 388, **p=0.016**),
driven by Javanese MSVAMP (p=1.8e-06). The same pooled test on the fitted
four is null (p=0.22), as it should be — they are at ceiling.

The sharpest single miss is **Irish MGSM: a genuine 9.6-point deficit, 6.5
predicted, 0 delivered.** Sinhala cannot test the law at all — its deficits
are 2.8 and −3.7, i.e. it is already at its ceiling.

**D11 must be restated.** It is not a law about deficits. It holds where the
text bridge works and fails where it does not, and scaling *visual* pretraining
buys nothing for a language whose *text* pathway is broken. The r=+0.989 was
fitted on four languages that all have working text bridges.

**The convergence is the finding.** The three languages where D11 fails —
**jv, ga, mn** — are exactly the three where E2's visual transfer was flat
(p=0.36). Two experiments, different benchmarks, different modalities,
different metrics, same three languages. Sinhala, the fourth LRL, is the one
that transfers well in E2 (86%) and is also the one with no deficit to test.

---

### H1 — Is the vision branch a tax on the text bridge? (2026-09-06) — NO

The control the launcher header asks for: identical stage 3 — same VQA data,
replay, epochs, lr — with `--no-vision`. Bengali:

| arm | visual alignment | MGSM | MSVAMP |
|---|---|---|---|
| stage 1 only | — | 11.6 | 34.1 |
| stage 3, `--no-vision` | none | 23.6 | 46.0 |
| stage 3 v3 | weak (11.5k pairs) | 35.2 | 54.5 |
| stage 3 v4 | strong (~111k pairs) | **62.0** | **64.5** |
| frozen-LLM ceiling | — | 74.8 | 69.6 |

**Monotone in visual alignment quality on both benchmarks.** "Vision costs,
and better vision costs less" is refuted in its strong form: removing vision
does not free the text bridge, it costs 38.4 MGSM points against the matched
v4 arm and 11.6 against the weaker v3 arm.

**The caveat that keeps this honest.** `--no-vision` does not merely remove an
input, it makes the VQA task *unanswerable* — ~34k examples per epoch asking
what colour a shirt is with no shirt. Some of the 38.4 is therefore "training
on an impossible task damages generation", not "pixels help maths". The
`--no-vision` arm is a **lower anchor, not a clean third dose**. The
confound-free dose-response is v3 → v4: both have real images, both are
answerable, only alignment quality differs — and that is D11, which for
Bengali is +26.8 MGSM.

**The confound was tested and does not hold** (2026-09-06, per-item files
recovered). `analysis/text_gen_health.py`, `--no-vision` against v3:

| | median chars | empty | no-extract | loops | clean-subset acc |
|---|---|---|---|---|---|
| MGSM `--no-vision` | 307 | 6.0% | 9.2% | 0.8% | **26.2%** (n=225) |
| MGSM v3 | 242 | 0.8% | 1.6% | 3.6% | **37.1%** (n=237) |
| MSVAMP `--no-vision` | 234 | 2.3% | 5.3% | 0.5% | **48.8%** (n=942) |
| MSVAMP v3 | 136 | 0.1% | 1.2% | 2.3% | **56.2%** (n=964) |

The prediction was that the unanswerable VQA task would push the model toward
short ungrounded answers. It did the opposite — outputs are **longer** (307 vs
242, 234 vs 136), not shorter, so the "collapsed to VQA-style short answers"
mechanism is refuted. Degeneration is genuinely higher (6.0% vs 0.8% empty)
but does not account for the gap: **on the clean subset the deficit survives
at 10.9 points on MGSM and 7.4 on MSVAMP.** H1's conclusion stands on cleaned
data, and the `--no-vision` arm can be reported as a real ablation rather than
a broken run.

**Closed against the matched v4 arm** (2026-09-06, `stage3_bn_dcl` per-item
files recovered). Paired McNemar on the same items, vision vs `--no-vision`:

| bench | n | vision-only right | no-vision-only right | Δ | p |
|---|---|---|---|---|---|
| MGSM | 250 | **102** | 6 | +38.4 | 1.3e-23 |
| MSVAMP | 1000 | **238** | 53 | +18.5 | 3.7e-29 |

The discordance is 17:1 on MGSM and 4.5:1 on MSVAMP — this is not a shifted
distribution, it is one arm solving problems the other cannot touch.

And on the clean subset, with degenerate generations dropped from **both**
arms, **the gap widens**: MGSM 26.2 → 65.3 (**+39.1**, raw +38.4) and MSVAMP
48.8 → 68.4 (**+19.6**, raw +18.5). Every route by which this could have been
an artefact is now closed: not short-answer collapse (outputs are longer), not
degeneration (the gap grows once it is removed), not unpaired comparison
(McNemar on identical items). **The vision branch is not a tax on the text
bridge; it is what makes the text bridge work.**

---

### Sufficiency is not alignment (2026-09-06) — REFUTES THE LOOSE MECHANISM CLAIM

E2 and E4 both ended with "the text bridge is what remains untested". Before
spending compute on that, the cheapest version of the claim was checked
against data already on disk: **is the text bridge simply worse for the
languages that fail?** Percentage of each language's own frozen-LLM ceiling
reached by the v4 mapping (raw extraction, no cleaning):

| lang | MGSM v4 | ceiling | % | MSVAMP v4 | ceiling | % | CVQA transfer retention |
|---|---|---|---|---|---|---|---|
| bn | 62.0 | 74.8 | 83% | 64.5 | 69.6 | 93% | (source) |
| de | 70.0 | 70.8 | 99% | 78.4 | 80.7 | 97% | (not in CVQA) |
| ru | 75.2 | 76.8 | 98% | 76.5 | 77.9 | 98% | 218% |
| zh | 68.0 | 72.0 | 94% | 80.4 | 80.9 | 99% | 48% |
| si | 32.4 | 37.6 | 86% | 39.7 | 35.0 | 113% | 86% |
| **jv** | 40.4 | 47.2 | **86%** | 45.9 | 53.6 | **86%** | **13%** |
| **mn** | 14.4 | 17.2 | **84%** | 26.0 | 27.0 | **96%** | **25%** |
| **ga** | 33.2 | 43.2 | **77%** | 57.3 | 61.4 | **93%** | **12%** |

**The text bridge is sufficient in every language, including all three that
fail.** Javanese reaches 86% of its ceiling on both benchmarks and transfers
at 13%; Chinese reaches 94–99% and transfers at 48%; Sinhala reaches 86%/113%
and transfers at 86%. There is no relationship.

So "jv/mn/ga fail because their text bridge is worse" is **false as stated**.
Whatever is broken is not decodability of the prefix by the frozen LLM.

**The distinction this forces.** Two different properties were being conflated:

- **Sufficiency** — can Gemma reason from the mapped prefix? Measured by % of
  ceiling. High everywhere (77–99%).
- **Alignment** — does language L's prefix land in the *same region* of prefix
  space as the source language's, so that a visual prefix tuned during
  Bengali stage 3 still composes with it? **Never measured.**

Transfer needs alignment; monolingual reasoning needs only sufficiency. That
is why a language can be fluent and untransferable at once, and it is the
reason the instrument has to be contrastive (retrieval@1, matched vs
mismatched margin) rather than a task score.

**Discriminating prediction for the retrieval@1 run**: the alignment score
must correlate with transfer retention and **not** with % of ceiling. If it
correlates with both, it is measuring general bridge quality, the mechanism
claim is unsupported, and the finding degrades to a description.

**Caveat on the dependent variable.** CVQA retention is noisy — n=200–412 per
language, ±2.8 on ΔV, and it disagrees with xGQA for zh (48% vs 100%), ko
(50% vs 91%) and id (67% vs 99%). Only the pooled jv/mn/ga group result
(n=935, p=0.36 flat, against n=1432, p=6.1e-16 for the strong group) is
solid enough to build on. Per-language CVQA retention must not be used as a
regression target.

---

### X1-X3 — the mechanism slate (2026-09-06) — PREDICTIONS RECORDED BEFORE LAUNCH

Everything below is written before any of the three runs. Each states what
would refute it, so a null result is reportable rather than embarrassing.

**X1 — source or target?** `job-scripts/source_ablation.sh`, evaluation only,
no training: all eleven `stage3_<L>_v4` checkpoints already exist. CVQA
transfer into jv/mn/ga (plus si as a positive control) from bn, id, ru and
zh. SRC=bn is free — those summaries exist and are skipped.

- *H_source* — jv/mn/ga fail because **Bengali** is a bad source for them.
  Predicts id→jv works where bn→jv does not (both Austronesian; Indonesian
  retains 99% on xGQA, so its own bridge is known healthy).
- *H_align* — they fail because **their own** text bridge is misaligned.
  Predicts flat from every source alike.
- **Refutation of the harness rather than the hypothesis**: if Sinhala also
  goes flat under a new source, something is wrong with the run, because it
  retains 86% from Bengali (n=225, p=3.6e-05). Check that before reading
  anything else.

Read with `analysis/source_ablation_report.py` — pooled paired McNemar over
the failing group, never per-language dV, for the reason in the caveat above.

**X2 — the instrument.** `alignment_score.py` + `job-scripts/alignment_score.sh`,
3h partition, no generation and no labels: NLLB encoder forwards plus a Gemma
embedding lookup over 1,000 held-out parallel sentences per language. Two
references — `bridge` (prefix(L) vs prefix(en), is the mapping
language-consistent?) and `llm` (prefix(L) vs Gemma's own embedding, does it
land where the LLM already represents that meaning?).

**Pre-registered discriminating prediction**: the score must correlate with
transfer retention and **must not** correlate with % of frozen-LLM ceiling.
Correlating with both means it is measuring general bridge quality, the
mechanism claim is unsupported, and the result degrades to description.
`analysis/alignment_vs_transfer.py` prints that verdict automatically —
Spearman with an exact permutation p, because n=9-11 languages.

Second refutation condition, structural: **the primary statistic is `margin`,
not retrieval@1.** The prefix space is 3584-dimensional; R@1 over N=1000 may
saturate at 1.000 for every language, which is the metric bottoming out, not
a finding.

X2 also scores `stage1_joint` when it exists — so it prices D12 **before**
eleven stage-3 runs. If the joint mapping does not raise jv/mn/ga's margin,
X3 will not fix their transfer either.

**X3 — the intervention.** Already implemented and chained:
`launch_joint.sh` (D12). One shared multilingual stage-1 mapping instead of
eleven per-language ones, everything else identical to v4.

**Pre-registered differential prediction**: it must lift **jv/mn/ga** and
leave **de/ru/zh flat** — those are at 97-99% of ceiling and 99-105% ΔV
retention, with nowhere to go. A uniform lift across all languages refutes
the alignment mechanism just as surely as no lift does, because it would mean
D12 improved something general rather than the specific thing X2 measures.
Judge with `analysis/gap_report.py v4 vj`, not with the mean.

**Launch order matters.** X1 first (no training, decides which hypothesis is
live), X2 second (cheap, and prices X3), X3 last (the expensive one).

---

### X1 — Source or target? (2026-09-06) — H_source WINS, AND E2's CONCLUSION WAS WRONG

CVQA transfer into jv/mn/ga from four sources, si as the positive control.
Evaluation only; every checkpoint already existed.

| source | jv | mn | ga | pooled ΔV (n=935) | retention | pooled p |
|---|---|---|---|---|---|---|
| bn | +1.35 | +1.92 | +0.92 | +1.39 | 17% | 0.36 |
| **id** | **+5.05** | **+4.81** | **+4.29** | **+4.71** | **56%** | **0.00085** |
| ru | +0.67 | +1.60 | +1.23 | +1.18 | 14% | 0.40 |
| zh | +4.38 | 0.00 | +6.44 | +3.64 | 44% | 0.011 |
| supervised | +10.10 | +7.69 | +7.36 | +8.34 | 100% | — |

Control si transfers from every source (86%, 100%, 97%, 89%; p from 3.6e-05
to 1.2e-06), so the harness is sound.

**E2's conclusion is refuted, and it was ours.** E2 recorded that jv/mn/ga
"fail completely" and hunted for a property of those languages — typology,
script, LLM competence, prior — that would explain it. All of that was
looking in the wrong place. **Transferability is a property of the
source–target pair, not of the target.** Indonesian extracts significant
visual signal in all three (pooled p=0.00085) where Bengali extracts none.
Refuting a target-intrinsic explanation needs exactly one source that works,
and Indonesian is it.

**What is NOT established.** The direct paired comparison between the two
sources on the same 935 items is **p=0.051** — id-better 110, bn-better 82.
Borderline. "id is significant and bn is not" is not the same claim as
"id > bn", and the difference-of-significance fallacy is the obvious way to
oversell this. What carries the weight instead is consistency: id retains
50%/62%/58% across three unrelated targets (Austronesian, Mongolic, Celtic),
each at p≈0.06-0.07 alone and 0.00085 pooled.

Chinese is not a general donor — it is the *best* source for Irish (+6.44,
87% retention, p=0.0055) and worthless for Mongolian (0.00, p=1.0). Russian
is flat everywhere despite being the best *target* in E2 (218% retention).
**Being a good target and being a good donor are different properties.**

**Bengali was a poor choice of source language, and that choice silently
shaped every transfer result in this project.**

---

### X2 — Alignment scoring (2026-09-06) — PREDICTION FAILED, AND THE DESIGN WAS CONFOUNDED

Scored `stage3_bn_dcl`'s text mapping on 1,000 held-out parallel sentences
per language.

| | outcome |
|---|---|
| bridge retrieval@1 | 0.779 (mn) to 0.996 (ru); did **not** saturate |
| vs transfer retention (must correlate) | rho=+0.50, permutation **p=0.18** |
| vs % of ceiling (must not correlate) | rho=+0.19, p=0.66 |

The pre-registered verdict is **FAILED**: alignment does not track transfer.
Javanese is the direct counterexample — retrieval@1 **0.960**, near the top of
the set, and 13% retention from Bengali.

**Two instrument problems, recorded so the re-run does not repeat them.**

1. `margin` was declared the primary statistic and is unusable. The prefix
   space is severely anisotropic: *mismatched* sentences sit at cosine
   **0.987**, so every margin is crushed into 0.007-0.010 and measures the
   residual scale, not alignment. The docstring anticipated saturation at the
   top; the failure was at the bottom. Retrieval@1 is the usable statistic
   here and is now the analysis default. A re-run should mean-center before
   the cosine.
2. The `llm` reference carries **no signal whatsoever** — retrieval@1 = 0.001
   at n=1000 (chance), median rank ~485, for all eleven languages. In
   hindsight this is expected: the mapping is trained so the LLM can *read*
   the prefix through attention, not so it lands on the embedding table's
   geometry. Drop it.

**The deeper problem is that X1 invalidated X2's design.** X2 correlated a
**target-intrinsic** measure (how well language L's bridge retrieves its own
translations) against a **pair-dependent** outcome (retention *from Bengali*).
X1 showed the outcome is a property of the pair. The experiment could not have
worked, and its null says nothing about alignment as a mechanism.

**The fix is a pair-level measure**: alignment between the *source* and
*target* prefixes, not target-to-English — the visual mapping was tuned while
the source's prefixes were in play, so that is the distance that should
matter. This needs a multi-way parallel corpus, which the stage-1 files are
not (each is a different L→English corpus, so L and S share no sentences).
**FLORES-200** is the right one: 1,012 dev sentences, all eleven languages,
and it is NLLB's own evaluation set. xGQA is multi-way parallel too but does
not cover jv/mn/ga/si, which are the languages in question.

**Prediction for the pair-level re-run**: source-target alignment should rank
id above bn and ru as a donor for jv/mn/ga, and should rank zh high for ga
and low for mn — the one clean dissociation X1 produced. If it cannot
reproduce that dissociation, the alignment mechanism should be abandoned
rather than re-instrumented a third time.

---

### Correction to the E2/E4 "convergence" claim (2026-09-06)

DESIGN.md previously recorded that E2 and E4 "converge" on jv/mn/ga and called
that the finding. **That claim is now half retracted.** E2's failure has a
source explanation (X1: Bengali is a poor donor; Indonesian is not). E4's
failure does not — it used each language's **own** supervised checkpoint, so
no source is involved, and why the v4 visual scale-up buys those languages
nothing is still open. Two different things go wrong for the same three
languages, and the most likely reason they are the same three is simply that
they are the lowest-resource in the set. They are not one mechanism.

---

### The question — corrected back to its anchor (2026-09-06)

Santiago caught a drift and he was right. The previous entry here promoted
donor selection to "the question". It is not the question; it is a finding
inside it. The anchor has been xGQA since the positioning sweep, and the
anchor is a literal call for methods. Verified at source (arXiv 2109.06082,
Pfeiffer, Geigle, Kamath, Steitz, Roth, Vulić, Gurevych), final sentence of
the abstract:

> "Our results suggest that simple cross-lingual transfer of multimodal
> models yields latent multilingual multimodal misalignment, **calling for
> more sophisticated methods for vision and multilingual language
> modeling.**"

**The question is that call.** Can zero-shot cross-lingual transfer for VQA
avoid the ~38-point collapse, and what makes the misalignment "latent"?

**The answer, and it is E3.** Decoupled bridges into a frozen LLM:
**+0.75 points instead of −38** on their benchmark, six unseen languages,
ΔV retained at 99.4% of the in-language supervised arm, six of seven
statistically indistinguishable from it, McNemar p < 1e-220 everywhere.

**E3 was wrongly demoted.** It was recorded above as "a control, not a
contribution" because the result follows from `mapping_vis` taking only
pixels. That reasoning was backwards. xGQA's whole finding is that this
transfer is *hard*; an architecture in which the difficulty does not arise is
exactly what a call for more sophisticated methods asks for. The
language-blind visual pathway is not a reason to discount the result — it is
**the mechanism**, and it answers their diagnosis directly: the misalignment
is "latent" because language and modality share one pathway, and it does not
occur when they do not.

The blind control is what stops this from being a degenerate win: the pixels
are demonstrably doing the work in every one of the seven languages, so the
system is not transferring a language prior that happens to score well.

**Where the answer stops, and this is X1's place in the paper.** All seven
xGQA languages are mid-to-high resource and well served by NLLB. Extend past
that set to jv/mn/ga/si — which xGQA does not cover — and transfer becomes
**donor-dependent**: 17% retention from Bengali, 56% from Indonesian, and
Bengali was chosen by default. That is the honest limit of the answer and a
new failure mode inside the same problem, not a different question.

So the paper's spine:

1. xGQA calls for more sophisticated methods. (their words)
2. Decoupling the bridges answers it: −38 becomes +0.75. (E3)
3. The mechanism is architectural and checkable in code, and it explains why
   their misalignment was latent. (`model.py:325`)
4. Beyond their language set the answer becomes conditional on donor choice.
   (X1)
5. Open: can donor quality be predicted label-free, before building
   anything? (X2b)

**Donor quality, measured** (`analysis/donor_matrix.py`, mean retention over
ga/jv/mn/si — the four targets every source was run on):

| donor | quality |
|---|---|
| **id** | **68%** |
| zh | 55% |
| ru | 35% |
| bn | 34% |

Donor quality must be averaged over a **common** target set. Averaging each
source over whatever it happened to be run on ranks Bengali first at 71%,
purely because Bengali was the only source run on the easy targets (pt, ru,
si, ko, zh) — which inverts X1's finding. The script enforces the common set
and prints which targets it used.

---

### X2b — pairwise alignment on FLORES-200 (2026-09-06) — IMPLEMENTED, NOT YET RUN

`pair_alignment.py` + `job-scripts/pair_alignment.sh`, with
`fetch_flores.py` to build the data.

X2's design was invalidated by X1: it correlated a target-only measure against
a pair-dependent outcome. X2b measures the pair — how close target T's prefix
is to source S's prefix on the *same* sentence — which requires a multi-way
parallel corpus. The stage-1 files cannot serve: each is a different
L→English corpus, so two languages share no sentence.

**FLORES-200 dev**, 997 sentences × 12 languages, verified line-aligned
(row 0 is the same Stanford-scientists sentence in all twelve). The HF
mirrors `facebook/flores` and `openlanguagedata/flores_plus` both answer
**401** without a token, and tokens do not belong in repo scripts, so the
fetcher pulls the canonical no-auth NLLB tarball
(`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, ~25 MB) and reads
members with `extractfile()` rather than `extractall()`.

Two fixes carried over from X2's failure: prefixes are **mean-centered per
language** before the cosine (mismatched sentences sat at 0.987 without it,
crushing every margin into a 0.007–0.010 band), and the `llm` reference is
dropped entirely — it scored exactly chance (retrieval@1 = 0.001 at n=1000)
for all eleven languages, because the mapping is trained so the LLM can
*read* the prefix through attention, not so it lands on the embedding
table's geometry.

**Pre-registered**: the pair score must rank id above bn and ru as a donor
for jv/mn/ga, and must reproduce X1's one clean dissociation — zh high for ga
(87% retention) and low for mn (0%). A correlation that misses that
dissociation is not the mechanism, and the alignment hypothesis should then be
abandoned rather than re-instrumented a third time.

It also scores `stage1_joint`, which prices D12 before eleven stage-3 runs:
if the shared mapping does not pull the languages' prefixes together, it will
not fix transfer either.

---

### X2b — first results (2026-09-06) — MY FIRST ANALYSIS WAS WRONG; VERDICT DEFERRED

Four checkpoints scored on FLORES-200 dev (997 sentences x 12 languages).

**The analysis error.** The first pass read every source's row out of
`pairalign_stage3_bn_dcl.json`. That is the wrong bridge: transferring from
source S runs **S's** mapping, so S's prefix space is the one the target has
to land in. Only the bn rows were right. `analysis/donor_matrix.py` now reads
`pairalign_stage3_<S>_v4.json` per source and marks any row that had to fall
back.

With the wrong file the global correlation looked significant (rho=+0.587,
p=0.0059) but the **within-target** correlation — does alignment pick the
*source* for a fixed target, which is the only question that matters — was
**−0.20**, i.e. the whole effect was target difficulty: si has both the
highest mean alignment (0.941) and the highest mean retention (92.9%), while
jv/mn/ga cluster at 0.89-0.90 and 27-44%.

**Corrected, for the two sources that have their own checkpoint:**

| pair | alignment (own ckpt) | (bn ckpt) | retention |
|---|---|---|---|
| bn→jv | 0.894 | 0.894 | 13.3 |
| bn→mn | 0.921 | 0.921 | 25.0 |
| bn→ga | 0.895 | 0.895 | 12.5 |
| bn→si | 0.959 | 0.959 | 85.7 |
| **id→jv** | **0.987** | 0.937 | **50.0** |
| **id→mn** | **0.979** | 0.901 | **62.5** |
| **id→ga** | **0.977** | 0.891 | **58.3** |
| **id→si** | **0.993** | 0.932 | **100.0** |

id beats bn on alignment **and** on retention in all four targets, 4/4.
Within-target Spearman rises from −0.20 to **+0.35** once id uses its own
file.

**The pre-registered verdict cannot be pronounced yet.** Both conditions
name ru and zh, and neither has been scored — `pairalign_stage3_ru_v4.json`
and `pairalign_stage3_zh_v4.json` do not exist. The earlier "FAILED" was
computed from the wrong bridge and is withdrawn. Score those two, then read
the verdict.

**A donor-level hypothesis is what the data actually suggests**, and it is
stronger than the pair-level one because it needs no target-side data at all:

| checkpoint | mean pairwise R@1 | donor quality (X1) |
|---|---|---|
| `stage1` (bn-only, pre-VQA) | 0.968 | — |
| **`stage1_joint`** (D12) | **0.986** | untested |
| `stage3_id_v4` | 0.983 | **68%** |
| `stage3_bn_dcl` | 0.918 | **34%** |

**Stage 3 in Bengali degrades cross-lingual alignment (0.968 → 0.918);
stage 3 in Indonesian does not (0.968 → 0.983).** Per-language, bn's stage 3
costs jv 0.957→0.892, ga 0.953→0.901, mn 0.955→0.906. That is a concrete
mechanism for X1's donor effect: training VQA in S pulls the shared text
mapping toward S, and how much collateral damage that does is a property of
S. n=2 donors, so this is an observation, not a test.

**Prediction, recorded now**: mean pairwise alignment of `stage3_<S>_v4`
predicts S's donor quality across all eleven sources. Both halves are cheap —
`pair_alignment.sh` with `CKPTS` covering the eleven, and job 20398782 is
already filling the donor matrix. If it holds, the "can you tell before you
build it" half of the question is answered **without any multimodal data or
evaluation in any target language**.

**And it re-motivates D12.** `stage1_joint` has the highest mean alignment of
any checkpoint measured (0.986), lifting exactly the languages bn's stage 3
damages (jv 0.981, ga 0.982, mn 0.977). The X3 gate is void — it assumed
alignment predicts transfer, which is what is still being decided — but X3's
own differential prediction (lift jv/mn/ga, leave de/ru/zh flat) stands on
its own and should simply be run.

---

---

### Corrections to the record (2026-09-07)

Three independent reviews on 2026-09-06/07 (Santiago, Claude, a second
reviewing agent) converged on the items below. Each was verified against
code or per-item results; the number that replaces the old one is given.

1. **E3's reference was Bengali's supervised arm.** The conventional
   reference is each target's own `stage3_<L>_v4`, and those checkpoints
   already existed. Recomputed from per-item files:

| lang | target-supervised full / ΔV | bn→L zero-shot full / ΔV | retention | D = ΔV_zs − ΔV_sup, two-sided 95% CI, image-cluster bootstrap | δ=1.0 | δ=1.75 |
|---|---|---|---|---|---|---|
| de | 51.52 / 18.57 | 49.24 / 17.59 | 94.7% | −0.98 [−1.79, −0.20] | inconclusive | non-inferior |
| ru | 49.68 / 17.47 | 48.39 / 17.28 | 98.9% | −0.19 [−1.07, +0.71] | non-inferior | non-inferior |
| zh | 48.75 / 16.04 | 48.88 / 16.77 | 104.5% | +0.72 [−0.14, +1.59] | non-inferior | non-inferior |
| pt | 51.15 / 18.87 | 48.65 / 16.76 | 88.8% | −2.11 [−3.03, −1.21] | materially inferior | inconclusive |
| id | 50.17 / 17.94 | 48.00 / 16.65 | 92.8% | −1.29 [−2.25, −0.29] | inconclusive | inconclusive |
| ko | 48.66 / 15.98 | 47.33 / 15.32 | 95.9% | −0.66 [−1.60, +0.30] | inconclusive | non-inferior |
| mean | 49.99 / 17.48 | 48.42 / 16.73 | **95.7%** | −0.75 | | |

   Generated by `analysis/e3_noninferiority.py` (B=4000, seed 1; one RNG per
   estimand and stratum with canonical ordering, so reordering or dropping
   targets never moves a primary interval, `analysis/test_invariance.py`;
   regions read on the image-cluster one-sided 5/95 bounds, which the script
   also prints; the script aborts on unequal item sets or unmapped ids).
   Pooled over the six targets with each image resampled jointly across its
   translations: D = −0.75, 95% CI [−1.21, −0.29]. Re-running with other seeds
   moves the second decimal; the regions did not change across seeds 0 and 1.

   Headline becomes: **−1.58 full-accuracy points and 95.7% of the
   target-supervised visual contribution with Bengali-only VQA
   supervision.** "6/7 indistinguishable" and "+0.75 vs source" are
   withdrawn as headlines; source-relative figures may appear as secondary.
   Non-inferiority (one-sided 95% lower bound; post hoc, so sensitivity
   only; three regions, image-cluster bootstrap, `analysis/e3_noninferiority.py`):
   at δ=1.0 ru and zh are non-inferior, pt is materially inferior, de/id/ko are
   inconclusive; at δ=1.75 de/ru/zh/ko are non-inferior and pt/id inconclusive.
   Clustering by image (398 images behind 12,578 questions) changes no verdict. "Statistically indistinguishable" is non-rejection,
   never equivalence.

2. **X1's p=0.051 was McNemar on full accuracy**, not on visual
   contribution. The estimand is the paired difference-in-differences
   (ΔV_id − ΔV_bn) on identical items, **image-cluster bootstrap** (CVQA ids
   are `<image>_<k>`, up to three questions per image; 935 items on 401
   images): pooled jv/mn/ga **+3.32, 95% CI [+0.32, +6.43]** (stratified by target); per language
   jv +3.70 [−2.03, +9.36], mn +2.88 [−2.61, +8.62], ga +3.37 [−1.19, +7.89]
   (`analysis/x1_did.py`, B=4000, seed 0; pooled interval stratified by
   target; one RNG per estimand and stratum, canonical order, invariance
   tested; the script aborts on duplicate ids, unequal item sets, ids
   without the `<image>_<k>` structure, or a pool outside the targets). Wording: "source choice
   matters; evidence for an Indonesian advantage is suggestive".
   Retention ratios divide by a supervised ΔV measured at n=200–412 and are
   descriptive only.

3. **The donor-level alignment table compared different lineages.**
   `stage3_id_v4` was warm-started from `outputs/stage1_id`
   (`train_stage3_all.sh:110`), which was never scored. "Indonesian stage 3
   preserves alignment 0.968 → 0.983" is **withdrawn**. Only Bengali's
   stage 1 → stage 3 (0.968 → 0.918) is longitudinal. Every source's own
   `stage1_<S>` must be scored before any drift claim.

4. **`donor_matrix.py` cannot compute donor quality once jv/mn/ga/si are
   sources**: `source_ablation.sh:105` skips source==target, so the
   common-target set is empty. Fix: donor panel restricted to the seven
   disjoint sources bn/de/ru/zh/pt/id/ko over jv/mn/ga/si, or a model with
   source and target effects. Classification by timestamp: discovery donors bn/id/ru/zh (outcomes
   inspected before the donor-level hypothesis was recorded). de/pt/ko are
   prospective **only if** the cluster audit shows that none of their
   outcomes was committed, produced or read before the freeze; see S1 Block
   D. Until that audit, all seven are treated as discovery data.

5. **"±0.4 noise floor" is one retrain pair, not a run distribution, and
   is wrong for MGSM/MSVAMP**: binomial SE alone is ±3.0 at n=250 and ±1.6
   at n=1000. E4's out-of-sample failure rests on jv MSVAMP (−6.5, ~3σ);
   the other cells are within noise. The D11 "law" was fitted on two real
   points (Bengali) plus six near-zero cells; it is **removed**, not
   restated.

6. **Wording that changes.** "No translated multimodal data" → "no
   target-language multimodal supervision" (stage 3 is NLLB-translated
   GQA). "Frozen-LLM ceiling" → "direct-prompt baseline", and "% of
   ceiling" is dropped as a denominator (si MSVAMP already exceeds it).
   "Language-blind visual pathway" → "question-independent visual
   features": SigLIP2 is itself trained on multilingual image-text data
   (arXiv 2502.14786), and stage 3 tunes `mapping_vis` on source-language
   VQA, so the downstream effect can be source-conditioned. "Label-free,
   before building anything" → "without target VQA labels or target
   images, after donor training". "Pre-registered" → "recorded in the
   shared repository before the result artifacts were committed" (e.g.
   2045930 → 08b8334, 1 h 26 min apart); `sacct -j <id>
   --format=JobID,Submit,Start,End` on the cluster can settle whether a run
   started after its prediction commit. "Cultural gap CVQA − xGQA" → a
   cross-benchmark difference, not interpretable causally.

7. **Claims dropped as not novel.** "Answers xGQA's call": Nooralahzadeh &
   Sennrich 2022, Liu et al. EACL 2023 Findings, Scheduled Unfreezing 2024,
   mBLIP, Centurio (ACL 2025), and our own D8 (Qwen3-VL zero-shot 53.0 on
   all seven languages) already show the −38 collapse is not a property of
   modern systems. "No instance of two bridges into one frozen LLM":
   X-LLM (arXiv 2305.04160). Related work to add: Translation Deserves
   Better (every stage-3 set is NLLB-translated GQA); LangRank, NN-Rank,
   Shah et al. 2024 (source selection); IAA, eP-ALM, Implicit Multimodal
   Alignment. Image-induced Fidelity Loss is already cited in D6.

**Spine, restated.** Not "decoupling solves xGQA". The honest statement
today, to be upgraded from "investigate whether" to "show that" only if
the S1 gate below passes:

> Using separately pretrained language and vision connectors into a frozen
> LLM, a checkpoint tuned for VQA in Bengali retains 95.7% of the
> target-supervised visual contribution across six unseen xGQA languages.
> Transfer varies with the VQA source language. We investigate whether
> this variation arises from source-conditioned connector co-adaptation
> and whether preserving pre-VQA alignment improves transfer.

The coupled-architecture baseline is a stated limitation, not a required
experiment, because the causal claim about xGQA's latent misalignment is
dropped. D11/H1 stay as the ablation that justifies the vision path and as
one mechanism paragraph; the mechanistic reasoning paper waits for a second
low-resource language and a second backbone.

---

### S1 — Specification v2.1.3: input-necessity audit, branch factorial, freeze factorial (2026-09-07) — FROZEN 2026-09-08, NOT RUN

**Revision history.** v1 → v2.1.3 in four review rounds on 2026-09-07
(Santiago, Claude, a second reviewing agent); the intermediate texts are in
git history, not here. What changed at the level of principle: endpoints
split into utility U and grounding Δ_ground; three-region non-inferiority
with substantive margins; per-stratum RNGs and fail-closed analysis scripts
with tests; one primary panel and endpoint per gate; G1 split into inference
dependence (A2) and training necessity (C5); a matched replay-only control
R0/R1 for any preservation loss; G3 as a separate, confirmatory-only
contribution; a compute ledger and a scope decision. No revision changed the
canonical E3 or X1 numbers. **Edit this entry, not the launchers, when the
design changes; then make the launcher match.**

**Freeze rule**: the commit that freezes this entry must exist **before any
new run of S1** (Blocks A–D and the confirmatory panel). Every launcher
**must** refuse to submit if the working tree is dirty or if `HEAD` does not
contain the freeze commit, and must write the spec SHA into the manifest at
submit time; none of this exists yet. "Before harvest" is not enough:
`source_ablation.sh:159` commits harvested results on the cluster by itself.
Job 20398782 predates the freeze and is **legacy / discovery** unless the
audit in Block D shows otherwise.

**Freeze record, 2026-09-08.** This entry is frozen by the commit titled
"docs: freeze S1 v2.1.3 experimental specification"; its hash is the **spec
SHA** every S1 manifest must carry. Verified in the tree at that commit:
the four documents (`SCIENCE.md`, `DESIGN.md`, `README.md`, `CLAUDE.md`),
`analysis/_boot.py`, `e3_noninferiority.py`, `x1_did.py`,
`test_invariance.py`, `test_fail_closed.py`, `power_sim.py` +
`audits/power_sim.json` (shared-image xGQA clusters), `block_d.py` +
`test_block_d.py` (paired bootstrap, four G3 conditions),
`inventory_cvqa.py` + `audits/cvqa_inventory.json` (sha256 `599afbf9…`
matches its `.sha256`), `audits/20398782.md`, `audits/cvqa_s1_prefreeze.json`
(job 20483784) and `audits/review_2026-09-08.md`; all four test suites pass.
The pre-declared "promising" rule for the optional confirmatory donor launch
was accepted by Santiago on 2026-09-08 as written under the Option 1
decision. After this commit nothing in S1 is edited in place: a change is a
new dated amendment below this entry that leaves the frozen text visible,
and any run submitted before the amendment is analysed under the text that
was frozen when it was submitted. No S1 job has been submitted yet; the
implementation prerequisites listed at the end of this entry come first.

#### Endpoints and statistics

- **Utility** U = Acc(correct image). **Grounding** Δ_ground =
  Acc(correct) − mean over permutations of Acc(shuffled), averaged *within
  item* first; permutations are never counted as independent observations.
  **Δ_gray** = Acc(correct) − Acc(gray canvas), secondary, continuity with
  every earlier "ΔV". **Δ_none** = Acc(correct) − Acc(no visual tokens),
  Block A only.
- **Every contrast is reported on both U and Δ_ground.** A change in
  Δ_ground with U falling is not an improvement; a bridge is "dispensable
  at inference" only if non-inferior on both.
- **Permutation rule**: three seeded derangements (0, 1, 2) on **every**
  panel. The unit of randomisation is the image, and xGQA has only 398 of
  them behind 12,578 questions, so question counts do not justify a single
  map anywhere. Shuffles permute *unique image ids* with no fixed
  points; every question of an image receives the same wrong image; the
  map is keyed by question id, so it is identical across xGQA languages,
  checkpoints and arms; on CVQA the derangement is drawn **within the target
  language, and within each country subset of a pooled unit**, never across
  subsets (mixing cultural pools would make Δ_ground a sum of instance
  mismatch and domain shift); each map is stored as
  `evaluation/shuffle_<panel>_seed<k>.json` with `original_image_id`,
  `assigned_image_id` and its sha256, and the hash goes into every
  manifest.
- **Pooling**: micro (item-pooled) for every pooled contrast; per-language
  macro reported as secondary. Bootstrap stratified by target, clustered by
  image where an image carries several questions (xGQA: 398 images), with
  **one cluster resample applied jointly to all arms and translations**
  (paired). 4,000 resamples, fixed seed.
- **Non-inferiority, three regions.** The operational statistic, the one
  `analysis/_boot.region` and every S1 script compute and report, is
  **D = candidate − reference** on the same items, in accuracy points,
  **negative = candidate worse**: non-inferior if the one-sided 5th
  percentile LB5(D) > −δ; materially inferior if the 95th percentile
  UB95(D) < −δ; inconclusive otherwise. The contrasts P1–P8 below are
  written the other way round (reference − candidate) for readability; each
  is the negation of the statistic that is computed, and every reported
  interval, including E3's −0.75 [−1.21, −0.29] and the power tables, is in
  the candidate − reference convention. Failing non-inferiority never means
  inferiority.
- **Margins are substantive, then power is computed; never the reverse.**
  δ_U = δ_G = **1.0 point** on every VQA panel: the largest loss this project
  treats as immaterial, below the smallest lever it ever accepted (D11's
  +1.32 xGQA). The reasoning-retention guard uses δ_R = **1.0 accuracy
  point** separately on MGSM and MSVAMP. Power at those margins: xGQA
  (12,578 items / 398 images,
  cluster half-width ≈ 0.8) can establish non-inferiority under equality;
  on the pooled CVQA jv/mn/ga panel (935 items / 401 images, half-width ≈ 3)
  **power under equality is low**: non-inferiority there needs A2 to be
  genuinely better, not merely equal. The three regions apply everywhere;
  the margin is not widened to make CVQA decidable, and the paper states
  the power. **Power, simulated** (`analysis/power_sim.py` → `audits/power_sim.json`;
  δ = 1.0, B = 400 bootstrap resamples, S = 120 Monte Carlo repetitions, seed
  7; exact NI rule: 5th percentile of the image-cluster bootstrap of D above
  −δ). Per-item paired differences D_i = d + ε_i, where ε_i is the observed,
  centred paired difference between two real arms on the same items (xGQA:
  bn zero-shot vs target-supervised; CVQA: bn-source vs id-source), and a
  new realisation flips the sign of every image's ε at once. Assumed
  quantities, from the real arms:

  | panel | endpoint | items / images | discordance | intra-image corr. of ε |
  |---|---|---|---|---|
  | xGQA bn/de/ko | grounding (Δ_gray proxy) | 37,734 / 398 shared images | 0.24 | 0.002 |
  | xGQA bn/de/ko | utility | 37,734 / 398 shared images | 0.16 | 0.004 |
  | CVQA jv/mn/ga | grounding (Δ_gray proxy) | 935 / 401 | 0.21 | ≈ 0 |
  | CVQA jv/mn/ga | utility | 935 / 401 | 0.14 | ≈ 0 |

  Δ_ground has never been measured, so its noise is proxied by Δ_gray's;
  grounding power is therefore optimistic. Dominant region and its
  probability per true effect d (candidate − reference; d = −2 is a true
  2-point loss of the candidate):

  | panel | endpoint | d = 0 | d = −0.5 | d = −1 | d = −2 | d = −3 | half-width |
  |---|---|---|---|---|---|---|---|
  | xGQA bn/de/ko | grounding | NI .93 | NI .55 | inc .93 | MI .95 | MI 1.00 | 0.58 |
  | xGQA bn/de/ko | utility | NI 1.00 | NI .69 | inc .90 | MI .99 | MI 1.00 | 0.47 |
  | CVQA jv/mn/ga | grounding | inc .79, NI .21 | inc .84 | inc .88 | inc .80, MI .18 | inc .57, MI .43 | 3.0 |
  | CVQA jv/mn/ga | utility | inc .68, NI .32 | inc .87 | inc .88 | inc .77, MI .23 | inc .53, MI .47 | 2.2 |

  Under equality xGQA establishes non-inferiority with probability 0.93–1.00,
  CVQA with 0.21–0.32; a true 2-point loss is detected on xGQA (0.95–0.99)
  and rarely on CVQA (0.18–0.23); a loss exactly at the margin is
  inconclusive everywhere, as it should be. (Recomputed 2026-09-08 with the
  398 shared xGQA image clusters perturbed and resampled jointly across the
  three translations, as the pooling rule above requires; the first artifact
  had treated the 1,194 (language, image) pairs as independent clusters and
  was slightly optimistic: NI under equality 0.96, MI at −2 0.97–0.98. CVQA
  images are disjoint across languages and its rows did not change.) **G3
  attainable power, correlation criterion only**, seven donors, α_conf = −1.0
  × rank(damage) + N(0, σ) points, exact one-sided permutation test at 0.10:
  P(p < 0.10) = 1.00 at σ = 0.5 and 1.0, 0.75 at σ = 2.0, 0.54 at σ = 3.0.
  This says nothing about the conjunction with the regret, pooled-grounding
  and utility guardrails, whose power depends on the unmeasured Δ_ground
  panel; MGSM/MSVAMP (δ_R) power is not simulated, since no paired per-item
  text results with a comparator arm exist yet. For the paper's appendix
  rerun with `--boot 2000 --sims 400`.
- **Seeds, inferential rule.** Replicated arms use the same paired seed
  list. Each contrast is computed per seed on identical items; the primary
  statistic is the mean over seeds, and its CI comes from the cluster
  bootstrap in which every resample recomputes the per-seed contrasts on the
  resampled clusters and averages them. This CI is **conditional on the three
  observed training seeds**; it does not estimate a population distribution
  over all possible training runs. Report the three seed-level estimates,
  their mean, SD and range next to it. Directional superiority claims must
  have the predicted sign in every seed. NI claims instead require the
  conditional pooled bound to pass **and** every seed-level point estimate
  of reference minus candidate to be below δ; a same-sign rule is not used
  for NI or for two-sided P4/P6/P7. Wording is "replicated across these three
  seeds", not an unrestricted recipe-population claim.
- **Hierarchy of claims, separated by family.** *Intervention primary*: G0,
  G1, G2, G4 and G5, combined as the loss-gate conjunction; training-arm
  claims remain conditional on one seed until their three-seed replication.
  *Donor-predictor primary*: G3-confirmatory on the new panel only;
  G3-current is exploratory. *Preservation-method primary*: the frozen R1
  versus matched-R0 contrast and its C2/utility guardrails below. These are
  three separate conclusions, not one omnibus paper-success test.
  *Secondary*: the components of P3–P8 that are not gate conditions, P4,
  P6, P7, the xGQA panels in Block A and si. *Exploratory*: every remaining
  cell. Every table labels its rows with one of these families.
- **Scope of wording**: conclusions are "in the evaluated panels". xGQA vs
  CVQA confounds resource level with benchmark, images and protocol; no
  resource-level conclusion is drawn from that comparison. A resource
  claim needs languages of several tiers *within* one benchmark and an
  external tier criterion (e.g. NLLB training volume or Joshi et al. 2020
  classes), which S1 does not attempt.
- Every number in the paper comes from a versioned script in
  `analysis/`. `analysis/e3_noninferiority.py` and `analysis/x1_did.py`
  exist as of this revision and reproduce the 2026-09-07 corrections.

#### Block A — input-necessity audit (eval-only)

Checkpoint `stage3_bn_dcl`, `--vis-layers "9,18,-1"`. Runs first; its
outcome is gate condition G1. Flags: `--prompt-mode
{question,instruction}`, `--no-text-branch`, `--shuffle-map PATH`,
`--no-image`.

| arm | LLM input | tests |
|---|---|---|
| A1 | X_f + V_f + T(question) | current system |
| A2 | V_f + T(question) | is the NLLB bridge used at inference? |
| A3 | X_f + V_f + T(instruction only) | does the bridge carry the question? |
| A4 | V_f + T(English question) | **original-English direct-prompt reference**, not a translate-test. xGQA: `build_xgqa_english.py` recovers GQA's original English question, shared by bn/de/ko, so A4 runs **once** there. CVQA: the dataset's `Translated Question` field is the English question (verified on the datasets-server rows API, 2026-09-07; `Question` is the native one), so A4 runs per target on CVQA jv/mn/ga/si. The active loader must keep both fields (see Data construction below) |

Panels: xGQA bn, de, ko (source, best, worst); CVQA jv, mn, ga, si.
Conditions, every panel: correct, shuffled 0/1/2, gray, no-image (6).

Count: xGQA 3 arms × 3 languages × 6 + A4 × 6 = **60 evals** at ~40 min
≈ 40 h; CVQA 4 arms × 4 languages × 6 = **96 evals** at ~5 min ≈ 8 h.
**≈ 48 h → four chained 12 h jobs.**

A2 and A3 are **inference ablations of a checkpoint trained with X_f**.
They measure that checkpoint's dependence on its inputs, not whether the
bridge was unnecessary during training. Call the A2 decision **G1-I**
(inference dependence). If G1-I is inconclusive, C5 is mandatory as a
separate **G1-T** test of training necessity; C5 can decide whether training
without the branch is viable, but it can never relabel the A2 inference
result. C5 is skipped if G1-I is dispensable and optional (not gate-relevant)
if G1-I is used.

Reading rule (P1 / G1-I): with D_U = U(A1) − U(A2) and D_G = Δ_ground(A1) −
Δ_ground(A2), per panel (xGQA pooled bn/de/ko; CVQA pooled jv/mn/ga; si
alone), at δ_U = δ_G = 1.0, the bridge is *dispensable at inference for this
checkpoint in that panel* only if A2 is non-inferior on **both** D_U and D_G;
*used* only if A2 is materially inferior on D_G (the primary endpoint);
otherwise inconclusive. **Primary panel for G1: CVQA pooled jv/mn/ga;
primary endpoint: Δ_ground; U is co-primary only for "dispensable".** The
xGQA panel and si are secondary and descriptive. Same rule
for A3 ("the bridge carries the question" only if A3 is non-inferior on
both).

**P2, grounding**: Δ_ground of A1 per target and pooled per panel, with its
95% CI; predicted **> 0** everywhere. On the primary panel P2 is also gate
condition **G0**: if LB95(Δ_ground(A1)) ≤ 0 there, no other gate is
evaluated, Blocks B–C are reported as exploratory, and the paper is the
descriptive version (Δ_gray sensitivity, no grounding claim). A target whose Δ_ground CI includes 0
has no demonstrated instance-specific grounding, whatever its Δ_gray says.
P1 and P2 are the two Block A primary contrasts; P3–P7 follow below, and P8
is conditional on C5.

#### Block B — branch factorial (eval-only)

Text branch × vision branch, each loaded from its own checkpoint.
`end_boundary` and `gate` live inside each `Mapping` and travel with
their branch.

| text \ vision | `stage2_dc_llava` (V2) | `stage3_bn_dcl` (V3_bn) | `stage3_id_v4` (V3_id) |
|---|---|---|---|
| `stage1` (T1_bn) | pre-VQA composition, bn lineage | vision-branch-associated (bn) | vision-branch-associated (id) |
| `stage1_id` (T1_id) | pre-VQA composition, id lineage | cross-lineage | vision-branch-associated (id) |
| `stage3_bn_dcl` (T3_bn) | text-branch-associated (bn) | bn checkpoint | swap |
| `stage3_id_v4` (T3_id) | text-branch-associated (id) | swap | id checkpoint |

Panels and controls:
- Transfer targets: CVQA jv, mn, ga, si. Conditions correct, shuffled
  0/1/2, gray (5). 12 cells × 4 × 5 = 240 evals.
- **CVQA-bn** is a *same-language, out-of-domain* control (CVQA is not GQA);
  12 × 5 = 60 evals.
- **Task positive controls** are xGQA-bn and xGQA-id (stage 3 trains on
  translated GQA): the four corner cells (bn/bn, id/id, both swaps) on
  xGQA-bn and xGQA-id, correct + gray: 4 × 2 × 2 = 16 evals ≈ 11 h.
  bn/bn on xGQA-bn and id/id on xGQA-id already exist and are reused.
- **No-X_f controls** (arm A2 applied) on three cells, bn/bn, id/id and
  T1_bn/V2, on the four transfer targets and bn, correct + shuffled 0/1/2:
  3 × 5 × 4 = 60 evals.

**Total ≈ 360 CVQA evals (~30 h) + 16 xGQA evals (~11 h) ≈ 41 h → four
chained 12 h jobs**, with `eval_matrix.py` loading NLLB, SigLIP2 and Gemma
once and swapping mapping state dicts.

Matrix-runner guarantees (each is a test in `tests/`):
1. loads only the requested branch from each file;
2. resets both `Mapping` modules to a pristine copy before every cell;
3. `stage1` and every `stage1_<L>` predate the gate (commit 4beae75,
   2026-08-24) and carry no `gate` key: they get `gate = 1.0` explicitly;
   any *other* missing or unexpected key is an error, never `strict=False`;
4. before the matrix runs, the bn/bn and id/id cells must reproduce the
   existing per-item predictions of `eval_cvqa_<L>_v4` exactly; a mismatch
   aborts the job.

Labels are **functional, not causal**: Block B says which branch the donor
difference *follows* when trained checkpoints are recombined; only Block C
can say that stage 3 *imprinted* it.

Contrasts, on both U and Δ_ground, pooled jv/mn/ga (micro), CI from the
paired cluster bootstrap. G(·,·) denotes the endpoint of a cell.

- **P3, functional localisation.**
  D_T = [G(T3_id, V2) − G(T1_id, V2)] − [G(T3_bn, V2) − G(T1_bn, V2)]
  (what stage-3 text training added, id lineage minus bn lineage, vision
  held at the shared pre-VQA V2);
  D_V = ½ Σ_{T ∈ {T1_bn, T1_id}} [G(T, V3_id) − G(T, V3_bn)]
  (vision imprint, id minus bn, text held at each lineage's pre-VQA stage 1).
  Primary statistic **D_T − D_V**. "The donor difference follows the text
  branch" iff **LB95(D_T) > 0 and LB95(D_T − D_V) > 0** (one-sided lower
  bounds, as in G2).
  "Text significant and vision not" is never the criterion.
- **P4, co-adaptation.**
  P4 = ½ [G(T3_bn, V3_bn) + G(T3_id, V3_id) − G(T3_bn, V3_id) − G(T3_id, V3_bn)].
  Two-sided; positive means matched pairs compose better than crossed
  pairs. No direction is predicted.

#### Block C — freeze factorial (training, Bengali, single seed = pilot)

Identical `stage1` + `stage2_dc_llava` initialisation, identical data
order (seed 13, order hash in the manifest), `S3_EPOCHS=2`, **no replay in
any arm** (a frozen text branch leaves replay with no trainable path, and
replay must not be confounded with the freeze factor).

| arm | `mapping_txt` | `mapping_vis` | note |
|---|---|---|---|
| C1 | trainable | trainable | |
| C2 | frozen | trainable | text-only pathway bit-identical to `stage1` (freeze covers `end_boundary` and `gate`; text evals carry no V_f): MGSM/MSVAMP = 11.6 / 34.1 and FLORES alignment = 0.968 **by construction** |
| C3 | trainable | frozen | |
| C4 | frozen | frozen | = pre-VQA composition; no training; a Block B cell |
| C5 | absent | trainable | **vision-only trained arm**; mandatory as G1-T iff G1-I is inconclusive, skipped if G1-I is dispensable, optional if G1-I is used. **Must be implemented**, and it is more than a flag: `train_stage3_vqa.py` forces `use_text_branch=True` (`:259`), every batch tokenises and feeds NLLB, and the load / freeze / gate paths assume `mapping_txt` exists. No FLORES alignment: it has no `mapping_txt`. Cost ~10 h per seed; three paired seeds whenever P8 is used to pass the loss gate or enter the paper |

v4 (`stage3_bn_dcl`, trained with replay) is an external reference, not a
cell. **Prediction, not a fact**: C1 without replay is expected to show a
reasoning collapse of the D2 kind; D2 used a different stage 2 and one
epoch, so the size is unknown. Block C speaks to transfer only.

Two implementation corrections **before** any Block C launch, both
recorded as properties of every existing checkpoint:
1. `train_stage3_vqa.py:356` calls `model.train()` after each validation
   pass, which puts the frozen NLLB encoder (dropout 0.1) back into train
   mode; SigLIP2 and Gemma have no dropout. Every existing checkpoint **with a
   text branch** (stage 1, stage 3) was trained with stochastic NLLB
   prefixes; stage-2 vision-only checkpoints never run NLLB. Block C forces the three towers
   to `eval()` after every `model.train()`; v4 is external, so the arms
   stay comparable among themselves.
2. The validation split is `random.shuffle` by row (`:235`), so questions
   of one GQA image land on both sides. Block C splits by `vg_image_id`.

Evaluate every arm on CVQA jv/mn/ga/si/bn (correct, shuffled 0/1/2, gray)
and xGQA-bn (correct, shuffled 0/1/2, gray); FLORES pair-alignment for
C1–C4 only. Cost: three runs (C1–C3) × ~10 h, plus ~10 h for C5 if triggered.

Contrasts, pooled jv/mn/ga, both endpoints, paired cluster bootstrap:
- **P5** = G(C2) − G(C1); predicted **> 0** on Δ_ground if stage-3 text
  drift is causal; U reported alongside, and xGQA-bn utility of C2 must be
  non-inferior to C1 at δ_U = 1.0 (source task retained).
- **P6** = G(C3) − G(C1); two-sided, no prediction.
- **P7** = (C4 − C3) − (C2 − C1); two-sided, no prediction.
- **P8 / G1-T** (conditional on C5): **D8 = G(C1) − G(C5)**, reference minus
  candidate, on both endpoints, so the P1 regions apply with the same sign
  (positive = the vision-only arm is worse). On the primary panel: the text
  branch is dispensable *in training* only if C5 is non-inferior to C1 on
  both U and Δ_ground (UB95(D8) < δ on each); then the text branch is
  dispensable **during training** and no preservation loss is pursued.
  Material inferiority on Δ_ground (LB95(D8_G) > δ_G) means it is needed
  during training and G1-T passes the loss gate. Anything else leaves G1-T
  unresolved and no loss is pursued. None of these outcomes changes G1-I.

**Two baselines are required for any preservation loss.** Matched replay-only
R0 isolates the effect of the loss; C2 is the simplest alignment-preserving
alternative it must at least match on transfer. C1 alone is neither: a
replay-trained loss versus no-replay C1 mixes two interventions. If R1 does
not improve on R0 and match C2, there is no preservation-loss method.

**Single seed means pilot.** Item bootstraps do not contain training
variability; P5–P7 from one trajectory are effects conditional on that
seed. If LB95(P5) > 0, C1 and C2 are replicated with three paired seeds before
any "show that"; C3 too if P6/P7 enter the paper; C5 uses three seeds whenever
P8 decides G1-T; and matched R0/R1 need three seeds of their own. The extra
seeds of `bn_v4` / `id_v4` estimate checkpoint stability only; they do not
replicate D9/D9b/D11 without their comparators.

#### Block D — donor predictor, two separate analyses (eval-only)

`pair_alignment.sh` with `CKPTS` = all eleven `stage1_<L>` (Bengali's is
`stage1`) + all eleven `stage3_<L>_v4` (Bengali's is `stage3_bn_dcl`) +
`stage1_joint`. Instrument note: `pair_alignment.py` excludes
`end_boundary` and the cosine cancels the scalar `gate`, so it measures the
directional geometry of the text MLP only; Block B swaps whole branches.
Centered R@1 **and** centered margin are both reported (R@1 is near
saturation for id).

- **D-pair, exploratory**: primary transfer endpoint Δ_ground(s,t), with U
  reported alongside, against centered-margin alignment(s,t); source and
  target fixed effects give the within-target reading and permutations are
  blocked by target. Centered R@1 is a secondary predictor. This analysis is
  not G3 and cannot rescue it.
- **D-donor, frozen.** *Predictor*: **centered margin** is primary (centered
  R@1 is secondary). *Damage* of donor S = mean over T ∈ {jv, mn, ga, si}
  of [margin_stage1_S(S→T) − margin_stage3_S(S→T)], read from S's **own**
  stage-1 and stage-3 checkpoints, positive = deterioration. This produces
  one fixed, committed ranking before any confirmatory outcome is evaluated.
  For any target panel P, define donor effect α_S(P) by the additive model
  Δ_ground(s,t) = μ + α_s + β_t, least squares with sum-to-zero constraints.
  Each (s,t) cell is the item-micro Δ_ground estimate (subset-stratified for
  pooled Spanish); target-language units receive equal weight in the model.
  P_current has four targets and a 7 × 4 matrix; its Spearman is exploratory.
  P_confirm has the twelve frozen language units and a 7 × 12 matrix;
  α_conf = α(P_confirm) is the only confirmatory outcome. A source fixed
  effect and a per-source constant are collinear, so this is separate from
  D-pair and is the declared macro-over-target exception to general micro
  pooling.

  *G3 test*: Spearman(damage, α_conf) over the seven donors, using standard
  midranks for ties and exact permutation of the seven α labels (5,040),
  one-sided (more damage → lower α), p < 0.10. The operational selector is
  the parameter-free rule "least damage = best donor"; a damage tie is
  broken by NLLB code only for selection, not for Spearman. The selected
  donor is fixed before evaluation. In each paired cluster-bootstrap
  replicate, resample images jointly over all seven donors within each
  target/subset, refit α_conf, and compute max_s α_s − α_selected. G3 requires
  the 95th percentile of that max-regret distribution ≤ 1.0 point. It also
  requires LB95 of the selected donor's pooled Δ_ground > 0 and a utility
  guardrail UB95(max_s α_s^U − α_selected^U) ≤ δ_U, calculated analogously.
  Inference is conditional on these twelve fixed languages and seven trained
  checkpoints; it does not generalise to an arbitrary language population.
  `donor_matrix.py` implements none of this (it uses Δ_gray and retention,
  auto-discovers every `zs*` file and permutes two-sided);
  `analysis/block_d.py` and synthetic tests **must be in the freeze commit**.
- **Δ_ground does not exist for any donor yet**: `source_ablation.sh:108`
  produced correct + gray only. Block D needs the shuffled condition for
  the seven donors × four targets × three seeds = **84 CVQA evals (~7 h)**,
  plus correct + gray for de/pt/ko if job 20398782 did not deliver them.
  Budget added to the roadmap.
- **Audit before classifying prospectivity**: `sacct -j 20398782
  --format=JobID,Submit,Start,End,State`, its log, the cluster clone's
  `git log` **and reflog**, and the mtimes of any partial
  `outputs/zeroshot_*` files (the job can write predictions and expire
  before its own commit), because `source_ablation.sh:159` commits
  harvested results by itself; "not harvested" cannot be asserted from the
  laptop. Until the audit says otherwise the job is legacy / discovery.
  **Audit outcome, 2026-09-07** (`Approach2/audits/20398782.md`): the job
  completed all 96 files (correct + gray only), auto-committed on the cluster
  at 01:30 −04:00 and was pushed as `1aafc1a` before any freeze commit.
  **de/pt/ko are not prospective**; the seven-donor analysis on the current
  targets is exploratory in its entirety, and G3 rests on the confirmatory
  panel alone. **Audit complete (cluster appendix collected 2026-09-07)**:
  submitted 03:43 UTC, before the hypothesis commit; auto-committed 05:30
  UTC; rebased and pushed 06:22 UTC. Then, by timestamp: discovery donors = bn, id, ru, zh (transfer
  inspected before the donor-level hypothesis was recorded on 2026-09-06);
  prospective donors = de, pt, ko **only if** their transfer results were
  neither produced, committed nor read before the freeze commit; otherwise
  none of the seven is prospective and the confirmatory targets carry G3
  alone.
  jv/mn/ga/si are excluded as donors (self-cells are skipped,
  `source_ablation.sh:105`).
- **Confirmatory targets**: rule = every CVQA `Subset` whose language has
  an NLLB-200 tag, is not one of the eleven, has ≥ 150 valid items and at
  least four unique images (enough for three distinct derangements). The
  inventory aborts on duplicate question ids; each of the three maps must
  have no fixed points and a distinct hash within every subset. **The
  freeze commit must contain the verified list.** `Stage3/load_vqa_eval_data.py`
  cannot produce it (it only configures the project's own codes and rejects
  others, `:219`); `Approach2/inventory_cvqa.py` (exists as of 2026-09-07;
  runs from any machine with network through the datasets-server rows API,
  validating every tag against the NLLB tokenizer's own token list) emits `Subset → language → NLLB tag → valid items / images → target unit`.
  Target unit = one per language; country subsets of one language form
  **one** unit with subset as a bootstrap stratum. **Cap, deterministic**:
  the 12 languages with the most valid items, ties broken by NLLB code.
  **Inventory run 2026-09-07** (`Approach2/inventory_cvqa.py` through the
  datasets-server rows API, tags validated against the NLLB tokenizer;
  `audits/cvqa_inventory.json`, sha256 `599afbf39481bae4…`): 10,374 rows, 39
  subsets, 31 languages, 20 eligible (Breton has no NLLB-200 tag; the
  eleven are excluded), panel of 12 with **5,202 items**:

  | unit | NLLB | items | subsets |
  |---|---|---|---|
| Spanish | `spa_Latn` | 2058 | 7 |
| Urdu | `urd_Arab` | 436 | 2 |
| Bulgarian | `bul_Cyrl` | 371 | 1 |
| Malay | `zsm_Latn` | 315 | 1 |
| Romanian | `ron_Latn` | 302 | 1 |
| Norwegian | `nob_Latn` | 299 | 1 |
| Swahili | `swh_Latn` | 273 | 1 |
| Minangkabau | `min_Latn` | 251 | 1 |
| Kinyarwanda | `kin_Latn` | 235 | 1 |
| Amharic | `amh_Ethi` | 234 | 1 |
| Oromo | `gaz_Latn` | 214 | 1 |
| Tamil | `tam_Taml` | 214 | 1 |

  Over the cap, eligible: Filipino, Japanese, Marathi, Hindi, Sundanese,
  Telugu, Igbo, Egyptian Arabic (200–203 items each). Every subset has ≥ 87
  images. **Images**: only 20–85% of rows per subset carry an `Image Source`
  URL; the rest are embedded in the 4.9 GB parquet and must be extracted from
  a local copy before evaluation, which the timed pilot has to include.
  *Budget*: the pre-freeze
  pilot (job 20483784, 2026-09-08) measured 0.337 s/item + 19.5 s load per
  invocation, so one (donor, condition) pass over the 5,202-item panel costs
  0.55 GPU-h and donor confirmation is 7 donors × 5 conditions = **19.3
  GPU-h** (the earlier 5-min-per-eval placeholders gave 35 h and 51 h; both
  are superseded). Spanish alone is 40% of the panel. Under Option 1 this
  runs only if Block D's exploratory test is promising (defined under the
  Option 1 decision below) and budget remains at A+12. The confirmatory panel is the **only**
  prospective test of the donor predictor (see G3): the seven-donor Spearman
  on the current targets is exploratory because four donors' outcomes shaped
  the hypothesis, and a three-donor ranking has chance 1/6. Candidate subsets
  from the dataset card, read through a summarizer and therefore **to be
  verified at source, not trusted**: Amharic, Egyptian Arabic, Bulgarian,
  Filipino, Hindi, Igbo, Kinyarwanda, Malay, Minangkabau, Norwegian, Oromo,
  Romanian, Spanish (six country subsets), Sundanese, Swahili, Tamil,
  Telugu, Urdu. Donor predictions for the verified list are written and
  committed before any of them is evaluated. All ten local CVQA languages
  already have results and cannot serve.

#### Loss gate, operationalised at A+12 days

This date is an **operational allocation gate**. G4/G5 initially come from
the single-seed Block-C pilot and do not become paper-level training-recipe
claims unless their three-seed criteria later pass.

The **loss gate** is G0, G1, G2, G4 and G5, all of them. **G3 is not part of
it**: the donor predictor is a separate contribution, decided only on the
confirmatory panel, and its failure does not invalidate an intervention that
helps. "G1 holds" means the branch is shown to be used at inference (G1-I)
or, after an inconclusive G1-I, needed during training (G1-T); dispensable or
unresolved never passes. Proceed to the preservation-method stage only if
the five hold:

| # | condition | statistic | threshold |
|---|---|---|---|
| G0 | grounding exists where the mechanism is tested | Block A, P2, primary panel | LB95(Δ_ground(A1)) > 0 on CVQA pooled jv/mn/ga. If it fails, do not launch Blocks B–C, do not evaluate G1/G2/G4/G5, and use the descriptive paper. Block D may continue as its separate contribution, but G3 has its own confirmatory grounding requirement |
| G1 | branch is used in the path that justifies preservation | G1-I = Block A/P1; G1-T = P8 only if G1-I is inconclusive | G1-I *used* (A2 materially inferior on Δ_ground) → **pass**. G1-I *dispensable* (A2 NI on Δ_ground and U) → stop, skip C5/loss. G1-I *inconclusive* → retain that label and run three-seed C5; G1-T *needed* (C5 materially inferior on Δ_ground) → **pass for the training intervention only**; G1-T *dispensable* (C5 NI on both endpoints) or inconclusive → stop. Neither P8 outcome rewrites G1-I |
| G2 | the donor difference follows the text branch | Block B, D_T − D_V on Δ_ground | **LB95(D_T) > 0 and LB95(D_T − D_V) > 0** (one-sided lower bounds; "excludes 0" would also pass an all-negative interval) |
| G3 | *(separate contribution, not in the loss gate)* damage predicts donor grounding on the fixed confirmatory panel | Block D-donor **on the confirmatory panel only** | apply the exact frozen D-donor procedure above: one-sided exact Spearman p < 0.10, paired max-regret UB95 ≤ 1.0, selected-donor pooled Δ_ground LB95 > 0, and utility-regret UB95 ≤ δ_U. Failure is a null for this contribution only; the current-target analysis remains exploratory |
| G4 | freezing text helps transfer **and** utility is retained | Block C, P5, primary panel | LB95(Δ_ground(C2) − Δ_ground(C1)) > 0 **and true non-inferiority on utility, UB95(U(C1) − U(C2)) < δ_U**. Under equality the utility half has low power on CVQA and the gate may fail for that reason alone; then the paper says "no utility loss above 1 point was detected" and does not claim retention. "Not materially inferior" is not accepted here |
| G5 | source task retained | Block C | xGQA-bn U(C2) non-inferior to U(C1) at δ_U = 1.0 |

Passing authorises **only** a matched method-stage experiment; it does not
authorise comparing a replay-trained loss directly with no-replay C1/C2.
Train three paired seeds of **R0** (same corrected C1 recipe plus reasoning
replay, no preservation loss) and **R1** (identical to R0 plus the frozen
preservation loss). Initialisation, data order, replay examples, optimizer,
steps and validation split are identical within each seed. Existing v4 is
external and descriptive because it used the old dropout/split behavior.

The preservation-method family succeeds only if all of the following hold:

1. the loss itself improves grounding over its matched replay control:
   LB95(Δ_ground(R1) − Δ_ground(R0)) > 0 on the primary panel and again on
   the confirmatory panel, with U(R1) non-inferior to U(R0) on both;
2. R1 is non-inferior to the simple freeze C2 on **both** Δ_ground and U on
   both panels;
3. reasoning recovery is attributed to replay only if **R0 beats C1** on MGSM
   and MSVAMP (LB95 > 0 for each paired contrast; both arms trainable, replay
   the only difference, which is D6's original comparison). R0 versus C2 is
   **not** informative here: C2's text-only pathway is bit-identical to
   stage 1, so its MGSM/MSVAMP are 11.6 / 34.1 by construction and any arm
   with replay beats it trivially. The preservation loss retains the recovery
   only if R1 is truly non-inferior to R0 at δ_R = 1.0 point on each
   benchmark (n = 250 / 1,000: MGSM is expected to have low power at δ_R =
   1.0 under equality; `power_sim.py` must quantify it). If its NI bound does not pass, the criterion fails and
   reasoning retention is **inconclusive**; absence of detected material harm
   is not reported as retention. v4 is context, not the matched control.

All bounds follow the three-seed conditional rule above. C1 is evaluated on
the confirmatory panel as the original-training reference, but R1 − C1 is
secondary because it mixes replay and the loss. Failure against R0 means the
loss has no demonstrated effect; failure against C2 means C2 remains the
method. Passing does **not** change "investigate whether" to "show that"
until the three-seed and confirmatory criteria are complete.

If the chain fails, the paper is: the 95.7% system result, source
dependence as exploratory, negative controls, no consolidated causal
explanation. No correlation is added to rescue the mechanism.

#### Compute ledger and minimum viable path (estimates, 2026-09-07)

All figures use S1's own unit costs on one H100: xGQA eval ≈ 40 min, CVQA
eval ≈ 5 min, one stage-3 training run ≈ 10 h, one pair-alignment job ≈ 3 h.
They are to be replaced by the timed pilot the roadmap requires; the ranking
of the rows will not change. **Measured 2026-09-08 (job 20483784, see the
pre-freeze note under Data construction):** a CVQA invocation costs 0.337
s/item + 19.5 s load, i.e. 2.8 min for an average panel unit of 433 items,
plus ≈10 min cold load once per allocation. The confirmatory-panel row and
the totals below use that measurement; the other rows keep their placeholders.

| stage | content | GPU-h |
|---|---|---|
| Block A | 60 xGQA + 96 CVQA evals (4 arms × 4 languages × 6, as specified in Block A) | 48 |
| Block B | 360 CVQA + 16 xGQA evals | 41 |
| Block D | 84 shuffled CVQA evals + 23 alignment scorings (11 stage 1 + 11 stage 3 + joint) | 10 |
| Block C pilot | C1–C3 training/evals plus C4 reuse and its 3 missing xGQA shuffles | 50 |
| C5 × 3 seeds, if G1-I is inconclusive | 3 × (10 + 6) | 48 |
| `bn_v4` / `id_v4` extra seeds | 4 × (10 + 3) | 52 |
| C1/C2 three-seed replication | 4 × (10 + 6) | 64 |
| R0/R1 × 3 seeds | 6 × (10 + 8) | 108 |
| confirmatory panel | 1,140 invocations (measured 09-08; was 95 at 5 min/eval); Option 1's 420 donor-confirmation calls alone = 19 | 52 |
| **full method path**, without optional checkpoint-stability seeds | | **≈ 373–421** |
| full path plus `bn_v4` / `id_v4` stability seeds | | **≈ 425–473** |
| **minimum scientific path**: A, B, D, C pilot | | **≈ 149** |
| minimum plus optional checkpoint-stability seeds | | **≈ 201** |

Window: 2026-09-08 to 10-06 is 28 days, i.e. 672 h of a single GPU with zero
queue time; the project's own constraints are minimal concurrency and 12 h
chained jobs, so realistic throughput in that window is on the order of
300–400 GPU-h. The implementation prerequisites (evaluator flags with a
prompt-only path, branch loading, `eval_matrix.py` and its tests, the shuffle
generator, manifests, the submit guard, C5 plumbing, the Block C fixes,
`inventory_cvqa.py`, `power_sim.py`, `block_d.py`) are about one week of
work, so Block A cannot start before roughly 09-15; the former fixed 09-20
gate becomes A+12, roughly 09-27 at the earliest, and the R0/R1 seeds plus the confirmatory
intervention evals (≈ 200 h) would have to run between 09-28 and 10-06.

**Consequence, stated before any result exists**: the full path does not fit
before 2026-10-12 under these constraints.

**Decision, 2026-09-07: Option 1.** Taken by Santiago after the reviewing
agents independently recommended it. What runs before the deadline: Blocks
A, B and D (D's 84 shuffled donor evals included, its seven-donor analysis
exploratory), the Block C single-seed pilot, and, only if capacity remains
after those, the extra seeds of `bn_v4` and `id_v4` (checkpoint-stability
work, never a prerequisite for any gate). What is declared **future work**
now, not after a result: R0/R1, the preservation loss and its confirmatory
intervention evals, the three-seed replication of C1/C2/C5.
**Option 1 override for C5 / G1-T**: if G1-I is inconclusive, C5 runs as a
single-seed pilot arm (its cost is in the C-pilot row), but the three paired
seeds that P8 requires to decide G1-T are future work; G1-T is therefore
**undecidable under Option 1**, G1 is reported as inconclusive, and the loss
gate cannot pass through the G1-T route. The confirmatory **donor** panel
runs only if Block D's exploratory test is promising and budget remains at
A+12. **"Promising" is pre-declared as**: the exploratory D-donor analysis
on P_current (7 × 4, `block_d.py`, the identical procedure and thresholds,
including the four G3 conditions with their bootstrap) returns a *defined*
one-sided exact p < 0.10 **and** the least-damage donor's exploratory
Δ_ground regret point estimate ≤ 1.0. If either fails, or the correlation is
undefined, the panel is not launched and G3 is reported as not tested;
nothing about P_current's outcome is allowed to alter the confirmatory
procedure itself.
The paper's method section is therefore the functional decomposition (Block
B) with the freeze-text pilot reported as exploratory; C2 is discussed as the
candidate method, never claimed. Every criterion in this entry stays as
written; what is not run is reported as not run. Recorded before the freeze
commit, as required:

- *Option 1, pre-declared scope*: the paper is the minimum scientific path.
  The 52 h of checkpoint-stability seeds run only if capacity remains. Block C is a
  single-seed pilot reported as exploratory; C2 is discussed as the candidate
  method, not claimed; R0/R1 and the confirmatory intervention evals are
  future work. The confirmatory **donor** evals (420, 19.3 GPU-h measured) run only if
  Block D's exploratory test is promising, as defined above, and budget
  remains on 09-27.
- *Option 2, full path*: requires either a second concurrent GPU through the
  post-implementation window **or** an extension beyond 10-12, subject to the
  timed pilot and queue. The full path is not reachable on one GPU by the
  current deadline; this does not make the minimum path unreachable.

Gate dates are re-stated relative to Block A's start (A+0): B and D at A+3
days, C pilot at A+7, loss gate at A+12. Whichever option is taken, the
v2.1.3 criteria stay as written; only what is *run* before the deadline
changes, and what is not run is reported as not run.

#### Roadmap

- Week of 09-08, before freeze: **done on 09-07**: the complete 20398782
  audit record (laptop and cluster-side evidence), `inventory_cvqa.py` and its inventory,
  `power_sim.py` and its output, `block_d.py` with synthetic tests, the
  Option 1 decision. `build_cvqa_s1.py` and
  `job-scripts/cvqa_s1_prefreeze.sh` implement the remaining two checks in
  one fail-closed 3 h job: native-query reconstruction is compared by id and
  canonical query hash against Bengali's current cluster JSONL, then the job
  extracts Spanish's embedded parquet images (the cold extraction is what is
  timed there) and times two correct-image evaluations of **Japanese**, a
  non-panel unit (full run and a `--limit` run, so per-item cost and
  model-load overhead are separated). **The timing unit is never a panel
  unit**: evaluating a donor on Spanish before the damage ranking is committed
  would burn the bn × Spanish cell of the prospective G3 test, and the
  builder's `attach-pilot` refuses any panel unit. Synthetic builder tests are in
  `analysis/test_cvqa_s1_builder.py`. **Done on 09-08**: job 20483784
  completed both checks (third attempt; see the data-construction note below
  for the two earlier failures) and its `audits/cvqa_s1_prefreeze.json` is
  committed as `d3bfc4a`. **Still open: commit "docs: freeze S1 v2.1.3
  experimental specification" before any new S1 `sbatch`.**
- After that freeze: implement evaluator flags, shuffle-map generator,
  `eval_matrix.py`, its tests, manifests and launcher submit guard; generate
  and verify map hashes; submit Block A. If G0 passes, submit Blocks B and D
  (including the 84 shuffled donor evals); otherwise do not launch B–C, while
  D may proceed independently under its own confirmatory grounding guard.
- A+7: Block C pilot (C1–C3, C5 single-seed only if G1-I is inconclusive;
  G1-T stays undecidable under Option 1, see the decision above); extra
  `bn_v4` / `id_v4` seeds are checkpoint-stability work, run only if capacity
  remains, not a prerequisite for the gate.
- A+12: loss gate (G0, G1, G2, G4, G5). G3 cannot be decided here; it waits
  for the confirmatory panel and does not block R0/R1.
- After A+12: under Option 1, write the minimum-path paper and run only the
  420 donor-confirmation invocations if the frozen contingency is met. Under
  Option 2, replicate C1/C2, train matched R0/R1, and run the full 1,140-call
  confirmatory panel. Lock the R1 recipe before either R0/R1 result is read.
  The loss enters the submission only if every frozen criterion finishes
  before the chosen deadline; otherwise it remains future work.
- In parallel, Approach 1 (Maryam): blind arms; bn → de/ru/zh zero-shot
  with her checkpoint; replay in her stage 3.
- Out of scope before 2026-10-12: coupled-architecture baseline
  (limitation); backbone expansion (sw/th); X3/D12 beyond exploratory.
  Also out of scope, carried over from the 2026-08-26 improvement queue
  (its items 1–3 became D9, D9b and D11): LLaVA sample 100k → 300k;
  culturally diverse stage-2 imagery (a hypothesis the CVQA − xGQA gap
  cannot test); Honeybee C-Abstractor for spatial; existence-QA for yes/no;
  an LwF-KL do-no-harm loss; a gate ramp-up schedule for D7's frontier.

#### Review corrections, 2026-09-08 (pre-freeze; accepted by Santiago the same day)

An independent review (`audits/review_2026-09-08.md`) reproduced E3 and X1
and found that two pre-freeze artifacts did not implement what this entry
promises. Corrections, all verified by synthetic tests that encode the
failure they fix:

- **`analysis/block_d.py`, rewritten (schema 2).** (i) The bootstrap drew
  images independently per donor because the RNG key contained the donor;
  seven identical donors gave regret 0 with UB95 = 25 points. Now one image
  draw per (target, subset) stratum is shared by every donor; identical
  donors give a regret distribution that is exactly 0. (ii) A constant
  damage or α vector made Spearman NaN, every permutation comparison false,
  and p = 0: a degenerate case became significant evidence. Now the
  correlation is reported as undefined, p is null and the verdict is
  `undefined`, never a pass. (iii) The verdict had three conditions; G3 has
  four. The utility regret UB95(max_s α^U_s − α^U_selected) ≤ δ_U is
  implemented from a second per-item endpoint, and the selected donor's
  pooled Δ_ground is item-micro over all targets (α stays macro over
  targets, the declared exception). (iv) The input carries subset, image
  and question ids and every donor must present the identical item universe,
  otherwise the analysis aborts; non-finite values abort. The production
  adapter from `eval_*` files remains post-freeze work: no evaluator
  produces the shuffled condition yet.
- **`analysis/power_sim.py`, xGQA pairing.** bn/de/ko are translations of
  the same questions on the same 398 images, and the pooling rule above
  requires one cluster resample applied jointly to all translations. The
  simulation had 1,194 independent (language, image) clusters for both the
  sign-flip perturbation and the bootstrap. Now one stratum of 398 shared
  image clusters; the artifact is regenerated (numbers above), the sign
  convention is explicit in the code, the artifact and this entry (see
  "Non-inferiority, three regions"), and the G3 grid is labelled as the
  correlation criterion only.
- **Specification reconciled** (this entry): Block A's ledger row now
  carries the 96 CVQA evals Block A specifies; the 35 h / 51 h donor
  estimates are superseded by the measured 19.3 GPU-h; Option 1 states the
  C5 / G1-T override, makes the `bn_v4` / `id_v4` seeds conditional on
  capacity everywhere, and pre-declares what "promising" means for the
  optional confirmatory donor launch. `README.md` (root and Approach 2)
  point to S1 and no longer propose LoRA.
- **Not changed, deliberately**: the scientific question, the architecture,
  Option 1, every threshold and margin. The review's remaining findings
  (evaluator flags, strict branch loading, `eval_matrix.py`, manifests,
  submit guard, the stage-3 completion marker and exact resume) block the
  corresponding launches, not the freeze; they are listed below.
- **Still missing for reproducibility**: a pinned environment capture of the
  validated cluster venv (`pip freeze`) with model and tokenizer revisions.
  To be added from a Rorqual login node.

#### Post-freeze implementation prerequisites (none exist yet)

- **Data construction, fail-closed.** The active `Stage3/load_vqa_eval_data.py`
  cannot build S1's panels: it configures only bn for xGQA and id/jv for CVQA
  (`:62–73`), it prefers CVQA's `Translated Question`, which is the **English**
  question, over the native `Question` (`:259`), and it drops `Subset` and the
  image id (`:265`). The existing CVQA result files carry native queries, so
  they came from Maryam's loader, not from this one. The S1 builder must emit
  `id`, `image_id` (the `<image>` prefix of `ID`), `subset`, `language`,
  `nllb_lang_tag`, `query` = native `Question`, `english_query` =
  `Translated Question`, `choices` = `Translated Options`, `answer_index`, and
  abort on any row whose image cannot be obtained (`Image Source` is a URL for
  external images and the literal `Self-open` for embedded ones, which need
  the parquet). Before the freeze, rebuild one existing panel and hash its
  `query` column against the file on the cluster; a mismatch blocks the freeze.
  `Approach2/build_cvqa_s1.py` now implements this schema directly from local
  parquet, never follows `Image Source`, verifies the inventory's item/image
  counts, and writes both canonical image-id files and question-id hardlinks
  for the legacy evaluator. `job-scripts/cvqa_s1_prefreeze.sh` runs the
  Bengali hash audit and cold Spanish extraction/pilot together. The code
  exists; its cluster report is not evidence until that job completes.
  **First run, job 20441017 (2026-09-07, HEAD `18b5536`): audit FAILED,
  freeze still blocked.** Bengali rebuilt from the parquet matched the
  cluster `cvqa/bn.jsonl` on all 286 ids (identical id-universe hash) and
  the existing queries are native Bengali, so the provenance claim above
  holds. The 6 mismatched items differed only in leading/trailing
  whitespace: the existing file keeps CVQA's raw `Question`, the builder
  applied `.strip()`. Decision: the builder now emits `query` and
  `english_query` verbatim (emptiness is still checked on the stripped
  value), because the prompts of every existing CVQA run contain that
  whitespace and new units must be built the same way; the audit itself
  stays byte-exact. Spanish (2058 items, 1155 images) and Japanese (203,
  94) extracted in under 4 min before the gate. The next submission runs
  from a new commit, so it starts a fresh scratch directory and the Spanish
  cold-extraction timing is unaffected.
  **Second run, job 20443726 (2026-09-07, HEAD `ea3b4e3`): query audit
  PASSED** (Bengali canonical query sha256 `e7342734…a086c`, identical to the
  cluster file), Spanish and Japanese rebuilt again in under 5 min, and both
  Japanese evaluations with `stage3_bn_dcl` completed (correct image: 57/203
  = 0.281; `--limit 40`: 12/40 = 0.300). **The job still exited 1**, at
  `attach-pilot`: the two-point subtraction gave 2.81 s/item and a load cost
  of −70 s. Cause, from the evaluator's own timestamps: the full run loaded
  the weights cold from Lustre (356 s from data load to "Loaded mapping"),
  the `--limit` run loaded them from the page cache (12 s); inference itself
  was 0.37 s/item in both runs (75 s/203 and 15 s/40). The two-point method
  assumes both runs pay the same load cost, and the first run in an
  allocation never does. Decision: the launcher now runs one untimed
  `--limit` warm-up before the timed pair, so both timed runs load from the
  cache; its wall time minus the per-item cost is recorded as
  `pilot.warmup_run.cold_load_seconds`, the first-load overhead each S1
  allocation pays once. It is recorded, not scaled into the GPU-hour
  estimate, because the number of allocations is a scheduling choice.
  `attach-pilot` refuses a warm-up that loaded faster than the timed runs.
  Expected from the next run: ≈0.37 s/item, warm load ≈12 s, cold load
  ≈350 s.
  **Third run, job 20483784 (2026-09-08, HEAD `4049c83`): COMPLETED, both
  checks pass; report committed as `d3bfc4a`.** Query audit PASS (0 missing,
  0 extra, 0 mismatched of 286). Cold Spanish build 194 s (35 s parquet
  hashing, 158 s scan/extract/write; 1,249 canonical images, 2,261 question
  links). Japanese, `stage3_bn_dcl`, correct image: 57/203 = 0.281 (full) and
  12/40 = 0.300 (`--limit`), same predictions as job 20443726. Measured
  costs, page cache warm: **0.337 s/item, 19.5 s model load per
  invocation**; cold first load 612 s (626 s warm-up minus 40 items), larger
  than the 356 s seen on 20443726, so the cold load is Lustre-bound and
  varies by node; budget ≈10 min per allocation. Scaled to the 12-unit,
  5,202-item panel: 1,989 s ≈ 0.55 GPU-h per (donor, condition) panel pass;
  **420 donor-confirmation invocations (Option 1) = 19.3 GPU-h**; the full
  1,140-call panel (Option 2) = 1,140 × 165.6 s ≈ 52 GPU-h. Both are about
  55% of the ledger's 5-min-per-eval placeholders (35 h and 95 h); the ledger
  row is updated below. Pre-freeze evidence is complete.
- **Evaluators abort on a missing image** instead of skipping it
  (`evaluate_vqa.py:174`, `evaluate_cvqa.py:140` skip today); incompatible flag
  combinations are rejected; NLLB and SigLIP2 are loaded only when their branch
  is used.
- **Block C launcher**: a dedicated script, not `train_stage3.sh` (one epoch,
  seed 42, no `--vis-layers` today); S1 needs two epochs, seed 13,
  `"9,18,-1"`, a dedicated deterministic sampler whose order hash goes into
  the manifest and survives a resumed job.

- Evaluators: `--prompt-mode {question,instruction}`, `--no-text-branch`,
  `--shuffle-map`, `--no-image`. **`--no-image` with `--no-text-branch` (A2/A4
  no-image cells) needs a prompt-only path**: `_build_prefix_raw` raises when
  neither `input_ids_mt` nor `pixel_values` is given (`model.py:347`).
- Branch loading `--txt-ckpt` / `--vis-ckpt`, explicit `gate = 1.0` for
  pre-gate checkpoints, strict keys otherwise; `eval_matrix.py` with its four
  guarantees as tests.
- C5: `use_text_branch=False` through `train_stage3_vqa.py`, batches that do
  not tokenise NLLB, and load / freeze / gate paths that tolerate a missing
  `mapping_txt`.
- Block C: frozen towers to `eval()` after `model.train()`; validation split
  by `vg_image_id`; matched R0/R1 replay arms with a frozen loss recipe.
- The shuffle-map generator under `Approach2/shuffle/`, launcher submit
  guard and manifests.
- `analysis/block_a.py`: P1/P2, G0/G1-I, U and Δ_ground from correct + three
  shuffles, validation of ids, manifests and map hashes across conditions,
  shuffles averaged within item, paired image-cluster bootstrap stratified by
  target, three regions, with invariance and fail-closed tests. **Must exist
  before Block A is submitted**; without it Block A yields predictions but
  no reproducible decision.
- Pre-freeze analysis artifacts that now exist: `Approach2/inventory_cvqa.py`
  (+ `audits/cvqa_inventory.json` and its sha256), `analysis/power_sim.py`
  (+ `audits/power_sim.json`), `analysis/block_d.py` (+ `test_block_d.py`).

#### Shuffle maps and git

`evaluation/` is gitignored (`.gitignore:34`), so the maps themselves are
not versioned. What is versioned, under `Approach2/shuffle/`: the
deterministic generator, the seeds, the sha256 of each panel's id universe
and the sha256 of every generated map. A launcher **must** regenerate the
maps, check the hashes, and abort on mismatch. None of `Approach2/shuffle/`,
the guards or the manifests exists yet; they are implementation work, not
verified properties.

#### Provenance

Every launcher **must** write a JSON manifest next to its outputs: experiment id,
hypothesis, estimand, prediction, source/targets, seed, **spec SHA** (the
S1-freeze commit), **code SHA**, dirty flag, full arguments, checkpoint
paths and sha256, data and split hashes, shuffle-map hashes,
`vis_layers`, decoding settings, SLURM job id. `eval_*.summary.json`
carries the same fields. Recommended commit order: (1) the four documents;
the five analysis files already present (`_boot.py`,
`e3_noninferiority.py`, `x1_did.py`, `test_invariance.py`,
`test_fail_closed.py`); the required pre-freeze `power_sim.py`, `block_d.py`
and their tests; `inventory_cvqa.py`, its machine-readable output/hash; and
the versioned 20398782 audit record, as "docs: freeze S1 v2.1.3 experimental
specification"; (2) implementation, tests and launchers. The freeze commit
does not exist until every item in (1) exists and agrees with this entry.

---

### Implementation record — 2026-09-09 (after S1 freeze; no protocol amendment)

Spec SHA remains `3b4faff4d673fad94cd3115b3f14212eeca9c12e`. The frozen S1
entry above is unchanged. The post-freeze evaluator controls, strict branch
loader, deterministic shuffle generator/registries, shared matrix runner,
submission/runtime provenance guards, and `analysis/block_a.py` now exist.
Stage 3 has optimizer-boundary snapshots, deterministic schedules, a distinct
completion marker and the corrected S1 tower/split/C5 paths. Operational
commands and the scope of verification are in `S1_IMPLEMENTATION.md`.

**Measured effect:** software verification only. The synthetic A plan has
156 cells and B has 376; per-item hash/condition failures abort. Tiny CPU
models reproduce the frozen full-prefix forward/generation path and the
production stage-3 loop produces identical mapping weights after interrupted
and uninterrupted training. These tests do not establish parity for the
real 9B checkpoints, GPU determinism or task accuracy. No S1 GPU job was
submitted and no hypothesis verdict changes. Santiago will capture the
Rorqual environment from his authenticated cluster session.

### Block A result — 2026-09-11 (job 20674060, spec SHA `3b4faff`)

156 cells, 27 h 53 min on one H100, report `audits/s1_A_analysis.json`
(committed `345ad4d`). Every gate applied exactly as frozen; nothing below
was chosen after seeing a number.

**Pipeline validation.** A1 on xGQA-bn returns utility **47.66**, identical
to the historical `eval_xgqa_bn_dcl` accuracy for the same checkpoint. The
rewritten S1 evaluator therefore agrees with the pre-S1 one on a real panel.

**G0 passes.** A1 Δ_ground on the primary CVQA panel jv/mn/ga is **+3.89
[+2.24, +5.64]**, LB5 > 0, so Blocks B and C are authorised. On xGQA it is
+19.01 [+18.00, +20.00]. This closes the caveat E2 and E3 have carried since
09-07: the visual contribution is instance-specific, not only "an image
beats a gray canvas". Per unit it is uneven: ga +8.69 [+5.88, +11.49], jv
+4.15 [+0.67, +7.78], **mn −1.39 [−4.01, +1.23]** and si (secondary) +17.04.
Mongolian shows no instance-specific grounding at all, which is evidence on
E4 and a limitation the paper must state: the pooled pass is carried by ga
and jv.

**G1-I inconclusive, and the reason matters.** A2 removes the text branch and
leaves the question in Gemma's own prompt. On xGQA, Δ_ground is **−0.22
[−0.48, +0.04], non-inferior** at δ=1 with 12,578 items per language, and
utility is −0.82 [−1.09, −0.55], which misses non-inferiority by **0.09 of a
point**. On CVQA pooled, +0.14 [−1.10, +1.44] and −0.11 [−1.29, +1.10], both
inconclusive as `power_sim.py` predicted for this panel (inconclusive with
probability 0.79–0.88 across the whole grid). The frozen rule requires both
endpoints non-inferior to call the branch dispensable, so the verdict is
**inconclusive**, recorded as such. What the numbers bound, separately from
the gate: the text bridge's entire contribution at inference is under one
point on either endpoint. A4, the English question straight into the prompt
with no bridge, scores **+36.90** utility on CVQA against A1's +35.72.

**A3: the bridge does not carry the question.** Removing the question from
the prompt and leaving only NLLB to carry it costs xGQA utility 47.66 →
**11.06** and Δ_ground 19.01 → **2.08**, both materially inferior; CVQA
utility −4.39 [−6.31, −2.35]. So the bridge is neither necessary nor
sufficient at inference: the question reaches the model as prompt tokens.

**What this does and does not decide.** It is an inference verdict on a
checkpoint *trained* with the bridge. DESIGN has separated these since v2.1.2
(G1-I versus G1-T) precisely so this reading is available: A cannot say the
bridge was unnecessary during training. That is Block C, and **C5, the arm
trained with no text branch at all, is now the most informative run in the
plan**: C5 ≈ C1 would mean the bridge is idle in training too; C5 < C1 would
mean it shapes training while remaining removable at inference. Under Option
1 the three paired seeds P8 needs stay future work, so G1-T remains
undecidable and C5 runs as a single-seed pilot arm, exactly as the Option 1
override says.

**Consequence for Block B, recorded before B runs.** B asks whether the donor
difference *follows* the text branch, with LB95(D_T) > 0 required. A has
bounded the branch's whole inference contribution on CVQA at roughly zero, so
a detectably positive D_T is now unlikely; substitution is not removal, so it
is not impossible. No criterion, margin or cell is changed. What changes is
**execution order only**: `s1_plan.py` now emits the eight P3/P4 cells on
jv/mn/ga first, so the ~4 h that can falsify the main hypothesis run before
the remaining ~19 h of factorial and controls. Cheapest falsifier first; the
full grid still runs.

**Not restated.** The question in `SCIENCE.md` §2 stands. A bounds one
channel; it does not answer whether the transfer variation comes from
source-conditioned connector co-adaptation.

### Environment change and the parity gate — 2026-09-11 (job 20919626)

Block B's first allocation aborted after nine minutes on its first parity
cell, `s1_B_cvqa_jv_T3_id__V3_id_correct`, item `5865939224274762645_2`.

**What the gate proved, and what it did not.** `query`, `choices` and
`answer_index` were identical to the historical `eval_cvqa_jv_zsid.jsonl`, so
the CVQA panels rebuilt by `build_cvqa_s1.py` reproduce the files the legacy
runs used, for Javanese as well as the Bengali unit the pre-freeze audit
checked. What differed were the choice log-likelihoods: −14.156, −12.585,
−12.452, −21.719 became −13.599, −12.275, −12.395, −21.756, shifts of 0.037
to 0.557 nats. The reference's top two choices were 0.133 apart, so the argmax
flipped.

**Corrected on 2026-09-12: the cause stated in the next paragraph is wrong.** The drift comes from the CVQA image files, not from transformers; the evidence is in the entry "Block B result, and the corrected cause of the CVQA parity drift". The original text is kept below as it was written.

**Cause.** The cluster venv now carries transformers 5.13.1; the reference
runs were produced under 4.x. Gemma 2 applies attention-logit soft-capping,
whose implementation differs across versions and attention backends. This is
consistent with the rest of Block A: xGQA reproduced the historical
`eval_xgqa_bn_dcl` accuracy exactly (47.66), because greedy decoding is
robust, while an argmax over four nearly tied log-likelihoods is not. In the
reference cells 4% of items have their top two choices within 0.1 nat and
10% within 0.2; resampling them with shifts of the observed size flips 4 to 9%
of predictions.

**Decision.** Exact per-item parity is not satisfiable across that boundary,
and dropping the gate would lose a real regression guard, so
`compare_predictions` is recalibrated rather than removed. Inputs stay
absolutely strict: the item universe, `query`, `choices` and `answer_index`
must be identical, which is what catches a wrong checkpoint, panel, prompt or
image. Open-ended cells still demand exact prediction equality. For choice
cells three guards replace exact equality: the 99th percentile of the per-item
score shift, measured over items that did **not** flip, must stay under 2.0
nats; disagreements must stay under 25% of the panel, against the 50 to 75%
a wrong checkpoint would give on four choices; and no disagreement may sit on
an item whose reference top-2 gap exceeds twice the largest shift observed on
an item that agreed. The drift is estimated on agreeing items on purpose: over
all items a broken pipeline would inflate the estimate until it excused its
own flips. Each cell prints its disagreement count, drift and both accuracies.

**Consequence for the record, and it is not small.** Numbers produced under
4.x and under 5.13.1 are not interchangeable on CVQA multiple choice. Every
S1 block computes all of its own cells in one environment, so its contrasts
are paired and unaffected. But **no contrast may mix a legacy CVQA number
with a new one**, and the legacy figures in this log (E3, X1, the D series)
stay valid only as the internally consistent set they were. The paper reports
the environment with the results.

**Not changed**: no gate, margin, endpoint or cell. `analysis/block_a.py` is
untouched and Block A's report stands, since all 156 of its cells ran in the
same environment.

### Block D implementation — 2026-09-11 (no protocol amendment)

The frozen Block D-donor procedure had an analysis (`analysis/block_d.py`) and
no way to feed it. That path now exists, in three pieces, and nothing in the
frozen text changed.

- **`s1_plan.py --block D`** emits the grid: seven donors, each a whole stage-3
  checkpoint (bn is `stage3_bn_dcl`, the rest are v4), on the four current
  transfer targets, correct plus the three seeded shuffles. **112 cells,
  about four hours.** The grey canvas is absent on purpose: D's endpoint is
  Δ_ground and Δ_gray is a different quantity. The historical correct-image
  files are not reused either, although they exist for exactly these donors and
  targets: they predate the move to transformers 5.x, so pairing an old correct
  with a new shuffled would not be a paired difference (see the entry above).
- **`analysis/block_d_input.py`** joins the matrix to the predictor. It
  authenticates every cell through `block_a.load_cells`, now shared, then emits
  per item `ground` = correct − mean of the three shuffles and `utility` =
  correct, keyed by subset and image so the bootstrap can resample images
  jointly across donors. Damage comes from each donor's OWN stage-1 and stage-3
  pair-alignment files, as the spec requires, and the adapter aborts with the
  exact `CKPTS=` line to run when one is missing rather than treating an absent
  margin as zero. Tests in `analysis/test_block_d_input.py`, including that its
  output is accepted by `block_d.analyse`.
- **Still to run before the plan is built**: eleven `pair_alignment.sh`
  scorings, `stage1_{id,ru,zh,de,pt,ko}` and `stage3_{ru,zh,de,pt,ko}_v4`. Only
  four of the fourteen checkpoints are scored today. **The scorings are
  committed before the D plan is built**, which is what fixes the donor ranking
  ahead of any outcome.

`pair_alignment.sh` and `evaluate_all.sh` no longer commit their harvest by
themselves, for the reason given under the environment entry: a harvest that
moves HEAD kills a queued S1 allocation. And S1 allocations now run from a
detached worktree of their own commit, so pulling in the main clone is safe.

### Block B analysis, one pre-declared reading, and the D13 and D12 runs — 2026-09-12

**Recorded before `analysis/block_b.py` is run on Block B's data.** B's 376
cells finished in job 20925480 (16 h 10 min, exit 0). Their outputs exist only
on the cluster; no contrast has been computed or looked at.

**Block B analysis.** `analysis/block_b.py` implements P3, P4 and G2 as frozen.
Each contrast is a fixed linear combination of cell endpoints taken item by
item on identical items, pooled micro over jv/mn/ga, with the CI from the
paired image-cluster bootstrap stratified by target and subset, 4,000
resamples; Δ_ground per item is correct minus the mean of the three shuffles.
It fails closed on an incomplete 376-cell grid, a checkpoint that is not the
arm's lineage, a text-branch flag that does not match the arm, and unpaired or
differing items. `analysis/test_block_b.py` fixes every contrast's arithmetic
in closed form, including a null factorial that must return exactly zero.

**Sign, clarified and not amended.** The non-inferiority paragraph says P1–P8
are written reference − candidate and are the negation of what is computed.
That sentence governs contrasts judged against the δ margin. P3 and P4 have no
margin: their formulas state their direction (Indonesian lineage minus Bengali
lineage; matched minus crossed pairs) and G2's own gate row fixes the sign as
LB95(D_T) > 0. `block_b.py` reports them exactly as written. Reading the
negation into them would invert G2 against its own row.

**One exploratory reading, declared now.** Block A bounded the text branch's
inference contribution near zero (09-11), which makes "the donor difference
follows the vision branch" the likelier outcome. No frozen criterion names
that reading; G2 defines only the text one. It is therefore declared here,
before the analysis runs, as an **exploratory** reading symmetric to G2 and
not as a gate: **LB95(D_V) > 0 and UB95(D_T − D_V) < 0**, on Δ_ground, primary
panel. It cannot pass or fail the loss gate and cannot rescue G2; if it holds
it is reported as a localisation observation from recombined checkpoints,
functional and not causal, as Block B's labels already say. The statement on
09-11 that such an outcome would be "a clean positive result" was made without
a criterion and is withdrawn in that form.

**D13, MetaMathQA replay: measured, not attributable.** Job 20925481,
`stage3_bn_mm30000` against the v4 reference `stage3_bn_dcl`, identical items:

| benchmark | v4 | mm30000 | lost / gained | exact McNemar |
|---|---|---|---|---|
| MGSM (n=250) | 62.00 | 35.20 | 76 / 9 | p = 2.4e-14 |
| MSVAMP (n=1000) | 64.50 | 50.70 | 193 / 55 | p = 3.7e-19 |
| xGQA full (n=12,578) | 47.66 | 47.54 | 875 / 859 | p = 0.72 |
| xGQA blind | 30.83 | 30.10 | | |

An answer-format artifact was checked and ruled out: `extract_math_answer`
accepts MetaMathQA's "The answer is:" form, and the lost items extract cleanly
and are wrong on the arithmetic. **The run was not a single variable**,
contrary to its launcher header: besides the replay pool, the stage-3 trainer
was rewritten on 09-09 (optimizer-boundary snapshots, deterministic replay
schedule) and the environment moved to transformers 5.13.1, whereas
`stage3_bn_dcl` was trained and evaluated before both. The drop cannot be
attributed to MetaMathQA until the GSM8K replay is reproduced with the current
trainer and environment. The measured statement is only that this run, as a
whole, lost about 27 MGSM and 14 MSVAMP points and left xGQA unchanged. MSVAMP
shares no source with GSM8K, so a loss of MGSM train-test overlap alone would
not explain the MSVAMP drop either.

**D12, pooling: training finished, and its comparison is confounded the same
way.** `launch_joint.sh` with `LANGS="jv mn ga de ru zh"`: job 20937497 trained
all six in 22 h 25 min and job 20937498 only verified completion (58 s);
`train_stage3_all.sh` exits non-zero on any failed language and both exited 0.
The `vj` arm was trained with the rewritten trainer under transformers 5.x.
The independent arm, the v4 checkpoints, was trained and evaluated before
both. **So `gap_report.py v4 vj` is not a single variable**, and on CVQA it
would also pair multiple-choice scores from two environments, which the 09-11
entry forbids. Recorded now so the pooling result is never read without it.
The evaluation half is removable by re-evaluating the six v4 checkpoints in
the current environment; the training half is removable only by retraining
the independent arm with the current trainer, about as long as the vj run.
The registered prediction in D12 and X3 stands as written; what changes is
that no verdict is drawn from a mixed-environment comparison.

**Evaluation half of the D12 confound: how it is removed.** `evaluate_all.sh`
gained `EVAL_TAG`. With `EVAL_TAG=tf5`, both arms are evaluated in the current
environment into tagged subdirectories and harvested as
`eval_<bench>_<L>[_BLIND]_<round>_tf5`, so the versioned 4.x-era v4 files are
never overwritten and a mixed read is impossible by name: the pooling
comparison is `gap_report.py v4_tf5 vj_tf5`, and the untagged `vj` files never
exist. Same six languages, same benchmarks, same launcher for both arms. The
training half of the confound (old trainer for v4, rewritten trainer for vj)
is not removed by this and stays stated with any pooling result.

### D13 and D12 controls, recorded before launch — 2026-09-12

Decided by Santiago the same day: reproduce the GSM8K replay in the current
environment, and retrain the independent arm of the pooling comparison with
the current trainer. Both are launched only after this entry is pushed.

**Replay volume, a correction to the D13 record.** The stage-3 trainer caps
every replay file at `--replay-max-rows-per-file`, default 10,000, a cap that
entered in `4beae75` on 2026-08-24, before D11 trained `stage3_bn_dcl`. So the
run tagged `mm30000` trained on a seeded 10,000-row sample of its 30,000
MetaMathQA rows, against GSM8K's 7,473, all of which are kept; the Bengali
translation file is capped at 10,000 rows in every arm, dcl included. D13
compared 10,000 against 7,473 math rows, not 30,000. The tag stays as it is
for traceability.

**D13 control: `REPLAY_SOURCE=gsm8k`, checkpoint `stage3_bn_gsm8k`.** Same
launcher, stage 1, stage 2, VQA data, translation replay, row cap, replay
interval, epochs, learning rate, trainer and environment as `mm30000`; only
the math pool differs, and it is the GSM8K file `stage3_bn_dcl` trained with.
Two paired comparisons on identical items, each with its own exact McNemar
test and point difference, never one read from the other's significance:
- **pool**: `mm30000` against `gsm8k`, on MGSM, MSVAMP and xGQA full;
- **trainer and environment**: `gsm8k` against `stage3_bn_dcl`, on the same.

What each outcome means, fixed now. If `gsm8k` does not differ significantly
from dcl on MGSM and MSVAMP while `mm30000` is significantly below `gsm8k`,
the MetaMathQA sample costs reasoning. If `gsm8k` is itself significantly
below dcl, the loss belongs at least in part to the trainer rewrite or to
transformers 5.x, and then **every checkpoint trained after 09-09 carries it,
including the pooled `vj` round and every Block C arm**: that outcome is the
more urgent one and would be raised before any C launch. If both differences
are significant, both are reported with their sizes and no single cause is
named.

**D12 control: `launch_v4r.sh`, round `v4r`.** Stage 3 for jv, mn, ga, de, ru
and zh, each from its own `stage1_<L>`, with `stage2_dc_llava`, replay, two
epochs and the current trainer: exactly the arguments `launch_joint.sh` gave
`vj`, minus the shared text mapping. The launcher requires all six own stage-1
mappings, unsets any exported `STAGE1_CKPT` so the control cannot silently
become a second joint arm, and chains its evaluations with `afterok` under
`EVAL_TAG=tf5`. `vj` is evaluated under the same tag.

**How D12 is read, fixed before any tf5 number exists.** D12 registered
`gap_report.py` and the differential prediction "lift jv/mn/ga, leave de/ru/zh
flat; a uniform lift refutes it as surely as no lift". `gap_report.py`'s groups
(low-resource bn jv mn si ga, higher-resource ru zh pt id ko) contain
languages neither arm trained; it reports them as missing and is shown for
completeness. The verdict is read on the six named languages, `vj_tf5` minus
`v4r_tf5`, paired on identical items with the image-cluster bootstrap:
- **lift**: CVQA pooled jv/mn/ga, LB5 > 0;
- **flat**: xGQA pooled de/ru/zh within the project's δ = 1.0, that is
  LB5 > −1.0 and UB95 < +1.0 (xGQA's 12,578 items per language can resolve
  that; CVQA's ru/zh cannot and are reported descriptively).
Supported only if both hold. Refuted if the lift fails, or if de/ru/zh lift by
at least as much as jv/mn/ga. Anything else is inconclusive. `v4_tf5` against
`v4r_tf5` measures the trainer's effect on the independent arm and is
descriptive.

### Block B result, and the corrected cause of the CVQA parity drift — 2026-09-12

**The cause recorded on 09-11 was wrong, and Block B's own parity cells refute
it.** Every gray-canvas parity cell reproduced its historical run exactly:
0 of 297, 312, 326, 225 and 286 predictions differ, score drift 0.000. The
gray canvas is generated in memory, so those cells run the same NLLB, SigLIP 2,
both mappings and Gemma 2 as the correct-image cells, under transformers
5.13.1. Had the library moved Gemma 2's numerics, they would drift too. Only
real images drift: the correct-image cells differ on 0 to 10 items with drift
p99 of 0.71 to 1.03 nats. xGQA, whose image files are the same in both
pipelines, reproduced exactly (47.66). The CVQA files are not the same: of 20
Javanese items compared between the legacy copy (`$DT/Stage3/data/cvqa/images`)
and the S1 panel (`$DT/Stage3/data/cvqa_s1/images`), **0 are byte-identical**,
all have the same size, and the largest per-pixel difference is 11 to 28 out of
255. `build_cvqa_s1.py` decodes every embedded image and re-encodes it as JPEG
at quality 95 (`canonical_image_bytes`); the legacy copies came through a
different pipeline. **The S1 CVQA panels feed the model different pixels.
Transformers 5.x is not implicated at inference.**

What changes because of it:
- **The rule stands, with a different reason.** No contrast may pair a CVQA
  number scored on the legacy image copy with one scored on an S1 panel. Every
  S1 block (A, B, D, the confirmatory panel) evaluates all of its arms on the
  S1 panels, so its contrasts are unaffected; the legacy CVQA figures (E2, X1,
  the D series) stay valid as their own internally consistent set.
- **A sensitivity to state in the paper.** Lossy re-encoding alone moved one
  CVQA cell by 1.05 accuracy points (the bn correct-image parity cell, 39.16
  historical against 38.11), the size of δ. CVQA multiple-choice accuracy is
  fragile at that scale, which is one more reason the project reads CVQA only
  pooled.
- **`compare_predictions` stays as recalibrated**; only its stated cause is
  corrected. Its tolerance was sized on this drift, and all ten Block B parity
  cells passed with at most 10 of 286 predictions differing.
- **D12.** `evaluate_all.sh` scores both arms on the legacy copy and the model
  path reproduces across versions, so the evaluation half of the D12 confound
  is very likely absent. The tagged re-evaluation is kept as a check, with a
  prediction fixed now: a `v4_tf5` evaluation reproduces the historical v4
  numbers exactly on xGQA and within a handful of items per CVQA cell. The
  training half (old trainer for v4, rewritten trainer for vj) is real, and is
  what round `v4r` removes.
- **D13.** MGSM and MSVAMP have no images, so evaluating under 5.x is no
  confound there either. The live confound is training alone: the rewritten
  trainer, and training numerics under 5.x, which the gray cells do not test.
  The GSM8K control addresses exactly that.
- **Retracted wording**, kept visible where it was written: "Gemma 2 applies
  attention-logit soft-capping, whose implementation differs across versions"
  (09-11 environment entry); "they predate the move to transformers 5.x" as the
  reason not to reuse historical correct-image files (09-11 Block D
  implementation entry); the same claim in the 09-12 D13/D12 entries. The
  decisions those sentences supported stand: historical correct-image CVQA
  files are still not reused in S1, now because their pixels differ.

**Block B result** (job 20925480; report `audits/s1_B_analysis.json`,
`1c34190`). Primary panel CVQA jv/mn/ga pooled, 935 items, paired
image-cluster bootstrap stratified by target and subset, 4,000 resamples.
Contrasts exactly as frozen; one-sided 5/95 bounds, and the two-sided 95% CI
for P4.

| contrast | Δ_ground | LB5 / UB95 | utility | LB5 / UB95 |
|---|---|---|---|---|
| D_T, stage-3 text imprint, id − bn, vision at V2 | +3.24 | +0.72 / +5.74 | +1.93 | −0.54 / +4.39 |
| D_V, vision imprint, id − bn, text at stage 1 | +3.24 | +1.93 / +4.59 | +4.65 | +3.19 / +6.24 |
| D_T − D_V | +0.00 | −2.75 / +2.81 | −2.73 | −5.60 / +0.11 |
| P4, matched − crossed | −0.87 | CI95 −2.29 / +0.58 | −1.66 | CI95 −3.06 / −0.32 |

- **G2 fails.** LB95(D_T) = +0.72 > 0, but LB95(D_T − D_V) = −2.75. The
  frozen text-localisation prediction is not supported and is recorded as
  failed.
- **The exploratory vision reading declared this morning fails too.**
  LB95(D_V) = +1.93 > 0, but UB95(D_T − D_V) = +2.81.
- **What the numbers do say.** Both connectors carry a positive
  Indonesian-minus-Bengali grounding difference, each with its lower bound
  above zero, and the difference between them is estimated at 0.00 with an
  interval 5.6 points wide. On this panel the donor difference is carried by
  both branches and cannot be localised to either. The 09-11 reasoning "if the
  text bridge carries nothing at inference, the imprint must live on the
  vision mapping" is refuted by D_T's positive lower bound and is withdrawn.
- **P4 on grounding is null; on utility it is negative** with a CI excluding
  zero: crossed pairs compose slightly better than matched ones, the opposite
  sign to co-adaptation. P4 predicted no direction; this is reported as
  measured, functional, one seed per checkpoint.
- **Secondary panels and controls, descriptive only.** CVQA-bn, the
  same-language out-of-domain control: D_V +7.98 [LB5 +4.76], D_T −0.93,
  D_T − D_V −8.92 [UB95 −3.22], so there the difference follows the vision
  branch clearly. Per target the pattern varies (mn: D_V +4.43 [+2.21], D_T
  +1.92 [−1.79]; ga: D_T +5.01 [+0.20], D_V +3.99 [+1.75]; jv: neither lower
  bound above zero) and single-language cells are not read alone. No-X_f
  controls: removing the text branch from the bn checkpoint moves grounding by
  +0.14 and utility by −0.11, the same values Block A measured for A2 on these
  cells, a cross-block consistency check; removing it from the id checkpoint
  **raises** grounding by +2.85 [+0.77, +4.91] and utility by +4.92 [+3.00,
  +6.85]. xGQA task controls: the matched bn checkpoint reproduces 47.66, and
  swapping either branch costs 2.9 to 3.5 utility points on xGQA-bn.
- Every label is functional, not causal, and conditional on one seed per
  checkpoint.

**What it does to the question, stated and not acted on.** SCIENCE.md §2
point 4 asks whether the source effect arises from source-conditioned
connector co-adaptation. P4 does not support co-adaptation in its registered
sense, and localisation is undetermined. The question is not restated here;
restating it is Santiago's decision, and if made it will be dated and recorded
in both files.

### Block D damage ranking, fixed before any Block D evaluation — 2026-09-13

Pair alignment for the seven donors' own checkpoints was scored in job
20993899 and committed as `1a74d1c` (the six `stage1_<L>`) and `9e43428` (the
five `stage3_<L>_v4`), before the Block D plan was built; `stage1`,
`stage3_bn_dcl` and `stage3_id_v4` were already in the repository. All
fourteen files carry a centered margin for every donor-to-target pair.
Damage, as frozen, is the mean over jv/mn/ga/si of the centered margin at
stage 1 minus the same margin at stage 3, each read from the donor's own files
by `analysis/block_d_input.py`; positive means deterioration. The same ranking
serves the exploratory panel and the confirmatory panel, because damage is
defined on these four targets in both.

| donor | damage | jv | mn | ga | si |
|---|---|---|---|---|---|
| zh | −0.01190 | −0.0129 | −0.0118 | −0.0105 | −0.0124 |
| de | −0.01005 | −0.0122 | −0.0097 | −0.0087 | −0.0096 |
| pt | −0.00910 | −0.0111 | −0.0090 | −0.0070 | −0.0093 |
| id | −0.00520 | −0.0052 | −0.0051 | −0.0055 | −0.0050 |
| ru | −0.00490 | −0.0060 | −0.0055 | −0.0028 | −0.0053 |
| ko | −0.00265 | −0.0029 | −0.0031 | −0.0012 | −0.0034 |
| bn | +0.00240 | +0.0030 | +0.0004 | +0.0054 | +0.0008 |

**Selected donor by the frozen rule, least damage: zh.** No two donors tie,
so the NLLB-code tie-break is not used.

Two properties of this predictor, recorded before any outcome exists:
- **Only Bengali's stage 3 lowered its centered margin to the four targets;
  the other six raised it.** On this panel "damage" is therefore mostly an
  improvement axis, and G3 reads as "donors whose stage 3 raised pair
  alignment more transfer better". The sign of the effect G3 tests is
  unchanged: more damage, lower α.
- **The spread is small**, 0.0143 in centered margin from zh to bn, and pair
  alignment has no measured noise floor: each checkpoint was scored once on
  997 FLORES sentences, with no repeated scoring and one training seed. The
  ranking is used exactly as frozen; its reliability is a stated limitation,
  not a reason to revise it.

### D13 control result: most of the drop is not the pool, and it is not yet a trainer effect — 2026-09-13

Job 20991975, `stage3_bn_gsm8k`, results `d2fc6ef`. Identical items, exact
McNemar; "gained / lost" counts items the later arm got right and wrong
relative to the earlier one.

| benchmark | dcl | gsm8k | mm30000 | gsm8k − dcl | mm30000 − gsm8k |
|---|---|---|---|---|---|
| MGSM (n=250) | 62.00 | 39.20 | 35.20 | −22.80, 8 / 65, p = 3.2e-12 | −4.00, 19 / 29, p = 0.19 |
| MSVAMP (n=1000) | 64.50 | 54.40 | 50.70 | −10.10, 65 / 166, p = 2.3e-11 | −3.70, 78 / 115, p = 0.009 |
| xGQA full (n=12,578) | 47.66 | 47.62 | 47.54 | −0.04, 858 / 863, p = 0.92 | −0.09, 242 / 253, p = 0.65 |
| xGQA blind | 30.83 | 30.55 | 30.10 | −0.28, p = 0.36 | −0.45, p = 0.016 |

Under the rule fixed on 09-12, both differences are significant on MSVAMP and
both are reported with their sizes; on MGSM only gsm8k − dcl is. **Most of
D13's MGSM drop (−26.8) is the gsm8k − dcl term (−22.8).** The MetaMathQA
sample adds −3.7 on MSVAMP, significant, and −4.0 on MGSM, not significant, and
it changes the output style: the median answer doubles in length and 27% of
its MGSM outputs run on for more than 40 characters after the stated answer,
against 2 to 4% for dcl and gsm8k.

**The gsm8k − dcl term is not yet a trainer effect.** The 09-12 rule attributed
it to "the trainer rewrite or transformers 5.x", and the 09-12 correction
entry stated that MGSM and MSVAMP carry no evaluation confound because they
have no images. That statement went further than the evidence and is withdrawn
in that form. The gray parity cells show that choice scoring reproduces, and
xGQA that greedy answers of one to three tokens reproduce; neither tests long
chain-of-thought generation, and `model.generate` changed on 09-09. It used to
slice `ids[:, prefix_len:]` whenever the returned sequence was longer than the
prefix, and now decodes the whole returned sequence. `evaluate_text.py` itself
is unchanged since the freeze.

What the static evidence says, before the deciding run:
- Since dcl's pilot (`b45552a`, 08-27) the stage-3 trainer changed on 09-05
  (`--no-vision`, offline flags) and was rewritten on 09-09. Replay dataset,
  collate, label construction (`replay_max_gen_len` 512 with EOS), loss scaling
  and every relevant default (replay every 3, 10,000-row cap, `max_gen_len` 64,
  `max_seq_len` 512) are the same. What differs: a deterministic replay
  schedule instead of a shuffled, cycled loader; gradient accumulation per
  epoch with a normalised partial final step, where the old counter ran across
  epochs; deterministic CUDA algorithms with TF32 off; and the stage-2 warm
  start, which now loads only `mapping_vis`, whereas the dcl-era trainer loaded
  every branch present in each checkpoint, stage 2 after stage 1.
- The gsm8k outputs read like a weaker model rather than a decoding artifact:
  coherent, about as long as dcl's (median 280 against 228 characters), stating
  an answer about as often (67% against 72%), rarely running on after it (3.6%
  against 2.4%), and a tolerant extractor keeps the gap (27.6% against 48.4%
  exact numeric match). The lost items contain reasoning errors.
- The two generation paths are demonstrably not equivalent for long outputs.
  The historical dcl outputs start mid-sentence in 6.8% of MGSM and 5.7% of
  MSVAMP items ("18 a day.\nThe goose earns $18 a day...", "= $0.23328..."),
  against 0.4% for gsm8k and under 1% for mm30000 under the current code: the
  old `ids[:, prefix_len:]` slice was cutting the start of long generations.
  That removes text before the answer rather than the answer itself, so it
  does not by itself explain a lower accuracy, but it shows evaluation code is
  a live candidate and not a formality.

**Deciding run, recorded before it is submitted.** `job-scripts/reeval_text.sh`
re-evaluates `stage3_bn_dcl` on MGSM and MSVAMP with the current code and
environment, as `dcl_tf5`. Two paired exact McNemar tests on each benchmark:
dcl_tf5 against the historical dcl run, and dcl_tf5 against gsm8k.
- **Training**: dcl_tf5 is significantly above gsm8k and not significantly
  below historical dcl, on both benchmarks. Then every checkpoint trained with
  the rewritten trainer carries the loss on reasoning, including `vj`, `v4r`
  and every Block C arm; the pooling verdict is unaffected because it is read
  on VQA with both arms on the same trainer; and **no Block C arm is launched
  until the cause is found**, starting with the stage-2 warm start and the
  replay schedule.
- **Evaluation**: dcl_tf5 is significantly below historical dcl and not
  significantly above gsm8k, on both benchmarks. Then no checkpoint is
  implicated, every MGSM and MSVAMP number produced since 09-09 is re-read, and
  `model.generate` is fixed before any further text evaluation.
- Anything else, including a split between benchmarks, is reported as a split
  with both sizes.

**One candidate excluded before the deciding run returns.** The keys of
`outputs/stage2_dc_llava/mapping/pytorch_model.bin` are `loss`, `mapping_vis`
and `step`: the stage-2 checkpoint has no text branch. The dcl-era trainer's
habit of loading every branch present in each warm-start checkpoint therefore
loaded nothing into `mapping_txt` from stage 2, and that warm-start difference
cannot explain the gsm8k − dcl gap. Also confirmed in production: Block D's
allocation (job 21009129) started from the detached worktree of its submitted
commit, `/scratch/santimn/s1_worktrees/807a08e...`.

### Deciding run: the reasoning loss is in training, not in evaluation — 2026-09-13

Job 21009320 re-evaluated `stage3_bn_dcl` on MGSM and MSVAMP with the current
code and environment (results `9efea67`, tag `dcl_tf5`). Identical items,
exact McNemar, "gained / lost" relative to the reference named second.

| benchmark | dcl historical | dcl_tf5 | gsm8k | dcl_tf5 − historical | dcl_tf5 − gsm8k |
|---|---|---|---|---|---|
| MGSM (n=250) | 62.00 | 60.80 | 39.20 | −1.20, 3 / 6, p = 0.51 | +21.60, 63 / 9, p = 4.2e-11 |
| MSVAMP (n=1000) | 64.50 | 65.00 | 54.40 | +0.50, 7 / 2, p = 0.18 | +10.60, 166 / 60, p = 1.1e-12 |

The extracted answer is identical between the historical and current
evaluation on 234 of 250 MGSM items and 977 of 1,000 MSVAMP items.

**Verdict under the rule fixed before this run: training.** On both
benchmarks dcl_tf5 is significantly above gsm8k and not significantly below
the historical dcl run. The current evaluation code reproduces dcl's reasoning
scores, including long chain-of-thought generation; the start-of-text slice
of the old `generate` changed how some outputs read, not what they answered.
The 22.8-point MGSM and 10.1-point MSVAMP gap between `stage3_bn_gsm8k` and
`stage3_bn_dcl` comes from how the checkpoint was trained.

Consequences, as pre-declared:
- **Every checkpoint trained with the rewritten trainer carries this loss on
  reasoning**: `stage3_bn_gsm8k`, `stage3_bn_mm30000`, the pooled round `vj`,
  the independent round `v4r` now training, and any Block C arm. Their MGSM and
  MSVAMP numbers are not comparable with any pre-09-09 checkpoint.
- **The pooling verdict is unaffected**: it is read on VQA, both arms share the
  trainer, and xGQA reproduced dcl within 0.04 points.
- **No Block C arm is launched until the cause is found and fixed.** Block C
  has no replay by design, but the cause is not yet known to be confined to
  replay.
- **D13 re-read**: the pool effect is the mm30000 − gsm8k row of the previous
  entry, −4.0 MGSM (not significant) and −3.7 MSVAMP (significant), plus a more
  verbose output style. The −26.8 first reported was mostly the trainer.
- Candidate already excluded: the stage-2 warm start (the checkpoint has no
  text branch). Next, the remaining training differences between the trainer
  at `b45552a` and at `d224c15`.

**Where the training loss can come from, and the run that splits it
(recorded before launch).** Read side by side, the trainer at `b45552a`
(dcl's pilot) and at `d224c15` run the same training loop: plain AdamW at the
same learning rate, no scheduler, no gradient clipping, no autocast, one replay
batch every three VQA batches, the same accumulation cadence, and the best
checkpoint chosen by VQA validation loss. What differs is small on paper: the
old replay stream was `itertools.cycle` over a shuffled loader, which repeats
its first shuffled pass in the same order forever, while the new one reshuffles
deterministically each cycle; the validation split is a different permutation;
and the new trainer turns on deterministic CUDA algorithms and turns TF32 off.
The stage-2 warm start is already excluded. None of these is an obvious
23-point cause, so the next step is an experiment rather than more reading.

`job-scripts/bisect_trainer.sh` trains Bengali stage 3 with the GSM8K replay
using the trainer at `287bae9`, the last commit before the rewrite, from a
detached worktree so its `model.py` and `common.py` come with it, in the
current venv, with the exact arguments of `stage3_bn_gsm8k`; it evaluates MGSM
and MSVAMP with the current code, which reproduces dcl. Tag `gsm8k_old287bae9`.
- **Near dcl** (not significantly below 62.0 / 64.5 and significantly above
  gsm8k, paired exact McNemar, both benchmarks): the rewrite causes the loss,
  and the next step bisects the 09-09 changes inside the trainer.
- **Near gsm8k** (significantly below dcl and not significantly above gsm8k,
  both benchmarks): training under the current environment causes the loss,
  whatever the code, and the next step is the environment (torch 2.13 and
  transformers 5.13.1 during training, where inference was shown to reproduce).
- Anything else is reported as a split with both sizes.

One operational caveat: the old `common.py` calls `torch.load` without
`weights_only`, which defaults to `True` in the current torch. Mapping
checkpoints load under that default; a resumed `training_state.pt` may not.
The job requests 12 h for a training of about 4 h, so a resume is not expected.

**Training logs, read before the bisect returns.** dcl's pilot (job 19754253,
old trainer) loaded 7,473 GSM8K and 10,000 translation replay rows, one replay
batch per three VQA batches, VQA validation loss 0.693 then 0.617, epoch 1
kept, and a final running replay loss of 1.750. The GSM8K control (job
20991975, rewritten trainer) reached VQA validation loss 0.681 then 0.639,
comparable, as xGQA's reproduction implies. It took **1 h 50 per epoch against
1 h 17**, 43% slower, which fits deterministic CUDA algorithms with TF32 off
and a replay schedule that reshuffles all 17,473 rows at every replay step;
that is a lead, not a cause. The rewritten trainer logs neither its replay row
counts nor its replay loss, so the most direct diagnostic is unavailable for
the runs already made. It is not added now: `v4r` is training from the main
clone, and editing the trainer would change the implementation fingerprint its
resume checks. It is added before any bisection inside the rewrite.

### Block D exploratory result: not promising, confirmatory panel not launched — 2026-09-13

Job 21009129 (112 cells, 3 h 08), analysis `4430465`: `audits/s1_D_input.json`
and `audits/s1_D_analysis.json`. P_current, seven donors on jv/mn/ga/si, 1,160
items on 547 images, paired image-cluster bootstrap over target × subset
strata, 2,000 resamples. Damage ranking as committed in `5befc65` before the
run.

| donor | damage | α on Δ_ground | α on utility |
|---|---|---|---|
| zh | −0.01190 | +0.69 | +1.21 |
| de | −0.01005 | −0.48 | −0.42 |
| pt | −0.00910 | +0.30 | −0.54 |
| id | −0.00520 | +0.86 | +0.71 |
| ru | −0.00490 | −1.41 | −0.98 |
| ko | −0.00265 | +0.19 | +0.24 |
| bn | +0.00240 | −0.15 | −0.23 |

- Spearman(damage, α) **−0.32**, in the predicted direction; exact one-sided
  permutation **p = 0.249**.
- Selected donor **zh**. Grounding regret **+0.17**, UB95 +1.98. Utility
  regret **0.00**, UB95 +1.26. Selected donor's pooled Δ_ground **+7.70**,
  LB95 +5.85.
- G3's four conditions, applied here only as the exploratory readout: p < 0.10
  fails; UB95 regret ≤ 1.0 fails; LB95 pooled grounding > 0 passes; UB95
  utility regret ≤ δ_U fails. `g3: fail` on P_current.

**Under the rule pre-declared on 09-08, the result is not promising**: that
rule required a defined p < 0.10 and a selected-donor regret point estimate of
at most 1.0. The regret point estimate passes (0.17) and p does not (0.249).
Under Option 1 the confirmatory twelve-unit panel is therefore **not
launched**, and G3 is reported as **not tested**, not as refuted: the
confirmatory test never ran. What the exploratory panel says is recorded as
measured: the least-damaged donor was a good choice here, its regret interval
is too wide to certify it, and damage does not rank the seven donors well
enough to reach p < 0.10. With seven donors the permutation test is coarse,
and `power_sim.py` gave it power of 0.54 to 1.00 across the simulated noise
levels, so this is weak evidence against the predictor, not strong evidence.
Four of the seven donors shaped the hypothesis, which is why this panel was
exploratory from the start.

### D12 reading: endpoint made explicit, and the analysis written before the data — 2026-09-13

The reading fixed on 09-12 named its thresholds ("lift: CVQA pooled jv/mn/ga,
LB5 > 0"; "flat: xGQA pooled de/ru/zh within ±1.0") but not the endpoint.
Fixed now, while `v4r_tf5` does not yet exist and `vj_tf5` has not been read:
**the primary endpoint is utility U, accuracy with the correct image**, which
is what D12 registered through `gap_report.py`. `evaluate_all.sh` scores the
correct image and the gray canvas only, so Δ_gray is reported descriptively
and Δ_ground does not exist for these runs. Nothing else changes: joint minus
independent, item-micro, paired on identical items, image-cluster bootstrap
with CVQA stratified by target and xGQA's three translations resampled as one
stratum of shared images; supported only if lift and flat both hold; refuted
if the lift fails or de/ru/zh lift by at least as much as jv/mn/ga; otherwise
inconclusive. `analysis/pooling_d12.py` implements exactly this, with
`analysis/test_pooling_d12.py` covering all four outcomes and the pairing
aborts.

### Bisect result: most of the reasoning loss is not the trainer rewrite — 2026-09-14

Job 21023176 (3 h 25), `stage3_bn_gsm8k_old287bae9`, results `d4dab51`: the
trainer at `287bae9` with its own `model.py` and `common.py`, trained in the
current environment with the arguments of the GSM8K control, evaluated with the
current code. Identical items, exact McNemar, "gained / lost" relative to the
reference named second.

| benchmark | dcl historical | dcl_tf5 | old trainer today | new trainer (gsm8k) | old − dcl | old − new |
|---|---|---|---|---|---|---|
| MGSM (n=250) | 62.00 | 60.80 | 43.60 | 39.20 | −18.40, 16 / 62, p = 1.5e-07 | +4.40, 27 / 16, p = 0.13 |
| MSVAMP (n=1000) | 64.50 | 65.00 | 58.00 | 54.40 | −6.50, 78 / 143, p = 1.5e-05 | +3.60, 95 / 59, p = 0.005 |

**Under the rule fixed before launch, this is a split.** MGSM meets "near
gsm8k": significantly below dcl and not significantly above the new trainer.
MSVAMP is significantly below dcl but significantly above the new trainer.
Both sizes are reported and no single cause is named:
- **The pre-rewrite code, trained today, already loses 18.4 of the 22.8 MGSM
  points and 6.5 of the 10.1 MSVAMP points.** The rewrite adds the rest,
  +4.4 on MGSM (not significant) and +3.6 on MSVAMP (significant), for which
  the 43% slowdown and the missing replay logging stay as leads.
- **The larger part appears whenever the dcl recipe is trained today, with
  either code.** Two explanations remain and this run cannot separate them:
  the training environment (torch 2.13 and transformers 5.13.1 during
  training, where inference was shown to reproduce exactly), or training
  variance. dcl is a single training run, no stage-3 recipe in this project has
  seed replicates, and reasoning scores have moved by more than 20 points with
  recipe changes before (D6, D11). Even the variance explanation would be a
  finding: it would mean a single training run cannot certify a reasoning
  number for this system.
- **Separating them costs one more training of about 3.5 h**: the same old
  trainer in the current environment with a second seed. Not launched here.

**Block C.** The block recorded on 09-13 was "until the cause is found and
fixed". That condition is not met: the cause is only partly located. Lifting
the block is a decision, recorded as a decision, not a consequence of this run.
What bears on it: Block C has no replay, is read on VQA transfer, and VQA
reproduced dcl within 0.1 point in every rewritten-trainer run compared with it.

### D12 pooling result: refuted — 2026-09-14

`analysis/pooling_d12.py` on `vj_tf5` (joint stage-1 mapping) minus `v4r_tf5`
(each language's own stage-1 mapping); both rounds trained by the same
rewritten trainer and scored by one launcher, one tag, one image copy. Report
`audits/d12_pooling_analysis.json`. Primary endpoint utility, as fixed on
09-13; bounds one-sided, image-cluster bootstrap, 4,000 resamples.

| panel | U, joint − independent | LB5 / UB95 |
|---|---|---|
| **lift**: CVQA jv/mn/ga pooled, 935 items | −1.18 | −2.99 / +0.65 |
| **flat**: xGQA de/ru/zh pooled, 37,734 items | −0.19 | −0.48 / +0.10 |

**Verdict: refuted.** The lift fails, so the registered differential prediction
fails; the flat half holds. Descriptive, not part of the verdict: Javanese is
significantly *worse* under the joint mapping, −4.04 [LB5 −7.09, UB95 −1.01];
Mongolian −0.64 and Irish +0.92 cross zero; CVQA ru/zh *gain* +2.54 [LB5 +0.39];
Δ_gray on jv/mn/ga pooled −2.35 [UB95 −0.10]; on the text benchmarks the joint
round scores higher for de/ru/zh by +1.2 to +5.6 points (n = 250 and 1,000, both
rounds carrying the rewritten trainer's reasoning loss). The `uniform_lift`
flag is also true on point estimates, but the verdict follows from the failed
lift alone.

Reading: sharing one text mapping across eleven languages does not help the
low-resource targets here, costs Javanese, and if anything helps the
higher-resource languages on CVQA, the opposite of what D12 registered. The
caveats stated at registration apply: one seed per language, and a joint stage
1 with about a tenth of each language's own exposure. Closed with evidence: do
not retry the joint stage-1 mapping as a low-resource lever without a new
reason.

### Decisions on 2026-09-14: Block C unblocked, and a seed replicate for the reasoning loss

Taken by Santiago after the bisect split and the D12 refutation.

**1. Block C is unblocked, as a decision.** The condition recorded on 09-13,
"until the cause is found and fixed", is not met: the bisect located only part
of the reasoning loss. The block is lifted anyway, for reasons that belong to
Block C's own design: its arms train without replay, it is read on VQA
transfer (P5–P8, G4, G5), VQA reproduced dcl within 0.1 point in every
rewritten-trainer run compared with it, and all four arms share one trainer
and one environment, so their contrasts are internally paired. What the
decision does not license: **no MGSM or MSVAMP number from a Block C arm is
compared with any earlier checkpoint, or read as a reasoning result**, until
the reasoning loss is explained. Arms: C1, C2, C3 and, because G1-I was
inconclusive, C5 as a single-seed pilot arm, seed 13, through
`s1_train_submit.py`; C4 needs no training and is Block B's `T1_bn__V2` cell.

**2. Seed replicate of the old-trainer run, reading fixed before launch.**
`bisect_trainer.sh` gained `SEED`; the replicate is the same trainer at
`287bae9`, same environment and arguments, seed 13 instead of 42, tag
`gsm8k_old287bae9_s13`. Two paired exact McNemar tests per benchmark: the
replicate against historical dcl, and against the seed-42 old-trainer run.
- **Environment**: on both benchmarks, significantly below dcl and not
  significantly different from the seed-42 run. Two runs of identical code in
  today's environment agree with each other and not with dcl.
- **Variance**: on both benchmarks, significantly different from the seed-42
  run. Two runs of identical code and environment disagree, so a single
  training run does not certify a reasoning number here, dcl's included.
- Anything else is mixed and reported with both sizes. One extra seed is a
  coarse probe: "not significantly different" is weak evidence of agreement,
  and the reading says so.

### Block C evaluation and analysis, written before any Block C checkpoint exists — 2026-09-14

The four trained arms were submitted today (C1 job 21060938, C2 21060942, C3
21060943, C5 21060946, seed 13) with no evaluation grid or analysis in place.
Both now exist, implemented from the frozen Block C text, before any arm has
finished training.

- **Grid** (`s1_plan.py --block C`): C1, C2, C3, C5 and the untrained C4
  (`stage1` text with `stage2_dc_llava` vision) on CVQA jv/mn/ga/si/bn and
  xGQA-bn, correct, three shuffles and gray: **150 cells**. C5 is evaluated
  with its text branch absent. The builder refuses to plan until every trained
  arm has written `complete.json`, because the best checkpoint appears after
  epoch 1 and does not mean the two-epoch run finished. The runner requires a
  passing G0, as for Block B.
- **Analysis** (`analysis/block_c.py`, tests in `analysis/test_block_c.py`):
  primary panel jv/mn/ga pooled micro, both endpoints, paired image-cluster
  bootstrap. P5 = G(C2) − G(C1), P6 = G(C3) − G(C1),
  P7 = (C4 − C3) − (C2 − C1), D8 = G(C1) − G(C5), each reported as written.
  **G4**: LB95(P5 on Δ_ground) > 0 and UB95(U(C1) − U(C2)) < 1.0. **G5**:
  xGQA-bn U(C2) non-inferior to U(C1) at 1.0. **G1-T readout** from D8:
  dispensable if UB95 < 1.0 on both endpoints, needed if LB95 on Δ_ground
  > 1.0, otherwise unresolved, always labelled a single-seed pilot and not
  decidable under Option 1.
- **Sign.** P5 to P7 are candidate minus reference; D8 and G4's utility half
  are reference minus candidate. The code reads each bound from the statistic
  as written. Where it derives a reference-minus-candidate bound from a
  candidate-minus-reference contrast, the equivalence is exact on the same
  resamples: the 95th percentile of X is minus the 5th percentile of −X.
- **As decided this morning**, no MGSM or MSVAMP number from a C arm is read as
  a reasoning result; Block C's frozen evaluation contains none.

### Seed replicate: the reasoning loss is the training environment — 2026-09-15

Job 21060746 (3 h 31), `stage3_bn_gsm8k_old287bae9_s13`, results `01d1479`:
the trainer at `287bae9`, current environment, seed 13, evaluated with the
current code. Identical items, exact McNemar, "gained / lost" relative to the
reference named second.

| benchmark | dcl | old trainer s42 | old trainer s13 | new trainer | s13 − dcl | s13 − s42 | s13 − new |
|---|---|---|---|---|---|---|---|
| MGSM (n=250) | 62.00 | 43.60 | 45.60 | 39.20 | −16.40, 16 / 57, p = 1.5e-06 | +2.00, 19 / 14, p = 0.49 | +6.40, 32 / 16, p = 0.029 |
| MSVAMP (n=1000) | 64.50 | 58.00 | 59.80 | 54.40 | −4.70, 84 / 131, p = 0.0016 | +1.80, 73 / 55, p = 0.13 | +5.40, 102 / 48, p = 1.2e-05 |

**Verdict under the rule fixed before launch: environment.** On both
benchmarks the replicate is significantly below dcl and not significantly
different from the seed-42 run. Two runs of identical code in today's
environment agree with each other and not with dcl. As the rule said, one extra
seed is a coarse probe, and "not significantly different" is weak evidence of
agreement; the verdict is the rule's, with that limit.

What this establishes:
- **Most of the reasoning loss comes from training in the current
  environment**, about 17 MGSM and 6 MSVAMP points averaged over the two old
  seeds, whichever code is used. "Environment" means everything that differs
  between the August training and today with the code held fixed: torch 2.13
  and transformers 5.13.1 at training time above all, since inference was shown
  to reproduce. It is not yet traced to a single component.
- **The 09-09 rewrite adds a smaller loss of its own**, now significant on both
  benchmarks against the replicate: about 5 MGSM and 5 MSVAMP points.
- **VQA is untouched by either**: xGQA reproduced dcl within 0.1 point in every
  run trained in the current environment. Block C, read on VQA, is unaffected,
  and the decision of 09-14 stands.
- **Consequence for the paper's reasoning results.** H1, D6, D9b and D11 rest on
  MGSM and MSVAMP numbers from single training runs in the August environment,
  which today's environment does not reproduce. They stay valid as measured,
  as a set, and cannot be compared with any reasoning number trained since. If
  they enter the paper, the environment is stated with them, and a claim that
  depends on their absolute level needs re-running its comparison inside one
  environment.
- **dcl as a draw.** A high draw under the old environment is not excluded,
  since that environment can no longer be sampled; what is excluded is that
  today's environment produces dcl-level reasoning by seed variation alone.

### S1 Block C: freezing the text mapping does not improve transfer. G4 fails, and the loss gate cannot pass — 2026-09-20

Job 21145732, `s1_eval`, 15 h 10, COMPLETED 2026-09-16; 150 of 150 cells; spec
SHA `3b4faff`; report `Approach2/audits/s1_C_analysis.json`. Primary panel CVQA
jv/mn/ga pooled micro, 935 items over 401 image clusters; xGQA-bn 12,578 items
over 398 clusters; 4,000 paired image-cluster resamples stratified by target
and subset. **One training seed per arm (13)**, so every bound is conditional
on those four trajectories and excludes training variance, as the frozen text
requires.

| arm | stage-3 text | stage-3 vision | CVQA U | CVQA Δ_ground [CI95] | xGQA-bn U | xGQA-bn Δ_ground |
|---|---|---|---|---|---|---|
| C1 | trainable | trainable | 37.86 | +5.10 [+2.94, +7.23] | 46.84 | +18.44 |
| C2 | frozen | trainable | 37.33 | +4.85 [+2.64, +7.05] | 46.96 | +18.10 |
| C3 | trainable | frozen | 40.11 | +6.45 [+4.14, +8.71] | 42.35 | +14.13 |
| C4 | frozen | frozen (no stage 3) | 40.43 | +6.42 [+4.36, +8.55] | 12.29 | +6.05 |
| C5 | absent | trainable | 37.01 | +5.38 [+3.20, +7.61] | 48.15 | +19.48 |

| contrast | endpoint | CVQA jv/mn/ga [CI95] | xGQA-bn U [CI95] |
|---|---|---|---|
| P5 = C2 − C1 | Δ_ground | **−0.25 [−2.04, +1.49]**, LB5 −1.74, UB95 +1.23 | — |
| P5 = C2 − C1 | U | −0.53 [−2.13, +0.99] | **+0.13 [−0.40, +0.64]** |
| P6 = C3 − C1 | Δ_ground | +1.35 [−1.03, +3.81] | — |
| P6 = C3 − C1 | U | +2.25 [−0.21, +4.69] | **−4.48 [−5.36, −3.55]** |
| P7 = (C4−C3) − (C2−C1) | Δ_ground | +0.21 [−2.73, +3.13] | — |
| P7 | U | +0.86 [−2.13, +3.86] | −30.19 [−31.68, −28.69] |
| D8 = C1 − C5 | Δ_ground | −0.29 [−2.70, +2.01] | — |
| D8 | U | +0.86 [−1.63, +3.28], UB95 +2.95 | −1.31 [−2.04, −0.62] |

**G4 fails, on both halves.** The rule was LB95(Δ_ground(C2) − Δ_ground(C1)) > 0
**and** UB95(U(C1) − U(C2)) < 1.0. Measured: LB95 = **−1.74** with the point
estimate at **−0.25**, and UB95 on utility = **1.85**.

**G5 passes.** UB95(xGQA-bn U(C1) − U(C2)) = **0.31** < 1.0; the source task is
retained by the freeze, as C2's construction predicts.

**G1-T: unresolved**, and undecidable under Option 1, which defers the three
paired C5 seeds P8 requires. D8's utility UB95 (+2.95) exceeds δ = 1.0, so
"dispensable" cannot be declared; Δ_ground's LB95 (−2.32) is far below δ, so
"needed" cannot either.

What this establishes:

- **Refuted: stage-3 drift of the text mapping is not what limits transfer.**
  The prediction recorded before the run was P5 > 0 on Δ_ground *if that drift
  is causal*. The point estimate is zero (−0.25) and UB95 is +1.23, so a
  benefit larger than 1.23 points is excluded at 95% — a bounded null, not an
  underpowered one, sitting 0.23 above the substantive margin. Freezing
  `mapping_txt` through stage 3 buys jv/mn/ga nothing.
- **Utility retention is not claimed.** The frozen text anticipated that under
  equality the utility half can fail on CVQA power alone, and it did:
  U(C1) − U(C2) = +0.53 with UB95 = 1.85. The paper says "no utility loss above
  1.85 points was detected" and claims no retention. "Not materially inferior"
  is not accepted for G4.
- **C2 is free and useless.** It costs nothing on the source task (G5) and
  gains nothing on the targets (P5). It stays a discussed candidate, never a
  claimed method, exactly as Option 1 pre-declared.
- **The loss gate cannot pass.** G0 pass, G1-I inconclusive, **G2 fail**
  (09-12), **G4 fail**, G5 pass; the gate is the conjunction of all five. R0/R1
  and the confirmatory intervention evals are therefore not authorised. They
  were already future work under Option 1, so the plan does not change: the
  method section is Block B's functional decomposition, with Block C reported
  as an exploratory single-seed pilot.
- **Exploratory, not predicted: the factor that moves is vision, and it trades
  against the source task.** C3, with vision frozen and text trained, sits
  above C1 on CVQA by +2.25 utility [−0.21, +4.69] and +1.35 Δ_ground
  [−1.03, +3.81] — neither excludes zero two-sided, and P6 was registered
  two-sided with no prediction, so no claim is made — while losing **−4.48
  [−5.36, −3.55]** utility on xGQA-bn, which is unambiguous. Freezing the
  vision mapping in stage 3 costs the source task about 4.5 points.
- **Exploratory: the untrained composition is the best arm on the targets.**
  C4 (no stage-3 VQA training at all) scores CVQA U 40.43 and Δ_ground +6.42
  [+4.36, +8.55], above every trained arm, and collapses on the source task
  (xGQA U 12.29 against C1's 46.84). Stage-3 VQA supervision in Bengali buys
  the source task and does not buy target grounding. Two limits on reading it:
  CVQA is multiple-choice, so an untrained arm can still answer while xGQA's
  open-ended format defeats it; and it is one seed, a Block B cell re-read
  here. No trainer boundary is crossed — all five arms share the same August
  `stage1` and `stage2_dc_llava` ancestors, the difference is whether stage 3
  ran today, and VQA was shown unaffected by the environment on 09-15.
- **Exploratory: removing the text branch during training does not hurt, and
  helps the source task.** C5 is indistinguishable from C1 on the primary panel
  and **above** it on xGQA-bn utility (D8 U −1.31 [−2.04, −0.62]); on CVQA-bn
  and si its grounding is higher (D8 Δ_ground −6.99 [−11.95, −2.23] and −4.59
  [−9.68, +0.42]). This echoes Block B, where removing the text branch from the
  id checkpoint raised grounding by +2.85. The point estimates are consistent
  with a branch that is dispensable in training too; the rule's verdict stays
  **unresolved** and neither this nor Block A rewrites G1-I.
- **Not run, and reported as not run**: the three paired C5 seeds of P8, the
  three-seed replication of C1/C2, and matched R0/R1.


### Paper framing changed by Santiago: Approach 1 is the method, S1 becomes the analysis — 2026-09-20

Recorded explicitly because CLAUDE.md forbids moving the frame silently. This
changes **what the paper proposes**, not the S1 question and not any S1 result.

**The decision.** With 22 days to ARR 10-12, the paper leads with Approach 1
(Maryam's NLLB → Qwen3-VL) as the proposed architecture. Approach 2
contributes the protocol-parity baselines, the controlled cross-architecture
comparison, and the analysis. Santiago's reason: he wants to present an
architecture, and Approach 1 is the line showing improvements.

**What S1 becomes.** Its answer stands as recorded — source-conditioned
connector co-adaptation is refuted on every arm tested (G2 fail 09-12, G3 not
promising 09-13, G4 fail 09-20, loss gate does not pass) — and moves from the
paper's spine to its analysis section. Nothing in SCIENCE.md is rewritten: a
question that was answered in the negative stays answered in the negative, and
the blocks are cited as pre-registered controls, which is what they are. They
are ablations and controls, not mechanistic interpretability.

**What cannot be claimed, whichever architecture leads.** Block A's A4 arm —
the English question straight into Gemma's prompt, no NLLB bridge — beats the
full Approach 2 pipeline (CVQA 36.90 vs 35.72, xGQA 50.20 vs 47.66), and A2
without the text branch is non-inferior. No paper of ours claims the NLLB text
bridge does work at inference.

**The number the paper now rests on, and it is not measured yet.** For an
Approach-1-led architecture paper the first question a reviewer asks is the
gain over its own backbone. What the record has, from D8 (2026-08-26, our port
of Maryam's `Baseline/evaluate.py`, reproducing her numbers to 0.04pp):

| | xGQA | CVQA |
|---|---|---|
| Qwen3-VL-8B zero-shot | 53.00 | 40.75 |
| M2RB (Approach 1) | 55.59 | 38.81 |
| v4 (Approach 2, frozen stack + 58M) | 47.73 | 44.02 |

So as last measured, Approach 1 adds **+2.59 on xGQA and −1.94 on CVQA over
its own zero-shot backbone**, while Approach 2 wins CVQA by 3.27 over that
backbone and 5.21 over Approach 1. These are arm-level averages over different
language sets and are not a paired test. **The blocking deliverable is the
paired, per-language, per-item comparison of the three arms with their blind
controls**, which needs Maryam's per-item outputs; ours exist
(`Approach2/results/qwen_{xgqa,cvqa}_<L>[_BLIND].jsonl`, 34 files, full 2×2,
same schema as our evaluator).

**The condition that sends the framing back.** If Approach 1's paired gain over
zero-shot Qwen3-VL is small on xGQA and negative on CVQA once measured on
identical items, "we propose this architecture" is not supportable and the
paper becomes the controlled comparison of the two routes, with each winning a
different benchmark. Recorded now so that outcome is a pre-declared branch and
not a late rescue. The same table decides either way, so no work is lost.

### Cross-architecture comparison, CVQA: the frozen stack beats its own reference VLM on identical items — 2026-09-20

First output of the framing above. `analysis/arch_compare.py` (+9 tests) takes
each arm as a filename template, so an arm from another approach plugs in
without editing it; report `audits/arch_compare_cvqa_qwen_v4.json`. Both arms
here were scored in August on the legacy CVQA image copy under transformers
4.x, so the pair is internally consistent and crosses neither boundary. 10
languages, **2,943 items over 1,370 image clusters**, blind control for every
cell, paired image-cluster bootstrap stratified by language, exact McNemar on
the discordant items.

| language | n | Qwen3-VL-8B zero-shot | v4 (frozen stack + 58M) | Δ full | Qwen dV | v4 dV |
|---|---|---|---|---|---|---|
| bn | 286 | 38.46 | 39.16 | +0.70 | +5.59 | +8.74 |
| ru | 200 | 41.00 | 42.00 | +1.00 | +6.00 | +5.50 |
| zh | 311 | 47.59 | 55.31 | +7.72 | +11.25 | +16.08 |
| pt | 284 | 46.83 | 45.77 | −1.06 | +10.92 | +11.97 |
| id | 412 | 41.50 | 40.29 | −1.21 | +7.04 | +8.74 |
| ko | 290 | 44.14 | 47.59 | +3.45 | +8.28 | +11.03 |
| jv | 297 | 38.72 | 40.07 | +1.35 | +5.39 | +10.10 |
| mn | 312 | 36.86 | 41.35 | +4.49 | +2.24 | +7.69 |
| si | 225 | 39.56 | 47.11 | +7.56 | +6.22 | +15.56 |
| ga | 326 | 32.82 | 35.58 | +2.76 | +6.75 | +7.36 |
| **pooled** | 2,943 | 40.71 | **43.22** | **+2.51 [+0.47, +4.51]** | +7.00 | **+10.23** |

Exact McNemar on full accuracy: 509 items only v4 gets, 435 only Qwen,
**p = 0.0175**. On the visual endpoint the gap is larger than on full accuracy,
**dV +3.23 [+1.23, +5.28]**, so it is not a trained answer-format prior: the
frozen stack extracts more from the image than the VLM it is measured against.
v4 wins 8 of 10 languages. This supersedes D8's "+1.8, pooled p = 0.135,
consistent trend, not significant": that was 10 languages without image
clustering and without the dV contrast, and it understated the effect.

**Consequence for the framing decided today.** The number an Approach-1-led
paper needs is the same table with Approach 1 as a third arm, which requires
Maryam's per-item outputs. As last measured (D8, arm-level averages) Approach 1
scores CVQA 38.81, below the 40.71 its own backbone gets zero-shot here, while
it leads xGQA at 55.59 against our 47.73. If that survives the paired
comparison, the pre-declared branch applies: the paper becomes the controlled
comparison of the two routes, each winning a different benchmark, and the
architecture claim is stated per benchmark rather than in general. xGQA needs
`--image-map` (its ids carry no image) and the script refuses without it, so
the xGQA half waits for a result file carrying `image_id`.
