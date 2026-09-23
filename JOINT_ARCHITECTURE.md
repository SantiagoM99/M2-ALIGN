# Joint architecture: merging Approach 1 and Approach 2

Written 2026-09-20, for the ARR 10-12 submission. Numbers here come from
`Approach2/DESIGN.md` and `Approach2/audits/`; each claim names its source.
Approach 1's code is read from this tree, whose `Stage1–3` are current to
**2026-08-11** — the September improvements are not in it, so everything said
about Approach 1's current configuration has to be confirmed against Maryam's
tree before it goes in the paper.

## 1. The two systems are the same shape, and neither trains its backbone

| | Approach 1 (Maryam) | Approach 2 (Santiago) |
|---|---|---|
| text encoder | NLLB-200 / M2M100, **frozen** | NLLB-200-600M, **frozen** |
| visual pathway | Qwen3-VL's native tower + DeepStack, **frozen** | SigLIP2-so400m-384, dense layers 9/18/−1, **frozen** |
| decoder | Qwen3-VL-8B, **frozen** | Gemma-2-9b-it, **frozen** |
| trainable | the `Mapping` (MLP + `end_boundary`) | two MLP mappings, 58.1M |
| question at inference | in Qwen's own prompt **and** through the bridge | in Gemma's own prompt **and** through the bridge |

`Stage3/model.py` sets `requires_grad = False` on the NLLB encoder, on
Qwen3-VL's LLM **and** on its vision tower; there is no LoRA and no unfreezing
anywhere in `Stage1–3`. So "Approach 2 is limited because the backbone cannot
be trained" is not a difference between the two lines: **neither trains a
backbone.** The difference is which backbone, and therefore what the trainable
mapping has to accomplish:

- Approach 1 starts from a model that already grounds images, so its mapping
  only has to deliver multilingual *text*. It inherits Qwen3-VL's visual
  ability for free.
- Approach 2 starts from a text-only LLM, so its mappings have to *create* the
  visual pathway. Everything visual it can do, it built.

That is why the two lines' measured contributions look so different, and why
"more potential" needs saying carefully. Approach 1 has more **absolute
headroom**, because its backbone is stronger and its remaining work is smaller.
Approach 2 has the larger **measured method contribution**, because its
visual pathway is its own: ΔV (accuracy with the image minus a gray canvas) is
+17.39 on xGQA and +10.23 on CVQA, created by 58M trainable parameters on top
of a decoder that cannot see at all.

## 2. What is measured today

Paired, identical items, blind control for every cell, image-cluster bootstrap
stratified by language, exact McNemar (`Approach2/analysis/arch_compare.py`;
reports in `Approach2/audits/arch_compare_*_qwen_v4.json`, DESIGN 2026-09-20).
Both arms in each row were scored by the same code, so each row is internally
matched.

| | Qwen3-VL-8B zero-shot | Approach 2 (v4) | difference |
|---|---|---|---|
| CVQA, 10 languages, 2,943 items, 1,370 image clusters | 40.71 | **43.22** | **+2.51 [+0.47, +4.51]**, p = 0.018 |
| CVQA ΔV | +7.00 | **+10.23** | +3.23 [+1.23, +5.28] |
| xGQA, 7 languages, 88,046 items, 2,786 clusters | **53.00** | 49.66 | **−3.34 [−3.76, −2.92]**, p = 8e-83 |
| xGQA ΔV | **+29.80** | +17.39 | −12.41 [−12.91, −11.91] |

Approach 1, reported 2026-09-23 with baseline and a1 at matched resolution,
CVQA scored open-ended, plus two new references (macro averages, not paired
tests):

| | xGQA, 7 langs | CVQA open, 10 langs |
|---|---|---|
| direct baseline | 51.39 | 39.16 |
| translate-then-test | 51.26 | **42.73** |
| **a1** | **57.67** | 38.53 |
| English (ceiling) | 58.70 | 44.48 |

a1 is **+6.28 over its baseline on xGQA**, +6.41 over translate-then-test, and
1.03 from gold English: 86% of the language gap recovered. On CVQA it is −0.63
against the same baseline and −4.20 against translate-then-test, flat on
jv/si/mn/ga (35.51 against 35.55) with the damage in bn (−3.14) and id (−5.58).
Under the letter protocol a1 is −11.63 below its baseline, which is a capability
of the frozen backbone that the mapping removes.

Putting the two lines side by side, same scoring convention on the open-ended
CVQA rows:

| CVQA open, macro | | xGQA, macro | |
|---|---|---|---|
| **Approach 2 (v4)** | **43.42** | **Approach 1 (a1)** | **57.67** |
| Qwen3-VL zero-shot (our port / hers) | 40.75 / 39.16 | Qwen3-VL zero-shot (our port / hers) | 53.00 / 51.39 |
| Approach 1 (a1) | 38.53 | Approach 2 (v4) | 49.66 |

**a1 owns xGQA, Approach 2 owns CVQA, and both facts have one explanation**
(section 4.1). A fairness point to settle: the baseline was matched *down* to
a1's resolution, and a reviewer will ask whether it was handicapped. We hold the
baseline at the higher resolution (xGQA 53.00, CVQA 40.75); a1 still wins xGQA
by +4.67 there, while CVQA becomes −2.22 instead of −0.63. Reporting both
resolutions is strictly better than reporting one.

**Why translate-then-test is competitive on CVQA and useless on xGQA.** xGQA's
non-English questions are machine translations of English, so TTT is a
translation of a translation and loses 7.44 against gold English; CVQA's
questions are natively authored and its English field is human, so TTT
translates once and lands 1.75 from the ceiling. **xGQA structurally flatters
mapping methods and CVQA does not**, which is why CVQA is this paper's honest
benchmark.

**The two benchmarks disagree.** On locally sourced, culture-specific images a
frozen 58M connector on a text-only LLM beats a native VLM; on GQA's Western
images with translated questions it loses, and the whole gap there is vision
extraction (ΔV), not answer format. No arm in this table is better in general,
so the paper states its claim per benchmark.

## 3. Protocol status

There are two CVQA protocols in this repository, and they do not measure the
same thing:

- **letter multiple choice** — `Baseline/evaluate_vqa.py` and
  `Stage3/evaluate_vqa.py` build `Choices: A. … B. …`, ask for the letter, and
  parse what is generated (`classify_mc`);
- **answer-choice log-likelihood** — `Approach2/evaluate_cvqa.py` and
  `Approach2/baseline_qwen/evaluate.py` score each choice by its
  length-normalized mean per-token log-probability and take the argmax; nothing
  is generated and nothing is parsed.

As of 2026-09-23 Maryam reports both, and her open-ended variant — average
log-probability per option — **is** the log-likelihood convention above. So the
open-ended CVQA rows of the two approaches are directly comparable, and the
concern raised on 09-20 is settled by her change. The letter rows are not
comparable with anything on the Approach 2 side.

Two things about the protocols are worth reporting as findings rather than
footnotes. The baseline scores **+27.73 higher under letters than open-ended**
(66.89 against 39.16), so much of CVQA's apparent difficulty is the protocol,
not the task. And, as Maryam observes, four options plus an image may be
answerable without reading the native question at all — which is precisely the
pathway the method exists to improve. That is measurable: run CVQA with the
question removed and the options kept. Next to the gray-canvas arm this bounds
what the benchmark measures from both sides, one blind arm for the image and one
for the question.

What still differs between the two lines and must be stated with every
cross-approach number: the CVQA image copy, and, for the Qwen arms only, the
resolution.

## 4. How to raise Approach 1's CVQA, cheapest first

Each lever names the measurement that predicts it.

**0. Done, 2026-09-23.** CVQA is now scored open-ended on her side too, so the
measurement question no longer blocks the rest. Its replacement as the cheapest
next measurement is the **question-blind arm** of section 3: image and options
kept, native question removed.

**1. Select the checkpoint on CVQA, not on xGQA (cost: zero if per-epoch
checkpoints were kept).** Block C, with the paired C4 − C1 contrast computed
post-hoc on 2026-09-23: an arm that received **no stage-3 VQA supervision at
all** is statistically indistinguishable from the trained arm on the CVQA
targets (Δ_ground **+1.32 [−1.13, +3.93]**, five per-unit panels all covering
zero) while being **12.39 [11.10, 13.61]** of grounding and **34.54 [33.00,
36.09]** of utility worse on the source benchmark. The supervision's entire
grounding benefit is confined to the benchmark it resembles; on culturally
sourced targets its effect is a null bounded within about ±4 points. It is not
a measured loss — an earlier draft of this document said the untrained arm
grounds the targets *better*, and the paired test does not support that. D9b
separately found epochs 1 → 2 worth +2.11 on xGQA, so more supervision keeps
buying the source-shaped benchmark. Predicted: the xGQA-selected checkpoint is
no better than epoch 1 on CVQA, and possibly worse; checking costs nothing.

**2. Sweep the strength of the bridge prefix at inference (cost: one inference
pass per λ).** Block A: removing the text branch from a checkpoint trained with
it is non-inferior on xGQA grounding (−0.22 [−0.48, +0.04]) and, on the
Indonesian checkpoint, *raises* CVQA grounding by +2.85 [+0.77, +4.91]; A4, the
English question straight into the prompt with no bridge at all, beats the full
pipeline (CVQA 36.90 vs 35.72, xGQA 50.20 vs 47.66). So the bridge may be
costing CVQA while the question is already in the prompt. λ = 0 must reproduce
zero-shot Qwen exactly, which also makes this the sanity check for the whole
pipeline.

**3. Change the donor language of the VQA supervision (cost: one stage-3
retrain).** X1: the source language of the VQA supervision changes transfer to
jv/mn/ga by +3.32 [+0.32, +6.43] on identical items, Indonesian over Bengali.
If Approach 1's stage 3 is supervised in Bengali, this is a measured ~3 points
on exactly the languages CVQA cares about.

**4. Add the SigLIP2 dense visual stream (cost: stage 2 + stage 3 for one new
mapping).** This is the merge proper, section 5. It targets ΔV, which is where
Approach 2's CVQA advantage actually lives (+10.23 against Qwen's +7.00), and
leaves Qwen's native pathway — which wins xGQA by +12.41 ΔV — untouched.

**Excluded as a lever:** prompt wording and the VQA prompt template. D3 already
adopted Maryam's prompt verbatim on the Approach 2 side and our port of her
baseline reproduces her xGQA to 0.04pp, so protocol parity there is not the
problem.

**A caution that applies to every CVQA number.** For Qwen zero-shot, CVQA ΔV is
+7.00 while xGQA ΔV is +29.80: most of CVQA accuracy is available without the
image. So a CVQA gain can come from the language prior rather than from
grounding, and only the blind arm separates them. Every CVQA claim in this
paper is reported with its blind control.

## 5. The merged architecture

One frozen VLM backbone, two trainable prefix mappings, and the backbone's own
visual pathway left intact:

```
question (native language) ──► NLLB-200 encoder (frozen) ──► MLP_txt ──┐
                                                                       │
image ──► SigLIP2-so400m-384, layers 9/18/−1 (frozen) ──► MLP_vis ─────┤──► prefix
                                                                       │
image ──► Qwen3-VL native tower + DeepStack (frozen) ──────────────────┤
                                                                       │
question (native language, in Qwen's own prompt) ─────────────────────-┘
                                          │
                                          ▼
                            Qwen3-VL-8B decoder (frozen)
```

Trainable: `MLP_txt`, `MLP_vis` and their boundary parameters — on the order of
58M, under 1% of the stack. Nothing in the three pretrained models is ever
updated, which is the property both lines already have and the one that makes
the ablations interpretable.

Curriculum, unchanged in shape from both lines: stage 1 text-only alignment,
stage 2 vision-only captions for `MLP_vis` into Qwen's embedding space, stage 3
joint VQA in a donor language, with the checkpoint selected on a
culture-specific dev criterion rather than on xGQA (lever 1).

**Why each stream is in the figure, with the number that put it there.**

| stream | justification |
|---|---|
| Qwen native vision | +29.80 ΔV on xGQA against our 17.39; removing it throws away the reason to use a VLM |
| SigLIP2 dense 9/18/−1 | our CVQA ΔV +10.23 against Qwen's +7.00, and D9's +2.97 xGQA, all of it ΔV |
| NLLB text bridge | Approach 1's +2.59 xGQA over its backbone. Note the bridge's inference contribution is bounded under one point in Approach 2 (Block A), so this stream is on probation and the paper must ablate it |
| question in the prompt | A3: with the question removed from the prompt and left only to the bridge, xGQA utility falls 47.66 → 11.06 |

**What has to be trained, and the honest cost.** `MLP_vis` must be trained
against Qwen's embedding space, so stage 2 cannot be warm-started from the
Gemma-side checkpoints; the ledger budgets about 10 GPU-h per trained arm and
stage 3 measured 3 h 20 for the Block C arms. Stage 1 and stage 2 for a new
decoder are the unmeasured part.

**Risks, named now.** The prefix grows by another visual stream, which costs
context and latency; the two visual streams may be redundant on xGQA and only
complementary on CVQA, which is exactly what the ablation has to show; and the
text bridge may turn out to be idle here as it is in Approach 2, in which case
the honest architecture is "a second visual expert injected into a frozen VLM"
and the paper says so.

## 6. What the paper can claim, and what is future work

Reachable by 10-12 with no new architecture: the controlled comparison of
section 2 with Approach 1 as a third arm, all three scored under one CVQA
protocol, per language, with blind controls; levers 0–2, which are inference
only; lever 3 if one stage-3 retrain fits. That is already a paper: two routes
to multilingual VQA, each winning a different benchmark, with the visual
endpoint isolating why.

The merged architecture of section 5 is the method contribution if its stage 2
and stage 3 land in time, and the natural next paper if they do not. S1's
blocks (`Approach2/DESIGN.md` S1, frozen `3b4faff`) supply the analysis
section: pre-registered controls on where the source-language imprint lives
(both connectors), whether preserving pre-VQA alignment helps (it does not,
G4 fails), and whether donor damage predicts donor quality (it does not).

**Still needed from Maryam**, and the critical path: per-item outputs for her
current system, per language, both benchmarks, with a blind arm; her
per-language item counts and data files; and her current training
configuration and parameter count.
