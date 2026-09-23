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

Approach 1, as last recorded (D8, 2026-08-26, arm-level averages, **not** a
paired test): xGQA 55.59, CVQA 38.81. Against its own zero-shot backbone that
is **+2.59 on xGQA and −1.90 on CVQA**.

**The two benchmarks disagree.** On locally sourced, culture-specific images a
frozen 58M connector on a text-only LLM beats a native VLM; on GQA's Western
images with translated questions it loses, and the whole gap there is vision
extraction (ΔV), not answer format. No arm in this table is better in general,
so the paper states its claim per benchmark.

## 3. A measurement question that has to be settled before any CVQA conclusion

There are two CVQA protocols in this repository:

- **letter multiple choice** — `Baseline/evaluate_vqa.py` and
  `Stage3/evaluate_vqa.py` build `Choices: A. … B. …` and ask the model to
  "answer with the letter", then parse the generated letter (`classify_mc`);
- **answer-choice log-likelihood** — `Approach2/evaluate_cvqa.py` and
  `Approach2/baseline_qwen/evaluate.py` score each choice's likelihood and take
  the argmax; nothing is generated and nothing is parsed.

The letter protocol is strictly harder: it requires letter-to-content binding,
it punishes a model that knows the answer but not the convention, and any
refusal or verbosity is scored wrong. The log-likelihood protocol cannot fail
to produce an answer. **Approach 1's 38.81 was produced by the letter
protocol.** Its comparison against its own backbone is internally matched
(40.75 is letter-MC too), so the −1.90 drop is real. But Approach 1's CVQA
number cannot be compared with Approach 2's 43.22, which is log-likelihood, and
no cross-approach CVQA claim should be made until both are scored the same way.
Re-scoring is inference only.

## 4. How to raise Approach 1's CVQA, cheapest first

Each lever names the measurement that predicts it.

**0. Score CVQA by answer-choice log-likelihood (cost: one inference pass).**
Section 3. This is a measurement fix, not a method change, and it is the first
thing to do because every other lever is read against it.

**1. Select the checkpoint on CVQA, not on xGQA (cost: zero if per-epoch
checkpoints were kept).** Block C, 2026-09-20: C4 — the composition with *no*
stage-3 VQA training at all — reaches Δ_ground +6.42 [+4.36, +8.55] and utility
40.43 on the CVQA targets, against the stage-3-trained C1's +5.10 and 37.86,
while collapsing on the source task (xGQA 12.29 against 46.84). **The paired
C4 − C1 contrast was not among Block C's frozen contrasts and is being computed
post-hoc (DESIGN 2026-09-23); until it lands, the two levels are reported as
levels and no ordering is claimed.** Stage-3 VQA supervision buys the source-style benchmark
and does not buy culture-specific grounding. D9b separately found epochs 1 → 2
worth +2.11 on xGQA. Together they predict that the xGQA-selected checkpoint is
close to the CVQA-worst one: evaluate epoch 1.

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
