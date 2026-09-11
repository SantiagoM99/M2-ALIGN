# Approach 2 — Scientific record

The state of the science: the question, every hypothesis, and whether it
stands. `DESIGN.md` is the chronological log with the numbers and the
reasoning; this file is the **current status**, kept short enough to read in
one sitting. When the two disagree, DESIGN.md's entry is the record and this
file is stale — fix it.

---

## 1. The question

Anchored on xGQA (arXiv 2109.06082 — Pfeiffer, Geigle, Kamath, Steitz, Roth,
Vulić, Gurevych). Final sentence of their abstract, verified at source:

> "Our results suggest that simple cross-lingual transfer of multimodal models
> yields latent multilingual multimodal misalignment, **calling for more
> sophisticated methods for vision and multilingual language modeling.**"

**Our question is that call:**

> Can zero-shot cross-lingual transfer for VQA avoid the ~38-point collapse,
> and what makes the misalignment "latent"?

Two halves, and both must be answered: **attribution** (what failed) and
**prediction** (can you tell before you build it).

**Superseded 2026-09-07.** The xGQA call has been answered by the field
(see DESIGN.md "Corrections to the record", item 7). The working question
is now the one in §2; xGQA remains the benchmark, not the question.

## 2. The claim chain

Restated 2026-09-07 after three independent reviews converged; the old
chain is kept in DESIGN.md for the record.

1. Two separately pretrained connectors, multilingual text and vision,
   into a frozen LLM; no target-language multimodal supervision.
2. A checkpoint tuned for VQA in Bengali retains **95.7%** of the
   target-supervised visual contribution across six unseen xGQA languages
   (−1.58 full-accuracy points). The contribution is Δ_gray, correct minus
   gray canvas. Instance-specific grounding **is now measured**: Block A
   gives Δ_ground +19.01 [+18.00, +20.00] on xGQA and +3.89 [+2.24, +5.64]
   on CVQA jv/mn/ga. *(E3, corrected reference; A-ground)*
3. Transfer to the CVQA targets jv/mn/ga varies with the VQA source language;
   S1 does not infer a resource-level effect from a cross-benchmark contrast.
   *(X1, difference-in-differences +3.32 [+0.32, +6.43], stratified image-cluster)*
4. **Under investigation**: does that variation arise from
   source-conditioned connector co-adaptation, and does preserving
   pre-VQA alignment improve transfer? *(DESIGN.md S1, operational gate at
   A+12 days; calendar date depends on Block A's launch)*

What the chain no longer claims: that decoupling caused the improvement
over xGQA's −38; that the visual pathway is language-blind; that the
NLLB bridge is necessary (Block A tests it); that donor quality is
predictable "before building anything".

## 3. How we work

The method and statistics rules live in `CLAUDE.md`; the experimental
contract, endpoints, gates and roadmap live in `DESIGN.md` → S1, **frozen on
2026-09-08** by the commit "docs: freeze S1 v2.1.3 experimental
specification" (its hash is the spec SHA of every S1 manifest). This file
does not duplicate them. One rule is restated because it produced most of
§5: **record a refutation as a refutation**; never soften a failed
prediction of ours into a partial success.

## 4. Hypothesis ledger

### Supported

| id | claim | evidence |
|---|---|---|
| **E3** | Bengali-only VQA supervision transfers to six unseen xGQA languages | vs each target's own `stage3_<L>_v4`: full 48.42 vs 49.99 (−1.58), ΔV 16.73 vs 17.48 (**95.7%**); de/pt/id significantly below, ru/zh/ko not; non-inferiority (post hoc, sensitivity; three regions): at δ=1 ru/zh non-inferior, pt materially inferior, de/id/ko inconclusive. Source-relative 99.4% is secondary. DESIGN 2026-09-07 |
| **E2** | Natural images beat a gray canvas (Δ_gray) in a language never seen with an image; instance-specific grounding untested | CVQA, 9 languages, pooled n=2657, p=**2.0e-13**. The grounding caveat is now closed by A-ground |
| **A-ground** | Instance-specific grounding exists: the correct image beats a *shuffled* image, not only a gray canvas | S1 Block A, job 20674060, arm A1: xGQA pooled bn/de/ko Δ_ground **+19.01 [+18.00, +20.00]**; CVQA pooled jv/mn/ga **+3.89 [+2.24, +5.64]**, LB5 > 0, so **G0 passes**. Per unit the picture is uneven: ga +8.69, jv +4.15, **mn −1.39 [−4.01, +1.23]**, i.e. Mongolian shows none. Validation: A1 on xGQA-bn reproduces the historical `eval_xgqa_bn_dcl` accuracy exactly (47.66), so the rewritten S1 evaluator agrees with the pre-S1 pipeline |
| **H1** | The vision branch is not a tax on the text bridge; better vision helps more | Paired McNemar vs matched v4 arm: MGSM 102-vs-6 (p=**1.3e-23**), MSVAMP 238-vs-53 (p=**3.7e-29**); gap **widens** on the clean subset; monotone across none → weak → strong alignment |
| **X1** | Source choice matters for jv/mn/ga; a target-intrinsic explanation is refuted | Paired difference-in-differences (ΔV_id − ΔV_bn) on identical items, image-cluster bootstrap, pooled jv/mn/ga **+3.32 [+0.32, +6.43]** (stratified by target); per language all three CIs cross zero; control si transfers from every source. Evidence for an Indonesian *advantage* is suggestive, not established |
| D6 | Reasoning replay in stage 3 | ACCEPTED |
| D9 | DenseConnector multi-layer vision features | +2.97 xGQA, all of it ΔV |
| D9b | Stage-3 epochs 1 → 2 | +2.11 xGQA, +4.8 MSVAMP |
| D11 | Stage-2 scale-up to LLaVA-Pretrain | +1.32 xGQA, **+26.4 MGSM**, +10.3 MSVAMP |

### Refuted

| id | claim | how it died |
|---|---|---|
| **D11-law** | "Gain = 0.71 × deficit" as a general law | Fitted r=+0.989 on bn/de/ru/zh, which was two real points (Bengali) plus six near-zero cells; out of sample on jv/si/ga/mn predicted **+1.8**, observed **−1.4**, with only jv MSVAMP (−6.5) beyond binomial noise. **Removed 2026-09-07**, not restated |
| **E2-conclusion** | "jv/mn/ga cannot receive visual transfer" | X1: they can, from Indonesian. Refuting a target-intrinsic explanation needs one source that works |
| **sufficiency** | "jv/mn/ga fail because their text bridge is worse" | Every language reaches 77–99% of its own direct-prompt baseline, including all three that fail. No relationship with transfer |
| **X2** | *Target-level* alignment predicts transfer | rho=+0.50, permutation p=0.18. Javanese: retrieval@1 **0.960** and 13% retention. *Design also confounded — see §5.* Superseded by X2b, which measures the pair |
| **H1-confound** | "`--no-vision` collapses the model to short VQA-style answers" | Outputs are **longer** (307 vs 242 chars), and the deficit survives cleaning |
| D7 | Zero-init prefix gate | REJECTED as trained (xGQA 19.41) |
| — | "Text-side DenseConnector" is novel | Already published: Puranegedara et al., arXiv 2508.09091 |
| — | "dcl would drift less than dc" | It drifted more |
| **text-bridge-necessary** | The NLLB text bridge is needed at inference by a checkpoint trained with it | S1 Block A arm A2 removes the branch and leaves the question in Gemma's own prompt. xGQA pooled: Δ_ground **−0.22 [−0.48, +0.04]**, non-inferior at δ=1; utility −0.82 [−1.09, −0.55], which misses non-inferiority by 0.09 of a point. CVQA pooled: +0.14 [−1.10, +1.44] and −0.11 [−1.29, +1.10]. A4, the English question straight into the prompt with no bridge at all, scores **+36.90** utility on CVQA against A1's +35.72. The frozen rule needs both endpoints non-inferior, so **G1-I is recorded as inconclusive**, not as dispensable; but the bridge's entire inference contribution is bounded under one point. This is an *inference* verdict on a checkpoint trained with the bridge, and says nothing about training (that is C5) |
| **text-bridge-sufficient** | The text bridge carries the question into the LLM | S1 Block A arm A3 removes the question from the prompt and leaves only NLLB to carry it. xGQA utility 47.66 → **11.06**, Δ_ground 19.01 → **2.08**, both materially inferior; CVQA utility −4.39 [−6.31, −2.35], materially inferior. The question reaches the model through its own prompt tokens |

### Open

| id | question | status |
|---|---|---|
| **X2b** | Does *pairwise* alignment predict pairwise transfer? | **Verdict deferred.** Four checkpoints scored. Corrected per-source reading: id beats bn on alignment and retention **4/4**; within-target Spearman +0.35. Not decidable until ru and zh have own-checkpoint alignment — both prospectively recorded conditions name them. The earlier "FAILED" came from the wrong bridge and is withdrawn |
| **donor-level** | Does a donor's stage 1 → stage 3 *damage* (centered margin) predict its donor effect? | **Live, untested, and a separate contribution** (not part of the intervention gate). Only Bengali's damage is measured; `stage3_id_v4` descends from the unscored `stage1_id`, so the earlier "Indonesian preserves alignment" is withdrawn. The seven-donor test on the current targets is exploratory (four donors shaped the hypothesis); the prospective test is the confirmatory panel with the damage ranking committed first. de/pt/ko count as prospective only if the cluster audit shows their results were neither produced, committed nor read |
| **X3 / D12** | Does a joint multilingual stage-1 mapping remove donor dependence? | Exploratory only (S1 v2.1.3): its joint stage 1 has ~1/10 the per-language exposure. Prospectively recorded **differential** prediction: must lift jv/mn/ga and leave de/ru/zh flat. A uniform lift refutes it as surely as no lift |
| **E4** | Why does better visual pretraining buy jv/mn/ga nothing? | Unexplained. Uses each language's own checkpoint, so no donor is involved — a different phenomenon from E2's failure. **Partial evidence 2026-09-11**: Block A shows Mongolian has no instance-specific grounding to improve (Δ_ground −1.39 [−4.01, +1.23]), while Irish has +8.69. For mn the question becomes why grounding is absent, not why it fails to grow |
| **preservation method** | Can an alignment-preservation loss retain transfer and reasoning? | Conditional on the S1 loss gate. It requires matched replay arms R0 (no loss) and R1 (loss), three paired seeds and the confirmatory panel. R1 versus no-replay C1/C2 alone cannot identify a loss effect |
| — | Donor matrix beyond 4 sources | Eval only, checkpoints exist |
| — | AlignVLM connector under our frozen setting | Required by the positioning sweep; a reviewer will say the interference is an MLP artifact |

## 5. Corrections and retractions

Kept visible on purpose; each was once written down as true. Numbers and
reasoning: `DESIGN.md` → "Corrections to the record (2026-09-07)".

- 2026-09-06: E3 demoted to "a control" and reinstated the same day; "the
  question is donor selection" retracted. Both frames were superseded on
  09-07 (§1–2).
- "E2 and E4 converge on one mechanism" — half retracted: two phenomena on
  the same three targets; a resource-level cause was never tested.
- X2 correlated a target-intrinsic measure with a pair-dependent outcome;
  its null says nothing about alignment.
- X2b's first "FAILED" read every source from Bengali's bridge — withdrawn;
  its global correlation (rho +0.587) was target difficulty, within-target
  it was −0.20.
- MindMerger citation: `MindMerger/run_training.py:35`, not a derived port.
- 2026-09-07: E3's reference was Bengali's own arm. Against each target's
  supervised checkpoint retention is 95.7%, not 99.4%; de/pt/id are
  significantly below; "6/7 indistinguishable" and "+0.75" withdrawn.
- X1's p=0.051 was McNemar on full accuracy; the estimand is the paired
  difference-in-differences, +3.32 [+0.32, +6.43] pooled.
- "Indonesian stage 3 preserves alignment 0.968 → 0.983" — withdrawn:
  `stage3_id_v4` descends from the unscored `stage1_id`.
- "±0.4 noise floor" withdrawn (one retrain pair); the D11 "law" removed.
- Wording withdrawn: "no translated multimodal data", "frozen-LLM ceiling",
  "language-blind visual pathway", "label-free before building",
  "pre-registered", "cultural gap", "no instance of two bridges into one
  frozen LLM" (X-LLM), "answers xGQA's call" (answered by the field and by
  our own D8).

## 6. Measurement conventions

Operational conventions live in `CLAUDE.md` (Statistics, Repo conventions)
and the analysis contract in `DESIGN.md` → S1 (Endpoints and statistics).
Two are restated here because the ledger above depends on them:

- **Δ_gray** = correct − gray canvas is *sensitivity*; **Δ_ground** =
  correct − shuffled image is *grounding*; **U** = Acc(correct) is
  *utility*. Every "ΔV" written before 2026-09-07 is Δ_gray, and no result
  in §4 yet demonstrates instance-specific grounding.
- **Non-rejection is not equivalence.** Margins δ_U = δ_G = δ_R = 1.0 point
  are substantive; the three regions (non-inferior, materially inferior,
  inconclusive) apply on every panel; margins chosen after the data are
  sensitivity analyses and are labelled so.
