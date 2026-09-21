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
4. **Answered in the negative, on every arm S1 tested.** The variation is
   not localised to one connector (Block B: D_T and D_V both positive, their
   difference 0.00, **G2 fails**), a donor's pre-VQA alignment damage does not
   predict its donor effect (Block D: Spearman −0.32, exact one-sided
   p = 0.249), and preserving pre-VQA alignment by freezing the stage-3 text
   mapping does not improve transfer (Block C: P5 on Δ_ground −0.25
   [−2.04, +1.49], UB95 +1.23, **G4 fails**). The S1 loss gate therefore does
   not pass, and no preservation loss is authorised. *(DESIGN.md 2026-09-12,
   09-13, 09-20)*

What the chain no longer claims: that decoupling caused the improvement
over xGQA's −38; that the visual pathway is language-blind; that the
NLLB bridge is necessary (Block A tests it); that donor quality is
predictable "before building anything".

### Paper framing, 2026-09-20

The research question above is unchanged and its answer stands. What changed is
what the **paper** proposes: Santiago decided on 2026-09-20 that Approach 1
(NLLB → Qwen3-VL) leads as the architecture and that S1's blocks become the
analysis section, with Approach 2 supplying protocol-parity baselines and the
controlled comparison. The frame is recorded in DESIGN.md 2026-09-20, including
the number that decision now rests on (Approach 1's paired gain over its own
zero-shot backbone, not yet measured on identical items) and the pre-declared
branch back to a two-route comparison paper if that gain is not there.

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
| **text-bridge-necessary** | The NLLB text bridge is needed at inference by a checkpoint trained with it | S1 Block A arm A2 removes the branch and leaves the question in Gemma's own prompt. xGQA pooled: Δ_ground **−0.22 [−0.48, +0.04]**, non-inferior at δ=1; utility −0.82 [−1.09, −0.55], which misses non-inferiority by 0.09 of a point. CVQA pooled: +0.14 [−1.10, +1.44] and −0.11 [−1.29, +1.10]. A4, the English question straight into the prompt with no bridge at all, scores **+36.90** utility on CVQA against A1's +35.72. The frozen rule needs both endpoints non-inferior, so **G1-I is recorded as inconclusive**, not as dispensable; but the bridge's entire inference contribution is bounded under one point. This is an *inference* verdict on a checkpoint trained with the bridge, and says nothing about training (that is C5). **C5 pilot, 2026-09-20**: an arm trained with no text branch at all is indistinguishable from C1 on the primary panel (D8 Δ_ground −0.29 [−2.70, +2.01], U +0.86 [−1.63, +3.28]) and **better** on the source task (xGQA-bn U −1.31 [−2.04, −0.62]); CVQA-bn and si grounding also favour it. G1-T stays **unresolved** under the frozen rule and undecidable under Option 1 (three paired seeds deferred), but nothing supports "needed in training" |
| **text-bridge-sufficient** | The text bridge carries the question into the LLM | S1 Block A arm A3 removes the question from the prompt and leaves only NLLB to carry it. xGQA utility 47.66 → **11.06**, Δ_ground 19.01 → **2.08**, both materially inferior; CVQA utility −4.39 [−6.31, −2.35], materially inferior. The question reaches the model through its own prompt tokens |
| **P3 / G2** | The source-language difference follows the text branch when trained checkpoints are recombined | S1 Block B, CVQA jv/mn/ga pooled, 935 items: D_T **+3.24** [LB5 +0.72], D_V **+3.24** [LB5 +1.93], D_T − D_V **+0.00** [LB5 −2.75]. G2 required both lower bounds above zero; the second is not. 2026-09-12 |
| **imprint-on-vision** (09-11 reasoning, never a criterion) | "If the text bridge is idle at inference, the source imprint must live on the vision mapping" | D_T's lower bound is +0.72, so the stage-3 text mapping carries a positive difference too. The exploratory reading declared before the analysis, LB95(D_V) > 0 and UB95(D_T − D_V) < 0, failed on UB95 = +2.81. Withdrawn 2026-09-12 |
| **parity-drift cause** (09-11) | CVQA parity drift comes from transformers 5.x changing Gemma 2's numerics | Gray-canvas parity cells reproduce exactly (0 flips, drift 0.000) and xGQA reproduces 47.66; only real CVQA images drift, and the S1 panel re-encodes them (0 of 20 byte-identical, max pixel difference 11–28/255). The cause is the image pixels. 2026-09-12 |
| **text-drift-causal** | Freezing the stage-3 text mapping preserves pre-VQA alignment and improves grounded transfer to jv/mn/ga | S1 Block C, job 21145732, P5 = G(C2) − G(C1) on CVQA jv/mn/ga pooled, 935 items over 401 image clusters: **−0.25 [−2.04, +1.49]**, LB95 −1.74, UB95 +1.23, so a benefit above 1.23 points is excluded and the point estimate is zero. Utility −0.53, UB95(U(C1) − U(C2)) = 1.85, so retention is not claimed either. **G4 fails on both halves**; with G2 already failed the loss gate cannot pass. The freeze does cost nothing on the source task (G5 passes, UB95 0.31), so C2 is free and useless. One seed per arm. 2026-09-20 |
| **X3 / D12** | A joint multilingual stage-1 mapping lifts jv/mn/ga and leaves de/ru/zh flat | Joint minus independent, both rounds on the same trainer and evaluation: CVQA jv/mn/ga utility **−1.18** [LB5 −2.99], so the lift fails; xGQA de/ru/zh −0.19 [−0.48, +0.10], flat. Javanese is significantly worse, −4.04 [UB95 −1.01]; CVQA ru/zh gain +2.54 [LB5 +0.39]. 2026-09-14 |

### Open

| id | question | status |
|---|---|---|
| **X2b** | Does *pairwise* alignment predict pairwise transfer? | **Verdict deferred.** Four checkpoints scored. Corrected per-source reading: id beats bn on alignment and retention **4/4**; within-target Spearman +0.35. Not decidable until ru and zh have own-checkpoint alignment — both prospectively recorded conditions name them. The earlier "FAILED" came from the wrong bridge and is withdrawn |
| **donor-level** | Does a donor's stage 1 → stage 3 *damage* (centered margin) predict its donor effect? | **Live, untested, and a separate contribution** (not part of the intervention gate). Only Bengali's damage is measured; `stage3_id_v4` descends from the unscored `stage1_id`, so the earlier "Indonesian preserves alignment" is withdrawn. The seven-donor test on the current targets is exploratory (four donors shaped the hypothesis); the prospective test is the confirmatory panel with the damage ranking committed first. de/pt/ko count as prospective only if the cluster audit shows their results were neither produced, committed nor read. **Ranking committed 2026-09-13, before any Block D evaluation**: zh, de, pt, id, ru, ko, bn; only bn's stage 3 lowered its alignment, the other six raised it; selected donor zh; spread 0.014 in centered margin with no measured noise floor. **Exploratory result 2026-09-13 (P_current)**: Spearman −0.32, exact one-sided p = 0.249; selected zh, grounding regret +0.17 [UB95 +1.98], utility regret 0.00 [UB95 +1.26]; 1 of 4 G3 conditions. **Not promising under the rule pre-declared 09-08, so the confirmatory panel is not launched under Option 1 and G3 is not tested**. Weak evidence against the predictor at seven donors, not a refutation |
| **X3 / D12** | Does a joint multilingual stage-1 mapping remove donor dependence? | Exploratory only (S1 v2.1.3): its joint stage 1 has ~1/10 the per-language exposure. Prospectively recorded **differential** prediction: must lift jv/mn/ga and leave de/ru/zh flat. A uniform lift refutes it as surely as no lift. **2026-09-12**: the six `vj` stage-3 runs (jv/mn/ga/de/ru/zh) are trained, with the rewritten trainer under transformers 5.x; the v4 arm predates both, so v4 against vj is not a single variable and draws no verdict until at least its evaluations are repeated in the current environment |
| **E4** | Why does better visual pretraining buy jv/mn/ga nothing? | Unexplained. Uses each language's own checkpoint, so no donor is involved — a different phenomenon from E2's failure. **Partial evidence 2026-09-11**: Block A shows Mongolian has no instance-specific grounding to improve (Δ_ground −1.39 [−4.01, +1.23]), while Irish has +8.69. For mn the question becomes why grounding is absent, not why it fails to grow |
| **B-localisation** | Where does the source-language difference live? | Both connectors carry it on the primary panel: D_T and D_V are each +3.24 with lower bounds above zero, and their difference is 0.00 [−2.75, +2.81]. On CVQA-bn it follows vision (D_T − D_V −8.92, UB95 −3.22). P4: grounding null; utility **−1.66** [−3.06, −0.32], so crossed pairs compose slightly better than matched. Removing the text branch from the id checkpoint raises grounding by +2.85 [+0.77, +4.91]. One seed per checkpoint; functional, not causal; Block C is what tests imprinting. **Block C, 2026-09-20**: the imprint is not removable by freezing. Freezing text changes nothing (P5 zero), freezing vision moves CVQA a little and costs xGQA-bn −4.48 utility, the interaction P7 is null (+0.21 [−2.73, +3.13]), and the untrained composition C4 grounds better on the targets (+6.42) than any stage-3-trained arm while collapsing on the source task (xGQA U 12.29 vs 46.84). Stage-3 supervision buys the source task, not target grounding |
| **D13** | Does a larger, more diverse math replay pool (MetaMathQA `GSM_*`, 30k) help reasoning without costing VQA? | **Measured 2026-09-12, not attributable.** Against `stage3_bn_dcl` on identical items: MGSM 62.0 → **35.2** (76 lost / 9 gained, p=2.4e-14), MSVAMP 64.5 → **50.7** (193 / 55, p=3.7e-19), xGQA unchanged (p=0.72). Not an answer-format artifact. Confounded with the 09-09 trainer rewrite and the move to transformers 5.x; needs the GSM8K replay reproduced in the current environment before any claim about the pool. The tag overstates the pool: a 10,000-row cap per replay file (since 2026-08-24) means it trained on 10,000 MetaMathQA rows against 7,473 GSM8K. **Controls launched 2026-09-12**: `stage3_bn_gsm8k` (pool alone) and round `v4r` for D12; readings fixed in DESIGN before launch. **Control result 2026-09-13**: GSM8K replay under the current trainer and environment also falls, gsm8k − dcl −22.8 MGSM (p=3.2e-12) and −10.1 MSVAMP (p=2.3e-11), xGQA unchanged; the pool adds −4.0 MGSM (p=0.19) and −3.7 MSVAMP (p=0.009). Whether the −22.8 is training or evaluation is decided by re-evaluating dcl with the current code, rule fixed in DESIGN before that run. **Deciding run 2026-09-13: training.** dcl re-evaluated with the current code scores 60.8 MGSM / 65.0 MSVAMP, not different from its historical 62.0 / 64.5 (p=0.51, 0.18) and far above gsm8k (p=4e-11, 1e-12). The rewritten stage-3 trainer loses reasoning; every checkpoint it trained is affected on MGSM/MSVAMP and Block C waits for the cause. **Bisect 2026-09-14, a split**: the pre-rewrite trainer trained today scores 43.6 MGSM / 58.0 MSVAMP, significantly below dcl (−18.4, −6.5) and above the new trainer by +4.4 (ns) / +3.6 (p=0.005). Most of the loss appears with either code when the recipe is trained today: training environment or training variance, not yet separated. **Seed replicate 2026-09-15: environment.** Old trainer, seed 13: 45.6 MGSM / 59.8 MSVAMP, significantly below dcl (−16.4, −4.7) and not different from seed 42 (+2.0 p=0.49, +1.8 p=0.13). Training in today's environment loses ~17 MGSM / ~6 MSVAMP; the rewrite adds ~5 / ~5. VQA unaffected. H1, D6, D9b, D11 reasoning numbers are August-environment single runs and not comparable with later training |
| **preservation method** | Can an alignment-preservation loss retain transfer and reasoning? | **Not authorised: the S1 loss gate does not pass** (G2 fail 09-12, G4 fail 09-20). Untested, and the premise it rests on has no support — freezing the text mapping outright, the strongest form of preservation available, changes transfer by zero. R0/R1, their three paired seeds and the confirmatory panel are future work; R1 versus no-replay C1/C2 alone could never identify a loss effect anyway |
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
