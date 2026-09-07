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

## 2. The claim chain

1. xGQA calls for more sophisticated methods. *(their words)*
2. Decoupling the bridges answers it: **−38 becomes +0.75**. *(E3)*
3. The mechanism is architectural and checkable in code — `mapping_vis`
   consumes only `encoder_vis(pixel_values)`, no language input, no
   cross-attention with the question (`model.py:325`) — so the visual pathway
   cannot be language-specific. Their misalignment is "latent" because
   language and modality share one pathway; it does not arise when they do
   not.
4. Beyond xGQA's language set the answer becomes conditional on **donor
   choice**. *(X1)*
5. **Open**: can donor quality be predicted label-free, before building
   anything? *(X2b)*

## 3. How we work

These are the rules that produced the corrections in §6. They are not
optional.

- **Pre-register the prediction and its refutation condition** before the run,
  in DESIGN.md. A null is then reportable instead of embarrassing.
- **Say what would refute it, including refutation of the instrument.** X2's
  entry named metric saturation as a failure mode; the metric failed at the
  other end instead. Name both ends.
- **Every VQA eval runs twice** — once with the real image, once with a grey
  384×384 canvas. ΔV = full − blind is the number that matters; raw accuracy
  can be a language prior.
- **Paired tests on identical items** (exact McNemar), never a comparison of
  two accuracies. "A is significant and B is not" is *not* "A > B" — test the
  difference directly and report that p as well.
- **Verify baselines in the upstream source**, never from memory or from a
  derived port. Cost of forgetting this once: a wrong citation of
  `Stage1/train.py:225` as MindMerger, when it is
  `MindMerger/run_training.py:35`.
- **Never ask a summarizer a leading question about a paper.** Pull the
  abstract and read it. One misleading answer already came from this.
- **Record a refutation as a refutation.** Do not soften a failed prediction
  of our own into a partial success.

## 4. Hypothesis ledger

### Supported

| id | claim | evidence |
|---|---|---|
| **E3** | Zero-shot cross-lingual transfer is essentially free in this architecture | 6 unseen languages, n=12,578 each, ΔV **99.4%** of in-language supervised, 6/7 indistinguishable by paired bootstrap, McNemar p < 1e-220 everywhere; absolute accuracy **+0.75** vs source |
| **E2** | The pixels genuinely contribute in a language never seen with an image | CVQA, 9 languages, pooled n=2657, p=**2.0e-13** |
| **H1** | The vision branch is not a tax on the text bridge; better vision helps more | Paired McNemar vs matched v4 arm: MGSM 102-vs-6 (p=**1.3e-23**), MSVAMP 238-vs-53 (p=**3.7e-29**); gap **widens** on the clean subset; monotone across none → weak → strong alignment |
| **X1** | Transferability is a property of the source–target **pair**, not of the target | Indonesian donates to jv/mn/ga at pooled p=**0.00085** where Bengali gives p=0.36; control si significant from all four sources |
| D6 | Reasoning replay in stage 3 | ACCEPTED |
| D9 | DenseConnector multi-layer vision features | +2.97 xGQA, all of it ΔV |
| D9b | Stage-3 epochs 1 → 2 | +2.11 xGQA, +4.8 MSVAMP |
| D11 | Stage-2 scale-up to LLaVA-Pretrain | +1.32 xGQA, **+26.4 MGSM**, +10.3 MSVAMP |

### Refuted

| id | claim | how it died |
|---|---|---|
| **D11-law** | "Gain = 0.71 × deficit" as a general law | Fitted r=+0.989 on bn/de/ru/zh; out of sample on jv/si/ga/mn predicted **+1.8**, observed **−1.4**, RMSE **4.65** against a ±0.4 noise floor. Sharpest miss: Irish MGSM, deficit 9.6, delivered 0. Holds only where the text bridge works |
| **E2-conclusion** | "jv/mn/ga cannot receive visual transfer" | X1: they can, from Indonesian. Refuting a target-intrinsic explanation needs one source that works |
| **sufficiency** | "jv/mn/ga fail because their text bridge is worse" | Every language reaches 77–99% of its own frozen-LLM ceiling, including all three that fail. No relationship with transfer |
| **X2** | *Target-level* alignment predicts transfer | rho=+0.50, permutation p=0.18. Javanese: retrieval@1 **0.960** and 13% retention. *Design also confounded — see §5.* Superseded by X2b, which measures the pair |
| **H1-confound** | "`--no-vision` collapses the model to short VQA-style answers" | Outputs are **longer** (307 vs 242 chars), and the deficit survives cleaning |
| D7 | Zero-init prefix gate | REJECTED as trained (xGQA 19.41) |
| — | "Text-side DenseConnector" is novel | Already published: Puranegedara et al., arXiv 2508.09091 |
| — | "dcl would drift less than dc" | It drifted more |

### Open

| id | question | status |
|---|---|---|
| **X2b** | Does *pairwise* alignment predict pairwise transfer? | **Verdict deferred.** Four checkpoints scored. Corrected per-source reading: id beats bn on alignment and retention **4/4**; within-target Spearman +0.35. Not decidable until ru and zh have own-checkpoint alignment — both pre-registered conditions name them. The earlier "FAILED" came from the wrong bridge and is withdrawn |
| **donor-level** | Does a donor's mean pairwise alignment predict its donor quality? | **Live, and the best candidate for the mechanism.** Stage 3 in Bengali degrades cross-lingual alignment 0.968 → **0.918**; in Indonesian it does not, 0.968 → **0.983**. Donor quality is 34% and 68% respectively. n=2 — an observation. Test across all eleven sources; needs no target-side data at all |
| **X3 / D12** | Does a joint multilingual stage-1 mapping remove donor dependence? | Chained and ready. Pre-registered **differential** prediction: must lift jv/mn/ga and leave de/ru/zh flat. A uniform lift refutes it as surely as no lift |
| **E4** | Why does better visual pretraining buy jv/mn/ga nothing? | Unexplained. Uses each language's own checkpoint, so no donor is involved — a different phenomenon from E2's failure |
| — | Donor matrix beyond 4 sources | Eval only, checkpoints exist |
| — | AlignVLM connector under our frozen setting | Required by the positioning sweep; a reviewer will say the interference is an MLP artifact |

## 5. Corrections and retractions

Kept visible on purpose. Anything here was once written down as true.

- **E3 demoted to "a control"** — retracted 2026-09-06. The reasoning was
  backwards: xGQA's finding is that this transfer is *hard*, so an
  architecture in which the difficulty does not arise is what a call for
  methods asks for. The language-blind visual path is the mechanism, not a
  reason to discount the result.
- **"The question is donor selection"** — retracted the same day. Donor
  selection is a finding *inside* the xGQA question, not a replacement for it.
  Caught by Santiago.
- **"E2 and E4 converge on one mechanism"** — half retracted. E2's failure has
  a donor explanation; E4's does not (own checkpoints, no donor). Two
  different things happening to the same three languages, most likely because
  they are the lowest-resource in the set.
- **X2's design was confounded** — it correlated a target-intrinsic measure
  against a pair-dependent outcome. Its null says nothing about alignment as
  a mechanism, only about that instrument.
- **"Near parity" with the ceiling** — corrected to 87% / 92%; the original
  compared a clean subset against a raw ceiling.
- **MindMerger citation** — `Stage1/train.py:225` was a derived port, not
  MindMerger. Correct: `MindMerger/run_training.py:35`.
- **X2b's first "FAILED" verdict** — withdrawn. Alignment was read from
  `pairalign_stage3_bn_dcl.json` for every source, but transferring from S
  runs **S's** mapping, so S's prefix space is the one the target must land
  in. Only the bn rows were right.
- **A significant global correlation that was entirely target difficulty** —
  with the wrong file, global rho=+0.587 (p=0.0059) while the *within-target*
  correlation was **−0.20**. Easy targets have both high alignment and high
  retention. Always report the within-target figure; `donor_matrix.py` now
  prints both.

## 6. Measurement conventions

- **Noise floors** from same-recipe retrains: ±0.4 on xGQA / MGSM / MSVAMP;
  **±2.8 on CVQA** at n≈286.
- **Never regress on per-language CVQA retention.** n=200–412, and it
  disagrees with xGQA for zh (48% vs 100%), ko (50% vs 91%) and id (67% vs
  99%). Only pooled group results at this scale are solid.
- **Donor quality must be averaged over a common target set.** Averaging each
  source over whatever it happened to be run on ranks Bengali first at 71%
  and inverts X1.
- **Pairwise alignment is read from the source's own checkpoint**
  (`pairalign_stage3_<S>_v4.json`), never from one shared file. Bengali's is
  named `_dcl`, so name resolution must try both.
- **Per-language ceilings, not English and not a mean**, are the denominator:
  frozen Gemma with `--no-mapping` on that language's own questions.
- **The BLIND suffix moves.** Supervised round files are
  `eval_<bench>_<L>_BLIND_<round>`; zero-shot files are
  `eval_<bench>_<L>_zs<src>_BLIND`. Derive neither from the other.
