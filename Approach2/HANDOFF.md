# Session handoff — 2026-09-06

Written to hand this work to a fresh session. Read `CLAUDE.md` (directives)
and `Approach2/SCIENCE.md` (question + hypothesis ledger) first; this file is
the **operational state** those two do not carry: what ran today, what is
built but not launched, and what to do next.

Branch `approach2-gemma-siglip`. Everything below is committed and pushed;
`git log 13f9b26..HEAD` is this session's full trail.

---

## 1. Where the project stands in one paragraph

The question is xGQA's literal call for "more sophisticated methods" for
zero-shot cross-lingual VQA (they report ~−38 accuracy points). **We answer
it**: decoupled text and vision bridges into a frozen Gemma give **+0.75
instead of −38** across six unseen languages, with ΔV at 99.4% of the
in-language supervised arm and six of seven statistically indistinguishable
from it. The mechanism is architectural and verifiable in code — `mapping_vis`
consumes only pixels (`model.py:325`), so the visual pathway cannot be
language-specific, which is *why* xGQA's misalignment was "latent". The
honest limit: beyond xGQA's six mid/high-resource languages, transfer becomes
**donor-dependent** — into jv/mn/ga you retain 17% from Bengali and 56% from
Indonesian, and Bengali was chosen by default. Whether donor quality can be
predicted label-free is the open question.

## 2. What ran today, and the results

| exp | question | outcome |
|---|---|---|
| **E3** | Does zero-shot transfer survive on xGQA? | **Yes.** 7 languages, n=12,578 each. ΔV +15.32 to +17.59 vs supervised bn +16.83 → **99.4% retention**; paired bootstrap says 6/7 indistinguishable, only ko lower (−1.51 [−2.46, −0.56]). McNemar p < 1e-220 everywhere. Absolute accuracy **rises** +0.75 going zero-shot |
| **E4** | Does D11's deficit law hold out of sample? | **No.** Refit on bn/de/ru/zh gives k=0.677, r=+0.989; on jv/si/ga/mn predicted +1.8, observed **−1.4**, RMSE 4.65 vs ±0.4 noise. Pooled paired: v4 **worse** than v3 on the LRLs (p=0.016) |
| **H1** | Is the vision branch a tax on the text bridge? | **No, decisively.** Paired McNemar vs matched v4: MGSM **102-vs-6** (p=1.3e-23), MSVAMP **238-vs-53** (p=3.7e-29). Gap *widens* on the clean subset (+39.1 MGSM, +19.6 MSVAMP). The confound I predicted (collapse to short answers) is refuted — outputs are *longer* |
| **X1** | Source or target? | **Source.** Indonesian donates to jv/mn/ga at pooled **p=0.00085** (56% retention) where Bengali gives p=0.36 (17%). Control si significant from all four sources |
| **X2** | Does target-level alignment predict transfer? | **Failed, and the design was confounded** — see §4 |

## 3. The two corrections made today — do not re-introduce them

1. **E3 was wrongly demoted to "a control".** The reasoning was backwards:
   xGQA's finding is that this transfer is *hard*, so an architecture in which
   the difficulty does not arise is exactly what a call for methods asks for.
   The language-blind visual path is the **mechanism**, not grounds to
   discount the result.
2. **The question is not "donor selection".** That drift was caught by
   Santiago. Donor selection is a finding *inside* the xGQA question. The
   anchor was verified at source (arXiv 2109.06082, final sentence of the
   abstract) — quote it from `SCIENCE.md` §1, do not paraphrase.

Also half-retracted: the claim that E2 and E4 "converge on one mechanism".
E2's failure has a donor explanation; E4's does not (it uses each language's
own checkpoint, so no donor is involved). **E4's failure is still
unexplained** and is the best unclaimed thread in the project.

## 4. Why X2 failed, and what replaces it

X2 scored, per language, how well its text bridge retrieves its own English
translations, then correlated that against retention *from Bengali*.
rho=+0.50, permutation p=0.18. Javanese is the direct counterexample:
retrieval@1 **0.960** and 13% retention.

Three separate problems, all recorded in DESIGN.md:

- **Design**: it correlated a *target-intrinsic* measure against a
  *pair-dependent* outcome. X1 proved the outcome depends on the pair, so the
  experiment could not have worked. Its null says nothing about alignment.
- **Metric**: `margin` was declared primary and is unusable — the prefix space
  is severely anisotropic (*mismatched* sentences sit at cosine **0.987**),
  crushing every margin into 0.007–0.010. retrieval@1 did not saturate
  (0.779–0.996) and is now the analysis default.
- **Reference**: the `llm` reference (prefix vs Gemma's embedding table)
  scored **exactly chance** (retrieval@1 = 0.001 at n=1000) for all eleven
  languages. The mapping is trained so the LLM can *read* the prefix through
  attention, not to match embedding geometry. Dropped.

**X2b** replaces it: pairwise alignment on FLORES-200, source↔target on the
same sentence, mean-centered per language. Built and verified, not yet run.

## 5. Built this session but NOT yet run

| file | status |
|---|---|
| `fetch_flores.py` | **Verified working** — ran it locally, 997 sentences × 12 languages, alignment spot-checked (row 0 is the same sentence in all twelve). HF mirrors answer 401 without a token, so it pulls the no-auth NLLB tarball. Must run on a **login node** |
| `pair_alignment.py`, `job-scripts/pair_alignment.sh` | X2b. Not run. Needs `evaluation/flores_dev.jsonl` first |
| `analysis/donor_matrix.py` | Works on today's data. Note the trap it fixes: donor quality **must** average over a common target set, else Bengali ranks first at 71% and inverts X1 |
| `job-scripts/source_ablation.sh` | Ran with 4 sources; the remaining 7 are eval-only |
| `launch_joint.sh` (X3/D12) | `train_stage1_joint` has run; the chain to eleven stage-3s has not been launched |

## 6. In flight as of the handoff (2026-09-06)

| job | what | walltime |
|---|---|---|
| **20398781** | X2b, pairwise FLORES alignment — also scores `stage1_joint` | 3h |
| **20398782** | donor matrix, sources `de pt ko si jv mn ga` into `jv mn ga si` | 12h |

`evaluation/flores_dev.jsonl` is built (997 sentences x 12 languages).

Check 20398782 actually received its env vars — `head -5
Approach2/logs/a2_srcabl_20398782.log` must show
`SOURCES=de pt ko si jv mn ga`. If it shows the default `bn id ru zh`, those
cells already exist, the job skips everything by idempotency and exits in
minutes having produced nothing. Resubmit with the assignments and `sbatch`
on one line.

**X3 is deliberately NOT launched** — it is gated on 20398781's verdict.

## 7. What to run next, in order

```bash
git pull

# 1. LOGIN NODE (compute nodes have no network)
python3 Approach2/fetch_flores.py --out-dir evaluation

# 2. X2b — 3h partition, also scores stage1_joint and so prices D12
DT=/scratch/santimn/datatransfer sbatch --export=ALL Approach2/job-scripts/pair_alignment.sh

# 3. Finish the donor matrix — evaluation only, no training
SOURCES="de pt ko si jv mn ga" TARGETS="jv mn ga si" \
  DT=/scratch/santimn/datatransfer sbatch --export=ALL Approach2/job-scripts/source_ablation.sh

# 4. X3 — only after reading step 2's verdict on stage1_joint
DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_joint.sh
```

Read results locally, from `Approach2/results/`:

```bash
python3 ../analysis/donor_matrix.py
python3 ../analysis/source_ablation_report.py
python3 ../analysis/alignment_vs_transfer.py
```

**Step 4 is gated on step 2 on purpose.** `launch_joint.sh` submits eleven
stage-3 runs. If the joint mapping does not pull the languages' prefixes
together in the pairwise score, it will not fix transfer either, and the
eleven runs are wasted.

## 8. Pre-registered predictions awaiting data

- **X2b**: the pair score must rank id above bn and ru as a donor for
  jv/mn/ga, **and** reproduce X1's one clean dissociation — zh high for ga
  (87% retention) and low for mn (0%). Missing that dissociation ⇒ abandon the
  alignment mechanism rather than re-instrument a third time.
- **X3/D12**: must lift jv/mn/ga and leave de/ru/zh **flat** (they are at
  97–99% of ceiling). A uniform lift refutes it as surely as no lift. Judge
  with `analysis/gap_report.py v4 vj`, never with the mean.

## 9. Known-open, unowned

- **Why E4's LRL failure happens.** Best unclaimed thread.
- The AlignVLM connector under our frozen setting — flagged as required by the
  positioning sweep; a reviewer will say the interference is an MLP artifact.
- The walkthrough artifact for Maryam
  (`https://claude.ai/code/artifact/c00865f5-098b-4d20-bed2-08ca21d25517`)
  still leads with the **old** visual-grounding framing and is now wrong in
  its headline. Needs rewriting against `SCIENCE.md` §2.
- Novelty estimate as of today: **~60%**. E3 answers the anchor paper's call
  and X1 is a real, actionable finding; the mechanism (why donor quality
  varies) is still missing, and that is what X2b and X3 are for.

## 10. Traps that already cost time

- Cluster Python is **3.11.5**: a backslash inside an f-string expression is a
  `SyntaxError` there and legal on a 3.12+ laptop. Check heredoc Python before
  submitting — this was caught once before launch and would have killed a 12h
  job at its final reporting step.
- Harvest **both** `eval_*.summary.json` and `eval_*.jsonl`. Five launchers
  shipped with only the former, which is why no paired test could be run on
  `--no-vision` until the files were re-copied by hand.
- The `BLIND` suffix moves: supervised is `..._<L>_BLIND_<round>`, zero-shot is
  `..._<L>_zs<src>_BLIND`. Deriving one from the other silently returns zero
  rows.
- Bengali's v4 checkpoint is `stage3_bn_dcl`, not `stage3_bn_v4`.
- CVQA is n=200–412 per language with a ±2.8 noise floor on ΔV. Never regress
  on per-language retention; pooled group tests only.
