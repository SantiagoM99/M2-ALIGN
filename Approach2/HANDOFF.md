# Session handoff — 2026-09-06

Written to hand this work to a fresh session. Read `CLAUDE.md` (directives)
and `Approach2/SCIENCE.md` (question + hypothesis ledger) first; this file is
the **operational state** those two do not carry: what ran, what is built but
not launched, and what to do next.

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
Indonesian, and Bengali was chosen by default. Why donor quality varies, and
whether it can be predicted label-free, is the live question.

## 2. What ran, and the results

| exp | question | outcome |
|---|---|---|
| **E3** | Does zero-shot transfer survive on xGQA? | **Yes.** 7 languages, n=12,578 each. ΔV +15.32 to +17.59 vs supervised bn +16.83 → **99.4% retention**; paired bootstrap says 6/7 indistinguishable, only ko lower (−1.51 [−2.46, −0.56]). McNemar p < 1e-220 everywhere. Absolute accuracy **rises** +0.75 going zero-shot |
| **E4** | Does D11's deficit law hold out of sample? | **No.** Refit on bn/de/ru/zh gives k=0.677, r=+0.989; on jv/si/ga/mn predicted +1.8, observed **−1.4**, RMSE 4.65 vs ±0.4 noise. Pooled paired: v4 **worse** than v3 on the LRLs (p=0.016) |
| **H1** | Is the vision branch a tax on the text bridge? | **No, decisively.** Paired McNemar vs matched v4: MGSM **102-vs-6** (p=1.3e-23), MSVAMP **238-vs-53** (p=3.7e-29). Gap *widens* on the clean subset. The predicted confound (collapse to short answers) is refuted — outputs are *longer* |
| **X1** | Source or target? | **Source.** Indonesian donates to jv/mn/ga at pooled **p=0.00085** (56% retention) where Bengali gives p=0.36 (17%). Control si significant from all four sources. Donor quality: id 68%, zh 55%, ru 35%, bn 34% |
| **X2** | Does *target-level* alignment predict transfer? | Failed, and the design was confounded by X1 — it correlated a target-intrinsic measure against a pair-dependent outcome |
| **X2b** | Does *pairwise* alignment predict transfer? | **Verdict deferred** — the first analysis was wrong (§3). Corrected, id beats bn on alignment and retention 4/4. Needs ru and zh scored |

## 3. Corrections made — do not re-introduce them

1. **E3 was wrongly demoted to "a control".** The reasoning was backwards:
   xGQA's finding is that this transfer is *hard*, so an architecture in which
   the difficulty does not arise is exactly what a call for methods asks for.
   The language-blind visual path is the **mechanism**, not grounds to
   discount the result.
2. **The question is not "donor selection".** That drift was caught by
   Santiago. Donor selection is a finding *inside* the xGQA question. The
   anchor was verified at source (arXiv 2109.06082, final sentence of the
   abstract) — quote it from `SCIENCE.md` §1, do not paraphrase.
3. **X2b's alignment must be read from the SOURCE's own checkpoint.**
   Transferring from S runs S's mapping, so S's prefix space is where the
   target has to land. The first pass read every row out of
   `pairalign_stage3_bn_dcl.json`, which is a different bridge for every
   source but bn. `analysis/donor_matrix.py` now resolves
   `pairalign_stage3_<S>_v4.json` per source and marks fallbacks with `*`.
4. **Always check the within-target correlation, not just the global one.**
   With the wrong file the global correlation looked significant
   (rho=+0.587, p=0.0059) while the within-target one was **−0.20** — the
   entire effect was target difficulty (si has both the highest alignment and
   the highest retention). `donor_matrix.py` now prints both.

Also half-retracted: the claim that E2 and E4 "converge on one mechanism".
E2's failure has a donor explanation; E4's does not (own checkpoints, no
donor). **E4's failure is still unexplained** and is the best unclaimed
thread in the project.

## 4. The live hypothesis — donor-level alignment

This is what X2b's data actually suggests, and it is stronger than the
pair-level version because it needs **no target-side data at all**:

| checkpoint | mean pairwise R@1 | donor quality (X1) |
|---|---|---|
| `stage1` (bn-only, pre-VQA) | 0.968 | — |
| **`stage1_joint`** (D12) | **0.986** | untested |
| `stage3_id_v4` | 0.983 | **68%** |
| `stage3_bn_dcl` | 0.918 | **34%** |

**Stage 3 in Bengali degrades cross-lingual alignment (0.968 → 0.918); stage 3
in Indonesian does not (0.968 → 0.983).** Per-language, bn's stage 3 costs
jv 0.957→0.892, ga 0.953→0.901, mn 0.955→0.906. Concrete mechanism for X1's
donor effect: training VQA in S pulls the shared text mapping toward S, and
how much collateral damage that does is a property of S. **n=2 donors — an
observation, not a test.**

**Recorded prediction**: mean pairwise alignment of `stage3_<S>_v4` predicts
S's donor quality across all eleven sources. Both halves are cheap and one is
already in flight. If it holds, the "can you tell before you build it" half
of the question is answered without any multimodal data or evaluation in any
target language.

**It also re-motivates D12.** `stage1_joint` has the highest mean alignment of
anything measured (0.986) and lifts exactly the languages bn's stage 3 damages
(jv 0.981, ga 0.982, mn 0.977).

## 5. Job state

| job | what | status |
|---|---|---|
| 20398781 | X2b, pairwise FLORES alignment, 4 checkpoints | **done** — `pairalign_*.json` in results |
| 20398782 | donor matrix, `de pt ko si jv mn ga` → `jv mn ga si` | in flight when this was written; check `sacct` |

`evaluation/flores_dev.jsonl` is built (997 sentences × 12 languages).
`stage1_joint` trained and is scored.

Check 20398782 received its env vars — `head -5
Approach2/logs/a2_srcabl_20398782.log` must show
`SOURCES=de pt ko si jv mn ga`. If it shows the default `bn id ru zh`, those
cells already exist, the job skips everything by idempotency and exits having
produced nothing. Resubmit with the assignments and `sbatch` on one line.

## 6. What to run next

```bash
git pull

# Score all eleven stage-3 checkpoints. This decides X2b's pre-registered
# verdict (which names ru and zh) AND tests the donor-level prediction in §4.
CKPTS="stage3_bn_dcl stage1 stage1_joint stage3_de_v4 stage3_ru_v4 stage3_zh_v4 stage3_pt_v4 stage3_id_v4 stage3_ko_v4 stage3_jv_v4 stage3_mn_v4 stage3_si_v4 stage3_ga_v4" \
  DT=/scratch/santimn/datatransfer sbatch --export=ALL Approach2/job-scripts/pair_alignment.sh

# X3 / D12 — no longer gated, see below
DT=/scratch/santimn/datatransfer bash Approach2/job-scripts/launch_joint.sh
```

Already-scored checkpoints are skipped. Read results locally from
`Approach2/results/`:

```bash
python3 ../analysis/donor_matrix.py            # transfer matrix + donor quality + alignment test
python3 ../analysis/source_ablation_report.py  # pooled paired McNemar per source
```

**The X3 gate is void.** It assumed alignment predicts transfer, which is
exactly what is still being decided. X3's own differential prediction stands
independently, so it should simply be run.

## 7. Pre-registered predictions awaiting data

- **X2b (pair level)**: must rank id above bn and ru as a donor for jv/mn/ga,
  **and** reproduce X1's one clean dissociation — zh high for ga (87%
  retention) and low for mn (0%). Not yet decidable: ru and zh have no
  own-checkpoint alignment. The earlier "FAILED" was computed from the wrong
  bridge and is **withdrawn**.
- **Donor level (§4)**: mean pairwise alignment of `stage3_<S>_v4` predicts
  S's donor quality, n=11 sources.
- **X3 / D12**: must lift jv/mn/ga and leave de/ru/zh **flat** (they are at
  97–99% of ceiling). A uniform lift refutes it as surely as no lift. Judge
  with `analysis/gap_report.py v4 vj`, never with the mean.

## 8. Known-open, unowned

- **Why E4's LRL failure happens.** Best unclaimed thread — it uses each
  language's own checkpoint, so no donor is involved.
- The AlignVLM connector under our frozen setting — flagged as required by the
  positioning sweep; a reviewer will say the interference is an MLP artifact.
- The walkthrough artifact for Maryam
  (`https://claude.ai/code/artifact/c00865f5-098b-4d20-bed2-08ca21d25517`)
  still leads with the **old** visual-grounding framing and is now wrong in
  its headline. Needs rewriting against `SCIENCE.md` §2.
- Novelty estimate: **~60%**. E3 answers the anchor paper's call and X1 is a
  real, actionable finding; the mechanism is still open, and §4 is the best
  candidate for it.

## 9. Traps that already cost time

- Cluster Python is **3.11.5**: a backslash inside an f-string expression is a
  `SyntaxError` there and legal on a 3.12+ laptop. Check heredoc Python before
  submitting — caught once just before launch, would have killed a 12h job at
  its final reporting step.
- Harvest **both** `eval_*.summary.json` and `eval_*.jsonl`. Five launchers
  shipped with only the former, which is why no paired test could be run on
  `--no-vision` until the files were re-copied by hand.
- The `BLIND` suffix moves: supervised is `..._<L>_BLIND_<round>`, zero-shot is
  `..._<L>_zs<src>_BLIND`. Deriving one from the other silently returns zero
  rows.
- Bengali's v4 checkpoint is `stage3_bn_dcl`, not `stage3_bn_v4`. Anything
  resolving checkpoint names must try both.
- CVQA is n=200–412 per language with a ±2.8 noise floor on ΔV. Never regress
  on per-language retention; pooled group tests only.
- Donor quality must be averaged over a **common** target set, else Bengali
  ranks first at 71% and inverts X1.
