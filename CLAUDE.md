# M2-ALIGN — working directives

Multilingual multimodal alignment. Active line is **Approach 2** on branch
`approach2-gemma-siglip`.

**Architecture in one line**: frozen NLLB-200-600M (text) + frozen
SigLIP2-so400m-384 (vision) → two trainable MLP mappings (58.1M params) →
frozen Gemma-2-9b-it. Prefix `[BOS] + X_f + b_txt + V_f + b_vis + T`.
Curriculum: stage 1 text-only, stage 2 vision-only, stage 3 joint VQA
warm-started from both. **Nothing in the three frozen models is ever updated.**

## Read these first

| file | what it is |
|---|---|
| `Approach2/SCIENCE.md` | **The question, every hypothesis, and whether it stands.** Start here. Includes retractions |
| `Approach2/DESIGN.md` | Chronological decision log: every decision, why, and the measured effect. The authoritative record. Its last entry, **S1**, is the experimental contract and carries the roadmap, the compute ledger and what to run next |
| `Approach2/audits/` | One record per audited run: evidence, verdict, what remains to check. The S1 freeze commit must include them |
| `Approach2/job-scripts/` | SLURM launchers, one per experiment. Headers state what each tests |
| `Approach2/analysis/` | Stdlib-only report scripts, run locally from `Approach2/results/` |

The question was restated on 2026-09-07 (DESIGN.md "Corrections to the
record"; SCIENCE.md §1–2): it is no longer xGQA's "call for more sophisticated
methods", which the field has answered, but source-conditioned connector
co-adaptation in a factorized frozen-backbone system. Do not restate or replace
it again without saying so explicitly and recording it in both files.

## Scientific method — not optional

- **Record the prediction and its refutation condition in DESIGN.md and commit
  it before the run is submitted**; launchers must refuse a dirty tree or a
  `HEAD` without the freeze commit and must write its hash into the manifest
  (prospective recording, not formal pre-registration; guard and manifest
  are still to be implemented). A null is then a result, not an embarrassment.
- **Name how the instrument could fail, at both ends.** A metric can saturate
  or bottom out; say what each would look like.
- **Every VQA eval feeding a primary contrast runs with the correct image,
  three seeded shuffled images on every panel (the unit is the image, and
  xGQA has only 398) and a grey 384×384 canvas**; secondary controls may be correct + grey, labelled
  Δ_gray-only. Δ_ground = correct − shuffled is the grounding statistic; Δ_gray = correct − grey is sensitivity only; always report utility
  U = Acc(correct) next to either. Raw accuracy can be a language prior.
- **Paired tests on identical items**: exact McNemar for a single pair;
  image-cluster bootstrap, stratified by target, for every CI and whenever an
  image carries several questions. "A is significant, B is not" is *not*
  "A > B" — test the difference and report that p too.
- **Verify baselines in the upstream source** (`~/Projects/NLP/Maryam/MindMerger`,
  `MERLIN`, `~/Projects/NLP/nlp_project`), never from memory or a derived port.
- **Never ask a summarizer a leading question about a paper.** Pull the
  abstract and read it.
- **Record a refutation as a refutation.** Never soften a failed prediction of
  ours into a partial success. Prefer correcting the record over defending it.
- Update `DESIGN.md` with every decision **and its measured effect**, and keep
  `SCIENCE.md`'s ledger in sync.

## Statistics

- **No measured noise floor exists.** Binomial SE alone: ±0.45 xGQA, ±3.0 MGSM
  (n=250), ±1.6 MSVAMP, ±2.5–2.8 CVQA per language. Recipe claims need ≥3
  seeds; item/cluster bootstraps exclude training variance and must say so.
  With three fixed seeds, report every seed plus mean/SD/range and describe
  the cluster CI as conditional on those trajectories, not as a population
  CI over training randomness.
- **Never regress on per-language CVQA retention** (n=200–412; disagrees with
  xGQA for zh, ko, id). Pooled group tests only, at that scale.
- The text reference is each language's own **direct-prompt baseline**
  (`--no-mapping` on its own questions), not English and not a mean. It is not
  a ceiling; do not report "% of ceiling".
- Donor quality is only comparable over a **disjoint donor panel**
  (bn/de/ru/zh/pt/id/ko over jv/mn/ga/si) or a source + target effects model;
  self-cells are skipped, so donors that are also targets break the common set.
- The zero-shot reference is the **target's own supervised checkpoint**, and
  "A significant, B not" is never "A > B": test the difference directly.
- The candidate experimental contract is `Approach2/DESIGN.md` **S1
  v2.1.3**; it is not frozen until its audit, inventory, power and Block-D
  analysis artifacts are committed. Edit the spec first, then the launcher.
- Non-inferiority margins are substantive (δ = 1.0 point), never derived from
  precision; CVQA panels are underpowered for non-inferiority and say so.
- Any preservation-loss arm trained with reasoning replay needs a matched
  replay-only control; comparison only with no-replay arms cannot identify
  the loss effect.

## Cluster (Alliance / Rorqual)

Project `def-annielee`. Repo at `/lustre09/project/6072380/santimn/M2-ALIGN`,
data at `DT=/scratch/santimn/datatransfer`.

- **Compute nodes have no internet.** Set `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and pass `--local-files-only`. This is also why
  harvest blocks `git commit` but never `git push` — pushing is done by hand
  from a login node. Anything needing network (dataset fetchers) runs on a
  login node.
- Modules **before** venv activation:
  `StdEnv/2023 python/3.11.5 cudacore/.12.2.2 arrow/21.0.0`, then
  `source $SCRATCH/venvs/m2-align/bin/activate`.
- **Python on the cluster is 3.11.5.** A backslash inside an f-string
  expression is a `SyntaxError` there and legal on a 3.12+ laptop. Check
  embedded heredoc Python before submitting.
- **Partitions are by walltime** (3/12/24/72/168h). Ask for 12h and chain;
  ask for 3h when the job really is short — it schedules far faster.
- **Minimal concurrent jobs.** Packed or chained is the default.
- **Launchers must be idempotent**: skip work whose output already exists, so
  a re-run only fills gaps.
- SLURM **freezes the batch script at submit time** — a queued job will not
  pick up later edits.
- Harvest **both** `eval_*.summary.json` **and** `eval_*.jsonl`. Without the
  per-item files no paired test can be run; five launchers once shipped with
  only the former.

## Repo conventions

- `evaluation/` is gitignored — **benchmark data never goes into git.**
- Results tables shared with collaborators are written in **English**; prose
  with Santiago is in **Spanish**.
- Analysis scripts are **stdlib only** (no torch/numpy) so they run on a
  laptop against `Approach2/results/`.
- Checkpoints are not interchangeable across `--vis-layers`; every consumer
  must pass the value the checkpoint was trained with (`"9,18,-1"` for v4).
- Bengali's v4 checkpoint is `stage3_bn_dcl`, not `stage3_bn_v4`.
- The `BLIND` suffix moves: supervised is `..._<L>_BLIND_<round>`, zero-shot
  is `..._<L>_zs<src>_BLIND`. Derive neither from the other.
- `stage1` and every `stage1_<L>` predate the gate (commit 4beae75,
  2026-08-24) and carry no `gate` key; a branch loader sets `gate = 1.0` for
  them explicitly and is strict about every other key.
- Every checkpoint with a text branch (stage 1, stage 3) was trained with NLLB
  dropout active (`train_stage3_vqa.py:356`); stage-2 vision-only checkpoints
  never run NLLB. S1 Block C fixes it for its own arms only.
- Pairwise alignment is read from the **source's own** checkpoint
  (`pairalign_stage3_<S>_v4.json`; Bengali's is `_dcl`), and drift against the
  source's own `stage1_<S>`, never against Bengali's stage 1.
- `donor_matrix.py` auto-discovers every `eval_cvqa_*_zs*.jsonl`. After
  pulling `1aafc1a` (job 20398782's results) do **not** run it until
  `analysis/block_d.py` exists and the freeze commit is in `HEAD`.
- Every reported number comes from a versioned script in `analysis/`. Current:
  `_boot.py`, `e3_noninferiority.py`, `x1_did.py`, `test_invariance.py`,
  `test_fail_closed.py`; report scripts run from `Approach2/results/`, the
  fail-closed test from anywhere.

## Security

- **Tokens and keys are credentials.** Never paste one into chat, a repo
  script, or a command line (it lands in `~/.bash_history` and in remote
  URLs) — enter them at the git prompt. `.tokens` stays gitignored;
  `~/.git-credentials` stays `chmod 600`.
- If a data source needs auth (FLORES on HF answers 401), find a no-auth
  canonical source rather than embedding a token.
- Read tar members with `extractfile()`, not `extractall()`.
- The user's email identifies him for authorship only; it goes to no other
  service.
