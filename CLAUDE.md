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
| `Approach2/DESIGN.md` | Chronological decision log: every decision, why, and the measured effect. The authoritative record |
| `Approach2/job-scripts/` | SLURM launchers, one per experiment. Headers state what each tests |
| `Approach2/analysis/` | Stdlib-only report scripts, run locally from `Approach2/results/` |

The question is anchored on xGQA's literal call for "more sophisticated
methods". Do not restate or replace it without saying so explicitly —
`SCIENCE.md` §5 records the one time that drift happened.

## Scientific method — not optional

- **Pre-register the prediction and its refutation condition in DESIGN.md
  before the run.** A null is then a result, not an embarrassment.
- **Name how the instrument could fail, at both ends.** A metric can saturate
  or bottom out; say what each would look like.
- **Every VQA eval runs twice**, second time with a grey 384×384 canvas.
  ΔV = full − blind is the number. Raw accuracy can be a language prior.
- **Paired tests on identical items** (exact McNemar). "A is significant, B is
  not" is *not* "A > B" — test the difference and report that p too.
- **Verify baselines in the upstream source** (`~/Projects/NLP/Maryam/MindMerger`,
  `MERLIN`, `~/Projects/NLP/nlp_project`), never from memory or a derived port.
- **Never ask a summarizer a leading question about a paper.** Pull the
  abstract and read it.
- **Record a refutation as a refutation.** Never soften a failed prediction of
  ours into a partial success. Prefer correcting the record over defending it.
- Update `DESIGN.md` with every decision **and its measured effect**, and keep
  `SCIENCE.md`'s ledger in sync.

## Statistics

- Noise floors from same-recipe retrains: **±0.4** on xGQA/MGSM/MSVAMP,
  **±2.8 on CVQA** at n≈286.
- **Never regress on per-language CVQA retention** (n=200–412; disagrees with
  xGQA for zh, ko, id). Pooled group tests only, at that scale.
- Denominator is each language's own frozen-LLM ceiling (`--no-mapping` on its
  own questions), not English and not a mean.
- Donor quality is only comparable over a **common target set**.

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
