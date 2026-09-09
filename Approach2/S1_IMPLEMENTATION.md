# Running the frozen S1 protocol

S1's spec SHA is `3b4faff4d673fad94cd3115b3f14212eeca9c12e`.
The code SHA is the clean HEAD used to prepare each submission. The frozen
DESIGN text is unchanged. This implementation has been tested locally on CPU;
no real S1 GPU evaluation or training has been run by this change.

## Local checks

Run with Python 3.11 and the model dependencies installed:

```bash
python -m unittest discover -s Approach2/tests -v
python Approach2/analysis/test_block_a.py
python Approach2/analysis/test_block_d.py
python Approach2/analysis/test_fail_closed.py
python Approach2/analysis/test_cvqa_s1_builder.py
```

The model tests create tiny random Gemma/encoder fixtures in memory. They do
not download checkpoints or run on GPU. They compare the full-prefix path
against `model.py` at the freeze commit, test prompt-only generation, branch
swaps, image permutations, the complete synthetic A matrix, authenticated
results, and interrupted versus uninterrupted stage-3 training. Analysis
scripts remain stdlib-only. The isolated CPU validation environment used
Python 3.11.5, PyTorch 2.6.0 and Transformers 4.51.3; it is not a claim about
the cluster's package versions.

## Capture Rorqual's environment

Santiago will run this from the authenticated login node, at the repo root:

```bash
bash Approach2/job-scripts/capture_environment.sh
```

It loads the documented modules and venv and writes
`Approach2/audits/rorqual_environment.json`. It records installed package
versions, modules and cached model revisions without dumping credentials,
loading models or using GPU. Commit this artifact before preparing a clean
S1 submission. Evaluation manifests additionally record package versions,
resolved frozen-model snapshots and configuration/tokenizer hashes. Mutable
local model directories also get weight-file hashes.

## Prepare data, maps and a plan without GPU

CVQA inputs must come from `build_cvqa_s1.py`: they must carry native `query`,
`english_query`, `image_id`, `subset`, choices and the gold index. Legacy
CVQA JSONLs without subsets are rejected under `--s1`. xGQA inputs must keep
question ids and `image_id` or `vg_image_id`; its English file must cover
exactly the same items and images. The guard rejects a partially matched
English file even if an older builder emitted it.

Create a local panels JSON whose entries point to actual files. Schema:

```json
{
  "xgqa": {
    "bn": {"data": "/absolute/path/xgqa/bn.jsonl", "images": "/absolute/path/gqa/images"},
    "de": {"data": "/absolute/path/xgqa/de.jsonl", "images": "/absolute/path/gqa/images"},
    "ko": {"data": "/absolute/path/xgqa/ko.jsonl", "images": "/absolute/path/gqa/images"},
    "en": {"data": "/absolute/path/xgqa/en.jsonl", "images": "/absolute/path/gqa/images"}
  },
  "cvqa": {
    "jv": {"data": "/absolute/path/cvqa_s1/javanese.jsonl", "images": "/absolute/path/cvqa_s1/images"},
    "mn": {"data": "/absolute/path/cvqa_s1/mongolian.jsonl", "images": "/absolute/path/cvqa_s1/images"},
    "ga": {"data": "/absolute/path/cvqa_s1/irish.jsonl", "images": "/absolute/path/cvqa_s1/images"},
    "si": {"data": "/absolute/path/cvqa_s1/sinhalese.jsonl", "images": "/absolute/path/cvqa_s1/images"}
  }
}
```

Place the filled file under ignored `evaluation/`, then:

```bash
python Approach2/s1_plan.py --block A \
  --panels evaluation/s1_panels.json \
  --checkpoints "$PWD/Approach2/outputs" \
  --results "$PWD/Approach2/outputs/s1_A" \
  --output evaluation/s1_A.plan.json
```

This creates **156 cells** (60 xGQA + 96 CVQA), ignored permutation maps in
`evaluation/`, and hash registries in `Approach2/shuffle/*.hashes.json`.
Commit the registries before submission. Each registry identifies the item
universe and all three deterministic maps; submit and runtime validation
regenerate the maps and reject differences. No benchmark content is committed.

A maps to the evaluator options as follows:

| Arm | Text branch | LLM question field | Prompt |
|---|---|---|---|
| A1 | present | native query | question |
| A2 | absent | native query | question |
| A3 | present | omitted from LLM prompt | instruction only |
| A4 | absent | English query; xGQA English file once | question |

Each arm has correct, shuffled 0/1/2, grey and no-image conditions. The A3
user instruction is `Answer with a single word or short phrase, in English.`
with the same system/chat wrapping; NLLB still receives the native question.
A2/A4 plus no-image runs only BOS and the prompt. Disabled encoders/tokenizers
are not loaded by standalone evaluations or invoked by matrix cells.
All original images are validated even in grey/no-image arms, so every
condition is evaluated on the same complete universe.

## Validate and submit evaluations

Preparing the submission is CPU-only and defaults to **not** calling sbatch:

```bash
python Approach2/s1_submit.py --plan evaluation/s1_A.plan.json \
  --submission "$PWD/Approach2/outputs/s1_A.submission.json"
```

It rejects a dirty tree, a HEAD without the freeze, missing analysis/tests,
untracked registries, invalid A grids, missing/corrupt images or conflicting
flags. It freezes the full plan, arguments, checkpoint/data/image hashes,
model revisions, environment and map identities before sbatch. Creating a
submission inside a tracked directory would dirty the tree, so use ignored
`outputs/` as shown.

To actually submit that prepared artifact, explicitly add `--submit`:

```bash
python Approach2/s1_submit.py --resume --submit \
  --submission "$PWD/Approach2/outputs/s1_A.submission.json"
```

To chain another 12-hour allocation, use the same command with
`--dependency afterany:JOB_ID`. Keep HEAD unchanged while the chain is active.
Every allocation validates the submitted code and input hashes before model
loading. Results with valid completion hashes are skipped; partial cells are
recomputed and attributed to the allocation that finished them. Per-cell
predictions, manifests, summaries and completion markers live together.
There is no automatic git commit or push in the S1 allocation.

## Analyse A before continuing

```bash
python Approach2/analysis/block_a.py \
  --submission Approach2/outputs/s1_A.submission.json \
  --output Approach2/outputs/s1_A.analysis.json
```

When harvesting to a laptop, copy the submission plus **all four files per
cell** (`.jsonl`, `.manifest.json`, `.summary.json`, `.complete.json`) and
pass `--results-dir` to override the cluster output directory. The analyser
checks identities, hashes, conditions, labels, map assignments and matched
configurations. It averages the three shuffled outcomes inside each item;
uses paired image-cluster bootstraps, joint across xGQA translations; reports
U, grounding, grey/no-image sensitivity and macro descriptives; and computes
G0/G1-I using the frozen primary CVQA panel. All candidate/reference
differences use candidate minus reference. G1-I is not evaluated when G0
fails, and G1-T remains unresolved under Option 1's deferred replication.

For B, add xGQA `id` and CVQA `bn` to the panels file and use `--block B`,
`--block-a-report` and optionally `--legacy-results` with `s1_plan.py`.
This creates **376 cells** (360 CVQA + 16 xGQA). B requires a passing G0 and
**10 mandatory matched-checkpoint parity cells**: bn/bn on CVQA-bn (correct
and grey, against `eval_cvqa_bn_dcl*.jsonl`, the only legacy CVQA results of
`stage3_bn_dcl`) and id/id on jv/mn/ga/si (against `eval_cvqa_{t}_zsid*.jsonl`
from `stage3_id_v4`). The legacy `eval_cvqa_{t}_zsbn*.jsonl` files were
produced by `stage3_bn_v4`, a different checkpoint, and no id-donor result
exists on CVQA-bn, so those cells have no reference and are not parity cells
(review 2026-09-09). Preparation and every allocation verify each reference's
`summary.json` names the expected checkpoint. Parity cells run and must match
historical per-item predictions before any crossed cell runs; a mismatch
aborts the allocation. The exact checkpoint parity is still to be tested with
real cluster weights.
The plan conservatively evaluates all sixteen xGQA task-control cells; the
frozen ledger already budgets them. Existing CVQA files serve as parity
references, not as unmanifested substitutes for primary S1 observations.

## Exact stage-3 continuation

`training_resume.py` and the stage-3 loop save explicit deterministic batch
schedules, replay position, optimizer state and Python/torch/CUDA RNG state.
Snapshots are written only after a completed optimizer update and gradient
reset, including an explicitly normalized partial final accumulation. A
resumed loader starts from the saved schedule suffix. Preprocessing is
deterministic and loader RNG is isolated from model dropout RNG. CUDA
execution requires deterministic algorithms; unsupported nondeterministic
operations fail instead of silently weakening the contract.

`complete.json` is written only after all epochs and validates the final
training snapshot and best checkpoint. A best checkpoint from epoch one is
not completion. Old snapshots lack enough state for exact continuation and
are rejected; use a new output directory for the corrected recipe. New
snapshots resume automatically when present. `--check-complete` loads no
model and returns 0 for verified completion, 3 for unfinished, and an error
for inconsistent artifacts. The packed legacy launcher uses this check and
no longer auto-commits harvested files.

S1 C pilot preparation also defaults to CPU-only:

```bash
python Approach2/s1_train_submit.py --arm C1 --seed 13 \
  --data-root "$DT" --checkpoints "$PWD/Approach2/outputs" \
  --output-dir "$PWD/Approach2/outputs/s1_C1_seed13" \
  --block-a-report Approach2/outputs/s1_A.analysis.json \
  --submission "$PWD/Approach2/outputs/s1_C1.submission.json"
```

Use `--resume --submit --submission ...` to submit the prepared artifact;
add `--dependency afterany:JOB_ID` for another allocation. Both submission
and runtime check the spec/code/input provenance. The launcher requests a
warning signal before walltime so the trainer can save at an optimizer
boundary. Abrupt termination falls back to the last complete atomic snapshot.

C1/C2/C3/C5 are supported; C4 is evaluation-only. Under `--s1`, towers stay
in eval mode after every `model.train()`, the validation split is by image,
there is no replay and training lasts two epochs. C5 has no text branch or
NLLB tokenization. No C arm is submitted by these instructions automatically;
Option 1's pilot and gate restrictions remain in effect. CPU tests establish
exact continuation of the real loop with toy components. Real H100 behavior,
throughput and checkpoint parity still need cluster validation.
