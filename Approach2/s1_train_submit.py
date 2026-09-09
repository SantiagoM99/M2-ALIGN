"""Prepare/submit/resume a provenance-checked S1 Block C single-seed pilot."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

from s1_contract import (
    ROOT,
    SPEC_SHA,
    atomic_json,
    digest,
    file_record,
    file_sha,
    git_state,
    read_json,
    read_rows,
)
from training_resume import epoch_batches, split_rows
from eval_runtime import environment_record, model_record


def build(a):
    state = git_state()
    gate = read_json(a.block_a_report)
    if gate.get("spec_sha") != SPEC_SHA or gate.get("gates", {}).get("G0") is not True:
        raise ValueError("Block C requires the authenticated Block A G0 pass")
    if a.arm == "C5" and gate["gates"].get("G1-I") == "dispensable":
        raise ValueError("C5 is skipped when G1-I is dispensable")
    dt = Path(a.data_root).resolve()
    ck = Path(a.checkpoints).resolve()
    data = dt / "Stage3/data/stage3b/bengali.jsonl"
    tr, va = split_rows(read_rows(data), 0.03, a.seed, True)
    checkpoints = {
        "mapping_txt": file_record(ck / "stage1/mapping/pytorch_model.bin"),
        "mapping_vis": file_record(ck / "stage2_dc_llava/mapping/pytorch_model.bin"),
    }
    args = [
        "--s1",
        "--data-path",
        str(data),
        "--images-dir",
        str(dt / "Stage3/data/gqa/images"),
        "--output-dir",
        str(Path(a.output_dir).resolve()),
        "--stage1-ckpt",
        checkpoints["mapping_txt"]["path"],
        "--stage2-ckpt",
        checkpoints["mapping_vis"]["path"],
        "--vis-layers",
        "9,18,-1",
        "--epochs",
        "2",
        "--seed",
        str(a.seed),
        "--train-batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--grad-accum",
        "16",
        "--lr",
        "2e-5",
        "--save-steps",
        "200",
        "--local-files-only",
    ]
    flag = {
        "C1": None,
        "C2": "--freeze-text-mapping",
        "C3": "--freeze-vision-mapping",
        "C5": "--no-text-branch",
    }[a.arm]
    if flag:
        args.append(flag)
    frozen_models = {
        "llm": model_record("google/gemma-2-9b-it"),
        "text": model_record("facebook/nllb-200-distilled-600M")
        if a.arm != "C5"
        else None,
        "vision": model_record("google/siglip2-so400m-patch14-384"),
    }
    for name, key in [
        ("llm", "--llm-path"),
        ("text", "--mt-path"),
        ("vision", "--vis-path"),
    ]:
        if frozen_models[name]:
            args.extend([key, frozen_models[name]["path"]])
    # Validate every training image and record its content before submission.
    from PIL import Image

    image_hashes = {}
    for r in tr + va:
        iid = str(r["vg_image_id"])
        if iid in image_hashes:
            continue
        candidates = [
            dt / "Stage3/data/gqa/images" / (iid + ext)
            for ext in (".jpg", ".jpeg", ".png")
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"missing training image {iid}")
        with Image.open(path) as im:
            im.verify()
        image_hashes[iid] = file_record(path)
    return {
        **state,
        "kind": "s1-training",
        "experiment_id": f"s1_{a.arm}_seed{a.seed}",
        "arm": a.arm,
        "seed": a.seed,
        "source": "bn",
        "targets": ["jv", "mn", "ga", "si", "bn"],
        "hypothesis": "S1 P5/P6/P7; C5 exploratory under Option 1",
        "prediction": "C2 improves grounding relative to C1; source utility non-inferior",
        "estimand": "candidate-minus-reference U and grounding, single-seed pilot",
        "arguments": args,
        "data": file_record(data),
        "checkpoints": checkpoints,
        "images": image_hashes,
        "frozen_models": frozen_models,
        "environment": environment_record(),
        "split_sha256": digest({"train": tr, "validation": va}),
        "order_hashes": {
            str(e): digest(epoch_batches(len(tr), 2, a.seed, e)) for e in range(2)
        },
        "shuffle_map_hashes": {},
        "vis_layers": "9,18,-1",
        "decoding": None,
        "block_a_report": file_record(a.block_a_report),
        "slurm_job_id": None,
    }


def validate_inputs(m):
    git_state(expected_code=m["code_sha"])
    if m["spec_sha"] != SPEC_SHA or m["kind"] != "s1-training":
        raise ValueError("invalid training submission")
    if m["environment"] != environment_record():
        raise ValueError("training environment changed since submission")
    for r in m["frozen_models"].values():
        if r is not None and model_record(r["identifier"]) != r:
            raise ValueError("frozen model revision changed since submission")
    for r in [
        m["data"],
        m["block_a_report"],
        *m["checkpoints"].values(),
        *m["images"].values(),
    ]:
        if file_sha(r["path"]) != r["sha256"]:
            raise ValueError("training input changed after submission")
    gate = read_json(m["block_a_report"]["path"])
    if not gate["gates"]["G0"]:
        raise ValueError("G0 failed")


def execute(path):
    m = read_json(path)
    validate_inputs(m)
    out = Path(m["arguments"][m["arguments"].index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "submission.manifest.json"
    if manifest.exists():
        prior = read_json(manifest)
        if {k: v for k, v in prior.items() if k != "slurm_job_id"} != {
            k: v for k, v in m.items() if k != "slurm_job_id"
        }:
            raise ValueError("training output belongs to a different submission")
    else:
        atomic_json(manifest, {**m, "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    # Exec retains the SLURM step's signal delivery to the training process.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            str(ROOT / "Approach2/train_stage3_vqa.py"),
            *m["arguments"],
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-submission")
    p.add_argument("--arm", choices=["C1", "C2", "C3", "C5"])
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--data-root")
    p.add_argument("--checkpoints")
    p.add_argument("--output-dir")
    p.add_argument("--block-a-report")
    p.add_argument("--submission")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dependency")
    a = p.parse_args()
    if a.execute_submission:
        execute(a.execute_submission)
        return
    for key in (
        ("submission",)
        if a.resume
        else (
            "arm",
            "data_root",
            "checkpoints",
            "output_dir",
            "block_a_report",
            "submission",
        )
    ):
        if not getattr(a, key):
            p.error(f"--{key.replace('_', '-')} is required")
    path = Path(a.submission).resolve()
    if a.resume:
        m = read_json(path)
        validate_inputs(m)
    else:
        if path.exists():
            raise ValueError("immutable submission already exists; use --resume")
        m = build(a)
        atomic_json(path, m)
    if a.submit:
        git_state(expected_code=m["code_sha"])
        command = ["sbatch", "--parsable"]
        if a.dependency:
            command.extend(["--dependency", a.dependency])
        job = subprocess.check_output(
            [
                *command,
                str(ROOT / "Approach2/job-scripts/train_stage3_s1.sh"),
                str(path),
            ],
            text=True,
        ).strip()
        atomic_json(
            str(path) + f".receipt.{job.split(';')[0]}.json",
            {
                "slurm_job_id": job,
                "spec_sha": SPEC_SHA,
                "code_sha": m["code_sha"],
                "submission_sha256": file_sha(path),
            },
        )
        print(job)
    else:
        print(f"Prepared {path}; no GPU job submitted")


if __name__ == "__main__":
    main()
