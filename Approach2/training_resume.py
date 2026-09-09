"""Deterministic batch schedules and atomic optimizer-boundary training snapshots."""

from __future__ import annotations
import os
from pathlib import Path
import random

from s1_contract import atomic_json, digest, file_sha, read_json


def epoch_batches(length, batch_size, seed, epoch, stream="vqa"):
    if length <= 0 or batch_size <= 0:
        raise ValueError("empty dataset/invalid batch size")
    order = list(range(length))
    random.Random(int(digest([seed, epoch, stream]), 16)).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, length, batch_size)]


def split_rows(rows, ratio, seed, by_image=True):
    if not 0 < ratio < 1:
        raise ValueError("validation ratio must be in (0,1)")
    if by_image:
        keys = sorted({str(r["vg_image_id"]) for r in rows})
        random.Random(seed).shuffle(keys)
        n = max(1, int(len(keys) * ratio))
        if n >= len(keys):
            raise ValueError("need at least two images")
        val = set(keys[:n])
        train = [r for r in rows if str(r["vg_image_id"]) not in val]
        valid = [r for r in rows if str(r["vg_image_id"]) in val]
    else:
        copied = list(rows)
        random.Random(seed).shuffle(copied)
        n = max(1, int(len(copied) * ratio))
        train, valid = copied[n:], copied[:n]
    if not train or not valid:
        raise ValueError("empty training/validation split")
    return train, valid


def save_snapshot(path, model, optimizer, progress, configuration):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None for p in params):
        raise ValueError("snapshot must be taken after optimizer.step and zero_grad")
    payload = {
        "schema_version": 2,
        "configuration_sha256": digest(configuration),
        "progress": progress,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mappings": {
            k: getattr(model, k).state_dict()
            for k in ("mapping_txt", "mapping_vis")
            if getattr(model, k, None) is not None
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def restore_snapshot(path, model, optimizer, configuration):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 2:
        raise ValueError(
            "legacy snapshot has no exact sampler state; cannot resume it as an exact run"
        )
    if state["configuration_sha256"] != digest(configuration):
        raise ValueError("resume configuration differs")
    for k, v in state["mappings"].items():
        getattr(model, k).load_state_dict(v, strict=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def completed(output_dir, configuration):
    root = Path(output_dir)
    marker = root / "complete.json"
    if not marker.exists():
        return False
    m = read_json(marker)
    if m["configuration_sha256"] != digest(configuration):
        raise ValueError("completed run has different configuration")
    for filename, expected in m["files"].items():
        path = root / filename
        if not path.exists() or file_sha(path) != expected:
            raise ValueError("completed training artifacts changed or disappeared")
    if m["epochs_completed"] != configuration["arguments"]["epochs"]:
        raise ValueError("wrong completed epoch count")
    return True


def mark_completed(output_dir, configuration, epochs):
    root = Path(output_dir)
    atomic_json(
        root / "complete.json",
        {
            "schema_version": 1,
            "configuration_sha256": digest(configuration),
            "epochs_completed": epochs,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "files": {
                p: file_sha(root / p)
                for p in ("training_state.pt", "mapping/pytorch_model.bin")
            },
        },
    )
