"""S1 identities, provenance and deterministic image permutations (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

SPEC_SHA = "3b4faff4d673fad94cd3115b3f14212eeca9c12e"
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"{path}: duplicate key {k}")
            result[k] = v
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def read_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    ids = [str(r.get("id", "")) for r in rows]
    if not rows or any(not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: empty data or empty/duplicate item ids")
    return rows


def item_identity(row, benchmark):
    q = str(row["id"])
    image = row.get("image_id", row.get("vg_image_id"))
    if image is None:
        raise ValueError(f"{q}: explicit image_id/vg_image_id required")
    image = str(image)
    if not image or Path(image).name != image:
        raise ValueError(f"{q}: invalid image id")
    subset = str(row.get("subset", "")) if benchmark == "cvqa" else "xgqa"
    if benchmark == "cvqa" and not subset:
        raise ValueError(f"{q}: CVQA subset required; rebuild with build_cvqa_s1.py")
    return {"id": q, "image_id": image, "subset": subset}


def universe(rows, benchmark):
    items = sorted((item_identity(r, benchmark) for r in rows), key=lambda r: r["id"])
    if not items or len({r["id"] for r in items}) != len(items):
        raise ValueError("empty or duplicate item universe")
    owner = {}
    for r in items:
        if owner.setdefault(r["image_id"], r["subset"]) != r["subset"]:
            raise ValueError("an image occurs in multiple subsets")
    return items


def shuffle_maps(rows, benchmark):
    """Three independently seeded rejection-sampled derangements per subset.

    Canonical ordering and no language label in xGQA's seed preserve pairing
    across translations. A repeated subset permutation is redrawn.
    """
    items = universe(rows, benchmark)
    groups = {}
    for r in items:
        groups.setdefault(r["subset"], set()).add(r["image_id"])
    used = {s: set() for s in groups}
    output = []
    for seed in SEEDS:
        assignments = {}
        for subset in sorted(groups):
            images = sorted(groups[subset])
            if len(images) < 4:
                raise ValueError(f"{subset}: at least four images required")
            rnd = random.Random(int(digest([seed, subset, images]), 16))
            while True:
                assigned = images.copy()
                rnd.shuffle(assigned)
                key = tuple(assigned)
                if key not in used[subset] and all(
                    a != b for a, b in zip(images, assigned)
                ):
                    break
            used[subset].add(key)
            assignments.update(zip(images, assigned))
        body = {
            "schema_version": 1,
            "seed": seed,
            "benchmark": benchmark,
            "id_universe_sha256": digest(items),
            "items": {
                r["id"]: {
                    "original_image_id": r["image_id"],
                    "assigned_image_id": assignments[r["image_id"]],
                    "subset": r["subset"],
                }
                for r in items
            },
        }
        output.append({**body, "sha256": digest(body)})
    return output


def validate_shuffle(payload, rows, benchmark):
    if payload.get("seed") not in SEEDS:
        raise ValueError("shuffle seed must be 0, 1 or 2")
    expected = shuffle_maps(rows, benchmark)[payload["seed"]]
    if payload != expected:
        raise ValueError(
            "shuffle does not match canonical deterministic map or item universe"
        )
    return payload


def git_state(root=ROOT, require_clean=True, expected_code=None):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", SPEC_SHA, head],
        capture_output=True,
    ).returncode:
        raise ValueError(f"HEAD must contain S1 freeze {SPEC_SHA}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if require_clean and dirty:
        raise ValueError("S1 requires a clean working tree (including untracked files)")
    if expected_code is not None and head != expected_code:
        raise ValueError("HEAD differs from submitted code SHA")
    return {"spec_sha": SPEC_SHA, "code_sha": head, "dirty": dirty}


def branch_paths(args):
    if args.extra_ckpt and (args.txt_ckpt or args.vis_ckpt):
        raise ValueError(
            "--extra-ckpt cannot be combined with explicit branch checkpoints"
        )
    return {
        "mapping_txt": args.txt_ckpt or args.ckpt,
        "mapping_vis": args.vis_ckpt or args.extra_ckpt or args.ckpt,
    }


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path)}


def result_complete(path, manifest):
    """A result is complete only with an authenticated completion sidecar."""
    path = Path(path)
    marker = Path(str(path) + ".complete.json")
    if not marker.exists():
        return False
    info = read_json(marker)
    summary = Path(str(path) + ".summary.json")
    if (
        info.get("manifest_sha256") != digest(manifest)
        or not path.exists()
        or not summary.exists()
        or info.get("predictions_sha256") != file_sha(path)
        or info.get("summary_sha256") != file_sha(summary)
    ):
        raise ValueError(f"{path}: completion/configuration/hash mismatch")
    return True
