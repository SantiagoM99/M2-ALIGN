"""Check that no image we train on is an image we evaluate on (stdlib).

The CG50 registration (DESIGN 2026-09-26) requires this before the run, not
after: CulturalGround's images come from Wikimedia Commons and CVQA's are
contributed by its annotators, so a file-level collision is unlikely — but
"unlikely" is not a control, and a single shared photograph would turn a
grounding gain into leakage.

Scope, stated because it bounds what a pass means. This compares **file
contents** by SHA-256. It catches the same photograph appearing in both sets,
whatever its filename. It does not catch the same *entity* photographed twice,
which is not leakage but a distribution match, and which the paper states
instead of testing. It also cannot catch a re-encoded copy of the same
photograph: byte equality is the test, so a JPEG requantised at a different
quality reads as a different image. That limit matters here because our own S1
builder re-encodes CVQA images, so this check runs against the image copy the
evaluation actually reads.

Exits non-zero when anything collides, so a launcher can gate on it.

    python3 Approach2/analysis/image_leakage.py \\
      --train-dir $DT/Stage3/data/gqa/images --train-prefix cg_ \\
      --eval-dir $DT/Stage3/data/cvqa/images \\
      --output Approach2/audits/cg_leakage.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHUNK = 1 << 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def index(directory: Path, prefix: str = "") -> dict[str, list[str]]:
    """content hash -> the names carrying it, for every file under `directory`."""
    if not directory.is_dir():
        raise SystemExit(f"image_leakage: not a directory: {directory}")
    out: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        out.setdefault(digest(path), []).append(path.name)
    if not out:
        raise SystemExit(f"image_leakage: no files under {directory} matching prefix {prefix!r}")
    return out


def collisions(train: dict, evaluation: dict) -> list[dict]:
    return [
        {"sha256": h, "train": train[h], "eval": evaluation[h]}
        for h in sorted(set(train) & set(evaluation))
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--train-prefix", default="", help="only files whose name starts with this")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--eval-prefix", default="")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    train = index(Path(a.train_dir), a.train_prefix)
    evaluation = index(Path(a.eval_dir), a.eval_prefix)
    found = collisions(train, evaluation)
    report = {
        "train_dir": a.train_dir,
        "train_prefix": a.train_prefix,
        "eval_dir": a.eval_dir,
        "eval_prefix": a.eval_prefix,
        "train_images": sum(len(v) for v in train.values()),
        "eval_images": sum(len(v) for v in evaluation.values()),
        "distinct_train_contents": len(train),
        "distinct_eval_contents": len(evaluation),
        "method": "SHA-256 over file contents; catches the same photograph, not the same entity",
        "collisions": found,
    }
    Path(a.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{report['train_images']} training images vs {report['eval_images']} evaluation images")
    print(f"-> {a.output}")
    if found:
        for c in found[:10]:
            print(f"LEAK {c['sha256'][:12]} train={c['train']} eval={c['eval']}")
        raise SystemExit(f"image_leakage: {len(found)} colliding image(s); the run is blocked")
    print("no file-level overlap")


if __name__ == "__main__":
    main()
