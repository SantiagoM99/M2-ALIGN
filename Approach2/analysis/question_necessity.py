"""Split CVQA into the items where the native question decides, and measure there.

Why this exists. On CVQA, a zero-shot VLM does *better* with the question removed
(+1.50, DESIGN 2026-09-26): the four answer choices plus the image are usually
enough, so the benchmark rewards ignoring the pathway a multilingual text bridge
exists to improve. A bridge therefore cannot show its value in the pooled average
however good it is — Maryam's a1 lands at +0.15 over its own baseline on CVQA
while gaining +6.75 on xGQA, and the across-language spread it compresses by 37%
on xGQA it does not compress at all on CVQA.

So the instrument has to be built where the pathway is load-bearing. Using a
**reference** system's two arms, each item is labelled:

    necessary   the reference answers it correctly with the question and wrongly
                without it — the question decided
    shortcut    correct both ways — the image and the choices sufficed
    other       wrong with the question (whether or not blind is right)

**The split is conditioned on the reference being correct**, so the reference
scores 100% on both `necessary` and `shortcut` by construction and must never be
one of the arms compared on them — it can only lose. The split is valid for
comparing two systems that are *neither* the reference: the reference defines the
items, the comparison happens between the others. Both the subset and the full
set are always reported together, because a gain that exists only on a subset is
a conditional gain and has to be labelled as one.

    python3 Approach2/analysis/question_necessity.py \\
      --reference-with 'qwen_cvqa_{L}.jsonl' \\
      --reference-blind 'qwen_cvqa_{L}_QBLIND.jsonl' \\
      --arm v4='eval_cvqa_{L}_v4.jsonl' \\
      --langs bn ru zh pt id ko jv mn si ga \\
      --output ../audits/question_necessity_cvqa.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import boot_stratified, pct  # noqa: E402

LABELS = ("necessary", "shortcut", "other")


def fail(message: str):
    raise SystemExit(f"question_necessity: {message}")


def read_scores(path: Path) -> dict[str, int]:
    if not path.is_file():
        fail(f"missing {path}")
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = str(row.get("id", "")).strip()
            if not item:
                fail(f"{path}:{lineno}: missing id")
            if not isinstance(row.get("correct"), bool):
                fail(f"{path}:{lineno}: 'correct' must be a JSON boolean")
            if item in out:
                fail(f"{path}: duplicate id {item}")
            out[item] = int(row["correct"])
    return out


def label_items(with_q: dict[str, int], blind: dict[str, int]) -> dict[str, str]:
    """Label every item the reference scored in both conditions."""
    shared = set(with_q) & set(blind)
    if not shared:
        fail("the reference's two arms share no items")
    labels = {}
    for item in sorted(shared):
        if with_q[item] and not blind[item]:
            labels[item] = "necessary"
        elif with_q[item] and blind[item]:
            labels[item] = "shortcut"
        else:
            labels[item] = "other"
    return labels


def cluster_of(item: str) -> str:
    """CVQA ids are `<image>_<question index>`, so the image is the prefix."""
    return item.rsplit("_", 1)[0]


def interval(values: dict[str, dict[str, int]], B: int, seed: int, label: str) -> dict:
    strata: dict = {}
    flat = []
    for lang, per_item in values.items():
        for item, value in per_item.items():
            strata.setdefault(lang, {}).setdefault(cluster_of(item), []).append(value)
            flat.append(value)
    if not flat:
        return {"estimate": None, "items": 0}
    samples = boot_stratified(strata, B, seed, label)
    return {
        "estimate": 100 * sum(flat) / len(flat),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "items": len(flat),
        "image_clusters": sum(len(c) for c in strata.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-with", required=True, help="template with {L}")
    p.add_argument("--reference-blind", required=True, help="template with {L}")
    p.add_argument("--arm", action="append", required=True, metavar="NAME=TEMPLATE")
    p.add_argument("--langs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    arms = {}
    for spec in a.arm:
        if "=" not in spec:
            fail(f"--arm {spec!r} is not NAME=TEMPLATE")
        name, template = spec.split("=", 1)
        if template in (a.reference_with, a.reference_blind):
            fail(
                f"--arm {name} is the reference itself. The split conditions on the "
                "reference's correctness, so it scores 100% on `necessary` and "
                "`shortcut` by construction and the comparison would be meaningless. "
                "Compare systems that are not the reference"
            )
        arms[name] = template

    labels, reference = {}, {}
    for lang in a.langs:
        with_q = read_scores(Path(a.reference_with.format(L=lang)))
        blind = read_scores(Path(a.reference_blind.format(L=lang)))
        labels[lang] = label_items(with_q, blind)
        reference[lang] = with_q

    counts = {label: 0 for label in LABELS}
    for lang in a.langs:
        for label in labels[lang].values():
            counts[label] += 1
    total = sum(counts.values())
    report = {
        "langs": list(a.langs),
        "definition": "necessary = reference right with the question and wrong without it; "
                      "shortcut = right both ways; other = wrong with the question",
        "split_defined_by": {"with": a.reference_with, "blind": a.reference_blind},
        "counts": counts,
        "shares": {k: round(100 * v / total, 2) for k, v in counts.items()},
        "boot": a.boot,
        "seed": a.seed,
        "arms": {},
    }

    for name, template in arms.items():
        scores = {lang: read_scores(Path(template.format(L=lang))) for lang in a.langs}
        entry = {}
        for label in LABELS + ("all",):
            values = {
                lang: {
                    item: scores[lang][item]
                    for item in labels[lang]
                    if item in scores[lang] and (label == "all" or labels[lang][item] == label)
                }
                for lang in a.langs
            }
            entry[label] = interval(values, a.boot, a.seed, f"qn:{name}:{label}")
        report["arms"][name] = entry

    # The reference's own accuracy on each subset, so a reader can see what the
    # split did to it: on `necessary` it is 100% and on `other` it is 0% by
    # construction, which is the honest way to show the split is not a ranking.
    report["reference_by_construction"] = {
        "necessary": "100% for the reference by definition",
        "other": "0% for the reference by definition",
        "shortcut": "100% for the reference by definition",
    }
    Path(a.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"split over {total} items: " + ", ".join(
        f"{k} {counts[k]} ({report['shares'][k]}%)" for k in LABELS))
    for name, entry in report["arms"].items():
        parts = []
        for label in ("all", "necessary", "shortcut"):
            e = entry[label]
            parts.append(f"{label} {e['estimate']:.2f} (n={e['items']})"
                         if e["estimate"] is not None else f"{label} n/a")
        print(f"{name:10s} " + "  ".join(parts))
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
