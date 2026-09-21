"""Compare architectures on identical items, per language, with blind controls (stdlib).

An architecture paper is judged on what its system adds over the backbone it is
built from, so every arm here is a filename template rather than one of our
round tags: a system from another approach, or from a collaborator, plugs in
without touching this file as long as it writes our per-item schema (`id` and
`correct`). The template spells the three things that vary — `{b}` benchmark,
`{L}` language, `{blind}` either empty or `_BLIND` — which covers both our
infixed naming (`eval_cvqa_jv_BLIND_v4.jsonl`) and the ported baseline's
suffixed one (`qwen_cvqa_jv_BLIND.jsonl`).

Two rules from CLAUDE.md shape the output. Full accuracy in a language the
model already knows can be a language prior, so every arm is reported with its
blind control and dV = full - blind, the part that is genuinely visual. And an
image carrying several questions is one unit, so pooled intervals resample
image clusters, stratified by language. CVQA ids carry their image
(`<image>_<question>`); xGQA ids do not, so pooled xGQA needs --image-map
pointing at a file whose rows carry `image_id`, and refuses without it rather
than pretending 12,578 questions are 12,578 independent draws over 398 images.

    python3 ../analysis/arch_compare.py \\
      --arm qwen=qwen_{b}_{L}{blind}.jsonl \\
      --arm a2_v4=eval_{b}_{L}{blind}_v4.jsonl \\
      --benchmark cvqa --output ../audits/arch_compare_cvqa.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import boot_stratified, pct  # noqa: E402

CONDITIONS = {"full": "", "blind": "_BLIND"}


def fail(message: str):
    raise SystemExit(f"arch_compare: {message}")


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
    if not out:
        fail(f"{path}: empty result file")
    return out


def read_image_map(path: Path) -> dict[str, str]:
    """id -> image, from any result file whose rows carry `image_id`."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "image_id" not in row:
                fail(f"{path}: rows carry no image_id, so it cannot serve as an image map")
            out[str(row["id"])] = str(row["image_id"])
    return out


def cluster_of(benchmark: str, item: str, image_map: dict[str, str]) -> str:
    if item in image_map:
        return image_map[item]
    if benchmark == "cvqa":
        return item.rsplit("_", 1)[0]
    fail(f"{benchmark} item {item} has no image: pass --image-map for pooled intervals")


def load_arm(template: str, benchmark: str, langs) -> dict:
    cells = {}
    for lang in langs:
        cells[lang] = {
            condition: read_scores(Path(template.format(b=benchmark, L=lang, blind=marker)))
            for condition, marker in CONDITIONS.items()
        }
    return cells


def shared_items(arms: dict, lang: str) -> list[str]:
    sets = [set(cell[lang][c]) for cell in arms.values() for c in CONDITIONS]
    common = set.intersection(*sets)
    if not common:
        fail(f"{lang}: the arms share no scored items")
    return sorted(common)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant pairs; 1.0 when there are none."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def endpoints(cell: dict, items) -> dict:
    full, blind = cell["full"], cell["blind"]
    return {
        "full": {i: full[i] for i in items},
        "dv": {i: full[i] - blind[i] for i in items},
    }


def interval(values: dict, clusters: dict, B: int, seed: int, label: str) -> dict:
    strata: dict = {}
    flat = []
    for lang, per_item in values.items():
        for item, value in per_item.items():
            strata.setdefault(lang, {}).setdefault(clusters[item], []).append(value)
            flat.append(value)
    samples = boot_stratified(strata, B, seed, label)
    return {
        "estimate": 100 * sum(flat) / len(flat),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "lb5": pct(samples, 0.05),
        "ub95": pct(samples, 0.95),
        "items": len(flat),
        "image_clusters": sum(len(c) for c in strata.values()),
    }


def compare(name_a: str, name_b: str, a: dict, b: dict) -> dict:
    """Arm a minus arm b, per item, plus the exact McNemar on full accuracy."""
    discordant_a = discordant_b = 0
    for lang in a:
        for item in a[lang]["full"]:
            if a[lang]["full"][item] and not b[lang]["full"][item]:
                discordant_a += 1
            elif b[lang]["full"][item] and not a[lang]["full"][item]:
                discordant_b += 1
    return {
        "arms": [name_a, name_b],
        "differences": {
            endpoint: {lang: {i: a[lang][endpoint][i] - b[lang][endpoint][i] for i in a[lang][endpoint]}
                       for lang in a}
            for endpoint in ("full", "dv")
        },
        "mcnemar": {
            f"{name_a}_only": discordant_a,
            f"{name_b}_only": discordant_b,
            "p_two_sided": mcnemar_exact(discordant_a, discordant_b),
        },
    }


def analyse(arms: dict, benchmark: str, langs, image_map: dict, B=4000, seed=0) -> dict:
    per_arm, clusters = {}, {}
    for lang in langs:
        items = shared_items(arms, lang)
        for item in items:
            clusters[item] = cluster_of(benchmark, item, image_map)
        for name, cell in arms.items():
            per_arm.setdefault(name, {})[lang] = endpoints(cell[lang], items)

    report = {
        "benchmark": benchmark,
        "languages": list(langs),
        "arms": sorted(arms),
        "endpoints": "full = accuracy with the image; dv = full - blind, the visual part",
        "clustering": "image_id where the rows carry it, else the CVQA id prefix",
        "boot": B,
        "seed": seed,
        "per_language": {},
        "pooled": {},
        "contrasts": {},
    }
    for name, cells in per_arm.items():
        report["per_language"][name] = {
            lang: {
                "n": len(cells[lang]["full"]),
                "full": 100 * sum(cells[lang]["full"].values()) / len(cells[lang]["full"]),
                "dv": 100 * sum(cells[lang]["dv"].values()) / len(cells[lang]["dv"]),
            }
            for lang in cells
        }
        report["pooled"][name] = {
            endpoint: interval({lang: cells[lang][endpoint] for lang in cells},
                               clusters, B, seed, f"{benchmark}:{name}:{endpoint}")
            for endpoint in ("full", "dv")
        }
    names = sorted(arms)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            c = compare(name_a, name_b, per_arm[name_a], per_arm[name_b])
            key = f"{name_a}-minus-{name_b}"
            report["contrasts"][key] = {
                "mcnemar": c["mcnemar"],
                **{endpoint: interval(c["differences"][endpoint], clusters, B, seed,
                                      f"{benchmark}:{key}:{endpoint}")
                   for endpoint in ("full", "dv")},
                "per_language_full": {
                    lang: 100 * sum(v.values()) / len(v)
                    for lang, v in c["differences"]["full"].items()
                },
            }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True,
                   help="NAME=TEMPLATE, e.g. qwen=qwen_{b}_{L}{blind}.jsonl")
    p.add_argument("--benchmark", required=True, choices=["cvqa", "xgqa"])
    p.add_argument("--langs", required=True, nargs="+")
    p.add_argument("--image-map", help="result file whose rows carry image_id (needed for xGQA)")
    p.add_argument("--output", required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    arms = {}
    for spec in a.arm:
        if "=" not in spec:
            fail(f"--arm {spec!r} is not NAME=TEMPLATE")
        name, template = spec.split("=", 1)
        if "{L}" not in template or "{blind}" not in template:
            fail(f"--arm {name}: the template needs {{L}} and {{blind}}")
        arms[name] = load_arm(template, a.benchmark, a.langs)
    if len(arms) < 2:
        fail("at least two arms are needed")

    image_map = read_image_map(Path(a.image_map)) if a.image_map else {}
    report = analyse(arms, a.benchmark, a.langs, image_map, a.boot, a.seed)
    Path(a.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for name in report["arms"]:
        pooled = report["pooled"][name]
        print(f"{name:12s} full {pooled['full']['estimate']:6.2f}  dV {pooled['dv']['estimate']:6.2f}")
    for key, c in sorted(report["contrasts"].items()):
        print(f"{key}: full {c['full']['estimate']:+.2f} {c['full']['ci95']}, "
              f"dV {c['dv']['estimate']:+.2f} {c['dv']['ci95']}, "
              f"McNemar p = {c['mcnemar']['p_two_sided']:.3g}")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
