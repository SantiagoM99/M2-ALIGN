"""D12 pooling verdict: joint stage-1 mapping (vj) against per-language (v4r), stdlib.

Implements the reading fixed in DESIGN.md on 2026-09-12 and its endpoint
clarification of 2026-09-13, before either tagged evaluation was read:
  * both arms trained by the same stage-3 trainer and scored by one launcher,
    under one tag, on one image copy (`evaluate_all.sh`, EVAL_TAG=tf5);
  * primary endpoint utility U = accuracy with the correct image, difference
    joint minus independent, item-micro, paired on identical items;
    Δ_gray (correct minus gray canvas) is descriptive; the grounding endpoint
    does not exist for these runs because no shuffled condition was scored;
  * **lift**: CVQA pooled jv/mn/ga, LB5 > 0;
  * **flat**: xGQA pooled de/ru/zh, LB5 > −1.0 and UB95 < +1.0;
  * supported only if both hold; refuted if the lift fails, or if de/ru/zh lift
    by at least as much as jv/mn/ga (point estimates); otherwise inconclusive.
The bootstrap resamples images: CVQA is stratified by target, and xGQA's
translations share their images, so its three languages form one stratum whose
clusters carry every translation of every question on an image.

    cd Approach2/results
    python3 ../analysis/pooling_d12.py --joint vj_tf5 --independent v4r_tf5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boot import boot_stratified, pct  # noqa: E402

LRL = ("jv", "mn", "ga")
HRL = ("de", "ru", "zh")
CVQA_DESCRIPTIVE = ("ru", "zh")
MARGIN = 1.0


def fail(message):
    raise SystemExit(f"pooling_d12: {message}")


def load(results: Path, bench: str, lang: str, tag: str, blind: bool) -> dict:
    path = results / f"eval_{bench}_{lang}{'_BLIND' if blind else ''}_{tag}.jsonl"
    if not path.is_file():
        fail(f"missing {path.name}")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            key = str(r["id"])
            if key in rows:
                fail(f"{path.name}: duplicate id {key}")
            if not isinstance(r.get("correct"), bool):
                fail(f"{path.name}: item {key} has no boolean 'correct'")
            image = r.get("image_id", r.get("vg_image_id"))
            if image is None and bench == "cvqa":
                image = key.rsplit("_", 1)[0]
            if image is None:
                fail(f"{path.name}: item {key} has no image id")
            rows[key] = {"correct": int(r["correct"]), "image": str(image)}
    if not rows:
        fail(f"{path.name} is empty")
    return rows


def paired(results, bench, langs, joint, independent, endpoint):
    """Per-item joint minus independent values, keyed by language."""
    out, images = {}, {}
    for lang in langs:
        cells = {}
        for arm, tag in (("joint", joint), ("independent", independent)):
            full = load(results, bench, lang, tag, blind=False)
            if endpoint == "gray":
                gray = load(results, bench, lang, tag, blind=True)
                if gray.keys() != full.keys():
                    fail(f"{bench}/{lang}/{tag}: gray and correct items differ")
                cells[arm] = {i: full[i]["correct"] - gray[i]["correct"] for i in full}
            else:
                cells[arm] = {i: full[i]["correct"] for i in full}
            images.setdefault(lang, {}).update({i: full[i]["image"] for i in full})
        if cells["joint"].keys() != cells["independent"].keys():
            fail(f"{bench}/{lang}: the two arms were scored on different items")
        out[lang] = {i: cells["joint"][i] - cells["independent"][i] for i in cells["joint"]}
    return out, images


def interval(values, images, bench, B, seed, label):
    strata = {}
    for lang in sorted(values):
        stratum = "xgqa-shared-images" if bench == "xgqa" else lang
        for i in sorted(values[lang]):
            strata.setdefault(stratum, {}).setdefault(images[lang][i], []).append(values[lang][i])
    samples = boot_stratified(strata, B, seed, label)
    flat = [v for lang in values for v in values[lang].values()]
    return {
        "estimate": 100.0 * sum(flat) / len(flat),
        "lb5": pct(samples, 0.05),
        "ub95": pct(samples, 0.95),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "items": len(flat),
        "image_clusters": sum(len(c) for c in strata.values()),
    }


def analyse(results: Path, joint: str, independent: str, B: int = 4000, seed: int = 0) -> dict:
    report = {"joint": joint, "independent": independent, "boot": B, "seed": seed,
              "difference": "joint minus independent, accuracy points", "panels": {}}
    for name, bench, langs in (("lrl-cvqa", "cvqa", LRL), ("hrl-xgqa", "xgqa", HRL),
                               ("hrl-cvqa-descriptive", "cvqa", CVQA_DESCRIPTIVE)):
        panel = {}
        for endpoint in ("U", "gray"):
            values, images = paired(results, bench, langs, joint, independent, endpoint)
            panel[endpoint] = {"pooled": interval(values, images, bench, B, seed, f"d12:{name}:{endpoint}")}
            for lang in langs:
                panel[endpoint][lang] = interval({lang: values[lang]}, images, bench, B, seed,
                                                 f"d12:{name}:{endpoint}:{lang}")
        report["panels"][name] = panel
    lift = report["panels"]["lrl-cvqa"]["U"]["pooled"]
    flat = report["panels"]["hrl-xgqa"]["U"]["pooled"]
    lift_holds = lift["lb5"] > 0
    flat_holds = flat["lb5"] > -MARGIN and flat["ub95"] < MARGIN
    uniform = flat["estimate"] >= lift["estimate"]
    if lift_holds and flat_holds:
        verdict = "supported"
    elif not lift_holds or (lift_holds and uniform):
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    report["verdict"] = {
        "lift_jv_mn_ga_LB5_gt_0": lift_holds,
        "flat_de_ru_zh_within_1": flat_holds,
        "uniform_lift": uniform,
        "D12": verdict,
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=".")
    p.add_argument("--joint", default="vj_tf5")
    p.add_argument("--independent", default="v4r_tf5")
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output")
    a = p.parse_args()
    report = analyse(Path(a.results_dir), a.joint, a.independent, a.boot, a.seed)
    text = json.dumps(report, indent=1)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=1))


if __name__ == "__main__":
    main()
