"""CG50 against C1: the two readings declared on 2026-10-08, kept apart (stdlib).

One training run answers two questions that must not be conflated, and both were
written down before the arm was built (DESIGN 2026-10-08):

1. **bn, country matched and language supervised.** Does the block CVQA's
   backbone gets wrong move when the supervised culture is the panel's own?
   Declared pass: Delta on the `other` subset with LB5 > 0, and the gain *not*
   concentrated in `shortcut`.
2. **jv/mn/ga, zero-shot and other cultures.** Does culturally grounded
   supervision in one donor language transfer to unseen languages whose
   countries differ? Declared pass: LB5(Delta_ground(CG50) - Delta_ground(C1))
   > 0 on CVQA jv/mn/ga pooled, with the source task retained,
   UB95(U(C1) - U(CG50)) < delta = 1.0 on xGQA-bn.

Reading 2 can pass while reading 1 fails, and the reverse, so the gates are
reported separately and neither is summarised into the other.

**Two things the registration left open, resolved here and labelled.** It wrote
"Delta on the `other` subset" without naming an endpoint: U is read as the
declared one, because the question is whether items the model answers wrongly
start being answered correctly, and grounding is reported beside it rather than
instead of it. And "not concentrated in `shortcut`" is qualitative; a difference
between two disjoint subsets is not a paired item contrast, so it is tested as
the difference of the two subset means under one joint resample of clusters, and
that operationalisation is post-hoc -- fixed at read time, not on 10-08.

The subsets come from a **reference** system's two arms, never from an arm being
compared: the reference scores 100% on `necessary` and `shortcut` by
construction (`question_necessity.py`). Qwen3-VL zero-shot is that reference.

Run on the login node, where the cells are:

    python3 Approach2/analysis/cg50.py \\
      --submission Approach2/outputs/s1_CG50.eval.submission.json \\
      --reference-with Approach2/results/qwen_cvqa_bn.jsonl \\
      --reference-blind Approach2/results/qwen_cvqa_bn_QBLIND.jsonl \\
      --output Approach2/audits/s1_CG50_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_contract import SPEC_SHA, atomic_json, file_sha  # noqa: E402
from _boot import pct, rng_for  # noqa: E402
from block_a import load_cells  # noqa: E402
from block_c import checkpoint_name, endpoints, interval  # noqa: E402
from question_necessity import label_items, read_scores  # noqa: E402

ARMS = {"C1": "s1_C1_seed13", "CG50": "cg50_seed13"}
CVQA_TARGETS = ("jv", "mn", "ga", "si", "bn")
PRIMARY = ("jv", "mn", "ga")
CONDITIONS = ("correct", "shuffled0", "shuffled1", "shuffled2", "gray")
SUBSETS = ("necessary", "shortcut", "other")
DELTA = 1.0
GAIN = {"CG50": 1.0, "C1": -1.0}
RETENTION = {"C1": 1.0, "CG50": -1.0}


def required_cells():
    req = set()
    for arm in ARMS:
        for t in CVQA_TARGETS:
            req.update(("cvqa", t, arm, c) for c in CONDITIONS)
        req.update(("xgqa", "bn", arm, c) for c in CONDITIONS)
    return req


def validate_grid(cells):
    """CG50 is read on C1's grid, so the grid has to be C1's exactly."""
    required = required_cells()
    if set(cells) != required:
        raise ValueError(
            f"CG50 needs its complete {len(required)}-cell grid: missing "
            f"{sorted(required - set(cells))[:3]}, extra {sorted(set(cells) - required)[:3]}"
        )
    refs = {}
    for (b, t, arm, c), cell in sorted(cells.items()):
        m, rows = cell["manifest"], cell["rows"]
        args = m["arguments"]
        if m["vis_layers"] != "9,18,-1":
            raise ValueError(f"{arm}/{c}: CG50 is read against dense vision layers 9,18,-1")
        if args["no_text_branch"]:
            raise ValueError(f"{arm}/{c}: both arms carry the text branch")
        if args["prompt_mode"] != "question" or args["question_field"] != "query":
            raise ValueError(f"{arm}/{c}: the prompt must carry the native question")
        if args["location_aware"]:
            raise ValueError(f"{arm}/{c}: location-aware is a different condition")
        for key, directory in (("mapping_txt", ARMS[arm]), ("mapping_vis", ARMS[arm])):
            record = m["checkpoints"].get(key)
            if not record or checkpoint_name(record) != directory:
                raise ValueError(f"{arm}/{c}: {key} is not loaded from {directory}")
        # Both arms must have been scored on one panel, or no difference is paired.
        ref = refs.setdefault((b, t), cell)
        if rows.keys() != ref["rows"].keys():
            raise ValueError(f"cvqa/{t}: item ids differ across cells of one target")
        for field in ("data", "id_universe_sha256", "image_sha256", "shuffle_map_hashes", "decoding"):
            if m[field] != ref["manifest"][field]:
                raise ValueError(f"{b}/{t}: {field} differs across cells of one target")
        if m.get("environment") != ref["manifest"].get("environment"):
            raise ValueError("runtime environment differs across cells")


def contrast(eps, targets, weights, endpoint):
    return {
        t: {
            i: sum(w * eps[arm][t][endpoint][i] for arm, w in weights.items())
            for i in eps["C1"][t][endpoint]
        }
        for t in targets
    }


def labels_for_panel(cells, reference_with, reference_blind):
    """Label every bn item by what the reference's question did for it."""
    with_q = read_scores(Path(reference_with))
    blind = read_scores(Path(reference_blind))
    labelled = label_items(with_q, blind)
    panel = cells[("cvqa", "bn", "C1", "correct")]["rows"]
    covered = {i: labelled[i] for i in panel if i in labelled}
    if len(covered) < 0.9 * len(panel):
        raise ValueError(
            f"the reference labels only {len(covered)} of {len(panel)} bn panel items; "
            "the reference arms and the S1 panel are not the same item universe"
        )
    return covered


def subset_mean_difference(cells, values, labels, first, second, B, seed, estimand):
    """LB/UB for (mean over `first`) - (mean over `second`), disjoint subsets.

    `boot_stratified` draws each stratum independently per estimand, so calling
    it once per subset would resample different clusters for the two and the
    difference would not be paired. Its contract is kept -- one RNG per
    (seed, estimand, stratum), canonical order, clusters resampled with
    replacement -- but one resample feeds both means.
    """
    rows = cells[("cvqa", "bn", "C1", "correct")]["rows"]
    strata: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for i, v in sorted(values.items()):
        if i not in labels:
            continue
        stratum = f"bn:{rows[i]['subset']}"
        cluster = str(rows[i]["image_id"])
        strata.setdefault(stratum, {}).setdefault(cluster, []).append((labels[i], v))
    if not strata:
        raise ValueError("no labelled items to compare")
    sums = [[0.0, 0, 0.0, 0] for _ in range(B)]
    for s in sorted(strata):
        clusters = strata[s]
        keys = sorted(clusters)
        rnd = rng_for(seed, estimand, s)
        for b in range(B):
            for _ in range(len(keys)):
                for name, value in clusters[keys[rnd.randrange(len(keys))]]:
                    if name == first:
                        sums[b][0] += value
                        sums[b][1] += 1
                    elif name == second:
                        sums[b][2] += value
                        sums[b][3] += 1
    samples = [
        100.0 * (a / na - c / nc)
        for a, na, c, nc in sums
        if na and nc
    ]
    if len(samples) < B // 2:
        raise ValueError(
            f"{estimand}: only {len(samples)} of {B} resamples contain both subsets; "
            "one of them is too small to compare"
        )
    counts = {
        name: sum(1 for i in values if labels.get(i) == name) for name in (first, second)
    }
    flat = {
        name: [v for i, v in values.items() if labels.get(i) == name]
        for name in (first, second)
    }
    return {
        "estimate": 100.0 * (
            sum(flat[first]) / len(flat[first]) - sum(flat[second]) / len(flat[second])
        ),
        "ci95": [pct(samples, 0.025), pct(samples, 0.975)],
        "lb5": pct(samples, 0.05),
        "ub95": pct(samples, 0.95),
        "items": counts,
        "resamples_used": len(samples),
    }


def analyse(cells, labels, B=4000, seed=0, overlap=None):
    if B <= 0:
        raise ValueError("bootstrap count must be positive")
    validate_grid(cells)
    eps = {
        arm: {
            **{t: endpoints(cells, "cvqa", t, arm) for t in CVQA_TARGETS},
        }
        for arm in ARMS
    }
    xgqa = {arm: {"bn": endpoints(cells, "xgqa", "bn", arm)} for arm in ARMS}

    report = {
        "spec_sha": SPEC_SHA,
        "boot": B,
        "seed": seed,
        "arms": ARMS,
        "difference": "CG50 minus C1, except retention which is C1 minus CG50, as declared",
        "inference": "one training seed per arm: conditional on these trajectories, no training variance",
        "subset_counts": {s: sum(1 for v in labels.values() if v == s) for s in SUBSETS},
        "readings": {},
    }

    # Reading 2, exactly as registered on 2026-09-26 and left unchanged on 10-08.
    gain = interval(cells, "cvqa", contrast(eps, PRIMARY, GAIN, "grounding"), B, seed, "CG50:primary:grounding")
    utility = interval(cells, "cvqa", contrast(eps, PRIMARY, GAIN, "U"), B, seed, "CG50:primary:U")
    retention = interval(
        cells, "xgqa", contrast(xgqa, ("bn",), RETENTION, "U"), B, seed, "CG50:xgqa-bn:retention"
    )
    report["readings"]["2_zero_shot_transfer"] = {
        "panel": "cvqa jv/mn/ga pooled",
        "grounding_gain": gain,
        "utility_gain": utility,
        "source_retention_xgqa_bn": retention,
        "passes": bool(gain["lb5"] > 0 and retention["ub95"] < DELTA),
        "condition": "LB5(dGround(CG50)-dGround(C1)) > 0 and UB95(U(C1)-U(CG50)) < 1.0",
    }

    # Reading 1, on bn, by what the reference's question did for each item.
    bn = {"panel": "cvqa bn", "full": {}, "subsets": {}}
    for endpoint in ("U", "grounding"):
        values = contrast(eps, ("bn",), GAIN, endpoint)
        bn["full"][endpoint] = interval(cells, "cvqa", values, B, seed, f"CG50:bn:{endpoint}")
        for name in SUBSETS:
            subset = {i: v for i, v in values["bn"].items() if labels.get(i) == name}
            if not subset:
                continue
            bn["subsets"].setdefault(name, {})[endpoint] = interval(
                cells, "cvqa", {"bn": subset}, B, seed, f"CG50:bn:{name}:{endpoint}"
            )
    # A reader must always produce a report: when one subset is empty or too
    # small for a joint resample, the difference is recorded as unavailable and
    # the gate fails on its own terms rather than the run dying here.
    try:
        other_minus_shortcut = subset_mean_difference(
            cells,
            contrast(eps, ("bn",), GAIN, "U")["bn"],
            labels,
            "other",
            "shortcut",
            B,
            seed,
            "CG50:bn:other-minus-shortcut:U",
        )
    except ValueError as exc:
        other_minus_shortcut = {"unavailable": str(exc)}
    bn["other_minus_shortcut_U"] = other_minus_shortcut
    other = bn["subsets"].get("other", {}).get("U")
    bn["passes"] = bool(
        other
        and other["lb5"] > 0
        and other_minus_shortcut.get("estimate", float("-inf")) >= 0
    )
    bn["condition"] = (
        "LB5(dU on `other`) > 0 and the gain not concentrated in `shortcut`; "
        "the second half is read as other-minus-shortcut >= 0, an operationalisation "
        "fixed at read time, not on 2026-10-08"
    )
    report["readings"]["1_bn_country_matched"] = bn

    # The answer-set objection, decomposed. `cg_entity_overlap.py` measured that
    # 10.49% of CVQA-bn gold answers sit in CG50's training rows against ~0% on
    # the zero-shot targets. A share bounds the contribution of memorised answers
    # -- it cannot exceed the share -- but only the split measures it, and the
    # number the paper needs is the gain on the items whose answer the arm never
    # saw.
    if overlap:
        groups = {"overlap", "clean"}
        unknown = sorted(set(overlap.values()) - groups)
        if unknown:
            raise ValueError(f"overlap labels must be overlap/clean, got {unknown}")
        split = {"counts": {g: sum(1 for v in overlap.values() if v == g) for g in sorted(groups)}}
        for endpoint in ("U", "grounding"):
            values = contrast(eps, ("bn",), GAIN, endpoint)
            for g in sorted(groups):
                subset = {i: v for i, v in values["bn"].items() if overlap.get(i) == g}
                if subset:
                    split.setdefault(g, {})[endpoint] = interval(
                        cells, "cvqa", {"bn": subset}, B, seed, f"CG50:bn:{g}:{endpoint}"
                    )
        try:
            split["overlap_minus_clean_U"] = subset_mean_difference(
                cells,
                contrast(eps, ("bn",), GAIN, "U")["bn"],
                overlap,
                "overlap",
                "clean",
                B,
                seed,
                "CG50:bn:overlap-minus-clean:U",
            )
        except ValueError as exc:
            split["overlap_minus_clean_U"] = {"unavailable": str(exc)}
        clean = split.get("clean", {}).get("U")
        split["survives_the_answer_set_objection"] = bool(clean and clean["lb5"] > 0)
        split["reading"] = (
            "the gain on `clean` items is the part no memorised answer can explain; "
            "exposure on `overlap` items may still require the image, so a positive "
            "overlap-minus-clean difference is not by itself evidence of memorisation"
        )
        report["readings"]["1b_answer_set_overlap"] = split

    # Contrasts answer the gates; a table needs the levels they are differences
    # of, so both arms are reported on both panels.
    report["levels"] = {}
    for name, b, targets in (("cvqa-bn", "cvqa", ("bn",)), ("cvqa-primary", "cvqa", PRIMARY)):
        report["levels"][name] = {
            arm: {
                ep: interval(
                    cells, b, {t: eps[arm][t][ep] for t in targets}, B, seed, f"CG50:{name}:{arm}:{ep}"
                )
                for ep in ("U", "grounding", "gray")
            }
            for arm in ARMS
        }
    report["levels"]["xgqa-bn"] = {
        arm: {
            ep: interval(
                cells, "xgqa", {"bn": xgqa[arm]["bn"][ep]}, B, seed, f"CG50:xgqa-bn:{arm}:{ep}"
            )
            for ep in ("U", "grounding", "gray")
        }
        for arm in ARMS
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--results-dir")
    ap.add_argument("--reference-with", required=True, help="the reference's with-question arm")
    ap.add_argument("--reference-blind", required=True, help="its question-blind arm")
    ap.add_argument("--output", required=True)
    ap.add_argument("--overlap-labels", help="per-item overlap/clean labels from cg_entity_overlap.py")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cells = load_cells(a.submission, a.results_dir, block="C")
    labels = labels_for_panel(cells, a.reference_with, a.reference_blind)
    overlap = None
    if a.overlap_labels:
        with open(a.overlap_labels, encoding="utf-8") as handle:
            overlap = json.load(handle).get("bn")
        if not overlap:
            raise SystemExit(f"{a.overlap_labels} carries no bn labels")
    report = analyse(cells, labels, a.boot, a.seed, overlap)
    report["submission_sha256"] = file_sha(a.submission)
    report["reference"] = {
        "with_question": str(a.reference_with),
        "question_blind": str(a.reference_blind),
    }
    atomic_json(a.output, report)
    print(
        json.dumps(
            {
                "reading_1_bn": report["readings"]["1_bn_country_matched"]["passes"],
                "reading_2_transfer": report["readings"]["2_zero_shot_transfer"]["passes"],
                "survives_answer_set_objection": report["readings"]
                .get("1b_answer_set_overlap", {})
                .get("survives_the_answer_set_objection"),
                "subset_counts": report["subset_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
