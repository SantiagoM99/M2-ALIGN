"""Build the exact frozen A/B cell grid and deterministic maps without GPU.

Panels JSON: {"xgqa": {"bn": {"data": "...", "images": "..."}, ...},
              "cvqa": {"jv": {"data": "...", "images": "..."}, ...}}
A needs xGQA bn/de/ko/en and CVQA jv/mn/ga/si. B needs xGQA bn/id and
CVQA jv/mn/ga/si/bn. Use build_cvqa_s1.py outputs with explicit subsets.
"""

import argparse
from pathlib import Path
from s1_contract import ROOT, SPEC_SHA, atomic_json, read_json, read_rows, shuffle_maps


def build(a):
    panels = read_json(a.panels)
    ck = lambda name: str(
        (Path(a.checkpoints) / name / "mapping/pytorch_model.bin").resolve()
    )
    txt = {
        "T1_bn": ck("stage1"),
        "T1_id": ck("stage1_id"),
        "T3_bn": ck("stage3_bn_dcl"),
        "T3_id": ck("stage3_id_v4"),
    }
    vis = {
        "V2": ck("stage2_dc_llava"),
        "V3_bn": ck("stage3_bn_dcl"),
        "V3_id": ck("stage3_id_v4"),
    }
    cells = []
    parity = []
    registered = {}

    def add(
        b, t, arm, condition, tc, vc, no_text=False, instruction=False, english=False
    ):
        panel = panels[b][t]
        key = (b, t)
        if key not in registered:
            maps = shuffle_maps(read_rows(panel["data"]), b)
            registry = ROOT / "Approach2/shuffle" / f"{b}_{t}.hashes.json"
            atomic_json(
                registry,
                {
                    "panel": f"{b}_{t}",
                    "benchmark": b,
                    "id_universe_sha256": maps[0]["id_universe_sha256"],
                    "maps": {str(m["seed"]): m["sha256"] for m in maps},
                },
            )
            for m in maps:
                atomic_json(
                    Path(a.maps_dir) / f"shuffle_{b}_{t}_seed{m['seed']}.json", m
                )
            registered[key] = registry
        cid = f"s1_{a.block}_{b}_{t}_{arm}_{condition}"
        argv = [
            "--s1",
            "--experiment-id",
            cid,
            "--arm",
            arm,
            "--source",
            "bn" if a.block == "A" else arm,
            "--target",
            t,
            "--data-path",
            str(Path(panel["data"]).resolve()),
            "--images-dir",
            str(Path(panel["images"]).resolve()),
            "--txt-ckpt",
            tc,
            "--vis-ckpt",
            vc,
            "--vis-layers",
            "9,18,-1",
            "--local-files-only",
            "--shuffle-registry",
            str(registered[key]),
            "--output-path",
            str((Path(a.results) / f"{cid}.jsonl").resolve()),
            "--hypothesis",
            "S1 P1/P2: input necessity and grounding"
            if a.block == "A"
            else "S1 P3/P4: functional branch localisation and co-adaptation",
            "--prediction",
            "A1 grounding > 0 on each panel; A2 necessity classified by frozen NI regions"
            if a.block == "A"
            else "D_T > 0 and D_T-D_V > 0; P4 two-sided",
            "--seed",
            "42",
        ]  # historical source checkpoint seed, not a new training run
        if no_text:
            argv += ["--no-text-branch"]
        if instruction:
            argv += ["--prompt-mode", "instruction"]
        if english:
            argv += ["--question-field", "english_query"]
        if condition == "gray":
            argv += ["--blind"]
        elif condition == "no-image":
            argv += ["--no-image"]
        elif condition.startswith("shuffled"):
            argv += [
                "--shuffle-map",
                str(
                    (
                        Path(a.maps_dir) / f"shuffle_{b}_{t}_seed{condition[-1]}.json"
                    ).resolve()
                ),
            ]
        cells.append({"id": cid, "benchmark": b, "argv": argv})
        return cid

    conditions = ["correct", "shuffled0", "shuffled1", "shuffled2", "gray"]
    if a.block == "A":
        for b, targets in [
            ("xgqa", ["bn", "de", "ko"]),
            ("cvqa", ["jv", "mn", "ga", "si"]),
        ]:
            for t in targets:
                for arm in (
                    ["A1", "A2", "A3"] if b == "xgqa" else ["A1", "A2", "A3", "A4"]
                ):
                    for c in conditions + ["no-image"]:
                        add(
                            b,
                            t,
                            arm,
                            c,
                            txt["T3_bn"],
                            vis["V3_bn"],
                            arm in ("A2", "A4"),
                            arm == "A3",
                            arm == "A4",
                        )
        for c in conditions + ["no-image"]:
            add("xgqa", "en", "A4", c, txt["T3_bn"], vis["V3_bn"], True)
    else:
        if not a.block_a_report:
            raise ValueError("Block B requires --block-a-report with G0 passing")
        for t in ["jv", "mn", "ga", "si", "bn"]:
            for tn, tp in txt.items():
                for vn, vp in vis.items():
                    for c in conditions:
                        cid = add("cvqa", t, tn + "__" + vn, c, tp, vp)
                        # Parity references must come from the SAME checkpoint the
                        # cell evaluates. Legacy `eval_cvqa_{t}_zsbn*` on jv/mn/ga/si
                        # were produced by stage3_bn_v4, not stage3_bn_dcl (= V3_bn),
                        # and no id-donor result exists on CVQA-bn; those cells have
                        # no reference and are not parity cells. That leaves bn/bn on
                        # bn (`eval_cvqa_bn_dcl*`) and id/id on the four targets.
                        donor = tn[3:] if tn in ("T3_bn", "T3_id") else None
                        if (
                            donor
                            and vn == "V3_" + donor
                            and c in ("correct", "gray")
                            and ((donor == "bn") == (t == "bn"))
                        ):
                            blind = "_BLIND" if c == "gray" else ""
                            tag = f"bn_dcl{blind}" if donor == "bn" else f"{t}_zsid{blind}"
                            parity.append(
                                {
                                    "cell_id": cid,
                                    "donor": donor,
                                    "expected_path": str(
                                        (
                                            Path(a.legacy_results)
                                            / f"eval_cvqa_{tag}.jsonl"
                                        ).resolve()
                                    ),
                                }
                            )
            for tn, vn in [("T3_bn", "V3_bn"), ("T3_id", "V3_id"), ("T1_bn", "V2")]:
                for c in conditions[:-1]:
                    add("cvqa", t, tn + "__" + vn + "__noX", c, txt[tn], vis[vn], True)
        for t in ["bn", "id"]:
            for tn in ["T3_bn", "T3_id"]:
                for vn in ["V3_bn", "V3_id"]:
                    for c in ["correct", "gray"]:
                        add("xgqa", t, tn + "__" + vn, c, txt[tn], vis[vn])
    if a.block == "B":
        cells = falsifier_first(cells)
    plan = {"spec_sha": SPEC_SHA, "block": a.block, "cells": cells, "parity": parity}
    if a.block_a_report:
        plan["block_a_report"] = str(Path(a.block_a_report).resolve())
    atomic_json(a.output, plan)
    print(
        f"{len(cells)} cells; maps generated. Commit the hash registries before s1_submit.py."
    )


# The eight cells P3 needs for D_T and D_V, on the three primary targets.
# D_T contrasts T1 against T3 with vision held at the shared pre-VQA V2;
# D_V contrasts V3_bn against V3_id with text held at each lineage's stage 1.
P3_CORE = {("T1_bn", "V2"), ("T1_id", "V2"), ("T3_bn", "V2"), ("T3_id", "V2"),
           ("T1_bn", "V3_bn"), ("T1_bn", "V3_id"), ("T1_id", "V3_bn"), ("T1_id", "V3_id")}
PRIMARY_TARGETS = ("jv", "mn", "ga")


def falsifier_first(cells):
    """Emit P3/P4's own cells before the rest of the factorial.

    Same grid, same criteria: only the order changes. Block A bounded the text
    branch's whole inference contribution at roughly zero on CVQA, so D_T is
    the statistic most likely to end the main hypothesis, and it needs eight
    of the twelve combinations on three of the five targets. Running those
    first spends ~4 h before the remaining ~19 h of controls instead of after.
    """
    def primary(cell):
        parts = cell["id"].split("_")
        arm = cell["argv"][cell["argv"].index("--arm") + 1]
        target = cell["argv"][cell["argv"].index("--target") + 1]
        pair = tuple(arm.split("__")[:2])
        return not (
            len(arm.split("__")) == 2
            and pair in P3_CORE
            and target in PRIMARY_TARGETS
        )

    return sorted(cells, key=primary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panels", required=True)
    p.add_argument("--block", choices=["A", "B"], required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--maps-dir", default=str(ROOT / "evaluation"))
    p.add_argument("--legacy-results", default=str(ROOT / "Approach2/results"))
    p.add_argument("--block-a-report")
    p.add_argument("--output", required=True)
    build(p.parse_args())
