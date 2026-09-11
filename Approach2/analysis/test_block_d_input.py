"""Synthetic tests for block_d_input.py: damage, per-item endpoints, and the
guarantee that what the adapter emits is accepted by block_d.analyse."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_d import SCHEMA_VERSION, analyse  # noqa: E402
from block_d_input import DONORS, NLLB, damage, endpoints  # noqa: E402


def expect_exit(needle, fn):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), f"expected abort mentioning {needle!r}, got {exc}"
        return
    raise AssertionError(f"expected abort mentioning {needle!r}")


def pairalign(path, label, margins):
    path.write_text(json.dumps({
        "label": label,
        "matrix": {"centered": {s: {t: {"margin": m} for t, m in row.items()}
                                for s, row in margins.items()}},
    }), encoding="utf-8")


def cell(rows):
    return {"manifest": {}, "rows": rows}


def main() -> None:
    targets = ["jv", "mn", "ga", "si"]

    # 1. damage = mean over targets of stage1 margin minus stage3 margin
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage1, stage3 = DONORS["bn"]
        pairalign(root / f"pairalign_{stage1}.json", stage1,
                  {"bn": {t: 0.80 for t in targets}})
        pairalign(root / f"pairalign_{stage3}.json", stage3,
                  {"bn": {"jv": 0.70, "mn": 0.60, "ga": 0.80, "si": 0.50}})
        d = damage(root, "bn", targets)
        assert abs(d["value"] - 0.15) < 1e-12, d          # (0.10+0.20+0.00+0.30)/4
        assert abs(d["per_target"]["si"] - 0.30) < 1e-12 and d["stage3_label"] == "stage3_bn_dcl"
        # a donor whose own checkpoints were never scored must say which to score
        expect_exit("pair_alignment.sh", lambda: damage(root, "ru", targets))
        # a missing source row in the matrix is not silently a zero
        pairalign(root / f"pairalign_{stage3}.json", stage3, {"id": {t: 0.1 for t in targets}})
        expect_exit("no centered margin", lambda: damage(root, "bn", targets))

    # 2. per-item endpoints: grounding is correct minus the mean of three shuffles
    def rows(flags):
        return {
            f"img{k}_{j}": {
                "subset": "('Javanese', 'Indonesia')", "image_id": f"img{k}", "correct": flags[k][j],
            }
            for k in range(len(flags)) for j in range(len(flags[k]))
        }

    cells = {
        ("cvqa", "jv", "D_bn", "correct"): cell(rows([[True, True], [True, False]])),
        ("cvqa", "jv", "D_bn", "shuffled0"): cell(rows([[True, False], [False, False]])),
        ("cvqa", "jv", "D_bn", "shuffled1"): cell(rows([[False, False], [False, False]])),
        ("cvqa", "jv", "D_bn", "shuffled2"): cell(rows([[False, False], [True, False]])),
    }
    out = endpoints(cells, "bn", "jv")
    subset = out["('Javanese', 'Indonesia')"]
    # item img0_0: correct 1, shuffles 1/0/0 -> 1 - 1/3
    assert abs(subset["img0"]["img0_0"]["ground"] - (1 - 1 / 3)) < 1e-12
    assert subset["img0"]["img0_0"]["utility"] == 1.0
    # item img1_1: correct 0, shuffles all 0 -> exactly 0, and utility 0
    assert subset["img1"]["img1_1"] == {"ground": 0.0, "utility": 0.0}
    # item img1_0: correct 1, shuffles 0/0/1 -> 1 - 1/3
    assert abs(subset["img1"]["img1_0"]["ground"] - (1 - 1 / 3)) < 1e-12
    assert sorted(subset) == ["img0", "img1"]

    # a shuffled cell that lost an item must abort, never silently drop it
    broken = dict(cells)
    short = {k: v for k, v in rows([[True, True], [True, False]]).items() if k != "img1_0"}
    broken[("cvqa", "jv", "D_bn", "shuffled1")] = cell(short)
    expect_exit("missing from a shuffled cell", lambda: endpoints(broken, "bn", "jv"))
    expect_exit("no correct-image cell", lambda: endpoints({}, "bn", "jv"))

    # 3. what the adapter emits must be what the analysis accepts
    donors = ["bn", "id", "ru", "zh", "de", "pt", "ko"]
    spec = {
        "schema_version": SCHEMA_VERSION,
        "donors": {d: NLLB[d] for d in donors},
        "damage": {d: 0.01 * i for i, d in enumerate(donors)},
        "delta_u": 1.0,
        "cells": {},
    }
    for i, d in enumerate(donors):
        spec["cells"][d] = {}
        for t in targets:
            # more damage -> lower grounding, the association G3 tests
            spec["cells"][d][t] = {
                f"({t}, 'X')": {
                    f"{t}_img{k}": {
                        f"{t}_img{k}_0": {
                            "ground": 1.0 - 0.1 * i if k % 2 else 0.0,
                            "utility": 1.0 if k % 3 else 0.0,
                        }
                    }
                    for k in range(12)
                }
            }
    result = analyse(spec, B=100, seed=0)
    assert result["selected_donor"] == "bn", result["selected_donor"]
    assert result["correlation_defined"] and result["exact_one_sided_p"] < 0.10
    assert result["g3"] in {"pass", "fail"}
    assert result["items"] == 4 * 12, result["items"]      # 4 targets x 12 images x 1 question
    assert result["images"] == 4 * 12, result["images"]
    assert result["strata"] == 4, result["strata"]        # one subset per target

    print("block_d_input: OK (damage from the donor's own checkpoints, missing-scoring abort, "
          "per-item grounding, dropped-item abort, adapter output accepted by block_d.analyse)")


if __name__ == "__main__":
    main()
