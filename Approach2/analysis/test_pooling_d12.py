"""Synthetic tests for pooling_d12.py: arithmetic, pairing, xGQA shared clusters, verdict rule."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooling_d12 import analyse  # noqa: E402

N = 60


def write(root, bench, lang, tag, blind, correct_fn):
    name = f"eval_{bench}_{lang}{'_BLIND' if blind else ''}_{tag}.jsonl"
    with (root / name).open("w") as f:
        for k in range(N):
            image = f"img{k // 3}"
            item = f"{image}_{k % 3}" if bench == "cvqa" else f"q{k}"
            row = {"id": item, "correct": bool(correct_fn(k))}
            if bench == "xgqa":
                row["image_id"] = image
            f.write(json.dumps(row) + "\n")


def build(root, lrl_gain, hrl_gain):
    """Independent arm right on items k < 30; joint adds `gain` items per language."""
    for bench, langs, gain in (("cvqa", ("jv", "mn", "ga"), lrl_gain), ("xgqa", ("de", "ru", "zh"), hrl_gain),
                               ("cvqa", ("ru", "zh"), 0)):
        for lang in langs:
            write(root, bench, lang, "v4r_tf5", False, lambda k: k < 30)
            write(root, bench, lang, "vj_tf5", False, lambda k, g=gain: k < 30 + g)
            for tag in ("v4r_tf5", "vj_tf5"):
                write(root, bench, lang, tag, True, lambda k: k < 10)


def expect_exit(needle, fn):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), exc
        return
    raise AssertionError(f"expected abort mentioning {needle!r}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        build(root, lrl_gain=18, hrl_gain=0)
        r = analyse(root, "vj_tf5", "v4r_tf5", B=300)
        lift = r["panels"]["lrl-cvqa"]["U"]["pooled"]
        assert abs(lift["estimate"] - 100 * 18 / N) < 1e-9, lift["estimate"]
        assert r["panels"]["hrl-xgqa"]["U"]["pooled"]["estimate"] == 0.0
        # xGQA translations share images: 20 clusters, not 60
        assert r["panels"]["hrl-xgqa"]["U"]["pooled"]["image_clusters"] == N // 3, r["panels"]["hrl-xgqa"]["U"]["pooled"]
        # gray endpoint: correct minus gray, both arms differ only through correct
        assert abs(r["panels"]["lrl-cvqa"]["gray"]["pooled"]["estimate"] - 100 * 18 / N) < 1e-9
        assert r["verdict"]["D12"] == "supported", r["verdict"]

        build(root, lrl_gain=0, hrl_gain=0)
        assert analyse(root, "vj_tf5", "v4r_tf5", B=200)["verdict"]["D12"] == "refuted"

        build(root, lrl_gain=18, hrl_gain=18)
        v = analyse(root, "vj_tf5", "v4r_tf5", B=300)["verdict"]
        assert v["uniform_lift"] and v["D12"] == "refuted", v

        build(root, lrl_gain=18, hrl_gain=6)
        v = analyse(root, "vj_tf5", "v4r_tf5", B=300)["verdict"]
        assert v["lift_jv_mn_ga_LB5_gt_0"] and not v["flat_de_ru_zh_within_1"] and v["D12"] == "inconclusive", v

        (root / "eval_cvqa_mn_vj_tf5.jsonl").unlink()
        expect_exit("missing eval_cvqa_mn_vj_tf5.jsonl", lambda: analyse(root, "vj_tf5", "v4r_tf5", B=10))
        build(root, lrl_gain=0, hrl_gain=0)
        lines = (root / "eval_xgqa_de_vj_tf5.jsonl").read_text().splitlines()[:-1]
        (root / "eval_xgqa_de_vj_tf5.jsonl").write_text("\n".join(lines) + "\n")
        expect_exit("different items", lambda: analyse(root, "vj_tf5", "v4r_tf5", B=10))

    print("pooling_d12: OK (lift arithmetic, xGQA shared-image clusters, gray endpoint, "
          "supported / refuted / uniform-refuted / inconclusive, missing-file and pairing aborts)")


if __name__ == "__main__":
    main()
