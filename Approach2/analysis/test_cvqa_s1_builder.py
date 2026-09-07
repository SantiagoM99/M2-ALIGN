"""Synthetic fail-closed tests for ``build_cvqa_s1.py`` (stdlib only).

Run from any directory:

    python3 Approach2/analysis/test_cvqa_s1_builder.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_cvqa_s1 import (  # noqa: E402
    build_selected_rows,
    canonical_blob,
    canonical_query_hash,
    command_attach_pilot,
    compare_queries,
    id_query_hash,
    sha256_bytes,
    sha256_file,
)


def raw(row_id: str, subset: str, native: str, english: str, image: bytes = b"image") -> dict:
    return {
        "ID": row_id,
        "Subset": subset,
        "Question": native,
        "Translated Question": english,
        "Translated Options": ["one", "two"],
        "Label": 0,
        "image": image,
        "__image_column": "image",
    }


def expect_exit(needle: str, fn) -> None:
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, exc)
    else:
        raise AssertionError(f"expected SystemExit containing {needle!r}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    es_mx = "('Spanish', 'Mexico')"
    bn_in = "('Bengali', 'India')"
    selected = {
        "Spanish": {
            "language": "Spanish",
            "nllb": "spa_Latn",
            "items": 2,
            "subsets": [{"subset": es_mx, "country": "Mexico", "items": 2, "images": 1}],
        },
        "Bengali": {
            "language": "Bengali",
            "nllb": "ben_Beng",
            "items": 1,
            "subsets": [{"subset": bn_in, "country": "India", "items": 1, "images": 1}],
        },
    }
    source = [
        raw("100_1", es_mx, "segunda pregunta", "second question"),
        raw("300_0", bn_in, "বাংলা প্রশ্ন", "Bengali question"),
        raw("100_0", es_mx, "primera pregunta", "first question"),
    ]
    materialised: list[tuple[str, str, bytes]] = []

    def sink(image_id: str, question_id: str, value) -> str:
        materialised.append((image_id, question_id, value))
        return "synthetic"

    rows, stats = build_selected_rows(source, selected, {"Spanish"}, sink)
    assert [row["id"] for row in rows["Spanish"]] == ["100_0", "100_1"]
    assert rows["Spanish"][0]["query"] == "primera pregunta"
    assert rows["Spanish"][0]["english_query"] == "first question"
    assert rows["Spanish"][0]["image_id"] == "100"
    assert rows["Spanish"][0]["nllb_lang_tag"] == "spa_Latn"
    assert [item[1] for item in materialised] == ["100_1", "100_0"]
    assert stats["Spanish"]["items"] == 2
    assert stats["Spanish"]["images"] == 1

    reversed_rows = list(reversed(rows["Spanish"]))
    assert canonical_query_hash(rows["Spanish"]) == canonical_query_hash(reversed_rows)
    assert id_query_hash(rows["Spanish"]) == id_query_hash(reversed_rows)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        current = root / "current.jsonl"
        write_jsonl(current, reversed_rows)
        comparison = compare_queries("Spanish", rows["Spanish"], current)
        assert comparison["passed"]
        assert comparison["rebuilt_canonical_query_sha256"] == comparison["existing_canonical_query_sha256"]

        changed = [dict(row) for row in reversed_rows]
        changed[0]["query"] = "wrong language or changed query"
        write_jsonl(current, changed)
        comparison = compare_queries("Spanish", rows["Spanish"], current)
        assert not comparison["passed"]
        assert comparison["mismatched_queries"] == 1
        write_jsonl(current, reversed_rows)

        built = root / "spanish.jsonl"
        built_jp = root / "japanese.jsonl"
        built_jp.write_text("synthetic-jp\n", encoding="utf-8")
        inventory = root / "inventory.json"
        images_root = root / "images"
        canonical_image = images_root / "by_image" / "100.jpg"
        predictions = root / "predictions.jsonl"
        summary_path = Path(str(predictions) + ".summary.json")
        checkpoint = root / "mapping.bin"
        report_path = root / "report.json"
        built.write_text("synthetic\n", encoding="utf-8")
        inventory.write_text("{}\n", encoding="utf-8")
        canonical_image.parent.mkdir(parents=True)
        canonical_image.write_bytes(b"canonical-image")
        predictions.write_text("{}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        summary_path.write_text(json.dumps({
            "benchmark": "cvqa",
            "scored": 2,
            "skipped": 0,
            "accuracy": 0.5,
            "data_path": str(built_jp),
            "ckpt": str(checkpoint),
            "blind": False,
        }), encoding="utf-8")
        limited_summary = root / "predictions_limit1.jsonl.summary.json"
        limited_summary.write_text(json.dumps({
            "benchmark": "cvqa",
            "scored": 1,
            "skipped": 0,
            "accuracy": 1.0,
            "data_path": str(built_jp),
            "ckpt": str(checkpoint),
            "blind": False,
        }), encoding="utf-8")
        base_report = {
            "kind": "cvqa_s1_prefreeze_data_audit",
            "git": {"head": "a" * 40, "dirty": False},
            "comparisons": [{
                "passed": True,
                "existing_path": str(current),
                "existing_file_sha256": sha256_file(current),
            }],
            "units": {
                "Spanish": {
                    "items": 2,
                    "images_extracted": True,
                    "output_path": str(built),
                    "output_sha256": sha256_file(built),
                },
                "Japanese": {
                    "items": 2,
                    "images_extracted": True,
                    "output_path": str(built_jp),
                    "output_sha256": sha256_file(built_jp),
                },
            },
            "images": {
                "root": str(images_root),
                "unique_images": 1,
                "canonical_files_written": 1,
                "image_manifest_sha256": sha256_bytes(canonical_blob([
                    ("100", sha256_file(canonical_image)),
                ])),
            },
            "inventory": {
                "path": str(inventory),
                "sha256": sha256_file(inventory),
                "confirmatory_panel_items": 10,
                "confirmatory_panel_units": ["Spanish"],
            },
            "timing_seconds": {"build_total": 5.0},
        }
        report_path.write_text(json.dumps(base_report), encoding="utf-8")
        attach_args = SimpleNamespace(
            report=str(report_path),
            eval_summary=str(summary_path),
            elapsed_seconds="4.0",
            unit="Japanese",
            extraction_unit="Spanish",
            limited_eval_summary=str(limited_summary),
            limited_elapsed_seconds="3.0",
            donor="bn",
            condition="correct",
            repo_root=None,
            checkpoint=str(checkpoint),
            vis_layers="9,18,-1",
            donors=7,
            conditions=5,
        )
        command_attach_pilot(attach_args)
        attached = json.loads(report_path.read_text(encoding="utf-8"))
        assert attached["pilot"]["seconds_per_item"] == 1.0, attached["pilot"]["seconds_per_item"]
        assert attached["pilot"]["load_seconds"] == 2.0, attached["pilot"]["load_seconds"]
        assert attached["pilot"]["extraction_unit"] == "Spanish"
        # per-item 1.0 s x 10 panel items x 35 + load 2.0 s x 35 invocations (one panel unit) = 420 s
        assert abs(attached["pilot"]["estimate"]["donor_confirmation_gpu_hours"] - 420 / 3600) < 1e-3  # stored rounded to 3 decimals
        assert attached["pilot"]["checkpoint_sha256"]
        assert attached["pilot"]["predictions_sha256"]

        # a confirmatory-panel unit must never be the timing unit
        report_path.write_text(json.dumps(base_report), encoding="utf-8")
        panel_args = SimpleNamespace(**{**vars(attach_args), "unit": "Spanish", "extraction_unit": "Spanish"})
        expect_exit("confirmatory-panel unit", lambda: command_attach_pilot(panel_args))

        # without a limited run the overhead folds into the per-item cost
        report_path.write_text(json.dumps(base_report), encoding="utf-8")
        single_args = SimpleNamespace(**{**vars(attach_args), "limited_eval_summary": None, "limited_elapsed_seconds": None})
        command_attach_pilot(single_args)
        single = json.loads(report_path.read_text(encoding="utf-8"))
        assert single["pilot"]["seconds_per_item"] == 2.0 and single["pilot"]["load_seconds"] == 0.0

        warm_report = dict(base_report)
        warm_report["images"] = {"unique_images": 1, "canonical_files_written": 0}
        report_path.write_text(json.dumps(warm_report), encoding="utf-8")
        expect_exit("not a cold extraction", lambda: command_attach_pilot(attach_args))

    duplicate = source + [raw("300_0", bn_in, "duplicate", "duplicate")]
    expect_exit(
        "duplicate CVQA question id",
        lambda: build_selected_rows(duplicate, selected, set(), None),
    )
    bad_native = list(source)
    bad_native[0] = raw("100_1", es_mx, "", "second question")
    expect_exit(
        "missing native Question",
        lambda: build_selected_rows(bad_native, selected, set(), None),
    )
    missing_image_reader = [dict(row) for row in source]
    missing_image_reader[0].pop("__image_column")
    expect_exit(
        "did not identify the image column",
        lambda: build_selected_rows(missing_image_reader, selected, {"Spanish"}, sink),
    )

    print("cvqa-s1-builder: OK (native query, schema, hashes, pilot provenance, two-point timing, panel-unit refusal, images, fail-closed guards)")


if __name__ == "__main__":
    main()
