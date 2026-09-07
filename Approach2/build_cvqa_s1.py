"""Build fail-closed CVQA panels for S1 directly from local parquet files.

This script has two jobs:

1. materialise one or more CVQA language units with the native ``Question``
   as ``query`` and the embedded parquet image as the source of truth; and
2. compare the canonical ``id -> query`` hash of a rebuilt existing unit
   against the JSONL already used on the cluster.

The pre-freeze SLURM job builds Spanish (the largest confirmatory unit),
rebuilds Bengali for the query audit, runs one timed Spanish evaluation, and
then attaches its timing to the report written here.  Compute nodes are
offline, so ``build`` accepts only local parquet files and never follows
``Image Source`` URLs.

Examples
--------

Build Spanish and Bengali, but extract images only for Spanish::

    python3 Approach2/build_cvqa_s1.py build \
      --parquet '/scratch/.../cvqa/*.parquet' \
      --unit Spanish --unit Bengali --extract-images-for Spanish \
      --compare-existing Bengali=/scratch/.../cvqa/bn.jsonl \
      --output-dir /scratch/.../cvqa_s1/data \
      --images-dir /scratch/.../cvqa_s1/images \
      --report Approach2/audits/cvqa_s1_prefreeze.json

Attach the elapsed time of a completed evaluator invocation::

    python3 Approach2/build_cvqa_s1.py attach-pilot \
      --report Approach2/audits/cvqa_s1_prefreeze.json \
      --repo-root . \
      --eval-summary /scratch/.../eval_cvqa_spanish.jsonl.summary.json \
      --elapsed-seconds 1234.5 --unit Spanish --donor bn --condition correct \
      --checkpoint Approach2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin \
      --vis-layers 9,18,-1
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable, Iterable, NoReturn


ID_RE = re.compile(r"^(?P<image>\d+)_(?P<question>\d+)$")
SCHEMA_VERSION = 1


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def canonical_blob(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower().replace("_", " ")).strip("_")
    if not value:
        fail(f"cannot derive a filename from unit {text!r}")
    return value


def normalise_unit(text: str) -> str:
    return " ".join(text.replace("_", " ").split()).casefold()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"{path}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing JSONL file: {path}")
    with handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{lineno}: invalid JSON: {exc}")
            if not isinstance(row, dict):
                fail(f"{path}:{lineno}: expected a JSON object")
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                fail(f"{path}:{lineno}: missing id")
            if row_id in seen:
                fail(f"{path}:{lineno}: duplicate id {row_id}")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        fail(f"{path}: no rows")
    return rows


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")


def canonical_query_hash(rows: list[dict]) -> str:
    """Hash the query column after canonical ordering by id.

    IDs are deliberately not part of this hash: the separately checked ID
    universe defines the order, making this exactly a query-column hash while
    remaining invariant to parquet shard or JSONL row order.
    """
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    return sha256_bytes(canonical_blob([row.get("query") for row in ordered]))


def id_query_hash(rows: list[dict]) -> str:
    ordered = sorted((str(row["id"]), row.get("query")) for row in rows)
    return sha256_bytes(canonical_blob(ordered))


def id_universe_hash(rows: list[dict]) -> str:
    return sha256_bytes(canonical_blob(sorted(str(row["id"]) for row in rows)))


def load_units(inventory_path: Path, requested: list[str]) -> tuple[dict, dict[str, dict]]:
    inventory = read_json(inventory_path)
    all_units = inventory.get("units")
    if not isinstance(all_units, list):
        fail(f"{inventory_path}: missing units list")
    by_name: dict[str, dict] = {}
    for unit in all_units:
        if not isinstance(unit, dict) or not unit.get("language"):
            fail(f"{inventory_path}: malformed unit")
        key = normalise_unit(str(unit["language"]))
        if key in by_name:
            fail(f"{inventory_path}: duplicate unit {unit['language']!r}")
        by_name[key] = unit
    selected: dict[str, dict] = {}
    for raw_name in requested:
        key = normalise_unit(raw_name)
        if key not in by_name:
            fail(f"unit {raw_name!r} is absent from {inventory_path}")
        name = str(by_name[key]["language"])
        selected[name] = by_name[key]
    return inventory, selected


def resolve_parquets(values: list[str]) -> list[Path]:
    found: set[Path] = set()
    for raw in values:
        for token in [part.strip() for part in raw.split(",") if part.strip()]:
            candidate = Path(token).expanduser()
            if candidate.is_dir():
                matches = candidate.rglob("*.parquet")
            else:
                matches = (Path(match) for match in glob.glob(str(candidate), recursive=True))
            for match in matches:
                if match.is_file():
                    found.add(match.resolve())
    paths = sorted(found, key=lambda path: str(path))
    if not paths:
        fail("--parquet did not resolve to any local parquet file")
    return paths


def parse_subset(value: str) -> tuple[str, str]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        fail(f"invalid CVQA Subset value {value!r}: {exc}")
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        fail(f"invalid CVQA Subset value {value!r}: expected (language, country)")
    return str(parsed[0]), str(parsed[1])


def iter_parquet_rows(paths: list[Path], wanted_subsets: set[str]) -> Iterable[dict]:
    """Yield selected raw rows without decoding every image in the dataset."""
    try:
        import pyarrow.dataset as parquet_dataset
    except ModuleNotFoundError:
        fail("pyarrow is required to read CVQA parquet files (load arrow/21.0.0 on Alliance)")

    dataset = parquet_dataset.dataset([str(path) for path in paths], format="parquet")
    names = set(dataset.schema.names)
    image_column = "image" if "image" in names else ("Image" if "Image" in names else None)
    required = {"ID", "Subset", "Question", "Translated Question", "Translated Options", "Label"}
    missing = sorted(required - names)
    if missing:
        fail(f"CVQA parquet schema is missing columns: {missing}; found {sorted(names)}")
    if image_column is None:
        fail(f"CVQA parquet schema has no embedded image/Image column; found {sorted(names)}")
    columns = sorted(required) + [image_column]
    expression = parquet_dataset.field("Subset").isin(sorted(wanted_subsets))
    scanner = dataset.scanner(
        columns=columns,
        filter=expression,
        batch_size=32,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row["__image_column"] = image_column
            yield row


def image_bytes(value) -> bytes:
    """Return encoded image bytes from a parquet Image feature value."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)) and raw:
            return bytes(raw)
        value = value.get("path")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.is_file():
            return path.read_bytes()
        fail(f"embedded image points to a missing local path: {path}")
    # This path supports a datasets.Image-decoded PIL object if a caller uses
    # the functions programmatically.  PyArrow normally returns {bytes,path}.
    if hasattr(value, "save") and hasattr(value, "convert"):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    fail(f"unsupported or empty embedded image value: {type(value).__name__}")


def canonical_image_bytes(data: bytes, label: str) -> bytes:
    """Decode and deterministically encode an RGB JPEG.

    Comparing decoded/canonicalised content avoids treating two equivalent
    copies of one CVQA image as different merely because their source PNG/JPEG
    metadata or compression differs.  It also makes the ``.jpg`` contract
    with the existing evaluator truthful.
    """
    if not data:
        fail(f"{label}: empty embedded image")
    try:
        from PIL import Image
    except ModuleNotFoundError:
        fail("Pillow is required to validate CVQA images")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            output = io.BytesIO()
            rgb.save(output, format="JPEG", quality=95, optimize=False, progressive=False)
            return output.getvalue()
    except Exception as exc:  # noqa: BLE001 - the row must fail closed
        fail(f"{label}: invalid embedded image: {exc}")


class ImageMaterializer:
    """Write one canonical image and compatibility hardlinks per question."""

    def __init__(self, images_dir: Path) -> None:
        self.images_dir = images_dir
        self.by_image = images_dir / "by_image"
        self.by_image.mkdir(parents=True, exist_ok=True)
        self.hashes: dict[str, str] = {}
        self.files_written = 0
        self.files_reused = 0
        self.question_links_written = 0

    def __call__(self, image_id: str, question_id: str, value) -> str:
        data = canonical_image_bytes(image_bytes(value), question_id)
        digest = sha256_bytes(data)
        previous = self.hashes.get(image_id)
        if previous is not None and previous != digest:
            fail(f"image id {image_id} has different embedded bytes across questions")
        self.hashes[image_id] = digest

        canonical = self.by_image / f"{image_id}.jpg"
        if canonical.exists():
            if sha256_file(canonical) != digest:
                fail(f"cached canonical image hash mismatch: {canonical}")
            self.files_reused += 1
        else:
            atomic_write(canonical, data)
            self.files_written += 1

        # evaluate_cvqa.py currently resolves <question id>.jpg.  Keeping a
        # hardlink makes this pre-freeze pilot compatible without duplicating
        # bytes; the post-freeze evaluator can use by_image/<image id>.jpg.
        question_path = self.images_dir / f"{question_id}.jpg"
        if question_path.exists():
            if sha256_file(question_path) != digest:
                fail(f"cached question image hash mismatch: {question_path}")
        else:
            try:
                os.link(canonical, question_path)
            except OSError:
                shutil.copyfile(canonical, question_path)
            self.question_links_written += 1
        return digest

    def manifest(self) -> dict:
        ordered = sorted(self.hashes.items())
        return {
            "root": str(self.images_dir.resolve()),
            "unique_images": len(ordered),
            "image_manifest_sha256": sha256_bytes(canonical_blob(ordered)),
            "canonical_files_written": self.files_written,
            "canonical_files_reused": self.files_reused,
            "question_links_written": self.question_links_written,
        }


Materialize = Callable[[str, str, object], str]


def build_selected_rows(
    raw_rows: Iterable[dict],
    selected: dict[str, dict],
    extract_units: set[str],
    materialize: Materialize | None,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    subset_to_unit: dict[str, str] = {}
    expected: dict[str, dict[str, dict]] = {}
    nllb: dict[str, str] = {}
    for language, unit in selected.items():
        tag = str(unit.get("nllb") or "").strip()
        if not tag:
            fail(f"inventory unit {language} has no NLLB tag")
        nllb[language] = tag
        expected[language] = {}
        for subset in unit.get("subsets", []):
            subset_name = str(subset.get("subset", ""))
            if not subset_name:
                fail(f"inventory unit {language} has a malformed subset")
            if subset_name in subset_to_unit:
                fail(f"inventory subset {subset_name!r} belongs to two units")
            subset_to_unit[subset_name] = language
            expected[language][subset_name] = subset

    rows_by_unit = {language: [] for language in selected}
    images_by_subset: dict[str, set[str]] = {name: set() for name in subset_to_unit}
    seen_ids: set[str] = set()
    for raw in raw_rows:
        subset = str(raw.get("Subset", ""))
        language = subset_to_unit.get(subset)
        if language is None:
            fail(f"parquet scanner yielded an unrequested subset: {subset!r}")
        subset_language, _country = parse_subset(subset)
        if normalise_unit(subset_language) != normalise_unit(language):
            fail(f"subset {subset!r} disagrees with inventory language {language!r}")

        row_id = str(raw.get("ID", "")).strip()
        match = ID_RE.fullmatch(row_id)
        if not match:
            fail(f"{language}: invalid or missing CVQA ID {row_id!r}")
        if row_id in seen_ids:
            fail(f"duplicate CVQA question id across selected units: {row_id}")
        seen_ids.add(row_id)
        image_id = match.group("image")

        # Both question fields are emitted verbatim: the Bengali JSONL that the
        # existing cluster runs used keeps CVQA's leading/trailing whitespace
        # (6 of 286 items), and the pre-freeze audit compares queries byte for
        # byte against it (job 20441017 failed on exactly those 6 items).
        query = str(raw.get("Question") or "")
        english_query = str(raw.get("Translated Question") or "")
        raw_choices = raw.get("Translated Options")
        if not query.strip():
            fail(f"{row_id}: missing native Question")
        if not english_query.strip():
            fail(f"{row_id}: missing Translated Question")
        if not isinstance(raw_choices, (list, tuple)) or not raw_choices:
            fail(f"{row_id}: missing Translated Options")
        choices = [str(choice) for choice in raw_choices]
        if any(not choice.strip() for choice in choices):
            fail(f"{row_id}: empty translated choice")
        try:
            answer_index = int(raw.get("Label"))
        except (TypeError, ValueError):
            fail(f"{row_id}: invalid Label {raw.get('Label')!r}")
        if not 0 <= answer_index < len(choices):
            fail(f"{row_id}: Label {answer_index} outside {len(choices)} choices")

        images_by_subset[subset].add(image_id)
        if language in extract_units:
            if materialize is None:
                fail("image extraction requested without a materializer")
            image_column = raw.get("__image_column")
            if not image_column:
                fail(f"{row_id}: parquet reader did not identify the image column")
            materialize(image_id, row_id, raw.get(str(image_column)))

        rows_by_unit[language].append({
            "id": row_id,
            "image_id": image_id,
            "subset": subset,
            "language": language,
            "nllb_lang_tag": nllb[language],
            "query": query,
            "english_query": english_query,
            "choices": choices,
            "answer_index": answer_index,
            "source_dataset": "afaji/cvqa:test",
        })

    stats: dict[str, dict] = {}
    for language, rows in rows_by_unit.items():
        rows.sort(key=lambda row: str(row["id"]))
        unit_expected = selected[language]
        if len(rows) != int(unit_expected["items"]):
            fail(f"{language}: rebuilt {len(rows)} rows, inventory expects {unit_expected['items']}")
        by_subset: dict[str, int] = {}
        for row in rows:
            by_subset[row["subset"]] = by_subset.get(row["subset"], 0) + 1
        for subset, definition in expected[language].items():
            got_items = by_subset.get(subset, 0)
            got_images = len(images_by_subset[subset])
            if got_items != int(definition["items"]):
                fail(f"{language}/{subset}: rebuilt {got_items} items, inventory expects {definition['items']}")
            if got_images != int(definition["images"]):
                fail(f"{language}/{subset}: rebuilt {got_images} images, inventory expects {definition['images']}")
        stats[language] = {
            "items": len(rows),
            "images": len({row["image_id"] for row in rows}),
            "subsets": len(by_subset),
            "id_universe_sha256": id_universe_hash(rows),
            "canonical_query_sha256": canonical_query_hash(rows),
            "id_query_sha256": id_query_hash(rows),
        }
    return rows_by_unit, stats


def compare_queries(language: str, rebuilt: list[dict], existing_path: Path) -> dict:
    existing = read_jsonl(existing_path)
    rebuilt_map = {str(row["id"]): row["query"] for row in rebuilt}
    existing_map = {str(row["id"]): row.get("query") for row in existing}
    rebuilt_ids, existing_ids = set(rebuilt_map), set(existing_map)
    missing = sorted(rebuilt_ids - existing_ids)
    extra = sorted(existing_ids - rebuilt_ids)
    mismatched = sorted(
        row_id for row_id in rebuilt_ids & existing_ids
        if rebuilt_map[row_id] != existing_map[row_id]
    )
    result = {
        "unit": language,
        "existing_path": str(existing_path.resolve()),
        "existing_file_sha256": sha256_file(existing_path),
        "rebuilt_items": len(rebuilt),
        "existing_items": len(existing),
        "rebuilt_id_universe_sha256": id_universe_hash(rebuilt),
        "existing_id_universe_sha256": id_universe_hash(existing),
        "rebuilt_canonical_query_sha256": canonical_query_hash(rebuilt),
        "existing_canonical_query_sha256": canonical_query_hash(existing),
        "rebuilt_id_query_sha256": id_query_hash(rebuilt),
        "existing_id_query_sha256": id_query_hash(existing),
        "missing_ids": len(missing),
        "extra_ids": len(extra),
        "mismatched_queries": len(mismatched),
        "examples": {
            "missing": missing[:5],
            "extra": extra[:5],
            "mismatched": [
                {
                    "id": row_id,
                    "rebuilt": rebuilt_map[row_id],
                    "existing": existing_map[row_id],
                }
                for row_id in mismatched[:5]
            ],
        },
    }
    result["passed"] = not missing and not extra and not mismatched
    return result


def parse_comparisons(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            fail(f"--compare-existing expects UNIT=PATH, got {value!r}")
        unit, raw_path = value.split("=", 1)
        unit = unit.strip()
        if not unit or not raw_path.strip():
            fail(f"--compare-existing expects UNIT=PATH, got {value!r}")
        if normalise_unit(unit) in {normalise_unit(name) for name in out}:
            fail(f"duplicate --compare-existing unit {unit!r}")
        out[unit] = Path(raw_path).expanduser()
    return out


def git_state(repo_root: Path | None) -> dict:
    if repo_root is None:
        return {"head": None, "dirty": None}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot record git state from {repo_root}: {exc}")
    return {"head": head, "dirty": bool(status.strip())}


def write_unit(path: Path, rows: list[dict]) -> str:
    blob = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    atomic_write(path, blob)
    return sha256_bytes(blob)


def command_build(args) -> None:
    started = time.monotonic()
    inventory_path = Path(args.inventory)
    requested = list(args.unit)
    comparisons = parse_comparisons(args.compare_existing)
    requested.extend(
        name for name in comparisons
        if normalise_unit(name) not in {normalise_unit(x) for x in requested}
    )
    if not requested:
        fail("request at least one --unit or --compare-existing")
    inventory, selected = load_units(inventory_path, requested)

    extract_keys = {normalise_unit(name) for name in args.extract_images_for}
    selected_keys = {normalise_unit(name) for name in selected}
    unknown_extract = sorted(extract_keys - selected_keys)
    if unknown_extract:
        fail(f"--extract-images-for contains unselected units: {unknown_extract}")
    extract_units = {name for name in selected if normalise_unit(name) in extract_keys}

    parquet_paths = resolve_parquets(args.parquet)
    hash_started = time.monotonic()
    parquet_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in parquet_paths
    ]
    parquet_hash_seconds = time.monotonic() - hash_started

    images_dir = Path(args.images_dir)
    image_materializer = ImageMaterializer(images_dir) if extract_units else None
    wanted_subsets = {
        str(subset["subset"])
        for unit in selected.values()
        for subset in unit.get("subsets", [])
    }
    scan_started = time.monotonic()
    rows_by_unit, unit_stats = build_selected_rows(
        iter_parquet_rows(parquet_paths, wanted_subsets),
        selected,
        extract_units,
        image_materializer,
    )
    scan_build_seconds = time.monotonic() - scan_started

    output_dir = Path(args.output_dir)
    for language, rows in rows_by_unit.items():
        path = output_dir / f"{slug(language)}.jsonl"
        unit_stats[language]["output_path"] = str(path.resolve())
        unit_stats[language]["output_sha256"] = write_unit(path, rows)
        unit_stats[language]["images_extracted"] = language in extract_units

    comparison_results = []
    for requested_name, existing_path in comparisons.items():
        matches = [name for name in selected if normalise_unit(name) == normalise_unit(requested_name)]
        if len(matches) != 1:
            fail(f"cannot resolve comparison unit {requested_name!r}")
        comparison_results.append(compare_queries(matches[0], rows_by_unit[matches[0]], existing_path))

    panel_total = sum(int(unit["items"]) for unit in inventory.get("confirmatory_panel", []))
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cvqa_s1_prefreeze_data_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_state(Path(args.repo_root).resolve() if args.repo_root else None),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
            "confirmatory_panel_items": panel_total,
            "confirmatory_panel_units": [
                str(unit.get("language")) for unit in inventory.get("confirmatory_panel", [])
            ],
        },
        "parquet": parquet_records,
        "units": unit_stats,
        "images": image_materializer.manifest() if image_materializer else None,
        "comparisons": comparison_results,
        "timing_seconds": {
            "parquet_hash": round(parquet_hash_seconds, 6),
            "scan_validate_extract_write": round(scan_build_seconds, 6),
            "build_total": round(time.monotonic() - started, 6),
        },
        "pilot": None,
    }
    failed = [result for result in comparison_results if not result["passed"]]
    atomic_json(Path(args.report), report)
    for language, stats in sorted(unit_stats.items()):
        print(
            f"{language}: items={stats['items']} images={stats['images']} "
            f"query_sha256={stats['canonical_query_sha256']} -> {stats['output_path']}"
        )
    for result in comparison_results:
        print(
            f"query audit {result['unit']}: {'PASS' if result['passed'] else 'FAIL'} "
            f"rebuilt={result['rebuilt_canonical_query_sha256']} "
            f"existing={result['existing_canonical_query_sha256']}"
        )
    print(f"report: {args.report}")
    if failed:
        fail(f"{len(failed)} query comparison(s) failed; freeze remains blocked")


def command_attach_pilot(args) -> None:
    report_path = Path(args.report)
    report = read_json(report_path)
    if report.get("kind") != "cvqa_s1_prefreeze_data_audit":
        fail(f"{report_path}: not a CVQA S1 pre-freeze report")
    git = report.get("git")
    if not isinstance(git, dict) or not git.get("head") or git.get("dirty") is not False:
        fail(f"{report_path}: data build did not come from a clean committed checkout")
    if args.repo_root:
        current_git = git_state(Path(args.repo_root).resolve())
        if current_git["dirty"] or current_git["head"] != git["head"]:
            fail(
                f"{report_path}: current checkout does not match clean build commit "
                f"{git['head']}"
            )
        if sha256_file(Path(__file__).resolve()) != report.get("script_sha256"):
            fail(f"{report_path}: builder script hash differs from the data-build record")
    inventory = report.get("inventory", {})
    inventory_path = Path(str(inventory.get("path", "")))
    if not inventory_path.is_file() or sha256_file(inventory_path) != inventory.get("sha256"):
        fail(f"{report_path}: inventory file is missing or changed")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        fail(f"{report_path}: no existing-panel query audit was recorded")
    if any(not result.get("passed") for result in comparisons):
        fail(f"{report_path}: query audit did not pass")
    for comparison in comparisons:
        existing_path = Path(str(comparison.get("existing_path", "")))
        if (
            not existing_path.is_file()
            or sha256_file(existing_path) != comparison.get("existing_file_sha256")
        ):
            fail(f"{report_path}: compared query reference is missing or changed")
    if args.condition != "correct":
        fail("the pre-freeze runtime pilot must use condition=correct")
    panel_units = {normalise_unit(str(name)) for name in inventory.get("confirmatory_panel_units", [])}
    if normalise_unit(args.unit) in panel_units:
        fail(
            f"timing unit {args.unit!r} is a confirmatory-panel unit: evaluating a donor on it "
            "before the damage ranking is committed would burn a prospective G3 cell (S1 Block D)"
        )
    extraction_stats = report.get("units", {}).get(args.extraction_unit)
    if not extraction_stats or not extraction_stats.get("images_extracted"):
        fail(f"{report_path}: extraction unit {args.extraction_unit!r} was not built with image extraction")
    if args.donors <= 0 or args.conditions <= 0:
        fail("--donors and --conditions must be positive")
    summary_path = Path(args.eval_summary)
    summary = read_json(summary_path)
    if summary.get("benchmark") != "cvqa":
        fail(f"{summary_path}: expected benchmark=cvqa")
    try:
        elapsed = float(args.elapsed_seconds)
    except ValueError:
        fail(f"invalid --elapsed-seconds {args.elapsed_seconds!r}")
    if elapsed <= 0:
        fail("--elapsed-seconds must be positive")
    scored = int(summary.get("scored", 0))
    skipped = int(summary.get("skipped", 0))
    if scored <= 0:
        fail(f"{summary_path}: evaluator scored no items")
    if skipped:
        fail(f"{summary_path}: evaluator skipped {skipped} items; pilot is invalid")
    unit_stats = report.get("units", {}).get(args.unit)
    if not unit_stats:
        fail(f"{report_path}: no built unit named {args.unit!r}")
    if not unit_stats.get("images_extracted"):
        fail(f"{report_path}: {args.unit} was not built with embedded-image extraction")
    if scored != int(unit_stats["items"]):
        fail(f"pilot scored {scored} items but built unit has {unit_stats['items']}")
    images = report.get("images")
    if not isinstance(images, dict) or int(images.get("unique_images", 0)) <= 0:
        fail(f"{report_path}: no extracted-image manifest")
    if int(images.get("canonical_files_written", 0)) != int(images["unique_images"]):
        fail(
            f"{report_path}: pilot was not a cold extraction "
            f"({images.get('canonical_files_written')} new / {images['unique_images']} images)"
        )
    images_root = Path(str(images.get("root", "")))
    canonical_files = sorted((images_root / "by_image").glob("*.jpg"))
    actual_image_manifest = sorted(
        (path.stem, sha256_file(path)) for path in canonical_files
    )
    if (
        len(canonical_files) != int(images["unique_images"])
        or sha256_bytes(canonical_blob(actual_image_manifest))
        != images.get("image_manifest_sha256")
    ):
        fail(f"{report_path}: extracted canonical image set is missing or changed")

    for _language, stats in report.get("units", {}).items():
        output_path = Path(str(stats.get("output_path", "")))
        if not output_path.is_file() or sha256_file(output_path) != stats.get("output_sha256"):
            fail(f"{report_path}: rebuilt unit output is missing or changed: {output_path}")

    built_data = Path(str(unit_stats["output_path"])).resolve()
    summary_data = Path(str(summary.get("data_path", ""))).resolve()
    if summary_data != built_data:
        fail(f"{summary_path}: evaluated {summary_data}, expected rebuilt data {built_data}")
    if bool(summary.get("blind")):
        fail(f"{summary_path}: pilot used --blind instead of the correct-image condition")
    checkpoint = Path(args.checkpoint).resolve()
    summary_checkpoint = Path(str(summary.get("ckpt", ""))).resolve()
    if summary_checkpoint != checkpoint:
        fail(f"{summary_path}: evaluated checkpoint {summary_checkpoint}, expected {checkpoint}")
    if not checkpoint.is_file():
        fail(f"missing pilot checkpoint: {checkpoint}")

    predictions_path = Path(str(summary_path).removesuffix(".summary.json"))
    if not predictions_path.is_file():
        fail(f"missing evaluator predictions paired with summary: {predictions_path}")
    panel_items = int(report["inventory"]["confirmatory_panel_items"])
    n_panel_units = len(report["inventory"].get("confirmatory_panel_units", [])) or 1
    if panel_items <= 0:
        fail(f"{report_path}: invalid confirmatory panel size {panel_items}")
    # Two-point timing: a full run and a --limit run of the same unit separate
    # the per-item cost from the fixed model-load overhead. Without the second
    # point the overhead is folded into the per-item cost (an upper bound).
    limited = None
    if args.limited_eval_summary:
        lim_summary = read_json(Path(args.limited_eval_summary))
        try:
            lim_elapsed = float(args.limited_elapsed_seconds)
        except (TypeError, ValueError):
            fail("--limited-elapsed-seconds is required with --limited-eval-summary")
        lim_scored = int(lim_summary.get("scored", 0))
        if lim_summary.get("benchmark") != "cvqa" or int(lim_summary.get("skipped", 0)):
            fail("limited pilot must be a cvqa run with no skipped items")
        if Path(str(lim_summary.get("data_path", ""))).resolve() != built_data:
            fail("limited pilot evaluated a different data file than the full pilot")
        if Path(str(lim_summary.get("ckpt", ""))).resolve() != checkpoint or bool(lim_summary.get("blind")):
            fail("limited pilot must use the same checkpoint and the correct-image condition")
        if not 0 < lim_scored < scored or lim_elapsed <= 0 or lim_elapsed >= elapsed:
            fail(f"limited pilot must score fewer items in less time: {lim_scored}/{scored}, {lim_elapsed}/{elapsed}")
        seconds_per_item = (elapsed - lim_elapsed) / (scored - lim_scored)
        load_seconds = elapsed - seconds_per_item * scored
        if seconds_per_item <= 0 or load_seconds < 0:
            fail(f"two-point timing is inconsistent: per-item {seconds_per_item:.3f}s, load {load_seconds:.1f}s")
        limited = {"eval_summary": str(Path(args.limited_eval_summary).resolve()), "scored": lim_scored,
                   "elapsed_seconds": round(lim_elapsed, 6)}
    else:
        seconds_per_item = elapsed / scored
        load_seconds = 0.0
    donor_conditions = int(args.donors) * int(args.conditions)
    invocations = donor_conditions * n_panel_units
    evaluation_seconds = seconds_per_item * panel_items * donor_conditions + load_seconds * invocations
    proportional_panel_seconds = seconds_per_item * panel_items + load_seconds * n_panel_units
    build_seconds = float(report["timing_seconds"]["build_total"])
    report["pilot"] = {
        "unit": args.unit,
        "extraction_unit": args.extraction_unit,
        "unit_is_confirmatory_panel_member": False,
        "limited_run": limited,
        "load_seconds": round(load_seconds, 3),
        "donor": args.donor,
        "condition": args.condition,
        "eval_summary": str(summary_path.resolve()),
        "eval_summary_sha256": sha256_file(summary_path),
        "evaluation_slurm_job_id": summary.get("slurm_job_id"),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "vis_layers": args.vis_layers,
        "scored": scored,
        "skipped": skipped,
        "accuracy": summary.get("accuracy"),
        "elapsed_seconds": round(elapsed, 6),
        "seconds_per_item": round(seconds_per_item, 9),
        "estimate": {
            "panel_items": panel_items,
            "donors": int(args.donors),
            "conditions": int(args.conditions),
            "proportional_seconds_per_panel_pass": round(proportional_panel_seconds, 3),
            "donor_confirmation_gpu_hours": round(evaluation_seconds / 3600.0, 3),
            "one_time_data_build_hours": round(build_seconds / 3600.0, 3),
            "total_hours_including_one_time_build": round((evaluation_seconds + build_seconds) / 3600.0, 3),
            "invocations": invocations,
            "note": (
                "Per-item cost from the timing unit scaled to the panel's item count, plus the "
                "measured model-load overhead once per (donor, condition, unit) invocation; "
                "without a --limit run the overhead is folded into the per-item cost."
            ),
        },
    }
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(report_path, report)
    print(
        f"pilot attached: {elapsed:.1f}s / {scored} {args.unit} items; "
        f"estimated donor confirmation={evaluation_seconds / 3600.0:.2f} GPU-h; "
        f"report={report_path}"
    )


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build units, extract embedded images, and compare queries")
    build.add_argument(
        "--parquet", action="append", required=True,
        help="local file/dir/glob; repeat or comma-separate",
    )
    build.add_argument("--inventory", default="Approach2/audits/cvqa_inventory.json")
    build.add_argument("--unit", action="append", default=[], help="inventory language unit; repeat")
    build.add_argument(
        "--extract-images-for", action="append", default=[],
        help="selected unit whose embedded images are materialised",
    )
    build.add_argument("--compare-existing", action="append", default=[], metavar="UNIT=PATH")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--images-dir", required=True)
    build.add_argument("--report", required=True)
    build.add_argument("--repo-root", default=None, help="record HEAD/dirty state from this checkout")
    build.set_defaults(func=command_build)

    attach = commands.add_parser("attach-pilot", help="attach evaluator timing and the scaled cost estimate")
    attach.add_argument("--report", required=True)
    attach.add_argument("--repo-root", required=True)
    attach.add_argument("--eval-summary", required=True)
    attach.add_argument("--elapsed-seconds", required=True)
    attach.add_argument("--unit", default="Japanese", help="timing unit; must NOT be a confirmatory-panel unit")
    attach.add_argument("--extraction-unit", default="Spanish", help="unit whose cold extraction was timed")
    attach.add_argument("--limited-eval-summary", default=None, help="summary of the --limit run for two-point timing")
    attach.add_argument("--limited-elapsed-seconds", default=None)
    attach.add_argument("--donor", default="bn")
    attach.add_argument("--condition", default="correct")
    attach.add_argument("--checkpoint", required=True)
    attach.add_argument("--vis-layers", required=True)
    attach.add_argument("--donors", type=int, default=7)
    attach.add_argument("--conditions", type=int, default=5)
    attach.set_defaults(func=command_attach_pilot)
    return top


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
