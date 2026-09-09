"""Shared fail-closed VQA evaluation, used by both CLIs and the S1 matrix."""

from __future__ import annotations

import argparse
import copy
import json
import importlib.metadata
import math
import os
import platform
from functools import lru_cache
from pathlib import Path

from s1_contract import (
    atomic_json,
    branch_paths,
    digest,
    file_record,
    file_sha,
    git_state,
    item_identity,
    read_json,
    read_rows,
    result_complete,
    shuffle_maps,
    universe,
    validate_shuffle,
)


def parser(benchmark):
    p = argparse.ArgumentParser(
        description=f"{benchmark} evaluation with S1 input controls"
    )
    p.add_argument("--data-path", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--image-cache-dir")
    p.add_argument("--output-path", required=True)
    for key in ("ckpt", "extra-ckpt", "txt-ckpt", "vis-ckpt"):
        p.add_argument("--" + key)
    p.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--vis-path", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--llm-path", default="google/gemma-2-9b-it")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--no-text-branch", action="store_true")
    p.add_argument(
        "--prompt-mode", choices=["question", "instruction"], default="question"
    )
    p.add_argument(
        "--question-field", choices=["query", "english_query"], default="query"
    )
    conditions = p.add_mutually_exclusive_group()
    conditions.add_argument("--blind", action="store_true")
    conditions.add_argument("--no-image", action="store_true")
    conditions.add_argument("--shuffle-map")
    p.add_argument("--shuffle-registry")
    p.add_argument("--max-vis-tokens", type=int, default=0)
    p.add_argument("--vis-layers", default="")
    p.add_argument("--max-mt-seq-len", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-gen-len", type=int, default=32)
    p.add_argument("--max-choice-len", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--s1", action="store_true")
    p.add_argument("--experiment-id")
    p.add_argument("--arm")
    p.add_argument("--source")
    p.add_argument("--target")
    p.add_argument(
        "--seed", type=int, default=13, help="training seed / checkpoint provenance"
    )
    p.add_argument("--hypothesis")
    p.add_argument("--prediction")
    p.add_argument(
        "--estimand", default="U; correct-minus-shuffled; correct-minus-grey"
    )
    p.set_defaults(benchmark=benchmark)
    return p


def validate_args(a):
    if a.prompt_mode == "instruction" and a.no_text_branch:
        raise ValueError(
            "instruction-only requires the text branch to carry the question"
        )
    if a.question_field == "english_query" and (
        not a.no_text_branch or a.prompt_mode != "question"
    ):
        raise ValueError(
            "English direct-prompt A4 requires --no-text-branch and question mode"
        )
    if sum(bool(v) for v in (a.blind, a.no_image, a.shuffle_map)) > 1:
        raise ValueError("image conditions are mutually exclusive")
    if (
        a.limit < 0
        or min(a.max_seq_len, a.max_mt_seq_len, a.max_gen_len, a.max_choice_len) <= 0
    ):
        raise ValueError("invalid sequence length/limit")
    paths = branch_paths(a)
    for name, used in [
        ("mapping_txt", not a.no_text_branch),
        ("mapping_vis", not a.no_image),
    ]:
        if used and not paths[name]:
            raise ValueError(f"{name}: checkpoint required for active branch")
    if a.s1:
        for name in (
            "experiment_id",
            "arm",
            "source",
            "target",
            "hypothesis",
            "prediction",
            "shuffle_registry",
        ):
            if not getattr(a, name):
                raise ValueError(f"S1 requires --{name.replace('_', '-')}")
        if a.limit:
            raise ValueError("S1 primary evaluations cannot use --limit")
        if a.vis_layers != "9,18,-1" or a.max_vis_tokens != 0 or a.no_chat_template:
            raise ValueError(
                "S1 requires vis-layers 9,18,-1, all visual tokens and chat template"
            )


def condition(a):
    if a.shuffle_map:
        return f"shuffled{read_json(a.shuffle_map)['seed']}"
    return "gray" if a.blind else "no-image" if a.no_image else "correct"


@lru_cache(maxsize=None)
def model_record(identifier):
    """Resolve cached revisions without network or loading model weights."""
    from transformers.utils.hub import cached_file

    config = Path(cached_file(identifier, "config.json", local_files_only=True))
    directory = config.parent
    parts = directory.parts
    revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    # Hub snapshots have a content-addressed revision. For mutable local
    # directories, fingerprint the weight files as well as configuration.
    patterns = ["*.json", "*.model"] + (
        ["*.safetensors", "*.bin"] if revision is None else []
    )
    files = {
        p.name: file_sha(p)
        for pattern in patterns
        for p in directory.glob(pattern)
        if p.is_file()
    }
    return {
        "identifier": identifier,
        "revision": revision,
        "path": str(directory),
        "files": files,
    }


def model_records(a):
    return {
        "llm": model_record(a.llm_path),
        "text": model_record(a.mt_path) if not a.no_text_branch else None,
        "vision": model_record(a.vis_path) if not a.no_image else None,
    }


def environment_record():
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "pillow",
                "sentencepiece",
            )
        },
    }


def prepare(a, image_cache=None):
    """Validate all original images for every condition, before loading towers."""
    from PIL import Image

    image_cache = {} if image_cache is None else image_cache
    validate_args(a)
    rows = read_rows(a.data_path)
    if a.limit:
        rows = rows[: a.limit]
    # Legacy CVQA files lack S1 identities. Only non-S1 use permits deriving
    # image id from the documented <image>_<question> format.
    if not a.s1:
        for r in rows:
            if a.benchmark == "cvqa":
                r.setdefault("image_id", str(r["id"]).rsplit("_", 1)[0])
                r.setdefault("subset", a.target or "legacy")
    identities = universe(rows, a.benchmark)
    registry = read_json(a.shuffle_registry) if a.shuffle_registry else None
    maps = shuffle_maps(rows, a.benchmark) if (registry or a.shuffle_map) else []
    if registry:
        if (
            registry["benchmark"] != a.benchmark
            or registry["id_universe_sha256"] != digest(identities)
            or registry["maps"] != {str(m["seed"]): m["sha256"] for m in maps}
        ):
            raise ValueError("shuffle registry differs from regenerated maps")
    shuffled = (
        validate_shuffle(read_json(a.shuffle_map), rows, a.benchmark)
        if a.shuffle_map
        else None
    )
    images = {}
    image_hashes = {}
    for r in rows:
        identity = item_identity(r, a.benchmark)
        iid = identity["image_id"]
        if iid not in images:
            candidates = [
                Path(a.images_dir) / (iid + ext) for ext in (".jpg", ".jpeg", ".png")
            ]
            if a.benchmark == "cvqa":
                candidates.append(Path(a.images_dir) / (str(r["id"]) + ".jpg"))
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise ValueError(f"missing original image {iid} (item {r['id']})")
            stat = path.stat()
            cache_key = (
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
            if cache_key not in image_cache:
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as exc:
                    raise ValueError(f"corrupt image {path}") from exc
                image_cache[cache_key] = file_sha(path)
            images[iid] = path
            image_hashes[iid] = image_cache[cache_key]
        if not isinstance(r.get("query"), str) or not r["query"].strip():
            raise ValueError("missing native query")
        if (
            not isinstance(r.get(a.question_field), str)
            or not r[a.question_field].strip()
        ):
            raise ValueError(f"missing question field {a.question_field}")
        if a.benchmark == "cvqa":
            choices = r.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or any(not isinstance(c, str) or not c.strip() for c in choices)
                or type(r.get("answer_index")) is not int
                or not 0 <= r["answer_index"] < len(choices)
            ):
                raise ValueError("invalid choices/answer_index")
        elif not isinstance(r.get("answer"), str) or not r["answer"].strip():
            raise ValueError("missing answer")
    return {
        "rows": rows,
        "images": images,
        "shuffled": shuffled,
        "id_universe_sha256": digest(identities),
        "image_sha256": digest(image_hashes),
        "shuffle_map_hashes": {str(m["seed"]): m["sha256"] for m in maps},
    }


def make_manifest(a, prepared, state=None):
    state = state or git_state(require_clean=a.s1)
    paths = branch_paths(a)
    return {
        **state,
        "schema_version": 1,
        "experiment_id": a.experiment_id,
        "arm": a.arm,
        "benchmark": a.benchmark,
        "source": a.source,
        "target": a.target,
        "seed": a.seed,
        "hypothesis": a.hypothesis,
        "prediction": a.prediction,
        "estimand": a.estimand,
        "condition": condition(a),
        "arguments": vars(a).copy(),
        "checkpoints": {k: file_record(v) if v else None for k, v in paths.items()},
        "frozen_models": model_records(a),
        "environment": environment_record(),
        "data": file_record(a.data_path),
        "split_sha256": prepared["id_universe_sha256"],
        "id_universe_sha256": prepared["id_universe_sha256"],
        "image_sha256": prepared["image_sha256"],
        "shuffle_map_hashes": prepared["shuffle_map_hashes"],
        "vis_layers": a.vis_layers,
        "decoding": {
            k: getattr(a, k)
            for k in (
                "max_gen_len",
                "max_choice_len",
                "max_seq_len",
                "max_mt_seq_len",
                "no_chat_template",
            )
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


class Runtime:
    """Towers load once; each cell starts with pristine mapping state."""

    def __init__(self, a, use_text=None, use_vision=None):
        import torch
        from transformers import AutoTokenizer, AutoImageProcessor, NllbTokenizer
        from model import DualEncoderMerger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        txt = not a.no_text_branch if use_text is None else use_text
        vis = not a.no_image if use_vision is None else use_vision
        llm_path = model_record(a.llm_path)["path"]
        mt_path = model_record(a.mt_path)["path"] if txt else None
        vis_path = model_record(a.vis_path)["path"] if vis else None
        self.tokenizer_mt = (
            NllbTokenizer.from_pretrained(mt_path, local_files_only=a.local_files_only)
            if txt
            else None
        )
        self.tokenizer_llm = AutoTokenizer.from_pretrained(
            llm_path, use_fast=True, local_files_only=a.local_files_only
        )
        if self.tokenizer_llm.pad_token is None:
            self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token
        self.tokenizer_llm.padding_side = "left"
        self.image_processor = (
            AutoImageProcessor.from_pretrained(
                vis_path, local_files_only=a.local_files_only
            )
            if vis
            else None
        )
        self.model = DualEncoderMerger(
            mt_path,
            vis_path,
            llm_path,
            a.max_gen_len,
            self.tokenizer_llm.bos_token_id,
            self.tokenizer_llm.pad_token_id,
            use_text_branch=txt,
            use_vision_branch=vis,
            max_vis_tokens=a.max_vis_tokens,
            vis_layers=a.vis_layers,
            local_files_only=a.local_files_only,
        ).to(self.device)
        self.pristine = {
            name: copy.deepcopy(getattr(self.model, name).state_dict())
            for name in ("mapping_txt", "mapping_vis")
            if getattr(self.model, name) is not None
        }

    def reset(self, a):
        from common import load_branch_checkpoint

        for name, state in self.pristine.items():
            getattr(self.model, name).load_state_dict(state, strict=True)
        for name, path in branch_paths(a).items():
            active = not a.no_text_branch if name == "mapping_txt" else not a.no_image
            if active:
                load_branch_checkpoint(path, getattr(self.model, name), name)
        self.model.max_gen_len = a.max_gen_len
        self.model.eval()

    def predict(self, a, prepared):
        import torch
        from PIL import Image
        from common import (
            _VQA_SYSTEM,
            format_chat_prompt,
            llm_input_features,
            mt_input_features,
        )
        from evaluate_cvqa import format_cvqa_chat
        from evaluate_vqa import row_nllb_tag, open_ended_correct

        self.reset(a)
        cell_condition = condition(a)
        for row in prepared["rows"]:
            ident = item_identity(row, a.benchmark)
            inputs = {}
            if not a.no_text_branch:
                ids, mask = mt_input_features(
                    [row["query"]],
                    [row_nllb_tag(row)],
                    self.tokenizer_mt,
                    a.max_mt_seq_len,
                    self.device,
                )
                inputs.update(input_ids_mt=ids, attention_mask_mt=mask)
            assigned = ident["image_id"]
            if prepared["shuffled"]:
                assigned = prepared["shuffled"]["items"][ident["id"]][
                    "assigned_image_id"
                ]
            if not a.no_image:
                if a.blind:
                    image = Image.new("RGB", (384, 384), (128, 128, 128))
                else:
                    with Image.open(prepared["images"][assigned]) as im:
                        image = im.convert("RGB")
                inputs["pixel_values"] = self.image_processor(
                    images=[image], return_tensors="pt"
                )["pixel_values"].to(self.device)
            if a.prompt_mode == "instruction":
                prompt = "Answer with a single word or short phrase, in English."
                if not a.no_chat_template and getattr(
                    self.tokenizer_llm, "chat_template", None
                ):
                    prompt = self.tokenizer_llm.apply_chat_template(
                        [{"role": "user", "content": f"{_VQA_SYSTEM}\n\n{prompt}"}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    bos = self.tokenizer_llm.bos_token
                    if bos and prompt.startswith(bos):
                        prompt = prompt[len(bos) :]
            else:
                formatter = (
                    format_cvqa_chat if a.benchmark == "cvqa" else format_chat_prompt
                )
                prompt = formatter(
                    self.tokenizer_llm, row[a.question_field], not a.no_chat_template
                )
            ids, mask = llm_input_features(
                [prompt],
                self.tokenizer_llm,
                a.max_seq_len,
                add_bos=False,
                add_eos=False,
                device=self.device,
            )
            inputs.update(input_ids_prompt=ids, mask_prompt=mask)
            result = {
                **ident,
                "query": row["query"],
                "condition": cell_condition,
                "assigned_image_id": assigned
                if not a.no_image and not a.blind
                else None,
            }
            with torch.inference_mode():
                if a.benchmark == "cvqa":
                    scores = []
                    for choice in row["choices"]:
                        labels, mask_label = llm_input_features(
                            [choice],
                            self.tokenizer_llm,
                            a.max_choice_len,
                            False,
                            False,
                            self.device,
                        )
                        score = -self.model(
                            labels=labels, mask_label=mask_label, **inputs
                        ).item()
                        if not math.isfinite(score):
                            raise ValueError("non-finite choice score")
                        scores.append(score)
                    pred = max(range(len(scores)), key=scores.__getitem__)
                    result.update(
                        choices=row["choices"],
                        answer_index=row["answer_index"],
                        pred_index=pred,
                        scores=scores,
                        correct=pred == row["answer_index"],
                    )
                else:
                    pred = self.model.generate(self.tokenizer_llm, **inputs)[0].strip()
                    result.update(
                        answer=row["answer"],
                        pred=pred,
                        correct=open_ended_correct(pred, row["answer"]),
                    )
            yield result


def evaluate(a, runtime=None, prepared=None, manifest=None):
    prepared = prepared or prepare(a)
    manifest = manifest or make_manifest(a, prepared)
    if result_complete(a.output_path, manifest):
        return read_json(a.output_path + ".summary.json")
    atomic_json(a.output_path + ".manifest.json", manifest)
    runtime = runtime or Runtime(a)
    tmp = Path(a.output_path + ".tmp")
    n = correct = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in runtime.predict(a, prepared):
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
            correct += r["correct"]
    if n != len(prepared["rows"]):
        raise ValueError("prediction count differs from complete input universe")
    summary = {
        **manifest,
        "scored": n,
        "skipped": 0,
        "correct": correct,
        "accuracy": correct / n,
        "blind": a.blind,
        "ckpt": a.ckpt,
        "data_path": a.data_path,
    }
    os.replace(tmp, a.output_path)
    atomic_json(a.output_path + ".summary.json", summary)
    atomic_json(
        a.output_path + ".complete.json",
        {
            "manifest_sha256": digest(manifest),
            "predictions_sha256": file_sha(a.output_path),
            "summary_sha256": file_sha(a.output_path + ".summary.json"),
        },
    )
    return summary


def main(benchmark):
    a = parser(benchmark).parse_args()
    print(json.dumps(evaluate(a), ensure_ascii=False))
