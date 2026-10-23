"""Does CG50's training set already contain CVQA's answers? (stdlib)

`audits/cg_leakage.json` hashes image bytes and says so in its own method field:
it catches the same photograph, not the same entity. On 2026-10-08 that was
enough, because entity overlap was declared a property to state rather than
test. The 10-22 result changes the stakes -- CG50 gains **+15.03** utility on
CVQA-bn, +18.18 on the block the backbone answers wrongly -- and a gain that
size has an obvious competing explanation: CulturalGround and CVQA are both
built from Wikidata entities, so the training rows may simply carry the panel's
answers.

CVQA's choices are English even where its questions are not, and
CulturalGround's answers are joined in English, so the two are directly
comparable as strings.

**The control is the point.** Overlap is measured on every CVQA target, not only
on bn. Reading 1 passed on bn and reading 2 failed on jv/mn/ga; if bn's answers
sit in the training set and the zero-shot targets' answers do not, that pattern
is an explanation of both results at once, and the claim "culturally grounded
supervision moves the knowledge block" has to become a claim about answer-set
exposure instead. If overlap is low on bn too, the gain survives this objection
and the paper can say so with a number.

This measures exposure, not causation: an overlapping answer may still require
the image to pick. Both are reported so the reader can see which.

    python3 Approach2/analysis/cg_entity_overlap.py \\
      --training $DT/Stage3/data/bn_cg50.jsonl \\
      --submission Approach2/outputs/s1_CG50.eval.submission.json \\
      --output Approach2/audits/cg_entity_overlap.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_contract import atomic_json, read_json  # noqa: E402

CULTURAL = "culturalground_refined_oe"


def normalise(text) -> str:
    """Casefold, drop accents and punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zঀ-৿]+", " ", s.casefold())
    return " ".join(s.split())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows


def training_exposure(path: Path) -> dict:
    """The answers and entity names the culturally grounded half actually carries."""
    rows = read_jsonl(path)
    cultural = [r for r in rows if r.get("source_dataset") == CULTURAL]
    if not cultural:
        raise SystemExit(
            f"{path} has no {CULTURAL} rows; this is not a mixed CG50 file"
        )
    answers = {normalise(r.get("answer", "")) for r in cultural}
    answers.discard("")
    # vg_image_id is `cg_<Qid>_<Commons file stem>`; the stem usually spells the
    # entity out, which is the only entity label the training rows carry.
    entities = set()
    for r in cultural:
        stem = str(r.get("vg_image_id", ""))
        stem = re.sub(r"^cg_Q\d+_?", "", stem)
        text = normalise(stem.replace("_", " "))
        if text:
            entities.add(text)
    return {
        "rows_total": len(rows),
        "rows_cultural": len(cultural),
        "distinct_answers": len(answers),
        "distinct_entity_labels": len(entities),
        "answers": answers,
        "entities": entities,
    }


def panel_paths(submission_path: Path) -> dict[str, str]:
    """The exact panel each CVQA cell was scored on, from the submission itself."""
    sub = read_json(submission_path)
    out = {}
    for manifest in sub["manifests"].values():
        if manifest["benchmark"] != "cvqa":
            continue
        target = manifest["target"]
        path = manifest["arguments"]["data_path"]
        if out.setdefault(target, path) != path:
            raise SystemExit(f"{target}: cells disagree on the panel path")
    if not out:
        raise SystemExit("the submission scores no CVQA cells")
    return out


def audit_panel(rows: list[dict], exposure: dict) -> dict:
    answers, entities = exposure["answers"], exposure["entities"]
    entity_blob = " | ".join(sorted(entities))
    gold_exact = gold_in_entity = any_choice = 0
    examples = []
    for r in rows:
        choices = [normalise(c) for c in r.get("choices", [])]
        if not choices:
            raise SystemExit("a CVQA row carries no choices")
        gold = choices[int(r["answer_index"])]
        hit = gold in answers
        gold_exact += hit
        in_entity = bool(gold) and gold in entity_blob
        gold_in_entity += in_entity
        any_choice += any(c in answers for c in choices)
        if hit and len(examples) < 5:
            examples.append(gold)
    n = len(rows)
    return {
        "items": n,
        "gold_answer_in_training_answers": gold_exact,
        "gold_answer_share": round(gold_exact / n, 4),
        "gold_answer_in_training_entity_labels": gold_in_entity,
        "gold_entity_share": round(gold_in_entity / n, 4),
        "any_choice_in_training_answers": any_choice,
        "any_choice_share": round(any_choice / n, 4),
        "example_overlapping_answers": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", required=True, help="the mixed CG50 stage-3 file")
    ap.add_argument("--submission", help="take panel paths from a CG50 eval submission")
    ap.add_argument("--panel", action="append", default=[], metavar="LANG=PATH",
                    help="a CVQA panel, repeatable; overrides --submission for that target")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    exposure = training_exposure(Path(a.training))
    panels = panel_paths(Path(a.submission)) if a.submission else {}
    for spec in a.panel:
        lang, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--panel wants LANG=PATH, got {spec!r}")
        panels[lang] = path
    if not panels:
        raise SystemExit("no panels: pass --submission or --panel")

    report = {
        "training": {
            "path": str(a.training),
            **{k: v for k, v in exposure.items() if k not in ("answers", "entities")},
        },
        "method": (
            "CVQA choices are English and CulturalGround answers are joined in English; "
            "strings are compared after casefolding, accent and punctuation removal. "
            "Exposure, not causation: an overlapping answer may still need the image."
        ),
        "panels": {},
    }
    for lang in sorted(panels):
        report["panels"][lang] = {
            "panel": panels[lang],
            **audit_panel(read_jsonl(Path(panels[lang])), exposure),
        }
    shares = {l: p["gold_answer_share"] for l, p in report["panels"].items()}
    report["gold_answer_share_by_target"] = shares
    if "bn" in shares and len(shares) > 1:
        others = [v for l, v in shares.items() if l != "bn"]
        report["bn_minus_other_targets"] = round(shares["bn"] - sum(others) / len(others), 4)
    atomic_json(a.output, report)
    print(json.dumps({k: report[k] for k in ("training", "gold_answer_share_by_target")}, indent=2))
    if "bn_minus_other_targets" in report:
        print("bn minus the mean of the other targets:", report["bn_minus_other_targets"])


if __name__ == "__main__":
    main()
