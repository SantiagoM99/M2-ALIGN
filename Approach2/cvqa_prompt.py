"""The CVQA prompt, in one place because two evaluators must build it identically.

Our stack (`evaluate_cvqa.py`) and the ported Qwen baseline
(`baseline_qwen/evaluate.py`) are compared on the same items, so any drift
between their prompts would show up as an architecture difference. Both built
`Question: {question}` from their own f-string until 2026-10-03; they now share
this module.

Stdlib only, so it can be imported and tested without torch.

**Location-aware prompting.** CVQA's own evaluation is a 2x2 — {location-aware,
location-agnostic} x {English, local} (Romero et al., 2406.05967) — and every
number in this project so far is the location-agnostic, local-language cell.
Stating the country is their condition, not an invention of ours; the exact
template is not theirs, because theirs shows the four options and our
open-ended protocol shows none.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


def location_sentence(country: str) -> str:
    """The country, as a sentence. The inventory writes `Sri_Lanka`."""
    return f"This picture is from {country.replace('_', ' ')}."


def build_cvqa_open_ended_prompt(question: str, country: str | None = None) -> str:
    """Question only, no visible options — matching Stage3/evaluate.py.

    Without `country` the string is byte-identical to the one every existing
    cell was scored with, which is what keeps those cells comparable.
    """
    if country:
        return f"{location_sentence(country)}\nQuestion: {question}"
    return f"Question: {question}"


AMBIGUOUS = object()


def country_lookup(inventory_path) -> dict:
    """NLLB tag -> country, from the committed CVQA inventory.

    Only the S1 panels carry CVQA's own `(language, country)` subset per row;
    the legacy evaluation copy stores the placeholder `"legacy"` (jobs 22448653
    and 22448654 died on exactly that, which is what the refusal is for). The
    inventory is the committed record of which subsets each language unit was
    built from, so it supplies the country for the legacy copy without anyone
    typing a country by hand.

    A unit spanning several countries maps to AMBIGUOUS rather than to its
    first one: CVQA's Chinese is China *and* Singapore, and Spanish is seven
    countries. Those languages need the per-row subset, i.e. the S1 panels.
    """
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    out = {}
    for unit in inventory.get("units", []):
        tag = unit.get("nllb")
        countries = set()
        for subset in unit.get("subsets", []):
            try:
                parsed = ast.literal_eval(str(subset.get("subset")))
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, tuple) and len(parsed) == 2:
                countries.add(str(parsed[1]))
        if not tag or not countries:
            continue
        out[tag] = countries.pop() if len(countries) == 1 else AMBIGUOUS
    if not out:
        raise ValueError(f"{inventory_path}: no language unit carries a usable subset")
    return out


def country_of(row: dict, lookup: dict | None = None, tag: str | None = None) -> str:
    """The country for one row: its own subset first, then the inventory.

    Refuses rather than guesses. A language is not a country here, and the
    placeholder `"legacy"` is not a subset.
    """
    subset = row.get("subset")
    if subset and str(subset) != "legacy":
        try:
            parsed = ast.literal_eval(str(subset))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"unparsable subset {subset!r}: {exc}") from exc
        if isinstance(parsed, tuple) and len(parsed) == 2 and str(parsed[1]).strip():
            return str(parsed[1])
        raise ValueError(f"subset {subset!r} is not (language, country)")
    tag = tag or row.get("nllb_lang_tag")
    if lookup and tag:
        country = lookup.get(tag)
        if country is AMBIGUOUS:
            raise ValueError(
                f"{tag} spans several CVQA countries, so the country must come from "
                "the row: run this language from an S1 panel, not the legacy copy"
            )
        if country:
            return country
    raise ValueError(
        "location-aware prompting found no country for row "
        f"{row.get('id')!r} (subset={subset!r}, tag={tag!r}); pass the CVQA inventory"
    )
