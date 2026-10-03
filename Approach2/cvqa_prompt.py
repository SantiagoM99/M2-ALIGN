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


def country_of(row: dict) -> str:
    """The country from CVQA's own `Subset` field, which our rows carry.

    Refuses rather than guesses: a language is not a country here. CVQA's
    Bengali subset is India, Chinese spans China and Singapore, and Spanish
    spans seven countries, so the value has to come from the row.
    """
    subset = row.get("subset")
    if not subset:
        raise ValueError(
            "location-aware prompting needs the CVQA `subset` field; "
            f"row {row.get('id')!r} has none"
        )
    try:
        parsed = ast.literal_eval(str(subset))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"unparsable subset {subset!r}: {exc}") from exc
    if not isinstance(parsed, tuple) or len(parsed) != 2 or not str(parsed[1]).strip():
        raise ValueError(f"subset {subset!r} is not (language, country)")
    return str(parsed[1])
