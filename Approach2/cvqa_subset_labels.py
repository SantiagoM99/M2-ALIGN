"""Per-item country labels for CVQA, so a panel can be split by culture (stdlib, network).

The country is not in the legacy rows (DESIGN 2026-10-05) and the committed
inventory holds counts per subset, not which item belongs to which. The pooled
cultural arm needs the item-level map for one reading in particular: Chinese has
two subsets, **China** and **Singapore**, the cultural rows are built from China
only, and Singapore is therefore a culture the arm never trained on in a
language it did train on. That is the one contrast that separates "the culture
matched" from "the language was supervised" (DESIGN 2026-11-02, prediction 2).

Output is shaped for `arch_compare.py --subset-labels`: {lang: {item_id:
country}}. Languages with a single subset yield one group, which is harmless and
keeps the file usable for every panel.

Reads afaji/cvqa through the datasets-server rows API, the same path
`inventory_cvqa.py` uses, so no images are downloaded and no dataset library is
needed. Login node: this needs network.

    python3 Approach2/cvqa_subset_labels.py --out Approach2/audits/cvqa_subset_labels.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory_cvqa import LANG_TO_NLLB, ROWS, get  # noqa: E402

# The ten languages this project scores, by NLLB tag. German is absent from CVQA.
TAG_TO_CODE = {
    "ben_Beng": "bn", "rus_Cyrl": "ru", "zho_Hans": "zh", "por_Latn": "pt",
    "ind_Latn": "id", "kor_Hang": "ko", "jav_Latn": "jv", "khk_Cyrl": "mn",
    "sin_Sinh": "si", "gle_Latn": "ga",
}
ID = re.compile(r"^\d+_\d+$")


def collect(fetch=get) -> tuple[dict, dict]:
    """{code: {item_id: country}} plus the row counts the caller should check."""
    labels: dict[str, dict[str, str]] = {}
    skipped = {"bad_id": 0, "other_language": 0}
    offset, total = 0, None
    while True:
        page = fetch(ROWS.format(o=offset))
        total = page.get("num_rows_total", total)
        rows = page.get("rows", [])
        if not rows:
            break
        for wrapper in rows:
            row = wrapper["row"]
            language, country = ast.literal_eval(str(row["Subset"]))
            code = TAG_TO_CODE.get(LANG_TO_NLLB.get(language.replace("_", " "), ""))
            if not code:
                skipped["other_language"] += 1
                continue
            item = str(row["ID"])
            if not ID.match(item):
                skipped["bad_id"] += 1
                continue
            labels.setdefault(code, {})[item] = country
        offset += len(rows)
        if total is not None and offset >= total:
            break
    if not labels:
        raise SystemExit("no rows for any scored language; the API shape changed")
    return labels, {"rows_read": offset, "num_rows_total": total, **skipped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Approach2/audits/cvqa_subset_labels.json")
    a = ap.parse_args()
    labels, counts = collect()
    Path(a.out).write_text(json.dumps(labels, ensure_ascii=False, indent=1, sort_keys=True))
    print(json.dumps(counts, indent=1))
    for code in sorted(labels):
        by_country: dict[str, int] = {}
        for country in labels[code].values():
            by_country[country] = by_country.get(country, 0) + 1
        print(f"{code}: {len(labels[code])} items  {by_country}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
