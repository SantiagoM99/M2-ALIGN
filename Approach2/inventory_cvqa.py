"""Inventory of CVQA subsets for the S1 confirmatory panel (stdlib, network).

Reads afaji/cvqa through the Hugging Face datasets-server rows API (no image
download), validates every NLLB-200 tag against the tokenizer's own special
token list fetched from the model repo, applies the S1 Block D rule, and
writes a machine-readable inventory plus its sha256.

Rule (DESIGN.md S1 v2.1.3, Block D): a target unit is one language; country
subsets of the same language pool into one unit with the subset as a
bootstrap stratum; eligible if the language has an NLLB-200 tag, is not one
of the project's eleven, the unit has >= 150 valid items and every subset
in it has >= 4 unique images (three distinct derangements). Cap: the 12
units with the most valid items, ties broken by NLLB code.

    python3 Approach2/inventory_cvqa.py --out Approach2/audits/cvqa_inventory.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows?dataset=afaji%2Fcvqa&config=default&split=test&offset={o}&length=100"
NLLB_TOKENS = "https://huggingface.co/facebook/nllb-200-distilled-600M/raw/main/special_tokens_map.json"

# CVQA language name -> NLLB-200 code (validated below against the tokenizer).
LANG_TO_NLLB = {
    "Amharic": "amh_Ethi", "Bengali": "ben_Beng", "Breton": "bre_Latn", "Bulgarian": "bul_Cyrl",
    "Chinese": "zho_Hans", "Egyptian Arabic": "arz_Arab", "Filipino": "tgl_Latn", "French": "fra_Latn",
    "Hindi": "hin_Deva", "Igbo": "ibo_Latn", "Indonesian": "ind_Latn", "Irish": "gle_Latn",
    "Japanese": "jpn_Jpan", "Javanese": "jav_Latn", "Kinyarwanda": "kin_Latn", "Korean": "kor_Hang",
    "Malay": "zsm_Latn", "Marathi": "mar_Deva", "Minangkabau": "min_Latn", "Mongolian": "khk_Cyrl",
    "Norwegian": "nob_Latn", "Oromo": "gaz_Latn", "Portuguese": "por_Latn", "Romanian": "ron_Latn",
    "Russian": "rus_Cyrl", "Sinhala": "sin_Sinh", "Spanish": "spa_Latn", "Sundanese": "sun_Latn",
    "Swahili": "swh_Latn", "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab",
}
ELEVEN = {"Bengali", "Russian", "German", "Chinese", "Portuguese", "Indonesian", "Korean",
          "Javanese", "Mongolian", "Sinhala", "Irish"}
MIN_ITEMS, MIN_IMAGES, CAP = 150, 4, 12


def get(url: str, tries: int = 4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Approach2/audits/cvqa_inventory.json")
    ap.add_argument("--recompute", default=None, help="re-apply the rule to a saved inventory JSON (no network)")
    a = ap.parse_args()
    valid_tags = set(get(NLLB_TOKENS)["additional_special_tokens"])
    subsets: dict[str, dict] = {}
    n_total = None
    offset = 0
    if a.recompute:
        saved = json.load(open(a.recompute))
        n_total = saved["num_rows_total"]
        for u in saved["units"]:
            for sub in u["subsets"]:
                d = {k: v for k, v in sub.items() if k not in ("subset", "country")}
                d["language"] = u["language"]; d["country"] = sub["country"]
                d["images"] = set(f"i{k}" for k in range(sub["images"]))
                subsets[sub["subset"]] = d
    while not a.recompute:
        page = get(ROWS.format(o=offset))
        n_total = page.get("num_rows_total", n_total)
        rows = page["rows"]
        if not rows:
            break
        for rr in rows:
            r = rr["row"]
            sub = str(r["Subset"])
            lang, country = ast.literal_eval(sub)
            d = subsets.setdefault(sub, {"language": lang, "country": country, "items": 0, "images": set(),
                                         "native_q": 0, "english_q": 0, "english_choices": 0, "image_url": 0,
                                         "bad_id": 0})
            i = str(r["ID"])
            m = re.match(r"^(\d+)_(\d+)$", i)
            if not m:
                d["bad_id"] += 1
                continue
            d["items"] += 1
            d["images"].add(m.group(1))
            d["native_q"] += bool((r.get("Question") or "").strip())
            d["english_q"] += bool((r.get("Translated Question") or "").strip())
            d["english_choices"] += bool(r.get("Translated Options"))
            d["image_url"] += str(r.get("Image Source") or "").startswith("http")
        offset += len(rows)
        if n_total is not None and offset >= n_total:
            break
    units: dict[str, dict] = {}
    for sub, d in subsets.items():
        d["images"] = len(d["images"])
        u = units.setdefault(d["language"], {"language": d["language"], "nllb": LANG_TO_NLLB.get(d["language"].replace("_", " ")),
                                                "subsets": [], "items": 0})
        u["subsets"].append({"subset": sub, "country": d["country"], **{k: v for k, v in d.items() if k not in ("language", "country")}})
        u["items"] += d["items"]
    for u in units.values():
        tag = u["nllb"]
        reasons = []
        if not tag or tag not in valid_tags:
            reasons.append("no NLLB-200 tag" if not tag else f"tag {tag} not in tokenizer")
        if u["language"] in ELEVEN:
            reasons.append("one of the eleven")
        if u["items"] < MIN_ITEMS:
            reasons.append(f"items {u['items']} < {MIN_ITEMS}")
        small = [s["subset"] for s in u["subsets"] if s["images"] < MIN_IMAGES]
        if small:
            reasons.append(f"subsets with < {MIN_IMAGES} images: {small}")
        u["eligible"] = not reasons
        u["excluded_because"] = reasons
    eligible = sorted([u for u in units.values() if u["eligible"]], key=lambda u: (-u["items"], u["nllb"]))
    panel = eligible[:CAP]
    out = {
        "dataset": "afaji/cvqa", "split": "test", "num_rows_total": n_total,
        "source_of_tags": NLLB_TOKENS, "rule": {"min_items": MIN_ITEMS, "min_images_per_subset": MIN_IMAGES, "cap": CAP,
                                                 "excluded_languages": sorted(ELEVEN)},
        "units": sorted(units.values(), key=lambda u: u["language"]),
        "confirmatory_panel": [{"language": u["language"], "nllb": u["nllb"], "items": u["items"],
                                "subsets": [s["subset"] for s in u["subsets"]]} for u in panel],
    }
    blob = json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1)
    open(a.out, "w").write(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    open(a.out.replace(".json", ".sha256"), "w").write(sha + "\n")
    print(f"rows={n_total} subsets={len(subsets)} languages={len(units)} eligible={len(eligible)} panel={len(panel)}")
    print(f"{'language':17s}{'nllb':10s}{'items':>6}{'subsets':>8}{'img/subset':>12}{'nativeQ':>8}{'enQ':>6}{'url%':>6}  status")
    for u in sorted(units.values(), key=lambda u: (-u["eligible"], -u["items"])):
        imgs = ",".join(str(s["images"]) for s in u["subsets"])
        nq = sum(s["native_q"] for s in u["subsets"]); eq = sum(s["english_q"] for s in u["subsets"])
        url = 100 * sum(s["image_url"] for s in u["subsets"]) / max(1, u["items"])
        status = "PANEL" if u in panel else ("eligible, over cap" if u["eligible"] else "; ".join(u["excluded_because"]))
        print(f"{u['language']:17s}{str(u['nllb']):10s}{u['items']:6d}{len(u['subsets']):8d}{imgs:>12}{nq:8d}{eq:6d}{url:6.0f}  {status}")
    print("sha256:", sha)


if __name__ == "__main__":
    main()
