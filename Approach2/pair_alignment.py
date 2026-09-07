"""Pairwise cross-lingual alignment of the text bridge, on FLORES-200.

The instrument X2 should have been
----------------------------------
X2 asked a target-only question — how well does language L's bridge retrieve
its own English translations — and correlated it against retention measured
from ONE source (Bengali). X1 then showed transfer is a property of the
source-target PAIR: Indonesian donates to jv/mn/ga (pooled p=0.00085) where
Bengali does not (p=0.36). A target-only measure cannot predict a pair-level
outcome, so X2's null said nothing about alignment as a mechanism.

This measures the quantity the mechanism is actually about: **how close is
target T's prefix to source S's prefix**, for the same sentence. The visual
mapping was tuned while the source's prefixes occupied the LLM's input
space, so that is the distance a transferred visual prefix has to survive.

FLORES-200 dev is multi-way parallel — line i is the same sentence in every
language — which the stage-1 files are not. Fetch it with fetch_flores.py.

Centering
---------
X2's margins were useless because the prefix space is severely anisotropic:
*mismatched* sentences sat at cosine 0.987, leaving every margin in a
0.007-0.010 band that measured the residual scale rather than alignment.
Each language's prefixes are therefore mean-centered before the cosine, which
is the standard correction. Both centered and raw numbers are reported so the
size of that artefact stays visible.

Usage
-----
    python3 Approach2/pair_alignment.py \
        --ckpt Approach2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin \
        --label stage3_bn_dcl \
        --flores evaluation/flores_dev.jsonl \
        --output Approach2/results/pairalign_stage3_bn_dcl.json \
        --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, NllbTokenizer

from common import load_mapping_checkpoint, mt_input_features, setup_logging
from model import DualEncoderMerger

TAGS = {
    "bn": "ben_Beng", "de": "deu_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn",
    "mn": "khk_Cyrl", "si": "sin_Sinh", "ga": "gle_Latn", "en": "eng_Latn",
}


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


@torch.inference_mode()
def prefixes(model, tokenizer_mt, texts, tag, max_len, device, batch_size):
    """Mean-pooled mapping_txt(NLLB_encoder(text)) per sentence. [N, llm_dim].

    The learnable end-boundary token is excluded: it is identical for every
    sentence, so it would add a constant that shrinks every distance alike.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids, mask = mt_input_features(chunk, [tag] * len(chunk), tokenizer_mt, max_len, device)
        enc = model.encoder_mt(input_ids=ids, attention_mask=mask,
                               output_hidden_states=False)[0]
        out.append(masked_mean(model.mapping_txt(enc.float()), mask).cpu())
    return torch.cat(out)


def pair_metrics(q: torch.Tensor, r: torch.Tensor) -> dict:
    """Retrieval of the matched sentence in r for each sentence in q."""
    qn = torch.nn.functional.normalize(q, dim=-1)
    rn = torch.nn.functional.normalize(r, dim=-1)
    sim = qn @ rn.T
    n = sim.size(0)
    gold = torch.arange(n)
    diag = sim[gold, gold]
    ranks = (sim > diag.unsqueeze(1)).sum(1)
    mism = (sim.sum(1) - diag) / (n - 1)
    return {
        "retrieval@1": round((ranks == 0).float().mean().item(), 4),
        "retrieval@5": round((ranks < 5).float().mean().item(), 4),
        "median_rank": int(ranks.median().item()) + 1,
        "margin": round((diag - mism).mean().item(), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--flores", required=True, help="flores_dev.jsonl from fetch_flores.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--langs", default="", help="ISO codes; default = every one present")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sentences")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-mt-seq-len", type=int, default=256)
    ap.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--llm-path", default="google/gemma-2-9b-it")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), "a2_pairalign")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(args.flores, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    wanted = set(args.langs.split()) if args.langs.strip() else set(TAGS)
    langs = [iso for iso, tag in TAGS.items() if iso in wanted and tag in rows[0]]
    logger.info("FLORES: %d sentences x %d languages (%s)", len(rows), len(langs),
                " ".join(langs))

    tokenizer_llm = AutoTokenizer.from_pretrained(
        args.llm_path, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_mt = NllbTokenizer.from_pretrained(
        args.mt_path, local_files_only=args.local_files_only)

    model = DualEncoderMerger(
        mt_path=args.mt_path, vis_path=None, llm_path=args.llm_path, max_gen_len=1,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_text_branch=True, use_vision_branch=False,
        local_files_only=args.local_files_only,
    ).to(device)
    load_mapping_checkpoint(args.ckpt, model, logger)
    model.eval()

    raw: dict[str, torch.Tensor] = {}
    for iso in langs:
        texts = [r[TAGS[iso]] for r in rows]
        raw[iso] = prefixes(model, tokenizer_mt, texts, TAGS[iso],
                            args.max_mt_seq_len, device, args.batch_size)
        logger.info("encoded %s", iso)
    cen = {iso: v - v.mean(0, keepdim=True) for iso, v in raw.items()}

    out: dict[str, dict] = {"centered": {}, "raw": {}}
    for src in langs:
        out["centered"][src] = {}
        out["raw"][src] = {}
        for tgt in langs:
            if src == tgt:
                continue
            out["centered"][src][tgt] = pair_metrics(cen[tgt], cen[src])
            out["raw"][src][tgt] = pair_metrics(raw[tgt], raw[src])

    payload = {
        "label": args.label,
        "ckpt": args.ckpt,
        "n_sentences": len(rows),
        "languages": langs,
        "note": "matrix[src][tgt] = retrieving tgt's sentence among src's, i.e. "
                "how well a prefix in tgt lands where src's prefixes live",
        "matrix": out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", args.output)

    print(f"\n=== {args.label}: centered retrieval@1, matrix[src][tgt] ===")
    print(f"{'src':<6}" + "".join(f"{t:>7}" for t in langs))
    for src in langs:
        cells = "".join(
            f"{'  --':>7}" if src == t else f"{out['centered'][src][t]['retrieval@1']:>7.3f}"
            for t in langs)
        print(f"{src:<6}{cells}")


if __name__ == "__main__":
    main()
