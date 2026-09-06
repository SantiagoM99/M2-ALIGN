"""X2 — Label-free alignment scoring of the text bridge, per language.

Why this exists
---------------
`DESIGN.md` ("Sufficiency is not alignment") measured that every language
reaches 77-99% of its own frozen-LLM ceiling on text reasoning, including the
three whose zero-shot multimodal transfer is flat (jv 86%/86% and 13%
retention; ga 77%/93% and 12%). So the bridge is *sufficient* everywhere and
sufficiency does not predict transfer. What transfer needs is different:
language L's prefix must land in the same region of prefix space that the
source language occupied while the visual mapping was being tuned, or the
visual prefix does not compose with it.

That property is relative, so the instrument has to be contrastive rather
than a task score — a matched pair must be closer than mismatched ones.
This follows `scripts/2027_eacl/sufficiency_scoring.py` in the ALIGNFREEZE
line (retrieval@1 over held-out parallel sentences) and MERLIN's Retrieval@5
layer analysis; the measure is established in this lineage.

Two references, because they answer different questions
-------------------------------------------------------
    bridge  prefix(s_L)  vs  prefix(s_en)   — is the bridge itself
            language-consistent? Both sides go through the same frozen NLLB
            encoder and the same trainable mapping, so this isolates whether
            the mapping puts translations in the same place.
    llm     prefix(s_L)  vs  emb_llm(s_en)  — does the bridge land where the
            frozen LLM already represents that meaning? Uses Gemma's input
            embedding table, which is what the prefix is spliced next to.

Pre-registered discriminating prediction (DESIGN.md, 2026-09-06)
----------------------------------------------------------------
The score must correlate with zero-shot transfer retention and must NOT
correlate with % of frozen-LLM ceiling. If it correlates with both it is
measuring general bridge quality, the alignment mechanism is unsupported,
and the result degrades from mechanism to description.

Which number to read
--------------------
`margin` (mean matched cosine minus mean mismatched cosine) is the primary
statistic. Retrieval@1 is reported because it is the established form in this
lineage, but the prefix space is 3584-dimensional and retrieval over N=1000
can saturate at 1.0 for every language, in which case it separates nothing.
The margin is continuous and does not saturate. If R@1 comes back at 1.000
across the board, that is not a result, it is the metric bottoming out.

Held-out data
-------------
Rows are taken from the END of each `<Language>_to_English.jsonl`. For every
language except the checkpoint's own training language this is moot — the
mapping in `stage3_bn_v4` never saw jv/mn/ga/de/... at all — but it keeps the
Bengali cell honest, which is the one that matters as the reference.

Usage
-----
    python3 Approach2/alignment_score.py \
        --ckpt Approach2/outputs/stage3_bn_v4/mapping/pytorch_model.bin \
        --label stage3_bn_v4 \
        --stage1-dir $DT/Stage1/data \
        --output Approach2/results/alignment_stage3_bn_v4.json \
        --n 1000 --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, NllbTokenizer

from common import load_mapping_checkpoint, mt_input_features, setup_logging
from model import DualEncoderMerger

# Human-readable stage-1 filename stem -> (iso code, NLLB tag). Matches
# train_stage1_text.py:LANGS_MAP_NLLB and evaluate_vqa.py:ISO_TO_NLLB.
LANGUAGES = {
    "Bengali": ("bn", "ben_Beng"),
    "German": ("de", "deu_Latn"),
    "Russian": ("ru", "rus_Cyrl"),
    "Chinese": ("zh", "zho_Hans"),
    "Portuguese": ("pt", "por_Latn"),
    "Indonesian": ("id", "ind_Latn"),
    "Korean": ("ko", "kor_Hang"),
    "Javanese": ("jv", "jav_Latn"),
    "Mongolian": ("mn", "khk_Cyrl"),
    "Sinhalese": ("si", "sin_Sinh"),
    "Irish": ("ga", "gle_Latn"),
}


def tail_rows(path: str, n: int) -> list[dict]:
    """Last *n* well-formed {source,target} rows of a JSONL file."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("source") and r.get("target"):
                rows.append(r)
    return rows[-n:]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding positions. hidden [B,S,D], mask [B,S] -> [B,D]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


@torch.inference_mode()
def bridge_prefix(model, tokenizer_mt, texts, tag, max_len, device, batch_size):
    """Mean-pooled mapping_txt(NLLB_encoder(text)) for each text. [N, llm_dim].

    The mapping's learnable end-boundary token is deliberately excluded: it is
    the same vector for every sentence, so including it would shrink every
    distance by a constant and inflate the scores.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids, mask = mt_input_features(chunk, [tag] * len(chunk), tokenizer_mt, max_len, device)
        enc = model.encoder_mt(
            input_ids=ids, attention_mask=mask, output_hidden_states=False
        )[0]
        out.append(masked_mean(model.mapping_txt(enc.float()), mask).cpu())
    return torch.cat(out)


@torch.inference_mode()
def llm_embed(model, tokenizer_llm, texts, max_len, device, batch_size):
    """Mean-pooled frozen-LLM input embeddings for each text. [N, llm_dim]."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer_llm(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
        emb = model.llm_embedding_layer(enc["input_ids"])
        out.append(masked_mean(emb.float(), enc["attention_mask"]).cpu())
    return torch.cat(out)


def retrieval_metrics(query: torch.Tensor, ref: torch.Tensor) -> dict:
    """Retrieval of the matched counterpart, plus the matched/mismatched margin.

    query[i] and ref[i] are the same sentence. A perfectly aligned bridge
    ranks ref[i] first for query[i] against all other ref[j] as distractors,
    so chance is 1/N and N is reported alongside.
    """
    q = torch.nn.functional.normalize(query, dim=-1)
    r = torch.nn.functional.normalize(ref, dim=-1)
    sim = q @ r.T                                   # [N, N]
    n = sim.size(0)
    gold = torch.arange(n)
    ranks = (sim > sim[gold, gold].unsqueeze(1)).sum(1)   # 0 = top-1
    matched = sim[gold, gold]
    off = sim.sum(1) - matched
    mismatched = off / (n - 1)
    return {
        "n": n,
        "retrieval@1": round((ranks == 0).float().mean().item(), 4),
        "retrieval@5": round((ranks < 5).float().mean().item(), 4),
        "median_rank": int(ranks.median().item()) + 1,
        "cos_matched": round(matched.mean().item(), 4),
        "cos_mismatched": round(mismatched.mean().item(), 4),
        "margin": round((matched - mismatched).mean().item(), 4),
        "chance@1": round(1.0 / n, 6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="mapping checkpoint to score")
    p.add_argument("--label", required=True, help="name for this checkpoint in the output")
    p.add_argument("--stage1-dir", required=True, help="dir with <Language>_to_English.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--langs", default="", help="ISO codes to score; default = all found")
    p.add_argument("--n", type=int, default=1000, help="held-out pairs per language")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-mt-seq-len", type=int, default=256)
    p.add_argument("--max-llm-seq-len", type=int, default=256)
    p.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--llm-path", default="google/gemma-2-9b-it")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    logger = setup_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), "a2_alignment"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wanted = set(args.langs.split()) if args.langs.strip() else None

    tokenizer_llm = AutoTokenizer.from_pretrained(
        args.llm_path, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_mt = NllbTokenizer.from_pretrained(
        args.mt_path, local_files_only=args.local_files_only)

    model = DualEncoderMerger(
        mt_path=args.mt_path,
        vis_path=None,
        llm_path=args.llm_path,
        max_gen_len=1,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        use_text_branch=True,
        use_vision_branch=False,
        local_files_only=args.local_files_only,
    ).to(device)
    load_mapping_checkpoint(args.ckpt, model, logger)
    model.eval()

    results: dict[str, dict] = {}
    for name, (iso, tag) in LANGUAGES.items():
        if wanted is not None and iso not in wanted:
            continue
        path = os.path.join(args.stage1_dir, f"{name}_to_English.jsonl")
        if not os.path.exists(path):
            logger.info("skip %s (%s): no file at %s", iso, name, path)
            continue
        rows = tail_rows(path, args.n)
        if len(rows) < 50:
            logger.info("skip %s: only %d usable rows", iso, len(rows))
            continue
        src = [r["source"] for r in rows]
        tgt = [r["target"] for r in rows]

        pref_l = bridge_prefix(model, tokenizer_mt, src, tag, args.max_mt_seq_len,
                               device, args.batch_size)
        pref_en = bridge_prefix(model, tokenizer_mt, tgt, "eng_Latn", args.max_mt_seq_len,
                                device, args.batch_size)
        emb_en = llm_embed(model, tokenizer_llm, tgt, args.max_llm_seq_len,
                           device, args.batch_size)

        results[iso] = {
            "language": name,
            "nllb_tag": tag,
            "bridge": retrieval_metrics(pref_l, pref_en),
            "llm": retrieval_metrics(pref_l, emb_en),
        }
        b, l = results[iso]["bridge"], results[iso]["llm"]
        logger.info("%s  bridge R@1 %.3f margin %+.4f | llm R@1 %.3f margin %+.4f (n=%d)",
                    iso, b["retrieval@1"], b["margin"], l["retrieval@1"], l["margin"], b["n"])

    payload = {
        "label": args.label,
        "ckpt": args.ckpt,
        "n_per_language": args.n,
        "held_out": "last N rows of each <Language>_to_English.jsonl",
        "languages": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s (%d languages)", args.output, len(results))

    print(f"\n=== {args.label} ===")
    print(f"{'lang':<6}{'bridge R@1':>12}{'bridge margin':>15}{'llm R@1':>10}{'llm margin':>12}")
    for iso, r in results.items():
        print(f"{iso:<6}{r['bridge']['retrieval@1']:>12.3f}{r['bridge']['margin']:>15.4f}"
              f"{r['llm']['retrieval@1']:>10.3f}{r['llm']['margin']:>12.4f}")


if __name__ == "__main__":
    main()
