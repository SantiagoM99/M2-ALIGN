"""Compare what two text mappings DO, not how far their weights moved.

`mapping_drift.py` measures ||W3 - W1||, which turned out to be the wrong
instrument: two 2-layer MLPs can sit 6% apart in weight space and compute
almost the same function, or sit equally far apart and compute different
ones. The D11 run showed both arms drifting ~6% from stage 1 with a cosine
of only +0.51 between their update directions — different destinations, not
different distances — so the question became functional, not geometric.

This encodes real benchmark questions with NLLB and pushes them through each
arm's `mapping_txt`, then reports, per arm pair:

  - cosine between the produced prefixes, token by token, averaged
  - relative L2 between prefixes
  - the same against the stage-1 mapping, which is the "no joint training"
    reference: an arm closer to stage 1 in FUNCTION preserved more of the
    translation-aligned representation, whatever its weights did

Gemma is never loaded — only the NLLB encoder and the mappings — so this is
minutes on one GPU and works on CPU for small --limit.

    python Approach2/analysis/mapping_function_drift.py \\
      --data evaluation/MGSM.jsonl --nllb-tag ben_Beng \\
      --ref  Approach2/outputs/stage1/mapping/pytorch_model.bin \\
      --arm  dc_e2=Approach2/outputs/stage3_bn_dc_e2/mapping/pytorch_model.bin \\
      --arm  dcl=Approach2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import Mapping  # noqa: E402


def load_mapping(path: str, in_dim: int, out_dim: int, device) -> Mapping:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "mapping_txt" not in payload:
        raise SystemExit(f"{path}: no mapping_txt")
    m = Mapping(in_dim, out_dim)
    m.load_state_dict(payload["mapping_txt"], strict=False)
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Functional drift of the text mapping.")
    ap.add_argument("--data", required=True, help="JSONL with a `question` field.")
    ap.add_argument("--nllb-tag", default="ben_Beng")
    ap.add_argument("--ref", required=True, help="Stage-1 mapping (reference).")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--llm-dim", type=int, default=3584, help="Gemma-2-9b hidden size.")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mt_path = args.mt_path
    if os.path.isdir(mt_path):
        subs = [d for d in os.listdir(mt_path) if os.path.isdir(os.path.join(mt_path, d))]
        if subs:
            mt_path = os.path.join(mt_path, subs[0])

    tok = AutoTokenizer.from_pretrained(mt_path, local_files_only=args.local_files_only)
    tok.src_lang = args.nllb_tag
    enc = AutoModelForSeq2SeqLM.from_pretrained(
        mt_path, local_files_only=args.local_files_only).get_encoder().to(device).eval()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["question"])
            if len(rows) >= args.limit:
                break
    print(f"{len(rows)} questions | tag {args.nllb_tag} | device {device}")

    maps = {"stage1": load_mapping(args.ref, enc.config.d_model, args.llm_dim, device)}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        maps[name] = load_mapping(path, enc.config.d_model, args.llm_dim, device)

    outs = {k: [] for k in maps}
    masks = []
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            batch = rows[i:i + 8]
            t = tok(batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_len).to(device)
            h = enc(**t).last_hidden_state.float()
            masks.append(t["attention_mask"].bool().cpu())
            for k, m in maps.items():
                outs[k].append(m(h).float().cpu())

    def pairwise(a_name: str, b_name: str) -> tuple[float, float]:
        cos_sum = l2_sum = ref_sum = n = 0.0
        for ai, bi, mk in zip(outs[a_name], outs[b_name], masks):
            a, b = ai[mk], bi[mk]                       # (tokens, llm_dim)
            cos_sum += torch.nn.functional.cosine_similarity(a, b, dim=-1).sum().item()
            l2_sum += (a - b).norm(dim=-1).pow(2).sum().item()
            ref_sum += a.norm(dim=-1).pow(2).sum().item()
            n += a.shape[0]
        return cos_sum / n, (l2_sum ** 0.5) / (ref_sum ** 0.5)

    names = [k for k in maps if k != "stage1"]
    print(f"\n{'pair':24} {'mean cosine':>12} {'rel L2':>9}")
    for nm in names:
        c, l = pairwise("stage1", nm)
        print(f"{'stage1 vs ' + nm:24} {c:12.4f} {l:9.2%}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c, l = pairwise(names[i], names[j])
            print(f"{names[i] + ' vs ' + names[j]:24} {c:12.4f} {l:9.2%}")
    print("\nHigher cosine to stage1 = more of the translation-aligned "
          "representation survived joint training.")


if __name__ == "__main__":
    main()
