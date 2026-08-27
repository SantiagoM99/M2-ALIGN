"""Category breakdown + McNemar between two paired xGQA runs.

Pairs per-item predictions (baseline vs new, full+blind) by id and reports
per-category accuracy, DeltaV, and paired McNemar (new full vs baseline
full). Categories from the gold answer, same taxonomy as the D8 analysis.

Usage (from repo root; defaults compare dc vs replay on bn):
    python Approach2/analysis/xgqa_category_breakdown.py \
        [baseline_full.jsonl baseline_blind.jsonl new_full.jsonl new_blind.jsonl]
"""
import json, math, collections, os, sys

R = "Approach2/results"

COLORS = {"red","blue","green","yellow","black","white","brown","gray","grey",
          "orange","pink","purple","tan","beige","gold","silver","blond","blonde",
          "dark","light blue","dark blue","light brown","dark brown","cream colored",
          "maroon","teal","khaki","brunette"}
MATERIALS = {"wood","wooden","metal","plastic","glass","leather","concrete","brick",
             "stone","cloth","fabric","paper","cardboard","rubber","steel","porcelain",
             "ceramic","denim","cotton","wool","marble","tile","clay","straw","wicker",
             "bamboo","copper","brass","iron","aluminum"}
SPATIAL = {"left","right","top","bottom","behind","front","above","below","under",
           "on top","background","foreground","middle","center","inside","outside",
           "on the left","on the right","in front of","near","far"}

def category(ans: str) -> str:
    a = ans.strip().lower()
    if a in ("yes", "no"):
        return "yes/no"
    if a in SPATIAL:
        return "spatial"
    if a in COLORS:
        return "color"
    if a in MATERIALS:
        return "material"
    return "object/other"

def load(name):
    d = {}
    with open(os.path.join(R, name), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["id"]] = r
    return d

files = sys.argv[1:5] or ["eval_xgqa_bn_replay.jsonl", "eval_xgqa_bn_replay_BLIND.jsonl",
                          "eval_xgqa_bn_dc.jsonl", "eval_xgqa_bn_dc_BLIND.jsonl"]
rp, rpb, dc, dcb = (load(f) for f in files)
print("baseline:", files[0], "| new:", files[2])

ids = set(dc) & set(dcb) & set(rp) & set(rpb)
print(f"paired items: {len(ids)} (dc={len(dc)} replay={len(rp)})\n")

stats = collections.defaultdict(lambda: {"n":0,"dc_f":0,"dc_b":0,"rp_f":0,"rp_b":0,
                                         "b":0,"c":0})  # b: rp right/dc wrong, c: dc right/rp wrong
tot = {"n":0,"dc_f":0,"dc_b":0,"rp_f":0,"rp_b":0,"b":0,"c":0}

for i in ids:
    cat = category(dc[i]["answer"])
    for s in (stats[cat], tot):
        s["n"]  += 1
        s["dc_f"] += dc[i]["correct"];  s["dc_b"] += dcb[i]["correct"]
        s["rp_f"] += rp[i]["correct"];  s["rp_b"] += rpb[i]["correct"]
        if rp[i]["correct"] and not dc[i]["correct"]: s["b"] += 1
        if dc[i]["correct"] and not rp[i]["correct"]: s["c"] += 1

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    z = (c - b) / math.sqrt(n)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p

hdr = f"{'category':<13}{'n':>7} | {'base fl':>8}{'new fl':>8}{'d_full':>8} | {'base dV':>7}{'new dV':>7}{'d_dV':>7} | {'b/c':>11}{'p':>10}"
print(hdr); print("-"*len(hdr))
order = sorted(stats, key=lambda k: -stats[k]["n"])
for cat in order + ["TOTAL"]:
    s = tot if cat == "TOTAL" else stats[cat]
    n = s["n"]
    rf, db_, rb, df = 100*s["rp_f"]/n, 100*s["dc_b"]/n, 100*s["rp_b"]/n, 100*s["dc_f"]/n
    dv_rp, dv_dc = rf - rb, df - db_
    p = mcnemar(s["b"], s["c"])
    print(f"{cat:<13}{n:>7} | {rf:>8.2f}{df:>8.2f}{df-rf:>+8.2f} | "
          f"{dv_rp:>7.2f}{dv_dc:>7.2f}{dv_dc-dv_rp:>+7.2f} | "
          f"{s['b']:>5}/{s['c']:<5}{p:>10.2g}")
