"""Across-seed spread of the cultural arm's gain (stdlib).

`arch_compare.py` reads one checkpoint per arm, so every bound it reports is
conditional on that one training trajectory. The cultural gain is a recipe
claim, which CLAUDE.md requires three seeds for, so this consumes one
`arch_compare` report per seed and reports the contrast per seed plus the mean,
SD and range across them.

It recomputes nothing: the per-seed estimates and their CIs come from reports
already written by a versioned script. What it adds is the across-seed form and
the checks that make the three comparable.

Pairing is verified rather than assumed. Every report must carry the same
contrast name, the same scored item count and the same languages, because a
replicate scored on a different item universe is not a replicate of this
contrast.

**The seeds share stages 1 and 2 by design, so the SD here bounds the stage-3
training variance of the intervention and not the pipeline's.** The output says
so in `scope`, and any text quoting the SD must repeat it.

    python3 Approach2/analysis/cg_seed_spread.py \\
      --report 42=audits/arch_compare_cvqa_cg_s42_donors.json \\
      --report 13=audits/arch_compare_cvqa_cg_s13_donors.json \\
      --report 7=audits/arch_compare_cvqa_cg_s7_donors.json \\
      --output Approach2/audits/cg_seed_spread_donors.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCOPE = (
    "All seeds warm-start from one shared stage-2 checkpoint, so this SD bounds "
    "the stage-3 training variance of the intervention, not the whole "
    "pipeline's. It is conditional on these trajectories and is not a "
    "population CI over training randomness."
)


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def read_report(path: Path, subset: str | None = None) -> dict:
    """One arch_compare report, optionally one labelled subset of it.

    A subset is read from `subsets.<name>`, which is where `--subset-labels`
    puts the clean and overlap arms. Asking for a subset a report does not carry
    is fatal rather than silently falling back to the full arm: a clean-subset
    claim read off the full items would be the memorisation claim the audit
    exists to rule out.
    """
    d = json.loads(path.read_text())
    contrasts = d.get("contrasts") or {}
    if len(contrasts) != 1:
        fail(f"{path} has {len(contrasts)} contrasts; expected exactly one")
    name, c = next(iter(contrasts.items()))
    if subset:
        available = c.get("subsets") or {}
        if subset not in available:
            fail(
                f"{path} has no subset {subset!r}; it carries "
                f"{sorted(available) or 'none'}. Re-run arch_compare.py with "
                "--subset-labels."
            )
        full = available[subset]
        name = f"{name} [{subset}]"
    else:
        full = c.get("full") or {}
    if "estimate" not in full:
        fail(f"{path} carries no estimate for {subset or 'the full arm'}")
    return {
        "contrast": name,
        "estimate": full["estimate"],
        "ci95": full.get("ci95"),
        "items": full.get("items"),
        # Only the full arm is broken down per language; a subset report carries
        # the pooled estimate alone, so the across-seed per-language table is
        # empty for a subset rather than wrong.
        "languages": sorted(c.get("per_language_full", {})) if not subset else [],
        "per_language": c.get("per_language_full", {}) if not subset else {},
    }


def spread(seeds: dict) -> dict:
    vals = [s["estimate"] for s in seeds.values()]
    if len(vals) < 3:
        fail(f"{len(vals)} seeds; a recipe claim needs at least three")
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    out = {
        "per_seed": {k: v["estimate"] for k, v in seeds.items()},
        "mean": mean,
        "sd": sd,
        "range": max(vals) - min(vals),
        "min": min(vals),
        "max": max(vals),
        "sd_over_mean": (sd / mean) if mean else None,
        "all_same_sign": all(v > 0 for v in vals) or all(v < 0 for v in vals),
        "scope": SCOPE,
    }
    langs = sorted({L for s in seeds.values() for L in s["per_language"]})
    per_lang = {}
    for L in langs:
        lv = [s["per_language"][L] for s in seeds.values() if L in s["per_language"]]
        if len(lv) == len(seeds):
            per_lang[L] = {
                "per_seed": {k: s["per_language"][L] for k, s in seeds.items()},
                "mean": statistics.mean(lv),
                "sd": statistics.stdev(lv) if len(lv) > 2 else None,
                "replicates_sign": all(v > 0 for v in lv) or all(v < 0 for v in lv),
            }
    out["per_language"] = per_lang
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", default=[], metavar="SEED=PATH",
                    help="an arch_compare report for one seed; repeat, three or more")
    ap.add_argument("--subset", help="read subsets.<name> instead of the full arm")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    seeds: dict[str, dict] = {}
    for spec in a.report:
        if "=" not in spec:
            fail(f"--report wants SEED=PATH, got {spec!r}")
        k, p = spec.split("=", 1)
        if k in seeds:
            fail(f"seed {k} given twice")
        seeds[k] = read_report(Path(p), a.subset)

    first = next(iter(seeds.values()))
    for k, s in seeds.items():
        for field in ("contrast", "items", "languages"):
            if s[field] != first[field]:
                fail(
                    f"seed {k} disagrees on {field} "
                    f"({s[field]!r} against {first[field]!r}); these reports do "
                    "not describe the same contrast on the same items"
                )

    out = {
        "contrast": first["contrast"],
        "subset": a.subset,
        "items": first["items"],
        "languages": first["languages"],
        "seeds": sorted(seeds),
        "per_seed_ci95": {k: v["ci95"] for k, v in seeds.items()},
        **spread(seeds),
    }
    Path(a.output).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{out['contrast']}  n={out['items']}  seeds={','.join(out['seeds'])}")
    for k in sorted(seeds):
        print(f"  seed {k:>3}: {out['per_seed'][k]:+6.2f}")
    print(f"  mean {out['mean']:+6.2f}  SD {out['sd']:.2f}  "
          f"range {out['range']:.2f}  SD/mean {out['sd_over_mean']:.1%}")
    print(f"  every seed same sign: {out['all_same_sign']}")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
