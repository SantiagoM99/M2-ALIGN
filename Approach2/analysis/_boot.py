"""Shared bootstrap helpers for the S1 analysis scripts (stdlib only).

Contract (S1 v2.1.3):
  * every (estimand, stratum) pair draws from its own RNG derived from the
    run seed, so the interval of a stratum never depends on which other
    strata are present or on their order;
  * strata and clusters are iterated in canonical (sorted) order;
  * resample b of a pooled statistic combines resample b of every stratum,
    item-weighted;
  * loaders abort on duplicate ids instead of overwriting.
"""
from __future__ import annotations

import hashlib
import json
import os
import random


def rng_for(seed: int, *labels: str) -> random.Random:
    key = ":".join([str(seed), *labels]).encode()
    return random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))


def load_items(tag: str) -> dict[str, int]:
    """eval_<tag>.jsonl -> {id: 0/1}; validate the minimal result schema."""
    p = f"eval_{tag}.jsonl"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    out: dict[str, int] = {}
    with open(p) as f:
        for lineno, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{p}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(r, dict):
                raise SystemExit(f"{p}:{lineno}: expected a JSON object")
            raw_id = r.get("id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
                raise SystemExit(f"{p}:{lineno}: missing or empty id")
            if "correct" not in r or not isinstance(r["correct"], bool):
                raise SystemExit(f"{p}:{lineno}: 'correct' must be a JSON boolean")
            i = str(raw_id)
            if i in out:
                raise SystemExit(f"{p}: duplicate id {i}; refusing to overwrite")
            out[i] = int(r["correct"])
    if not out:
        raise SystemExit(f"{p}: empty result file")
    return out


def pct(v: list[float], q: float) -> float:
    if not v:
        raise ValueError("percentile of an empty sample")
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def boot_stratified(strata: dict[str, dict[str, list[int]]], B: int, seed: int, estimand: str) -> list[float]:
    """strata[stratum][cluster] = per-item values.

    Within each stratum, clusters are resampled with replacement using an RNG
    derived from (seed, estimand, stratum); resample b of the pooled statistic
    is the item-weighted mean over all strata's b-th resamples.
    """
    if B <= 0:
        raise ValueError("B must be positive")
    if not strata:
        raise ValueError("at least one stratum is required")
    tot = [0.0] * B
    cnt = [0] * B
    for s in sorted(strata):
        cl = strata[s]
        if not cl:
            raise ValueError(f"stratum {s!r} has no clusters")
        if any(not values for values in cl.values()):
            raise ValueError(f"stratum {s!r} contains an empty cluster")
        keys = sorted(cl)
        C = len(keys)
        rnd = rng_for(seed, estimand, s)
        for b in range(B):
            for _ in range(C):
                v = cl[keys[rnd.randrange(C)]]
                tot[b] += sum(v)
                cnt[b] += len(v)
    return [100.0 * t / c for t, c in zip(tot, cnt)]


def region(lb5: float, ub95: float, delta: float) -> str:
    if lb5 > -delta:
        return "non-inferior"
    if ub95 < -delta:
        return "mat. inferior"
    return "inconclusive"
