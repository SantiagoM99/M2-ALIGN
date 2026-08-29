"""How far did stage 3 pull the text mapping away from its stage-1 solution?

DESIGN.md D11 proposes a mechanism for why scaling *vision* data improved
*text* reasoning: stage 3 trains both mappings into one frozen LLM, so a
poorly aligned vision mapping puts large gradients on the shared input space
and the text mapping drifts off its stage-1 solution to compensate. Better
visual alignment removes that pressure.

That mechanism makes a falsifiable prediction, stated here before the
measurement: **the arm with the better stage-2 checkpoint (dcl, ~111k pairs)
should show LESS text-mapping drift than the arm with the weaker one (dc_e2,
11.5k), despite both running an identical stage 3.** If drift is equal or
larger, the mechanism is wrong and D11's explanation has to be rewritten —
the effect would still stand, but as an unexplained one.

Drift is relative L2, ||W3 - W1|| / ||W1||, per tensor and pooled, plus the
cosine between each arm's update direction (do the two arms move the mapping
the same way, or somewhere else entirely?).

Usage (on the cluster, from the repo root):
    python3 Approach2/analysis/mapping_drift.py \\
      --ref  Approach2/outputs/stage1/mapping/pytorch_model.bin \\
      --arm  dc_e2=Approach2/outputs/stage3_bn_dc_e2/mapping/pytorch_model.bin \\
      --arm  dcl=Approach2/outputs/stage3_bn_dcl/mapping/pytorch_model.bin

Add --branch mapping_vis (with --ref the stage-2 checkpoint) for the vision
side. Needs torch; run it in the venv, no GPU required.
"""
from __future__ import annotations

import argparse

import torch


def get_branch(path: str, branch: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if branch not in payload:
        raise SystemExit(f"{path}: no '{branch}' (has {sorted(k for k in payload)})")
    return {k: v.float() for k, v in payload[branch].items()}


def flat(sd: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([sd[k].reshape(-1) for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser(description="Text-mapping drift between arms.")
    ap.add_argument("--ref", required=True, help="Reference checkpoint (stage 1).")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="Repeatable: an arm's stage-3 checkpoint.")
    ap.add_argument("--branch", default="mapping_txt",
                    choices=["mapping_txt", "mapping_vis"])
    ap.add_argument("--per-tensor", action="store_true")
    args = ap.parse_args()

    ref = get_branch(args.ref, args.branch)
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = get_branch(path, args.branch)

    # `gate` (D7) exists only in checkpoints trained after it was added, so
    # stage-1 references predate it. Drift is measured over the shared
    # weights; the gate is reported on its own below, since it is a scalar
    # with a different meaning (1.0 = identity, i.e. untouched).
    shared = set(ref)
    for sd in arms.values():
        shared &= set(sd)
    extra = (set(ref) | set().union(*(set(sd) for sd in arms.values()))) - shared
    if extra:
        print(f"note: keys absent from at least one checkpoint, excluded from drift: "
              f"{sorted(extra)}\n")

    keys = sorted(shared)
    if not keys:
        raise SystemExit("no shared parameters between reference and arms")
    rflat = flat(ref, keys)
    print(f"branch = {args.branch} | reference = {args.ref}")
    print(f"parameters = {rflat.numel():,}  ||W1|| = {rflat.norm():.4f}\n")

    deltas = {}
    print(f"{'arm':10} {'||dW||':>10} {'rel drift':>11}")
    for name, sd in arms.items():
        d = flat(sd, keys) - rflat
        deltas[name] = d
        print(f"{name:10} {d.norm():10.4f} {(d.norm()/rflat.norm()).item():10.2%}")

    names = list(arms)
    if len(names) >= 2:
        print("\ncosine between update directions:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = deltas[names[i]], deltas[names[j]]
                cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                print(f"  {names[i]} vs {names[j]}: {cos:+.4f}")

    gates = {n: sd["gate"] for n, sd in arms.items() if "gate" in sd}
    if gates:
        print("\ngate (1.0 = identity, the prefix passes through untouched):")
        for n, g in gates.items():
            print(f"  {n:10} {g.flatten().tolist()}")

    if args.per_tensor:
        print("\nper-tensor relative drift:")
        head = "".join(f"{n:>12}" for n in names)
        print(f"{'tensor':32}{head}")
        for k in keys:
            row = "".join(
                f"{((arms[n][k]-ref[k]).norm()/ref[k].norm()).item():11.2%} " for n in names
            )
            print(f"{k:32}{row}")


if __name__ == "__main__":
    main()
