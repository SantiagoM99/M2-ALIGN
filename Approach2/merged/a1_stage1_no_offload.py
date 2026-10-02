"""Run her Stage 1 unchanged, with DeepSpeed's CPU offload turned off.

Her `tools/deepspeed_config.get_train_ds_config` defaults to `offload=True` and
`Stage1/train.py` calls it positionally, so there is no flag to turn it off. That
default sends the optimizer state to CPU, which makes DeepSpeed instantiate
`DeepSpeedCPUAdam`, which JIT-compiles an AVX-512 kernel — and that build fails on
this cluster, because the toolchain compiles with `-march=x86-64-v3` while the
source needs AVX-512 ("target specific option mismatch", job 22218332).

Turning the offload off is a fix rather than a workaround. Only the mapping is
trainable here, so the optimizer state is small and there is nothing to gain by
keeping it in host memory; the optimizer, its hyperparameters and the ZeRO stage
are unchanged, so this moves *where* the state lives, not what training computes.

Her code is not edited — it runs from the worktree as committed, and this script
patches the one default from outside, then calls her `main`. The deepspeed
launcher takes this file in place of hers and forwards the same arguments.

    A1_ROOT=$SCRATCH/a1 deepspeed --master_port 50010 \\
      Approach2/merged/a1_stage1_no_offload.py --deepspeed <her Stage 1 args>
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("A1_ROOT", Path(os.environ.get("SCRATCH", ".")) / "a1"))
    stage1 = root / "Stage1"
    if not (stage1 / "train.py").is_file():
        raise SystemExit(
            f"a1_stage1_no_offload: no Stage1/train.py under {root}. Set A1_ROOT to the "
            "worktree of her branch (git worktree add $SCRATCH/a1 upstream/parallel)"
        )
    # Her launcher runs from inside Stage1, so `from tools...` resolves there.
    sys.path.insert(0, str(stage1))
    os.chdir(stage1)

    from tools import deepspeed_config

    original = deepspeed_config.get_train_ds_config

    @functools.wraps(original)
    def without_offload(*args, **kwargs):
        kwargs["offload"] = False
        config = original(*args, **kwargs)
        print("a1_stage1_no_offload: offload=False; "
              f"offload_optimizer={config.get('zero_optimization', {}).get('offload_optimizer')}",
              flush=True)
        return config

    deepspeed_config.get_train_ds_config = without_offload

    # Her argparse and logging live inside `if __name__ == "__main__"`, so the
    # script has to be executed, not imported. runpy does that with the module
    # already patched, and her `from tools.deepspeed_config import ...` then binds
    # the wrapper rather than the original.
    import runpy

    runpy.run_path(str(stage1 / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
