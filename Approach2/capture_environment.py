"""Read-only environment inventory. Run on the authenticated Rorqual login node.

Captures packages, module versions, and cached model revisions without network,
model loading, credentials, environment dumps, or GPU computation.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform

from s1_contract import atomic_json, file_sha, git_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    packages = sorted(
        [
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions()
        ],
        key=lambda d: d["name"].lower(),
    )
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    snapshots = []
    for repo in [
        "models--google--gemma-2-9b-it",
        "models--google--siglip2-so400m-patch14-384",
        "models--facebook--nllb-200-distilled-600M",
    ]:
        for root in (cache / "hub" / repo, cache / "transformers" / repo):
            refs = root / "refs"
            if not refs.exists():
                continue
            for ref in sorted(refs.rglob("*")):
                if ref.is_file():
                    revision = ref.read_text().strip()
                    path = root / "snapshots" / revision
                    configs = {
                        str(f.relative_to(path)): file_sha(f)
                        for f in path.glob("*.json")
                    }
                    snapshots.append(
                        {
                            "model_cache": repo,
                            "ref": str(ref.relative_to(refs)),
                            "revision": revision,
                            "configuration_hashes": configs,
                        }
                    )
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.sys.executable,
        "modules": os.environ.get("LOADEDMODULES", "").split(":"),
        "packages": packages,
        "cached_models": snapshots,
        "git": git_state(require_clean=False),
    }
    atomic_json(a.output, result)
    print(
        f"Wrote {a.output}: {len(packages)} packages, {len(snapshots)} cached model refs; no GPU used"
    )


if __name__ == "__main__":
    main()
