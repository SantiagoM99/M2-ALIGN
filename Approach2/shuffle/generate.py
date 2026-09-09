"""Generate ignored maps and a tracked hash registry before S1 submission."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_contract import atomic_json, digest, read_json, read_rows, shuffle_maps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--benchmark", choices=["xgqa", "cvqa"], required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--output-dir", default="evaluation")
    p.add_argument("--registry", required=True)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    if Path(a.panel).name != a.panel:
        p.error("panel must be a simple filename component")
    maps = shuffle_maps(read_rows(a.data_path), a.benchmark)
    entry = {
        "panel": a.panel,
        "benchmark": a.benchmark,
        "id_universe_sha256": maps[0]["id_universe_sha256"],
        "maps": {str(m["seed"]): m["sha256"] for m in maps},
    }
    if a.verify:
        if read_json(a.registry) != entry:
            raise ValueError("registry does not match regenerated maps")
    else:
        atomic_json(a.registry, entry)
    for m in maps:
        atomic_json(Path(a.output_dir) / f"shuffle_{a.panel}_seed{m['seed']}.json", m)
    print(digest(entry))


if __name__ == "__main__":
    main()
