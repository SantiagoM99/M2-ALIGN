"""Order- and presence-invariance tests for the S1 analysis scripts (stdlib).

Run from Approach2/results:  python3 ../analysis/test_invariance.py
Asserts that every per-target row and the pooled row of x1_did.py and
e3_noninferiority.py are byte-identical when --targets is reordered, and
that X1's pooled row is unchanged when a non-pool target is dropped. Uses
B=800 so it runs in seconds; the invariance property does not depend on B.
"""
from __future__ import annotations

import subprocess
import sys

HERE = __file__.rsplit("/", 1)[0]


def run(script: str, *args: str) -> list[str]:
    out = subprocess.run([sys.executable, f"{HERE}/{script}", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{script} failed: {out.stderr.strip()}")
    return [l for l in out.stdout.splitlines() if l and not l.startswith(("X1 DiD", "E3 vs", "lang", "target", "Three", "Regions"))]


def rows(lines: list[str]) -> dict[str, str]:
    return {l.split()[0]: l for l in lines}


def main() -> None:
    B = "800"
    a = rows(run("x1_did.py", "--boot", B, "--targets", "jv", "mn", "ga", "si"))
    b = rows(run("x1_did.py", "--boot", B, "--targets", "si", "ga", "mn", "jv"))
    c = rows(run("x1_did.py", "--boot", B, "--targets", "ga", "jv", "mn", "--pool", "ga", "jv", "mn"))
    for k in ("jv", "mn", "ga", "si", "pooled"):
        assert a[k] == b[k], f"x1 {k} changed under reordering:\n{a[k]}\n{b[k]}"
    for k in ("jv", "mn", "ga", "pooled"):
        assert a[k] == c[k], f"x1 {k} changed when si was dropped:\n{a[k]}\n{c[k]}"
    e = rows(run("e3_noninferiority.py", "--boot", B, "--targets", "de", "ru", "zh", "pt", "id", "ko"))
    f = rows(run("e3_noninferiority.py", "--boot", B, "--targets", "ko", "id", "pt", "zh", "ru", "de"))
    for k in ("de", "ru", "zh", "pt", "id", "ko", "pooled"):
        assert e[k] == f[k], f"e3 {k} changed under reordering:\n{e[k]}\n{f[k]}"
    g = rows(run("e3_noninferiority.py", "--boot", B, "--targets", "de", "ru"))
    for k in ("de", "ru"):
        assert e[k] == g[k], f"e3 {k} changed when other targets were dropped"
    print("invariance: OK (x1 reorder, x1 drop-si, e3 reorder, e3 subset)")


if __name__ == "__main__":
    main()
