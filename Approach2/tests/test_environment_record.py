"""The environment guard must see a library that imports but has no dist-info."""
import importlib.metadata
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    import eval_runtime as er

    fake = types.ModuleType("PIL")
    fake.__version__ = "9.5.0.post2"
    sys.modules["PIL"] = fake
    real = importlib.metadata.version

    def no_metadata(name):
        if name in ("pillow", "nowhere"):
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    importlib.metadata.version = no_metadata
    try:
        assert er._package_version("pillow") == "9.5.0.post2", er._package_version("pillow")
        assert er._package_version("nowhere") is None
    finally:
        importlib.metadata.version = real
    print("environment record: OK (import fallback for pillow, None for a truly absent package)")


if __name__ == "__main__":
    main()
