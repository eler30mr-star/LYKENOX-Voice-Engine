"""Inspect the isolated NNSVS runtime without importing the main app."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


PACKAGES = ["nnsvs", "torch", "hydra-core", "librosa", "pysinsy", "nnmnkwii", "pyworld", "h5py"]
MODULES = ["nnsvs", "torch", "hydra", "librosa", "pysinsy", "nnmnkwii", "pyworld", "h5py"]


def version(package: str) -> str | None:
    """Return installed package version or None."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def compiler_status() -> dict[str, object]:
    """Check whether MSVC build tools required for compiled dependencies exist."""

    vc_root = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    vcvars = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
    candidates = sorted(vc_root.glob("*\\bin\\Hostx64\\x64\\cl.exe")) if vc_root.exists() else []
    if not vcvars.exists():
        return {"ok": False, "reason": "vcvars64.bat not found"}
    if not candidates:
        return {"ok": False, "reason": "cl.exe not found under Visual Studio MSVC tools"}
    return {"ok": True, "vcvars64": str(vcvars), "cl": str(candidates[-1])}


def devtest_status(root: Path) -> dict[str, object]:
    """Report the official NIT-SONG070 recipe state without running it."""

    recipe = root / "tools" / "nnsvs_env" / "nnsvs_source" / "recipes" / "nit-song070" / "dev-48k-world"
    data_ok = (recipe / "data" / "list" / "train_no_dev.list").exists()
    stage1_dump = recipe / "dump" / "yoko" / "org" / "train_no_dev"
    return {
        "recipe": "nit-song070/dev-48k-world",
        "stage0_data_prep": "OK" if data_ok else "not_run",
        "stage1_feature_generation": "FAIL" if data_ok and not stage1_dump.exists() else "OK",
        "stage1_error": "ModuleNotFoundError: No module named 'parallel_wavegan'" if data_ok and not stage1_dump.exists() else None,
    }


def main() -> None:
    """Print JSON runtime status for the NNSVS backend env."""

    versions = {package: version(package) for package in PACKAGES}
    imports = {}
    errors = {}
    for module in MODULES:
        try:
            importlib.import_module(module)
            imports[module] = True
        except Exception as exc:  # noqa: BLE001 - diagnostic tool reports exact import failure.
            imports[module] = False
            errors[module] = f"{type(exc).__name__}: {exc}"
    torch_device = {"cuda": False, "xpu": False}
    if imports.get("torch"):
        import torch

        torch_device = {
            "cuda": bool(torch.cuda.is_available()),
            "xpu": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        }
    root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            {
                "available": bool(imports.get("nnsvs") and imports.get("torch")),
                "python": sys.version,
                "versions": versions,
                "imports": imports,
                "errors": errors,
                "compiler": compiler_status(),
                "official_devtest": devtest_status(root),
                "device": "cpu",
                "torch_device": torch_device,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
