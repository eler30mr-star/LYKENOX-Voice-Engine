"""Inspect the isolated NNSVS runtime without importing the main app."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys


PACKAGES = ["nnsvs", "torch", "hydra-core", "librosa", "pysinsy", "nnmnkwii", "pyworld"]


def version(package: str) -> str | None:
    """Return installed package version or None."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    """Print JSON runtime status for the NNSVS backend env."""

    versions = {package: version(package) for package in PACKAGES}
    imports = {}
    errors = {}
    for module in ["nnsvs", "torch", "hydra", "librosa", "pysinsy", "nnmnkwii", "pyworld"]:
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
    print(
        json.dumps(
            {
                "available": bool(imports.get("nnsvs") and imports.get("torch")),
                "python": sys.version,
                "versions": versions,
                "imports": imports,
                "errors": errors,
                "device": "cpu",
                "torch_device": torch_device,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
