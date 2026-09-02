from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch


PACKAGES = ["numpy", "scipy", "networkx", "pandas", "scikit-learn", "torch", "matplotlib", "osmnx", "geopandas"]


def main():
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpus": os.cpu_count(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads_for_experiment": 1,
        "packages": {},
    }
    for package in PACKAGES:
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"][package] = None
    destination = Path("outputs/paper_summary/environment.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
