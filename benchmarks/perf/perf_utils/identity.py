# SPDX-License-Identifier: MIT
"""Record the AKLR source, the GPUs and the serving packages a measurement ran on."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def aklr_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or "unknown"


def source_dirty() -> bool:
    """Exclude unknown source states from baseline promotion."""
    if "AKLR_SOURCE_DIRTY" in os.environ:
        return os.environ["AKLR_SOURCE_DIRTY"] != "false"
    done = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode != 0 or bool(done.stdout.strip())


def observed_hardware() -> dict:
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPUs are required")
    count = torch.cuda.device_count()
    return {
        "devices": [
            {"name": p.name, "arch": p.gcnArchName.split(":", 1)[0], "total_memory": p.total_memory}
            for p in (torch.cuda.get_device_properties(i) for i in range(count))
        ],
        "peer_access": [
            [i == j or torch.cuda.can_device_access_peer(i, j) for j in range(count)] for i in range(count)
        ],
    }


def server_packages(framework: str) -> dict:
    try:
        return {
            name: importlib.metadata.version(name)
            for name in (framework, "torch", "triton", "amd-aiter", "flydsl")
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"serving package identity is unavailable: {error}") from error
