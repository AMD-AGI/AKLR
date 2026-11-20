# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping
from typing import Final

from packaging.specifiers import SpecifierSet
from packaging.version import Version

__all__ = ["LIBRARIES", "toolchain_version", "unmet_requirement"]

LIBRARIES: Final = {"aiter": "amd-aiter", "mori": "amd_mori"}


def toolchain_version(name: str) -> str | None:
    if name in LIBRARIES:
        try:
            return importlib.metadata.version(LIBRARIES[name])
        except importlib.metadata.PackageNotFoundError:
            return None
    try:
        if name == "hip":
            import torch

            hip = torch.version.hip
            return hip.split("-")[0] if hip else None
        return importlib.import_module(name).__version__
    except ImportError:
        return None


def unmet_requirement(requires: Mapping[str, str]) -> str | None:
    for name, specifier in requires.items():
        version = toolchain_version(name)
        if version is None:
            return f"{name} is not installed, {specifier} is required"
        if not SpecifierSet(specifier).contains(Version(version).base_version):
            return f"{name} {version} is not {specifier}"
    return None
