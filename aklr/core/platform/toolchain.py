# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping
from typing import Final

from packaging.specifiers import SpecifierSet
from packaging.version import Version

__all__ = ["toolchain_version", "unmet_requirement"]

# libraries read from their installed distribution, not imported: importing mori sizes its heap
_DISTRIBUTIONS: Final = {"mori": "amd_mori"}


def toolchain_version(name: str) -> str | None:
    """``name``'s installed version, None when it is missing: a language's module
    ``__version__``, for ``hip`` the HIP torch was built with, for a library its
    distribution's version."""
    if name == "hip":
        return _hip_version()
    if name in _DISTRIBUTIONS:
        try:
            return importlib.metadata.version(_DISTRIBUTIONS[name])
        except importlib.metadata.PackageNotFoundError:
            return None
    try:
        return importlib.import_module(name).__version__
    except ImportError:
        return None


def _hip_version() -> str | None:
    """``torch.version.hip`` without its build suffix ("7.15.26333-abc" -> "7.15.26333");
    None without a ROCm torch."""
    try:
        import torch
    except ImportError:
        return None
    hip = torch.version.hip
    return hip.split("-")[0] if hip else None


def unmet_requirement(requires: Mapping[str, str]) -> str | None:
    """Why the installed toolchains miss ``requires`` (``kernel.requires``), None when
    they meet it; a dev or rc build counts as its release."""
    for name, specifier in requires.items():
        version = toolchain_version(name)
        if version is None:
            return f"{name} is not installed, {specifier} is required"
        if not SpecifierSet(specifier).contains(Version(version).base_version):
            return f"{name} {version} is not {specifier}"
    return None
