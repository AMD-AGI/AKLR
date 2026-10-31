# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
from collections.abc import Mapping

from packaging.specifiers import SpecifierSet
from packaging.version import Version

__all__ = ["toolchain_version", "unmet_requirement"]


def toolchain_version(name: str) -> str | None:
    """The ``__version__`` of the toolchain module ``name`` (a kernel language), None
    when it does not import."""
    try:
        return importlib.import_module(name).__version__
    except ImportError:
        return None


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
