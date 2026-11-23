# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import hashlib
import importlib
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

from aklr.core.contract import spec
from aklr.core.contract.semantics import sorted_key_json

if TYPE_CHECKING:
    from aklr.core.contract.op import OpContract

__all__ = [
    "OP_PACKAGE_PATTERN",
    "REPOSITORY_SCHEMA_PATH_PATTERN",
    "SCHEMA_PATH_PATTERN",
    "SCHEMA_ROOTS",
    "committed_schema_paths",
    "import_entry",
    "legacy_aiter_path",
    "op_api_modules",
    "op_contracts",
    "op_dir",
    "op_dirs",
    "op_module",
    "op_package_of",
    "op_packages",
    "schema_op_dir",
    "schema_relative_path",
    "schema_uuid",
]

OP_PACKAGE_PATTERN = re.compile(r"aklr\.plugins(?:\.[a-z][a-z0-9_]*){2,}")
# schema_relative_path's layout, relative to an op directory; main's legacy solution
# files impls/<language>/<dir>/<stem>.json are not schemas
SCHEMA_PATH_PATTERN: Final = r"^impls/(?P<language>[^/]+)/(?P<name>[^/]+)/(?P<uuid>[0-9a-f]{16})\.json$"
# the same, relative to the repository: under an op directory of aklr/ops or aklr/plugins
REPOSITORY_SCHEMA_PATH_PATTERN: Final = r"^aklr/(?:ops|plugins(?:/[^/]+)+)/[^/]+/" + SCHEMA_PATH_PATTERN[1:]
SCHEMA_ROOTS: Final = ("aklr/ops", "aklr/plugins")
_MOVED_FAMILIES = frozenset(
    {
        "attention",
        "fused_moe",
        "gemm",
        "kda_conv",
        "mega_moe",
        "mhc",
        "mla",
        "mla_prefill",
        "quant",
        "rms_norm",
        "topk",
    }
)
_AKLR_DIR: Final = Path(__file__).resolve().parents[2]


def legacy_aiter_path(path: PurePosixPath) -> PurePosixPath | None:
    """The exact pre-migration counterpart, for reading historical Git trees and reports."""
    parts = path.parts
    if len(parts) >= 4 and parts[:3] == ("aklr", "plugins", "aiter") and parts[3] in _MOVED_FAMILIES:
        return PurePosixPath("aklr", "ops", *parts[3:])
    return None


def op_packages() -> tuple[str, ...]:
    """Every installed plugin family which publishes contracts."""
    return tuple(sorted(_plugin_op_packages().values()))


def op_dirs(plugins_dir: Path) -> tuple[Path, ...]:
    """Family directories in this checkout or installation, without importing any code."""
    families = []
    for directory, children, files in os.walk(plugins_dir):
        children[:] = [name for name in children if name not in {"impls", "__pycache__"}]
        if "plugin.py" in files or "contract.py" in files:
            path = Path(directory)
            if len(path.relative_to(plugins_dir).parts) >= 2:
                families.append(path)
    return tuple(sorted(families))


@functools.cache
def _plugin_op_packages() -> dict[str, str]:
    """Installed family packages by globally unique op type."""
    packages: dict[str, str] = {}
    plugins_dir = _AKLR_DIR / "plugins"
    for path in op_dirs(plugins_dir):
        parts = path.relative_to(plugins_dir).parts
        op_type, package = parts[-1], ".".join(("aklr", "plugins", *parts))
        if op_type in packages:
            raise ValueError(f"{package}: op_type {op_type!r} is taken by {packages[op_type]}")
        packages[op_type] = package
    return packages


def op_package_of(op_type: str) -> str:
    """The unique installed plugin family for an op type."""
    try:
        return _plugin_op_packages()[op_type]
    except KeyError:
        raise ValueError(f"unknown op_type {op_type!r}") from None


def op_module(op_package: str, submodule: str) -> ModuleType:
    """``<op_package>.<submodule>``; ``submodule`` can be plugin, reference, baseline, group or a legacy module."""
    return importlib.import_module(f"{op_package}.{submodule}")


@functools.cache
def op_dir(op_package: str) -> Path:
    """Locate a family without importing its partially constructed module."""
    path = _AKLR_DIR.parent.joinpath(*op_package.split("."))
    if not OP_PACKAGE_PATTERN.fullmatch(op_package) or not path.is_dir():
        raise ModuleNotFoundError(f"No op package {op_package!r}", name=op_package)
    return path


@functools.cache
def op_api_modules(op_package: str) -> tuple[str, str]:
    """Definition and public import module names, including mixed-layout installations."""
    directory = op_dir(op_package)
    definition = "plugin" if (directory / "plugin.py").is_file() else "api"
    public = "api" if (directory / "api.py").is_file() else definition
    return f"{op_package}.{definition}", f"{op_package}.{public}"


def op_contracts(op_package: str) -> tuple[OpContract, ...]:
    """Read the family's publication once, including families from the earlier layout."""
    definition, _ = op_api_modules(op_package)
    if definition.endswith(".plugin"):
        return importlib.import_module(definition).family.contracts
    return op_module(op_package, "contract").CONTRACTS


def import_entry(entry: str) -> Any:
    """The object a dotted path names, split at its last dot; ``ValueError`` if it has
    no dot."""
    module, _, name = entry.rpartition(".")
    if not module or not name:
        raise ValueError(f"not a dotted entry: {entry!r}")
    return getattr(importlib.import_module(module), name)


def schema_relative_path(schema_name: str, language: str, uuid: str) -> PurePosixPath:
    """The schema file's path relative to its op directory."""
    return PurePosixPath("impls", language, schema_name, f"{uuid}.json")


def schema_op_dir(path: Path) -> Path:
    """The op directory of the schema file at ``path``, absolute or relative as
    ``path``."""
    return path.parents[3]


def committed_schema_paths(op_dir: Path, schema_name: str | None = None) -> list[Path]:
    """The sorted ``impls/<language>/<schema_name>/*.json`` files of an op directory,
    for every ``spec.LANGUAGES`` language (every name when ``schema_name`` is None)."""
    paths: list[Path] = []
    for language in spec.LANGUAGES:
        if schema_name is None:
            paths += (op_dir / "impls" / language).glob("*/*.json")
        else:
            paths += (op_dir / "impls" / language / schema_name).glob("*.json")
    return sorted(p for p in paths if p.is_file())


def schema_uuid(schema_name: str, authored_parts: Mapping[str, Any]) -> str:
    """First 16 hex digits of the sha256 of ``[schema_name, authored_parts]``: authored
    edits move the file; measurements and workload order never do."""
    text = sorted_key_json([schema_name, authored_parts])
    return hashlib.sha256(text.encode()).hexdigest()[:16]
