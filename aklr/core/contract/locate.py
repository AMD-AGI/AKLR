# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import hashlib
import importlib
import pkgutil
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final

from aklr.core.contract import spec
from aklr.core.contract.semantics import sorted_key_json

__all__ = [
    "OP_PACKAGE_PATTERN",
    "REPOSITORY_SCHEMA_PATH_PATTERN",
    "SCHEMA_PATH_PATTERN",
    "SCHEMA_ROOTS",
    "committed_schema_paths",
    "import_entry",
    "op_dir",
    "op_module",
    "op_package_of",
    "op_packages",
    "schema_op_dir",
    "schema_relative_path",
    "schema_uuid",
]

OP_PACKAGE_PATTERN = re.compile(r"aklr\.(?:ops|plugins(?:\.[a-z][a-z0-9_]*)+)\.[a-z][a-z0-9_]*")
# schema_relative_path's layout, relative to an op directory; main's legacy solution
# files impls/<language>/<dir>/<stem>.json are not schemas
SCHEMA_PATH_PATTERN: Final = r"^impls/(?P<language>[^/]+)/(?P<name>[^/]+)/(?P<uuid>[0-9a-f]{16})\.json$"
# the same, relative to the repository: under an op directory of aklr/ops or aklr/plugins
REPOSITORY_SCHEMA_PATH_PATTERN: Final = r"^aklr/(?:ops|plugins(?:/[^/]+)+)/[^/]+/" + SCHEMA_PATH_PATTERN[1:]
SCHEMA_ROOTS: Final = ("aklr/ops", "aklr/plugins")
_AKLR_DIR: Final = Path(__file__).resolve().parents[2]


def op_packages() -> tuple[str, ...]:
    """Every ``aklr.ops.<pkg>`` package and every plugin op package, sorted."""
    ops = importlib.import_module("aklr.ops")
    ops_packages = (f"aklr.ops.{m.name}" for m in pkgutil.iter_modules(ops.__path__) if m.ispkg)
    return tuple(sorted([*ops_packages, *_plugin_op_packages().values()]))


@functools.cache
def _plugin_op_packages() -> dict[str, str]:
    """``op_type -> package`` of every package under ``aklr/plugins`` holding a
    ``contract.py``: its last name, unique across ``aklr/ops`` and ``aklr/plugins``."""
    packages: dict[str, str] = {}
    plugins_dir = _AKLR_DIR / "plugins"
    for path in sorted(plugins_dir.rglob("contract.py")):
        parts = path.parent.relative_to(plugins_dir).parts
        op_type, package = parts[-1], ".".join(("aklr", "plugins", *parts))
        if op_type in packages or (_AKLR_DIR / "ops" / op_type).is_dir():
            raise ValueError(f"{package}: op_type {op_type!r} is taken")
        packages[op_type] = package
    return packages


def op_package_of(op_type: str) -> str:
    """The package of ``op_type``: a plugin op's, else ``aklr.ops.<op_type>``."""
    return _plugin_op_packages().get(op_type, f"aklr.ops.{op_type}")


def op_module(op_package: str, submodule: str) -> ModuleType:
    """``<op_package>.<submodule>``; ``submodule`` is contract, api, reference or baseline."""
    return importlib.import_module(f"{op_package}.{submodule}")


def op_dir(op_package: str) -> Path:
    """The op package's directory."""
    return Path(op_module(op_package, "contract").__file__).parent


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
