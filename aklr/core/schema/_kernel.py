###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
#
###############################################################################

import ast
import hashlib
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Annotated, Dict, Final, List, Literal, Optional

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import Field, model_validator

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.toolchain import LIBRARIES, unmet_requirement
from aklr.core.schema._frozen_model import NonEmptyStr, frozen_dataclass

Language = Literal["flydsl", "hip", "hsaco", "triton"]
TOOLCHAINS: Final = {"flydsl": "flydsl", "hip": "hip", "hsaco": "hip", "triton": "triton"}


@frozen_dataclass
class SourceFile:
    path: NonEmptyStr
    content: NonEmptyStr

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(f"Invalid source path (absolute path not allowed): {self.path}")
        if ".." in src_path.parts:
            raise ValueError(f"Invalid source path (parent directory traversal not allowed): {self.path}")
        return self

    @property
    def imports(self) -> set[str]:
        if not self.path.endswith(".py"):
            return set()
        try:
            tree = ast.parse(self.content)
        except (SyntaxError, ValueError, MemoryError):
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.partition(".")[0])
        return names


@frozen_dataclass
class Build:
    language: NonEmptyStr
    target_hardware: Annotated[List[str], Field(min_length=1)]
    entry_point: NonEmptyStr
    requires: Dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    dependencies: List[NonEmptyStr] = Field(default_factory=list)
    destination_passing_style: bool = True
    binding: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "Build":
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f'Invalid entry point format: {self.entry_point}. Expected "<file_path>::<function_name>".'
            )
        return self

    @model_validator(mode="after")
    def _validate_target_hardware(self) -> "Build":
        for name in self.target_hardware:
            try:
                DeviceTarget(*name.split("/"))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid target_hardware entry '{name}'. Expected '<arch>' or "
                    "'<arch>/<hardware>', e.g. 'gfx950' or 'gfx950/MI355X'."
                ) from e
        return self

    @model_validator(mode="after")
    def _validate_requires(self) -> "Build":
        for toolchain, specifier in self.requires.items():
            try:
                SpecifierSet(specifier)
            except InvalidSpecifier as e:
                raise ValueError(f"Invalid requires entry for '{toolchain}': {e}") from e
        return self

    def runs_on(self, target: DeviceTarget) -> bool:
        return (
            DeviceTarget.matches_target(
                target, [DeviceTarget(*name.split("/")) for name in self.target_hardware]
            )
            and unmet_requirement(self.requires) is None
        )


@frozen_dataclass
class Kernel:
    name: NonEmptyStr
    author: NonEmptyStr
    build: Build
    sources: Annotated[List[SourceFile], Field(min_length=1)]
    description: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Kernel":
        seen_paths = set()
        for source in self.sources:
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.build.entry_point.split("::")[0]

        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

        return self

    @property
    def requires_error(self) -> Optional[str]:
        imported = {name for source in self.sources for name in source.imports}
        toolchain = TOOLCHAINS.get(self.build.language, self.build.language)
        for name in (toolchain, *sorted(imported.intersection(LIBRARIES))):
            if name not in self.build.requires:
                return f'requires names no {name} version, such as {{"{name}": ">=<developed on>"}}'
        return None

    def get_entry_path(self) -> Path:
        return Path(self.build.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        return self.build.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        entry_path = self.build.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    @cached_property
    def _hash_cache(self) -> str:
        h = hashlib.sha1()
        for s in (
            self.build.language,
            self.build.entry_point,
            self.build.binding if self.build.binding else "",
            str(self.build.destination_passing_style),
            *self.build.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())
        return h.hexdigest()

    def with_unique_name(self) -> "Kernel":
        return replace(self, name=f"{self.name}_{self.hash()[:8]}")

    def hash(self) -> str:
        return self._hash_cache

    def __hash__(self) -> int:  # pragma: no cover
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        if not isinstance(other, Kernel):
            return NotImplemented
        return self._hash_cache == other._hash_cache
