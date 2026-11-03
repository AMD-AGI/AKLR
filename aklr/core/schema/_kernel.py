###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

import hashlib
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import Field, model_validator

from aklr.core.platform.device import DeviceTarget
from aklr.core.schema._frozen_model import NonEmptyStr, frozen_dataclass

# build.language and impls/<language>/ values; a new language is a code PR first, then a schema PR
Language = Literal["flydsl", "hip", "triton"]


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


@frozen_dataclass
class Build:
    language: NonEmptyStr
    target_hardware: Annotated[List[str], Field(min_length=1)]
    entry_point: NonEmptyStr
    dependencies: List[NonEmptyStr] = Field(default_factory=list)
    destination_passing_style: bool = True  # True: outputs come as the last arguments; False: returned
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

    def runs_on(self, target: DeviceTarget) -> bool:
        return DeviceTarget.matches_target(
            target, [DeviceTarget(*name.split("/")) for name in self.target_hardware]
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
        """SHA1 of the kernel's content, name excluded; the build cache's key."""
        return self._hash_cache

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial wrapper
        if not isinstance(other, Kernel):
            return NotImplemented
        return self._hash_cache == other._hash_cache
