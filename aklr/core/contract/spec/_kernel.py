# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import keyword
from functools import cached_property
from typing import Annotated, Any, Final, Literal, get_args

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from aklr.core.contract.semantics import sorted_key_json
from aklr.core.contract.semantics._frozen_model import FrozenModel
from aklr.core.contract.spec._sources import SourceFile
from aklr.core.platform.device import DeviceTarget

# kernel.language and impls/<language>/: a new language is a code PR first, then a schema PR.
Language = Literal["flydsl", "hip", "triton"]
LANGUAGES: Final = get_args(Language)
# each language's source suffixes, the entry file's first
SOURCE_SUFFIXES: Final = {"flydsl": (".py",), "hip": (".hip", ".h"), "triton": (".py",)}


class KernelSpec(FrozenModel):
    """A schema's ``kernel``. ``sources`` is empty in an export directory and filled by
    the author; ``spec.write_sources`` writes them."""

    language: Language
    # deduplicated and sorted; a target with no known field is a ValueError
    targets: Annotated[
        tuple[DeviceTarget, ...],
        Field(min_length=1),
        AfterValidator(DeviceTarget.normalize_targets),
    ]
    # the toolchain versions the kernel needs, by language: {"triton": ">=3.8"}; a kernel
    # names at least its own language (requires_error)
    requires: dict[Language, str] = {}  # noqa: RUF012
    # who wrote the kernel: a person, team or agent; "": not recorded
    author: str = ""
    # "<path>::<symbol>"
    entry_point: str
    sources: tuple[SourceFile, ...]

    @field_validator("requires")
    @classmethod
    def _check_requires(cls, requires: dict[str, str]) -> dict[str, str]:
        for name, specifier in requires.items():
            with contextlib.suppress(InvalidSpecifier):
                if SpecifierSet(specifier):  # an empty one requires nothing
                    continue
            raise ValueError(f"{name}: {specifier!r} is not a version specifier such as '>=3.8'")
        return requires

    @field_validator("author")
    @classmethod
    def _check_author(cls, author: str) -> str:
        if author != author.strip() or not author.isprintable():
            raise ValueError(f"{author!r} is not one line without surrounding spaces")
        return author

    @field_validator("entry_point")
    @classmethod
    def _check_entry_point(cls, entry_point: str) -> str:
        symbol = entry_point.partition("::")[2]
        if entry_point.count("::") != 1:
            raise ValueError(f"{entry_point!r} is not <path>::<symbol>")
        if not symbol.isidentifier() or keyword.iskeyword(symbol):
            raise ValueError(f"{symbol!r} is not a Python identifier")
        return entry_point

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, sources: tuple[SourceFile, ...]) -> tuple[SourceFile, ...]:
        paths = [s.path for s in sources]
        if paths != sorted(set(paths)):
            raise ValueError(f"paths are not unique and ascending: {paths}")
        return sources

    @model_validator(mode="after")
    def _check_paths(self) -> KernelSpec:
        suffixes = SOURCE_SUFFIXES[self.language]
        paths = {s.path for s in self.sources}
        if foreign := sorted(p for p in paths if not p.endswith(suffixes)):
            raise ValueError(f"paths do not end in {' or '.join(suffixes)}: {foreign}")
        # a Python kernel is one flat package; C++ includes may nest
        if ".py" in suffixes and (nested := sorted(p for p in paths if "/" in p)):
            raise ValueError(f"paths are not single segments: {nested}")
        if not self.entry_file.endswith(suffixes[0]):
            raise ValueError(f"entry_point file {self.entry_file!r} is not a {suffixes[0]}")
        if paths and self.entry_file not in paths:
            raise ValueError(f"entry_point file {self.entry_file!r} is not in sources")
        return self

    @field_serializer("targets")
    def _serialize_targets(self, targets: tuple[DeviceTarget, ...]) -> list[dict[str, str]]:
        return [{k: v for k, v in dataclasses.asdict(t).items() if v is not None} for t in targets]

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.requires:
            data.pop("requires", None)
        if not self.author:
            data.pop("author", None)
        return data

    @property
    def requires_error(self) -> str | None:
        """Why ``requires`` names no version of the kernel's own language; None when it
        does or there are no sources (a definition)."""
        if not self.sources or self.language in self.requires:
            return None
        return f'kernel.requires names no {self.language} version, such as {{"{self.language}": ">=<developed on>"}}'

    @property
    def entry_file(self) -> str:
        return self.entry_point.partition("::")[0]

    @property
    def entry_symbol(self) -> str:
        return self.entry_point.partition("::")[2]

    def same_kernel(self, other: KernelSpec) -> bool:
        """Equal to ``other`` but perhaps for ``author``, which only records who wrote
        it: validation's test of an unchanged kernel."""
        return self.model_copy(update={"author": ""}) == other.model_copy(update={"author": ""})

    @cached_property
    def content_hash(self) -> str:
        """sha256 of the entry point and the sources, so editing ``targets``, ``requires``
        or ``author`` rebuilds nothing; never in the json."""
        parts: dict[str, Any] = {
            "entry_point": self.entry_point,
            "sources": [[s.path, s.content] for s in self.sources],
        }
        return hashlib.sha256(sorted_key_json(parts).encode()).hexdigest()
