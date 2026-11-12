###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

from aklr.core.validation.stages.validation_scope import SCHEMA_PATH_PATTERN

__all__ = [
    "MAX_DETAIL_CHARACTERS", "Block", "BulletList", "Heading", "Paragraph", "Table",
    "detail_text", "escape", "one_line", "schema_short_name",
]  # fmt: skip

MAX_DETAIL_CHARACTERS: Final = 400
_MARKDOWN_SPECIAL_CHARACTERS: Final = frozenset("\\`*_[]<>|@#&~!")


class Block(ABC):
    @abstractmethod
    def lines(self) -> list[str]: ...


@dataclass(frozen=True)
class Heading(Block):
    level: int
    text: str

    def lines(self) -> list[str]:
        return [f"{'#' * self.level} {self.text}"]


@dataclass(frozen=True)
class Paragraph(Block):
    text: str

    def lines(self) -> list[str]:
        return [self.text] if self.text else []


@dataclass(frozen=True)
class Table(Block):
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def lines(self) -> list[str]:
        if not self.rows:
            return []
        separator = ("---",) * len(self.columns)
        return ["| " + " | ".join(cells) + " |" for cells in (self.columns, separator, *self.rows)]


@dataclass(frozen=True)
class BulletList(Block):
    items: tuple[str, ...]

    def lines(self) -> list[str]:
        return [f"- {item}" for item in self.items]


def escape(text: object) -> str:
    """Markdown-safe text: each special character as its &#N; entity, line breaks as spaces."""
    return "".join(
        (f"&#{ord(ch)};" if ch in _MARKDOWN_SPECIAL_CHARACTERS else " " if ch in "\r\n" else ch)
        for ch in str(text)
    )


def detail_text(detail: str) -> str:
    if len(detail) > MAX_DETAIL_CHARACTERS:
        detail = detail[:MAX_DETAIL_CHARACTERS] + "…"
    return escape(detail)


def one_line(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def schema_short_name(schema_path: str) -> str:
    match = SCHEMA_PATH_PATTERN.fullmatch(schema_path)
    if match is None:
        return escape(schema_path)
    return escape(f"{match['language']}/{match['name']}/{match['uuid'][:8]}")
