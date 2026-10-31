# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import codecs
import os
import re
import stat
import tempfile
import tokenize
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import AfterValidator
from pydantic_core import PydanticCustomError

from aklr.core.contract.semantics._frozen_model import FrozenModel

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails

_SOURCE_PATH_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.(py|hip|h)")
_FORBIDDEN_CHARACTER_PATTERN = re.compile("[\x00-\x09\x0b-\x1f\x7f\x85\u2028\u2029]")


def _check_source_path(path: str) -> str:
    if not _SOURCE_PATH_PATTERN.fullmatch(path) or path.rpartition("/")[2] == "__init__.py":
        raise ValueError(
            f"{path!r} is not a relative path of <identifier> segments ending in "
            "<identifier>.py, .hip or .h, nor an __init__.py"
        )
    return path


def _check_source_content(content: str) -> str:
    """One file's text: utf-8, no control character but ``\\n``, one final newline and
    no blank line at the end."""
    if not content:
        raise ValueError("empty file")
    if content.startswith("\ufeff"):
        raise PydanticCustomError("source_line", "byte order mark", {"line_number": 1})
    if found := _FORBIDDEN_CHARACTER_PATTERN.search(content):
        raise PydanticCustomError(
            "source_line",
            f"forbidden character {found.group()!r}",
            {"line_number": content.count("\n", 0, found.start()) + 1},
        )
    lines = content.split("\n")
    if lines[-1]:
        raise PydanticCustomError("source_line", "no final newline", {"line_number": len(lines)})
    if len(lines) > 1 and not lines[-2]:
        raise PydanticCustomError("source_line", "blank line at the end", {"line_number": len(lines) - 1})
    # Python honours an encoding declared in the first two lines: utf-8 only.
    for line_number in range(1, min(len(lines) - 1, 2) + 1):
        readline = iter([f"{line}\n".encode() for line in lines[:line_number]]).__next__
        try:
            encoding, _ = tokenize.detect_encoding(readline)
        except SyntaxError as e:  # an unknown encoding name
            raise PydanticCustomError("source_line", e.msg, {"line_number": line_number}) from None
        # detect_encoding normalises only utf-8 spellings like "utf_8"; "utf8" comes back as is
        if codecs.lookup(encoding).name != "utf-8":
            raise PydanticCustomError(
                "source_line",
                f"encoding declaration {encoding!r} is not utf-8",
                {"line_number": line_number},
            )
    return content


class SourceFile(FrozenModel):
    path: Annotated[str, AfterValidator(_check_source_path)]
    # the file's whole text, as on main
    content: Annotated[str, AfterValidator(_check_source_content)]


def write_sources(sources: Sequence[SourceFile], directory: Path) -> None:
    """Write each source at its path under ``directory``."""
    for source in sources:
        path = directory / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomically(path, source.content.encode())


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` by ``data`` at once (a failure leaves the old file), keeping its
    permission bits; a new file gets 0o644 (``mkstemp`` alone would leave 0o600)."""
    permission_bits = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            os.fchmod(file.fileno(), permission_bits)
            file.write(data)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def decode_text(where: str, data: bytes) -> str:
    """``data`` decoded; else ``ValueError("<where>:<line_number>: not UTF-8: ...")``."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        line_number = data.count(b"\n", 0, e.start) + 1
        raise ValueError(f"{where}:{line_number}: not UTF-8: {e.reason}") from None


def source_error_parts(detail: ErrorDetails) -> tuple[int | None, str]:
    """``(line_number, reason)`` of one ``ValidationError`` item; ``line_number`` is
    ``None`` off ``content``. Never parses ``msg``."""
    loc, ctx = detail["loc"], detail.get("ctx", {})
    line_number = None
    if loc[-1:] == ("content",) and "line_number" in ctx:
        line_number = ctx["line_number"]
    reason = str(ctx["error"]) if detail["type"] == "value_error" else detail["msg"]
    return line_number, reason
