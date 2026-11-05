# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A model's launch recipes in the sglang cookbook, at one commit."""

from __future__ import annotations

import http.client
import json
import re
import subprocess
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import ConfigDict, Field, ValidationError

from aklr.core.contract.spec import FrozenModel

__all__ = ["COOKBOOK_REPO", "Cookbook", "CookbookCell", "MatchValue"]

COOKBOOK_REPO: Final = "sgl-project/sglang"
_CONFIG_DIR: Final = "docs/src/snippets/configs"
_FETCH_TIMEOUT_SECONDS: Final = 60
# a JavaScript literal of plain values: whitespace, comments, strings (a template with no
# substitution or escape, strings joined by +), keys, true/false/null, numbers and
# punctuation; a value of anything else (a function, a template with a substitution) reads as
# null, and anything else elsewhere (a spread, a variable) fails
_JS_TOKEN: Final = re.compile(
    r"""(?P<skip>\s+|//[^\n]*|/\*.*?\*/)"""
    r"""|(?P<string>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')"""
    r"""|(?P<template>`[^`\\$]*`)"""
    r"""|(?P<key>[A-Za-z_$][\w$]*(?=\s*:))"""
    r"""|(?P<word>(?:true|false|null)\b)"""
    r"""|(?P<number>-?\d[\w.+-]*)"""
    r"""|(?P<punct>[\[\]{},:+])""",
    re.S,
)
_JS_COMMENT_OR_STRING: Final = re.compile(
    r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`""", re.S
)
# a non-literal value's tokens, to find where it ends: a `,`, `]` or `}` outside its brackets
_JS_ANY_TOKEN: Final = re.compile(
    r"""\s+|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`"""
    r"""|[\[\](){},]|[^\s"'`\[\](){},]+""",
    re.S,
)

# any other one-character escape (\/, \q) is the character itself; a backslash-newline
# continues the line
_JS_ONE_CHARACTER_ESCAPES: Final = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v",
    "0": "\0", "\n": ""}  # fmt: skip
# an escape of a JavaScript string: \xHH, \uHHHH, \u{H...}, or one character
_JS_ESCAPE: Final = re.compile(r"\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|[\s\S])")

MatchValue = str | int | float | bool


class CookbookCell(FrozenModel):
    """One launch recipe: the selection it answers, its node count, env and server flags."""

    # the cookbook's other keys (specCollapsePp, ...) are the page's, not the launch's
    model_config = ConfigDict(frozen=True, extra="ignore")

    match: dict[str, MatchValue]
    nnodes: int = Field(default=1, ge=1)
    env: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    verified: bool | None = None
    warn: str | None = None
    redirect: bool | str | None = None

    @property
    def notes(self) -> str:
        """The cell's cautions for the progress and the report; empty when it has none."""
        notes = [] if self.verified is not False else ["verified: false"]
        notes += [
            f"{name}: {value}" for name, value in (("redirect", self.redirect), ("warn", self.warn)) if value
        ]
        return "; ".join(notes)


class Cookbook(FrozenModel):
    """The cells of one model configuration (``<vendor>/<model>``) at commit ``sha``."""

    source: str
    sha: str
    config: str
    cells: tuple[CookbookCell, ...]
    docker_images: dict[str, str]

    @classmethod
    def fetch(cls, source: str, config: str, repo: str = COOKBOOK_REPO) -> Cookbook:
        """``config`` from a local sglang checkout at ``source``, else from ``repo`` at the
        branch, tag or SHA ``source``; a fetch or format fault raises ``ValueError``."""
        relative_path = f"{_CONFIG_DIR}/{config}.jsx"
        try:
            # a directory only when it is an sglang checkout: a ref named like a local directory stays a ref
            if (Path(source) / _CONFIG_DIR).is_dir():
                text = (Path(source) / relative_path).read_text(encoding="utf-8")
                done = subprocess.run(
                    ["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
                )
                sha = done.stdout.strip() if done.returncode == 0 else "local"
            else:
                sha = cls.pinned(source, repo)
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/{relative_path}"
                with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
                    text = response.read().decode()
        except (OSError, subprocess.SubprocessError, http.client.HTTPException) as error:
            raise ValueError(f"cookbook {config} at {source}: {error}") from error
        return cls.parse(text, source=source, sha=sha, config=config)

    @staticmethod
    def pinned(source: str, repo: str = COOKBOOK_REPO) -> str:
        """``source`` with a branch or tag resolved to its commit, so every fetch reads one
        commit; a local checkout or a SHA as it is. ``ValueError`` when it resolves to none."""
        if (Path(source) / _CONFIG_DIR).is_dir() or re.fullmatch(r"[0-9a-f]{40}", source):
            return source
        try:
            done = subprocess.run(
                ["git", "ls-remote", f"https://github.com/{repo}", f"refs/heads/{source}",
                 f"refs/tags/{source}^{{}}", f"refs/tags/{source}"],
                capture_output=True, text=True, timeout=_FETCH_TIMEOUT_SECONDS, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"{repo}: {source}: {error}") from error
        refs = {name: sha for sha, name in (line.split() for line in done.stdout.splitlines())}
        # an annotated tag's commit is its peeled ref
        names = (f"refs/heads/{source}", f"refs/tags/{source}^{{}}", f"refs/tags/{source}")
        if done.returncode != 0 or not (sha := next((refs[n] for n in names if n in refs), "")):
            raise ValueError(f"{repo}: no branch or tag {source}: {done.stderr.strip()}")
        return sha

    @classmethod
    def parse(cls, text: str, *, source: str, sha: str, config: str) -> Cookbook:
        """The cookbook of a ``.jsx`` config's text; a format it does not know raises ``ValueError``."""
        where = f"cookbook {config} at {sha}"
        try:
            if (cells := cls._literal(text, "cells")) is None:
                raise ValueError("no `cells: [` at the start of a line")
            docker_images = cls._literal(text, "dockerImages") or {}
            return cls(
                source=source,
                sha=sha,
                config=config,
                cells=tuple(CookbookCell.model_validate(cell) for cell in cells),
                docker_images=docker_images,
            )
        except (ValueError, TypeError, ValidationError) as error:
            raise ValueError(f"{where}: {error}") from error

    @classmethod
    def _literal(cls, text: str, key: str) -> object:
        """The value of the ``key: [...]`` or ``key: {...}`` that starts a line; ``None`` if
        no line does."""
        # in the text with its comments and strings blanked, so a commented-out example is no literal
        code = _JS_COMMENT_OR_STRING.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
        starts = list(re.finditer(rf"^[ \t]*{key}[ \t]*:[ \t]*(?=[\[{{])", code, re.M))
        if len(starts) > 1:
            raise ValueError(f"{key}: {len(starts)} literals")
        if not starts:
            return None
        parts: list[str] = []
        depth, position = 0, starts[0].end()
        while True:
            token = _JS_TOKEN.match(text, position)
            if token is None and parts[-1] == ":":
                n_open = 0
                while (token := _JS_ANY_TOKEN.match(text, position)) is not None:
                    if n_open == 0 and token.group() in ",]}":
                        break
                    n_open += (token.group() in "([{") - (token.group() in ")]}")
                    position = token.end()
                parts.append("null")
                continue
            if token is None:
                line = text.count("\n", 0, position) + 1
                raise ValueError(
                    f"{key}: not a plain literal at line {line}: {text[position : position + 40]!r}"
                )
            position = token.end()
            value, kind = token.group(), token.lastgroup
            if kind == "skip":
                continue
            if kind in ("string", "template"):
                # JavaScript's escapes, not Python's: \/ is /, \q is q, \a is a
                value = json.dumps(
                    _JS_ESCAPE.sub(
                        lambda m: (
                            chr(int(m[1][1:].strip("{}"), 16))
                            if m[1][0] in "xu"
                            else _JS_ONE_CHARACTER_ESCAPES.get(m[1], m[1])
                        ),
                        value[1:-1],
                    )
                )
                if parts[-1] == "+":
                    parts.pop()
                    value = json.dumps(json.loads(parts.pop()) + json.loads(value))
            elif kind == "key":
                value = json.dumps(value)
            elif value in "]}":
                if parts[-1] == ",":  # a trailing comma
                    parts.pop()
                depth -= 1
            elif value in "[{":
                depth += 1
            parts.append(value)
            if depth == 0:
                return json.loads("".join(parts))

    def cell(self, match: Mapping[str, MatchValue]) -> CookbookCell:
        """The one cell whose ``match`` equals ``match``."""
        cells = [c for c in self.cells if c.match == dict(match)]
        if len(cells) != 1:
            raise ValueError(f"cookbook {self.config} at {self.sha}: {len(cells)} cells match {dict(match)}")
        return cells[0]

    def docker_image(self, match: Mapping[str, MatchValue]) -> str | None:
        """The image the cookbook names for ``match``: the ``dockerImages`` key of most
        ``|``-joined match values."""
        values = {str(v) for v in match.values()}
        keys = [k for k in self.docker_images if set(k.split("|")) <= values]
        return self.docker_images[max(keys, key=lambda k: k.count("|"))] if keys else None
