# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A model's launch recipes in the sglang cookbook, at one commit."""

from __future__ import annotations

import http.client
import json
import re
import shlex
import subprocess
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from pydantic import ConfigDict, Field, ValidationError

from aklr.core.contract.spec import FrozenModel

__all__ = ["Cookbook", "CookbookCell", "MatchValue"]

MatchValue = str | int | float | bool


class CookbookLiteralParser:
    """Read plain JavaScript values from cookbook JSX without executing its code."""

    def __init__(self) -> None:
        # Parse plain JavaScript literals; unsupported values become null, other syntax fails.
        self._token = re.compile(
            r"""(?P<skip>\s+|//[^\n]*|/\*.*?\*/)"""
            r"""|(?P<string>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')"""
            r"""|(?P<template>`[^`\\$]*`)"""
            r"""|(?P<key>[A-Za-z_$][\w$]*(?=\s*:))"""
            r"""|(?P<word>(?:true|false|null)\b)"""
            r"""|(?P<number>-?\d[\w.+-]*)"""
            r"""|(?P<punct>[\[\]{},:+])""",
            re.S,
        )
        self._comment_or_string = re.compile(
            r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`""", re.S
        )
        # a non-literal value's tokens, to find where it ends: a `,`, `]` or `}` outside its brackets
        self._expression_token = re.compile(
            r"""\s+|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`"""
            r"""|[\[\](){},]|[^\s"'`\[\](){},]+""",
            re.S,
        )

        # any other one-character escape (\/, \q) is the character itself; a backslash-newline
        # continues the line
        self._character_escapes = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v",
            "0": "\0", "\n": ""}  # fmt: skip
        # an escape of a JavaScript string: \xHH, \uHHHH, \u{H...}, or one character
        self._escape = re.compile(r"\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|[\s\S])")

    def read(self, text: str, key: str) -> object:
        """The value of the ``key: [...]`` or ``key: {...}`` that starts a line; ``None`` if
        no line does."""
        # in the text with its comments and strings blanked, so a commented-out example is no literal
        code = self._comment_or_string.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
        starts = list(re.finditer(rf"^[ \t]*{key}[ \t]*:[ \t]*(?=[\[{{])", code, re.M))
        if len(starts) > 1:
            raise ValueError(f"{key}: {len(starts)} literals")
        if not starts:
            return None
        parts: list[str] = []
        depth, position = 0, starts[0].end()
        while True:
            token = self._token.match(text, position)
            if token is None and parts[-1] == ":":
                position = self._skip_expression(text, position)
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
                value = json.dumps(self._decode_string(value))
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

    def _decode_string(self, value: str) -> str:
        """Decode JavaScript escapes, including unknown single-character escapes."""
        return self._escape.sub(
            lambda match: (
                chr(int(match[1][1:].strip("{}"), 16))
                if match[1][0] in "xu"
                else self._character_escapes.get(match[1], match[1])
            ),
            value[1:-1],
        )

    def _skip_expression(self, text: str, position: int) -> int:
        """Skip an unsupported value up to the delimiter outside its brackets."""
        depth = 0
        while (token := self._expression_token.match(text, position)) is not None:
            if depth == 0 and token.group() in ",]}":
                break
            depth += (token.group() in "([{") - (token.group() in ")]}")
            position = token.end()
        return position


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


class CookbookParser(FrozenModel):
    """One parser toggle declared by the cookbook's playground."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    flag: str


class Cookbook(FrozenModel):
    """The cells of one model configuration (``<vendor>/<model>``) at commit ``sha``."""

    source: str
    sha: str
    config: str
    cells: tuple[CookbookCell, ...]
    docker_images: dict[str, str]
    parsers: tuple[CookbookParser, ...] = ()

    @classmethod
    def fetch(cls, source: str, config: str, repo: str = "sgl-project/sglang") -> Cookbook:
        """``config`` from a local sglang checkout at ``source``, else from ``repo`` at the
        branch, tag or SHA ``source``; a fetch or format fault raises ``ValueError``."""
        config_dir = "docs/src/snippets/configs"
        relative_path = f"{config_dir}/{config}.jsx"
        try:
            # a directory only when it is an sglang checkout: a ref named like a local directory stays a ref
            if (Path(source) / config_dir).is_dir():
                text = (Path(source) / relative_path).read_text(encoding="utf-8")
                done = subprocess.run(
                    ["git", "-C", source, "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                sha = done.stdout.strip() if done.returncode == 0 else "local"
            else:
                sha = cls.pinned(source, repo)
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/{relative_path}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    text = response.read().decode()
        except (OSError, subprocess.SubprocessError, http.client.HTTPException) as error:
            raise ValueError(f"cookbook {config} at {source}: {error}") from error
        return cls.parse(text, source=source, sha=sha, config=config)

    @staticmethod
    def pinned(source: str, repo: str = "sgl-project/sglang") -> str:
        """``source`` with a branch or tag resolved to its commit, so every fetch reads one
        commit; a local checkout or a SHA as it is. ``ValueError`` when it resolves to none."""
        if (Path(source) / "docs/src/snippets/configs").is_dir() or re.fullmatch(r"[0-9a-f]{40}", source):
            return source
        try:
            done = subprocess.run(
                ["git", "ls-remote", f"https://github.com/{repo}", f"refs/heads/{source}",
                 f"refs/tags/{source}^{{}}", f"refs/tags/{source}"],
                capture_output=True, text=True, timeout=60, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"{repo}: {source}: {error}") from error
        if done.returncode != 0:
            raise ValueError(
                f"{repo}: {source}: git ls-remote failed ({done.returncode}): {done.stderr.strip()}"
            )
        refs = {name: sha for sha, name in (line.split() for line in done.stdout.splitlines())}
        # an annotated tag's commit is its peeled ref
        names = (f"refs/heads/{source}", f"refs/tags/{source}^{{}}", f"refs/tags/{source}")
        if not (sha := next((refs[n] for n in names if n in refs), "")):
            raise ValueError(f"{repo}: no branch or tag {source}: {done.stderr.strip()}")
        return sha

    @classmethod
    def parse(cls, text: str, *, source: str, sha: str, config: str) -> Cookbook:
        """The cookbook of a ``.jsx`` config's text; a format it does not know raises ``ValueError``."""
        where = f"cookbook {config} at {sha}"
        try:
            parser = CookbookLiteralParser()
            if (cells := parser.read(text, "cells")) is None:
                raise ValueError("no `cells: [` at the start of a line")
            docker_images = parser.read(text, "dockerImages") or {}
            features = parser.read(text, "playgroundFeatures")
            if features is None:
                features = {}
            if not isinstance(features, dict) or not isinstance(parsers := features.get("parsers", {}), dict):
                raise ValueError("playgroundFeatures and its parsers must be objects")
            items = tuple(CookbookParser.model_validate(item) for item in parsers.get("items", []))
            if len({item.id for item in items}) != len(items):
                raise ValueError("duplicate playground parser IDs")
            return cls(
                source=source,
                sha=sha,
                config=config,
                cells=tuple(CookbookCell.model_validate(cell) for cell in cells),
                docker_images=docker_images,
                parsers=items,
            )
        except (ValueError, TypeError, ValidationError) as error:
            raise ValueError(f"{where}: {error}") from error

    def cell(self, match: Mapping[str, MatchValue]) -> CookbookCell:
        """The one cell whose ``match`` equals ``match``."""
        cells = [c for c in self.cells if c.match == dict(match)]
        if len(cells) != 1:
            raise ValueError(f"cookbook {self.config} at {self.sha}: {len(cells)} cells match {dict(match)}")
        return cells[0]

    def parser_args(self, choices: Mapping[str, bool]) -> dict[str, str | bool | None]:
        """Resolve explicit parser toggles to cookbook flags, or reject unknown choices."""
        args: dict[str, str | bool | None] = {}
        for name, enabled in choices.items():
            where = f"cookbook {self.config} at {self.sha}: matches.parsers.{name}"
            item = next((item for item in self.parsers if item.id == name), None)
            if item is None:
                raise ValueError(f"{where}: no such parser")
            tokens = shlex.split(item.flag)
            if not tokens:
                raise ValueError(f"{where}: empty flag")
            flag, equals, value = tokens[0].partition("=")
            if (
                not re.fullmatch(r"--[A-Za-z][\w-]*", flag)
                or len(tokens) > 2
                or (equals and len(tokens) != 1)
            ):
                raise ValueError(f"{where}: expected one flag with at most one value: {item.flag!r}")
            if len(tokens) == 2 and tokens[1].startswith("--"):
                raise ValueError(f"{where}: expected a value, got {tokens[1]!r}")
            args[flag] = (value if equals else tokens[1] if len(tokens) == 2 else True) if enabled else None
        return args

    def docker_image(self, match: Mapping[str, MatchValue]) -> str | None:
        """The image the cookbook names for ``match``: the ``dockerImages`` key of most
        ``|``-joined match values."""
        values = {str(v) for v in match.values()}
        keys = [k for k in self.docker_images if set(k.split("|")) <= values]
        return self.docker_images[max(keys, key=lambda k: k.count("|"))] if keys else None
