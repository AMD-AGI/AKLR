# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A model's launch recipes in the sglang cookbook, at one commit."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import ConfigDict, ValidationError

from aklr.core.contract.spec import FrozenModel

__all__ = ["COOKBOOK_REPO", "Cookbook", "CookbookCell", "MatchValue"]

COOKBOOK_REPO: Final = "sgl-project/sglang"
_CONFIG_DIR: Final = "docs/src/snippets/configs"
_FETCH_TIMEOUT_SECONDS: Final = 60
# a JavaScript literal of plain values: whitespace, comments, strings, keys, true/false/null,
# numbers and punctuation; anything else (a spread, a call, a template, a variable) fails
_JS_TOKEN: Final = re.compile(
    r"""(?P<skip>\s+|//[^\n]*|/\*.*?\*/)"""
    r"""|(?P<string>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')"""
    r"""|(?P<key>[A-Za-z_$][\w$]*(?=\s*:))"""
    r"""|(?P<word>(?:true|false|null)\b)"""
    r"""|(?P<number>-?\d[\w.+-]*)"""
    r"""|(?P<punct>[\[\]{},:])""",
    re.S,
)

MatchValue = str | int | float | bool


class CookbookCell(FrozenModel):
    """One launch recipe: the selection it answers, its node count, env and server flags."""

    # the cookbook's other keys (specCollapsePp, ...) are the page's, not the launch's
    model_config = ConfigDict(frozen=True, extra="ignore")

    match: dict[str, MatchValue]
    nnodes: int = 1
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
            if Path(source).is_dir():
                text = (Path(source) / relative_path).read_text(encoding="utf-8")
                done = subprocess.run(
                    ["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
                )
                sha = done.stdout.strip() if done.returncode == 0 else "local"
            else:
                sha = source
                if not re.fullmatch(r"[0-9a-f]{40}", source):
                    done = subprocess.run(
                        ["git", "ls-remote", f"https://github.com/{repo}", f"refs/heads/{source}",
                         f"refs/tags/{source}^{{}}", f"refs/tags/{source}"],
                        capture_output=True, text=True, timeout=_FETCH_TIMEOUT_SECONDS, check=False,
                    )  # fmt: skip
                    refs = {name: sha for sha, name in (line.split() for line in done.stdout.splitlines())}
                    # an annotated tag's commit is its peeled ref
                    names = (f"refs/heads/{source}", f"refs/tags/{source}^{{}}", f"refs/tags/{source}")
                    sha = next((refs[name] for name in names if name in refs), "")
                    if done.returncode != 0 or not sha:
                        raise ValueError(f"{repo}: no branch or tag {source}: {done.stderr.strip()}")
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/{relative_path}"
                with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
                    text = response.read().decode()
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"cookbook {config} at {source}: {error}") from error
        return cls.parse(text, source=source, sha=sha, config=config)

    @classmethod
    def parse(cls, text: str, *, source: str, sha: str, config: str) -> Cookbook:
        """The cookbook of a ``.jsx`` config's text; a format it does not know raises ``ValueError``."""
        where = f"cookbook {config} at {sha}"
        try:
            cells = cls._literal(text, "cells")
            docker_images = cls._literal(text, "dockerImages") or {}
            return cls(
                source=source,
                sha=sha,
                config=config,
                cells=tuple(CookbookCell.model_validate(cell) for cell in cells or ()),
                docker_images=docker_images,
            )
        except (ValueError, TypeError, ValidationError) as error:
            raise ValueError(f"{where}: {error}") from error

    @staticmethod
    def _literal(text: str, key: str) -> object:
        """The value of the ``key: [...]`` or ``key: {...}`` that starts a line; ``None`` if
        no line does."""
        starts = list(re.finditer(rf"^[ \t]*{key}[ \t]*:[ \t]*(?=[\[{{])", text, re.M))
        if len(starts) > 1:
            raise ValueError(f"{key}: {len(starts)} literals")
        if not starts:
            return None
        parts: list[str] = []
        depth, position = 0, starts[0].end()
        while True:
            token = _JS_TOKEN.match(text, position)
            if token is None:
                line = text.count("\n", 0, position) + 1
                raise ValueError(
                    f"{key}: not a plain literal at line {line}: {text[position : position + 40]!r}"
                )
            position = token.end()
            value, kind = token.group(), token.lastgroup
            if kind == "skip":
                continue
            if kind == "string":
                value = json.dumps(ast.literal_eval(value))
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
