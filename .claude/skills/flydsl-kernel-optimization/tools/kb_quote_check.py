#!/usr/bin/env python3
"""Check that every quote in an evidence sidecar is where it says it is.

A sidecar is a tab-separated file drafted beside a card's evidence lines, one row per measured number:

    card_line <TAB> E<n> <TAB> A|B|C <TAB> source <TAB> quote

`source` is one of
    /abs/path/file:LINE                  a file on this machine
    git:/abs/repo@REV:path/in/repo:LINE  a file at a revision
    commit:/abs/repo@REV                 a commit message
and `quote` is text copied from it. A quote matches when, with runs of whitespace and Markdown emphasis
ignored, it occurs within two lines of LINE (anywhere in a commit message). Rows of class C have no source.

    python tools/kb_quote_check.py work/sikl/kb_evidence/flydsl_gemm_small_m.tsv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WINDOW = 2
FILE_RE = re.compile(r"^(/.+):(\d+)$")
GIT_RE = re.compile(r"^git:(/[^@]+)@([^:]+):(.+):(\d+)$")
COMMIT_RE = re.compile(r"^commit:(/[^@]+)@(\S+)$")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[*_`]", "", text).split())


def _window(lines: list[str], line: int) -> str:
    return " ".join(lines[max(0, line - 1 - WINDOW) : line + WINDOW])


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def source_text(source: str) -> str:
    """The text a quote must occur in."""
    if match := FILE_RE.match(source):
        path, line = match.groups()
        return _window(
            Path(path).read_text(encoding="utf-8", errors="replace").splitlines(),
            int(line),
        )
    if match := GIT_RE.match(source):
        repo, rev, path, line = match.groups()
        return _window(_git(repo, "show", f"{rev}:{path}").splitlines(), int(line))
    if match := COMMIT_RE.match(source):
        repo, rev = match.groups()
        return _git(repo, "log", "-1", "--format=%B", rev)
    raise ValueError(f"unknown source form: {source}")


def check(sidecar: Path) -> list[str]:
    problems = []
    for number, raw in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cells = raw.split("\t")
        if len(cells) != 5:
            problems.append(f"{sidecar}:{number}: expected 5 tab-separated cells, got {len(cells)}")
            continue
        _, ident, grade, source, quote = cells
        if grade == "C":
            continue
        try:
            text = source_text(source)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            problems.append(f"{sidecar}:{number}: {ident}: cannot read {source}: {error}")
            continue
        if _norm(quote) not in _norm(text):
            problems.append(f"{sidecar}:{number}: {ident}: quote not found at {source}: {quote[:70]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sidecars", nargs="+", type=Path)
    args = parser.parse_args(argv)
    problems = [p for sidecar in args.sidecars for p in check(sidecar)]
    print("\n".join(problems) if problems else "every quote found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
