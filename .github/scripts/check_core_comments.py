#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail on any comment or docstring in aklr/core's Python: run from the repository root.

Two kinds of comment stay: the licence header at the top of a file, and a bare tool directive.
"""

import ast
import re
import sys
import tokenize
from pathlib import Path

HEADER = re.compile(r"#!.*|#{10,}|#|# SPDX-License-Identifier: .+|# Copyright .+")
# a directive carries no prose; only `structure: allow` must state its reason
DIRECTIVE = re.compile(
    r"# (noqa(: [A-Z]+\d+(, [A-Z]+\d+)*)?"
    r"|(type|pyright): ignore(\[[\w, -]+\])?"
    r"|pragma: no cover"
    r"|fmt: (skip|off|on)"
    r"|structure: allow \S.*)"
)


def findings(path: Path) -> list[str]:
    found = []
    in_header = True
    with path.open("rb") as file:
        for token in tokenize.tokenize(file.readline):
            if token.type == tokenize.COMMENT:
                if not (in_header and HEADER.fullmatch(token.string)) and not DIRECTIVE.fullmatch(token.string):
                    found.append(f"{path}:{token.start[0]}: comment: {token.string}")
            elif token.type not in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE):
                in_header = False
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if ast.get_docstring(node, clean=False) is not None:
                found.append(f"{path}:{node.body[0].lineno}: docstring")
    return found


def main() -> int:
    found = [line for path in sorted(Path("aklr/core").rglob("*.py")) for line in findings(path)]
    print("\n".join(found), file=sys.stderr, end="\n" if found else "")
    return bool(found)


if __name__ == "__main__":
    sys.exit(main())
