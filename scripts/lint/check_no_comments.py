#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail on a docstring or a comment in the given files; the license header and tool directives stay."""

import ast
import re
import sys
import tokenize

ALLOWED = re.compile(
    r"#+|# SPDX-License-Identifier: .+|# Copyright .+|# (noqa|type: ignore|pyright: ignore|fmt: skip)\b.*"
)


def main() -> int:
    errors = []
    for path in sys.argv[1:]:
        with tokenize.open(path) as file:
            source = file.read()
        errors += [
            f"{path}:{node.lineno}: docstring"
            for node in ast.walk(ast.parse(source, path))
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        errors += [
            f"{path}:{token.start[0]}: comment"
            for token in tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
            if token.type == tokenize.COMMENT and not ALLOWED.fullmatch(token.string)
        ]
    print("\n".join(sorted(errors)), file=sys.stderr, end="\n" if errors else "")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
