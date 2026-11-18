#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the layout rules of tests/AGENTS.md that ruff cannot: run from the repository root."""

import ast
import sys
from pathlib import Path


def main() -> int:
    errors = []
    for path in sorted(Path("tests").rglob("*.py")):
        is_test = path.name.startswith("test_")
        package = Path("aklr", *path.parts[1:-1])
        if is_test and not package.is_dir():
            errors.append(f"{path}: no {package}/ to test; a test's directory mirrors aklr/")
        if not (path.parent / "__init__.py").is_file():
            errors.append(f"{path}: no __init__.py in {path.parent}/")
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import | ast.ImportFrom):
                # `from tests.x import test_y` names the module in node.names, not node.module
                module = getattr(node, "module", None)
                names = [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
                names += [module] if module else []
                for name in names:
                    last = name.rpartition(".")[2]
                    # any spelling: `tests.x.test_y`, `.test_y`, bare `test_y`
                    if last.startswith("test_") or last == "conftest":
                        errors.append(
                            f"{path}:{node.lineno}: imports {name}; shared code goes in a helper module, "
                            "shared fixtures in conftest.py"
                        )
            elif isinstance(node, ast.FunctionDef) and not is_test and path.name != "conftest.py":
                for decorator in node.decorator_list:
                    if "fixture" in ast.unparse(decorator):
                        errors.append(
                            f"{path}:{node.lineno}: fixture {node.name} outside conftest.py and test_*.py"
                        )
    print("\n".join(errors), file=sys.stderr, end="\n" if errors else "")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
