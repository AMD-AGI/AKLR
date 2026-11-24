#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the layout rules of tests/AGENTS.md that ruff cannot: run from the repository root."""

import ast
import sys
from pathlib import Path


def main() -> int:
    errors = []
    paths = sorted(Path("tests").rglob("*.py"))
    for directory in sorted({parent for path in paths for parent in path.parents if parent.parts}):
        if not (directory / "__init__.py").is_file():
            errors.append(f"{directory}/: no __init__.py")
    for path in paths:
        is_test = path.name.startswith("test_")
        # tests/benchmarks/ tests benchmarks/ at the repository root, outside aklr/
        root = Path() if path.parts[1] == "benchmarks" else Path("aklr")
        package = root.joinpath(*path.parts[1:-1])
        if is_test and not package.is_dir():
            errors.append(f"{path}: no {package}/ to test; a test's directory mirrors aklr/")
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import | ast.ImportFrom):
                # only import statements are seen: importlib.import_module("...") is not
                module = getattr(node, "module", None) or ""
                modules = [module] if module else []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    # `from pkg import name`: name is a module only if pkg/name.py exists
                    parts = module.split(".") if module else []
                    # shortcut: looks under the repository root and beside the file only, a module on
                    # another sys.path entry is not seen; upgrade if tests start importing that way
                    roots = [Path(), path.parent]
                    if node.level:  # a level above the repository resolves nowhere
                        roots = [path.parents[node.level - 1]] if node.level <= len(path.parents) else []
                    modules += [
                        f"{module}.{alias.name}"
                        for alias in node.names
                        if any(root.joinpath(*parts, f"{alias.name}.py").is_file() for root in roots)
                    ]
                for name in modules:
                    last = name.rpartition(".")[2]
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
