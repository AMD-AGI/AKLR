###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import ast
import dis
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType, FunctionType
from typing import Any


@dataclass(frozen=True)
class FunctionSource:
    """How one kind of schema function (reference, check, init) is kept as standalone source."""

    preamble: str
    entry: str = "run"

    def render(self, fn: Callable) -> str:
        """The standalone source of fn: preamble, the project functions it calls, fn, bound to entry."""
        functions = (*self.dependencies(fn), fn)
        for f in functions:
            if f.__name__ == "<lambda>" or f.__code__.co_freevars:
                raise ValueError(
                    f"{f.__qualname__} must be a plain def without free variables to be kept as source"
                )
        parts = [self.preamble]
        parts += [textwrap.dedent(inspect.getsource(f)) for f in functions]
        if fn.__name__ != self.entry:
            parts.append(f"{self.entry} = {fn.__name__}\n")
        return "\n".join(parts)

    def dependencies(self, fn: Callable) -> tuple[FunctionType, ...]:
        imported = {
            (alias.asname or alias.name).partition(".")[0]
            for node in ast.parse(self.preamble).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        package = fn.__module__.partition(".")[0]
        seen: set[Callable] = {fn}
        ordered: list[FunctionType] = []

        def visit(f: Callable) -> None:
            codes = [f.__code__]
            for code in codes:
                # nested defs and comprehensions: code objects in co_consts, visited by this same loop
                codes += [c for c in code.co_consts if isinstance(c, CodeType)]
                for instruction in dis.get_instructions(code):
                    name = instruction.argval
                    if instruction.opname != "LOAD_GLOBAL" or name in imported or name not in f.__globals__:
                        continue
                    dep = f.__globals__[name]
                    if not (isinstance(dep, FunctionType) and dep.__module__.partition(".")[0] == package):
                        raise ValueError(
                            f"{f.__qualname__} uses global '{name}' that the standalone source cannot bind"
                        )
                    if dep not in seen:
                        seen.add(dep)
                        visit(dep)
                        ordered.append(dep)

        visit(fn)
        return tuple(ordered)

    def validate(self, source: str) -> None:
        try:
            module = ast.parse(source, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Source must be valid Python code: {e}") from e
        bound = {node.name for node in module.body if isinstance(node, ast.FunctionDef)} | {
            target.id
            for node in module.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if self.entry not in bound:
            raise ValueError(f"Source must bind a top-level name '{self.entry}'")
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(m.partition(".")[0] == "aklr" for m in modules):
                raise ValueError("Source must not import aklr")

    def load(self, source: str, filename: str) -> Callable:
        namespace: dict[str, Any] = {}
        code = compile(source, filename, "exec")
        exec(code, namespace)  # noqa: S102 schema sources are trusted repo files
        return namespace[self.entry]


FUNCTION_SOURCES: dict[str, FunctionSource] = {
    "reference": FunctionSource(
        preamble=(
            "from __future__ import annotations\nimport math\nimport torch\nimport torch.nn.functional as F\n"
        ),
    ),
    "check": FunctionSource(
        preamble=("from __future__ import annotations\nimport torch\n"),
    ),
    "init": FunctionSource(
        preamble=("from __future__ import annotations\nimport math\nimport torch\n"),
    ),
}
