# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The standalone source form of a schema function.

A schema keeps ``reference``, ``check`` and ``init`` as Python source so it
dumps to json; :class:`FunctionSource` renders a callable into that source
and loads the source back into a callable. The project functions it calls are
found from its code and inlined ahead of it. A new kind of schema function is one more :data:`FUNCTION_SOURCES` entry;
a kind that needs a different layout subclasses :class:`FunctionSource`.
"""

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
    """How one kind of schema function is kept as standalone source.

    Attributes
    ----------
    preamble:
        Imports put ahead of the function so the source runs on its own.
    entry:
        Top-level name the source binds the function to.
    """

    preamble: str
    entry: str = "run"

    def render(self, fn: Callable) -> str:
        """Return the standalone source of *fn*, bound to :attr:`entry`.

        Raises
        ------
        ValueError
            If *fn* or a dependency is a lambda or a closure, whose source does
            not run alone.
        """
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
        """Return the functions *fn* calls transitively, callees first.

        A dependency is a global name loaded by the code of *fn* (nested defs
        and comprehensions included) bound to a plain function of the top-level
        package of *fn* and not imported by :attr:`preamble`. Override to change
        what is inlined.

        Raises
        ------
        ValueError
            If a function loads any other global that :attr:`preamble` does
            not import, so the standalone source could not bind it.
        """
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
        """Check that *source* is valid Python binding :attr:`entry` at top level.

        Raises
        ------
        ValueError
            If the source does not parse or does not bind :attr:`entry`.
        """
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
        """Execute *source* and return the function bound to :attr:`entry`.

        *filename* names the source in tracebacks.
        """
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
"""The source form of each schema function field, keyed by field name."""
