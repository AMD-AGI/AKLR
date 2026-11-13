# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SC1-SC2, the audit of an hsaco kernel after SA0-SA4: its Python reaches the GPU only
through ``aklr.core.jit.hsaco``, which loads code objects and launches them on torch's
current stream. Lexical, on each parsed source; what the code objects do is caught by bench,
and reflection by review."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import TYPE_CHECKING, Final

from aklr.core.validation.stages._report._finding import Finding, FindingCode

if TYPE_CHECKING:
    from aklr.core.contract.spec import KernelSpec

__all__ = ["check_hsaco_kernel"]

# SC1: besides sibling sources (relative imports), the launcher, torch and pure computation
_LAUNCHER: Final = "aklr.core.jit.hsaco"
_MODULES: Final = frozenset(
    {
        *("__future__", "base64", "collections", "dataclasses", "enum", "functools"),
        *("itertools", "math", "struct", "torch", "typing", "zlib"),
    }
)
# SC1 and SC2: torch's ways to native code
_TORCH_NATIVE: Final = (
    *("torch._C", "torch.classes.load_library", "torch.cuda.cudart"),
    *("torch.ops.load_library", "torch.utils.cpp_extension"),
)
# SC2: code from a string, and a module by name
_BUILTINS: Final = frozenset({"__import__", "compile", "eval", "exec", "globals", "vars"})


def check_hsaco_kernel(schema_path: str, kernel: KernelSpec) -> tuple[Finding, ...]:
    """Every SC1-SC2 rejection of the hsaco kernel of the schema at ``schema_path``; a source
    that does not parse is SA0's."""
    rejections = set()
    for source in kernel.sources:
        try:
            tree = ast.parse(source.content, filename=source.path)
        except (SyntaxError, ValueError, MemoryError):
            continue
        rejections |= {(source.path, line, rule, what) for rule, line, what in _rejections(tree)}
    return tuple(
        Finding(
            code=FindingCode.SOURCE_REJECTED,
            schema_path=schema_path,
            axes=None,
            detail=f"SC{rule} {path}:{line} {what}",
        )
        for path, line, rule, what in sorted(rejections)
    )


def _rejections(tree: ast.Module) -> Iterator[tuple[int, int, str]]:
    """(rule, line, what) of each rejection in one source."""
    # the module or member each allowed import binds a name to
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _allowed(alias.name):
                    yield 1, node.lineno, f"imports {alias.name}"
                elif alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[alias.name.partition(".")[0]] = alias.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                name = f"{node.module}.{alias.name}"
                if not (_allowed(node.module) and _allowed(name) or name == _LAUNCHER):
                    yield 1, node.lineno, f"imports {name}"
                else:
                    bound[alias.asname or alias.name] = name
    # an attribute chain once, whole
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BUILTINS:
            yield 2, node.lineno, f"uses {node.id}"
        elif isinstance(node, ast.Attribute) and id(node) not in inner:
            qualname = _qualname(node, bound)
            if qualname is not None and qualname.startswith(_TORCH_NATIVE):
                yield 2, node.lineno, f"uses {qualname}"


def _allowed(module: str) -> bool:
    if module.startswith(_TORCH_NATIVE):
        return False
    return module == _LAUNCHER or module.partition(".")[0] in _MODULES


def _qualname(node: ast.Attribute, bound: dict[str, str]) -> str | None:
    """The dotted name of ``node`` from what its root name is bound to; None when an allowed
    import binds no root."""
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in bound:
        return None
    return ".".join((bound[node.id], *reversed(attrs)))
