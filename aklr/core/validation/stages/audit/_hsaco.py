###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""SC1-SC2, the audit of an hsaco kernel after SA0-SA4: its Python reaches the GPU only through
``aklr.core.jit.hsaco``, which loads code objects and launches them on torch's current stream. Lexical, on each
parsed source; what the code objects do is caught by bench, and reflection by review."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from aklr.core.schema._kernel import Kernel

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


def check_hsaco_kernel(kernel: Kernel) -> set[tuple[str, int, int, str]]:
    """SC1-SC2 rejections, (path, line, rule, what); a source that does not parse is SA0's."""
    rejections = set()
    for source in kernel.sources:
        try:
            tree = ast.parse(source.content, filename=source.path)
        except (SyntaxError, ValueError, MemoryError):
            continue
        # the module or member each allowed import binds a name to
        bound = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _allowed(alias.name):
                        rejections.add((source.path, node.lineno, 1, f"imports {alias.name}"))
                    elif alias.asname:
                        bound[alias.asname] = alias.name
                    else:
                        bound[alias.name.partition(".")[0]] = alias.name.partition(".")[0]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    name = f"{node.module}.{alias.name}"
                    if not (_allowed(node.module) and _allowed(name) or name == _LAUNCHER):
                        rejections.add((source.path, node.lineno, 1, f"imports {name}"))
                    else:
                        bound[alias.asname or alias.name] = name
        # an attribute chain once, whole
        inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _BUILTINS:
                rejections.add((source.path, node.lineno, 2, f"uses {node.id}"))
            elif isinstance(node, ast.Attribute) and id(node) not in inner:
                # the dotted name from what an allowed import bound its root to
                attrs, root = [], node
                while isinstance(root, ast.Attribute):
                    attrs.append(root.attr)
                    root = root.value
                if isinstance(root, ast.Name) and root.id in bound:
                    qualname = ".".join((bound[root.id], *reversed(attrs)))
                    if qualname.startswith(_TORCH_NATIVE):
                        rejections.add((source.path, node.lineno, 2, f"uses {qualname}"))
    return rejections


def _allowed(module: str) -> bool:
    if module.startswith(_TORCH_NATIVE):
        return False
    return module == _LAUNCHER or module.partition(".")[0] in _MODULES
