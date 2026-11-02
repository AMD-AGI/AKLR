# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import builtins
import copy
import types
from collections.abc import Callable
from typing import Any

__all__ = [
    "compile_closure",
    "fill_expr",
    "fill_statements",
    "join_tests",
    "name_node",
    "shape_tuple",
]


def compile_closure(module: ast.Module, func_name: str, scope: dict[str, Any], origin: str) -> Callable:
    """The function ``func_name`` (without defaults) that ``module`` defines, with
    globals ``scope`` and ``origin`` in tracebacks."""
    tree = ast.fix_missing_locations(module)
    # The module is assembled from AKLR's own metadata, never from runtime input.
    code = compile(tree, origin, "exec")
    (fn_code,) = (c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == func_name)
    # torch.compile inlines these functions (an aiter proxy inside a compiled sglang layer) and
    # reads their globals' __builtins__, which FunctionType does not add as exec does.
    scope.setdefault("__builtins__", builtins)
    fn = types.FunctionType(fn_code, scope, func_name)
    return fn


def fill_expr(template: str, **nodes: Any) -> ast.expr:
    """The expression ``template`` with its placeholders ``nodes`` filled in."""
    return _Placeholders(nodes).visit(ast.parse(template, mode="eval")).body


def fill_statements(template: str, **nodes: Any) -> list[ast.stmt]:
    """The statements of ``template`` with its placeholders ``nodes`` filled in."""
    return _Placeholders(nodes).visit(ast.parse(template)).body


def name_node(identifier: str) -> ast.Name:
    """``identifier`` loaded."""
    return ast.Name(identifier, ast.Load())


def shape_tuple(shape: tuple[str | int, ...]) -> ast.Tuple:
    """A tuple loading each dim of ``shape``: an int as a constant, a str as a name."""
    dims = [ast.Constant(d) if isinstance(d, int) else name_node(d) for d in shape]
    return ast.Tuple(dims, ast.Load())


def join_tests(operator: ast.boolop, tests: list[ast.expr]) -> ast.expr:
    """``tests`` joined by ``operator``; a single test as it is."""
    return tests[0] if len(tests) == 1 else ast.BoolOp(operator, tests)


class _Placeholders(ast.NodeTransformer):
    def __init__(self, nodes: dict[str, Any]) -> None:
        self.nodes = nodes

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id not in self.nodes:
            return node
        new = self.nodes[node.id]
        if isinstance(new, str):
            return ast.Name(new, node.ctx)
        if isinstance(new, ast.AST):
            return copy.deepcopy(new)
        return ast.Constant(new)
