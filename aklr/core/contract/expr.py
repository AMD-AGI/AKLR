# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import copy
import functools
import json
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple, NoReturn

from aklr.core.contract.closure import compile_closure, fill_statements

__all__ = [
    "ExprAst",
    "ExprError",
    "check_scope",
    "closure_globals",
    "compile",
    "json_file_text",
    "parse",
    "reject_json_constant",
    "to_ast",
]

# name -> argument count; reserved: never a symbol
_FUNCS = types.MappingProxyType({"ceil_div": 2, "min": 2, "max": 2, "bit_length": 1})
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_STR_CMPOPS = (ast.Eq, ast.NotEq, ast.In)  # the only operators a str may meet
_LITERALS = (int, float, str, bool)


class ExprError(ValueError):
    pass


class ExprAst(NamedTuple):
    text: str
    tree: ast.expr
    symbols: frozenset[str]  # e.g. "m", "a.dtype", "layout_variant"


def parse(text: str) -> ExprAst:
    """Check ``text`` against the grammar; context-free (scopes are ``check_scope``'s)."""
    # Single line and ASCII: ``_parenthesised`` reads the text by column offsets.
    if not text.isascii() or "\n" in text or "\r" in text:
        raise ExprError(f"expression must be single-line ASCII: {text!r}")
    try:
        tree = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError) as e:
        raise ExprError(f"{text!r}: {type(e).__name__}: {e}") from None
    _check(tree, text)
    return ExprAst(text, tree, frozenset(name for _, name in _symbols(tree)))


def check_scope(e: ExprAst, int_symbols: frozenset[str], str_symbols: frozenset[str]) -> None:
    """Every symbol is in scope; a str symbol is only an operand of
    ``==``/``!=``/``in``."""
    unknown = e.symbols - int_symbols - str_symbols
    if unknown:
        raise ExprError(f"{e.text!r}: unknown symbols {sorted(unknown)}")
    compared = {
        id(operand)
        for n in ast.walk(e.tree)
        if isinstance(n, ast.Compare) and isinstance(n.ops[0], _STR_CMPOPS)
        for operand in (n.left, *n.comparators)
    }
    for node, name in _symbols(e.tree):
        if name in str_symbols and id(node) not in compared:
            raise ExprError(f"{e.text!r}: str symbol {name!r} outside ==, != or in")


@functools.lru_cache(maxsize=4096)
def compile(text: str) -> tuple[Callable[[Mapping[str, Any]], Any], frozenset[str]]:
    """``(closure over a symbol -> value mapping, the symbols it reads)``."""
    e = parse(text)
    symbols = ast.Name("symbols", ast.Load())
    body = to_ast(e, lambda s: ast.Subscript(symbols, ast.Constant(s), ast.Load()))
    module = ast.Module(
        fill_statements("def evaluate(symbols):\n    return BODY", BODY=body),
        type_ignores=[],
    )
    fn = compile_closure(module, "evaluate", closure_globals(), f"<expr {text!r}>")

    def closure(symbols: Mapping[str, Any]) -> Any:
        try:
            return fn(symbols)
        except KeyError as missing:  # the only lookup is symbols[symbol]
            raise ExprError(f"{text!r}: unbound symbol {missing}") from None

    return closure, e.symbols


def to_ast(e: ExprAst, symbol: Callable[[str], ast.expr]) -> ast.expr:
    """A fresh Python AST of ``e`` with ``symbol(name)`` in place of every symbol."""
    return _Lower(symbol).visit(copy.deepcopy(e.tree))


def closure_globals() -> dict[str, Any]:
    """Fresh globals for lowered expressions: ``min``, ``max``, ``bit_length``, no
    builtins."""
    return {"__builtins__": {}, "min": min, "max": max, "bit_length": int.bit_length}


def json_file_text(obj: Any) -> str:
    """The text of a committed json file: insertion order, indent 2, number lists on one
    line, one final newline."""
    return _json_text(obj, 0) + "\n"


def reject_json_constant(constant: str) -> NoReturn:
    """``json.loads``'s ``parse_constant``: NaN and infinities raise ``ValueError``."""
    raise ValueError(f"non-finite number {constant}")


def _json_text(value: Any, depth: int) -> str:
    indent = "  " * (depth + 1)
    if isinstance(value, dict) and value:
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"json keys must be str: {list(value)!r}")
        start, end = "{}"
        items = [
            f"{indent}{json.dumps(key, ensure_ascii=False)}: {_json_text(v, depth + 1)}"
            for key, v in value.items()
        ]
    elif isinstance(value, list | tuple) and value:
        if all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
            return json.dumps(value, allow_nan=False)
        start, end = "[]"
        items = [indent + _json_text(v, depth + 1) for v in value]
    else:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return start + "\n" + ",\n".join(items) + "\n" + "  " * depth + end


def _check(node: ast.expr, text: str, str_ok: bool = False) -> None:
    match node:
        case ast.IfExp():
            for child in (node.test, node.body, node.orelse):
                _check(child, text)
        case ast.BoolOp():
            for value in node.values:
                nested = isinstance(value, ast.BoolOp) or (
                    isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                )
                if nested and not _parenthesised(value, text):
                    raise ExprError(f"{text!r}: parenthesise and/or/not inside and/or")
                _check(value, text)
        case ast.UnaryOp(op=ast.Not() | ast.USub()):
            _check(node.operand, text)
        case ast.Compare(ops=[ast.In()], comparators=[ast.Tuple(elts=[_, *_]) as right]):
            _check(node.left, text, str_ok=True)
            for elt in right.elts:
                _check_literal(elt, text)
        case ast.Compare(ops=[ast.In()], comparators=[ast.Constant() as right]) if _parenthesised(
            right, text
        ):
            _check(node.left, text, str_ok=True)
            _check_literal(right, text)
        case ast.Compare(ops=[op], comparators=[right]) if isinstance(op, _CMPOPS):
            str_ok = isinstance(op, _STR_CMPOPS)
            _check(node.left, text, str_ok)
            _check(right, text, str_ok)
        case ast.BinOp(op=op) if isinstance(op, _BINOPS):
            _check(node.left, text)
            _check(node.right, text)
        case ast.Call(func=ast.Name(id=func), args=args, keywords=[]) if (
            func in _FUNCS and len(args) == _FUNCS[func]
        ):
            for arg in node.args:
                _check(arg, text)
        case ast.Name(id=name) if name not in _FUNCS:
            pass
        case ast.Attribute(value=ast.Name(id=name), attr="dtype") if (
            name not in _FUNCS
            # one lexical token: no space around the dot
            and text[node.col_offset : node.end_col_offset] == f"{name}.dtype"
        ):
            pass
        case ast.Constant(value=value) if type(value) in _LITERALS:
            if isinstance(value, str) and not str_ok:
                raise ExprError(f"{text!r}: a str is only an operand of ==, != or in")
        case _:
            raise ExprError(f"{text!r}: not in the grammar: {ast.unparse(node)!r}")


def _check_literal(node: ast.expr, text: str) -> None:
    if not isinstance(node, ast.Constant):
        raise ExprError(f"{text!r}: in takes literals only")
    _check(node, text, str_ok=True)


def _parenthesised(node: ast.expr, text: str) -> bool:
    before = text[: node.col_offset].rstrip()
    after = text[node.end_col_offset :].lstrip()
    return before.endswith("(") and after.startswith(")")


def _symbols(node: ast.expr) -> Iterator[tuple[ast.expr, str]]:
    if isinstance(node, ast.Attribute):
        yield node, f"{node.value.id}.{node.attr}"
    elif isinstance(node, ast.Name):
        if node.id not in _FUNCS:  # a reserved name is a call's func
            yield node, node.id
    else:
        for child in ast.iter_child_nodes(node):
            yield from _symbols(child)


class _Lower(ast.NodeTransformer):
    def __init__(self, symbol: Callable[[str], ast.expr]):
        self.symbol = symbol

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return node if node.id in _FUNCS else self.symbol(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        return self.symbol(f"{node.value.id}.{node.attr}")

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if node.func.id != "ceil_div":
            return node
        a, b = node.args
        floor = ast.BinOp(ast.UnaryOp(ast.USub(), a), ast.FloorDiv(), b)
        return ast.UnaryOp(ast.USub(), floor)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.Constant):
            node.comparators = [ast.Tuple(node.comparators, ast.Load())]
        return node
