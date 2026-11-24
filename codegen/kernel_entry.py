# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The entry of a kernel's sources: the function dispatch calls with a schema's call keys."""

from __future__ import annotations

import ast
import re
import tokenize
from collections.abc import Mapping
from io import StringIO

__all__ = ["KernelEntry"]

# a param of a torch op's schema string: its type, then its name
_HIP_PARAM = r"((?:Tensor(?:\([a-z]!?\))?|float|int|bool|ScalarType|str)\??\s+)(\w+)"
_HIP_STRING = r'("(?:[^"\\]|\\.)*")'


class KernelEntry:
    """The entry ``entry_point`` (``<file>::<symbol>``) of a kernel written in ``language``, in ``sources``."""

    def __init__(self, language: str, entry_point: str, sources: Mapping[str, str]) -> None:
        self.language, self.sources = language, dict(sources)
        self.file, _, self.symbol = entry_point.partition("::")

    def retargeted(
        self, rename: Mapping[str, str], call_keys: tuple[str, ...], constants: Mapping[str, str]
    ) -> dict[str, str]:
        """The sources with the entry taking ``call_keys``: its params renamed by ``rename`` wherever it
        names them, and one no longer passed bound to its value in ``constants`` (python literal text).
        ``ValueError`` where that cannot be done without changing what the kernel computes. A kernel
        without sources has none to change."""
        if not self.sources:
            return {}
        retarget = self._hip if self.language == "hip" else self._python
        return {**self.sources, self.file: retarget(rename, call_keys, constants)}

    def _python(self, rename: Mapping[str, str], call_keys: tuple[str, ...], constants: Mapping[str, str]) -> str:  # structure: allow S4 one language's rules, chosen in retargeted  # fmt: skip
        content, symbol = self.sources[self.file], self.symbol
        (fn,) = [n for n in ast.parse(content).body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        a = fn.args
        if a.args or a.posonlyargs or a.vararg or a.kwarg or a.defaults or any(a.kw_defaults):
            raise ValueError(f"{symbol} takes more than plain keyword-only params")
        params = [x.arg for x in a.kwonlyargs]
        rename = {old: new for old, new in rename.items() if old in params and old != new}
        new_params = [rename.get(p, p) for p in params]
        gone = [p for p in params if rename.get(p, p) not in call_keys]
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | set(params)
        if taken := sorted(n for n in rename.values() if n in names and n not in rename):
            raise ValueError(f"{symbol} already names {taken}")
        lines = content.split("\n")
        edits: list[tuple[int, int, str, str]] = []  # line, utf-8 column, old text, new text
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in rename:
                edits.append((node.lineno, node.col_offset, node.id, rename[node.id]))
            elif isinstance(node, ast.arg) and node.arg in rename:
                edits.append((node.lineno, node.col_offset, node.arg, rename[node.arg]))
            elif isinstance(node, ast.FunctionDef | ast.Lambda) and node is not fn:
                inner = node.args
                if {x.arg for x in (*inner.args, *inner.kwonlyargs, *inner.posonlyargs)} & rename.keys():
                    raise ValueError(f"a function nested in {symbol} takes a param of its name")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == symbol:
                # the entry calling itself: its keywords are its params
                if set(call_keys) - set(new_params):
                    raise ValueError(f"{symbol} calls itself and is to take new params")
                for keyword in node.keywords:
                    if keyword.arg in gone:
                        if keyword.lineno != keyword.end_lineno:
                            raise ValueError(f"{symbol} passes itself {keyword.arg} over several lines")
                        raw = lines[keyword.lineno - 1].encode()
                        end = keyword.end_col_offset + (
                            2 if raw[keyword.end_col_offset :].startswith(b", ") else 0
                        )
                        edits.append(
                            (keyword.lineno, keyword.col_offset, raw[keyword.col_offset : end].decode(), "")
                        )
                    elif keyword.arg in rename:
                        edits.append((keyword.lineno, keyword.col_offset, keyword.arg, rename[keyword.arg]))
        # a name inside a removed keyword goes with it
        removed = [(line, column, len(old.encode())) for line, column, old, new in edits if not new]
        for line, column, old, new in sorted(set(edits), reverse=True):
            if new and any(at == line and start <= column < start + size for at, start, size in removed):
                continue
            raw = lines[line - 1].encode()
            lines[line - 1] = (raw[:column] + new.encode() + raw[column + len(old.encode()) :]).decode()
        if set(new_params) == set(call_keys):
            return "\n".join(lines)
        # the header again with the call keys, and what is no longer passed bound where the body reads it
        body_names = {n.id for statement in fn.body for n in ast.walk(statement) if isinstance(n, ast.Name)}
        bound = []
        for param in gone:
            if param in body_names:
                if param not in constants:
                    raise ValueError(f"{symbol} reads {param}, which is no longer passed and has no constant")
                bound.append(f"{param} = {constants[param]}")
        header = f"def {symbol}(*, {', '.join(call_keys)}):"
        if len(header) > 110:
            header = f"def {symbol}(\n    *,\n" + "".join(f"    {key},\n" for key in call_keys) + "):"
        depth, last = 0, fn.lineno - 1  # the header ends at the first ':' outside brackets
        for token in tokenize.generate_tokens(StringIO("\n".join(lines[fn.lineno - 1 :])).readline):
            depth += (token.string in ("(", "[", "{")) - (token.string in (")", "]", "}"))
            if token.type == tokenize.OP and token.string == ":" and depth == 0:
                last += token.end[0] - 1
                break
        first = fn.body[0]
        is_docstring = isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str)
        after_docstring = lines[last + 1 : first.end_lineno] if is_docstring else []
        rest = lines[first.end_lineno if is_docstring else last + 1 :]
        indent = " " * first.col_offset
        return "\n".join(
            [
                *lines[: fn.lineno - 1],
                *header.split("\n"),
                *after_docstring,
                *(indent + b for b in bound),
                *rest,
            ]
        )

    def _hip(self, rename: Mapping[str, str], call_keys: tuple[str, ...], constants: Mapping[str, str]) -> str:  # structure: allow S4 one language's rules, chosen in retargeted  # fmt: skip
        content, symbol = self.sources[self.file], self.symbol
        registered = re.search(r"m\.def\(\s*((?:\"[^\"]*\"\s*)+),\s*&" + symbol + r"\s*\)", content)
        if registered is None:
            raise ValueError(f"no m.def of {symbol}")
        schema = registered.group(1)
        names = [name for _, name in re.findall(_HIP_PARAM, "".join(re.findall(r'"([^"]*)"', schema)))]
        rename = {old: new for old, new in rename.items() if old in names and old != new}
        for gone in [n for n in names if rename.get(n, n) not in call_keys]:
            if constants.get(gone) not in ("True", "False"):
                raise ValueError(f"{symbol} takes {gone}, which is no longer passed: only a bool is bound")
            # out of the schema string and of the function's params, a constant of its body instead
            content = content.replace(schema, schema := re.sub(rf"bool\s+{gone},\s*", "", schema))
            head = re.search(r"\bvoid\s+" + symbol + r"\s*\((.*?)\)\s*\{", content, re.S)
            content = (
                content[: head.start(1)]
                + re.sub(rf"bool\s+{gone},\s*", "", head.group(1))
                + content[head.end(1) : head.end()]
                + f"\n    const bool {gone} = {constants[gone].lower()};"
                + content[head.end() :]
            )
            names.remove(gone)
        if sorted(rename.get(n, n) for n in names) != sorted(call_keys):
            raise ValueError(f"{symbol} takes {names}, to be renamed into {list(call_keys)}")
        # in the schema string a param's name follows its type; an alias set, Tensor(a!), is no name
        renamed = re.sub(_HIP_PARAM, lambda m: m.group(1) + rename.get(m.group(2), m.group(2)), schema)
        content = content.replace(schema, renamed)
        start = re.search(r"\bvoid\s+" + symbol + r"\s*\(", content)
        if start is None:
            raise ValueError(f"no definition of {symbol}")
        depth, end = 0, content.index("{", start.end())
        while True:
            depth += {"{": 1, "}": -1}.get(content[end], 0)
            end += 1
            if depth == 0:
                break
        function = content[start.start() : end]
        words = set(re.findall(r"\b\w+\b", re.sub(_HIP_STRING, "", function)))
        if not rename or any(new in words and new not in rename for new in rename.values()):
            # the function keeps its own names: the op binds the schema's params to them by position
            return content
        # in a string literal only a name of several letters is one: "a" is a word too
        in_strings = {old: new for old, new in rename.items() if len(old) > 1}
        pieces = re.split(_HIP_STRING, function)
        for i, piece in enumerate(pieces):
            names_here = in_strings if i % 2 else rename
            if names_here:
                pattern = r"(?<![\w.>])(" + "|".join(map(re.escape, names_here)) + r")\b(?!\s*\()"
                pieces[i] = re.sub(pattern, lambda m, names_here=names_here: names_here[m.group(1)], piece)
        return content[: start.start()] + "".join(pieces) + content[end:]
