# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from aklr.core.contract import expr
from aklr.core.contract.closure import (
    compile_closure,
    fill_expr,
    fill_statements,
    name_node,
)
from aklr.core.contract.spec import ShapeByAttrSpec, TensorSpec
from aklr.core.contract.spec._points import case_key

if TYPE_CHECKING:
    import torch

    from aklr.core.contract.fastkey._codec import FastKeyCodec
    from aklr.core.contract.op import OpContract


class ClosureGenerator(ABC):
    """Generates one hot-path closure of a contract and the scope it runs in:
    ``expr.closure_globals()``, ``_device`` and the subclass's ``scope``."""

    def __init__(
        self,
        codec: FastKeyCodec,
        device: torch.device | None,
        scope: dict[str, Any],
    ) -> None:
        self.codec = codec
        self.device = device
        self.scope = {**expr.closure_globals(), **scope, "_device": device}
        # Contract names are closure locals: none may shadow ``call``/``key``/scope names or start with ``_``.
        c = codec.contract
        names = [*c.axes, *c.attrs, *c.inputs, *c.outputs]
        clash = [n for n in names if n in ("call", "key") or n[0] == "_" or n in self.scope]
        if clash:
            raise ValueError(f"{c.contract_id}: {clash} clash with generated names")

    @property
    def contract(self) -> OpContract:
        return self.codec.contract

    @abstractmethod
    def generate(self) -> Callable:
        """The compiled closure."""

    def symbol(self, name: str) -> ast.expr:
        """A contract-scope expression symbol as the closure holds it:
        ``layout_variant`` and ``<t>.dtype`` (adding ``_N_<t>`` to the scope) by
        index; else a local."""
        if name == "layout_variant":
            names = ast.Constant(self.contract.layout_variants.names())
            return fill_expr("N[_layout_variant_idx]", N=names)
        tensor, dot, _ = name.partition(".")
        if dot:  # "<t>.dtype", the only dotted symbol of the contract scope
            names = {**self.codec.dtype_names[tensor], None: None}
            self.scope[f"_N_{tensor}"] = MappingProxyType(names)
            return fill_expr("N[D]", N=f"_N_{tensor}", D=f"_dt_{tensor}")
        return name_node(name)  # axis, derived axis or attr

    def derived_axis_statements(self) -> list[ast.stmt]:
        """``<axis> = <expr>`` for each derived axis, in declaration order."""
        return [
            s
            for axis, a in self.contract.axes.items()
            if a.kind == "derived"
            for s in fill_statements("A = X", A=axis, X=expr.to_ast(expr.parse(a.expr), self.symbol))
        ]

    def by_case(self, spec: TensorSpec, exprs: list[ast.expr]) -> ast.expr:
        """``exprs[i]`` (one per shape case of ``spec``) chosen by the shape's attr."""
        if not isinstance(spec.shape, ShapeByAttrSpec):
            (x,) = exprs
            return x
        name = spec.shape.attr_name
        value_of = {case_key(v): v for v in self.contract.attrs[name].values}
        *branches, (_, result) = zip(spec.shape.cases, exprs, strict=True)
        for key, x in reversed(branches):
            v = ast.Constant(value_of[key])
            result = fill_expr("X if A == V else Y", X=x, A=name, V=v, Y=result)
        return result

    def compile(self, func_name: str, param: str, body: list[ast.stmt]) -> Callable:
        """``def <func_name>(<param>): <body>`` compiled in the scope."""
        module = ast.parse(f"def {func_name}({param}):\n    pass")
        module.body[0].body = body
        origin = f"<{func_name}:{self.contract.contract_id}>"
        return compile_closure(module, func_name, self.scope, origin)
