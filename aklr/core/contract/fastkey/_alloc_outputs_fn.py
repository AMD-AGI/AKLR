# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING

from aklr.core.contract.closure import (
    fill_expr,
    fill_statements,
    join_tests,
    name_node,
    shape_tuple,
)
from aklr.core.contract.fastkey._generator import ClosureGenerator

if TYPE_CHECKING:
    import torch

    from aklr.core.contract.fastkey._codec import FastKeyCodec


class AllocOutputsFnGenerator(ClosureGenerator):
    """Generates ``alloc_outputs(key, call)``; ``device`` ``None``: torch's default."""

    def __init__(self, codec: FastKeyCodec, device: torch.device | None) -> None:
        import torch

        super().__init__(codec, device, {"_empty": torch.empty})

    def generate(self) -> Callable:
        """Bind the outputs; if one it may make is ``None``, unpack ``key``, derive the
        axes and store ``_empty(<shape>, ...)`` into ``call``; return the result."""
        outputs = self.contract.outputs
        body = [s for o in outputs for s in fill_statements("O = call[K]", O=o, K=ast.Constant(o))]
        allocated = self.contract.allocated_outputs
        if allocated:
            store_names = [ast.Name(n, ast.Store()) for n in self.codec.key_names]
            make = fill_statements("K = key", K=ast.Tuple(store_names, ast.Store()))
            make += self.derived_axis_statements()
            for o in allocated:
                spec = outputs[o]
                shapes = []
                for shape in spec.shape_cases():
                    empty = fill_expr("_empty((), dtype=D, device=_device)", D=f"_dt_{o}")
                    if shape:  # rank 0 keeps ``()``
                        empty.args = shape_tuple(shape).elts
                    shapes.append(empty)
                x = self.by_case(spec, shapes)
                stored = fill_statements("call[K] = O = X", K=ast.Constant(o), O=o, X=x)
                if len(allocated) > 1:
                    stored = [ast.If(fill_expr("O is None", O=o), stored, [])]
                make += stored
            missing = [fill_expr("O is None", O=o) for o in allocated]
            body.append(ast.If(join_tests(ast.Or(), missing), make, []))
        names = list(map(name_node, outputs))
        result = names[0] if len(names) == 1 else ast.Tuple(names, ast.Load())
        body.append(ast.Return(result))
        return self.compile("alloc_outputs", "key, call", body)
