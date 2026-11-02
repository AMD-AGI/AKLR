# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import ast

import torch

from aklr.core.contract.closure import compile_closure, fill_statements


def _closure():
    body = fill_statements("def _f(args, kwargs):\n    return len(args) + kwargs['k']")
    return compile_closure(ast.Module(body, type_ignores=[]), "_f", {}, "<test>")


def test_closure_globals_carry_builtins() -> None:
    """A generated function's globals hold ``__builtins__``, as exec's would."""
    assert "__builtins__" in _closure().__globals__


def test_torch_compile_inlines_a_closure() -> None:
    """Dynamo inlines a generated function called from a compiled one (an aiter proxy
    inside a compiled sglang layer) instead of failing on ``KeyError: __builtins__``."""
    closure = _closure()

    @torch.compile(backend="eager", dynamic=True)
    def compiled(x: torch.Tensor) -> torch.Tensor:
        return x * closure((x, x), {"k": 1})

    x = torch.ones(3)
    assert torch.equal(compiled(x), x * 3)
