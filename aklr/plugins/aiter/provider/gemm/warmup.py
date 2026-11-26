# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a framework plugin's warm-up shares for the GEMMs (16-bit and per-token fp8): the Points this device's
committed kernels serve, and one scratch call of each through aiter's op, as the framework reaches it.

An AKLR kernel compiles at its first eager call; a step with more rows than the framework's own warm-up reaches
would first reach its kernels while serving, and stall every request in that step. A plugin calls each served
kernel once on scratch operands at each m a workload of its schema holds, for the weight shapes its model holds:
no other kernel is built."""

from __future__ import annotations

from typing import Any

import torch

from aklr.core.contract import semantics
from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.runtime.dispatch import DispatcherRegistry

__all__ = [
    "call_once",
    "call_once_a8w8",
    "gemm_a8w8_dispatcher",
    "gemm_dispatcher",
    "served_gemm_points",
    "weight_shapes",
]


def gemm_dispatcher() -> Any:
    """The gemm@1.0 dispatcher, None where it serves nothing."""
    dispatcher = DispatcherRegistry.default().dispatchers().get("gemm@1.0")
    return None if dispatcher is None or dispatcher.table is None else dispatcher


def gemm_a8w8_dispatcher() -> Any:
    """The gemm_a8w8@1.0 dispatcher (the per-token fp8 linears), None where it serves nothing."""
    dispatcher = DispatcherRegistry.default().dispatchers().get("gemm_a8w8@1.0")
    return None if dispatcher is None or dispatcher.table is None else dispatcher


def served_gemm_points(dispatcher: Any) -> list[Any]:
    """The Points of the committed GEMM kernels this device serves: every m a workload holds."""
    codec = FastKeyCodec.from_contract(dispatcher.contract)
    return [codec.to_point(key) for key in dispatcher.table.runnables]


def _aklr_served(aiter: Any) -> bool:
    """Whether AKLR's kernel answered the aiter call just made: no interface in this aiter, or no provider enabled,
    and aiter ran its own."""
    decision = aiter.plugin.last_decision() if hasattr(aiter, "plugin") else None
    return getattr(decision, "provider", None) == "aklr"


def call_once(point: Any, device: torch.device) -> bool:
    """One call of aiter's GEMM at ``point`` (b as [n, k]: aiter takes no other), on zeros; whether AKLR's kernel
    served it."""
    import aiter
    from aiter.tuned_gemm import gemm_a16w16

    def zeros(shape, name):
        return torch.zeros(shape, dtype=semantics.torch_dtypes(point.dtypes[name])[0], device=device)

    m, n, k = (point.axes[a] for a in ("m", "n", "k"))
    gemm_a16w16(
        zeros((m, k), "a"),
        zeros((n, k), "b"),
        zeros((n,), "bias") if point.attrs["has_bias"] else None,
        semantics.torch_dtypes(point.dtypes["out"])[0],
    )
    return _aklr_served(aiter)


def call_once_a8w8(point: Any, device: torch.device) -> bool:
    """One call of aiter's per-token fp8 GEMM at ``point`` (b as [n, k]), on zeros with unit scales; whether AKLR's
    kernel served it. aiter has the op for a shuffled weight without a bias alone (``gemm_a8w8_bpreshuffle``): a
    Point of another layout, or with a bias, no aiter call reaches, and it is not called."""
    if point.layout_variant_name != "b_shuffled" or point.attrs["has_bias"]:
        return False
    import aiter
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle

    def dtype(name):
        return semantics.torch_dtypes(point.dtypes[name])[0]

    m, n, k = (point.axes[a] for a in ("m", "n", "k"))
    gemm_a8w8_bpreshuffle(
        torch.zeros((m, k), dtype=dtype("a"), device=device),
        torch.zeros((n, k), dtype=dtype("b"), device=device),
        torch.ones((m, 1), dtype=dtype("a_scale"), device=device),
        torch.ones((n, 1), dtype=dtype("b_scale"), device=device),
        dtype=dtype("out"),
    )
    return _aklr_served(aiter)


def weight_shapes(models: Any) -> set[tuple[int, int]]:
    """Every 2-D weight shape, either way round, of ``models`` (those that are modules): the (n, k) a GEMM of this
    rank can take."""
    shapes = set()
    for model in models:
        if isinstance(model, torch.nn.Module):
            for tensor in [*model.parameters(), *model.buffers()]:
                if tensor.dim() == 2:
                    shapes |= {tuple(tensor.shape), tuple(tensor.shape)[::-1]}
    return shapes
