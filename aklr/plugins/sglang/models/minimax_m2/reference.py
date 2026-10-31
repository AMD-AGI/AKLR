# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in fp32: every input decoded to fp32 and nothing
rounded or quantized in between (the kernels' bf16, fp8 and fp4 steps are theirs) but what
the checkpoints' FP8 scheme quantizes: an fp8 linear's input (_fp8_linear). The MXFP4
experts' activations stay unquantized: the scale rounding of their blocks is the kernel's
(aiter rounds a block's maximum to nearest even, others up), its error left to the bound.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["minimax_m2_fp8_moe_reference", "minimax_m2_mxfp4_moe_reference"]

_FP8_BLOCK = 128
_MXFP4_BLOCK = 32


def _fp32(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Callable[..., Tensor]:
    """Tensor argument ``name`` (its ``index`` only, if given) in fp32."""

    def get(name: str, index: Any = None) -> Tensor:
        tensor = row_major_args[name]
        if index is not None:
            tensor = tensor[index]
        return to_float64(tensor, point.dtypes[name]).float()

    return get


def _routed(point: Any, x: Tensor, router: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Each token's topk experts by sigmoid score plus correction_bias, and their weights:
    the unbiased scores renormalized to sum 1."""
    scores = (x @ router.T).sigmoid()
    ids = (scores + bias).topk(point.attrs["topk"], dim=-1).indices
    weights = scores.gather(-1, ids)
    return ids, weights / weights.sum(dim=-1, keepdim=True)


def _held_experts(
    point: Any,
    row_major_args: Mapping[str, Any],
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    mlp: Callable[[Tensor, int], Tensor],
) -> Tensor:
    """The routes to the experts this rank holds, ``mlp(rows, e)`` held expert e (w1's
    index) on its rows, weighed and summed; the routes to the others add nothing."""
    import torch

    local = ids - row_major_args["expert_offset"]
    out = torch.zeros_like(x)
    for e in local.unique().tolist():
        if not 0 <= e < point.axes["n_experts"]:
            continue
        tokens, slots = (local == e).nonzero(as_tuple=True)
        out.index_add_(0, tokens, weights[tokens, slots, None] * mlp(x[tokens], e))
    return out


def _fp8_linear(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    """x weight^T of an fp8 weight [n, k] with its scale per 128 x 128 block, x taken as the
    checkpoint's FP8 scheme quantizes it (sglang's fp8 MoE on aiter, per_128x128): in bf16,
    then each 1 x 128 group in fp8 at a scale of its amax / 448."""
    import torch

    n, k = weight.shape
    block_scale = scale.repeat_interleave(_FP8_BLOCK, 0)[:n]
    weight = weight * block_scale.repeat_interleave(_FP8_BLOCK, 1)[:, :k]
    fp8 = torch.float8_e4m3fn
    groups = x.bfloat16().float().unflatten(-1, (-1, _FP8_BLOCK))
    group_scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    group_scale = group_scale.clamp(min=torch.finfo(torch.float32).tiny)
    x = ((groups / group_scale).to(fp8).float() * group_scale).flatten(-2)
    return x @ weight.T


def _silu_and_mul(gate_up: Tensor) -> Tensor:
    import torch

    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def minimax_m2_fp8_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token's routed experts this rank holds, both linears of each an fp8 one
    (_fp8_linear), weighed by the renormalized scores and summed."""
    f = _fp32(point, row_major_args, to_float64)
    x = f("hidden_states")
    ids, weights = _routed(point, x, f("router_weight"), f("correction_bias"))

    def mlp(rows: Tensor, e: int) -> Tensor:
        gate_up = _fp8_linear(rows, f("w1", e), f("w1_scale", e))
        return _fp8_linear(_silu_and_mul(gate_up), f("w2", e), f("w2_scale", e))

    return _held_experts(point, row_major_args, x, ids, weights, mlp).double()


def minimax_m2_mxfp4_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token's routed experts this rank holds, their MXFP4 weights dequantized per 32
    values, weighed by the renormalized scores and summed."""
    f = _fp32(point, row_major_args, to_float64)
    x = f("hidden_states")
    ids, weights = _routed(point, x, f("router_weight"), f("correction_bias"))

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        gate_up = rows @ weight("w1", e).T
        return _silu_and_mul(gate_up) @ weight("w2", e).T

    return _held_experts(point, row_major_args, x, ids, weights, mlp).double()
