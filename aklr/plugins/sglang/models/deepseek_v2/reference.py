# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the deepseek_v2 contracts say, in torch, in fp32: every input decoded to fp32 and nothing
rounded or quantized in between but what the contracts round: the router's logits, in the dtype
sglang's MoEGate returns them in, and for fp8 experts an fp8 linear's input (_fp8_linear)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["deepseek_v2_fp8_moe_reference", "deepseek_v2_mxfp4_moe_reference"]


_FP8_BLOCK = 128
_MXFP4_BLOCK = 32


def _fp32(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Callable[..., Tensor]:
    """Tensor argument ``name`` (its ``index`` only, if given) in fp32."""

    def get(name: str, index: Any = None) -> Tensor:
        tensor = row_major_args[name]
        if index is not None:
            tensor = tensor[index]
        return decode(tensor, point.dtypes[name]).float()

    return get


def _silu_and_mul(gate_up: Tensor) -> Tensor:
    """silu(gate) * up, gate and up the halves of ``gate_up``."""
    import torch

    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _routed(
    point: Any,
    row_major_args: Mapping[str, Any],
    logits: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Each token's topk experts by sigmoid score plus ``bias``, and their weights."""
    scores = logits.sigmoid()
    ids = (scores + bias).topk(point.attrs["topk"], dim=-1).indices
    weights = scores.gather(-1, ids)
    if point.attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * row_major_args["routed_scaling_factor"]


def _experts(
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    mlp: Callable[[Tensor, int], Tensor],
) -> Tensor:
    """Each token's experts, ``mlp(rows, e)`` expert e on its rows, weighed and summed."""
    import torch

    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        out.index_add_(0, tokens, weights[tokens, slots, None] * mlp(x[tokens], e))
    return out


def _with_shared_slot(point: Any, ids: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """The routing with the shared expert appended: the last slot, at weight 1."""
    import torch

    shared = torch.full_like(ids[:, :1], point.axes["n_routed"])
    return (
        torch.cat([ids, shared], dim=-1),
        torch.cat([weights, torch.ones_like(weights[:, :1])], dim=-1),
    )


def _routing(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """Each token's routed experts and the shared one (the last slot) at weight 1. Rounding
    point: the router's logits, exact products of hidden_states and router_weight summed in
    float64, round once to hidden_states' dtype, bf16."""
    args, f = row_major_args, _fp32(point, row_major_args, decode)
    logits = (
        decode(args["hidden_states"], point.dtypes["hidden_states"]).double()
        @ decode(args["router_weight"], point.dtypes["router_weight"]).double().T
    )
    ids, weights = _routed(point, args, rounded_to_bfloat16(logits).float(), f("correction_bias"))
    return _with_shared_slot(point, ids, weights)


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


def deepseek_v2_fp8_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """On hidden_states, each token's routed experts and the shared one (the last slot) at
    weight 1 (_routing), both linears of each an fp8 one (_fp8_linear)."""
    f = _fp32(point, row_major_args, decode)
    ids, weights = _routing(point, row_major_args, decode)

    def mlp(rows: Tensor, e: int) -> Tensor:
        gate_up = _fp8_linear(rows, f("w1", e), f("w1_scale", e))
        return _fp8_linear(_silu_and_mul(gate_up), f("w2", e), f("w2_scale", e))

    return _experts(f("hidden_states"), ids, weights, mlp)


def deepseek_v2_mxfp4_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """On hidden_states, each token's routed experts and the shared one (the last slot) at
    weight 1 (_routing), the MXFP4 weights dequantized per 32 values."""
    f = _fp32(point, row_major_args, decode)
    ids, weights = _routing(point, row_major_args, decode)

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _silu_and_mul(rows @ weight("w1", e).T) @ weight("w2", e).T

    return _experts(f("hidden_states"), ids, weights, mlp)
