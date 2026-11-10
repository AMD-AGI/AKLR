# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in fp32: every input decoded to fp32 and nothing rounded or
quantized in between but what a contract rounds: the router's logits, in the dtype sglang's
MoEGate returns them in, and in deepseek_v2_fp8_moe each fp8 linear's input, as the checkpoint's
FP8 scheme quantizes it (_fp8_linear)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["deepseek_v2_fp8_moe_reference", "deepseek_v2_mxfp4_moe_reference"]


_MXFP4_BLOCK = 32
_FP8_BLOCK = 128


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


def _mlp(x: Tensor, gate_up: Tensor, down: Tensor) -> Tensor:
    """down(silu(gate) * up), gate and up the halves of x gate_up^T."""
    import torch

    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


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


def deepseek_v2_mxfp4_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """On hidden_states, each token's routed experts and the shared one (the last slot) at
    weight 1, the MXFP4 weights dequantized per 32 values. Rounding point: the router's logits,
    exact products of hidden_states and router_weight summed in float64, round once to
    hidden_states' dtype, bf16."""
    args, f = row_major_args, _fp32(point, row_major_args, decode)
    hidden = args["hidden_states"]
    logits = (
        decode(hidden, point.dtypes["hidden_states"]).double()
        @ decode(args["router_weight"], point.dtypes["router_weight"]).double().T
    )
    x = f("hidden_states")
    ids, weights = _routed(point, args, rounded_to_bfloat16(logits).float(), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, weight("w1", e), weight("w2", e))

    return _experts(x, ids, weights, mlp)


def _first_largest(values: Tensor, k: int) -> Tensor:
    """The indices of each row's k largest values, the lower index first among equals."""
    import torch

    return torch.sort(values, dim=-1, descending=True, stable=True).indices[..., :k]


def _grouped_routed(
    point: Any,
    row_major_args: Mapping[str, Any],
    logits: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Each token's topk experts by sigmoid score plus ``bias`` within its topk_group best
    groups, a group scored by its two best biased scores summed, and their weights."""
    import torch

    attrs = point.attrs
    scores = logits.sigmoid()
    biased = scores + bias
    groups = biased.unflatten(-1, (attrs["num_expert_group"], -1))
    group_scores = groups.topk(2, dim=-1).values.sum(dim=-1)
    kept = torch.zeros_like(group_scores, dtype=torch.bool)
    kept.scatter_(-1, _first_largest(group_scores, attrs["topk_group"]), True)
    allowed = kept.repeat_interleave(groups.shape[-1], dim=-1)
    ids = _first_largest(biased.masked_fill(~allowed, float("-inf")), attrs["topk"])
    weights = scores.gather(-1, ids)
    if attrs["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * row_major_args["routed_scaling_factor"]


def _fp8_linear(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    """x weight^T of an fp8 weight [n, k] with its scale per 128 x 128 block, x taken as the
    checkpoint's FP8 scheme quantizes it (sglang's fp8 MoE on aiter, per_128x128): in bf16,
    then each 1 x 128 group in fp8 at a scale of its amax / 448."""
    import torch

    n, k = weight.shape
    block_scale = scale.repeat_interleave(_FP8_BLOCK, 0)[:n]
    weight = weight * block_scale.repeat_interleave(_FP8_BLOCK, 1)[:, :k]
    fp8 = torch.float8_e4m3fn
    groups = x.bfloat16().to(x.dtype).unflatten(-1, (-1, _FP8_BLOCK))
    group_scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    group_scale = group_scale.clamp(min=torch.finfo(torch.float32).tiny)
    x = ((groups / group_scale).to(fp8).to(x.dtype) * group_scale).flatten(-2)
    return x @ weight.T


def deepseek_v2_fp8_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """On hidden_states, each token's routed experts (_grouped_routed) and the shared one (the
    last slot) at weight 1, both linears of each expert an fp8 one. Rounding points: the
    router's logits, exact products of hidden_states and router_weight summed in float64,
    round once to hidden_states' dtype, bf16; each linear's input rounds to bf16, then to fp8
    per 1 x 128 group (_fp8_linear)."""
    args, f = row_major_args, _fp32(point, row_major_args, decode)
    logits = (
        decode(args["hidden_states"], point.dtypes["hidden_states"]).double()
        @ decode(args["router_weight"], point.dtypes["router_weight"]).double().T
    )
    x = f("hidden_states")
    ids, weights = _grouped_routed(point, args, rounded_to_bfloat16(logits).float(), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def mlp(rows: Tensor, e: int) -> Tensor:
        import torch

        gate, up = _fp8_linear(rows, f("w1", e), f("w1_scale", e)).chunk(2, dim=-1)
        return _fp8_linear(torch.nn.functional.silu(gate) * up, f("w2", e), f("w2_scale", e))

    return _experts(x, ids, weights, mlp)
