# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What deepseek_v2_mxfp4_moe's contract says, in torch, in fp32: every input decoded to fp32 and nothing
rounded or quantized in between but what the contract rounds: the router's logits, in the dtype
sglang's MoEGate returns them in."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["deepseek_v2_mxfp4_moe_reference"]


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
    exact products of hidden_states and router_weight summed in float64, round to
    hidden_states' dtype."""
    args, f = row_major_args, _fp32(point, row_major_args, decode)
    hidden = args["hidden_states"]
    logits = (
        decode(hidden, point.dtypes["hidden_states"]).double()
        @ decode(args["router_weight"], point.dtypes["router_weight"]).double().T
    )
    x = f("hidden_states")
    ids, weights = _routed(point, args, logits.to(hidden.dtype).float(), f("correction_bias"))
    ids, weights = _with_shared_slot(point, ids, weights)

    def weight(name: str, e: int) -> Tensor:
        scale = f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)
        return f(name, e) * scale

    def mlp(rows: Tensor, e: int) -> Tensor:
        return _mlp(rows, weight("w1", e), weight("w2", e))

    return _experts(x, ids, weights, mlp).double()
