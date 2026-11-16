# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in the dtype the inputs decode to: nothing rounded
or quantized in between (the kernels' bf16 steps are theirs) but the router's logits,
rounded once to bf16 as the block's gate returns them, which pick the experts, and what
the checkpoint's FP8 scheme quantizes: each expert linear's input (_fp8_input). The MXFP4
experts' activations stay unquantized: kernels round a block's scale from its maximum in
different ways (to nearest or up) and sglang's own small-M kernel keeps them bf16, so that
error is left to the bound."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "qwen2_moe_fp8_experts_reference",
    "qwen2_moe_fp8_router_experts_reference",
    "qwen2_moe_mxfp4_moe_reference",
    "qwen2_moe_router_reference",
]

_BLOCK = 128  # the FP8 scheme's block: 128 x 128 of a weight, 1 x 128 of an input
_MXFP4_BLOCK = 32  # an MXFP4 weight's block: 1 x 32


def _router_logits(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """hidden_states router_weight^T, exact (float64), rounded to bf16, in the decoded
    dtype."""
    x = decode(row_major_args["hidden_states"], point.dtypes["hidden_states"])
    w = decode(row_major_args["router_weight"], point.dtypes["router_weight"])
    return rounded_to_bfloat16(x.double() @ w.double().T).to(x.dtype)


def _routed(logits: Tensor, topk: int) -> tuple[Tensor, Tensor]:
    """Each row's ``topk`` experts by descending logit, a tie going to the lower id, and
    their weights: the softmax of their logits, which is the softmax over every expert
    renormalized over them. Returns (weights, ids)."""
    import torch

    ids = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :topk]
    return logits.gather(-1, ids).softmax(dim=-1), ids


def _fp8_input(x: Tensor) -> Tensor:
    """``x`` as an fp8 linear of the checkpoint's scheme takes it (sglang's
    per-token-group quant): in bf16, then each 1 x 128 group in fp8 at a float32 scale of
    its amax / 448, dequantized."""
    import torch

    fp8 = torch.float8_e4m3fn
    groups = rounded_to_bfloat16(x).float().unflatten(-1, (-1, _BLOCK))
    scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    return ((groups / scale).to(fp8).to(x.dtype) * scale.to(x.dtype)).flatten(-2)


def _dequantized(weight: Tensor, scale: Tensor) -> Tensor:
    """An fp8 weight [rows, cols] (decoded) times its scale per 128 x 128 block."""
    rows, cols = weight.shape
    blocks = scale.repeat_interleave(_BLOCK, 0)[:rows].repeat_interleave(_BLOCK, 1)
    return weight * blocks[:, :cols]


def _experts(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
    weights: Tensor,
    ids: Tensor,
    shared_slot: int,
) -> Tensor:
    """This rank's share of the block's output: each token's routed experts ``ids``
    weighed by ``weights`` and the shared expert (``shared_slot``, the slot after the
    routed ones) weighed by sigmoid(hidden_states shared_gate_weight^T), each expert
    down(silu(gate(x)) * up(x)) with both linears' inputs quantized (_fp8_input)."""
    import torch

    def f(name: str, e: int | None = None) -> Tensor:
        tensor = row_major_args[name]
        return decode(tensor if e is None else tensor[e], point.dtypes[name])

    x = f("hidden_states")
    shared_weight = (x @ f("shared_gate_weight")[0]).sigmoid()
    ids = torch.cat([ids, torch.full_like(ids[:, :1], shared_slot)], dim=-1)
    weights = torch.cat([weights, shared_weight[:, None]], dim=-1)
    x_fp8 = _fp8_input(x)
    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        w1 = _dequantized(f("w1", e), f("w1_scale", e))
        w2 = _dequantized(f("w2", e), f("w2_scale", e))
        gate, up = (x_fp8[tokens] @ w1.T).chunk(2, dim=-1)
        expert = _fp8_input(torch.nn.functional.silu(gate) * up) @ w2.T
        out.index_add_(0, tokens, weights[tokens, slots, None] * expert)
    return out


def qwen2_moe_router_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(topk_weights, topk_ids) of the bf16 logits (_router_logits, _routed)."""
    logits = _router_logits(point, row_major_args, decode)
    weights, ids = _routed(logits, point.axes["topk"])
    return weights, ids.double()


def qwen2_moe_fp8_experts_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The routed experts topk_ids at topk_weights and the gated shared one (_experts)."""
    weights = decode(row_major_args["topk_weights"], point.dtypes["topk_weights"])
    ids = row_major_args["topk_ids"].long()
    shared_slot = point.axes["n_experts"] - 1
    return _experts(point, row_major_args, decode, weights, ids, shared_slot)


def qwen2_moe_fp8_router_experts_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The router's experts (qwen2_moe_router_reference) and the gated shared one
    (_experts)."""
    logits = _router_logits(point, row_major_args, decode)
    weights, ids = _routed(logits, point.attrs["topk"])
    shared_slot = point.axes["n_routed"]
    return _experts(point, row_major_args, decode, weights, ids, shared_slot)


def _silu_mlp(x: Tensor, gate_up: Tensor, down: Tensor) -> Tensor:
    """down(silu(gate) * up), gate and up the halves of x gate_up^T."""
    import torch

    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


def qwen2_moe_mxfp4_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """The router's experts (_router_logits, _routed) weighed and summed, each on its MXFP4
    weights dequantized per 32 values and the unquantized input, plus the BF16 shared expert
    weighed by sigmoid(hidden_states shared_gate_weight^T). Rounding point: the router's
    logits alone (_router_logits)."""
    import torch

    def f(name: str, e: int | None = None) -> Tensor:
        tensor = row_major_args[name]
        return decode(tensor if e is None else tensor[e], point.dtypes[name])

    def mxfp4(name: str, e: int) -> Tensor:
        return f(name, e) * f(f"{name}_scale", e).repeat_interleave(_MXFP4_BLOCK, dim=-1)

    x = f("hidden_states")
    weights, ids = _routed(_router_logits(point, row_major_args, decode), point.attrs["topk"])
    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        routed = _silu_mlp(x[tokens], mxfp4("w1", e), mxfp4("w2", e))
        out.index_add_(0, tokens, weights[tokens, slots, None] * routed)
    shared_weight = (x @ f("shared_gate_weight")[0]).sigmoid()
    return out + shared_weight[:, None] * _silu_mlp(x, f("shared_w1"), f("shared_w2"))
