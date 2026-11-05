# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch, in the dtype the inputs decode to: nothing rounded or
quantized in between (the kernels' float32 and bf16 steps are theirs) but the router's
logits, rounded once to bf16 as the block's gate returns them, which pick the experts, and
what the checkpoint's FP8 scheme quantizes: each expert linear's input (_fp8_input)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["qwen3_moe_fp8_moe_reference"]

_BLOCK = 128  # the FP8 scheme's block: 128 x 128 of a weight, 1 x 128 of an input


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


def qwen3_moe_fp8_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """This rank's share of the block's output: each token's routed experts (the bf16
    logits' top-k, _router_logits and _routed) weighed by their renormalized softmax, each
    expert down(silu(gate(x)) * up(x)) with both linears' inputs quantized (_fp8_input)."""
    import torch

    def f(name: str, e: int | None = None) -> Tensor:
        tensor = row_major_args[name]
        return decode(tensor if e is None else tensor[e], point.dtypes[name])

    weights, ids = _routed(_router_logits(point, row_major_args, decode), point.attrs["topk"])
    x = f("hidden_states")
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
