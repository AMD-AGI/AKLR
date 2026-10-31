# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["fused_moe_reference"]

_BLOCK = 32
_SWIGLU_BF16_ROWS = 256  # aiter's GPTOSS_SWIGLU_MXFP4_BF16_BOUND default


def _mxfp4(x: Tensor, grid: Tensor) -> Tensor:
    import torch

    blocks = x.unflatten(-1, (-1, _BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    finite = amax.isfinite()
    mantissa, exponent = torch.frexp(torch.where(finite, amax, 0.0).clamp_min(1e-10))
    exponent = (exponent - 3 + (mantissa > 0.75).int()).clamp(-127, 127)
    scale = torch.where(finite, torch.exp2(exponent.double()), float("nan"))
    scaled = torch.where(finite, blocks, 0.0) / scale
    m = scaled.abs()
    lo = torch.bucketize(m, grid, right=True) - 1  # the largest code <= m
    hi = (lo + 1).clamp_max(len(grid) - 1)
    below, above = m - grid[lo], grid[hi] - m
    up = (above < below) | ((above == below) & (hi % 2 == 0))
    codes = torch.where(up, grid[hi], grid[lo]) * scaled.sign()
    return (codes * scale).flatten(-2)


def _mxfp8(x: Tensor) -> Tensor:
    import torch

    blocks = x.unflatten(-1, (-1, _BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    finite = amax.isfinite()
    mantissa, exponent = torch.frexp(torch.where(finite, amax, 0.0) / 448.0)
    exponent = (exponent - 1 + (mantissa != 0.5).int()).clamp(-126, 127)
    scale = torch.where(finite, torch.exp2(exponent.double()), float("nan"))
    scaled = torch.where(finite, blocks, 0.0) / scale
    codes = scaled.to(torch.float8_e4m3fn).double()
    return (codes * scale).flatten(-2)


def _weight(
    row_major_args: Mapping[str, Any],
    name: str,
    expert: int,
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    w = to_float64(row_major_args[name][expert], "float4_e2m1")
    scale = to_float64(row_major_args[f"{name}_scale"][expert], "float8_e8m0")
    return w * scale.repeat_interleave(_BLOCK, dim=-1)


def fused_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    to_float64: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each token through its top-k experts, weighted by its routing weights, summed.

    Rounded as aiter: both stages' activations to MXFP4 per 1 x 32 (MXFP8 when
    act_quant is 8), kept unrounded under gate_mode 'interleave' and for swiglu below
    _SWIGLU_BF16_ROWS rows; the gated product to hidden_states' dtype.
    """
    import torch
    import torch.nn.functional as F

    hidden, ids = row_major_args["hidden_states"], row_major_args["topk_ids"].long()
    if point.attrs["has_expert_mask"]:
        # A global id to its slot in w1, as aiter's sorting maps it; -1 off this rank.
        local = row_major_args["expert_mask"].long()
        slot = local.cumsum(0) - 1
        ids = torch.where(local[ids] == 1, slot[ids], -1)
    if point.attrs["has_num_local_tokens"]:
        ids[int(row_major_args["num_local_tokens"][0]) :] = -1  # rows not received
    # e2m1's 8 magnitudes, ascending: byte i holds code i in its low nibble.
    codes = torch.arange(16, dtype=torch.uint8, device=hidden.device)
    fp4_grid = to_float64(codes, "float4_e2m1")[0::2][:8]
    activation, stage1 = point.attrs["activation"], point.attrs["doweight_stage1"]
    limit = row_major_args.get("swiglu_limit")
    if point.attrs["gate_mode"] == "interleave" or (
        activation == "swiglu" and point.axes["n_tokens"] < _SWIGLU_BF16_ROWS
    ):  # aiter keeps bf16 activations
        quantize = lambda t: t
    elif point.attrs["act_quant"] == 8:
        quantize = _mxfp8
    else:
        quantize = functools.partial(_mxfp4, grid=fp4_grid)
    beta, linear_beta = row_major_args.get("beta"), row_major_args.get("linear_beta")
    x = quantize(to_float64(hidden, point.dtypes["hidden_states"]))
    weights = to_float64(row_major_args["topk_weights"], point.dtypes["topk_weights"])
    out = x.new_zeros((*ids.shape, x.shape[-1]))
    for expert in ids.unique().tolist():
        if expert < 0:  # another rank's route adds nothing
            continue
        routed = ids == expert
        tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
        projected = x[tokens] @ _weight(row_major_args, "w1", expert, to_float64).mT
        if stage1:
            projected = projected * weights[routed].unsqueeze(-1)
        gate, up = projected.chunk(2, dim=-1)  # w1 stores [gate | up] along its rows
        if limit is not None:  # aiter's torch_moe_stage1 order: clamp, then activate
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        if activation == "silu":
            gated = F.silu(gate) * up
        elif activation == "gelu":
            gated = F.gelu(gate) * up
        elif activation == "swiglu":  # gpt-oss's: alpha 1.702, up biased by 1
            gated = gate * torch.sigmoid(1.702 * gate) * (up + 1)
        else:
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            gated = gate * (linear_beta * torch.tanh(up / linear_beta))
        gated = gated.to(hidden.dtype).double()
        w2 = _weight(row_major_args, "w2", expert, to_float64)
        out[routed] = quantize(gated) @ w2.mT
    if not stage1:
        out = out * weights.unsqueeze(-1)
    return out.sum(dim=1)
