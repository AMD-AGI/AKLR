###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aklr.ops.gemm.reference import (
    unshuffle_e8m0_scales,
    unshuffle_tiles,
    unshuffle_weight,
)

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "deinterleave_gate_up_scales",
    "deinterleave_gate_up_weight",
    "fused_moe_gate_up_interleaved_reference",
    "fused_moe_reference",
    "fused_moe_shuffled_16x16_reference",
]


def _mxfp4(x: Tensor, grid: Tensor) -> Tensor:
    import torch

    blocks = x.unflatten(-1, (-1, 32))
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

    blocks = x.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    finite = amax.isfinite()
    mantissa, exponent = torch.frexp(torch.where(finite, amax, 0.0) / 448.0)
    exponent = (exponent - 1 + (mantissa != 0.5).int()).clamp(-126, 127)
    scale = torch.where(finite, torch.exp2(exponent.double()), float("nan"))
    scaled = torch.where(finite, blocks, 0.0) / scale
    codes = scaled.to(torch.float8_e4m3fn).double()
    return (codes * scale).flatten(-2)


def deinterleave_gate_up_weight(stored: Tensor, gate_up: bool) -> Tensor:
    """aiter ``shuffle_weight(is_guinterleave=True)`` undone on expert weights: 16-row
    by 64-byte tiles; with ``gate_up`` the gate and up halves alternate tile by tile."""
    rows, cols = stored.shape[-2:]
    if gate_up:
        split, order = (2, rows // 32, 16, cols // 64, 4, 16), (1, 0, 3, 4, 2, 5)
    else:
        split, order = (rows // 16, 16, cols // 64, 4, 16), (0, 2, 3, 1, 4)
    return unshuffle_tiles(stored, split, order)


def deinterleave_gate_up_scales(stored: Tensor, experts: int, rows: int, cols: int, gate_up: bool) -> Tensor:
    """aiter ``shuffle_scale(is_guinterleave=True)`` undone on the 2-D ``(experts *
    rows, padded cols)`` scales: the ``(experts, rows, cols)`` ones."""
    padded_cols = stored.shape[-1]
    x = stored.reshape(experts, rows, padded_cols)
    if gate_up:
        split = (2, rows // 32, 16, padded_cols // 8, 2, 4)
        order = (1, 3, 5, 2, 4, 0)
    else:
        split, order = (rows // 32, 2, 16, padded_cols // 8, 2, 4), (0, 3, 5, 2, 4, 1)
    return unshuffle_tiles(x, split, order)[..., :cols].contiguous()


def _weight(w: Tensor, scale: Tensor, expert: int) -> Tensor:
    import torch

    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    e2m1 = torch.tensor(magnitudes, dtype=torch.float64, device=w.device)
    e2m1 = torch.cat((e2m1, -e2m1))
    codes = w[expert].view(torch.uint8).long()
    decoded = e2m1[torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)]
    codes = scale[expert].view(torch.uint8)
    scale = torch.exp2(codes.double() - 127).masked_fill(codes == 255, math.nan)
    return decoded * scale.repeat_interleave(32, dim=-1)


def fused_moe_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=None,
    w1_scale=None,
    w2_scale=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    swiglu_limit=None,
    act_quant=None,
) -> Tensor:
    """Each token through its top-k experts, weighted by its routing weights, summed."""
    import torch
    import torch.nn.functional as F

    ids = topk_ids.long()
    if expert_mask is not None:
        # A global id to its slot in w1, as aiter's sorting maps it; -1 off this rank.
        local = expert_mask.long()
        slot = local.cumsum(0) - 1
        ids = torch.where(local[ids] == 1, slot[ids], -1)
    if num_local_tokens is not None:
        ids[int(num_local_tokens[0]) :] = -1  # rows not received
    # e2m1's 8 magnitudes, ascending
    fp4_grid = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float64,
        device=hidden_states.device,
    )
    quantize = {
        None: lambda t: t,
        "mxfp8": _mxfp8,
        "mxfp4": lambda t: _mxfp4(t, fp4_grid),
    }[act_quant]
    x = quantize(hidden_states.double())
    weights = topk_weights.double()
    out = x.new_zeros((*ids.shape, x.shape[-1]))
    for expert in ids.unique().tolist():
        if expert < 0:  # another rank's route adds nothing
            continue
        routed = ids == expert
        tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
        projected = x[tokens] @ _weight(w1, w1_scale, expert).mT
        gate, up = projected.chunk(2, dim=-1)  # w1 stores [gate | up] along its rows
        if swiglu_limit is not None:  # aiter's torch_moe_stage1: clamp, then activate
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        if activation == "silu":
            gated = F.silu(gate) * up
        elif activation == "gelu":
            gated = F.gelu(gate) * up
        else:
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            gated = gate * (linear_beta * torch.tanh(up / linear_beta))
        gated = gated.to(hidden_states.dtype).double()
        out[routed] = quantize(gated) @ _weight(w2, w2_scale, expert).mT
    return (out * weights.unsqueeze(-1)).sum(dim=1)


def fused_moe_shuffled_16x16_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=None,
    w1_scale=None,
    w2_scale=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    swiglu_limit=None,
    act_quant=None,
) -> Tensor:
    """The row-major maths on weights unshuffled (16 x 16) and scales unshuffled."""
    return fused_moe_reference(
        hidden_states,
        unshuffle_weight(w1, (16, 16)),
        unshuffle_weight(w2, (16, 16)),
        topk_weights,
        topk_ids,
        expert_mask,
        activation,
        unshuffle_e8m0_scales(w1_scale, w1.shape[-2], w1.shape[-1] * 2 // 32),
        unshuffle_e8m0_scales(w2_scale, w2.shape[-2], w2.shape[-1] * 2 // 32),
        beta,
        linear_beta,
        num_local_tokens,
        swiglu_limit,
        act_quant,
    )


def fused_moe_gate_up_interleaved_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=None,
    w1_scale=None,
    w2_scale=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    swiglu_limit=None,
    act_quant=None,
) -> Tensor:
    """The row-major maths on gate-up interleaved weights and 2-D scales undone."""
    experts, gate_up_rows, model_bytes = w1.shape
    model_dim, inter_bytes = w2.shape[-2:]
    return fused_moe_reference(
        hidden_states,
        deinterleave_gate_up_weight(w1, gate_up=True),
        deinterleave_gate_up_weight(w2, gate_up=False),
        topk_weights,
        topk_ids,
        expert_mask,
        activation,
        deinterleave_gate_up_scales(w1_scale, experts, gate_up_rows, model_bytes * 2 // 32, gate_up=True),
        deinterleave_gate_up_scales(w2_scale, experts, model_dim, inter_bytes * 2 // 32, gate_up=False),
        beta,
        linear_beta,
        num_local_tokens,
        swiglu_limit,
        act_quant,
    )
