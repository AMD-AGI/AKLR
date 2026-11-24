# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import torch.nn.functional as F

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import to_float32

__all__ = ["SWIGLU_BF16_ROWS", "FusedMoeReference", "mxfp4_weight", "mxfp8"]

_BLOCK = 32
SWIGLU_BF16_ROWS = 256  # aiter's GPTOSS_SWIGLU_MXFP4_BF16_BOUND default


def _mxfp4(x, grid):
    blocks = x.unflatten(-1, (-1, _BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    finite = amax.isfinite()
    mantissa, exponent = torch.frexp(torch.where(finite, amax, 0.0).clamp_min(1e-10))
    exponent = (exponent - 3 + (mantissa > 0.75).int()).clamp(-127, 127)
    scale = torch.where(finite, torch.exp2(exponent.to(x.dtype)), float("nan"))
    scaled = torch.where(finite, blocks, 0.0) / scale
    m = scaled.abs()
    lo = torch.bucketize(m, grid, right=True) - 1  # the largest code <= m
    hi = (lo + 1).clamp_max(len(grid) - 1)
    below, above = m - grid[lo], grid[hi] - m
    up = (above < below) | ((above == below) & (hi % 2 == 0))
    codes = torch.where(up, grid[hi], grid[lo]) * scaled.sign()
    return (codes * scale).flatten(-2)


def mxfp8(x):
    """``x`` rounded as aiter rounds activations to MXFP8: fp8 e4m3 at an e8m0 scale per 1 x 32."""
    blocks = x.unflatten(-1, (-1, _BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    finite = amax.isfinite()
    mantissa, exponent = torch.frexp(torch.where(finite, amax, 0.0) / 448.0)
    exponent = (exponent - 1 + (mantissa != 0.5).int()).clamp(-126, 127)
    scale = torch.where(finite, torch.exp2(exponent.to(x.dtype)), float("nan"))
    scaled = torch.where(finite, blocks, 0.0) / scale
    codes = scaled.to(torch.float8_e4m3fn).to(x.dtype)
    return (codes * scale).flatten(-2)


def mxfp4_weight(weight, scale):
    """An MXFP4 weight in float32, dequantized by its e8m0 scale per 32 values."""
    return to_float32(weight, "float4_e2m1") * to_float32(scale, "float8_e8m0").repeat_interleave(
        _BLOCK, dim=-1
    )


class FusedMoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        activation,
        doweight_stage1,
        gate_mode,
        act_quant,
        expert_mask=None,
        num_local_tokens=None,
        swiglu_limit=None,
        beta=None,
        linear_beta=None,
        **_,
    ):
        """Each token through its top-k experts, weighted by its routing weights, summed.

        Rounded as aiter: both stages' activations to MXFP4 per 1 x 32 (MXFP8 when
        act_quant is 8), kept unrounded under gate_mode 'interleave' and for swiglu below
        SWIGLU_BF16_ROWS rows; the gated product to hidden_states' dtype.
        """
        ids = topk_ids.long().clone()  # .long() may return topk_ids itself, and ids is written
        if expert_mask is not None:
            # A global id to its slot in w1, as aiter's sorting maps it; -1 off this rank.
            local = expert_mask.long()
            slot = local.cumsum(0) - 1
            ids = torch.where(local[ids] == 1, slot[ids], -1)
        if num_local_tokens is not None:
            ids[int(num_local_tokens[0]) :] = -1  # rows not received
        # e2m1's 8 magnitudes, ascending: byte i holds code i in its low nibble.
        codes = torch.arange(16, dtype=torch.uint8, device=hidden_states.device)
        fp4_grid = to_float32(codes, "float4_e2m1")[0::2][:8].contiguous()
        if gate_mode == "interleave" or (
            activation == "swiglu" and hidden_states.shape[0] < SWIGLU_BF16_ROWS
        ):  # aiter keeps bf16 activations
            quantize = lambda t: t
        elif act_quant == 8:
            quantize = mxfp8
        else:
            quantize = functools.partial(_mxfp4, grid=fp4_grid)
        x = quantize(hidden_states.float())
        weights = topk_weights.float()
        out = x.new_zeros((*ids.shape, x.shape[-1]))
        for expert in ids.unique().tolist():
            if expert < 0:  # another rank's route adds nothing
                continue
            routed = ids == expert
            tokens = routed.nonzero(as_tuple=True)[0]  # topk_ids rows are distinct
            projected = x[tokens] @ mxfp4_weight(w1[expert], w1_scale[expert]).mT
            if doweight_stage1:
                projected = projected * weights[routed].unsqueeze(-1)
            gate, up = projected.chunk(2, dim=-1)  # w1 stores [gate | up] along its rows
            if swiglu_limit is not None:  # aiter's torch_moe_stage1 order: clamp, then activate
                gate, up = gate.clamp(max=swiglu_limit), up.clamp(-swiglu_limit, swiglu_limit)
            if activation == "silu":
                gated = F.silu(gate) * up
            elif activation == "gelu":
                gated = F.gelu(gate) * up
            elif activation == "swiglu":  # gpt-oss's: alpha 1.702, up biased by 1
                gated = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            else:
                gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
                gated = gate * (linear_beta * torch.tanh(up / linear_beta))
            gated = gated.to(hidden_states.dtype).to(x.dtype)
            out[routed] = quantize(gated) @ mxfp4_weight(w2[expert], w2_scale[expert]).mT
        if not doweight_stage1:
            out = out * weights.unsqueeze(-1)
        return {"out": out.sum(dim=1).to(hidden_states.dtype)}
