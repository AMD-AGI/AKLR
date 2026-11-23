# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in fp32: every input decoded to fp32 and nothing
rounded or quantized in between (the kernels' bf16, fp8 and fp4 steps are theirs) but what
the checkpoints' FP8 scheme quantizes: an fp8 linear's input (_fp8_linear). The MXFP4
experts' activations stay unquantized: the scale rounding of their blocks is the kernel's
(aiter rounds a block's maximum to nearest even, others up), its error left to the bound.
"""

import torch

from aklr.core.contract.reference import OpReference
from aklr.ops.fused_moe.reference import mxfp4_weight

__all__ = ["MinimaxM2Fp8MoeReference", "MinimaxM2Mxfp4MoeReference"]

_FP8_BLOCK = 128


def _routed(x, router, bias, topk):
    """Each token's topk experts by sigmoid score plus correction_bias, and their weights:
    the unbiased scores renormalized to sum 1."""
    scores = (x @ router.T).sigmoid()
    ids = (scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(-1, ids)
    return ids, weights / weights.sum(dim=-1, keepdim=True)


def _held_experts(x, ids, weights, mlp, expert_offset, n_held):
    """The routes to the ``n_held`` experts this rank holds, from ``expert_offset`` on,
    ``mlp(rows, e)`` held expert e (w1's index) on its rows, weighed and summed; the routes to
    the others add nothing."""
    local = ids - expert_offset
    out = torch.zeros_like(x)
    for e in local.unique().tolist():
        if not 0 <= e < n_held:
            continue
        tokens, slots = (local == e).nonzero(as_tuple=True)
        out.index_add_(0, tokens, weights[tokens, slots, None] * mlp(x[tokens], e))
    return out


def _fp8_linear(x, weight, scale):
    """x weight^T of an fp8 weight [n, k] with its scale per 128 x 128 block, x taken as the
    checkpoint's FP8 scheme quantizes it (sglang's fp8 MoE on aiter, per_128x128): in bf16,
    then each 1 x 128 group in fp8 at a scale of its amax / 448."""
    n, k = weight.shape
    block_scale = scale.repeat_interleave(_FP8_BLOCK, 0)[:n]
    weight = weight * block_scale.repeat_interleave(_FP8_BLOCK, 1)[:, :k]
    fp8 = torch.float8_e4m3fn
    groups = x.bfloat16().float().unflatten(-1, (-1, _FP8_BLOCK))
    group_scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    group_scale = group_scale.clamp(min=torch.finfo(torch.float32).tiny)
    x = ((groups / group_scale).to(fp8).float() * group_scale).flatten(-2)
    return x @ weight.T


def _silu_and_mul(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


class MinimaxM2Fp8MoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        router_weight,
        correction_bias,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk,
        expert_offset,
        **_,
    ):
        """Each token's routed experts this rank holds, both linears of each an fp8 one
        (_fp8_linear), weighed by the renormalized scores and summed."""
        x = hidden_states.float()
        ids, weights = _routed(x, router_weight.float(), correction_bias.float(), topk)

        def mlp(rows, e):
            gate_up = _fp8_linear(rows, w1[e].float(), w1_scale[e].float())
            return _fp8_linear(_silu_and_mul(gate_up), w2[e].float(), w2_scale[e].float())

        return {
            "out": _held_experts(x, ids, weights, mlp, expert_offset, w1.shape[0]).to(hidden_states.dtype)
        }


class MinimaxM2Mxfp4MoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        router_weight,
        correction_bias,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk,
        expert_offset,
        **_,
    ):
        """Each token's routed experts this rank holds, their MXFP4 weights dequantized per 32
        values, weighed by the renormalized scores and summed."""
        x = hidden_states.float()
        ids, weights = _routed(x, router_weight.float(), correction_bias.float(), topk)

        def mlp(rows, e):
            gate_up = rows @ mxfp4_weight(w1[e], w1_scale[e]).T
            return _silu_and_mul(gate_up) @ mxfp4_weight(w2[e], w2_scale[e]).T

        return {
            "out": _held_experts(x, ids, weights, mlp, expert_offset, w1.shape[0]).to(hidden_states.dtype)
        }
