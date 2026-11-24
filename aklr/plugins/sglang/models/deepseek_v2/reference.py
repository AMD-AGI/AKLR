# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in fp32: every input decoded to fp32 and nothing rounded or
quantized in between but what a contract rounds: the router's logits, in the dtype sglang's
MoEGate returns them in, and in deepseek_v2_fp8_moe each fp8 linear's input, as the checkpoint's
FP8 scheme quantizes it (_fp8_linear)."""

import torch

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import rounded_to_bfloat16
from aklr.ops.fused_moe.reference import mxfp4_weight

__all__ = ["DeepseekV2Fp8MoeReference", "DeepseekV2Mxfp4MoeReference"]

_FP8_BLOCK = 128


def _mlp(x, gate_up, down):
    """down(silu(gate) * up), gate and up the halves of x gate_up^T."""
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


def _routed(logits, bias, topk, renormalize, routed_scaling_factor):
    """Each token's topk experts by sigmoid score plus ``bias``, and their weights."""
    scores = logits.sigmoid()
    ids = (scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * routed_scaling_factor


def _experts(x, ids, weights, mlp):
    """Each token's experts, ``mlp(rows, e)`` expert e on its rows, weighed and summed."""
    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        out.index_add_(0, tokens, weights[tokens, slots, None] * mlp(x[tokens], e))
    return out


def _with_shared_slot(ids, weights, n_routed):
    """The routing with the shared expert appended: the last slot, at weight 1."""
    shared = torch.full_like(ids[:, :1], n_routed)
    return (
        torch.cat([ids, shared], dim=-1),
        torch.cat([weights, torch.ones_like(weights[:, :1])], dim=-1),
    )


class DeepseekV2Mxfp4MoeReference(OpReference):
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
        renormalize,
        routed_scaling_factor,
        **_,
    ):
        """On hidden_states, each token's routed experts and the shared one (the last slot) at
        weight 1, the MXFP4 weights dequantized per 32 values. Rounding point: the router's
        logits round to hidden_states' dtype, bf16."""
        x = hidden_states.float()
        logits = rounded_to_bfloat16(x @ router_weight.float().T)
        ids, weights = _routed(logits, correction_bias.float(), topk, renormalize, routed_scaling_factor)
        ids, weights = _with_shared_slot(ids, weights, router_weight.shape[0])

        def mlp(rows, e):
            return _mlp(rows, mxfp4_weight(w1[e], w1_scale[e]), mxfp4_weight(w2[e], w2_scale[e]))

        return {"out": _experts(x, ids, weights, mlp).to(hidden_states.dtype)}


def _first_largest(values, k):
    """The indices of each row's k largest values, the lower index first among equals."""
    return torch.sort(values, dim=-1, descending=True, stable=True).indices[..., :k]


def _grouped_routed(logits, bias, topk, num_expert_group, topk_group, renormalize, routed_scaling_factor):
    """Each token's topk experts by sigmoid score plus ``bias`` within its topk_group best
    groups, a group scored by its two best biased scores summed, and their weights."""
    scores = logits.sigmoid()
    biased = scores + bias
    groups = biased.unflatten(-1, (num_expert_group, -1))
    group_scores = groups.topk(2, dim=-1).values.sum(dim=-1)
    kept = torch.zeros_like(group_scores, dtype=torch.bool)
    kept.scatter_(-1, _first_largest(group_scores, topk_group), True)
    allowed = kept.repeat_interleave(groups.shape[-1], dim=-1)
    ids = _first_largest(biased.masked_fill(~allowed, float("-inf")), topk)
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * routed_scaling_factor


def _fp8_linear(x, weight, scale):
    """x weight^T of an fp8 weight [n, k] with its scale per 128 x 128 block, x taken as the
    checkpoint's FP8 scheme quantizes it (sglang's fp8 MoE on aiter, per_128x128): in bf16,
    then each 1 x 128 group in fp8 at a scale of its amax / 448."""
    n, k = weight.shape
    block_scale = scale.repeat_interleave(_FP8_BLOCK, 0)[:n]
    weight = weight * block_scale.repeat_interleave(_FP8_BLOCK, 1)[:, :k]
    fp8 = torch.float8_e4m3fn
    groups = x.bfloat16().to(x.dtype).unflatten(-1, (-1, _FP8_BLOCK))
    group_scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    group_scale = group_scale.clamp(min=torch.finfo(torch.float32).tiny)
    x = ((groups / group_scale).to(fp8).to(x.dtype) * group_scale).flatten(-2)
    return x @ weight.T


class DeepseekV2Fp8MoeReference(OpReference):
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
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
        **_,
    ):
        """On hidden_states, each token's routed experts (_grouped_routed) and the shared one
        (the last slot) at weight 1, both linears of each expert an fp8 one. Rounding points:
        the router's logits round to hidden_states' dtype, bf16; each linear's input rounds to
        bf16, then to fp8 per 1 x 128 group (_fp8_linear)."""
        x = hidden_states.float()
        logits = rounded_to_bfloat16(x @ router_weight.float().T)
        ids, weights = _grouped_routed(
            logits,
            correction_bias.float(),
            topk,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor,
        )
        ids, weights = _with_shared_slot(ids, weights, router_weight.shape[0])

        def mlp(rows, e):
            gate, up = _fp8_linear(rows, w1[e].float(), w1_scale[e].float()).chunk(2, dim=-1)
            return _fp8_linear(torch.nn.functional.silu(gate) * up, w2[e].float(), w2_scale[e].float())

        return {"out": _experts(x, ids, weights, mlp).to(hidden_states.dtype)}
