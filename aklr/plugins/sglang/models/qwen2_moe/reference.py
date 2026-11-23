# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in the dtype the inputs decode to: nothing rounded
or quantized in between (the kernels' bf16 steps are theirs) but the router's logits,
rounded once to bf16 as the block's gate returns them, which pick the experts, and what
the checkpoint's FP8 scheme quantizes: each expert linear's input (_fp8_input). The MXFP4
experts' activations stay unquantized: kernels round a block's scale from its maximum in
different ways (to nearest or up) and sglang's own small-M kernel keeps them bf16, so that
error is left to the bound."""

import torch

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import rounded_to_bfloat16
from aklr.ops.fused_moe.reference import mxfp4_weight

__all__ = [
    "Qwen2MoeFp8ExpertsReference",
    "Qwen2MoeFp8RouterExpertsReference",
    "Qwen2MoeMxfp4MoeReference",
    "Qwen2MoeRouterReference",
]

_BLOCK = 128  # the FP8 scheme's block: 128 x 128 of a weight, 1 x 128 of an input


def _logits(hidden_states, router_weight):
    """hidden_states router_weight^T, rounded to bf16, in float32."""
    return rounded_to_bfloat16(hidden_states.float() @ router_weight.float().T)


def _routed(logits, topk):
    """Each row's ``topk`` experts by descending logit, a tie going to the lower id, and
    their weights: the softmax of their logits, which is the softmax over every expert
    renormalized over them. Returns (weights, ids)."""
    ids = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :topk]
    return logits.gather(-1, ids).softmax(dim=-1), ids


def _fp8_input(x):
    """``x`` as an fp8 linear of the checkpoint's scheme takes it (sglang's
    per-token-group quant): in bf16, then each 1 x 128 group in fp8 at a float32 scale of
    its amax / 448, dequantized."""
    fp8 = torch.float8_e4m3fn
    groups = rounded_to_bfloat16(x).float().unflatten(-1, (-1, _BLOCK))
    scale = groups.abs().amax(-1, keepdim=True) / torch.finfo(fp8).max
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    return ((groups / scale).to(fp8).to(x.dtype) * scale.to(x.dtype)).flatten(-2)


def _dequantized(weight, scale):
    """An fp8 weight [rows, cols] (decoded) times its scale per 128 x 128 block."""
    rows, cols = weight.shape
    blocks = scale.repeat_interleave(_BLOCK, 0)[:rows].repeat_interleave(_BLOCK, 1)
    return weight * blocks[:, :cols]


def _experts(hidden_states, shared_gate_weight, w1, w1_scale, w2, w2_scale, weights, ids, shared_slot):
    """This rank's share of the block's output: each token's routed experts ``ids``
    weighed by ``weights`` and the shared expert (``shared_slot``, the slot after the
    routed ones) weighed by sigmoid(hidden_states shared_gate_weight^T), each expert
    down(silu(gate(x)) * up(x)) with both linears' inputs quantized (_fp8_input)."""
    x = hidden_states.float()
    shared_weight = (x @ shared_gate_weight.float()[0]).sigmoid()
    ids = torch.cat([ids, torch.full_like(ids[:, :1], shared_slot)], dim=-1)
    weights = torch.cat([weights, shared_weight[:, None]], dim=-1)
    x_fp8 = _fp8_input(x)
    out = torch.zeros_like(x)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        gate_up = _dequantized(w1[e].float(), w1_scale[e].float())
        down = _dequantized(w2[e].float(), w2_scale[e].float())
        gate, up = (x_fp8[tokens] @ gate_up.T).chunk(2, dim=-1)
        expert = _fp8_input(torch.nn.functional.silu(gate) * up) @ down.T
        out.index_add_(0, tokens, weights[tokens, slots, None] * expert)
    return out.to(hidden_states.dtype)


class Qwen2MoeRouterReference(OpReference):
    def reference(self, *, hidden_states, router_weight, topk_ids, **_):
        """topk_weights and topk_ids of the bf16 logits (_logits, _routed); the logits are
        topk_id_scores, for check."""
        logits = _logits(hidden_states, router_weight)
        weights, ids = _routed(logits, topk_ids.shape[-1])
        return {
            "topk_weights": weights,
            "topk_ids": ids.to(topk_ids.dtype),
            "topk_id_scores": logits.to(torch.bfloat16),
        }

    def check(self, outputs, expected, criteria):
        """``OpReference.check``, but an id that differs from the reference's stands for it where
        the two experts' logits are equal or one bf16 value apart, in a row without a repeated id:
        a kernel sums in another order, so a logit at a rounding midpoint can round to the next
        value."""
        ids, reference_ids = outputs["topk_ids"], expected["topk_ids"]
        tied = _tied(ids, reference_ids, expected["topk_id_scores"])
        return super().check(
            {**outputs, "topk_ids": torch.where(tied, reference_ids, ids)}, expected, criteria
        )


def _tied(ids, reference_ids, logits):
    """Where an id is in range, in a row without a repeated id, and its bf16 logit equals the
    reference id's or is the next bf16 value above or below it."""
    n = logits.shape[-1]
    a, y = ids.long(), reference_ids.long()  # gather's index dtype
    in_range = (a >= 0) & (a < n)
    a = a.clamp(0, n - 1)
    # sign and magnitude to one integer line, +0 and -0 one value: adjacent values differ by 1
    bits = logits.view(torch.int16).long()
    order = torch.where(bits < 0, -(bits & 0x7FFF), bits & 0x7FFF)
    within = (order.gather(-1, a) - order.gather(-1, y)).abs() <= 1
    ordered = a.sort(-1).values
    distinct = (ordered[..., 1:] != ordered[..., :-1]).all(-1, keepdim=True)
    return in_range & distinct & within


class Qwen2MoeFp8ExpertsReference(OpReference):
    def reference(
        self, *, hidden_states, topk_weights, topk_ids, shared_gate_weight, w1, w1_scale, w2, w2_scale, **_
    ):
        """The routed experts topk_ids at topk_weights and the gated shared one, in w1's last
        slot (_experts)."""
        weights, ids = topk_weights.float(), topk_ids.long()
        return {
            "out": _experts(
                hidden_states, shared_gate_weight, w1, w1_scale, w2, w2_scale, weights, ids, w1.shape[0] - 1
            )
        }


class Qwen2MoeFp8RouterExpertsReference(OpReference):
    def reference(
        self, *, hidden_states, router_weight, shared_gate_weight, w1, w1_scale, w2, w2_scale, topk, **_
    ):
        """The router's experts (_logits, _routed) and the gated shared one, in the slot after
        the routed ones (_experts)."""
        weights, ids = _routed(_logits(hidden_states, router_weight), topk)
        return {
            "out": _experts(
                hidden_states,
                shared_gate_weight,
                w1,
                w1_scale,
                w2,
                w2_scale,
                weights,
                ids,
                router_weight.shape[0],
            )
        }


def _silu_mlp(x, gate_up, down):
    """down(silu(gate) * up), gate and up the halves of x gate_up^T."""
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


class Qwen2MoeMxfp4MoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        router_weight,
        w1,
        w1_scale,
        w2,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_gate_weight,
        topk,
        **_,
    ):
        """The router's experts (_logits, _routed) weighed and summed, each on its MXFP4
        weights dequantized per 32 values and the unquantized input, plus the BF16 shared
        expert weighed by sigmoid(hidden_states shared_gate_weight^T). Rounding point: the
        router's logits alone (_logits)."""
        x = hidden_states.float()
        weights, ids = _routed(_logits(hidden_states, router_weight), topk)
        out = torch.zeros_like(x)
        for e in ids.unique().tolist():
            tokens, slots = (ids == e).nonzero(as_tuple=True)
            routed = _silu_mlp(x[tokens], mxfp4_weight(w1[e], w1_scale[e]), mxfp4_weight(w2[e], w2_scale[e]))
            out.index_add_(0, tokens, weights[tokens, slots, None] * routed)
        shared_weight = (x @ shared_gate_weight.float()[0]).sigmoid()
        out = out + shared_weight[:, None] * _silu_mlp(x, shared_w1.float(), shared_w2.float())
        return {"out": out.to(hidden_states.dtype)}
