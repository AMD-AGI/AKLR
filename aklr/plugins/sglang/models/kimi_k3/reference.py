# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What kimi_k3_latent_moe's contract says, in torch, in fp32: every input decoded to fp32 and
nothing rounded or quantized in between but what the contract rounds."""

import torch

from aklr.core.contract.reference import OpReference
from aklr.core.contract.semantics import rounded_to_bfloat16, to_float32
from aklr.plugins.aiter.provider.fused_moe.reference import mxfp4_weight, mxfp8

__all__ = [
    "KimiK3LatentMoeReference",
    "KimiK3MlaPrefillGateReference",
    "KimiK3MlaVcOutputGateReference",
    "KimiK3MoeFrontGemmReference",
    "KimiK3RoutedMoeMxfp8Reference",
]

_BLOCK = 32


def _situ(gate, up):
    return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))


class KimiK3LatentMoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        front_weight,
        correction_bias,
        w1,
        w1_scale,
        w2,
        w2_scale,
        shared_w2,
        topk,
        front_fp32=False,
        **_,
    ):
        """latent and shared. Rounding points: the front projection rounds to
        bf16, or with front_fp32 only its latent input z does; each routed linear's input goes to
        MXFP8 per 1 x 32 and the routed SiTU product to bf16, as aiter's a8w4 experts take them."""
        front = hidden_states.float() @ front_weight.float().T
        if not front_fp32:
            front = rounded_to_bfloat16(front)
        si, routed = shared_w2.shape[1], correction_bias.shape[0]
        gate, up, logits, z = front.split((si, si, routed, front.shape[-1] - 2 * si - routed), dim=-1)
        z = rounded_to_bfloat16(z)

        shared = rounded_to_bfloat16(_situ(gate, up)) @ shared_w2.float().T

        scores = logits.sigmoid()
        ids = (scores + correction_bias.float()).topk(topk, dim=-1, sorted=True).indices
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        x = mxfp8(z)
        latent = torch.zeros_like(z)
        for e in ids.unique().tolist():
            tokens, slots = (ids == e).nonzero(as_tuple=True)
            g, u = (x[tokens] @ mxfp4_weight(w1[e], w1_scale[e]).T).chunk(2, dim=-1)
            h = rounded_to_bfloat16(_situ(g, u))
            latent.index_add_(
                0, tokens, weights[tokens, slots, None] * (mxfp8(h) @ mxfp4_weight(w2[e], w2_scale[e]).T)
            )
        return {"latent": latent.to(hidden_states.dtype), "shared": shared.to(hidden_states.dtype)}


class KimiK3RoutedMoeMxfp8Reference(OpReference):
    def reference(
        self, *, a, a_scale, topk_ids, topk_weights, w1, w1_scale, w2, w2_scale, beta, linear_beta, **_
    ):
        """out, the routed experts' weighed sum. Rounding points: the input is the MXFP8 it is given; each
        expert's SiTUv2 product to bf16, then to MXFP8 per 1 x 32 for the down projection, as sglang's
        k3_a8w4 experts take it."""
        beta, linear_beta = float(beta), float(linear_beta)
        x = a.float() * to_float32(a_scale, "float8_e8m0").repeat_interleave(_BLOCK, dim=-1)
        ids, weights = topk_ids.long(), topk_weights.float()
        out = torch.zeros_like(x)
        for e in ids.unique().tolist():
            tokens, slots = (ids == e).nonzero(as_tuple=True)
            g, u = (x[tokens] @ mxfp4_weight(w1[e], w1_scale[e]).T).chunk(2, dim=-1)
            situ = (
                beta * torch.tanh(g / beta) * torch.sigmoid(g) * (linear_beta * torch.tanh(u / linear_beta))
            )
            h = rounded_to_bfloat16(situ)
            out.index_add_(
                0, tokens, weights[tokens, slots, None] * (mxfp8(h) @ mxfp4_weight(w2[e], w2_scale[e]).T)
            )
        return {"out": out.to(torch.bfloat16)}


class KimiK3MoeFrontGemmReference(OpReference):
    def reference(self, *, x, weight, **_):
        """out = x @ weight.T, the bf16 values decoded exactly and each dot product summed in float32."""
        return {"out": x.float() @ weight.float().T}


class KimiK3MlaVcOutputGateReference(OpReference):
    def reference(self, *, latent, vc_weight, gate, **_):
        """Rounding points: the value projection and the sigmoid each round to bf16 before
        their product."""
        values = rounded_to_bfloat16(torch.einsum("thk,hvk->thv", latent.float(), vc_weight.float())).flatten(
            1
        )
        return {"output": (values * rounded_to_bfloat16(torch.sigmoid(gate.float()))).to(latent.dtype)}


class KimiK3MlaPrefillGateReference(OpReference):
    def reference(self, *, q, k, v, gate, qo_indptr, sm_scale, **_):
        """Rounding points: the attention output and the sigmoid each round to bf16 before
        their product."""
        dtype, q, k, v = q.dtype, q.float(), k.float(), v.float()
        attention = torch.empty_like(v)
        bounds = qo_indptr.long().tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            scores = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]) * float(sm_scale)
            causal = torch.ones(end - start, end - start, dtype=torch.bool, device=q.device).tril()
            weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            attention[start:end] = torch.einsum("hqk,khd->qhd", weights, v[start:end])
        output = rounded_to_bfloat16(attention).flatten(1) * rounded_to_bfloat16(torch.sigmoid(gate.float()))
        return {"output": output.to(dtype)}
