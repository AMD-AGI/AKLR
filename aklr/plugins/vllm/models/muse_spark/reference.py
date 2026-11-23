# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch: the RMSNorm unrounded, rounded to hidden_states' dtype
as vLLM's fused_rmsnorm_expand writes it, then each route through fused_moe's reference
(its MXFP4 activations and bf16 intermediate) with a weight of one."""

import torch

from aklr.core.contract.reference import OpReference
from aklr.ops.fused_moe.reference import FusedMoeReference

__all__ = ["MuseSparkMoeReference"]


class MuseSparkMoeReference(OpReference):
    def reference(
        self,
        *,
        hidden_states,
        norm_weight,
        topk_ids,
        expert_mask,
        w1,
        w1_scale,
        w2,
        w2_scale,
        eps,
        gain_center,
        **_,
    ):
        """Each row normalized, then each of its routes through its expert, unweighted."""
        x = hidden_states.float()
        gain = norm_weight.float() + gain_center
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        normalized = (x * rstd * gain).to(hidden_states.dtype)
        n_tokens, topk = topk_ids.shape
        ids = topk_ids.reshape(n_tokens * topk, 1)
        out = FusedMoeReference().reference(
            hidden_states=normalized.repeat_interleave(topk, dim=0),
            topk_ids=ids,
            topk_weights=torch.ones(ids.shape, dtype=torch.float32, device=ids.device),
            expert_mask=expert_mask,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            activation="silu",
            doweight_stage1=False,
            gate_mode="separated",
            act_quant=4,
        )["out"]
        return {"out": out.view(n_tokens, topk, -1)}
