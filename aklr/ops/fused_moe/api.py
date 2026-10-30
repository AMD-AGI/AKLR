# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import (
    aklr_route,
    attr_from_env,
    must_default,
    none_if_falsy,
    unused,
)
from aklr.ops.fused_moe.contract import FUSED_MOE

if TYPE_CHECKING:
    import torch

__all__ = ["fused_moe"]


@api(FUSED_MOE)
def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weights,
    topk_ids,
    expert_mask: torch.Tensor | None = None,  # EP
    activation=None,
    quant_type=None,
    doweight_stage1=False,
    # following for quant
    w1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: torch.Tensor | None = None,  # [expert(local_expert:EP), 1, inter_dim]
    bias1=None,
    bias2=None,
    # clamped SwiGLU, swiglu and SiTUv2 activation maths, not tuning knobs
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = None,
    out_dtype=None,
    num_local_tokens: torch.Tensor | None = None,  # EP: [1] int32 rows received
): ...


@aklr_route("aiter.fused_moe.fused_moe")
def _(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask=None,
    activation=None,
    quant_type=None,
    doweight_stage1=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_size_M=None,
    num_local_tokens=None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode=None,
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
    shared_expert_id=-1,
    stage2_scatter=None,
):
    unused(moe_sorting_dispatch_policy, splitk)  # scheduling; the registry decides
    # a layout AKLR does not describe
    must_default(block_size_M, hidden_pad, intermediate_pad)
    must_default(shared_w1, shared_w2, shared_w1_scale, shared_w2_scale, shared_expert_id)
    must_default(stage2_scatter)  # writes stage 2 elsewhere, on the gfx1250 path
    # not in the contract: activation scales, biases
    must_default(a1_scale, a2_scale, bias1, bias2)
    # aiter clamps only at a truthy limit; vLLM passes 0.0 for none
    none_if_falsy(swiglu_limit)
    return fused_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weight,
        topk_ids=topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        activation=activation,
        # situv2's activation quant is chosen by aiter's environment, not by the call
        act_quant=attr_from_env(activation, when="situv2", env={"AITER_SITUV2_A8W4": 8}),
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        gate_mode=gate_mode,
        expert_mask=expert_mask,
        beta=beta,
        linear_beta=linear_beta,
        swiglu_limit=swiglu_limit,
        num_local_tokens=num_local_tokens,
        out_dtype=dtype,
    )
