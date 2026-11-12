# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MoE layers on ROCm, where ``vllm::rocm_aiter_fused_moe`` reaches aiter's fused MoE."""

from __future__ import annotations

from aklr.plugins import aiter as aklr_aiter
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm._aiter_ops._rocm_aiter_fused_moe_impl",
    "aiter.fused_moe.fused_moe",
    versions=("0.33.0rc1.dev910",),
)
def _fused_moe(
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
    gate_mode="separated",
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
    shared_expert_id=-1,
    stage2_scatter=None,
    output=None,
    quant_type_a=None,
    quant_dtype_a=None,
    quant_dtype_a2=None,
    clamp_shared=True,
    herd_topk=0,
    herd_routed_scaling_factor=1.0,
):
    return aklr_aiter.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_expert_id=shared_expert_id,
        stage2_scatter=stage2_scatter,
        output=output,
        quant_type_a=quant_type_a,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
        herd_topk=herd_topk,
    )
