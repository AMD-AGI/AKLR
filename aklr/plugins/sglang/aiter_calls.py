# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes on the aiter functions sglang calls, for every model that reaches them.

sglang looks ``fused_moe`` up in aiter at each call, so hooking aiter's name reaches every call. The
parameters are those of the aiter in the sglang release's image; the arguments go by name to
:mod:`aklr.plugins.aiter`."""

from __future__ import annotations

from aklr.plugins import aiter as aklr_aiter
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("aiter.fused_moe.fused_moe", versions=("0.5.22",))
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
        # the parameters sglang 0.5.22's aiter has: no caller's buffer, no activation quant types
        output=None,
        quant_type_a=None,
        quant_dtype_a=None,
        quant_dtype_a2=None,
    )
