# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The route for the ``fused_moe`` call in sglang's aiter MoE runner (``AiterRunnerCore.run``)."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.fused_moe.api import fused_moe
from aklr.plugins.sglang.hook import sglang_call_route

__all__: list[str] = []


# The parameters of every aiter fused_moe that sglang images ship: the oldest release's, in its
# order, then the ones later releases added. sglang passes output and quant_dtype_a only where
# the installed aiter has them.
@sglang_call_route(
    "sglang.srt.layers.moe.moe_runner.aiter.AiterRunnerCore.run",
    "aiter.fused_moe.fused_moe",
    versions=("0.5.22",),
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
    # Fall back on arguments the op does not take. moe_sorting_dispatch_policy and splitk only
    # choose how aiter schedules the work; clamp_shared and herd_routed_scaling_factor only apply
    # with shared experts or herd_topk, which fall back.
    if (
        block_size_M is not None
        or hidden_pad
        or intermediate_pad
        or a1_scale is not None
        or a2_scale is not None
        or bias1 is not None
        or bias2 is not None
        or shared_w1 is not None
        or shared_w2 is not None
        or shared_expert_id != -1
        or stage2_scatter is not None
        or quant_type_a is not None
        or quant_dtype_a is not None
        or quant_dtype_a2 is not None
        or herd_topk
    ):
        return FallbackReason.OUT_OF_CONTRACT
    return fused_moe(
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
        swiglu_limit=swiglu_limit or None,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        out_dtype=dtype,
        num_local_tokens=num_local_tokens,
        out=output,
    )
