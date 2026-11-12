# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused MoE, ``aiter.fused_moe.fused_moe``, served by AKLR's ``fused_moe`` op."""

from __future__ import annotations

import functools
import inspect

from aklr.core.contract.spec import attr_value
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.fused_moe.api import fused_moe as aklr_fused_moe
from aklr.plugins.aiter.situv2_a4w4_moe.api import situv2_a4w4_moe

__all__ = ["fused_moe"]


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    expert_mask,
    activation,
    quant_type,
    doweight_stage1,
    w1_scale,
    w2_scale,
    a1_scale,
    a2_scale,
    block_size_M,
    num_local_tokens,
    dtype,
    hidden_pad,
    intermediate_pad,
    bias1,
    bias2,
    swiglu_limit,
    beta,
    linear_beta,
    gate_mode,
    shared_w1,
    shared_w2,
    shared_expert_id,
    stage2_scatter,
    output,
    quant_type_a,
    quant_dtype_a,
    quant_dtype_a2,
    herd_topk=0,
):
    """AKLR's routing-weighted sum of the chosen experts, or why aiter computes it. aiter's
    sorting policy and split-K pick its schedule, not the result, and are not taken."""
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
    act_quant = 4
    # aiter picks SiTUv2's activation dtype at each call (AITER_SITUV2_A8W4, AITER_SITUV2_A4W4): fp8 is
    # fused_moe's, on sglang's gate-up interleaved weights; fp4 is vLLM's, whose a16w4 shuffle keeps gate and up
    # apart and is marked so on the weight by aiter from 0.1.28 on
    if attr_value(activation) == "situv2":
        from aiter import dtypes

        q_dtype = _activation_dtype(
            hidden_states, w1, activation=activation, quant_type=quant_type, gate_mode=gate_mode
        )
        if q_dtype == dtypes.fp8:
            act_quant = 8
        elif (
            q_dtype == dtypes.fp4x2
            and getattr(w1, "gate_up_interleave", None) is False
            and attr_value(gate_mode) == "separated"
            and expert_mask is None
            and num_local_tokens is None
            and not doweight_stage1
        ):
            return situv2_a4w4_moe(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale,
                w2_scale,
                beta,
                linear_beta,
                out_dtype=dtype,
                out=output,
            )
        else:
            return FallbackReason.OUT_OF_CONTRACT
    return aklr_fused_moe(
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
        # vLLM passes 0.0 for no limit
        swiglu_limit=swiglu_limit or None,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        out_dtype=dtype,
        num_local_tokens=num_local_tokens,
        act_quant=act_quant,
        out=output,
    )


def _activation_dtype(hidden_states, w1, *, activation, quant_type, gate_mode):
    """The dtype aiter's fused MoE quantizes this call's activations to: asked as it asks."""
    from aiter.fused_moe import resolve_activation_dtype

    extra = {}
    if _takes_gate_up_interleave():
        extra["gate_up_interleave"] = getattr(w1, "gate_up_interleave", None)
    return resolve_activation_dtype(
        quant_type,
        w1.dtype,
        activation=activation,
        gate_mode=gate_mode,
        M=hidden_states.shape[0],
        hidden_dtype=hidden_states.dtype,
        has_a1_scale=False,
        **extra,
    )


@functools.cache
def _takes_gate_up_interleave() -> bool:
    """Whether the installed aiter (0.1.28 on) reads the weights' gate-up interleave."""
    from aiter.fused_moe import resolve_activation_dtype

    return "gate_up_interleave" in inspect.signature(resolve_activation_dtype).parameters
