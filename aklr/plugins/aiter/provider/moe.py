# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused MoE, ``fused_moe``, served by AKLR's ``fused_moe``. aiter routes the layer beneath its
public ``fused_moe``: the call arrives with aiter's enums as ints and expert parallelism's stage-2 scatter
as the ``ep_*`` fields, and AKLR takes of it what :func:`aklr.plugins.aiter.fused_moe.op_kwargs` maps for
the public function."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.fused_moe.api import fused_moe
from aklr.plugins.aiter.fused_moe import op_kwargs
from aklr.plugins.aiter.provider._base import offer, selector, served


@selector("fused_moe")
def _select_fused_moe(args, kwargs):
    (
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
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
        _moe_sorting_dispatch_policy,
        dtype,
        hidden_pad,
        intermediate_pad,
        bias1,
        bias2,
        swiglu_limit,
        beta,
        linear_beta,
        gate_mode,
        ep_arena_handle,
        ep_combine_input_offset,
        ep_slot_stride_bytes,
        ep_max_tokens_per_rank,
        ep_world_size,
        ep_combine_quant,
        ep_source_token_map,
        output,
        quant_type_a,
        quant_dtype_a,
        quant_dtype_a2,
        herd_topk,
        _herd_routed_scaling_factor,
        deterministic,
    ) = args
    # block_size_M other than -1 pins aiter's tile; the stage-2 scatter (ep_*) writes the rows into the peers'
    # buffers, where AKLR returns them; deterministic asks for a fixed order of the top-k sums, which AKLR's
    # is not. aiter's sorting policy only schedules its own sort, and the HERD routing factor only scales
    # herd_topk's candidates.
    if (
        block_size_M != -1
        or hidden_pad
        or intermediate_pad
        or a1_scale is not None
        or a2_scale is not None
        or bias1 is not None
        or bias2 is not None
        or ep_source_token_map is not None
        or ep_arena_handle
        or ep_combine_input_offset
        or ep_slot_stride_bytes
        or ep_max_tokens_per_rank
        or ep_world_size
        or ep_combine_quant
        or quant_type_a is not None
        or quant_dtype_a is not None
        or quant_dtype_a2 is not None
        or herd_topk
        or deterministic
    ):
        return None
    call = _op_kwargs(
        hidden_states,
        w1,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        num_local_tokens,
        dtype,
        swiglu_limit,
        beta,
        linear_beta,
        gate_mode,
        output,
    )
    if isinstance(call, FallbackReason):
        return None
    return offer(fused_moe.serving(hidden_states, w1, w2, topk_weight, topk_ids, **call), _run_fused_moe)


def _op_kwargs(
    hidden_states,
    w1,
    expert_mask,
    activation,
    quant_type,
    doweight_stage1,
    w1_scale,
    w2_scale,
    num_local_tokens,
    dtype,
    swiglu_limit,
    beta,
    linear_beta,
    gate_mode,
    output,
):
    """``op_kwargs`` of the call: aiter's custom-op schema carries its enums as their values."""
    from aiter import ActivationType, QuantType

    return op_kwargs(
        hidden_states,
        w1,
        expert_mask=expert_mask,
        activation=ActivationType(activation),
        quant_type=QuantType(quant_type),
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        num_local_tokens=num_local_tokens,
        dtype=dtype,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        output=output,
    )


def _drop_presort():
    """aiter's own fused MoE consumes, beneath its route, the sort a caller registered for the next call
    (``register_presorted_moe``). AKLR sorts by itself and drops it, or it would apply to the next call aiter
    computes, whose routing buffers are often this call's."""
    from aiter.presorted_moe import pop_presorted_moe

    pop_presorted_moe()


def _run_fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
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
    moe_sorting_dispatch_policy,
    dtype,
    hidden_pad,
    intermediate_pad,
    bias1,
    bias2,
    swiglu_limit,
    beta,
    linear_beta,
    gate_mode,
    ep_arena_handle,
    ep_combine_input_offset,
    ep_slot_stride_bytes,
    ep_max_tokens_per_rank,
    ep_world_size,
    ep_combine_quant,
    ep_source_token_map,
    output,
    quant_type_a,
    quant_dtype_a,
    quant_dtype_a2,
    herd_topk,
    herd_routed_scaling_factor,
    deterministic,
):
    # first, so that no failure below leaves it registered
    _drop_presort()
    call = _op_kwargs(
        hidden_states,
        w1,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        num_local_tokens,
        dtype,
        swiglu_limit,
        beta,
        linear_beta,
        gate_mode,
        output,
    )
    # [M, model_dim], in output when given
    return served(fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **served(call)))
