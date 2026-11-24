# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused MoE, ``fused_moe``, served by AKLR's ``fused_moe``, and SiTUv2 on MXFP4 activations
(vLLM's Kimi-K3) by ``situv2_a4w4_moe``. aiter routes the layer beneath its public ``fused_moe``: the
call arrives with aiter's enums as ints and expert parallelism's stage-2 scatter as the ``ep_*`` fields,
and AKLR takes of it what :func:`op_kwargs` maps."""

from __future__ import annotations

import functools
import inspect

from aklr.core.contract.spec import attr_value
from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.fused_moe.api import fused_moe
from aklr.ops.fused_moe.reference import SWIGLU_BF16_ROWS
from aklr.ops.situv2_a4w4_moe.api import situv2_a4w4_moe
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
        _splitk,
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
    # is not. aiter's sorting policy only schedules its own sort, the HERD routing factor only scales
    # herd_topk's candidates, and splitk asks aiter for a split of K, which moves only the order of the sums.
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
    a4w4 = _situv2_a4w4_kwargs(
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
    if isinstance(a4w4, FallbackReason):
        return None
    if a4w4 is not None:
        return offer(
            situv2_a4w4_moe.serving(hidden_states, w1, w2, topk_weight, topk_ids, **a4w4), _run_fused_moe
        )
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


def _situv2_a4w4_kwargs(
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
    """``situv2_a4w4_moe``'s keyword arguments, after its five tensors, for a call aiter computes as SiTUv2
    on MXFP4 activations, as vLLM's Kimi-K3 runs it (AITER_SITUV2_A4W4, gate and up rows separated, an e8m0
    scale per 32); why AKLR declines such a call; or ``None`` for any other call."""
    from aiter import ActivationType, QuantType, dtypes

    # aiter's enums are pybind11's: each lookup is a new object, equal by value only
    activation, quant_type = ActivationType(activation), QuantType(quant_type)
    if (
        activation != ActivationType.Situv2
        or quant_type != QuantType.per_1x32
        or attr_value(gate_mode) != "separated"
    ):
        return None
    picked = _activation_dtype(
        hidden_states, w1, activation=activation, quant_type=quant_type, gate_mode=gate_mode
    )
    if picked != dtypes.fp4x2:
        return None
    # every expert routed, weighted after stage 2, no limit (vLLM's 0.0), both scales and both betas
    if (
        expert_mask is not None
        or doweight_stage1
        or num_local_tokens is not None
        or swiglu_limit
        or w1_scale is None
        or w2_scale is None
        or beta is None
        or linear_beta is None
    ):
        return FallbackReason.OUT_OF_CONTRACT
    return {
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "beta": beta,
        "linear_beta": linear_beta,
        "out_dtype": dtype,
        "out": output,
    }


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
    splitk,
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
    a4w4 = _situv2_a4w4_kwargs(
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
    if a4w4 is not None:
        return served(situv2_a4w4_moe(hidden_states, w1, w2, topk_weight, topk_ids, **served(a4w4)))
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


def op_kwargs(
    hidden_states,
    w1,
    *,
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
    """AKLR's ``fused_moe`` keyword arguments, after its five tensors, for a call of aiter's fused MoE
    passing nothing else, or why aiter computes it. ``activation`` and ``quant_type`` are aiter's
    enums."""
    act_quant = 4
    name = attr_value(activation)
    # aiter picks SiTUv2's activation dtype at each call (AITER_SITUV2_A8W4, AITER_SITUV2_A4W4): fp8 is
    # fused_moe's, on sglang's gate-up interleaved weights; any other goes back to aiter
    if name == "situv2":
        from aiter import dtypes

        q_dtype = _activation_dtype(
            hidden_states, w1, activation=activation, quant_type=quant_type, gate_mode=gate_mode
        )
        if q_dtype == dtypes.fp8:
            act_quant = 8
        else:
            return FallbackReason.OUT_OF_CONTRACT
    # fused_moe rounds SwiGLU's activations to fp4 from SWIGLU_BF16_ROWS rows on, aiter from its
    # GPTOSS_SWIGLU_MXFP4_BF16_BOUND: a call they would round differently goes back to aiter
    elif name == "swiglu" and attr_value(gate_mode) == "separated":
        rounds = _aiter_rounds_swiglu(hidden_states, w1, activation=activation, quant_type=quant_type)
        if rounds != (hidden_states.shape[0] >= SWIGLU_BF16_ROWS):
            return FallbackReason.OUT_OF_CONTRACT
    return {
        "expert_mask": expert_mask,
        "activation": activation,
        "quant_type": quant_type,
        "doweight_stage1": doweight_stage1,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        # vLLM passes 0.0 for no limit
        "swiglu_limit": swiglu_limit or None,
        "beta": beta,
        "linear_beta": linear_beta,
        "gate_mode": gate_mode,
        "out_dtype": dtype,
        "num_local_tokens": num_local_tokens,
        "act_quant": act_quant,
        "out": output,
    }


def _aiter_rounds_swiglu(hidden_states, w1, *, activation, quant_type) -> bool:
    """Whether aiter's fused MoE rounds this SwiGLU call's activations (gate and up separated) to fp4."""
    import importlib

    from aiter import dtypes

    aiter_moe = importlib.import_module("aiter.fused_moe")
    if hasattr(aiter_moe, "resolve_activation_dtype"):
        q_dtype = _activation_dtype(
            hidden_states, w1, activation=activation, quant_type=quant_type, gate_mode="separated"
        )
        return q_dtype == dtypes.fp4x2
    # an aiter without the resolver (CI's) picks it inline, by the same bound
    return hidden_states.shape[0] >= aiter_moe._SWIGLU_MXFP4_BF16_BOUND


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
