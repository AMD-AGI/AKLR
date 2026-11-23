# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Map AITER fused-MoE arguments to their AKLR contract values."""

from __future__ import annotations

import functools
import inspect

from aklr.core.contract.spec import attr_value
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.aiter.fused_moe.reference import _SWIGLU_BF16_ROWS


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
    # fused_moe rounds SwiGLU's activations to fp4 from _SWIGLU_BF16_ROWS rows on, aiter from its
    # GPTOSS_SWIGLU_MXFP4_BF16_BOUND: a call they would round differently goes back to aiter
    elif name == "swiglu" and attr_value(gate_mode) == "separated":
        rounds = _aiter_rounds_swiglu(hidden_states, w1, activation=activation, quant_type=quant_type)
        if rounds != (hidden_states.shape[0] >= _SWIGLU_BF16_ROWS):
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
