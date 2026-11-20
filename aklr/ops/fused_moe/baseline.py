###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the fused_moe op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations

import os


def aiter_moe_act_quant(activation, swiglu_limit, n_tokens):
    if activation == "swiglu":  # bf16 below aiter's GPTOSS_SWIGLU_MXFP4_BF16_BOUND rows, 256 by default
        return None if n_tokens < 256 else "mxfp4"
    if swiglu_limit is not None:
        return None
    return "mxfp8" if activation == "situv2" else "mxfp4"


def fused_moe_baseline(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation="silu",
    act_quant=None,
    w1_scale=None,
    w2_scale=None,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    gate_mode="separated",
    use_shuffled_weight=False,
    **_,
):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    if act_quant != aiter_moe_act_quant(activation, swiglu_limit, hidden_states.shape[0]):
        return None  # aiter picks the activation quant by its own rules
    if activation == "swiglu":
        if swiglu_limit != 7.0:
            return None  # aiter's swiglu_and_mul clamps at gpt-oss's 7, whatever the call passes
    elif swiglu_limit is not None and swiglu_limit < 10:
        return None  # aiter's CK-Tile decode path does not clamp
    if activation == "gelu" and expert_mask is not None:
        return None  # aiter has no gelu a4w4 under expert parallelism
    w1.is_shuffled = w2.is_shuffled = use_shuffled_weight
    # the environment sglang runs aiter with, read by aiter at call time
    env = {}
    if activation == "situv2":
        env["AITER_SITUV2_A8W4"] = "1"  # SiTUv2's mxfp8 activations (act_quant)
    if gate_mode == "interleave":
        # DeepSeek-V4's serving recipes (sglang's cookbook, InferenceX): e4m3 activations at every row count;
        # aiter's default bound of 256 has no kernel for the clamped SwiGLU's bf16 activations at 16-128 rows,
        # nor at 7 and 14 for V4-Pro's inter_dim 384
        env["AITER_BF16_FP8_MOE_BOUND"] = "0"

    def call():
        saved = {var: os.environ.get(var) for var in env}
        os.environ.update(env)
        try:
            return fused_moe(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_mask=expert_mask,
                activation=getattr(ActivationType, activation.capitalize()),
                quant_type=QuantType.per_1x32,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                num_local_tokens=num_local_tokens,
                swiglu_limit=swiglu_limit,
                beta=beta,
                linear_beta=linear_beta,
                gate_mode=gate_mode,
            )
        finally:
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value

    return call
