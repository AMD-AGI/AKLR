# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["fused_moe_baseline"]


def fused_moe_baseline(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation="silu",
    quant_type=None,
    doweight_stage1=False,
    w1_scale=None,
    w2_scale=None,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode="separated",
    out_dtype=None,
    num_local_tokens=None,
    act_quant=4,
    out=None,
):
    """aiter's fused_moe, in the environment sglang runs it with; that environment, not
    the call, gives situv2 its 8-bit activations."""
    import os

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    is_situv2 = activation == "situv2"
    if act_quant != (8 if is_situv2 else 4):
        raise NotImplementedError(f"aiter's fused_moe: act_quant={act_quant} with activation={activation!r}")
    env = {"AITER_SITUV2_A8W4": "1"} if is_situv2 else {}
    if gate_mode == "interleave":
        # sglang's DeepSeek-V4 recipe (aklr/benchmarks/models/deepseekv4/README.md): aiter's
        # default of 256 has no kernel for the clamped SwiGLU's bf16 activations at 32-128 rows
        env["AITER_BF16_FP8_MOE_BOUND"] = "16"
    # the contract spells aiter's enums by their lower-case names
    activation, quant_type = (
        {m.name.lower(): m for m in enum.__members__.values()}[value]
        for enum, value in ((ActivationType, activation), (QuantType, quant_type))
    )
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
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            num_local_tokens=num_local_tokens,
            dtype=out_dtype,
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
