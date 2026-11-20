# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's fused_moe in the environment vLLM's Kimi-K3 recipe runs it with (VLLM_ROCM_USE_AITER_MOE_SITUV2=1, which
vLLM turns into AITER_SITUV2_A4W4=1; AITER_FLYDSL_STAGE2_FP8=1): that environment, not the call, gives SiTUv2 its
MXFP4 activations."""

from __future__ import annotations

import os

__all__ = ["situv2_a4w4_moe_baseline"]

_ENV = {"AITER_SITUV2_A4W4": "1", "AITER_SITUV2_A8W4": "0", "AITER_FLYDSL_STAGE2_FP8": "1"}


def situv2_a4w4_moe_baseline(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    beta,
    linear_beta,
    out_dtype=None,
    out=None,
):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    saved = {var: os.environ.get(var) for var in _ENV}
    os.environ.update(_ENV)
    try:
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            dtype=out_dtype,
            beta=beta,
            linear_beta=linear_beta,
            gate_mode="separated",
        )
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
