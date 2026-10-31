# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["fused_moe_baseline"]

# sglang's DeepSeek-V4 recipe (aklr/benchmarks/models/deepseekv4/README.md): aiter's
# default of 256 has no kernel for the clamped SwiGLU's bf16 activations at 32-128 rows
_INTERLEAVE_ENV = {"AITER_BF16_FP8_MOE_BOUND": "16"}


def fused_moe_baseline(**call):
    """aiter's fused_moe through its route, in the environment sglang runs it with."""
    from aklr.core.runtime import seam
    from aklr.ops.fused_moe.contract import FUSED_MOE

    (route,) = seam.contract_routes(FUSED_MOE)
    bound = route.backend_bind(FUSED_MOE, call)
    env = dict(bound.env)
    if call["gate_mode"] == "interleave":
        env |= _INTERLEAVE_ENV
    with seam.apply_env(env):
        return bound.backend_fn(*bound.args, **bound.kwargs)
