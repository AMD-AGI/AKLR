# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.plugins.aiter.situv2_a4w4_moe.contract import SITUV2_A4W4_MOE

__all__ = ["situv2_a4w4_moe"]


@api(SITUV2_A4W4_MOE)
def situv2_a4w4_moe(
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
): ...
