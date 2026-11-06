# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.mega_moe.contract import MEGA_MOE

__all__ = ["mega_moe"]


@api(MEGA_MOE)
def mega_moe(
    hidden_states,
    w1,  # [local experts, 2 * inter_dim, model_dim], fp4x2
    w2,  # [local experts, model_dim, inter_dim], fp4x2
    topk_weights,
    topk_ids,  # global expert ids
    w1_scale=None,
    w2_scale=None,
    ep_size=1,
    swiglu_limit=10.0,
    out_dtype=None,
): ...
