###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.mega_moe.schema import MEGA_MOE

__all__ = ["mega_moe"]

# Positional parameters are the schema's inputs, then its outputs, in order.


@aklr_api(MEGA_MOE)
def mega_moe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    ep_size,
    swiglu_limit,
    out=None,
): ...
