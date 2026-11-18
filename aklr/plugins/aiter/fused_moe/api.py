###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.plugins.aiter.fused_moe.schema import (
    FUSED_MOE_VARIANTS,
)

__all__ = ["fused_moe"]

# Positional parameters are the schema's inputs, then its outputs, in order; the rest pick the schema.


# aiter's swiglu reads the 16 x 16 shuffled storage only, its SiTUv2 the gate-up interleaved one
@aklr_api(FUSED_MOE_VARIANTS)
def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=None,
    act_quant=None,
    w1_scale=None,
    w2_scale=None,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    num_local_tokens=None,
    out=None,
    gate_mode="separated",
    use_shuffled_weight=False,
): ...
