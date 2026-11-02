###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.fused_moe.schema import fused_moe_op_schema

__all__ = ["fused_moe"]

# Positional parameters are the schema's inputs, then its outputs, in order; the rest pick the schema.


@aklr_api(fused_moe_op_schema)
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
