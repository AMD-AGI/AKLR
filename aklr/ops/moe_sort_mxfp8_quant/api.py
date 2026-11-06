# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.moe_sort_mxfp8_quant.contract import MOE_SORT_MXFP8_QUANT

__all__ = ["moe_sort_mxfp8_quant"]


@api(MOE_SORT_MXFP8_QUANT)
def moe_sort_mxfp8_quant(
    hidden_states,
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    sorted_scales,
    n_experts,
    block_size=32,
): ...
