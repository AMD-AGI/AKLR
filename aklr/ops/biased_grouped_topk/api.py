###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.biased_grouped_topk.schema import BIASED_GROUPED_TOPK

__all__ = ["biased_grouped_topk"]


@aklr_api(BIASED_GROUPED_TOPK)
def biased_grouped_topk(
    router_logits,
    correction_bias,
    renormalize,
    routed_scaling_factor=1.0,
    topk_weights=None,
    topk_ids=None,
    *,
    topk,
    num_expert_group=1,
    topk_group=1,
): ...
