###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.topk.schema import BIASED_GROUPED_TOPK, TOPK_TRANSFORM_PAGED

__all__ = ["biased_grouped_topk", "topk_transform_paged"]


@aklr_api(TOPK_TRANSFORM_PAGED)
def topk_transform_paged(
    scores,
    seq_lens,
    page_tables,
    k,
    out_page_indices,
    *,
    page_size=1,
    out_raw_indices=None,
): ...


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
