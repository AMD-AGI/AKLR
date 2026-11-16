# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.ops.topk.contract import BIASED_GROUPED_TOPK, TOPK_TRANSFORM_PAGED

__all__ = ["biased_grouped_topk", "topk_transform_paged"]


@api(BIASED_GROUPED_TOPK)
def biased_grouped_topk(
    router_logits,
    correction_bias,
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    renormalize,
    routed_scaling_factor=1.0,
): ...


@api(TOPK_TRANSFORM_PAGED)
def topk_transform_paged(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    metadata,
    out_raw_indices=None,
): ...
