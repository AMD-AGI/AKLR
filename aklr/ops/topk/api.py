# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route, must_default
from aklr.ops.topk.contract import TOPK_TRANSFORM_PAGED

__all__ = ["topk_transform_paged"]


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


@aklr_route("sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2")
def _(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    metadata,
    out_raw_indices=None,
):
    must_default(out_raw_indices)  # a second raw output is not in the contract
    topk_transform_paged(
        scores=scores,
        seq_lens=seq_lens,
        metadata=metadata,
        out_page_indices=out_page_indices,
        page_size=page_size,
        page_tables=page_tables,
    )
