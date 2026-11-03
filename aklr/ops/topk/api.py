# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
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
