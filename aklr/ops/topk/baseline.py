# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["topk_transform_paged_baseline"]


def topk_transform_paged_baseline(scores, seq_lens, page_tables, out_page_indices, page_size, metadata):
    """sglang's DeepSeek-V4 paged top-k, into the caller's ``out_page_indices``."""
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

    topk_transform_paged_v2(scores, seq_lens, page_tables, out_page_indices, page_size, metadata)
    return out_page_indices
