# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["biased_grouped_topk_baseline", "topk_transform_paged_baseline"]


def biased_grouped_topk_baseline(
    router_logits,
    correction_bias,
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    renormalize,
    routed_scaling_factor=1.0,
):
    """aiter's biased grouped top-k, into the caller's outputs."""
    from aiter.ops.topk import biased_grouped_topk

    biased_grouped_topk(
        router_logits,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


def topk_transform_paged_baseline(scores, seq_lens, page_tables, out_page_indices, page_size, metadata):
    """sglang's DeepSeek-V4 paged top-k, into the caller's ``out_page_indices``."""
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

    topk_transform_paged_v2(scores, seq_lens, page_tables, out_page_indices, page_size, metadata)
    return out_page_indices
