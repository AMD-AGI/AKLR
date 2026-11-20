###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baselines of the topk op: the backend calls its schemas' standards time a kernel against."""

from __future__ import annotations


def topk_transform_baseline(scores, seq_lens, page_tables, out_page_indices, page_size=1, **_):
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

    metadata = plan_topk_v2(seq_lens)
    return lambda: topk_transform_paged_v2(
        scores=scores,
        seq_lens=seq_lens,
        page_tables=page_tables,
        out_page_indices=out_page_indices,
        page_size=page_size,
        metadata=metadata,
    )


def biased_grouped_topk_baseline(
    router_logits,
    correction_bias,
    renormalize,
    routed_scaling_factor=1.0,
    topk_weights=None,
    topk_ids=None,
    num_expert_group=1,
    topk_group=1,
    **_,
):
    from aiter.ops.topk import biased_grouped_topk

    return lambda: biased_grouped_topk(
        gating_output=router_logits,
        correction_bias=correction_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        need_renorm=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
