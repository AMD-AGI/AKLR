# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["biased_grouped_topk_baseline"]


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
