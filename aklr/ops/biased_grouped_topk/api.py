# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import api
from aklr.core.runtime.seam import aklr_route
from aklr.ops.biased_grouped_topk.contract import BIASED_GROUPED_TOPK

__all__ = ["biased_grouped_topk"]


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


@aklr_route("aiter.ops.topk.biased_grouped_topk")
def _(
    gating_output,
    correction_bias,
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    need_renorm,
    routed_scaling_factor=1.0,
):
    biased_grouped_topk(
        router_logits=gating_output,
        correction_bias=correction_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=need_renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
