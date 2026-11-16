# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MoE routing, ``topk_gating``, served by AKLR's ``biased_grouped_topk`` in one group:
sigmoid scores, experts picked by score plus bias, weighted by the score alone."""

from __future__ import annotations

from aklr.ops.topk.api import biased_grouped_topk
from aklr.plugins.aiter.provider._base import offer, selector, served


@selector("topk_gating")
def _select_topk_gating(args, kwargs):
    (
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        need_renorm,
        routed_scaling_factor,
        score_func,
        num_shared_experts,
        num_valid_rows,
    ) = args
    if score_func != "sigmoid" or correction_bias is None or num_shared_experts or num_valid_rows is not None:
        return None
    return offer(
        biased_grouped_topk.serving(
            gating_output,
            correction_bias,
            topk_weights,
            topk_indices,
            1,
            1,
            need_renorm,
            routed_scaling_factor,
        ),
        _run_topk_gating,
    )


def _run_topk_gating(
    topk_weights,
    topk_indices,
    gating_output,
    correction_bias,
    need_renorm,
    routed_scaling_factor,
    score_func,
    num_shared_experts,
    num_valid_rows,
):
    served(
        biased_grouped_topk(
            gating_output,
            correction_bias,
            topk_weights,
            topk_indices,
            1,
            1,
            need_renorm,
            routed_scaling_factor,
        )
    )
