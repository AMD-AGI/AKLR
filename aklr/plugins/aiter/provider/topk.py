# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's MoE routing by sigmoid scores and a bias, served by AKLR's ``biased_grouped_topk``: aiter's
``topk_gating`` in one group, and its ``biased_grouped_topk`` with every group kept. Experts are picked by
score plus bias, weighted by the score alone."""

from __future__ import annotations

import os

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
    if (
        score_func != "sigmoid"
        or correction_bias is None
        or num_shared_experts
        or num_valid_rows is not None
        or _herd()
    ):
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


@selector("biased_grouped_topk")
def _select_biased_grouped_topk(args, kwargs):
    (
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        routed_scaling_factor,
        num_valid_rows,
    ) = args
    # AKLR keeps every group (aiter scores the groups wrongly when it drops some), has no padding rows, and
    # takes the bias in the scores' dtype
    if (
        num_valid_rows is not None
        or topk_group != num_expert_group
        or correction_bias.dtype != gating_output.dtype
        or (num_expert_group == 1 and _herd())
    ):
        return None
    return offer(
        biased_grouped_topk.serving(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            routed_scaling_factor,
        ),
        _run_biased_grouped_topk,
    )


def _run_biased_grouped_topk(
    gating_output,
    correction_bias,
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    need_renorm,
    routed_scaling_factor,
    num_valid_rows,
):
    # aiter's contract returns None: topk_weights and topk_ids are the results, written in place
    served(
        biased_grouped_topk(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            routed_scaling_factor,
        )
    )


def _herd() -> bool:
    """Whether aiter routes by HERD (``AITER_FLYDSL_USE_HERD=1``, FlyDSL on gfx950): for its model profiles,
    one group, top-(k+1) candidates and a drop by batch popularity rather than the exact top-k AKLR
    computes."""
    return os.environ.get("AITER_FLYDSL_USE_HERD", "0") == "1"
