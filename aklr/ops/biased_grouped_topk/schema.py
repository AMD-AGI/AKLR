# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.biased_grouped_topk.reference import biased_grouped_topk_reference

__all__ = ["BIASED_GROUPED_TOPK"]


def _biased_grouped_topk_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # the fp32 weights to 1e-4; the ids exactly
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=1e-4 if rtol is None else rtol,
        atol=1e-4 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


BIASED_GROUPED_TOPK = Schema(
    op_type="biased_grouped_topk",
    description=(
        "Pick topk experts per token by sigmoid(router_logits) + correction_bias, among "
        "the topk_group of num_expert_group equal groups whose two best biased scores "
        "sum highest; weigh them with the unbiased sigmoid scores, divided by their sum "
        "when renormalize, times routed_scaling_factor."
    ),
    axes={
        "n_tokens": Var("Tokens routed in this call."),
        "n_experts": Const("Experts scored per token."),
        "topk": Const("Experts chosen per token."),
        "num_expert_group": Const("Expert groups.", abbrev="ng"),
        "topk_group": Const("Groups kept per token.", abbrev="kg"),
    },
    constraints=[
        "n_experts % num_expert_group == 0",
        "topk_group <= num_expert_group",
        "topk <= topk_group * (n_experts // num_expert_group)",
        "router_logits.dtype in ('bfloat16', 'float32')",
        "correction_bias.dtype == router_logits.dtype",
    ],
    inputs={
        "router_logits": Tensor(["n_tokens", "n_experts"], description="Router logits per token."),
        "correction_bias": Tensor(
            ["n_experts"],
            description="Per-expert bias added to the scores for selection only.",
        ),
        "topk": Scalar("int32", description="Experts chosen per token."),
        "num_expert_group": Scalar("int32", description="Expert groups."),
        "topk_group": Scalar("int32", description="Groups kept per token."),
        "renormalize": Scalar("bool", description="Whether the weights are divided by their sum."),
        "routed_scaling_factor": Scalar("float32", description="Multiplies every weight.", optional=True),
    },
    outputs={
        "topk_weights": Tensor(
            ["n_tokens", "topk"],
            dtype="float32",
            description=("Unbiased sigmoid scores of the chosen experts, renormalized and scaled."),
        ),
        "topk_ids": Tensor(
            ["n_tokens", "topk"],
            dtype="int32",
            description="Chosen experts, by descending biased score.",
        ),
    },
    reference=biased_grouped_topk_reference,
    check=_biased_grouped_topk_check,
)
