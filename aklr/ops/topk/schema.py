###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import torch

from aklr.core.schema import Const, Scalar, Schema, Standard, Tensor, Var, default_check
from aklr.ops.topk.baseline import biased_grouped_topk_baseline, topk_transform_baseline
from aklr.ops.topk.reference import biased_grouped_topk_reference, topk_transform_paged_reference

__all__ = ["BIASED_GROUPED_TOPK", "TOPK_TRANSFORM_PAGED"]


def _topk_transform_paged_check(reference_outputs, actual_outputs, **_unused):
    # a full row's selections may come in any order: compare each row sorted
    reference, actual = (
        o[0] if isinstance(o, (list, tuple)) else o for o in (reference_outputs, actual_outputs)
    )
    if reference.shape != actual.shape:
        return False
    return default_check(
        reference.sort(dim=-1).values,
        actual.sort(dim=-1).values,
        min_cos_sim=None,
    )


TOPK_TRANSFORM_PAGED = Schema(
    op_type="topk_transform_paged",
    description=(
        "Select each row's k highest scores among its first seq_lens "
        "candidates; write their positions, or their page slots, padded with -1."
    ),
    axes={
        "n_tokens": Var("Rows, one per token."),
        "n_candidates": Var("Candidate scores per row."),
        "k": Const("Candidates selected per row.", abbrev="k"),
        "n_pages": Var("Page table columns."),
        "page_size": Const("Slots per page.", abbrev="ps"),
    },
    constraints=[
        "page_size >= 1",
        # every slot fits in int32
        "page_tables is None or n_pages * page_size <= 2147483648",
    ],
    inputs={
        "scores": Tensor(
            ["n_tokens", "n_candidates"],
            dtype="float32",
            description="Candidate scores per row.",
        ),
        "seq_lens": Tensor(
            ["n_tokens"],
            dtype="int32",
            description="Valid candidates per row, a prefix the page table covers.",
        ),
        "page_tables": Tensor(
            ["n_tokens", "n_pages"],
            dtype="int32",
            optional=True,
            description="Page of each page_size candidates, distinct within a row.",
        ),
        "k": Scalar("int32", description="Candidates selected per row."),
        "page_size": Scalar("int32", optional=True, description="Slots per page."),
    },
    outputs={
        "out_page_indices": Tensor(
            ["n_tokens", "k"],
            dtype="int32",
            description=(
                "Selected positions, or page_tables slots when passed; -1 after a short row's entries."
            ),
        )
    },
    standard=Standard(
        reference=topk_transform_paged_reference,
        check=_topk_transform_paged_check,
        baseline=topk_transform_baseline,
    ),
)


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


def _biased_grouped_topk_init(
    *, n_tokens, n_experts, topk, num_expert_group, topk_group, device="cuda", seed=0
):
    torch.manual_seed(seed)
    return {
        "router_logits": torch.randn(n_tokens, n_experts, device=device).to(torch.bfloat16),
        # ties on the biased score are possible; the reference breaks them to the lower expert
        "correction_bias": (torch.randn(n_experts, device=device) * 0.1).to(torch.bfloat16),
        "topk": topk,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "renormalize": True,
        "routed_scaling_factor": 1.0,
    }


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
    standard=Standard(
        reference=biased_grouped_topk_reference,
        check=_biased_grouped_topk_check,
        init=_biased_grouped_topk_init,
        baseline=biased_grouped_topk_baseline,
    ),
)
