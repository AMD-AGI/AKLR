# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["BIASED_GROUPED_TOPK", "CONTRACTS"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

BIASED_GROUPED_TOPK = OpContract(
    contract_id="biased_grouped_topk@1.0",
    description=(
        "Pick topk experts per token by sigmoid(router_logits) + correction_bias; weigh "
        "them with the unbiased sigmoid scores, divided by their sum when renormalize, "
        "times routed_scaling_factor. Expert groups are all kept (topk_group == "
        "num_expert_group), so grouping selects nothing."
    ),
    op_type="biased_grouped_topk",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 65536, is_var=True),
        "n_experts": axis("Experts scored per token.", 1, 1024),
        "topk": axis("Experts chosen per token.", 1, 1024),
    },
    attrs={
        "num_expert_group": attr(
            "Expert groups; every group is kept, so they select nothing.",
            1,
            2,
            4,
            8,
            param_name="num_expert_group",
        ),
        "topk_group": attr("Groups kept: all of them.", 1, 2, 4, 8, param_name="topk_group"),
        "renormalize": attr(
            "Whether the weights are divided by their sum.",
            False,
            True,
            param_name="renormalize",
        ),
    },
    constraints=(
        "topk <= n_experts",
        # aiter scores the groups wrongly when it drops some (2 or 4 groups)
        "topk_group == num_expert_group",
        "n_experts % num_expert_group == 0",
    ),
    inputs={
        "router_logits": TensorSpec(
            description="Router logits per token.",
            shape=("n_tokens", "n_experts"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT32)),
            # Ties on the biased score are possible (no tie-free bf16 distribution yet):
            # the lower expert ranks first.
            distribution=FiniteDistribution(),
        ),
        "correction_bias": TensorSpec(
            description="Per-expert bias added to the scores for selection only.",
            shape=("n_experts",),
            dtype=DTypeSpec(same_as_tensor="router_logits"),
            distribution=FiniteDistribution(),
        ),
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies every weight.",
            dtype=DType.FLOAT32,
            interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
            default=1.0,
        ),
    },
    outputs={
        "topk_weights": TensorSpec(
            description=("Unbiased sigmoid scores of the chosen experts, renormalized and scaled."),
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-4),
        ),
        "topk_ids": TensorSpec(
            description="Chosen experts, by descending biased score.",
            shape=("n_tokens", "topk"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (BIASED_GROUPED_TOPK,)
