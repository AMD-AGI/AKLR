# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
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

__all__ = ["CONTRACTS", "DEEPSEEK_V4_ROUTER"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

DEEPSEEK_V4_ROUTER = OpContract(
    contract_id="deepseek_v4_router@1.0",
    description=(
        "sglang's select_experts on ROCm for a sqrtsoftplus router whose shared experts are fused "
        "into the top-k (DeepSeek-V4): each token's n_slots - n_shared experts by sqrt(softplus(router_logits)) "
        "+ correction_bias, weighed with the unbiased scores, divided by their sum when "
        "renormalize, times routed_scaling_factor; then n_shared columns of shared experts "
        "n_experts + s at weight shared_scale. Rows at or past num_token_non_padded, when given, "
        "keep the shared ids and take routed ids 0 and weights 0."
    ),
    op_type="moe_topk",
    axes={
        "n_tokens": axis("Tokens routed in this call.", 1, 65536, is_var=True),
        "n_experts": axis("Routed experts scored per token.", 1, 1024),
        "n_slots": axis("Experts per token: the routed ones, then the n_shared shared ones.", 2, 64),
    },
    attrs={
        "n_shared": attr(
            "Shared experts appended to every token, after its n_slots - n_shared routed ones.",
            1,
            param_name="n_shared",
        ),
        "renormalize": attr(
            "Whether the routed weights are divided by their sum.",
            True,
            False,
            param_name="renormalize",
        ),
        "has_num_token_non_padded": attr(
            "Whether num_token_non_padded is passed.", False, True, default=False, abbrev="padded"
        ),
    },
    # aiter's wave64 topk_gating holds n_experts / 64 experts per lane
    constraints=("n_slots - n_shared <= n_experts", "n_experts % 64 == 0"),
    inputs={
        "router_logits": TensorSpec(
            description="Router logits per token.",
            shape=("n_tokens", "n_experts"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT32)),
            # Ties on the biased score are possible: no tie-free bf16 distribution yet.
            distribution=FiniteDistribution(),
        ),
        "correction_bias": TensorSpec(
            description="Per-expert bias added to the scores for selection only.",
            shape=("n_experts",),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT32)),
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
        "num_token_non_padded": TensorSpec(
            description="Rows that are real tokens; later rows are padding.",
            shape=(1,),
            dtype=DTypeSpec(any_of=(DType.INT32, DType.INT64)),
            present_when="has_num_token_non_padded",
            distribution=IntRangeDistribution(lo=0, hi="n_tokens"),
        ),
        "routed_scaling_factor": ScalarSpec(
            description="Multiplies every routed weight.",
            dtype=DType.FLOAT32,
            interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
            default=1.0,
        ),
        "shared_scale": ScalarSpec(
            description="The weight of every shared expert column.",
            dtype=DType.FLOAT32,
            interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
            default=1.0,
        ),
    },
    outputs={
        "topk_weights": TensorSpec(
            description=(
                "Per token: the routed experts' unbiased scores, renormalized and scaled, then "
                "shared_scale per shared expert; 0 on padded rows."
            ),
            shape=("n_tokens", "n_slots"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=ToleranceBound(absolute_tolerance=1e-4, relative_tolerance=1e-4),
        ),
        "topk_ids": TensorSpec(
            description=(
                "Per token: the routed experts by descending biased score (0 on padded rows), "
                "then shared experts n_experts .. n_experts + n_shared - 1."
            ),
            shape=("n_tokens", "n_slots"),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (DEEPSEEK_V4_ROUTER,)
