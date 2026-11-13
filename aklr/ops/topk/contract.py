# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    ElementOrder,
    FiniteDistribution,
    IntRangeDistribution,
    PermutationDistribution,
    ToleranceBound,
    TopkStaticPlanDistribution,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["BIASED_GROUPED_TOPK", "CONTRACTS", "TOPK_TRANSFORM_PAGED"]

_INT32 = DTypeSpec(any_of=(DType.INT32,))
_PAGE_SIZES = tuple(2**i for i in range(32))  # sglang's uint32 powers of two

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

BIASED_GROUPED_TOPK = OpContract(
    contract_id="biased_grouped_topk@1.0",
    description=(
        "Pick topk experts per token by sigmoid(router_logits) + correction_bias; weigh "
        "them with the unbiased sigmoid scores, divided by their sum when renormalize, "
        "times routed_scaling_factor. Expert groups are all kept (topk_group == "
        "num_expert_group), so grouping selects nothing."
    ),
    op_type="topk",
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

TOPK_TRANSFORM_PAGED = OpContract(
    contract_id="topk_transform_paged@1.0",
    description=(
        "Select each row's topk highest scores among its first seq_lens "
        "candidates; write their positions, or their page slots, padded with -1."
    ),
    op_type="topk",
    axes={
        "n_tokens": axis("Rows, one per token.", 1, 65536, is_var=True),
        "n_candidates": axis(
            "Candidate scores per row, 4-float aligned rows.",
            4,
            2**24,
            multiple_of=4,
            is_var=True,
        ),
        "n_plan_rows": axis("Plan rows: n_tokens + 1.", 2, 65537, is_var=True),
        "topk": axis("Candidates selected per row.", 1, 2048),
        "n_pages": axis("Page table columns.", 0, 2**31, is_var=True),
    },
    attrs={
        "page_size": attr("Slots per page.", *_PAGE_SIZES, param_name="page_size"),
        "has_page_tables": attr("Whether page_tables is passed.", False, True, default=False),
    },
    constraints=(
        "n_plan_rows == n_tokens + 1",
        "n_pages * page_size <= 2147483648",  # every slot fits in int32
    ),
    inputs={
        "scores": TensorSpec(
            description="Candidate scores per row.",
            shape=("n_tokens", "n_candidates"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=PermutationDistribution(),
        ),
        "seq_lens": TensorSpec(
            description="Valid candidates per row, a prefix the page table covers.",
            shape=("n_tokens",),
            dtype=_INT32,
            distribution=(
                (
                    "has_page_tables",
                    IntRangeDistribution(lo="0", hi="min(n_candidates, n_pages * page_size)"),
                ),
                (None, IntRangeDistribution(lo="0", hi="n_candidates")),
            ),
        ),
        "page_tables": TensorSpec(
            description="Page of each page_size candidates, distinct within a row.",
            shape=("n_tokens", "n_pages"),
            dtype=_INT32,
            present_when="has_page_tables",
            distribution=IntRangeDistribution(lo="0", hi="n_pages - 1", distinct=True),
        ),
        "metadata": TensorSpec(
            description="sglang's top-k metadata (plan_topk_v2), static threshold.",
            shape=("n_plan_rows", 2),
            dtype=_INT32,
            distribution=TopkStaticPlanDistribution(),
        ),
    },
    outputs={
        "out_page_indices": TensorSpec(
            description=(
                "Selected positions, or page_tables slots when passed; -1 after a short row's entries."
            ),
            shape=("n_tokens", "topk"),
            dtype=_INT32,
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    # Ties and a full row's order are the kernel's choice.
    compare=CompareSpec(
        determinism=DeterminismLevel.BOUNDED,
        element_order=ElementOrder.ANY_IN_FULL_ROWS,
    ),
)

CONTRACTS = (BIASED_GROUPED_TOPK, TOPK_TRANSFORM_PAGED)
