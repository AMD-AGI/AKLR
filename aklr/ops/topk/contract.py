# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DeterminismLevel,
    DType,
    ElementOrder,
    IntRangeDistribution,
    PermutationDistribution,
    TopkStaticPlanDistribution,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    attr,
    axis,
)

__all__ = ["CONTRACTS", "TOPK_TRANSFORM_PAGED"]

_INT32 = DTypeSpec(any_of=(DType.INT32,))
_PAGE_SIZES = tuple(2**i for i in range(32))  # sglang's uint32 powers of two

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

CONTRACTS = (TOPK_TRANSFORM_PAGED,)
