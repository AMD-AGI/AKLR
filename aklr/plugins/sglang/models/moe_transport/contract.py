# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
)

__all__ = ["CONTRACTS", "DEEPSEEK_V4_DP_MOE_GATHER"]

DEEPSEEK_V4_DP_MOE_GATHER = OpContract(
    contract_id="deepseek_v4_dp_moe_gather@1.0",
    description=(
        "The local rows a DP-attention rank contributes to the TP MoE's all-gather, as sglang's "
        "MoeRowTransport prepares them on ROCm (DeepSeek-V4): rows at or past num_token_non_padded "
        "are padding, set to zero in hidden_states, and out takes hidden_states' rows, the padding "
        "ones zero, as the gather's input buffer holds them."
    ),
    op_type="moe_transport",
    axes={
        "n_rows": axis("Rows of this rank's slot of the gather, padding included.", 1, 65536, is_var=True),
        "hidden": axis("Elements per row.", 1, 65536),
    },
    attrs={},
    inputs={
        "hidden_states": TensorSpec(
            description=(
                "This rank's rows; its padding rows are set to zero in place, as sglang masks them "
                "before the gather. Padding rows may hold any bits (stale values of a replayed graph)."
            ),
            shape=("n_rows", "hidden"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16, DType.FLOAT32)),
            distribution=FiniteDistribution(),
            mutable=True,
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
        "num_token_non_padded": TensorSpec(
            description="Rows that are real tokens; later rows are padding (sglang passes a 0-d tensor).",
            shape=(1,),
            can_squeeze=True,
            dtype=DTypeSpec(any_of=(DType.INT32, DType.INT64)),
            distribution=IntRangeDistribution(lo=0, hi="n_rows"),
        ),
    },
    outputs={
        "out": TensorSpec(
            description="hidden_states' rows, the padding ones zero: the gather's input.",
            shape=("n_rows", "hidden"),
            dtype=DTypeSpec(same_as_tensor="hidden_states"),
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (DEEPSEEK_V4_DP_MOE_GATHER,)
