# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LogUniformDistribution,
    OutputRoundingBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
)

__all__ = ["ADD_RMS_NORM", "CONTRACTS"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_SAME_AS_X = DTypeSpec(same_as_tensor="x")
_ROWS = ("n_tokens", "hidden_size")

ADD_RMS_NORM = OpContract(
    contract_id="add_rms_norm@1.0",
    description=(
        "residual_out = x + residual; normalized = residual_out * "
        "rsqrt(mean(residual_out**2) + epsilon) * weight, the mean over the hidden axis."
    ),
    op_type="rms_norm",
    axes={
        "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
        "hidden_size": axis("Width normalized.", 1, 8192),
    },
    inputs={
        "x": TensorSpec(
            description="The layer's output.",
            shape=_ROWS,
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=FiniteDistribution(),
        ),
        "residual": TensorSpec(
            description="The residual stream.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            distribution=FiniteDistribution(),
        ),
        "weight": TensorSpec(
            description="Scale of each normalized column.",
            shape=("hidden_size",),
            dtype=_SAME_AS_X,
            # about [0.5, 1.5]
            distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
            lifetime="weight",
        ),
        "epsilon": ScalarSpec(
            description="Added to the mean square.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, _FLOAT32_LARGEST),
            default=1e-6,
        ),
    },
    outputs={
        "normalized": TensorSpec(
            description="The normalized sum, scaled by weight.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            error_bounds=OutputRoundingBound(),
        ),
        "residual_out": TensorSpec(
            description="x + residual.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            error_bounds=OutputRoundingBound(),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (ADD_RMS_NORM,)
