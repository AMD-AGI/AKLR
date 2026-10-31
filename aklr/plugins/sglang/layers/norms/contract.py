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

__all__ = ["CONTRACTS", "RMS_NORM", "RMS_NORM_ADD"]

_SAME_AS_X = DTypeSpec(same_as_tensor="x")
_ROWS = ("n_tokens", "hidden_size")
_AXES = {
    "n_tokens": axis("Rows, one per token.", 1, 65536, is_var=True),
    "hidden_size": axis("Normalized width.", 1, 65536),
}
_X = TensorSpec(
    description="Rows to normalize.",
    shape=_ROWS,
    dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
    distribution=FiniteDistribution(),
)
_WEIGHT = TensorSpec(
    description="Per-column scale.",
    shape=("hidden_size",),
    dtype=_SAME_AS_X,
    distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
    lifetime="weight",
)
_EPS = ScalarSpec(
    description="Added to the mean square.",
    dtype=DType.FLOAT32,
    interval=(2.0**-149, DTYPE_TRAITS["float32"].largest_finite_value),
    default=1e-6,
)
_NORMED = TensorSpec(
    description="Normalized rows times weight.",
    shape=_ROWS,
    dtype=_SAME_AS_X,
    error_bounds=OutputRoundingBound(),
)
_ROW_MAJOR_VARIANTS = LayoutVariantsSpec(by_name={"row_major": LayoutVariant()})

RMS_NORM = OpContract(
    contract_id="rms_norm@1.0",
    description="RMSNorm over the last axis, computed in float32.",
    op_type="norms",
    axes=_AXES,
    inputs={"x": _X, "weight": _WEIGHT, "eps": _EPS},
    outputs={"out": _NORMED},
    layout_variants=_ROW_MAJOR_VARIANTS,
)

RMS_NORM_ADD = OpContract(
    contract_id="rms_norm_add@1.0",
    description=(
        "RMSNorm over the last axis of x + residual, computed in float32 from the "
        "unrounded sum; the sum is returned too."
    ),
    op_type="norms",
    axes=_AXES,
    inputs={
        "x": _X,
        "residual": TensorSpec(
            description="Added to x before the norm.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            distribution=FiniteDistribution(),
        ),
        "weight": _WEIGHT,
        "eps": _EPS,
    },
    outputs={
        "normed": _NORMED,
        "residual_out": TensorSpec(
            description="x + residual.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            error_bounds=OutputRoundingBound(),
        ),
    },
    layout_variants=_ROW_MAJOR_VARIANTS,
)

CONTRACTS = (RMS_NORM, RMS_NORM_ADD)
