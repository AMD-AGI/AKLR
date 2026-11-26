# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LogUniformDistribution,
    tolerance,
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
from aklr.plugins.aiter.provider.rms_norm.baseline import RmsnormBaseline, RmsnormWithAddBaseline
from aklr.plugins.aiter.provider.rms_norm.reference import RmsnormReference, RmsnormWithAddReference

__all__ = ["CONTRACTS", "RMSNORM", "RMSNORM_WITH_ADD"]

_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_SAME_AS_INPUT = DTypeSpec(same_as_tensor="input")
_ROWS = ("n_tokens", "hidden_size")
_AXES = {
    "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
    "hidden_size": axis("Width normalized.", 1, 8192),
}
_INPUT_DTYPE = DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16))
_WEIGHT = TensorSpec(
    description="Scale of each normalized column.",
    shape=("hidden_size",),
    dtype=_SAME_AS_INPUT,
    # about [0.5, 1.5]
    distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
    lifetime="weight",
)
_EPSILON = ScalarSpec(
    description="Added to the mean square.",
    dtype=DType.FLOAT32,
    interval=(2.0**-149, _FLOAT32_LARGEST),
    default=1e-6,
)
# aiter's variants of the op that AKLR's kernels do not compute: only their off value is in the contract,
# so a kernel takes no such param
_USE_MODEL_SENSITIVE_RMSNORM = attr(
    "aiter's model-sensitive RMSNorm; 0 is the plain one.",
    0,
    default=0,
    param_name="use_model_sensitive_rmsnorm",
    optional=True,
)


def _output(description: str) -> TensorSpec:
    return TensorSpec(
        description=description,
        shape=_ROWS,
        dtype=_SAME_AS_INPUT,
        error_bounds=tolerance(*_INPUT_DTYPE.any_of),
    )


RMSNORM_WITH_ADD = OpContract(
    contract_id="rmsnorm_with_add@1.0",
    description=(
        "aiter's rmsnorm_with_add: residual_out = input + residual_in; out = residual_out * "
        "rsqrt(mean(residual_out**2) + epsilon) * weight, the mean over the hidden axis."
    ),
    op_type="rms_norm",
    axes=_AXES,
    inputs={
        "input": TensorSpec(
            description="The layer's output.",
            shape=_ROWS,
            dtype=_INPUT_DTYPE,
            distribution=FiniteDistribution(),
        ),
        "residual_in": TensorSpec(
            description="The residual stream.",
            shape=_ROWS,
            dtype=_SAME_AS_INPUT,
            distribution=FiniteDistribution(),
        ),
        "weight": _WEIGHT,
        "epsilon": _EPSILON,
    },
    outputs={
        "out": _output("The normalized sum, scaled by weight."),
        "residual_out": _output("input + residual_in."),
    },
    attrs={
        "gemma_norm": attr(
            "Gemma's RMSNorm, scaling by 1 + weight; false is the plain one.",
            False,
            default=False,
            param_name="gemma_norm",
            optional=True,
        ),
        "use_model_sensitive_rmsnorm": _USE_MODEL_SENSITIVE_RMSNORM,
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=RmsnormWithAddReference(),
    baseline=RmsnormWithAddBaseline(),
)

RMSNORM = OpContract(
    contract_id="rmsnorm@1.0",
    description=(
        "aiter's rmsnorm: normalized = input * rsqrt(mean(input**2) + epsilon) * weight, "
        "the mean over the hidden axis."
    ),
    op_type="rms_norm",
    axes=_AXES,
    inputs={
        "input": TensorSpec(
            description="Rows to normalize.",
            shape=_ROWS,
            dtype=_INPUT_DTYPE,
            distribution=FiniteDistribution(),
        ),
        "weight": _WEIGHT,
        "epsilon": _EPSILON,
    },
    outputs={"normalized": _output("The normalized rows, scaled by weight.")},
    attrs={"use_model_sensitive_rmsnorm": _USE_MODEL_SENSITIVE_RMSNORM},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=RmsnormReference(),
    baseline=RmsnormBaseline(),
)

CONTRACTS = (RMSNORM, RMSNORM_WITH_ADD)
