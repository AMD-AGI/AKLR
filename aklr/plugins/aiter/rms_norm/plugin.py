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
from aklr.plugins import Plugin
from aklr.plugins.aiter._candidate import offer, served

__all__ = ["ADD_RMS_NORM", "RMS_NORM", "add_rms_norm", "declare", "family", "rms_norm"]


_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_SAME_AS_X = DTypeSpec(same_as_tensor="x")
_ROWS = ("n_tokens", "hidden_size")
_AXES = {
    "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
    "hidden_size": axis("Width normalized.", 1, 8192),
}
_X_DTYPE = DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16))
_WEIGHT = TensorSpec(
    description="Scale of each normalized column.",
    shape=("hidden_size",),
    dtype=_SAME_AS_X,
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


def _output(description: str) -> TensorSpec:
    return TensorSpec(
        description=description, shape=_ROWS, dtype=_SAME_AS_X, error_bounds=OutputRoundingBound()
    )


ADD_RMS_NORM = OpContract(
    contract_id="add_rms_norm@1.0",
    description=(
        "residual_out = x + residual; normalized = residual_out * "
        "rsqrt(mean(residual_out**2) + epsilon) * weight, the mean over the hidden axis."
    ),
    op_type="rms_norm",
    axes=_AXES,
    inputs={
        "x": TensorSpec(
            description="The layer's output.", shape=_ROWS, dtype=_X_DTYPE, distribution=FiniteDistribution()
        ),
        "residual": TensorSpec(
            description="The residual stream.",
            shape=_ROWS,
            dtype=_SAME_AS_X,
            distribution=FiniteDistribution(),
        ),
        "weight": _WEIGHT,
        "epsilon": _EPSILON,
    },
    outputs={
        "normalized": _output("The normalized sum, scaled by weight."),
        "residual_out": _output("x + residual."),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

RMS_NORM = OpContract(
    contract_id="rms_norm@1.0",
    description="normalized = x * rsqrt(mean(x**2) + epsilon) * weight, the mean over the hidden axis.",
    op_type="rms_norm",
    axes=_AXES,
    inputs={
        "x": TensorSpec(
            description="Rows to normalize.", shape=_ROWS, dtype=_X_DTYPE, distribution=FiniteDistribution()
        ),
        "weight": _WEIGHT,
        "epsilon": _EPSILON,
    },
    outputs={"normalized": _output("The normalized rows, scaled by weight.")},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)


family = Plugin("aiter")
family.define(ADD_RMS_NORM)
family.define(RMS_NORM)
family.build()


@family.op
def add_rms_norm(x, residual, weight, epsilon, normalized, residual_out): ...


@family.op
def rms_norm(x, weight, epsilon): ...


def _select_rmsnorm(args, kwargs):
    x, weight, epsilon, use_model_sensitive_rmsnorm = args
    if use_model_sensitive_rmsnorm:
        return None
    return offer(rms_norm.serving(x, weight, epsilon), _run_rmsnorm)


def _run_rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm):
    return served(rms_norm(input, weight, epsilon))


def _select_rmsnorm_with_add(args, kwargs):
    out, x, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm = args
    if gemma_norm or use_model_sensitive_rmsnorm:
        return None
    return offer(
        add_rms_norm.serving(x, residual_in, weight, epsilon, out, residual_out),
        _run_rmsnorm_with_add,
    )


def _run_rmsnorm_with_add(
    out, input, residual_in, residual_out, weight, epsilon, gemma_norm, use_model_sensitive_rmsnorm
):
    served(add_rms_norm(input, residual_in, weight, epsilon, out, residual_out))


def declare(plugin: Plugin) -> None:
    plugin.provide("rmsnorm", version=2, select=_select_rmsnorm)
    plugin.provide("rmsnorm_with_add", version=1, select=_select_rmsnorm_with_add)
