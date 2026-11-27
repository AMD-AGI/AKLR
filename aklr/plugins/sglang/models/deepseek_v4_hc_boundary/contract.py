# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4's mHC sublayer boundary at decode rows on HIP, as sglang's ``_hc_boundary_partials`` runs it
(``kernels/ops/layernorm/mhc_boundary_hip.py``): the pending hc_post of the sublayer output onto the residual
copies (with post), the collapse of the copies by the previous pre, and the split-K partials of the mixing
statistics the following norm launch reduces."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, FiniteDistribution, ToleranceBound
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseek_v4_hc_boundary.baseline import HcBoundaryBaseline
from aklr.plugins.sglang.models.deepseek_v4_hc_boundary.reference import HcBoundaryReference

__all__ = ["CONTRACTS", "DEEPSEEK_V4_HC_BOUNDARY", "DEEPSEEK_V4_HC_BOUNDARY_POST"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_F32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FINITE = FiniteDistribution()
_EXACT_BF16 = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)
_PARTIALS = ToleranceBound(absolute_tolerance=1e-3, relative_tolerance=1e-4)

_AXES = {
    "n_tokens": axis("Rows in this call.", 1, 65536, is_var=True),
    "hc_mult": axis("Residual copies: 4.", 4, 4),
    "hidden_size": axis("Hidden size, a multiple of the slice width.", 64, 65536, multiple_of=64),
    "n_mix": axis("Mixing coefficients per row: (2 + hc_mult) * hc_mult.", 24, 24),
    "n_slices": axis(
        "Partials per row: hidden columns of every copy in slices of hidden_size // n_slices.", 1, 4096
    ),
    "flat_size": derived_axis("Columns of hc_fn: hc_mult * hidden_size.", "hc_mult * hidden_size"),
}
_RESIDUAL = TensorSpec(
    description="The residual copies.",
    shape=("n_tokens", "hc_mult", "hidden_size"),
    dtype=_BF16,
    distribution=_FINITE,
)
_PRE = TensorSpec(
    description="The previous pre: each copy's collapse weight.",
    shape=("n_tokens", "hc_mult"),
    dtype=_F32,
    distribution=_FINITE,
)
_HC_FN = TensorSpec(
    description="The mixing projection, fp32.",
    shape=("n_mix", "flat_size"),
    dtype=_F32,
    distribution=_FINITE,
    lifetime="weight",
)
_Y = TensorSpec(
    description="The collapse sum_k pre[k] * copy_k (fp32 in copy order), bf16.",
    shape=("n_tokens", "hidden_size"),
    dtype=_BF16,
    error_bounds=_EXACT_BF16,
)
_PART_MIX = TensorSpec(
    description="Per slice s: sum over its columns of every copy of copy * hc_fn, fp32.",
    shape=("n_slices", "n_tokens", "n_mix"),
    dtype=_F32,
    error_bounds=_PARTIALS,
)
_PART_SQ = TensorSpec(
    description="Per slice s: sum over its columns of every copy of copy**2, fp32.",
    shape=("n_slices", "n_tokens"),
    dtype=_F32,
    error_bounds=_PARTIALS,
)

DEEPSEEK_V4_HC_BOUNDARY = OpContract(
    contract_id="deepseek_v4_hc_boundary@1.0",
    description=(
        "With no pending post: y = sum_k pre_prev[k] * residual[:, k] in fp32 (copy order), rounded to bf16; per "
        "slice s of n_slices over the hidden columns, part_mix[s] = sum_k residual[:, k, slice] hc_fn[:, k * H + "
        "slice]^T and part_sq[s] = sum_k sum residual[:, k, slice]**2, fp32."
    ),
    op_type="deepseek_v4_hc_boundary",
    axes=_AXES,
    constraints=("hidden_size % n_slices == 0",),
    inputs={"residual": _RESIDUAL, "pre_prev": _PRE, "hc_fn": _HC_FN},
    outputs={"y": _Y, "part_mix": _PART_MIX, "part_sq": _PART_SQ},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=HcBoundaryReference(),
    baseline=HcBoundaryBaseline(),
)

DEEPSEEK_V4_HC_BOUNDARY_POST = OpContract(
    contract_id="deepseek_v4_hc_boundary_post@1.0",
    description=(
        "With the pending post of x: residual_out[:, k] = post[k] * x + sum_j comb[j, k] * residual[:, j] (fp32, "
        "multiplies and adds in that order, none contracted), rounded to bf16; then y, part_mix and part_sq as "
        "deepseek_v4_hc_boundary's on residual_out."
    ),
    op_type="deepseek_v4_hc_boundary",
    axes=_AXES,
    constraints=("hidden_size % n_slices == 0",),
    inputs={
        "x": TensorSpec(
            description="The sublayer's output rows.",
            shape=("n_tokens", "hidden_size"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "residual": _RESIDUAL,
        "post": TensorSpec(
            description="The pending post per copy.",
            shape=("n_tokens", "hc_mult"),
            dtype=_F32,
            distribution=_FINITE,
        ),
        "comb": TensorSpec(
            description="The pending comb, [j, k] mixes copy j into copy k.",
            shape=("n_tokens", "hc_mult", "hc_mult"),
            dtype=_F32,
            distribution=_FINITE,
        ),
        "pre_prev": _PRE,
        "hc_fn": _HC_FN,
    },
    outputs={
        "residual_out": TensorSpec(
            description="The residual copies after the post, bf16.",
            shape=("n_tokens", "hc_mult", "hidden_size"),
            dtype=_BF16,
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
        "y": _Y,
        "part_mix": _PART_MIX,
        "part_sq": _PART_SQ,
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=HcBoundaryReference(),
    baseline=HcBoundaryBaseline(),
)

CONTRACTS = (DEEPSEEK_V4_HC_BOUNDARY, DEEPSEEK_V4_HC_BOUNDARY_POST)
