# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One mHC sublayer boundary of DeepSeek-V4 on ROCm, with the RMSNorm that follows it.

sglang (``forward_hc_pre_from_prev_fused_boundary``) defers each sublayer's hc_post to the next
boundary and collapses the four streams with the pre mix of the boundary before. One boundary:

1. ``residual_out[t, k] = post_in[t, k] * x[t] + sum_j comb_in[t, j, k] * residual[t, j]``, rounded
   to bf16 (the previous sublayer's deferred hc_post);
2. ``y[t] = sum_k pre_prev[t, k] * residual_out[t, k]``, rounded to bf16;
3. the mix logits ``residual_out[t].flatten() @ hc_fn.T`` scaled by the rsqrt of the mean square of
   ``residual_out[t]``, then sigmoid / Sinkhorn into ``pre`` / ``post`` / ``comb`` of this boundary;
4. ``norm = rmsnorm(y, norm_weight)`` rounded to bf16 and, for the MXFP8 contract, quantized per 32
   values to E4M3FN at the smallest power-of-two scale >= amax / 448 (E8M0 exponent bytes).
"""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    OutputRoundingBound,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)

__all__ = [
    "CONTRACTS",
    "DEEPSEEK_V4_HC_BOUNDARY_NORM",
    "DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8",
]

_F32 = DTypeSpec(any_of=(DType.FLOAT32,))
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_FINITE = FiniteDistribution()
# sglang's own mHC kernels and aiter's mHC tests: allclose(atol=1e-2, rtol=1e-2)
_TOLERANCE = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)

_AXES = {
    "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
    "hidden_size": axis("Hidden size of one stream.", 64, 65536, multiple_of=64),
    "projection_size": derived_axis("Mix projection input: the four streams.", "4 * hidden_size"),
    "n_groups": derived_axis("MXFP8 groups of 32 per row.", "hidden_size // 32"),
}


def _epsilon(description: str, default: float) -> ScalarSpec:
    return ScalarSpec(
        description=description,
        dtype=DType.FLOAT32,
        interval=(2.0**-149, _FLOAT32_LARGEST),
        default=default,
    )


_INPUTS = {
    "x": TensorSpec(
        description="The previous sublayer's output, whose hc_post was deferred to this boundary.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "residual": TensorSpec(
        description="The four residual streams before that hc_post.",
        shape=("n_tokens", 4, "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "post_in": TensorSpec(
        description="The previous sublayer's post mix: weight of x in new stream k.",
        shape=("n_tokens", 4),
        dtype=_F32,
        # 2 * sigmoid: about [0.2, 1.9]
        distribution=LogUniformDistribution(log2_lo="-2.32", log2_hi="0.9"),
    ),
    "comb_in": TensorSpec(
        description="The previous sublayer's combination mix: weight of stream j in new stream k, at [j, k].",
        shape=("n_tokens", 4, 4),
        dtype=_F32,
        # Sinkhorn output: in [1/16, 1]
        distribution=LogUniformDistribution(log2_lo="-4", log2_hi="0"),
    ),
    "pre_prev": TensorSpec(
        description="The previous boundary's pre mix: weight of new stream k in this sublayer's input.",
        shape=("n_tokens", 4),
        dtype=_F32,
        # sigmoid + eps: about [0.1, 1]
        distribution=LogUniformDistribution(log2_lo="-3.32", log2_hi="0"),
    ),
    "hc_fn": TensorSpec(
        description="Projects the flattened new streams to 24 mix logits: pre, post, then comb.",
        shape=(24, "projection_size"),
        dtype=_F32,
        distribution=LecunNormalDistribution(fan_in="projection_size"),
        lifetime="weight",
    ),
    "hc_scale": TensorSpec(
        description="Scale of the pre, post and comb logits.",
        shape=(3,),
        dtype=_F32,
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="1"),
        lifetime="weight",
    ),
    "hc_base": TensorSpec(
        description="Bias of the 24 mix logits.",
        shape=(24,),
        dtype=_F32,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "norm_weight": TensorSpec(
        description="Weight of the RMSNorm of y.",
        shape=("hidden_size",),
        dtype=_BF16,
        # about [0.5, 1.5]
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
        lifetime="weight",
    ),
    "rms_eps": _epsilon("Added to the mean square of the new streams (DeepSeek-V4: rms_norm_eps).", 1e-6),
    "norm_eps": _epsilon("Added to the mean square of y (the norm's variance_epsilon).", 1e-6),
    "hc_eps": _epsilon("Added to each pre mix after its sigmoid and in each Sinkhorn normalization.", 1e-6),
    "sinkhorn_iters": ScalarSpec(
        description="Sinkhorn normalizations of the combination mix.",
        dtype=DType.INT32,
        interval=(1, 2**31 - 1),
        default=20,
    ),
}

_BASE_OUTPUTS = {
    "residual_out": TensorSpec(
        description="The new streams: the deferred hc_post applied, in residual's dtype.",
        shape=("n_tokens", 4, "hidden_size"),
        dtype=_BF16,
        error_bounds=_TOLERANCE,
    ),
    "norm": TensorSpec(
        description="RMSNorm of y, the next sublayer's input.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        error_bounds=_TOLERANCE,
    ),
}
_MIX_OUTPUTS = {
    "pre": TensorSpec(
        description="This boundary's pre mix: sigmoid(...) + hc_eps.",
        shape=("n_tokens", 4),
        dtype=_F32,
        error_bounds=_TOLERANCE,
    ),
    "post": TensorSpec(
        description="This boundary's post mix: 2 * sigmoid(...).",
        shape=("n_tokens", 4),
        dtype=_F32,
        error_bounds=_TOLERANCE,
    ),
    "comb": TensorSpec(
        description="This boundary's combination mix after Sinkhorn, at [j, k].",
        shape=("n_tokens", 4, 4),
        dtype=_F32,
        error_bounds=_TOLERANCE,
    ),
}
_MXFP8_OUTPUTS = {
    "quantized": TensorSpec(
        description="norm as E4M3FN codes of its 32-value groups.",
        shape=("n_tokens", "hidden_size"),
        dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
        error_bounds=OutputRoundingBound(),
    ),
    "scales": TensorSpec(
        description="E8M0 scale of each group at [token, group]: its exponent byte clamped to [1, 254].",
        shape=("n_tokens", "n_groups"),
        dtype=DTypeSpec(any_of=(DType.FLOAT8_E8M0,)),
        error_bounds=OutputRoundingBound(),
    ),
}
_ROW_MAJOR = LayoutVariantsSpec(by_name={"row_major": LayoutVariant()})
# a norm one bf16 ulp off (another summation order) moves an MXFP8 code across a rounding edge
# about once in a million values
_COMPARE = CompareSpec(max_violation_fraction=1e-4, determinism=DeterminismLevel.BOUNDED)

DEEPSEEK_V4_HC_BOUNDARY_NORM = OpContract(
    contract_id="deepseek_v4_hc_boundary_norm@1.0",
    description=(
        "DeepSeek-V4 mHC sublayer boundary (deferred hc_post, collapse with the previous pre, mix "
        "statistics, reduce and Sinkhorn) and the RMSNorm of the collapsed row."
    ),
    op_type="deepseek_v4",
    axes=_AXES,
    inputs=_INPUTS,
    outputs={**_BASE_OUTPUTS, **_MIX_OUTPUTS},
    layout_variants=_ROW_MAJOR,
    compare=_COMPARE,
)

DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8 = OpContract(
    contract_id="deepseek_v4_hc_boundary_norm_mxfp8@1.0",
    description=(
        "deepseek_v4_hc_boundary_norm, and the norm also as the MXFP8 operand (E4M3FN codes and E8M0 "
        "scales per 32 values) of the projections that read it."
    ),
    op_type="deepseek_v4",
    axes=_AXES,
    inputs=_INPUTS,
    outputs={**_BASE_OUTPUTS, **_MXFP8_OUTPUTS, **_MIX_OUTPUTS},
    layout_variants=_ROW_MAJOR,
    compare=_COMPARE,
)

CONTRACTS = (DEEPSEEK_V4_HC_BOUNDARY_NORM, DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8)
