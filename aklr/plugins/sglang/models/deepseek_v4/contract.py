# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
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
    "DEEPSEEK_V4_HC_PRE_NORM",
]


_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))


# DeepSeek-V4's mHC sublayer boundary on ROCm. sglang (forward_hc_pre_from_prev_fused_boundary)
# defers each sublayer's hc_post to the next boundary and collapses the four streams with the pre
# mix of the boundary before. One boundary, t a token, j and k streams:
# 1. residual_out[t, k] = post_in[t, k] * x[t] + sum_j comb_in[t, j, k] * residual[t, j], rounded
#    to bf16 (the previous sublayer's deferred hc_post);
# 2. y[t] = sum_k pre_prev[t, k] * residual_out[t, k], rounded to bf16;
# 3. this boundary's mixes: the 24 logits residual_out[t].flatten() @ hc_fn.T times the rsqrt of
#    the mean square of residual_out[t] (+ rms_eps), then sigmoid / 2 * sigmoid / softmax and
#    Sinkhorn into pre / post / comb;
# 4. norm = RMSNorm(y) rounded to bf16 and, for the MXFP8 contract, its E4M3FN codes per 32 values
#    at the smallest power-of-two scale >= amax / 448 (E8M0).
# deepseek_v4_hc_pre_norm is the boundary without a pending hc_post (steps 2-4 on residual itself),
# as the DSpark draft stage starts each layer.

_F32 = DTypeSpec(any_of=(DType.FLOAT32,))

_FINITE = FiniteDistribution()

# sglang's own mHC kernel tests and aiter's: allclose(atol=1e-2, rtol=1e-2)
_HC_TOLERANCE = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)

_HC_AXES = {
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


_PENDING_POST_INPUTS = {
    "x": TensorSpec(
        description="The previous sublayer's output, whose hc_post was deferred to this boundary.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
}

_RESIDUAL = TensorSpec(
    description="The four residual streams (before the deferred hc_post, when there is one).",
    shape=("n_tokens", 4, "hidden_size"),
    dtype=_BF16,
    distribution=_FINITE,
)

_POST_MIX_INPUTS = {
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
}

_MIX_INPUTS = {
    "pre_prev": TensorSpec(
        description="The previous boundary's pre mix: weight of stream k in this sublayer's input.",
        shape=("n_tokens", 4),
        dtype=_F32,
        # sigmoid + eps: about [0.1, 1]
        distribution=LogUniformDistribution(log2_lo="-3.32", log2_hi="0"),
    ),
    "hc_fn": TensorSpec(
        description="Projects the flattened streams to 24 mix logits: pre, post, then comb.",
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
    "rms_eps": _epsilon("Added to the mean square of the streams (DeepSeek-V4: rms_norm_eps).", 1e-6),
    "norm_eps": _epsilon("Added to the mean square of y (the norm's variance_epsilon).", 1e-6),
    "hc_eps": _epsilon("Added to each pre mix after its sigmoid and in each Sinkhorn normalization.", 1e-6),
    "sinkhorn_iters": ScalarSpec(
        description="Sinkhorn normalizations of the combination mix.",
        dtype=DType.INT32,
        interval=(1, 2**31 - 1),
        default=20,
    ),
}

_RESIDUAL_OUT = {
    "residual_out": TensorSpec(
        description="The new streams: the deferred hc_post applied.",
        shape=("n_tokens", 4, "hidden_size"),
        dtype=_BF16,
        error_bounds=_HC_TOLERANCE,
    ),
}

_NORM_OUT = {
    "norm": TensorSpec(
        description="RMSNorm of y, the next sublayer's input.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        error_bounds=_HC_TOLERANCE,
    ),
}

_MXFP8_OUTS = {
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

_MIX_OUTS = {
    "pre": TensorSpec(
        description="This boundary's pre mix: sigmoid(...) + hc_eps.",
        shape=("n_tokens", 4),
        dtype=_F32,
        error_bounds=_HC_TOLERANCE,
    ),
    "post": TensorSpec(
        description="This boundary's post mix: 2 * sigmoid(...).",
        shape=("n_tokens", 4),
        dtype=_F32,
        error_bounds=_HC_TOLERANCE,
    ),
    "comb": TensorSpec(
        description="This boundary's combination mix after Sinkhorn, at [j, k].",
        shape=("n_tokens", 4, 4),
        dtype=_F32,
        error_bounds=_HC_TOLERANCE,
    ),
}

_ROW_MAJOR = LayoutVariantsSpec(by_name={"row_major": LayoutVariant()})

# a norm one bf16 ulp off (another summation order) moves an MXFP8 code across a rounding edge
# about once in a million values
_HC_COMPARE = CompareSpec(max_violation_fraction=1e-4)

DEEPSEEK_V4_HC_BOUNDARY_NORM = OpContract(
    contract_id="deepseek_v4_hc_boundary_norm@1.0",
    description=(
        "DeepSeek-V4 mHC sublayer boundary (the deferred hc_post, the collapse with the previous pre "
        "mix, this boundary's pre / post / comb mixes) and the RMSNorm of the collapsed row."
    ),
    op_type="deepseek_v4",
    axes=_HC_AXES,
    inputs={**_PENDING_POST_INPUTS, "residual": _RESIDUAL, **_POST_MIX_INPUTS, **_MIX_INPUTS},
    outputs={**_RESIDUAL_OUT, **_NORM_OUT, **_MIX_OUTS},
    layout_variants=_ROW_MAJOR,
    compare=_HC_COMPARE,
)

DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8 = OpContract(
    contract_id="deepseek_v4_hc_boundary_norm_mxfp8@1.0",
    description=(
        "deepseek_v4_hc_boundary_norm, and the norm also as the MXFP8 operand (E4M3FN codes and E8M0 "
        "scales per 32 values) of the projections that read it."
    ),
    op_type="deepseek_v4",
    axes=_HC_AXES,
    inputs={**_PENDING_POST_INPUTS, "residual": _RESIDUAL, **_POST_MIX_INPUTS, **_MIX_INPUTS},
    outputs={**_RESIDUAL_OUT, **_NORM_OUT, **_MXFP8_OUTS, **_MIX_OUTS},
    layout_variants=_ROW_MAJOR,
    compare=_HC_COMPARE,
)

DEEPSEEK_V4_HC_PRE_NORM = OpContract(
    contract_id="deepseek_v4_hc_pre_norm@1.0",
    description=(
        "DeepSeek-V4 mHC sublayer boundary without a deferred hc_post: the collapse of the streams "
        "with the previous pre mix, their pre / post / comb mixes, and the RMSNorm of the collapsed row."
    ),
    op_type="deepseek_v4",
    axes={n: a for n, a in _HC_AXES.items() if n != "n_groups"},
    inputs={"residual": _RESIDUAL, **_MIX_INPUTS},
    outputs={**_NORM_OUT, **_MIX_OUTS},
    layout_variants=_ROW_MAJOR,
)

CONTRACTS = (DEEPSEEK_V4_HC_BOUNDARY_NORM, DEEPSEEK_V4_HC_BOUNDARY_NORM_MXFP8, DEEPSEEK_V4_HC_PRE_NORM)
