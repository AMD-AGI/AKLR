# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's mHC sublayer boundary without a pending post, together with the norm that follows it.

sglang runs this as two launches (``kernels/ops/layernorm/mhc_boundary_hip.py``): ``_hc_boundary_partials``
collapses the residual copies under the previous hyper-connection's ``pre`` and takes split-K partials of the
mixing statistics, then ``rmsnorm_with_sinkhorn`` reduces those partials, runs the Sinkhorn normalization into
this boundary's ``pre`` / ``post`` / ``comb`` and RMS-norms the collapse. Both launches are below the GPU's
per-kernel floor at decode rows, so at TP4 c1 they cost 15% of a verify step. This contract is the pair.
"""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DeterminismLevel,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseek_v4_hc_pre_norm.baseline import (
    HcPreNormBaseline,
    HcPreNormMxfp8Baseline,
    HcPreNormPostBaseline,
    HcPreNormPostMxfp8Baseline,
)
from aklr.plugins.sglang.models.deepseek_v4_hc_pre_norm.reference import HcPreNormReference

__all__ = [
    "CONTRACTS",
    "DEEPSEEK_V4_HC_PRE_NORM",
    "DEEPSEEK_V4_HC_PRE_NORM_MXFP8",
    "DEEPSEEK_V4_HC_PRE_NORM_POST",
    "DEEPSEEK_V4_HC_PRE_NORM_POST_MXFP8",
]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_F32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_U8 = DTypeSpec(any_of=(DType.UINT8,))
_FINITE = FiniteDistribution()
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
# the mixes are sigmoids and a Sinkhorn-normalized softmax: aiter's mHC bound
_MIX_TOLERANCE = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)
_BF16_TOLERANCE = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)

_AXES = {
    "n_tokens": axis("Rows in this call.", 1, 65536, is_var=True),
    "hc_mult": axis("Residual copies: 4.", 4, 4),
    "hidden_size": axis("Hidden size of one copy, a multiple of 64.", 64, 65536, multiple_of=64),
    "n_mix": axis("Mixing coefficients per row: (2 + hc_mult) * hc_mult.", 24, 24),
    "flat_size": derived_axis("Columns of hc_fn: hc_mult * hidden_size.", "hc_mult * hidden_size"),
    "n_blocks": derived_axis("Scales of a normed row: one per 32 columns.", "hidden_size // 32"),
}


def _epsilon(description: str) -> ScalarSpec:
    return ScalarSpec(
        description=description,
        dtype=DType.FLOAT32,
        interval=(2.0**-149, _FLOAT32_LARGEST),
        default=1e-6,
    )


_INPUTS = {
    "residual": TensorSpec(
        description="The residual copies.",
        shape=("n_tokens", "hc_mult", "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "pre_prev": TensorSpec(
        description="The previous hyper-connection's pre: each copy's weight in the collapse.",
        shape=("n_tokens", "hc_mult"),
        dtype=_F32,
        # about [0.1, 1.1]: a sigmoid plus hc_eps
        distribution=LogUniformDistribution(log2_lo="-3.32", log2_hi="0.14"),
    ),
    "hc_fn": TensorSpec(
        description="The mixing projection of the flattened copies, fp32.",
        shape=("n_mix", "flat_size"),
        dtype=_F32,
        distribution=LecunNormalDistribution(fan_in="flat_size"),
        lifetime="weight",
    ),
    "hc_scale": TensorSpec(
        description="Scale of the pre, post and comb logits, in that order.",
        shape=(3,),
        dtype=_F32,
        distribution=LogUniformDistribution(log2_lo="0", log2_hi="0"),
        lifetime="weight",
    ),
    "hc_base": TensorSpec(
        description="Bias of the n_mix logits: hc_mult pre, hc_mult post, then hc_mult**2 comb row-major.",
        shape=("n_mix",),
        dtype=_F32,
        distribution=_FINITE,
        lifetime="weight",
    ),
    "norm_weight": TensorSpec(
        description="Weight of the RMS norm the collapse feeds.",
        shape=("hidden_size",),
        dtype=_BF16,
        # about [0.5, 1.5]
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
        lifetime="weight",
    ),
    "rms_eps": _epsilon("Added to the mean square of the flattened copies, before the mixing logits."),
    "hc_eps": _epsilon("Added to each pre after its sigmoid and in each Sinkhorn normalization."),
    "norm_eps": _epsilon("Added to the mean square of the collapse, in its RMS norm."),
    "post_multiplier": ScalarSpec(
        description="Multiplies each post after its sigmoid.",
        dtype=DType.FLOAT32,
        interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
        default=2.0,
    ),
    "sinkhorn_iters": ScalarSpec(
        description="Sinkhorn normalizations of the comb mix.",
        dtype=DType.INT32,
        interval=(1, 2**31 - 1),
        default=20,
    ),
}

_OUTPUTS = {
    "y": TensorSpec(
        description="The collapse sum_k pre_prev[k] * copy_k, summed in copy order in fp32, bf16.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        error_bounds=_BF16_TOLERANCE,
    ),
    "out_norm": TensorSpec(
        description="RMSNorm(y) * norm_weight, bf16.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        error_bounds=_BF16_TOLERANCE,
    ),
    "pre": TensorSpec(
        description="This boundary's pre: sigmoid of its logits plus hc_eps.",
        shape=("n_tokens", "hc_mult"),
        dtype=_F32,
        error_bounds=_MIX_TOLERANCE,
    ),
    "post": TensorSpec(
        description="This boundary's post: post_multiplier times the sigmoid of its logits.",
        shape=("n_tokens", "hc_mult"),
        dtype=_F32,
        error_bounds=_MIX_TOLERANCE,
    ),
    "comb": TensorSpec(
        description="This boundary's comb, [j, k] mixing copy j into copy k after the Sinkhorn iterations.",
        shape=("n_tokens", "hc_mult", "hc_mult"),
        dtype=_F32,
        error_bounds=_MIX_TOLERANCE,
    ),
}

_POST_INPUTS = {
    "x": TensorSpec(
        description="The sublayer's output rows, the pending post's.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE,
    ),
    "pending_post": TensorSpec(
        description="The previous boundary's post, one weight of x per copy.",
        shape=("n_tokens", "hc_mult"),
        dtype=_F32,
        # about [0.1, 2]: twice a sigmoid
        distribution=LogUniformDistribution(log2_lo="-3.32", log2_hi="1"),
    ),
    "pending_comb": TensorSpec(
        description="The previous boundary's comb, [j, k] mixing copy j into copy k.",
        shape=("n_tokens", "hc_mult", "hc_mult"),
        dtype=_F32,
        # about [1/16, 1]: a Sinkhorn-normalized softmax
        distribution=LogUniformDistribution(log2_lo="-4", log2_hi="0"),
    ),
}
_RESIDUAL_OUT = TensorSpec(
    description="The residual copies after the pending post, bf16.",
    shape=("n_tokens", "hc_mult", "hidden_size"),
    dtype=_BF16,
    error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
)
_QUANTIZED = {
    "quantized": TensorSpec(
        description="Each normed element over its block's scale, fp8 e4m3.",
        shape=("n_tokens", "hidden_size"),
        dtype=_FP8,
        error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
    ),
    "scales": TensorSpec(
        description="The ue8m0 exponent byte of each 32 columns' scale: scale = 2**(byte - 127).",
        shape=("n_tokens", "n_blocks"),
        dtype=_U8,
        error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
    ),
}

_DESCRIPTION = (
    "With flat = residual flattened over the copies, rsqrt = 1 / sqrt(mean(flat**2) + rms_eps) and "
    "logits = (flat @ hc_fn^T) * rsqrt * hc_scale + hc_base, split hc_mult / hc_mult / hc_mult**2: "
    "pre = sigmoid(pre logits) + hc_eps, post = post_multiplier * sigmoid(post logits), and comb the "
    "row softmax of the comb logits plus hc_eps, then sinkhorn_iters alternating column and row "
    "normalizations, each denominator taking hc_eps. Independently, y = sum_k pre_prev[k] * copy_k "
    "(fp32, copy order) rounded to bf16, and out_norm = y / sqrt(mean(y**2) + norm_eps) * norm_weight."
)

DEEPSEEK_V4_HC_PRE_NORM = OpContract(
    contract_id="deepseek_v4_hc_pre_norm@1.0",
    description=_DESCRIPTION,
    op_type="deepseek_v4_hc_pre_norm",
    axes=_AXES,
    inputs=_INPUTS,
    outputs=_OUTPUTS,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=HcPreNormReference(),
    baseline=HcPreNormBaseline(),
)

_MXFP8_DESCRIPTION = (
    " out_norm is also quantized per 32 columns to fp8 e4m3 by the smallest power of two >= their amax "
    "/ 448 (amax floored at 1e-10, the exponent clamped to [1, 254]): quantized holds out_norm / scale "
    "clamped to +-448, and scales the scale's ue8m0 exponent byte."
)

DEEPSEEK_V4_HC_PRE_NORM_MXFP8 = OpContract(
    contract_id="deepseek_v4_hc_pre_norm_mxfp8@1.0",
    description=_DESCRIPTION + _MXFP8_DESCRIPTION,
    op_type="deepseek_v4_hc_pre_norm",
    axes=_AXES,
    inputs=_INPUTS,
    outputs={**_OUTPUTS, **_QUANTIZED},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=HcPreNormReference(),
    baseline=HcPreNormMxfp8Baseline(),
)

_POST_DESCRIPTION = (
    "residual_out[:, k] = pending_post[k] * x + sum_j pending_comb[j, k] * residual[:, j] (float32, "
    "multiplies and adds in that order, none contracted), rounded to bf16; then "
    + _DESCRIPTION[0].lower()
    + _DESCRIPTION[1:]
    .replace("residual flattened", "residual_out flattened")
    .replace("pre_prev[k] * copy_k", "pre_prev[k] * residual_out[:, k]")
)

DEEPSEEK_V4_HC_PRE_NORM_POST = OpContract(
    contract_id="deepseek_v4_hc_pre_norm_post@1.0",
    description=_POST_DESCRIPTION,
    op_type="deepseek_v4_hc_pre_norm",
    axes=_AXES,
    inputs={**_POST_INPUTS, **_INPUTS},
    outputs={"residual_out": _RESIDUAL_OUT, **_OUTPUTS},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=HcPreNormReference(),
    baseline=HcPreNormPostBaseline(),
)

DEEPSEEK_V4_HC_PRE_NORM_POST_MXFP8 = OpContract(
    contract_id="deepseek_v4_hc_pre_norm_post_mxfp8@1.0",
    description=_POST_DESCRIPTION + _MXFP8_DESCRIPTION,
    op_type="deepseek_v4_hc_pre_norm",
    axes=_AXES,
    inputs={**_POST_INPUTS, **_INPUTS},
    outputs={"residual_out": _RESIDUAL_OUT, **_OUTPUTS, **_QUANTIZED},
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
    reference=HcPreNormReference(),
    baseline=HcPreNormPostMxfp8Baseline(),
)

CONTRACTS = (
    DEEPSEEK_V4_HC_PRE_NORM,
    DEEPSEEK_V4_HC_PRE_NORM_MXFP8,
    DEEPSEEK_V4_HC_PRE_NORM_POST,
    DEEPSEEK_V4_HC_PRE_NORM_POST_MXFP8,
)
