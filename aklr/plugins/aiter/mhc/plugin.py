# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DeterminismLevel,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.plugins import Plugin
from aklr.plugins.aiter._candidate import offer, served

__all__ = [
    "MHC_FUSED_POST_PRE",
    "MHC_POST",
    "MHC_PRE",
    "declare",
    "family",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
]


_F32 = DTypeSpec(any_of=(DType.FLOAT32,))
_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_RESIDUAL = DTypeSpec(same_as_tensor="residual")
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value

_FINITE_DISTRIBUTION = FiniteDistribution()
# about [0.1, 0.9]
_FRACTION_DISTRIBUTION = LogUniformDistribution(log2_lo="-3.32", log2_hi="-0.15")
# as aiter's mHC tests: allclose(atol=1e-2, rtol=1e-2)
_TOLERANCE_BOUND = ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)

_POST_AXES = {
    "n_tokens": axis("Tokens.", 1, 65536, is_var=True),
    "hidden_size": axis("Hidden size of one stream.", 1, 65536),
}
_PRE_AXES = {
    **_POST_AXES,
    "projection_size": derived_axis("Projection input: four streams.", "4 * hidden_size"),
}
_ROW_MAJOR_VARIANTS = LayoutVariantsSpec(by_name={"row_major": LayoutVariant()})
_NORM_ATTRS = {"has_norm_weight": attr("Whether norm_weight is passed.", False, True, default=False)}


def _epsilon(description: str) -> ScalarSpec:
    return ScalarSpec(
        description=description,
        dtype=DType.FLOAT32,
        interval=(2.0**-149, _FLOAT32_LARGEST),
        default=1e-6,
    )


_RESIDUAL = TensorSpec(
    description="Four residual streams.",
    shape=("n_tokens", 4, "hidden_size"),
    dtype=_BF16,
    distribution=_FINITE_DISTRIBUTION,
)
_POST_INPUTS = {
    "x": TensorSpec(
        description="The layer's output.",
        shape=("n_tokens", "hidden_size"),
        dtype=_BF16,
        distribution=_FINITE_DISTRIBUTION,
    ),
    "residual": _RESIDUAL,
    "post_mix": TensorSpec(
        description="Weight of x in each new stream.",
        shape=("n_tokens", 4, 1),
        dtype=_F32,
        can_squeeze=True,
        distribution=_FRACTION_DISTRIBUTION,
    ),
    "comb_mix": TensorSpec(
        description="Weight of old stream i in new stream j, at [i, j].",
        shape=("n_tokens", 4, 4),
        dtype=_F32,
        # in [1/16, 1]
        distribution=LogUniformDistribution(log2_lo="-4", log2_hi="0"),
    ),
}
_PRE_INPUTS = {
    "proj_weight": TensorSpec(
        description="Projects the flattened streams to 24 mix logits.",
        shape=(24, "projection_size"),
        dtype=_F32,
        distribution=LecunNormalDistribution(fan_in="projection_size"),
        lifetime="weight",
    ),
    "mix_scale": TensorSpec(
        description="Scale of the pre, post and combination logits.",
        shape=(3,),
        dtype=_F32,
        distribution=LogUniformDistribution(log2_lo="0", log2_hi="0"),
        lifetime="weight",
    ),
    "mix_bias": TensorSpec(
        description="Bias of the 24 mix logits.",
        shape=(24,),
        dtype=_F32,
        distribution=_FINITE_DISTRIBUTION,
        lifetime="weight",
    ),
    "rms_eps": _epsilon("Added to the mean square of the streams."),
    "pre_eps": _epsilon("Added to each pre mix after its sigmoid."),
    "sinkhorn_eps": _epsilon("Added in each Sinkhorn normalization."),
    "post_multiplier": ScalarSpec(
        description="Multiplies each post mix after its sigmoid.",
        dtype=DType.FLOAT32,
        interval=(-_FLOAT32_LARGEST, _FLOAT32_LARGEST),
        default=1.0,
    ),
    "sinkhorn_iters": ScalarSpec(
        description="Sinkhorn normalizations of the combination mix.",
        dtype=DType.INT32,
        interval=(1, 2**31 - 1),
        default=20,
    ),
    "norm_weight": TensorSpec(
        description="Weight of the rms norm of layer_input.",
        shape=("hidden_size",),
        dtype=_BF16,
        present_when="has_norm_weight",
        # about [0.5, 1.5]
        distribution=LogUniformDistribution(log2_lo="-1", log2_hi="0.58"),
        lifetime="weight",
    ),
    # always present: aiter passes norm_eps even without norm_weight
    "norm_eps": _epsilon("Added to the mean square of layer_input."),
}

_LAYER_INPUT = TensorSpec(
    description="The streams summed under the pre mix.",
    shape=("n_tokens", "hidden_size"),
    dtype=_SAME_AS_RESIDUAL,
    error_bounds=_TOLERANCE_BOUND,
)
_NEXT_STREAMS = TensorSpec(
    description="The new streams.",
    shape=("n_tokens", 4, "hidden_size"),
    dtype=_SAME_AS_RESIDUAL,
    error_bounds=_TOLERANCE_BOUND,
)
_POST_MIX = TensorSpec(
    description="Post mix of the next mHC post.",
    shape=("n_tokens", 4, 1),
    dtype=_F32,
    error_bounds=_TOLERANCE_BOUND,
)
_COMBINATION_MIX = TensorSpec(
    description="Combination mix of the next mHC post.",
    shape=("n_tokens", 4, 4),
    dtype=_F32,
    error_bounds=_TOLERANCE_BOUND,
)

MHC_PRE = OpContract(
    contract_id="mhc_pre@1.0",
    description=(
        "Mix four residual streams into an optionally rms-normed layer input; derive the post mixes."
    ),
    op_type="mhc",
    axes=_PRE_AXES,
    attrs=_NORM_ATTRS,
    inputs={"residual": _RESIDUAL, **_PRE_INPUTS},
    outputs={
        "post_mix": _POST_MIX,
        "comb_mix": _COMBINATION_MIX,
        "layer_input": _LAYER_INPUT,
    },
    layout_variants=_ROW_MAJOR_VARIANTS,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

MHC_POST = OpContract(
    contract_id="mhc_post@1.0",
    description="Mix four residual streams and a layer output into four new streams.",
    op_type="mhc",
    axes=_POST_AXES,
    inputs=_POST_INPUTS,
    outputs={"out": _NEXT_STREAMS},
    layout_variants=_ROW_MAJOR_VARIANTS,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)

MHC_FUSED_POST_PRE = OpContract(
    contract_id="mhc_fused_post_pre@1.0",
    description=(
        "mHC post, then the next mHC pre on the new streams rounded to residual's "
        "dtype; aiter's return order (next_post_mix, next_comb_mix, layer_input, "
        "next_residual), not sglang's residual-first order."
    ),
    op_type="mhc",
    axes=_PRE_AXES,
    attrs=_NORM_ATTRS,
    inputs={
        **_POST_INPUTS,
        # main's (tokens, streams): its flydsl kernel aborts on (T, 4, 1)
        "post_mix": _POST_INPUTS["post_mix"].model_copy(
            update={"shape": ("n_tokens", 4), "can_squeeze": False}
        ),
        **_PRE_INPUTS,
    },
    outputs={
        "next_post_mix": _POST_MIX,
        "next_comb_mix": _COMBINATION_MIX,
        "layer_input": _LAYER_INPUT,
        "next_residual": _NEXT_STREAMS,
    },
    layout_variants=_ROW_MAJOR_VARIANTS,
    compare=CompareSpec(determinism=DeterminismLevel.BOUNDED),
)


family = Plugin("aiter")
family.define(MHC_PRE)
family.define(MHC_POST)
family.define(MHC_FUSED_POST_PRE)
family.build()


@family.op
def mhc_pre(
    residual,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
): ...


@family.op
def mhc_post(x, residual, post_mix, comb_mix, *, out=None): ...


@family.op
def mhc_fused_post_pre(
    x,
    residual,
    post_mix,
    comb_mix,
    proj_weight,
    mix_scale,
    mix_bias,
    *,
    rms_eps=1e-6,
    pre_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_multiplier=1.0,
    sinkhorn_iters=20,
    norm_weight=None,
    norm_eps=1e-6,
): ...


def _kwargs(rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps):
    return {
        "rms_eps": rms_eps,
        "pre_eps": hc_pre_eps,
        "sinkhorn_eps": hc_sinkhorn_eps,
        "post_multiplier": hc_post_mult_value,
        "sinkhorn_iters": sinkhorn_repeat,
        "norm_weight": norm_weight,
        "norm_eps": norm_eps,
    }


def _select_mhc_pre(args, kwargs):
    residual, fn, hc_scale, hc_base, *scalars, _large_m_splitk, w_preshuffle_bf16, res_preshuffle = args
    # the shuffled fn or residual layouts are not AKLR's; large_m_splitk only picks aiter's schedule
    if w_preshuffle_bf16 or res_preshuffle:
        return None
    return offer(mhc_pre.serving(residual, fn, hc_scale, hc_base, **_kwargs(*scalars)), _run_mhc_pre)


def _run_mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
    large_m_splitk,
    w_preshuffle_bf16,
    res_preshuffle,
):
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    # (post_mix, comb_mix, layer_input), aiter's order
    return served(mhc_pre(residual, fn, hc_scale, hc_base, **scalars))


def _post_mix(post_layer_mix):
    """aiter's post mix, (T, 4, 1) or (T, 4), as AKLR's (T, 4); ``None`` for another shape."""
    if post_layer_mix.dim() == 3 and post_layer_mix.shape[-1] == 1:
        return post_layer_mix.squeeze(-1)
    return post_layer_mix if post_layer_mix.dim() == 2 else None


def _select_mhc_post(args, kwargs):
    out, x, residual, post_layer_mix, comb_res_mix, _store_nt, res_preshuffle = args
    post_mix = _post_mix(post_layer_mix)
    if out is None or res_preshuffle or post_mix is None:
        return None
    return offer(mhc_post.serving(x, residual, post_mix, comb_res_mix, out=out), _run_mhc_post)


def _run_mhc_post(out, x, residual, post_layer_mix, comb_res_mix, store_nt, res_preshuffle):
    served(mhc_post(x, residual, _post_mix(post_layer_mix), comb_res_mix, out=out))


def _select_mhc_fused_post_pre(args, kwargs):
    (
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        w_preshuffle_bf16,
        res_preshuffle,
    ) = args
    post_mix = _post_mix(post_layer_mix)
    # the shuffled fn or residual layouts are not AKLR's
    if w_preshuffle_bf16 or res_preshuffle or post_mix is None:
        return None
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    return offer(
        mhc_fused_post_pre.serving(
            layer_input, residual_in, post_mix, comb_res_mix, fn, hc_scale, hc_base, **scalars
        ),
        _run_mhc_fused_post_pre,
    )


def _run_mhc_fused_post_pre(
    layer_input,
    residual_in,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
    w_preshuffle_bf16,
    res_preshuffle,
):
    scalars = _kwargs(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps
    )
    # (post_mix, comb_mix, layer_input, next_residual), aiter's order
    return served(
        mhc_fused_post_pre(
            layer_input,
            residual_in,
            _post_mix(post_layer_mix),
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            **scalars,
        )
    )


def declare(plugin: Plugin) -> None:
    plugin.provide("mhc_post", version=1, select=_select_mhc_post)
    plugin.provide("mhc_pre", version=1, select=_select_mhc_pre)
    plugin.provide("mhc_fused_post_pre", version=1, select=_select_mhc_fused_post_pre)
