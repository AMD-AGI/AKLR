###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import math

import torch

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.mhc.reference import (
    mhc_fused_post_pre_reference,
    mhc_post_reference,
    mhc_pre_reference,
)

__all__ = ["MHC_FUSED_POST_PRE", "MHC_POST", "MHC_PRE"]


def _mhc_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # aiter's mHC tests: allclose(atol=1e-2, rtol=1e-2)
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=1e-2 if rtol is None else rtol,
        atol=1e-2 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


MHC_PRE = Schema(
    op_type="mhc_pre",
    description=(
        "Mix four residual streams into an optionally rms-normed layer input; derive the post mixes."
    ),
    axes={
        "n_tokens": Var("Tokens."),
        "hidden_size": Const("Hidden size of one stream."),
        "n_streams": Const("Residual streams.", abbrev="", value=4),
        "n_mix_logits": Const("Mix logits: 4 pre, 4 post, 16 combination.", abbrev="", value=24),
        "n_mix_scales": Const("Pre, post and combination.", abbrev="", value=3),
        "one": Const("A post mix per stream.", abbrev="", value=1),
        "projection_size": Const("Projection input: n_streams * hidden_size.", abbrev=""),
    },
    constraints=[
        "projection_size == n_streams * hidden_size",
        "sinkhorn_iters >= 1",
    ],
    inputs={
        "residual": Tensor(
            ["n_tokens", "n_streams", "hidden_size"],
            dtype="bfloat16",
            description="Four residual streams.",
        ),
        "proj_weight": Tensor(
            ["n_mix_logits", "projection_size"],
            dtype="float32",
            description="Projects the flattened streams to the mix logits.",
        ),
        "mix_scale": Tensor(
            ["n_mix_scales"],
            dtype="float32",
            description="Scale of the pre, post and combination logits.",
        ),
        "mix_bias": Tensor(["n_mix_logits"], dtype="float32", description="Bias of the mix logits."),
        "rms_eps": Scalar(
            "float32",
            description="Added to the mean square of the streams.",
            optional=True,
        ),
        "pre_eps": Scalar(
            "float32",
            description="Added to each pre mix after its sigmoid.",
            optional=True,
        ),
        "sinkhorn_eps": Scalar(
            "float32",
            description="Added in each Sinkhorn normalization.",
            optional=True,
        ),
        "post_multiplier": Scalar(
            "float32",
            description="Multiplies each post mix after its sigmoid.",
            optional=True,
        ),
        "sinkhorn_iters": Scalar(
            "int32",
            description="Sinkhorn normalizations of the combination mix.",
            optional=True,
        ),
        "norm_weight": Tensor(
            ["hidden_size"],
            dtype="bfloat16",
            optional=True,
            description="Weight of the rms norm of layer_input.",
        ),
        # aiter passes norm_eps even without norm_weight
        "norm_eps": Scalar(
            "float32",
            description="Added to the mean square of layer_input.",
            optional=True,
        ),
    },
    outputs={
        "post_mix": Tensor(
            ["n_tokens", "n_streams", "one"],
            dtype="float32",
            description="Post mix of the next mHC post.",
        ),
        "comb_mix": Tensor(
            ["n_tokens", "n_streams", "n_streams"],
            dtype="float32",
            description="Combination mix of the next mHC post.",
        ),
        "layer_input": Tensor(
            ["n_tokens", "hidden_size"],
            description="The streams summed under the pre mix.",
            dtype_from="residual",
        ),
    },
    reference=mhc_pre_reference,
    check=_mhc_check,
)

MHC_POST = Schema(
    op_type="mhc_post",
    description="Mix four residual streams and a layer output into four new streams.",
    axes={
        "n_tokens": Var("Tokens."),
        "hidden_size": Const("Hidden size of one stream."),
        "n_streams": Const("Residual streams.", abbrev="", value=4),
        "one": Const("A post mix per stream.", abbrev="", value=1),
    },
    inputs={
        "x": Tensor(
            ["n_tokens", "hidden_size"],
            dtype="bfloat16",
            description="The layer's output.",
        ),
        "residual": Tensor(
            ["n_tokens", "n_streams", "hidden_size"],
            dtype="bfloat16",
            description="Four residual streams.",
        ),
        "post_mix": Tensor(
            ["n_tokens", "n_streams", "one"],
            dtype="float32",
            description="Weight of x in each new stream.",
        ),
        "comb_mix": Tensor(
            ["n_tokens", "n_streams", "n_streams"],
            dtype="float32",
            description="Weight of old stream i in new stream j, at [i, j].",
        ),
    },
    outputs={
        "out": Tensor(
            ["n_tokens", "n_streams", "hidden_size"],
            dtype="bfloat16",
            dtype_from="out",
            description="The new streams.",
        )
    },
    reference=mhc_post_reference,
    check=_mhc_check,
)


def _mhc_fused_post_pre_init(*, n_tokens, hidden_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    projection_size = 4 * hidden_size

    def log_uniform(shape, lo, hi):
        return torch.exp2(torch.empty(shape, device=device).uniform_(lo, hi))

    return {
        "x": torch.randn(n_tokens, hidden_size, device=device).to(torch.bfloat16),
        "residual": torch.randn(n_tokens, 4, hidden_size, device=device).to(torch.bfloat16),
        "post_mix": log_uniform((n_tokens, 4, 1), -3.32, -0.15),
        "comb_mix": log_uniform((n_tokens, 4, 4), -4.0, 0.0),
        "proj_weight": torch.randn(24, projection_size, device=device) / math.sqrt(projection_size),
        "mix_scale": torch.ones(3, device=device),
        "mix_bias": torch.randn(24, device=device),
        "rms_eps": 1e-6,
        "pre_eps": 1e-6,
        "sinkhorn_eps": 1e-6,
        "post_multiplier": 2.0,
        "sinkhorn_iters": 20,
        "norm_weight": log_uniform(hidden_size, -1.0, 0.58).to(torch.bfloat16),
        "norm_eps": 1e-6,
    }


MHC_FUSED_POST_PRE = Schema(
    op_type="mhc_fused_post_pre",
    description=(
        "mHC post, then the next mHC pre on the new streams rounded to residual's "
        "dtype; aiter's return order (next_post_mix, next_comb_mix, layer_input, "
        "next_residual), not sglang's residual-first order."
    ),
    axes={
        "n_tokens": Var("Tokens."),
        "hidden_size": Const("Hidden size of one stream."),
        "n_streams": Const("Residual streams.", abbrev="", value=4),
        "n_mix_logits": Const("Mix logits: 4 pre, 4 post, 16 combination.", abbrev="", value=24),
        "n_mix_scales": Const("Pre, post and combination.", abbrev="", value=3),
        "one": Const("A post mix per stream.", abbrev="", value=1),
        "projection_size": Const("Projection input: n_streams * hidden_size.", abbrev=""),
    },
    constraints=[
        "projection_size == n_streams * hidden_size",
        "sinkhorn_iters >= 1",
    ],
    inputs={
        "x": Tensor(
            ["n_tokens", "hidden_size"],
            dtype="bfloat16",
            description="The layer's output.",
        ),
        "residual": Tensor(
            ["n_tokens", "n_streams", "hidden_size"],
            dtype="bfloat16",
            description="Four residual streams.",
        ),
        "post_mix": Tensor(
            ["n_tokens", "n_streams", "one"],
            dtype="float32",
            description="Weight of x in each new stream.",
        ),
        "comb_mix": Tensor(
            ["n_tokens", "n_streams", "n_streams"],
            dtype="float32",
            description="Weight of old stream i in new stream j, at [i, j].",
        ),
        "proj_weight": Tensor(
            ["n_mix_logits", "projection_size"],
            dtype="float32",
            description="Projects the flattened streams to the mix logits.",
        ),
        "mix_scale": Tensor(
            ["n_mix_scales"],
            dtype="float32",
            description="Scale of the pre, post and combination logits.",
        ),
        "mix_bias": Tensor(["n_mix_logits"], dtype="float32", description="Bias of the mix logits."),
        "rms_eps": Scalar(
            "float32",
            description="Added to the mean square of the streams.",
            optional=True,
        ),
        "pre_eps": Scalar(
            "float32",
            description="Added to each pre mix after its sigmoid.",
            optional=True,
        ),
        "sinkhorn_eps": Scalar(
            "float32",
            description="Added in each Sinkhorn normalization.",
            optional=True,
        ),
        "post_multiplier": Scalar(
            "float32",
            description="Multiplies each post mix after its sigmoid.",
            optional=True,
        ),
        "sinkhorn_iters": Scalar(
            "int32",
            description="Sinkhorn normalizations of the combination mix.",
            optional=True,
        ),
        "norm_weight": Tensor(
            ["hidden_size"],
            dtype="bfloat16",
            optional=True,
            description="Weight of the rms norm of layer_input.",
        ),
        # aiter passes norm_eps even without norm_weight
        "norm_eps": Scalar(
            "float32",
            description="Added to the mean square of layer_input.",
            optional=True,
        ),
    },
    outputs={
        "next_post_mix": Tensor(
            ["n_tokens", "n_streams", "one"],
            dtype="float32",
            description="Post mix of the next mHC post.",
        ),
        "next_comb_mix": Tensor(
            ["n_tokens", "n_streams", "n_streams"],
            dtype="float32",
            description="Combination mix of the next mHC post.",
        ),
        "layer_input": Tensor(
            ["n_tokens", "hidden_size"],
            description="The streams summed under the pre mix.",
            dtype_from="residual",
        ),
        "next_residual": Tensor(
            ["n_tokens", "n_streams", "hidden_size"],
            description="The new streams.",
            dtype_from="residual",
        ),
    },
    reference=mhc_fused_post_pre_reference,
    check=_mhc_check,
    init=_mhc_fused_post_pre_init,
)
