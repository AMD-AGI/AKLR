###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

import torch

from aklr.core.schema import Const, Scalar, Schema, Tensor, Var, default_check
from aklr.ops.rms_norm.reference import add_rms_norm_reference, rms_norm_reference

__all__ = ["ADD_RMS_NORM", "RMS_NORM"]


def _add_rms_norm_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # the output's unit roundoff: B = 2u|y| in bfloat16, rounding the output only
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=2**-8 if rtol is None else rtol,
        atol=0.0 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _add_rms_norm_init(*, n_tokens, hidden_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_tokens, hidden_size, device=device).to(torch.bfloat16)
    residual = torch.randn(n_tokens, hidden_size, device=device).to(torch.bfloat16)
    log2_weight = torch.empty(hidden_size, device=device).uniform_(-1.0, 0.58)
    return {
        "x": x,
        "residual": residual,
        "weight": torch.exp2(log2_weight).to(torch.bfloat16),
        "epsilon": 1e-6,
        "normalized": torch.empty_like(x),
        "residual_out": torch.empty_like(x),
    }


def _rms_norm_init(*, n_tokens, hidden_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_tokens, hidden_size, device=device).to(torch.bfloat16)
    log2_weight = torch.empty(hidden_size, device=device).uniform_(-1.0, 0.58)
    return {
        "x": x,
        "weight": torch.exp2(log2_weight).to(torch.bfloat16),
        "epsilon": 1e-6,
        "normalized": torch.empty_like(x),
    }


ADD_RMS_NORM = Schema(
    op_type="add_rms_norm",
    description=(
        "residual_out = x + residual; normalized = residual_out * "
        "rsqrt(mean(residual_out**2) + epsilon) * weight, the mean over the hidden axis."
    ),
    axes={
        "n_tokens": Var("Tokens."),
        "hidden_size": Const("Width normalized."),
    },
    constraints=[
        "epsilon > 0",
        "x.dtype in ('bfloat16', 'float16')",
        "residual.dtype == x.dtype and weight.dtype == x.dtype",
    ],
    inputs={
        "x": Tensor(["n_tokens", "hidden_size"], description="The layer's output."),
        "residual": Tensor(["n_tokens", "hidden_size"], description="The residual stream."),
        "weight": Tensor(["hidden_size"], description="Scale of each normalized column."),
        "epsilon": Scalar("float32", description="Added to the mean square."),
    },
    outputs={
        "normalized": Tensor(
            ["n_tokens", "hidden_size"],
            description="The normalized sum, scaled by weight.",
            dtype_from="x",
        ),
        "residual_out": Tensor(["n_tokens", "hidden_size"], description="x + residual.", dtype_from="x"),
    },
    reference=add_rms_norm_reference,
    check=_add_rms_norm_check,
    init=_add_rms_norm_init,
)

RMS_NORM = Schema(
    op_type="rms_norm",
    description="normalized = x * rsqrt(mean(x**2) + epsilon) * weight, the mean over the hidden axis.",
    axes={
        "n_tokens": Var("Tokens."),
        "hidden_size": Const("Width normalized."),
    },
    constraints=[
        "epsilon > 0",
        "x.dtype in ('bfloat16', 'float16')",
        "weight.dtype == x.dtype",
    ],
    inputs={
        "x": Tensor(["n_tokens", "hidden_size"], description="Rows to normalize."),
        "weight": Tensor(["hidden_size"], description="Scale of each normalized column."),
        "epsilon": Scalar("float32", description="Added to the mean square."),
    },
    outputs={
        "normalized": Tensor(
            ["n_tokens", "hidden_size"],
            description="The normalized rows, scaled by weight.",
            dtype_from="x",
        ),
    },
    reference=rms_norm_reference,
    check=_add_rms_norm_check,
    init=_rms_norm_init,
)
