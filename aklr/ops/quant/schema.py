# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.schema import Const, Schema, Standard, Tensor, Var, default_check
from aklr.ops.quant.baseline import per_1x32_mx_quant_hip_baseline, per_group_quant_hip_baseline
from aklr.ops.quant.reference import per_group_quant_fp8_reference, per_group_quant_mxfp8_reference

__all__ = ["PER_GROUP_QUANT_FP8", "PER_GROUP_QUANT_MXFP8"]


def _exact_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    # the reference rounds as the op does: every value and scale is exact
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=0.0 if rtol is None else rtol,
        atol=0.0 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _per_group_quant_init(*, n_tokens, hidden_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    return {"x": torch.randn(n_tokens, hidden_size, device=device).to(torch.bfloat16)}


_AXES_DESCRIPTION = "Rows quantized independently."

PER_GROUP_QUANT_FP8 = Schema(
    op_type="per_group_quant_fp8",
    description=(
        "Quantize each group of 128 BF16 values to E4M3FN using its absolute maximum, "
        "floored at 1e-10. Scales are float32 dequantization scales stored group-major."
    ),
    axes={
        "n_tokens": Var(_AXES_DESCRIPTION),
        "hidden_size": Const("Elements per row, divisible by 128."),
        "n_groups": Const(
            "Groups of 128 elements per row: hidden_size // 128.",
            abbrev="",
            extract=lambda call: call["x"].shape[1] // 128,  # only the outputs carry it
        ),
    },
    constraints=["hidden_size % 128 == 0", "n_groups == hidden_size // 128", "x.dtype == 'bfloat16'"],
    inputs={"x": Tensor(["n_tokens", "hidden_size"], description="Finite BF16 activations.")},
    outputs={
        "quantized": Tensor(
            ["n_tokens", "hidden_size"], description="E4M3FN quantized activations.", dtype="float8_e4m3fn"
        ),
        "scales": Tensor(
            ["n_groups", "n_tokens"],
            description="Dequantization scale at [group, row], with rows contiguous.",
            dtype="float32",
        ),
    },
    standard=Standard(
        reference=per_group_quant_fp8_reference,
        check=_exact_check,
        init=_per_group_quant_init,
        baseline=per_group_quant_hip_baseline,
    ),
)

PER_GROUP_QUANT_MXFP8 = Schema(
    op_type="per_group_quant_mxfp8",
    description=(
        "Quantize groups of 32 BF16 values to E4M3FN with row-major E8M0 "
        "dequantization scales, rounding each scale up to a power of two."
    ),
    axes={
        "n_tokens": Var(_AXES_DESCRIPTION),
        "hidden_size": Const("Elements per row, divisible by 32."),
        "n_groups": Const(
            "Groups of 32 elements per row: hidden_size // 32.",
            abbrev="",
            extract=lambda call: call["x"].shape[1] // 32,  # only the outputs carry it
        ),
    },
    constraints=["hidden_size % 32 == 0", "n_groups == hidden_size // 32", "x.dtype == 'bfloat16'"],
    inputs=PER_GROUP_QUANT_FP8.inputs,
    outputs={
        "quantized": PER_GROUP_QUANT_FP8.outputs["quantized"],
        "scales": Tensor(
            ["n_tokens", "n_groups"],
            description="E8M0 exponent bytes at [row, group]; scale is 2**(byte - 127).",
            dtype="uint8",
        ),
    },
    standard=Standard(
        reference=per_group_quant_mxfp8_reference,
        check=_exact_check,
        init=_per_group_quant_init,
        baseline=per_1x32_mx_quant_hip_baseline,
    ),
)
