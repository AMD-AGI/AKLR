# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DotProductBound,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    OutputRoundingBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "DEEPSEEK_V4_WO_A_SPLIT_K", "DEEPSEEK_V4_WO_A_SPLIT_K_MXFP8"]

# DeepSeek-V4's wo_a absorb GEMM in sglang's gfx950 split-K regime (gfx95_batched_gemm_bf16_fp8_grid):
# the per-group projection of the attention output, at most 64 rows (the decode and verify rows of one rank)
_WO_A_AXES = {
    "n_tokens": axis("Rows of this call; sglang takes its split-K regime up to 64.", 1, 64, is_var=True),
    "n_groups": axis("Output groups on this rank: o_groups over the attention TP size.", 1, 64),
    "rank": axis("Output columns per group (o_lora_rank).", 32, 4096, multiple_of=32),
    "in_dim": axis(
        "Reduction length per group, n_heads * head_dim / o_groups; split eight ways in 256-wide steps.",
        2048,
        65536,
        multiple_of=2048,
    ),
    "out_dim": derived_axis("Output columns, every group's rank columns in group order.", "n_groups * rank"),
}

_WO_A_INPUTS = {
    "x": TensorSpec(
        description="The attention output, in_dim columns per group.",
        shape=("n_tokens", "n_groups", "in_dim"),
        dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
        distribution=FiniteDistribution(),
    ),
    "w": TensorSpec(
        description="wo_a's weight of each group, (rank, in_dim).",
        shape=("n_groups", "rank", "in_dim"),
        dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
        distribution=LecunNormalDistribution(fan_in="in_dim"),
        lifetime="weight",
    ),
}

_WO_A_GRID = (
    "the fp8-grid rule of wo_b's operand: per 32 columns, e = the exponent of the smallest power of two "
    ">= max(amax, 1e-10) / 448 (in float32), and each value / 2**e rounded to fp8 e4m3, saturating at 448"
)

# The sum is exact here; a float32 kernel's last bit moves the bf16 rounding, which now and then
# moves a value one step of the fp8 grid (or a group's exponent).
_WO_A_COMPARE = CompareSpec(max_violation_fraction=1e-3)

DEEPSEEK_V4_WO_A_SPLIT_K = OpContract(
    contract_id="deepseek_v4_wo_a_split_k@1.0",
    description=(
        "sglang's gfx950 split-K wo_a GEMM of DeepSeek-V4 (_batched_gemm_split_k) with a bf16 output: "
        "out[t, g * rank + r] = sum_d x[t, g, d] * w[g, r, d] rounded to bf16, then, with fp8_grid, "
        f"onto {_WO_A_GRID}, times 2**e."
    ),
    op_type="gfx95_batched_gemm_bf16_fp8_grid",
    axes=_WO_A_AXES,
    attrs={
        "fp8_grid": attr(
            "Whether the output is rounded onto wo_b's fp8 grid (still held in bf16).",
            False,
            True,
            param_name="fp8_grid",
        ),
    },
    inputs=_WO_A_INPUTS,
    outputs={
        "out": TensorSpec(
            description="x @ w.T per group, the groups side by side; on the fp8 grid when fp8_grid.",
            shape=("n_tokens", "out_dim"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,), same_as_tensor="x"),
            error_bounds=DotProductBound(length="in_dim", accumulate_dtype=DType.FLOAT32, max_split_k=8),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_WO_A_COMPARE,
)

DEEPSEEK_V4_WO_A_SPLIT_K_MXFP8 = OpContract(
    contract_id="deepseek_v4_wo_a_split_k_mxfp8@1.0",
    description=(
        "sglang's gfx950 split-K wo_a GEMM of DeepSeek-V4 (_batched_gemm_split_k) handing wo_b MXFP8: "
        "y[t, g * rank + r] = sum_d x[t, g, d] * w[g, r, d] rounded to bf16, then quantized by "
        f"{_WO_A_GRID}; the fp8 codes and each group's 2**e."
    ),
    op_type="gfx95_batched_gemm_bf16_fp8_grid",
    axes={
        **_WO_A_AXES,
        "n_scales": derived_axis("One scale per 32 output columns.", "n_groups * rank // 32"),
    },
    inputs=_WO_A_INPUTS,
    outputs={
        "quantized": TensorSpec(
            description="y / 2**e of its 32-column group, rounded to fp8 e4m3.",
            shape=("n_tokens", "out_dim"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            error_bounds=OutputRoundingBound(),
        ),
        "scales": TensorSpec(
            description="2**e of each 32-column group of y; sglang holds the e8m0 byte as uint8.",
            shape=("n_tokens", "n_scales"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E8M0,)),
            error_bounds=OutputRoundingBound(),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=_WO_A_COMPARE,
)

CONTRACTS = (DEEPSEEK_V4_WO_A_SPLIT_K, DEEPSEEK_V4_WO_A_SPLIT_K_MXFP8)
