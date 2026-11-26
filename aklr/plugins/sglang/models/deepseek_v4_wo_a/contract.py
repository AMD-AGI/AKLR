# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1's grouped output-LoRA projection (wo_a) on gfx950, as sglang's
``wo_a_fp8_grid_matmul`` runs it when wo_b takes MXFP8 activations: each group's slice of the
attention output times its wo_a block, rounded to bf16, then quantized per 32 columns to fp8 e4m3
with a ue8m0 scale, wo_b's operand."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    SqnrBound,
    StridedRows,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseek_v4_wo_a.baseline import DeepseekV4WoAMxfp8Baseline
from aklr.plugins.sglang.models.deepseek_v4_wo_a.reference import DeepseekV4WoAMxfp8Reference

__all__ = ["CONTRACTS", "DEEPSEEK_V4_WO_A_MXFP8"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))


def _storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="storage", expected_value=value),)


_FINITE = FiniteDistribution()

DEEPSEEK_V4_WO_A_MXFP8 = OpContract(
    contract_id="deepseek_v4_wo_a_mxfp8@1.0",
    description=(
        "y[t, g * o_lora_rank + r] = sum_d o[t, g, d] * wo_a[g, r, d], rounded to bf16, then each 32 "
        "consecutive columns of a row quantized to fp8 e4m3 by the smallest power of two >= their "
        "amax / 448 (amax floored at 1e-10, exponent clamped to [1, 254]): quantized holds y / scale, "
        "clamped to +-448, and scales the scale's ue8m0 exponent byte."
    ),
    op_type="deepseek_v4_wo_a",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "n_groups": axis("Output groups on this rank.", 1, 64),
        "o_lora_rank": axis("Columns of a group's projection.", 32, 8192, multiple_of=32),
        "group_dim": axis(
            "The reduction: a group's slice of the attention output.", 128, 65536, multiple_of=128
        ),
        "out_dim": derived_axis("Columns of the output: every group's.", "n_groups * o_lora_rank"),
        "out_blocks": derived_axis(
            "Scales per output row: one per 32 columns.", "n_groups * o_lora_rank // 32"
        ),
    },
    inputs={
        "o": TensorSpec(
            description="The attention output per token, as n_groups slices of group_dim.",
            shape=("n_tokens", "n_groups", "group_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "wo_a": TensorSpec(
            description="wo_a's weight, one [o_lora_rank, group_dim] block per group.",
            shape=("n_groups", "o_lora_rank", "group_dim"),
            dtype=_BF16,
            distribution=_FINITE,
            lifetime="weight",
        ),
    },
    outputs={
        # Two correct kernels sum in other fp32 orders: a bf16 value can round to its neighbour, and a
        # group's amax can cross a power of two, which halves or doubles its 32 codes. The codes' SQNR
        # bound leaves room for that, a scale may move by one.
        "quantized": TensorSpec(
            description="Each output element over its group's scale, fp8 e4m3.",
            shape=("n_tokens", "out_dim"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            error_bounds=SqnrBound(min_sqnr_db=25.0),
        ),
        "scales": TensorSpec(
            description="The ue8m0 exponent byte of each 32 columns' scale: scale = 2**(byte - 127).",
            shape=("n_tokens", "out_blocks"),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            error_bounds=ToleranceBound(absolute_tolerance=1, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
            # sglang's o: rows that may lie apart in a wider buffer
            "strided_rows": LayoutVariant(
                layouts={"o": StridedRows()}, detect_facts=_storage_facts("strided_rows")
            ),
        }
    ),
    reference=DeepseekV4WoAMxfp8Reference(),
    baseline=DeepseekV4WoAMxfp8Baseline(),
)

CONTRACTS = (DEEPSEEK_V4_WO_A_MXFP8,)
