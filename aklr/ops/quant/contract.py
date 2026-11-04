# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dynamic group quantization with explicit scale storage order."""

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, FiniteDistribution, OutputRoundingBound
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)

PER_GROUP_QUANT_FP8 = OpContract(
    contract_id="per_group_quant_fp8@1.0",
    description=(
        "Quantize each group of 128 BF16 values to E4M3FN using its absolute maximum, "
        "floored at 1e-10. Scales are float32 dequantization scales stored group-major."
    ),
    op_type="quant",
    axes={
        "n_tokens": axis("Rows quantized independently.", 1, 131072, is_var=True),
        "hidden_size": axis("Elements per row, divisible by 128.", 128, 21504, multiple_of=128),
        "n_groups": derived_axis("Groups of 128 elements per row.", "hidden_size // 128"),
    },
    inputs={
        "x": TensorSpec(
            description="Finite BF16 activations.",
            shape=("n_tokens", "hidden_size"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16,)),
            distribution=FiniteDistribution(),
        ),
    },
    outputs={
        "quantized": TensorSpec(
            description="E4M3FN quantized activations.",
            shape=("n_tokens", "hidden_size"),
            dtype=DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,)),
            error_bounds=OutputRoundingBound(),
        ),
        "scales": TensorSpec(
            description="Dequantization scale at [group, row], with rows contiguous.",
            shape=("n_groups", "n_tokens"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            error_bounds=OutputRoundingBound(),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

PER_GROUP_QUANT_MXFP8 = OpContract(
    contract_id="per_group_quant_mxfp8@1.0",
    description=(
        "Quantize groups of 32 BF16 values to E4M3FN with row-major E8M0 "
        "dequantization scales, rounding each scale up to a power of two."
    ),
    op_type="quant",
    axes={
        "n_tokens": axis("Rows quantized independently.", 1, 131072, is_var=True),
        "hidden_size": axis("Elements per row, divisible by 32.", 32, 21504, multiple_of=32),
        "n_groups": derived_axis("Groups of 32 elements per row.", "hidden_size // 32"),
    },
    inputs=PER_GROUP_QUANT_FP8.inputs,
    outputs={
        "quantized": PER_GROUP_QUANT_FP8.outputs["quantized"],
        "scales": TensorSpec(
            description="E8M0 exponent bytes at [row, group]; scale is 2**(byte - 127).",
            shape=("n_tokens", "n_groups"),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (PER_GROUP_QUANT_FP8, PER_GROUP_QUANT_MXFP8)
