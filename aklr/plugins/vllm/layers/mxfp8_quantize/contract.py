# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MXFP8 activation quantizer (``mxfp8_e4m3_quantize`` in
``model_executor/layers/quantization/utils/mxfp8_utils.py``, its Triton kernel on ROCm): groups of 32 BF16
values to E4M3FN with row-major E8M0 scale bytes, the byte clamped to [0, 254]."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DType, FiniteDistribution, ToleranceBound, tolerance
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.vllm.layers.mxfp8_quantize.baseline import Mxfp8E4m3QuantizeBaseline
from aklr.plugins.vllm.layers.mxfp8_quantize.reference import Mxfp8E4m3QuantizeReference

__all__ = ["CONTRACTS", "MXFP8_E4M3_QUANTIZE"]

MXFP8_E4M3_QUANTIZE = OpContract(
    contract_id="mxfp8_e4m3_quantize@1.0",
    description=(
        "vLLM's MXFP8 quantization of BF16 activations: per row and group of 32 values, the scale byte is "
        "ceil(log2(max(amax, FLT_MIN) / 448)) + 127 clamped to [0, 254], and the values are x * 2**(127 - byte) "
        "rounded to E4M3FN. Scales are row-major [row, group]."
    ),
    op_type="mxfp8_quantize",
    axes={
        "n_tokens": axis("Rows quantized independently.", 1, 131072, is_var=True),
        "hidden_size": axis("Elements per row, divisible by 32.", 32, 65536, multiple_of=32),
        "n_groups": derived_axis("Groups of 32 elements per row.", "hidden_size // 32"),
    },
    inputs={
        "x": TensorSpec(
            description="Finite BF16 activations, rows contiguous.",
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
            error_bounds=tolerance(DType.FLOAT8_E4M3FN),
        ),
        "scales": TensorSpec(
            description="E8M0 exponent bytes at [row, group]; the dequantization scale is 2**(byte - 127).",
            shape=("n_tokens", "n_groups"),
            dtype=DTypeSpec(any_of=(DType.UINT8,)),
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    reference=Mxfp8E4m3QuantizeReference(),
    baseline=Mxfp8E4m3QuantizeBaseline(),
)

CONTRACTS = (MXFP8_E4M3_QUANTIZE,)
