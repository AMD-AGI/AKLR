# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DotProductBound, DType, FiniteDistribution
from aklr.core.contract.spec import DTypeSpec, LayoutVariant, LayoutVariantsSpec, TensorSpec, axis

__all__ = ["CONTRACTS", "KIMI_K3_TINY_GEMM"]

_SAME_AS_X = DTypeSpec(same_as_tensor="x")

KIMI_K3_TINY_GEMM = OpContract(
    contract_id="kimi_k3_tiny_gemm@1.0",
    description="Kimi-K3's 16-bit gemm on torch: out = x @ w.T.",
    op_type="kimi_k3_tiny_gemm",
    axes={
        "m": axis("Tokens.", 1, 2**20, is_var=True),
        "n": axis("Output columns.", 16, 262144, multiple_of=8),
        "k": axis("Reduction length.", 128, 65536, multiple_of=128),
    },
    inputs={
        "x": TensorSpec(
            description="Activations.",
            shape=("m", "k"),
            dtype=DTypeSpec(any_of=(DType.BFLOAT16, DType.FLOAT16)),
            distribution=FiniteDistribution(),
        ),
        "w": TensorSpec(
            description="Weights.",
            shape=("n", "k"),
            dtype=_SAME_AS_X,
            distribution=FiniteDistribution(),
            lifetime="weight",
        ),
    },
    outputs={
        "out": TensorSpec(
            description="x @ w.T.",
            shape=("m", "n"),
            dtype=_SAME_AS_X,
            error_bounds=DotProductBound(length="k", accumulate_dtype=DType.FLOAT32, max_split_k=16),
        )
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
)

CONTRACTS = (KIMI_K3_TINY_GEMM,)
