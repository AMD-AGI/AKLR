# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import aklr_api
from aklr.ops.gemm.schema import (
    bmm_bf16_nn_op_schema,
    bmm_bf16_nt_op_schema,
    gemm_a8w8_nt_op_schema,
    gemm_bf16_nt_op_schema,
    gemm_fp8_nt_blockscaled_op_schema,
    gemm_mxfp4_nt_op_schema,
)

__all__ = [
    "bmm_bf16_nn",
    "bmm_bf16_nt",
    "gemm_a8w8_nt",
    "gemm_bf16_nt",
    "gemm_fp8_nt_blockscaled",
    "gemm_mxfp4_nt",
]

# Tensor parameters are named as the schema's inputs and outputs; the others pick the
# schema (b's storage, scale layout) or the output dtype.


@aklr_api(gemm_bf16_nt_op_schema)
def gemm_bf16_nt(a, b, bias=None, out=None, out_dtype=torch.bfloat16, use_shuffled_weight=False): ...


@aklr_api(gemm_a8w8_nt_op_schema)
def gemm_a8w8_nt(
    a,
    b,
    a_scale,
    b_scale,
    bias=None,
    out=None,
    out_dtype=torch.bfloat16,
    use_shuffled_weight=False,
): ...


@aklr_api(gemm_fp8_nt_blockscaled_op_schema)
def gemm_fp8_nt_blockscaled(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode="MN",
    out=None,
    out_dtype=None,
    use_shuffled_weight=False,
): ...


@aklr_api(gemm_mxfp4_nt_op_schema)
def gemm_mxfp4_nt(
    a,
    b,
    a_descale,
    b_descale,
    bias=None,
    out=None,
    out_dtype=torch.bfloat16,
    use_shuffled_weight=False,
): ...


@aklr_api(bmm_bf16_nt_op_schema)
def bmm_bf16_nt(A, B, out=None, out_dtype=torch.bfloat16, use_shuffled_weight=False): ...


@aklr_api(bmm_bf16_nn_op_schema)
def bmm_bf16_nn(A, B, out=None, out_dtype=torch.bfloat16, use_shuffled_weight=False): ...
