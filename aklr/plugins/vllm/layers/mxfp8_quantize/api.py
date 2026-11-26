# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.layers.mxfp8_quantize.contract import MXFP8_E4M3_QUANTIZE

__all__ = ["mxfp8_e4m3_quantize"]

_GEMM = "vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm"


@api(MXFP8_E4M3_QUANTIZE)
def mxfp8_e4m3_quantize(x): ...


@vllm_route(
    f"{_GEMM}._rocm_mxfp8_block32_gemm_quant_impl", f"{_GEMM}.mxfp8_e4m3_quantize", versions=("0.34.0",)
)
def _mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, alignment=0):
    """The activation quant of vLLM's MXFP8 linear on 32x32 block-scaled weights (DeepSeek-V4/V4.1) past
    the token counts its GEMM quantizes in its K loop."""
    import torch

    if (
        is_sf_swizzled_layout
        or alignment
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or x.shape[1] % 32
        or not x.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    # (quantized, scales) as vLLM's returns them, or why vLLM keeps the call
    return mxfp8_e4m3_quantize(x)
