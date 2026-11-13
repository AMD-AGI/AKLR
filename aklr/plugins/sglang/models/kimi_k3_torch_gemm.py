# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's unquantized GEMMs that sglang runs on torch, not aiter (``srt/models/kimi_k3.py``).

The MoE front (router, latent down and the shared experts' gate_up in one weight) and the shared
experts' down projection go through ``_k3_bf16_gemm``; KDA's beta and forget-gate projections go
through ``kimi_k3_tiny_gemm``, which the model imports at each call. Both are ``x @ weight.T``."""

from __future__ import annotations

from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("sglang.srt.models.kimi_k3._k3_bf16_gemm", versions=("0.5.22",))
def _k3_bf16_gemm(x, weight, out=None, out_dtype=None):
    if out_dtype is None and out is not None:
        out_dtype = out.dtype
    return gemm(x, weight, out_dtype=out_dtype, out=out)


@sglang_route("sglang.kernels.ops.gemm.kimi_k3_tiny_gemm", versions=("0.5.22",))
def _kimi_k3_tiny_gemm(x, w):
    return gemm(x, w)
