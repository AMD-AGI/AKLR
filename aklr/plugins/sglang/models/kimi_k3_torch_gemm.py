# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's unquantized GEMMs that sglang runs outside its linear layers (``srt/models/kimi_k3.py``).

The MoE front (router, latent down and the shared experts' gate_up in one weight) and the shared
experts' down projection go through ``_k3_bf16_gemm``; KDA's beta and forget-gate projections go
through ``kimi_k3_tiny_gemm``, which the model imports at each call. Both are ``x @ weight.T``.

On 0.5.22 and 0.5.23 ``_k3_bf16_gemm`` runs torch. On ROCm, 0.5.24.dev809 (Kimi-K3.1-Pro's image) first
offers it to aiter (``srt/models/kimi_k3_amd/gemm.py``): a new output to aiter's tuned table where the table
holds the shape, an output given or of another dtype (the fp32 MoE front, the shared down into its buffer)
to aiter's FlyDSL kernel, launched directly; torch where aiter does not take it. That direct launch is below
the op aiter routes, so aiter's plugin never sees it: these GEMMs reach AKLR only through this route."""

from __future__ import annotations

from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("sglang.srt.models.kimi_k3._k3_bf16_gemm", versions=("0.5.22", "0.5.23", "0.5.24.dev809"))
def _k3_bf16_gemm(x, weight, out=None, out_dtype=None):
    if out_dtype is None and out is not None:
        out_dtype = out.dtype
    return gemm(x, weight, out_dtype=out_dtype, out=out)


@sglang_route("sglang.kernels.ops.gemm.kimi_k3_tiny_gemm", versions=("0.5.22", "0.5.23"))
def _kimi_k3_tiny_gemm(x, w):
    return gemm(x, w)
