# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

__all__ = ["deepseek_v4_wo_a_split_k_baseline", "deepseek_v4_wo_a_split_k_mxfp8_baseline"]


def _wo_a_split_k(x, w, out, fp8_grid, scale):
    import inspect

    from sglang.kernels.ops.gemm import gfx95_batched_gemm_bf16_fp8_grid as gemm

    n_tokens, _, in_dim = x.shape
    if not gemm._split_k_applies(n_tokens, in_dim, w.shape[1]):
        raise NotImplementedError("outside sglang's split-K regime")
    # sglang's own function, also where the route hooks it
    inspect.unwrap(gemm._batched_gemm_split_k)(x, w, out, fp8_grid, scale)


def deepseek_v4_wo_a_split_k_baseline(x, w, fp8_grid=False, out_dtype=None, out=None):
    """sglang 0.5.23's gfx950 split-K wo_a GEMM, _batched_gemm_split_k: the K-split partial
    kernel into a float32 scratch, then the reduce kernel (split order, bf16, fp8 grid), into
    the caller's out."""
    import torch

    if out is None:
        out = torch.empty((x.shape[0], w.shape[0] * w.shape[1]), dtype=torch.bfloat16, device=x.device)
    _wo_a_split_k(x, w, out, fp8_grid, None)
    return out


def deepseek_v4_wo_a_split_k_mxfp8_baseline(x, w, quantized=None, scales=None):
    """sglang 0.5.23's gfx950 split-K wo_a GEMM, _batched_gemm_split_k, emitting fp8: the
    K-split partial kernel, then the reduce kernel quantizing the bf16 sum, into the caller's
    quantized and scales (the e8m0 bytes, as sglang's uint8)."""
    import torch

    shape = (x.shape[0], w.shape[0] * w.shape[1])
    if quantized is None:
        quantized = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    if scales is None:
        scales = torch.empty((shape[0], shape[1] // 32), dtype=torch.float8_e8m0fnu, device=x.device)
    _wo_a_split_k(x, w, quantized, False, scales.view(torch.uint8))
    return quantized, scales
