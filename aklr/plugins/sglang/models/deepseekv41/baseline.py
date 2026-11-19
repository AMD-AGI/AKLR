# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23.dev20261111 runs on gfx950 for DeepSeek-V4.1 where no monokernel
serves the step."""

from __future__ import annotations

__all__ = ["deepseekv41_mxfp8_linear_baseline", "deepseekv41_sparse_mla_baseline"]


def deepseekv41_mxfp8_linear_baseline(x, weight, weight_scale, x_scale=None, output=None):
    """aiter_mxfp8_blockscaled_linear as Fp8LinearMethod calls it: sglang's quantization of a
    bf16 x, then aiter's blockscale GEMM at the kernel its tuned tables pick."""
    import torch
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear

    return aiter_mxfp8_blockscaled_linear(
        input=x,
        weight=weight,
        weight_scale_ue8m0=weight_scale.view(torch.uint8),
        input_scale=None if x_scale is None else x_scale.view(torch.uint8),
    )


def deepseekv41_sparse_mla_baseline(
    q,
    kv_cache,
    indices,
    sinks,
    sm_scale,
    extra_kv_cache=None,
    extra_indices=None,
    rope_freqs=None,
    positions=None,
    kv_cache_layout="row_major",
    output=None,
):
    """aiter_sparse_decode_fwd as DeepseekV4HipRadixBackend calls it: aiter's sparse decode
    kernel, split over the KV below 1024 tokens with sglang's reduce folding the rotation in,
    unsplit from there with sglang's rotation after it."""
    from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd

    attended, _ = aiter_sparse_decode_fwd(
        q,
        kv_cache,
        indices,
        sinks,
        sm_scale,
        extra_k_cache=extra_kv_cache,
        extra_indices_in_kvcache=extra_indices,
        inv_rope=None if rope_freqs is None else (rope_freqs, positions),
    )
    return attended
