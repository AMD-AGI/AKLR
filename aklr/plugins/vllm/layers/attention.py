# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# TODO(ruibzhan): remove this file once AITER's plugin works inside vLLM custom ops
# under torch.compile and CUDA graphs and preserves measured speedups for all
# workloads served by this route.

"""vLLM's ROCM_AITER_FA attention backend, where it reaches aiter's triton unified attention."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.attention.api import unified_attention
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


# vLLM imports aiter's backward-compatible path, which may alias the current module.
@vllm_route(
    "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionImpl.forward",
    "aiter.ops.triton.unified_attention.unified_attention",
    "aiter.ops.triton.attention.unified_attention.unified_attention",
    versions=("0.33.0rc1.dev910",),
)
def _unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache=False,
    skip_reduce=False,
    backend=None,
):
    if (
        q_descale is not None
        or k_descale is not None
        or v_descale is not None
        or q_scales is not None
        or alibi_slopes is not None
        or output_scale is not None
        or qq_bias is not None
        or sinks is not None
        or shuffled_kv_cache
        or skip_reduce
    ):
        return FallbackReason.OUT_OF_CONTRACT
    result = unified_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        softmax_scale,
        max_seqlen_q=int(max_seqlen_q),
        window_size=(int(window_size[0]), int(window_size[1])),
        causal=bool(causal),
        # vLLM's caches with the K and V halves of each page side by side
        kv_cache_storage="row_major" if k.is_contiguous() and v.is_contiguous() else "kv_split_pages",
        softcap=float(softcap),
        out=out,
    )
    # aiter's returns nothing: it writes out
    return result if isinstance(result, FallbackReason) else None
