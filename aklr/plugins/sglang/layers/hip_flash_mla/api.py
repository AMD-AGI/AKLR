# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla import flash_mla_with_kvcache
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("sglang.srt.layers.attention.hip_flash_mla.aiter_sparse_decode_fwd", versions=("0.5.23",))
def _aiter_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    attn_sink,
    softmax_scale,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    inv_rope=None,
    topk_length=None,
    extra_topk_length=None,
    **_,
):
    """The decode attention and its inverse RoPE as one flash_mla_with_kvcache call, returned
    as sglang returns them, (output [b, s, heads, 512], None), for DeepSeek-V4's fp8 pools
    when the model passes inv_rope; sglang folds the top-k lengths into the index lists before
    this call, so topk_length and extra_topk_length are not read, as sglang does not read them
    here."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers.attention import hip_flash_mla

    b, s = q.shape[:2]
    if (
        inv_rope is None
        # V4.1's compact pools take sglang's compact kernel
        or k_cache.shape[-1] in (KVLayout.V41.bytes_per_token, KVLayout.V41_FP4.bytes_per_token)
        # deterministic inference pins a split count, hence the combine order, below 1024 rows
        or (
            b * s < hip_flash_mla._AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS
            and hip_flash_mla.hip_attn_kv_splits() > 0
        )
    ):
        return FallbackReason.OUT_OF_CONTRACT
    rope_freqs, positions = inv_rope
    result = flash_mla_with_kvcache(
        q,
        k_cache,
        indices,
        extra_kv_cache=extra_k_cache,
        extra_sparse_indices=extra_indices_in_kvcache,
        sm_scale=softmax_scale,
        sinks=attn_sink,
        kv_cache_format="dsv4_fp8",
        return_lse=False,
        kv_cache_layout="padded_pages",
        rope_freqs=rope_freqs,
        positions=positions.reshape(-1)[: b * s],
    )
    if isinstance(result, FallbackReason):
        return result
    return result[0], None
