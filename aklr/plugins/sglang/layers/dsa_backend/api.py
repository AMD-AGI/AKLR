# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend.contract import DSA_SPARSE_MLA_DECODE

__all__ = ["dsa_sparse_mla_decode"]


@api(DSA_SPARSE_MLA_DECODE)
def dsa_sparse_mla_decode(q, kv_cache, topk_indices, sm_scale): ...


# _forward_triton_decode, not forward_decode: sglang still picks the DSA decode
# implementation (triton, HIP's default), and this is the one the baseline measures.
@sglang_route(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_triton_decode",
    versions=">=0.5.22,<0.5.23",
)
def _forward_triton_decode(self, q_nope, q_rope, kv_cache, v_head_dim, page_table_1, sm_scale):
    """The decode's attention as one op call on the whole q, the layer's KV pool as rows
    and the top-k slots (page_table_1), returned as sglang's kernel returns it:
    [1, tokens, heads, v_head_dim], v_head_dim being q_nope's width (sglang's kernel
    asserts it). Speculative decoding (MTP) and a q not held as one buffer fall back to
    sglang; the contract checks the rest (fp8 q and pool, shapes, the top-k's width, the
    pool's size)."""
    q = _whole_query(q_nope, q_rope)
    if q is None or _speculative():
        return FallbackReason.OUT_OF_CONTRACT
    output = dsa_sparse_mla_decode(
        q=q,
        kv_cache=kv_cache.view(-1, kv_cache.shape[-1]),
        topk_indices=page_table_1,
        sm_scale=sm_scale,
    )
    if isinstance(output, FallbackReason):
        return output
    return output.unsqueeze(0)


def _whole_query(q_nope, q_rope):
    """The [tokens, heads, latent + rope] q that forward_decode splits into the two views
    it passes (q_all, as the fused RoPE and cache write leaves it); None for two tensors
    apart."""
    tokens, heads, latent = q_nope.shape
    width = latent + q_rope.shape[-1]
    if not (
        q_rope.dtype == q_nope.dtype
        and q_rope.shape[:2] == q_nope.shape[:2]
        and q_nope.stride() == q_rope.stride() == (heads * width, width, 1)
        and q_rope.untyped_storage().data_ptr() == q_nope.untyped_storage().data_ptr()
        and q_rope.storage_offset() == q_nope.storage_offset() + latent
    ):
        return None
    return q_nope.as_strided((tokens, heads, width), q_nope.stride())


def _speculative() -> bool:
    """Whether sglang decodes speculatively (MTP): its draft steps stay on sglang."""
    from sglang.srt.runtime_context import get_spec

    return get_spec().speculative_algorithm is not None
