# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.ops.mla import flash_mla_with_kvcache
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.mla_decode.contract import KIMI_K3_MLA_DECODE

__all__ = ["kimi_k3_mla_decode"]


@api(KIMI_K3_MLA_DECODE)
def kimi_k3_mla_decode(q, kv_cache, kv_indptr, kv_indices, sm_scale=192**-0.5, output=None): ...


@sglang_route(
    "sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode", versions=("0.5.22",)
)
def _forward_decode(
    self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None, score_mod=None, aux_tensors=None
):
    """MLA decode with a page-size-1 latent cache and no extras as one op, after sglang's own KV
    write; every other decode runs sglang's forward_decode."""
    if (
        not self.use_mla
        or self.dcp_size > 1
        or self.page_size != 1
        or self.enable_deterministic
        or sinks is not None
        or score_mod is not None
        or aux_tensors is not None
        or layer.k_scale is not None
        or layer.tp_k_head_num != 1
        or layer.logit_cap
        or (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
        or (layer.xai_temperature_len or -1) > 0
    ):
        return FallbackReason.OUT_OF_CONTRACT
    metadata = self.forward_metadata
    arguments = {
        "q": q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
        "kv_cache": self.token_to_kv_pool.get_key_buffer(layer.layer_id),
        "kv_indptr": metadata.kv_indptr,
        "kv_indices": metadata.kv_indices,
        "sm_scale": layer.scaling,
    }
    reason = kimi_k3_mla_decode.fallback_reason(**arguments)
    if reason is not None:
        return reason
    if save_kv_cache:
        from sglang.srt.layers.attention.triton_backend import KVWriteLoc

        loc = KVWriteLoc.for_batch(forward_batch, swa_loc=metadata.swa_out_cache_loc)
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)
    output = kimi_k3_mla_decode(**arguments)
    if isinstance(output, FallbackReason):
        raise RuntimeError(f"kimi_k3_mla_decode declined after its KV write: {output}")
    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


# the lse sglang gives a row that sees none of this rank's tokens; the cross-rank merge weighs it as nothing
_EMPTY_LSE = -1e30


@sglang_route(
    "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend._forward_verify_asm_cprr",
    versions=("0.5.22",),
)
def _forward_verify_asm_cprr(self, q, layer, k_descale, n_rows):
    """DCP target verify on aiter's cp schedule as one op call: each query row reads this rank's shard
    of its request, prefix and window, up to its own global position, which is the causal mask.
    Returns the partial and its natural-log lse for sglang's cross-rank merge."""
    import torch

    if layer.logit_cap or self.page_size != 1:
        return FallbackReason.OUT_OF_CONTRACT
    metadata = self.forward_metadata
    heads, table = layer.tp_q_head_num, metadata.verify_token_table
    kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
    if kv_cache.dtype == torch.uint8:
        kv_cache = kv_cache.view(self.kv_cache_dtype)
    arguments = {
        "q": q.view(n_rows, 1, heads, layer.qk_head_dim),
        "kv_cache": kv_cache.view(-1, 1, 1, layer.qk_head_dim),
        "block_table": table,
        "head_dim_v": layer.v_head_dim,
        "sm_scale": layer.scaling,
        "kv_cache_format": "fp8_e4m3",
        "kv_scale": self._resolve_fp8_kv_scale_float(layer, k_descale),
        "return_lse": True,
    }
    stand_in = table.new_empty((n_rows,))
    reason = flash_mla_with_kvcache.fallback_reason(cache_seqlens=stand_in, **arguments)
    if reason is not None:
        return reason
    lens, empty = _causal_shard_lens(self, metadata, n_rows)
    output, lse = flash_mla_with_kvcache(cache_seqlens=lens, **arguments)
    return output.view(n_rows, heads, -1), lse.view(n_rows, heads).masked_fill_(empty[:, None], _EMPTY_LSE)


def _causal_shard_lens(backend, metadata, n_rows):
    """Per query row, this rank's tokens at or before the row's global position: with global position p
    on rank p % size, row i of a request of global length L (window included) reads its first
    ceil((L - q_len + i + 1 - rank) / size) entries of the shard. The kernel reads a row's first table
    entry even at length 0, which sglang leaves unwritten for a request with an empty shard, so those
    rows point it at the pool's padding slot 0. Computed once per forward, every layer reads the same;
    a cuda graph capture recomputes them, since sglang warms up on the capture's metadata first."""
    import torch
    from sglang.srt.runtime_context import get_parallel

    capturing = torch.cuda.is_current_stream_capturing()
    cached = getattr(metadata, "_aklr_causal_shard_lens", None)
    if cached is not None and cached[0] == capturing:
        return cached[1:]

    size, rank, q_len = backend.dcp_world_size, get_parallel().attn_dcp_rank, metadata.max_q_len
    indptr = metadata.global_kv_indptr
    bs = n_rows // q_len
    visible = (indptr[1 : bs + 1] - indptr[:bs] - q_len + 1 - rank + size - 1).unsqueeze(1)
    visible = visible + torch.arange(q_len, dtype=indptr.dtype, device=indptr.device)
    lens = (visible.clamp_min(0) // size).view(n_rows).to(torch.int32)
    empty = lens == 0
    metadata.verify_token_table[:n_rows, 0].masked_fill_(empty, 0)
    metadata._aklr_causal_shard_lens = (capturing, lens, empty)
    return lens, empty
