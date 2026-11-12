# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSPARK target verify under decode context parallelism that sglang's aiter backend runs on aiter's ASM MLA
(``_forward_verify_asm_cprr``): AKLR's ``flash_mla_with_kvcache`` op over this rank's shard.

A request's ``q_len`` draft rows attend to this rank's tokens of the request, the prefix and the draft window alike.
Tokens are dealt to the ranks one by one by global position, and row ``i`` sees the positions up to its own. sglang
plans each row's token table (``verify_token_table``: the request's shard, repeated per row) and the requests' global
lengths (``global_kv_indptr``); the route turns them into each row's count of visible tokens on this rank. aiter pads
the gathered heads up to a head count its kernel schedules; the op takes them as they are."""

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla.api import flash_mla_with_kvcache
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []

# the lse sglang gives a row that sees no token of this rank, which the cross-rank merge weighs as nothing
_EMPTY_LSE = -1e30


@sglang_route(
    "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend._forward_verify_asm_cprr",
    versions=("0.5.22",),
)
def _forward_verify_asm_cprr(self, q, layer, k_descale, n_rows):
    from sglang.srt.layers.attention import aiter_backend

    fm = self.forward_metadata
    table, global_indptr = fm.verify_token_table, fm.global_kv_indptr
    if table is None or global_indptr is None:
        return FallbackReason.OUT_OF_CONTRACT
    q_len = fm.max_q_len
    batch = n_rows // q_len
    dcp, rank = self.dcp_world_size, aiter_backend.get_parallel().attn_dcp_rank
    # row i of a request of global length g sits at position g - q_len + i; this rank holds the positions
    # rank, rank + dcp, ..., so (position - rank) // dcp + 1 of them up to there
    lengths = (global_indptr[1 : batch + 1] - global_indptr[:batch]).to(torch.int32)
    positions = lengths[:, None] - q_len + torch.arange(q_len, dtype=torch.int32, device=lengths.device)
    seen = torch.where(
        positions >= rank,
        torch.div(positions - rank, dcp, rounding_mode="floor") + 1,
        torch.zeros_like(positions),
    ).reshape(n_rows)
    num_heads = layer.tp_q_head_num
    result = flash_mla_with_kvcache(
        q.reshape(n_rows, 1, num_heads, layer.qk_head_dim),
        self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(-1, 1, 1, layer.qk_head_dim),
        block_table=table[:n_rows],
        cache_seqlens=seen,
        head_dim_v=layer.v_head_dim,
        sm_scale=layer.scaling,
        kv_cache_format="fp8_e4m3",
        kv_scale=self._resolve_fp8_kv_scale_float(layer, k_descale),
        return_lse=True,
    )
    if isinstance(result, FallbackReason):
        return result
    out, lse = result
    lse = lse.view(n_rows, num_heads).masked_fill(seen[:, None] == 0, _EMPTY_LSE)
    return out.view(n_rows, num_heads, layer.v_head_dim), lse
