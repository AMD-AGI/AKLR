# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
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
