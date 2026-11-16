# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import sys

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend_hip.contract import DSA_SPARSE_MLA_DCP_PARTIAL

__all__ = ["dsa_sparse_mla_dcp_partial"]

_MODULE = "sglang.srt.layers.attention.dsa.dsa_backend_hip"


@api(DSA_SPARSE_MLA_DCP_PARTIAL)
def dsa_sparse_mla_dcp_partial(
    q_nope, q_rope, kv_cache, indices, topk_length, sm_scale=192**-0.5, output=None, lse=None
): ...


@sglang_route(f"{_MODULE}.forward_triton_dcp_partial", versions=("0.5.24.dev809",))
def _forward_triton_dcp_partial(backend, q_nope, q_rope, kv_cache, page_table_1, layer):
    """A DCP rank's decode or verify partial as one op call where sglang runs its gfx950 kernel:
    sglang's localization of the global top-k, then the op; (o [rows, heads, 512], natural-log lse
    [rows, heads]) as the target returns them. Every other partial stays sglang's."""
    module = sys.modules[_MODULE]
    if not (
        module._gfx950_decode_enabled()
        and module.supports_sparse_mla_decode_gfx950(q_nope, kv_cache, layer.v_head_dim)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    pool = kv_cache.view(torch.uint8)
    arguments = {"kv_cache": pool.view(-1, pool.shape[-1]), "sm_scale": layer.scaling}
    # what the op is given below: q made contiguous, dcp_localize_compact's outputs
    stand_ins = {
        "q_nope": q_nope.new_empty(q_nope.shape),
        "q_rope": q_rope.new_empty(q_rope.shape),
        "indices": torch.empty_like(page_table_1),
        "topk_length": page_table_1.new_empty(page_table_1.shape[0], dtype=torch.int32),
    }
    reason = dsa_sparse_mla_dcp_partial.fallback_reason(**stand_ins, **arguments)
    if reason is not None:
        return reason
    indices, topk_length = module.dcp_localize_compact(page_table_1)
    # the DCP all-gather leaves q as transposed views of one buffer: the op takes it row-major
    return dsa_sparse_mla_dcp_partial(
        q_nope=q_nope.contiguous(),
        q_rope=q_rope.contiguous(),
        indices=indices,
        topk_length=topk_length,
        **arguments,
    )
