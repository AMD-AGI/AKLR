# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.mqa_logits.contract import FP8_MQA_LOGITS, ROW_ALIGN

__all__ = ["fp8_mqa_logits"]


@api(FP8_MQA_LOGITS)
def fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends, logits, no_values=None): ...


# sglang imports it under aiter's compatibility name inside Indexer._get_topk_ragged, the
# prefill (extend without speculation) logits of the DSA indexer
@sglang_route("aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits", versions=("0.5.22", "0.5.23"))
def _fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends, clean_logits=True):
    """The logits as one op call into the [rows, keys rounded up to 256] float32 buffer aiter
    allocates, returned as aiter returns it: its first keys columns; a call asking for -inf
    outside the windows stays on aiter."""
    import torch

    if clean_logits:
        return FallbackReason.OUT_OF_CONTRACT
    # the chunked path slices the window bounds at any row; ops take 16-byte aligned tensors
    cu_starts, cu_ends = (t if t.data_ptr() % 16 == 0 else t.clone() for t in (cu_starts, cu_ends))
    n_keys = KV.shape[0]
    width = -(-n_keys // ROW_ALIGN) * ROW_ALIGN
    logits = torch.empty((Q.shape[0], width), dtype=torch.float32, device=Q.device)
    result = fp8_mqa_logits(
        q=Q,
        kv=KV,
        kv_scales=kv_scales,
        weights=weights,
        cu_starts=cu_starts,
        cu_ends=cu_ends,
        logits=logits,
    )
    if isinstance(result, FallbackReason):
        return result
    return logits[:, :n_keys]
