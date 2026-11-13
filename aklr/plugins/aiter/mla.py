# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's ASM MLA decode, ``aiter.mla.mla_decode_fwd``, served by AKLR's ``mla_decode_fwd`` op."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla.api import mla_decode_fwd as aklr_mla_decode_fwd

__all__ = ["mla_decode_fwd"]


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    *,
    page_size,
    nhead_kv,
    sm_scale,
    logit_cap,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    return_logits,
    return_lse,
    g_kv_indptr,
    cp_world_size,
):
    """AKLR's ``(o, lse)`` for one query a request over a page-size-1 pool, or why aiter computes
    it. aiter returns its split scratch first, which neither framework reads; AKLR returns ``o``
    there. aiter's schedule (work and reduce maps, splits, batch modes) is not taken: it only
    orders aiter's own kernel's work."""
    if (
        max_seqlen_q != 1
        or page_size != 1
        or nhead_kv != 1
        or logit_cap > 0
        or num_kv_splits_indptr is not None
        or return_logits
        or not return_lse
        or q_scale is None
        or kv_scale is None
        or g_kv_indptr is not None
        or cp_world_size != 1
        or q.dim() != 3
        or q.shape[0] != kv_indptr.shape[0] - 1
    ):
        return FallbackReason.OUT_OF_CONTRACT
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return aklr_mla_decode_fwd(
        q, kv_buffer, kv_indptr, kv_indices, q_scale, kv_scale, sm_scale=float(sm_scale), o=o
    )
