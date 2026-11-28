# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.layers.mla_query_heads.contract import MLA_QUERY_HEADS

__all__ = ["mla_query_heads"]


@api(MLA_QUERY_HEADS)
def mla_query_heads(q, out_heads=128, out_dtype=None, out=None): ...


@vllm_route(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLAImpl._forward_mqa",
    "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLAHelper.get_mla_padded_q",
    versions=("0.34.0",),
)
def _get_mla_padded_q(num_heads, q, target_heads=None):
    """vLLM pads the decode query to the head count aiter's decode kernel takes. Where that count is no multiple of the
    query's and the rows are dense (Kimi-K3's 96 gathered heads for the 128-head kernel), it tiles the heads, two torch
    launches: the op is that tiling. Every other way vLLM pads (no padding, a repeat of each head, a view) stays
    vLLM's."""
    if (
        target_heads is None
        or q.dim() != 3
        or q.shape[1] != num_heads
        or target_heads <= num_heads
        or target_heads % num_heads == 0
        or q.stride(0) == target_heads * q.stride(1)
        or not q.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    return mla_query_heads(q, out_heads=target_heads)
