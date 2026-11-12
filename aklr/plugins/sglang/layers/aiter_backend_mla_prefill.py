# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA prefill sglang's aiter backend runs on aiter's fp8 path (``srt/layers/attention/aiter_backend.py``):
AKLR's ``mla_prefill_varlen`` op.

sglang plans aiter's work once a forward, then at each MLA layer tiles the heads up to one aiter's
reduce instantiates, casts q, k and v to fp8 and runs aiter's ASM prefill and reduce. The route
takes the layer's call whole: the offsets from the plan, the rest from the call."""

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla_prefill.api import mla_prefill_varlen
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route(
    "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend._mla_fp8_prefill_attn_ps",
    versions=("0.5.22",),
)
def _mla_fp8_prefill_attn_ps(self, q, k, v, layer, ps):
    # the chunked prefix's passes need the LSE, and its prefix chunks attend without the causal mask
    if ps.need_lse or not ps.is_causal:
        return FallbackReason.OUT_OF_CONTRACT
    # without a prefix the keys are the queries' own rows
    offsets_k = ps.kv_indptr if k.shape[0] != q.shape[0] else None
    # kv_b's output leaves v a strided view: ask before copying it whole
    stand_in = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    args = {"qkv_fp8": True, "cu_seqlens_k": offsets_k, "sm_scale": layer.scaling}
    if (reason := mla_prefill_varlen.fallback_reason(q, k, stand_in, ps.qo_indptr, **args)) is not None:
        return reason
    output = mla_prefill_varlen(q, k, v.contiguous(), ps.qo_indptr, **args)
    return output if isinstance(output, FallbackReason) else (output, None)
