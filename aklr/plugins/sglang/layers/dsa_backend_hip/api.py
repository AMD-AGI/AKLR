# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_backend_hip.contract import DSA_SPARSE_MLA_FP8_DS_PREFILL, KV_ROW_BYTES

__all__ = ["dsa_sparse_mla_fp8_ds_prefill"]


@api(DSA_SPARSE_MLA_FP8_DS_PREFILL)
def dsa_sparse_mla_fp8_ds_prefill(
    q_nope, q_rope, kv_cache, indices, sm_scale=192**-0.5, output=None, lse=None
): ...


# sglang's own gfx950 kernel (python/sglang/kernels/jit/csrc/attention/sparse_mla_prefill_gfx950.cuh);
# triton_sparse_mla_fwd imports it at each call, once supports_sparse_mla_prefill_gfx950 holds
@sglang_route(
    "sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950.sparse_mla_prefill_gfx950",
    versions=("0.5.24.dev809",),
)
def _sparse_mla_prefill_gfx950(q_nope, q_rope, kv, indices, sm_scale, d_v=512, return_lse=False, stages=2):
    """One op call, as the target's own wrapper hands its kernel the call: the pool as [slots, 656] bytes, the
    indices as [tokens, topk] int32; q's two parts dense; ([1, tokens, heads, 512] bf16[, lse]) as the target
    returns them. sglang calls it only where supports_sparse_mla_prefill_gfx950 holds (656-byte rows, bf16 q,
    at most 16 heads). The pipeline depth (stages) is the target's own choice, not the op's."""
    if d_v != 512:
        return FallbackReason.OUT_OF_CONTRACT
    rows = indices.squeeze(1) if indices.dim() == 3 else indices
    arguments = {
        "q_nope": q_nope,
        "q_rope": q_rope,
        "kv_cache": kv.view(torch.uint8).reshape(kv.shape[0], KV_ROW_BYTES),
        "indices": rows if rows.dtype == torch.int32 else rows.to(torch.int32),
        "sm_scale": float(sm_scale),
    }
    # sglang hands q's two parts as views of one [tokens, heads, 576] q, which the op takes dense: ask it
    # with dense stand-ins before copying them
    strided = [n for n in ("q_nope", "q_rope", "indices") if not arguments[n].is_contiguous()]
    if strided:
        reason = dsa_sparse_mla_fp8_ds_prefill.fallback_reason(
            **arguments
            | {n: torch.empty_like(arguments[n], memory_format=torch.contiguous_format) for n in strided}
        )
        if reason is not None:
            return reason
        arguments |= {n: arguments[n].contiguous() for n in strided}
    result = dsa_sparse_mla_fp8_ds_prefill(**arguments)
    if isinstance(result, FallbackReason):
        return result
    output, token_lse = result
    return (output.unsqueeze(0), token_lse) if return_lse else output.unsqueeze(0)
