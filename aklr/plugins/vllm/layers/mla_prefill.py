# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's MLA prefill over cached context on its ROCM_AITER_FA prefill backend
(``v1/attention/backends/mla/prefill/aiter_flash_attn.py``): the causal pass over a step's new tokens and the pass
over each context chunk, each returning (output, lse) for ``merge_attn_states``, on AKLR's ``mla_prefill_varlen_lse``
op in fp8 (q, k and v rounded to float8_e4m3fn) instead of aiter's bf16 flash attention.

vLLM reads the lse as [heads, tokens]; the op writes [tokens, heads], handed over transposed. Where AKLR declines a
pass it runs on vLLM's bf16 flash attention, and a batch of several sequences runs on bf16 flash attention inside the
op's kernel; both are logged once per kind. A prefill without cached context stays on vLLM's own fp8 path, and so do
passes under 256 query rows (a decode step's extends)."""

from __future__ import annotations

import logging

import torch

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.mla_prefill.api import mla_prefill_varlen_lse
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []

_LOGGER = logging.getLogger(__name__)
_LOGGED: set[tuple] = set()
_VERSIONS = ("0.33.0rc1.dev910",)
# Fewer query rows stay on vLLM: a decode step's extends (a few tokens over the whole context) reach these passes
# too, and the op's fp8 kernel, built for prefill chunks, casts the whole context and plans per shape on each call.
# The op's workloads start at 257 rows.
_MIN_ROWS = 256


def _log_once(key: tuple, message: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _LOGGER.warning(message, *args)


def _aligned(offsets):
    """offsets, or a copy where its data does not start on the 16 bytes AKLR's dispatch requires."""
    return offsets if offsets is None or offsets.data_ptr() % 16 == 0 else offsets.clone()


def _lse_pass(q, k, v, offsets_q, offsets_k, causal, scale):
    """(output, lse as [heads, tokens]) of one pass on mla_prefill_varlen_lse, or why AKLR declines it."""
    if q.shape[0] < _MIN_ROWS:
        _log_once(
            ("small", causal),
            "AKLR mla_prefill_varlen_lse: passes under %d query rows stay on vLLM",
            _MIN_ROWS,
        )
        return FallbackReason.OUT_OF_CONTRACT
    args = {"qkv_fp8": True, "causal": causal, "cu_seqlens_k": offsets_k, "sm_scale": scale}
    stand_in = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    reason = mla_prefill_varlen_lse.fallback_reason(q, k, stand_in, offsets_q, **args)
    if reason is None:
        if offsets_q.shape[0] > 2:
            _log_once(
                ("several", causal),
                "AKLR mla_prefill_varlen_lse: a batch of several sequences (causal=%s) runs on bf16 flash attention "
                "inside the AKLR kernel, not fp8",
                causal,
            )
        result = mla_prefill_varlen_lse(q, k, v.contiguous(), offsets_q, **args)
        if not isinstance(result, FallbackReason):
            output, lse = result
            return output, lse.t().contiguous()
        reason = result
    _log_once(
        (reason, causal, q.shape[0], k.shape[0]),
        "AKLR mla_prefill_varlen_lse declined (%s) a prefill pass (causal=%s, %d queries, %d keys): it runs on "
        "vLLM's bf16 flash attention",
        reason.value,
        causal,
        q.shape[0],
        k.shape[0],
    )
    return reason


@vllm_route(
    "vllm.model_executor.layers.attention.mla_attention.MLACommonBaseImpl.forward_mha",
    "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn.AiterFlashAttnPrefillBackend.run_prefill_new_tokens",
    versions=_VERSIONS,
)
def _run_prefill_new_tokens(self, q, k, v, return_softmax_lse, out=None, output_scale=None):
    # with cached context only (the lse merges it); a caller's buffer or quantized output stays on vLLM
    if not return_softmax_lse or out is not None or output_scale is not None or q.dtype != torch.bfloat16:
        return FallbackReason.OUT_OF_CONTRACT
    return _lse_pass(q, k, v, _aligned(self._prefill_metadata.query_start_loc), None, True, self.scale)


@vllm_route(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLAImpl._compute_prefill_context",
    "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn.AiterFlashAttnPrefillBackend.run_prefill_context_chunk",
    versions=_VERSIONS,
)
def _run_prefill_context_chunk(self, chunk, q, k, v, out=None):
    if out is not None or q.dtype != torch.bfloat16:
        return FallbackReason.OUT_OF_CONTRACT
    return _lse_pass(q, k, v, _aligned(chunk.query_start_loc), _aligned(chunk.cu_seq_lens), False, self.scale)
