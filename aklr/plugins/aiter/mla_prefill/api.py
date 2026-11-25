# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import aklr_api
from aklr.plugins.aiter.mla_prefill.schema import (
    MLA_PREFILL_VARLEN_LSE_VARIANTS,
    MLA_PREFILL_VARLEN_VARIANTS,
    SPARSE_PREFILL_MXFP8,
)

__all__ = ["mla_prefill_varlen", "mla_prefill_varlen_lse", "sparse_prefill_mxfp8"]


@aklr_api(SPARSE_PREFILL_MXFP8)
def sparse_prefill_mxfp8(
    q_nope,
    q_rope,
    prefix_nope,
    prefix_rope,
    prefix_indices,
    prefix_indptr,
    extend_nope,
    extend_rope,
    extend_indices,
    extend_indptr,
    sinks,
    softmax_scale=512**-0.5,
    output=None,
): ...


@aklr_api(MLA_PREFILL_VARLEN_VARIANTS)
def mla_prefill_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    qkv_fp8=True,
    cu_seqlens_k=None,
    sm_scale=None,
    output=None,
): ...


@aklr_api(MLA_PREFILL_VARLEN_LSE_VARIANTS)
def mla_prefill_varlen_lse(
    q,
    k,
    v,
    cu_seqlens_q,
    qkv_fp8=True,
    causal=True,
    cu_seqlens_k=None,
    sm_scale=None,
    output=None,
    lse=None,
): ...
