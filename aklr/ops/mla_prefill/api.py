# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.runtime.dispatch import api

from .contract import MLA_PREFILL_VARLEN, SPARSE_PREFILL_MXFP8


@api(SPARSE_PREFILL_MXFP8)
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
    *,
    softmax_scale=512**-0.5,
): ...


@api(MLA_PREFILL_VARLEN)
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
