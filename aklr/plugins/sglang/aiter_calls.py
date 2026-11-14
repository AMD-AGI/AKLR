# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routes on the aiter functions sglang calls, for every model that reaches them.

sglang looks ``fused_moe`` up in aiter at each call, so hooking aiter's name reaches every call. The
route takes the parameters of CI's aiter, followed by the ones the aiter in sglang's release image
adds: sglang passes ``output`` when that aiter has it, for MoRI EP's combine buffer. The arguments
go by name to :mod:`aklr.plugins.aiter`.

sglang calls ``mm`` on aiter's ``tgemm``, so hooking aiter's name reaches every call; sglang's hook registry
also rebinds the names other modules imported, as the unquantized linear layers import aiter's Triton GEMM,
which they call for at most 64 rows where the weight's K is at most 2048 (at most 512 rows where K is at most
512). Those routes take the parameters of the aiter in the sglang release's image.

sglang's hook registry rebinds aiter's ``mla_decode_fwd`` where sglang's modules imported it, so that route
reaches every call; its parameters are those of the aiter in the sglang release's image."""

from __future__ import annotations

import torch

from aklr.plugins import aiter as aklr_aiter
from aklr.plugins.sglang.hook import sglang_route

__all__: list[str] = []


@sglang_route("aiter.fused_moe.fused_moe", versions=("0.5.22",))
def _fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask=None,
    activation=None,
    quant_type=None,
    doweight_stage1=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_size_M=None,
    num_local_tokens=None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode="separated",
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
    shared_expert_id=-1,
    stage2_scatter=None,
    output=None,
    quant_type_a=None,
    quant_dtype_a=None,
    quant_dtype_a2=None,
):
    return aklr_aiter.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=gate_mode,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_expert_id=shared_expert_id,
        stage2_scatter=stage2_scatter,
        output=output,
        quant_type_a=quant_type_a,
        quant_dtype_a=quant_dtype_a,
        quant_dtype_a2=quant_dtype_a2,
    )


@sglang_route("aiter.tuned_gemm.tgemm.mm", versions=("0.5.22",))
def _tgemm_mm(inp, weights, bias=None, otype=None, scale_a=None, scale_b=None, scale_c=None):
    return aklr_aiter.mm(
        inp, weights, bias=bias, otype=otype, scale_a=scale_a, scale_b=scale_b, scale_c=scale_c
    )


# aiter.ops.triton.gemm_a16w16 is a backward-compatible path whose module is a second copy; the
# unquantized linear layers' name for the function is the defining module's
@sglang_route("aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16", versions=("0.5.22",))
def _triton_gemm_a16w16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    config=None,
    activation=None,
    skip_reduce=False,
    kernel_type="bandwidth_bound",
    backend=None,
):
    return aklr_aiter.gemm_a16w16(
        x, w, bias=bias, dtype=dtype, y=y, activation=activation, skip_reduce=skip_reduce
    )


# the MLA decode under decode context parallelism (aiter_backend's DCP decode and DSpark verify)
@sglang_route("aiter.mla.mla_decode_fwd", versions=("0.5.22",))
def _mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size=1,
    nhead_kv=1,
    sm_scale=None,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
    intra_batch_mode=False,
    return_logits=False,
    return_lse=False,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    causal=True,
):
    return aklr_aiter.mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        page_size=page_size,
        nhead_kv=nhead_kv,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        num_kv_splits_indptr=num_kv_splits_indptr,
        q_scale=q_scale,
        kv_scale=kv_scale,
        return_logits=return_logits,
        return_lse=return_lse,
        g_kv_indptr=g_kv_indptr,
        cp_world_size=cp_world_size,
    )
