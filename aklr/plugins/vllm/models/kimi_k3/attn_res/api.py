# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.attn_res.contract import KIMI_K3_ATTN_RES, KIMI_K3_ATTN_RES_FP8

__all__ = ["kimi_k3_attn_res", "kimi_k3_attn_res_fp8"]


@api(KIMI_K3_ATTN_RES)
def kimi_k3_attn_res(
    prefix,
    bank,
    score_norm_weight,
    score_proj_weight,
    n_valid_rows=8,
    write_bank=False,
    addend=None,
    norm_weight=None,
    score_eps=1e-5,
    norm_eps=1e-5,
    out_dtype=None,
    out=None,
): ...


@api(KIMI_K3_ATTN_RES_FP8)
def kimi_k3_attn_res_fp8(
    prefix,
    bank,
    score_norm_weight,
    score_proj_weight,
    n_valid_rows=8,
    write_bank=False,
    addend=None,
    norm_weight=None,
    score_eps=1e-5,
    norm_eps=1e-5,
): ...


@vllm_route(
    "vllm.models.kimi_k3.amd.linear._apply_attn_res",
    "vllm.models.kimi_k3.amd.linear.attn_res",
    versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
)
def _attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
):
    """Every attention residual of Kimi-K3's decoder layers and the one before its final norm, and with quant_dtype
    the per-token fp8 out for the linear after it (as (out, scale)). vLLM writes the bank only at row num_blocks."""
    # fnuz fp8 (vLLM's 224 for its largest value) no contract states
    if block_write_idx not in (-1, num_blocks) or quant_dtype not in (None, torch.float8_e4m3fn):
        return FallbackReason.OUT_OF_CONTRACT
    return (kimi_k3_attn_res if quant_dtype is None else kimi_k3_attn_res_fp8)(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        n_valid_rows=num_blocks,
        write_bank=block_write_idx >= 0,
        addend=delta,
        norm_weight=output_norm_weight,
        score_eps=eps,
        norm_eps=output_norm_eps,
    )
