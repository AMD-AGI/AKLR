# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.attn_res.contract import KIMI_K3_ATTN_RES

__all__ = ["kimi_k3_attn_res"]


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


@vllm_route(
    "vllm.models.kimi_k3.amd.linear._apply_attn_res",
    "vllm.models.kimi_k3.amd.linear.attn_res",
    versions=("0.33.0rc1.dev910",),
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
    """Every attention residual of Kimi-K3's decoder layers and the one before its final norm.
    vLLM writes the bank only at row num_blocks."""
    # the fp8 output for a linear that takes per-token fp8 input, which no contract states
    if quant_dtype is not None or block_write_idx not in (-1, num_blocks):
        return FallbackReason.OUT_OF_CONTRACT
    return kimi_k3_attn_res(
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
