# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.attn_residual.contract import ADD_ATTN_RES, ATTN_RES

__all__ = ["add_attn_res", "attn_res"]


@api(ATTN_RES)
def attn_res(
    prefix,
    bank,
    score_weight,
    norm_weight,
    score_eps,
    norm_eps,
    n_valid_rows=8,
    write_bank=False,
    normalized=None,
): ...


@api(ADD_ATTN_RES)
def add_attn_res(
    prefix,
    addend,
    bank,
    score_weight,
    norm_weight,
    score_eps,
    norm_eps,
    n_valid_rows=8,
    write_bank=False,
    normalized=None,
    prefix_out=None,
): ...


@sglang_route("sglang.kernels.ops.attention.attn_res_hip.attn_res_hip", versions=("0.5.22",))
def _attn_res_hip(
    prefix_sum,
    bank,
    cw,
    ow,
    out,
    nvb,
    score_eps,
    out_eps,
    *,
    addend=None,
    prefix_out=None,
    write_prefix=False,
):
    # without ow the mix itself is the result (DSpark's capture of the stream), which no contract
    # states; an addend without prefix_out is sglang's error to raise
    if ow is None or (addend is not None and prefix_out is None):
        return FallbackReason.OUT_OF_CONTRACT
    common = {"n_valid_rows": nvb, "write_bank": write_prefix, "normalized": out}
    if addend is None:
        result = attn_res(prefix_sum, bank, cw, ow, score_eps, out_eps, **common)
    else:
        result = add_attn_res(
            prefix_sum, addend, bank, cw, ow, score_eps, out_eps, prefix_out=prefix_out, **common
        )
    # sglang's function returns nothing: the kernel wrote out, prefix_out and the bank
    return result if isinstance(result, FallbackReason) else None
