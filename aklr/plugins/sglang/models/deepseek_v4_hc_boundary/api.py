# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4_hc_boundary.contract import (
    DEEPSEEK_V4_HC_BOUNDARY,
    DEEPSEEK_V4_HC_BOUNDARY_POST,
)

__all__ = ["deepseek_v4_hc_boundary", "deepseek_v4_hc_boundary_post"]

# Columns of every copy one program owns. sglang slices 64 (_HC_BOUNDARY_BLOCK_K) and reads the count
# back from part_sq.shape[0], so the slicing is AKLR's to choose: at decode rows 80 programs of 2 warps
# leave the machine idle, and a narrower slice fills it. The op's schemas carry this count as n_slices.
SLICE_WIDTH = 64


@api(DEEPSEEK_V4_HC_BOUNDARY)
def deepseek_v4_hc_boundary(residual, pre_prev, hc_fn, y=None, part_mix=None, part_sq=None): ...


@api(DEEPSEEK_V4_HC_BOUNDARY_POST)
def deepseek_v4_hc_boundary_post(
    x, residual, post, comb, pre_prev, hc_fn, residual_out=None, y=None, part_mix=None, part_sq=None
): ...


@sglang_route("sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials", versions=("0.5.23",))
def _hc_boundary_partials(
    x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, *, hc_mult, prefill=None
):
    """DeepSeek-V4's mHC boundary below sglang's prefill rows, with the previous pre's collapse, as one op call into
    sglang's residual_out and y; the partials keep sglang's 64-column slices, so the norm launch that reduces them
    is sglang's own. The gfx950 prefill kernel's rows and the boundary without a collapse stay on sglang."""
    import torch
    from sglang.kernels.ops.layernorm import mhc_boundary_hip as hip

    m, _, h = residual.shape
    has_post = x is not None
    if (
        prefill
        or (prefill is None and hip._hc_boundary_use_prefill(m, has_post, pre_prev is not None, x))
        or pre_prev is None
        or y is None
        or m == 0
        or h % SLICE_WIDTH
        or not residual.is_contiguous()
        or (has_post and not x.is_contiguous())
    ):
        return FallbackReason.OUT_OF_CONTRACT
    slices, mix = h // SLICE_WIDTH, hc_fn.shape[0]
    part_mix = torch.empty((slices, m, mix), dtype=torch.float32, device=residual.device)
    part_sq = torch.empty((slices, m), dtype=torch.float32, device=residual.device)
    if has_post:
        out = deepseek_v4_hc_boundary_post(
            x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, part_mix, part_sq
        )
    else:
        out = deepseek_v4_hc_boundary(residual, pre_prev, hc_fn, y, part_mix, part_sq)
    if isinstance(out, FallbackReason):
        return out
    return out[-2], out[-1]
