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
# leave the machine idle (the 2 MiB projection read at 303 GB/s, 4% of the device), and a narrower slice
# fills it: 16 columns, 320 programs, is 1.3-1.6x on MI355X. The op's schemas carry the count as n_slices.
NARROW_SLICE_WIDTH, WIDE_SLICE_WIDTH, NARROW_MAX_ROWS = 16, 64, 48


def slice_width(n_rows: int, has_post: bool) -> int:
    """16 columns (320 programs for DeepSeek-V4.1) for a pending post up to NARROW_MAX_ROWS rows,
    sglang's 64 otherwise.

    The count is AKLR's to choose -- sglang reads it back from part_sq.shape[0] -- but the norm that
    reduces the partials walks every slice, so a narrower one only pays where the boundary itself has
    enough work. With a pending post it is 1.52x at 6 rows, 1.46x at 12 and 1.36x at 24 (aklr validate,
    cold graph replay, MI355X), and end to end at TP4 concurrency 32 the op carries +2.7% of
    throughput. Without a post -- only TP4 at the rows its fused all-reduce covers, so concurrency 1 --
    the boundary is 1.15x alone but the step loses 1.2%, so sglang's width stays. Past
    NARROW_MAX_ROWS the rows already fill the machine (0.89x at 192)."""
    return NARROW_SLICE_WIDTH if has_post and n_rows <= NARROW_MAX_ROWS else WIDE_SLICE_WIDTH


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
        or h % slice_width(m, has_post)
        or not residual.is_contiguous()
        or (has_post and not x.is_contiguous())
    ):
        return FallbackReason.OUT_OF_CONTRACT
    slices, mix = h // slice_width(m, has_post), hc_fn.shape[0]
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
