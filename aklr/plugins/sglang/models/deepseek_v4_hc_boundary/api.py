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

# Columns of every copy one program owns, by row count. sglang slices 64 and reads the count back
# from part_sq.shape[0], so the slicing is AKLR's to choose -- but the norm launch that follows
# reduces every slice, so what has to be minimised is the pair, not the boundary. Measured on MI355X
# (work_tp4/pair_{nopost,post}.log, the pair against sglang's own pair):
#
#   rows            6      12      48     192     384
#   32 columns   1.14x   1.14x   1.04x   0.96x   1.02x
#   64 columns   1.08x   1.09x   1.09x   1.12x   1.04x
#   128 columns  0.89x   0.89x   0.90x   1.01x   1.15x
#
# (with a pending post; without one the same widths win, at 1.07x / 1.07x / 1.00x.) Tuning the
# boundary alone picks 16 columns everywhere and loses the step: it is 1.3-1.6x on its own and up to
# 0.89x on the pair.
# 128 columns only with a pending post: without one the boundary has too little work to fill that
# few programs (0.67-0.93x on its own past 256 rows), and a width the op then declines would leave
# sglang's own kernel running on it.
_SLICE_WIDTHS = ((24, 32), (256, 64), (1 << 30, 128))


def slice_width(n_rows: int, has_post: bool) -> int:
    """The hidden columns one program owns at this row count."""
    width = next(width for limit, width in _SLICE_WIDTHS if n_rows <= limit)
    return width if has_post else min(width, 64)


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
    sglang's residual_out and y; the partials are sliced by ``slice_width``, which the norm launch that
    reduces them reads back from their shape. The gfx950 prefill kernel's rows and the boundary without a collapse stay on sglang."""
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
