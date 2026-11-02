# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Under DP attention, the zeroing of a DP gather's pad rows in one launch (sglang's arange,
compare and masked fill per MoE layer). ``AKLR_DSV41_DP_PAD_MASK=0`` keeps sglang's."""

from __future__ import annotations

import functools
import os

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV = "AKLR_DSV41_DP_PAD_MASK"
TRANSPORT = "sglang.srt.models.deepseek_common.moe_transport"
_NOTED = False

SGLANG_READS = {
    "srt/models/deepseek_common/moe_transport.py": ("_mask_dp_pad_rows(local_hidden_states, forward_batch)",),
    "srt/model_executor/forward_batch_info.py": ("num_token_non_padded",),
}


@functools.cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_pad_rows(x, n_real, stride, cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        if row >= tl.load(n_real):
            c = tl.arange(0, BLOCK)
            for c0 in range(0, cols, BLOCK):
                tl.store(
                    x + row * stride + c0 + c, tl.zeros((BLOCK,), x.dtype.element_ty), mask=c0 + c < cols
                )

    return mask_pad_rows


def _fits(x: torch.Tensor, n_real) -> bool:
    return (
        isinstance(n_real, torch.Tensor)
        and n_real.numel() == 1
        and n_real.dtype in (torch.int32, torch.int64)
        and n_real.device == x.device
        and x.is_cuda
        and x.dim() == 2
        and x.stride(1) == 1
    )


@sglang_route(f"{TRANSPORT}._mask_dp_pad_rows", versions=VERSIONS, around=True)
def _mask_dp_pad_rows(orig, local_hidden_states, forward_batch):
    n_real = forward_batch.num_token_non_padded
    x = local_hidden_states
    if (
        not serving()
        or os.environ.get(ENV, "1") == "0"
        or n_real is None
        or x.shape[0] == 0
        or not _fits(x, n_real)
    ):
        return orig(local_hidden_states, forward_batch)
    _kernel()[(x.shape[0],)](x, n_real, x.stride(0), x.shape[1], BLOCK=1024)
    global _NOTED
    if not _NOTED:
        _NOTED = True
        stats.set_info("dp_pad_mask", "one launch")
    return None
