# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Under DP attention, the zeroing of a DP gather's pad rows in one launch (sglang's arange,
compare and masked fill per MoE layer). ``AKLR_DSV41_DP_PAD_MASK=0`` keeps sglang's.

The FP8 DP gather of prefill rows (sglang's ``SGLANG_ENABLE_DP_GATHER_FP8``) quantizes through
``_run_per_token_group_quant_8bit_kernel``, whose group-quant kernel sglang imports on CUDA only;
on ROCm the route runs the same quantization through sglang's Triton kernel instead."""

from __future__ import annotations

import functools
import os
import sys

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV = "AKLR_DSV41_DP_PAD_MASK"
TRANSPORT = "sglang.srt.models.deepseek_common.moe_transport"
FP8_KERNEL = "sglang.kernels.ops.quantization.fp8_kernel"
_NOTED = False
_NOTED_QUANT = False

SGLANG_READS = {
    "srt/models/deepseek_common/moe_transport.py": ("_mask_dp_pad_rows(local_hidden_states, forward_batch)",),
    "srt/model_executor/forward_batch_info.py": ("num_token_non_padded",),
    "kernels/ops/quantization/fp8_kernel.py": (
        "_run_per_token_group_quant_8bit_kernel(x, x_q, x_s, group_size, eps, fp8_min, fp8_max, *, "
        "scale_ue8m0, fuse_silu_and_mul, masked_m)",
        "per_token_group_quant",
        "_per_token_group_quant_8bit_raw(x, group_size, eps, dtype, ..., output_q, output_s)",
    ),
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


@sglang_route(f"{FP8_KERNEL}._run_per_token_group_quant_8bit_kernel", versions=VERSIONS, around=True)
def _group_quant(
    orig, x, x_q, x_s, group_size, eps, fp8_min, fp8_max, *, scale_ue8m0, fuse_silu_and_mul, masked_m
):
    kernel = sys.modules[FP8_KERNEL]
    if (
        hasattr(kernel, "per_token_group_quant")
        or not serving()
        or scale_ue8m0
        or fuse_silu_and_mul
        or masked_m is not None
        or x_q.dtype != torch.float8_e4m3fn
    ):
        return orig(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0=scale_ue8m0,
            fuse_silu_and_mul=fuse_silu_and_mul,
            masked_m=masked_m,
        )
    kernel._per_token_group_quant_8bit_raw(x, group_size, eps, dtype=x_q.dtype, output_q=x_q, output_s=x_s)
    global _NOTED_QUANT
    if not _NOTED_QUANT:
        _NOTED_QUANT = True
        stats.set_info("fp8_group_quant", "sglang's Triton kernel (its JIT kernel is CUDA-only)")
    return None
