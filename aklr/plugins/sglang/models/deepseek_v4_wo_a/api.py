# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4_wo_a.contract import DEEPSEEK_V4_WO_A_MXFP8

__all__ = ["deepseek_v4_wo_a_mxfp8"]


@api(DEEPSEEK_V4_WO_A_MXFP8)
def deepseek_v4_wo_a_mxfp8(o, wo_a, quantized=None, scales=None, *, storage="row_major"): ...


@sglang_route(
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_hip.wo_a_fp8_grid_matmul", versions=("0.5.23",)
)
def _wo_a_fp8_grid_matmul(o, wo_a, fp8_grid, emit_fp8=False):
    """V4.1's wo_a on gfx950 when wo_b takes MXFP8 activations, returned as sglang's Mxfp8Activation;
    the bf16 and fp8-grid outputs, a process without sglang's fork of the GEMM, and deterministic
    inference (the op's K split depends on the row count) stay on sglang."""
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as dense
    from sglang.srt.runtime_context import get_exec

    if (
        fp8_grid
        or not emit_fp8
        or dense._wo_a_fp8_grid_gemm is None
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return FallbackReason.OUT_OF_CONTRACT
    out = deepseek_v4_wo_a_mxfp8(o, wo_a, storage="strided_rows")
    if isinstance(out, FallbackReason):
        return out
    quantized, scales = out
    return dense.Mxfp8Activation(quantized, scales)
