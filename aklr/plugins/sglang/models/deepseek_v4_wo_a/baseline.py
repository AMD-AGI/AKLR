# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["DeepseekV4WoAMxfp8Baseline"]


class DeepseekV4WoAMxfp8Baseline(OpBaseline):
    def baseline(self, *, o, wo_a, quantized=None, scales=None, storage="row_major"):
        """What sglang 0.5.23's wo_a_fp8_grid_matmul runs on gfx950 for an MXFP8 wo_b: its fork of
        aiter's batched GEMM with the quantization in the epilogue, split eight ways along K up to 64
        rows."""
        from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import batched_gemm_bf16_fp8_grid

        q, s = batched_gemm_bf16_fp8_grid(o, wo_a, fp8_grid=False, emit_fp8=True)
        return {"quantized": q, "scales": s}
