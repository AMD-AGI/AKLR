# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.baseline import OpBaseline

__all__ = ["Mxfp8E4m3QuantizeBaseline"]


class Mxfp8E4m3QuantizeBaseline(OpBaseline):
    def baseline(self, *, x, quantized=None, scales=None):
        """What vLLM 0.34's mxfp8_e4m3_quantize runs on ROCm for a 2D row-major activation: its Triton
        kernel."""
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_triton

        quantized, scales = _mxfp8_e4m3_quantize_triton(x)
        return {"quantized": quantized, "scales": scales}
