# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.ops.gemm.api import gemm
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []


@vllm_route(
    "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
    "vllm.models.muse_spark.amd.model.MuseSparkTorchLinearMethod.apply",
    versions=("0.33.0rc1.dev910",),
)
def _torch_linear(self, layer, x, bias=None):
    """The MoE pooler and unpooler, which vLLM keeps off aiter's tuned GEMM (its split-K
    kernels for these shapes accumulate in bf16) and runs through F.linear."""
    return gemm(x, layer.weight, bias=bias)
