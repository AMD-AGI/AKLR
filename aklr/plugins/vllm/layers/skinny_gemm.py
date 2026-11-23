# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.plugins import Plugin
from aklr.plugins.aiter.gemm.plugin import gemm

__all__: list[str] = []


def _wv_split_krc(a, b, cu_count, bias=None):
    """vLLM's skinny GEMM on gfx950 for 10 to 128 tokens, ahead of aiter where it fits."""
    return gemm(a, b, bias=bias)


def declare(plugin: Plugin) -> None:
    plugin.intercept(
        "vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl",
        "vllm._custom_ops.wvSplitKrc",
        versions=("0.32.1rc1.dev276", "0.33.0rc1.dev910", "0.33.0", "0.34.0"),
        replace=_wv_split_krc,
    )
