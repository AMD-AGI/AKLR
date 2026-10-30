# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.ops.gemm import bmm, gemm_a4w4_mxfp4, gemm_a8w8, gemm_a8w8_blockwise_scaled
from aklr.ops.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from aklr.ops.mla import flash_mla_with_kvcache, fp8_paged_mqa_logits
from aklr.ops.topk import topk_transform_paged

__all__ = [
    "bmm",
    "flash_mla_with_kvcache",
    "fp8_paged_mqa_logits",
    "gemm_a4w4_mxfp4",
    "gemm_a8w8",
    "gemm_a8w8_blockwise_scaled",
    "load_routes",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "topk_transform_paged",
]


def load_routes() -> None:
    """Import every op's ``api`` module so its routes register, then proxy aiter's routed
    entry points."""
    from aklr.core.contract.locate import op_module
    from aklr.core.contract.registry import ContractRegistry
    from aklr.core.runtime.seam import aklr_proxy

    contracts = ContractRegistry.default().contracts()
    for package in sorted({c.op_package for c in contracts}):
        op_module(package, "api")
    aklr_proxy("aiter")
