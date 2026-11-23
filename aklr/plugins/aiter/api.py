# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native AITER calls served by this provider.

Use the original AITER module path for module-specific overloads, in particular
the Triton GEMMs. Resolving an export returns the host function itself, including
its compile guard, argument defaults and native fallback.
"""

from importlib import import_module

_MODULES = {
    "biased_grouped_topk": "aiter.ops.topk",
    "causal_conv1d_split_qkv_triton_tile_fn": "aiter.ops.triton.gated_delta_net.causal_conv1d_prefill",
    "deepgemm_fp8_paged_mqa_logits": "aiter.ops.triton.attention.pa_mqa_logits",
    "flash_attn_varlen_func": "aiter.ops.mha",
    "fused_moe": "aiter.fused_moe",
    "gemm_a16w16": "aiter.tuned_gemm",
    "gemm_a16w16_out": "aiter.tuned_gemm",
    "gemm_a8w8_blockscale": "aiter.ops.gemm_op_a8w8",
    "gemm_a8w8_blockscale_bpreshuffle": "aiter.ops.gemm_op_a8w8",
    "gemm_a8w8_bpreshuffle": "aiter.ops.gemm_op_a8w8",
    "mhc_fused_post_pre": "aiter.ops.mhc",
    "mhc_post": "aiter.ops.mhc",
    "mhc_pre": "aiter.ops.mhc",
    "mla_decode_fwd": "aiter.mla",
    "mla_gluon": "aiter.ops.triton.gluon.mla_gluon",
    "mla_sparse_prefill_v4": "aiter.ops.mla_sparse_prefill",
    "per_1x32_mx_quant_hip": "aiter.ops.quant",
    "per_group_quant_hip": "aiter.ops.quant",
    "rms_norm": "aiter.ops.rmsnorm",
    "rmsnorm2d_fwd": "aiter.ops.rmsnorm",
    "rmsnorm2d_fwd_with_add": "aiter.ops.rmsnorm",
    "topk_gating": "aiter.ops.topk",
}
__all__ = sorted(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULES[name]), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
