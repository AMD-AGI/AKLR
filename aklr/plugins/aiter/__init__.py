###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""AKLR as a provider of aiter's kernel plugin interface (``aklr.plugins.aiter.provider``).

aiter routes its ops to the providers enabled for it; sglang and vLLM reach AKLR through it once they
enable that interface, with no route of their own on an aiter function. The ops it serves are the
op packages beside it, one per op family (``README.md``)."""

# an api named as its package (fused_moe, mega_moe) stays reached through the package
from aklr.plugins.aiter import fused_moe, gemm, mega_moe  # noqa: F401
from aklr.plugins.aiter.attention import paged_attention_decode, unified_attention
from aklr.plugins.aiter.gemm import (
    bmm_bf16_nn,
    bmm_bf16_nt,
    gemm_a8w8_nt,
    gemm_bf16_nt,
    gemm_fp8_nt_blockscaled,
    gemm_mxfp4_nt,
)
from aklr.plugins.aiter.kda_conv import kimi_k3_kda_conv
from aklr.plugins.aiter.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from aklr.plugins.aiter.mla import (
    flash_mla_with_kvcache,
    fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_verify,
    mla_decode_fwd,
    mla_decode_fwd_cprr,
)
from aklr.plugins.aiter.mla_prefill import mla_prefill_varlen, mla_prefill_varlen_lse, sparse_prefill_mxfp8
from aklr.plugins.aiter.quant import per_group_quant_fp8, per_group_quant_mxfp8
from aklr.plugins.aiter.rms_norm import add_rms_norm, rms_norm
from aklr.plugins.aiter.topk import biased_grouped_topk, topk_transform_paged

__all__ = [
    "add_rms_norm",
    "biased_grouped_topk",
    "bmm_bf16_nn",
    "bmm_bf16_nt",
    "flash_mla_with_kvcache",
    "fp8_paged_mqa_logits",
    "fp8_paged_mqa_logits_verify",
    "gemm_a8w8_nt",
    "gemm_bf16_nt",
    "gemm_fp8_nt_blockscaled",
    "gemm_mxfp4_nt",
    "kimi_k3_kda_conv",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "mla_decode_fwd",
    "mla_decode_fwd_cprr",
    "mla_prefill_varlen",
    "mla_prefill_varlen_lse",
    "paged_attention_decode",
    "per_group_quant_fp8",
    "per_group_quant_mxfp8",
    "rms_norm",
    "sparse_prefill_mxfp8",
    "topk_transform_paged",
    "unified_attention",
]
