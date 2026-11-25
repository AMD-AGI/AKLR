# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The absorbed MLA steps of sglang's ROCm forward: in
``DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare``, the qkv_a projection and the two
latent RMSNorms; around the attention of ``forward_absorb_rocm_core``, with kv_b_proj in fp8 at
one scale per tensor, the query's absorb, RoPE and KV-cache write before it and the value
absorb after it."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseek_common.baseline import MlaQkvANormBaseline
from aklr.plugins.sglang.models.deepseek_common.reference import MlaQkvANormReference

__all__ = ["CONTRACTS", "MLA_QKV_A_NORM"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FINITE = FiniteDistribution()
# The latent RMSNorms' weights: drawn in about [0.5, 1.5], as rms_norm's
_NORM_WEIGHT = LogUniformDistribution(log2_lo="-1", log2_hi="0.58")
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
# A normalized latent is rounded to bf16 twice, as sglang rounds it: the qkv latent (a bf16
# tensor) before the norm, then the norm's output, each up to 2**-8 of the value; B = 2**-12 +
# 4 * 2**-8 * |y| against the reference's bf16 leaves one more 2**-8 for the projection's fp32
# sums and one for the reference's own rounding, and the absolute part covers them where an
# output is near 0. A CPU model of sglang's and these kernels' arithmetic (an IEEE fp32 MFMA
# chain, or one keeping 14 bits below each MFMA's largest product, split-K or not, hidden 6144
# and 16384, 1 to 256 tokens) peaks at 0.65 of B. A missing or swapped norm weight, the two
# latents normalized together or by the other's rms, a dropped K group or a wrong token row each
# fail.
_NORMALIZED = ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-7 + 2**-7)
# The RoPE key is the projection rounded once to bf16, as the reference's is: B = 2**-6 +
# 3 * 2**-8 * |y|, the absolute part for the fp32 sums where a key cancels to near 0 (the same
# model peaks at 0.49).
_PROJECTED = ToleranceBound(absolute_tolerance=2**-6, relative_tolerance=2**-8 + 2**-7)

MLA_QKV_A_NORM = OpContract(
    contract_id="mla_qkv_a_norm@1.0",
    description=(
        "sglang's qkv_a projection and latent RMSNorms of an MLA attention with a compressed "
        "query, on one rank (the qkv latent forward_absorb_rocm_prepare fetches, and its norms): "
        "each token's hidden state times w_qkv_a, the qkv latent; its first q_lora_rank columns, "
        "q's latent, and its next kv_lora_rank columns, the KV latent, each times "
        "rsqrt(mean(x**2) + eps) of its own columns and its norm weight; its last columns, the "
        "RoPE key, as projected."
    ),
    op_type="deepseek_common",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "hidden_dim": axis("The hidden size: the projection's reduction.", 1, 16384),
        "q_lora_rank": axis("q's latent: the qkv latent's first part.", 1, 8192),
        "kv_lora_rank": axis("The KV latent: the qkv latent's next part.", 1, 8192),
        "latent_dim": axis("A qkv latent row: q's latent, the KV latent, then the RoPE key.", 3, 32768),
        "rope_dim": derived_axis(
            "The RoPE key: the qkv latent's last part.", "latent_dim - q_lora_rank - kv_lora_rank"
        ),
    },
    constraints=("rope_dim >= 1",),
    inputs={
        "hidden_states": TensorSpec(
            description="Each token's hidden state, the attention's input.",
            shape=("n_tokens", "hidden_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "w_qkv_a": TensorSpec(
            description="fused_qkv_a_proj_with_mqa's weight: row c gives the qkv latent's column c.",
            shape=("latent_dim", "hidden_dim"),
            dtype=_BF16,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "q_norm_weight": TensorSpec(
            description="q_a_layernorm's weight, one per column of q's latent.",
            shape=("q_lora_rank",),
            dtype=_BF16,
            distribution=_NORM_WEIGHT,
            lifetime="weight",
        ),
        "kv_norm_weight": TensorSpec(
            description="kv_a_layernorm's weight, one per column of the KV latent.",
            shape=("kv_lora_rank",),
            dtype=_BF16,
            distribution=_NORM_WEIGHT,
            lifetime="weight",
        ),
        "eps": ScalarSpec(
            description="Added to each mean square: both norms' variance_epsilon.",
            dtype=DType.FLOAT32,
            interval=(2.0**-149, _FLOAT32_LARGEST),
            default=1e-5,
        ),
    },
    outputs={
        "q_latent": TensorSpec(
            description="Each token's q latent, normalized: q_b_proj's and the indexer's input.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_BF16,
            error_bounds=_NORMALIZED,
        ),
        "kv_latent": TensorSpec(
            description="Each token's KV latent, normalized: its KV cache row's latent part.",
            shape=("n_tokens", "kv_lora_rank"),
            dtype=_BF16,
            error_bounds=_NORMALIZED,
        ),
        "k_rope": TensorSpec(
            description="Each token's RoPE key as projected, before RoPE.",
            shape=("n_tokens", "rope_dim"),
            dtype=_BF16,
            error_bounds=_PROJECTED,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
    reference=MlaQkvANormReference(),
    baseline=MlaQkvANormBaseline(),
)


CONTRACTS = (MLA_QKV_A_NORM,)
