# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The absorbed MLA steps of sglang's ROCm forward: in
``DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare``, the qkv_a projection and the two
latent RMSNorms, with a bf16 or a block-fp8 projection; around the attention of
``forward_absorb_rocm_core``, with kv_b_proj in fp8 at one scale per tensor, the query's absorb,
RoPE and KV-cache write before it and the value absorb after it."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
    DType,
    FiniteDistribution,
    LecunNormalDistribution,
    LogUniformDistribution,
    Preshuffle,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins.sglang.models.deepseek_common.baseline import MlaQkvANormBaseline, MlaQkvANormFp8Baseline
from aklr.plugins.sglang.models.deepseek_common.reference import (
    FP8_GROUP,
    MlaQkvANormFp8Reference,
    MlaQkvANormReference,
)

__all__ = ["CONTRACTS", "FP8_GROUP", "MLA_QKV_A_NORM", "MLA_QKV_A_NORM_FP8"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
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


# A block-fp8 checkpoint's 128 x 128 weight scales, as gemm_a8w8_blockwise_scaled draws them
_BLOCK_SCALE = LogUniformDistribution(log2_lo="-3", log2_hi="0")
# The fp8 projection's outputs, against the reference's in their declared dtypes. The hidden
# state's fp8 codes are the reference's bit for bit (the same per-group quantization); the sums
# are not: gfx950's fp8 MFMA adds a group's products below fp32 and the kernels split K. The
# normalized latents keep _NORMALIZED (a CI-image bench of 1 to 80 tokens against an unrounded
# reference: peak 2**-14 + 2**-8.4 * |y|). The RoPE key, rounded once, stayed within 2**-14 +
# 2**-8 * |y| of it; B = 2**-12 + 3 * 2**-8 * |y| leaves one more 2**-8 for the reference's own
# rounding.
_PROJECTED_FP8 = ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-8 + 2**-7)
# q's latent in fp8 codes, quantized from its bf16 value: a bf16 step of the latent, or of its
# group's amax, may move a code across an fp8 rounding boundary (one fp8 step, up to 2**-3 of the
# code, plus the code's own rounding, 2**-4); a small code follows the latent's absolute error,
# 448 / amax times larger (the same bench: peak 2.4e-3 on codes below 2**-6)
_Q_CODES = ToleranceBound(absolute_tolerance=2**-7, relative_tolerance=2**-3 + 2**-4)
# a group's scale, its bf16 amax / 448: as close as the normalized latent (the same bench: peak
# 2**-7)
_Q_SCALES = ToleranceBound(absolute_tolerance=0.0, relative_tolerance=2**-6)

MLA_QKV_A_NORM_FP8 = OpContract(
    contract_id="mla_qkv_a_norm_fp8@1.0",
    description=(
        "sglang's qkv_a projection and latent RMSNorms of an MLA attention with a compressed "
        "query and a block-fp8 checkpoint, on one rank, and q_b_proj's input quantization: each "
        "token's hidden state, quantized per 128-wide group to fp8 at max(amax, 1e-10) / 448, "
        "times w_qkv_a at its scale per 128 x 128 block, the qkv latent; its first q_lora_rank "
        "columns, q's latent, and its next kv_lora_rank columns, the KV latent, each times "
        "rsqrt(mean(x**2) + eps) of its own columns and its norm weight; its last columns, the "
        "RoPE key, as projected. q's normalized latent, rounded to bf16, also quantized per "
        "128-wide group to fp8 at max(amax, 1e-10) / 448, its scales group by group."
    ),
    op_type="deepseek_common",
    axes={
        "n_tokens": axis("Tokens in this call.", 1, 65536, is_var=True),
        "hidden_dim": axis("The hidden size: the projection's reduction.", 128, 16384, multiple_of=128),
        "q_lora_rank": axis("q's latent: the qkv latent's first part.", 128, 8192, multiple_of=128),
        "kv_lora_rank": axis("The KV latent: the qkv latent's next part.", 1, 8192),
        "latent_dim": axis("A qkv latent row: q's latent, the KV latent, then the RoPE key.", 3, 32768),
        "rope_dim": derived_axis(
            "The RoPE key: the qkv latent's last part.", "latent_dim - q_lora_rank - kv_lora_rank"
        ),
        "hidden_groups": derived_axis("128-wide groups of the hidden size.", "hidden_dim // 128"),
        "latent_blocks": derived_axis("128-row blocks of w_qkv_a.", "ceil_div(latent_dim, 128)"),
        "q_groups": derived_axis("128-wide groups of q's latent.", "q_lora_rank // 128"),
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
            dtype=_FP8,
            distribution=LecunNormalDistribution(fan_in="hidden_dim"),
            lifetime="weight",
        ),
        "w_qkv_a_scale": TensorSpec(
            description="w_qkv_a's dequant scale per 128 x 128 block (its weight_scale_inv).",
            shape=("latent_blocks", "hidden_groups"),
            dtype=_FP32,
            distribution=_BLOCK_SCALE,
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
            description="Each token's q latent, normalized: the indexer's input.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_BF16,
            error_bounds=_NORMALIZED,
        ),
        "q_fp8": TensorSpec(
            description="q_latent's fp8 codes, per 128-wide group: q_b_proj's input.",
            shape=("n_tokens", "q_lora_rank"),
            dtype=_FP8,
            error_bounds=_Q_CODES,
        ),
        "q_scale": TensorSpec(
            description=(
                "q_fp8's scale per group, group-major (row g holds group g of every token): "
                "transposed, the column-major scale aiter's preshuffled GEMM reads."
            ),
            shape=("q_groups", "n_tokens"),
            dtype=_FP32,
            error_bounds=_Q_SCALES,
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
            error_bounds=_PROJECTED_FP8,
        ),
    },
    # row-major, or as sglang holds the weight for aiter's preshuffled GEMM on gfx950:
    # shuffle_weight's 16 x 16 tiles, the scales as the checkpoint stores them
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(
                detect_facts=(
                    LayoutFact(location="tensor_attr", name="w_qkv_a.is_shuffled", expected_value=False),
                )
            ),
            "shuffled_16x16": LayoutVariant(
                layouts={"w_qkv_a": Preshuffle(tile=(16, 16))},
                detect_facts=(
                    LayoutFact(location="tensor_attr", name="w_qkv_a.is_shuffled", expected_value=True),
                ),
            ),
        }
    ),
    compare=CompareSpec(determinism="bitwise"),
    reference=MlaQkvANormFp8Reference(),
    baseline=MlaQkvANormFp8Baseline(),
)

CONTRACTS = (MLA_QKV_A_NORM, MLA_QKV_A_NORM_FP8)
