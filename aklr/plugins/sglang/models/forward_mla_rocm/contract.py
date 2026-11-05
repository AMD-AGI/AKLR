# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The absorbed MLA steps around the attention of sglang's ROCm forward
(``DeepseekMLARocmForwardMixin.forward_absorb_rocm_core``), with kv_b_proj in fp8 at one
scale per tensor: the query's absorb, RoPE and KV-cache write before the attention, and the
value absorb after it."""

from __future__ import annotations

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LogUniformDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)

__all__ = ["CONTRACTS", "FP8_GROUP", "MLA_KC_ROPE_CACHE", "MLA_VC"]

_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()
# The width of the groups sglang's fp8 absorb quantizes its activations in (aiter's a8w8
# batched GEMM with group_size=128): a group's fp8 codes at a scale of max(amax, 1e-10) / 448.
FP8_GROUP = 128
# kv_b_proj's per-tensor scale (sglang's input_to_float8: amax / 448 of the bf16 weight),
# drawn where the absorbed products of N(0, 1) inputs stay inside fp8's range
_W_SCALE = LogUniformDistribution(log2_lo=-6, log2_hi=0)
# cos and sin of the rotary angles: drawn in (0, 1], the magnitudes of a real table
_ROTARY = LogUniformDistribution(log2_lo=-8, log2_hi=0)
# The KV pool's bytes that 32-bit offsets address: rows past them stay on sglang.
_POOL_BYTES = 2**31 - 2
# fp8 outputs of bf16 math: one fp8 ulp (2**-3 of the value) for the kernels' bf16 steps, which
# may round a value across an fp8 rounding boundary, and 2**-5 for a RoPE pair whose two
# products, rounded to bf16 at up to |q| apart, cancel
_FP8_OF_BF16 = ToleranceBound(absolute_tolerance=2**-5, relative_tolerance=2**-3)

MLA_KC_ROPE_CACHE = OpContract(
    contract_id="mla_kc_rope_cache@1.0",
    description=(
        "sglang's absorbed MLA query and KV-cache write on one rank, kv_b_proj in fp8 at one "
        "scale per tensor (_fused_bmm_rope_cat_and_cache in forward_absorb_rocm_core): each "
        "head's position-free query part, quantized per 128-wide group to fp8 at "
        "max(amax, 1e-10) / 448, times w_kc and w_scale, then the head's RoPE part rotated, "
        "as q_cat; each token's KV latent and rotated RoPE key written to its KV pool row in "
        "the pool's fp8, unscaled. RoPE rotates GPT-J pairs (2i, 2i + 1) by the token's "
        "position's cos and sin. A row whose slot is 0 is padding: it writes nothing and its "
        "q_cat is 0."
    ),
    op_type="forward_mla_rocm",
    axes={
        "n_tokens": axis(
            "Tokens in this call, one query each, padding rows included.", 1, 65536, is_var=True
        ),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "qk_nope_dim": axis("A query head's position-free part: the absorb's reduction.", 1, 512),
        "kv_lora_rank": axis(
            "The KV latent: a KV row's first part, and q_cat's.", 128, 1024, multiple_of=128
        ),
        "kv_dim": axis("A KV row and a q_cat head: the latent, then the RoPE part.", 130, 1152),
        "latent_dim": axis("A qkv latent row: q's latent, the KV latent, then the RoPE key.", 131, 16384),
        "n_slots": axis("Rows of the KV pool.", 1, 2**31 - 1, is_var=True, any_size=True),
        "n_positions": axis("Rows of the rotary tables.", 1, 2**31 - 1, is_var=True, any_size=True),
        "rope_dim": derived_axis("The RoPE part of a query head and of a KV row.", "kv_dim - kv_lora_rank"),
        "qk_head_dim": derived_axis(
            "A query head: its position-free part, then its RoPE part.", "qk_nope_dim + rope_dim"
        ),
        "rope_pairs": derived_axis("The RoPE part's pairs, one rotary angle each.", "rope_dim // 2"),
    },
    constraints=(
        "rope_dim >= 2",
        "rope_dim % 2 == 0",
        "latent_dim > kv_dim",
        # distinct slots, 0 among them
        "n_tokens <= n_slots",
        f"n_slots * kv_dim <= {_POOL_BYTES}",
        f"n_positions * rope_pairs <= {2**31 - 1}",
    ),
    inputs={
        "q": TensorSpec(
            description="Each token's query per head, q_b_proj's output: the position-free part, then the RoPE part.",
            shape=("n_tokens", "n_heads", "qk_head_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "qkv_latent": TensorSpec(
            description=(
                "Each token's qkv_a projection, sglang's qkv latent: q's latent, the KV latent, "
                "then the RoPE key, of which only the RoPE key is read."
            ),
            shape=("n_tokens", "latent_dim"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "kv_latent": TensorSpec(
            description="Each token's KV latent after kv_a_layernorm.",
            shape=("n_tokens", "kv_lora_rank"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "w_kc": TensorSpec(
            description=(
                "kv_b_proj's key part per head, absorbed (sglang's w_kc, transposed): row c "
                "maps the position-free query part to latent channel c."
            ),
            shape=("n_heads", "kv_lora_rank", "qk_nope_dim"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w_scale": TensorSpec(
            description="kv_b_proj's scale, one per tensor.",
            shape=(),
            dtype=_FP32,
            distribution=_W_SCALE,
            lifetime="weight",
        ),
        "cos_cache": TensorSpec(
            description="cos of each position's rotary angles, one per pair.",
            shape=("n_positions", "rope_pairs"),
            dtype=_BF16,
            distribution=_ROTARY,
            lifetime="weight",
        ),
        "sin_cache": TensorSpec(
            description="sin of each position's rotary angles, one per pair.",
            shape=("n_positions", "rope_pairs"),
            dtype=_BF16,
            distribution=_ROTARY,
            lifetime="weight",
        ),
        "positions": TensorSpec(
            description="Each token's position.",
            shape=("n_tokens",),
            dtype=_INT64,
            distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
        "out_cache_loc": TensorSpec(
            description=(
                "Each token's KV pool row, distinct but for 0: a padding row's (sglang's graph "
                "padding, which several rows may hold)."
            ),
            shape=("n_tokens",),
            dtype=_INT64,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "kv_cache": TensorSpec(
            description="This layer's KV pool: a row per slot, the latent then the RoPE part.",
            shape=("n_slots", "kv_dim"),
            dtype=_FP8,
            distribution=_FINITE,
            mutable=True,
            error_bounds=_FP8_OF_BF16,
        ),
    },
    outputs={
        "q_cat": TensorSpec(
            description=(
                "Each head's absorbed query in the latent's basis, then its rotated RoPE part, "
                "in the pool's dtype; 0 for a padding row."
            ),
            shape=("n_tokens", "n_heads", "kv_dim"),
            dtype=DTypeSpec(same_as_tensor="kv_cache"),
            error_bounds=_FP8_OF_BF16,
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

MLA_VC = OpContract(
    contract_id="mla_vc@1.0",
    description=(
        "sglang's absorbed MLA value projection on one rank, kv_b_proj in fp8 at one scale per "
        "tensor, for a bf16 o_proj (rocm_absorb_v_bmm): each head's attention output, "
        "quantized per 128-wide group to fp8 at max(amax, 1e-10) / 448, times w_vc and "
        "w_scale; the heads side by side, o_proj's input."
    ),
    op_type="forward_mla_rocm",
    axes={
        "n_tokens": axis("Tokens in this call, padding rows included.", 1, 65536, is_var=True),
        "n_heads": axis("Query heads on this rank.", 1, 128),
        "kv_lora_rank": axis(
            "The latent: the attention output's width per head.", 128, 1024, multiple_of=128
        ),
        "v_head_dim": axis("A head's value channels.", 1, 1024),
        "out_dim": derived_axis("The heads' value channels side by side.", "n_heads * v_head_dim"),
    },
    inputs={
        "attn_output": TensorSpec(
            description="Each head's attention output, in the latent's basis.",
            shape=("n_tokens", "n_heads", "kv_lora_rank"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "w_vc": TensorSpec(
            description=(
                "kv_b_proj's value part per head, absorbed (sglang's w_vc, transposed): row v "
                "maps the latent to value channel v."
            ),
            shape=("n_heads", "v_head_dim", "kv_lora_rank"),
            dtype=_FP8,
            distribution=_FINITE,
            lifetime="weight",
        ),
        "w_scale": TensorSpec(
            description="kv_b_proj's scale, one per tensor.",
            shape=(),
            dtype=_FP32,
            distribution=_W_SCALE,
            lifetime="weight",
        ),
    },
    outputs={
        "attn_bmm_output": TensorSpec(
            description="Each token's value channels, head by head.",
            shape=("n_tokens", "out_dim"),
            dtype=_BF16,
            # the codes are the reference's: fp32 sums in another order, and bf16 rounding
            error_bounds=ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-8),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

CONTRACTS = (MLA_KC_ROPE_CACHE, MLA_VC)
