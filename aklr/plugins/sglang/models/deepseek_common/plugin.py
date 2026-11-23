# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The absorbed MLA steps of sglang's ROCm forward: in
``DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare``, the qkv_a projection and the two
latent RMSNorms; around the attention of ``forward_absorb_rocm_core``, with kv_b_proj in fp8 at
one scale per tensor, the query's absorb, RoPE and KV-cache write before it and the value
absorb after it."""

from __future__ import annotations

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    DTYPE_TRAITS,
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
    ScalarSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin

__all__ = [
    "FP8_GROUP",
    "MLA_KC_ROPE_CACHE",
    "MLA_QKV_A_NORM",
    "MLA_VC",
    "declare",
    "family",
    "mla_kc_rope_cache",
    "mla_qkv_a_norm",
    "mla_vc",
]


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
# The latent RMSNorms' weights: drawn in about [0.5, 1.5], as rms_norm's
_NORM_WEIGHT = LogUniformDistribution(log2_lo="-1", log2_hi="0.58")
_FLOAT32_LARGEST = DTYPE_TRAITS["float32"].largest_finite_value
# A normalized latent is rounded to bf16 twice, as sglang rounds it: the qkv latent (a bf16
# tensor) before the norm, then the norm's output, each up to 2**-8 of the value; B = 2**-12 +
# 3 * 2**-8 * |y| leaves one more 2**-8 for the projection's fp32 sums, and the absolute part
# covers them where an output is near 0. A CPU model of sglang's and these kernels' arithmetic
# (an IEEE fp32 MFMA chain, or one keeping 14 bits below each MFMA's largest product, split-K or
# not, hidden 6144 and 16384, 1 to 256 tokens) peaks at 0.65 of B. A missing or swapped norm
# weight, the two latents normalized together or by the other's rms, a dropped K group or a
# wrong token row each fail.
_NORMALIZED = ToleranceBound(absolute_tolerance=2**-12, relative_tolerance=2**-7)
# The RoPE key is the projection rounded once to bf16: B = 2**-6 + 2 * 2**-8 * |y|, the
# absolute part for the fp32 sums where a key cancels to near 0 (the same model peaks at 0.49).
_PROJECTED = ToleranceBound(absolute_tolerance=2**-6, relative_tolerance=2**-8)

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
)

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
    op_type="deepseek_common",
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
            description="kv_b_proj's per-tensor scale, one element: sglang's 0-d w_scale, viewed.",
            shape=(1,),
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
    op_type="deepseek_common",
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
            description="kv_b_proj's per-tensor scale, one element: sglang's 0-d w_scale, viewed.",
            shape=(1,),
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
            # The codes are the reference's bit for bit (aiter's prequant kernel quantizes the
            # same way); the sums are not: gfx950's fp8 MFMA adds a group's 128 products below
            # fp32 (see gemm's contract), an absolute error that shows where an output cancels
            # to near 0. Under absolute 2**-12, CI run 41962335788 (MI355X) failed 14 of 17
            # kernels on it: 1 to 133 elements in 21 of 108 input sets (violation fraction
            # 3.8e-6 to 2.5e-4), peak 2.85x, unbiased. An MFMA keeping 14 bits below a group's
            # largest product reproduces that (21 to 30 of 108 simulated sets, peak 3.6x to
            # 4.4x) and stays under 2**-9 (peak 0.73x), where a code of 128 or more one fp8
            # step off, or a missing scale, quantization or row, still fails every set.
            error_bounds=ToleranceBound(absolute_tolerance=2**-9, relative_tolerance=2**-8),
        ),
    },
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)


family = Plugin("sglang")
family.define(MLA_QKV_A_NORM)
family.define(MLA_KC_ROPE_CACHE)
family.define(MLA_VC)
family.build()


@family.op
def mla_qkv_a_norm(hidden_states, w_qkv_a, q_norm_weight, kv_norm_weight, eps=1e-5): ...


@family.op
def mla_kc_rope_cache(
    q,
    qkv_latent,
    kv_latent,
    w_kc,
    w_scale,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    kv_cache,
): ...


@family.op
def mla_vc(attn_output, w_vc, w_scale): ...


def _forward_absorb_rocm_prepare(
    self,
    positions,
    hidden_states,
    forward_batch,
    zero_allocator,
    llama_4_scaling=None,
    prev_topk_indices=None,
):
    """forward_absorb_rocm_prepare where its compressed-query branch normalizes the qkv latent
    with aiter's fused bf16 RMSNorm (a bf16 q_b_proj): mla_qkv_a_norm for the qkv latent's
    projection, split and norms, then the original's steps (q_b_proj, the indexer, the query
    absorb unless the core fuses it, RoPE where prepare applies it). sglang keeps the call when
    the op declines it, and when the core's route would serve the fused absorb with
    mla_kc_rope_cache: that op reads the RoPE key inside the whole qkv latent, which this op
    does not produce."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as rocm

    if _prepare_falls_back(self, hidden_states, rocm):
        return FallbackReason.OUT_OF_CONTRACT
    arguments = {
        "hidden_states": hidden_states,
        "w_qkv_a": self.fused_qkv_a_proj_with_mqa.weight,
        "q_norm_weight": self.q_a_layernorm.weight,
        "kv_norm_weight": self.kv_a_layernorm.weight,
        "eps": self.q_a_layernorm.variance_epsilon,
    }
    reason = mla_qkv_a_norm.fallback_reason(**arguments)
    if reason is not None:
        return reason
    fuse_bmm_rope_cache = (
        not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        and rocm._can_fuse_bmm_rope_cat_and_cache(self)
    )
    if fuse_bmm_rope_cache and _core_serves_kc(
        self, hidden_states, positions, forward_batch, llama_4_scaling, rocm
    ):
        return FallbackReason.OUT_OF_CONTRACT
    result = mla_qkv_a_norm(**arguments)
    if isinstance(result, FallbackReason):
        return result
    # the original's steps from here, on its branch without an alt stream, DCP, the deep-gemm
    # BMM or a LoRA: nothing returns a FallbackReason after the op
    q, k_nope, k_rope = result
    q_lora = q if self.use_dsa else None
    topk_indices = None
    k_nope = k_nope.unsqueeze(1)
    q = self.q_b_proj_forward(q)
    if q_lora is not None:
        if self.should_run_indexer(prev_topk_indices):
            topk_indices = self.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )
        else:
            topk_indices = rocm.maybe_capture_indexer_topk(self.layer_id, prev_topk_indices)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # the RoPE key in a tensor of its own, not a view of the qkv latent
    k_pe = k_rope.unsqueeze(1)
    if fuse_bmm_rope_cache:
        q_nope_out = None
    else:
        q_nope_out = rocm.rocm_absorb_q_bmm(self, q_nope, is_capture_mode=get_is_capture_mode())
        q_nope_out = q_nope_out.transpose(0, 1)
    fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
    # the original's DCP decode term is false without DCP
    if self.rotary_emb is not None and (
        (not fuse_rope_for_trtllm_mla)
        and (not self._skip_rope_for_dsa_tilelang_fused())
        and (not self._skip_rope_for_aiter_fused_mla())
        and (
            not rocm._use_aiter
            or not rocm._is_gfx95_supported
            or self.use_dsa
            or (
                self.current_attention_backend not in rocm.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
                and self.current_attention_backend != "aiter"
            )
        )
    ):
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        q_nope if fuse_bmm_rope_cache else None,
    )


def _prepare_falls_back(attn, hidden_states, rocm) -> bool:
    """Whether forward_absorb_rocm_prepare computes other than the route: no compressed query
    or no fused qkv_a projection (prepare_qkv_latent's other branch), a quantized, biased or
    LoRA projection, a qkv latent sglang already holds, would gather (input-scattered
    attention) or projects from other hidden states, another norm branch than aiter's fused
    bf16 one (the alt stream, an mxfp4 or block-fp8 q_b_proj, aiter off), two epsilons, DCP, the
    deep-gemm BMM or a kv_b LoRA."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.runtime_context import get_forward, get_lora

    projection = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    inputs = get_forward().attn_inputs
    return (
        attn.q_lora_rank is None
        or projection is None
        or hasattr(attn, "q_a_proj")
        or type(projection.quant_method) is not UnquantizedLinearMethod
        or projection.bias is not None
        or get_lora().enable_lora
        or rocm.get_attn_tp_context().input_scattered
        or inputs is None
        or inputs.hidden_states_local is not hidden_states
        or inputs.qkv_latent_ is not None
        or attn.alt_stream is not None
        or not rocm._use_aiter
        or (rocm._use_aiter_gfx95 and attn.q_b_proj.weight.dtype == torch.uint8)
        or (rocm._use_aiter_gfx95 and rocm._is_block_scale_fp8(attn.q_b_proj))
        or attn.q_a_layernorm.variance_epsilon != attn.kv_a_layernorm.variance_epsilon
        or attn.dcp_size != 1
        or attn.use_deep_gemm_bmm
        or rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        or rocm.is_kv_b_lora_active(attn)
    )


def _core_serves_kc(attn, hidden_states, positions, forward_batch, llama_4_scaling, rocm) -> bool:
    """Whether the core's route would hand mla_kc_rope_cache the call sglang's own prepare
    leaves it, asked with empty stand-ins of that prepare's query, qkv latent and KV latent."""
    if _core_falls_back(attn, forward_batch, llama_4_scaling, rocm):
        return False
    tokens, like = hidden_states.shape[0], {"dtype": hidden_states.dtype, "device": hidden_states.device}
    rope_start = attn.q_lora_rank + attn.kv_lora_rank
    q = torch.empty((tokens, attn.num_local_heads, attn.qk_head_dim), **like)
    latent = torch.empty((tokens, rope_start + attn.qk_rope_head_dim), **like)
    k_nope = torch.empty((tokens, 1, attn.kv_lora_rank), **like)
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    kc_arguments = _kc_arguments(
        attn, q_nope, q_pe, k_nope, latent[:, rope_start:].unsqueeze(1), positions, forward_batch
    )
    return kc_arguments is not None and mla_kc_rope_cache.fallback_reason(**kc_arguments) is None


def _forward_absorb_rocm_core(
    self,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    forward_batch,
    zero_allocator,
    positions,
    topk_indices,
    llama_4_scaling,
    q_nope_unabsorbed=None,
):
    """The decode branch of forward_absorb_rocm_core whose query absorb prepare left to it
    (q_nope_unabsorbed, a DSA decode on gfx950): the fused absorb, RoPE and KV write as
    mla_kc_rope_cache, sglang's attention, the value absorb as mla_vc, then o_proj. An op
    that declines the call leaves its step to sglang's own function; with neither taking it,
    sglang runs the whole method."""
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as rocm

    if q_nope_unabsorbed is None or _core_falls_back(self, forward_batch, llama_4_scaling, rocm):
        return FallbackReason.OUT_OF_CONTRACT
    kc_arguments = _kc_arguments(self, q_nope_unabsorbed, q_pe, k_nope, k_pe, positions, forward_batch)
    kc_reason = (
        FallbackReason.OUT_OF_CONTRACT
        if kc_arguments is None
        else mla_kc_rope_cache.fallback_reason(**kc_arguments)
    )
    vc_arguments = _vc_arguments(self, rocm)
    vc_reason = (
        FallbackReason.OUT_OF_CONTRACT
        if vc_arguments is None
        else mla_vc.fallback_reason(attn_output=_attention_stand_in(self, q_pe), **vc_arguments)
    )
    if kc_reason is not None and vc_reason is not None:
        return kc_reason
    # the original's steps from here: nothing returns a FallbackReason after the KV write
    q_cat = None if kc_reason is not None else mla_kc_rope_cache(**kc_arguments)
    if q_cat is None or isinstance(q_cat, FallbackReason):
        q_cat, _, k_rope, _ = rocm._fused_bmm_rope_cat_and_cache(
            self, q_nope_unabsorbed, q_pe, k_nope, k_pe, positions, forward_batch.out_cache_loc
        )
    else:
        # the rotated key is in the KV pool: the attention of a call that saves no KV
        # reads no k_rope
        k_rope = None
    attn_output = self.attn_mqa(
        q_cat,
        None,
        None,
        forward_batch,
        q_rope=None,
        k_rope=k_rope,
        save_kv_cache=False,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )
    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
    attn_bmm_output = None if vc_reason is not None else mla_vc(attn_output=attn_output, **vc_arguments)
    if attn_bmm_output is None or isinstance(attn_bmm_output, FallbackReason):
        attn_bmm_output = rocm.rocm_absorb_v_bmm(self, attn_output)
    output, _ = self.o_proj(attn_bmm_output)
    if self.next_skip_topk is None:
        return output
    # topk_indices for the next layer when the index cache is on
    return (output, topk_indices) if self.next_skip_topk else (output, None)


def _core_falls_back(attn, forward_batch, llama_4_scaling, rocm) -> bool:
    """Whether forward_absorb_rocm_core computes other than the route: another branch than
    the fused RoPE-and-cache one with the absorbed query passed whole to the attention, a
    llama-4 scaling or LoRA on kv_b_proj, DCP, a KV scale (the ops write the pool unscaled)
    or NeoX RoPE pairs (the ops rotate GPT-J's)."""
    mode = forward_batch.forward_mode
    k_scale = attn.attn_mqa.k_scale
    k_scale_float = getattr(attn.attn_mqa, "k_scale_float", None)
    return (
        attn.current_attention_backend not in rocm.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        or attn.rotary_emb is None
        or not attn._skip_rope_for_dsa_tilelang_fused()
        or not (mode.is_decode_or_idle() or mode.is_target_verify())
        or llama_4_scaling is not None
        or attn.dcp_size != 1
        or attn.use_deep_gemm_bmm
        or rocm._SGLANG_EXPERIMENTAL_LORA_OPTI
        or rocm.is_kv_b_lora_active(attn)
        or (k_scale is not None and k_scale_float != 1.0)
        or attn.rotary_emb.is_neox_style
    )


def _kc_arguments(attn, q_nope, q_pe, k_nope, k_pe, positions, forward_batch):
    """mla_kc_rope_cache's arguments as sglang holds them, zero-copy; None for a tensor not
    laid out as the projections and the pools leave them."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    rotary = attn.rotary_emb
    kv_cache = get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id)
    arguments = {
        "q": _whole_rows(q_nope, q_pe),
        "qkv_latent": _qkv_latent(k_pe, attn.q_lora_rank, attn.kv_lora_rank),
        "kv_latent": _rows(k_nope),
        "w_kc": attn.w_kc.transpose(1, 2),
        "w_scale": _scale(attn.w_scale),
        "cos_cache": _rows(getattr(rotary, "cos_cache", None)),
        "sin_cache": _rows(getattr(rotary, "sin_cache", None)),
        "positions": positions,
        "out_cache_loc": forward_batch.out_cache_loc,
        "kv_cache": _rows(kv_cache),
    }
    return None if any(value is None for value in arguments.values()) else arguments


def _vc_arguments(attn, rocm):
    """mla_vc's weights as sglang holds them, for rocm_absorb_v_bmm's fp8 branch into a bf16
    o_proj; None where it takes another branch."""
    w_scale = _scale(attn.w_scale)
    if not (
        rocm._use_aiter_gfx95
        and attn.w_kc.dtype == torch.float8_e4m3fn
        and attn.o_proj.weight.dtype != torch.uint8
        and not rocm._is_block_scale_fp8(attn.o_proj)
        and w_scale is not None
    ):
        return None
    return {"w_vc": attn.w_vc.transpose(1, 2), "w_scale": w_scale}


def _rows(tensor):
    """A contiguous tensor as [rows, everything else], a view; None for another tensor."""
    if not (isinstance(tensor, torch.Tensor) and tensor.dim() >= 2 and tensor.is_contiguous()):
        return None
    return tensor.view(tensor.shape[0], -1)


def _scale(w_scale):
    """kv_b_proj's per-tensor scale, sglang's 0-d tensor, as the contracts' one element (a
    view of it); None for another value."""
    return w_scale.view(1) if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 0 else None


def _attention_stand_in(attn, q_pe):
    """An empty tensor shaped as the attention's output, to ask mla_vc before the KV write."""
    return torch.empty(
        (q_pe.shape[0], attn.num_local_heads, attn.kv_lora_rank), dtype=q_pe.dtype, device=q_pe.device
    )


def _whole_rows(first, second):
    """The [tokens, heads, a + b] tensor whose two adjacent views prepare split q into; None
    for two tensors apart."""
    tokens, heads, width = first.shape
    whole = width + second.shape[-1]
    if not (
        second.dtype == first.dtype
        and second.shape[:2] == first.shape[:2]
        and first.stride() == second.stride() == (heads * whole, whole, 1)
        and second.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
        and second.storage_offset() == first.storage_offset() + width
    ):
        return None
    return first.as_strided((tokens, heads, whole), first.stride())


def _qkv_latent(k_pe, q_lora_rank, kv_lora_rank):
    """The [tokens, q_lora_rank + kv_lora_rank + rope] qkv latent whose last columns k_pe
    [tokens, 1, rope] views (fetch_qkv_latent's, split by prepare); None otherwise."""
    tokens, _, rope = k_pe.shape
    offset = q_lora_rank + kv_lora_rank
    width = offset + rope
    if not (k_pe.stride(0) == width and k_pe.stride(-1) == 1 and k_pe.storage_offset() >= offset):
        return None
    return k_pe.as_strided((tokens, width), (width, 1), k_pe.storage_offset() - offset)


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm.DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_absorb_rocm_prepare,
    )
    plugin.wrap(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm.DeepseekMLARocmForwardMixin.forward_absorb_rocm_core",
        versions=("0.5.22", "0.5.23"),
        replace=_forward_absorb_rocm_core,
    )
