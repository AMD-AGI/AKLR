# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang runs on aiter on gfx950 for the parts of DeepseekV2DecoderLayer and
DeepseekModelNextN the ops replace: DeepseekV2MoE.forward_normal, the dense MLP, the attention on
its sparse absorbed path (sglang 0.5.22, the DSA backend's triton kernels) and eh_proj.
"""

from __future__ import annotations

import functools

__all__ = [
    "deepseek_v2_attention_baseline",
    "deepseek_v2_attention_indexer_baseline",
    "deepseek_v2_dense_layer_baseline",
    "deepseek_v2_eh_proj_baseline",
    "deepseek_v2_fp8_attention_indexer_baseline",
    "deepseek_v2_fp8_dense_layer_baseline",
    "deepseek_v2_fp8_moe_layer_baseline",
    "deepseek_v2_fp8_moe_layer_shared_apart_baseline",
    "deepseek_v2_moe_baseline",
    "deepseek_v2_moe_layer_baseline",
    "deepseek_v2_unquantized_moe_layer_baseline",
    "deepseek_v2_unquantized_moe_layer_shared_apart_baseline",
]


def _aiter(fn):
    """aiter's own body, not AKLR's stand-in: the speedup is over aiter."""
    proxy = getattr(fn, "aklr_proxy", None)
    return fn if proxy is None else proxy.backend_fn


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baselines replay: it is chosen at import,
    from SGLANG_USE_AITER=1 on gfx950 with ROCm 7.2 or later."""
    from sglang.srt.layers.quantization import fp8_utils, unquant

    if not (fp8_utils._use_aiter_bpreshuffle_gfx95 and unquant._use_aiter):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950, ROCm >= 7.2")


def _rms_norm(x, weight, eps):
    """sglang's RMSNorm on aiter without a residual (RMSNorm.forward_aiter)."""
    from aiter import rmsnorm2d_fwd

    return rmsnorm2d_fwd(x, weight, eps)


def _linear(x, weight, scale):
    """sglang's linear on aiter: fp8 weights (with a scale) through aiter_w8a8_block_fp8_linear,
    bf16 ones through tgemm.mm, as UnquantizedLinearMethod."""
    if scale is None:
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight, None, otype=x.dtype)
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    return aiter_w8a8_block_fp8_linear(x, weight, [128, 128], scale)


def _in_dtype(bias, dtype):
    """``bias`` in ``dtype``, cast once per tensor as sglang caches it
    (TopKConfig.correction_bias_for_dtype): sglang's fp32 bias reaches top-k in bf16."""
    if bias.dtype == dtype:
        return bias
    cast = getattr(bias, "_aklr_in_dtype", None)
    if cast is None or cast.dtype != dtype:
        cast = bias._aklr_in_dtype = bias.to(dtype)
    return cast


def _route(
    hidden_states,
    router_weight,
    correction_bias,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    routed_scaling_factor,
):
    """The router's bf16 logits (MoEGate on HIP: aiter_dsv3_router_gemm, tgemm.mm), aiter's
    grouped top-k on them with the bias in their dtype: the routing weights, then the expert
    ids."""
    import torch
    from aiter import biased_grouped_topk

    logits = _linear(hidden_states, router_weight, None)
    shape, device = (hidden_states.shape[0], topk), hidden_states.device
    weights = torch.empty(shape, dtype=torch.float32, device=device)
    ids = torch.empty(shape, dtype=torch.int32, device=device)
    _aiter(biased_grouped_topk)(
        logits,
        _in_dtype(correction_bias, logits.dtype),
        weights,
        ids,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    return weights, ids


def deepseek_v2_moe_layer_baseline(**call):
    """post_attention_layernorm, then DeepseekV2MoE.forward_normal on it with MXFP4 experts,
    the shared one in their last slot (_shared_slot_moe at per_1x32)."""
    from aiter import QuantType

    return _shared_slot_moe(QuantType.per_1x32, **call)


def deepseek_v2_fp8_moe_layer_baseline(**call):
    """post_attention_layernorm, then DeepseekV2MoE.forward_normal on it with fp8 experts,
    the shared one in their last slot (_shared_slot_moe at per_128x128: fused_moe quantizes
    the activations per 1 x 128)."""
    from aiter import QuantType

    return _shared_slot_moe(QuantType.per_128x128, **call)


def deepseek_v2_moe_baseline(**call):
    """DeepseekV2MoE.forward_normal with MXFP4 experts, the shared one in their last slot
    (_shared_slot_experts at per_1x32)."""
    from aiter import QuantType

    return _shared_slot_experts(QuantType.per_1x32, **call)


def _shared_slot_moe(quant_type, hidden_states, norm_weight, norm_eps, **experts):
    """post_attention_layernorm, then _shared_slot_experts on it; the stream added."""
    x = _rms_norm(hidden_states, norm_weight, norm_eps)
    return hidden_states + _shared_slot_experts(quant_type, x, **experts)


def _shared_slot_experts(
    quant_type,
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w2,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
    w1_scale=None,
    w2_scale=None,
    out_dtype=None,
    out=None,
):
    """_route's top-k on ``hidden_states``, sglang's shared-slot append and aiter's fused_moe
    on the experts at ``quant_type`` (with their scales, if quantized); one rank: no
    all-reduce."""
    _check_aiter_path()
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        fused_append_shared_experts,
    )

    weights, ids = _route(
        hidden_states,
        router_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    ids, weights = fused_append_shared_experts(ids, weights, 1, 1.0, router_weight.shape[0])
    return _aiter(fused_moe)(
        hidden_states,
        w1,
        w2,
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _mlp(x, gate_up_weight, down_weight, gate_up_scale=None, down_scale=None):
    """DeepseekV2MLP as sglang runs it on HIP: gate_up_proj and down_proj as its linears (fp8
    ones with their scales), sgl_kernel's silu_and_mul between them."""
    import torch
    from sgl_kernel import silu_and_mul

    gate_up = _linear(x, gate_up_weight, gate_up_scale)
    act = torch.empty(
        (gate_up.shape[0], gate_up.shape[1] // 2),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    silu_and_mul(gate_up, act)
    return _linear(act, down_weight, down_scale)


def deepseek_v2_unquantized_moe_layer_baseline(**call):
    """post_attention_layernorm, then DeepseekV2MoE.forward_normal on it with unquantized
    experts, the shared one in their last slot (_shared_slot_moe at No)."""
    from aiter import QuantType

    return _shared_slot_moe(QuantType.No, **call)


def deepseek_v2_unquantized_moe_layer_shared_apart_baseline(**call):
    """post_attention_layernorm, then DeepseekV2MoE.forward_normal on it with unquantized
    experts and the shared expert apart (_shared_apart_moe at No)."""
    from aiter import QuantType

    return _shared_apart_moe(QuantType.No, **call)


def deepseek_v2_fp8_moe_layer_shared_apart_baseline(**call):
    """post_attention_layernorm, then DeepseekV2MoE.forward_normal on it with fp8 experts and
    an fp8 shared expert apart (_shared_apart_moe at per_128x128)."""
    from aiter import QuantType

    return _shared_apart_moe(QuantType.per_128x128, **call)


def _shared_apart_moe(
    quant_type,
    hidden_states,
    norm_weight,
    norm_eps,
    router_weight,
    correction_bias,
    w1,
    w2,
    shared_gate_up_weight,
    shared_down_weight,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
    w1_scale=None,
    w2_scale=None,
    shared_gate_up_scale=None,
    shared_down_scale=None,
    out_dtype=None,
    out=None,
):
    """post_attention_layernorm, then the shared expert (_mlp, fp8 linears with their scales),
    _route's top-k, aiter's fused_moe on the experts at ``quant_type``, their sum; the stream
    added (one rank: no all-reduce)."""
    _check_aiter_path()
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe

    x = _rms_norm(hidden_states, norm_weight, norm_eps)
    shared = _mlp(
        x,
        shared_gate_up_weight,
        shared_down_weight,
        shared_gate_up_scale,
        shared_down_scale,
    )
    weights, ids = _route(
        x,
        router_weight,
        correction_bias,
        topk,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    routed = _aiter(fused_moe)(
        x,
        w1,
        w2,
        weights,
        ids,
        quant_type=quant_type,
        activation=ActivationType.Silu,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    return hidden_states + (routed + shared)


def deepseek_v2_dense_layer_baseline(
    hidden_states,
    norm_weight,
    norm_eps,
    gate_up_weight,
    down_weight,
    out_dtype=None,
    out=None,
):
    """post_attention_layernorm, then _mlp on it; the stream added (one rank: no
    all-reduce)."""
    _check_aiter_path()
    return hidden_states + _mlp(_rms_norm(hidden_states, norm_weight, norm_eps), gate_up_weight, down_weight)


def deepseek_v2_fp8_dense_layer_baseline(
    hidden_states,
    norm_weight,
    norm_eps,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
    out_dtype=None,
    out=None,
):
    """post_attention_layernorm, then _mlp on it with fp8 linears; the stream added (one
    rank: no all-reduce)."""
    _check_aiter_path()
    x = _rms_norm(hidden_states, norm_weight, norm_eps)
    return hidden_states + _mlp(x, gate_up_weight, down_weight, gate_up_scale, down_scale)


def _unquantized_linear(x, weight):
    """UnquantizedLinearMethod.apply on aiter: on gfx950 the decode shapes
    _prefer_triton_a16w16 takes (K up to 2048) through aiter's triton gemm_a16w16, the rest
    through tgemm.mm (_linear)."""
    from sglang.srt.layers.quantization import unquant

    if unquant._is_gfx95 and unquant._prefer_triton_a16w16(x, weight):
        return unquant._aiter_triton_a16w16(x, weight, None, dtype=x.dtype)
    return _linear(x, weight, None)


def _layer_linear(x, weight, scale):
    """An attention linear as its quant method runs it: fp8 (with a scale) through _linear's
    aiter_w8a8_block_fp8_linear, bf16 through _unquantized_linear."""
    if scale is None:
        return _unquantized_linear(x, weight)
    return _linear(x, weight, scale)


def _mla_qkv(
    x,
    qkv_a_weight,
    qkv_a_scale,
    q_a_norm_weight,
    kv_a_norm_weight,
    eps,
    q_b_weight,
    q_b_scale,
):
    """forward_absorb_rocm_prepare up to q_b, from the normed stream x: fused_qkv_a_proj_with_mqa
    and q_b_proj as their quant methods run them (_layer_linear: fp8 activations quantized in
    aiter_w8a8_block_fp8_linear), both RMSNorms in one fused_qk_rmsnorm: q, q_lora, k_latent,
    k_pe. sglang's fused RMSNorm + fp8 quant paths stay off for block fp8 linears:
    _is_block_scale_fp8 reads weight_scale, which they keep as weight_scale_inv."""
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        fused_qk_rmsnorm_bf16,
    )

    q_lora_rank, kv_lora_rank = q_a_norm_weight.shape[0], kv_a_norm_weight.shape[0]
    qkv = _layer_linear(x, qkv_a_weight, qkv_a_scale)
    q_a, kv, k_pe = qkv.split([q_lora_rank, kv_lora_rank, qkv.shape[-1] - q_lora_rank - kv_lora_rank], dim=-1)
    q_lora, k_latent = fused_qk_rmsnorm_bf16(q_a, q_a_norm_weight, eps, kv, kv_a_norm_weight, eps)
    q = _layer_linear(q_lora, q_b_weight, q_b_scale)
    return q, q_lora, k_latent, k_pe


@functools.cache
def _unit_scale(device):
    """attn_mqa.k_scale of an fp8 KV cache stored unscaled."""
    import torch

    return torch.ones((), dtype=torch.float32, device=device)


def _mla_sparse(
    q,
    k_latent,
    k_pe,
    positions,
    out_cache_loc,
    kv_cache,
    topk_indices,
    sm_scale,
    cos_cache,
    sin_cache,
    w_uk_t,
    w_uv,
    w_scale,
    o_proj_weight,
    o_proj_scale,
    residual,
    decode,
):
    """forward_absorb_rocm_core after q_b: the W_UK product, RoPE and the KV rows' write into
    kv_cache, emitting q in fp8 (fp8 weights in one aiter kernel,
    fused_fp8_bmm_rope_cat_and_cache_mla; bf16 ones through torch.bmm, then
    fused_qk_rope_cat_and_cache_mla), the DSA backend's triton sparse MLA (a ``decode``'s
    triton_sparse_mla_decode_splitk, _forward_triton_decode; target-verify's and
    draft-extend's triton_sparse_mla_fwd, forward_extend, its max_topk_length the top-k's
    width as the CUDA graphs' metadata, max_seq_len_k the page table's width, gives it), the
    W_UV product (fp8 through aiter's batched fp8 gemm at sglang's decode tiles,
    _absorb_bmm_config; bf16 through torch.bmm), then o_proj as its quant method runs it
    (_layer_linear), plus the residual stream (one rank: no all-reduce)."""
    import torch
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
    )
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )
    from sglang.srt.layers.rocm_linear_utils import (
        fused_fp8_bmm_rope_cat_and_cache_mla,
        fused_qk_rope_cat_and_cache_mla,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        _absorb_bmm_config,
    )

    n_tokens, (n_heads, latent, nope) = q.shape[0], w_uk_t.shape
    q_nope, q_pe = q.view(n_tokens, n_heads, -1).split([nope, q.shape[1] // n_heads - nope], -1)
    cache = kv_cache.view(kv_cache.shape[0], 1, -1)
    k_rope_cache = (
        k_latent.unsqueeze(1),
        k_pe.unsqueeze(1),
        cache,
        out_cache_loc,
        positions,
        cos_cache.view(cos_cache.shape[0], 1, 1, -1),
        sin_cache.view(sin_cache.shape[0], 1, 1, -1),
    )
    if w_scale is None:  # bf16: sglang's w_kc is w_uk_t transposed
        q_latent = torch.bmm(q_nope.transpose(0, 1), w_uk_t.transpose(1, 2))
        q_cat, *_ = fused_qk_rope_cat_and_cache_mla(
            q_latent.transpose(0, 1),
            q_pe,
            *k_rope_cache,
            _unit_scale(q.device),
            False,
            q_out_dtype=kv_cache.dtype,
        )
    else:
        q_cat, *_ = fused_fp8_bmm_rope_cat_and_cache_mla(
            q_nope.transpose(0, 1),
            w_uk_t,
            w_scale,
            q_pe,
            *k_rope_cache,
            group_size=128,
            k_scale=_unit_scale(q.device),
            q_out_dtype=kv_cache.dtype,
        )
    attend = (
        triton_sparse_mla_decode_splitk
        if decode
        else functools.partial(
            triton_sparse_mla_fwd,
            topk_length=None,
            max_topk_length=topk_indices.shape[-1],
        )
    )
    o_latent = attend(
        q_nope=q_cat[..., :latent],
        q_rope=q_cat[..., latent:],
        kv=cache,
        indices=topk_indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=latent,
    )
    o = torch.empty(n_tokens, n_heads, w_uv.shape[1], dtype=torch.bfloat16, device=q.device)
    o_latent = o_latent.view(n_tokens, n_heads, latent)
    if w_scale is None:  # bf16: sglang's w_vc is w_uv transposed
        torch.bmm(o_latent.transpose(0, 1), w_uv.transpose(1, 2), out=o.transpose(0, 1))
    else:
        fp8_bmm(
            X=o_latent,
            WQ=w_uv,
            w_scale=w_scale,
            group_size=128,
            YQ=o,
            transpose_bm=True,
            transpose_bm_in=True,
            dtype=torch.bfloat16,
            config=_absorb_bmm_config(n_heads, n_tokens, w_uv.shape[1]),
        )
    return residual + _layer_linear(o.flatten(1), o_proj_weight, o_proj_scale)


# The fewest rows of a target-verify (one request, MTP's 3 draft tokens and the last one): an
# fp8-W_UK (main model) call with fewer decodes. The call does not name its forward mode.
_MIN_VERIFY_ROWS = 4


def deepseek_v2_attention_baseline(
    hidden_states,
    input_norm_weight,
    input_norm_eps,
    qkv_a_weight,
    qkv_a_scale,
    q_a_norm_weight,
    kv_a_norm_weight,
    eps,
    q_b_weight,
    q_b_scale,
    positions,
    out_cache_loc,
    kv_cache,
    topk_indices,
    sm_scale,
    cos_cache,
    sin_cache,
    w_uk_t,
    w_uv,
    w_scale,
    o_proj_weight,
    o_proj_scale,
    output=None,
):
    """input_layernorm, _mla_qkv on the normed stream, then _mla_sparse on its q, k_latent and
    k_pe, the stream added (one rank: no all-reduce). A carried top-k's call decodes on MTP's
    layer (bf16 W_UK: its draft steps after the first) and below _MIN_VERIFY_ROWS rows, and
    is a target-verify otherwise."""
    _check_aiter_path()
    decode = w_scale is None or hidden_states.shape[0] < _MIN_VERIFY_ROWS
    q, _, k_latent, k_pe = _mla_qkv(
        _rms_norm(hidden_states, input_norm_weight, input_norm_eps),
        qkv_a_weight,
        qkv_a_scale,
        q_a_norm_weight,
        kv_a_norm_weight,
        eps,
        q_b_weight,
        q_b_scale,
    )
    return _mla_sparse(
        q,
        k_latent,
        k_pe,
        positions,
        out_cache_loc,
        kv_cache,
        topk_indices,
        sm_scale,
        cos_cache,
        sin_cache,
        w_uk_t,
        w_uv,
        w_scale,
        o_proj_weight,
        o_proj_scale,
        hidden_states,
        decode,
    )


def _index_k(x, wk, wk_scale, k_norm_weight, k_norm_bias, k_norm_eps):
    """The indexer's k before RoPE, from the normed stream x: wk as its quant method runs it
    (_layer_linear), then k_norm, sglang's bf16 LayerNorm through aiter's layernorm2d_fwd
    (forward_hip)."""
    from aiter import layernorm2d_fwd

    k = _layer_linear(x, wk, wk_scale)
    return layernorm2d_fwd(k, k_norm_weight, k_norm_bias, k_norm_eps)


def _index_rope(q_rope, k_rope, positions, cos_cache, sin_cache):
    """The indexer's RoPE in place, GPT-J style, as aiter's RotaryEmbedding.forward_hip runs
    it: rope_cached_positions_2c_fwd_inplace on q's and k's rope values."""
    import aiter

    n_tokens, rope_dim = positions.shape[0], q_rope.shape[-1]
    aiter.rope_cached_positions_2c_fwd_inplace(
        q_rope.view(1, n_tokens, -1, rope_dim),
        k_rope.view(1, n_tokens, -1, rope_dim),
        cos_cache.view(cos_cache.shape[0], 1, 1, -1),
        sin_cache.view(sin_cache.shape[0], 1, 1, -1),
        positions.view(1, n_tokens),
        1,
        reuse_freqs_front_part=True,
        nope_first=False,
    )


@functools.cache
def _head_gate(heads_scale, softmax_scale):
    """Indexer._get_logits_head_gate after weights_proj, compiled as sglang compiles it
    (torch.compile, dynamic shapes): the heads' weights times heads_scale, q's scales and
    softmax_scale, in fp32."""
    import torch

    @torch.compile(dynamic=True)
    def gate(weights, q_scale):
        weights = weights * heads_scale
        return weights.unsqueeze(-1) * q_scale * softmax_scale

    return gate


def _indexer_qk(
    x,
    q_lora,
    positions,
    out_cache_loc,
    pages,
    wq_b,
    wq_b_scale,
    wk,
    wk_scale,
    weights_proj,
    k_norm_weight,
    k_norm_bias,
    k_norm_eps,
    cos_cache,
    sin_cache,
):
    """Indexer.forward_cuda's standard path up to its scores, from the normed stream x
    (_get_q_k_bf16, act_quant, _store_index_k_cache, _get_logits_head_gate): wq_b as its
    quant method runs it, k through _index_k, both turned by _index_rope, then
    fast_hadamard_transform; q's quant through act_quant with ue8m0 scales, k's into the
    pages through aiter's indexer_k_quant_and_cache (ue8m0 scales, preshuffled), then the
    heads' weights (weights_proj as UnquantizedLinearMethod runs it, then _head_gate): q in
    fp8, the weights."""
    from aiter.ops.cache import indexer_k_quant_and_cache
    from fast_hadamard_transform import hadamard_transform
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant

    n_heads, head_dim = weights_proj.shape[0], wk.shape[0]
    rope_dim = 2 * cos_cache.shape[1]
    query = _layer_linear(q_lora, wq_b, wq_b_scale).view(-1, n_heads, head_dim)
    key = _index_k(x, wk, wk_scale, k_norm_weight, k_norm_bias, k_norm_eps)
    _index_rope(query[..., :rope_dim], key[:, :rope_dim], positions, cos_cache, sin_cache)
    query = hadamard_transform(query, scale=head_dim**-0.5)
    key = hadamard_transform(key, scale=head_dim**-0.5)
    q_fp8, q_scale = act_quant(query, 128, "ue8m0")
    indexer_k_quant_and_cache(key, pages, out_cache_loc, 128, "ue8m0", preshuffle=True)
    weights = _unquantized_linear(x, weights_proj)
    return q_fp8, _head_gate(n_heads**-0.5, head_dim**-0.5)(weights, q_scale)


def _gfx950_indexer(
    x,
    wq_b,
    wk,
    weights_proj,
    k_norm_weight,
    k_norm_bias,
    k_norm_eps,
    cos_cache,
    sin_cache,
    topk,
    fp8_dtype,
):
    """The layer's gfx950 fused indexer (Gfx950FusedIndexer), made once per weights as
    Indexer.__init__ makes it, with the workspace it takes there (as wide as the RoPE table's
    positions and MTP's draft tokens, in pages, 4 pages more); None where sglang leaves it
    off: wq_b, wk or weights_proj not in bf16, shapes the kernels do not take, no
    preshuffled paged MQA in aiter, or kernels that do not build. Where it is on, the fused
    fp8 writer is off."""
    import types

    import torch
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fused_decode, loader
    from sglang.srt.layers.attention.dsa.utils import (
        aiter_can_use_preshuffle_paged_mqa,
    )

    params = (wq_b, weights_proj, k_norm_weight, k_norm_bias, cos_cache, sin_cache)
    cached = getattr(wk, "_aklr_gfx950", None)
    if cached is not None and cached[1] == k_norm_eps and all(a is b for a, b in zip(cached[0], params)):
        return cached[2]
    shape = (
        wk.shape[0],
        2 * cos_cache.shape[1],
        weights_proj.shape[0],
        topk,
        wq_b.shape[1],
        wk.shape[1],
    )
    supported = (
        fused_decode.HEAD_DIM,
        fused_decode.ROPE_DIM,
        fused_decode.N_HEADS,
        fused_decode.INDEX_TOPK,
        fused_decode.Q_LORA_RANK,
        fused_decode.HIDDEN_SIZE,
    )
    state = None
    if (
        all(w.dtype == torch.bfloat16 for w in (wq_b, wk, weights_proj))
        and fused_decode.supported_hardware()
        and shape == supported
        and aiter_can_use_preshuffle_paged_mqa()
        and loader.modules_or_none() is not None
    ):
        layer = types.SimpleNamespace
        state = fused_decode.Gfx950FusedIndexer(
            layer(
                n_heads=weights_proj.shape[0],
                softmax_scale=wk.shape[0] ** -0.5,
                wq_b=layer(weight=wq_b),
                wk=layer(weight=wk),
                weights_proj=layer(weight=weights_proj),
                k_norm=layer(weight=k_norm_weight, bias=k_norm_bias, variance_epsilon=k_norm_eps),
                rotary_emb=layer(cos_cache=cos_cache, sin_cache=sin_cache),
            )
        )
        pages = -(-(cos_cache.shape[0] + 8) // 64) + 4
        fused_decode.prealloc_workspace(device=x.device, max_cols=pages * 64, fp8_dtype=fp8_dtype)
    wk._aklr_gfx950 = (params, k_norm_eps, state)
    return state


def _gfx950_topk(state, x, q_lora, positions, out_cache_loc, index_k_cache, seq_lens, page_table):
    """Indexer._gfx950_fused_decode's four launches on a decode's or a target-verify's rows
    (one GEMV for q, k and the heads' weights; k_norm, RoPE, the Hadamard transform, the
    quants and k's store; the scores; their top-k as slots), or None where it declines the
    call: more rows than MAX_ROWS, or a page table wider than the workspace."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fused_decode

    rows, page_size = x.shape[0], fused_decode.PAGE_SIZE
    if not (
        1 <= rows <= fused_decode.MAX_ROWS
        and state.ensure_workspace(
            device=x.device,
            max_cols=page_size * page_table.shape[1],
            fp8_dtype=index_k_cache.dtype,
        )
    ):
        return None
    return state.run(
        x=x,
        q_lora=q_lora,
        positions=positions,
        out_cache_loc=out_cache_loc,
        kv_cache=index_k_cache.view(-1, page_size, fused_decode.CACHE_TOK_STRIDE),
        kv_cache_read=index_k_cache,
        seqlens_int32=seq_lens,
        row_ends_int32=seq_lens,
        page_table_64=page_table,
        rows=rows,
    )


def _topk_plan(seq_lens):
    """topk_transform_paged_v2's plan of seq_lens, made once per tensor as sglang makes it once
    per forward (DSAMetadata.topk_v2_plan) for every layer."""
    plan = getattr(seq_lens, "_aklr_topk_plan", None)
    if plan is None:
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        plan = seq_lens._aklr_topk_plan = plan_topk_v2(seq_lens)
    return plan


def _page_table_1(page_table):
    """The page table at page size 1 (DSAMetadata.page_table_1: each page's 64 slots), made
    once per tensor as sglang makes it once per forward for every layer."""
    table = getattr(page_table, "_aklr_page_table_1", None)
    if table is None:
        import torch

        slots = torch.arange(64, dtype=page_table.dtype, device=page_table.device)
        table = (page_table.unsqueeze(-1) * 64 + slots).flatten(1)
        page_table._aklr_page_table_1 = table
    return table


@functools.cache
def _cu_seqlens_q(rows, device):
    """DSAMetadata.cu_seqlens_q of the CUDA graphs: one query a row, arange(rows + 1)."""
    import torch

    return torch.arange(rows + 1, dtype=torch.int32, device=device)


def _max_len(seq_lens):
    """seq_lens' largest, read once per tensor as sglang reads it once per forward (from
    seq_lens_cpu, to pick the decode CUDA graph)."""
    longest = getattr(seq_lens, "_aklr_max_len", None)
    if longest is None:
        longest = seq_lens._aklr_max_len = int(seq_lens.max())
    return longest


def _topk(logits, seq_lens, page_table, topk):
    """DSAIndexerMetadata.topk_transform on HIP, the scores' top-k as slots:
    fast_topk_transform_fused over the page table at page size 1 (sglang turns
    SGLANG_OPT_USE_TOPK_V2 off for DSA models on HIP unless it is set), or
    topk_transform_paged_v2 over the page table where SGLANG_OPT_USE_TOPK_V2 is set on.
    """
    from sglang.srt.environ import envs

    rows, v2 = logits.shape[0], envs.SGLANG_OPT_USE_TOPK_V2
    if v2.is_set() and v2.get():
        import torch
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

        out = logits.new_empty((rows, topk), dtype=torch.int32)
        topk_transform_paged_v2(logits, seq_lens, page_table, out, 64, _topk_plan(seq_lens))
        return out
    from sgl_kernel import fast_topk_transform_fused

    return fast_topk_transform_fused(
        score=logits,
        lengths=seq_lens,
        page_table_size_1=_page_table_1(page_table),
        cu_seqlens_q=_cu_seqlens_q(rows, logits.device),
        topk=topk,
    )


def deepseek_v2_attention_indexer_baseline(
    hidden_states,
    input_norm_weight,
    input_norm_eps,
    qkv_a_weight,
    qkv_a_scale,
    q_a_norm_weight,
    kv_a_norm_weight,
    eps,
    q_b_weight,
    q_b_scale,
    wq_b,
    wk,
    weights_proj,
    k_norm_weight,
    k_norm_bias,
    k_norm_eps,
    index_cos_cache,
    index_sin_cache,
    positions,
    out_cache_loc,
    index_k_cache,
    seq_lens,
    page_table,
    kv_cache,
    sm_scale,
    cos_cache,
    sin_cache,
    w_uk_t,
    w_uv,
    w_scale,
    o_proj_weight,
    o_proj_scale,
    topk_indices,
    output=None,
    *,
    wq_b_scale=None,
    wk_scale=None,
):
    """input_layernorm; _mla_qkv on the normed stream; then Indexer.forward_cuda as it runs
    on HIP for the call's forward mode (wq_b and wk with their scales, if fp8): MTP's layer
    (bf16 W_UK) runs the indexer in its draft extend only, and the main model's call
    decodes below _MIN_VERIFY_ROWS rows and is a target-verify otherwise. A decode in which
    no token has more than topk positions replays the dense decode CUDA graph
    (_should_skip_logits_computation): k alone (_index_k, RoPE beside a dummy q, the
    Hadamard transform) stored through aiter's indexer_k_quant_and_cache, then the top-k of
    zero scores. A target-verify or another decode on the gfx950 fused indexer
    (_gfx950_indexer) runs its four launches. The rest run the standard path (_indexer_qk),
    then aiter_paged_mqa_logits over the page table's width and the top-k as slots (_topk).
    Never aiter's fused fp8 writer: it stores k unrotated, and the routes fall back where it
    is on. Then _mla_sparse on q, k_latent and k_pe over the slots, the stream added (one
    rank: no all-reduce)."""
    _check_aiter_path()
    import torch
    from aiter.ops.cache import indexer_k_quant_and_cache
    from fast_hadamard_transform import hadamard_transform
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import aiter_paged_mqa_logits

    n_tokens, topk = hidden_states.shape[0], topk_indices.shape[1]
    draft_extend = w_scale is None
    decode = not draft_extend and n_tokens < _MIN_VERIFY_ROWS
    x = _rms_norm(hidden_states, input_norm_weight, input_norm_eps)
    q, q_lora, k_latent, k_pe = _mla_qkv(
        x,
        qkv_a_weight,
        qkv_a_scale,
        q_a_norm_weight,
        kv_a_norm_weight,
        eps,
        q_b_weight,
        q_b_scale,
    )
    pages = index_k_cache.view(index_k_cache.shape[0], 64, -1)
    fused = _gfx950_indexer(
        x,
        wq_b,
        wk,
        weights_proj,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        index_cos_cache,
        index_sin_cache,
        topk,
        index_k_cache.dtype,
    )
    slots = None
    if decode and _max_len(seq_lens) <= topk:
        index_k = _index_k(x, wk, wk_scale, k_norm_weight, k_norm_bias, k_norm_eps)
        k_rope = index_k[:, : 2 * index_cos_cache.shape[1]]
        dummy_q = torch.empty_like(k_rope)
        _index_rope(dummy_q, k_rope, positions, index_cos_cache, index_sin_cache)
        index_k = hadamard_transform(index_k, scale=index_k.shape[1] ** -0.5)
        indexer_k_quant_and_cache(index_k, pages, out_cache_loc, 128, "ue8m0", preshuffle=True)
        scores = torch.zeros(n_tokens, topk, dtype=torch.float32, device=x.device)
        slots = _topk(scores, seq_lens, page_table, topk)
    elif fused is not None and not draft_extend:
        slots = _gfx950_topk(
            fused,
            x,
            q_lora,
            positions,
            out_cache_loc,
            index_k_cache,
            seq_lens,
            page_table,
        )
    if slots is None:
        q_fp8, weights = _indexer_qk(
            x,
            q_lora,
            positions,
            out_cache_loc,
            pages,
            wq_b,
            wq_b_scale,
            wk,
            wk_scale,
            weights_proj,
            k_norm_weight,
            k_norm_bias,
            k_norm_eps,
            index_cos_cache,
            index_sin_cache,
        )
        logits = aiter_paged_mqa_logits(
            q_fp8,
            pages.unsqueeze(2),
            weights.squeeze(2),
            seq_lens,
            page_table,
            64 * page_table.shape[1],
            preshuffle=True,
            kv_block_size=64,
        )
        slots = _topk(logits, seq_lens, page_table, topk)
    attended = _mla_sparse(
        q,
        k_latent,
        k_pe,
        positions,
        out_cache_loc,
        kv_cache,
        slots,
        sm_scale,
        cos_cache,
        sin_cache,
        w_uk_t,
        w_uv,
        w_scale,
        o_proj_weight,
        o_proj_scale,
        hidden_states,
        decode,
    )
    return attended, slots


def deepseek_v2_fp8_attention_indexer_baseline(**call):
    """deepseek_v2_attention_indexer_baseline on GLM-5.3's layers: every linear but
    weights_proj in fp8 through aiter_w8a8_block_fp8_linear, wq_b and wk with wq_b_scale and
    wk_scale. The gfx950 fused indexer stays off (it takes bf16 wq_b and wk), so the
    indexer runs the standard path."""
    return deepseek_v2_attention_indexer_baseline(**call)


def deepseek_v2_eh_proj_baseline(embeds, hidden, enorm_weight, hnorm_weight, eps, weight, output=None):
    """DeepseekModelNextN's start on HIP: sglang's RMSNorm of each input, torch.cat, then
    eh_proj as sglang's bf16 ReplicatedLinear (_linear), one rank."""
    _check_aiter_path()
    import torch

    x = torch.cat(
        [_rms_norm(embeds, enorm_weight, eps), _rms_norm(hidden, hnorm_weight, eps)],
        dim=-1,
    )
    return _linear(x, weight, None)
