# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on HIP (SGLANG_USE_AITER=1, aiter's preshuffled index-K pages) for
the indexer's preparation in Indexer.forward_cuda's standard path (_get_q_k_bf16, act_quant,
_store_index_k_cache, _get_logits_head_gate) and in _forward_cuda_k_only (_get_k_bf16,
_store_index_k_cache), call for call."""

from __future__ import annotations

import functools
from types import SimpleNamespace

from aklr.plugins.sglang.layers.dsa_indexer.contract import HEAD_DIM, PAGE_SIZE, ROPE_DIM

__all__ = ["dsa_indexer_key_prepare_baseline", "dsa_indexer_prepare_baseline"]

# Indexer's quantization: block_size and scale_fmt as deepseek_v2 builds it
_QUANT_BLOCK = 128
_SCALE_FMT = "ue8m0"


def _check_aiter_path() -> None:
    """Raise unless sglang takes the aiter path the baseline replays, chosen at import."""
    from sglang.srt.layers.quantization import unquant

    if not unquant._use_aiter:
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on HIP")


def _linear(x, weight):
    """ReplicatedLinear's UnquantizedLinearMethod.apply on aiter: aiter's triton gemm_a16w16
    for the decode shapes _prefer_triton_a16w16 takes, else tgemm.mm."""
    from sglang.srt.layers.quantization import unquant

    if unquant._is_gfx95 and unquant._prefer_triton_a16w16(x, weight):
        return unquant._aiter_triton_a16w16(x, weight, None, dtype=x.dtype)
    return unquant.tgemm.mm(x, weight, None, otype=x.dtype)


def _layer_norm(x, weight, bias, eps):
    """k_norm, sglang's LayerNorm.forward_hip: aiter's layernorm2d_fwd when the dtypes agree."""
    import torch
    from sglang.srt.layers import layernorm

    if (
        layernorm._has_aiter_layer_norm
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == weight.dtype
    ):
        return layernorm.layer_norm(x, weight, bias, eps)
    return torch.nn.functional.layer_norm(x.to(weight.dtype), (x.shape[-1],), weight, bias, eps).to(x.dtype)


def _rope(positions, query, key, cos_cache, sin_cache):
    """The indexer's rotary_emb (aiter's RotaryEmbedding, interleaved) on views of query and
    key, in place: its forward_hip on the tables as aiter holds them, [rows, 1, 1, pairs]."""
    from aiter.rotary_embedding import RotaryEmbedding

    rotary = SimpleNamespace(
        cos_cache=cos_cache.view(cos_cache.shape[0], 1, 1, -1),
        sin_cache=sin_cache.view(sin_cache.shape[0], 1, 1, -1),
        is_neox_style=False,
        head_size=ROPE_DIM,
        rotary_dim=ROPE_DIM,
    )
    RotaryEmbedding.forward_hip(rotary, positions, query, key)


def _rotate_activation(x):
    """Indexer's Hadamard transform: dsa_indexer.rotate_activation."""
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    return rotate_activation(x)


def _key(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, positions, cos_cache, sin_cache, query=None):
    """The bf16 key, and the query when given (_get_q_k_bf16), else with a dummy query's RoPE
    (_get_k_bf16): wk, k_norm, RoPE on both in place, the Hadamard transform on each."""
    import torch

    key = _layer_norm(_linear(x, wk), k_norm_weight, k_norm_bias, k_norm_eps)
    k_rope = key[..., :ROPE_DIM]
    q_rope = torch.empty_like(k_rope) if query is None else query[..., :ROPE_DIM]
    _rope(positions, q_rope, k_rope, cos_cache, sin_cache)
    if query is not None:
        query = _rotate_activation(query)
    return _rotate_activation(key), query


def _store(key, index_k_cache, out_cache_loc):
    """_store_index_k_cache on aiter: indexer_k_quant_and_cache into the preshuffled pages."""
    from aiter.ops.cache import indexer_k_quant_and_cache

    assert index_k_cache.shape[1:] == (PAGE_SIZE, HEAD_DIM + 4)
    indexer_k_quant_and_cache(
        key, index_k_cache, out_cache_loc.contiguous(), _QUANT_BLOCK, _SCALE_FMT, preshuffle=True
    )


@functools.cache
def _head_gate():
    """_get_logits_head_gate after weights_proj, compiled as sglang compiles it (one fused
    kernel): weights * n_heads^-0.5, times each head's q_scale and softmax_scale."""
    import torch

    @torch.compile(dynamic=True)
    def head_gate(weights, q_scale, n_heads_scale: float, softmax_scale: float):
        weights = weights * n_heads_scale
        return weights.unsqueeze(-1) * q_scale * softmax_scale

    return head_gate


def dsa_indexer_prepare_baseline(
    x,
    q_lora,
    wq_b,
    wk,
    weights_proj,
    k_norm_weight,
    k_norm_bias,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    index_k_cache,
    k_norm_eps=1e-6,
    q_fp8=None,
    head_weights=None,
):
    """forward_cuda's standard path up to the scores: (q_fp8, head_weights), the keys stored;
    act_quant is sglang's tilelang kernel on HIP."""
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant

    _check_aiter_path()
    n_heads = weights_proj.shape[0]
    query = _linear(q_lora, wq_b).view(-1, n_heads, HEAD_DIM)
    key, query = _key(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, positions, cos_cache, sin_cache, query)
    quantized, q_scale = act_quant(query, _QUANT_BLOCK, _SCALE_FMT)
    _store(key, index_k_cache, out_cache_loc)
    weights = _head_gate()(_linear(x, weights_proj), q_scale, n_heads**-0.5, HEAD_DIM**-0.5)
    return quantized, weights.squeeze(-1)


def dsa_indexer_key_prepare_baseline(
    x,
    wk,
    k_norm_weight,
    k_norm_bias,
    cos_cache,
    sin_cache,
    positions,
    out_cache_loc,
    index_k_cache,
    k_norm_eps=1e-6,
    no_values=None,
):
    """_forward_cuda_k_only's standard branch: the keys stored, and no values."""
    import torch

    _check_aiter_path()
    key, _ = _key(x, wk, k_norm_weight, k_norm_bias, k_norm_eps, positions, cos_cache, sin_cache)
    _store(key, index_k_cache, out_cache_loc)
    return torch.empty(0, dtype=torch.float32, device=x.device) if no_values is None else no_values
