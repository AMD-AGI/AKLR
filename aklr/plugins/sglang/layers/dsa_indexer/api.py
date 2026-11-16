# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.dsa_indexer.contract import (
    DSA_INDEXER_KEY_PREPARE,
    DSA_INDEXER_PREPARE,
    HEAD_DIM,
    PAGE_SIZE,
    ROPE_DIM,
)

__all__ = ["dsa_indexer_key_prepare", "dsa_indexer_prepare"]

# Indexer's quantization, as deepseek_v2 builds it: one scale per 128 values, rounded up to a
# power of two
_QUANT_BLOCK = 128
_SCALE_FMT = "ue8m0"


@api(DSA_INDEXER_PREPARE)
def dsa_indexer_prepare(
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
): ...


@api(DSA_INDEXER_KEY_PREPARE)
def dsa_indexer_key_prepare(
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
): ...


@sglang_route("sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda", versions=("0.5.22",))
def _forward_cuda(self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
    """forward_cuda's standard path with its preparation (_get_q_k_bf16, act_quant,
    _store_index_k_cache, _get_logits_head_gate) as the op, then sglang's scores and top-k.
    sglang keeps every other branch: the key-only one (routed on _forward_cuda_k_only), its
    gfx950 fused decode, the graph split ops, the fused writers, dual streams and CP."""
    import torch
    from sglang.srt.layers.attention.dsa import dsa_indexer as indexer
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner import get_is_capture_mode

    hidden = _hidden_bf16(x)
    mode = forward_batch.forward_mode
    if (
        hidden is None
        or not isinstance(q_lora, torch.Tensor)
        or _module_falls_back(self, ("wq_b", "wk", "weights_proj"))
        or _batch_falls_back(indexer, forward_batch)
        or indexer.is_graph_dsa_split_op_surface(forward_batch)
        or self.dsa_enable_prefill_cp
        # the key-only branch, whose route is _forward_cuda_k_only's
        or self._should_skip_logits_computation(forward_batch)
        or _gfx950_fused_decode_may_serve(self, mode, hidden)
    ):
        return FallbackReason.OUT_OF_CONTRACT
    enable_dual_stream = (
        self.alt_stream is not None
        and get_is_capture_mode()
        and 0 < q_lora.shape[0] <= indexer.DUAL_STREAM_TOKEN_THRESHOLD
    )
    if enable_dual_stream:
        return FallbackReason.OUT_OF_CONTRACT
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    pool, index_k_cache = _index_k_cache(layer_id)
    if metadata is None or index_k_cache is None:
        return FallbackReason.OUT_OF_CONTRACT
    arguments = _key_arguments(self, hidden, positions, forward_batch, index_k_cache) | {
        "q_lora": q_lora,
        "wq_b": self.wq_b.weight,
        "weights_proj": self.weights_proj.weight,
    }
    reason = dsa_indexer_prepare.fallback_reason(**arguments)
    if reason is not None:
        return reason
    _invalidate(pool, layer_id)
    result = dsa_indexer_prepare(**arguments)
    if isinstance(result, FallbackReason):
        return result
    q_fp8, head_weights = result
    # [tokens, heads, 1], as _get_logits_head_gate returns them
    weights = head_weights.unsqueeze(-1)
    if forward_batch.seq_lens.numel() == 0:
        topk_result = torch.full(
            (hidden.shape[0], self.index_topk), -1, dtype=torch.int, device=hidden.device
        )
    elif mode.is_decode_or_idle() or mode.is_target_verify() or mode.is_draft_extend_v2():
        topk_result = self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)
    else:
        topk_result = self._get_topk_ragged(False, forward_batch, layer_id, q_fp8, weights, metadata)
    topk_result = indexer._broadcast_indexer_topk_from_rank0(topk_result)
    return indexer.maybe_capture_indexer_topk(layer_id, topk_result)


@sglang_route(
    "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._forward_cuda_k_only",
    versions=("0.5.22",),
)
def _forward_cuda_k_only(
    self,
    x,
    positions,
    forward_batch,
    layer_id,
    act_quant,
    metadata,
    return_indices=True,
    *,
    num_tokens=None,
    topk_result=None,
):
    """_forward_cuda_k_only's standard branch with its keys (_get_k_bf16,
    _store_index_k_cache) as the op, then sglang's select-all top-k on zero logits. The graph
    split ops' calls (num_tokens) and the fused writers' branches stay on sglang."""
    import torch
    from sglang.srt.layers.attention.dsa import dsa_indexer as indexer

    hidden = _hidden_bf16(x)
    mode = forward_batch.forward_mode
    if (
        num_tokens is not None
        or hidden is None
        or not (mode.is_extend_without_speculative() or mode.is_decode_or_idle())
        or _module_falls_back(self, ("wk",))
        or _batch_falls_back(indexer, forward_batch)
        # sglang would make it contiguous on the batch itself
        or not forward_batch.out_cache_loc.is_contiguous()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    pool, index_k_cache = _index_k_cache(layer_id)
    if index_k_cache is None:
        return FallbackReason.OUT_OF_CONTRACT
    arguments = _key_arguments(self, hidden, positions, forward_batch, index_k_cache)
    reason = dsa_indexer_key_prepare.fallback_reason(**arguments)
    if reason is not None:
        return reason
    _invalidate(pool, layer_id)
    result = dsa_indexer_key_prepare(**arguments)
    if isinstance(result, FallbackReason):
        return result
    if not return_indices:
        return None
    # the select-all top-k: with every length within index_topk, zero logits pick each slot
    seq_lens_expanded = metadata.get_seqlens_expanded()
    dummy_logits = torch.zeros(
        seq_lens_expanded.shape[0], self.index_topk, dtype=torch.float32, device=hidden.device
    )
    raw_topk_result = metadata.topk_transform(dummy_logits, self.index_topk)
    if topk_result is not None:
        topk_result[: raw_topk_result.shape[0]] = raw_topk_result
        return None
    return raw_topk_result


def _hidden_bf16(x):
    """The bf16 hidden states _get_q_k_bf16 and _get_k_bf16 project: x, or the third of
    fused_rms_fp8_group_quant's (fp8, scale, bf16); None for anything else."""
    import torch

    if isinstance(x, tuple):
        x = x[2] if len(x) == 3 else None
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return None
    return x


def _module_falls_back(module, linears) -> bool:
    """Whether the layer prepares its index keys other than the op: keys not Hadamard-rotated
    (a fused writer), an RMSNorm k_norm, NeoX RoPE or rotary tables the op cannot read, another
    quantization, or a LoRA adapter on one of ``linears``."""
    import torch
    from sglang.srt.layers.layernorm import LayerNorm

    rotary = module.rotary_emb
    tables = [getattr(rotary, name, None) for name in ("cos_cache", "sin_cache")]
    k_norm = module.k_norm
    return (
        module.use_dsa_indexer_fusion
        or module.use_aiter_fused_fp8_writer
        or not isinstance(k_norm, LayerNorm)
        or not (k_norm.elementwise_affine and k_norm.use_bias)
        or module.is_neox_style
        or (module.head_dim, module.rope_head_dim) != (HEAD_DIM, ROPE_DIM)
        or (module.block_size, module.scale_fmt) != (_QUANT_BLOCK, _SCALE_FMT)
        or not all(
            isinstance(t, torch.Tensor)
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            and t.device == k_norm.weight.device
            and t.shape[-1] == ROPE_DIM // 2
            for t in tables
        )
        or any(getattr(getattr(module, name, None), "set_lora", False) for name in linears)
    )


def _batch_falls_back(indexer, forward_batch) -> bool:
    """Whether the call is one the op's branch does not reach in sglang: inside a piecewise or
    breakable CUDA graph (sglang's split ops), or with context parallelism (CP gathers the
    keys); and whether sglang stores the keys elsewhere than aiter's preshuffled fp8 pages."""
    return (
        indexer._is_in_piecewise_or_breakable_cuda_graph()
        or forward_batch.attn_cp_metadata is not None
        or indexer.is_cp_active(forward_batch)
        or not (indexer._use_aiter and indexer._use_aiter_preshuffle)
        or indexer._is_fp8_fnuz
    )


def _gfx950_fused_decode_may_serve(module, mode, hidden) -> bool:
    """Whether sglang's gfx950 fused decode indexer (four kernels, logits and top-k included)
    may take the call: it is left to sglang, which falls through to its standard path when it
    declines."""
    if not module.use_gfx950_fused_indexer or not (mode.is_decode_or_idle() or mode.is_target_verify()):
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import MAX_ROWS

    return hidden.shape[0] <= MAX_ROWS


def _index_k_cache(layer_id):
    """(pool, the layer's index-K pages [pages, PAGE_SIZE, HEAD_DIM + 4] in fp8), as
    _store_index_k_cache views them; (pool, None) where sglang would store elsewhere (another
    page size, a layer-split pool that owns only some layers)."""
    import torch
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    if pool.index_page_size != PAGE_SIZE or hasattr(pool, "_is_layer_owned"):
        return pool, None
    buffer = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
    if not buffer.is_contiguous() or buffer.numel() % (PAGE_SIZE * (HEAD_DIM + 4)):
        return pool, None
    return pool, buffer.view(-1, PAGE_SIZE, HEAD_DIM + 4).view(torch.float8_e4m3fn)


def _invalidate(pool, layer_id) -> None:
    """As _store_index_k_cache does before its store."""
    if hasattr(pool, "invalidate_index_buffer_for_layer"):
        pool.invalidate_index_buffer_for_layer(layer_id)


def _key_arguments(module, hidden, positions, forward_batch, index_k_cache):
    """The key's arguments of both ops: the layer's wk, k_norm and rotary tables, the batch's
    positions and slots (aiter stores the first of them, one a key row)."""
    rotary = module.rotary_emb
    out_cache_loc = forward_batch.out_cache_loc
    if out_cache_loc.shape[0] > hidden.shape[0]:
        out_cache_loc = out_cache_loc[: hidden.shape[0]]
    return {
        "x": hidden,
        "wk": module.wk.weight,
        "k_norm_weight": module.k_norm.weight,
        "k_norm_bias": module.k_norm.bias,
        "cos_cache": rotary.cos_cache.view(-1, ROPE_DIM // 2),
        "sin_cache": rotary.sin_cache.view(-1, ROPE_DIM // 2),
        "positions": positions,
        "out_cache_loc": out_cache_loc,
        "index_k_cache": index_k_cache,
        "k_norm_eps": module.k_norm.variance_epsilon,
    }
