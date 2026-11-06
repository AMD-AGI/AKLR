# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4.contract import (
    DEEPSEEK_V4_ATTENTION_Q_PREP,
    DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE,
)

__all__ = ["deepseek_v4_attention_q_prep", "deepseek_v4_sparse_mla_decode_inv_rope"]


@api(DEEPSEEK_V4_ATTENTION_Q_PREP)
def deepseek_v4_attention_q_prep(
    qkv_a,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    freqs,
    positions,
    swa_loc,
    kv_cache,
    q,
    page_size=256,
    q_eps=9.99999968265522e-21,
    kv_eps=9.99999968265522e-21,
    q_lora_quant=None,
    q_lora_scale=None,
): ...


@sglang_route("sglang.srt.models.deepseek_v4.MQALayer._forward_prepare", versions=("0.5.23",))
def _forward_prepare(
    self,
    x,
    positions,
    forward_batch,
    attn_backend,
    q_out=None,
    x_quant=None,
    q_rope_out=None,
    k_nope_out=None,
    k_rope_out=None,
):
    """sglang's wqkv_a, then q_norm, wq_b and the K store (which ropes q) as one op call, then
    the compressor and indexer as sglang runs them; returned as sglang returns them, (q, None).
    Only the arm sglang takes on ROCm gfx950 with the plain SWA store and a fused query RoPE;
    everything else stays on sglang."""
    import torch
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
    from sglang.srt.models import deepseek_v4 as dv4

    if _prepare_falls_back(dv4, self, x, forward_batch, q_out):
        return FallbackReason.OUT_OF_CONTRACT
    pool = dv4.get_token_to_kv_pool()
    n_tokens, rank, device = x.shape[0], self.q_lora_rank, x.device
    arguments = {
        "q_norm_weight": self.q_norm.weight.data,
        "kv_norm_weight": self.kv_norm.weight.data,
        "wq_b": self.wq_b.weight,
        "wq_b_scale": self.wq_b.weight_scale_mx_e8m0.view(torch.float8_e8m0fnu),
        "freqs": torch.view_as_real(self.freqs_cis).flatten(-2),
        "positions": positions,
        "swa_loc": attn_backend.get_swa_out_cache_loc(forward_batch),
        "kv_cache": pool.get_swa_raw_buffer(self.layer_id),
        "page_size": pool.swa_page_size,
        "q_eps": self.q_norm.variance_epsilon,
        "kv_eps": self.eps,
        "q": torch.empty((n_tokens, self.n_local_heads, self.head_dim), dtype=torch.bfloat16, device=device),
        "q_lora_quant": torch.empty((n_tokens, rank), dtype=torch.float8_e4m3fn, device=device),
        "q_lora_scale": torch.empty((n_tokens, rank // 32), dtype=torch.float8_e8m0fnu, device=device),
    }
    op = deepseek_v4_attention_q_prep
    # wqkv_a's output, a stand-in to ask the op before the GEMM runs
    stand_in = torch.empty((n_tokens, rank + self.head_dim), dtype=torch.bfloat16, device=device)
    reason = op.fallback_reason(qkv_a=stand_in, **arguments)
    if reason is not None:
        return reason
    del stand_in
    qkv_a, _ = self.wqkv_a(x_quant if x_quant is not None else x)
    out = op(qkv_a=qkv_a, **arguments)
    if isinstance(out, FallbackReason):
        return out
    del qkv_a
    q, q_lora_quant, q_lora_scale = out
    # what the indexer reads: wq_b's operand, as sglang's compute_q_b hands it on
    q_lora = Mxfp8Activation(q_lora_quant, q_lora_scale.view(torch.uint8))
    if self.compress_ratio in (1, 2) and (self.compressor is not None or self.indexer is not None):
        if dv4._low_ratio_sources_break(attn_backend, forward_batch):
            dv4.bcg_deepseek_v4_low_ratio_sources(self, x, q_lora, positions)
        else:
            attn_backend.forward_low_ratio_sources(
                layer=self, x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch
            )
    else:
        if self.indexer is not None:
            self.indexer(
                x=x, q_lora=q_lora, forward_batch=forward_batch, attn_backend=attn_backend, q_lora_quant=None
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)
    return q, None


def _prepare_falls_back(dv4, layer, x, forward_batch, q_out) -> bool:
    """Whether sglang prepares this call other than as the op computes it: off ROCm gfx950,
    without the fused wqkv_a, on unified KV, context parallelism, the fused QK norm-RoPE
    store or a replay-tail graph; when the K store does not rope q (a q_out buffer, a q head
    norm, SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE); when q_norm does not emit wq_b's E4M3FN
    operand, or wq_b is not on aiter's group-32 MXFP8 GEMM (a preshuffled or native weight,
    a bias, gathered or sequence-parallel output, the FlyDSL GEMM's rows); or with a uniform
    FP8 or non-V4 pool."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers import linear
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.layers.quantization import fp8_hip
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense as gfx95_dense

    hip = dv4._hip
    if not (dv4._is_hip and hip._is_gfx95_supported and layer.fuse_wqa_wkv):
        return True
    if (
        is_unified_kv_triton()
        or layer.use_fused_qk_norm_rope
        or dv4.is_cp_active(forward_batch)
        or dv4.in_replay_tail_graph()
        or not hip.fuses_q_rope_into_k_store(layer, q_out, unified=False, use_cp=False)
        or not layer.fused_rmsnorm_fake_quant
    ):
        return True
    wq_b = layer.wq_b
    if not layer._wq_b_native_consumer_checked:
        # as sglang's q_norm_fake_quant decides it, on first use
        layer._wq_b_native_consumer = gfx95_dense._mxfp8_consumer(wq_b)
        layer._wq_b_native_consumer_checked = True
    method = wq_b.quant_method
    if not (
        layer._wq_b_native_consumer
        and isinstance(method, Fp8LinearMethod)
        and not method.use_marlin
        and method.mxfp8_dense_backend.is_gfx95_mxfp8_aiter()
        and not getattr(wq_b, "mxfp8_aiter_preshuffled", False)
        and wq_b.bias is None
        and not wq_b.gather_output
        and not linear.get_forward().sp_active
        and (x.shape[0] < fp8_hip._FLYDSL_MXFP8_MIN_M or wq_b.weight_scale_mx_flydsl is None)
    ):
        return True
    pool = dv4.get_token_to_kv_pool()
    return pool.uniform_fp8 or KVLayout.parse(pool.kv_layout) is not KVLayout.V4


@api(DEEPSEEK_V4_SPARSE_MLA_DECODE_INV_ROPE)
def deepseek_v4_sparse_mla_decode_inv_rope(
    q,
    kv_cache,
    sparse_indices,
    sinks,
    rope_freqs,
    positions,
    kv_cache_layout="row_major",
    extra_kv_cache=None,
    extra_sparse_indices=None,
    sm_scale=0.04419417382415922,
    output=None,
): ...


@sglang_route("sglang.srt.layers.attention.hip_flash_mla.aiter_sparse_decode_fwd", versions=("0.5.23",))
def _aiter_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    attn_sink,
    softmax_scale,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    inv_rope=None,
    topk_length=None,
    extra_topk_length=None,
    **_,
):
    """The decode attention and its inverse RoPE as one op call, returned as sglang returns
    them, (output [b, s, heads, 512], None), for DeepSeek-V4's fp8 pools when the model
    passes inv_rope; sglang folds the top-k lengths into the index lists before this call, so
    topk_length and extra_topk_length are not read, as sglang does not read them here."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers.attention import hip_flash_mla

    b, s, h, d = q.shape
    n = b * s
    if (
        inv_rope is None
        # V4.1's compact pools take sglang's compact kernel
        or k_cache.shape[-1] in (KVLayout.V41.bytes_per_token, KVLayout.V41_FP4.bytes_per_token)
        # deterministic inference pins a split count, hence the combine order, below 1024 rows
        or (
            n < hip_flash_mla._AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS and hip_flash_mla.hip_attn_kv_splits() > 0
        )
    ):
        return FallbackReason.OUT_OF_CONTRACT
    rope_freqs, positions = inv_rope
    output = deepseek_v4_sparse_mla_decode_inv_rope(
        q=q.reshape(n, h, d),
        kv_cache=k_cache,
        sparse_indices=indices.reshape(n, -1),
        sinks=attn_sink,
        rope_freqs=rope_freqs,
        positions=positions.reshape(-1)[:n],
        extra_kv_cache=extra_k_cache,
        extra_sparse_indices=None if extra_k_cache is None else extra_indices_in_kvcache.reshape(n, -1),
        sm_scale=softmax_scale,
        kv_cache_layout="padded_pages",
    )
    if isinstance(output, FallbackReason):
        return output
    return output.view(b, s, h, d), None
