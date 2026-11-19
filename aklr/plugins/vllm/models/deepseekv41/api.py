# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's routes to the DeepSeek-V4.1 ops, for the steps its mono decode kernels do not take.

Of a layer's two launches only the attention front (``deepseekv41_mono_k1``) is vLLM's too.
Its attention back multiplies MXFP8 rows by an MXFP8 wo_a and sums wo_b's output over the TP
ranks before the FFN seam, and its MoE keeps the shared expert as an MXFP8 MLP beside 384
FP4 slots: ``deepseekv41_mono_k2_attention`` and ``deepseekv41_mono_k2_ffn`` say neither, so
no route asks for them. Their kernels are reached one by one: the sparse MLA and the MXFP8
linears here, the routed experts through aiter's fused_moe."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins.sglang.models.deepseekv41 import (
    deepseekv41_mono_k1,
    deepseekv41_mxfp8_linear,
    deepseekv41_sparse_mla,
)
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []

# The InferenceX cookbook's nightly (4fab5a07) reports 0.33.0rc1.dev910
_VLLM = ("0.33.0rc1.dev910", "0.34.0")
# An fp8_ds_mla record: 576 data bytes, 8 scale bytes
_RECORD = 584
_ATTENTION = "vllm.models.deepseek_v41.amd.rocm.DeepseekV41ROCMAiterMLAAttention"
_MONO = "vllm.models.deepseek_v41.amd.mono_decode.MonoDecodeLayer"
_LINEAR = "vllm.model_executor.kernels.linear.mxfp8.rocm_native.RocmDotScaledMxfp8LinearKernel"


@vllm_route(f"{_ATTENTION}._forward_decode", f"{_ATTENTION}._aiter_sparse_mla", versions=_VLLM)
def _aiter_sparse_mla(
    self,
    q,
    output,
    swa_k_cache,
    swa_indices,
    swa_indptr,
    compressed_k_cache,
    topk_indices,
    topk_indptr,
    positions=None,
    output_mxfp8=None,
):
    """A decode's or a prefill's sparse MLA (both reach aiter's kernel through this method),
    asked for as one op call: the window cache and the compressed one in place, a token's
    indices at the width vLLM packed its ragged lists from."""
    import torch
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    swa, tokens = metadata[self.swa_cache_layer.prefix], q.shape[0]
    if swa_indices is swa.decode_swa_ragged_indices:
        width = swa.decode_swa_width
    elif swa_indices is swa.prefill_swa_ragged_indices:
        width = swa.prefill_swa_indices.numel() // swa.num_prefill_tokens
    else:
        return FallbackReason.OUT_OF_CONTRACT

    def pool(cache, page_size):
        # a page of 584-byte records in each block's row, which the layer's other caches share
        if (
            cache.dtype != torch.uint8
            or cache.shape[1:] != (page_size, _RECORD)
            or cache.stride()[1:] != (_RECORD, 1)
        ):
            return None
        return cache.unsqueeze(2)

    pools = {"kv_cache": pool(swa_k_cache, swa.block_size)}
    arguments = {
        "q": q.unsqueeze(1),
        "indices": swa_indices.new_empty((tokens, 1, width)),
        "sinks": self.attn_sink[: q.shape[1]].float(),
        "sm_scale": self.scale,
    }
    if compressed_k_cache is not None:
        block_size = metadata[self.compressed_cache_prefix].block_size
        pools["extra_kv_cache"] = pool(compressed_k_cache, block_size // self.compress_ratio)
        arguments["extra_indices"] = topk_indices.new_empty((tokens, 1, self.topk_indices_buffer.shape[-1]))
    if positions is not None:
        arguments["rope_freqs"] = self.rotary_emb.cos_sin_cache
        # vLLM's positions past a first row start 8 bytes past the 16 AKLR's dispatch requires
        arguments["positions"] = positions if positions.data_ptr() % 16 == 0 else positions.clone()
    if any(cache is None for cache in pools.values()):
        return FallbackReason.OUT_OF_CONTRACT
    # shortcut: the op is asked for, which records the call, and the attention is left to
    # vLLM, whose kernel stores the rows as MXFP8 on gfx950 (the op's are bf16) and reads
    # ragged indices and a cos-then-sin RoPE table (they stand in by shape here); run the op
    # with both converted and vLLM's quantization after it once a kernel serves it
    reason = deepseekv41_sparse_mla.fallback_reason(**arguments, **pools, kv_cache_layout="strided_pages")
    return reason or FallbackReason.OUT_OF_CONTRACT


@vllm_route(_MONO, f"{_MONO}.__call__", versions=_VLLM)
def _mono_decode(
    self,
    layer,
    x,
    positions,
    residual,
    post_mix,
    res_mix,
    pre_mix,
    engram_hashes=None,
    engram_mask=None,
):
    """A decode-only step's layer where vLLM's mono kernels do not take the step (more than
    48 rows): its attention front as the launch a monokernel runs it in, asked for with the
    layer's own tensors on this rank. The route is declared on the class, whose source holds
    the call it replaces: the layer calls its mono object."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    attn, tokens = layer.attn, x.shape[0]
    metadata = get_forward_context().attn_metadata if is_forward_context_available() else None
    swa = metadata.get(attn.swa_cache_layer.prefix) if isinstance(metadata, dict) else None
    if (
        self.ffn_only
        or swa is None
        or swa.num_prefills
        or not swa.num_decodes
        or swa.num_decode_tokens > tokens
        # vLLM's own answer for the step: no reason, and its kernels take it
        or self._decode_step(layer, tokens)[0] is None
        or len(
            {layer.attn_norm.variance_epsilon, attn.q_norm.variance_epsilon, attn.kv_norm.variance_epsilon}
        )
        != 1
    ):
        return FallbackReason.OUT_OF_CONTRACT
    owed = {}
    if residual is None:
        # the embedding in every stream (stream 0 the input), or a pipeline stage's streams
        # under the pre-mix carried in: no post is owed
        if x.dim() == 3:
            residual = x
            if pre_mix is not None:
                owed["prev_pre_mix"] = pre_mix.reshape(tokens, -1)
        else:
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
    else:
        owed["prev_pre_mix"] = pre_mix.reshape(tokens, -1)
        if layer.engram is None or engram_hashes is None:
            owed.update(x=x, post_mix=post_mix.reshape(tokens, -1), comb_mix=res_mix)
    # shortcut: the launch is asked for, which records the layer's call, and the layer is
    # left to vLLM (an Engram layer's settled streams and the cos-then-sin RoPE table stand
    # in by shape); run it here once a kernel serves it and the layer's back has its ops
    reason = deepseekv41_mono_k1.fallback_reason(
        residual=residual,
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm_weight=layer.attn_norm.weight,
        wqkv=attn.fused_wqa_wkv.weight,
        wqkv_scale=attn.fused_wqa_wkv.weight_scale,
        q_norm_weight=attn.q_norm.weight,
        kv_norm_weight=attn.kv_norm.weight,
        wq_b=attn.wq_b.weight,
        wq_b_scale=attn.wq_b.weight_scale,
        rope_freqs=attn.rotary_emb.cos_sin_cache,
        positions=positions[:tokens],
        hc_rms_eps=layer.rms_norm_eps,
        hc_eps=layer.hc_eps,
        norm_eps=layer.attn_norm.variance_epsilon,
        sinkhorn_iters=layer.hc_sinkhorn_iters,
        **owed,
    )
    return reason or FallbackReason.OUT_OF_CONTRACT


@vllm_route(_LINEAR, f"{_LINEAR}.apply_weights", versions=_VLLM)
def _apply_weights(self, layer, x, bias=None):
    """A linear whose weight keeps its 32 x 32 scale blocks, as one op call on the rows of x
    (bf16, or the MXFP8 rows of a QuantizedActivation with their scales), then vLLM's bias,
    returned at x's leading shape. The other weights are vLLM's dot_scaled and bf16 paths.
    Declared on the class: its callers are every quantization scheme's linear method."""
    import torch

    weight = layer.weight
    quantized = None if isinstance(x, torch.Tensor) else x
    rows, dtype, shape = (x, x.dtype, x.shape) if quantized is None else (x.data, x.orig_dtype, x.orig_shape)
    rows = rows.reshape(-1, shape[-1])
    if (
        layer.weight_scale.shape[0] == weight.shape[0]
        or weight.element_size() != 1
        or dtype != torch.bfloat16
        or not rows.shape[0]
    ):
        return FallbackReason.OUT_OF_CONTRACT
    arguments = {"x": rows, "weight": weight, "weight_scale": layer.weight_scale}
    if quantized is not None:
        arguments["x_scale"] = quantized.scale.reshape(rows.shape[0], -1)
    out = deepseekv41_mxfp8_linear(**arguments)
    if isinstance(out, FallbackReason):
        return out
    out = out.view(*shape[:-1], weight.shape[0])
    return out if bias is None else out + bias
