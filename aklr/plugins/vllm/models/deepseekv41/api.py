# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM's routes to the DeepSeek-V4.1 ops: the kernels of vLLM's own chain, reached one by one.
The sparse MLA, the MXFP8 linears, wo_a's groups and the compressor's KV/score projection here;
the routed experts through aiter's fused_moe."""

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason
from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.models.deepseekv41 import (
    deepseekv41_mxfp8_linear,
    deepseekv41_sparse_mla,
)
from aklr.plugins.vllm.hook import vllm_route

__all__: list[str] = []

# The InferenceX cookbook's nightly (4fab5a07) reports 0.33.0rc1.dev910
_VLLM = ("0.33.0rc1.dev910", "0.34.0")
# An fp8_ds_mla record: 576 data bytes, 8 scale bytes
_RECORD = 584
_ROCM = "vllm.models.deepseek_v41.amd.rocm"
_ATTENTION = f"{_ROCM}.DeepseekV41ROCMAiterMLAAttention"
_KV_SCORE = "vllm.models.deepseek_v4.compressor.DeepseekV4CompressorKVScoreLinear"
_MONO = "vllm.models.deepseek_v41.amd.mono_decode"
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


@vllm_route(f"{_ATTENTION}._o_proj", f"{_ROCM}.rocm_mxfp8_wo_a_bmm", versions=_VLLM)
def _mxfp8_wo_a_bmm(a, a_scale, wo_a, n_groups, o_lora_rank):
    """wo_a on the MXFP8 rows the attention stored, where vLLM's own chain runs the layer: an
    MXFP8 linear a group, on the group's columns of a and its rows of the weight, returned
    group-major as vLLM's kernel writes them. A group's columns are copied out of a's rows
    (the op takes row-major rows), and so are the outputs into one tensor. The route replaces
    the name the attention looks up: vLLM's custom op keeps its own reference to the kernel."""
    import torch

    weight, weight_scale, tokens = wo_a.weight, wo_a.weight_scale, a.shape[0]
    # a scale per weight row is vLLM's other loaded form, outside the linear's 32 x 32 blocks
    if (
        not tokens
        or weight_scale.shape[0] * 32 != weight.shape[0]
        or weight.shape[0] != n_groups * o_lora_rank
    ):
        return FallbackReason.OUT_OF_CONTRACT
    a, a_scale = a.view(tokens, n_groups, -1), a_scale.view(tokens, n_groups, -1)
    weight, weight_scale = (
        weight.view(n_groups, o_lora_rank, -1),
        weight_scale.view(n_groups, -1, weight_scale.shape[1]),
    )
    outs = []
    for group in range(n_groups):
        out = deepseekv41_mxfp8_linear(
            x=a[:, group].contiguous(),
            x_scale=a_scale[:, group].contiguous(),
            weight=weight[group],
            weight_scale=weight_scale[group],
        )
        if isinstance(out, FallbackReason):
            return out
        outs.append(out)
    return torch.cat(outs, dim=1)


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


@vllm_route(_KV_SCORE, f"{_KV_SCORE}.forward", versions=_VLLM)
def _kv_score_forward(self, input_):
    """The compressor's fused KV/score projection where vLLM runs a layer's input projections
    one after another: bf16 rows on a bf16 weight, fp32 out, as one gemm."""
    import torch

    return gemm(input_, self.weight, out_dtype=torch.float32)


# shortcut: vLLM's other two chains call torch.mm on the same weight inline, and torch.mm is no
# route target (a builtin without a signature, shared by the whole process); the routes below ask
# the op for the projection, which records the call, and leave the chain to vLLM. Compute it here
# once vLLM calls the projection through DeepseekV4CompressorKVScoreLinear.forward there too.
@vllm_route(_ATTENTION, f"{_ATTENTION}._forward_csa2_full", versions=_VLLM)
def _forward_csa2_full(self, hidden_states, positions):
    """The projection inside the chain vLLM forks over its streams, on the layers with a compressor."""
    import torch

    weight = self.compressor.fused_wkv_wgate.weight
    return (
        gemm.fallback_reason(hidden_states, weight, out_dtype=torch.float32) or FallbackReason.OUT_OF_CONTRACT
    )


@vllm_route(_MONO, f"{_MONO}._select", versions=_VLLM)
def _select(layer, positions, normed, qr):
    """The projection inside a decode step's top-k selection, on the layers with a compressor."""
    import torch

    if layer.attn.compressor is None:
        return FallbackReason.OUT_OF_CONTRACT
    weight = layer.attn.compressor.fused_wkv_wgate.weight
    return gemm.fallback_reason(normed, weight, out_dtype=torch.float32) or FallbackReason.OUT_OF_CONTRACT
