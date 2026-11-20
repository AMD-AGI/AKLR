# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.ops.gemm.api import gemm
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseekv41.contract import (
    DEEPSEEKV41_MXFP8_LINEAR,
    DEEPSEEKV41_SPARSE_MLA,
    MXFP8_BLOCK,
)

__all__ = ["deepseekv41_mxfp8_linear", "deepseekv41_sparse_mla"]

# The release, and the InferenceX cookbook's image for DeepSeek-V4.1: the targets are the same text in both
_SGLANG = ("0.5.23", "0.5.23.dev20261111")
# A slot of DeepSeek-V4's packed fp8 pool; the V4.1 compact pools take sglang's other kernel
_SLOT_BYTES = 584


@api(DEEPSEEKV41_MXFP8_LINEAR)
def deepseekv41_mxfp8_linear(
    x, weight, weight_scale, x_scale=None, weight_layout="row_major", output=None
): ...


@api(DEEPSEEKV41_SPARSE_MLA)
def deepseekv41_sparse_mla(
    q,
    kv_cache,
    indices,
    sinks,
    sm_scale,
    extra_kv_cache=None,
    extra_indices=None,
    rope_freqs=None,
    positions=None,
    kv_cache_layout="row_major",
    output=None,
): ...


@sglang_route("sglang.srt.layers.attention.hip_flash_mla.aiter_sparse_decode_fwd", versions=_SGLANG)
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
    """The layer's attention as one op call, a token a row, returned as sglang returns it. The
    callers fold topk_length into the indices as -1 for this kernel, which reads no length."""
    if k_cache.shape[-1] != _SLOT_BYTES:
        return FallbackReason.OUT_OF_CONTRACT
    batch, queries, heads, width = q.shape
    tokens = batch * queries
    arguments = {
        "q": q.reshape(tokens, 1, heads, width),
        "kv_cache": k_cache,
        "indices": indices.reshape(tokens, 1, -1),
        "sinks": attn_sink,
        "sm_scale": softmax_scale,
        "kv_cache_layout": "padded_pages",
    }
    if extra_k_cache is not None:
        arguments["extra_kv_cache"] = extra_k_cache
        arguments["extra_indices"] = extra_indices_in_kvcache.reshape(tokens, 1, -1)
    if inv_rope is not None:
        arguments["rope_freqs"], positions = inv_rope
        positions = positions.view(-1)[:tokens]
        # a slice of the batch's int64 positions starts 8 bytes past the 16 AKLR's dispatch requires at an odd row
        arguments["positions"] = positions if positions.data_ptr() % 16 == 0 else positions.clone()
    output = deepseekv41_sparse_mla(**arguments)
    if isinstance(output, FallbackReason):
        return output
    return output.view(batch, queries, heads, width), None


@sglang_route(
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix.DeepseekV4HipRadixBackend._forward_prefill_opus",
    versions=_SGLANG,
)
def _forward_prefill_opus(
    self, q, layer_id, compress_ratio, forward_batch, core_attn_metadata, attn_sink, inv_rope
):
    """A chunk of 1024 tokens or more, which sglang attends with OPUS over bf16 copies of the rows:
    the op call its shorter chunks make, on the same SWA window and top-k slots of the fp8 pools."""
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import _fold_lengths_for_aiter_sparse

    pool, tokens = self.token_to_kv_pool, q.shape[0]
    indices = core_attn_metadata.swa_page_indices[:tokens]
    slot_bytes = {pool.swa_kv_pool.kv_cache_total_dim}
    if compress_ratio:
        slot_bytes.add(pool.get_extra_key_bytes_per_token(layer_id))
    # the V4.1 compact pools are not the op's; OPUS itself takes no chunk longer than its metadata
    if slot_bytes != {_SLOT_BYTES} or indices.shape[0] != tokens:
        return FallbackReason.OUT_OF_CONTRACT
    # each pool's pages, sliced out of their padded rows as sglang's shorter chunks slice them
    page_size = pool.swa_page_size
    cache = pool.get_swa_key_buffer_radix(layer_id)[:, : page_size * _SLOT_BYTES]
    cache = cache.view(-1, page_size, 1, _SLOT_BYTES)
    extra_cache = extra_indices = extra_lengths = None
    if compress_ratio:
        page_size = pool.get_extra_key_page_size(layer_id)
        extra_cache = pool.get_extra_key_buffer(layer_id)[:, : page_size * _SLOT_BYTES]
        extra_cache = extra_cache.view(-1, page_size, 1, _SLOT_BYTES)
        extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)[:tokens].unsqueeze(1)
        extra_lengths = core_attn_metadata.sparse_topk_lengths(compress_ratio)[:tokens]
    # OPUS reads each token's lengths; the op reads none, so sglang's own fold marks the slots past them
    indices, extra_indices = _fold_lengths_for_aiter_sparse(
        core_attn_metadata,
        compress_ratio,
        indices.unsqueeze(1),
        core_attn_metadata.swa_topk_lengths[:tokens],
        extra_indices,
        extra_lengths,
    )
    served = _aiter_sparse_decode_fwd(
        q.unsqueeze(1), cache, indices, attn_sink, self.softmax_scale, extra_cache, extra_indices, inv_rope
    )
    return served if isinstance(served, FallbackReason) else served[0].squeeze(1)


@sglang_route("sglang.kernels.ops.gemm.mxfp8_aiter_gfx95.aiter_mxfp8_blockscaled_linear", versions=_SGLANG)
def _aiter_mxfp8_blockscaled_linear(
    input, weight, weight_scale_ue8m0, input_scale=None, bias=None, weight_preshuffled=False
):
    """The linear as one op call on the rows of input, then sglang's bias, returned at input's
    leading shape. A bf16 input is the op's; any other unquantized one sglang casts first."""
    import torch

    rows = input.reshape(-1, input.shape[-1])
    if weight_preshuffled or not rows.shape[0] or (input_scale is None and input.dtype != torch.bfloat16):
        return FallbackReason.OUT_OF_CONTRACT
    arguments = {"x": rows, "weight": weight, "weight_scale": weight_scale_ue8m0}
    if input_scale is not None:
        arguments["x_scale"] = input_scale.reshape(rows.shape[0], -1)
    out = deepseekv41_mxfp8_linear(**arguments)
    if isinstance(out, FallbackReason):
        return out
    if bias is not None:
        out = out + bias
    return out.view(*input.shape[:-1], weight.shape[0])


@sglang_route("sglang.srt.layers.quantization.fp8_hip._apply_flydsl_mxfp8", versions=_SGLANG)
def _apply_flydsl_mxfp8(layer, x, bias, input_scale):
    """A linear of 2048 rows or more, which sglang's aiter route runs on its FlyDSL kernel: the same
    linear, on the plain weight and the compact block scales that kernel's own were laid out from."""
    return _aiter_mxfp8_blockscaled_linear(x, layer.weight, layer.weight_scale_mx_e8m0, input_scale, bias)


# the nightly's alone: 0.5.23 has no 32-wide wo_a, and runs the 128-wide one without this function
@sglang_route(
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_wo_a_fp8._wo_a_bmm", versions=("0.5.23.dev20261111",)
)
def _wo_a_bmm(o_fp8, weight, o_scale, weight_scale):
    """The wo_a projection's BMM as one op call a group, on the group's weight as sglang holds it
    for aiter, returned as aiter returns it: a token's groups side by side."""
    import torch

    # V4's 128-wide blocks keep a row-major weight under scales the op does not take
    if (
        not (o_fp8.ndim == o_scale.ndim == weight.ndim == weight_scale.ndim == 3)
        or not o_fp8.shape[0]
        or weight.shape[-1] != MXFP8_BLOCK * weight_scale.shape[-1]
    ):
        return FallbackReason.OUT_OF_CONTRACT
    outputs = []
    for group in range(weight.shape[0]):
        # a group's rows lie a token apart: copied, as the op takes rows side by side
        out = deepseekv41_mxfp8_linear(
            x=o_fp8[:, group].contiguous(),
            x_scale=o_scale[:, group].contiguous(),
            weight=weight[group],
            weight_scale=weight_scale[group],
            weight_layout="shuffled_16x16",
        )
        if isinstance(out, FallbackReason):
            return out
        outputs.append(out)
    return torch.stack(outputs, dim=1)


@sglang_route("sglang.srt.models.deepseek_v4._apply_wo_a_bf16_matmul", versions=_SGLANG)
def _apply_wo_a_bf16_matmul(
    o,
    wo_a,
    is_decode,
    is_target_verify=False,
    fuse_mxfp8_quant=False,
    is_prefill=False,
    fast_path=False,
    fp8_grid=False,
    emit_fp8=False,
):
    """The wo_a projection a monokernel's prefill keeps in bf16, as one GEMM a group, returned as
    sglang's einsum returns it: a token's groups side by side. Decode and target-verify rows are
    asked of gemm, which records them, and left to sglang's kernels, which return other forms."""
    import torch

    if not (o.ndim == wo_a.ndim == 3) or o.shape[1] != wo_a.shape[0]:
        return FallbackReason.OUT_OF_CONTRACT
    # a group's rows lie a token apart: asked with a row-major stand-in, copied only for a call the op takes
    stand_in = torch.empty((o.shape[0], o.shape[2]), dtype=o.dtype, device=o.device)
    outputs = []
    for group, weight in enumerate(wo_a):
        if (reason := gemm.fallback_reason(stand_in, weight)) is not None:
            return reason
        if is_decode or is_target_verify:
            continue
        out = gemm(o[:, group].contiguous(), weight)
        if isinstance(out, FallbackReason):
            return out
        outputs.append(out)
    return torch.stack(outputs, dim=1) if outputs else FallbackReason.OUT_OF_CONTRACT


@sglang_route("sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k", versions=_SGLANG)
def _rocm_router_gemv_split_k(x, w):
    """The GEMV sglang runs at 64 rows or fewer for the MoE router's logits and the indexer's head
    weights, asked of gemm as the product it is: x @ w.T in fp32."""
    import torch

    # shortcut: the op is asked for, which records the call, and the GEMV is left to sglang, whose
    # callers sum the split-K partials it returns in their own launch; return a kernel's product as
    # the one partial once a kernel serves it with that sum's batch invariance
    return gemm.fallback_reason(x, w, out_dtype=torch.float32) or FallbackReason.OUT_OF_CONTRACT


@sglang_route("sglang.kernels.ops.gemm.bf16_fp32.linear_bf16_fp32", versions=_SGLANG)
def _linear_bf16_fp32(x, y, *, hpc_kernel_min_m=None):
    """The bf16 linear sglang returns in fp32, the compressor's KV and score projections among its
    callers, as the GEMM it is. hpc_kernel_min_m picks a kernel no ROCm build has."""
    import torch

    return gemm(x, y, out_dtype=torch.float32)


@sglang_route("sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Indexer.forward_wk", versions=_SGLANG)
def _forward_wk(self, latent):
    """The indexer's key projection, asked of gemm as the product it is: latent @ wk.T."""
    # shortcut: the op is asked for, which records the call, and the product is left to sglang, whose
    # small-row kernel keeps a reduction order of its own; run gemm once a kernel serves it in that order
    return gemm.fallback_reason(latent, self.wk.weight) or FallbackReason.OUT_OF_CONTRACT


@sglang_route("sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Compressor.project", versions=_SGLANG)
def _project(self, x):
    """An uncompressed layer's KV projection, asked of gemm as the product it is: x @ wkv.T. A
    compressed layer's projections reach gemm through sglang's bf16 linear with fp32 output."""
    # shortcut: recorded and left to sglang, which returns the pair project's callers split; serve it
    # from here once a kernel takes the shape
    if self.compress_ratio == 1:
        gemm.fallback_reason(x, self.wkv.weight)
    return FallbackReason.OUT_OF_CONTRACT


@sglang_route(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_compress_fused",
    versions=_SGLANG,
)
def _low_ratio_compress_fused(self, layer, x, req, pos, *, draft_len=1, extend_offsets=None, projected=None):
    """A decode or target-verify step's KV projection on an uncompressed layer, asked of gemm as the
    product it is: x @ wkv.T. sglang calls wkv inline there, so the route sits on the step."""
    # shortcut: recorded and left to sglang, whose step norms, rotates and stores the projection in
    # one kernel; pass a kernel's product as `projected` once one serves the shape
    weight = getattr(getattr(getattr(layer, "compressor", None), "wkv", None), "weight", None)
    if getattr(layer, "compress_ratio", None) == 1 and projected is None and weight is not None:
        gemm.fallback_reason(x, weight)
    return FallbackReason.OUT_OF_CONTRACT
