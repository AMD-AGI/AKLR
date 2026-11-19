# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models import moe_blocks
from aklr.plugins.sglang.models.deepseekv41.contract import (
    DEEPSEEKV41_MONO_K1,
    DEEPSEEKV41_MONO_K2_ATTENTION,
    DEEPSEEKV41_MONO_K2_FFN,
    DEEPSEEKV41_MXFP8_LINEAR,
    DEEPSEEKV41_SPARSE_MLA,
)

__all__ = [
    "deepseekv41_mono_k1",
    "deepseekv41_mono_k2_attention",
    "deepseekv41_mono_k2_ffn",
    "deepseekv41_mxfp8_linear",
    "deepseekv41_sparse_mla",
    "moe_forward_normal",
]

# The release, and the InferenceX cookbook's image for DeepSeek-V4.1: the targets are the same text in both
_SGLANG = ("0.5.23", "0.5.23.dev20261111")
# A slot of DeepSeek-V4's packed fp8 pool; the V4.1 compact pools take sglang's other kernel
_SLOT_BYTES = 584


@api(DEEPSEEKV41_MXFP8_LINEAR)
def deepseekv41_mxfp8_linear(x, weight, weight_scale, x_scale=None, output=None): ...


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


@api(DEEPSEEKV41_MONO_K1)
def deepseekv41_mono_k1(
    residual,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    wqkv,
    wqkv_scale,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    rope_freqs,
    positions,
    x=None,
    post_mix=None,
    comb_mix=None,
    prev_pre_mix=None,
    hc_rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    sinkhorn_iters=20,
    q=None,
    residual_out=None,
    pre_mix_out=None,
    post_mix_out=None,
    comb_mix_out=None,
    normed=None,
    q_latent=None,
    k_nope=None,
    k_nope_scale=None,
    k_rope=None,
): ...


@api(DEEPSEEKV41_MONO_K2_ATTENTION)
def deepseekv41_mono_k2_attention(
    q,
    kv_cache,
    indices,
    sinks,
    sm_scale,
    wo_a,
    wo_b,
    wo_b_scale,
    residual,
    post_mix,
    comb_mix,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    extra_kv_cache=None,
    extra_indices=None,
    rope_freqs=None,
    positions=None,
    wo_a_scale=None,
    peer_x=None,
    kv_cache_layout="row_major",
    hc_rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    sinkhorn_iters=20,
    ffn_input=None,
    residual_out=None,
    pre_mix_out=None,
    post_mix_out=None,
    comb_mix_out=None,
): ...


@sglang_route(
    "sglang.srt.models.deepseek_v4_platform_stage.forward_hc_pre_from_prev_fused_boundary",
    versions=("0.5.23.dev20261111",),
)
def _forward_hc_pre_from_prev_fused_boundary(
    layer,
    positions,
    hidden_states,
    input_ids,
    forward_batch,
    input_ids_global,
    prev_pre,
    pending_post,
    defer_post,
):
    """A decode or verify step's layer where no monokernel took the step, on a rank holding
    every head: its attention front and back as the two launches a monokernel runs them in,
    each asked for with the layer's own tensors (the back's inputs the front writes stand in
    by shape)."""
    import torch
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

    attn, mode = layer.self_attn, forward_batch.forward_mode
    linears = (attn.wqkv_a, attn.wq_b, attn.wo_b) if attn.fuse_wqa_wkv else ()
    if (
        not (mode.is_decode_or_idle() or mode.is_target_verify())
        or not linears
        or attn.q_head_norm
        or attn.attn_tp_size != 1
        or any(getattr(linear, "mxfp8_aiter_preshuffled", False) for linear in linears)
        or any(getattr(linear, "weight_scale_mx_e8m0", None) is None for linear in linears)
        or len({layer.input_layernorm.variance_epsilon, attn.q_norm.variance_epsilon, attn.eps}) != 1
    ):
        return FallbackReason.OUT_OF_CONTRACT
    x, residual, post, comb = pending_post or (None, hidden_states, None, None)
    tokens, pool = residual.shape[0], get_token_to_kv_pool()
    if not tokens or residual.dim() != 3:
        return FallbackReason.OUT_OF_CONTRACT
    seam = {
        "residual": residual,
        "hc_rms_eps": layer.rms_norm_eps,
        "hc_eps": layer.hc_eps,
        "norm_eps": attn.eps,
        "sinkhorn_iters": layer.hc_sinkhorn_iters,
    }
    rope = {
        "rope_freqs": torch.view_as_real(attn.freqs_cis).flatten(-2),
        "positions": positions[:tokens],
    }
    front = deepseekv41_mono_k1.fallback_reason(
        **seam,
        **rope,
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm_weight=layer.input_layernorm.weight,
        wqkv=attn.wqkv_a.weight,
        wqkv_scale=attn.wqkv_a.weight_scale_mx_e8m0,
        q_norm_weight=attn.q_norm.weight,
        kv_norm_weight=attn.kv_norm.weight,
        wq_b=attn.wq_b.weight,
        wq_b_scale=attn.wq_b.weight_scale_mx_e8m0,
        x=x,
        post_mix=None if post is None else post.view(tokens, -1),
        comb_mix=comb,
        prev_pre_mix=prev_pre,
    )

    if attn.wo_a.weight.dtype != torch.bfloat16:
        # shortcut: sglang keeps wo_a in fp8 on gfx950 unless a monokernel backend loads the
        # layer, and the back launch's contract has a bf16 wo_a: only the front is asked for;
        # ask for the back too once its contract has the MXFP8 wo_a
        return front or FallbackReason.OUT_OF_CONTRACT

    def pages(buffer, page_size):
        # the pool's rows are pages padded to 576 bytes; the op takes each page's slots
        if buffer.dim() == 4:
            return buffer
        return buffer.view(torch.uint8).as_strided(
            (buffer.shape[0], page_size, 1, _SLOT_BYTES), (buffer.stride(0), _SLOT_BYTES, _SLOT_BYTES, 1)
        )

    metadata, ratio = get_attn_backend().forward_metadata.core_attn_metadata, attn.compress_ratio
    extra = {}
    if ratio:
        extra = {
            "extra_kv_cache": pages(
                pool.get_extra_key_buffer(attn.layer_id), pool.get_extra_key_page_size(attn.layer_id)
            ),
            "extra_indices": metadata.sparse_page_indices(ratio)[:tokens].reshape(tokens, 1, -1),
        }
    mixes = residual.new_empty((tokens, 4, 4), dtype=torch.float32)
    back = deepseekv41_mono_k2_attention.fallback_reason(
        **seam,
        **rope,
        **extra,
        q=residual.new_empty((tokens, 1, attn.n_local_heads, 512)),
        kv_cache=pages(pool.get_swa_raw_buffer(attn.layer_id), pool.get_swa_key_page_size()),
        indices=metadata.swa_page_indices[:tokens].reshape(tokens, 1, -1),
        sinks=attn.attn_sink.float(),
        sm_scale=512**-0.5,
        kv_cache_layout="padded_pages",
        wo_a=attn.wo_a.weight,
        wo_b=attn.wo_b.weight,
        wo_b_scale=attn.wo_b.weight_scale_mx_e8m0,
        post_mix=mixes[:, 0],
        comb_mix=mixes,
        pre_mix=mixes[:, 0],
        hc_fn=layer.hc_ffn_fn,
        hc_scale=layer.hc_ffn_scale,
        hc_base=layer.hc_ffn_base,
        norm_weight=layer.post_attention_layernorm.weight,
    )
    # shortcut: the launches are asked for, which records the layer's two calls, and the layer
    # is left to sglang; run them here (the KV store and an index source's indexer between
    # them) once a kernel serves both
    return front or back or FallbackReason.OUT_OF_CONTRACT


@api(DEEPSEEKV41_MONO_K2_FFN)
def deepseekv41_mono_k2_ffn(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_w1=None,
    shared_w1_scale=None,
    shared_w2=None,
    shared_w2_scale=None,
    topk=6,
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    output=None,
): ...


def moe_forward_normal(moe, hidden_states, skip_shared_experts, use_vision_topk, leave_output):
    """DeepseekV2MoE.forward_normal for a DeepSeek-V4.1 layer, which deepseek_v2's route on it
    hands here: the router, top-k and experts, the fused shared one last, as one op call; then
    what forward_normal does with the experts' output: handed on unsummed with leave_output
    (a later step sums the ranks), else its all-reduce."""
    if skip_shared_experts or use_vision_topk or _moe_falls_back(moe):
        return FallbackReason.OUT_OF_CONTRACT
    experts, routing = moe.experts, moe.topk.topk_config
    out = deepseekv41_mono_k2_ffn(
        hidden_states=hidden_states,
        router_weight=moe.gate.weight,
        correction_bias=routing.correction_bias,
        w1=experts.w13_weight,
        w1_scale=experts.w13_weight_scale_inv,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale_inv,
        topk=routing.top_k - 1,
        routed_scaling_factor=moe.routed_scaling_factor,
        swiglu_limit=experts.moe_runner_config.swiglu_limit,
    )
    if isinstance(out, FallbackReason):
        return out
    if leave_output:
        from sglang.srt.models.deepseek_v2 import MoEOutput

        return MoEOutput(
            routed=out,
            shared=None,
            experts=experts,
            routed_scaling_factor=moe.routed_scaling_factor,
            shared_is_replicated=False,
        )
    return moe._finalize_normal_output(out, None)


def _moe_falls_back(moe) -> bool:
    """Whether ``moe`` runs other than as deepseekv41_mono_k2_ffn takes it: DeepSeek-V4.1's
    ungrouped sqrtsoftplus routing with a bias, renormalized and scaled in the top-k, one shared
    expert fused as the last slot, and FP4 experts with interleaved gate and up rows on aiter's
    runner at MXFP8 activations, clamped, unpadded, at sglang's standard dispatch."""
    experts, routing = moe.experts, moe.topk.topk_config
    method = experts.quant_method
    runner = getattr(method, "runner", None)
    return (
        moe_blocks.runtime_falls_back()
        or moe.is_hash
        or moe.num_fused_shared_experts != 1
        or moe._fuse_shared_experts_inside_sbo
        or not moe.routed_scaling_factor_fused_in_topk
        or routing.use_grouped_topk
        or routing.custom_routing_function is not None
        or routing.scoring_func != "sqrtsoftplus"
        or routing.correction_bias is None
        or not routing.renormalize
        or routing.num_shared_topk_slots != 1
        or moe.topk.waterfill_balancer is not None
        or type(experts).__name__ != "FusedMoE"
        or type(experts.dispatcher).__name__ != "StandardDispatcher"
        or experts.dispatcher.expert_mask_gpu is not None
        or type(method).__name__ != "Fp8MoEMethod"
        or not method.is_fp4_expert
        or not getattr(experts, "fp8_activations", False)
        or not getattr(experts.w13_weight, "is_shuffled", False)
        or runner is None
        or not runner.runner_backend.is_aiter()
        or runner.config.activation != "silu"
        or not runner.config.gate_up_interleaved
        or not runner.config.swiglu_limit
        or runner.config.no_combine
        or getattr(experts, "hidden_pad", 0)
        or getattr(experts, "intermediate_pad", 0)
    )
