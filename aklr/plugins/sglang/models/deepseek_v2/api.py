# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v2.contract import (
    DEEPSEEK_V2_ATTENTION,
    DEEPSEEK_V2_ATTENTION_INDEXER,
    DEEPSEEK_V2_DENSE_LAYER,
    DEEPSEEK_V2_EH_PROJ,
    DEEPSEEK_V2_FP8_ATTENTION_INDEXER,
    DEEPSEEK_V2_FP8_DENSE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER,
    DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART,
    DEEPSEEK_V2_MOE,
    DEEPSEEK_V2_MOE_LAYER,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER,
    DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART,
)

__all__ = [
    "deepseek_v2_attention",
    "deepseek_v2_attention_indexer",
    "deepseek_v2_dense_layer",
    "deepseek_v2_eh_proj",
    "deepseek_v2_fp8_attention_indexer",
    "deepseek_v2_fp8_dense_layer",
    "deepseek_v2_fp8_moe_layer",
    "deepseek_v2_fp8_moe_layer_shared_apart",
    "deepseek_v2_moe",
    "deepseek_v2_moe_layer",
    "deepseek_v2_unquantized_moe_layer",
    "deepseek_v2_unquantized_moe_layer_shared_apart",
]


@api(DEEPSEEK_V2_MOE)
def deepseek_v2_moe(
    hidden_states,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@api(DEEPSEEK_V2_MOE_LAYER)
def deepseek_v2_moe_layer(
    hidden_states,
    norm_weight,
    norm_eps,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@api(DEEPSEEK_V2_FP8_MOE_LAYER)
def deepseek_v2_fp8_moe_layer(
    hidden_states,
    norm_weight,
    norm_eps,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@api(DEEPSEEK_V2_FP8_MOE_LAYER_SHARED_APART)
def deepseek_v2_fp8_moe_layer_shared_apart(
    hidden_states,
    norm_weight,
    norm_eps,
    router_weight,
    correction_bias,
    w1,
    w1_scale,
    w2,
    w2_scale,
    shared_gate_up_weight,
    shared_gate_up_scale,
    shared_down_weight,
    shared_down_scale,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func,
    routed_scaling_factor,
): ...


@api(DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER)
def deepseek_v2_unquantized_moe_layer(
    hidden_states,
    norm_weight,
    norm_eps,
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
): ...


@api(DEEPSEEK_V2_UNQUANTIZED_MOE_LAYER_SHARED_APART)
def deepseek_v2_unquantized_moe_layer_shared_apart(
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
): ...


@api(DEEPSEEK_V2_DENSE_LAYER)
def deepseek_v2_dense_layer(hidden_states, norm_weight, norm_eps, gate_up_weight, down_weight): ...


@api(DEEPSEEK_V2_FP8_DENSE_LAYER)
def deepseek_v2_fp8_dense_layer(
    hidden_states,
    norm_weight,
    norm_eps,
    gate_up_weight,
    gate_up_scale,
    down_weight,
    down_scale,
): ...


@api(DEEPSEEK_V2_ATTENTION)
def deepseek_v2_attention(
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
): ...


@api(DEEPSEEK_V2_ATTENTION_INDEXER)
def deepseek_v2_attention_indexer(
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
): ...


@api(DEEPSEEK_V2_FP8_ATTENTION_INDEXER)
def deepseek_v2_fp8_attention_indexer(
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
    wq_b_scale,
    wk,
    wk_scale,
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
): ...


@api(DEEPSEEK_V2_EH_PROJ)
def deepseek_v2_eh_proj(embeds, hidden, enorm_weight, hnorm_weight, eps, weight): ...


@sglang_route(
    "sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.forward",
    versions=">=0.5.22,<0.5.23",
)
def _decoder_layer(
    self,
    positions,
    hidden_states,
    forward_batch,
    zero_allocator,
    gemm_output_zero_allocator=None,
    llama_4_scaling=None,
    prev_topk_indices=None,
    captured_last_layer_outputs=None,
    next_full_attention_layer_id=None,
):
    """Decode, target-verify and the MTP draft's extend of a layer as two op calls on its
    residual stream, sglang's norms and all-reduces folded in: the attention from
    input_layernorm on (with the indexer's top-k in the layers that run one), o_proj's output
    summed over the ranks and added to the stream, then the MoE (in a dense layer, the MLP) from
    post_attention_layernorm on, summed and added the same way. The layer takes the stream
    whole from forward_batch.residual_stream (_stream_value) and writes the stream after it
    back as the residual with nothing pending, the form sglang's next layer reads its input
    from and its final norm normalizes; until both ops have run, the residual stream stays as
    the previous layer left it, for sglang to run the layer on if one falls back. Both ops
    are asked first (fallback_reason) on a tensor shaped as the layer's input: forming the
    input completes the pending sum, wasted if sglang then runs the layer."""
    from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

    if _decoder_layer_falls_back(
        self,
        forward_batch,
        llama_4_scaling,
        prev_topk_indices,
        captured_last_layer_outputs,
    ):
        return FallbackReason.OUT_OF_CONTRACT
    attn, residual_stream = self.self_attn, forward_batch.residual_stream
    input_like = _layer_input_like(residual_stream, hidden_states)
    args = _attention_args(attn, input_like, self.input_layernorm, positions, forward_batch)
    runs_indexer = attn.should_run_indexer(prev_topk_indices)
    if runs_indexer:
        attention = _attention_indexer_call(attn, args, forward_batch)
        if isinstance(attention, FallbackReason):
            return attention
    else:
        attention = (deepseek_v2_attention, {**args, "topk_indices": prev_topk_indices})
    mlp = _mlp_call(self, input_like)
    for op, kwargs in (attention, mlp):
        if (reason := op.fallback_reason(**kwargs)) is not None:
            return reason
    (attention_op, attention_args), (mlp_op, mlp_args) = attention, mlp
    layer_input = _stream_value(residual_stream, hidden_states)
    attended = attention_op(**(attention_args | {"hidden_states": layer_input}))
    if isinstance(attended, FallbackReason):
        return attended
    topk_indices = prev_topk_indices
    if runs_indexer:
        attended, topk_indices = attended
    # captured as sglang captures it, once the op serves the layer
    topk_indices = maybe_capture_indexer_topk(attn.layer_id, topk_indices)
    out = mlp_op(**(mlp_args | {"hidden_states": attended}))
    if isinstance(out, FallbackReason):
        return out
    return residual_stream.write(out), topk_indices if attn.next_skip_topk else None


def _layer_input_like(residual_stream, hidden_states):
    """A tensor shaped as the layer's input, without forming it: ``hidden_states``, or the
    output pending on ``residual_stream`` (``hidden_states`` may be its handle)."""
    pending = residual_stream.pending
    return hidden_states if pending is None else pending.value


def _stream_value(residual_stream, hidden_states):
    """The whole stream ahead of a layer, from sglang's ResidualStream and the previous
    layer's ``hidden_states``: the layer stack's input or the residual a layer wrote as they
    are, else the residual plus the pending contribution with its sum completed, from copies
    (snapshot), which leaves the contribution pending in ``residual_stream``."""
    if residual_stream.pending is None:
        residual_stream.check(hidden_states)
        return hidden_states
    return residual_stream.snapshot(hidden_states)


def _decoder_layer_falls_back(self, forward_batch, llama_4_scaling, prev_topk_indices, captured) -> bool:
    """When the layer route falls back to sglang: a capture of the layer's outputs, the
    residual stream not the layer's to take whole (_residual_stream_falls_back), the
    attention off its ops (_attention_falls_back), the MoE off its layer ops
    (_moe_falls_back), a dense layer's MLP off its plain path (_mlp_falls_back), and ranks
    other than one TP group for both halves (DP attention, EP)."""
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    mlp = self.mlp
    return (
        captured is not None
        or _residual_stream_falls_back(self, forward_batch)
        or _attention_falls_back(self.self_attn, forward_batch, llama_4_scaling, prev_topk_indices)
        or (_moe_falls_back(mlp) if self.is_layer_sparse else _mlp_falls_back(mlp))
        or is_dp_attention_enabled()
    )


def _residual_stream_falls_back(self, forward_batch) -> bool:
    """When the layer does not take its residual stream (forward_batch.residual_stream)
    whole, as the rows of the batch on every rank of the TP group: no stream; pipeline
    stages (the next stage takes a residual apart from the hidden states, which a stage
    ending in a written stream does not send); two-batch overlap's microbatches; a batch
    variant other than the ordinary one; attention on its TP slices of the rows or on a TP
    group short of the whole (DP or CP attention); and a contribution pending on the stream
    other than a plain add of a sum (a deferred MoE finalize, which only its producer
    completes)."""
    from sglang.srt.layers.layer_boundary.contracts import BatchVariant
    from sglang.srt.layers.layer_boundary.output import DeferredFinalize
    from sglang.srt.runtime_context import get_parallel

    residual_stream, parallel = forward_batch.residual_stream, get_parallel()
    boundary = self.attn_boundary
    if (
        residual_stream is None
        or parallel.pp_size > 1
        or forward_batch.can_run_tbo
        or boundary.plan.variant_for(forward_batch) is not BatchVariant.ORDINARY
        or boundary.input_on_attn_tp_slices
        or parallel.attn_tp_size != parallel.tp_size
    ):
        return True
    pending = residual_stream.pending
    return pending is not None and (
        not pending.update.is_plain_add or isinstance(pending.owed, DeferredFinalize)
    )


def _attention_falls_back(attn, forward_batch, llama_4_scaling, prev_topk_indices) -> bool:
    """When the layer's DeepseekV2AttentionMLA leaves its attention ops: any path but sglang's
    sparse absorbed one in decode, target-verify and the MTP draft's extend, a top-k of
    positions (the backend's own transform), a KV row not 576 fp8 values of scale 1, RoPE not
    aiter's interleaved tables, a W_UK/W_UV scale that is a number other than 1, LoRA, DCP or
    Llama-4 scaling, and a layer that carries no top-k (sglang attends dense then)."""
    import torch
    from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        AttnForwardMethod,
    )
    from sglang.srt.runtime_context import get_parallel

    mode = forward_batch.forward_mode
    if (
        not (mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2())
        or not attn.use_dsa
        or attn.dispatch_attn_forward_method(forward_batch) != AttnForwardMethod.MLA_ROCM
    ):
        return True
    backend, pool, rope = get_attn_backend(), get_token_to_kv_pool(), attn.rotary_emb
    return (
        not getattr(backend, "use_fused_topk", False)
        or getattr(backend, "hisparse_coordinator", None) is not None
        or getattr(pool, "dsa_kv_cache_store_fp8", False)
        or getattr(pool, "kv_cache_dim", None) != attn.kv_lora_rank + attn.qk_rope_head_dim
        or attn.attn_mqa.k_scale_float != 1.0
        or not hasattr(rope, "cos_cache")
        or rope.is_neox_style
        or attn.q_a_layernorm.variance_epsilon != attn.kv_a_layernorm.variance_epsilon
        or (not torch.is_tensor(attn.w_scale) and attn.w_scale != 1.0)
        or llama_4_scaling is not None
        or get_parallel().dcp_enabled
        or is_kv_b_lora_active(attn)
        or (not attn.should_run_indexer(prev_topk_indices) and prev_topk_indices is None)
    )


def _attention_args(attn, stream, input_norm, positions, forward_batch) -> dict:
    """sglang's DeepseekV2AttentionMLA on the residual stream as the attention ops' arguments
    but the top-k and the indexer's; the ops store the tokens' KV rows into sglang's cache.
    """
    import torch
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    qkv_a, q_b, o_proj = attn.fused_qkv_a_proj_with_mqa, attn.q_b_proj, attn.o_proj
    kv_cache = get_token_to_kv_pool().get_key_buffer(attn.layer_id)
    rope = attn.rotary_emb
    return {
        "hidden_states": stream,
        "input_norm_weight": input_norm.weight,
        "input_norm_eps": input_norm.variance_epsilon,
        "qkv_a_weight": qkv_a.weight,
        "qkv_a_scale": getattr(qkv_a, "weight_scale_inv", None),
        "q_a_norm_weight": attn.q_a_layernorm.weight,
        "kv_a_norm_weight": attn.kv_a_layernorm.weight,
        "eps": attn.q_a_layernorm.variance_epsilon,
        "q_b_weight": q_b.weight,
        "q_b_scale": getattr(q_b, "weight_scale_inv", None),
        "positions": positions,
        "out_cache_loc": forward_batch.out_cache_loc,
        "kv_cache": kv_cache.view(-1, kv_cache.shape[-1]),
        "sm_scale": attn.attn_mqa.scaling,
        "cos_cache": rope.cos_cache.view(rope.cos_cache.shape[0], -1),
        "sin_cache": rope.sin_cache.view(rope.sin_cache.shape[0], -1),
        "w_uk_t": attn.w_kc.transpose(1, 2),
        "w_uv": attn.w_vc.transpose(1, 2),
        # sglang leaves bf16 ones at the float 1.0
        "w_scale": attn.w_scale if torch.is_tensor(attn.w_scale) else None,
        "o_proj_weight": o_proj.weight,
        "o_proj_scale": getattr(o_proj, "weight_scale_inv", None),
    }


def _attention_indexer_call(attn, args, forward_batch):
    """deepseek_v2_attention_indexer (deepseek_v2_fp8_attention_indexer for fp8 indexer weights)
    and its arguments on the attention's ``args``, where it serves sglang's indexer: the op
    returns the stream after the attention and the indexer's top-k, storing the tokens' index
    K into sglang's cache too; else the reason."""
    import torch
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    indexer, layer_id, stream = attn.indexer, attn.layer_id, args["hidden_states"]
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    if _attention_indexer_falls_back(indexer, forward_batch, metadata, stream.shape[0]):
        return FallbackReason.OUT_OF_CONTRACT
    index_k_cache = get_token_to_kv_pool().get_index_k_with_scale_buffer(layer_id=layer_id)
    rope = indexer.rotary_emb
    op, scales = deepseek_v2_attention_indexer, {}
    if indexer.wq_b.weight.dtype == torch.float8_e4m3fn:
        op = deepseek_v2_fp8_attention_indexer
        scales = {
            "wq_b_scale": indexer.wq_b.weight_scale_inv,
            "wk_scale": indexer.wk.weight_scale_inv,
        }
    return op, {
        **args,
        "wq_b": indexer.wq_b.weight,
        "wk": indexer.wk.weight,
        **scales,
        "weights_proj": indexer.weights_proj.weight,
        "k_norm_weight": indexer.k_norm.weight,
        "k_norm_bias": indexer.k_norm.bias,
        "k_norm_eps": indexer.k_norm.variance_epsilon,
        "index_cos_cache": rope.cos_cache.view(rope.cos_cache.shape[0], -1),
        "index_sin_cache": rope.sin_cache.view(rope.sin_cache.shape[0], -1),
        "index_k_cache": index_k_cache.view(index_k_cache.shape[0], -1).view(torch.float8_e4m3fn),
        "seq_lens": metadata.get_seqlens_expanded(),
        "page_table": metadata.get_page_table_64(),
        "topk_indices": torch.empty(
            (stream.shape[0], indexer.index_topk),
            dtype=torch.int32,
            device=stream.device,
        ),
    }


def _attention_indexer_falls_back(indexer, forward_batch, metadata, n_tokens) -> bool:
    """When the attention indexer ops do not serve sglang's indexer: an indexer other than the
    plain paged one on HIP (IndexerKPool's pooled keys, the fused wk and weights_proj, a
    piecewise or breakable graph, context parallel, LoRA on weights_proj, scales that are not
    powers of 2, an RMSNorm k_norm, RoPE not aiter's interleaved tables, no sequences, or the
    k-only path of short contexts), no DSA metadata, index K stored or read other than in
    aiter's preshuffled pages of 64 keys at ue8m0 scales (aiter's fused fp8 writer stores them
    unrotated, from an fp32 k_norm), a pool that splits or broadcasts them by layer, scores
    other than aiter's or with initial or local tokens masked, and a top-k other than the
    fused paged v2 kernel's of at most 2048 per row or broadcast from rank 0.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_indexer import (
        Indexer,
        _is_hip,
        _is_in_piecewise_or_breakable_cuda_graph,
        _use_aiter,
        _use_aiter_preshuffle,
    )
    from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import (
        DSAIndexerMetadata,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.layernorm import LayerNorm
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    if not _is_hip or not isinstance(indexer, Indexer) or not isinstance(metadata, DSAIndexerMetadata):
        return True
    pool, attn_metadata = get_token_to_kv_pool(), metadata.attn_metadata
    return (
        indexer.use_dsa_indexer_fusion
        or _is_in_piecewise_or_breakable_cuda_graph()
        or indexer.dsa_enable_prefill_cp
        or getattr(forward_batch, "attn_cp_metadata", None) is not None
        or getattr(indexer.weights_proj, "set_lora", False)
        or indexer.block_size != indexer.head_dim
        or type(indexer.k_norm) is not LayerNorm
        or not hasattr(indexer.rotary_emb, "cos_cache")
        or indexer.rotary_emb.is_neox_style
        or forward_batch.seq_lens.numel() == 0
        or indexer._should_skip_logits_computation(forward_batch)
        or not (_use_aiter and _use_aiter_preshuffle)
        or indexer.use_aiter_fused_fp8_writer
        or indexer.scale_fmt != "ue8m0"
        or pool.page_size != 64
        or getattr(pool, "index_page_size", None) != 64
        or attn_metadata.page_size != 64
        or any(
            hasattr(pool, n)
            for n in (
                "invalidate_index_buffer_for_layer",
                "_is_layer_owned",
                "get_broadcastable_index_k_with_scale_buffer",
            )
        )
        or not indexer.paged_mqa_logits_backend.is_aiter()
        or indexer.num_init_tokens != 0
        or indexer.num_local_tokens != 0
        or not envs.SGLANG_DSA_FUSE_TOPK.get()
        or metadata.force_unfused_topk
        or not metadata.topk_backend.should_use_topk_v2()
        or metadata.topk_transform_method != TopkTransformMethod.PAGED
        or not 0 < indexer.index_topk <= 2048
        or attn_metadata.real_page_table.shape[0] != n_tokens
        or envs.SGLANG_DSA_TOPK_BROADCAST.get()
    )


def _moe_falls_back(mlp) -> bool:
    """When a sparse layer's DeepseekV2MoE leaves its layer ops: EP, routing off its plain path
    (_routing_falls_back), experts neither MXFP4, fp8 nor unquantized or off aiter's fused_moe
    (_runner_falls_back); MXFP4 experts without the shared expert in their last slot; fp8
    ones off their 128 x 128 blocks (_fp8_experts_fall_back), or with the shared expert
    neither in their last slot nor apart as fp8 linears (_shared_apart_falls_back);
    unquantized experts on another backend or not shuffled for aiter, or with the shared
    expert neither in their last slot nor apart as bf16 linears."""
    experts = mlp.experts
    if mlp.moe_ep_size != 1 or _routing_falls_back(mlp):
        return True
    if _experts_format(experts) == "Fp8MoEMethod":
        shared_fused = mlp.num_fused_shared_experts == 1
        return (not shared_fused and _shared_apart_falls_back(mlp, fp8=True)) or _fp8_experts_fall_back(
            experts
        )
    if not _unquantized(experts):
        return (
            _experts_format(experts) != "QuarkW4A4MXFp4MoE"
            or mlp.num_fused_shared_experts != 1
            or _runner_falls_back(getattr(experts.scheme, "runner", None), experts)
        )
    method = experts.quant_method
    backend = method.runner.runner_backend
    shared_fused = mlp.num_fused_shared_experts == 1
    return (
        backend.is_triton_kernels()
        or backend.is_deep_gemm()
        or method.use_flashinfer_cutlass
        or method.use_flashinfer_trtllm_moe
        or experts._skip_aiter_moe_shuffle
        or _runner_falls_back(method._aiter_runner, experts)
        or (not shared_fused and _shared_apart_falls_back(mlp, fp8=False))
    )


def _shared_apart_falls_back(mlp, fp8: bool) -> bool:
    """When DeepseekV2MoE's shared expert is not a plain DeepseekV2MLP apart from the experts,
    summed with them before the all-reduce: fused into them, absent, whole on every rank, with
    a SwiGLU limit or NVFP4's fused gate_up, or with linears other than bf16 ones (``fp8``:
    fp8 ones with block scales, _fp8_linear)."""
    import torch

    shared = getattr(mlp, "shared_experts", None)
    if mlp.num_fused_shared_experts != 0 or shared is None:
        return True
    gate_up, down = shared.gate_up_proj, shared.down_proj
    if fp8:
        has_linears = _fp8_linear(gate_up) and _fp8_linear(down)
    else:
        has_linears = gate_up.weight.dtype == down.weight.dtype == torch.bfloat16
    return (
        not has_linears
        or mlp._shared_expert_tp1
        or shared.swiglu_limit is not None
        or getattr(shared, "_enable_nvfp4_gemm_swiglu_fusion", False)
    )


def _routing_falls_back(mlp) -> bool:
    """When DeepseekV2MoE's routing leaves its plain path: hash routing, a correction bias of
    its own for image tokens (the vision top-k), deterministic inference, and anything that
    records, remaps or overrides the routing."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    exec_ = get_exec()
    return (
        mlp.is_hash
        or mlp.gate.e_score_correction_bias_vl is not None
        or exec_.deterministic.enable_deterministic_inference
        or exec_.moe.enable_eplb
        or exec_.moe.expert_distribution_recorder_mode is not None
        or exec_.features.enable_return_routed_experts
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def _runner_falls_back(runner, experts) -> bool:
    """When FusedMoE ``experts`` run other than on aiter's fused_moe as the MoE layer ops do:
    no aiter ``runner``, an activation other than gated silu, the routing weights on the
    inputs, no combine, gemm1's alpha or clamp, expert biases or an expert mask."""
    if runner is None:
        return True
    config = runner.config
    return (
        config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or config.gemm1_alpha is not None
        or config.gemm1_clamp_limit is not None
        or getattr(experts, "w13_weight_bias", None) is not None
        or getattr(experts, "w2_weight_bias", None) is not None
        or experts.dispatcher.expert_mask_gpu is not None
    )


def _fp8_experts_fall_back(experts) -> bool:
    """When Fp8MoEMethod ``experts`` run other than as the fp8 MoE layer op does: weights not at
    a scale per 128 x 128 block (MXFP8, fp4 experts, per-channel scales), or off aiter's
    fused_moe (_runner_falls_back)."""
    method = experts.quant_method
    runner = getattr(method, "runner", None)
    return (
        not method.block_quant
        or method.use_mxfp8
        or method.is_fp4_expert
        or list(method.weight_block_size) != [128, 128]
        or _runner_falls_back(runner, experts)
        or not runner.runner_backend.is_aiter()
    )


def _experts_format(experts) -> str:
    """The class that names FusedMoE ``experts``' format, by sglang's rule: their scheme's, else
    (fp8, unquantized, a LoRA wrapper) their quant method's."""
    return type(getattr(experts, "scheme", None) or experts.quant_method).__name__


def _unquantized(experts) -> bool:
    """Whether FusedMoE ``experts`` hold bf16 weights, sglang's UnquantizedFusedMoEMethod."""
    return _experts_format(experts) == "UnquantizedFusedMoEMethod"


def _routing_args(mlp) -> dict:
    """sglang's DeepseekV2MoE's router and routing as the MoE layer ops' arguments."""
    routing = mlp.topk.topk_config
    grouped = routing.use_grouped_topk
    return {
        "router_weight": mlp.gate.weight,
        "correction_bias": routing.correction_bias,
        "topk": routing.top_k - routing.num_fused_shared_experts,
        "num_expert_group": routing.num_expert_group if grouped else 1,
        "topk_group": routing.topk_group if grouped else 1,
        "renormalize": routing.renormalize,
        "scoring_func": routing.scoring_func,
        "routed_scaling_factor": mlp.routed_scaling_factor,
    }


def _moe_layer_args(mlp) -> dict:
    """sglang's DeepseekV2MoE with MXFP4 experts as deepseek_v2_moe_layer's arguments past the
    norm (deepseek_v2_moe's past hidden_states): the router, the experts and the
    routing."""
    experts = mlp.experts
    return {
        **_routing_args(mlp),
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale,
    }


def _mlp_call(layer, hidden_states):
    """The layer's MoE (in a dense layer, its MLP) from post_attention_layernorm on
    ``hidden_states``, as its op and arguments."""
    post_norm, mlp = layer.post_attention_layernorm, layer.mlp
    norm = {
        "hidden_states": hidden_states,
        "norm_weight": post_norm.weight,
        "norm_eps": post_norm.variance_epsilon,
    }
    if not layer.is_layer_sparse:
        return _dense_layer_call(mlp, norm)
    if _unquantized(mlp.experts):
        return _unquantized_moe_layer_call(mlp, norm)
    if _experts_format(mlp.experts) == "Fp8MoEMethod":
        return _fp8_moe_layer_call(mlp, norm)
    return deepseek_v2_moe_layer, {**norm, **_moe_layer_args(mlp)}


def _fp8_moe_layer_call(mlp, norm: dict):
    """A sparse layer's DeepseekV2MoE with fp8 experts from ``norm``
    (post_attention_layernorm's arguments) on, as its op and arguments: the shared expert in
    their last slot, or apart from them."""
    args = {**norm, **_fp8_moe_layer_args(mlp)}
    if mlp.num_fused_shared_experts:
        return deepseek_v2_fp8_moe_layer, args
    gate_up, down = mlp.shared_experts.gate_up_proj, mlp.shared_experts.down_proj
    return deepseek_v2_fp8_moe_layer_shared_apart, {
        **args,
        "shared_gate_up_weight": gate_up.weight,
        "shared_gate_up_scale": gate_up.weight_scale_inv,
        "shared_down_weight": down.weight,
        "shared_down_scale": down.weight_scale_inv,
    }


def _fp8_moe_layer_args(mlp) -> dict:
    """sglang's DeepseekV2MoE with fp8 experts as deepseek_v2_fp8_moe_layer's arguments past
    the norm: the router, the experts in shuffle_weight's 16 x 16 tiles with their scales per
    128 x 128 block, and the routing."""
    experts = mlp.experts
    return {
        **_routing_args(mlp),
        "w1": experts.w13_weight,
        "w1_scale": experts.w13_weight_scale_inv,
        "w2": experts.w2_weight,
        "w2_scale": experts.w2_weight_scale_inv,
    }


def _unquantized_moe_layer_call(mlp, norm: dict):
    """A sparse layer's DeepseekV2MoE with unquantized experts from ``norm``
    (post_attention_layernorm's arguments) on, as its op and arguments: the shared expert in
    their last slot, or apart from them."""
    args = {**norm, **_unquantized_moe_layer_args(mlp)}
    if mlp.num_fused_shared_experts:
        return deepseek_v2_unquantized_moe_layer, args
    shared = mlp.shared_experts
    return deepseek_v2_unquantized_moe_layer_shared_apart, {
        **args,
        "shared_gate_up_weight": shared.gate_up_proj.weight,
        "shared_down_weight": shared.down_proj.weight,
    }


def _unquantized_moe_layer_args(mlp) -> dict:
    """sglang's DeepseekV2MoE with unquantized experts as deepseek_v2_unquantized_moe_layer's
    arguments past the norm: the router, the experts in shuffle_weight's 16 x 16 tiles as
    sglang's aiter path holds them, and the routing."""
    experts = mlp.experts
    # sglang shuffles the experts for its aiter runner into their parameters, which drops
    # shuffle_weight's mark
    experts.w13_weight.is_shuffled = experts.w2_weight.is_shuffled = True
    return {
        **_routing_args(mlp),
        "w1": experts.w13_weight,
        "w2": experts.w2_weight,
    }


def _dense_layer_call(mlp, norm: dict):
    """A dense layer's DeepseekV2MLP from ``norm`` (post_attention_layernorm's arguments) on,
    as its op and arguments: the fp8 op for fp8 linears with block scales, else the bf16
    one."""
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    if _fp8_linear(gate_up):
        return deepseek_v2_fp8_dense_layer, {
            **norm,
            "gate_up_weight": gate_up.weight,
            "gate_up_scale": gate_up.weight_scale_inv,
            "down_weight": down.weight,
            "down_scale": down.weight_scale_inv,
        }
    return deepseek_v2_dense_layer, {
        **norm,
        "gate_up_weight": gate_up.weight,
        "down_weight": down.weight,
    }


def _fp8_linear(linear) -> bool:
    """Whether ``linear`` holds an fp8 weight with its scale per block (weight_scale_inv)."""
    import torch

    return (
        linear.weight.dtype == torch.float8_e4m3fn and getattr(linear, "weight_scale_inv", None) is not None
    )


def _mlp_falls_back(mlp) -> bool:
    """When a dense layer's DeepseekV2MLP leaves its plain path on HIP: weights neither both
    bf16 nor both fp8 with block scales (_fp8_linear), a SwiGLU limit, NVFP4's fused gate_up,
    and an MLP each rank runs whole (replicated: moe_dense_tp_size 1)."""
    import torch

    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    is_bf16 = gate_up.weight.dtype == down.weight.dtype == torch.bfloat16
    return (
        not (is_bf16 or (_fp8_linear(gate_up) and _fp8_linear(down)))
        or mlp.swiglu_limit is not None
        or getattr(mlp, "_enable_nvfp4_gemm_swiglu_fusion", False)
        or mlp.is_replicated
    )


@sglang_route(
    "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward_normal",
    versions=">=0.5.22,<0.5.23",
)
def _moe_forward_normal(
    self,
    hidden_states,
    gemm_output_zero_allocator=None,
    input_ids=None,
    input_ids_global=None,
    skip_shared_experts=False,
    num_token_non_padded=None,
    use_vision_topk=False,
    return_moe_output=False,
):
    """A DeepseekV2MoE's router, top-k and experts (the shared one among them) as one op
    call on hidden_states, in the layers the layer route leaves to sglang: the op of the
    experts' format (_moe_call); then forward_normal's end, sglang's post-experts
    all-reduce summing the output over the ranks unless a later step does
    (reduce_results)."""
    if _forward_normal_falls_back(
        self,
        skip_shared_experts,
        num_token_non_padded,
        use_vision_topk,
        return_moe_output,
    ):
        return FallbackReason.OUT_OF_CONTRACT
    call = _moe_call(self, hidden_states)
    if isinstance(call, FallbackReason):
        return call
    op, arguments = call
    out = op(**arguments)
    if isinstance(out, FallbackReason) or not self.reduce_results:
        return out
    from sglang.srt.layers.moe import post_experts_all_reduce

    return post_experts_all_reduce(out)


def _forward_normal_falls_back(
    mlp, skip_shared_experts, num_token_non_padded, use_vision_topk, return_moe_output
) -> bool:
    """When DeepseekV2MoE.forward_normal leaves its op: the shared expert skipped or run
    beside the experts' combine (single-batch overlap), padded rows routed to nothing
    (num_token_non_padded), the vision top-k, the output asked for in pieces (MoEOutput),
    and the MoE layer ops' cases (_moe_falls_back)."""
    return (
        skip_shared_experts
        or num_token_non_padded is not None
        or use_vision_topk
        or return_moe_output
        or mlp._fuse_shared_experts_inside_sbo
        or _moe_falls_back(mlp)
    )


def _moe_call(mlp, hidden_states):
    """DeepseekV2MoE ``mlp``'s op on ``hidden_states`` and its arguments, by the experts'
    format: deepseek_v2_moe for MXFP4 experts with the shared one in their last slot;
    OUT_OF_CONTRACT for any other format."""
    if _experts_format(mlp.experts) == "QuarkW4A4MXFp4MoE":
        return deepseek_v2_moe, {"hidden_states": hidden_states, **_moe_layer_args(mlp)}
    return FallbackReason.OUT_OF_CONTRACT


@sglang_route(
    "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.forward",
    versions=">=0.5.22,<0.5.23",
)
def _model_nextn(self, input_ids, positions, forward_batch, input_embeds=None):
    """The MTP draft layer as sglang runs it on HIP, its decoder's input from one op call
    (deepseek_v2_eh_proj: both norms and eh_proj, the ranks' shares summed in the op), which
    starts the layer's residual stream; shared_head's norm takes what the decoder leaves in
    it (residual_batch.final_norm)."""
    import torch
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.utils import BumpAllocator

    if _model_nextn_falls_back(self, input_ids, forward_batch, input_embeds):
        return FallbackReason.OUT_OF_CONTRACT
    hidden_states = deepseek_v2_eh_proj(
        embeds=self.embed_input_ids(input_ids, forward_batch),
        hidden=forward_batch.spec_info.hidden_states,
        enorm_weight=self.enorm.weight,
        hnorm_weight=self.hnorm.weight,
        eps=self.enorm.variance_epsilon,
        weight=self.eh_proj.weight,
    )
    if isinstance(hidden_states, FallbackReason):
        return hidden_states
    zero_allocator = BumpAllocator(buffer_size=2, dtype=torch.float32, device=input_ids.device)
    residual_batch.start(forward_batch)
    index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
    with get_global_expert_distribution_recorder().disable_this_region():
        hidden_states, topk_indices = self.decoder(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            prev_topk_indices=index_topk_share.topk_indices,
        )
    hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
    hidden_states = residual_batch.final_norm(hidden_states, forward_batch, self.shared_head.norm)
    index_topk_share.update(topk_indices)
    index_topk_share.publish()
    return hidden_states


def _model_nextn_falls_back(self, input_ids, forward_batch, input_embeds) -> bool:
    """When the MTP draft layer's route falls back to sglang: embeddings passed in or
    multimodal ones, a rotation of the hidden states (NPU), no tokens, norms of two
    epsilons, and DP attention (each rank then drafts its own requests, while the op sums
    the ranks' shares of one set of tokens)."""
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    return (
        input_embeds is not None
        or forward_batch.mm_input_embeds is not None
        or self.rot_weight is not None
        or input_ids.shape[0] == 0
        or self.enorm.variance_epsilon != self.hnorm.variance_epsilon
        or is_dp_attention_enabled()
    )
