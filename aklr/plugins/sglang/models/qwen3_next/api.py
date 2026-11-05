# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.qwen3_5.api import gated_attention_decode_group_fp8_quant
from aklr.plugins.sglang.models.qwen3_5.contract import ROTARY_DIM
from aklr.plugins.sglang.models.qwen3_next.contract import (
    QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT,
    QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT,
)

__all__ = ["qwen3_next_gdn_decode_group_fp8_quant", "qwen3_next_gdn_prefill_group_fp8_quant"]


@api(QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT)
def qwen3_next_gdn_prefill_group_fp8_quant(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    cu_seqlens,
    has_initial_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
    max_seq_len=1024,
): ...


@api(QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT)
def qwen3_next_gdn_decode_group_fp8_quant(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
): ...


@sglang_route("sglang.srt.models.qwen3_next.Qwen3GatedDeltaNet.forward", versions=("0.5.22",))
def _gdn_forward(self, hidden_states, forward_batch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if forward_batch.forward_mode == ForwardMode.DECODE:
        run = _decode
    elif forward_batch.forward_mode == ForwardMode.EXTEND:
        run = _prefill
    else:
        return FallbackReason.OUT_OF_CONTRACT
    if _layer_falls_back(self):
        return FallbackReason.OUT_OF_CONTRACT
    # the input projections take hidden_states whole; the op needs only its rows
    hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    return run(self, hidden_states, hidden, forward_batch, get_attn_backend().linear_attn_backend)


def _layer_falls_back(layer) -> bool:
    """Whether the layer computes other than the ops (LoRA, other activations or another
    gated norm), or its out_proj cannot take the ops' values and row-major per-head
    scales (_takes_fp8_groups)."""
    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
    from sglang.srt.runtime_context import get_lora

    norm = layer.norm
    gated_rms_norm = (
        type(norm) is FusedRMSNormGated and norm.bias is None
        if type(norm) is not RMSNormGated
        else norm.norm_before_gate and norm.group_size is None and norm.bias is None
    )
    return (
        get_lora().enable_lora
        or layer.attn.activation != "silu"
        or not gated_rms_norm
        or norm.activation not in ("silu", "swish")
        or not _takes_fp8_groups(layer.out_proj)
    )


def _takes_fp8_groups(linear) -> bool:
    """Whether ``linear`` takes FP8 values with row-major per-1x128 scales: block-FP8 on
    aiter's CK GEMM only, since sglang hands aiter's Triton GEMM those scales transposed."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )

    method = linear.quant_method
    return (
        type(method) is Fp8LinearMethod
        and method.block_quant
        and not method.use_mxfp8
        and not method.block_fp8_as_mxfp8
        and method.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        and not use_aiter_triton_gemm_w8a8_tuned_gfx950(*linear.weight.shape)
    )


def _prefill(layer, hidden_states, hidden, forward_batch, backend):
    """sglang's input projections, then the op, its values and per-head scales into out_proj."""
    metadata, lengths = backend.forward_metadata, forward_batch.extend_seq_lens_cpu
    if (
        # a capture would replay this call's lengths
        torch.cuda.is_current_stream_capturing()
        # sglang would also write the states' prefix-cache snapshots, or score multi-item prompts
        or metadata.has_mamba_track_mask
        or backend.mis_metadata is not None
        # rows past the sequences (DP attention's padding), or lengths of another batch
        or lengths is None
        or len(lengths) != forward_batch.batch_size
        or sum(lengths) != hidden.shape[0]
    ):
        return FallbackReason.OUT_OF_CONTRACT
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.attn.layer_id)
    arguments = _layer_inputs(layer, cache) | {
        "cache_indices": metadata.mamba_cache_indices,
        "cu_seqlens": metadata.query_start_loc,
        "has_initial_state": forward_batch.extend_prefix_lens > 0,
        "max_seq_len": max(lengths),
    }
    op = qwen3_next_gdn_prefill_group_fp8_quant
    reason = op.fallback_reason(**_projection_stand_ins(arguments, hidden), **arguments)
    if reason is not None:
        return reason
    qkvz, ba = layer._forward_input_proj(hidden_states)
    result = op(projected_qkvz=qkvz, projected_ba=ba, **arguments)
    if isinstance(result, FallbackReason):
        return result
    _, values, scales = result
    return layer.out_proj((values, scales))[0]


def _decode(layer, hidden_states, hidden, forward_batch, backend):
    """Graph capture too: sglang's projections, the op, then sglang's state tracking and
    out_proj."""
    metadata = backend.forward_metadata
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.attn.layer_id)
    # ReplaySSM keeps per-slot ring buffers that the op does not update
    ring = (cache.replayssm_d, cache.replayssm_k, cache.replayssm_g)
    if any(t is not None for t in (*ring, metadata.replayssm_write_pos, metadata.replayssm_force_flush)):
        return FallbackReason.OUT_OF_CONTRACT
    # padded graph rows hold sglang's -1 slot, whose state the op leaves alone
    arguments = _layer_inputs(layer, cache) | {"cache_indices": metadata.mamba_cache_indices}
    op = qwen3_next_gdn_decode_group_fp8_quant
    reason = op.fallback_reason(**_projection_stand_ins(arguments, hidden), **arguments)
    if reason is not None:
        return reason
    qkvz, ba = layer._forward_input_proj(hidden_states)
    result = op(projected_qkvz=qkvz, projected_ba=ba, **arguments)
    if isinstance(result, FallbackReason):
        return result
    backend._track_mamba_state_decode(
        forward_batch,
        arguments["conv_state"],
        arguments["delta_state"],
        arguments["cache_indices"],
        layer.attn.layer_id,
    )
    _, values, scales = result
    return layer.out_proj((values, scales))[0]


def _layer_inputs(layer, cache):
    """The arguments the ops take from the layer and its state pools."""
    weight, bias = layer.attn.conv_weights, layer.attn.bias
    return {
        "conv_state": cache.conv[0],
        "delta_state": cache.temporal,
        "conv_weight": weight,
        "conv_bias": _no_bias(weight.shape[0], weight.dtype, weight.device) if bias is None else bias,
        "a_log": layer.A_log,
        "dt_bias": layer.dt_bias,
        "norm_weight": layer.norm.weight,
        "scale": layer.attn.head_k_dim**-0.5,
        "eps": layer.norm.eps,
    }


def _projection_stand_ins(arguments, hidden):
    """Empty stand-ins of sglang's projections, to ask an op before they run; passed as a
    temporary, so that the allocator hands their memory on to the projections."""
    _, heads, value_dim, _ = arguments["delta_state"].shape
    rows, width = hidden.shape[0], arguments["conv_state"].shape[1] + heads * value_dim
    return {
        "projected_qkvz": torch.empty((rows, width), dtype=hidden.dtype, device=hidden.device),
        "projected_ba": torch.empty((rows, 2 * heads), dtype=hidden.dtype, device=hidden.device),
    }


@cache
def _no_bias(size, dtype, device):
    """The convolution bias the ops take: the layer has none."""
    return torch.zeros(size, dtype=dtype, device=device)


@sglang_route(
    "sglang.srt.models.qwen3_next.Qwen3HybridAttentionDecoderLayer.self_attention", versions=("0.5.22",)
)
def _attention_self_attention(self, positions, hidden_states, forward_batch):
    """Decode as HIP runs it (forward_prepare_native, the aiter backend's forward_decode,
    sigmoid_gate_mul, then o_proj's own FP8 quantization of its BF16 input), graph capture
    too: sglang's qkv_proj, qwen3_5's gated attention op, then sglang's o_proj on the op's
    FP8 values and scales."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import qwen3_next as model

    if not model._is_hip or not forward_batch.forward_mode.is_decode() or forward_batch.spec_info is not None:
        return FallbackReason.OUT_OF_CONTRACT
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    if positions.dim() != 1 or _attention_falls_back(self, backend):
        return FallbackReason.OUT_OF_CONTRACT
    hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    rows = hidden.shape[0]
    key_cache, value_cache = backend.token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
    metadata = backend.forward_metadata
    arguments = {
        "q_norm_weight": self.q_norm.weight,
        "k_norm_weight": self.k_norm.weight,
        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
        "cache_locations": forward_batch.out_cache_loc[:rows],
        "key_cache": key_cache,
        "value_cache": value_cache,
        "kv_indptr": metadata.kv_indptr,
        "kv_indices": metadata.kv_indices,
        "eps": self.q_norm.variance_epsilon,
        "scale": backend.scale,
    }
    op = gated_attention_decode_group_fp8_quant
    reason = op.fallback_reason(**_attention_stand_ins(self, hidden, positions), **arguments)
    if reason is not None:
        return reason
    qkv, _ = self.qkv_proj(hidden_states)
    # the op's temporal, height and width rows, each the one row Qwen3-Next rotates by, which
    # makes its interleaved mRoPE plain NeoX RoPE; row-major, as the op reads them
    result = op(projected_qkv=qkv, positions=positions.expand(3, -1).contiguous(), **arguments)
    if isinstance(result, FallbackReason):
        return result
    values, scales = result
    return self.o_proj((values.view(rows, -1), scales))[0]


def _attention_falls_back(layer, backend) -> bool:
    """Whether sglang computes other than the op: LoRA or sequence parallelism; another
    backend or decode path than paged_attention_ragged over one-token pages (MLA, unified or
    sliding-window attention, a 5-D pool, context parallelism, a logit cap); KV scales other
    than 1 (sglang writes at the layer's and reads at 1); another rotary embedding than NeoX
    RoPE on the op's 64 channels; norm epsilons that differ; or an o_proj that cannot take
    the op's FP8 groups (_takes_fp8_groups, and no row-wise FP8 copy for its small-M path,
    which only BF16 inputs take)."""
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.runtime_context import get_forward, get_lora

    attn, rope = layer.attn, layer.rotary_emb
    unit_scales = all(
        scale is None or as_float == 1.0
        for scale, as_float in ((attn.k_scale, attn.k_scale_float), (attn.v_scale, attn.v_scale_float))
    )
    return (
        get_lora().enable_lora
        or get_forward().sp_active
        or not layer.attn_output_gate
        or type(backend) is not AiterAttnBackend
        or backend.use_mla
        or backend.use_triton_unified_attention
        or backend.kv_cache_is_vectorized_5d
        or backend.use_sliding_window_kv_pool
        or backend.dcp_world_size > 1
        or bool(attn.logit_cap)
        or attn.sliding_window_size != -1
        or attn.qk_head_dim != attn.v_head_dim
        or not unit_scales
        or type(rope) is not RotaryEmbedding
        or rope._force_native
        or not rope.is_neox_style
        or rope.rotary_dim != ROTARY_DIM
        or rope.head_size != layer.head_dim
        or not (type(layer.q_norm) is type(layer.k_norm) is GemmaRMSNorm)
        or layer.k_norm.variance_epsilon != layer.q_norm.variance_epsilon
        or not _takes_fp8_groups(layer.o_proj)
        or getattr(layer.o_proj.weight, "_ptpc_weight", None) is not None
    )


def _attention_stand_ins(layer, hidden, positions):
    """Empty stand-ins of qkv_proj's output and of the op's three position rows, to ask the
    op before they are made; passed as a temporary, so that the allocator hands their memory on."""
    width = 2 * layer.q_size + 2 * layer.kv_size
    return {
        "projected_qkv": torch.empty(
            (hidden.shape[0], width), dtype=layer.q_norm.weight.dtype, device=hidden.device
        ),
        "positions": torch.empty((3, positions.shape[0]), dtype=positions.dtype, device=positions.device),
    }
