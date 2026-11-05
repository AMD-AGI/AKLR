# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
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
    scales: block-FP8 on aiter's CK GEMM only, since sglang hands aiter's Triton GEMM
    those scales transposed."""
    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )
    from sglang.srt.runtime_context import get_lora

    norm, method = layer.norm, layer.out_proj.quant_method
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
        or not (
            type(method) is Fp8LinearMethod
            and method.block_quant
            and not method.use_mxfp8
            and not method.block_fp8_as_mxfp8
            and method.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        )
        or use_aiter_triton_gemm_w8a8_tuned_gfx950(*layer.out_proj.weight.shape)
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
