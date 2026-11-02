# Created by OpenAI
from functools import cache

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.qwen3_5.contract import (
    GDN_DECODE,
    GDN_DECODE_GROUP_FP8_QUANT,
    GDN_PREFILL_GROUP_FP8_QUANT,
)

__all__ = ["gdn_decode", "gdn_decode_group_fp8_quant", "gdn_prefill_group_fp8_quant"]


@api(GDN_PREFILL_GROUP_FP8_QUANT)
def gdn_prefill_group_fp8_quant(
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
    max_seq_len=2048,
): ...


@api(GDN_DECODE_GROUP_FP8_QUANT)
def gdn_decode_group_fp8_quant(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
): ...


@api(GDN_DECODE)
def gdn_decode(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
): ...


@sglang_route("sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward", versions=("0.5.22",))
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
    # AMD's fused AR+norm hands a (bf16, fp8, scale) tuple, whole to sglang's input projections
    hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    return run(self, hidden_states, hidden, forward_batch, get_attn_backend().linear_attn_backend)


def _layer_falls_back(layer) -> bool:
    """Whether the layer computes other than the ops (LoRA, its activations), or its out_proj
    cannot take the FP8 op's values and row-major per-head scales: block-FP8 on aiter's CK GEMM
    only, since sglang hands aiter's Triton GEMM those scales transposed."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )
    from sglang.srt.runtime_context import get_lora

    method = layer.out_proj.quant_method
    return (
        get_lora().enable_lora
        or layer.attn.activation != "silu"
        or layer.norm.activation != "swish"
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
    """sglang's input projections, then the FP8 op, its values and per-head scales into out_proj."""
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
    weight = layer.attn.conv_weights
    arguments = _layer_inputs(layer, cache) | {
        "cache_indices": metadata.mamba_cache_indices,
        "cu_seqlens": metadata.query_start_loc,
        "has_initial_state": forward_batch.extend_prefix_lens > 0,
        "conv_bias": _no_bias(weight.shape[0], weight.dtype, weight.device),
        "max_seq_len": max(lengths),
    }
    op = gdn_prefill_group_fp8_quant
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
    """Graph capture too: sglang's projections, the FP8 op or else the BF16 op, then sglang's
    state tracking and out_proj."""
    metadata = backend.forward_metadata
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.attn.layer_id)
    # ReplaySSM keeps per-slot ring buffers that the ops do not update
    ring = (cache.replayssm_d, cache.replayssm_k, cache.replayssm_g)
    if any(t is not None for t in (*ring, metadata.replayssm_write_pos, metadata.replayssm_force_flush)):
        return FallbackReason.OUT_OF_CONTRACT
    # padded graph rows hold sglang's -1 slot, whose state the ops leave alone
    arguments = _layer_inputs(layer, cache) | {"cache_indices": metadata.mamba_cache_indices}
    reasons, projected = [], None
    for op in (gdn_decode_group_fp8_quant, gdn_decode):
        reason = op.fallback_reason(**_projection_stand_ins(arguments, hidden), **arguments)
        if reason is not None:
            reasons.append(reason)
            continue
        if projected is None:
            qkvz, ba = layer._forward_input_proj(hidden_states)
            projected = {"projected_qkvz": qkvz, "projected_ba": ba}
        result = op(**projected, **arguments)
        if not isinstance(result, FallbackReason):
            break
        reasons.append(result)
    else:
        return reasons[0]
    backend._track_mamba_state_decode(
        forward_batch,
        arguments["conv_state"],
        arguments["delta_state"],
        arguments["cache_indices"],
        layer.attn.layer_id,
    )
    if op is gdn_decode:
        result = result.reshape(hidden.shape[0], -1)
    return layer.out_proj(result)[0]


def _layer_inputs(layer, cache):
    """The arguments the ops take from the layer and its state pools."""
    return {
        "conv_state": cache.conv[0],
        "delta_state": cache.temporal,
        "conv_weight": layer.attn.conv_weights,
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
    """The prefill op's convolution bias: the layer has none."""
    return torch.zeros(size, dtype=dtype, device=device)
