# Created by OpenAI
from functools import cache

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.qwen3_5.contract import (
    GATED_ATTENTION_DECODE_GROUP_FP8_QUANT,
    GDN_DECODE,
    GDN_DECODE_GROUP_FP8_QUANT,
    GDN_PREFILL_GROUP_FP8_QUANT,
    MROPE_SECTION,
    ROTARY_DIM,
)

__all__ = [
    "gated_attention_decode_group_fp8_quant",
    "gdn_decode",
    "gdn_decode_group_fp8_quant",
    "gdn_prefill_group_fp8_quant",
]


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


@api(GATED_ATTENTION_DECODE_GROUP_FP8_QUANT)
def gated_attention_decode_group_fp8_quant(
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    kv_indptr,
    kv_indices,
    eps=1e-6,
    scale=256**-0.5,
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
    if not _takes_fp8_groups(self.out_proj):
        # BF16 projections (an MXFP4 checkpoint keeps its GDN layers unquantized): decode on the
        # BF16 op alone, whose output out_proj takes as sglang's own decode hands it
        if run is _prefill or not _unquantized_projections(self):
            return FallbackReason.OUT_OF_CONTRACT
        run = _decode_unquantized
    # AMD's fused AR+norm hands a (bf16, fp8, scale) tuple, whole to sglang's input projections
    hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    return run(self, hidden_states, hidden, forward_batch, get_attn_backend().linear_attn_backend)


def _layer_falls_back(layer) -> bool:
    """Whether the layer computes other than the ops: LoRA, or other activations."""
    from sglang.srt.runtime_context import get_lora

    return get_lora().enable_lora or layer.attn.activation != "silu" or layer.norm.activation != "swish"


def _unquantized_projections(layer) -> bool:
    """Whether sglang runs the layer's input projections and out_proj unquantized: out_proj then
    takes the BF16 normalized output, as sglang's own decode hands it."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return all(
        type(linear.quant_method) is UnquantizedLinearMethod
        for linear in (layer.in_proj_qkvz, layer.in_proj_ba, layer.out_proj)
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


def _decode_unquantized(layer, hidden_states, hidden, forward_batch, backend):
    """_decode on the BF16 op alone, for a layer whose projections sglang runs unquantized."""
    return _decode(layer, hidden_states, hidden, forward_batch, backend, fp8_groups=False)


def _decode(layer, hidden_states, hidden, forward_batch, backend, *, fp8_groups=True):
    """Graph capture too: sglang's projections, the FP8 op or else the BF16 op (the BF16 op
    alone without ``fp8_groups``), then sglang's state tracking and out_proj."""
    metadata = backend.forward_metadata
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.attn.layer_id)
    # ReplaySSM keeps per-slot ring buffers that the ops do not update
    ring = (cache.replayssm_d, cache.replayssm_k, cache.replayssm_g)
    if any(t is not None for t in (*ring, metadata.replayssm_write_pos, metadata.replayssm_force_flush)):
        return FallbackReason.OUT_OF_CONTRACT
    # padded graph rows hold sglang's -1 slot, whose state the ops leave alone
    arguments = _layer_inputs(layer, cache) | {"cache_indices": metadata.mamba_cache_indices}
    reasons, projected = [], None
    for op in (gdn_decode_group_fp8_quant, gdn_decode) if fp8_groups else (gdn_decode,):
        reason = op.fallback_reason(**_projection_stand_ins(arguments, hidden), **arguments)
        if reason is not None:
            reasons.append(reason)
            continue
        if projected is None:
            qkvz, ba = layer._forward_input_proj(hidden_states)
            if not fp8_groups:
                # up to 64 rows, sglang runs both BF16 projections as one GEMM and hands on
                # column slices of its output; the ops read them row-major
                qkvz, ba = qkvz.contiguous(), ba.contiguous()
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


@sglang_route("sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention", versions=("0.5.22",))
def _attention_self_attention(self, positions, hidden_states, forward_batch):
    """Decode as HIP runs it (forward_prepare_fused_gate, the aiter backend's forward_decode,
    fused_sigmoid_mul), graph capture too: sglang's qkv_proj, the op, then sglang's o_proj
    on the op's FP8 values and scales."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import qwen3_5 as model

    if not model._is_hip or not forward_batch.forward_mode.is_decode() or forward_batch.spec_info is not None:
        return FallbackReason.OUT_OF_CONTRACT
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    if _attention_falls_back(self, backend):
        return FallbackReason.OUT_OF_CONTRACT
    # AMD's fused AR+norm hands a (bf16, fp8, scale) or (fp8, scale) tuple: qkv_proj takes
    # the part forward_prepare_fused_gate selects for it
    if model._use_aiter and isinstance(hidden_states, tuple):
        hidden_states = model._select_fused_ar_input_for_linear(hidden_states, self.qkv_proj)
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
    # a decode graph hands the rows of its wider positions buffer: the op reads them row-major
    result = op(projected_qkv=qkv, positions=positions.contiguous(), **arguments)
    if isinstance(result, FallbackReason):
        return result
    values, scales = result
    return self.o_proj((values.view(rows, -1), scales))[0]


def _attention_falls_back(layer, backend) -> bool:
    """Whether sglang computes other than the op: LoRA or sequence parallelism; another
    backend or decode path than paged_attention_ragged over one-token pages (MLA, unified or
    sliding-window attention, a 5-D pool, context parallelism, a logit cap); KV scales other
    than 1 (sglang writes at the layer's and reads at 1); another rotary embedding than
    interleaved NeoX mRoPE on the op's sections; norm epsilons that differ; or an o_proj
    that cannot take the op's FP8 groups (_takes_fp8_groups, and no row-wise FP8 copy for
    its small-M path, which only BF16 inputs take)."""
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
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
        or type(rope) is not MRotaryEmbedding
        or rope._force_native
        or not rope.is_neox_style
        or not rope.mrope_interleaved
        or rope.mrope_interleaved_glm
        or tuple(rope.mrope_section or ()) != MROPE_SECTION
        or rope.rotary_dim != ROTARY_DIM
        or rope.head_size != layer.head_dim
        or not (type(layer.q_norm) is type(layer.k_norm) is GemmaRMSNorm)
        or layer.k_norm.variance_epsilon != layer.q_norm.variance_epsilon
        or not _takes_fp8_groups(layer.o_proj)
        or getattr(layer.o_proj.weight, "_ptpc_weight", None) is not None
    )


def _attention_stand_ins(layer, hidden, positions):
    """Empty stand-ins of qkv_proj's output and of the row-major positions, to ask the op
    before they are made; passed as a temporary, so that the allocator hands their memory on."""
    width = 2 * layer.q_size + 2 * layer.kv_size
    return {
        "projected_qkv": torch.empty(
            (hidden.shape[0], width), dtype=layer.q_norm.weight.dtype, device=hidden.device
        ),
        "positions": torch.empty_like(positions, memory_format=torch.contiguous_format),
    }
