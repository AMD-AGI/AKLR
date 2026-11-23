# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next's Gated DeltaNet core between sglang's input and output projections: the
causal convolution, the gated delta recurrence, the gated RMSNorm and the grouped FP8
quantization of out_proj's input, for prefill and for decode; and its full-attention
layers' gated decode attention over their FP8 KV cache.

sglang's Qwen3-Next holds the Q/K/V/Z and B/A projections grouped by key head: each key
head's Q, its K, then the V and the Z of its value heads; and each key head's B, then
its A. Qwen3.5 holds them contiguous, so the qwen3_5 contracts do not describe these
inputs. A model is a scenario: Qwen3-Next-80B-A3B at attention TP4 has 4 key heads,
8 value heads, key_dim and value_dim 128.

The full-attention layers compute what qwen3_5's gated attention decode does, except that
Qwen3-Next rotates by one position per row (plain NeoX RoPE, not interleaved mRoPE), so
the qwen3_5 contract does not describe its positions."""

from functools import cache

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    BoundedCuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    CompareSpec,
    DTypeSpec,
    LayoutVariant,
    LayoutVariantsSpec,
    ScalarSpec,
    TensorSpec,
    attr,
    axis,
    derived_axis,
)
from aklr.core.runtime.dispatch import FallbackReason
from aklr.plugins import Plugin
from aklr.plugins.sglang.models.qwen3_5.plugin import (
    GATED_ATTENTION_DECODE_GROUP_FP8_QUANT,
    ROTARY_DIM,
    gated_attention_decode_group_fp8_quant,
)

__all__ = [
    "QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT",
    "QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT",
    "QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT",
    "declare",
    "family",
    "qwen3_next_gated_attention_decode_group_fp8_quant",
    "qwen3_next_gdn_decode_group_fp8_quant",
    "qwen3_next_gdn_prefill_group_fp8_quant",
]


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_FP32 = DTypeSpec(any_of=(DType.FLOAT32,))
_FP8 = DTypeSpec(any_of=(DType.FLOAT8_E4M3FN,))
_INT32 = DTypeSpec(any_of=(DType.INT32,))
_INT64 = DTypeSpec(any_of=(DType.INT64,))
_FINITE = FiniteDistribution()


def _tensor(description, shape, dtype, *, mutable=False, bound=None, distribution=_FINITE):
    return TensorSpec(
        description=description,
        shape=shape,
        dtype=dtype,
        distribution=distribution,
        mutable=mutable,
        error_bounds=() if bound is None else bound,
    )


# the heads on one rank of the attention TP group, and the widths they give
_HEAD_AXES = {
    "n_v_heads": axis("Value heads on this rank.", 1, 64),
    "key_dim": axis("Key head width.", 16, 128),
    "value_dim": axis("FP8 group and value-head width.", 128, 128),
    "conv_dim": axis("Convolution channels: every Q, then every K, then every V.", 160, 8192),
    "value_width": derived_axis("Total value width.", "n_v_heads * value_dim"),
    "n_k_heads": derived_axis("Key heads on this rank.", "(conv_dim - value_width) // (2 * key_dim)"),
    "projection_dim": derived_axis(
        "Q/K/V/Z projection width: per key head, 2 * key_dim + 2 * value_dim * n_v_heads // n_k_heads.",
        "conv_dim + value_width",
    ),
    "gate_dim": derived_axis("B/A projection width.", "2 * n_v_heads"),
}
_HEAD_CONSTRAINTS = (
    "conv_dim > value_width",
    "(conv_dim - value_width) % (2 * key_dim) == 0",
    "n_k_heads >= 1",
    "n_v_heads % n_k_heads == 0",
)
_PROJECTIONS = {
    "projected_qkvz": _tensor(
        "Q/K/V/Z projections grouped by key head: each key head's Q, its K, then the V "
        "and the Z of its value heads.",
        ("n_tokens", "projection_dim"),
        _BF16,
    ),
    "projected_ba": _tensor(
        "B/A projections grouped by key head: the B, then the A, of its value heads.",
        ("n_tokens", "gate_dim"),
        _BF16,
    ),
}
_CONV_STATE = _tensor(
    "Live three-token convolution histories, channels as conv_dim orders them.",
    ("n_slots", "conv_dim", 3),
    _BF16,
    mutable=True,
    bound=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
)
_WEIGHTS = {
    "conv_weight": _tensor("Four causal convolution taps, oldest first.", ("conv_dim", 4), _BF16),
    "conv_bias": _tensor("Convolution bias.", ("conv_dim",), _BF16),
    "a_log": _tensor("Log decay magnitude.", ("n_v_heads",), _FP32),
    "dt_bias": _tensor("Gate bias.", ("n_v_heads",), _BF16),
    "norm_weight": _tensor("Gated RMSNorm weights.", ("value_dim",), _BF16),
    "scale": ScalarSpec(
        description="Query scale.",
        dtype=DType.FLOAT32,
        interval=(0.0883883461356163, 128**-0.5),
        default=128**-0.5,
    ),
    "eps": ScalarSpec(
        description="Gated RMSNorm epsilon.",
        dtype=DType.FLOAT32,
        interval=(9.999999974752427e-7, 1e-6),
        default=1e-6,
    ),
}
_OUTPUTS = {
    "normalized": TensorSpec(
        description="Gated normalized BF16 values.",
        shape=("n_tokens", "n_v_heads", "value_dim"),
        dtype=_BF16,
        # sglang's causal_conv1d kernels round each tap's product to BF16 before the FP32
        # sum; these kernels and the reference sum exact products. A Q/K/V channel can then
        # differ by one BF16 ulp and a core element by up to ~2**-7 of the core's RMS, which
        # the gated norm multiplies by |norm_weight * SiLU(z)| (~5 at N(0, 1) inputs), so
        # at small elements the relative term does not cover it. Under (0.005, 0.02)
        # sglang's own path failed CI's unit test (run 41950799326, MI355X): decode
        # violation_fraction 0.000977, peak_bound_multiple 1.46; prefill 0.000977, 2.21.
        # 0.04 is 2**-7 at a gate of 5; on CI's inputs, a CPU emulation of sglang's path
        # that gives those numbers peaks at 0.33 (decode) and 0.55 (prefill) of it
        error_bounds=ToleranceBound(absolute_tolerance=0.04, relative_tolerance=0.02),
    ),
    "values": TensorSpec(
        description="Grouped E4M3FN values.",
        shape=("n_tokens", "value_width"),
        dtype=_FP8,
        error_bounds=ToleranceBound(absolute_tolerance=2, relative_tolerance=0.125),
    ),
    "scales": TensorSpec(
        description="One FP32 scale per token and value head.",
        shape=("n_tokens", "n_v_heads"),
        dtype=_FP32,
        error_bounds=ToleranceBound(absolute_tolerance=1e-7, relative_tolerance=0.02),
    ),
}
_STEP = (
    "the width-four causal convolution of the history and the token's Q/K/V channels "
    "adds the bias to the FP32 sum of the four products, oldest tap first, and its SiLU "
    "is rounded to BF16. beta, the sigmoid of B, is rounded to BF16, and the decay is "
    "exp(-exp(a_log) * softplus(A + dt_bias)). Each value head uses its key head's Q "
    "and K, L2-normalized with epsilon 1e-6; the gated delta rule updates the FP32 "
    "delta_state in [value,key] order, and the core output, Q times the state times "
    "scale, is rounded to BF16. The core is gated RMS-normalized with norm_weight and "
    "SiLU of the head's Z, rounded to BF16, and quantized per value head to E4M3FN with "
    "scale=max(amax,1e-10)/448"
)

QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT = OpContract(
    contract_id="qwen3_next_gdn_prefill_group_fp8_quant@1.0",
    op_type="qwen3_next",
    description=(
        "Qwen3-Next GDN prefill from Q/K/V/Z and B/A projections grouped by key head. "
        "cu_seqlens partitions the tokens into positive sequences with actual maximum "
        "max_seq_len; each sequence uses its distinct cache_indices slot, and a fresh "
        "one (has_initial_state false) starts from zero history and state, ignoring the "
        f"slot's old values. For each token, {_STEP}; Q and K are rounded to BF16 after "
        "their normalization. Each sequence's slot keeps its last three tokens and its "
        "final state; unindexed slots and all other inputs are preserved."
    ),
    axes={
        "n_tokens": axis("Tokens in this prefill.", 1, 16384, is_var=True),
        "n_sequences": axis("Sequences, each with its own state slot.", 1, 2047, is_var=True),
        "n_slots": axis("Allocated state slots.", 1, 2047, is_var=True, any_size=True),
        **_HEAD_AXES,
        "n_starts": axis("Sequence boundaries.", 2, 2048, is_var=True),
    },
    attrs={
        "max_seq_len": attr(
            "Actual maximum sequence length.",
            1024,
            2,
            4096,
            5120,
            default=1024,
            param_name="max_seq_len",
        ),
    },
    constraints=(
        *_HEAD_CONSTRAINTS,
        "n_slots >= n_sequences",
        "n_starts == n_sequences + 1",
        "n_tokens >= n_sequences + max_seq_len - 1",
        "n_tokens <= n_sequences * max_seq_len",
    ),
    inputs={
        **_PROJECTIONS,
        "conv_state": _CONV_STATE,
        "delta_state": _tensor(
            "Live recurrent matrices, value dimension first.",
            ("n_slots", "n_v_heads", "value_dim", "key_dim"),
            _FP32,
            mutable=True,
            # as qwen3_5's prefill: sglang's own prefill puts Q/K/V channels one BF16
            # ulp from the reference and rounds again in its chunked recurrence, 3.1e-2
            # off a 0.59 entry and 6.2e-2 off a 3.1 one on small Points
            bound=ToleranceBound(absolute_tolerance=4e-2, relative_tolerance=1e-2),
        ),
        "cache_indices": _tensor(
            "Distinct state slots for the sequences.",
            ("n_sequences",),
            _INT32,
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        "cu_seqlens": _tensor(
            "Cumulative sequence boundaries.",
            ("n_starts",),
            _INT32,
            distribution=BoundedCuSeqlensDistribution(total="n_tokens", max_seqlen="max_seq_len"),
        ),
        "has_initial_state": _tensor(
            "Continue each slot's history and state, or start from zero.",
            ("n_sequences",),
            DTypeSpec(any_of=(DType.BOOL,)),
        ),
        **_WEIGHTS,
    },
    outputs=_OUTPUTS,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT = OpContract(
    contract_id="qwen3_next_gdn_decode_group_fp8_quant@1.0",
    op_type="qwen3_next",
    description=(
        "Qwen3-Next GDN decode, one token per row, from Q/K/V/Z and B/A projections "
        "grouped by key head. A row whose cache index is -1 is padding: it changes no "
        "state, and its core output is zero. Every other row uses its "
        f"distinct slot: {_STEP}. Q and K stay FP32 after their normalization, and the "
        "slot's history shifts the row in. Unindexed slots and all other inputs are "
        "preserved."
    ),
    axes={
        "n_tokens": axis("Decode rows: one token each, padding rows included.", 1, 2048, is_var=True),
        "n_slots": axis("Allocated state slots.", 1, 2047, is_var=True, any_size=True),
        **_HEAD_AXES,
    },
    constraints=(
        *_HEAD_CONSTRAINTS,
        # drawn indices are distinct, -1 among them at most once
        "n_tokens <= n_slots + 1",
    ),
    inputs={
        **_PROJECTIONS,
        "conv_state": _CONV_STATE,
        "delta_state": _tensor(
            "Live recurrent matrices, value dimension first.",
            ("n_slots", "n_v_heads", "value_dim", "key_dim"),
            _FP32,
            mutable=True,
            # as qwen3_5's decode: one step's change is |update x key|, and a Q/K/V
            # channel a correct kernel rounds to the neighbouring BF16 value moves it
            # by 2**-8 of that
            bound=ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2),
        ),
        "cache_indices": _tensor(
            "Each row's distinct state slot, or -1 for a padding row.",
            ("n_tokens",),
            _INT32,
            distribution=IntRangeDistribution(lo=-1, hi="n_slots - 1", distinct=True),
        ),
        **_WEIGHTS,
    },
    outputs=_OUTPUTS,
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)

# qwen3_5's gated attention decode: its inputs but the positions, its outputs and bounds
_QWEN3_5_ATTENTION = GATED_ATTENTION_DECODE_GROUP_FP8_QUANT
# The FP8 pools' bytes that the kernels' 32-bit offsets address: larger pools stay on sglang.
_POOL_BYTES = 2**31

QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT = OpContract(
    contract_id="qwen3_next_gated_attention_decode_group_fp8_quant@1.0",
    op_type="qwen3_next",
    description=(
        "Qwen3-Next full-attention decode on one rank, between qkv_proj and o_proj, one token "
        "per row. projected_qkv holds each query head's query then its output gate, then the "
        "KV heads' keys, then their values. Q and K are RMS-normalized per head with weight "
        "1 + norm weight and rounded to BF16; each head's first 64 channels take NeoX RoPE at "
        "the row's position, from cos_sin_cache's row of 32 cosines then 32 sines, rounded to "
        "BF16. Each row writes its K and V, cast to E4M3FN, at its cache_locations slot of "
        "key_cache and value_cache; rows sharing a slot leave one of their values there. Query "
        "head h then attends, through KV head h // (n_heads // n_kv_heads), the written slots "
        "kv_indices[kv_indptr[r]:kv_indptr[r + 1]], at most max_context of them, with scores "
        "q . k * scale; the softmax-weighted sum of their values is rounded to BF16, multiplied "
        "by sigmoid of the gate and rounded to BF16, then quantized in groups of 128 channels to "
        "E4M3FN with scale=max(amax,1e-10)/448. Unwritten slots and all other inputs are "
        "preserved."
    ),
    axes=dict(_QWEN3_5_ATTENTION.axes),
    attrs={
        "max_context": attr(
            "The most slots any row attends: the server's context length, rounded up to a "
            "capacity the kernels are built for.",
            8192,
            16384,
            65536,
            131072,
            default=8192,
            param_name="max_context",
        ),
    },
    constraints=(
        "(qkv_dim - 2 * n_kv_heads * head_dim) % (2 * head_dim) == 0",
        "n_heads >= 1",
        "n_heads % n_kv_heads == 0",
        "(n_heads * head_dim) % 128 == 0",
        "n_offsets == n_tokens + 1",
        # every row attends at least one slot and at most max_context
        "n_kv_tokens >= n_tokens + max_context - 1",
        "n_kv_tokens <= n_tokens * max_context",
        "n_slots >= n_tokens",
        f"n_slots * n_kv_heads * head_dim < {_POOL_BYTES}",
    ),
    inputs={
        **{n: _QWEN3_5_ATTENTION.inputs[n] for n in ("projected_qkv", "q_norm_weight", "k_norm_weight")},
        "positions": _tensor(
            "Each row's position.",
            ("n_tokens",),
            _INT64,
            distribution=IntRangeDistribution(lo=0, hi="n_positions - 1"),
        ),
        **{n: _QWEN3_5_ATTENTION.inputs[n] for n in ("cos_sin_cache", "cache_locations")},
        "key_cache": _tensor(
            "This layer's key pool.",
            ("n_slots", "n_kv_heads", "head_dim"),
            _FP8,
            mutable=True,
            # sglang 0.5.23 rotates Qwen3-Next's Q and K on HIP with sgl_kernel's
            # rotary_embedding in BF16 arithmetic: cos and sin, each product and the sum
            # rounded to BF16. Where a rotated key nearly cancels, that is ~2**-8 of the
            # pair's magnitude off the reference's FP32 rotation, several FP8 steps at the
            # smallest keys, past qwen3_5's 2**-9. An FP64 emulation of sglang's path over
            # 1500 seeded input sets at 1 to 256 TP4 rows (23M written key elements) needs
            # at most 0.047 beside the relative 0.125; 2**-6 fails 1-15% of the sets at 32 to
            # 256 rows, 2**-5 up to 1%. Kernels wrong on purpose (GPT-J pairs, the sine's sign,
            # no K norm, an unwritten slot) stay 22x or more past 2**-4.
            bound=ToleranceBound(absolute_tolerance=2**-4, relative_tolerance=0.125),
        ),
        "value_cache": _QWEN3_5_ATTENTION.inputs["value_cache"],
        "kv_indptr": _tensor(
            "Where each row's slots start in kv_indices, then where the last row's end.",
            ("n_offsets",),
            _INT32,
            # bench draws the longest row at max_context, the case a capacity must hold
            distribution=BoundedCuSeqlensDistribution(total="n_kv_tokens", max_seqlen="max_context"),
        ),
        **{n: _QWEN3_5_ATTENTION.inputs[n] for n in ("kv_indices", "eps", "scale")},
    },
    # sglang's own decode is lossier than the reference, as for qwen3_5's op (whose outputs
    # these are): aiter's paged_attention_ragged rounds Q and each 256-token partition's
    # softmax weights to E4M3. An FP64 emulation of Qwen3-Next's sglang path comes to
    # 20.8-27.1 dB (values) and 25.3-39.7 dB (scales) at its TP4 and TP2 heads.
    outputs=dict(_QWEN3_5_ATTENTION.outputs),
    layout_variants=LayoutVariantsSpec(by_name={"row_major": LayoutVariant()}),
    compare=CompareSpec(determinism="bitwise"),
)


family = Plugin("sglang")
family.define(QWEN3_NEXT_GDN_PREFILL_GROUP_FP8_QUANT)
family.define(QWEN3_NEXT_GDN_DECODE_GROUP_FP8_QUANT)
family.define(QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT)
family.build()


@family.op
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


@family.op
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


@family.op
def qwen3_next_gated_attention_decode_group_fp8_quant(
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
    max_context=8192,
): ...


def _gdn_forward(self, hidden_states, forward_batch):
    # sglang 0.5.23 relabels a decode under DP attention a one-token extend, the decode kept as
    # logical_forward_mode: such a batch stays on sglang (0.5.22 has no logical mode)
    if (
        getattr(forward_batch, "logical_forward_mode", forward_batch.forward_mode)
        != forward_batch.forward_mode
    ):
        return FallbackReason.OUT_OF_CONTRACT
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


def _attention_self_attention(self, positions, hidden_states, forward_batch):
    """Decode as HIP runs it (forward_prepare_native, the aiter backend's forward_decode,
    fused_sigmoid_mul, then o_proj's own FP8 quantization of its BF16 input), graph capture
    too: sglang's qkv_proj, an op (_attention_ops), then sglang's o_proj on the op's FP8
    values and scales."""
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
        "eps": self.q_norm.variance_epsilon,
        "scale": backend.scale,
    }
    reasons, qkv = [], None
    for op, asked, made in _attention_ops(positions, metadata.kv_indices, rows, backend.max_context_len):
        reason = op.fallback_reason(projected_qkv=_qkv_stand_in(self, hidden), **arguments, **asked)
        if reason is not None:
            reasons.append(reason)
            continue
        if qkv is None:
            qkv, _ = self.qkv_proj(hidden_states)
        result = op(projected_qkv=qkv, **arguments, **made())
        if not isinstance(result, FallbackReason):
            values, scales = result
            return self.o_proj((values.view(rows, -1), scales))[0]
        reasons.append(result)
    return reasons[0]


def _attention_ops(positions, kv_indices, rows, context_len):
    """The ops the route asks, in order, each with the arguments to ask it with and a function
    making those it is called with:
    - Qwen3-Next's own, at the capacity sglang's context length rounds up to, which bounds
      every row; its kv_indices are the first rows * capacity entries, which hold every row's
      slots (a decode graph's buffer runs longer);
    - qwen3_5's, on the one position row Qwen3-Next rotates by, given as its temporal, height
      and width rows, which makes its interleaved mRoPE plain NeoX RoPE (row-major, as the
      op reads them; asked with an empty stand-in, a temporary)."""
    capacity = _capacity(context_len)
    if capacity is not None:
        own = {"positions": positions, "kv_indices": kv_indices[: rows * capacity], "max_context": capacity}
        yield qwen3_next_gated_attention_decode_group_fp8_quant, own, lambda: own
    rows_of_three = (3, positions.shape[0])
    stand_in = torch.empty(rows_of_three, dtype=positions.dtype, device=positions.device)
    yield (
        gated_attention_decode_group_fp8_quant,
        {"positions": stand_in, "kv_indices": kv_indices},
        lambda: {"positions": positions.expand(rows_of_three).contiguous(), "kv_indices": kv_indices},
    )


def _capacity(context_len):
    """The least max_context of Qwen3-Next's op at or above ``context_len``, or None."""
    if context_len is None:
        return None
    capacities = QWEN3_NEXT_GATED_ATTENTION_DECODE_GROUP_FP8_QUANT.attrs["max_context"].values
    return min((c for c in capacities if c >= context_len), default=None)


def _attention_falls_back(layer, backend) -> bool:
    """Whether sglang computes other than the ops: LoRA or sequence parallelism; another
    backend or decode path than paged_attention_ragged over one-token pages (MLA, unified or
    sliding-window attention, a 5-D pool, context parallelism, a logit cap); KV scales other
    than 1 (sglang writes at the layer's and reads at 1); another rotary embedding than NeoX
    RoPE on the ops' 64 channels; norm epsilons that differ; or an o_proj that cannot take
    the ops' FP8 groups (_takes_fp8_groups, and no row-wise FP8 copy for its small-M path,
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


def _qkv_stand_in(layer, hidden):
    """An empty stand-in of qkv_proj's output, to ask an op before it is made; passed as a
    temporary, so that the allocator hands its memory on."""
    width = 2 * layer.q_size + 2 * layer.kv_size
    return torch.empty((hidden.shape[0], width), dtype=layer.q_norm.weight.dtype, device=hidden.device)


def declare(plugin: Plugin) -> None:
    plugin.wrap(
        "sglang.srt.models.qwen3_next.Qwen3GatedDeltaNet.forward",
        versions=("0.5.22", "0.5.23"),
        replace=_gdn_forward,
    )
    plugin.wrap(
        "sglang.srt.models.qwen3_next.Qwen3HybridAttentionDecoderLayer.self_attention",
        versions=("0.5.22", "0.5.23"),
        replace=_attention_self_attention,
    )
