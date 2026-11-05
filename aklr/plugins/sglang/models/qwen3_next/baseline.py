# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.22 runs on ROCm between Qwen3GatedDeltaNet's input projections and
out_proj: the grouped Q/K/V/Z and B/A unpack, the indexed convolution, the gated delta
recurrence (packed decode, or chunked prefill), FusedRMSNormGated, then the AITER
per-1x128 FP8 quantization that the block-FP8 out_proj applies to its BF16 input.

The prefill gives fresh slots a zero state itself, to implement the contract
independently of sglang's allocator, and hands the convolution the host maximum as its
launch geometry, as sglang's own lengths give it, without a timed device-to-host read."""


def _heads(conv_state, delta_state):
    _slots, heads, value_dim, key_dim = delta_state.shape
    key_heads = (conv_state.shape[1] - heads * value_dim) // (2 * key_dim)
    return key_heads, heads, key_dim, value_dim


def _normalize_quantize(core, z, norm_weight, eps, rows):
    """FusedRMSNormGated's rms_norm_gated, then out_proj's per-1x128 FP8 quantization."""
    import aiter
    from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated

    heads, value_dim = z.shape[-2:]
    normed = rms_norm_gated(
        core.reshape(-1, value_dim), z.reshape(-1, value_dim), norm_weight, None, "silu", eps=eps
    )
    normed = normed.reshape(rows, heads, value_dim)
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    values, scales = quantize(normed.reshape(rows, -1), quant_dtype=aiter.dtypes.fp8)
    return normed, values, scales


def qwen3_next_gdn_prefill_group_fp8_quant_baseline(
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
    normalized=None,
    values=None,
    scales=None,
):
    import torch
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    key_heads, heads, key_dim, value_dim = _heads(conv_state, delta_state)
    rows = projected_qkvz.shape[0]
    mixed, z, b, a = fused_qkvzba_split_reshape_cat(
        projected_qkvz,
        projected_ba,
        key_heads,
        heads,
        key_dim,
        value_dim,
    )
    mixed = causal_conv1d_fn(
        mixed.transpose(0, 1),
        conv_weight,
        conv_bias,
        conv_states=conv_state,
        query_start_loc=cu_seqlens,
        seq_lens_cpu=[max_seq_len] * cache_indices.numel(),
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    ).transpose(0, 1)[:rows]
    q, k, v = fused_qkv_split_gdn_prefill(mixed, key_heads, key_heads, heads, key_dim, key_dim, value_dim)
    g, beta = fused_gdn_gating(a_log, a, b, dt_bias)
    # sglang allocates fresh slots zeroed; a standalone call must not see stale state
    initial = delta_state[cache_indices]
    delta_state[cache_indices] = torch.where(has_initial_state[:, None, None, None], initial, 0)
    core, _, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=delta_state,
        initial_state_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        inplace_update=True,
    )
    return _normalize_quantize(core, z, norm_weight, eps, rows)


def qwen3_next_gdn_decode_group_fp8_quant_baseline(
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
    normalized=None,
    values=None,
    scales=None,
):
    """Padding rows (-1) keep their state and give a zero core, as in sglang."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkvzba_split_reshape_cat
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    key_heads, heads, key_dim, value_dim = _heads(conv_state, delta_state)
    rows = projected_qkvz.shape[0]
    mixed, z, b, a = fused_qkvzba_split_reshape_cat(
        projected_qkvz,
        projected_ba,
        key_heads,
        heads,
        key_dim,
        value_dim,
    )
    mixed = causal_conv1d_update(
        mixed,
        conv_state,
        conv_weight,
        conv_bias,
        "silu",
        conv_state_indices=cache_indices,
    )
    core = mixed.new_empty(rows, 1, heads, value_dim)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=delta_state,
        out=core,
        ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return _normalize_quantize(core, z, norm_weight, eps, rows)
