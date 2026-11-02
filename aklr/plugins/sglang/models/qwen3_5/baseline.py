# Created by OpenAI
"""SGLang 0.5.22 Triton prefill and native block-128 FP8 activation quantization.

Projection GEMMs and the output projection are outside this op on both paths.
Fresh recurrent slots are initialized here to implement the contract independently
of SGLang's allocator. The host maximum gives the convolution its native launch
geometry without a timed device-to-host read.

Decode replays SGLang 0.5.22's ROCm decode kernels between the projections; the
FP8 decode op adds the AITER quantization its output projection would run.
"""


def gdn_prefill_group_fp8_quant_baseline(
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
    max_seq_len=2048,
    scale=128**-0.5,
    eps=1e-6,
    normalized=None,
    values=None,
    scales=None,
):
    import torch
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    _slots, heads, value_dim, key_dim = delta_state.shape
    key_heads = (conv_state.shape[1] - heads * value_dim) // (2 * key_dim)
    rows = projected_qkvz.shape[0]
    mixed, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
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
    q, k, v = fused_qkv_split_gdn_prefill(
        mixed,
        key_heads,
        key_heads,
        heads,
        key_dim,
        key_dim,
        value_dim,
    )
    g, beta = fused_gdn_gating(a_log, a, b, dt_bias)
    # Normal serving allocates fresh slots zeroed. A standalone contract call
    # must give fresh slots the same initial value even with stale pool contents.
    initial = delta_state[cache_indices]
    delta_state[cache_indices] = torch.where(
        has_initial_state[:, None, None, None],
        initial,
        0,
    )
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
    normed, _, _ = _layer_norm_fwd(
        core.reshape(-1, value_dim),
        norm_weight,
        None,
        eps,
        z=z.reshape(-1, value_dim),
        is_rms_norm=True,
        activation="swish",
    )
    normed = normed.reshape(rows, heads, value_dim)
    quantized, factors = per_token_group_quant_fp8(normed.reshape(rows, -1), value_dim)
    return normed, quantized, factors


def _decode_normalized(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale,
    eps,
):
    """SGLang 0.5.22's ROCm decode: unpack, indexed Conv1D update, packed recurrence
    and gated RMSNorm; padding rows (-1) keep their state and give a zero core."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    _slots, heads, value_dim, key_dim = delta_state.shape
    key_heads = (conv_state.shape[1] - heads * value_dim) // (2 * key_dim)
    rows = projected_qkvz.shape[0]
    mixed, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
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
        None,
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
    normed, _, _ = _layer_norm_fwd(
        core.reshape(-1, value_dim),
        norm_weight,
        None,
        eps,
        z=z.reshape(-1, value_dim),
        is_rms_norm=True,
        activation="swish",
    )
    return normed.reshape(rows, heads, value_dim)


def gdn_decode_group_fp8_quant_baseline(
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
    values=None,
    scales=None,
):
    """The decode above, then the per-1x128 AITER quantization that SGLang's block
    FP8 output projection applies to its BF16 input (row-major scales)."""
    import aiter

    normed = _decode_normalized(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        scale,
        eps,
    )
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    return quantize(normed.reshape(normed.shape[0], -1), quant_dtype=aiter.dtypes.fp8)


def gdn_decode_baseline(
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
    normalized=None,
):
    """SGLang 0.5.22's ROCm decode up to the BF16 input of its output projection."""
    return _decode_normalized(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        scale,
        eps,
    )
