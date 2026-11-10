# Created by OpenAI
"""SGLang 0.5.22 Triton prefill and native block-128 FP8 activation quantization.

Projection GEMMs and the output projection are outside this op on both paths.
Fresh recurrent slots are initialized here to implement the contract independently
of SGLang's allocator. The host maximum gives the convolution its native launch
geometry without a timed device-to-host read.

Decode replays SGLang 0.5.22's ROCm decode kernels between the projections; the
FP8 decode op adds the AITER quantization its output projection would run.

The gated attention decode replays what Qwen3_5AttentionDecoderLayer.self_attention runs
between qkv_proj and o_proj on HIP with an FP8 KV cache on the aiter backend; the FP8 op
adds the AITER quantization of the block-FP8 o_proj, the BF16 op ends at o_proj's BF16
input.
"""

import functools

from aklr.plugins.sglang.models.qwen3_5.contract import MROPE_SECTION, ROTARY_DIM

# sglang's aiter backend sizes its decode partitions for the served context
# (--context-length): 131072 here, half of Qwen3.5's 262144-token default, which launches
# fewer of them
_SERVED_CONTEXT = 131072
_PARTITION_SIZE = 256  # sglang's _AITER_PARTITION_SIZE_ROCM


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


@functools.cache
def _unit_scale(device):
    """The aiter backend's KV scale tensor, 1.0, which an FP8 checkpoint's own unit
    KV scales equal."""
    import torch

    return torch.ones(1, dtype=torch.float32, device=device)


@functools.cache
def _last_page_lengths(rows, device):
    """The aiter backend's kv_last_page_len: one-token pages."""
    import torch

    return torch.ones(rows, dtype=torch.int32, device=device)


def _gated_attention(
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
    eps,
    scale,
):
    """forward_prepare_fused_gate's fused Gemma QK norm and gate split, the rotary
    module's Triton mRoPE, the aiter backend's forward_decode (its fused FP8 KV write,
    then paged_attention_ragged over one-token pages at unit KV scales) and
    fused_sigmoid_mul: the BF16 rows sglang hands o_proj."""
    import aiter
    import torch
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash
    from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate

    rows, device = projected_qkv.shape[0], projected_qkv.device
    _, kv_heads, head_dim = key_cache.shape
    heads = (projected_qkv.shape[1] - 2 * kv_heads * head_dim) // (2 * head_dim)
    width, kv_width = heads * head_dim, kv_heads * head_dim
    query_gate, key, value = projected_qkv.split([2 * width, kv_width, kv_width], dim=-1)
    query, key, gate = fused_qk_gemma_rmsnorm_with_gate(
        query_gate, key, q_norm_weight, k_norm_weight, eps, head_dim, heads
    )
    query, key, gate = (t.view(rows, -1) for t in (query, key, gate))
    triton_mrope_fused(
        query,
        key,
        cos_sin_cache,
        positions,
        list(MROPE_SECTION),
        head_dim,
        ROTARY_DIM,
        True,
        False,
        True,
        None,
    )
    unit = _unit_scale(device)
    key_pages = key_cache.view(-1, 1, kv_heads, head_dim)
    value_pages = value_cache.view(-1, 1, kv_heads, head_dim)
    launch_reshape_and_cache_flash(
        key.view(-1, kv_heads, head_dim),
        value.view(-1, kv_heads, head_dim),
        key_pages,
        value_pages,
        cache_locations,
        k_scale=unit,
        v_scale=unit,
    )
    partitions = -(-_SERVED_CONTEXT // _PARTITION_SIZE)
    # the backend's float32 workspace: partial outputs, then exp sums and max logits
    workspace = torch.empty(rows * heads * partitions * (head_dim + 2) * 4, dtype=torch.uint8, device=device)
    attended = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    aiter.paged_attention_ragged(
        attended.view(-1, heads, head_dim),
        workspace,
        query.view(-1, heads, head_dim),
        key_pages,
        value_pages,
        scale,
        kv_indptr,
        kv_indices,
        _last_page_lengths(rows, device),
        1,
        partitions,
        None,
        "fp8_e4m3",
        "NHD",
        0.0,
        unit,
        unit,
        None,
        _PARTITION_SIZE,
    )
    fused_sigmoid_mul(attended, gate, inplace=True)
    return attended


def gated_attention_decode_group_fp8_quant_baseline(
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
    values=None,
    scales=None,
):
    """The gated attention above, then the per-1x128 AITER quantization the block-FP8
    o_proj applies to its BF16 input (row-major scales)."""
    import aiter

    attended = _gated_attention(
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
        eps,
        scale,
    )
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    quantized, factors = quantize(attended, quant_dtype=aiter.dtypes.fp8)
    return quantized.view(attended.shape[0], -1, key_cache.shape[-1]), factors


def gated_attention_decode_baseline(
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
    gated=None,
):
    """The gated attention above: sglang's BF16 input to an o_proj that takes BF16."""
    return _gated_attention(
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
        eps,
        scale,
    )
