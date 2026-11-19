# Created by OpenAI
"""SGLang 0.5.23 Triton prefill and native block-128 FP8 activation quantization.

Projection GEMMs and the output projection are outside this op on both paths; the BF16
prefill op ends at the unquantized output projection's BF16 input.
Fresh recurrent slots are initialized here to implement the contract independently
of SGLang's allocator. The host maximum gives the convolution its native launch
geometry without a timed device-to-host read.

Decode replays SGLang 0.5.23's ROCm decode kernels between the projections; the
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


def _prefill_normalized(
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
    max_seq_len,
    scale,
    eps,
):
    """SGLang 0.5.23's Triton prefill up to the BF16 input of its output projection."""
    import torch
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
    from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

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
    return normed.reshape(rows, heads, value_dim)


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
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    normed = _prefill_normalized(
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
        max_seq_len,
        scale,
        eps,
    )
    rows, _, value_dim = normed.shape
    quantized, factors = per_token_group_quant_fp8(normed.reshape(rows, -1), value_dim)
    return normed, quantized, factors


def gdn_prefill_baseline(
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
):
    """The prefill above, ending at the BF16 input of an unquantized out_proj."""
    return _prefill_normalized(
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
        max_seq_len,
        scale,
        eps,
    )


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
    """SGLang 0.5.23's ROCm decode: unpack, indexed Conv1D update, packed recurrence
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
    """SGLang 0.5.23's ROCm decode up to the BF16 input of its output projection."""
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


def _verify_core(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    inter_indices,
    inter_ssm,
    inter_conv,
    conv_weight,
    a_log,
    dt_bias,
    scale,
    draft_tokens,
):
    """SGLang 0.5.23's ROCm chain verify up to its gated norm: unpack, the indexed conv update
    writing the deduplicated window through its overlapping view, and the recurrence writing
    each draft token's state (gdn_backend.forward_extend, TritonGDNKernel.target_verify), without
    the index comparison select_verify_intermediate_state_indices launches for nothing. Returns
    the core and Z."""
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    _slots, heads, value_dim, key_dim = delta_state.shape
    key_heads = (conv_state.shape[1] - heads * value_dim) // (2 * key_dim)
    rows, sequences = projected_qkvz.shape[0], cache_indices.shape[0]
    mixed, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        projected_qkvz, projected_ba, key_heads, heads, key_dim, value_dim
    )
    window_rows, channels, width = inter_conv.shape
    window = inter_conv.as_strided(
        (window_rows, draft_tokens, channels, width - draft_tokens + 1), (channels * width, 1, width, 1)
    )
    mixed = causal_conv1d_update(
        mixed.view(sequences, draft_tokens, -1).transpose(1, 2),
        conv_state,
        conv_weight,
        None,
        "silu",
        conv_state_indices=cache_indices,
        intermediate_conv_window=window,
        intermediate_state_indices=inter_indices,
    )
    mixed = mixed.transpose(1, 2).reshape(rows, -1)
    q, k, v = mixed.split([key_heads * key_dim, key_heads * key_dim, heads * value_dim], dim=-1)
    # sglang's backend sizes padding requests (-1) at zero tokens; this one runs them on a zero state
    core = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.view(1, rows, key_heads, key_dim),
        k=k.view(1, rows, key_heads, key_dim),
        v=v.view(1, rows, heads, value_dim),
        b=b,
        initial_state_source=delta_state,
        initial_state_indices=cache_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=_verify_starts(sequences, draft_tokens, projected_qkvz.device),
        disable_state_update=True,
        intermediate_states_buffer=inter_ssm,
        intermediate_state_indices=inter_indices,
        cache_steps=draft_tokens,
    )
    return core.reshape(-1, value_dim), z.reshape(-1, value_dim)


@functools.cache
def _verify_starts(sequences, draft_tokens, device):
    """The backend's query_start_loc: draft_tokens rows a request."""
    import torch

    return torch.arange(0, (sequences + 1) * draft_tokens, draft_tokens, dtype=torch.int32, device=device)


def _verify_normalized(norm_weight, eps, core, z, rows, **quant):
    from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd

    normed, _, _ = _layer_norm_fwd(
        core, norm_weight, None, eps, z=z, is_rms_norm=True, activation="swish", **quant
    )
    return normed if quant else normed.reshape(rows, -1, core.shape[-1])


def gdn_verify_group_fp8_quant_baseline(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    inter_indices,
    inter_ssm,
    inter_conv,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
    draft_tokens=4,
    projection_storage="row_major",
    values=None,
    scales=None,
):
    """The verify above, gated RMSNorm (RMSNormGated), then the per-1x128 AITER quantization
    that SGLang's block FP8 output projection applies to its BF16 input."""
    import aiter

    core, z = _verify_core(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        inter_indices,
        inter_ssm,
        inter_conv,
        conv_weight,
        a_log,
        dt_bias,
        scale,
        draft_tokens,
    )
    rows = projected_qkvz.shape[0]
    normed = _verify_normalized(norm_weight, eps, core, z, rows)
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    return quantize(normed.reshape(rows, -1), quant_dtype=aiter.dtypes.fp8)


def gdn_verify_token_fp8_quant_baseline(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    inter_indices,
    inter_ssm,
    inter_conv,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
    draft_tokens=4,
    projection_storage="row_major",
    values=None,
    scales=None,
):
    """The verify above, then the gated RMSNorm that quantizes each row to FP8 for a Quark
    per-token FP8 out_proj (Qwen3_5GatedDeltaNet.forward's quant_heads path)."""
    core, z = _verify_core(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        inter_indices,
        inter_ssm,
        inter_conv,
        conv_weight,
        a_log,
        dt_bias,
        scale,
        draft_tokens,
    )
    heads = delta_state.shape[1]
    return _verify_normalized(norm_weight, eps, core, z, projected_qkvz.shape[0], quant_heads=heads)


def gdn_verify_baseline(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    inter_indices,
    inter_ssm,
    inter_conv,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
    draft_tokens=4,
    projection_storage="row_major",
    normalized=None,
):
    """The verify above up to the BF16 input of its output projection."""
    core, z = _verify_core(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        inter_indices,
        inter_ssm,
        inter_conv,
        conv_weight,
        a_log,
        dt_bias,
        scale,
        draft_tokens,
    )
    return _verify_normalized(norm_weight, eps, core, z, projected_qkvz.shape[0])


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


def _prepare(
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    eps,
):
    """forward_prepare_fused_gate's fused Gemma QK norm and gate split, the rotary module's
    Triton mRoPE, and the aiter backend's fused FP8 KV write at unit KV scales: the query,
    key, value and gate rows the attention reads."""
    from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
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
    return query, key, value, gate


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
    """_prepare, the aiter backend's forward_decode (paged_attention_ragged over one-token
    pages at unit KV scales) and fused_sigmoid_mul: the BF16 rows sglang hands o_proj."""
    import aiter
    import torch
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    query, _, _, gate = _prepare(
        projected_qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        eps,
    )
    rows, device = projected_qkv.shape[0], projected_qkv.device
    _, kv_heads, head_dim = key_cache.shape
    heads = query.shape[1] // head_dim
    width = heads * head_dim
    key_pages = key_cache.view(-1, 1, kv_heads, head_dim)
    value_pages = value_cache.view(-1, 1, kv_heads, head_dim)
    unit = _unit_scale(device)
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


def _gated_attention_prefill(
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    cu_seqlens,
    eps,
    scale,
    max_seq_len,
):
    """_prepare, the aiter backend's forward_extend without a prefix (gfx950, head width 256,
    FP8 pool: flash_attn_varlen_fp8_pertensor_func on Q, K and V cast to E4M3FN at unit
    scales) and fused_sigmoid_mul: the BF16 rows sglang hands o_proj."""
    import torch
    from aiter import flash_attn_varlen_fp8_pertensor_func
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    query, key, value, gate = _prepare(
        projected_qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        eps,
    )
    rows, device = projected_qkv.shape[0], projected_qkv.device
    _, kv_heads, head_dim = key_cache.shape
    unit = _unit_scale(device)
    fp8 = torch.float8_e4m3fn
    attended = flash_attn_varlen_fp8_pertensor_func(
        query.contiguous().view(rows, -1, head_dim).to(fp8),
        key.contiguous().view(rows, kv_heads, head_dim).to(fp8),
        value.contiguous().view(rows, kv_heads, head_dim).to(fp8),
        unit,
        unit,
        unit,
        cu_seqlens,
        cu_seqlens,
        max_seq_len,
        max_seq_len,
        softmax_scale=scale,
        causal=True,
    )
    attended = attended.to(torch.bfloat16).view(rows, -1)
    fused_sigmoid_mul(attended, gate, inplace=True)
    return attended


def gated_attention_prefill_group_fp8_quant_baseline(
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    cu_seqlens,
    eps=1e-6,
    scale=256**-0.5,
    max_seq_len=1024,
    values=None,
    scales=None,
):
    """The gated prefill above, then the per-1x128 AITER quantization of the block-FP8
    o_proj (row-major scales)."""
    import aiter

    attended = _gated_attention_prefill(
        projected_qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        cu_seqlens,
        eps,
        scale,
        max_seq_len,
    )
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    quantized, factors = quantize(attended, quant_dtype=aiter.dtypes.fp8)
    return quantized.view(attended.shape[0], -1, key_cache.shape[-1]), factors


def gated_attention_prefill_baseline(
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    cu_seqlens,
    eps=1e-6,
    scale=256**-0.5,
    max_seq_len=1024,
    gated=None,
):
    """The gated prefill above: sglang's BF16 input to an o_proj that takes BF16."""
    return _gated_attention_prefill(
        projected_qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        cu_seqlens,
        eps,
        scale,
        max_seq_len,
    )
