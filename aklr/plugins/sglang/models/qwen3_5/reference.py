# Created by OpenAI
"""Sequential mathematical oracle; deliberately independent of the Gluon kernels.

Inputs use contiguous Q/K/V/Z and B/A projections. An FP32 recurrence provides the
numerical oracle; BF16 interfaces and the FP8 quantization boundary are explicit.
This implementation is for correctness and must not be used as a speed baseline.
"""

import itertools

import torch
import torch.nn.functional as F

from aklr.core.contract.reference import OpReference

__all__ = [
    "MROPE_SECTION",
    "ROTARY_DIM",
    "GatedAttentionGroupFp8QuantReference",
    "GatedAttentionReference",
    "GdnDecodeGroupFp8QuantReference",
    "GdnDecodeReference",
    "GdnPrefillGroupFp8QuantReference",
    "GdnPrefillReference",
]

# A full-attention head's rotary part: NeoX RoPE on its first 64 channels, whose 32
# frequencies Qwen3.5's interleaved mRoPE splits over the temporal, height and width positions
ROTARY_DIM = 64
MROPE_SECTION = (11, 11, 10)


def _prefill(
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
    *,
    scale,
    eps=1e-6,
    **_,
):
    slots, heads, value_dim, key_dim = delta_state.shape
    channels = conv_state.shape[1]
    key_heads = (channels - heads * value_dim) // (2 * key_dim)
    if heads % key_heads or norm_weight.numel() != value_dim:
        raise ValueError("unsupported head dimensions")
    rows = projected_qkvz.shape[0]
    starts = cu_seqlens.cpu().tolist()
    indices = cache_indices.cpu().tolist()
    initial = has_initial_state.cpu().tolist()
    if (
        len(starts) != len(indices) + 1
        or len(initial) != len(indices)
        or starts[0] != 0
        or starts[-1] != rows
        or any(a >= b for a, b in itertools.pairwise(starts))
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= slots for i in indices)
    ):
        raise ValueError("invalid sequence or state-slot metadata")
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    output = torch.empty((rows, heads, value_dim), dtype=torch.bfloat16, device=projected_qkvz.device)
    ratio = heads // key_heads
    q_width, k_width = key_heads * key_dim, key_heads * key_dim
    taps = conv_weight.shape[1]
    if conv_state.shape[2] != taps - 1:
        raise ValueError("convolution history and taps disagree")
    for seq, slot in enumerate(indices):
        history = conv_state[slot].clone() if initial[seq] else torch.zeros_like(conv_state[slot])
        state = (
            delta_state[slot].float()
            if initial[seq]
            else torch.zeros_like(delta_state[slot], dtype=torch.float32)
        )
        for token in range(starts[seq], starts[seq + 1]):
            raw = projected_qkvz[token, :channels]
            window = torch.cat((history, raw[:, None]), dim=1)
            conv = (window.float() * conv_weight.float()).sum(dim=1) + conv_bias.float()
            activated = F.silu(conv).to(torch.bfloat16)
            history = window[:, 1:]
            q = activated[:q_width].reshape(key_heads, key_dim).float()
            k = activated[q_width : q_width + k_width].reshape(key_heads, key_dim).float()
            q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
            k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
            q, k = q.repeat_interleave(ratio, 0), k.repeat_interleave(ratio, 0)
            v = activated[q_width + k_width :].reshape(heads, value_dim).float()
            b, a = projected_ba[token].float().split(heads)
            beta = b.sigmoid().to(torch.bfloat16).float()
            decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(a + dt_bias.float()))
            state = state * decay[:, None, None]
            prediction = (state * k[:, None, :]).sum(dim=-1)
            update = (v - prediction) * beta[:, None]
            state = state + update[:, :, None] * k[:, None, :]
            core = ((state * q[:, None, :]).sum(dim=-1) * scale).to(torch.bfloat16).float()
            gate = projected_qkvz[token, channels:].reshape(heads, value_dim).float()
            normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
            output[token] = (normalized * norm_weight.float() * F.silu(gate)).to(torch.bfloat16)
        conv_result[slot] = history
        state_result[slot] = state.float()
    scales = output.float().abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (output.float() / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return output, conv_result, state_result, values.reshape(rows, -1), scales


def _decode(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    *,
    scale,
    eps=1e-6,
    **_,
):
    """One decode token per row, in float32 between the BF16 and FP8 rounding points.

    Rounding points: each convolution tap's product to BF16, summed in FP32 oldest
    tap first (as SGLang's decode convolution does); the SiLU convolution output, beta
    and the core output to BF16 (the tensors SGLang keeps between its decode
    kernels); the gated RMS-normalized output to BF16, then the per-head E4M3FN
    values. A row with cache index -1 is padding: its core output is zero and no
    state changes.
    """
    slots, heads, value_dim, key_dim = delta_state.shape
    channels = conv_state.shape[1]
    key_heads = (channels - heads * value_dim) // (2 * key_dim)
    if heads % key_heads or norm_weight.numel() != value_dim:
        raise ValueError("unsupported head dimensions")
    if conv_state.shape[2] != conv_weight.shape[1] - 1:
        raise ValueError("convolution history and taps disagree")
    rows = projected_qkvz.shape[0]
    indices = cache_indices.tolist()
    live = [i for i in indices if i != -1]
    if len(indices) != rows or len(set(live)) != len(live) or any(i < -1 or i >= slots for i in indices):
        raise ValueError("invalid state-slot metadata")
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    core = torch.zeros((rows, heads, value_dim), dtype=torch.float32, device=projected_qkvz.device)
    ratio, width = heads // key_heads, key_heads * key_dim
    for row, slot in enumerate(indices):
        if slot == -1:
            continue
        window = torch.cat((conv_state[slot], projected_qkvz[row, :channels, None]), 1)
        # each tap's product rounds to BF16; FP32 sums them, the oldest tap first
        taps = (window.float() * conv_weight.float()).to(torch.bfloat16).float()
        conv = ((taps[:, 0] + taps[:, 1]) + taps[:, 2]) + taps[:, 3]
        activated = F.silu(conv.float()).to(torch.bfloat16).float()
        conv_result[slot] = window[:, 1:]
        q = activated[:width].reshape(key_heads, key_dim)
        k = activated[width : 2 * width].reshape(key_heads, key_dim)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q, k = q.repeat_interleave(ratio, 0), k.repeat_interleave(ratio, 0)
        v = activated[2 * width :].reshape(heads, value_dim)
        b, a = projected_ba[row].float().split(heads)
        beta = b.sigmoid().to(torch.bfloat16).float()
        decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(a + dt_bias.float()))
        state = delta_state[slot].float() * decay[:, None, None]
        update = (v - (state * k[:, None, :]).sum(dim=-1)) * beta[:, None]
        state = state + update[:, :, None] * k[:, None, :]
        core[row] = (state * q[:, None, :]).sum(dim=-1).to(torch.bfloat16).float()
        state_result[slot] = state.float()
    gate = projected_qkvz[:, channels:].reshape(rows, heads, value_dim).float()
    normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
    normalized = (normalized * norm_weight.float() * F.silu(gate)).to(torch.bfloat16)
    scales = normalized.float().abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (normalized.float() / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return normalized, conv_result, state_result, values.reshape(rows, -1), scales


class GdnPrefillGroupFp8QuantReference(OpReference):
    def reference(self, **call):
        """The recurrence (_prefill); the BF16 output and its FP8 values and scales, then both
        final state pools."""
        normalized, conv_state, delta_state, values, scales = _prefill(**call)
        return {
            "normalized": normalized,
            "values": values,
            "scales": scales,
            "conv_state": conv_state,
            "delta_state": delta_state,
        }


class GdnPrefillReference(OpReference):
    def reference(self, **call):
        """The recurrence (_prefill); the BF16 output, then both final state pools."""
        normalized, conv_state, delta_state, _, _ = _prefill(**call)
        return {"normalized": normalized, "conv_state": conv_state, "delta_state": delta_state}


class GdnDecodeGroupFp8QuantReference(OpReference):
    def reference(self, **call):
        """The decode (_decode); the FP8 values and scales, then both final state pools."""
        _, conv_state, delta_state, values, scales = _decode(**call)
        return {"values": values, "scales": scales, "conv_state": conv_state, "delta_state": delta_state}


class GdnDecodeReference(OpReference):
    def reference(self, **call):
        """The decode (_decode); the BF16 output, then both final state pools."""
        normalized, conv_state, delta_state, _, _ = _decode(**call)
        return {"normalized": normalized, "conv_state": conv_state, "delta_state": delta_state}


def _rounded(x, dtype):
    """``x`` rounded to ``dtype``, kept in its own float dtype."""
    return x.to(dtype).to(x.dtype)


def _mrope_axes(device):
    """Which of positions' rows each rotary frequency reads: interleaved mRoPE gives
    frequency f the height row when f % 3 == 1 and f < 3 * sections[1], the width row
    when f % 3 == 2 and f < 3 * sections[2], else the temporal row."""
    _, height, width = MROPE_SECTION
    axes = [
        1 if f % 3 == 1 and f < 3 * height else 2 if f % 3 == 2 and f < 3 * width else 0
        for f in range(ROTARY_DIM // 2)
    ]
    return torch.tensor(axes, device=device)


def _gemma_norm_rope(x, weight, eps, cos, sin):
    """Per-head RMSNorm with weight 1 + weight, rounded to BF16, then NeoX RoPE on the
    first ROTARY_DIM channels, rounded to BF16; ``x`` (rows, heads, head_dim), ``cos`` and
    ``sin`` (rows, ROTARY_DIM // 2)."""
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + weight)
    normalized = _rounded(normalized, torch.bfloat16)
    half = ROTARY_DIM // 2
    first, second = normalized[..., :half], normalized[..., half:ROTARY_DIM]
    cos, sin = cos[:, None, :], sin[:, None, :]
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((_rounded(rotated, torch.bfloat16), normalized[..., ROTARY_DIM:]), dim=-1)


# prefill query rows a reference step attends at once: bounds its FP32 score block
_QUERY_BLOCK = 256


def _gated_attention(
    *,
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    cache_locations,
    key_cache,
    value_cache,
    eps,
    scale,
    kv_indptr=None,
    kv_indices=None,
    cu_seqlens=None,
    **_,
):
    """The gated attention (decode, or prefill without kv_indptr) in float32: each query
    head's attention output rounded to BF16 times sigmoid of its gate, before that product's
    own rounding, as (rows, heads, head_dim); then both pools as the call leaves them.

    Rounding points: the normalized and the rotated Q and K to BF16 (the tensors sglang
    keeps between its kernels), the keys and values to E4M3FN in the pools, and the
    attention output to BF16. Rows write their slots before any row attends.
    """
    qkv_dim, (kv_heads, head_dim) = projected_qkv.shape[1], key_cache.shape[1:]
    kv_width = kv_heads * head_dim
    heads = (qkv_dim - 2 * kv_width) // (2 * head_dim)
    width = heads * head_dim
    qkv = projected_qkv.float()
    rows, device = qkv.shape[0], qkv.device
    query_gate = qkv[:, : 2 * width].reshape(rows, heads, 2, head_dim)
    keys = qkv[:, 2 * width : 2 * width + kv_width].reshape(rows, kv_heads, head_dim)
    values = qkv[:, 2 * width + kv_width :].reshape(rows, kv_heads, head_dim)

    table = cos_sin_cache.float()
    frequencies = torch.arange(ROTARY_DIM // 2, device=device)
    rotary_positions = positions.index_select(0, _mrope_axes(device)).T
    cos = table[rotary_positions, frequencies]
    sin = table[rotary_positions, frequencies + ROTARY_DIM // 2]
    query = _gemma_norm_rope(query_gate[:, :, 0], q_norm_weight.float(), eps, cos, sin)
    keys = _gemma_norm_rope(keys, k_norm_weight.float(), eps, cos, sin)

    # .float() may return its input itself: the pools are written in copies
    pool_dtype = key_cache.dtype
    key_cache = key_cache.float().clone()
    value_cache = value_cache.float().clone()
    key_cache[cache_locations] = _rounded(keys, torch.float8_e4m3fn)
    value_cache[cache_locations] = _rounded(values, torch.float8_e4m3fn)

    group = heads // kv_heads
    attention = torch.empty((rows, kv_heads, group, head_dim), dtype=qkv.dtype, device=device)
    if kv_indptr is not None:
        starts, indices = kv_indptr.tolist(), kv_indices
        for row in range(rows):
            read = indices[starts[row] : starts[row + 1]]
            q = query[row].reshape(kv_heads, group, head_dim)
            scores = torch.einsum("kgd,tkd->kgt", q, key_cache[read]) * scale
            attention[row] = torch.einsum("kgt,tkd->kgd", scores.softmax(-1), value_cache[read])
    else:
        # prefill: each row reads its sequence's slots up to its own, in blocks of query rows
        starts = cu_seqlens.tolist()
        for begin, end in itertools.pairwise(starts):
            read = cache_locations[begin:end]
            keys, values = key_cache[read], value_cache[read]
            for first in range(begin, end, _QUERY_BLOCK):
                last = min(first + _QUERY_BLOCK, end)
                q = query[first:last].reshape(-1, kv_heads, group, head_dim)
                scores = torch.einsum("rkgd,tkd->rkgt", q, keys[: last - begin]) * scale
                causal = torch.arange(first - begin, last - begin, device=device)[:, None] >= torch.arange(
                    last - begin, device=device
                )
                scores = scores.masked_fill(~causal[:, None, None, :], float("-inf"))
                attention[first:last] = torch.einsum(
                    "rkgt,tkd->rkgd", scores.softmax(-1), values[: last - begin]
                )
    attention = _rounded(attention.reshape(rows, heads, head_dim), torch.bfloat16)
    return (
        attention * torch.sigmoid(query_gate[:, :, 1]),
        key_cache.to(pool_dtype),
        value_cache.to(pool_dtype),
    )


class GatedAttentionGroupFp8QuantReference(OpReference):
    def reference(self, **call):
        """The gated attention (_gated_attention), decode or prefill, in float32; the grouped FP8
        values and scales, then both pools as the call leaves them.

        Rounding points: the normalized and the rotated Q and K to BF16 (the tensors sglang
        keeps between its kernels), the keys and values to E4M3FN in the pools, the attention
        output and the gated output to BF16, the values to E4M3FN. Rows write their slots before
        any row attends.
        """
        product, key_cache, value_cache = _gated_attention(**call)
        gated = _rounded(product, torch.bfloat16)
        rows = gated.shape[0]
        groups = gated.reshape(rows, -1, 128)
        scales = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
        quantized = (groups / scales[..., None]).clamp(-448.0, 448.0).reshape(gated.shape)
        return {
            "values": quantized.to(torch.float8_e4m3fn),
            "scales": scales,
            "key_cache": key_cache,
            "value_cache": value_cache,
        }


class GatedAttentionReference(OpReference):
    def reference(self, **call):
        """The gated attention (_gated_attention), decode or prefill, in float32; the gated output,
        each row's heads side by side, then both pools as the call leaves them.

        Rounding points: the normalized and the rotated Q and K to BF16 (the tensors sglang
        keeps between its kernels), the keys and values to E4M3FN in the pools, the attention
        output and the gated output to BF16. Rows write their slots before any row attends.
        """
        product, key_cache, value_cache = _gated_attention(**call)
        return {
            "gated": product.reshape(product.shape[0], -1).to(torch.bfloat16),
            "key_cache": key_cache,
            "value_cache": value_cache,
        }
