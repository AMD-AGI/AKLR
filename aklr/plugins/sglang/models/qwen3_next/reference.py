# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The contracts' computation in float32, between their BF16 and FP8 rounding points, written
from the math and independent of the kernels and of sglang's.

GDN rounding points: the convolution's SiLU to BF16; in prefill, the L2-normalized Q and K
to BF16 too; beta and the core output to BF16; the gated RMS-normalized output to BF16; the
grouped FP8 values to E4M3FN. The gated attention decode reference's rounding points are in its
docstring."""

import itertools

import torch
import torch.nn.functional as F

from aklr.core.contract.reference import OpReference

__all__ = [
    "ROTARY_DIM",
    "Qwen3NextGatedAttentionDecodeGroupFp8QuantReference",
    "Qwen3NextGdnDecodeGroupFp8QuantReference",
    "Qwen3NextGdnPrefillGroupFp8QuantReference",
]

# A full-attention head's rotary part: NeoX RoPE on its first 64 channels
ROTARY_DIM = 64


def _heads(conv_state, delta_state, norm_weight):
    _slots, heads, value_dim, key_dim = delta_state.shape
    key_heads = (conv_state.shape[1] - heads * value_dim) // (2 * key_dim)
    if key_heads < 1 or heads % key_heads or norm_weight.numel() != value_dim:
        raise ValueError("unsupported head dimensions")
    return key_heads, heads, key_dim, value_dim


def _ungroup(projected_qkvz, projected_ba, key_heads, heads, key_dim, value_dim):
    """The convolution input (every Q, every K, every V), Z, B and A of each token, from
    projections grouped by key head: what sglang's fused_qkvzba_split_reshape_cat gives."""
    rows, ratio = projected_qkvz.shape[0], heads // key_heads
    grouped = projected_qkvz.reshape(rows, key_heads, 2 * key_dim + 2 * ratio * value_dim)
    q, k, v, z = grouped.split([key_dim, key_dim, ratio * value_dim, ratio * value_dim], dim=-1)
    mixed = torch.cat([t.reshape(rows, -1) for t in (q, k, v)], dim=-1)
    b, a = projected_ba.reshape(rows, key_heads, 2 * ratio).split([ratio, ratio], dim=-1)
    return mixed, z.reshape(rows, heads, value_dim), b.reshape(rows, heads), a.reshape(rows, heads)


def _convolve(window, conv_weight, conv_bias):
    """SiLU of the bias plus the sum of the taps' products, rounded to BF16;
    ``window`` holds each channel's four inputs, oldest first, in its last dim."""
    return _rounded(F.silu((window * conv_weight).sum(-1) + conv_bias), torch.bfloat16)


def _split_qkv(activated, key_heads, heads, key_dim, value_dim):
    width = key_heads * key_dim
    lead = activated.shape[:-1]
    q = activated[..., :width].reshape(*lead, key_heads, key_dim)
    k = activated[..., width : 2 * width].reshape(*lead, key_heads, key_dim)
    v = activated[..., 2 * width :].reshape(*lead, heads, value_dim)
    return q, k, v


def _l2_normalized(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def _gates(b, a, a_log, dt_bias):
    beta = _rounded(torch.sigmoid(b), torch.bfloat16)
    decay = torch.exp(-torch.exp(a_log) * F.softplus(a + dt_bias))
    return beta, decay


def _normalize_quantize(core, z, norm_weight, eps):
    normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
    normalized = _rounded(normalized * norm_weight * F.silu(z), torch.bfloat16)
    scales = normalized.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (normalized / scales[..., None]).clamp(-448.0, 448.0)
    return normalized, values.reshape(normalized.shape[0], -1), scales


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
    key_heads, heads, key_dim, value_dim = _heads(conv_state, delta_state, norm_weight)
    if conv_state.shape[2] != conv_weight.shape[1] - 1:
        raise ValueError("convolution history and taps disagree")
    rows, slots = projected_qkvz.shape[0], delta_state.shape[0]
    starts = cu_seqlens.cpu().tolist()
    indices = cache_indices.cpu().tolist()
    initial = has_initial_state.cpu().tolist()
    if (
        len(starts) != len(indices) + 1
        or len(initial) != len(indices)
        or starts[0] != 0
        or starts[-1] != rows
        or any(s >= e for s, e in itertools.pairwise(starts))
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= slots for i in indices)
    ):
        raise ValueError("invalid sequence or state-slot metadata")
    mixed, z, b, a = _ungroup(projected_qkvz, projected_ba, key_heads, heads, key_dim, value_dim)
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    activated = torch.empty_like(mixed)
    for seq, slot in enumerate(indices):
        begin, end = starts[seq], starts[seq + 1]
        history = conv_state[slot] if initial[seq] else torch.zeros_like(conv_state[slot])
        inputs = torch.cat((history, mixed[begin:end].T), dim=1)
        activated[begin:end] = _convolve(
            inputs.unfold(1, 4, 1), conv_weight[:, None, :], conv_bias[:, None]
        ).T
        conv_result[slot] = inputs[:, -3:]
    q, k, v = _split_qkv(activated, key_heads, heads, key_dim, value_dim)
    q = _rounded(_l2_normalized(q), torch.bfloat16).repeat_interleave(heads // key_heads, -2)
    k = _rounded(_l2_normalized(k), torch.bfloat16).repeat_interleave(heads // key_heads, -2)
    beta, decay = _gates(b, a, a_log, dt_bias)
    # the recurrence, every sequence at once: step t takes each sequence's t-th token
    lengths = [e - s for s, e in itertools.pairwise(starts)]
    state = torch.stack(
        [
            delta_state[slot] if initial[seq] else torch.zeros_like(delta_state[slot])
            for seq, slot in enumerate(indices)
        ]
    )
    core = mixed.new_empty((rows, heads, value_dim))
    for t in range(max(lengths)):
        live = [seq for seq, n in enumerate(lengths) if n > t]
        token = torch.tensor([starts[seq] + t for seq in live], device=mixed.device)
        s = state[live] * decay[token][:, :, None, None]
        update = (v[token] - (s * k[token][:, :, None, :]).sum(-1)) * beta[token][:, :, None]
        s = s + update[..., None] * k[token][:, :, None, :]
        state[live] = s
        core[token] = _rounded((s * q[token][:, :, None, :]).sum(-1) * scale, torch.bfloat16)
    for seq, slot in enumerate(indices):
        state_result[slot] = state[seq]
    normalized, values, scales = _normalize_quantize(core, z, norm_weight, eps)
    return normalized, conv_result, state_result, values, scales


def _decode(
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
    *,
    scale,
    eps=1e-6,
    **_,
):
    """One token per row; a row with cache index -1 is padding: its core output is zero
    and no state changes."""
    key_heads, heads, key_dim, value_dim = _heads(conv_state, delta_state, norm_weight)
    if conv_state.shape[2] != conv_weight.shape[1] - 1:
        raise ValueError("convolution history and taps disagree")
    rows, slots = projected_qkvz.shape[0], delta_state.shape[0]
    indices = cache_indices.cpu().tolist()
    live = [row for row, i in enumerate(indices) if i != -1]
    slot = [indices[row] for row in live]
    if len(indices) != rows or len(set(slot)) != len(slot) or any(i < -1 or i >= slots for i in indices):
        raise ValueError("invalid state-slot metadata")
    mixed, z, b, a = _ungroup(projected_qkvz, projected_ba, key_heads, heads, key_dim, value_dim)
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    core = mixed.new_zeros((rows, heads, value_dim))
    if live:
        live_rows = torch.tensor(live, device=mixed.device)
        slots_of = torch.tensor(slot, device=mixed.device)
        window = torch.cat((conv_state[slots_of], mixed[live_rows][:, :, None]), dim=2)
        activated = _convolve(window, conv_weight, conv_bias)
        conv_result[slots_of] = window[:, :, 1:]
        q, k, v = _split_qkv(activated, key_heads, heads, key_dim, value_dim)
        ratio = heads // key_heads
        q = (_l2_normalized(q) * scale).repeat_interleave(ratio, -2)
        k = _l2_normalized(k).repeat_interleave(ratio, -2)
        beta, decay = _gates(b[live_rows], a[live_rows], a_log, dt_bias)
        state = delta_state[slots_of] * decay[:, :, None, None]
        update = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
        state = state + update[..., None] * k[:, :, None, :]
        core[live_rows] = _rounded((state * q[:, :, None, :]).sum(-1), torch.bfloat16)
        state_result[slots_of] = state
    normalized, values, scales = _normalize_quantize(core, z, norm_weight, eps)
    return normalized, conv_result, state_result, values, scales


def _floats(call):
    """``call`` with its float tensors in float32."""
    return {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in call.items()}


def _gdn_outputs(call, normalized, conv_state, delta_state, values, scales):
    """The outputs, then both final state pools, each in its dtype."""
    return {
        "normalized": normalized.to(torch.bfloat16),
        "values": values.to(torch.float8_e4m3fn),
        "scales": scales,
        "conv_state": conv_state.to(call["conv_state"].dtype),
        "delta_state": delta_state.to(call["delta_state"].dtype),
    }


class Qwen3NextGdnPrefillGroupFp8QuantReference(OpReference):
    def reference(self, **call):
        """The recurrence (_prefill) in float32; the outputs, then both final state pools."""
        return _gdn_outputs(call, *_prefill(**_floats(call)))


class Qwen3NextGdnDecodeGroupFp8QuantReference(OpReference):
    def reference(self, **call):
        """The decode (_decode) in float32; the outputs, then both final state pools."""
        return _gdn_outputs(call, *_decode(**_floats(call)))


def _rounded(x, dtype):
    """``x`` rounded to ``dtype``, kept in its own float dtype."""
    return x.to(dtype).to(x.dtype)


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


class Qwen3NextGatedAttentionDecodeGroupFp8QuantReference(OpReference):
    def reference(
        self,
        *,
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
        **_,
    ):
        """The contract's decode in float32; the grouped FP8 values and scales, then both pools
        as the call leaves them.

        Rounding points: the normalized and the rotated Q and K to BF16 (the tensors sglang
        keeps between its kernels), the keys and values to E4M3FN in the pools, the attention
        output and the gated output to BF16, the values to E4M3FN. Rows write their slots
        before any row attends.
        max_context bounds how many slots a row attends and does not change the result.
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

        # each row's cosines and sines, at its one position
        angles = cos_sin_cache.float()[positions]
        cos, sin = angles[:, : ROTARY_DIM // 2], angles[:, ROTARY_DIM // 2 : ROTARY_DIM]
        q_weight = q_norm_weight.float()
        k_weight = k_norm_weight.float()
        query = _gemma_norm_rope(query_gate[:, :, 0], q_weight, eps, cos, sin)
        keys = _gemma_norm_rope(keys, k_weight, eps, cos, sin)

        # .float() may return its input itself: the pools are written in copies
        pool_dtype = key_cache.dtype
        key_cache = key_cache.float().clone()
        value_cache = value_cache.float().clone()
        key_cache[cache_locations] = _rounded(keys, torch.float8_e4m3fn)
        value_cache[cache_locations] = _rounded(values, torch.float8_e4m3fn)

        starts = kv_indptr.tolist()
        group = heads // kv_heads
        attention = torch.empty((rows, kv_heads, group, head_dim), dtype=qkv.dtype, device=device)
        for row in range(rows):
            read = kv_indices[starts[row] : starts[row + 1]]
            q = query[row].reshape(kv_heads, group, head_dim)
            scores = torch.einsum("kgd,tkd->kgt", q, key_cache[read]) * scale
            attention[row] = torch.einsum("kgt,tkd->kgd", scores.softmax(-1), value_cache[read])
        attention = _rounded(attention.reshape(rows, heads, head_dim), torch.bfloat16)
        gated = _rounded(attention * torch.sigmoid(query_gate[:, :, 1]), torch.bfloat16)

        groups = gated.reshape(rows, -1, 128)
        scales = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
        quantized = (groups / scales[..., None]).clamp(-448.0, 448.0).reshape(rows, heads, head_dim)
        return {
            "values": quantized.to(torch.float8_e4m3fn),
            "scales": scales,
            "key_cache": key_cache.to(pool_dtype),
            "value_cache": value_cache.to(pool_dtype),
        }
