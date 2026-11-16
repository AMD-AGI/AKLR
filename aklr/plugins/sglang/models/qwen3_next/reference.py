# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The contracts' computation in float64 between their BF16 and FP8 rounding points,
written from the math and independent of the kernels and of sglang's.

Rounding points: the convolution's SiLU to BF16; in prefill, the L2-normalized Q and K
to BF16 too; beta and the core output to BF16; the gated RMS-normalized output to BF16,
then the per-head E4M3FN values. The convolution sums the bias and the four products
exactly, oldest tap first. The recurrence runs in float64 and the final states round
to FP32.

The gated attention decode reference computes in the float dtype ``decode`` gives, with
the rounding points its docstring lists."""

import inspect
import itertools

import torch
import torch.nn.functional as F

from aklr.plugins.sglang.models.qwen3_5.contract import ROTARY_DIM

__all__ = [
    "qwen3_next_gated_attention_decode_group_fp8_quant_reference",
    "qwen3_next_gdn_decode_group_fp8_quant_reference",
    "qwen3_next_gdn_prefill_group_fp8_quant_reference",
    "reference_decode",
    "reference_prefill",
]


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
    """SiLU of the bias plus the exact sum of the taps' products, rounded to BF16;
    ``window`` holds each channel's four inputs, oldest first, in its last dim."""
    total = (window.double() * conv_weight.double()).sum(-1) + conv_bias.double()
    return F.silu(total).to(torch.bfloat16).double()


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
    beta = torch.sigmoid(b.double()).to(torch.bfloat16).double()
    decay = torch.exp(-torch.exp(a_log.double()) * F.softplus(a.double() + dt_bias.double()))
    return beta, decay


def _normalize_quantize(core, z, norm_weight, eps):
    normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
    normalized = (normalized * norm_weight.double() * F.silu(z.double())).to(torch.bfloat16)
    scales = normalized.double().abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (normalized.double() / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return normalized, values.reshape(normalized.shape[0], -1), scales


def reference_prefill(
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
    activated = torch.empty(mixed.shape, dtype=torch.float64, device=mixed.device)
    for seq, slot in enumerate(indices):
        begin, end = starts[seq], starts[seq + 1]
        history = conv_state[slot] if initial[seq] else torch.zeros_like(conv_state[slot])
        inputs = torch.cat((history, mixed[begin:end].T), dim=1)
        activated[begin:end] = _convolve(
            inputs.unfold(1, 4, 1), conv_weight[:, None, :], conv_bias[:, None]
        ).T
        conv_result[slot] = inputs[:, -3:]
    q, k, v = _split_qkv(activated, key_heads, heads, key_dim, value_dim)
    q = _l2_normalized(q).to(torch.bfloat16).double().repeat_interleave(heads // key_heads, -2)
    k = _l2_normalized(k).to(torch.bfloat16).double().repeat_interleave(heads // key_heads, -2)
    beta, decay = _gates(b, a, a_log, dt_bias)
    # the recurrence, every sequence at once: step t takes each sequence's t-th token
    lengths = [e - s for s, e in itertools.pairwise(starts)]
    state = torch.stack(
        [
            delta_state[slot].double()
            if initial[seq]
            else torch.zeros_like(delta_state[slot], dtype=torch.float64)
            for seq, slot in enumerate(indices)
        ]
    )
    core = torch.empty((rows, heads, value_dim), dtype=torch.float64, device=mixed.device)
    for t in range(max(lengths)):
        live = [seq for seq, n in enumerate(lengths) if n > t]
        token = torch.tensor([starts[seq] + t for seq in live], device=mixed.device)
        s = state[live] * decay[token][:, :, None, None]
        update = (v[token] - (s * k[token][:, :, None, :]).sum(-1)) * beta[token][:, :, None]
        s = s + update[..., None] * k[token][:, :, None, :]
        state[live] = s
        core[token] = ((s * q[token][:, :, None, :]).sum(-1) * scale).to(torch.bfloat16).double()
    for seq, slot in enumerate(indices):
        state_result[slot] = state[seq].float()
    normalized, values, scales = _normalize_quantize(core, z, norm_weight, eps)
    return normalized, conv_result, state_result, values, scales


def reference_decode(
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
    core = torch.zeros((rows, heads, value_dim), dtype=torch.float64, device=mixed.device)
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
        state = delta_state[slots_of].double() * decay[:, :, None, None]
        update = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
        state = state + update[..., None] * k[:, :, None, :]
        core[live_rows] = (state * q[:, :, None, :]).sum(-1).to(torch.bfloat16).double()
        state_result[slots_of] = state.float()
    normalized, values, scales = _normalize_quantize(core, z, norm_weight, eps)
    return normalized, conv_result, state_result, values, scales


def _reference(function, point, row_major_args, decode):
    names = inspect.signature(function).parameters
    normalized, conv, state, values, scales = function(
        **{k: v for k, v in row_major_args.items() if k in names}
    )
    results = {
        "normalized": normalized,
        "values": values,
        "scales": scales,
        "conv_state": conv,
        "delta_state": state,
    }
    return tuple(decode(results[n], point.dtypes[n]) for n in results)


def qwen3_next_gdn_prefill_group_fp8_quant_reference(point, row_major_args, decode):
    """The outputs, then both final state pools."""
    return _reference(reference_prefill, point, row_major_args, decode)


def qwen3_next_gdn_decode_group_fp8_quant_reference(point, row_major_args, decode):
    """The outputs, then both final state pools."""
    return _reference(reference_decode, point, row_major_args, decode)


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


def qwen3_next_gated_attention_decode_group_fp8_quant_reference(point, row_major_args, decode):
    """The contract's decode in ``decode``'s float dtype; the grouped FP8 values (before
    their E4M3FN rounding) and scales, then both pools as the call leaves them.

    Rounding points: the normalized and the rotated Q and K to BF16 (the tensors sglang
    keeps between its kernels), the keys and values to E4M3FN in the pools, the attention
    output and the gated output to BF16. Rows write their slots before any row attends.
    max_context bounds how many slots a row attends and does not change the result.
    """
    args, dtypes = row_major_args, point.dtypes
    qkv_dim, kv_heads, head_dim = (point.axes[n] for n in ("qkv_dim", "n_kv_heads", "head_dim"))
    kv_width = kv_heads * head_dim
    heads = (qkv_dim - 2 * kv_width) // (2 * head_dim)
    width = heads * head_dim
    eps, scale = args["eps"], args["scale"]
    qkv = decode(args["projected_qkv"], dtypes["projected_qkv"])
    rows, device = qkv.shape[0], qkv.device
    query_gate = qkv[:, : 2 * width].reshape(rows, heads, 2, head_dim)
    keys = qkv[:, 2 * width : 2 * width + kv_width].reshape(rows, kv_heads, head_dim)
    values = qkv[:, 2 * width + kv_width :].reshape(rows, kv_heads, head_dim)

    # each row's cosines and sines, at its one position
    angles = decode(args["cos_sin_cache"], dtypes["cos_sin_cache"])[args["positions"]]
    cos, sin = angles[:, : ROTARY_DIM // 2], angles[:, ROTARY_DIM // 2 : ROTARY_DIM]
    q_weight = decode(args["q_norm_weight"], dtypes["q_norm_weight"])
    k_weight = decode(args["k_norm_weight"], dtypes["k_norm_weight"])
    query = _gemma_norm_rope(query_gate[:, :, 0], q_weight, eps, cos, sin)
    keys = _gemma_norm_rope(keys, k_weight, eps, cos, sin)

    # decode may return its input itself: the pools are written in copies
    key_cache = decode(args["key_cache"], dtypes["key_cache"]).clone()
    value_cache = decode(args["value_cache"], dtypes["value_cache"]).clone()
    slots = args["cache_locations"]
    key_cache[slots] = _rounded(keys, torch.float8_e4m3fn)
    value_cache[slots] = _rounded(values, torch.float8_e4m3fn)

    starts = args["kv_indptr"].tolist()
    indices = args["kv_indices"]
    group = heads // kv_heads
    attention = torch.empty((rows, kv_heads, group, head_dim), dtype=qkv.dtype, device=device)
    for row in range(rows):
        read = indices[starts[row] : starts[row + 1]]
        q = query[row].reshape(kv_heads, group, head_dim)
        scores = torch.einsum("kgd,tkd->kgt", q, key_cache[read]) * scale
        attention[row] = torch.einsum("kgt,tkd->kgd", scores.softmax(-1), value_cache[read])
    attention = _rounded(attention.reshape(rows, heads, head_dim), torch.bfloat16)
    gated = _rounded(attention * torch.sigmoid(query_gate[:, :, 1]), torch.bfloat16)

    groups = gated.reshape(rows, -1, 128)
    scales = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
    quantized = (groups / scales[..., None]).clamp(-448.0, 448.0).reshape(rows, heads, head_dim)
    return quantized, scales, key_cache, value_cache
