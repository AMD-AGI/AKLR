# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA from its recurrence, token by token, in the dtype ``decode`` gives, in the order of sglang's recurrent kernel
(q and k normalized, q scaled; the state decayed, then the delta, then the output); no rounding is part of the op
(sglang's chunk_kda rounds the normalized q and k to bf16, an error left to compare). A prefill walks all its
sequences side by side, one token of each a step."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.sglang.layers.kda_triton.contract import L2_NORM_EPS, TRACK_CHUNK

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "kimi_k3_sgl_kda_decode_reference",
    "kimi_k3_sgl_kda_prefill_reference",
    "kimi_k3_sgl_kda_prefill_track_reference",
]


def _operands(point, args, decode):
    """Each token's normalized and scaled q, normalized k, v, decay exp(g) along k and beta, (n_tokens, heads, dim)
    and (n_tokens, heads); the state cache decoded, a copy, and its (slot, head, v, k) view."""
    import torch

    dtypes = point.dtypes
    heads, dim = point.axes["n_heads"], point.axes["head_dim"]
    q, k, v = decode(args["qkv"], dtypes["qkv"]).unflatten(1, (3, heads)).unbind(1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * args["scale"]
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    a = decode(args["a_log"], dtypes["a_log"]).exp()[:, None]
    gate = args["lower_bound"] * torch.sigmoid(
        a * (decode(args["raw_g"], dtypes["raw_g"]) + decode(args["g_bias"], dtypes["g_bias"]))
    )
    beta = torch.sigmoid(decode(args["raw_beta"], dtypes["raw_beta"])[..., 0])
    state_cache = decode(args["state_cache"], dtypes["state_cache"]).clone()
    return q, k, v, gate.exp(), beta, state_cache, state_cache.view(-1, heads, dim, dim)


def _step(state, q, k, v, decay, beta):
    """One token's step of (..., head, v, k) states, q, k, v and decay (..., head, dim), beta (..., head): (the new
    states, out)."""
    state = state * decay[..., None, :]
    delta = (v - (state * k[..., None, :]).sum(-1)) * beta[..., None]
    state = state + delta[..., :, None] * k[..., None, :]
    return state, (state * q[..., None, :]).sum(-1)


def kimi_k3_sgl_kda_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(out, state_cache) as the call leaves them."""
    import torch

    q, k, v, decay, beta, state_cache, states = _operands(point, row_major_args, decode)
    slots = row_major_args["state_indices"].long().to(states.device)
    taken = slots >= 0
    # a -1 slot: a zero state, its out still given, nothing written
    start = torch.where(taken[:, None, None, None], states[slots.clamp(min=0)], 0.0)
    new, out = _step(start, q, k, v, decay, beta)
    states[slots[taken]] = new[taken]
    return out, state_cache


def _walk(point, row_major_args, decode, *, track):
    """Every sequence from its slot's state (zero where the slot is -1), its last state back to the slot; with
    ``track``, its state after its first 64 x track_chunk_idx[s] tokens to track_state[s] where that is inside it."""
    import torch

    args = row_major_args
    q, k, v, decay, beta, state_cache, states = _operands(point, args, decode)
    device = states.device
    offsets = args["cu_seqlens"].tolist()
    starts = torch.tensor(offsets[:-1], device=device)
    lengths = torch.tensor(offsets[1:], device=device) - starts
    slots = args["state_indices"].long().to(device)
    taken = slots >= 0
    state = torch.where(taken[:, None, None, None], states[slots.clamp(min=0)], 0.0)
    snapshot = None
    if track:
        snapshot = decode(args["track_state"], point.dtypes["track_state"]).clone()
        snaps = snapshot.view(state.shape)
        at = args["track_chunk_idx"].long().to(device) * TRACK_CHUNK
        snapped = (at >= 0) & (at < lengths)  # c = -1, or at or past the end: none
    out = torch.empty_like(v)
    longest = int(lengths.max()) if lengths.numel() else 0
    for j in range(longest):
        if track:
            hit = snapped & (at == j)  # the state after j tokens
            snaps[hit] = state[hit]
        live = j < lengths
        t = (starts + j).clamp(max=v.shape[0] - 1)
        new, o = _step(state, q[t], k[t], v[t], decay[t], beta[t])
        state = torch.where(live[:, None, None, None], new, state)
        out[t[live]] = o[live]
    states[slots[taken]] = state[taken]
    return out, state_cache, snapshot


def kimi_k3_sgl_kda_prefill_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(out, state_cache) as the call leaves them."""
    out, state_cache, _ = _walk(point, row_major_args, decode, track=False)
    return out, state_cache


def kimi_k3_sgl_kda_prefill_track_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(out, state_cache, track_state) as the call leaves them."""
    return _walk(point, row_major_args, decode, track=True)
