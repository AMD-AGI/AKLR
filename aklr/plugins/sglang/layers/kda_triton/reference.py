# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA from its recurrence, token by token, in float32, in the order of sglang's recurrent kernel
(q and k normalized, q scaled; the state decayed, then the delta, then the output); the one rounding is the
output's, to its dtype (sglang's chunk_kda also rounds the normalized q and k to bf16, an error left to compare).
A prefill walks all its sequences side by side, one token of each a step."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = [
    "L2_NORM_EPS",
    "TRACK_CHUNK",
    "KimiK3SglKdaDecodeReference",
    "KimiK3SglKdaPrefillReference",
    "KimiK3SglKdaPrefillTrackReference",
]

# x / sqrt(sum(x**2) + eps): fla's l2norm_fwd and the recurrent kernel's in-kernel norm
L2_NORM_EPS = 1e-6
# tokens of one chunk of chunk_kda, the unit of sglang's prefix-cache snapshots (track_chunk_idx)
TRACK_CHUNK = 64


def _operands(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, scale, lower_bound):
    """Each token's normalized and scaled q, normalized k, v, decay exp(g) along k and beta, (n_tokens, heads, dim)
    and (n_tokens, heads); the state cache in float32, a copy, and its (slot, head, v, k) view."""
    heads, dim = raw_g.shape[1], raw_g.shape[2]
    q, k, v = qkv.float().unflatten(1, (3, heads)).unbind(1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * scale
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    gate = lower_bound * torch.sigmoid(a_log.float().exp()[:, None] * (raw_g.float() + g_bias.float()))
    beta = torch.sigmoid(raw_beta.float()[..., 0])
    states = state_cache.float().clone()
    return q, k, v, gate.exp(), beta, states, states.view(-1, heads, dim, dim)


def _step(state, q, k, v, decay, beta):
    """One token's step of (..., head, v, k) states, q, k, v and decay (..., head, dim), beta (..., head): (the new
    states, out)."""
    state = state * decay[..., None, :]
    delta = (v - (state * k[..., None, :]).sum(-1)) * beta[..., None]
    state = state + delta[..., :, None] * k[..., None, :]
    return state, (state * q[..., None, :]).sum(-1)


class KimiK3SglKdaDecodeReference(OpReference):
    def reference(
        self, *, qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, scale, lower_bound, **_
    ):
        """out and state_cache as the call leaves them."""
        q, k, v, decay, beta, cache, states = _operands(
            qkv, raw_g, raw_beta, a_log, g_bias, state_cache, scale, lower_bound
        )
        slots = state_indices.long()
        taken = slots >= 0
        # a -1 slot: a zero state, its out still given, nothing written
        start = torch.where(taken[:, None, None, None], states[slots.clamp(min=0)], 0.0)
        new, out = _step(start, q, k, v, decay, beta)
        states[slots[taken]] = new[taken]
        return {"out": out.to(qkv.dtype), "state_cache": cache}


def _walk(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound,
          track_chunk_idx=None, track_state=None):  # fmt: skip
    """Every sequence from its slot's state (zero where the slot is -1), its last state back to the slot; with a
    snapshot, its state after its first 64 x track_chunk_idx[s] tokens to track_state[s] where that is inside it.
    Returns out, the state cache and the snapshots (None without)."""
    q, k, v, decay, beta, cache, states = _operands(
        qkv, raw_g, raw_beta, a_log, g_bias, state_cache, scale, lower_bound
    )
    device = states.device
    offsets = cu_seqlens.tolist()
    starts = torch.tensor(offsets[:-1], device=device)
    lengths = torch.tensor(offsets[1:], device=device) - starts
    slots = state_indices.long()
    taken = slots >= 0
    state = torch.where(taken[:, None, None, None], states[slots.clamp(min=0)], 0.0)
    snapshot = None
    if track_state is not None:
        snapshot = track_state.float().clone()
        snaps = snapshot.view(state.shape)
        at = track_chunk_idx.long() * TRACK_CHUNK
        snapped = (at >= 0) & (at < lengths)  # c = -1, or at or past the end: none
    out = torch.empty_like(v)
    longest = int(lengths.max()) if lengths.numel() else 0
    for j in range(longest):
        if snapshot is not None:
            hit = snapped & (at == j)  # the state after j tokens
            snaps[hit] = state[hit]
        live = j < lengths
        t = (starts + j).clamp(max=v.shape[0] - 1)
        new, o = _step(state, q[t], k[t], v[t], decay[t], beta[t])
        state = torch.where(live[:, None, None, None], new, state)
        out[t[live]] = o[live]
    states[slots[taken]] = state[taken]
    return out.to(qkv.dtype), cache, snapshot


class KimiK3SglKdaPrefillReference(OpReference):
    def reference(
        self,
        *,
        qkv,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        cu_seqlens,
        scale,
        lower_bound,
        **_,
    ):
        """out and state_cache as the call leaves them."""
        out, cache, _ = _walk(
            qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound
        )
        return {"out": out, "state_cache": cache}


class KimiK3SglKdaPrefillTrackReference(OpReference):
    def reference(
        self,
        *,
        qkv,
        raw_g,
        raw_beta,
        a_log,
        g_bias,
        state_cache,
        state_indices,
        cu_seqlens,
        track_chunk_idx,
        track_state,
        scale,
        lower_bound,
        **_,
    ):
        """out, state_cache and track_state as the call leaves them."""
        out, cache, snapshot = _walk(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens,
                                     scale, lower_bound, track_chunk_idx, track_state)  # fmt: skip
        return {"out": out, "state_cache": cache, "track_state": snapshot}
