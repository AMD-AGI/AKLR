# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA speculative decode from its recurrence, token by token, in float32 and in the order of vLLM's recurrent kernel
(q and k normalized, multiplied by the reciprocal square root of each norm, then q scaled; the state decayed, then the
delta, then the output); the output rounds to qkv's dtype, the state cache stays float32."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["L2_NORM_EPS", "KimiK3KdaDecodeSpecReference"]

# x * rsqrt(sum(x**2) + eps): vLLM's recurrent kernel
L2_NORM_EPS = 1e-6


def _operands(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, scale, lower_bound):
    """Each token's normalized q (scaled) and k, v, decay exp(g) and beta, (n_tokens, heads, dim) and
    (n_tokens, heads); the state cache in float32, a copy, and its (slot, head, v, k) view."""
    heads, dim = raw_g.shape[1], raw_g.shape[2]
    q, k, v = qkv.float().unflatten(1, (3, heads)).unbind(1)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    a = a_log.float().exp()[:, None]
    gate = lower_bound * torch.sigmoid(a * (raw_g.float() + g_bias.float()))
    beta = torch.sigmoid(raw_beta.float()[..., 0])
    # .float() may return its input: the cache is written in a copy
    cache = state_cache.float().clone()
    return q, k, v, gate.exp(), beta, cache, cache.view(-1, heads, dim, dim)


def _step(state, q, k, v, decay, beta):
    """One token's step of a (head, v, k) state: (the new state, out)."""
    state = state * decay[:, None, :]
    delta = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
    state = state + delta[:, :, None] * k[:, None, :]
    return state, (state * q[:, None, :]).sum(-1)


class KimiK3KdaDecodeSpecReference(OpReference):
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
        num_accepted_tokens,
        scale,
        lower_bound,
        **_,
    ):
        """out and state_cache as the call leaves them: each window from its start slot's state, every token's state
        to its slot."""
        q, k, v, decay, beta, cache, states = _operands(
            qkv, raw_g, raw_beta, a_log, g_bias, state_cache, scale, lower_bound
        )
        slots = state_indices.tolist()
        accepted = num_accepted_tokens.tolist()
        window = len(slots) // len(accepted)
        out = torch.zeros_like(v)
        for s, n_acc in enumerate(accepted):
            first = s * window
            start = slots[first + n_acc - 1]
            if start <= 0:  # vLLM's padding: zero outputs, no slot touched
                continue
            state = states[start].clone()
            for t in range(first, first + window):
                state, out[t] = _step(state, q[t], k[t], v[t], decay[t], beta[t])
                if slots[t] > 0:
                    states[slots[t]] = state
        return {"out": out.to(qkv.dtype), "state_cache": cache.to(state_cache.dtype)}
