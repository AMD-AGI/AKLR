# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA decode from its recurrence, token by token, in float32 and in the order of vLLM's recurrent
kernels (q and k normalized, as the packed decode kernel divides by each norm and the speculative one multiplies by its
reciprocal square root, then q scaled; the state decayed, then the delta, then the output); no rounding is part of the
op."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import to_float32
from aklr.plugins.vllm.models.kimi_k3.kda_decode.contract import L2_NORM_EPS

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_decode_reference", "kimi_k3_kda_decode_spec_reference"]


def _operands(point, args, *, by_rsqrt):
    """Each token's normalized q (scaled) and k, v, decay exp(g) and beta, (n_tokens, heads, dim) and
    (n_tokens, heads); the state cache in float32, a copy, and its (slot, head, v, k) view. ``by_rsqrt``: x times
    rsqrt(sum(x**2) + eps), else x divided by sqrt(sum(x**2) + eps)."""
    import torch

    dtypes = point.dtypes
    heads, dim = point.axes["n_heads"], point.axes["head_dim"]
    q, k, v = to_float32(args["qkv"], dtypes["qkv"]).unflatten(1, (3, heads)).unbind(1)
    if by_rsqrt:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * args["scale"]
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    else:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_NORM_EPS) * args["scale"]
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_NORM_EPS)
    a = to_float32(args["a_log"], dtypes["a_log"]).exp()[:, None]
    gate = args["lower_bound"] * torch.sigmoid(
        a * (to_float32(args["raw_g"], dtypes["raw_g"]) + to_float32(args["g_bias"], dtypes["g_bias"]))
    )
    beta = torch.sigmoid(to_float32(args["raw_beta"], dtypes["raw_beta"])[..., 0])
    state_cache = to_float32(args["state_cache"], dtypes["state_cache"]).clone()
    return q, k, v, gate.exp(), beta, state_cache, state_cache.view(-1, heads, dim, dim)


def _step(state, q, k, v, decay, beta):
    """One token's step of a (head, v, k) state: (the new state, out)."""
    state = state * decay[:, None, :]
    delta = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
    state = state + delta[:, :, None] * k[:, None, :]
    return state, (state * q[:, None, :]).sum(-1)


def kimi_k3_kda_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """(out, state_cache) as the call leaves them."""
    import torch

    q, k, v, decay, beta, state_cache, states = _operands(point, row_major_args, by_rsqrt=False)
    out = torch.zeros_like(v)
    for t, slot in enumerate(row_major_args["state_indices"].tolist()):
        if slot <= 0:  # vLLM's null block and padding: a zero output, no slot touched
            continue
        states[slot], out[t] = _step(states[slot], q[t], k[t], v[t], decay[t], beta[t])
    return out, state_cache


def kimi_k3_kda_decode_spec_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """(out, state_cache) as the call leaves them: each window from its start slot's state, every token's state to its
    slot."""
    import torch

    q, k, v, decay, beta, state_cache, states = _operands(point, row_major_args, by_rsqrt=True)
    slots = row_major_args["state_indices"].tolist()
    accepted = row_major_args["num_accepted_tokens"].tolist()
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
    return out, state_cache
