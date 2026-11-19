# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA prefill from its recurrence, token by token, in the dtype ``decode`` gives; no rounding
is part of the op."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.plugins.vllm.models.kimi_k3.kda_prefill.contract import CHECKPOINT_CHUNK, L2_NORM_EPS

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["kimi_k3_kda_prefill_checkpoint_reference", "kimi_k3_kda_prefill_reference"]


def _l2_normalized(x: Tensor) -> Tensor:
    return x * (x.square().sum(dim=-1, keepdim=True) + L2_NORM_EPS).rsqrt()


def kimi_k3_kda_prefill_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """(out, state_cache) as the call leaves them."""
    out, state_cache, _ = _walk(point, row_major_args, decode, checkpoint=False)
    return out, state_cache


def kimi_k3_kda_prefill_checkpoint_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """(out, state_cache, checkpoint_state) as the call leaves them."""
    return _walk(point, row_major_args, decode, checkpoint=True)


def _walk(point, row_major_args, decode, *, checkpoint):
    """Every sequence token by token from its slot (or zero), its last state back to the slot; with ``checkpoint``,
    its state after 64 x checkpoint_chunks[s] tokens to checkpoint_state[s] when that is inside the sequence."""
    import torch

    args, dtypes = row_major_args, point.dtypes
    heads, dim = point.axes["n_heads"], point.axes["head_dim"]
    q = _l2_normalized(decode(args["q"], dtypes["q"])) * args["scale"]
    k = _l2_normalized(decode(args["k"], dtypes["k"]))
    v = decode(args["v"], dtypes["v"])
    a_log = decode(args["a_log"], dtypes["a_log"])
    g_bias = decode(args["g_bias"], dtypes["g_bias"])
    gate = args["lower_bound"] * torch.sigmoid(
        a_log.exp()[:, None] * (decode(args["raw_g"], dtypes["raw_g"]) + g_bias)
    )
    decay = gate.exp()  # (n_tokens, n_heads, head_dim), along k
    beta = torch.sigmoid(decode(args["raw_beta"], dtypes["raw_beta"]))[..., None, None]
    state_cache = decode(args["state_cache"], dtypes["state_cache"]).clone()
    states = state_cache.view(-1, heads, dim, dim)  # (slot, head, v, k)
    offsets = args["cu_seqlens"].tolist()
    slots = args["state_indices"].tolist()
    resumes = args["has_initial_state"].tolist()
    snapshot, snaps = None, None
    if checkpoint:
        snapshot = decode(args["checkpoint_state"], dtypes["checkpoint_state"]).clone()
        snaps = snapshot.view(-1, heads, dim, dim)
        chunks = args["checkpoint_chunks"].tolist()
    out = torch.empty_like(v)
    for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
        state = states[slot].clone() if resume else torch.zeros_like(states[slot])
        start, end = offsets[seq], offsets[seq + 1]
        at = start + CHECKPOINT_CHUNK * chunks[seq] if checkpoint else None
        for t in range(start, end):
            state *= decay[t, :, None, :]
            update = (v[t, :, :, None] - state @ k[t, :, :, None]) * beta[t]
            state += update @ k[t, :, None, :]
            out[t] = (state @ q[t, :, :, None])[..., 0]
            if at is not None and start < at < end and t + 1 == at:
                snaps[seq] = state
        states[slot] = state
    return out, state_cache, snapshot
