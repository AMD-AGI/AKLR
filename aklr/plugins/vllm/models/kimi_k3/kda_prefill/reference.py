# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA prefill from its recurrence, token by token, in float32; the output rounds to q's dtype, the states stay
float32."""

import torch

from aklr.core.contract.reference import OpReference

__all__ = [
    "CHECKPOINT_CHUNK",
    "L2_NORM_EPS",
    "KimiK3KdaPrefillCheckpointReference",
    "KimiK3KdaPrefillReference",
]

# x / sqrt(sum(x**2) + eps): vLLM's Triton prefill path and recurrent kernel, and aiter's KDA
L2_NORM_EPS = 1e-6
# tokens a checkpoint chunk counts: vLLM's KDA_CHECKPOINT_ALIGNMENT, its fused kernels' chunk
CHECKPOINT_CHUNK = 64


def _l2_normalized(x):
    return x * (x.square().sum(dim=-1, keepdim=True) + L2_NORM_EPS).rsqrt()


class KimiK3KdaPrefillReference(OpReference):
    def reference(self, **call):
        """out and state_cache as the call leaves them."""
        out, state_cache, _ = _walk(**call)
        return {"out": out, "state_cache": state_cache}


class KimiK3KdaPrefillCheckpointReference(OpReference):
    def reference(self, **call):
        """out, state_cache and checkpoint_state as the call leaves them."""
        out, state_cache, checkpoint_state = _walk(**call)
        return {"out": out, "state_cache": state_cache, "checkpoint_state": checkpoint_state}


def _walk(
    *,
    q,
    k,
    v,
    raw_g,
    raw_beta,
    a_log,
    g_bias,
    state_cache,
    state_indices,
    has_initial_state,
    cu_seqlens,
    scale,
    lower_bound,
    checkpoint_chunks=None,
    checkpoint_state=None,
    **_,
):
    """Every sequence token by token from its slot (or zero), its last state back to the slot; with a
    ``checkpoint_state``, its state after 64 x checkpoint_chunks[s] tokens to checkpoint_state[s] when that is inside
    the sequence (a point at or past its end is outside the contract, and writes nothing here). (out, state_cache,
    checkpoint_state or None) in their dtypes."""
    heads, dim = q.shape[1], q.shape[2]
    qn = _l2_normalized(q.float()) * scale
    kn = _l2_normalized(k.float())
    vf = v.float()
    gate = lower_bound * torch.sigmoid(a_log.float().exp()[:, None] * (raw_g.float() + g_bias.float()))
    decay = gate.exp()  # (n_tokens, n_heads, head_dim), along k
    beta = torch.sigmoid(raw_beta.float())[..., None, None]
    # .float() may return its input: the states are written in copies
    cache = state_cache.float().clone()
    states = cache.view(-1, heads, dim, dim)  # (slot, head, v, k)
    offsets = cu_seqlens.tolist()
    slots = state_indices.tolist()
    resumes = has_initial_state.tolist()
    snapshot, snaps, chunks = None, None, None
    if checkpoint_state is not None:
        snapshot = checkpoint_state.float().clone()
        snaps = snapshot.view(-1, heads, dim, dim)
        chunks = checkpoint_chunks.tolist()
    out = torch.empty_like(vf)
    for seq, (slot, resume) in enumerate(zip(slots, resumes, strict=True)):
        state = states[slot].clone() if resume else torch.zeros_like(states[slot])
        start, end = offsets[seq], offsets[seq + 1]
        at = start + CHECKPOINT_CHUNK * chunks[seq] if chunks is not None else None
        for t in range(start, end):
            state *= decay[t, :, None, :]
            update = (vf[t, :, :, None] - state @ kn[t, :, :, None]) * beta[t]
            state += update @ kn[t, :, None, :]
            out[t] = (state @ qn[t, :, :, None])[..., 0]
            if at is not None and start < at < end and t + 1 == at:
                snaps[seq] = state
        states[slot] = state
    return (
        out.to(q.dtype),
        cache.to(state_cache.dtype),
        None if snapshot is None else snapshot.to(checkpoint_state.dtype),
    )
