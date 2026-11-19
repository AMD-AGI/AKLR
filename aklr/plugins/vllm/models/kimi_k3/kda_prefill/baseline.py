# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's KDA prefill, ``chunk_kimi_delta_attn``: the fastest one a vLLM-free image holds (vLLM's
fused HIP kernels are built into vLLM's ``_C``). With every activation in its kernels and the
safe gate, aiter runs its two-kernel FlashKDA path (chunks of 32), cutting long sequences into
segments that run side by side. It takes and returns dense V-first states, so the slots are
gathered before it and written back after it, as vLLM's Triton backend does around its own."""

from __future__ import annotations

__all__ = ["kimi_k3_kda_prefill_baseline", "kimi_k3_kda_prefill_checkpoint_baseline"]


def kimi_k3_kda_prefill_baseline(
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
    state_storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
):
    """aiter's ``chunk_kimi_delta_attn`` between a gather of the sequences' slots (zero where a
    sequence starts fresh) and the scatter of their last states."""
    import torch
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    heads, dim = q.shape[1], q.shape[2]
    # (slot, head, v, k); a view, so the scatter below writes state_cache
    states = state_cache.unflatten(1, (heads, dim))
    slots = state_indices.long()
    resumed = has_initial_state[:, None, None, None]
    initial_state = torch.where(resumed, states[slots], 0.0)
    o, final_state = chunk_kimi_delta_attn(
        q[None],
        k[None],
        v[None],
        raw_g[None],
        raw_beta[None],
        A_log=a_log,
        dt_bias=g_bias.flatten(),
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )
    states[slots] = final_state
    if out is None:
        return o[0]
    out.copy_(o[0])
    return out


# (start, cut, end) of every sequence, cut = end without a snapshot, by the identity and version of cu_seqlens and
# checkpoint_chunks: the last call's only
_CUTS: dict = {}


def _checkpoint_cuts(cu_seqlens, checkpoint_chunks):
    """Each sequence's (start, cut, end), read on the host outside a CUDA graph capture and kept for the capture of
    the same call that follows it (the bench calls a baseline eagerly before it captures it), where reading is not
    allowed."""
    import torch

    from aklr.plugins.vllm.models.kimi_k3.kda_prefill.contract import CHECKPOINT_CHUNK

    key = (cu_seqlens.data_ptr(), cu_seqlens._version, cu_seqlens.numel(), checkpoint_chunks.data_ptr(),
           checkpoint_chunks._version)  # fmt: skip
    if torch.cuda.is_current_stream_capturing():
        if key not in _CUTS:
            raise RuntimeError(
                "kimi_k3_kda_prefill_checkpoint's baseline reads its cut points on the host: call it once with the "
                "same tensors outside the capture first"
            )
        return _CUTS[key]
    bounds, chunks = cu_seqlens.tolist(), checkpoint_chunks.tolist()
    cuts = []
    for s, c in enumerate(chunks):
        at = bounds[s] + CHECKPOINT_CHUNK * c
        cuts.append((bounds[s], at if bounds[s] < at < bounds[s + 1] else bounds[s + 1], bounds[s + 1]))
    _CUTS.clear()
    _CUTS[key] = tuple(cuts)
    return _CUTS[key]


def kimi_k3_kda_prefill_checkpoint_baseline(
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
    checkpoint_chunks,
    checkpoint_state,
    state_storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
):
    """aiter's ``chunk_kimi_delta_attn`` on each sequence alone, in two calls when it takes a snapshot: up to the
    snapshot point, whose last state is the snapshot, then the rest from there. The cut points are read on the host
    (``_checkpoint_cuts``) and the calls are dense (no cu_seqlens, which aiter reads on the host behind a one-entry
    cache), so a CUDA graph captures the call that ran once before it: a baseline of validation, not a kernel that
    serves."""
    import torch
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    heads, dim = q.shape[1], q.shape[2]
    states = state_cache.unflatten(1, (heads, dim))
    snaps = checkpoint_state.unflatten(1, (heads, dim))
    slots = state_indices.long()
    initial_state = torch.where(has_initial_state[:, None, None, None], states.index_select(0, slots), 0.0)
    o = torch.empty_like(v) if out is None else out

    def walk(a, b, initial):
        part, final = chunk_kimi_delta_attn(
            q[None, a:b], k[None, a:b], v[None, a:b], raw_g[None, a:b], raw_beta[None, a:b],
            A_log=a_log, dt_bias=g_bias.flatten(), scale=scale, initial_state=initial, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True,
            lower_bound=lower_bound, state_v_first=True,
        )  # fmt: skip
        o[a:b] = part[0]
        return final

    finals = []
    for s, (start, cut, end) in enumerate(_checkpoint_cuts(cu_seqlens, checkpoint_chunks)):
        final = initial_state[s : s + 1]
        if cut > start:
            final = walk(start, cut, final)
        if cut < end:
            snaps[s] = final[0]
            final = walk(cut, end, final)
        finals.append(final)
    if finals:
        states.index_copy_(0, slots, torch.cat(finals).to(states.dtype))
    return o
