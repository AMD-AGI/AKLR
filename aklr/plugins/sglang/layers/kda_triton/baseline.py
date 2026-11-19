# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's Triton KDA backend runs (``TritonKDAKernel``), on the operands as ``KDAAttnBackend`` hands them
over: q, k and v the [1, n, heads, dim] views of the packed convolution rows, the gate [1, n, heads, dim] (a prefill's)
or [n, heads * dim] (a decode's), beta [1, n, heads], the layer's A_log and flat dt_bias, the state pool [slots,
heads, v, k].

- Prefill: fla's ``chunk_kda`` with q and k normalized, the gate and beta activated inside (beta_is_raw), the state
  read and written in place at the slots, as ``TritonKDAKernel.extend`` calls it on sglang 0.5.22 (0.5.23 on gfx950
  first tries its HIP prefill, which is not this baseline); with a snapshot, its fp32 track_state written at
  track_chunk_idx, as the KDA backend asks it under the mamba radix cache's extra buffer.
- Decode: ``fused_sigmoid_gating_delta_rule_update`` with is_kda and the safe gate, one token a sequence through
  sglang's query_start_loc (0, 1, .., n), as ``TritonKDAKernel.decode`` calls it."""

from __future__ import annotations

__all__ = [
    "kimi_k3_sgl_kda_decode_baseline",
    "kimi_k3_sgl_kda_prefill_baseline",
    "kimi_k3_sgl_kda_prefill_track_baseline",
]

# each decode batch's query_start_loc (0, 1, .., n), by device and token count: made once, not in a timed call
_DECODE_OFFSETS: dict = {}


def _views(qkv, raw_g, raw_beta, a_log, g_bias, state_cache):
    """sglang's operands: q, k, v, the gate and beta with its leading batch dim, A_log [1, 1, heads, 1], dt_bias flat,
    the pool [slots, heads, v, k]."""
    heads, dim = raw_g.shape[1], raw_g.shape[2]
    q, k, v = (x[None] for x in qkv.unflatten(1, (3, heads)).unbind(1))
    return (
        q,
        k,
        v,
        raw_g[None],
        raw_beta[None, :, :, 0],
        a_log.view(1, 1, heads, 1),
        g_bias.flatten(),
        state_cache.unflatten(1, (heads, dim)),
    )


def _chunk_kda(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound,
               track_state=None, track_chunk_idx=None):  # fmt: skip
    """chunk_kda's out [1, n_tokens, heads, dim] (with a snapshot it returns the per-chunk states too: dropped)."""
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    q, k, v, g, beta, A_log, dt_bias, ssm_states = _views(qkv, raw_g, raw_beta, a_log, g_bias, state_cache)
    track = track_state is not None
    result = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=ssm_states,
        initial_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=True,
        output_intermediate_states=track,
        track_state=track_state.unflatten(1, ssm_states.shape[1:3]) if track else None,
        track_chunk_idx=track_chunk_idx,
        chunk_indices=None,
        chunk_offsets=None,
    )
    return result[0] if track else result


def kimi_k3_sgl_kda_prefill_baseline(
    qkv,
    raw_g,
    raw_beta,
    a_log,
    g_bias,
    state_cache,
    state_indices,
    cu_seqlens,
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
):
    """sglang's out [n_tokens, n_heads, head_dim]; state_cache is written in place, as in sglang."""
    o = _chunk_kda(
        qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound
    )
    if out is None:
        return o[0]
    out.copy_(o[0])
    return out


def kimi_k3_sgl_kda_prefill_track_baseline(
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
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
):
    """sglang's out [n_tokens, n_heads, head_dim] of a prefill that snapshots; state_cache and track_state are written
    in place, as in sglang (its per-chunk states h, which the KDA backend does not read once it passes track_state,
    are dropped)."""
    o = _chunk_kda(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound,
                   track_state=track_state, track_chunk_idx=track_chunk_idx)  # fmt: skip
    if out is None:
        return o[0]
    out.copy_(o[0])
    return out


def _fused_sigmoid_gating_delta_rule_update():
    """sglang's recurrent KDA/GDN kernel: under ``kernels/ops/attention/linear`` from 0.5.23, ``fla`` before."""
    try:
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
    except ModuleNotFoundError:
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
    return fused_sigmoid_gating_delta_rule_update


def kimi_k3_sgl_kda_decode_baseline(
    qkv,
    raw_g,
    raw_beta,
    a_log,
    g_bias,
    state_cache,
    state_indices,
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
):
    """sglang's out [n_tokens, n_heads, head_dim]; state_cache is written in place, as in sglang."""
    import torch

    n_tokens = qkv.shape[0]
    q, k, v, _, b, A_log, dt_bias, ssm_states = _views(qkv, raw_g, raw_beta, a_log, g_bias, state_cache)
    key = (qkv.device, n_tokens)
    if key not in _DECODE_OFFSETS:
        _DECODE_OFFSETS[key] = torch.arange(n_tokens + 1, dtype=torch.int32, device=qkv.device)
    o = _fused_sigmoid_gating_delta_rule_update()(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_g.flatten(1),  # the decode gate as the KDA backend hands it over: [n, heads * dim]
        b=b,
        initial_state_source=ssm_states,
        initial_state_indices=state_indices,
        cu_seqlens=_DECODE_OFFSETS[key],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=lower_bound,
    )
    if out is None:
        return o[0]
    out.copy_(o[0])
    return out
