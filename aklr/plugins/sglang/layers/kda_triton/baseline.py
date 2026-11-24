# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's Triton KDA backend runs (``TritonKDAKernel``), on the operands as ``KDAAttnBackend`` hands them
over: q, k and v the [1, n, heads, dim] views of the packed convolution rows, the gate [1, n, heads, dim], beta
[1, n, heads], the layer's A_log and flat dt_bias, the state pool [slots, heads, v, k].

- Prefill: fla's ``chunk_kda`` with q and k normalized, the gate and beta activated inside (beta_is_raw), the state
  read and written in place at the slots, as ``TritonKDAKernel.extend`` calls it on sglang 0.5.22 (0.5.23 on gfx950
  first tries its HIP prefill, which is not this baseline); with a snapshot, its fp32 track_state written at
  track_chunk_idx, as the KDA backend asks it under the mamba radix cache's extra buffer."""

from aklr.core.contract.baseline import OpBaseline

__all__ = [
    "KimiK3SglKdaPrefillBaseline",
    "KimiK3SglKdaPrefillTrackBaseline",
]


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
    if v.is_contiguous():
        # one token's view of its packed row: chunk_kda's v.contiguous() would be the view itself, and 0.5.23's
        # writes its output there (o = v), into qkv; a longer view is copied there by sglang already
        v = v.clone()
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


class KimiK3SglKdaPrefillBaseline(OpBaseline):
    def baseline(
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
        storage="row_major",
        scale=128**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        """sglang's out [n_tokens, n_heads, head_dim], which chunk_kda allocates; state_cache is written in place,
        as in sglang."""
        o = _chunk_kda(
            qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound
        )
        return {"out": o[0]}


class KimiK3SglKdaPrefillTrackBaseline(OpBaseline):
    def baseline(
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
        storage="row_major",
        scale=128**-0.5,
        lower_bound=-5.0,
        out_dtype=None,
        out=None,
    ):
        """sglang's out [n_tokens, n_heads, head_dim] of a prefill that snapshots; state_cache and track_state are
        written in place, as in sglang (its per-chunk states h, which the KDA backend does not read once it passes
        track_state, are dropped)."""
        o = _chunk_kda(qkv, raw_g, raw_beta, a_log, g_bias, state_cache, state_indices, cu_seqlens, scale, lower_bound,
                       track_state=track_state, track_chunk_idx=track_chunk_idx)  # fmt: skip
        return {"out": o[0]}
