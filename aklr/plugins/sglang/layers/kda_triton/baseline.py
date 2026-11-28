# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang's Triton KDA backend runs (``TritonKDAKernel``), on the operands as ``KDAAttnBackend`` hands them
over: q, k and v the [1, n, heads, dim] views of the packed convolution rows, the gate [1, n, heads, dim], beta
[1, n, heads], the layer's A_log and flat dt_bias, the state pool [slots, heads, v, k].

- Prefill: fla's ``chunk_kda`` with q and k normalized, the gate and beta activated inside (beta_is_raw), the state
  read and written in place at the slots, as ``TritonKDAKernel.extend`` calls it on sglang 0.5.22; with a snapshot,
  its fp32 track_state written at track_chunk_idx, as the KDA backend asks it under the mamba radix cache's extra
  buffer. sglang 0.5.23 on gfx950 runs its own HIP prefill (kda_gfx950_prefill) first, on by default; there the route
  leaves every call to sglang, so chunk_kda is the engine's path wherever the route serves (0.5.22, or 0.5.23 with
  SGLANG_ROCM_KDA_HIP_PREFILL=0).
- Chain verify in ReplaySSM ring mode (sglang 0.5.23, ``KDAAttnBackend._forward_target_verify``): its fused kernel
  takes no ring on gfx950, so the backend runs the draft tokens' convolution (``causal_conv1d_update`` with its
  windows) and the Triton verify kernel with the ring write (``TritonKDAKernel.target_verify``), two launches."""

import functools

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = [
    "KimiK3SglKdaPrefillBaseline",
    "KimiK3SglKdaPrefillTrackBaseline",
    "KimiK3SglKdaVerifyBaseline",
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


@functools.cache
def _chain_starts(requests: int, steps: int, device):
    """sglang's query_start_loc of dense chains (the backend's metadata holds it; made once here too)."""
    return torch.arange(requests + 1, dtype=torch.int32, device=device) * steps


@functools.cache
def _triton_kernel():
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

    return TritonKDAKernel()


class KimiK3SglKdaVerifyBaseline(OpBaseline):
    def baseline(
        self,
        *,
        mixed_qkv,
        conv_weight,
        conv_state,
        cache_indices,
        conv_window,
        window_indices,
        a,
        b,
        a_log,
        dt_bias,
        ssm_state,
        ring_v,
        ring_k,
        ring_g,
        ring_beta,
        storage="row_major",
        scale=128**-0.5,
        lower_bound=None,
        softplus_beta=None,
        softplus_threshold=None,
        output=None,
    ):
        """The dense chain branch of KDAAttnBackend._forward_target_verify on sglang 0.5.23, on the pools as it
        holds them: the convolution of each request's chain with its windows, then the Triton verify kernel from
        the slots' states, writing the rings; its own output [n_tokens, heads, dim] (a padding request's rows as
        sglang leaves them: it reads none)."""
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

        n_tokens, heads, dim = a.shape
        requests, steps = cache_indices.shape[0], conv_window.shape[1]
        convolved = causal_conv1d_update(
            mixed_qkv.flatten(1).view(requests, steps, -1).transpose(1, 2),
            conv_state.transpose(-1, -2),
            conv_weight,
            None,
            activation="silu",
            conv_state_indices=cache_indices,
            intermediate_conv_window=conv_window.transpose(-1, -2),
            intermediate_state_indices=window_indices,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )
        rows = convolved.transpose(1, 2).reshape(n_tokens, -1)
        q, k, v = (t.unflatten(-1, (heads, dim)).unsqueeze(0) for t in rows.split(heads * dim, dim=-1))
        out = _triton_kernel().target_verify(
            A_log=a_log.view(1, 1, heads, 1),
            dt_bias=dt_bias.flatten(),
            q=q,
            k=k,
            v=v,
            a=a.unsqueeze(0),
            b=b.squeeze(-1).unsqueeze(0),
            ssm_states=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=_chain_starts(requests, steps, cache_indices.device),
            intermediate_states_buffer=None,
            intermediate_state_indices=window_indices,
            cache_steps=steps,
            retrieve_parent_token=None,
            lower_bound=lower_bound,
            cache_ring=True,
            replayssm_rawv=ring_v,
            replayssm_rawk=ring_k,
            replayssm_g=ring_g,
            replayssm_beta=ring_beta,
        )
        return {"output": out[0]}
