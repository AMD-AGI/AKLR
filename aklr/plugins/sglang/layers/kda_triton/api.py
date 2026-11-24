# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import importlib

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.kda_triton.contract import (
    KIMI_K3_SGL_KDA_PREFILL,
    KIMI_K3_SGL_KDA_PREFILL_TRACK,
)

__all__ = ["kimi_k3_sgl_kda_prefill", "kimi_k3_sgl_kda_prefill_track"]

_TRITON_KDA = "sglang.srt.layers.attention.linear.kernels.kda_triton.TritonKDAKernel"


@api(KIMI_K3_SGL_KDA_PREFILL)
def kimi_k3_sgl_kda_prefill(
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
): ...


@api(KIMI_K3_SGL_KDA_PREFILL_TRACK)
def kimi_k3_sgl_kda_prefill_track(
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
): ...


def _packed(q, k, v, heads, dim):
    """q, k and v [1, n, heads, dim] as the one row-major (n, 3 heads, dim) view of the convolution rows they split,
    else None."""
    step = heads * dim * q.element_size()
    if (
        q.dim() != 4
        or q.shape[0] != 1
        or k.shape != q.shape
        or v.shape != q.shape
        or q.stride() != k.stride()
        or q.stride() != v.stride()
        or q.stride()[1:] != (3 * heads * dim, dim, 1)
        or (k.data_ptr() - q.data_ptr(), v.data_ptr() - q.data_ptr()) != (step, 2 * step)
    ):
        return None
    return q[0].as_strided((q.shape[1], 3 * heads, dim), (3 * heads * dim, dim, 1))


def _layer_operands(q, b, A_log, dt_bias, ssm_states, cache_indices):
    """(name, failed) of what the route takes as sglang's KDA backend passes it: bf16 rows, beta [1, n, heads] dense
    in its heads, A_log and dt_bias per head, the float32 pool [slots, heads, v, k] dense, int32 slots."""
    n, heads, dim = q.shape[1], q.shape[2], q.shape[3]
    return (
        ("q|k|v not bf16", q.dtype != torch.bfloat16),
        ("beta", tuple(b.shape) != (1, n, heads) or b.stride(2) != 1 or b.dtype != torch.bfloat16),
        ("A_log", A_log.numel() != heads or A_log.dtype != torch.float32 or not A_log.is_contiguous()),
        (
            "dt_bias",
            dt_bias.numel() != heads * dim or dt_bias.dtype != torch.float32 or not dt_bias.is_contiguous(),
        ),
        ("state pool", ssm_states.dim() != 4 or tuple(ssm_states.shape[1:]) != (heads, dim, dim)),
        ("state pool not dense", ssm_states.dtype != torch.float32 or not ssm_states.is_contiguous()),
        (
            "slots",
            cache_indices.dim() != 1
            or cache_indices.dtype != torch.int32
            or not cache_indices.is_contiguous(),
        ),
    )


def _common(qkv, raw_g, b, A_log, dt_bias, ssm_states, heads, dim):
    """The ops' leading operands: qkv, the gate (n, heads, dim), beta (n, heads, 1), a_log, g_bias and the pool as
    (slots, heads * dim, dim) rows."""
    return (
        qkv,
        raw_g,
        b[0].unsqueeze(-1),
        A_log.view(heads),
        dt_bias.view(heads, dim),
        ssm_states.view(ssm_states.shape[0], heads * dim, dim),
    )


def _storage(b):
    """The ops' layout of beta [1, n, heads]: its rows apart by more than a row (a column of sglang's [f_a | b]
    projection, sglang's on K3) or dense."""
    return "row_major" if b.stride(1) == b.shape[2] else "strided_beta"


@functools.cache
def _sglang_hip_prefill() -> bool:
    """Whether sglang runs its own HIP KDA prefill (0.5.23's kda_gfx950_prefill, on by default on gfx950), which its
    TritonKDAKernel.extend tries before chunk_kda: those calls stay with sglang, since the op is timed against
    chunk_kda, not against it."""
    kda_triton = importlib.import_module(_TRITON_KDA.rpartition(".")[0])
    hip = getattr(kda_triton, "kda_gfx950_prefill", None)
    return hip is not None and bool(hip.available())


@sglang_route(f"{_TRITON_KDA}.extend", versions=("0.5.22", "0.5.23"))
def _extend(
    self,
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
    **kwargs,
):
    """Kimi-K3's KDA prefill as the KDA backend's forward_extend asks the Triton kernel for it (chunk_kda): the packed
    convolution rows, the raw gate [1, n, heads, dim] and beta, the layer's A_log, dt_bias and safe gate, the pool read
    and written in place at the slots; with the mamba radix cache's extra buffer, the fp32 snapshots (track_state at
    track_chunk_idx) through kimi_k3_sgl_kda_prefill_track. Returns what sglang's does: the output [1, n, heads, dim],
    and with return_intermediate_states no per-chunk states (the backend reads track_state, or the final state where
    every tracked row is chunk-aligned). Stays with sglang: a breakable prefill CUDA graph (padded chunk tables) or any
    capture, a spec-decode extend, a call wanting the per-chunk states h, and sglang's own HIP prefill where it runs."""
    track_state, track_chunk_idx = kwargs.get("track_state"), kwargs.get("track_chunk_idx")
    h_src = kwargs.get("track_ssm_h_src")
    if (
        A_log is None
        or dt_bias is None
        or lower_bound is None
        or not beta_is_raw
        or kwargs.get("is_spec_decode", False)
        or kwargs.get("chunk_indices") is not None
        or kwargs.get("chunk_offsets") is not None
        or (q.device.type == "cuda" and torch.cuda.is_current_stream_capturing())
        or (track_state is None) != (track_chunk_idx is None)
        # the per-chunk states h asked for: no snapshot buffer, and unaligned rows to track
        or (return_intermediate_states and track_state is None and (h_src is None or h_src.numel() > 0))
        or q.dim() != 4
        or _sglang_hip_prefill()
    ):
        return FallbackReason.OUT_OF_CONTRACT
    n, heads, dim = q.shape[1], q.shape[2], q.shape[3]
    n_seqs = cache_indices.shape[0] if cache_indices.dim() == 1 else -1
    qkv = _packed(q, k, v, heads, dim)
    lengths = kwargs.get("extend_seq_lens_cpu")
    checks = (
        ("q|k|v not one packed row", qkv is None),
        *_layer_operands(q, beta, A_log, dt_bias, ssm_states, cache_indices),
        ("gate", tuple(g.shape) != (1, n, heads, dim) or not g.is_contiguous() or g.dtype != torch.bfloat16),
        ("offsets", tuple(query_start_loc.shape) != (n_seqs + 1,) or query_start_loc.dtype != torch.int32
         or not query_start_loc.is_contiguous()),
        # the lengths on the host: rows past the sequences (DP attention's padding), or another batch's
        ("lengths", lengths is None or len(lengths) != n_seqs or sum(lengths) != n),
        ("snapshot", track_state is not None and (
            tuple(track_state.shape) != (n_seqs, heads, dim, dim) or track_state.dtype != torch.float32
            or not track_state.is_contiguous() or tuple(track_chunk_idx.shape) != (n_seqs,)
            or track_chunk_idx.dtype != torch.int32 or not track_chunk_idx.is_contiguous())),
    )  # fmt: skip
    if any(failed for _, failed in checks):
        return FallbackReason.OUT_OF_CONTRACT
    common = _common(qkv, g[0], beta, A_log, dt_bias, ssm_states, heads, dim)
    options = {"storage": _storage(beta), "scale": dim**-0.5, "lower_bound": float(lower_bound)}
    if track_state is None:
        result = kimi_k3_sgl_kda_prefill(*common, cache_indices, query_start_loc, **options)
    else:
        snapshot = track_state.view(n_seqs, heads * dim, dim)
        result = kimi_k3_sgl_kda_prefill_track(
            *common, cache_indices, query_start_loc, track_chunk_idx, snapshot, **options
        )
    if isinstance(result, FallbackReason):
        return result
    out = result.unsqueeze(0)
    return (out, None) if return_intermediate_states else out
