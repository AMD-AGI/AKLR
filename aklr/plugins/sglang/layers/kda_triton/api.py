# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import importlib
import logging

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.kda_triton.contract import (
    KIMI_K3_SGL_KDA_PREFILL,
    KIMI_K3_SGL_KDA_PREFILL_TRACK,
    KIMI_K3_SGL_KDA_VERIFY,
)

__all__ = ["kimi_k3_sgl_kda_prefill", "kimi_k3_sgl_kda_prefill_track", "kimi_k3_sgl_kda_verify"]

_LOGGER = logging.getLogger(__name__)
_SEEN: set[str] = set()


def _once(reason: str) -> None:
    """Log each distinct reason the verify route hands a call back to sglang, once per process."""
    if reason not in _SEEN:
        _SEEN.add(reason)
        _LOGGER.warning("aklr kda verify route falls back: %s", reason)


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
    in its heads, its elements within int32 offsets of its first (the kernels' offsets), A_log and dt_bias per head,
    the float32 pool [slots, heads, v, k] dense, int32 slots."""
    n, heads, dim = q.shape[1], q.shape[2], q.shape[3]
    return (
        ("q|k|v not bf16", q.dtype != torch.bfloat16),
        ("beta", tuple(b.shape) != (1, n, heads) or b.stride(2) != 1 or b.dtype != torch.bfloat16),
        ("beta past int32 offsets", (n - 1) * b.stride(1) + heads - 1 >= 2**31),
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
    """Whether sglang runs its own HIP KDA prefill (0.5.23's kda_gfx950_prefill, on by default on gfx950, off with
    SGLANG_ROCM_KDA_HIP_PREFILL=0), which its TritonKDAKernel.extend tries before chunk_kda: then every call stays with
    sglang, since the op is timed against chunk_kda, not against it."""
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
    capture, a spec-decode extend, a call wanting the per-chunk states h, and every call where sglang's own HIP prefill
    runs. So the route serves on 0.5.22; on 0.5.23 on gfx950, whose HIP prefill is on by default, it leaves every call
    to sglang and serves only with that prefill turned off (SGLANG_ROCM_KDA_HIP_PREFILL=0)."""
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


@api(KIMI_K3_SGL_KDA_VERIFY)
def kimi_k3_sgl_kda_verify(
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
): ...


@sglang_route(
    "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend._forward_target_verify",
    versions=("0.5.23",),
)
def _forward_target_verify(self, layer, forward_batch, mixed_qkv, a, b):
    """sglang 0.5.23's chain verify under --enable-linear-replayssm-spec: on gfx950 its fused
    kernel takes no ring, so the convolution and the gated delta rule run as separate kernels
    (the Triton verify kernel's, also where the nv_cutedsl backend falls to it off CUDA); the op
    takes the same pools as they are stored and the projections where they are. Trees, ragged
    windows, the per-step snapshot mode, a convolution bias, the other verify kernels and
    projections laid out otherwise stay on sglang."""
    from sglang.srt.layers.attention.linear import kda_backend, utils

    fm = self.forward_metadata
    dispatcher = self.kernel_dispatcher
    cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    spec = forward_batch.spec_info
    steps, seq_len = spec.draft_token_num, mixed_qkv.shape[0]
    heads, dim = layer.num_v_heads, layer.head_k_dim
    checks = (
        ("no ReplaySSM ring", getattr(cache, "replayssm_rawv", None) is None),
        ("per-step state snapshots", getattr(cache, "intermediate_ssm", None) is not None),
        ("ragged verify layout", spec.ragged_verify_layout is not None),
        (
            "tree verify",
            fm.retrieve_next_token is not None
            or fm.retrieve_next_sibling is not None
            or fm.retrieve_parent_token is not None,
        ),
        ("convolution bias", layer.bias is not None),
        ("fused accept state", getattr(self, "accept_lens_pool", None) is not None),
        # the unfused chain is the Triton kernel's, which the nv_cutedsl backend also falls to off CUDA
        (
            "verify kernel",
            type(dispatcher.verify_kernel).__name__ != "TritonKDAKernel"
            or (dispatcher.verify_backend.is_nv_cutedsl() and kda_backend.is_cuda()),
        ),
        (f"{seq_len} tokens in chains of {steps}", seq_len % steps != 0),
        ("head counts", (layer.num_q_heads, layer.num_k_heads) != (heads, heads)),
        ("head sizes", (layer.head_q_dim, layer.head_v_dim) != (dim, dim)),
        (
            f"convolution input of {tuple(mixed_qkv.shape)}",
            mixed_qkv.dim() != 2 or mixed_qkv.shape[1] != 3 * heads * dim,
        ),
        # the kernels load a row's channels eight at a time, 16 bytes aligned
        (
            f"convolution rows {mixed_qkv.stride(0)} channels apart",
            mixed_qkv.stride(-1) != 1 or mixed_qkv.stride(0) % 8 != 0,
        ),
        (f"gate of {tuple(a.shape)}", a.numel() != seq_len * heads * dim),
        ("gate not dense", not a.is_contiguous()),
        ("beta's heads not dense", b.stride(-1) != 1),
        (f"beta of {tuple(b.shape)}", b.numel() != seq_len * heads),
        ("A_log", layer.A_log.numel() != heads),
        ("dt_bias", layer.dt_bias.numel() != heads * dim),
    )
    failed = [name for name, bad in checks if bad]
    if failed:
        _once(", ".join(failed))
        return FallbackReason.OUT_OF_CONTRACT
    requests = seq_len // steps
    slots = fm.mamba_cache_indices
    # sglang's select_verify_intermediate_state_indices, its mask computed only where it is read
    lines = self.verify_intermediate_state_indices
    if utils.pp_spec_stable_rows_enabled():
        lines = utils.select_verify_intermediate_state_indices(
            lines,
            forward_batch.req_pool_indices,
            slots[: fm.query_start_loc.shape[0] - 1] >= 0,
            self.req_to_token_pool.size,
        )
    arguments = {
        # sglang's projections where they are: the convolution input and beta rows of its fused projections'
        # outputs (a dense tensor is such rows too), the gate dense
        "storage": "strided_rows",
        "mixed_qkv": mixed_qkv.unflatten(-1, (3 * heads, dim)),
        "a": a.reshape(seq_len, heads, dim),
        "b": b.reshape(seq_len, heads).unsqueeze(-1),
        "conv_weight": layer.conv_weights,
        "conv_state": cache.conv[0],
        "cache_indices": slots[:requests],
        "conv_window": cache.intermediate_conv_window[0],
        "window_indices": lines[:requests],
        "a_log": layer.A_log.reshape(heads),
        "dt_bias": layer.dt_bias.reshape(heads, dim),
        "ssm_state": cache.temporal,
        "ring_v": cache.replayssm_rawv,
        "ring_k": cache.replayssm_rawk,
        "ring_g": cache.replayssm_g,
        "ring_beta": cache.replayssm_beta,
        "scale": dim**-0.5,
        **(
            {"lower_bound": layer.lower_bound}
            if layer.lower_bound is not None
            else {"softplus_beta": 1.0, "softplus_threshold": 20.0}
        ),
    }
    output = kimi_k3_sgl_kda_verify(**arguments)
    if isinstance(output, FallbackReason):
        _once(_with_shapes(output, arguments))
        return output
    return output.view(1, seq_len, heads, dim)


def _with_shapes(reason, arguments) -> str:
    """The op's reason with each tensor's shape and dtype, for the log."""
    tensors = ", ".join(
        f"{n} {tuple(t.shape)} {str(t.dtype).rpartition('.')[2]}"
        for n, t in arguments.items()
        if isinstance(t, torch.Tensor)
    )
    return f"{reason.value}: {tensors}"
