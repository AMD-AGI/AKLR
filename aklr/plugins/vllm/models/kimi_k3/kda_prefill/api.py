# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

import torch

from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.kda_prefill.contract import (
    CHECKPOINT_CHUNK,
    KIMI_K3_KDA_PREFILL,
    KIMI_K3_KDA_PREFILL_CHECKPOINT,
)

__all__ = ["kimi_k3_kda_prefill", "kimi_k3_kda_prefill_checkpoint"]


@api(KIMI_K3_KDA_PREFILL)
def kimi_k3_kda_prefill(
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
): ...


@api(KIMI_K3_KDA_PREFILL_CHECKPOINT)
def kimi_k3_kda_prefill_checkpoint(
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
): ...


# The table's per-shape refusals (no kernel measured at this length, or none fast enough), remembered so that a
# declined layer skips the stand-ins and the table's key; exporting counts every call, so nothing is kept then
_TABLE_REASONS = frozenset(
    (
        FallbackReason.NO_WORKLOAD,
        FallbackReason.NOT_QUALIFIED,
        FallbackReason.NOT_MEASURED,
        FallbackReason.NO_SCHEMA,
    )
)
_DECLINED: dict[tuple, FallbackReason] = {}
# shapes the table served last time: their layer skips the question (dispatch asks it again anyway, so a stale entry
# costs the launches made before it, never a wrong result)
_SERVED: set[tuple] = set()
# checkpoint staging rows, one buffer per (row shape, device, stream) grown to the most sequences seen, reused layer to
# layer: each layer's snapshot copy is queued on that stream before the next layer's kernel writes them; never kept
# from inside a graph capture
_STAGING: dict[tuple, torch.Tensor] = {}


@functools.cache
def _remembers():
    """Whether the route keeps the table's answers: not while exporting (RuntimeEnv's AKLR_EXPORT), read once."""
    return not RuntimeEnv().is_export_enabled


def _remember(key, reason):
    if _remembers() and reason in _TABLE_REASONS:
        if len(_DECLINED) >= 4096:
            _DECLINED.clear()
        _DECLINED[key] = reason
    return reason


def _served(key, result):
    """Remember (or forget) that the table served ``key``; returns ``result``."""
    if not _remembers():
        return result
    if isinstance(result, FallbackReason):
        _SERVED.discard(key)
    else:
        if len(_SERVED) >= 4096:
            _SERVED.clear()
        _SERVED.add(key)
    return result


def _stream_id(device):
    return torch._C._cuda_getCurrentRawStream(device.index) if device.type == "cuda" else 0


def _staging(n, rows, device):
    """Zero-free float32 staging rows [n, *rows] on device: outside a capture, the first n of a kept buffer."""
    if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return torch.empty((n, *rows), dtype=torch.float32, device=device)
    key = (rows, device.index, _stream_id(device))
    buf = _STAGING.get(key)
    if buf is None or buf.shape[0] < n:
        if buf is None and len(_STAGING) >= 16:
            _STAGING.clear()
        buf = _STAGING[key] = torch.empty((n, *rows), dtype=torch.float32, device=device)
    return buf[:n]


@functools.cache
def _plan():
    """_plan's (checkpoint_plan, scatter_rows), imported at the first snapshot (Triton's kernels)."""
    from aklr.plugins.vllm.models.kimi_k3.kda_prefill._plan import checkpoint_plan, scatter_rows

    return checkpoint_plan, scatter_rows


def _checkpoint_plan(offsets, indices, null):
    """vLLM's checkpoint offsets in chunks of CHECKPOINT_CHUNK (0 where none is taken: not a whole chunk, or no
    cache row to write it to) and each sequence's cache row (-1 where none is taken), in one launch."""
    return _plan()[0](offsets, indices, null, CHECKPOINT_CHUNK)


@functools.cache
def _null_block_id():
    """vLLM's NULL_BLOCK_ID: the cache row no request owns, which vLLM's kernel takes for no snapshot."""
    try:
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    except ImportError:
        return 0
    return NULL_BLOCK_ID


def _contiguous_stand_in(x):
    """A contiguous tensor of x's shape, dtype and device for the dispatch question alone (its values are never
    read): a view of x's own storage when it fits there, else a new one."""
    strides = [1] * x.dim()
    for i in range(x.dim() - 2, -1, -1):
        strides[i] = strides[i + 1] * x.shape[i + 1]
    if (
        x.is_contiguous()
        or (x.storage_offset() + x.numel()) * x.element_size() <= x.untyped_storage().nbytes()
    ):
        return x.as_strided(x.shape, strides)
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@vllm_route(
    "vllm.models.kimi_k3.amd.kda.KimiK3DeltaAttention._forward",
    "vllm.models.kimi_k3.amd.kda.chunk_kda_prefill",
    versions=("0.34.0", "0.32.1rc1.dev276", "0.0.0+mum.14d1ec2336"),
)
def _chunk_kda_prefill(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    g_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    lower_bound=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets=None,
    backend="triton",
    out=None,
    checkpoint_state=None,
    checkpoint_offsets=None,
    checkpoint_state_indices=None,
    state_cache=None,
    state_indices=None,
    has_initial_state=None,
):
    """Kimi-K3's KDA prefill as its layer calls it on the fused backend: the paged state read and
    written in place (state_cache), q and k normalized and the gate and beta activated inside, and
    vLLM's mid-prefill state snapshots (checkpoint_*, under prefix caching) taken by
    kimi_k3_kda_prefill_checkpoint. vLLM's other backends and calls that pass or want a state outside
    the cache stay with vLLM. Returns what vLLM does there: the output, [1, n_tokens, n_heads,
    head_dim], and no final state.

    Where vLLM 0.34.0 reaches it (kda.py, KimiK3DeltaAttention): on every step holding a prefill when
    the layer's prefill backend is fused, i.e. additional_config kda_prefill_backend "fused", or "auto"
    where aiter's Gluon chunk_kda does not serve (it does with a float32 recurrent state, a gate lower
    bound, VLLM_ROCM_USE_AITER, CDNA4 and aiter.ops.triton.attention.chunk_kda importable). The
    snapshot calls (kimi_k3_kda_prefill_checkpoint) come with prefix caching, for which vLLM sets
    mamba_cache_mode "align" on a hybrid model, and kda_prefill_checkpoint on (its default): a step
    whose metadata asks for a snapshot calls with backend "fused" whatever the configured backend is
    (the Gluon one hands those steps over where vLLM has the fused kernel). 0.32.1rc1.dev276 passes the
    configured backend there and takes snapshots only on the fused one."""
    checkpoint = (checkpoint_state, checkpoint_offsets, checkpoint_state_indices)
    if (
        backend != "fused"
        or not use_qk_l2norm_in_kernel
        # all three or none, as kda.py passes them
        or (any(c is None for c in checkpoint) and any(c is not None for c in checkpoint))
        or initial_state is not None
        or output_final_state
        or state_cache is None
        or state_indices is None
        or has_initial_state is None
        or cu_seqlens is None
        or g_bias is None
        or lower_bound is None
        or q.dim() != 4
        or q.shape[0] != 1
    ):
        return FallbackReason.OUT_OF_CONTRACT
    n_heads, head_dim = q.shape[2], q.shape[3]
    # a slot's heads must sit v by v, k by k, so that they view as one dense block of rows
    slot = state_cache.shape[1:]
    if (
        slot not in ((n_heads, head_dim, head_dim), (n_heads * head_dim, head_dim))
        or state_cache.stride()[1:] != (head_dim * head_dim, head_dim, 1)[-len(slot) :]
        or not g_bias.is_contiguous()
        or g_bias.numel() != n_heads * head_dim
        or (checkpoint_state is not None and checkpoint_state.shape[1:] != state_cache.shape[1:])
        or (checkpoint_state is not None and checkpoint_state.stride()[1:] != state_cache.stride()[1:])
    ):
        return FallbackReason.OUT_OF_CONTRACT
    if scale is None:
        scale = k.shape[-1] ** -0.5
    memo = (q.shape, q.stride(), q.dtype, raw_beta.stride(), state_cache.shape, state_cache.stride(),
            cu_seqlens.shape, checkpoint_offsets is None, scale, lower_bound, out is None)  # fmt: skip
    if memo in _DECLINED:
        return _DECLINED[memo]
    # served last time: skip the question and its stand-ins, dispatch asks again anyway
    served = memo in _SERVED
    rows = (n_heads * head_dim, head_dim)
    beta = raw_beta[0]
    common = [
        q[0],
        k[0],
        v[0],
        raw_g[0],
        # a column slice of the layer's fused projection: asked as its contiguous stand-in, copied only to be served
        beta.contiguous() if served else _contiguous_stand_in(beta),
        A_log,
        g_bias.view(n_heads, head_dim),
        # each slot's heads, v by v, in a page that also holds the convolution state
        state_cache.view(state_cache.shape[0], *rows),
        state_indices,
        has_initial_state,
        cu_seqlens,
    ]
    options = {"state_storage": "strided_rows", "scale": scale, "lower_bound": lower_bound}
    options["out"] = None if out is None else out[0]
    if checkpoint_offsets is None:
        # most calls fall back (no workload at their size): ask first, so that a fallback costs no launch
        if not served:
            reason = kimi_k3_kda_prefill.fallback_reason(*common, **options)
            if reason is not None:
                return _remember(memo, reason)
            common[4] = beta.contiguous()
        result = _served(memo, kimi_k3_kda_prefill(*common, **options))
    else:
        # the same question first, on uninitialized stand-ins (an allocation, no launch); then vLLM's offsets in
        # chunks, 0 where none is taken (not a whole chunk, or no row to write it to); each snapshot lands in a
        # staging row, then those taken in their cache rows at once, and no row for one not taken (as vLLM's
        # kernel, which skips a negative or NULL_BLOCK_ID row): nothing waits on the device
        staging = _staging(checkpoint_offsets.shape[0], rows, q.device)
        if not served:
            stand_in = torch.empty(checkpoint_offsets.shape, dtype=torch.int32, device=q.device)
            reason = kimi_k3_kda_prefill_checkpoint.fallback_reason(*common, stand_in, staging, **options)
            if reason is not None:
                return _remember(memo, reason)
            common[4] = beta.contiguous()
        # one launch for both (a prefill asks this on every KDA layer; a chain of elementwise ops here cost a dozen
        # launches a layer, and the lagging host of one rank the group's allreduces); a staging row not taken is
        # never read nor copied, so the staging needs no zeroing
        chunks, rows_to = _checkpoint_plan(checkpoint_offsets, checkpoint_state_indices, _null_block_id())
        result = _served(memo, kimi_k3_kda_prefill_checkpoint(*common, chunks, staging, **options))
        if not isinstance(result, FallbackReason):
            # one launch: each taken snapshot into its cache row, the rows -1 left alone
            _plan()[1](staging, rows_to, checkpoint_state.view(checkpoint_state.shape[0], *rows))
    if isinstance(result, FallbackReason):
        return result
    return (out if out is not None else result.unsqueeze(0)), None
