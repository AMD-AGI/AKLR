# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.kda_decode.contract import KIMI_K3_KDA_DECODE, KIMI_K3_KDA_DECODE_SPEC

__all__ = ["kimi_k3_kda_decode", "kimi_k3_kda_decode_spec"]

_LOGGER = logging.getLogger(__name__)
_SEEN: set[str] = set()
# The op's output rows, one buffer per (device, stream, heads, head_dim, dtype), its first n_tokens rows a call's:
# vLLM copies them out (into its core_attn_out) right after the call, on the same stream, before the next layer's
# call writes them. Grown only outside a graph capture, so that a capture allocates nothing; never freed, since a
# captured graph keeps writing the buffer it was captured with (a grown-out one is kept in _RETIRED)
_OUT: dict[tuple, torch.Tensor] = {}
_RETIRED: list[torch.Tensor] = []


def _once(reason: str) -> None:
    """Log each distinct reason the route hands a call back to vLLM, once per process."""
    if reason not in _SEEN:
        _SEEN.add(reason)
        _LOGGER.warning("aklr kda_decode route falls back: %s", reason)


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _stream_id(device: torch.device) -> int:
    return torch._C._cuda_getCurrentRawStream(device.index) if device.type == "cuda" else 0


def _out_rows(n: int, heads: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    """The first n rows [n, heads, dim] of this stream's output buffer; None inside a capture that finds it too
    small (outside one it grows, by doubling)."""
    key = (device, _stream_id(device), heads, dim, dtype)
    buf = _OUT.get(key)
    if buf is None or buf.shape[0] < n:
        if _capturing(device):
            return None
        if buf is not None:
            _RETIRED.append(buf)
        size = n if buf is None else max(n, 2 * buf.shape[0])
        buf = _OUT[key] = torch.empty((size, heads, dim), dtype=dtype, device=device)
    return buf[:n]


def _gate_and_state_checks(raw_g, raw_beta, A_log, dt_bias, initial_state, n, heads, dim):
    """(name, failed) of what both routes take as vLLM's layer passes it: the gate [1, n, heads, dim] dense in a
    token, beta [1, n, heads] dense in its heads, A_log and dt_bias per head, the float32 state dense in a slot."""
    return (
        ("raw_g shape", tuple(raw_g.shape) != (1, n, heads, dim)),
        ("raw_g not dense in a token", raw_g.stride()[2:] != (dim, 1)),
        ("raw_beta shape", tuple(raw_beta.shape) != (1, n, heads)),
        ("raw_beta heads not dense", raw_beta.stride(2) != 1),
        ("state not dense in a slot", initial_state.stride()[1:] != (dim * dim, dim, 1)),
        ("A_log", A_log.shape != (heads,)),
        ("dt_bias", dt_bias.numel() != heads * dim or not dt_bias.is_contiguous()),
        ("gate dtypes", (raw_g.dtype, raw_beta.dtype) != (torch.bfloat16,) * 2),
        ("state not float32", initial_state.dtype != torch.float32),
    )


def _dense(indices, what):
    """indices as a dense tensor: itself, a fresh copy outside a graph capture, None inside one (the route copies
    nothing there; vLLM's graphs pass dense ones anyway)."""
    if indices.is_contiguous():
        return indices
    if _capturing(indices.device):
        _once(f"{what} of strides {indices.stride()} inside a graph capture")
        return None
    # a fresh copy, since .contiguous() keeps a one-element vector's stride
    return torch.empty_like(indices, memory_format=torch.contiguous_format).copy_(indices)


@api(KIMI_K3_KDA_DECODE)
def kimi_k3_kda_decode(
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
): ...


@vllm_route(
    "vllm.models.kimi_k3.amd.kda.KimiK3DeltaAttention._forward",
    "vllm.models.kimi_k3.amd.kda.fused_recurrent_kda_packed_decode",
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _fused_recurrent_kda_packed_decode(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    state_indices,
    scale=None,
):
    """Kimi-K3's KDA layer runs a pure-decode batch (no speculative tokens) through vLLM's packed decode kernel after
    the convolution update: the packed q|k|v rows [n, 3 * heads * dim], the gate [1, n, heads, dim], beta [1, n,
    heads] (a column of the fused projection), the paged recurrent state [slots, heads, dim, dim] read and written in
    place at state_indices. Taken where they are in kimi_k3_kda_decode's contract; returns what vLLM's does: the
    output [1, n, heads, dim] and the state."""
    if "entered" not in _SEEN:
        _SEEN.add("entered")
        _LOGGER.warning("aklr kda_decode route reached")
    ranks = (mixed_qkv.dim(), raw_g.dim(), raw_beta.dim(), initial_state.dim(), state_indices.dim())
    if lower_bound is None or ranks != (2, 4, 3, 4, 1):
        _once(f"out of contract (lower_bound {lower_bound}, ranks {ranks})")
        return FallbackReason.OUT_OF_CONTRACT
    n, (heads, dim_v, dim) = mixed_qkv.shape[0], initial_state.shape[1:]
    checks = (
        ("v and k heads differ", dim_v != dim),
        ("mixed_qkv width", mixed_qkv.shape[1] != 3 * heads * dim),
        ("mixed_qkv not dense in a row", mixed_qkv.stride(1) != 1),
        ("mixed_qkv not bf16", mixed_qkv.dtype != torch.bfloat16),
        *_gate_and_state_checks(raw_g, raw_beta, A_log, dt_bias, initial_state, n, heads, dim),
        ("state_indices length", state_indices.shape[0] != n),
        ("state_indices not int32", state_indices.dtype != torch.int32),
    )
    failed = [name for name, bad in checks if bad]
    if failed:
        _once(f"out of contract ({', '.join(failed)}): mixed_qkv {tuple(mixed_qkv.shape)}/{mixed_qkv.stride()} "
              f"{mixed_qkv.dtype}, raw_g {tuple(raw_g.shape)}/{raw_g.stride()}, raw_beta {tuple(raw_beta.shape)}/"
              f"{raw_beta.stride()}, state {tuple(initial_state.shape)}/{initial_state.stride()} "
              f"{initial_state.dtype}, state_indices {tuple(state_indices.shape)} {state_indices.dtype}")  # fmt: skip
        return FallbackReason.OUT_OF_CONTRACT
    device = mixed_qkv.device
    # vLLM's eager batches name their slots by a column of the block table (stride > 1), its graphs by a dense
    # vector: the contract's is dense
    state_indices = _dense(state_indices, "state_indices")
    if state_indices is None:
        return FallbackReason.OUT_OF_CONTRACT
    out = _out_rows(n, heads, dim, mixed_qkv.dtype, device)
    if out is None:
        _once(f"no output rows for {n} tokens kept outside the graph capture")
        return FallbackReason.NOT_WARMED
    result = kimi_k3_kda_decode(
        mixed_qkv.unflatten(1, (3 * heads, dim)),  # (n, 3 heads, dim), rows of the conv output
        raw_g[0],
        raw_beta[0].unsqueeze(-1),  # (n, heads, 1), a column of the fused projection
        A_log,
        dt_bias.view(heads, dim),
        initial_state.flatten(1, 2),  # (slots, heads * dim, dim), each slot a row of a page
        state_indices,
        storage="strided_rows",
        scale=dim**-0.5 if scale is None else scale,
        lower_bound=lower_bound,
        out=out,
    )
    if isinstance(result, FallbackReason):
        _once(
            f"{result.value}: {n} tokens, mixed_qkv stride {mixed_qkv.stride()}, raw_g stride {raw_g.stride()}, "
            f"raw_beta stride {raw_beta.stride()}, state {tuple(initial_state.shape)}/{initial_state.stride()}"
        )
        return result
    return result.unsqueeze(0), initial_state


@api(KIMI_K3_KDA_DECODE_SPEC)
def kimi_k3_kda_decode_spec(
    qkv,
    raw_g,
    raw_beta,
    a_log,
    g_bias,
    state_cache,
    state_indices,
    num_accepted_tokens,
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    out_dtype=None,
    out=None,
): ...


def _packed(q, k, v, heads, dim):
    """q, k and v [1, n, heads, dim] as the one (n, 3 heads, dim) view of the packed rows they split, else None."""
    step = heads * dim * q.element_size()
    if (
        q.stride() != k.stride()
        or q.stride() != v.stride()
        or q.stride()[2:] != (dim, 1)
        or q.stride(1) < 3 * heads * dim
        or (k.data_ptr() - q.data_ptr(), v.data_ptr() - q.data_ptr()) != (step, 2 * step)
    ):
        return None
    return q[0].as_strided((q.shape[1], 3 * heads, dim), (q.stride(1), dim, 1))


@vllm_route(
    "vllm.models.kimi_k3.amd.kda.KimiK3DeltaAttention._forward",
    "vllm.models.kimi_k3.amd.kda.fused_recurrent_kda",
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _fused_recurrent_kda(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens=None,
    uniform_sequence_length=None,
    out=None,
    fuse_gate=None,
):
    """Kimi-K3's KDA layer runs a speculative batch through vLLM's recurrent kernel after the convolution update: q, k
    and v [1, n, heads, dim] split from the packed rows, every sequence a window of uniform_sequence_length tokens
    (vLLM's DSPARK verify windows), each starting from the state in the slot of its accepted token
    (ssm_state_indices[s, num_accepted_tokens[s] - 1]) and writing each token's state to its slot. Taken where they
    are in kimi_k3_kda_decode_spec's contract; returns what vLLM's does: the output [1, n, heads, dim] (out when given)
    and the state. The non-speculative decodes of a mixed batch (one-dimensional ssm_state_indices) stay with vLLM."""
    window = uniform_sequence_length
    ranks = tuple(x.dim() for x in (q, k, v, raw_g, raw_beta, initial_state, ssm_state_indices))
    if (
        num_accepted_tokens is None
        or window is None
        or lower_bound is None
        or fuse_gate is False
        or ranks != (4, 4, 4, 4, 3, 4, 2)
    ):
        _once(f"out of contract (num_accepted_tokens {num_accepted_tokens is not None}, uniform_sequence_length "
              f"{window}, lower_bound {lower_bound}, fuse_gate {fuse_gate}, ranks {ranks})")  # fmt: skip
        return FallbackReason.OUT_OF_CONTRACT
    if "entered spec" not in _SEEN:
        _SEEN.add("entered spec")
        _LOGGER.warning("aklr kda_decode speculative route reached")
    n, (heads, dim_v, dim) = q.shape[1], initial_state.shape[1:]
    n_seqs = cu_seqlens.shape[0] - 1
    qkv = _packed(q, k, v, heads, dim)
    checks = (
        ("v and k heads differ", dim_v != dim),
        ("q|k|v not one packed row", qkv is None),
        ("q, k, v shapes", tuple(q.shape) != (1, n, heads, dim) or k.shape != q.shape or v.shape != q.shape),
        ("q|k|v not bf16", q.dtype != torch.bfloat16),
        *_gate_and_state_checks(raw_g, raw_beta, A_log, dt_bias, initial_state, n, heads, dim),
        ("windows", n_seqs < 1 or n != n_seqs * window),
        ("slots narrower than a window", ssm_state_indices.shape[1] < window or ssm_state_indices.stride(1) != 1),
        ("fewer slot rows or accepted counts than sequences",
         ssm_state_indices.shape[0] < n_seqs or num_accepted_tokens.dim() != 1
         or num_accepted_tokens.shape[0] < n_seqs),
        ("index dtypes", (ssm_state_indices.dtype, num_accepted_tokens.dtype) != (torch.int32,) * 2),
        ("out", out is not None and (tuple(out.shape) != (1, n, heads, dim) or not out.is_contiguous())),
    )  # fmt: skip
    failed = [name for name, bad in checks if bad]
    if failed:
        _once(f"out of contract ({', '.join(failed)}): q {tuple(q.shape)}/{q.stride()} {q.dtype}, raw_g "
              f"{tuple(raw_g.shape)}/{raw_g.stride()}, raw_beta {tuple(raw_beta.shape)}/{raw_beta.stride()}, state "
              f"{tuple(initial_state.shape)}/{initial_state.stride()} {initial_state.dtype}, ssm_state_indices "
              f"{tuple(ssm_state_indices.shape)}/{ssm_state_indices.stride()} {ssm_state_indices.dtype}, "
              f"{n_seqs} sequences of {window}")  # fmt: skip
        return FallbackReason.OUT_OF_CONTRACT
    # the windows' slots, sequence by sequence: vLLM's graphs pass dense [n_seqs, window] rows, its eager batches the
    # block table's
    slots = _dense(ssm_state_indices[:n_seqs, :window], "ssm_state_indices")
    accepted = _dense(num_accepted_tokens[:n_seqs], "num_accepted_tokens")
    if slots is None or accepted is None:
        return FallbackReason.OUT_OF_CONTRACT
    rows = out[0] if out is not None else _out_rows(n, heads, dim, q.dtype, q.device)
    if rows is None:
        _once(f"no output rows for {n} tokens kept outside the graph capture")
        return FallbackReason.NOT_WARMED
    result = kimi_k3_kda_decode_spec(
        qkv,
        raw_g[0],
        raw_beta[0].unsqueeze(-1),  # (n, heads, 1), a column of the fused projection
        A_log,
        dt_bias.view(heads, dim),
        initial_state.flatten(1, 2),  # (slots, heads * dim, dim), each slot a row of a page
        slots.view(-1),
        accepted,
        storage="strided_rows",
        scale=dim**-0.5,
        lower_bound=lower_bound,
        out=rows,
    )
    if isinstance(result, FallbackReason):
        _once(f"{result.value}: {n_seqs} sequences of {window}, q stride {q.stride()}, raw_g stride {raw_g.stride()}, "
              f"raw_beta stride {raw_beta.stride()}, state {tuple(initial_state.shape)}/{initial_state.stride()}")  # fmt: skip
        return result
    return (out if out is not None else result.unsqueeze(0)), initial_state
