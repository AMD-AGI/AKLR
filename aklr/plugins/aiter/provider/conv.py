# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter's prefill causal conv1d with the q/k/v split (``causal_conv1d_split_qkv``, its Triton tile
kernel, which Kimi-K3's KDA prefill calls on vLLM), served by AKLR's ``kimi_k3_kda_conv``. aiter takes
x as [dim, n_tokens] and the state as (slots, dim, state_len): on vLLM, views of the packed q|k|v
projection rows and of its SD state pages, which AKLR's contract takes as the strided rows they are."""

from __future__ import annotations

import torch

from aklr.ops.kda_conv.api import kimi_k3_kda_conv
from aklr.plugins.aiter.provider._base import offer, selector, served

_WIDTH = 4  # the contract's taps


def _vector(t, dtype) -> bool:
    return t is not None and t.dim() == 1 and t.dtype == dtype


def _dense(t, *, copy: bool):
    """``t`` with a stride of one: vLLM can hand a column of a wider table (a one-element one too,
    whose stride ``.contiguous()`` keeps). select asks with an unfilled one, which it does not read."""
    if t.stride(0) == 1:
        return t
    dense = torch.empty_like(t, memory_format=torch.contiguous_format)
    return dense.copy_(t) if copy else dense


def _operands(x, conv_states, k_dim, cache_indices, has_initial_state, query_start_loc, *, copy: bool):
    """AKLR's operands over aiter's: x's rows (n_tokens, 3, proj), each slot's first width - 1 state
    rows (n_slots, width - 1, dim; vLLM's slot keeps more for speculative decoding) and dense index
    vectors."""
    rows = x.t().unflatten(1, (3, k_dim))
    state = conv_states.transpose(1, 2)[:, : _WIDTH - 1]
    vectors = (_dense(t, copy=copy) for t in (cache_indices, has_initial_state, query_start_loc))
    return rows, state, *vectors


@selector("causal_conv1d_split_qkv")
def _select_causal_conv1d_split_qkv(args, kwargs):
    (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        k_dim,
        v_dim,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        _block_m,
        _block_n,
        _num_warps,
        _metadata,
        _preserve_input_dtype,
    ) = args
    # block_m, block_n and num_warps tile aiter's kernel, and metadata memoizes its chunk schedule.
    # aiter applies silu (swish is its other name) or nothing, makes up the index vectors when given
    # none, reads x in the state's dtype and returns q, k and v in either (preserve_input_dtype), and
    # skips a sequence whose slot is pad_slot_id, a slot when it is not negative. AKLR's contract:
    # silu, a bf16 x and state of one dtype, x channel-contiguous (the transpose of q|k|v rows), the
    # state vLLM's SD pages, float32 taps, width 4
    if (
        bias is not None
        or activation not in ("silu", "swish")
        or k_dim != v_dim
        or (pad_slot_id is not None and pad_slot_id >= 0)
        or x.dim() != 2
        or x.shape[0] != 3 * k_dim
        or x.stride(0) != 1
        or weight.dtype != torch.float32
        or tuple(weight.shape) != (x.shape[0], _WIDTH)
        or conv_states.dim() != 3
        or conv_states.dtype != x.dtype
        or conv_states.shape[1] != x.shape[0]
        or conv_states.stride(1) != 1
        or conv_states.shape[2] < _WIDTH - 1
        or not _vector(cache_indices, torch.int32)
        or not _vector(has_initial_state, torch.bool)
        or not _vector(query_start_loc, torch.int32)
    ):
        return None
    rows, state, indices, resumes, offsets = _operands(
        x, conv_states, k_dim, cache_indices, has_initial_state, query_start_loc, copy=False
    )
    return offer(
        kimi_k3_kda_conv.serving(rows, weight, state, indices, resumes, offsets, storage="strided_rows"),
        _run_causal_conv1d_split_qkv,
    )


def _run_causal_conv1d_split_qkv(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    k_dim,
    v_dim,
    cache_indices,
    has_initial_state,
    activation,
    pad_slot_id,
    block_m,
    block_n,
    num_warps,
    metadata,
    preserve_input_dtype,
):
    rows, state, indices, resumes, offsets = _operands(
        x, conv_states, k_dim, cache_indices, has_initial_state, query_start_loc, copy=True
    )
    q, k, v = served(kimi_k3_kda_conv(rows, weight, state, indices, resumes, offsets, storage="strided_rows"))
    return q, k, v
