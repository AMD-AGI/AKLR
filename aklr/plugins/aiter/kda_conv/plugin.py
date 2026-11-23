# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA prefill short convolution as vLLM runs it on ROCm: ``kda.py``'s fused q/k/v causal conv1d (aiter's
``causal_conv1d_split_qkv_triton_tile_fn``) over the packed q|k|v projection, every sequence's tokens back to back, each
starting from its slot of the paged convolution state (or from zeros) and leaving its last width - 1 inputs there.

vLLM's operands as AKLR takes them: the packed projection is the q|k|v columns of a wider row (``x`` of
(n_tokens, 3, proj), rows apart by more than a row), and the convolution state is vLLM's SD layout (per slot, its
width - 1 inputs by channel, dim innermost) in pages shared with the recurrent state (slots apart by more than a slot)."""

from __future__ import annotations

import torch

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import (
    CuSeqlensDistribution,
    DType,
    FiniteDistribution,
    IntRangeDistribution,
    LecunNormalDistribution,
    StridedRows,
    ToleranceBound,
)
from aklr.core.contract.spec import (
    DTypeSpec,
    LayoutFact,
    LayoutVariant,
    LayoutVariantsSpec,
    TensorSpec,
    axis,
    derived_axis,
)
from aklr.plugins import Plugin
from aklr.plugins.aiter._candidate import offer, served

__all__ = ["KIMI_K3_KDA_CONV", "declare", "family", "kimi_k3_kda_conv"]


def _storage_facts(value: str) -> tuple[LayoutFact, ...]:
    return (LayoutFact(location="arg", name="storage", expected_value=value),)


_BF16 = DTypeSpec(any_of=(DType.BFLOAT16,))
_SAME_AS_X = DTypeSpec(same_as_tensor="x")
_FINITE = FiniteDistribution()
# a bf16 rounding of a float32 sum of four products and the silu: about 2**-8 relative, and small outputs absolute
_OUT_BOUND = ToleranceBound(absolute_tolerance=2**-9, relative_tolerance=2**-7)

KIMI_K3_KDA_CONV = OpContract(
    contract_id="kimi_k3_kda_conv@1.0",
    description=(
        "Kimi-K3's KDA short convolution, every sequence's tokens back to back: sequence s owns rows "
        "cu_seqlens[s]:cu_seqlens[s + 1] of x and slot state_indices[s] of conv_state. Per channel c of the packed "
        "q|k|v (c = part * proj + j for x[:, part, j]), the window is conv_state[slot, :, c] (when "
        "has_initial_state[s], else width - 1 zeros) then the sequence's inputs; token t's y[t, c] = "
        "silu(sum_j weight[c, j] * window[t + j]) in float32, rounded to x's dtype; the window's last width - 1 "
        "entries are written back to conv_state[slot, :, c]. q = y[:, :proj], k = y[:, proj:2 * proj], "
        "v = y[:, 2 * proj:]. Slots no sequence names are left unchanged."
    ),
    op_type="kda_conv",
    axes={
        "n_tokens": axis("Tokens of every sequence, back to back.", 1, 131072, is_var=True),
        "n_seqs": axis("Sequences.", 1, 1024, is_var=True),
        "n_offsets": axis("Sequence offsets: sequences + 1.", 2, 1025, is_var=True),
        # reached through state_indices: any cache size serves
        "n_slots": axis("Slots of the convolution state.", 1, 65536, is_var=True, any_size=True),
        "proj": axis("Channels of each of q, k and v on this rank.", 128, 8192, multiple_of=128),
        "width": axis("The convolution's taps.", 4, 4),
        "dim": derived_axis("Channels of the packed q|k|v.", "3 * proj"),
        "state_len": derived_axis("Inputs a slot keeps.", "width - 1"),
    },
    constraints=("n_offsets == n_seqs + 1", "n_seqs <= n_tokens", "n_seqs <= n_slots"),
    inputs={
        "x": TensorSpec(
            description="The packed q|k|v projection before the convolution: x[t, 0] is q's, x[t, 1] k's, x[t, 2] v's.",
            shape=("n_tokens", 3, "proj"),
            dtype=_BF16,
            distribution=_FINITE,
        ),
        "weight": TensorSpec(
            description="Each packed channel's taps, the oldest input's first; float32, as vLLM keeps Kimi-K3's.",
            shape=("dim", "width"),
            dtype=DTypeSpec(any_of=(DType.FLOAT32,)),
            distribution=LecunNormalDistribution(fan_in="width"),
            lifetime="weight",
        ),
        "conv_state": TensorSpec(
            description="The paged convolution state, vLLM's SD layout: per slot, its last width - 1 inputs, oldest "
            "first, each a row of every packed channel (vLLM's slot may hold more rows, for speculative decoding: the "
            "first width - 1 are these).",
            shape=("n_slots", "state_len", "dim"),
            dtype=_SAME_AS_X,
            distribution=_FINITE,
            mutable=True,
            # copies of inputs
            error_bounds=ToleranceBound(absolute_tolerance=0, relative_tolerance=0),
        ),
        "state_indices": TensorSpec(
            description="Each sequence's slot, distinct.",
            shape=("n_seqs",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=IntRangeDistribution(lo=0, hi="n_slots - 1", distinct=True),
        ),
        # a device tensor in vLLM: no Point tells fresh sequences from resumed ones
        "has_initial_state": TensorSpec(
            description="Whether each sequence resumes from its slot's state, else from zeros.",
            shape=("n_seqs",),
            dtype=DTypeSpec(any_of=(DType.BOOL,)),
            distribution=_FINITE,
        ),
        "cu_seqlens": TensorSpec(
            description="Where each sequence's tokens start, then where the last ends.",
            shape=("n_offsets",),
            dtype=DTypeSpec(any_of=(DType.INT32,)),
            distribution=CuSeqlensDistribution(min_seqlen=1, total="n_tokens"),
        ),
    },
    outputs={
        "q": TensorSpec(
            description="Queries after the convolution.",
            shape=("n_tokens", "proj"),
            dtype=_SAME_AS_X,
            error_bounds=_OUT_BOUND,
        ),
        "k": TensorSpec(
            description="Keys after the convolution.",
            shape=("n_tokens", "proj"),
            dtype=_SAME_AS_X,
            error_bounds=_OUT_BOUND,
        ),
        "v": TensorSpec(
            description="Values after the convolution.",
            shape=("n_tokens", "proj"),
            dtype=_SAME_AS_X,
            error_bounds=_OUT_BOUND,
        ),
    },
    layout_variants=LayoutVariantsSpec(
        by_name={
            "row_major": LayoutVariant(detect_facts=_storage_facts("row_major")),
            # vLLM's: x the q|k|v columns of a wider projection row, conv_state slots apart by a page
            "strided_rows": LayoutVariant(
                layouts={"x": StridedRows(), "conv_state": StridedRows()},
                detect_facts=_storage_facts("strided_rows"),
            ),
        }
    ),
)


family = Plugin("aiter")
family.define(KIMI_K3_KDA_CONV)
family.build()


@family.op
def kimi_k3_kda_conv(
    x,
    weight,
    conv_state,
    state_indices,
    has_initial_state,
    cu_seqlens,
    storage="row_major",
    out_dtype=None,
    out=None,
): ...


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


def declare(plugin: Plugin) -> None:
    plugin.provide("causal_conv1d_split_qkv", version=1, select=_select_causal_conv1d_split_qkv)
