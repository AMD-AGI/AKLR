# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM runs: aiter's ``causal_conv1d_split_qkv_triton_tile_fn`` with vLLM's block_m=8, on x as vLLM hands it
([dim, n_tokens], a transposed view of the token rows) and the convolution state as vLLM's (n_slots, dim, state_len)
view. vLLM passes its prefill metadata, whose chunk schedule is built once per step: the baseline keeps one per call
shape, so that no timed call builds it (a host sync)."""

from __future__ import annotations

from types import SimpleNamespace

__all__ = ["kimi_k3_kda_conv_baseline"]

_SCHEDULES: dict[tuple, SimpleNamespace] = {}


def kimi_k3_kda_conv_baseline(
    x,
    weight,
    conv_state,
    state_indices,
    has_initial_state,
    cu_seqlens,
    storage="row_major",
    q=None,
    k=None,
    v=None,
):
    """aiter's q, k and v; conv_state is written in place, as in vLLM."""
    import inspect

    from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn as conv

    proj = x.shape[2]
    key = (cu_seqlens.data_ptr(), x.shape[0], cu_seqlens.shape[0])
    metadata = _SCHEDULES.setdefault(key, SimpleNamespace(nums_dict={}))
    extra = (
        {"preserve_input_dtype": True} if "preserve_input_dtype" in inspect.signature(conv).parameters else {}
    )
    return conv(
        x.flatten(1).t(),
        weight,
        None,
        conv_state.transpose(1, 2),
        cu_seqlens,
        proj,
        proj,
        cache_indices=state_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        block_m=8,
        metadata=metadata,
        **extra,
    )
