###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The Baseline of the kda_conv op: the backend call its schemas' standards time a kernel against."""

from __future__ import annotations


def causal_conv1d_split_qkv_baseline(
    x, weight, conv_state, state_indices, has_initial_state, cu_seqlens, **_
):
    """What vLLM runs: aiter's causal_conv1d_split_qkv_triton_tile_fn with vLLM's block_m=8, on x as vLLM hands
    it ((dim, n_tokens), a transposed view of the token rows) and the convolution state as vLLM's
    (n_slots, dim, state_len) view, written in place. vLLM passes its prefill metadata, whose chunk schedule
    is built once a step: one is kept here, so that only the first call builds it (a host sync)."""
    import inspect
    from types import SimpleNamespace

    from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn

    proj = x.shape[2]
    metadata = SimpleNamespace(nums_dict={})
    parameters = inspect.signature(causal_conv1d_split_qkv_triton_tile_fn).parameters
    extra = {"preserve_input_dtype": True} if "preserve_input_dtype" in parameters else {}
    return lambda: causal_conv1d_split_qkv_triton_tile_fn(
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
