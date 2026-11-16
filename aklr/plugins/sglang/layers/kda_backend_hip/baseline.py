# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's gfx950 chain verify, ``kda_verify_gfx950`` in ring mode, one launch per call."""

from __future__ import annotations

from aklr.plugins.sglang.layers.kda_backend_hip.contract import HEAD_DIM

__all__ = ["kda_chain_verify_ring_baseline"]


def kda_chain_verify_ring_baseline(
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
    scale=HEAD_DIM**-0.5,
    lower_bound=None,
    softplus_beta=None,
    softplus_threshold=None,
    output=None,
):
    """sglang's call on the same pools, viewed as kda_chain_verify_gfx950 hands them over; its own
    output, whose padding requests' rows it leaves unwritten (sglang reads none of them)."""
    from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import kda_verify_gfx950

    n_tokens, heads = b.shape
    softplus = (
        {}
        if lower_bound is not None
        else {"softplus_beta": softplus_beta, "softplus_threshold": softplus_threshold}
    )
    result = kda_verify_gfx950(
        mixed_qkv=mixed_qkv,
        conv_weight=conv_weight,
        conv_state=conv_state.transpose(-1, -2),
        intermediate_conv_window=conv_window.transpose(-1, -2),
        intermediate_state_indices=window_indices,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        ssm_states=ssm_state,
        cache_indices=cache_indices,
        intermediate_states_buffer=None,
        scale=scale,
        T=conv_window.shape[1],
        num_q_heads=heads,
        num_v_heads=heads,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        lower_bound=lower_bound,
        cache_ring=True,
        replayssm_rawv=ring_v,
        replayssm_rawk=ring_k,
        replayssm_g=ring_g,
        replayssm_beta=ring_beta,
        **softplus,
    )
    return result.view(n_tokens, heads, HEAD_DIM)
