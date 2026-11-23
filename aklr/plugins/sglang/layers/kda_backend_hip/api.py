# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.layers.kda_backend_hip.contract import KDA_CHAIN_VERIFY_RING

__all__ = ["kda_chain_verify_ring"]


@api(KDA_CHAIN_VERIFY_RING)
def kda_chain_verify_ring(
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
    scale=128**-0.5,
    lower_bound=None,
    softplus_beta=None,
    softplus_threshold=None,
    output=None,
): ...


@sglang_route(
    "sglang.kernels.ops.attention.linear.kda_verify_gfx950.kda_verify_gfx950",
    versions=("0.5.24.dev809",),
)
def _kda_verify_gfx950(
    mixed_qkv,
    conv_weight,
    conv_state,
    intermediate_conv_window,
    intermediate_state_indices,
    a,
    b,
    A_log,  # noqa: N803 -- sglang's name
    dt_bias,
    ssm_states,
    cache_indices,
    intermediate_states_buffer,
    scale,
    T,  # noqa: N803 -- sglang's name
    num_q_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    lower_bound=None,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    use_qk_l2norm_in_kernel=True,
    cache_ring=False,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
):
    """The ring mode, as kda_chain_verify_gfx950 calls it once sglang's own checks pass: the op on
    the pools' storage (conv_state and the windows as sglang allocates them, [.., 3, dim]); the
    target's [1, seq_len, heads, head_v_dim] output. The per-step snapshot mode, and a call the
    target would refuse, stay on sglang."""
    seq_len, heads = mixed_qkv.shape[0], num_v_heads
    if (
        not cache_ring
        or intermediate_states_buffer is not None
        or not use_qk_l2norm_in_kernel
        or num_q_heads != num_v_heads
        or head_k_dim != head_v_dim
        or a.numel() != seq_len * heads * head_k_dim
        or b.numel() != seq_len * heads
        or A_log.numel() != heads
        or dt_bias.numel() != heads * head_k_dim
    ):
        return FallbackReason.OUT_OF_CONTRACT
    arguments = {
        "mixed_qkv": mixed_qkv,
        "conv_weight": conv_weight,
        "conv_state": conv_state.transpose(-1, -2),
        "cache_indices": cache_indices,
        "conv_window": intermediate_conv_window.transpose(-1, -2),
        "window_indices": intermediate_state_indices,
        "a": a.reshape(seq_len, heads, head_k_dim),
        "b": b.reshape(seq_len, heads),
        "a_log": A_log.reshape(heads),
        "dt_bias": dt_bias.reshape(heads, head_k_dim),
        "ssm_state": ssm_states,
        "ring_v": replayssm_rawv,
        "ring_k": replayssm_rawk,
        "ring_g": replayssm_g,
        "ring_beta": replayssm_beta,
        "scale": scale,
        **(
            {"lower_bound": lower_bound}
            if lower_bound is not None
            else {"softplus_beta": softplus_beta, "softplus_threshold": softplus_threshold}
        ),
    }
    # sglang's projections are often rows of a wider output (the split of its fused in-proj),
    # which the op takes dense: ask it with dense stand-ins before copying them
    strided = [n for n in ("mixed_qkv", "a", "b") if not arguments[n].is_contiguous()]
    if strided:
        reason = kda_chain_verify_ring.fallback_reason(
            **arguments
            | {n: torch.empty_like(arguments[n], memory_format=torch.contiguous_format) for n in strided}
        )
        if reason is not None:
            return reason
        arguments |= {n: arguments[n].contiguous() for n in strided}
    output = kda_chain_verify_ring(**arguments)
    if isinstance(output, FallbackReason):
        return output
    return output.view(1, seq_len, heads, head_v_dim)
