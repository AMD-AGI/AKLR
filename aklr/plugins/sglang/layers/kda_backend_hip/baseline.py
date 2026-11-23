# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang's gfx950 chain verify, ``kda_verify_gfx950`` in ring mode, one launch per call: the copy of its kernel
in _kda_verify_gfx950.hip, built here as a torch extension, so a sglang without it (CI's release) runs the
baseline too; tests/plugins/sglang/test_kda_verify_baseline.py holds the copy to sglang 0.5.24.dev809's bits."""

from __future__ import annotations

import functools
from pathlib import Path

from aklr.plugins.sglang.layers.kda_backend_hip.contract import HEAD_DIM

__all__ = ["kda_chain_verify_ring_baseline"]

_SOURCE = Path(__file__).with_name("_kda_verify_gfx950.hip")
_LIBRARY = "aklr_baseline_kda_verify_gfx950"


@functools.cache
def _op():
    """The copy's op, its library built on the first call for this process's GPU."""
    import torch

    from aklr.core.jit import hip_extension
    from aklr.core.platform.device import DeviceTarget
    from aklr.core.runtime.builder.builder import get_build_cache_path

    arch = DeviceTarget.current().arch
    if arch is None:
        raise RuntimeError("kda_verify_gfx950: no GPU arch to build for")
    root = get_build_cache_path() / "baselines" / _LIBRARY
    hip_extension.load_library(_LIBRARY, {"kda_verify_gfx950.hip": _SOURCE.read_text()}, root, arch)
    return getattr(torch.ops, _LIBRARY).run.default


def _octet_rows(t):
    """t as sglang's wrapper hands it over: rows 16-byte aligned for the kernel's 8-channel loads."""
    if t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.contiguous()


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
    """sglang's call on the same pools, viewed as kda_chain_verify_gfx950 hands them over, through
    kda_verify_gfx950's ring path: BV 4 for up to two requests, else 8; its own output, whose padding
    requests' rows it leaves unwritten (sglang reads none of them)."""
    n_tokens, heads = b.shape
    steps = conv_window.shape[1]
    o = mixed_qkv.new_empty(n_tokens, heads, HEAD_DIM)
    _op()(
        _octet_rows(mixed_qkv),
        conv_weight,
        conv_state.transpose(-1, -2),
        cache_indices,
        conv_window.transpose(-1, -2),
        window_indices,
        _octet_rows(a.reshape(n_tokens, heads * HEAD_DIM)),
        _octet_rows(b.reshape(n_tokens, heads)),
        a_log,
        dt_bias,
        ssm_state,
        ring_v,
        ring_k,
        ring_g,
        ring_beta,
        o,
        heads,
        4 if n_tokens // steps <= 2 else 8,
        steps,
        lower_bound is not None,
        float(scale),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(softplus_beta) if softplus_beta is not None else 1.0,
        float(softplus_threshold) if softplus_threshold is not None else 20.0,
    )
    return o
