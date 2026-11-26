# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sglang 0.5.24.dev809's gfx950 sparse MLA prefill, ``sparse_mla_prefill_gfx950`` as triton_sparse_mla_fwd calls
it on a prefill, LSE asked for, one launch per call: the copy of its kernel in _sparse_mla_prefill_gfx950.hip,
built here as a torch extension, so a sglang without it (CI's release) runs the baseline too;
tests/plugins/sglang/test_sparse_mla_prefill_baseline.py holds the copy to sglang's own kernel bit for bit."""

import functools
from pathlib import Path

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = ["DsaSparseMlaFp8DsPrefillBaseline"]

_SOURCE = Path(__file__).with_name("_sparse_mla_prefill_gfx950.hip")
_LIBRARY = "aklr_baseline_sparse_mla_prefill_gfx950"
# sparse_mla_prefill_gfx950's default pipeline depth, the one triton_sparse_mla_fwd runs
_STAGES = 2


@functools.cache
def _op():
    """The copy's op, its library built on the first call for this process's GPU."""
    from aklr.core.jit import hip_extension
    from aklr.core.platform.device import DeviceTarget
    from aklr.core.runtime.builder.builder import get_build_cache_path

    arch = DeviceTarget.current().arch
    if arch is None:
        raise RuntimeError("sparse_mla_prefill_gfx950: no GPU arch to build for")
    root = get_build_cache_path() / "baselines" / _LIBRARY
    hip_extension.load_library(_LIBRARY, {"sparse_mla_prefill_gfx950.hip": _SOURCE.read_text()}, root, arch)
    return getattr(torch.ops, _LIBRARY).run.default


class DsaSparseMlaFp8DsPrefillBaseline(OpBaseline):
    def baseline(self, *, q_nope, q_rope, kv_cache, indices, sm_scale=192**-0.5, output=None, lse=None):
        """sglang's wrapper's call: its own output and LSE, the pool's rows and the indices as they come."""
        n_tokens, n_heads, _ = q_nope.shape
        out = torch.empty(n_tokens, n_heads, 512, dtype=torch.bfloat16, device=q_nope.device)
        token_lse = torch.empty(n_tokens, n_heads, dtype=torch.float32, device=q_nope.device)
        _op()(q_nope, q_rope, kv_cache, indices, out, token_lse, float(sm_scale), 1, _STAGES)
        return {"output": out, "lse": token_lse}
