# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A ratio-2 compressor's projection from the indexer query's launch: within ``given``,
``DeepseekV41Compressor.project_fused`` returns the ``wkv_gate`` GEMV's fp32 sums that
launch computed (as ``linear_bf16_fp32``, another order); elsewhere it is sglang's."""

from __future__ import annotations

import contextlib
import contextvars

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

SGLANG_READS = {
    "srt/layers/attention/dsv4/dsv41_sparse.py": (
        "DeepseekV41Compressor",
        "project_fused(x)",
        "use_fused_gate",
        "wkv_gate",
    ),
    "kernels/ops/gemm/bf16_fp32.py": ("linear_bf16_fp32",),
}

_GIVEN = contextvars.ContextVar("aklr_dsv41_compressor_kv_score", default=None)
# where a captured step's ratio-2 projection came from
INFO = "deepseek_v4.compressor:projection"
_NOTED: set[str] = set()


@contextlib.contextmanager
def given(compressor, x, kv_score):
    """``compressor.project_fused(x)`` is ``kv_score`` inside (no-op for None)."""
    token = _GIVEN.set(None if kv_score is None else (compressor, x, kv_score))
    try:
        yield
    finally:
        _GIVEN.reset(token)


@sglang_route(
    "sglang.srt.layers.attention.dsv4.dsv41_sparse.DeepseekV41Compressor.project_fused",
    versions=VERSIONS,
    around=True,
)
def _project_fused(orig, self, x):
    held = _GIVEN.get()
    if held is not None and held[0] is self and held[1] is x:
        _note("in the indexer query's launch")
        return held[2]
    if held is not None:
        _note("sglang's (the query's launch held another input)")
    return orig(self, x)


def _note(how: str) -> None:
    if how not in _NOTED and torch.cuda.is_current_stream_capturing():
        _NOTED.add(how)
        stats.set_info(INFO, "; ".join(sorted(_NOTED)))
