# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23 runs on HIP for the DSA indexer's prefill logits: aiter's fp8_mqa_logits
without the -inf fill, as Indexer._get_topk_ragged calls it."""

from __future__ import annotations

__all__ = ["fp8_mqa_logits_baseline"]


class _Empty:
    """aiter's module torch, whose empty returns ``buffer`` for its shape: aiter allocates its logits
    as [rows, keys rounded up to 256], the buffer the op writes, so it writes there at no extra copy."""

    def __init__(self, torch, buffer):
        self._torch, self._buffer = torch, buffer

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def empty(self, size, *args, **kwargs):
        if tuple(size) == tuple(self._buffer.shape):
            return self._buffer
        return self._torch.empty(size, *args, **kwargs)


def fp8_mqa_logits_baseline(q, kv, kv_scales, weights, cu_starts, cu_ends, logits, no_values=None):
    """aiter's logits (clean_logits=False) into this call's logits buffer."""
    from aiter.ops.triton.attention import fp8_mqa_logits as module

    torch = module.torch
    module.torch = _Empty(torch, logits)
    try:
        module.fp8_mqa_logits(q, kv, kv_scales, weights, cu_starts, cu_ends, clean_logits=False)
    finally:
        module.torch = torch
    return logits.new_empty(0) if no_values is None else no_values
