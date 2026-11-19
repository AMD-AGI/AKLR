# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The KDA prefill route's checkpoint plan in one launch: per sequence, its snapshot point in chunks and its cache
row, from vLLM's offsets and state indices (device tensors)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["checkpoint_plan"]


@triton.jit
def _plan_kernel(
    offsets_ptr, indices_ptr, chunks_ptr, rows_ptr, n, null, CHUNK: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    mask = i < n
    off = tl.load(offsets_ptr + i, mask=mask, other=0).to(tl.int64)
    idx = tl.load(indices_ptr + i, mask=mask, other=-1).to(tl.int64)
    taken = (off > 0) & (off % CHUNK == 0) & (idx >= 0) & (idx != null)
    tl.store(chunks_ptr + i, tl.where(taken, off // CHUNK, 0).to(tl.int32), mask=mask)
    tl.store(rows_ptr + i, tl.where(taken, idx, null), mask=mask)


def checkpoint_plan(offsets, indices, null, chunk):
    """(chunks as int32, rows as int64): offsets // chunk and indices where a snapshot is taken (offsets a positive
    multiple of chunk, indices a row other than null), 0 and null elsewhere."""
    if offsets.device.type != "cuda":  # the same, op by op (the route's tests on CPU)
        taken = (offsets > 0) & (offsets % chunk == 0) & (indices >= 0) & (indices != null)
        chunks = torch.where(taken, offsets // chunk, 0).to(torch.int32)
        return chunks, torch.where(taken, indices, null).long()
    n = offsets.shape[0]
    chunks = torch.empty(n, dtype=torch.int32, device=offsets.device)
    rows = torch.empty(n, dtype=torch.int64, device=offsets.device)
    if n:
        _plan_kernel[(1,)](
            offsets,
            indices,
            chunks,
            rows,
            n,
            int(null),
            CHUNK=chunk,
            BLOCK=max(16, triton.next_power_of_2(n)),
        )
    return chunks, rows
