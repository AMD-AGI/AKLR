# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The KDA prefill route's checkpoint plan and snapshot copy, one launch each: per sequence, its snapshot point in
chunks and its cache row, from vLLM's offsets and state indices (device tensors); then each snapshot taken into its
cache row, none written for a sequence that takes none."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["checkpoint_plan", "scatter_rows"]

_ROW_BLOCK = 4096  # elements a program of the snapshot copy moves


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
    tl.store(rows_ptr + i, tl.where(taken, idx, -1), mask=mask)


@triton.jit
def _scatter_kernel(src_ptr, rows_ptr, dst_ptr, row_elems, dst_stride, BLOCK: tl.constexpr):
    s = tl.program_id(0)
    row = tl.load(rows_ptr + s)
    if row >= 0:
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < row_elems
        x = tl.load(src_ptr + s.to(tl.int64) * row_elems + offs, mask=mask)
        tl.store(dst_ptr + row * dst_stride + offs, x, mask=mask)


def checkpoint_plan(offsets, indices, null, chunk):
    """(chunks as int32, rows as int64): offsets // chunk and indices where a snapshot is taken (offsets a positive
    multiple of chunk, indices a row other than null), 0 and -1 elsewhere, as vLLM's kernel reads its own (a
    negative row takes none)."""
    if offsets.device.type != "cuda":  # the same, op by op (the route's tests on CPU)
        taken = (offsets > 0) & (offsets % chunk == 0) & (indices >= 0) & (indices != null)
        chunks = torch.where(taken, offsets // chunk, 0).to(torch.int32)
        return chunks, torch.where(taken, indices, -1).long()
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


def scatter_rows(src, rows, dst):
    """dst[rows[i]] = src[i] where rows[i] >= 0, no row written for a negative one: src [n, ...] dense, rows int64
    [n], dst [slots, ...] with each row dense (rows any distance apart)."""
    if src.device.type != "cuda":
        taken = rows >= 0
        dst[rows[taken]] = src[taken]
        return
    n, row_elems = src.shape[0], src[0].numel()
    if n:
        _scatter_kernel[(n, triton.cdiv(row_elems, _ROW_BLOCK))](
            src, rows, dst, row_elems, dst.stride(0), BLOCK=_ROW_BLOCK
        )
