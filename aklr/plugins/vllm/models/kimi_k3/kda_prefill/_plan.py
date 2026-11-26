# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The KDA prefill route's checkpoint plan and snapshot copy, one FlyDSL launch each: per sequence, its snapshot point
in chunks and its cache row, from vLLM's offsets and state indices (device tensors); then each snapshot taken into its
cache row, none written for a sequence that takes none."""

from __future__ import annotations

from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler import from_c_void_p
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T
from flydsl.expr.typing import Vector as Vec

from aklr.core.jit import flydsl_cache

__all__ = ["checkpoint_plan", "scatter_rows"]

_THREADS = 256
# 16-byte pieces of a snapshot one thread of the copy moves, and the int32 words a workgroup moves with them
_PASSES = 4
_ROW_BLOCK = 4 * _PASSES * _THREADS
# the compiled launches by (launch, the element types of its pointers): FlyDSL 0.5 traces a jit function at every
# call, so the first call of each is compiled and kept (as aklr.core.jit.flydsl_cache keeps the kernels' own)
_COMPILED: dict[tuple, Any] = {}


def _ld(p, off, ty=None):
    return fx.ptr_load(fx.add_offset(p, off)) if ty is None else fx.ptr_load(fx.add_offset(p, off), ty)


def _st(val, p, off):
    fx.ptr_store(val, fx.add_offset(p, off))


@flyc.kernel
def _plan_kernel(offsets, indices, chunks, rows, n: fx.Int32, null: fx.Int32, chunk: fx.Int32):
    """Sequence i's snapshot point in chunks and its cache row; 0 and -1 where it takes none."""
    i = Int32(fx.block_idx.x) * _THREADS + Int32(fx.thread_idx.x)
    if i < n:
        off = Int64(_ld(offsets, i))
        idx = Int64(_ld(indices, i))
        size = Int64(chunk)
        taken = (off > Int64(0)) & (off % size == Int64(0)) & (idx >= Int64(0)) & (idx != Int64(null))
        _st(Int32(taken.select(off // size, Int64(0))), chunks, i)
        _st(taken.select(idx, Int64(-1)), rows, i)


@flyc.kernel
def _scatter_kernel(src, rows, dst, row_words: fx.Int32, dst_stride: fx.Int32):
    """Snapshot s's 16-byte pieces of this workgroup's block into its cache row; nothing for a negative row."""
    s = Int32(fx.block_idx.x)
    row = Int64(_ld(rows, s))
    if row >= Int64(0):
        first = Int32(fx.block_idx.y) * _ROW_BLOCK + Int32(fx.thread_idx.x) * 4
        for p in range_constexpr(_PASSES):
            word = first + p * (4 * _THREADS)
            if word < row_words:
                piece = Vec(_ld(src, Int64(s) * Int64(row_words) + Int64(word), T.i32x4))
                _st(piece, dst, row * Int64(dst_stride) + Int64(word))


@flyc.jit
def _launch_plan(
    offsets, indices, chunks, rows, n: fx.Int32, null: fx.Int32, chunk: fx.Int32, stream: fx.Stream
):
    _plan_kernel(offsets, indices, chunks, rows, n, null, chunk).launch(
        grid=((n + _THREADS - 1) // _THREADS, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def _launch_scatter(
    src,
    rows,
    dst,
    row_words: fx.Int32,
    dst_stride: fx.Int32,
    n: fx.Int32,
    blocks: fx.Int32,
    stream: fx.Stream,
):
    _scatter_kernel(src, rows, dst, row_words, dst_stride).launch(
        grid=(n, blocks, 1), block=(_THREADS, 1, 1), stream=stream
    )


def _pointer(tensor, dtype=None):
    """The tensor's first element as FlyDSL's pointer argument, read as ``dtype`` elements (its own by default)."""
    element, size = {torch.int32: (Int32, 4), torch.int64: (Int64, 8)}[dtype or tensor.dtype]
    return from_c_void_p(
        element, tensor.data_ptr(), assumed_align=size
    )  # FlyDSL takes a multiple of the element


def _launch(fn, key: tuple, args: tuple, device: torch.device) -> None:
    """One launch on the device's current stream: the jit function's own call before FlyDSL 0.5 (it keeps its
    trace), else the compiled launch of this signature, compiled by its first call."""
    args = (*args, fx.Stream(torch._C._cuda_getCurrentRawStream(device.index)))
    if not flydsl_cache.is_needed():
        fn(*args)
        return
    compiled = _COMPILED.get((fn, key))
    if compiled is None:
        compiled = flyc.compile(fn, *args)  # traces, compiles or loads, and launches this call
        if compiled is not None:  # None under FLYDSL_COMPILE_ONLY
            _COMPILED[(fn, key)] = compiled
    else:
        compiled(*args)


def checkpoint_plan(offsets, indices, null, chunk):
    """(chunks as int32, rows as int64): offsets // chunk and indices where a snapshot is taken (offsets a positive
    multiple of chunk, indices a row other than null), 0 and -1 elsewhere, as vLLM's kernel reads its own (a
    negative row takes none)."""
    device_types = (torch.int32, torch.int64)
    if (
        offsets.device.type != "cuda"
        or offsets.dtype not in device_types
        or indices.dtype not in device_types
        or not -(2**31) <= int(null) < 2**31
    ):  # the same, op by op (the route's tests on CPU; no dtype or id vLLM passes)
        taken = (offsets > 0) & (offsets % chunk == 0) & (indices >= 0) & (indices != null)
        chunks = torch.where(taken, offsets // chunk, 0).to(torch.int32)
        return chunks, torch.where(taken, indices, -1).long()
    n = offsets.shape[0]
    chunks = torch.empty(n, dtype=torch.int32, device=offsets.device)
    rows = torch.empty(n, dtype=torch.int64, device=offsets.device)
    if n:
        offsets, indices = offsets.contiguous(), indices.contiguous()
        _launch(
            _launch_plan,
            (offsets.dtype, indices.dtype),
            (
                _pointer(offsets),
                _pointer(indices),
                _pointer(chunks),
                _pointer(rows),
                n,
                int(null),
                int(chunk),
            ),
            offsets.device,
        )
    return chunks, rows


def scatter_rows(src, rows, dst):
    """dst[rows[i]] = src[i] where rows[i] >= 0, no row written for a negative one: src [n, ...] dense, rows int64
    [n], dst [slots, ...] with each row dense (rows any distance apart)."""
    n, row_bytes = src.shape[0], src[0].numel() * src.element_size()
    dst_stride = dst.stride(0) * dst.element_size()
    if (
        src.device.type != "cuda"
        or src.dtype != dst.dtype
        or row_bytes % 16
        or dst_stride % 4
        or dst_stride >= 2**33
        or not (src.is_contiguous() and rows.is_contiguous() and rows.dtype == torch.int64)
    ):  # the same, op by op (the route's tests on CPU; no layout the route hands over)
        taken = rows >= 0
        dst[rows[taken]] = src[taken]
        return
    if n:
        blocks = -(-(row_bytes // 4) // _ROW_BLOCK)
        _launch(
            _launch_scatter,
            (),
            (_pointer(src, torch.int32), _pointer(rows), _pointer(dst, torch.int32), row_bytes // 4, dst_stride // 4,
             n, blocks),
            src.device,
        )  # fmt: skip
