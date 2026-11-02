# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 GEMV of the mega kernel: 16 rows of a row-major e4m3 weight [N, K] (vLLM's layout,
E8M0 scales in 32x32 blocks) against a tile's MXFP8 rows in LDS, over a K window (a split-K
part); wave w takes steps w, w + 8, ..., summed in wave order by ``row_sum``."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aklr.plugins.sglang.models.deepseek_v4.kernels.common.aiter_flydsl import (
    buffer_ops as bo,
)

from .device import THREADS, WAVES, mfma_scaled, rsrc, tile_map, traced

# a token's LDS row of MXFP8 words / E8M0 codes, padded by 16 B so 16 rows read
# by one MFMA operand do not share a bank group
LDS_PAD = 4
RED = WAVES * 64 * 4  # one task's accumulators in ``red`` (f32)


def lds_row(words):
    return words + LDS_PAD


def lds_at(u, row):
    """Element u of a span of rows ``row`` wide, at the ``lds_row`` stride."""
    return u + u // row * LDS_PAD


def _steps(c, nsteps, split, tiled):
    """This wave's steps of the window: [(local step, live)]."""
    wave = c["wave"]
    if const_expr(split is None):
        per = (nsteps + WAVES - 1) // WAVES
        ragged = nsteps % WAVES != 0
        return [(wave + WAVES * i, (wave + WAVES * i < nsteps) if ragged else None) for i in range(per)]
    assert split <= WAVES and nsteps % split == 0
    per = nsteps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def _step_load(c, r_w, r_ws, k, row0, st, pred=None):
    """Global step ``st``'s weight operand of rows row0 .. +16: two 16 B chunks a lane (K
    bytes 16 j .. and 64 + 16 j .., row row0 + lane % 16) and its K block's E8M0 byte.
    ``pred`` false: masked (no memory traffic, zeros)."""
    lane = c["lane"]
    j = lane // 16
    row = row0 + lane % 16
    base = row * k + 128 * st + 16 * j
    lo = fx.Vector(bo.buffer_load(r_w, base // 4, vec_width=4, dtype=T.i32))
    hi = fx.Vector(bo.buffer_load(r_w, (base + 64) // 4, vec_width=4, dtype=T.i32))
    kg = k // 32
    wsi = (row0 // 32) * kg + st * 4 + j
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def gemv_loads(c, w, ws, k, row0, split=None, tiled=False, kbeg=0, nsteps=None, pred=None):
    """The weight operands of rows row0 .. +16 (``tiled``: row0 + 16 (wave //
    split)) over steps kbeg .. kbeg + nsteps for this wave, all in flight
    (``pred`` false: masked off, no memory traffic)."""
    nsteps = k // 128 if nsteps is None else nsteps
    if const_expr(pred is None):
        r_w, r_ws = rsrc(w), rsrc(ws)
    else:
        # a masked-off CTA reads through a zero-size buffer: zeros, no traffic
        nb = pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
        r_w, r_ws = rsrc(w, nb), rsrc(ws, nb)
    if const_expr(tiled):
        row0 = row0 + 16 * (c["wave"] // split)
    return [
        _step_load(
            c,
            r_w,
            r_ws,
            k,
            row0,
            kbeg + (st if live is None else fx.min(st, nsteps - 1)),
            pred,
        )
        for st, live in _steps(c, nsteps, split, tiled)
    ]


def _step_mfma(c, nsteps, xl, xsl, st, ops, acc, live, rows):
    lane = c["lane"]
    lo, hi, sa = ops
    x_row, s_row = lds_row(nsteps * 32), lds_row(nsteps * 4)
    j = lane // 16
    col = fx.min(lane % 16, rows - 1)
    if const_expr(live is not None):
        st = fx.min(st, nsteps - 1)
    av = fx.Vector.from_elements([lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32)
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xv = fx.Vector.from_elements([xlo[d] for d in range(4)] + [xhi[d] for d in range(4)], fx.Int32)
    sb = fx.ptr_load(xsl + (col * s_row + st * 4 + j))
    if const_expr(live is not None):
        # a step past the end adds zero at unit scale
        av = fx.Vector.from_elements([live.select(av[d], fx.Int32(0)) for d in range(8)], fx.Int32)
        sa = live.select(sa, fx.Int32(127))
        sb = live.select(sb, fx.Int32(127))
    return mfma_scaled(av, xv, acc, sa, sb)


def gemv_mfmas(c, k, xl, xsl, red, ops, split=None, tiled=False, rows=16, nsteps=None):
    """``gemv_loads``' operands against ``rows`` tokens' MXFP8 rows of the
    window in LDS (``xl`` words, ``xsl`` codes, rows ``lds_row`` of the window)
    -> ``red[wave][lane]`` (a 16 x 16 C: row 4 (lane // 16) + i, token lane % 16)."""
    lane, wave = c["lane"], c["wave"]
    nsteps = k // 128 if nsteps is None else nsteps
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_steps(c, nsteps, split, tiled)):
        acc = _step_mfma(c, nsteps, xl, xsl, st, ops[i], acc, live, rows)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements(
            [(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(4)],
            fx.Float32,
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)


def row_sum(red, r, t, waves=None):
    """Row r, token t of the waves' accumulators, summed in wave order."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


def span_items(width, n):
    """A [n, width] span of 16 B items: rows a round of THREADS threads, rounds."""
    wq = width // 4
    rpr = THREADS // wq
    assert rpr >= 1, width
    return wq, rpr, (n + rpr - 1) // rpr


def rows_loads(
    c,
    words_base,
    codes_base,
    stride_w,
    off_w,
    words,
    stride_g,
    off_g,
    groups,
    t0,
    n,
    pred=None,
):
    """Tokens t0 .. t0 + n's published MXFP8 row slices issued, not waited on, 16 B a
    lane-item. Plain loads: their flags are seen and nothing in the launch read these
    lines before. ``pred`` false: masked off."""
    tid = c["tid"]
    nb = None if pred is None else pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    out = []
    for base, stride, off, width in (
        (words_base, stride_w, off_w, words),
        (codes_base, stride_g, off_g, groups),
    ):
        wq, rpr, rounds = span_items(width, n)
        row0, col = tile_map(tid, wq)
        a0 = col * 4 + off + t0 * stride
        r = rsrc(base, nb)
        regs = []
        for i in range(rounds):
            row = row0 + rpr * i
            if (i + 1) * rpr > n or rpr * wq < THREADS:
                row = fx.min(row, n - 1)
            regs.append(fx.Vector(bo.buffer_load(r, row * stride + a0, vec_width=4, dtype=T.i32)))
        out.append(regs)
    return out


@traced
def _store_span(c, regs, width, dst, n):
    tid = c["tid"]
    wq, rpr, _ = span_items(width, n)
    row0, col = tile_map(tid, wq)
    lrow = lds_row(width)
    base = dst + (row0 * lrow + col * 4)
    for i in range_constexpr(len(regs)):
        if const_expr((i + 1) * rpr <= n and rpr * wq == THREADS):
            fx.ptr_store(regs[i], base + i * rpr * lrow)
        else:
            # infera-flash: no branch (rows past n rewrite row n - 1), so the load is not sunk behind later weights'
            row = fx.min(row0 + rpr * i, n - 1)
            fx.ptr_store(regs[i], dst + (row * lrow + col * 4))


def rows_store(c, regs, words, groups, xl, xsl, n):
    """``rows_loads``' registers into LDS rows 0 .. n (``lds_row`` strides)."""
    _store_span(c, regs[0], words, xl, n)
    _store_span(c, regs[1], groups, xsl, n)
    gpu.barrier()
