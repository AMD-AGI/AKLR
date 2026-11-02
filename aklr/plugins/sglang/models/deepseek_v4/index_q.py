# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The indexer's query for more rows than ``index_q_pack``'s eight a group, in one Triton launch:
the FP8 GEMM on the matrix cores, then the same RoPE, FP4 packing, head weights and compressor
GEMV as the HIP kernel (``hipops/csrc/dsv41/index_q.hip``)."""

from __future__ import annotations

import functools

import torch

QRANK, DIM = 1280, 128
# a q program's columns: one 32-column chunk of a head (one ue8m0 scale)
CHUNK = 32
# a weights or gate program's output columns
SMALL_N = 16
MAX_ROWS = 64


@functools.cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def pow2_ceil(x):
        bits = x.to(tl.uint32, bitcast=True)
        e = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        return ((e + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def bf16r(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def fp4_round_abs(a):
        r = tl.full(a.shape, 6.0, tl.float32)
        r = tl.where(a <= 5.0, 4.0, r)
        r = tl.where(a < 3.5, 3.0, r)
        r = tl.where(a <= 2.5, 2.0, r)
        r = tl.where(a < 1.75, 1.5, r)
        r = tl.where(a <= 1.25, 1.0, r)
        r = tl.where(a < 0.75, 0.5, r)
        r = tl.where(a <= 0.25, 0.0, r)
        return r

    @triton.jit
    def x_dot(x, src, r, live, n, nmask, dim, BM: tl.constexpr, BK: tl.constexpr):
        """x [rows, dim] . src[n]^T in fp32: the head weights' (src = weights_proj) or the
        compressor's (src = gate_w)."""
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, 16), dtype=tl.float32)
        for k0 in range(0, dim, BK):
            a = tl.load(x + r[:, None] * dim + k0 + k[None, :], mask=live[:, None], other=0.0)
            b = tl.load(src + n[None, :] * dim + k0 + k[:, None], mask=nmask[None, :], other=0.0)
            acc += tl.dot(a, b)
        return acc

    @triton.jit
    def index_q_kernel(
        qr,
        w,
        ws,
        x,
        wp,
        pos,
        table,
        q_fp4,
        q_scale,
        weights,
        gate_w,
        gate_out,
        rows,
        positions,
        weight_scale,
        dim,
        gate_n,
        HEADS: tl.constexpr,
        PAIRS: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        QK: tl.constexpr,
        HAS_GATE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = tl.program_id(1) * BM + tl.arange(0, BM)
        live = r < rows
        n_q = HEADS * 4
        n_w = HEADS // 16
        if pid < n_q:
            h = pid // 4
            j = pid % 4
            c = tl.arange(0, 32)
            col = h * 128 + j * 32 + c
            kk = tl.arange(0, QK)
            qacc = tl.zeros((BM, 32), dtype=tl.float32)
            for k0 in range(0, 1280, QK):
                # e4m3 values at power-of-two scales are bf16 values: the products are exact in
                # fp32 either way, the sums in another order than the HIP kernel's
                a = tl.load(qr + r[:, None] * 1280 + k0 + kk[None, :], mask=live[:, None], other=0.0)
                a = tl.reshape(a.to(tl.float32), (BM, QK // 32, 32))
                sa = pow2_ceil(tl.maximum(tl.max(tl.abs(a), axis=2), 1e-4) * (1.0 / 448.0))
                a8 = tl.minimum(tl.maximum(a / sa[:, :, None], -448.0), 448.0).to(tl.float8e4nv)
                a = tl.reshape(a8.to(tl.float32) * sa[:, :, None], (BM, QK)).to(tl.bfloat16)
                b = tl.load(w + col[None, :] * 1280 + k0 + kk[:, None]).to(tl.float8e4nv, bitcast=True)
                gk = tl.arange(0, QK // 32)
                sw = (tl.load(ws + (h * 4 + j) * 40 + k0 // 32 + gk).to(tl.uint32) << 23).to(
                    tl.float32, bitcast=True
                )
                b = tl.reshape(tl.reshape(b.to(tl.float32), (QK // 32, 32, 32)) * sw[:, None, None], (QK, 32))
                qacc += tl.dot(a, b.to(tl.bfloat16))
            y = bf16r(qacc)
            rope0: tl.constexpr = 128 - 2 * PAIRS
            if j * 32 >= rope0:
                p = tl.load(pos + r, mask=live, other=0)
                p = tl.where((p >= 0) & (p < positions), p, 0)
                pair = (j * 32 - rope0) // 2 + tl.arange(0, 16)
                cs = table + (p[:, None] * PAIRS + pair[None, :]) * 2
                cos = tl.load(cs, mask=live[:, None], other=0.0)
                sin = tl.load(cs + 1, mask=live[:, None], other=0.0)
                x0, x1 = tl.split(tl.reshape(y, (BM, 16, 2)))
                y = tl.reshape(tl.join(bf16r(x0 * cos - x1 * sin), bf16r(x0 * sin + x1 * cos)), (BM, 32))
            amax = tl.max(tl.abs(y), axis=1)
            s = pow2_ceil(tl.maximum(amax, 6.0 * 1.1754943508222875e-38) * (1.0 / 6.0))
            v = tl.minimum(tl.maximum(y / s[:, None], -6.0), 6.0)
            fq = bf16r(tl.where(v < 0, -fp4_round_abs(tl.abs(v)), fp4_round_abs(tl.abs(v))) * s[:, None])
            a2 = tl.max(tl.abs(fq), axis=1)
            bits = tl.maximum(a2 / 6.0, 1.0e-4).to(tl.uint32, bitcast=True)
            e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
            e = tl.minimum(tl.maximum(e, 1), 254)
            s2 = (e << 23).to(tl.float32, bitcast=True)
            u = tl.minimum(tl.maximum(fq / s2[:, None], -6.0), 6.0)
            mag_f = fp4_round_abs(tl.abs(u))
            mb = mag_f.to(tl.uint32, bitcast=True)
            mag = tl.where(
                mb >= 0x3F800000, (((mb >> 23) - 126) << 1) | ((mb >> 22) & 1), (mb != 0).to(tl.uint32)
            )
            code = tl.where(mag == 0, 0, mag | tl.where(u < 0, 8, 0).to(tl.uint32))
            lo, hi = tl.split(tl.reshape(code, (BM, 16, 2)))
            byte = (lo | (hi << 4)).to(tl.uint8)
            i16 = tl.arange(0, 16)
            tl.store(q_fp4 + (r[:, None] * HEADS + h) * 64 + j * 16 + i16[None, :], byte, mask=live[:, None])
            tl.store(q_scale + r * 256 + j * 64 + (h % 16) * 4 + h // 16, e.to(tl.uint8), mask=live)
            if HEADS < 64:
                if h < 16:
                    z = tl.arange(0, 4)
                    zero = live[:, None] & (z[None, :] >= HEADS // 16)
                    tl.store(
                        q_scale + r[:, None] * 256 + j * 64 + h * 4 + z[None, :],
                        tl.zeros((BM, 4), dtype=tl.uint8),
                        mask=zero,
                    )
        elif pid < n_q + n_w:
            hn = (pid - n_q) * 16 + tl.arange(0, 16)
            wacc = x_dot(x, wp, r, live, hn, hn < HEADS, dim, BM, BK)
            tl.store(
                weights + r[:, None] * HEADS + hn[None, :],
                (bf16r(wacc) * weight_scale).to(tl.bfloat16),
                mask=live[:, None],
            )
        elif HAS_GATE:
            gn = (pid - n_q - n_w) * 16 + tl.arange(0, 16)
            gmask = gn < gate_n
            gacc = x_dot(x, gate_w, r, live, gn, gmask, dim, BM, BK)
            tl.store(gate_out + r[:, None] * gate_n + gn[None, :], gacc, mask=live[:, None] & gmask[None, :])

    return index_q_kernel


def index_q(qr, w, ws, x, wp, positions, table, weight_scale: float, gate_w=None) -> tuple:
    """``index_q_pack``'s outputs (q_fp4, q_scale, weights[, gate_out]) for 1 to ``MAX_ROWS`` rows."""
    rows, heads = qr.shape[0], w.shape[0] // DIM
    pairs = table.shape[1]
    device = qr.device
    # index_q_pack's dtypes: e2m1 pairs as int8, ue8m0 scales as uint8
    q_fp4 = torch.empty((rows, heads, DIM // 2), dtype=torch.int8, device=device)
    q_scale = torch.empty((rows, 1, 4, 16, 4), dtype=torch.uint8, device=device)
    weights = torch.empty((rows, heads), dtype=torch.bfloat16, device=device)
    gate_n = 0 if gate_w is None else gate_w.shape[0]
    gate_out = torch.empty((rows, max(gate_n, 1)), dtype=torch.float32, device=device)
    bm = 16
    grid = (heads * 4 + heads // 16 + -(-gate_n // SMALL_N), -(-rows // bm))
    _kernel()[grid](
        qr,
        w,
        ws,
        x,
        wp,
        positions,
        table,
        q_fp4.view(torch.uint8),
        q_scale,
        weights,
        wp if gate_w is None else gate_w,
        gate_out,
        rows,
        table.shape[0],
        float(weight_scale),
        x.shape[1],
        gate_n,
        HEADS=heads,
        PAIRS=pairs,
        BM=bm,
        BK=256,
        QK=256,
        HAS_GATE=gate_w is not None,
        num_warps=4 if rows <= 32 else 2,
        num_stages=2,
    )
    out = (q_fp4, q_scale, weights)
    return out + (gate_out,) if gate_w is not None else out


def supported(qr, w, x, table) -> bool:
    """Whether ``index_q`` takes these shapes (as ``index_q_pack`` does, past its eight rows)."""
    rows, heads = qr.shape[0], w.shape[0] // DIM
    rope0 = DIM - 2 * table.shape[1]
    return (
        0 < rows <= MAX_ROWS
        and qr.shape[1] == QRANK
        and w.shape == (heads * DIM, QRANK)
        and heads % 16 == 0
        and heads <= 64
        and rope0 % CHUNK == 0
        and x.shape[1] % 128 == 0
    )
