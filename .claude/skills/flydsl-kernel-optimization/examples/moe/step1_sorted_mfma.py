"""MoE step 1 -- routed tokens sorted by expert into 16-row blocks; both stages become MFMA GEMMs.

What it adds over step 0: grouping. The host sorts the T*TOPK routed items by expert (common.moe_sort):
items of one expert become consecutive rows, each expert padded to a multiple of BM=16 rows with the
sentinel id T*TOPK, plus one expert id per 16-row block (-1 past the last real block). Now a block of
16 rows shares one expert, so each weight tile read from memory feeds 16 rows through the matrix core:
  stage 1, grid (I/64 column blocks, MAX_BLOCKS row blocks), N fastest: a 4-wave workgroup computes a
    16x64 block of gate AND of up for one expert (v_mfma_f32_16x16x32_bf16, waves split N as in the gemm
    ladder), A rows gathered from x by token id, W1 rows loaded straight into MFMA registers; the
    epilogue applies silu(g)*u and stores h in sorted row order (bf16);
  stage 2, grid (D/64, MAX_BLOCKS): 16x64 blocks of h_sorted @ W2[e]^T; the epilogue multiplies each row
    by its routing weight and writes it (fp32) to that item's row of `partial` (T*TOPK, D);
  reduce, grid (T): out[t] = sum over the token's TOPK rows of `partial`, in a fixed order.
The gather and the padding cost no branches: x and `partial` are bounded buffer descriptors
(`make_buffer_tensor(..., max_size=False)`), so sentinel rows read zeros and their stores are dropped
by the hardware. Blocks past the last real one exit on `block_expert < 0`; nothing reads a count on
the host to size the grid (MAX_BLOCKS is an upper bound, FlyDSL mxfp_moe/gemm1.py:59-67).

What it saves (estimated): each expert's weights are read once per 16-row block instead of once per
routed item: with ~16 items per expert here, about 12-16 blocks read W1/W2 -- roughly 36 MiB instead of
384 MiB -- and the dot products run on the matrix core. What it costs: the host sort (a device->host
copy, a Python loop and an upload: a synchronization no CUDA graph can capture), padding rows that
compute zeros (up to 15 per expert), an fp32 partial buffer (512 KiB) and a third kernel launch. The
gathered A rows are re-read by all 4 waves (no LDS): small next to the weights at 16 rows.
Card: ../../grouped_gemm/flydsl_grouped_moe_inference.md sections 2, 5, 6; ../../moe/flydsl_moe_decode.md.

Run: python step1_sorted_mfma.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from common import (
    DEVICE,
    bench,
    check,
    moe_inputs,
    moe_reference,
    moe_sort,
)
from common import MOE_BM as BM
from common import MOE_D as D
from common import MOE_E as E
from common import MOE_I as I
from common import MOE_MAX_BLOCKS as MAX_BLOCKS
from common import MOE_T as T
from common import MOE_TOPK as TOPK
from flydsl.expr import range_constexpr

THREADS = 256
BN1, BN2, BK = (
    64,
    64,
    32,
)  # stage-1 h columns, stage-2 output columns per workgroup; K per MFMA


def silu_mul(g, u):
    return g * (1.0 / (1.0 + fx.math.exp(-g))) * u


@flyc.kernel
def moe_stage1_kernel(
    x: fx.Tensor,
    w1: fx.Tensor,
    sorted_ids: fx.Tensor,
    block_expert: fx.Tensor,
    h: fx.Tensor,
    tiled_mma: fx.TiledMma,
):
    tid = fx.thread_idx.x
    nblk, blk, _ = fx.block_idx
    e = block_expert[blk]

    def expert_block():
        gx = fx.rocdl.make_buffer_tensor(x, max_size=False)  # bounded: sentinel token T reads zeros
        w1e = fx.Tensor(fx.make_view(fx.get_iter(w1) + e * (2 * I * D), fx.make_layout((2 * I, D), (D, 1))))
        gw = fx.rocdl.make_buffer_tensor(w1e)
        tWg = fx.flat_divide(gw, fx.make_tile(BN1, BK))[None, None, nblk, None]  # gate rows
        tWu = fx.flat_divide(gw, fx.make_tile(BN1, BK))[None, None, I // BN1 + nblk, None]  # up rows
        tH = fx.flat_divide(fx.rocdl.make_buffer_tensor(h), fx.make_tile(BM, BN1))[None, None, blk, nblk]

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
        pWg, pWu = thr_g2r_B.partition_S(tWg), thr_g2r_B.partition_S(tWu)

        frag_A = thr_mma.make_fragment_A(fx.flat_divide(gx, fx.make_tile(BM, BK))[None, None, 0, 0])
        frag_Bg = thr_mma.make_fragment_B(tWg[None, None, 0])
        frag_Bu = thr_mma.make_fragment_B(tWu[None, None, 0])
        frag_G = thr_mma.make_fragment_C(tH)
        frag_U = thr_mma.make_fragment_C(tH)

        # Which (row, k) of the 16x32 A tile this lane holds: partition coordinate tensors like data.
        a_row = fx.get_scalar(thr_mma.partition_A(fx.make_view(0, fx.make_layout((BM, BK), (1, 0))))[0])
        a_col = fx.get_scalar(thr_mma.partition_A(fx.make_view(0, fx.make_layout((BM, BK), (0, 1))))[0])
        token = sorted_ids[blk * BM + a_row] // TOPK  # the sentinel T*TOPK -> token T: out of bounds
        a_off = token * D + a_col

        frag_Bg_retile, frag_Bu_retile = thr_g2r_B.retile(frag_Bg), thr_g2r_B.retile(frag_Bu)
        a_frag = frag_A[None, 0, 0]  # this lane's 8 consecutive k of one A row

        frag_G.fill(0.0)
        frag_U.fill(0.0)
        for kt in range(D // BK):
            a_src = fx.make_view(fx.get_iter(gx) + (a_off + kt * BK), fx.make_layout(8, 1))
            fx.copy(buf_copy, a_src, a_frag)  # gathered row, 16 B
            fx.copy(buf_copy, pWg[None, None, None, kt], frag_Bg_retile)
            fx.copy(buf_copy, pWu[None, None, None, kt], frag_Bu_retile)
            fx.gemm(tiled_mma, frag_G, frag_A, frag_Bg, frag_G)
            fx.gemm(tiled_mma, frag_U, frag_A, frag_Bu, frag_U)

        frag_H = fx.make_fragment_like(frag_G, fx.BFloat16)
        frag_H.store(silu_mul(frag_G.load(), frag_U.load()).to(fx.BFloat16))
        out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        thr_r2g = fx.make_tiled_copy_C(out_copy, tiled_mma).get_slice(tid)
        fx.copy(out_copy, thr_r2g.retile(frag_H), thr_r2g.partition_S(tH))

    # A runtime `if` carries every local whose method it calls; keep its body a plain call.
    if e >= 0:  # blocks past the last real one exit
        expert_block()


@flyc.kernel
def moe_stage2_kernel(
    h: fx.Tensor,
    w2: fx.Tensor,
    sorted_ids: fx.Tensor,
    block_expert: fx.Tensor,
    topk_w: fx.Tensor,
    partial: fx.Tensor,
    tiled_mma: fx.TiledMma,
):
    tid = fx.thread_idx.x
    nblk, blk, _ = fx.block_idx
    e = block_expert[blk]

    def expert_block():
        w2e = fx.Tensor(fx.make_view(fx.get_iter(w2) + e * (D * I), fx.make_layout((D, I), (I, 1))))
        tA = fx.flat_divide(fx.rocdl.make_buffer_tensor(h), fx.make_tile(BM, BK))[None, None, blk, None]
        tB = fx.flat_divide(fx.rocdl.make_buffer_tensor(w2e), fx.make_tile(BN2, BK))[None, None, nblk, None]
        gp = fx.rocdl.make_buffer_tensor(partial, max_size=False)  # bounded: sentinel rows are dropped
        gw = fx.rocdl.make_buffer_tensor(topk_w, max_size=False)  # ... and read weight 0

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
        pA, pB = thr_g2r_A.partition_S(tA), thr_g2r_B.partition_S(tB)
        frag_A = thr_mma.make_fragment_A(tA[None, None, 0])
        frag_B = thr_mma.make_fragment_B(tB[None, None, 0])
        frag_Y = thr_mma.make_fragment_C(fx.flat_divide(gp, fx.make_tile(BM, BN2))[None, None, 0, 0])

        frag_A_retile, frag_B_retile = thr_g2r_A.retile(frag_A), thr_g2r_B.retile(frag_B)

        frag_Y.fill(0.0)
        for kt in range(I // BK):
            fx.copy(buf_copy, pA[None, None, None, kt], frag_A_retile)
            fx.copy(buf_copy, pB[None, None, None, kt], frag_B_retile)
            fx.gemm(tiled_mma, frag_Y, frag_A, frag_B, frag_Y)

        # Epilogue: row m of the block is routed item sorted_ids[blk*BM + m]; scale by its weight, store.
        c_rows = thr_mma.partition_C(fx.make_view(0, fx.make_layout((BM, BN2), (1, 0))))
        c_cols = thr_mma.partition_C(fx.make_view(0, fx.make_layout((BM, BN2), (0, 1))))
        y = frag_Y.load()
        for v in range_constexpr(4):
            item = sorted_ids[blk * BM + fx.get_scalar(c_rows[v])]
            gp[item, nblk * BN2 + fx.get_scalar(c_cols[v])] = gw[item] * y[v]

    if e >= 0:
        expert_block()


@flyc.kernel
def moe_reduce_kernel(partial: fx.Tensor, out: fx.Tensor):
    tid = fx.thread_idx.x
    t = fx.block_idx.x
    for o in range_constexpr(D // THREADS):
        dcol = o * THREADS + tid
        acc = fx.Float32(0.0)
        for j in range_constexpr(TOPK):  # fixed order: deterministic
            acc = acc + partial[t * TOPK + j, dcol]
        out[t, dcol] = acc.to(fx.BFloat16)


@flyc.jit
def fused_moe(
    x: fx.Tensor,
    w1: fx.Tensor,
    w2: fx.Tensor,
    sorted_ids: fx.Tensor,
    block_expert: fx.Tensor,
    topk_w: fx.Tensor,
    h: fx.Tensor,
    partial: fx.Tensor,
    out: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    x = fx.Tensor(fx.make_view(fx.get_iter(x), fx.make_layout((T, D), (D, 1))))
    w1 = fx.Tensor(fx.make_view(fx.get_iter(w1), fx.make_layout(E * 2 * I * D, 1)))
    w2 = fx.Tensor(fx.make_view(fx.get_iter(w2), fx.make_layout(E * D * I, 1)))
    sorted_ids = fx.Tensor(fx.make_view(fx.get_iter(sorted_ids), fx.make_layout(MAX_BLOCKS * BM, 1)))
    block_expert = fx.Tensor(fx.make_view(fx.get_iter(block_expert), fx.make_layout(MAX_BLOCKS, 1)))
    topk_w = fx.Tensor(fx.make_view(fx.get_iter(topk_w), fx.make_layout(T * TOPK, 1)))
    h = fx.Tensor(fx.make_view(fx.get_iter(h), fx.make_layout((MAX_BLOCKS * BM, I), (I, 1))))
    partial = fx.Tensor(fx.make_view(fx.get_iter(partial), fx.make_layout((T * TOPK, D), (D, 1))))
    out = fx.Tensor(fx.make_view(fx.get_iter(out), fx.make_layout((T, D), (D, 1))))
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),  # four waves along N, 16 columns each
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    moe_stage1_kernel(x, w1, sorted_ids, block_expert, h, tiled_mma).launch(
        grid=(I // BN1, MAX_BLOCKS, 1), block=(THREADS, 1, 1), stream=stream
    )
    moe_stage2_kernel(h, w2, sorted_ids, block_expert, topk_w, partial, tiled_mma).launch(
        grid=(D // BN2, MAX_BLOCKS, 1), block=(THREADS, 1, 1), stream=stream
    )
    moe_reduce_kernel(partial, out).launch(grid=(T, 1, 1), block=(THREADS, 1, 1), stream=stream)


def main():
    x, w1, w2, topk_ids, topk_w = moe_inputs()
    h = torch.empty(MAX_BLOCKS * BM, I, dtype=torch.bfloat16, device=DEVICE)
    partial = torch.empty(T * TOPK, D, dtype=torch.float32, device=DEVICE)
    out = torch.empty(T, D, dtype=torch.bfloat16, device=DEVICE)

    def run():
        sorted_ids, block_expert = moe_sort(topk_ids)  # host-side routing sort (moe/step2 moves it on chip)
        fused_moe(x, w1, w2, sorted_ids, block_expert, topk_w, h, partial, out)

    run()
    ref = None if DEVICE == "cpu" else moe_reference(x, w1, w2, topk_ids, topk_w)
    ok = check("moe step1", out, ref)
    if ok:
        bench(run)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
