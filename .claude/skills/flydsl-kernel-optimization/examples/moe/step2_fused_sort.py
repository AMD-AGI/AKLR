"""MoE step 2 -- the routing sort folded into stage 1: no host round trip, one node less.

What it adds over step 1: stage 1 takes `topk_ids` instead of host-sorted arrays. Every stage-1
workgroup rebuilds, in LDS, just enough of the padded sort to know its own 16-row block (as the card's
MXFP4 decode kernel does; ../../moe/flydsl_moe_decode.md section 2):
  1. each of the first T*TOPK threads loads one routed item's expert into LDS; the block's 16 row slots
     are seeded with the sentinel;
  2. every thread scans the T*TOPK experts in LDS (broadcast reads): its item's rank among earlier items
     of the same expert, and (threads < E) one expert's item count;
  3. from the E counts, each thread finds which expert owns block `blk` and the rank of its first row
     (blocks are expert-major, each expert padded to a multiple of 16 rows -- the same order as
     common.moe_sort);
  4. the items of that expert whose rank falls in the block write their id into its row slot.
The expert id comes out of LDS in a VGPR; `readfirstlane` makes it wave-uniform, so the per-expert weight
descriptor stays in SGPRs (a descriptor built from VGPRs puts every buffer load in a readfirstlane
loop). The workgroups of column block 0 publish the block's row ids and expert id for stage 2.
Every conditional store is branch-free: an index that should not be written is pointed past a bounded
buffer (dropped by the hardware) or at a spare LDS slot.

What it saves (estimated): the host sort -- a device->host copy, a Python loop and an upload, tens of
microseconds and a synchronization that blocks CUDA-graph capture -- and with it every torch kernel
it launched; the whole MoE is three graph-capturable FlyDSL kernels. On the card's MXFP4 decode
kernel, folding the sort (one node of three) cut the time 6-12% (field). What it costs: every stage-1
workgroup repeats the sort: O(T*TOPK) work per thread (128 items, read as 32 broadcast ds_read_b128;
an estimated 1-2 us), 1.1 KiB of LDS and three barriers; fine at decode sizes, not past THREADS routed
items. The card's kernel builds the same sort with an LDS-atomic histogram and DPP wave scans (cheaper per
thread, more code); FlyDSL's moe_sorting_kernel (histogram, prefix sum, scatter) runs 1-4 kernels by batch size.
Card: ../../moe/flydsl_moe_decode.md sections 1-2; ../../gemm/flydsl_gemm_datapath.md section 8.

Run: python step2_fused_sort.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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
)
from common import MOE_BM as BM
from common import MOE_D as D
from common import MOE_E as E
from common import MOE_I as I
from common import MOE_MAX_BLOCKS as MAX_BLOCKS
from common import MOE_T as T
from common import MOE_TOPK as TOPK
from flydsl.expr import range_constexpr, rocdl

THREADS = 256
BN1, BN2, BK = (
    64,
    64,
    32,
)  # stage-1 h columns, stage-2 output columns per workgroup; K per MFMA
N_ITEMS = T * TOPK  # routed items; the in-LDS sort gives one to each thread
SENTINEL = N_ITEMS  # padding row id: token T, out of bounds everywhere
NO_EXPERT = E  # expert id of the threads past the last item: matches no block (block ids are -1..E-1)
assert N_ITEMS <= THREADS


@fx.struct
class SortStorage:
    expert_of: fx.Array[fx.Int32, THREADS, 16]  # expert of item i (NO_EXPERT past the last item)
    count: fx.Array[fx.Int32, E + 1, 16]  # items per expert (+1 spare slot)
    rows: fx.Array[fx.Int32, BM + 1, 16]  # this block's item ids (+1 spare slot)


def silu_mul(g, u):
    return g * (1.0 / (1.0 + fx.math.exp(-g))) * u


@flyc.kernel
def moe_stage1_kernel(
    x: fx.Tensor,
    w1: fx.Tensor,
    topk_ids: fx.Tensor,
    sorted_ids: fx.Tensor,
    block_expert: fx.Tensor,
    h: fx.Tensor,
    tiled_mma: fx.TiledMma,
):
    tid = fx.thread_idx.x
    nblk, blk, _ = fx.block_idx
    lds = fx.SharedAllocator().allocate(SortStorage).peek()
    s_expert = lds.expert_of.view(fx.make_layout(THREADS, 1))
    s_count = lds.count.view(fx.make_layout(E + 1, 1))
    s_rows = lds.rows.view(fx.make_layout(BM + 1, 1))

    # 1. load the routing; seed the block's row slots with the sentinel.
    my_e = (tid < N_ITEMS).select(
        fx.rocdl.make_buffer_tensor(topk_ids, max_size=False)[tid], fx.Int32(NO_EXPERT)
    )
    s_expert[tid] = my_e
    s_rows[(tid < BM).select(tid, BM)] = fx.Int32(SENTINEL)
    fx.gpu.barrier()

    # 2. rank of this thread's item among earlier items of its expert; thread e < E counts expert e.
    rank = fx.Int32(0)
    count = fx.Int32(0)
    for i in range(N_ITEMS):
        e_i = s_expert[i]
        rank = rank + ((e_i == my_e) & (fx.Int32(i) < tid)).select(fx.Int32(1), fx.Int32(0))
        count = count + (e_i == tid).select(fx.Int32(1), fx.Int32(0))
    s_count[(tid < E).select(tid, E)] = count
    fx.gpu.barrier()

    # 3. which expert owns block `blk`, and the rank of the block's first row within that expert.
    block_e = fx.Int32(-1)
    first_rank = fx.Int32(0)
    start = fx.Int32(0)
    for e in range_constexpr(E):
        nb = (s_count[e] + (BM - 1)) // BM
        hit = (blk >= start) & (blk < start + nb)
        block_e = hit.select(fx.Int32(e), block_e)
        first_rank = hit.select((blk - start) * BM, first_rank)
        start = start + nb
    block_e = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, block_e))  # uniform -> SGPR

    # 4. items of that expert whose rank falls in this block fill its rows (others hit the spare slot).
    mine = (my_e == block_e) & (rank >= first_rank) & (rank < first_rank + BM)
    s_rows[mine.select(rank - first_rank, BM)] = tid
    fx.gpu.barrier()

    # Column block 0 publishes the block for stage 2; masked lanes write past the bounded buffers.
    publish = nblk == 0
    g_sorted = fx.rocdl.make_buffer_tensor(sorted_ids, max_size=False)
    g_sorted[(publish & (tid < BM)).select(blk * BM + tid, MAX_BLOCKS * BM)] = s_rows[
        (tid < BM).select(tid, BM)
    ]
    g_bexp = fx.rocdl.make_buffer_tensor(block_expert, max_size=False)
    g_bexp[(publish & (tid == 0)).select(blk, MAX_BLOCKS)] = block_e
    e = block_e

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
        token = s_rows[a_row] // TOPK  # from LDS now; the sentinel -> token T: out of bounds
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
    topk_ids: fx.Tensor,
    topk_w: fx.Tensor,
    sorted_ids: fx.Tensor,
    block_expert: fx.Tensor,
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
    topk_ids = fx.Tensor(fx.make_view(fx.get_iter(topk_ids), fx.make_layout(T * TOPK, 1)))
    topk_w = fx.Tensor(fx.make_view(fx.get_iter(topk_w), fx.make_layout(T * TOPK, 1)))
    h = fx.Tensor(fx.make_view(fx.get_iter(h), fx.make_layout((MAX_BLOCKS * BM, I), (I, 1))))
    partial = fx.Tensor(fx.make_view(fx.get_iter(partial), fx.make_layout((T * TOPK, D), (D, 1))))
    out = fx.Tensor(fx.make_view(fx.get_iter(out), fx.make_layout((T, D), (D, 1))))
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),  # four waves along N, 16 columns each
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    moe_stage1_kernel(x, w1, topk_ids, sorted_ids, block_expert, h, tiled_mma).launch(
        grid=(I // BN1, MAX_BLOCKS, 1), block=(THREADS, 1, 1), stream=stream
    )
    moe_stage2_kernel(h, w2, sorted_ids, block_expert, topk_w, partial, tiled_mma).launch(
        grid=(D // BN2, MAX_BLOCKS, 1), block=(THREADS, 1, 1), stream=stream
    )
    moe_reduce_kernel(partial, out).launch(grid=(T, 1, 1), block=(THREADS, 1, 1), stream=stream)


def main():
    x, w1, w2, topk_ids, topk_w = moe_inputs()
    sorted_ids = torch.empty(MAX_BLOCKS * BM, dtype=torch.int32, device=DEVICE)  # written by stage 1
    block_expert = torch.empty(MAX_BLOCKS, dtype=torch.int32, device=DEVICE)  # written by stage 1
    h = torch.empty(MAX_BLOCKS * BM, I, dtype=torch.bfloat16, device=DEVICE)
    partial = torch.empty(T * TOPK, D, dtype=torch.float32, device=DEVICE)
    out = torch.empty(T, D, dtype=torch.bfloat16, device=DEVICE)

    def run():
        fused_moe(x, w1, w2, topk_ids, topk_w, sorted_ids, block_expert, h, partial, out)

    run()
    ref = None if DEVICE == "cpu" else moe_reference(x, w1, w2, topk_ids, topk_w)
    ok = check("moe step2", out, ref)
    if ok:
        bench(run)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
