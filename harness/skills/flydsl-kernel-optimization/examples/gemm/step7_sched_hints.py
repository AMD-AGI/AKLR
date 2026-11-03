"""GEMM step 7 -- scheduling hints: two fences and a sched_group_barrier table per K step.

What it adds over step 6: three `fx.rocdl` scheduling directives in `k_step`; no instruction is added.
  1. `sched_barrier(0)` right after the next tile's DMA and B loads: LLVM may no longer sink those loads
     behind the MFMAs (in step 6's ISA it had moved the four B loads into the middle of the MFMA stream,
     shortening the prefetch); now all eight are issued at the top and have a whole K step to land.
  2. `hot_loop_scheduler()`: 4 A reads ahead, then 8 groups of {4 MFMAs, the next 2 A reads}. Each
     group's MFMAs (2 A fragments x 2 B columns) run while the next group's ds_read_b128s are in flight:
     the waits in the loop become lgkmcnt(3)/(2) instead of (1)/(0).
  3. `sched_barrier(0)` closing the K step: this stage's reads and MFMAs stay before the next barrier,
     and the peeled last tile no longer hoists all its LDS reads (step 5 peaked at 216 VGPRs there).
Hints are filled greedily and a table that does not fit is dropped silently: the obvious table
"1 read, 2 MFMAs" made LLVM wait on every read at once and put two MFMAs on one accumulator back to
back (compile-only ISA of this kernel). Read the scheduled ISA after every table change.

What it saves (estimated): the exposed LDS latency between a ds_read and its MFMAs (in step 6 every
MFMA pair waits on reads issued just before it) and the exposed B-load latency; a few percent on this
compute-bound shape, either sign until measured (card field results: -4% to +4% per table; race it,
do not port it). What it costs: 8 more VGPRs than fences alone (two more A fragments live; 128 here,
still 4 waves per SIMD by registers) and a table that must be re-derived for any other tile.
Card: ../../gemm/flydsl_gemm_schedule_and_locality.md section 3; ../../optimize/gemm-optimization.md section 5.

Run: python step7_sched_hints.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
from common import (
    bench,
    check,
    gemm_inputs,
    gemm_reference,
    shuffle_weight,
)
from flydsl.expr import range_constexpr, rocdl

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 64
THREADS = 256
DMA_ELEMS = 8
STAGES = 2
K_TILES = K // BK
A_READS = (BM // 16) * (BK // 32)  # ds_read_b128 per wave per K tile: 16, each feeding 2 MFMAs


def lds_layout(rows, cols):
    """Row-major (rows, cols) with 16-byte chunk index XOR (row % 8): Swizzle<3, 3, 3> on bf16 offsets."""
    return fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((rows, cols), (1, 0)),
    )


@fx.struct
class SharedStorage:
    a: fx.Array[fx.BFloat16, STAGES * BM * BK, 16]


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    gA = fx.rocdl.make_buffer_tensor(A)
    gB = fx.rocdl.make_buffer_tensor(B)  # B: the preshuffled buffer, viewed as logical (N, K)
    gC = fx.rocdl.make_buffer_tensor(C)
    tB = fx.flat_divide(gB, fx.make_tile(BN, BK))[None, None, bid_y, None]
    tC = fx.flat_divide(gC, fx.make_tile(BM, BN))[None, None, bid_x, bid_y]

    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    a_layout = lds_layout(BM, BK)
    sA = [fx.make_view(lds.a.ptr + s * BM * BK, a_layout) for s in range(STAGES)]

    wave_off = fx.Int32(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(tid // 64 * 64 * DMA_ELEMS)))
    dma_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    def dma_a(k_tile, stage):
        lds_ptr = lds.a.ptr + stage * BM * BK + wave_off
        for i in range_constexpr(BM * BK // (THREADS * DMA_ELEMS)):
            chunk = THREADS * i + tid
            row = chunk // (BK // DMA_ELEMS)
            col = chunk % (BK // DMA_ELEMS) * DMA_ELEMS
            src_col = fx.get_scalar(fx.crd2idx((row, col), a_layout)) % BK
            src = fx.make_view(
                fx.get_iter(gA) + ((bid_x * BM + row) * K + k_tile * BK + src_col),
                fx.make_layout(1, 1),
            )
            fx.copy_atom_call(dma_atom, src, fx.make_view(lds_ptr, fx.make_layout(1, 1)))
            lds_ptr = lds_ptr + THREADS * DMA_ELEMS
        rocdl.asyncmark()

    buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_s2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
    thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
    pA_s2r = [thr_s2r_A.partition_S(s) for s in sA]
    pB_g = thr_g2r_B.partition_S(tB)

    frag_A = thr_mma.make_fragment_A(sA[0])
    frag_B = [thr_mma.make_fragment_B(tB[None, None, 0]) for _ in range(STAGES)]  # register double buffer
    frag_C = thr_mma.make_fragment_C(tC)
    frag_A_retile = thr_s2r_A.retile(frag_A)
    frag_B_retile = [thr_g2r_B.retile(f) for f in frag_B]

    def load_b(k_tile, stage):
        fx.copy(buf_copy, pB_g[None, None, None, k_tile], frag_B_retile[stage])

    def hot_loop_scheduler():
        rocdl.sched_dsrd(4)
        for g in range_constexpr(A_READS // 2):
            rocdl.sched_mfma(4)
            if fx.const_expr(g + 2 < A_READS // 2):
                rocdl.sched_dsrd(2)
        rocdl.sched_barrier(0)  # 3: nothing of this K step moves past here

    def k_step(read_stage, next_k):
        rocdl.wait_asyncmark(0)  # RAW: this wave's DMA into read_stage has landed
        fx.gpu.barrier()  # ... every wave's has; WAR: every wave's LDS reads of the other stage are done
        if fx.const_expr(next_k is not None):
            dma_a(next_k, read_stage ^ 1)
            load_b(next_k, read_stage ^ 1)
        rocdl.sched_barrier(0)  # 1: the loads above stay ahead of every MFMA below
        for ki in range_constexpr(BK // 32):
            fx.copy(
                uni_copy,
                pA_s2r[read_stage][None, None, ki],
                frag_A_retile[None, None, ki],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, ki],
                frag_B[read_stage][None, None, ki],
                frag_C,
            )
        hot_loop_scheduler()  # 2 and 3

    frag_C.fill(0.0)
    dma_a(0, 0)
    load_b(0, 0)
    for kk in range(K_TILES // 2 - 1):
        k_step(0, kk * 2 + 1)
        k_step(1, kk * 2 + 2)
    k_step(0, K_TILES - 1)
    k_step(1, None)

    out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    thr_r2g_C = fx.make_tiled_copy_C(out_copy, tiled_mma).get_slice(tid)
    frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
    frag_C_out.store(frag_C.load().to(fx.BFloat16))
    fx.copy(out_copy, thr_r2g_C.retile(frag_C_out), thr_r2g_C.partition_S(tC))


@flyc.jit
def gemm(
    A: fx.Tensor,
    B_shuffled: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    C = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
    # The (N/16, K/32, 4, 16, 8) buffer seen as logical (n, k): n = (n_lane, n_blk), k = (elem, k_lane, k_blk).
    B = fx.Tensor(
        fx.make_view(
            fx.get_iter(B_shuffled),
            fx.make_layout(((16, N // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512))),
        )
    )
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    gemm_kernel(A, B, C, tiled_mma).launch(grid=(M // BM, N // BN, 1), block=(THREADS, 1, 1), stream=stream)


def main():
    A, B, C = gemm_inputs(M, N, K)
    B_shuffled = shuffle_weight(B)  # once, on the host side of the call
    gemm(A, B_shuffled, C)
    ok = check("gemm step7", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B_shuffled, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
