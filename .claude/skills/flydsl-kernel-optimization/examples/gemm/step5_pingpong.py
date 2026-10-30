"""GEMM step 5 -- two LDS stages (ping-pong): the next K tile's DMA overlaps this tile's MFMAs.

What it adds over step 4: a second LDS buffer per operand and a reordered loop. Tile t is read from
stage t % 2 while the DMA for tile t+1 fills the other stage. Each K step is
  wait_asyncmark(0)   this wave's DMA for tile t has landed
  gpu.barrier()       ... every wave's has, and every wave has finished reading tile t-1 (the other stage)
  DMA tile t+1 -> other stage; asyncmark
  MFMAs on tile t
so one barrier per K tile guards both hazards (step 4 needed two). The loop body handles two tiles,
stage 0 then stage 1, so every stage index is a compile-time constant (FlyDSL preshuffle_gemm's
`pipeline_2stage` shape); the last two tiles are peeled because nothing is left to prefetch.

The barrier is `fx.gpu.barrier()`, not a bare `rocdl.s_barrier()`: gpu.barrier carries the workgroup
fence, and in this file's ISA every `s_barrier` comes after the stage's ds_reads have been waited for
(`lgkmcnt(0)`). A bare s_barrier is no fence: a ds_read of the stage may still be in flight when other
waves start the DMA that overwrites it, a race no single-wave test catches. FlyDSL's gemm_a16w16
instead pins each stage's MFMAs with `rocdl.sched_barrier(0)` -- step 7.

What it saves (estimated): the global -> LDS latency (several hundred cycles, over a thousand from HBM)
is hidden behind the previous tile's 32 MFMAs per wave instead of being paid in full on every K step, and
one barrier per tile goes away. What it costs: LDS doubles to 64 KiB per workgroup (two workgroups per
CU at most), and the compile-only ISA peaks at 216 VGPRs -- not in the loop (v0-v119) but in the peeled
last tile, where LLVM hoists all 20 LDS reads ahead of the MFMAs: 2 waves per SIMD instead of 4 (step 7's
closing fence brings it back down).
Card: ../../gemm/flydsl_gemm_mainloop.md (the 2-stage ping-pong baseline); ../../optimize/gemm-optimization.md
section 2; ../../optimize/prefetch-data-load.md.

Run: python step5_pingpong.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
from common import bench, check, gemm_inputs, gemm_reference
from flydsl.expr import range_constexpr, rocdl

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 64
THREADS = 256
DMA_ELEMS = 8
STAGES = 2
K_TILES = K // BK


def lds_layout(rows, cols):
    """Row-major (rows, cols) with 16-byte chunk index XOR (row % 8): Swizzle<3, 3, 3> on bf16 offsets."""
    return fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((rows, cols), (1, 0)),
    )


@fx.struct
class SharedStorage:
    a: fx.Array[fx.BFloat16, STAGES * BM * BK, 16]
    b: fx.Array[fx.BFloat16, STAGES * BN * BK, 16]


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    gA = fx.rocdl.make_buffer_tensor(A)
    gB = fx.rocdl.make_buffer_tensor(B)
    gC = fx.rocdl.make_buffer_tensor(C)
    tC = fx.flat_divide(gC, fx.make_tile(BM, BN))[None, None, bid_x, bid_y]

    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    a_layout, b_layout = lds_layout(BM, BK), lds_layout(BN, BK)
    sA = [fx.make_view(lds.a.ptr + s * BM * BK, a_layout) for s in range(STAGES)]
    sB = [fx.make_view(lds.b.ptr + s * BN * BK, b_layout) for s in range(STAGES)]

    wave_off = fx.Int32(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(tid // 64 * 64 * DMA_ELEMS)))
    dma_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    def dma_tile(g_iter, row0, k_tile, lds_ptr, layout):
        lds_ptr = lds_ptr + wave_off
        for i in range_constexpr(BM * BK // (THREADS * DMA_ELEMS)):
            chunk = THREADS * i + tid
            row = chunk // (BK // DMA_ELEMS)
            col = chunk % (BK // DMA_ELEMS) * DMA_ELEMS
            src_col = fx.get_scalar(fx.crd2idx((row, col), layout)) % BK
            src = fx.make_view(
                g_iter + ((row0 + row) * K + k_tile * BK + src_col),
                fx.make_layout(1, 1),
            )
            fx.copy_atom_call(dma_atom, src, fx.make_view(lds_ptr, fx.make_layout(1, 1)))
            lds_ptr = lds_ptr + THREADS * DMA_ELEMS

    def dma_stage(k_tile, stage):
        dma_tile(fx.get_iter(gA), bid_x * BM, k_tile, lds.a.ptr + stage * BM * BK, a_layout)
        dma_tile(fx.get_iter(gB), bid_y * BN, k_tile, lds.b.ptr + stage * BN * BK, b_layout)
        rocdl.asyncmark()

    buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_s2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
    thr_s2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
    pA_s2r = [thr_s2r_A.partition_S(s) for s in sA]
    pB_s2r = [thr_s2r_B.partition_S(s) for s in sB]

    frag_A = thr_mma.make_fragment_A(sA[0])
    frag_B = thr_mma.make_fragment_B(sB[0])
    frag_C = thr_mma.make_fragment_C(tC)
    frag_A_retile = thr_s2r_A.retile(frag_A)
    frag_B_retile = thr_s2r_B.retile(frag_B)

    def k_step(read_stage, next_k):
        rocdl.wait_asyncmark(0)  # RAW: this wave's DMA into read_stage has landed
        fx.gpu.barrier()  # ... every wave's has; WAR: every wave's LDS reads of the other stage are done
        if fx.const_expr(next_k is not None):
            dma_stage(next_k, read_stage ^ 1)  # prefetch: lands while the MFMAs below run
        for ki in range_constexpr(BK // 32):
            fx.copy(
                uni_copy,
                pA_s2r[read_stage][None, None, ki],
                frag_A_retile[None, None, ki],
            )
            fx.copy(
                uni_copy,
                pB_s2r[read_stage][None, None, ki],
                frag_B_retile[None, None, ki],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, ki],
                frag_B[None, None, ki],
                frag_C,
            )

    frag_C.fill(0.0)
    dma_stage(0, 0)  # prologue
    for kk in range(K_TILES // 2 - 1):  # tiles 2kk (stage 0) and 2kk+1 (stage 1)
        k_step(0, kk * 2 + 1)
        k_step(1, kk * 2 + 2)
    k_step(0, K_TILES - 1)  # peeled tail: the last prefetch ...
    k_step(1, None)  # ... and the last tile

    out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    thr_r2g_C = fx.make_tiled_copy_C(out_copy, tiled_mma).get_slice(tid)
    frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
    frag_C_out.store(frag_C.load().to(fx.BFloat16))
    fx.copy(out_copy, thr_r2g_C.retile(frag_C_out), thr_r2g_C.partition_S(tC))


@flyc.jit
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
    C = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    gemm_kernel(A, B, C, tiled_mma).launch(grid=(M // BM, N // BN, 1), block=(THREADS, 1, 1), stream=stream)


def main():
    A, B, C = gemm_inputs(M, N, K)
    gemm(A, B, C)
    ok = check("gemm step5", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
