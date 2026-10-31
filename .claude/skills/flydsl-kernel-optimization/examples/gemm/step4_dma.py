"""GEMM step 4 -- global -> LDS by DMA (buffer_load_dwordx4 ... lds) instead of through registers.

What it adds over step 3: the A and B tiles no longer pass through VGPRs. Each lane issues one
`fx.rocdl.cdna4.BufferLoadAsyncLDS128b` copy per 16 bytes: the memory unit writes the data straight into
LDS at M0 + 16 * lane, where M0 is the wave's base address (wave-uniform, `readfirstlane`). Completion is
tracked by the program, not by the compiler: `rocdl.asyncmark()` closes the group of DMAs just issued and
`rocdl.wait_asyncmark(0)` waits until no group is outstanding; then `s_barrier` makes every wave's part
visible to all. (`fx.rocdl.BufferCopyLDS128b` is the compiler-tracked form: the compiler then drains
vmcnt(0) before any LDS read, which would serialize step 5's prefetch behind every read.)

The swizzle moves to the global side. The DMA always fills LDS linearly (lane order), so the lane that
fills physical chunk c of row r must fetch the logical column that the swizzled layout stores there:
c ^ (r % 8) chunks, which is also what `crd2idx` of the same composed layout returns (XOR is its own
inverse) -- the readers keep using the composed view unchanged, so writer and reader still share one
layout object (FlyDSL `gemm_a16w16_gfx950.py:async_load_to_lds` does the same).

What it saves (estimated): the 8 ds_write_b128 per thread per K tile and the vmcnt waits that fed them
(the loop's s_waitcnt drop from 20 to 4), i.e. VALU/LDS issue slots; LDS write bandwidth is the same.
On a bf16 8-wave loop at M=16384, staging through VGPRs instead of DMA measured 10-15% slower (card
mainloop, field; timing-only builds). The usual register saving does not show here: VGPRs go 112 -> 120
because each lane keeps its 8 DMA source offsets live. What it costs: an s_mov to M0 per DMA (most
followed by an s_nop the hazard recognizer inserts) and hand-written completion waits -- forget one and
the MFMAs read stale LDS.
Card: ../../gemm/flydsl_gemm_datapath.md sections 4-5; ../../gemm/flydsl_gemm_mainloop.md (async copy).

Run: python step4_dma.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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
DMA_ELEMS = 8  # one 16-byte DMA per lane = 8 bf16


def lds_layout(rows, cols):
    """Row-major (rows, cols) with 16-byte chunk index XOR (row % 8): Swizzle<3, 3, 3> on bf16 offsets."""
    return fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((rows, cols), (1, 0)),
    )


@fx.struct
class SharedStorage:
    a: fx.Array[fx.BFloat16, BM * BK, 16]
    b: fx.Array[fx.BFloat16, BN * BK, 16]


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
    sA = fx.make_view(lds.a.ptr, a_layout)
    sB = fx.make_view(lds.b.ptr, b_layout)

    # Each wave fills 64 lanes x 16 B = 1 KiB per pass; its base is wave-uniform (goes to M0).
    wave_off = fx.Int32(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(tid // 64 * 64 * DMA_ELEMS)))
    dma_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    def dma_tile(g_iter, row0, k_tile, lds_ptr, layout):
        lds_ptr = lds_ptr + wave_off
        for i in range_constexpr(BM * BK // (THREADS * DMA_ELEMS)):
            chunk = THREADS * i + tid  # physical 16-byte chunk this lane fills, in LDS order
            row = chunk // (BK // DMA_ELEMS)
            col = chunk % (BK // DMA_ELEMS) * DMA_ELEMS
            src_col = fx.get_scalar(fx.crd2idx((row, col), layout)) % BK  # = col ^ 8 * (row % 8)
            src = fx.make_view(
                g_iter + ((row0 + row) * K + k_tile * BK + src_col),
                fx.make_layout(1, 1),
            )
            fx.copy_atom_call(dma_atom, src, fx.make_view(lds_ptr, fx.make_layout(1, 1)))
            lds_ptr = lds_ptr + THREADS * DMA_ELEMS

    buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_s2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
    thr_s2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
    pA_s2r = thr_s2r_A.partition_S(sA)
    pB_s2r = thr_s2r_B.partition_S(sB)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(tC)
    frag_A_retile = thr_s2r_A.retile(frag_A)
    frag_B_retile = thr_s2r_B.retile(frag_B)

    frag_C.fill(0.0)
    for k in range(K // BK):
        dma_tile(fx.get_iter(gA), bid_x * BM, k, lds.a.ptr, a_layout)
        dma_tile(fx.get_iter(gB), bid_y * BN, k, lds.b.ptr, b_layout)
        rocdl.asyncmark()  # close this tile's DMA group
        rocdl.wait_asyncmark(0)  # ... and wait for it (this wave's part)
        rocdl.s_barrier()  # every wave's part has landed
        for ki in range_constexpr(BK // 32):
            fx.copy(uni_copy, pA_s2r[None, None, ki], frag_A_retile[None, None, ki])
            fx.copy(uni_copy, pB_s2r[None, None, ki], frag_B_retile[None, None, ki])
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, ki],
                frag_B[None, None, ki],
                frag_C,
            )
        fx.gpu.barrier()  # every wave is done reading before the next DMA overwrites LDS

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
    ok = check("gemm step4", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
