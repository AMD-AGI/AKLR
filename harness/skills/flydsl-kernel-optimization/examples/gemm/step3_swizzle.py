"""GEMM step 3 -- XOR-swizzled LDS layout (16-byte granularity).

What it adds over step 2: the LDS views of A and B become `make_composed_layout(Swizzle<3,3,3>, row-major)`
instead of plain row-major. On the element offset of a 128x64 bf16 tile, Swizzle<3,3,3> XORs bits 6..8
(row % 8) into bits 3..5 (which 16-byte chunk of the 128-byte row): chunk' = chunk ^ (row % 8). Whole
16-byte chunks move, so every ds_read_b128 / ds_write_b128 stays one instruction. Writer (the G2S copy)
and readers (the MFMA operand copies) both go through the same composed view, so they agree by
construction; nothing else in the kernel changes. This is the layout FlyDSL's bf16 kernels use
(`kernels/gemm/gemm_a16w16_gfx950_utils.py:make_lds_layout`, preshuffle_gemm's sync path).

Why (gfx950 LDS: 64 banks x 4 B; a ds_read_b128 is served 16 lanes at a time): an MFMA operand read
has lanes 0-15 reading rows 0-15 at the same 16-byte chunk. Row-major with 128-byte rows maps those rows
onto two 4-bank groups: worked through for the lane groups of a b128 phase, a 4-way conflict on every
operand read (step 2). After the swizzle the 16 lanes of a phase land on 16 different 4-bank groups:
conflict-free; the G2S writes stay conflict-free.
What it saves (estimated): up to 4x on the LDS read time of each K tile, which in step 2 sits between
the barrier and the MFMAs with nothing overlapping it. What it costs: one VALU XOR per base address
(4 v_bitop3_b32, outside the loop), no extra LDS bytes. Check with the SQ_LDS_BANK_CONFLICT counter, not
with the ISA: the ds_read/ds_write counts do not change.
Card: ../../gemm/flydsl_gemm_datapath.md sections 2-3; ../../optimize/gemm-optimization.md section 3.

Run: python step3_swizzle.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
from common import bench, check, gemm_inputs, gemm_reference

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 64
THREADS = 256


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
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_mma: fx.TiledMma,
    tiled_copy_g2s: fx.TiledCopy,
):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    gA = fx.rocdl.make_buffer_tensor(A)
    gB = fx.rocdl.make_buffer_tensor(B)
    gC = fx.rocdl.make_buffer_tensor(C)
    tA = fx.flat_divide(gA, fx.make_tile(BM, BK))[None, None, bid_x, None]  # (BM, BK, k-tiles)
    tB = fx.flat_divide(gB, fx.make_tile(BN, BK))[None, None, bid_y, None]  # (BN, BK, k-tiles)
    tC = fx.flat_divide(gC, fx.make_tile(BM, BN))[None, None, bid_x, bid_y]  # (BM, BN)

    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    sA = fx.make_view(lds.a.ptr, lds_layout(BM, BK))  # swizzled (was row-major)
    sB = fx.make_view(lds.b.ptr, lds_layout(BN, BK))

    buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)  # global <-> registers
    uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)  # LDS <-> registers
    thr_mma = tiled_mma.thr_slice(tid)
    thr_g2s = tiled_copy_g2s.get_slice(tid)
    thr_s2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
    thr_s2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)

    pA_g, pA_s = thr_g2s.partition_S(tA), thr_g2s.partition_D(sA)
    pB_g, pB_s = thr_g2s.partition_S(tB), thr_g2s.partition_D(sB)
    pA_s2r = thr_s2r_A.partition_S(sA)
    pB_s2r = thr_s2r_B.partition_S(sB)

    reg_A = fx.make_fragment_like(pA_s)  # staging registers for global -> LDS
    reg_B = fx.make_fragment_like(pB_s)
    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(tC)
    frag_A_retile = thr_s2r_A.retile(frag_A)
    frag_B_retile = thr_s2r_B.retile(frag_B)

    frag_C.fill(0.0)
    for k in range(K // BK):
        fx.copy(buf_copy, pA_g[None, None, None, k], reg_A)
        fx.copy(buf_copy, pB_g[None, None, None, k], reg_B)
        fx.copy(uni_copy, reg_A, pA_s)
        fx.copy(uni_copy, reg_B, pB_s)
        fx.gpu.barrier()  # the tile is in LDS for every wave
        for ki in fx.range_constexpr(BK // 32):
            fx.copy(uni_copy, pA_s2r[None, None, ki], frag_A_retile[None, None, ki])
            fx.copy(uni_copy, pB_s2r[None, None, ki], frag_B_retile[None, None, ki])
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, ki],
                frag_B[None, None, ki],
                frag_C,
            )
        fx.gpu.barrier()  # every wave is done reading before the next tile overwrites LDS

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
        fx.make_layout((1, 4, 1), (0, 1, 0)),  # four waves along N
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    # Global -> LDS copy: 8 threads x 16 B cover one 128-byte row, 32 rows per pass (preshuffle_gemm's G2S).
    thrs_k = BK // 8
    thrs_m = THREADS // thrs_k
    tiled_copy_g2s = fx.make_tiled_copy(
        fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
        fx.make_layout(((thrs_k, thrs_m), (1, 8)), ((thrs_m * 8, 1), (1, thrs_m))),
        fx.make_tile(thrs_m, BK),
    )
    gemm_kernel(A, B, C, tiled_mma, tiled_copy_g2s).launch(
        grid=(M // BM, N // BN, 1), block=(THREADS, 1, 1), stream=stream
    )


def main():
    A, B, C = gemm_inputs(M, N, K)
    gemm(A, B, C)
    ok = check("gemm step3", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
