"""GEMM step 2 -- a 128x128 workgroup tile on four waves, A and B staged through LDS.

What it adds over step 1: on-chip reuse. A 256-thread workgroup (4 waves) owns a 128x128 block of C and
walks K in 64-wide tiles. Per tile, all 256 threads copy the 128x64 A tile and the 128x64 B tile
global -> registers -> LDS (one 16-byte load and one 16-byte LDS write per 8 elements, rows read whole
128-byte lines), a barrier publishes them, and each wave then reads its MFMA operands from LDS. The four
waves split N (`make_layout((1, 4, 1), (0, 1, 0))`, the split FlyDSL's preshuffle_gemm uses): wave w owns
columns 16w..16w+15 of every 64, so every wave reads all of A and a quarter of B. Per K tile a wave
issues 8 (M) x 2 (N) x 2 (K) = 32 MFMAs from 16 + 4 LDS reads. The LDS layout is plain row-major.

What it saves (estimated): each A element fetched from global now feeds 128 output columns and each B
element 128 output rows (step 1: 16), an 8x cut in global/L2 traffic per MAC, and 32 MFMAs per K step
per wave instead of 1. What it costs (estimated): 32 KiB of LDS per workgroup; two barriers per K tile;
the loads, LDS writes and MFMAs of one tile run back to back with nothing overlapping them; and the
row-major LDS layout: each row is 128 B, exactly one pass over the 64 banks, so the rows an MFMA operand
read touches (lanes 0-15 = rows 0-15, same 16-byte chunk) collide -- worked through for the 16-lane
phases of ds_read_b128, every read is a 4-way bank conflict (the G2S writes, 8 lanes per row, are
conflict-free). Step 3 removes it.
Card: ../../gemm/flydsl_gemm_datapath.md sections 1, 7; ../../optimize/gemm-optimization.md section 1.

Run: python step2_lds.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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
    sA = fx.make_view(lds.a.ptr, fx.make_ordered_layout((BM, BK), (1, 0)))  # row-major
    sB = fx.make_view(lds.b.ptr, fx.make_ordered_layout((BN, BK), (1, 0)))

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
    ok = check("gemm step2", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
