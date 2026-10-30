"""GEMM step 1 -- one wave per 16x16 tile of C on the matrix core (v_mfma_f32_16x16x32_bf16).

What it adds over step 0: the MFMA. A 64-lane workgroup (one wave) owns a 16x16 block of C and walks K
in steps of 32. Per step every lane loads 8 consecutive bf16 of one A row and 8 of one B row (16 bytes
each, one buffer_load_dwordx4) straight from global memory into the MFMA operand registers, and the
wave issues one v_mfma_f32_16x16x32_bf16 (16*16*32 MACs). The layout API does the bookkeeping:
`make_tiled_mma` describes which lane holds which (m, k) / (n, k) / (m, n) element, and
`make_tiled_copy_A/B/C` derive copies whose per-lane slices match those registers.

Lane map of the bf16 16x16x32 atom (printed from `tiled_mma.tv_layout_A_tiled` etc.):
  A, B: lane l holds row l%16, k = 8*(l//16) .. +7;   C: lane l holds column l%16, rows 4*(l//16) .. +3.

What it saves and costs (estimated): 16x fewer load instructions per MAC than step 0 (two 16-byte loads
per 8192 MACs) and the work moves to the matrix core, so expect one to two orders of magnitude over
step 0. Still far from peak: the MFMAs wait on their own global loads (the source prefetches nothing;
LLVM's 4x unroll of the K loop keeps up to 8 loads in flight, and that is all the latency hiding), every
A row is re-read by all N/16 workgroups of its row and every B row by all M/16 of its column (no on-chip
reuse beyond the caches), and the C stores are 2-byte scalar stores.
Card: ../../optimize/gemm-optimization.md section 6.2 (MFMA atoms); ../../gemm/flydsl_gemm_mainloop.md section 5.

Run: python step1_mfma.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
from common import bench, check, gemm_inputs, gemm_reference

M, N, K = 4096, 4096, 4096
BM, BN, BK = 16, 16, 32  # one MFMA atom per workgroup tile


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    gA = fx.rocdl.make_buffer_tensor(A)
    gB = fx.rocdl.make_buffer_tensor(B)
    gC = fx.rocdl.make_buffer_tensor(C)
    tA = fx.flat_divide(gA, fx.make_tile(BM, BK))[None, None, bid_x, None]  # (BM, BK, k-tiles)
    tB = fx.flat_divide(gB, fx.make_tile(BN, BK))[None, None, bid_y, None]  # (BN, BK, k-tiles)
    tC = fx.flat_divide(gC, fx.make_tile(BM, BN))[None, None, bid_x, bid_y]  # (BM, BN)

    buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)  # 16 B per lane
    thr_mma = tiled_mma.thr_slice(tid)
    thr_g2r_A = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
    thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)
    pA = thr_g2r_A.partition_S(tA)
    pB = thr_g2r_B.partition_S(tB)

    frag_A = thr_mma.make_fragment_A(tA[None, None, 0])
    frag_B = thr_mma.make_fragment_B(tB[None, None, 0])
    frag_C = thr_mma.make_fragment_C(tC)
    frag_A_retile = thr_g2r_A.retile(frag_A)  # the same registers, seen with the copy's layout
    frag_B_retile = thr_g2r_B.retile(frag_B)

    frag_C.fill(0.0)
    for k in range(K // BK):
        fx.copy(buf_copy, pA[None, None, None, k], frag_A_retile)
        fx.copy(buf_copy, pB[None, None, None, k], frag_B_retile)
        fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)

    # Epilogue: f32 -> bf16 in registers, then one 2-byte store per element (the MFMA C layout).
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
        fx.make_layout((1, 1, 1), (0, 0, 0)),  # one wave
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),  # K order of the bf16 K=32 atom
    )
    gemm_kernel(A, B, C, tiled_mma).launch(grid=(M // BM, N // BN, 1), block=(64, 1, 1), stream=stream)


def main():
    A, B, C = gemm_inputs(M, N, K)
    gemm(A, B, C)
    ok = check("gemm step1", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
