"""GEMM step 6 -- B preshuffled on the host and loaded straight into MFMA registers.

What it adds over step 5: B leaves LDS. The host reorders B once (`shuffle_weight`, FlyDSL v0.3.2's
`tests/utils.py` layout (N/16, K/32, 4, 16, 8)) and the kernel views that buffer as the logical (N, K)
matrix again (the hierarchical layout of `preshuffle_gemm.py:775-780`). An MFMA B fragment -- lane l
holds row l%16, k 8*(l//16)..+7 -- is then 64 lanes x 16 B of one contiguous 1 KiB block, so each wave
fetches its own B columns with fully coalesced `buffer_load_dwordx4` straight into the registers the
MFMA reads. B is double-buffered in registers (`frag_B[stage]`), prefetched one K tile ahead next to
A's DMA; the compiler tracks those loads (vmcnt) itself. A keeps the LDS ping-pong of step 5.

Why it is legal to skip LDS for B here: with the waves split along N (step 2) no two waves of a
workgroup read the same B column, so LDS bought B no reuse inside the workgroup -- only a detour.
(For an operand several waves share, bypassing LDS re-reads it per wave: card datapath section 7.)

What it saves (estimated): half of the LDS traffic and LDS capacity (64 -> 32 KiB, so LDS no longer
limits a CU to two workgroups), 4 of the 20 ds_read_b128 per wave per K tile, and B's DMA M0 setup;
VGPRs drop 216 -> 138 (the peeled tail has fewer reads to hoist). What it costs: B now lives in
registers (two buffers of 16 VGPRs); a host-side reorder of B, paid once for a weight that does not change
(inference) or on every call when it does (training); an unshuffled B gives silently wrong results.
In the ISA LLVM sinks the four B loads into the middle of the MFMA stream to save registers, halving
their prefetch distance; step 7 pins them.
Card: ../../gemm/flydsl_gemm_datapath.md section 10; ../../optimize/gemm-optimization.md sections 4.2, 12.

Run: python step6_preshuffle_b.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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

    def k_step(read_stage, next_k):
        rocdl.wait_asyncmark(0)  # RAW: this wave's DMA into read_stage has landed
        fx.gpu.barrier()  # ... every wave's has; WAR: every wave's LDS reads of the other stage are done
        if fx.const_expr(next_k is not None):
            dma_a(next_k, read_stage ^ 1)
            load_b(next_k, read_stage ^ 1)
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
    ok = check("gemm step6", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B_shuffled, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
