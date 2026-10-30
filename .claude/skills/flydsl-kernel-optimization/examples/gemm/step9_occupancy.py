"""GEMM step 9 -- pin the register budget: occupancy attributes on the kernel.

What it adds over step 8: two kernel attributes at launch, no code change --
  value_attrs={"rocdl.waves_per_eu": 4, "llvm.passthrough": [["amdgpu-agpr-alloc", "0"]]}
"4 waves per SIMD" caps the register allocation at 512 / 4 = 128 per lane; `amdgpu-agpr-alloc=0` tells
LLVM this kernel needs no AGPRs, so all 128 may be arch VGPRs. Step 8's epilogue had pushed the kernel
to 130 VGPRs (in the main loop; the epilogue itself is small), one granule past the 128 tier.

Why it matters here (derived from the ISA, grid card section 1): 4 waves per workgroup, 32 KiB of LDS,
so workgroups per CU = waves per SIMD by registers: 130 VGPRs -> 3, 128 -> 4. M = N = 4096 at 128x128
is 1024 workgroups on 256 CUs: 3 per CU gives 768 slots and 2 dispatch rounds (the second one-third
full); 4 per CU gives one round. What it saves (estimated): up to a third of the time at this shape,
less what a fourth co-resident workgroup per CU slows each one down; at shapes with many full rounds,
only whatever the extra wave buys (possibly nothing for a compute-bound loop). Steps 5, 6 and 8 sit in
the 2- or 3-wave tier (see the README table) -- read their timings with that in mind.

What it costs and how it fails (compile-only ISA of this kernel): with `rocdl.waves_per_eu` alone the
allocator split the 128 registers 64 arch / 64 AGPR and spilled 96 VGPRs to scratch (164 B), with
`agpr-alloc=0` it fits in 128 with no spill and one more s_nop. A `maxnreg` compile hint does not change
the binary, and a bare "passthrough" key is dropped silently (../../language/limits.md): read back
`.vgpr_count`, `.agpr_count` and `.vgpr_spill_count` after setting any attribute. A cap below what the
kernel needs turns into spills, not into the occupancy you asked for.
Card: ../../gemm/flydsl_gemm_mainloop.md sections 4, 7; ../../gemm/flydsl_gemm_grid_pricing.md sections 1, 4.

Run: python step9_occupancy.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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


@fx.union
class SharedStorage:
    a: fx.Array[fx.BFloat16, STAGES * BM * BK, 16]  # main loop: the A stages
    c: fx.Array[fx.BFloat16, BM * BN, 16]  # epilogue: the C tile, in the same bytes


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    gA = fx.rocdl.make_buffer_tensor(A)
    gB = fx.rocdl.make_buffer_tensor(B)  # B: the preshuffled buffer, viewed as logical (N, K)
    gC = fx.rocdl.make_buffer_tensor(C)
    tB = fx.flat_divide(gB, fx.make_tile(BN, BK))[None, None, bid_y, None]
    tC = fx.flat_divide(gC, fx.make_tile(BM, BN))[None, None, bid_x, bid_y]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    a_ptr = storage.a.peek().ptr
    a_layout = lds_layout(BM, BK)
    sA = [fx.make_view(a_ptr + s * BM * BK, a_layout) for s in range(STAGES)]

    wave_off = fx.Int32(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(tid // 64 * 64 * DMA_ELEMS)))
    dma_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)

    def dma_a(k_tile, stage):
        lds_ptr = a_ptr + stage * BM * BK + wave_off
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

    # Epilogue: registers -> LDS in the MFMA layout, then LDS -> registers -> global 16 bytes per lane.
    frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
    frag_C_out.store(frag_C.load().to(fx.BFloat16))
    sC = fx.make_view(storage.c.peek().ptr, fx.make_ordered_layout((BM, BN), (1, 0)))
    thr_r2s_C = fx.make_tiled_copy_C(
        fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
    ).get_slice(tid)
    fx.gpu.barrier()  # every wave is done reading A: C overwrites those bytes
    fx.copy(
        fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16),
        thr_r2s_C.retile(frag_C_out),
        thr_r2s_C.partition_D(sC),
    )
    fx.gpu.barrier()  # the whole C tile is in LDS

    thrs_n = BN // 8  # 16 lanes x 16 B per 256-byte row
    thrs_m = THREADS // thrs_n
    tiled_copy_s2g = fx.make_tiled_copy(
        uni_copy,
        fx.make_layout(((thrs_n, thrs_m), (1, 8)), ((thrs_m * 8, 1), (1, thrs_m))),
        fx.make_tile(thrs_m, BN),
    )
    thr_s2g = tiled_copy_s2g.get_slice(tid)
    pC_s = thr_s2g.partition_S(sC)
    reg_C = fx.make_fragment_like(pC_s)
    fx.copy(uni_copy, pC_s, reg_C)  # ds_read_b128
    fx.copy(buf_copy, reg_C, thr_s2g.partition_D(tC))  # buffer_store_dwordx4


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
    gemm_kernel(
        A,
        B,
        C,
        tiled_mma,
        value_attrs={
            "rocdl.waves_per_eu": 4,
            "llvm.passthrough": [["amdgpu-agpr-alloc", "0"]],
        },
    ).launch(grid=(M // BM, N // BN, 1), block=(THREADS, 1, 1), stream=stream)


def main():
    A, B, C = gemm_inputs(M, N, K)
    B_shuffled = shuffle_weight(B)  # once, on the host side of the call
    gemm(A, B_shuffled, C)
    ok = check("gemm step9", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B_shuffled, C), flops=2 * M * N * K)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
