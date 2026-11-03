"""GEMM step 0 -- one output element per thread, scalar loads.

C (M, N) = A (M, K) @ B (N, K)^T, bf16 in and out, fp32 accumulate (all gemm/step* files).

What it adds: the baseline. A 256-thread workgroup owns a 16x16 block of C; each thread walks K for
its one element, reading A[m, k] and B[n, k] element by element from global memory and doing one fp32
multiply-add per pair. No MFMA, no sharing of a load between threads except through the caches.
The source loads one bf16 at a time; LLVM unrolls the loop 8x and merges the loads into one
global_load_dwordx4 per operand per 8 k (compile-only ISA), then waits for both before the 8 FMAs.

What it costs (estimated): one 16-byte load per operand per 8 FMAs and a full memory round trip per 8 k,
with 16 different B rows per 16 lanes; the loop is latency-bound long before it is compute-bound and
the matrix cores are idle. Expect well under 1% of gfx950's ~2.5 PFLOP/s bf16 dense peak (estimated;
`bench` prints the measured number).

Run: python step0_naive.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
"""

# ruff: noqa: B008  (FlyDSL's `stream: fx.Stream = fx.Stream(None)` default)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
from common import bench, check, gemm_inputs, gemm_reference

M, N, K = 4096, 4096, 4096
TILE = 16  # 16 x 16 outputs per workgroup, one per thread


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx
    m = bid_x * TILE + tid // TILE
    n = bid_y * TILE + tid % TILE  # neighbouring threads -> neighbouring columns of C

    acc = fx.Float32(0.0)
    for k in range(K):  # a runtime loop; `acc` is carried through it
        acc = acc + A[m, k].to(fx.Float32) * B[n, k].to(fx.Float32)
    C[m, n] = acc.to(fx.BFloat16)


@flyc.jit
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    # Static row-major views: shapes and strides become compile-time constants.
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
    C = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
    gemm_kernel(A, B, C).launch(grid=(M // TILE, N // TILE, 1), block=(TILE * TILE, 1, 1), stream=stream)


def main():
    A, B, C = gemm_inputs(M, N, K)
    gemm(A, B, C)
    ok = check("gemm step0", C, None if A.device.type == "cpu" else gemm_reference(A, B))
    if ok:
        bench(lambda: gemm(A, B, C), flops=2 * M * N * K, iters=3)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
