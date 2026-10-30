"""MoE step 0 -- GEMVs: stage 1 one workgroup per (token, chosen expert), stage 2 one per token.

bf16 fused MoE, all moe/step* files: x (T, D); the router picks TOPK experts per token with weights w;
W1 (E, 2I, D) holds each expert's gate rows then up rows, W2 (E, D, I) its down projection.
  h[t, j] = silu(W1g[e] x[t]) * (W1u[e] x[t])          (stage 1, bf16 intermediate)
  out[t]  = sum_j w[t, j] * W2[e] h[t, j]              (stage 2), with e = topk_ids[t, j].
Sizes (common.py): T=64 tokens, E=8 experts, TOPK=2, D=1024, I=512 -- a decode-sized batch.

What it is: the baseline. Stage 1 launches one 256-thread workgroup per routed item (t, j) -- 128 of
them; each thread computes two of the item's I outputs, i.e. two gate and two up dot products of
length D, reading x[t] and its expert's weight rows element by element (LLVM merges 4 into one
global_load_dwordx2). Stage 2 launches one workgroup
per token; each thread owns four outputs and walks the token's TOPK items in order, so the weighted
sum is a plain fixed-order loop (deterministic, no atomics, no extra kernel).

What it costs (estimated): every routed item streams its expert's whole W1 (2 MiB) and, in stage 2,
W2 (1 MiB): 128 x 2 MiB + 128 x 1 MiB = 384 MiB of weight reads for 24 MiB of distinct weights, a
16x redundancy that the 4 MiB-per-XCD L2 can only partly absorb; lanes read different weight rows, so
each load touches 64 different cache lines. No matrix core. Expect memory-bound at a small fraction of
HBM bandwidth (estimated; `bench` prints the measured time).
Card: ../../moe/flydsl_moe_decode.md; ../../grouped_gemm/flydsl_grouped_moe_inference.md section 2.

Run: python step0_gemv.py  (COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950 to only compile).
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
from common import MOE_D as D
from common import MOE_E as E
from common import MOE_I as I
from common import MOE_T as T
from common import MOE_TOPK as TOPK
from flydsl.expr import range_constexpr

THREADS = 256


def silu_mul(g, u):
    return g * (1.0 / (1.0 + fx.math.exp(-g))) * u


@flyc.kernel
def moe_stage1_kernel(x: fx.Tensor, w1: fx.Tensor, topk_ids: fx.Tensor, h: fx.Tensor):
    tid = fx.thread_idx.x
    item = fx.block_idx.x  # routed item t * TOPK + j
    t = item // TOPK
    e = topk_ids[item]
    for o in range_constexpr(I // THREADS):
        i = o * THREADS + tid
        g = fx.Float32(0.0)
        u = fx.Float32(0.0)
        for d in range(D):
            xv = x[t, d].to(fx.Float32)
            g = g + w1[e, i, d].to(fx.Float32) * xv
            u = u + w1[e, I + i, d].to(fx.Float32) * xv
        h[item, i] = silu_mul(g, u).to(fx.BFloat16)


@flyc.kernel
def moe_stage2_kernel(h: fx.Tensor, w2: fx.Tensor, topk_ids: fx.Tensor, topk_w: fx.Tensor, out: fx.Tensor):
    tid = fx.thread_idx.x
    t = fx.block_idx.x
    for o in range_constexpr(D // THREADS):
        dcol = o * THREADS + tid
        acc = fx.Float32(0.0)
        for j in range_constexpr(TOPK):  # fixed order: the sum is the same on every run
            item = t * TOPK + j
            e = topk_ids[item]
            y = fx.Float32(0.0)
            for i in range(I):
                y = y + w2[e, dcol, i].to(fx.Float32) * h[item, i].to(fx.Float32)
            acc = acc + topk_w[item] * y
        out[t, dcol] = acc.to(fx.BFloat16)


@flyc.jit
def fused_moe(
    x: fx.Tensor,
    w1: fx.Tensor,
    w2: fx.Tensor,
    topk_ids: fx.Tensor,
    topk_w: fx.Tensor,
    h: fx.Tensor,
    out: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    x = fx.Tensor(fx.make_view(fx.get_iter(x), fx.make_layout((T, D), (D, 1))))
    w1 = fx.Tensor(fx.make_view(fx.get_iter(w1), fx.make_layout((E, 2 * I, D), (2 * I * D, D, 1))))
    w2 = fx.Tensor(fx.make_view(fx.get_iter(w2), fx.make_layout((E, D, I), (D * I, I, 1))))
    topk_ids = fx.Tensor(fx.make_view(fx.get_iter(topk_ids), fx.make_layout(T * TOPK, 1)))
    topk_w = fx.Tensor(fx.make_view(fx.get_iter(topk_w), fx.make_layout(T * TOPK, 1)))
    h = fx.Tensor(fx.make_view(fx.get_iter(h), fx.make_layout((T * TOPK, I), (I, 1))))
    out = fx.Tensor(fx.make_view(fx.get_iter(out), fx.make_layout((T, D), (D, 1))))
    moe_stage1_kernel(x, w1, topk_ids, h).launch(grid=(T * TOPK, 1, 1), block=(THREADS, 1, 1), stream=stream)
    moe_stage2_kernel(h, w2, topk_ids, topk_w, out).launch(
        grid=(T, 1, 1), block=(THREADS, 1, 1), stream=stream
    )


def main():
    x, w1, w2, topk_ids, topk_w = moe_inputs()
    h = torch.empty(T * TOPK, I, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty(T, D, dtype=torch.bfloat16, device=DEVICE)
    fused_moe(x, w1, w2, topk_ids, topk_w, h, out)
    ref = None if DEVICE == "cpu" else moe_reference(x, w1, w2, topk_ids, topk_w)
    ok = check("moe step0", out, ref)
    if ok:
        bench(lambda: fused_moe(x, w1, w2, topk_ids, topk_w, h, out))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
