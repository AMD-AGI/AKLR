"""Helpers the example steps share. Nothing here is the technique a step teaches.

- the run mode: ``COMPILE_ONLY=1`` compiles for ``ARCH`` (gfx950) on CPU tensors and launches nothing;
  otherwise tensors live on the GPU and every step is checked against torch;
- the correctness check and its tolerance, and a small CUDA-event timer;
- the host-side B preshuffle that ``gemm/step6`` onward consumes (a copy of FlyDSL v0.3.2
  ``tests/utils.py:shuffle_weight``, so the examples run from the wheel alone);
- the MoE problem: inputs, routing, the host-side routing sort of ``moe/step1``, and the torch reference.
"""

import os

import torch

COMPILE_ONLY = os.environ.get("COMPILE_ONLY", "0") == "1"
DEVICE = "cpu" if COMPILE_ONLY else "cuda"

# Tolerance for a bf16 output with fp32 accumulation, against an fp32 reference:
# |out - ref| <= ATOL + RTOL * |ref|. bf16 rounding of the output alone is <= 2^-8 relative (0.4%).
ATOL = 1e-2
RTOL = 1e-2


def check(name, out, ref, atol=ATOL, rtol=RTOL):
    """Print the max error of ``out`` against ``ref``; return True when every element is within tolerance."""
    if COMPILE_ONLY:
        print(f"{name}: compiled for {os.environ.get('ARCH', '?')} (COMPILE_ONLY=1); not run, not checked")
        return True
    out32, ref32 = out.float(), ref.float()
    err = (out32 - ref32).abs()
    violation = (err - (atol + rtol * ref32.abs())).max().item()
    ok = violation <= 0
    print(
        f"{name}: max abs error {err.max().item():.4g} (max |ref| {ref32.abs().max().item():.4g}), "
        f"tolerance {atol:g} + {rtol:g}*|ref| -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def bench(fn, flops=None, warmup=3, iters=20):
    """Median time of ``fn()`` over ``iters`` CUDA-event-timed calls; prints it (and TFLOP/s when given)."""
    if COMPILE_ONLY:
        return None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1e3)
    us = sorted(times)[len(times) // 2]
    extra = f", {flops / us / 1e6:.1f} TFLOP/s" if flops else ""
    print(f"  median {us:.1f} us over {iters} calls (hot L2, no flush){extra}")
    return us


def shuffle_weight(x, layout=(16, 16)):
    """FlyDSL v0.3.2 ``tests/utils.py:shuffle_weight`` (16-bit path): (N, K) -> the same bytes ordered
    (N/16, K/32, 4, 16, 8), so one 16-byte load per lane lands in the MFMA B-operand registers.
    """
    IN, IK = layout
    BK = IK * 2
    KP = 16 // x.element_size()
    BN = IN
    assert x.shape[-2] % BN == 0 and x.shape[-1] % BK == 0
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // KP, KP)
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)


# ---------------------------------------------------------------------------------------------------
# MoE problem shared by moe/step*: bf16 x (T, D); router -> top-k experts and weights per token;
# W1 (E, 2I, D) = [gate rows; up rows]; W2 (E, D, I).  out[t] = sum_j w[t,j] * W2[e] (silu(g) * u).
# ---------------------------------------------------------------------------------------------------
MOE_T, MOE_E, MOE_TOPK, MOE_D, MOE_I = 64, 8, 2, 1024, 512
MOE_BM = 16  # rows per sorted block = tile-M of the MFMA steps (the "sort unit")
# Upper bound on padded M-blocks: each expert pads at most BM-1 rows (FlyDSL mxfp_moe/gemm1.py:59-67).
MOE_MAX_BLOCKS = (MOE_T * MOE_TOPK + min(MOE_T * MOE_TOPK, MOE_E) * (MOE_BM - 1) + MOE_BM - 1) // MOE_BM


def moe_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(MOE_T, MOE_D, generator=g).to(torch.bfloat16)
    w1 = (torch.randn(MOE_E, 2 * MOE_I, MOE_D, generator=g) / MOE_D**0.5).to(torch.bfloat16)
    w2 = (torch.randn(MOE_E, MOE_D, MOE_I, generator=g) / MOE_I**0.5).to(torch.bfloat16)
    logits = torch.randn(MOE_T, MOE_E, generator=g)
    topk_w, topk_ids = torch.topk(torch.softmax(logits, dim=-1), MOE_TOPK, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).float()
    return [t.to(DEVICE) for t in (x, w1, w2, topk_ids.to(torch.int32), topk_w)]


def moe_reference(x, w1, w2, topk_ids, topk_w):
    """fp32 reference of the fused MoE (the bf16 intermediate h is rounded like the kernels round it)."""
    out = torch.zeros(MOE_T, MOE_D, dtype=torch.float32, device=x.device)
    for t in range(MOE_T):
        for j in range(MOE_TOPK):
            e = int(topk_ids[t, j])
            gu = w1[e].float() @ x[t].float()
            h = (torch.nn.functional.silu(gu[:MOE_I]) * gu[MOE_I:]).to(torch.bfloat16).float()
            out[t] += topk_w[t, j] * (w2[e].float() @ h)
    return out


def moe_sort(topk_ids):
    """Host-side routing sort (moe/step1): routed items t*TOPK+j grouped by expert in ascending item order,
    each expert padded to a multiple of BM with the sentinel T*TOPK.  Returns sorted_ids (MAX_BLOCKS*BM,)
    and block_expert (MAX_BLOCKS,), -1 past the last real block.  moe/step2 builds the same arrays on the GPU.
    """
    flat = topk_ids.flatten().cpu()
    sentinel = MOE_T * MOE_TOPK
    sorted_ids = torch.full((MOE_MAX_BLOCKS * MOE_BM,), sentinel, dtype=torch.int32)
    block_expert = torch.full((MOE_MAX_BLOCKS,), -1, dtype=torch.int32)
    row = 0
    for e in range(MOE_E):
        items = torch.nonzero(flat == e).flatten().to(torch.int32)
        if items.numel() == 0:
            continue
        sorted_ids[row : row + items.numel()] = items
        nblk = (items.numel() + MOE_BM - 1) // MOE_BM
        block_expert[row // MOE_BM : row // MOE_BM + nblk] = e
        row += nblk * MOE_BM
    return sorted_ids.to(topk_ids.device), block_expert.to(topk_ids.device)


# ---------------------------------------------------------------------------------------------------
# GEMM problem shared by gemm/step*: C (M, N) = A (M, K) @ B (N, K)^T, bf16 in and out, fp32 accumulate.
# ---------------------------------------------------------------------------------------------------
def gemm_inputs(M, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(M, K, generator=g).to(torch.bfloat16).to(DEVICE)
    B = torch.randn(N, K, generator=g).to(torch.bfloat16).to(DEVICE)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)
    return A, B, C


def gemm_reference(A, B):
    return A.float() @ B.float().t()
