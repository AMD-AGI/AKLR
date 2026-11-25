---
title: "MiniMax-M2 MoE layer in Gluon, fp8 experts with 128 × 128 block scales, model dim 3072, 256 routed experts, top-8 — nine kernels by token count for a rank that holds all 256 experts or 64 of them: the fp32 router on VALU dots, fp32 MFMAs or three bf16 terms; job queues from relaxed atomics or an atomic-free counting sort; both GEMM inputs quantized to fp8 per 1 × 128 in the kernels"
kind: case
ops: [moe, fused_moe, grouped_gemm]
dtypes: [fp8, bf16, fp32]
regimes: ["tokens 1-16", "tokens 32-256", "tokens 1024-16384"]
accuracy: bounded
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over what SGLang runs for the layer (`MiniMaxM2MoE.forward_normal` on aiter), per size, one run of the ruler per kernel", value: "4.115, 3.755, 2.452, 1.841, 1.732 at 1, 2, 4, 8, 16 tokens; 1.158, 1.173, 1.106, 1.133 at 32, 64, 128, 256; 1.257, 1.248, 1.348, 1.372, 1.388 at 1024, 2048, 4096, 8192, 16384", gpu: MI355X, workload: "MM256-FP8: all 256 experts on the rank, intermediate dim 384", date: "2026-11"}
  - {metric: "speedup over what SGLang runs for the layer (`MiniMaxM2MoE.forward_normal` on aiter), per size, one run of the ruler per kernel", value: "3.137, 2.912, 1.672, 1.466, 1.325 at 1, 2, 4, 8, 16 tokens; 1.167, 1.102, 1.146, 1.164 at 32, 64, 128, 256; 1.216, 1.485, 1.681, 1.770, 1.785 at 1024, 2048, 4096, 8192, 16384", gpu: MI355X, workload: "MM64-FP8: 64 of the 256 experts on the rank, from expert offset 192, intermediate dim 1536", date: "2026-11"}
---

# MiniMax-M2 MoE layer in Gluon with fp8 experts (model dim 3072, 256 routed experts)

Nine Gluon kernels for MiniMax-M2's MoE layer on one rank with fp8 experts, a family of the Triton and Gluon kernel
library that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page says which family
to start from; [triton/moe](../triton/moe.md) explains the techniques and names the kernels by the labels used
below: MM256-FP8 on a rank that holds all 256 experts, MM64-FP8 on a rank that holds 64 of them, then the token
counts (`MM256-FP8/m256`, `MM64-FP8/m1–m16`). The fp8 quantization of a 1 × 128 group and the fp32 accuracy of the
routers are explained in [triton/numerics](../triton/numerics.md) §2 and §5. This page has the family's kernels by
size, their code, and what was measured. A kernel here is one host entry with the `@gluon.jit` functions it launches;
a route is one (token, expert) pair, a job a run of up to BM routes of one expert, the rows of one gate-up tile, and
a ticket the position that a relaxed `atomic_add` on an expert's counter returns
([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1, §2).

The layer is MiniMax-M2's MoE layer on one rank: an fp32 router (sigmoid plus correction bias, top-8 of 256,
renormalized), then the experts, with no shared expert. On a rank that holds 64 of the experts, those from
`expert_offset` on, `out` is the rank's share: the routes to the other experts add nothing
([triton/moe](../triton/moe.md) §1). The library's MXFP4 family computes it with MXFP4 experts
([cases/mxfp4_moe_gluon_dim3072](mxfp4_moe_gluon_dim3072.md)); this family:

The same layer with e4m3 experts and fp32 scales per 128 × 128 block, both GEMM inputs quantized per 1 × 128 in the kernels: MM256-FP8 at intermediate dim 384 (5 kernels) and MM64-FP8 at intermediate dim 1536 (4 kernels), 28 sizes. The baseline runs the router as `tgemm.mm` with its first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_128x128` *(field, E1)*.

## Applies when

- The op is the layer above at its shape: model dim 3072, 256 routed experts, top-8, with all 256 experts on the
  rank at intermediate dim 384 or 64 of them at intermediate dim 1536. Its operands are bf16 hidden states
  (tokens × 3072), the fp32 router weight (256 × 3072) and correction bias, the e4m3 gate-up and down weights of the
  experts held, both in aiter's 16 × 16 shuffled layout, with an fp32 scale per 128 × 128 block, and `expert_offset`;
  `out` is bf16.
- The path replaced is SGLang's `MiniMaxM2MoE.forward_normal`. A speedup is against a baseline that reconstructs
  SGLang's path on aiter, the calls above, and does not replay SGLang's process-wide small-batch sort patch;
  all-to-all backends, EPLB and deterministic inference are not covered and stay on SGLang's path
  ([triton/moe](../triton/moe.md) §1).
- Each rank has 14 sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192 and 16384 tokens. A kernel is
  written for a fixed list of them (§ Mechanism); another count has no kernel.
- The family is held to `bounded` accuracy: each output stays within its error bound against the reference, and
  nothing is asked across runs. The bound on `out` is an SQNR floor of 30 dB ([triton/numerics](../triton/numerics.md)
  §1), and all 28 sizes passed their accuracy check *(field, E2)*. Six of the nine kernels hold
  an integer atomic, and every kernel of the library with an atomic is `bounded`
  ([triton/numerics](../triton/numerics.md) §8, [triton/kernels/kernel_library](../triton/kernels/kernel_library.md)
  §2): the atomics place routes, and the values are summed in a fixed order ([triton/moe](../triton/moe.md) §1). The
  library's 31 kernels with an atomic still gave byte-identical outputs at every size in 10 runs
  *(field, E3)*.

## Counter-signals

- **The weakest sizes are at 32–256 tokens**: 1.106–1.173 at MM256-FP8 and 1.102–1.167 at MM64-FP8
  *(field, E5)*. MM256-FP8 at 1 token read 4.115 and, when a later run of the ruler re-measured it,
  3.855, the largest drop among the sizes that run re-measured *(field, E5)*.
- **Router accuracy is shown on samples and selected inputs.** Sampled projections matched an exact sum at all 28
  sizes, but the inline bf16 terms of the routers on bf16 MFMAs were not checked one by one
  *(field, E6)*; complete logits were compared with an fp64 matmul at selected
  branches only, among them MM256-FP8's VALU routers at 1, 2, 4 and 16 tokens, its fp32-MFMA router at 32 and both
  ranks' three-term routers for 1024–16384 tokens *(field, E7)*. Three bf16 terms
  alone do not make a router fp32-accurate for every input, and none of this family's four routers on bf16 MFMAs
  corrects residuals (read from the sources; their rounding is in § Mechanism and in § Code, entry 2): § Mechanism
  names an fp64 correction for the MM64-FP4 kernels only. Nothing guarantees the logits for every input, or the top-8
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §3).
- **Untested**: the job counts of the relaxed-atomic job queues
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §3). Under graph capture the MM64-FP8 kernel
  for 1024–16384 tokens replayed with new inputs accurately and kept the same grids over its 8 eager launches at two
  input sets *(field, E9)*; other routings, worst-case sizing and the device early
  exit are untested ([triton/authoring](../triton/authoring.md) §8).
- **Not compared, or derived and not measured.** No measurement compares the atomic-free counting sort with a ticket
  sort ([triton/moe](../triton/moe.md) §6). Why the MM64-FP8 down runs plain fp8 `mfma` [16, 16, 32] at 64 tokens and
  `mfma_scaled` [16, 16, 128] at 32, and what the switch costs at a fixed token count, was not measured
  ([triton/gluon/mfma](../triton/gluon/mfma.md) §2). The program orders of the MM256-FP8 kernels, the job list walked
  in 8 contiguous stripes at 256 tokens and an M tile's N tiles kept on one XCD from 1024, are derived from the
  formulas, not measured ([triton/moe](../triton/moe.md) §7).
- **`expert_offset`.** The MM256-FP8 kernels ignore it, which is right while the rank holds all 256 experts
  ([triton/moe](../triton/moe.md) §12). The MM64-FP8 kernels take it as a constexpr, one value per process
  ([triton/authoring](../triton/authoring.md) §5); every timed size of theirs ran offset 192, and every accuracy
  input set except one ran 0 ([triton/moe](../triton/moe.md) §1).
- **A 64-expert rank still writes every token's row.** About 10% of tokens have no route to a held expert
  (estimated, uniform routing); their `out` row is zero and must still be written. The MM64-FP8 combine for 32–64
  tokens loops over all 8 slots of every token and adds nothing for a route to an expert the rank does not hold
  ([triton/moe](../triton/moe.md) §12).
- **Weight loads without `.cg` at 4, 8 and 16 tokens (MM64-FP8).** The kernel loads its expert weights there with no
  cache hint, and adding `.cg` was faster in every pair, in time and in speedup: time without / with `.cg` 1.023,
  1.062 and 1.096, one run *(field, E10)* ([triton/gluon/memory](../triton/gluon/memory.md)
  §6).
- **The inline-asm DPP scan showed no gain.** In the MM64-FP8 kernel for 128–256 tokens, replacing the DPP scan of
  the job-table builder by `gl.associative_scan` (112 against 67 instructions, compiled) gave B/A speedups of
  0.994–1.007, with no consistent sign over 3 runs and no paired A/A measurement of the noise
  *(field, E11)* ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §9).

## Mechanism

| Tokens | MM256-FP8 | MM64-FP8 | Launches | Decomposition |
|---|---|---|---|---|
| 1–16 | one kernel (1, 2, 4, 8, 16) | one kernel (1, 2, 4, 8, 16) | MM256-FP8 3; MM64-FP8 4 (1–4 tokens), 6 (8–16) *(field, E12)* | router on fp32 VALU, no MFMA; 180 sampled projections matched an exact sum *(field, E6)*; MM256-FP8 runs the down and the combine in one launch of 8-warp CTAs, one per (token, column tile) *(field, E12)* |
| 32–256 | three kernels (32, 64), (128), (256) | two kernels (32, 64), (128, 256) | 5–6 *(field, E12)* | router on fp32 MFMAs at MM256-FP8 for (32, 64) and (128) and at MM64-FP8 for (32, 64), on bf16 MFMAs at MM256-FP8 for (256) and at MM64-FP8 for (128, 256); 3456 sampled projections matched an exact sum, the inline bf16 terms not checked one by one *(field, E6)*; MM64-FP8 at 256 tokens launches a persistent gate-up on a 48 × 96 grid *(field, E12)* |
| 1024–16384 | one kernel (1024, 2048, 4096, 8192, 16384) | one kernel (the same five) | 6–8 *(field, E12)* | a prepare launch first (MM64-FP8, which prepares its inputs; MM256-FP8 from 8192 tokens, which prepares the router and quantizes) or the router first (MM256-FP8 up to 4096); router; selection and sort (MM256-FP8: launches over chunks of constexpr size; MM64-FP8: an atomic-free SWAR counting sort, a rank-count launch and a prefix scatter); gate-up with fused SwiGLU + fp8 quantization; down; combine |

- **Techniques.** One warp per route under `AMDMFMALayout(..., warps_per_cta=[8, 1, 1])`, the route as the batch
  dimension ([triton/moe](../triton/moe.md) §9); a counting sort with four 8-bit counters per 32-bit word and one
  `gl.associative_scan` ([triton/moe](../triton/moe.md) §6); inline asm in 6 kernels: `v_rcp_f32` in all six, `v_exp_f32` in five,
  `v_readlane_b32` in four, `s_ff1_i32_b64` in three, `v_max_f64` / `v_min_f64` only in MM64-FP8 for 128–256 tokens
  ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §4, §6). The routers on bf16 MFMAs ([cases/mxfp4_moe_gluon_dim3072](mxfp4_moe_gluon_dim3072.md) and here) split the fp32 weight into
  three bf16 terms rounded per kernel: RTZ, RTZ, RTZ with an fp64 correction of flagged residuals in the MM64-FP4 kernels for 2048, 4096,
  6144, 8192 and 16384 tokens; RTZ, RNE, RNE in the MM256-FP8 kernel for 1024–16384; RNE, RNE, RNE without the correction in the MM64-FP8 kernel for 1024–16384 ([triton/numerics](../triton/numerics.md)
  §5). Random and cancellation inputs leave no residual; in two inputs built to have one (MM64-FP4 for 6144 and for 8192 tokens), 16 flags and
  256 residual entries were set, and the correction moved those logits from 0 to 2^-23, the fp64 value *(field, E7)*.
  Three bf16 terms alone do not make a router fp32-accurate for every input.
- **Resources** *(field, E13)*: 188 compiled kernels; `num_warps` 1 (73), 2 (13), 4 (96), 8 (6); `waves_per_eu` 1 in 4; VGPR
  14–327, AGPRs in 2 (35, 71); LDS 0–131.9 KiB; an 8-SGPR spill in one route-packing kernel; scaled 16 × 16 × 128 MFMA in 47,
  bf16 16 × 16 × 32 in 13, scaled 32 × 32 × 64 in 10, fp8 16 × 16 × 32 in 9, fp32 16 × 16 × 4 in 7.
  The compiled kernels counted here are compiled-cache records, one per compilation of a `@gluon.jit` function, so
  the same binary can be counted more than once: they are not the nine kernels of the table. VGPR is a compiled
  kernel's `.vgpr_count`, architectural VGPRs plus AGPRs
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1).

## Code

### 1. The rank of each route among the routes of its expert, four 8-bit counters per word (MM64-FP8, 1024–16384 tokens)

The MM64-FP8 kernel for 1024–16384 tokens sorts its routes by expert with no atomic, which fixes the row order
([triton/moe](../triton/moe.md) §6). This is its rank-count launch, the SWAR rank of every route among the
same-expert routes of a 512-route block; the prefix-scatter launch that follows adds to it the expert's counts in
the earlier blocks and the padded counts of the earlier experts, and stores the route at that row. Four experts
share a 32-bit word of 8-bit counters, at bit `shift` each, so one `gl.associative_scan` along the block, less the
route's own increment, counts the earlier routes of four experts at once; `Ids` holds each route's expert id, `R` is
the route count, `OFFSET` the rank's `expert_offset`, `E` the experts it holds and `group` the words this program
counts. `route` is the block's 512 route indices, `layout` the layout of the 4 words × 512 routes tile and `_add`
the combine function of the scan, an addition. The 8-bit counters hold because a block's 512 routes are
512 // TOPK = 64 tokens, each choosing an expert once: the kernel asserts
`gl.static_assert(512 % TOPK == 0 and 512 // TOPK < 256)` (TOPK 3 fails it).

```python
ids = gl.load(Ids + route, route < R, -1) - OFFSET
matches = (route[None, :] < R) & (ids[None, :] >= 0) & (ids[None, :] < E)
matches &= group[:, None] == ids[None, :] // 4
shift = (ids & 3) * 8
increment = gl.full((512,), 1, gl.uint32, gl.SliceLayout(0, layout)) << shift
packed = gl.where(matches, increment[None, :], 0)
prefix = gl.associative_scan(packed, 1, _add) - packed
rank = prefix >> shift[None, :] & 255
```

Not this: a block that can hold more than 255 routes of one expert. The count carries into the next expert's
counter, and that expert's ranks come out too high: in a test kernel holding a copy of this rank count, a block of
300 routes of one expert followed by 212 of the next gave every route of the next expert a wrong rank, the first
four 1, 2, 3, 4 where 0, 1, 2, 3 are right *(field, E18)*.

### 2. The router's fp32 weight as three bf16 terms, one bf16 MFMA accumulator each (MM256-FP8, 256 tokens)

The K loop of the MM256-FP8 router for 256 tokens: the hidden states are bf16, so only the fp32 weight tile `b` is
split, into three bf16 terms that feed three bf16 MFMAs, one accumulator each. The two truncations (`rtz`) take 16
significant bits and leave at most 8, so the three terms hold the fp32 weight exactly unless a lower term falls
below bf16's normal range, and a bf16 × bf16 product in fp32's normal range is exact in fp32
*(field, E3)*: rounding happens only in the accumulation. The accumulators are summed after the
loop as `high + (middle + low)`. `a` and `b` are the tiles in the MFMA operand layouts and `CHUNK` the K steps of the
program's K split; the split is redone on every call, since the weight is an input of the call. RTZ to bf16 is a bit
truncation (S1), which turns a NaN whose payload sits in the low 16 bits into Inf: route
non-finite weights around it. This router rounds its three terms RTZ, RTZ, RNE (a `.to(gl.bfloat16)` with no
rounding argument rounds to nearest even); the MM64-FP8 router for 128–256 tokens (entry 6) rounds all three RNE and
accumulates the two lower terms together. Neither corrects residuals (read from the sources).

```python
# MM256-FP8/m256, the router's K loop: the fp32 weight tile b as three bf16 terms, one bf16 MFMA accumulator each
    for step in gl.static_range(CHUNK):
        # ... (a and b loaded and converted to the MFMA operand layouts)
        b0 = b.to(gl.bfloat16, fp_downcast_rounding="rtz")
        residual = b - b0.to(gl.float32)
        b1 = residual.to(gl.bfloat16, fp_downcast_rounding="rtz")
        b2 = (residual - b1.to(gl.float32)).to(gl.bfloat16)
        high = gl.amd.cdna4.mfma(a, b0, high)
        middle = gl.amd.cdna4.mfma(a, b1, middle)
        low = gl.amd.cdna4.mfma(a, b2, low)
    acc = high + (middle + low)
```

Not this: `bf16x3` in place of the hand-written split. Plain Triton's `input_precision="bf16x3"` splits both
operands, and under Gluon it changes nothing: `mfma` compiled to the same ISA with `TRITON_F32_DEFAULT=bf16x3` as
without it *(field, E14)*.

Explained in [triton/moe](../triton/moe.md) §5.1, [triton/numerics](../triton/numerics.md) §5 and [triton/gluon/mfma](../triton/gluon/mfma.md) §2.5.

### 3. fp8 quantization per 1 × 128 group, the amax taken on bf16 bit patterns (MM64-FP8, 64 tokens)

The quantizer of the MM64-FP8 kernel for 32–64 tokens, whose down launch quantizes each 1 × 128 group of its bf16
input as it loads it. Per group: the amax floored at 1e-10, the scale amax / 448 as a rounded product corrected with
its residual, `v_rcp_f32` refined by one Newton step, then `x * inverse` cast to e4m3 without a clamp. The values
were already rounded to bf16, so the amax can be an integer max of their bit patterns with the sign cleared,
non-negative bf16 patterns ordering like their values; the down launch takes that path (`NARROW_PEAK`) at 64 tokens.
`x` holds the groups in fp32, 128 values along axis 1, and `DTYPE` is the e4m3 type.

```python
@gluon.jit
def _max_bf16_magnitude_bits(a, b):
    return gl.maximum(a, b)

@gluon.jit
def _quantize_fp8_groups(x, DTYPE: gl.constexpr, NARROW_PEAK: gl.constexpr = False):
    if NARROW_PEAK:  # x holds bf16-rounded values: amax as an integer max of their bit patterns, sign cleared
        bits = (x.to(gl.bfloat16).to(gl.uint16, bitcast=True) & 32767).to(gl.uint16)
        peak_bits = gl.reduce(bits, 1, _max_bf16_magnitude_bits)
        peak = peak_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
    else:
        peak = gl.max(gl.abs(x), 1)
    peak = gl.maximum(peak, 1e-10)
    first = peak * (1.0 / 448.0)  # the scale as a rounded product,
    remainder = gl.fma(-first, 448.0, peak)  # corrected with its residual
    scale = gl.fma(remainder, 1.0 / 448.0, first)
    inverse = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", (scale,), dtype=gl.float32, is_pure=True, pack=1
    )
    error = gl.fma(-scale, inverse, 1.0)  # one Newton step on the reciprocal
    inverse = gl.fma(inverse, error, inverse)
    q = (x * inverse[:, None]).to(DTYPE)
    return (q, scale)
```

Not this: the bit-pattern max where a NaN must not win. A NaN pattern orders above Inf; map the patterns above 0x7f80
to 0 there, as the Qwen3-Next GDN decode kernels do.

Explained in [triton/numerics](../triton/numerics.md) §2.

### 4. The weight load's cache hint chosen per job (MM256-FP8, 256 tokens)

The gate-up worker of the MM256-FP8 kernel for 256 tokens picks the cache hint of its weight loads per job, a job
being a run of up to `BM` routes of one expert, the rows of one gate-up tile. An expert with more than `BM` routes
has more than one job, and its jobs read the same weight tiles; the planner launch sets bit 24 of such a job's
descriptor (`packed |= gl.where(count[:, None] > BM, 1 << 24, 0)`) and the worker branches on it, dropping `.cg` for
those jobs. The hint is a constexpr string, so each branch makes its own call of `_run_w13_job`, the gate-up of one
job; `Jobs` is the job list the planner wrote. Inside `_run_w13_job` the gate and up weight loads of the first
128-K group are `.ca` in both branches; the hint applies from the second group on.

```python
packed = gl.load(Jobs + job).to(gl.uint32)
if packed & 1 << 24 != 0:
    _run_w13_job(
        # ... (19 arguments, the same in both calls)
        ".ca",
    )
else:
    _run_w13_job(
        # ... (the same 19 arguments)
        ".cg",
    )
```

Not this: `.cg` on weights that a program on another XCD re-reads in the same pass; other re-reads are settled per
size ([triton/moe](../triton/moe.md) §7). At the two re-read sizes measured, both on other kernels, the hint lost:
DS512 at 256 tokens by 0.2–0.7% (4 runs) and MM256-FP4 at 16384 tokens by 17% (one run)
*(field, E10)*.

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §6.

### 5. Three K steps' operands in registers, rotated in a rolled loop (MM64-FP8, 8 tokens)

The fp8 projection of the MM64-FP8 kernel for 1–16 tokens loads its operands straight into the MFMA operand layouts,
so its look-ahead needs no LDS: it keeps up to three K steps' operand tuples (A, B and both scales) in flight in
registers and rotates them. This is its three-tuple branch, which the up projection takes at 8 tokens: a trip loads
the tuple of the K group three ahead (`na`, `nb`, `nsa`, `nsb`), accumulates the oldest (`a`, `b`, `sa`, `sb`) and
moves the two waiting tuples up; `groups` is the K groups of the program's K split. The rotation is loop-carried, so
the loop stays a rolled `range`. The depth is set per token count (up projection: 2 tuples at 1, 4 and 16 tokens, 3
at 8); compiled, the second tuple cost 16–24 VGPRs and the third none *(field, E15)*.

```python
for step in range(groups - 3):
    na, nb, nsa, nsb = _load_fp8_tile(
        # ... (X, XS, W, WS and the tile's row, expert and column indices)
        split * groups + step + 3,
        # ... (K, WN, WK, BUFFER_LOAD, WEIGHT_CACHE)
    )
    acc = _accumulate_fp8(a, b, sa, sb, acc)
    a, b, sa, sb = (a1, b1, sa1, sb1)
    a1, b1, sa1, sb1 = (a2, b2, sa2, sb2)
    a2, b2, sa2, sb2 = (na, nb, nsa, nsb)
```

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §12.

### 6. A slot's relaxed reads retired before its refill, in a one-wave ring (MM64-FP8 router, 128–256 tokens)

One step of the two-slot LDS ring of the MM64-FP8 router for 128–256 tokens, a one-warp launch (slot 0 shown;
`_copy_router_panel` issues the two async copies of a slot, the hidden tile into `ax0` and the weight tile into
`bw0`, and commits them as one group). The reads are relaxed (`load_shared_relaxed`), so the order of a slot's
refill after its reads is the kernel's to keep, and with one wave `gl.barrier()` emits neither a wait nor an
`s_barrier`. The router therefore writes an inline `s_waitcnt lgkmcnt(0)` with a `~{memory}` clobber between the
relaxed reads and the async copy that refills the same slot with the panel two steps ahead; the asm is impure and
its `=v` output is a dummy.

```python
a = gl.amd.cdna4.async_copy.load_shared_relaxed(ax0, ad)  # relaxed reads of the slot
b = gl.amd.cdna4.async_copy.load_shared_relaxed(bw0, bd)
gl.inline_asm_elementwise(  # retire them; the `=v` output is a dummy
    "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
    "=v,~{memory}",
    (),
    dtype=gl.int32,
    is_pure=False,
    pack=1,
)
if step + 2 < steps:  # then the refill of the same slot
    _copy_router_panel(
        ax0, bw0, X, W, xo + (step + 2) * BK, wo + (step + 2) * BK
    )
```

Not this: `gl.barrier()` alone between the relaxed reads and the refill. In synthetic one-wave 2-slot rings of
bank-conflicted 16 KiB tiles, 10–16% of programs then read refilled data, and none did with `s_waitcnt lgkmcnt(0)`
before the refill *(field, E16)* ([triton/gluon/memory](../triton/gluon/memory.md) §11,
[triton/gemm](../triton/gemm.md) §5).

Explained in [triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §8.

## Result

- **Speedups** over the baseline (§ Applies when) *(field, E17)*: 1.102 (MM64-FP8,
  64 tokens) to 4.115 (MM256-FP8, 1 token) over the 28 sizes. Across the library's DeepSeek-V2 and MiniMax-M2 MoE
  families (this one, [cases/mxfp4_moe_gluon_dim6144](mxfp4_moe_gluon_dim6144.md) and
  [cases/mxfp4_moe_gluon_dim3072](mxfp4_moe_gluon_dim3072.md)) the 32–256-token sizes are the weakest band whatever
  the design, and inside one 1–16-token kernel the value falls as tokens grow: by band in
  [triton/moe](../triton/moe.md) §2 and [triton/tuning](../triton/tuning.md) §14.

Per size, each kernel's values from one run of the ruler *(field, E5)*:

| Tokens | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MM256-FP8 | 4.115 | 3.755 | 2.452 | 1.841 | 1.732 | 1.158 | 1.173 | 1.106 | 1.133 | 1.257 | 1.248 | 1.348 | 1.372 | 1.388 |
| MM64-FP8 | 3.137 | 2.912 | 1.672 | 1.466 | 1.325 | 1.167 | 1.102 | 1.146 | 1.164 | 1.216 | 1.485 | 1.681 | 1.770 | 1.785 |

Each value belongs to the whole kernel at that size. Repeat runs of unchanged sources moved a ratio by 2% at the
median and 6% at p90 *(field, E5)*: gaps of that order between sizes or kernels say nothing about
designs ([triton/moe](../triton/moe.md) §11).

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, each kernel's `run` replayed on the host with its launches recorded, no GPU] the 24 fp8 GEMM sizes one launch each, the N 3072 kernel for M = 6144 rejecting independently strided A and B; at the 89 MoE sizes of 58 kernels the gate-up launch, its grid and the source of its W1; the baselines' calls: MiniMax-M2 routers `tgemm.mm` with the first operand cast to fp32, then `moe_fused_gate` and `fused_moe` at `per_1x32` (MXFP4) or `per_128x128` (FP8); GDN decode a conv update, the recurrence, the norm and the quantization; GDN prefill `causal_conv1d_fn`, `fused_gdn_gating`, `chunk_gated_delta_rule`, the norm and the quantization (`per_token_group_quant_fp8` in Qwen3.5)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto); accuracy against each output's bound, speedup against what sglang runs for the op] every size with a recorded accuracy result passed: 198 of the 242 sizes of the library's 154 kernels, among them gated attention 9, Qwen3.5 GDN decode fp8 8 and prefill 21, MiniMax-M2 fp8 28, Qwen2-MoE fp8 7; the 44 sizes of the 20 Qwen3-Next kernels, which had no recorded result then, all passed in a later run of the ruler, the run their stored speedups are from; speedups: the MM64-FP4 kernel with the software e2m1 encoder 1.19 at 128 tokens and 1.194 at 256; the Qwen2-MoE fp8 router + experts kernels with a float64 router dot 1.172 at 1 token (on the VALU) and 1.063 at 32 (on the fp64 MFMA), 1.152 and 1.063 in a second run of the ruler with unchanged sources (image lmsysorg/sglang:v0.5.22-rocm10-mi35x)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, runs on one GPU] the accuracy check on a `bitwise` and a `bounded` copy of one kernel ran the first input set 3 times (a first capture, the accuracy run, the repeat run) and 2 times, and bit flips planted in those runs produced no finding; bf16 × bf16 products in fp32's normal range exact for 748591 VALU and 16384 MFMA products; 512 crafted values near bf16 midpoints, alone and with 512 random values (1024): plain fp64 → fp32 → bf16 wrong on 260 either way, the nudge of the 1-token Qwen2-MoE router + experts kernel on 0; repeated calls on the same inputs: the 24 a8w8 blockwise kernels 0 non-identical outputs in 20 runs per size, the bf16 GEMM N 2624 × K 6144 1 bit pattern in 20 runs at m = 2, 4, 8, 64, the 31 bounded kernels with atomics byte-identical at every size in 10 runs (the DS512 kernel for 4096 tokens of E4 in 50); a 4096-program ticket kernel in a graph: second replay offset by 4096 without a zeroing node, identical with one (3 runs)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the kernel called 100 times with synchronization, its runtime records saved and their bytes hashed on CPU] the target block on the 128-row body (32 × 32 × 64 MFMAs) in all 100 calls, one hash of its route partials; at 15, 16, 17, 31, 32 and 33 live rows, each with the body that count selects and with the next larger one forced, the target's up codes, scales and down partials and token 0's 6144 outputs identical within each live count (1,040 hashes) (the DS512 kernel for 4096 tokens; the same check, in a run of its own, on the DS512 kernel for 256 tokens: the target on the 16-row body in all 100 calls, one hash of its route partials, no bit of its up codes, scales or down partials changed at those live rows)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush, Kineto), speedup over the baseline of SGLang's path on aiter] speedup per size, the implementation in parentheses: DeepSeek-V2/GLM DS256: 1 token 2.139 (DS256/m1), 2 2.186 (DS256/m2), 4 1.867 (DS256/m4), 8 1.521 and 16 1.245 (DS256/m8–m16), 32 1.086 (DS256/m32), 128 1.030 (DS256/m128), 256 1.160 (DS256/m256), 1024 1.402, 2048 1.329 and 3072 1.407 (DS256/m1024–m3072), 4096 1.420 and 4192 1.378 (DS256/m4096–m4192); DS512: 1 token 2.093 and 2 1.829 (DS512/m1–m2), 4 1.324 (DS512/m4), 8 1.184 (DS512/m8), 16 1.097 (DS512/m16), 256 1.043 (DS512/m256), 1024 1.152 (DS512/m1024), 2048 1.391 (DS512/m2048), 3072 1.235 (DS512/m3072), 4096 1.161 (DS512/m4096), 4192 1.166 (DS512/m4192); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (MM256-FP8/m1–m16); 32 1.158 and 64 1.173 (MM256-FP8/m32–m64); 128 1.106 (MM256-FP8/m128); 256 1.133 (MM256-FP8/m256); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (MM256-FP8/m1024–m16384); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (MM64-FP8/m1–m16); 32 1.167 and 64 1.102 (MM64-FP8/m32–m64); 128 1.146 and 256 1.164 (MM64-FP8/m128–m256); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (MM64-FP8/m1024–m16384); MXFP4, 256 experts: 1 token 3.444 (MM256-FP4/m1), 2 3.309 (MM256-FP4/m2), 4, 8, 16 2.549, 1.977, 1.595 (MM256-FP4/m4–m16), 32 1.179 (MM256-FP4/m32), 64 1.230 (MM256-FP4/m64), 128 1.210 (MM256-FP4/m128), 256 1.178 (MM256-FP4/m256), 1024 1.352 (MM256-FP4/m1024), 2048 1.272 and 4096 1.261 (MM256-FP4/m2048–m4096), 6144 1.415 (MM256-FP4/m6144), 8192 1.423 (MM256-FP4/m8192), 16384 1.389 (MM256-FP4/m16384); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (MM64-FP4/m1–m2), 4 2.554 (MM64-FP4/m4), 8 2.113 and 16 1.555 (MM64-FP4/m8–m16), 32 1.231 and 64 1.275 (MM64-FP4/m32–m64), 128 1.190 and 256 1.194 (MM64-FP4/m128–m256), 1024 1.252 (MM64-FP4/m1024), 2048 1.318 (MM64-FP4/m2048), 4096 1.590 (MM64-FP4/m4096), 6144 1.689 (MM64-FP4/m6144), 8192 1.735 (MM64-FP4/m8192), 16384 1.619 (MM64-FP4/m16384); Qwen2-MoE router+experts: 1 token 1.172 (Q2-RE/m1), 2 1.145 (Q2-RE/m2), 4 1.119 (Q2-RE/m4), 32 1.063 (Q2-RE/m32); router: 8 tokens 1.074 (Q2-R/m8), 16 1.219 (Q2-R/m16), 64 1.295 (Q2-R/m64), 128 1.417 (Q2-R/m128), 256 1.967 (Q2-R/m256), 2232 1.416 (Q2-R/m2232), 4096 1.226 (Q2-R/m4096); fp8 experts: 8 tokens 1.208 (Q2-E/m8), 64 1.171 (Q2-E/m64), 128 1.076 (Q2-E/m128); the values come from eight runs of the ruler, each implementation's from one run; a later run re-measured 57 of the 96 sizes, by −6.3% (MM256-FP8 at 1 token, 4.115 → 3.855) to +7.0% (Q2-R at 16 tokens, 1.219 → 1.304), median change 0.9%; over repeat runs of the ruler on unchanged sources, the ratios at the same size differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 sizes, at most 54.8 % (Q2-RE at 1 token)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one input set per size, the router launch's projection sampled against `math.fsum` with a tolerance per sample, its ISA read; not timed] 58 MiniMax-M2 router projections (hidden bf16 [M, 3072], weight fp32 [256, 3072], M from 1 to 16384): all 8520 sampled comparisons within tolerance, none non-finite; by subset (overlapping): MXFP4 1–16 tokens 10 sizes / 492 comparisons, MXFP4 1024–16384 12 / 912, FP8 1–16 10 / 180, FP8 32–256 8 / 3456, bf16-MFMA routers 18 / 1896; no MFMA in the routers at 1–16 tokens; bf16 MFMA in the MM64-FP4 kernels for 2048, 4096, 6144, 8192 and 16384 tokens, the MM256-FP8 kernels for 256 and for 1024–16384, and the MM64-FP8 kernels for 128–256 and for 1024–16384; fp32 MFMA in every other router above 16 tokens; the three bf16 terms of the MM64-FP4 routers checked on samples, the FP8 routers' inline terms not (MiniMax-M2 MoE kernels)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, as in E8] 13 router branches at selected sizes (among them the VALU routers of MM256-FP8 at 1, 2, 4 and 16 tokens, the fp32 MFMA of MM256-FP8 at 32 and of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens, the bf16 three-term MFMA of MM64-FP4 for 6144 and for 8192 and of MM256-FP8 and MM64-FP8 for 1024–16384), a seed-0 random and a cancellation input each (26 inputs), plus two inputs built with nonzero weight residuals; on the 26 the kernel's error against fp64 no larger than the fp32 matmul's in L2 on 25 (the cancellation input of MM256-FP4 for 6144 tokens 1.0000934×), in max-abs on 23 (the cancellation inputs of MM256-FP4 for 2048–4096, for 6144 and for 8192 tokens 1.0319×, 1.0018×, 1.0451×), element by element on 3; in the two residual inputs (MM64-FP4 at 6144 and at 8192 tokens) 16 flags and 256 residual entries set, the fp64 correction ran, logits 0 without it and 2^-23 with it, equal to fp64 and to the fp32 matmul: error 0 on both (MiniMax-M2 MoE kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, complete router logits of the library kernel, of a GPU fp32 matmul (IEEE fp32 MFMA, no TF32) and of an fp64 matmul on the same inputs] MM64-FP4 at 2048 and at 4096 tokens and MM256-FP4 at 1024, 4 input distributions × 8 seeds each: the kernel's L2 error against fp64 no larger than the fp32 matmul's in 96 of 96 inputs, its max-abs error in 93 (MiniMax-M2 MXFP4 MoE kernels)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, each kernel captured into a CUDA graph, replayed and checked for accuracy; eager launches recorded by the profiler] MM256-FP4 for 16384, for 8192 and for 2048–4096 tokens and MM64-FP8 for 1024–16384: captured graphs replayed with new inputs, accurate; eager grids identical at two input sets, 6, 6, 6 and 8 launches; other routings, worst-case sizing and the device early exit not tested
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (MM64-FP4 and DS512 at 256 tokens N=4; DS256 at 2 tokens and the store A/Bs N=3), the ruler (Kineto), 4 A/B rounds per size, in time and in speedup over the baseline] weight-load `.cg` on and off at 12 sizes of 7 MoE kernels, 10 whose routing reads each weight tile about once (mean readers < 2) and 2 re-read; the ISA changed only in cache bits; accuracy passed; unchanged-binary controls within 0.37%; time without/with `.cg`: three 1-token sizes 0.995–1.004 with pairs both ways, MM256-FP8 at 2 tokens 1.037 (speedup 1.020, 3 of 4 speedup pairs agreeing), MM64-FP8 at 4/8/16 tokens 1.023/1.062/1.096, MM64-FP4 at 128 tokens 1.018 (speedup 1.016), at 256 tokens 1.028, DS256 at 2048 tokens 1.016, DS512 at 256 tokens 0.998, MM256-FP4 at 16384 tokens 0.854; `.cg` faster in every pair, in time and speedup, at six of the read-once sizes, slower in every pair at none; flipping the kernel's own hint won in every pair at MM64-FP8 at 4/8/16 tokens (time 0.909–0.982) and MM64-FP4 at 128 tokens (0.978–0.988); repeats: MM64-FP4 at 256 tokens 2.6–3.5% slower without `.cg` and DS512 at 256 tokens 0.2–0.7% slower with it (4 runs each), DS256 at 2 tokens 0.0–1.2% (3 runs, every pair the same way in two); removing `.cs` from the n = 3072 fp8 GEMM's output stores cost 1.7–1.9% at m = 6144 and gained 0.1–0.7% at m = 8193, every pair (3 runs) (MM64-FP8, MM256-FP8, MM64-FP4, MM256-FP4, DS256, DS512 and the n = 3072 fp8 GEMM)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler (Kineto), A/B, 4 rounds per arm per run] DPP asm scan (A) against `gl.associative_scan` (B, E forced to 63), both accurate in every round; job-table builder compiled: A 6 DPP adds, 0 `ds_bpermute`, 67 instructions, B 6 `ds_bpermute`, 112 instructions; A / B µs per run at 128 tokens 153.25 / 153.44, 153.88 / 153.52, 153.06 / 153.08, at 256 tokens 158.25 / 159.74, 159.27 / 157.51, 156.29 / 156.48; B/A speedup 0.9939, 0.9987, 0.9951 at 128 tokens and 0.9942, 1.0066, 0.9996 at 256; the per-round speedups of the two arms overlap in every run (MM64-FP8, 128 and 256 tokens)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one call per size with its device launches recorded by the profiler] MiniMax-M2 MXFP4: 3 launches at the 10 sizes of 1–16 tokens, the last a down with the weighted combine, the router over K = 3072 in every kernel but MM256-FP4 for 4–16 tokens; 5 at 32–256 tokens; 6–8 at 1024–16384, the down one design in every MM256-FP4 kernel but the one for 2048–4096 tokens, which has its own; MiniMax-M2 FP8: MM256-FP8 3 at 1–16 tokens, its down-and-combine launch 8 warps on a (token, column tile) grid, MM64-FP8 4 at 1–4 and 6 at 8–16, 5–6 at 32–256 with MM64-FP8 at 256 launching a persistent gate-up on 48 × 96 programs, MM256-FP8 6 at 1024–4096 and 7 at 8192–16384, MM64-FP8 8; GDN prefill at all 34 sizes 5–8 launches in the order prepare, chunk factors, summaries for long sequences, one sequential launch, epilogue; at the 21 Qwen3.5 sizes the final conv history the last three entries of the old history (zeros when fresh) followed by the inputs (MoE and GDN prefill kernels)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E14 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, compiled, no GPU launch] default `gl.clamp`: one `v_med3_f32`; with `PropagateNan.ALL` one `v_cmp_u_f32` and one `v_cndmask_b32` more; `gl.maximum` with `ALL`: one `v_maximum3_f32`; a 16 × 16 × 64 fp32 `tl.dot`, both operands split by `bf16x3`: 6 bf16 MFMAs against 16 fp32 with `ieee`; Gluon `mfma` under `TRITON_F32_DEFAULT=bf16x3`: the same ISA as without it (test kernels, not library kernels)
- E15 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] `wait_group(n)` with `n` a kernel argument or a `range` variable: `CompilationError` at trace, "create_async_wait_group(): incompatible function arguments", a constexpr compiles; a 1-tuple grown in a `range` loop: "'scf.for' op different number of region iter_args and yielded values: 1 != 2", a fixed-length queue compiles; [triton/gluon/memory](../triton/gluon/memory.md) §12's rolled-ring structure for an fp8 block-scaled GEMM, M = 32, N = 4096, K = 7168, 16 columns, 2 waves, 6 slots, one step per trip: 325 instructions (1417 unrolled), 56 VGPRs, 45056 B of LDS, loop `vmcnt(12)`, drain `vmcnt(9)`, `(6)`, `(3)`, `(0)`; a weight scale `gl.load`ed at its group: `vmcnt(0)` and `vmcnt(12)` in the loop; a tuple of scales loaded with their groups' copies: `vmcnt(3)`, `(7)`, `(16)` at one step per trip, `(14)`–`(19)` at 5; a weight scale from a 1-D LDS table: 60 VGPRs, 45312 B, `vmcnt(12)`; 8 slots, 7 steps per trip: 57344 B, 88 VGPRs, `vmcnt(18)`; unrolled 96 VGPRs and 64 SGPRs, rolled 31 SGPRs, no spill; a `[64, 32]` fp32 table with a rank-2 layout: `.index(g)` fails, "rank must be equal to or one less than the shape size. Got 2 and 1", with a rank-1 layout the allocation or `.store` fails ("source has rank 2 but memdesc encoding has rank 1"); per group in a `range` loop, `.gather` (a 2-D table with `gl.full((32, 1), g)`, or a flattened table) or a rank-1 `_reinterpret` view's `.index(g)`: one 32-bit word per trip, 1 `ds_read_b32` in the one-trip loop, wider reads in the 8× unrolled one; 1 `s_barrier` per trip without a `gl.barrier()` after the staging, 0 with it; a uniform `gl.load` after an async copy or a `gl.store` → `global_load_dword`, also in a 4-trip `range` loop that LLVM unrolled; `WS.to(gl.uint64).to(gl.pointer_type(gl.float32, 4))` alone fails to compile, through [triton/gluon/memory](../triton/gluon/memory.md) §13's inline asm 4 `s_load_dword` after a copy or a store; MM64-FP8's fp8 projection kernel, M = 8, BN = 32, 2 warps: 68, 84, 84 VGPRs at look-ahead depth 1, 2, 3 (66, 90, 88 at one split), no spill (test kernels and MM64-FP8 at 1–16 tokens)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one process, 400 rounds with the variants in random order, outputs compared with their source tiles on the GPU; ISA of the launched binaries] 11 one-wave Gluon rings, each step `wait_group`, `load_shared_relaxed` of a slot, then its refill: with `gl.barrier()` every binary read before its DMA with no `lgkmcnt` wait between, and the static and rolled 2-slot 16 KiB bank-conflicted rings read refilled data in 400 of 400 launches (9.8–16.2 % of programs), the other nine in none; with an inline `s_waitcnt lgkmcnt(0)` added, zero wrong in all 11, 0.988–1.013× the time with `gl.barrier()` alone (synthetic rings, 4096 programs × 16 steps)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] MiniMax-M2 FP8 MoE: lowest 1.102 at MM64-FP8 (intermediate dim 1536), 64 tokens, highest 4.115 at MM256-FP8 (intermediate dim 384), 1 token, in two runs of the ruler (model dim 3072, 256 routed experts)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, a Gluon test kernel holding a copy of the rank count of §1, run on 512-route blocks (64 experts, expert offset 192, TOPK 8), its ranks compared with a stable ranking computed on the host] a block of 300 routes of one expert followed by 212 of the next: every rank of the next expert wrong, the first four 1, 2, 3, 4 against 0, 1, 2, 3, and 44 of the first expert's 300 ranks wrong; random routings at 64, 4000 and 4096 tokens: every rank equal to the stable ranking; a build with TOPK 3 fails at the static assert (test kernel)
- S1 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2127 fp32 → E4M3FN on CDNA4 via `scalePk4DowncastToFp8<CvtScaleF32PkFp8F32Op>`, scale 1.0 (:561), RTNE only (:2111); an fp8 cast marks its function with `SetFP8ClampingAttr` (:2473-2481); fp32 → bf16 RTNE in packed pairs on CDNA4 (:106), RTZ a 16-bit shift (:73-77)
