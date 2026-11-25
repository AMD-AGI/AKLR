---
title: "DeepSeek-V2 / GLM MXFP4 MoE layer in Gluon, model dim 6144, 256 routed experts, top-8 and a shared expert, expert intermediate dim 256 or 512 — no sort and the top-k recomputed in the gate-up programs at 1–16 tokens, but for one kernel; ticket-placed expert jobs and a fixed-order reduce at 32–256; sorted expert tiles of run-time height from 1024"
kind: case
ops: [moe, fused_moe, grouped_gemm, gemm, topk]
dtypes: [mxfp4, bf16, fp32]
regimes: ["tokens 1-16", "tokens 32-256", "tokens 1024-4192"]
accuracy: bounded
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over SGLang's path on aiter (router GEMM, biased_grouped_topk, fused_moe), per size, one reference run of the ruler per kernel", value: "1.030 (DS256, 128 tokens) to 2.186 (DS256, 2 tokens) over the 24 sizes", gpu: MI355X, workload: "expert intermediate dim 256 (DS256) at 1, 2, 4, 8, 16, 32, 128, 256, 1024, 2048, 3072, 4096 and 4192 tokens; 512 (DS512) at 1, 2, 4, 8, 16, 256, 1024, 2048, 3072, 4096 and 4192", date: "2026-11"}
---

# DeepSeek-V2 / GLM MXFP4 MoE layer in Gluon (model dim 6144, 256 routed experts and a shared expert)

Nineteen Gluon kernels for one op, a whole MoE layer on one rank, a family of the Triton and Gluon kernel library
that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page says which family to start
from; [triton/moe](../triton/moe.md) explains the techniques, and names kernels by the labels used below: DS256 and
DS512 for the rank's expert intermediate dimension, then the token count (`DS256/m2`; a range, as in
`DS256/m8–m16`, is one kernel written for several counts). This page has the family's kernels by size, their code,
and what was measured.

The layer and its baseline:

SGLang's `DeepseekV2MoE` on one rank: bf16 router logits, the sigmoid top-8 of 256 with an fp32 correction bias, weights
renormalized × 2.5, the shared expert as slot 256 at weight 1, MXFP4 experts (e2m1 with e8m0 per 32, aiter's 16 × 16
shuffled weights) and their weighted sum. This rank's expert intermediate dimension is 256 or 512 (DS256 and DS512, the labels of [triton/moe](../triton/moe.md) §1); 19 kernels, 24 sizes. The baseline
replays aiter's router GEMM, `biased_grouped_topk` and `fused_moe`. Triton side: [triton/moe](../triton/moe.md); op card:
[../moe/INDEX.md](../moe/INDEX.md).

## Applies when

- The op is the whole layer call on one rank at this shape: `hidden_states` (tokens × 6144) and the router weight
  (256 × 6144) in bf16, an fp32 correction bias per routed expert, and the gate-up and down weights of 257 experts,
  the shared one last, in e2m1 with an e8m0 scale per 32 values, the weights in the 16 × 16 preshuffle that the
  paragraph above calls aiter's (its byte order: [triton/gluon/mfma](../triton/gluon/mfma.md) §7, which did not
  compare it with aiter's `shuffle_weight`) and the scales in aiter's `shuffle_scale` order (the same section). The
  gate-up weight holds an expert's gate rows, then its up rows. `out` (tokens × 6144) is bf16, this rank's share of
  the layer's output.
- The path replaced is SGLang's `DeepseekV2MoE.forward_normal`. A speedup is against a baseline that reconstructs
  SGLang's path on aiter: the router GEMM, `biased_grouped_topk`, the shared-expert append and `fused_moe`.
  All-to-all backends, EPLB and deterministic inference are not covered and stay on SGLang's path. The `fused_moe`
  step alone has FlyDSL kernels on these 16 × 16 shuffled weights
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §3; a case:
  [fused_moe_shuffled_16x16](fused_moe_shuffled_16x16.md)).
- A kernel is written for one token count or a few (§ Mechanism): DS256 has 9 kernels for 1, 2, 4, 8, 16, 32, 128,
  256, 1024, 2048, 3072, 4096 and 4192 tokens, DS512 10 kernels for 1, 2, 4, 8, 16, 256, 1024, 2048, 3072, 4096 and
  4192.
- The family is held to `bounded` accuracy: `out` stays within its error bound against the reference, an SQNR floor
  of 10 dB, and nothing is asked across runs. The sources of 12 of the 19 kernels hold relaxed integer atomics
  (tickets, job counters, bitmasks), and the library holds every kernel with an atomic to `bounded`: ticket order
  is not deterministic, so the kernels let tickets place rows and keep every float sum in a fixed order, and none
  has a float atomic ([triton/moe](../triton/moe.md) §6, §10, [triton/numerics](../triton/numerics.md) §8).
- The reference rounds the router logits to bf16 before the sigmoid, adds the correction bias in fp32 and keeps the
  activations in fp32. Its `torch.topk` fixes no tie order; the kernels take the lower id
  ([triton/moe](../triton/moe.md) §5.1, [triton/numerics](../triton/numerics.md) §9).

## Counter-signals

- **DS256 and DS512 each have their lowest value at 32–256 tokens**, where the kernel and the baseline both stream
  nearly every expert's weights (the link is inferred): DS256 measured 1.086 at 32 tokens, 1.030 at 128 and 1.160 at
  256, DS512 1.043 at 256 *(field, E1)*. Outside that band the lowest are DS512 at 16 tokens (1.097)
  and at 1024 (1.152).
- **Sizes with no kernel.** A count outside the 24 has none: DS512 at 32–128 tokens, DS256 at 64, both between 256
  and 1024 and above 4192.
- **One run of the ruler per kernel.** Run against itself, DS256 at 2 tokens spread 0.9–3.2% over 3 runs in four
  sessions *(field, E2)* and 3.05% over 14 runs on the rocprofiler-sdk timer
  *(field, E3)*: one run does not resolve a difference of about 1% at these sizes. A later run of
  the ruler re-measured 57 of the library's 96 MoE sizes and moved them by −6.3% to +7.0%, median 0.9%
  *(field, E1)*.
- **Little room under the bound.** The kernels quantize both expert-GEMM inputs to MXFP4, which the reference leaves
  unquantized. Inside the reference that rounding alone comes to 13.45 dB at 256 tokens
  *(field, E4)* against the floor of 10 dB, so a further independent error has to stay above
  about 12.6 dB (estimated) ([triton/numerics](../triton/numerics.md) §1).
- **The scenario is written as literals, and no kernel of the family asserts it.** 256 experts are `(256,)` vectors
  and top-8 plus the shared expert `row * 9` (DS256/m2); the routed weights are multiplied by 2.5 in device code;
  DS256/m8–m16 sums 12 router splits in its top-k. The host code of DS256/m2 ran at other shapes without an error,
  and a call with another scenario computes the wrong thing without one: read every literal, not only your size's
  branch ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1, [triton/moe](../triton/moe.md) §12).
- **Branches a size does not take.** A kernel grown from a source for several token counts keeps paths that its own
  count does not take: DS256/m1 a 2-token projection and a non-peeled top-k, the DS512 kernel for 256 tokens a host
  branch for sizes it is not written for. Only the branch that a size's constexprs select was compiled and checked
  (§ Code, entry 7, shows lines that are not traced at all). A kernel's own description can miscount too:
  DS256/m4's lists two launches, and a trace of its call shows three *(field, E5)*.
- **Read from the source, not run.** DS256/m4096–m4192 aliases the router logits with the activation buffer and the
  quantized input with the partials; its host code writes, reads and overwrites each aliased buffer in launch
  order, and no GPU run tested the aliasing ([triton/moe](../triton/moe.md) §12). The XCD placement behind the
  program orders (§ Code, entry 8; the down of DS256/m8–m16) is derived from the formulas: where a program ran was
  not measured on a MoE kernel ([triton/moe](../triton/moe.md) §7). For a kernel started from this family,
  [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §3 lists the shared expert's combine as
  untested; this page has no measurement of that combine alone.
- **The fusions were timed on toy kernels, not on these.** Recomputing the top-k in the consumers, SwiGLU in the
  gate-up epilogue and down + combine in one program each saved time on the ruler's timer in a toy top-4-of-32 MoE;
  on synthetic fp8 VALU shapes with 256 experts and top-8 the three together took longer to replay than the unfused
  K-split design ([triton/moe](../triton/moe.md) §2, §4, §9). No kernel of the family runs the same design with a
  separate selection launch, so its speedups do not isolate a fusion.

## Mechanism

Kernels with their token counts in parentheses; DS512 has no kernel at 32–128.

| Tokens | DS256 | DS512 | Launches | Decomposition |
|---|---|---|---|---|
| 1–2 | a kernel per count (1), (2) | one kernel (1, 2) | 3–4 | router (split-K tiles, or one full-K program per expert at DS256, 2 tokens) beside the hidden-state MXFP4 quantization (192 + 24 programs at DS256, 1 token); the gate-up of every route split along K (4 splits at DS256, 12 at DS512 at 1 token) and finished by a small launch, or whole (DS512 at 2); a last launch of down and combine over (token, column tile): 16 columns at DS256, 32 and 64 at DS512 at 1 and 2 |
| 4–16 | two kernels (4), (8, 16) | a kernel per count (4), (8), (16) | 3 (DS512 at 16 tokens: 5) | router + quantization; one launch runs the gate-up of every route, no separate activation launch; down + combine. Inter 512 at 16 tokens: a front launch, a selection launch that writes masks, two grouped-expert launches, a combine |
| 32–256 | a kernel per count (32), (128), (256) | one kernel (256) | 5 (DS512: 6) | router, quantization and counter initialization in the first launches; a selection launch on relaxed atomics; gate-up over expert jobs (16 to 128 rows at DS512; DS256 at 128 and 256 tokens derives the job from the counts); down; a last, atomic-free launch reduces the parts and writes `out` |
| 1024–4192 | two kernels (1024, 2048, 3072), (4096, 4192) | a kernel per count (1024), (2048), (3072), (4096), (4192) | 6 | full-K router + quantization; route selection; a prepare launch writes job lists (at DS512, 1024–3072 tokens: full 128-row tiles first, then tails by falling height down to 16 rows, the mix changing with the routing); grouped gate-up; routed down; the shared expert's down, which writes `out` |

- **Techniques.** Router tiles and the MXFP4 quantization of the hidden states share one launch, split by program id
  ([triton/moe](../triton/moe.md) §3). At 1–16 tokens every kernel but DS512 at 16 tokens selects experts in the kernel with a rank loop; at DS256, 2 tokens,
  each gate-up program recomputes its route's expert with rank + 1 argmax rounds instead of waiting for a top-k launch
  ([triton/moe](../triton/moe.md) §4, [../moe/few_tokens.md](../moe/few_tokens.md)). A gate-up N tile holds matching gate and up
  rows; in full-K gate-up designs such as DS256 at 256 tokens, SwiGLU and the MXFP4 requantization run in its epilogue ([triton/moe](../triton/moe.md) §7, §8). At DS512, 1–2 tokens, the down
  multiplies a token's routes (MFMA rows) by their experts' column bands and keeps the diagonal, weighted and summed in
  slot order in registers ([triton/moe](../triton/moe.md) §9). At 32–256 tokens, the gate-up programs of DS256 at 256 tokens scan the per-expert
  counts with `gl.associative_scan` and make the job descriptor wave-uniform with `v_readfirstlane_b32`
  ([triton/gluon/wave_ops](../triton/gluon/wave_ops.md) §6). MXFP4 operands go to `mfma_scaled` with e8m0 scales in
  `get_mfma_scale_layout` layouts; activations are packed with `v_cvt_scalef32_pk_fp4_f32` inline asm
  ([triton/numerics](../triton/numerics.md) §3).
- **Resources** *(field, E6)*: 116 compiled-cache records, not distinct binaries ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1); `num_warps` 1 (57), 2 (8), 4 (51);
  `waves_per_eu` unset in all; VGPR 22–280, AGPRs in 2 (16, 24); LDS 0–74.8 KiB; no spills; scaled 16 × 16 × 128 MFMA in
  54, bf16 16 × 16 × 32 in 23, scaled 32 × 32 × 64 in 8.
- **Two calls, launch by launch** *(field, E5)*. DS256 at 2 tokens: router and quantization (304
  programs: a full-K router program per expert, the rest MXFP4 quantization); selection and gate-up (2306: 2
  selection and 2304 gate-up programs); activation and requantization (72); down and weighted combine (768). DS256
  at 256 tokens: router projection; selection (top-8, relaxed atomic tickets, and the hidden-state quantization,
  which in the DS256 kernels for 128 and for 256 tokens shares this launch, not the router's: read from their
  sources); gate-up over expert tiles; down over expert tiles; a fixed-order reduce.
- **The shared expert** (id 256, weight 1) is the token's ninth route at up to 16 tokens. At 32–256 tokens its jobs
  run first, its down result is stored straight into `out`, and the reduce adds the 8 routed partials on top of it
  (DS256 at 256 tokens). From 1024 tokens its down epilogue gathers each token's 8 routed partials and writes
  `out`, so there is no combine launch (the DS256 kernel for 4096 and 4192 tokens)
  ([triton/moe](../triton/moe.md) §10).

## Code

Entries 1–4 and 6–8 are from DS256 kernels, entries 5 and 9 from the DS512 kernel for 1–2 tokens.

### 1. A gate-up program recomputes its route's expert (DS256, 2 tokens)

The selection is recomputed per program: at 2 tokens the gate-up programs of the DS256 kernel do not wait for a
top-k launch, and each routed-route program calls this helper with its token (`row`) and its route's slot (`rank`).
`_routing_scores` returns the 256 expert ids `e`, four per lane, and their `score`, the sigmoid of the bf16-rounded
logit plus the fp32 correction bias; at 2 tokens `SPLITS` is 1 and `Logits` holds that sigmoid, which the router
launch stored per expert. `_next_expert` is one argmax round, the lowest available id among the experts that hold
the maximum score; the loop retires each winner and runs `rank` + 1 rounds in all. Run over 16384 one-wave programs,
this election took 9.7–10.7 µs at slots 0–4 and 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5, 6 and 7, in 3 runs
*(field, E5)*. The programs that publish `Ids` and `Weights` select with the same helpers; entry 3
says what happens when the two selections differ ([triton/moe](../triton/moe.md) §4.1, §5.3).

```python
def _expert_at_rank(Logits, Bias, row, rank, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    e, _, score = _routing_scores(Logits, Bias, row, SPLITS)
    available = gl.full((256,), True, gl.int1, layout)
    idx = _next_expert(score, available, e)
    for j in range(rank):
        available = available & (e != idx)
        score = gl.where(e == idx, -float("inf"), score)
        idx = _next_expert(score, available, e)
    return idx
```

### 2. Router tiles and the input's MXFP4 quantization in one grid, split by program id (DS256, 1 token)

Two phases that read only the call's inputs can share one grid, each program taking its role from its id; the role
test is a scalar, so it is uniform over the program. In the first launch of the DS256 kernel for 1 token, 192
one-wave programs each compute a 16 × 16 router tile over one K split of 512 (16 expert tiles × 12 splits) and store
its fp32 partial, beside 24 programs that quantize the token to MXFP4 *(field, E5)*. `BN` is the
width of an expert tile and `SPLITS` the split count; `Y` holds the fp32 partials, indexed (token, split, expert),
`om` and `on` are the tile's rows and expert columns, and `om < M` masks the rows past the token count `M`; at 1
token `GROUPED` is false, so the `elif` arm is not compiled. `_quantize_groups(values, shared)` returns the e2m1
codes and e8m0 scales of the 32-value groups, and it is called twice because the routed experts and the shared
expert take different exponent rules ([triton/numerics](../triton/numerics.md) §3). On a toy router GEMV beside a
4-program elementwise phase, one grid took 0.495–0.506× the time of two launches on the ruler's timer in 4 runs, and
0.755–0.835× the replay time in three of them but 1.154× in the fourth *(field, E7)*.

```python
    program = gl.program_id(0)
    if program < 256 // BN * SPLITS:
        tile = program % (256 // BN)
        split = program // (256 // BN)
        # ... (a 16 × BN router tile over this split's K on bf16 MFMAs, accumulated in acc)
        gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
    elif GROUPED and program == 256 // BN * SPLITS:
        # ... (one program initializes the per-expert group masks)
    else:
        quant = program - 256 // BN * SPLITS - (1 if GROUPED else 0)
        # ... (this program's row and K groups loaded as fp32 values)
        routed, routed_scale = _quantize_groups(values, False)
        shared, shared_scale = _quantize_groups(values, True)
        # ... (both quantizations and their scales stored)
```

Explained in [triton/moe](../triton/moe.md) §3.

### 3. The top-k recomputed from the router's split partials in every gate-up program (DS256, 1 token)

At 1 token the DS256 kernel runs 1153 programs in its selection and gate-up launch *(field, E5)*:
program 0 publishes `Ids` and `Weights` for the down launch, the 1024 routed-route programs (8 routes × 4 K splits ×
32 column tiles) each recompute their route's expert with this helper, and the 128 shared-expert ones take id 256.
The helper sums the router's 12 fp32 split partials in split order, rounds once to bf16, the reference's rounding
point, adds the bias to the sigmoid and runs `rank` + 1 elections. `_elect_rank` is the election by ballot, in which
the lowest id wins a tie ([triton/moe](../triton/moe.md) §5.3). The kernel maps job j to slot 7 − j
(`route = gl.where(job < 8, 7 - job, 8)`), so the selections with the most rounds start first; no source says why,
and no measurement compares the orders.

```python
@gluon.jit
def _expert_at_rank_peeled(Logits, Bias, row, rank, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    e = gl.arange(0, 256, layout)
    logits = gl.load(Logits + row * SPLITS * 256 + e)
    for part in gl.static_range(1, SPLITS):
        logits += gl.load(Logits + (row * SPLITS + part) * 256 + e)
    logits = logits.to(gl.bfloat16).to(gl.float32)
    probability = 1.0 / (1.0 + gl.exp(-logits))
    score = probability + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    idx = _elect_rank(score, available, e)
    for j in range(rank):
        available = available & (e != idx)
        score = gl.where(e == idx, -float("inf"), score)
        idx = _elect_rank(score, available, e)
    return idx
```

Not this: a gate-up copy of the selection whose bf16 rounding or tie rule differs from the publishing program's.
With only the gate-up copy changed, producer and consumer disagreed on a slot's expert on 12 of 13 inputs
*(field, E8)*, and in the accuracy check the copy without the bf16 rounding failed while the copy
with the higher-id tie rule passed *(field, E9)*.

Explained in [triton/moe](../triton/moe.md) §4.1.

### 4. A gate-up program derives its job from the per-expert counts (DS256, 256 tokens)

The DS256 kernel for 256 tokens has no job-list launch: each gate-up program finds its tile from the per-expert
counts, which the relaxed tickets of the selection launch left in `Counts`. `TM` is the tile height and `PAD`, the
token count rounded up to `TM`, an expert's region of the sorted buffer; `tile` is the program's tile, counted after
the `shared_jobs` tiles of the shared expert; `M` is the token count, and the experts' counters are
`_NATIVE_COUNT_STRIDE` = 16 int32 apart at this size (the `M <= 128` arm, 32 apart, is not taken at 256 tokens).
Two prefix sums share one `gl.associative_scan` by bit packing, an expert's tiles (`chunks`) in the low 10 bits and
its routes above them, with `gl.static_assert` on the ranges; `gl.max` then keeps the one expert whose tiles hold
`tile`. The descriptor holds live rows − 1 in bits 0–4, the tile `e · PAD/TM + chunk` in bits 5–16 and its first
activation row from bit 17; after the excerpt `v_readfirstlane_b32` in inline assembly makes it wave-uniform, and
the column-0 program stores it for the down launch. No measurement compares deriving the job with a job list that
the selection launch writes.

```python
        e = gl.arange(0, 256, scan_layout)
        count = gl.load(Counts + e * (32 if M <= 128 else _NATIVE_COUNT_STRIDE))
        chunks = ((count + TM - 1).to(gl.uint32) // TM).to(gl.int32)
        end = gl.associative_scan(chunks + (count << 10), 0, _add)
        begin = (end & 1023) - chunks
        row_begin = (end >> 10) - count
        chunk = tile - shared_jobs - begin
        packed = (
            (e * (PAD // TM) + chunk) * 32 + gl.minimum(TM, count - chunk * TM) - 1 + (row_begin + chunk * TM << 17)
        )
        descriptor = gl.max(gl.where((chunk >= 0) & (chunk < chunks), packed, -1), 0)
```

Not this: the 4-warp `gl.max` result used as it is where the base of a `buffer_load` depends on it. Compiled on a
test kernel, such a base turned the load into a waterfall loop (`v_readfirstlane_b32`, `s_and_saveexec_b64`,
`s_cbranch_execnz`) *(field, E10)*.

Explained in [triton/moe](../triton/moe.md) §4.2.

### 5. Down and combine in one program: a token's routes against their experts' column slices (DS512, 1–2 tokens)

The DS512 kernel for 1–2 tokens is a whole call in 3–4 launches. Its last launch runs one program per (token,
column tile) that does the down projection and the combine: `_tiny_projection` puts the token's routes in the
MFMA's A rows and the `BN`-column slices of their 8 experts side by side in B, and the `gl.gather` keeps the
diagonal blocks, route r against expert r. The loop weighs them and sums them in slot order in registers, without
partials in memory or atomics. The second `_tiny_projection` call is the shared expert's; after the excerpt its row
is rounded to bf16 and added.

```python
    acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, False, CACHE)
    vn = gl.arange(0, 8 * BN, gl.SliceLayout(0, mma))
    diagonal = gl.gather(acc, vn[None, :] // BN, 0).reshape((8, BN))
    # ... (diagonal converted to a layout ep that holds one row per route; value starts at 0 in it)
    for rank in gl.static_range(8):
        part = gl.gather(diagonal, gl.full((1, BN), rank, gl.int32, ep), 0)
        weight = gl.load(Weights + token * 8 + rank)
        value += part * weight
    shared_acc = _tiny_projection(X, XS, W, WS, Ids, token, column, N, K, BN, BK, WARPS, True, CACHE)
```

Explained in [triton/moe](../triton/moe.md) §9.

### 6. One scratch allocation carved into typed regions (DS256, 128 tokens)

The DS256 kernel for 128 tokens allocates its scratch inside the call, on the input's device and sized from the
call's shapes, as one `torch.empty` uint8 arena. It carves the arena into 11 typed regions, each starting on a
64 KiB boundary; `specifications` lists their shapes and dtypes, among them the job list, the per-expert counts, the
sorted routes, the router logits and the quantized activations with their scales. `torch.empty` holds whatever was
there, so what a launch reads before it writes is written by an earlier launch of the call: program 0 of the router
launch stores 0 into the per-expert counts, from which the selection launch takes its tickets.

```python
# a DeepSeek-V2 MXFP4 MoE: one uint8 arena, each typed region starting on a 64 KiB boundary
regions = []
total_bytes = 0
for shape, dtype in specifications:  # ((jobs_count,), torch.int32), ((m, 8), torch.float32), ...
    region_bytes = dtype.itemsize * prod(shape)
    regions.append((total_bytes, region_bytes, shape, dtype))
    total_bytes += triton.cdiv(region_bytes, 65536) * 65536
arena = torch.empty((total_bytes,), dtype=torch.uint8, device=x.device)
return [arena.narrow(0, offset, size).view(dtype).view(shape) for offset, size, shape, dtype in regions]
```

Not this: a region carved at an offset that is not a multiple of 16 bytes. Its tensor loses the `D` key and the
launch compiles another binary: on a probe kernel, a view at offset 8 of a 4 KiB buffer kept only the `S` key
*(field, E11)*.

Explained in [triton/authoring](../triton/authoring.md) §6.

### 7. Lines after a `return` are not traced (DS256, 2 tokens)

`_quantized_bytes` packs a group's activations into fp4 bytes in the DS256 kernel for 2 tokens: `x` holds the values
and `a` their magnitudes over the group's scale; the helper applies the sign of `x` to `a`, converts each pair with
`v_cvt_scalef32_pk_fp4_f32` and returns the bytes. Triton does not trace what follows a `return` in the same block
(S1), so the three sign-OR lines under `return packed` are dead code: an edit there changes
no instruction. The same holds for a branch that the size's constexprs do not select. To see whether a change is
compiled, put `gl.static_print` in the branch or diff the amdgcn of two builds.

```python
@gluon.jit
def _quantized_bytes(x, a):
    # ... the sign of x applied to a, a split into lo and hi ...
    packed = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]",
        constraints="=v,v,v",
        args=[lo, hi],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    ).to(gl.uint8)
    return packed  # the block ends here: an edit to the three lines below changes no instruction
    sign = gl.convert_layout(gl.where(x < 0, 8, 0).to(gl.uint8), pack_layout)
    sign_lo, sign_hi = gl.split(sign.reshape((x.shape[0], 16, 2)))
    return packed | sign_lo | sign_hi << 4
```

Explained in [triton/debugging](../triton/debugging.md) §3.

### 8. A grouped walk of the router's tiles (DS256, 4192 tokens)

gfx950 deals workgroups round-robin over 8 XCDs, each with its own L2: program i ran on XCD i % 8 on the default
stream *(field, E12)* and on XCD (i + c) % 8 on a new one
*(field, E13)*, so programs whose ids are congruent mod 8 share an XCD. Above 4096 tokens,
at 4192 in this family, the DS256 router maps the 256 program ids of its 64 row tiles of 64 tokens, the first 4096
tokens, to a (row tile, column tile) with this helper, called with 4 column tiles and `GROUP = 8`: 8 consecutive
ids take 8 row tiles of one column, and ids `GROUP` apart take the next column of the same row tile, so the 4
column tiles of one row tile share an XCD; the 12 programs of the last 96 tokens walk the rows within a column.
That placement is derived from the code and was not measured. At 1024–4096 tokens the router walks the rows within
a column, `(pid % ROWS, pid // ROWS)`. The gate-up launch of the two DS256 kernels from 1024 tokens orders its
(job, column tile) programs with the same helper and `GROUP = 8` (read from the source).

```python
@gluon.jit
def _grouped_tile(pid, ROWS: gl.constexpr, COLS: gl.constexpr, GROUP: gl.constexpr):
    gl.static_assert(ROWS % GROUP == 0)
    row = pid // (GROUP * COLS) * GROUP + pid % GROUP  # GROUP consecutive ids: GROUP row tiles of one column
    column = pid // GROUP % COLS  # ids GROUP apart: the next column of the same row tile
    return (row, column)
```

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §14.

### 9. Shuffled e8m0 scale words unpacked into (row, group) bytes (DS512, 1–2 tokens)

The weights' e8m0 scales arrive in aiter's `shuffle_scale` order, in which one 32-bit word holds the bytes of rows
n and n + 16 at groups g and g + 4. The gate-up of the DS512 kernel for 1–2 tokens loads a tile's scale words whole
(`sw`), splits each word into its four bytes with two `gl.join` levels, which add two trailing axes of 2, and
permutes the six axes back to (row, group) order. The reshape gives the `[BN, BK // 32]` scale tile of the B operand
of `mfma_scaled`: `BN` weight rows by the 32-value groups of one K step.

```python
@gluon.jit
def _unpack_bytes(words):
    b0, b1 = (words.to(gl.uint8), (words >> 8).to(gl.uint8))
    b2, b3 = ((words >> 16).to(gl.uint8), (words >> 24).to(gl.uint8))
    return gl.join(gl.join(b0, b2), gl.join(b1, b3))  # two join levels: two new trailing axes of 2

raw = sw.reshape((BN // 32, BK // 256, 4, 16))
scales = gl.permute(_unpack_bytes(raw), (0, 5, 3, 1, 4, 2))  # back to (row, group) order
scales = scales.reshape((BN, BK // 32))
```

Not this: the scale words only reshaped to `[BN, BK // 32]`. On a test kernel (one 16 × 16 MXFP4 tile, K 512) that
gave a relative error of 0.94 *(field, E14)*.

Explained in [triton/gluon/mfma](../triton/gluon/mfma.md) §7.

## Result

- **Speedups** over the baseline above *(field, E15)*: 1.030 (DS256, 128 tokens) to 2.186 (DS256, 2 tokens) over
  the 24 sizes.

Per size, each kernel's value from one run of the ruler *(field, E1)*; a dash is a token count with
no kernel:

| Tokens | DS256 | DS512 |
|---|---|---|
| 1 | 2.139 | 2.093 |
| 2 | 2.186 | 1.829 |
| 4 | 1.867 | 1.324 |
| 8 | 1.521 | 1.184 |
| 16 | 1.245 | 1.097 |
| 32 | 1.086 | — |
| 128 | 1.030 | — |
| 256 | 1.160 | 1.043 |
| 1024 | 1.402 | 1.152 |
| 2048 | 1.329 | 1.391 |
| 3072 | 1.407 | 1.235 |
| 4096 | 1.420 | 1.161 |
| 4192 | 1.378 | 1.166 |

- **Where the time goes at the two ends.** At 2 tokens the DS256 call took 19.06–19.61 µs in its 4 launches,
  4.76–4.90 µs per launch (4 runs) *(field, E5)*, against a separately measured floor of
  4.03–4.07 µs per launch on the ruler's timer *(field, E7)*: the reading fits a launch cost and
  does not isolate it. In 14 runs on the rocprofiler-sdk timer the kernel took 18.9–19.3 µs and the baseline
  41.8–42.5 µs *(field, E3)*; the baseline enqueues 8–11 device operations at 1–2 tokens, these
  kernels 3–4 *(field, E5)*. At 256 tokens the DS512 call reads all experts' weights, about
  1.29 GB, in 225.5–236.8 µs (4 runs) *(field, E5)*, about 80–95% of a 6.0–6.8 TB/s stream
  (estimated), which the baseline pays too.
- **The tile order, changed alone.** In one run, at 2048 tokens, the DS256 kernel for 1024–3072 tokens (A) was
  timed against a copy with only its tile order changed (B), the whole call: mean B/A 1.02654, the 95% t interval of
  the paired mean ratio 1.0104–1.0428, an interval within that run and not across runs, and every B sample slower
  than every A sample in 4 balanced pairs *(field, E16)*. No L2 counter was read,
  so the cause is unknown ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §4).
- **Other changes of one constant measured on these kernels** are in the general pages: the gate-up K split at 1–2
  tokens and `.cg` on the weight loads per size ([triton/moe](../triton/moe.md) §7), `enable_fp_fusion` and the
  router-logit rounding at DS512, 8 tokens ([triton/moe](../triton/moe.md) §8, §5.1), the weight panel held across
  activation tiles at DS256, 2048 tokens ([triton/moe](../triton/moe.md) §9).

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush, Kineto), speedup over the baseline of SGLang's path on aiter] speedup per size, the implementation in parentheses: DeepSeek-V2/GLM DS256: 1 token 2.139 (DS256/m1), 2 2.186 (DS256/m2), 4 1.867 (DS256/m4), 8 1.521 and 16 1.245 (DS256/m8–m16), 32 1.086 (DS256/m32), 128 1.030 (DS256/m128), 256 1.160 (DS256/m256), 1024 1.402, 2048 1.329 and 3072 1.407 (DS256/m1024–m3072), 4096 1.420 and 4192 1.378 (DS256/m4096–m4192); DS512: 1 token 2.093 and 2 1.829 (DS512/m1–m2), 4 1.324 (DS512/m4), 8 1.184 (DS512/m8), 16 1.097 (DS512/m16), 256 1.043 (DS512/m256), 1024 1.152 (DS512/m1024), 2048 1.391 (DS512/m2048), 3072 1.235 (DS512/m3072), 4096 1.161 (DS512/m4096), 4192 1.166 (DS512/m4192); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (MM256-FP8/m1–m16); 32 1.158 and 64 1.173 (MM256-FP8/m32–m64); 128 1.106 (MM256-FP8/m128); 256 1.133 (MM256-FP8/m256); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (MM256-FP8/m1024–m16384); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (MM64-FP8/m1–m16); 32 1.167 and 64 1.102 (MM64-FP8/m32–m64); 128 1.146 and 256 1.164 (MM64-FP8/m128–m256); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (MM64-FP8/m1024–m16384); MXFP4, 256 experts: 1 token 3.444 (MM256-FP4/m1), 2 3.309 (MM256-FP4/m2), 4, 8, 16 2.549, 1.977, 1.595 (MM256-FP4/m4–m16), 32 1.179 (MM256-FP4/m32), 64 1.230 (MM256-FP4/m64), 128 1.210 (MM256-FP4/m128), 256 1.178 (MM256-FP4/m256), 1024 1.352 (MM256-FP4/m1024), 2048 1.272 and 4096 1.261 (MM256-FP4/m2048–m4096), 6144 1.415 (MM256-FP4/m6144), 8192 1.423 (MM256-FP4/m8192), 16384 1.389 (MM256-FP4/m16384); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (MM64-FP4/m1–m2), 4 2.554 (MM64-FP4/m4), 8 2.113 and 16 1.555 (MM64-FP4/m8–m16), 32 1.231 and 64 1.275 (MM64-FP4/m32–m64), 128 1.190 and 256 1.194 (MM64-FP4/m128–m256), 1024 1.252 (MM64-FP4/m1024), 2048 1.318 (MM64-FP4/m2048), 4096 1.590 (MM64-FP4/m4096), 6144 1.689 (MM64-FP4/m6144), 8192 1.735 (MM64-FP4/m8192), 16384 1.619 (MM64-FP4/m16384); Qwen2-MoE router+experts: 1 token 1.172 (Q2-RE/m1), 2 1.145 (Q2-RE/m2), 4 1.119 (Q2-RE/m4), 32 1.063 (Q2-RE/m32); router: 8 tokens 1.074 (Q2-R/m8), 16 1.219 (Q2-R/m16), 64 1.295 (Q2-R/m64), 128 1.417 (Q2-R/m128), 256 1.967 (Q2-R/m256), 2232 1.416 (Q2-R/m2232), 4096 1.226 (Q2-R/m4096); fp8 experts: 8 tokens 1.208 (Q2-E/m8), 64 1.171 (Q2-E/m64), 128 1.076 (Q2-E/m128); the values come from eight runs of the ruler, each implementation's from one run; a later run re-measured 57 of the 96 sizes, by −6.3% (MM256-FP8 at 1 token, 4.115 → 3.855) to +7.0% (Q2-R at 16 tokens, 1.219 → 1.304), median change 0.9%; over repeat runs of the ruler on unchanged sources, the ratios at the same size differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 sizes, at most 54.8 % (Q2-RE at 1 token)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4 sessions, the ruler, one GPU, the implementation against itself and an identical copy] first session: Q2-R/m256 at 256 tokens 1.93, 1.92, 1.94; DS256/m2 at 2 tokens 2.18, 2.19, 2.17; second, (max − min)/mean of the first 3 runs: Q2-R/m256 0.91 %, DS256/m2 0.94 %; over all 8 runs (A and the copy) 0.95 % and 1.27 %; third and fourth, first 3 runs: Q2-R/m256 0.98 % and 2.46 %, DS256/m2 2.39 % and 3.17 %
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=14 runs per implementation, the ruler (rocprofiler-sdk), each run a new process, the implementation against itself] speedup: Q2-R/m256 at 256 tokens 1.972–2.008, (max − min)/mean 1.81 %; DS256/m2 at 2 tokens 2.169–2.236, 3.05 %, its time 18.9–19.3 µs and the baseline's 41.8–42.5 µs
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, runs on one GPU: the bound formulas applied to built outputs, or the accuracy check run on a library kernel and on an edited copy] a 0/0 tolerance on bf16: B = 2^-8·|y|, the round-to-nearest output passed, 1% of elements one step further failed (fraction over the bound 0.00995); MXFP4 round trip (ceil e8m0, RNE e2m1) of both expert-GEMM inputs inside the references, first input set: DeepSeek-V2 13.45 dB at 256 tokens, MiniMax-M2 13.18 dB at 128; an fp32 dot-product rounding bound at K = 6144: fp32 sums in 1, 6 and 16 parts passed at peak 0.69; bias test, n = 10^6, every error +0.5·u·|y|: |mean|/rms 0.80, the output-rounding and dot-product bounds failed, the tolerance, the SQNR floor and a bound relative to a same-mean baseline passed; RTZ to bf16 of positive values 0.68; the check's large-magnitude input set, every floating-point input scaled by one exact power of two, the largest that keeps the tensors and the dot-product outputs in range: 2^54 for the N 2624 × K 6144 bf16 GEMM and 2^55 for the N 6144 × K 2048 bf16 GEMM at all 10 sizes; Qwen3.5 GDN decode bounds: 65536 codes moved one e4m3 step and 512 scales moved one ulp all passed; a8w8 block scales, 2400 drawn: log2 from −2.999 to −0.0009, none an integer; failures: `0 · (1 / gl.exp2(-127))` 256 of 256 non-finite mismatches, a 0xFF scale, an RTZ'd NaN, saturated 448, 65504 and fp32 maximum, NaN against 448 one each; a bf16 running sum over K = 2048 fraction over the bound 0.941 at peak 58.4, a stray fp4 nibble 0.438, an unwritten output 1 at peak 128; missing tap rounding, a contracted product and x·(1/s) at injected ties 1.4e-4, 9.5e-5, 1.2e-4 at peaks 1.84, 1.95, 2.67 under bounds chosen for the test; fp8 P on P·V 23.1 dB; MXFP4 18.77 dB, 17.07 with an e5m2 rounding first; the Qwen2-MoE router kernel for 256 tokens, seed 0: fp32 logits rounded once to bf16, then ranked, gave the reference's top-10 ids for all 256 tokens, the unrounded sum ranked changed 97 tokens and failed the id comparison (fraction of ids differing 0.0809) in both checked runs of each run; the a8w8 GEMM N 7168 × K 2048 at m = 64: `dot * sx * sw` against `dot * (sx * sw)` in K stages 0–2 of 8 (stages 3–7 multiply by a precomputed sx · sw in both arms), 2 checked runs per arm in each run: all 12 passed, worst fraction over the bound 0
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the implementations' call times N=4, the top-k probes N=3), one GPU: launch traces with torch.profiler, the ruler's timer (Kineto) and bare graph replay on test kernels, the ruler on the implementations and on copies with one change (A/B: 4 alternated rounds per arm)] launch traces of the 96 MoE sizes: launches per call as in [triton/moe](../triton/moe.md) §1; every size compiles on its first call, none on its second, and 136 of 471 launches carry no token-count constexpr; DS256/m2 at 2 tokens grids 304, 2306, 72, 768; DS256/m1 at 1 token router launch 216 = 192 + 24 programs, selection and gate-up launch 1153; all 32 DeepSeek and MiniMax sizes from 1024 tokens sort before the gate-up; none of 185 kernel names is a quantizer alone; routers of 512 programs in MM256-FP4/m128, MM256-FP4/m256, MM256-FP8/m256, MM64-FP8/m128–m256 and of one full-K program per (token block, expert) in MM256-FP8/m1–m16, MM256-FP4/m2, MM64-FP4/m8–m16 above 1 token; Q2-E/m64 at 64 grids of 3072 and 4096, MM64-FP8/m128–m256 at 256 a persistent W13 kernel on 48 × 96; the same grids on two routings and graph replay equal to eager at 82 sizes; split-K finish launches in DS256/m2, DS256/m1, DS512/m1–m2 at 1 token and Q2-E/m8; the baseline enqueues 8–11 device operations at 1–2 tokens, these implementations 3–4; docstring lists differ from the trace at 7 sizes in 5 of 11 implementations; one trivial kernel reads 4.21 µs on the ruler's timer, each added one 4.06 µs, and a bare graph replay pays 1.815–1.844 µs per extra kernel; call times of the unchanged implementations, mean of 2 measurements of the ruler's timer, 4 runs: DS256/m2 at 2 tokens 19.06–19.61 µs (4.76–4.90 µs per launch), DS512/m256 at 256 225.5–236.8 µs, MM64-FP4/m128–m256 at 256 103.6–105.4 µs, MM256-FP4/m8192 at 8192 505.7–518.6 µs, an earlier batch of runs 225.8, 102.9 and 515.6 µs for DS512/m256, MM64-FP4/m128–m256 and MM256-FP4/m8192; DS256/m2 at 2 tokens, gate-up K splits 4 → 2 (grid 2306 → 1154): 2.183 → 2.175, 19.20 → 19.28 µs, runs 2.167–2.195 against 2.157–2.191, paired 95 % confidence interval of the ratio 0.991–1.001; top-8 of 256 at 16384 tokens: one wave per token 12.96–13.18 µs with no `s_barrier`, four waves 37.68–38.32 µs with 31, the same ids, no scratch in either; the keyed min (retired scores at −inf) 0 wrong of 4000 rows; MM64-FP8/m1–m16 exact at 8 and 16 tokens; MM64-FP4/m8–m16 at 2 tokens keeps 8 through its id tie rule, at 8 tokens selects 9 on a tied 8th and 9th (16 tokens not run); DS256/m2's election (the form of [triton/moe](../triton/moe.md) §4.1) over 16384 programs 9.7–10.7 µs at slots 0–4, 11.3–11.4, 12.8–13.0 and 14.3–14.4 µs at slots 5–7, R² of a linear fit 0.724–0.774; with MM64-FP4/m4096's combine lanes disagree on 10000 of 10000 NaN rows and 0 of 10000 finite ones; MM256-FP8/m1–m16 and MM256-FP4/m2 give ±0 and every NaN payload one key each, Q2-RE/m1 gives 7 bf16 NaNs 7 keys; an MFMA-only fp16 loop with 4 useful rows: `instr_shape=[4, 64, 64]` 2432 µs, [16, 16, 32] 2810 µs; every finite E4M3FN code converts exactly to bf16; integer-scratch patterns over 100 calls: Q2-E/m64 at 64 tokens 9 and Q2-E/m128 at 128 4 in one run, 15 and 8 in another; 31 implementations with atomics, 20 runs each, one output each (the 96 MoE sizes; probe kernels)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and the one-grid router N=4), one GPU: 1 GiB stream probes timed as 2^30 bytes over the median of 9 rounds of 10 launches; the ruler's timer (Kineto) and bare graph replay on test kernels and on DS256/m2; the ruler A/B, 4 alternated rounds per arm] a 1 GiB bf16 tensor read once per launch by 2048 programs of 4 warps, 16 bytes per lane: 6013, 6009, 6009 GB/s plain with `buffer_load` (5994, 6000, 5993 with `gl.load`), 6823, 6805, 6802 GB/s with `.cg` (6803, 6791, 6785), so 6.0 and 6.8 TB/s; DS256/m2 at 2 tokens: 4 launches (router, selection and gate-up, activation, down) in every trace, call 19.00, 19.17, 19.04 µs, 4.75, 4.79, 4.76 µs per kernel against a per-launch floor (slope over chains of 1–8 trivial kernels) of 4.041, 4.029, 4.066 µs, so the 4 floors are 84–85 % of the call; 4 chained trivial kernels 16.41, 16.44, 16.42 µs, one kernel doing their 4 updates 4.30, 4.30, 4.32 µs; a 1 GiB `.cg` stream over 256–8192 programs of 1, 2, 4 or 8 warps with 1–8 independent 16-byte loads per lane: configurations with equal bytes in flight within 1.104× of each other (the widest at 4 MB: 256 programs × 4 warps × 4 loads 5697–5756 GB/s, 512 × 4 × 2 6287–6333 GB/s), 256 × 4 × 8 over 2048 × 4 × 1 1.067–1.070; at 3 MB 384 × 4 × 2 4920–4962 GB/s, 768 × 2 × 2 4898–4947, 96 × 4 × 8 3945–3995 (0.80×), 192 × 1 × 16 2454–2467 (0.495–0.499×); Q2-R/m256 at 256 tokens, B/A speedup: split 4 → 2 (256 → 128 CTAs) 0.749, 0.748, 0.753 (9.93–10.03 → 13.26–13.32 µs), 4 → 8 (512 CTAs) 0.961, 0.967, 0.978, look-ahead 2 → 1 0.889, 0.881, 0.890; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization), on the ruler's timer: top-k recomputed in the gate-up and down programs instead of a selection launch saves 3.07, 2.89, 3.22, 3.40 µs at 2 tokens and 3.44, 3.22, 3.41, 3.87 µs at 16 (replayed at 2 tokens 1.24–1.80 µs, call-time ratio 0.843–0.889), down and combine in one program per (token, 64-column tile) instead of a down and a combine launch 2.94, 2.97, 3.32, 3.23 µs at 2 tokens and 3.16, 3.18, 3.38, 3.73 µs at 16 (not replayed); a toy router GEMV (64 programs, 2 rows, K 2048) beside a 4-program elementwise phase in one grid: outputs equal, one grid over two launches 0.499, 0.495, 0.506, 0.498 on the ruler's timer, 0.835, 0.775, 0.755 and 1.154 (9.360 against 8.110 µs) replayed; a two-role test kernel compiles at 256 VGPRs and 32 B of LDS, its streaming role alone at 16 VGPRs and no LDS, its heavy role at 256 and 32 B (probe and toy kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, copies of DS256/m1's selection and expert code run directly at 1 token on natural and saved inputs, frozen reference ids and slot weights] four saved inputs with near or exact ties: replacing only slot 7's expert (by an unselected one) in both gate-up and down moved `out` 6.21, 112.2, 6.13 and 9.47 times as far in L2 as removing only the bf16 rounding of the activation product, total errors against the reference 1.22, 3.03, 1.23 and 1.28×; changing only the gate-up copy's bf16 rounding or tie rule: producer and consumer disagreed on a slot's expert on 12 of 13 natural inputs, both sides changed alike: agreement on 36 of 36; the 12 split partials summed in reverse order on four saved inputs: ids, weights and outputs unchanged
- E9 [2026-11, gfx950 (MI355X), Triton not recorded (MI355X image), N=1, the ruler with its accuracy check (the unchanged implementations twice), one input set per size, copies of the implementations with one change; the selection copies also run directly] Q2-R/m256 at 256 tokens with token 0's first two ids swapped, everything else bitwise equal: a mismatch on `topk_ids` (violation fraction 2/2560), the unchanged implementation passed twice; DS512/m8 at 8 tokens ranking the unrounded logits: passed twice, no token's experts changed, error change unresolved; a gated-attention kernel at 32 rows with the attention kept in fp32 before the gate: passed twice, its scales' relative L2 against the float64 reference without that rounding 0.0032334 → 0.0032690; DS256/m1 at 1 token, the gate-up copy without the bf16 rounding of the summed logits: 6 of 16 tokens took another expert, the accuracy check failed; with the higher id winning ties: 4 tokens, the accuracy check passed
- E10 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] `x_ptr + e * (7168 * 2048)`, `e` an int32 from `gl.load`: `mul i32 %13, 14680064` in the LLVM IR, with `e.to(gl.int64)` `mul nsw i64`; `buffer_load` from a base offset by a program id, a kernel argument or a scalar `gl.load`: 1 `v_readfirstlane_b32`, no loop, by a 1-warp `gl.max` none, by a 4-warp `gl.max` 5 `v_readfirstlane_b32`, `s_and_saveexec_b64`, `s_cbranch_execnz`; bf16 offsets + 128 → `offset:256`, + 2048 → a second offset VGPR, + 2047 → 8 `buffer_load_ushort`; steps of 16 and 4080 bytes in the immediate; the 16×16 preshuffle offset from an i32 tile argument: 28 VALU (int32), 22 (`.to(gl.uint32)`), 28 (int32 with `gl.assume(tile >= 0)`); 2 bf16 per lane, int32 or uint32 offsets: 1 `buffer_load_dword` / `global_load_dword` either way; bf16 `[32, 64]`, 16 B per lane, base `X + slot * stride`: `stride` unmarked → 8 `buffer_load_ushort` and the async copy "failed to translate module to LLVM IR", `stride` "D", constexpr 4104 or `gl.multiple_of(slot * stride, 16)` → 1 `buffer_load_dwordx4` (`… lds`); a uniform `gl.load` ahead of any copy or store → `s_load_dword` (test kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; planted copies on one GPU checked against the op's reference in captured graphs with poisoned outputs (a first capture and two replays with new inputs, fresh captures per input set, a repeat, a check after the timing) and timed by the ruler, and direct calls against the op's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the original kernel 0; a copy of an fp8 GEMM (N 3072, m = 16) storing 8 of its 16 rows: wrong values at the first capture with 24576 = 8 × 3072 non-finite mismatches, the original none (one run); the poison filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; a `run` keeping its first output in a module-level dict passed a source scan and gave wrong values at the first capture, the fresh captures, the repeat and the check after the timing; a math error: wrong values at the captures and after the timing; a changed reduction order: the same in one run, only a slower time in the other; state kept across calls: wrong values after the timing only; no difference between repeated runs in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / the ruler's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (copies of a plain-Triton RMSNorm at hidden 6144 and of an fp8 GEMM, probe kernels)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, 1 GiB read flush before each launch, 3 alternated rounds of 9 launches] every program's XCC id (`s_getreg`) equal to its id mod 8 on the default stream, in 4096-program launches and in 10 grids (1-D and multi-axis), 0 mismatches; 4 MiB L2 per XCD reported, a re-read per byte 2.22–2.23× slower at 8 than at 2 MiB per XCD; consecutive ids 21.3–21.6 µs (64 KB chunks) and 154.4–160.5 µs (512 KB), ids congruent mod 8 11.7–12.1 and 74.1–76.8 µs; ratio per round 1.76–1.82 and 2.07–2.17, round medians per run 1.78–1.80 and 2.08–2.12 (21.3–21.5 → 11.8–12.0, 156.3–159.0 → 74.4–74.6); one reader per chunk 0.98–1.01; the XCD remap of a bf16 GEMM at N 2624 × K 6144 a bijection for each grid of 1 to 699 programs; n = 7168 GEMM at m = 64 without its program-id permutation, the ruler (Kineto), 4 + 4 rounds, N=1: 6.262 → 6.262 µs, speedup ratio 1.001 (2048 programs × 4 warps, 8 readers per chunk; the two GEMMs named)
- E13 [2026-11, gfx950 (MI355X), HIP 7.15, N=2, one process, `HW_REG_XCC_ID` read back by every workgroup] default stream: XCC id = id % 8 for 1-D grids of 255–8192 workgroups and a 37 × 11 grid; on streams created in the process: (id + 7) % 8 with one created stream, 6 and 7 with two, 5, 6 and 7 with three, 5 on all four with four; a graph replayed on a stream took that stream's offset (one capture stream tested) (test kernel)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, Gluon test kernels run twice, against float64 (`mfma_scaled` error as a fraction of Σ|a·b| or of the largest output), values of the newer run] `mfma_scaled` on 16x16x128 tiles: real-scale e4m3 × e4m3 and e4m3 × e2m1 at `k_width` 16 within 1.387e-05 and 2.589e-08 of Σ|a·b|, all nine format pairs with real scales within 1.622e-05; A `k_width` 16 against B 32: 1.198 of the largest output; fp8 `k_width` 32 with real scales 0.795, against e2m1 with unit scales 1.12; one K = 256 call bitwise equal to two chained K = 128 calls in 64 of 64 trials at `k_width` 16 and 32. One MXFP4 16x16 tile, K = 512, random scales in 126..128: scales passed transposed ([16,16]) compile, relative error 0.8424; a linear scale layout with lane bits on the groups first 1.078; A's row-3 scales all 0xFF: output row 3 all NaN, the other rows exact; masked scale bytes 127 and words 0x7F7F7F7F exact; `scaled_upcast` to bf16 bitwise for all 16 codes with s in {120, 127, 130} constant along a row, 0.5625 of the elements right with s changing every two; whole scale words only reshaped 0.9401 (newer run); aiter `dynamic_mxfp4_quant(shuffle=True)` against a reference packing of the e8m0 scales padded to (256, 8) and shuffled, at (M, K) = (33, 256), (255, 288), (1, 64): valid bytes equal, of 5631 pad bytes 4735 zero and 11 0x7F; aiter `gemm_a4w4`, N = K = 256 at M = 1, 33, 255: 0xFF in the row-padding scale bytes changed no output. Random 64x64x1024 bf16 trials, each variant run 3 times: 16x16x32 against 32x32x16 at `k_width` 8 bitwise equal in 64 of 64; `k_width` 8 against 16 different in 64 of 64, at most 1.625 ulp of Σ|w||x|, each variant repeatable; bf16 × bf16 products in fp32's normal range equal to float64 bitwise, 16384 through a bf16 MFMA and 748591 on the VALU
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] DeepSeek-V2 MXFP4 MoE: lowest 1.030 at DS256, 128 tokens, highest 2.186 at DS256, 2 tokens, in two runs of the ruler (model dim 6144, 256 routed experts, aiter's 16 × 16 shuffled weights)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, the ruler with a separate compile cache per arm, 4 balanced AB/BA pairs per size (12 at N 3072 M 1 and M 16 and at N 7168 M 2), 95% t interval of the paired mean ratio] B is A with only the tile order changed: 12 sizes, 58 kernels compiled, no accuracy finding; mean B/A and intervals per size as tabled in [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §4; N 3072 M 256 B/A 0.97012 (6.0875 → 5.906 µs, interval 0.9689–0.9713); N 7168 M 64 B/A 0.99305 (0.9856–1.0005), inside ±1.5%; N 3072 M 1 1.00434 (0.9895–1.0196) and M 16 1.00803 (0.9953–1.0211), unresolved against ±1.5% and ±1.3%; N 7168 M 2 1.02350 (1.0101–1.0372); tile maps under `pid % 8` estimated on CPU, no L2 counter read (fp8 GEMM and DeepSeek-V2 MoE kernels)
- S1 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:516 a block stops at its own `return`; a constexpr `if` traces only its taken block (:995-996), and a `return` inside it moves tracing into a dead block, so the statements after the `if` are traced (:551-554); the byte-packing helper of DS256 at 2 tokens has `return packed` before its sign-OR lines ([triton/debugging](../triton/debugging.md) §3); a Qwen3-Next GDN kernel keeps its rank-2 and rank-3 delta rules apart because "the unified version traced both ranks and failed on the rank it was not given"
