---
title: "Qwen2-MoE block in Gluon, model dim 4096, 512 routed experts, top-10 — a split-K router of exactly 256 programs at 8–256 tokens with per-row top-10; fp8 experts with the routing given; router + experts in one call with fp64 logits rounded once"
kind: case
ops: [moe, fused_moe, grouped_gemm, gemm, topk]
dtypes: [bf16, fp8, fp64]
regimes: ["tokens 1-32 (router + experts)", "tokens 8-128 (experts)", "tokens 8-4096 (router)"]
accuracy: bounded
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup over what SGLang runs for the block, lowest and highest size per op; each kernel's value from one reference run of the ruler (the router's and the router + experts' in one run, the experts' two in two others)", value: "router 1.074 (8 tokens) to 1.967 (256); fp8 experts 1.076 (128) to 1.208 (8); router + experts 1.063 (32) to 1.172 (1)", gpu: MI355X, workload: "router at 8, 16, 64, 128, 256, 2232 and 4096 tokens; fp8 experts at 8, 64 and 128; router + experts at 1, 2, 4 and 32", date: "2026-11"}
---

# Qwen2-MoE block in Gluon: router, fp8 experts, router + experts (model dim 4096, 512 routed experts)

Fourteen Gluon kernels for three ops of one MoE block, a family of the Triton and Gluon kernel library that
[triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps. That page says which family to start
from; [triton/moe](../triton/moe.md) explains the techniques, and names kernels by the labels used below: Q2-R the
router, Q2-E the experts, Q2-RE the router + experts, then the token count (`Q2-RE/m1`). This page has the family's
kernels by size, their code, and what was measured. A kernel here is one host entry with the `@gluon.jit` functions
it launches; a route is one (token, expert) pair
([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1, §2).

The block:

Qwen3.5-397B-A17B's MoE block at TP4: model dim 4096, 512 routed experts, top-10; the
experts ops add inter dim 256, one shared expert weighted by `sigmoid(hidden · shared_gate_weight)` (513 experts in
all) and e4m3 experts with fp32 128 × 128 block scales. The router + experts op computes in one call what the router op
followed by the experts op computes.

## Applies when

- The op is one of the three above, at this block's shape: model dim 4096, 512 routed experts, top-10.
- Operands: the router takes the hidden states (bf16, tokens × 4096) and the router weight (bf16, 512 × 4096) and
  writes `topk_weights` (fp32) and `topk_ids` (int32), tokens × 10 each. The experts op takes the hidden states, those
  two outputs, `shared_gate_weight` (bf16, 1 × 4096) and the 513 experts' e4m3 weights (gate rows then up rows,
  512 × 4096 per expert; down, 4096 × 256) with their fp32 scales per 128 × 128 block, and writes `out` (bf16,
  tokens × 4096). The router + experts op takes the router's two inputs, `shared_gate_weight` and the same weights and
  scales, and writes `out`.
- The path replaced is SGLang's `Qwen2MoeSparseMoeBlock._forward_router_experts`; a speedup is against what SGLang
  runs for the block. The baseline reconstructs SGLang's path on aiter ([triton/moe](../triton/moe.md) §1): where the
  router weight is 512 × 4096 and SGLang's `smallm_router_supported(hidden_states)` accepts the call, its gfx950
  small-M router, which appends the gated shared expert itself, otherwise the gate, `fused_topk` and the append; then
  SGLang's gfx950 small-M MoE kernel where it accepts the call, otherwise aiter's `fused_moe`. All-to-all backends,
  EPLB and deterministic inference are not covered and stay on SGLang's path.
- A kernel is written for one token count: the router has seven, the experts three, the router + experts four
  (§ Mechanism). Another count has no kernel among these fourteen. The router + experts op later grew from 4 to 11
  kernels and from 4 to 24 token counts, 1–16384 tokens; none of the kernels added has a speedup measured on the
  ruler of § Result ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2).
- The family is held to `bounded` accuracy: each output stays within its error bound against the reference, and
  nothing is asked across runs. The router's `topk_ids` must match the reference's exactly, its `topk_weights` are
  held to a tolerance and `out` to an SQNR bound ([triton/moe](../triton/moe.md) §1). The routers at 8–256 tokens
  still gave bitwise identical outputs over 20 repeats *(field, E1)*.

## Counter-signals

- **The thinnest margins** are the router at 8 tokens (1.074), the experts at 128 (1.076) and the router + experts
  at 32 (1.063) *(field, E2)*.
- **At 2232 and 4096 tokens the router's design is read from the source, and its behaviour on ties and over
  repeated runs is untested.** There the design is a large-M GEMM with multi-row selection, and what the table says
  of it comes from the source; the run of 20 repeats with ties in every row covered the routers at 8–256 tokens and
  not these two *(field, E1)*. Both sizes were timed on the ruler (§ Result), a
  run of which also checks the outputs against the op's accuracy requirement ([triton/tuning](../triton/tuning.md)
  §1). Whether the split sums of the 8–256-token routers round once is untested too.
- **The router + experts kernels were written against the former float64 reference.** The fp32 one needs no fp64
  products at these token counts (§ Mechanism). The page describes and excerpts these kernels as measured; none of
  the kernels the op has since holds fp64 (read from their sources).

## Mechanism

| Op | Tokens, a kernel each | Launches | Decomposition |
|---|---|---|---|
| router: bf16 logits, top-10 ids and softmax weights | 8, 16, 64, 128, 256, 2232, 4096 | 2 | a split-K projection, exactly 256 programs at 8–256 tokens, then per-row selection over the stored bf16 logits: top-10 with lower-id ties, softmax weights within 2.3e-8 of those 10 logits, outputs bitwise identical over 20 repeats at 8–256 tokens *(field, E1)* (whether the split sums round once is untested); at 2232 and 4096 a large-M GEMM with multi-row selection (read from the source; these two were timed on the ruler, § Result, and were not in the run of 20 repeats with ties) |
| fp8 experts: routing given, 513 experts, aiter's 16 × 16 shuffled weights | 8, 64, 128 | 3; 5; 5 | routing given; 8 tokens: per-route split-K gate-up, activation quantization, down with the ordered 11-route sum; 64 and 128: a route sort in LDS (by one CTA at 64), then gate-up, quantization, down, ordered reduce (at 64: persistent stages over device-counted tasks) |
| fp8 router + experts, the same weights | 1, 2, 4, 32 (as measured, written against a float64 reference; with an fp32 reference these token counts need no fp64) | 4; 4; 4; 7 | router logits in fp64 (VALU), rounded once to bf16; every gate-up CTA replays the top-k of its row; activation quantization; one CTA per output tile multiplies all of a token's routes in one MFMA and keeps the diagonal; at 32 tokens fp64 MFMA router partials, a counting sort, sparse gate-up, quantization, grouped down, ordered sum |

- **Techniques.** One `@gluon.constexpr_function` gives the host its grid and the kernel its tile geometry
  ([triton/tuning](../triton/tuning.md) §14); `llvm_fn_attrs` on the router's launches ([triton/tuning](../triton/tuning.md) §9); the top-k replayed per CTA ([triton/moe](../triton/moe.md) §4); fp64 products for a router,
  written against a float64 reference, on the VALU at 1–4 tokens and on fp64 MFMAs at 32 ([triton/numerics](../triton/numerics.md)
  §5); a one-CTA counting sort with `atomic_scatter_add` on an LDS descriptor ([triton/moe](../triton/moe.md) §6).
- **Resources** *(field, E3)*: router 14 compiled kernels, `num_warps` 1, 4 or 16, VGPR 41–192, LDS 0–130 KiB, bf16 MFMA;
  experts 13 compiled kernels, 1–8 warps, `waves_per_eu` 2 or 4 in 3, VGPR 16–110, LDS 0–12 KiB, bf16 and fp16 MFMA; router +
  experts 19 compiled kernels, 1–4 warps, VGPR 19–128, LDS 0–16 KiB, scaled 16 × 16 × 128, fp64 16 × 16 × 4 and fp16 MFMA. No
  AGPRs, no spills.
  The compiled kernels counted here are compiled-cache records, one per compilation of a `@gluon.jit` function, so
  the same binary can be counted more than once: they are not the fourteen kernels of the table. VGPR is a compiled
  kernel's `.vgpr_count`, architectural VGPRs plus AGPRs
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1).

## Code

### 1. Round an fp64 logit to bf16 once (router + experts, 1 token)

Written against the former float64 reference (§ Counter-signals), the router + experts kernels as measured compute
the logits as float64 products and round them once to bf16, that reference's rounding point. Casting float64 to fp32
and then to bf16 rounds twice: on 512 crafted near-midpoint logits that erred on 260, this helper on none
*(field, E4)*.

```python
# Q2-RE/m1: float64 value -> bf16; an fp32 halfway between two bf16 values first moves one fp32 step toward the float64 value
@gluon.jit
def _round_to_bf16(value):
    approximate = value.to(gl.float32)
    bits = approximate.to(gl.int32, bitcast=True)
    midpoint = (bits & 65535) == 32768
    inexact = value != approximate.to(gl.float64)
    toward_positive = value > approximate.to(gl.float64)
    step = gl.where(toward_positive == (approximate > 0), 1, -1)
    corrected = (bits + step).to(gl.float32, bitcast=True)
    return gl.where(midpoint & inexact, corrected, approximate).to(gl.bfloat16)
```

Not this: the float64 value cast to fp32 and then to bf16, the double rounding above.

Explained in [triton/moe](../triton/moe.md) §5.1.

### 2. An integer key that orders by score, then by the lower id (router + experts, 1 token)

In the 1-token kernel as measured, the top-10 takes one `gl.max` per round over keys built from the bf16 logit's
bits and the expert ids: the score's
bits made monotonic, −0 folded into 0, and the inverted expert id in the 9 low bits, so the maximum key is the
highest score and, on a tie, the lowest id.

```python
    bits = logits.to(gl.uint16, bitcast=True).to(gl.uint32)
    bits = gl.where((bits & 32767) == 0, 0, bits)  # -0 folded into 0
    ordered = gl.where((bits & 32768) != 0, 65535 - bits, bits | 32768)  # negative scores below positive ones
    return (ordered << 9) | (511 - experts).to(gl.uint32)  # the inverted id below the score: the lower id wins a tie
```

Explained in [triton/moe](../triton/moe.md) §5.3.

### 3. One function gives the host its grid and the kernel its geometry (router, 256 tokens)

The split, expert tile, K tile, look-ahead and routing group come from one `@gluon.constexpr_function` of the row
count. The kernel reads it as `config: gl.constexpr = _shape_config(M)`; the host calls `_shape_config(rows)` and
sizes the grid from `config.split` and `config.expert_tile`, so both read the same split and expert tile. Its arms
for 64 and 128 rows are branches this kernel keeps and does not take at its one size
([triton/moe](../triton/moe.md) §12). The 64- and 128-token routers are other kernels with no such function: each
writes its row count's values as literals and conditionals on the row count.

```python
# fields of _ShapeConfig: split, expert_tile, k_tile, lookahead, routing_group
@gluon.constexpr_function
def _shape_config(rows):
    """One source of truth for host launches and compile-time kernel geometry."""
    if rows == 64:
        return _ShapeConfig(8, 32, 128, 1, 16)
    if rows == 128:
        return _ShapeConfig(4, 32, 128, 2, 4)
    return _ShapeConfig(4, 64, 64, 2, 8)
```

Explained in [triton/tuning](../triton/tuning.md) §3.

### 4. The async-copy layout read off the padded LDS layout (router, 2232 tokens)

A direct-to-LDS copy writes a wave's lanes one after another, 16 bytes per lane, so the copy layout has to hand
each lane the bytes that land at its slot. The 2232-token router takes the padded destination layout and reads the
copy layout off its `offset_bases`: the first bases are one lane's 16 bytes (`bases[:3]`, 8 bf16 elements), the
next 6 the 64 lanes, the next log2(`num_warps`) the waves, the rest registers.

```python
@constexpr_function
def _copy_layout(shared, num_warps):
    """Map 128-bit async copies onto the operand's padded LDS layout."""
    bases = shared.offset_bases
    wave_end = 9 + num_warps.bit_length() - 1
    return gl.DistributedLinearLayout(
        bases[:3] + bases[wave_end:], bases[3:9], bases[9:wave_end], [], shared.shape,
    )
```

Explained in [triton/gluon/layouts](../triton/gluon/layouts.md) §6.4.

### 5. An LDS ring refilled at the top of the step (router, 256 tokens)

The ring issues the refill of stage `stage + AHEAD` at the top of the step, before the wait and the barrier, so the
slot refilled must not be one a wave still reads: with `AHEAD` 2 the ring holds `DEPTH = AHEAD + 2` slots. The
reads are relaxed (`load_shared_relaxed`), so the barriers between a slot's last read and its refill are the ones
the kernel writes. `_prefetch_stage` issues a stage's two async copies, one per operand, and commits them as one
group; `STAGES` is the number of K tiles in the program's K split, `a_buffers` and `b_buffers` are the ring's slots
for the two operands, and `a_dot` is the MFMA's layout for its A operand.

```python
for stage in gl.static_range(-AHEAD, STAGES):
    if stage + AHEAD < STAGES:
        _prefetch_stage(
            a_buffers[(stage + AHEAD) % DEPTH],
            b_buffers[(stage + AHEAD) % DEPTH],
            # ... (the two tensors and their offsets for stage + AHEAD)
        )
    if stage >= 0:
        gl.amd.cdna4.async_copy.wait_group(min(AHEAD, STAGES - stage - 1))
        gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffers[stage % DEPTH], a_dot)
```

Not this: `DEPTH = AHEAD + 1` with the refill at the top of the step and no `gl.barrier()` before it. On synthetic
four-wave rings with relaxed reads, 2 groups ahead in 3 slots gave wrong sums in every launch; 4 slots, or 3 plus a
barrier before the refill, gave none *(field, E5)*. The 128-token router's source keeps such a
barrier for a 3-slot ring, under `if DEPTH <= AHEAD + 1 and stage >= DEPTH - AHEAD:`; at its own size the ring has
4 slots and the branch is not compiled.

Explained in [triton/gluon/memory](../triton/gluon/memory.md) §12.

## Result

- **Speedups** over what SGLang runs for the block *(field, E2)*: router 1.074 (8 tokens) to 1.967 (256); experts 1.076 (128) to 1.208 (8); router +
  experts 1.063 (32) to 1.172 (1).

Per size, each kernel's value from one run of the ruler; a dash where the op has no kernel at that count:

| Tokens | Router *(field, E6)* | fp8 experts *(field, E6)* | Router + experts *(field, E6)* |
|---|---|---|---|
| 1 | — | — | 1.172 |
| 2 | — | — | 1.145 |
| 4 | — | — | 1.119 |
| 8 | 1.074 | 1.208 | — |
| 16 | 1.219 | — | — |
| 32 | — | — | 1.063 |
| 64 | 1.295 | 1.171 | — |
| 128 | 1.417 | 1.076 | — |
| 256 | 1.967 | — | — |
| 2232 | 1.416 | — | — |
| 4096 | 1.226 | — | — |

A later run of the ruler re-measured the router at 16 tokens at 1.304 (+7.0%), the largest rise among the sizes it
re-measured; over repeat runs of the ruler on unchanged sources the ratios at one size differ by 2.03% at the median
and by at most 54.8%, at the router + experts' 1 token *(field, E6)*.

What moving the 256-token router's split-K and look-ahead did to its speedup: [triton/tuning](../triton/tuning.md) §3.
The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the library's routers run 20 times on equal inputs with ties in every row] Qwen2-MoE routers at 8–256 tokens (five kernels): projection grid 256 programs; outputs bitwise identical over the 20 repeats; ids the top-10 of the stored bf16 logits, lower id first on ties; softmax weights within 2.3e-8 of those 10 logits; 2232 and 4096 tokens and once-rounded split sums not tested
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] Qwen2-MoE: router 1.074 at 8 tokens and 1.967 at 256, in one run; fp8 experts 1.076 at 128 and 1.208 at 8, in two other runs; router + experts 1.063 at 32 and 1.172 at 1, in the router's run (model dim 4096, 512 routed experts, top-10)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the toy MoE N=4), test kernels and copies of the implementations' code on one GPU] DeepSeek's reference with both GEMMs' activations MXFP4-quantized moves `out` by rel L2 0.2103, max abs 31.19; a MM256-FP4/m64-form copy at 32, 128 and 256 tokens puts each route in a job of up to 16 rows of one expert, and over 20 runs at 256 tokens the tickets differ in 19 while the slot-order combine keeps one bit pattern (a float `atomic_add` in arrival order gives 20); 1000 programs of 8 ids in a 64-lane layout leave every counter at 1000; 512 crafted near-midpoint logits: f64 → f32 → bf16 wrong on 260, with the midpoint fix on 0; those 512 and 512 random values through Q2-RE/m1's helper alone: f64 → f32 → bf16 wrong on 260 of the 1024, the helper on 0; an MM64-FP4/m1024 copy storing only the rows with a local route: 239616 non-finite mismatches in `out`; a MM256-FP4/m64 copy without its fill of `Sorted`: the copy crashes, exit −6; under `launch_cooperative_grid` a counter barrier over 256 workgroups of 1024 threads finished in 0.2 ms, every program seeing every store; 4096 workgroups were refused (`hipErrorCooperativeLaunchTooLarge`); on a 2.5 GiB tensor `buffer_load` read the data at byte 2^31 − 64 and 0 at 2^31 + 64 with no fault, `gl.load` both; a toy decode MoE (32 experts, top-4, H 256, I 128, every gate-up program owning full K, no requantization) at 2 and 16 tokens against no fusion, 4 runs: each fusion alone, the other two off, saves on the ruler's timer (Kineto) top-k recomputed in the gate-up, down and combine programs 3.30–4.19 µs, SwiGLU in the gate-up epilogue 4.04–4.64 µs, down and combine in one program 4.19–4.43 µs (their replay times not recorded); the three together give call-time ratios of 0.667–0.751 replayed and 0.426–0.462 on the ruler's timer, rel L2 against no fusion 4.5e-8 and 5.1e-8; in the same runs, each taken as the last of the three with the other two on in both arms, top-k 2.89–3.87 µs and down and combine 2.94–3.73 µs (test kernels)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, two GPU runs of synthetic one- and four-wave rings with relaxed reads and a single buffer; 2048 programs, 200 launches per delay setting, 4 settings per run (one wave: 0, 20, 39, 39 dependent loads; four waves: spin delays 0, 1, 4, 16 µs)] one-wave ring refilling the just-read slot without a barrier: 2.8–5.1 % of programs read refilled data in every launch, zero wrong in 1600 launches with it; four waves, 2 groups ahead in 3 slots: wrong sums in every launch, a pinned variant 48–89 % of outputs wrong, zero wrong with 4 slots or 3 plus a barrier; single buffer without the barrier: wrong in all 200 launches at each delay (38–72 % of outputs), zero wrong in 1600 with it; the n3072 m = 8192 kernel without that barrier passed its accuracy check in six runs
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (graph replay, 1 GiB read flush, Kineto), speedup over the baseline of SGLang's path on aiter] speedup per size, the implementation in parentheses: DeepSeek-V2/GLM DS256: 1 token 2.139 (DS256/m1), 2 2.186 (DS256/m2), 4 1.867 (DS256/m4), 8 1.521 and 16 1.245 (DS256/m8–m16), 32 1.086 (DS256/m32), 128 1.030 (DS256/m128), 256 1.160 (DS256/m256), 1024 1.402, 2048 1.329 and 3072 1.407 (DS256/m1024–m3072), 4096 1.420 and 4192 1.378 (DS256/m4096–m4192); DS512: 1 token 2.093 and 2 1.829 (DS512/m1–m2), 4 1.324 (DS512/m4), 8 1.184 (DS512/m8), 16 1.097 (DS512/m16), 256 1.043 (DS512/m256), 1024 1.152 (DS512/m1024), 2048 1.391 (DS512/m2048), 3072 1.235 (DS512/m3072), 4096 1.161 (DS512/m4096), 4192 1.166 (DS512/m4192); MiniMax-M2 FP8, 256 experts: 1, 2, 4, 8, 16 tokens 4.115, 3.755, 2.452, 1.841, 1.732 (MM256-FP8/m1–m16); 32 1.158 and 64 1.173 (MM256-FP8/m32–m64); 128 1.106 (MM256-FP8/m128); 256 1.133 (MM256-FP8/m256); 1024–16384 1.257, 1.248, 1.348, 1.372, 1.388 (MM256-FP8/m1024–m16384); FP8, 64 of 256 experts (expert_offset 192): 1, 2, 4, 8, 16 tokens 3.137, 2.912, 1.672, 1.466, 1.325 (MM64-FP8/m1–m16); 32 1.167 and 64 1.102 (MM64-FP8/m32–m64); 128 1.146 and 256 1.164 (MM64-FP8/m128–m256); 1024–16384 1.216, 1.485, 1.681, 1.770, 1.785 (MM64-FP8/m1024–m16384); MXFP4, 256 experts: 1 token 3.444 (MM256-FP4/m1), 2 3.309 (MM256-FP4/m2), 4, 8, 16 2.549, 1.977, 1.595 (MM256-FP4/m4–m16), 32 1.179 (MM256-FP4/m32), 64 1.230 (MM256-FP4/m64), 128 1.210 (MM256-FP4/m128), 256 1.178 (MM256-FP4/m256), 1024 1.352 (MM256-FP4/m1024), 2048 1.272 and 4096 1.261 (MM256-FP4/m2048–m4096), 6144 1.415 (MM256-FP4/m6144), 8192 1.423 (MM256-FP4/m8192), 16384 1.389 (MM256-FP4/m16384); MXFP4, 64 of 256 experts (expert_offset 192): 1 token 3.633 and 2 2.797 (MM64-FP4/m1–m2), 4 2.554 (MM64-FP4/m4), 8 2.113 and 16 1.555 (MM64-FP4/m8–m16), 32 1.231 and 64 1.275 (MM64-FP4/m32–m64), 128 1.190 and 256 1.194 (MM64-FP4/m128–m256), 1024 1.252 (MM64-FP4/m1024), 2048 1.318 (MM64-FP4/m2048), 4096 1.590 (MM64-FP4/m4096), 6144 1.689 (MM64-FP4/m6144), 8192 1.735 (MM64-FP4/m8192), 16384 1.619 (MM64-FP4/m16384); Qwen2-MoE router+experts: 1 token 1.172 (Q2-RE/m1), 2 1.145 (Q2-RE/m2), 4 1.119 (Q2-RE/m4), 32 1.063 (Q2-RE/m32); router: 8 tokens 1.074 (Q2-R/m8), 16 1.219 (Q2-R/m16), 64 1.295 (Q2-R/m64), 128 1.417 (Q2-R/m128), 256 1.967 (Q2-R/m256), 2232 1.416 (Q2-R/m2232), 4096 1.226 (Q2-R/m4096); fp8 experts: 8 tokens 1.208 (Q2-E/m8), 64 1.171 (Q2-E/m64), 128 1.076 (Q2-E/m128); the values come from eight runs of the ruler, each implementation's from one run; a later run re-measured 57 of the 96 sizes, by −6.3% (MM256-FP8 at 1 token, 4.115 → 3.855) to +7.0% (Q2-R at 16 tokens, 1.219 → 1.304), median change 0.9%; over repeat runs of the ruler on unchanged sources, the ratios at the same size differ by 2.03 % at the median, 6.2 % at p90 and by more than 5 % in 16 % of 1716 pairs at 170 sizes, at most 54.8 % (Q2-RE at 1 token)
