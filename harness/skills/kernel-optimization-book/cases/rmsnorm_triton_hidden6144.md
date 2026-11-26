---
title: "RMSNorm and add + RMSNorm in plain Triton, hidden size 6144 — the row as three 2048-wide chunks in registers, one program per row, `.cg` loads and `.wt` stores; from 1024 rows each RMSNorm program takes m // 512 rows with the next row's loads pipelined"
kind: case
ops: [norm]
dtypes: [bf16]
regimes: ["tokens 3-4096 (RMSNorm)", "tokens 4-256 (add + RMSNorm)"]
accuracy: bitwise
gens: [gfx950]
triton: 3.8.0
basis: measured N=1
measured:
  - {metric: "speedup of RMSNorm over aiter's `rmsnorm2d_fwd`, per size, one reference run of the ruler", value: "1.006 (4 tokens) to 1.212 (2048); 1.006–1.024 at 3–64 tokens, below the bar of these runs", gpu: MI355X, workload: "hidden size 6144, 14 token counts from 3 to 4096", date: "2026-11"}
  - {metric: "speedup of add + RMSNorm over aiter's `rmsnorm2d_fwd_with_add`, per size, one reference run of the ruler", value: "1.024 (8 tokens, below the bar of these runs) to 1.093 (256)", gpu: MI355X, workload: "hidden size 6144, 7 token counts from 4 to 256", date: "2026-11"}
---

# RMSNorm and add + RMSNorm in plain Triton (hidden size 6144)

Two plain `@triton.jit` kernels at hidden size 6144, one for RMSNorm and one for add + RMSNorm, a family of the
Triton and Gluon kernel library that [triton/kernels/kernel_library](../triton/kernels/kernel_library.md) maps; with
the two GEMMs of [cases/bf16_triton_small_m](bf16_triton_small_m.md) they are the library's four `@triton.jit`
kernels. Why plain Triton fits them, and what its compiler adds (coalescing, `num_stages` pipelining, buffer
instructions): [triton/authoring](../triton/authoring.md) §2. This page has the kernels by size, the RMSNorm
kernel's row loop, and what was measured. RMSNorm measured below the bar at 3–64 tokens, and for add + RMSNorm a
HIP kernel measured better at 32 and 128 tokens (§ Counter-signals).

## Applies when

- The op is RMSNorm over rows of 6144 values, `normalized = x · rsqrt(mean(x²) + epsilon) · weight` with the mean
  over the row, or add + RMSNorm at the same hidden size: the residual add and the norm in one call, with the sum
  and the normalized rows as outputs ([hip/memory_bound](../hip/memory_bound.md) describes that op for a HIP
  kernel). The RMSNorm kernel takes bf16 `x` and `weight` and a runtime `epsilon`, and writes bf16.
- The paths replaced are aiter's `rmsnorm2d_fwd` for RMSNorm and `rmsnorm2d_fwd_with_add` for add + RMSNorm; a
  speedup is against them.
- The token count is one of the kernel's sizes: 3, 4, 6, 8, 12, 16, 18, 64, 128, 256, 512, 1024, 2048 or 4096 for
  RMSNorm, and 4, 8, 16, 32, 64, 128 or 256 for add + RMSNorm. The library has norms at hidden size 6144 only.
- The family is held to `bitwise` accuracy: every run on the same inputs gives the same bytes as the first, and the
  output's error bound, a tolerance of atol = rtol = 1e-2 for a bf16 output, holds besides
  ([triton/numerics](../triton/numerics.md) §1). The kernels use no atomic, and each repeated bit for bit at its
  first size ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2).

## Counter-signals

- **RMSNorm measured below the bar at 3–64 tokens.** It read 1.006–1.024 there, below the bar of these runs
  ([triton/tuning](../triton/tuning.md) §1); a later run of the ruler read 1.023–1.029 at 3–18 tokens, below the
  bar again, and 1.040 at 64 *(field, E1)*. A size counts on its own speedup, not
  on the mean over sizes ([triton/authoring](../triton/authoring.md) §11).
- **add + RMSNorm has thin margins too.** Its lowest size, 1.024 at 8 tokens, is below the bar
  *(field, E1)*, and its 4-token size stored exactly 1.030 in the last of nine
  reference runs *(field, E2)*.
- **At 32 and 128 tokens a HIP kernel measured better for add + RMSNorm.** The add + RMSNorm shape also has a HIP
  kernel, and at each size the kernel to beat is the best measured one of any language. The HIP kernel measured
  1.038 and 1.067 at 32 and 128 tokens against the Triton kernel's 1.035 and 1.062, so it is the better one there,
  and it alone covers 512–8192 tokens *(field, E3)*. A new kernel at a shared
  size has to beat the best one already there. The HIP kernel's sizes carry another epsilon
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §1); what moved a HIP add + RMSNorm at this
  hidden size: [hip/memory_bound](../hip/memory_bound.md).
- **One launch on a timer that reads about 4 µs for a trivial kernel.** The call times of add + RMSNorm and of its
  baseline on the ruler's timer (Kineto) are in § Result; the same runs measured a slope of 4.029–4.066 µs per
  added trivial kernel *(field, E4)*. Read both sides' times, not only their ratio
  ([triton/tuning](../triton/tuning.md) §1, [pitfalls/profiled_timer_floor](../pitfalls/profiled_timer_floor.md)).
- **In HIP the same store bits paid behind a flush and lost without one.** The stores of these kernels carry
  `sc0 sc1`, the bits `.wt` sets *(field, E5)*
  ([triton/gluon/memory](../triton/gluon/memory.md) §6). On a HIP add + RMSNorm at hidden 6144 those bits on both
  output stores carried nearly all of that kernel's gain over aiter ([hip/memory_bound](../hip/memory_bound.md) §1),
  but without a flush before each call, at 16384 tokens and hidden 6144 and 8192, they took 10–11% longer than
  aiter's store bits (1.096× at hidden 6144) *(field, E6)*.
- **`epsilon` has to stay a runtime argument.** A copy of the RMSNorm kernel with `epsilon` hard-coded to 1e-5
  passed every check at its sizes, yet called with `epsilon` 1e-3 on inputs of scale 0.01 it was off by up to 224%,
  the original kernel exact *(field, E7)* ([triton/authoring](../triton/authoring.md) §4).
- **The host has arms no size of the kernel takes.** The RMSNorm host picks the chunk width, the rows per program
  and the warp count from the token count and the hidden size: it has arms for hidden sizes up to 1024 and up to
  4096, and one padded chunk for a hidden size that is not three times a power of two. Hidden size 6144 takes none
  of them.

## Mechanism

- **RMSNorm**: one kernel for RMSNorm (3–4096 tokens) and one for add + RMSNorm (4–256
  tokens), hidden size 6144. The 6144-wide row as three 2048-wide chunks in registers, the sum of squares in fp32, `.cg` loads, `.wt`
  stores, one program per row *(field, E5)*; RMSNorm from 1024 rows gives each program `m // 512` rows, the next
  row's loads pipelined and the weight loaded once:
  entry 1 of § Code.

| Kernel | Tokens | Rows per program | Programs | Row loop |
|---|---|---|---|---|
| RMSNorm | 3, 4, 6, 8, 12, 16, 18, 64, 128, 256, 512 | 1 | one per row | one trip |
| | 1024, 2048, 4096 | 2, 4, 8 (`m // 512`) | 512 | `num_stages` 2: the next row's loads pipelined |
| add + RMSNorm | 7 counts from 4 to 256 | 1 | one per row | — |

- **One launch per call, every size from one source.** The RMSNorm host branches on the token count: one row per
  program below 1024 rows, grid-stride from 1024 rows
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §2, §3).
- **Three power-of-two chunks, not one padded chunk.** A row of 6144 is held as three 2048-wide chunks. Padded to
  one 8192-wide chunk, the RMSNorm kernel stayed accurate and took 1.022–1.037× the split time at 1024–4096 tokens
  and 0.988–1.023× at 3–512, faster in one of 3 runs *(field, E8)*; a minimal test RMSNorm
  with one row per program, at hidden 6144 and 4096 tokens, took 0.94× its split time when padded
  *(field, E9)*: time both ([triton/gluon/layouts](../triton/gluon/layouts.md) §2).
- **fp32 arithmetic, no matrix instruction.** In the RMSNorm kernel each chunk is converted to fp32 as it is loaded;
  the squares of the three chunks are added and summed, divided by the hidden size, and `rsqrt` of that plus
  `epsilon` scales the row, which is multiplied by the weight and rounded to the output dtype at the store. The
  add + RMSNorm kernel adds `residual` to each fp32 chunk, stores that sum rounded to bf16 as `residual_out`, and
  normalizes the unrounded fp32 sum. Neither kernel uses a matrix instruction
  ([triton/kernels/kernel_library](../triton/kernels/kernel_library.md) §4).
- **Resources** *(field, E5)*: 6 compiled kernels, 4 warps, VGPR 49–79, 16 B of
  LDS. They load with `buffer_load_dwordx4`, no `global_load`.

## Code

### 1. A grid-stride row loop with the next row's loads pipelined (RMSNorm, 1024–4096 tokens)

From the RMSNorm kernel's one kernel function. Below 1024 rows the host launches one program per row, with `LOOP`
and `NS` 1; from 1024 rows it launches `cdiv(m, LOOP)` programs of `LOOP = m // 512` rows each (512 programs at
1024, 2048 and 4096 tokens) with `NS` 2, the `num_stages` of the loop, which pipelines the next row's loads. A
program takes rows `program_id`, `program_id + num_programs` and so on, `rmask` drops a row past `M`, and the three
weight chunks are loaded once, before the loop. `cols` is `tl.arange(0, CHUNK)` with `CHUNK` 2048, `cmask` is
`cols < H` with `H` the hidden size, and `x0` is the first of the row's three chunks, converted to fp32 as it is
loaded; the loads of the other two follow the excerpt.

```python
for it in tl.range(0, LOOP, num_stages=NS):
    row = tl.program_id(0) + it * tl.num_programs(0)
    rmask = cmask & (row < M)
    base = row.to(tl.int64) * H + cols
    x0 = tl.load(x_ptr + base, mask=rmask, other=0.0, cache_modifier=".cg").to(tl.float32)
```

Not this: that loop with fewer stages. It slowed RMSNorm at 1024, 2048 and 4096 tokens by 1.047–1.120×,
1.092–1.095× and 1.051–1.052× (one run, 4 rounds per arm in a fixed order, which may bias the smaller effects; add +
RMSNorm not tested) *(field, E10)*.

Explained in [triton/authoring](../triton/authoring.md) §2.

## Result

- **Speedups** over aiter *(field, E1)*: RMSNorm 1.006 (4 tokens) to 1.212
  (2048), below the bar of these runs ([triton/tuning](../triton/tuning.md) §1) at 3–64 tokens (a later run of the
  ruler read 1.023–1.029 at 3–18 tokens and 1.040 at 64); add + RMSNorm 1.024 (8 tokens, also below the bar) to
  1.093 (256).
- **RMSNorm by size range in another reference run of the ruler** *(field, E11)*:
  1.023–1.029 at its 7 sizes of 3–18 tokens and 1.040–1.214 at 64–4096 tokens.
- **Call time**, the ruler's timer (Kineto), 3 runs *(field, E4)*: add + RMSNorm 4.07–4.24 µs
  against its baseline's 4.19–4.72 µs at its 7 sizes.
- **Run to run**, the median gap between two reference runs of a size is 0.9% for the plain-Triton GEMMs and
  RMSNorm *(field, E2)*.

The ruler, its timer and the bar a size is counted against: [triton/tuning](../triton/tuning.md) §1.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] RMSNorm 1.006 at 4 tokens to 1.212 at 2048, 1.006–1.024 at 3–64 tokens; add + RMSNorm 1.024 at 8 tokens to 1.093 at 256, in another run; a later run of the ruler: RMSNorm 1.023–1.029 at 3–18 tokens and 1.040 at 64 (hidden size 6144)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 to 8 reference runs per size, reference runs of the ruler (Kineto), stored speedup against the op's baseline] the absolute log ratio between any two stored values of the same sources at the same size: median 2.2%, p90 6.3%, p95 8.1%, 17.5% of pairs above 5% (1314 pairs over the 122 Triton/Gluon sizes with two or more runs); per-family median 0.9% plain-Triton GEMM and RMSNorm, 2.0% Qwen3.5 GDN prefill, 2.3% DeepSeek-V2, 2.4% MiniMax-M2 FP8, 2.6% MiniMax-M2 MXFP4, 2.7% Qwen3.5 GDN decode, 3.1% fp8 GEMM, 3.7% Qwen2-MoE; 9 sizes stored values on both sides of 1.03, and 3 stored exactly 1.030 in the last run (add + RMSNorm at 4 tokens, DSA with 64 heads at 8, DS256 at 128); the 12 fp8 GEMM sizes at m ≤ 64 with two or more runs moved 4.0–11.8% (max ÷ min over 3–7 runs) (nine reference runs)
- E3 [2026-11, gfx950 (MI355X), FlyDSL ? and HIP kernels beside Triton 3.8.0 ones, N=1, a reference run of the ruler, stored speedup against the op's baseline] one run: the FlyDSL bf16 GEMM N 6144 × K 2048 M 1 1.191, M 2, 4, 8, 16, 32, 64 1.045, 0.403, 0.445, 0.392, 0.361, 0.360, M 128–4096 0.450–0.864; the FlyDSL bf16 GEMM N 2624 × K 6144 M 1 1.134, M 2, 4, 8, 16, 32, 64 0.420, 0.415, 0.419, 0.396, 0.410, 0.553, M 128–4096 0.623–0.998; the HIP add + RMSNorm (hidden size 6144) 1.038 at 32 tokens, 1.067 at 128, 1.040–1.072 at 512–8192; the Triton add + RMSNorm 1.035 at 32, 1.062 at 128
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler (Kineto) on one GPU, per-run median call times; 4 runs of the ruler per kernel in each run, 2 for add + RMSNorm and for each size of the N 6144 × K 2048 bf16 GEMM] add + RMSNorm at its 7 sizes, 4 to 256 tokens: candidate 4.07–4.24, baseline 4.19–4.72 µs; fp8 GEMM N=3072 m = 16 (1 launch) 4.21–4.33 µs; Qwen3.5 GDN decode 2 tokens (2 launches, baseline 5) 8.55–8.87 against 21.20–21.78 µs, ratio 2.44–2.51, mean speedup 2.470–2.474 against 5/2; Qwen2 router 8 tokens (2 launches) 8.33–8.51 µs; the floor slope measured beside them 4.029–4.066 µs; per-launch excess of the run means 0.181–0.237 (GEMM), 0.273–0.345 (GDN), 0.140–0.171 (router) µs, single runs up to 0.41; DS256, 2 tokens (4 launches) 19.00–19.17 µs (runs 18.85–19.34), 2.79–3.09 µs above 4 floors; the N 6144 × K 2048 bf16 GEMM (25.17 MB of weight) 5.94–5.98, 5.95–6.00, 6.04–6.09, 6.34–6.37 µs at m = 2, 4, 8, 16, and the N=7168 fp8 GEMM at m = 64 6.24–6.36 µs for 15.74 MB moved, 2.48–2.52 TB/s
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, every kernel built and run once, each with its own `TRITON_CACHE_DIR`, on 4 GPUs; the cached `.amdgcn` and `.json` read] `num_warps`, `waves_per_eu`, VGPR (`.vgpr_count`, AGPRs included), AGPR, LDS, spills, MFMA opcodes and the count of every opcode (loads, stores and atomics included, buffer loads into LDS apart) of every kernel function of the library (a `@gluon.jit` or `@triton.jit` def of a kernel's sources) compiled while running each of the library's 154 kernels once: 897 compiled-cache records among 1709 compiled kernels, one per compilation, not deduplicated; N 3072 fp8 GEMM LDS 9, 9, 12, 36, 48, 49 KiB at M 1, 2, 4, 8, 16, 32; N 7168 at M 16, 32, 64 64.0, 66.1, 72.0 KiB; MM64-FP4 at 2048, 4096, 6144, 8192, 16384 tokens: 0 atomic instructions (global, buffer, flat, ds), popcounts (`s_bcnt`) 4, 2, 4, 8, 8; every GDN prefill kernel (26) holds `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_16x16x4_f32`; 45 GDN decode compiled kernels, 0 MFMA; RMSNorm loads with the nt bit, stores sc0 sc1 only, 16 B of LDS (all 154 Triton kernels)
- E6 [2026-11, gfx950 (MI355X), HIP 7.15 (hipcc), N=5, per-kernel CUDA-graph timer (1 GiB read before each call), also with a kernel reading the outputs after each call, and as a 40-call graph back to back; arms differing only in the stores' cache bits] time with `sc0 sc1` over time with no cache bits, flushed: add + RMSNorm hidden 2048 0.903–0.979 and hidden 8192 0.946–0.972 at 16 / 1024 / 16384 tokens; bf16 add 0.963 / 0.953 / 1.002 at 2 / 32 / 256 MiB per tensor; MXFP8 quantization, hidden 7168, 1.427 / 1.713 at 1024 / 16384 tokens. Over AITER's bits at 16384 tokens, hidden 6144 / 8192: back to back 1.096 / 1.114, outputs read next 1.107 / 1.061 (the HIP add + RMSNorm of [hip/memory_bound](../hip/memory_bound.md), a bf16 add and an MXFP8 quantization in HIP)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; planted copies on one GPU checked against the op's reference in captured graphs with poisoned outputs (a first capture and two replays with new inputs, fresh captures per input set, a repeat, a check after the timing) and timed by the ruler, and direct calls against the op's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the original kernel 0; a copy of an fp8 GEMM (N 3072, m = 16) storing 8 of its 16 rows: wrong values at the first capture with 24576 = 8 × 3072 non-finite mismatches, the original none (one run); the poison filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; a `run` keeping its first output in a module-level dict passed a source scan and gave wrong values at the first capture, the fresh captures, the repeat and the check after the timing; a math error: wrong values at the captures and after the timing; a changed reduction order: the same in one run, only a slower time in the other; state kept across calls: wrong values after the timing only; no difference between repeated runs in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / the ruler's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (copies of a plain-Triton RMSNorm at hidden 6144 and of an fp8 GEMM, probe kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, the ruler (Kineto), one GPU, 4 alternated rounds per arm] plain-Triton RMSNorm at hidden 6144 with the row padded to one 8192 chunk instead of three 2048 chunks: accurate at all 14 token counts in every run; padded over split kernel time in the three runs 1.024–1.037, 1.022–1.033, 1.028–1.036 at 1024–4096 tokens; 0.988–0.992, 1.010–1.016, 1.007–1.011 at 3–128; 0.998–1.005, 1.021–1.023, 1.016–1.019 at 256–512 (the library's RMSNorm kernel for hidden 6144)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one process, CUDA events after a 1 GiB read flush, median of 36 readings per arm over 3 passes of 12 rotated rounds] a minimal one-row-per-program RMSNorm (bf16, 4 warps) at hidden 6144 and 4096 tokens, the row as one power-of-two chunk with a mask against three chunks: pad over split 0.9384, pad slower in 2 of 36 rounds; outputs within 1 bf16 ulp of an fp64 reference (a test kernel)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU: `num_stages` A/B 4 runs per arm, the Gluon bf16 GEMM skeleton of [triton/authoring](../triton/authoring.md) §10 against the plain-Triton N 6144 × K 2048 GEMM and the skeleton with one-step K prefetch 3 runs per arm in a fixed order; Gluon fp8 GEMMs against aiter's plain-Triton kernel in 5 flushed graph rounds in one process; the skeleton's direct calls against the op's reference, 20 eager calls, 20 replays and a second process per input] fewer stages: the N 6144 × K 2048 bf16 GEMM at m = 2, 4, 8, 16 1.533–1.625× slower, RMSNorm at 1024, 2048, 4096 tokens 1.047–1.120×, 1.092–1.095×, 1.051–1.052× slower; unshuffled E4M3 fp8 GEMMs with fp32 block scales (N 3072, K 1536) at m = 2, 8, 16 0.777–0.781×, 0.755–0.764×, 0.759–0.768× of aiter's time, both within bounds; the skeleton at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64 with one positive and one signed input each: bitwise stable, 0–1 bf16 ulp on the positive input, 0.960–0.984 of the bound on the signed one, 8192 B of `metadata.shared` passed at every launch while the amdgcn's group segment read 0; checked and timed at m = 2, 4, 8, 16, 32, 64: no correctness, build or capture failure, all six sizes in each of 3 runs at speedup 0.373–0.434 over the baseline; 2.882–3.158× the time of the plain-Triton N 6144 × K 2048 GEMM at m = 2, 4, 8, 16, and 1.695–1.729× faster with one-step K prefetch
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] a plain-Triton RMSNorm at hidden 6144: 1.023–1.029 at its 7 sizes of 3–18 tokens and 1.040–1.214 at 64–4096 tokens
