---
title: "bf16 GEMM, narrow N (32–256) × K 6144 — whole dot products per wave or workgroup at M ≤ 8, split-K with a fixed-order fold at M ≥ 2048; N 64 × K 4096 at m 4–128 with the whole K in one workgroup"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 2-8", "M 16-1024", "M 2048-4096", "M 4-128 (N 64 × K 4096)"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.565–1.195; 2 of 13 Points ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "gemm_n128_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.490–1.201; 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096)", gpu: MI355X, workload: "gemm_n256_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.694–1.841; 3 of 13 Points ≥ 1.03 (m 4, 8, 2048)", gpu: MI355X, workload: "gemm_n32_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.557–1.094; 2 of 13 Points ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "gemm_n128_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.492–1.195; 4 of 13 Points ≥ 1.03 (m 2, 4, 8, 4096)", gpu: MI355X, workload: "gemm_n256_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.711–1.907; 4 of 13 Points ≥ 1.03 (m 4, 8, 128, 2048)", gpu: MI355X, workload: "gemm_n32_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "6d7cfec (#102, CI run 42549655045, measurement profile 19)", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.046–1.330 at m 4–128; m 1, 2 (1.014, 1.019) not listed", gpu: MI355X, workload: "gemm_n64_k4096 kernel of file 0837d8ad62cf3731, measured on a file with the same kernel, apart from taking the current stream without naming the device, and m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
  - aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/0837d8ad62cf3731.json
---

# bf16 GEMM at narrow N, K 6144: one dot product per wave at small M

Three of this page's four FlyDSL `gemm@1.0` files are the GLM-5.2 set, one schema file per N. Each schema
embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. "m" is
the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0;
  `GEMM` in `aklr/ops/gemm/contract.py`). Every reduction these kernels run at a served Point adds in a fixed order.
- K = 6144 with N = 32, 128 or 256: a weight of 384 KiB, 1.5 MiB or 3 MiB, a narrow-N row bound by launch,
  reduction and memory latency rather than by reading `b` (`../gemm/small_m.md` §2).
- Served Points on main as of #102 (6d7cfec, CI run 42549655045, measurement profile 19): n128 m 2, 4; n256 m 2, 4,
  8, 4096; n32 m 4, 8, 128, 2048. § Result shows an earlier run. Each file lists 13 Points, m = 1, 2, 4, …, 4096.
- The yardstick is aiter, except where a second file is served: `gemm_n128_k6144` and `gemm_n256_k6144` each have
  one, listing m 256–2048 and m 256–1024 and served only where CI measures it at the threshold or above. They, and
  `gemm_n64_k4096` at m 256–4096, are in [bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md) § N 64–256 at
  m 256–4096.

## Counter-signals

- **The middle of the M range was below the threshold in all three files in an earlier CI run of the predecessor repository:** n128 at m ≥ 8
  0.565–0.977 *(field, E1)*, n256 at m 16–1024 0.490–0.970 *(field, E2)*, n32 at m 16–1024 0.694–1.012
  *(field, E3)*. As of #102, n32 is served at m 128 (1.081) *(field, E8)*. These Points run split-K
  MFMA paths, and all but n256 at m = 1024 first pack `b` into MFMA fragment order on every call (n128
  `_make_pack_b` through `prepare_b`; n256 `_run_pack_wave_b` on the path for m 16–512; n32
  `_compile_weight_pack` for m ≥ 16), a second launch over the whole weight. The served Points that pack are n32
  at m = 2048, at 1.062 *(field, E3)*, and, as of #102, n32 at m = 128 *(field, E8)*.
- **m = 1 is not served in any file** (0.993–1.002 *(field, E1, E2, E3)*), nor n32 at m = 2: its 4×4×4 MFMA path
  with 16 K partitions across 128 workgroups reads 0.997 *(field, E3)*.
- **A reduction in arrival order at n128, m = 16.** 24 K partitions add their fp32 MFMA tiles into an fp64
  workspace with `BufferAtomicAdd(Float64)` (`_epilogue_atomic_sum`); the last producer converts to bf16. Adding
  fp32 partials in fp64 is exact only while all partials and running sums fit one 53-bit window, which a wide
  magnitude spread breaks (estimated), so the output can depend on arrival order under a bitwise contract
  (`../gemm/correctness.md` §8). CI's check passed at that Point, which is not
  served (0.631 *(field, E1)*). Do not copy it.
- **Counters that live across calls.** n32's split paths (m = 2 and m 16–2048, the served m = 2048 included) zero
  their arrival counters once per shape and rely on `atomicrmw uinc_wrap` returning each to 0 at its last
  arrival; n128 at m ≥ 512 advances a counter by one generation per call and never resets it. `SKILL.md`'s Stop
  table asks for scratch rewritten by every call before it is read.
- **Not the compile-function form.** `run` keeps built launches in a module-level dict (`_RUN_CACHE`), not in a
  `functools.cache` function, so compilation and scratch allocation run inside `run` on a Point's first call
  (`aklr/ops/README.md` § The kernel section and ABI).
- **Stale docstrings.** n128's says the packed weight is refreshed for a new or modified weight, n32's that `b` is
  packed on first use and repacked when replaced or mutated; both kernels pack on every call.

## Mechanism

**m ≤ 8 (n128 m 2, 4; n256 m 2, 4, 8; n32 m 4, 8): the whole K of an output element stays in one workgroup.** No
MFMA, no split across workgroups, `b` read in its row-major layout with 16 B buffer loads straight into
registers, fp32 accumulation, one launch:

| | n128 `_make_tiny_gemm` | n256 `_gemv` | n32 `_compile_dot` |
|---|---|---|---|
| Work per workgroup | one column, 1 row (m 2) or 2 rows (m 4); 4 waves | one output element; 4 waves | one output element; 1 wave |
| Workgroups | 128 × m/rows = 256 at m 2 and 4 | m × 256: 512, 1024, 2048 | m × 32: 128, 256 |
| Inner product | each thread loads its 24 bf16 of the `b` row once and reuses them for the workgroup's rows; m 2: fp32 products summed in the thread; m 4: `v_dot2_f32_bf16` in 4 independent chains | 3 steps of 16 B of `a` and `b` per thread; `v_dot2_f32_bf16` into 4 fp32 accumulators | 12 steps of 16 B per lane; `v_dot2_f32_bf16` into 4 fp32 accumulators |
| Fold | 6-step xor shuffle per wave, then thread 0 sums the 4 waves' partials from LDS in wave order | the same | 6-step xor shuffle; lane 0 stores |

With N this narrow, an MFMA tile covers few columns and the grid is small (`../gemm/grid_pricing.md` §1,
`../gemm/small_m.md` §3); here the whole K of each output element stays in one workgroup and the fold never
leaves it (`../gemm/small_m.md` §5: splitting N across warps needs no LDS reduction, splitting K
does).

**m ≥ 2048: split-K MFMA with a fold in a fixed order.**

- n256, m 2048 and 4096 (`_gemm_pipeline`): 64×64 tile with 4 waves (m 2048) or 128×128 with 8 waves (m 4096);
  `a` and `b`, unpacked, go by `cdna4.BufferLoadAsyncLDS128b` into a 2-stage LDS ring and from there by 16 B reads
  into `mfma_f32_16x16x32_bf16`. Split-K 4 writes fp32 partials in fragment order; a second launch
  (`_reduce_packed_parts`) sums partitions 0–3 in order and stores bf16. 512 and 256 workgroups.
- n32, m 2048 (`_compile_hierarchical`): a 32×32 output tile per workgroup, K split over its 8 waves and over 4
  workgroups (256 workgroups of 512 threads); `b` packed per call (384 KiB); `mfma_f32_16x16x32_bf16`. The 8 wave
  partials are summed through LDS in wave order; owner waves store an fp32 plane and take a ticket from a wrapping
  counter (`atomicrmw uinc_wrap`, agent scope); the last of the 4 sums planes 0–3 in order and stores bf16, all in
  the GEMM launch.

Fold forms: a separate reduce launch against an in-launch last-arriver (`../gemm/grid_pricing.md` §2,
`../gemm/small_m.md` §4); both are bitwise-stable because the order is fixed
(`../gemm/correctness.md` §8).

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). The table is an earlier CI run of the predecessor repository; bold: at or
above 1.03 (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables) in that run. Per Point:

The predecessor repository's CI ran an editable install whose `setup.py` replaced any other FlyDSL with its pin, so
these runs used FlyDSL 0.3.2, not the CI image's 0.3.4.1 (not checked in the run's log;
`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | ≥ 1.03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n128 *(field, E1)* | 1.002 | **1.195** | **1.182** | 0.600 | 0.631 | 0.565 | 0.584 | 0.662 | 0.650 | 0.747 | 0.738 | 0.942 | 0.977 | 2 of 13 |
| n256 *(field, E2)* | 0.997 | **1.065** | **1.201** | **1.125** | 0.490 | 0.500 | 0.494 | 0.531 | 0.595 | 0.784 | 0.970 | **1.047** | **1.069** | 5 of 13 |
| n32 *(field, E3)* | 0.993 | 0.997 | **1.257** | **1.841** | 0.694 | 0.695 | 0.752 | 1.012 | 0.979 | 0.947 | 0.974 | **1.062** | 0.811 | 3 of 13 |

10 of the 39 Points were at or above 1.03 in that run. As of #102 (6d7cfec, CI run 42549655045, measurement profile
19) 10 are served, not the same 10: n256 no longer at m 2048, n32 also at m 128 *(field, E8)*. The `speedup` in each
schema's `workloads` is the current value.

## N 64 × K 4096 at m 4–128: the whole K in one workgroup, on `v_dot2`

`gemm_n64_k4096/0837d8ad62cf3731.json` lists m 4, 8, …, 128 (512 KiB of weight) with the design of the m ≤ 8 paths
above: no MFMA, no LDS ring, `b` in its row-major layout loaded straight into registers, the whole K of an output in
one workgroup. No other file serves these Points, so the yardstick is aiter. As of #125 CI reads 1.020–1.286:
dispatch serves m 8–128, not m 4 (`qualifies()` in `aklr/core/contract/spec/_workloads.py`; `AKLR_MIN_SPEEDUP`,
default 1.03, `aklr/ops/README.md` § Environment variables). Narrow N at larger m:
[bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md). m 1 and 2 are not listed: within 2% of aiter
(1.014, 1.019) *(field, E4)*.

- Each workgroup owns an R × C block of outputs. Every thread takes a strided set of 8-element K chunks of those rows
  and columns, issues all its 16 B loads before the first use, and accumulates with `v_dot2_f32_bf16` in fp32.
- Per m: 1 × 1 outputs and one wave (m 4, 256 workgroups); 2 × 2 and one wave (m 8, 16, 32: 128, 256 and 512
  workgroups); 4 × 4 and 4 × 8 with eight waves (m 64, 128: 256 workgroups). Workgroup ids are remapped so that the
  ids of one XCD cover one block of rows × columns of the tile grid and its L2 reads a fraction of A and of B
  (`../gemm/schedule_and_locality.md` §2).
- Fold, in a fixed order: at m 4, six `shuffle_xor` steps across the wave; with several outputs per workgroup, a
  reduce-scatter that exchanges lane halves with `permlane32_swap` and `permlane16_swap`, then DPP for the lower lane
  bits; with eight waves, the wave sums meet in LDS and are added in wave order. One 2-byte store per output, `sc0
  sc1`.
- The launch requests kernel-argument preload (`amdgpu-kernarg-preload-count` 16) and passes the three tensors with
  static layouts (`flyc.from_torch_tensor`). In the campaign's compiler output that preloaded all three pointers
  (kernarg size 24, 8 user SGPRs), where dynamic-layout arguments preloaded only the first *(field, E7)*; the delivered
  build's metadata was not read.

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: 1.046–1.330 over aiter, highest at m 8,
thin at m 4 *(field, E4)*. The kernel read 4.08–4.20 µs per round at every m from 1 to 128 while aiter read
4.19–5.95 *(field, E5)*: the gain is aiter's time growing with m, and whether profile 19 has a floor near 4.1 µs was
not measured (`../pitfalls/aklr_dispatch_floor.md`). Under stress every listed Point gave the output of the call
alone *(field, E6)*. Reproduce with
`python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/0837d8ad62cf3731.json` (§ Reproduce); to roll back,
delete it: its Points are then served by no file, and the caller runs its own path.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
```

The report goes to `aklr_validation/report.md`; a run outside CI checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: dispatch stops serving that file's Points. `AKLR_DISABLE=gemm@1.0` stops serving the
whole contract, every other bf16 GEMM schema included. A re-measured Point below 1.03 stops being served
without a rollback, and a `failed` accuracy is never served. Roll back for a wrong result CI does not catch, for
example after a FlyDSL or LLVM upgrade, or a counter left off its expected value by an aborted launch (n32 at
m = 2048).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.565–1.195, 2 of 13 Points ≥ 1.03 (m 2, 4) (gemm_n128_k6144, m 1–4096)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.490–1.201, 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096) (gemm_n256_k6144, m 1–4096)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the predecessor repository's CI] speedup over aiter per Point 0.694–1.841, 3 of 13 Points ≥ 1.03 (m 4, 8, 2048) (gemm_n32_k6144, m 1–4096)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays (32 at m 1, 2) with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] an earlier file (the kernel of 0837d8ad62cf3731 except `torch.cuda.current_stream()` without `a.device`, also listing m 1, 2): 1.014, 1.019, 1.046, 1.330, 1.186, 1.207, 1.233, 1.313 at m 1, 2, 4, 8, 16, 32, 64, 128; m 1, 2 demoted `not_faster`; accuracy passed
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E4] per-round times, kernel 4.08–4.20 µs at every m (m 1 4.16–4.20, m 128 4.09–4.16); aiter 4.19–4.33 (m 1), 4.35–4.42 (m 4), 5.43–5.95 (m 8), 5.36–5.52 (m 128) (`candidate_milliseconds`, `versus_milliseconds`)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with the schema, each try compared byte for byte with an eager call alone] 0837d8ad62cf3731 as its predecessor, whose one difference was `torch.cuda.current_stream()` without `a.device`, every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=1, compiler output read in the forge-loop campaign, not timed] gemm_n64_k4096 with `amdgpu-kernarg-preload-count` 16: dynamic-layout tensor arguments preloaded only the first pointer (`preload_length 2`); static-layout `flyc.from_torch_tensor` arguments preloaded all three (kernarg size 24, user SGPR count 8)
- E8 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI perf run, measurement profile 19] Points at or above 1.03 as of #102: n128 m 2, 4 (0.557–1.094); n256 m 2, 4, 8, 4096 (0.492–1.195; m 2048 1.021); n32 m 4, 8, 128, 2048 (0.711–1.907; m 128 1.081) (6d7cfec, CI run 42549655045; each schema's `workloads`)
- S1 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/0837d8ad62cf3731.json#kernel.py:111 `_CONFIG` (rows, columns, waves, XCD row and column split) per m; XCD remap at :144-171; loads before use at :178-186; `fdot2_f32_bf16` at :188-207; `shuffle_xor` fold at :216-219, `_butterfly` at :79; wave fold in LDS at :225-237; store `sc0 sc1` at :25, :45; preload hint at :250-253
