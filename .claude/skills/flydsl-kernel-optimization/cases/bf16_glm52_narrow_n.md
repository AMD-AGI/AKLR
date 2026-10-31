---
title: "bf16 GEMM, narrow N (32–256) × K 6144 — whole dot products per wave or workgroup at M ≤ 8, split-K with a fixed-order fold at M ≥ 2048"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 2-8", "M 16-1024", "M 2048-4096"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.565–1.195; 2 of 13 Points ≥ 1.03 (m 2, 4)", gpu: MI355X, workload: "gemm_n128_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.490–1.201; 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096)", gpu: MI355X, workload: "gemm_n256_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.694–1.841; 3 of 13 Points ≥ 1.03 (m 4, 8, 2048)", gpu: MI355X, workload: "gemm_n32_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
  - aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
  - aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
---

# bf16 GEMM at narrow N, K 6144: one dot product per wave at small M

Three FlyDSL kernels for the `gemm@1.0` contract on GLM-5.2 shapes (PR #43), one schema file per N. Each schema
embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. "m" is
the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0;
  `GEMM` in `aklr/ops/gemm/contract.py`). Every reduction these kernels run at a served Point adds in a fixed order.
- K = 6144 with N = 32, 128 or 256: a weight of 384 KiB, 1.5 MiB or 3 MiB, a narrow-N row bound by launch,
  reduction and memory latency rather than by reading `b` (`../gemm/flydsl_gemm_small_m.md` §2).
- Served Points: n128 m 2, 4; n256 m 2, 4, 8, 2048, 4096; n32 m 4, 8, 2048. Each file holds 13 Points,
  m = 1, 2, 4, …, 4096.
- Each schema name has only this file, so at every Point the yardstick is aiter.

## Counter-signals

- **The middle of the M range is served in none of the three files:** n128 at m ≥ 8 0.565–0.977 *(field, E1)*,
  n256 at m 16–1024 0.490–0.970 *(field, E2)*, n32 at m 16–1024 0.694–1.012 *(field, E3)*. These Points run split-K
  MFMA paths, and all but n256 at m = 1024 first pack `b` into MFMA fragment order on every call (n128
  `_make_pack_b` through `prepare_b`; n256 `_run_pack_wave_b` on the path for m 16–512; n32
  `_compile_weight_pack` for m ≥ 16), a second launch over the whole weight. The one served Point that packs is
  n32 at m = 2048, at 1.062 *(field, E3)*.
- **m = 1 is not served in any file** (0.993–1.002 *(field, E1, E2, E3)*), nor n32 at m = 2: its 4×4×4 MFMA path
  with 16 K partitions across 128 workgroups reads 0.997 *(field, E3)*.
- **A reduction in arrival order at n128, m = 16.** 24 K partitions add their fp32 MFMA tiles into an fp64
  workspace with `BufferAtomicAdd(Float64)` (`_epilogue_atomic_sum`); the last producer converts to bf16. Adding
  fp32 partials in fp64 is exact only while all partials and running sums fit one 53-bit window, which
  cancellation or a wide magnitude spread breaks (estimated), so the output can depend on arrival order under a
  bitwise contract (`../gemm/flydsl_gemm_correctness.md` §8). CI's check passed at that Point, which is not
  served (0.631 *(field, E1)*). Do not copy it.
- **Counters that live across calls.** n32's split paths (m = 2 and m ≥ 16, the served m = 2048 included) zero
  their arrival counters once per shape and rely on `atomicrmw uinc_wrap` returning each to 0 at its last
  arrival; n128 at m ≥ 512 advances a counter by one generation per call and never resets it. `SKILL.md`'s Stop
  table asks for scratch rewritten by every call before it is read, or `state_bytes` with an epoch.
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

With N this narrow, an MFMA tile covers few columns and the grid is small (`../gemm/flydsl_gemm_grid_pricing.md` §1,
`../gemm/flydsl_gemm_small_m.md` §3); here every output element gets its own wave or workgroup and the fold never
leaves it (`../gemm/flydsl_gemm_small_m.md` §5: splitting N across warps needs no LDS reduction, splitting K
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

Fold forms: a separate reduce launch against an in-launch last-arriver (`../gemm/flydsl_gemm_grid_pricing.md` §2,
`../gemm/flydsl_gemm_small_m.md` §4); both are bitwise-stable because the order is fixed
(`../gemm/flydsl_gemm_correctness.md` §8).

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). Bold: served (≥ 1.03, `AKLR_MIN_SPEEDUP`,
`aklr/ops/README.md` § Environment variables). Per Point:

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n128 *(field, E1)* | 1.002 | **1.195** | **1.182** | 0.600 | 0.631 | 0.565 | 0.584 | 0.662 | 0.650 | 0.747 | 0.738 | 0.942 | 0.977 | 2 of 13 |
| n256 *(field, E2)* | 0.997 | **1.065** | **1.201** | **1.125** | 0.490 | 0.500 | 0.494 | 0.531 | 0.595 | 0.784 | 0.970 | **1.047** | **1.069** | 5 of 13 |
| n32 *(field, E3)* | 0.993 | 0.997 | **1.257** | **1.841** | 0.694 | 0.695 | 0.752 | 1.012 | 0.979 | 0.947 | 0.974 | **1.062** | 0.811 | 3 of 13 |

10 of the 39 Points are served; aiter runs the rest.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n128_k6144/0bac580d3d76a3df.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n256_k6144/0d33213684622a71.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n32_k6144/87f4f44793d84725.json
```

The report goes to `aklr_validation/report.md`; a local run checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: dispatch falls back to aiter at that file's Points. `AKLR_DISABLE=gemm@1.0` sends the
whole contract to aiter, every other bf16 GEMM schema included. A re-measured Point below 1.03 stops being served
without a rollback, and a `failed` accuracy is never served. Roll back for a wrong result CI does not catch, for
example after a FlyDSL or LLVM upgrade, or a counter left off its expected value by an aborted launch (n32 at
m = 2048).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.565–1.195, 2 of 13 Points ≥ 1.03 (m 2, 4) (gemm_n128_k6144, m 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.490–1.201, 5 of 13 Points ≥ 1.03 (m 2, 4, 8, 2048, 4096) (gemm_n256_k6144, m 1–4096; 5571f4589, CI run 40613440602)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.694–1.841, 3 of 13 Points ≥ 1.03 (m 4, 8, 2048) (gemm_n32_k6144, m 1–4096; 5571f4589, CI run 40613440602)
