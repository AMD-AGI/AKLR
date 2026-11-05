---
title: "bf16 GEMM, wide N (2624–6144) × K 2048–6144 — one wave per output column on the unpacked weight at M ≤ 2; a weight repack per call loses above"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-2", "M 4-4096"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, CI, per Point", value: "0.418–1.148; 1 of 13 Points ≥ 1.03 and served (m 1); at m 2, 4, 8, 64 a Triton kernel of the same name is served", gpu: MI355X, workload: "gemm_n2624_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.250–1.465; 2 of 13 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "gemm_n3072_k6144, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "102877ca0 (CI run 40154665266); file added by 97d5eb0b3 (#81)", date: "2026-10-25"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.355–1.198; 2 of 13 Points ≥ 1.03 (m 1, 2), 1 served (m 1); at m 2–64 a Triton kernel of the same name is served", gpu: MI355X, workload: "gemm_n6144_k2048, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
  - {metric: "speedup over aiter, CI, per Point", value: "0.418–1.208; 2 of 13 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "gemm_n6144_k3072, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", run_ref: "5571f4589 (CI run 40613440602); file added by 97d5eb0b3 (#81)", date: "2026-10-28"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n2624_k6144/4001df477e6c67f6.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3072_k6144/a85da0998ddc31b1.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k2048/eb749c07b94008ab.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k3072/d7fed2d1260b5f45.json
  - aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json
  - aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json
---

# bf16 GEMM at wide N: a wave per output column at M ≤ 2

Four FlyDSL kernels for the `gemm@1.0` contract on GLM-5.2 shapes (PR #43), one schema file per shape. Each schema
embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. Two of
the shapes also have a Triton kernel under the same name: n2624_k6144 (the QKV projection, PR #93) and n6144_k2048
(the output projection, PR #95). "m" is the Point's M.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0;
  `GEMM` in `aklr/ops/gemm/contract.py`). The served paths keep each output's whole K in one wave.
- N × K = 2624 × 6144, 3072 × 6144, 6144 × 2048, 6144 × 3072: a weight of 30.75, 36, 24 and 36 MiB, a streaming
  row bound by the read of `b` (`../gemm/small_m.md` §2).
- m = 1 and 2 only. FlyDSL is served at n2624 m 1, n3072 m 1, 2, n6144_k2048 m 1, n6144_k3072 m 1, 2. The Triton
  kernels are served at n2624 m 2, 4, 8, 64 and n6144_k2048 m 2–64; no kernel serves the other Points.
- The yardstick at a Point is the faster of the kernel AKLR serves there and aiter: aiter, except at the Points
  the Triton kernels serve.

## Counter-signals

- **Repacking the weight on every call loses at every Point that does it.** Above the m ≤ 2 path each launch
  first runs a pack kernel that rewrites the whole `b` into MFMA order, then the MFMA GEMM on the packed copy
  (n2624 `_run_pack`, m ≥ 2; n3072 `_pack`, m ≥ 4; n6144_k2048 `_run_pack`, m ≥ 4; n6144_k3072
  `_run_pack_weights` or `_run_pack_lds_weights`, m ≥ 4). Those Points read n2624 0.418–0.996, n3072 0.250–0.735,
  n6144_k2048 0.355–0.861, n6144_k3072 0.418–0.711 *(field, E1, E2, E3, E4)*.
  - **The cost at these shapes:** the pack reads the weight and writes the packed copy, and the GEMM reads the copy
    again: about three passes over 24–36 MiB instead of one, plus a second launch per call (estimated). The cost
    of the pack alone is not measured: no arm runs the same GEMM without it.
  - The kernel cannot keep the packed copy between calls (`SKILL.md`, Stop table), and `gemm@1.0` has only the
    `row_major` variant, so no pre-shuffled weight can reach it; the host-side preshuffle of
    `../gemm/datapath.md` §10 pays only when done once. At these weight sizes, read `b` in its (n, k)
    layout.
- **At m 2–64 the Triton kernels win on the two shapes that have one.** They read the unpacked weight once per
  call: n2624 m 2, 4, 8, 64 at 1.077–1.195 *(field, E5)* and n6144_k2048 m 2–64 at 1.078–1.285 *(field, E6)*. At
  n6144_k2048, m = 2, the FlyDSL m ≤ 2 path itself reads 1.075 *(field, E3)* against Triton's 1.139
  *(field, E6)*, about 0.94× the served kernel (estimated from the two CI ratios).
- **Stale docstring.** n6144_k2048's says packed weights are reused while their identity and version stay
  unchanged; the code packs on every call.
- **Not the compile-function form.** `run` keeps built launches in a module-level dict (`_RUN_CACHE`), not in a
  `functools.cache` function, so compilation and scratch allocation run inside `run` on a Point's first call
  (`aklr/ops/README.md` § The kernel section and ABI).

## Mechanism

At every served FlyDSL Point: **one wave owns one output column and reads that column's weight row, unpacked,
once.** Each lane issues 16 B buffer loads of the `b` row and of each `a` row straight into registers (no LDS, no
MFMA), accumulates in a few independent fp32 chains, and the wave reduces them; one launch, no split across waves
or workgroups, so the order of the sum is fixed. The weight is read once per call at full width, which is what a
streaming row needs (`../gemm/small_m.md` §2); the grid is N/4 or N/8 workgroups, 656–1536, more
than the 256 CUs (`../gemm/grid_pricing.md` §1). `../gemm/small_m.md` §5 keeps `b`
on the LDS path for MFMA tiles; at m ≤ 2 these kernels issue no MFMA.

| | n2624_k6144 m 1 | n3072_k6144 m 1, 2 | n6144_k2048 m 1 | n6144_k3072 m 1, 2 |
|---|---|---|---|---|
| Function | `_vector_gemv` | `_small_dot` | `_dot_gemm` | `_tiny_product` |
| Workgroup | 4 waves, 4 columns | 4 waves, 4 columns | 4 waves, 4 columns | 4 waves (m 1) or 8 (m 2), one column each |
| Workgroups | 656 | 768 | 1536 | 1536 (m 1), 768 (m 2) |
| K steps per lane | 12 × 16 B, all 12 issued before the first dot | 12 × 16 B | 4 × 16 B | 6 × 16 B; at m 2 both rows' `a` loads issued with the `b` load |
| Products | `v_dot2_f32_bf16` into 4 fp32 chains | m 1: fp32 FMA on 8-wide vectors; m 2: `v_dot2_f32_bf16` into 4 chains, the weight base in a scalar buffer descriptor | `v_dot2_f32_bf16` into 4 chains | `v_dot2_f32_bf16` into 4 chains |
| Wave reduction | DPP `row_shr` / `row_bcast` adds (`_wave_sum_dpp`), the sum in lane 63 | `fx.coop.warp_reduce` | 6-step xor shuffle | balanced local sum, then `warp_reduce` |

## Result

`speedup` as CI writes it into each schema's `workloads`: a lower bound of the CUDA-graph time ratio aiter /
kernel (`aklr/core/validation/stages/_bench/_report.py`). Dispatch serves each Point with the file of the highest
`speedup`, from 1.03 (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables). Bold: the file served
there. Per Point:

CI installs the repository with `--no-deps`, so these runs used the FlyDSL of the CI image, 0.3.4.1, not the
repository's pin 0.3.2 (`../pitfalls/measured_a_different_build.md`).

| File | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | Served |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FlyDSL n2624_k6144 *(field, E1)* | **1.148** | 0.437 | 0.432 | 0.438 | 0.418 | 0.422 | 0.555 | 0.615 | 0.795 | 0.765 | 0.996 | 0.871 | 0.822 | 1 of 13 |
| Triton n2624_k6144 *(field, E5)* | – | **1.081** | **1.077** | **1.100** | – | – | **1.195** | – | – | – | – | – | – | 4 of 4 |
| FlyDSL n3072_k6144 *(field, E2)* | **1.465** | **1.331** | 0.250 | 0.257 | 0.265 | 0.303 | 0.301 | 0.315 | 0.370 | 0.431 | 0.606 | 0.626 | 0.735 | 2 of 13 |
| FlyDSL n6144_k2048 *(field, E3)* | **1.198** | 1.075 | 0.383 | 0.425 | 0.378 | 0.355 | 0.375 | 0.459 | 0.548 | 0.635 | 0.766 | 0.685 | 0.861 | 1 of 13 |
| Triton n6144_k2048 *(field, E6)* | – | **1.139** | **1.146** | **1.285** | **1.137** | **1.078** | **1.105** | – | – | – | – | – | – | 6 of 6 |
| FlyDSL n6144_k3072 *(field, E4)* | **1.208** | **1.119** | 0.420 | 0.418 | 0.435 | 0.421 | 0.424 | 0.491 | 0.532 | 0.564 | 0.711 | 0.696 | 0.701 | 2 of 13 |

"–": the Triton file has no workload at that Point. The FlyDSL files serve 6 of their 52 Points; at n6144_k2048,
m = 2, the FlyDSL file passes 1.03 but the Triton file is served.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n2624_k6144/4001df477e6c67f6.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n3072_k6144/a85da0998ddc31b1.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6144_k2048/eb749c07b94008ab.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6144_k3072/d7fed2d1260b5f45.json
```

The Triton files validate the same way. The report goes to `aklr_validation/report.md`; a local run checks the file,
the served numbers are CI's (`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits
the file: with the base set, bench measures only schemas that are not at the base, so on an unchanged checkout it
skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the FlyDSL schema file: its m 1 (and m 2) Points go unserved, while the Triton files keep the Points
they serve. `AKLR_DISABLE=gemm@1.0` stops serving the whole contract, every other bf16 GEMM schema included. A
re-measured Point below 1.03 stops being served without a rollback, and a `failed` accuracy is never served. Roll
back for a wrong result CI does not catch, for example after a FlyDSL or LLVM upgrade.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.418–1.148, 1 of 13 Points ≥ 1.03 (m 1); m ≥ 2 (weight repacked per call) 0.418–0.996 (gemm_n2624_k6144 FlyDSL, m 1–4096; 5571f4589, CI run 40613440602)
- E2 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.250–1.465, 2 of 13 Points ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.250–0.735 (gemm_n3072_k6144 FlyDSL, m 1–4096; 102877ca0, CI run 40154665266)
- E3 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.355–1.198, 2 of 13 Points ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.355–0.861 (gemm_n6144_k2048 FlyDSL, m 1–4096; 5571f4589, CI run 40613440602)
- E4 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, AKLR CI] speedup over aiter per Point 0.418–1.208, 2 of 13 Points ≥ 1.03 (m 1, 2); m ≥ 4 (repacked) 0.418–0.711 (gemm_n6144_k3072 FlyDSL, m 1–4096; 5571f4589, CI run 40613440602)
- E5 [2026-10, gfx950, Triton, N=1, AKLR CI] speedup over aiter per Point 1.077–1.195 at m 2, 4, 8, 64, all served (gemm_n2624_k6144 Triton; 5571f4589, CI run 40613440602)
- E6 [2026-10, gfx950, Triton, N=1, AKLR CI] speedup over aiter per Point 1.078–1.285 at m 2–64, all served (gemm_n6144_k2048 Triton; 5571f4589, CI run 40613440602)
