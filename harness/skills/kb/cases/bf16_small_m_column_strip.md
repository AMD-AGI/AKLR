---
title: "bf16 GEMM, N 6144 × K 4096 and K 6144 at m 1–128 — one narrow column strip of B per workgroup on a full round, its waves splitting every K tile; at K 6144, m 64–128, K split four ways across workgroups and folded in a second launch"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-32", "M 64-128"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1, AKLR validate measurement profile 19
measured:
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.130–1.290; 8 of 8 Points at or above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n6144_k4096 file 3e2804d64ce6b4c7, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.085–1.309; 7 of 7 Points at or above AKLR_MIN_SPEEDUP; m 64 left out, where the predecessor runs a one-launch fold", gpu: MI355X, workload: "gemm_n6144_k6144 file f73a587b360cf0f3, measured as its predecessor, whose code differs only at m 64, m 1, 2, 4, 8, 16, 32, 128", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k4096/3e2804d64ce6b4c7.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k6144/f73a587b360cf0f3.json
---

# bf16 GEMM at m 1–128, N 6144 × K 4096 and K 6144: one column strip of B per workgroup

Two FlyDSL kernels for the `gemm@1.0` contract on 48 and 72 MiB weights, one file per schema name, each listing
m 1, 2, 4, …, 128. Each file embeds its kernel as `kernel.sources[0].content` (`kernel.py`); the function names below
are in that file. "m" is the Point's M. N 6144 × K 4096 runs an extended version of the `_build_skinny` strip builder of
`gemm_n6144_k16384`; N 6144 × K 6144 has a strip kernel of its own (`_build_strip`) and, at m 64 and 128, a split-K
kernel (`_build_splitk`). The band kernels of the same decode sizes, which split M × N × K differently, are in
[k3_bf16_band_stream.md](k3_bf16_band_stream.md); the split-K folds of the Kimi-K3 K 7168 kernels in
[k3_bf16_splitk_wide_n.md](k3_bf16_splitk_wide_n.md).

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16, `trans_b`,
  no bias, bf16 `out`. The schema's `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0).
  Every reduction these kernels run at a listed Point adds in a fixed order, with no atomics on data.
- N = 6144 with K = 4096 or 6144: 48 or 72 MiB of `b`, against at most 1.5 MiB of `a` at these m (estimated from the
  shapes), so each kernel gives every CU its own strip of `b`.
- No other file serves these Points, so the yardstick is aiter. Dispatch serves a Point whose CI `speedup` reaches
  `AKLR_MIN_SPEEDUP`, default 1.03 (`qualifies()` in `aklr/core/contract/spec/_workloads.py`; `aklr/ops/README.md`
  § Environment variables); every listed Point of both files reaches it in CI, 1.081–1.316, so all 16 are served (#125).

## Counter-signals

- **m 64 and 128 at K 6144 take two launches.** On this kernel a one-launch fold, in which the last of the four
  splits to count in on a launch-tagged counter adds the planes, read 18.30 µs against 19.38 for the fold launch at
  m 64, and 24.05 against 23.82 at m 128 *(field, E4)*. AKLR's per-kernel timer gives the same order at m 64 but reads
  the fold launch's short second kernel at its floor; on the wall time of back-to-back calls the one-launch fold was
  the slower at both m, 19.29 against 18.50 µs at m 64 *(field, E5)*. Its counter word came from a `torch.empty`
  workspace the call had not written, which `SKILL.md`'s Stop table rules out, so both Points take the second
  launch. At m 64 the one-launch fold measured 1.272 over aiter *(field, E2)*.
- **The thinnest margins** are K 6144 at m 32 (1.085) and K 4096 at m 128 (1.130) *(field, E1, E2)*.
- **Fallback.** N 6144 × K 4096 sends m > 128 to `torch.matmul`, N 6144 × K 6144 any m outside its tables; no listed
  Point reaches either.

## Mechanism

Shared by both strip kernels (all of N 6144 × K 4096; N 6144 × K 6144 at m ≤ 32):
- **256 workgroups, each owning a strip of 24 output columns** (48 at K 4096, m 64 and 128, where two workgroups own
  the two row halves of a 48-column strip) over all of K. A and B go into LDS by `buffer_load_dwordx4 … lds` into
  XOR-swizzled 16 B chunks, B with `sc0 nt` and A with `sc0`, through one ring that every wave of the workgroup reads,
  with an `s_waitcnt vmcnt` and an `s_barrier` per K tile (`../gemm/datapath.md` §2, §5).
- **The K split is inside every K tile:** each of the W waves takes a 1/W slice of the tile's K, so each wave holds
  partials of the whole strip. After the loop the waves write their fp32 partials to a padded LDS scratch and the
  threads add them over the waves in order, then round once (`../gemm/correctness.md` §8).
- **The K loop is an `scf.for`** with `sched_vmem`, `sched_dsrd` and `sched_mfma` hints
  (`../gemm/schedule_and_locality.md` §3), and A is first in the MFMA.

**N 6144 × K 4096** (`3e2804d6…`, `_build_skinny`), four waves:
- The MFMA runs over 32 columns (48 at m 64, 128). With a 24-column strip, the second B DMA of each K tile covers
  rows 16–31 of the stage; rows 24–31, past the strip, read zero through an offset past the descriptor, and those
  columns are not stored.
- Ring: 12 slots at m ≤ 16, 10 at m 32 (BK 128), 4 at BK 256 (m 64), 5 at BK 128 (m 128).
- Workgroup t takes index kid = (t mod 8)·32 + t/8. At m ≤ 32 kid is the strip; at m 64 and 128 the strip is kid / 2
  and kid mod 2 its row half, so the two halves of a 48-column strip get ids 8 apart. The workgroups whose ids are
  equal mod 8, which run on one XCD, stream one contiguous slab of B. Workgroup kid starts its K walk at tile
  (kid / G)·R mod the tile count, with (G, R) = (32, 4), (64, 7), (32, 6), (32, 6), (8, 1), (32, 3) for m ≤ 4, 8, 16,
  32, 64, 128 (`../gemm/small_m.md` §6).
- Fold: every thread adds two adjacent columns over the four waves in order and stores them as one 4 B word. The
  three pointer arguments are preloaded into SGPRs.

**N 6144 × K 6144** (`f73a587b…`):
- m ≤ 32, `_build_strip`, eight waves, each taking an eighth of every K tile; BK 256 with 10 slots (m ≤ 8), 512 with 4
  (m 16), 256 with 4 (m 32). It differs from the K 4096 strip in three ways: workgroup b takes strip b (no
  renumbering); each B stage holds only the 24 strip rows, so MFMA columns 24–31 read the next stage's bytes and are
  never stored; and when the last DMA of a stage covers fewer than all eight waves' worth of chunks, only the first
  waves issue it. Workgroups 8r to 8r + 7 start on K tile r mod the tile count (`bid >> 3`), meant to keep the B
  stream from moving in lockstep. Each thread adds one output over the eight waves in order and stores 2 bytes.
- m 64 and 128, `_build_splitk`: 256 workgroups = 64 column groups of 96 columns × 4 K slices of 1536; eight waves as
  4 (rows) × 2 (columns), so no wave splits K inside the workgroup; BK 128 with 3 slots (m 64), 64 with 5 (m 128).
  Workgroup b takes K slice (b mod 8) / 2, meant to keep a slice's workgroups on one pair of XCDs (the code calls it
  a placement hint). A workgroup gathers its waves' sub-tiles through padded LDS and writes them, at the default
  cache policy, as its fp32 partial plane to the call's `torch.empty` workspace, which holds the four planes only.
- m 64 and 128, second launch: a fold kernel adds the four planes in split order and rounds once, 16 rows × 96
  columns per workgroup (256 workgroups at m 64, 512 at m 128); every element it reads was written by the first
  launch of the same call.
- No kernel-argument preload.

## Result

`speedup` here is AKLR validate's one-sided lower bound of the aiter / kernel time ratio (measurement profile 19,
`MEASUREMENT_PROFILE` in `aklr/core/validation/stages/_measure.py`; aiter's tuned `gemm_a16w16` through
`gemm_baseline`), on one MI355X at main 0e02a81 with FlyDSL 0.3.2. N 6144 × K
6144 was measured as its predecessor, whose code differs only at m 64 (§ Counter-signals). Dispatch reads the
`speedup` CI writes: 1.204–1.316 at K 4096 and 1.081–1.300 at K 6144 (m 64, the fold launch, 1.117), all 16 Points
served (#125).

| File | Points | Speedup over aiter | Highest |
|---|---|---|---|
| N 6144 × K 4096 `3e2804d6…` *(field, E1)* | m 1–128 | 1.130–1.290 | 1.290 at m 4 |
| N 6144 × K 6144 `f73a587b…` *(field, E2)* | m 1–32, 128 | 1.085–1.309 | 1.309 at m 128 (split-K, two launches) |

Every Point passed accuracy. Kernel times at K 6144 ran from 12.91 µs (m 1) to 24.23 µs (m 128) *(field, E2)*.
Under stress every listed Point gave the output of the call alone, K 6144 as its predecessor *(field, E3)*.

## Reproduce

On an MI355X, from the repository root, one file at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6144_k4096/3e2804d64ce6b4c7.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n6144_k6144/f73a587b360cf0f3.json
```

The report goes to `aklr_validation/report.md` (`aklr/ops/README.md` § Commands). Add `--base origin/main` only on a
branch that adds the file: with the base set, bench measures only schemas that are not at the base
(`aklr/core/validation/stages/validation_scope.py`). The local result does not count; CI's is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points are then served by no file, since the other file of each name has no
kernel, and the caller runs its own path. `AKLR_DISABLE=gemm@1.0` stops serving the whole contract, every other bf16
GEMM schema included. A Point CI measures below the threshold stops being served without a rollback.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] 3e2804d64ce6b4c7: 1.278, 1.287, 1.290, 1.225, 1.286, 1.251, 1.218, 1.130 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n6144_k4096)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E1; times are the median of a Point's per-round times] the predecessor of f73a587b360cf0f3, the same code but for a one-launch fold on a launch-tagged counter at m 64: 1.166, 1.106, 1.109, 1.114, 1.194, 1.085, 1.272, 1.309 at m 1, 2, 4, 8, 16, 32, 64, 128; kernel 12.91 µs at m 1, 24.23 at m 128; accuracy passed (gemm_n6144_k6144)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] 3e2804d64ce6b4c7 and the predecessor of f73a587b360cf0f3 (E2), every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ; a full model's decode graph was not run
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=?, the campaign's interleaved A/B, timer not stated in the record] N 6144 × K 6144: the one-launch tagged fold with batched loads and its own quarter from registers 18.30 µs against 19.38 for the fold launch at m 64, 24.05 against 23.82 at m 128
- E5 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=10 rounds in one process, AKLR's `bench_cudagraph` (per-kernel times summed per call) and the wall time per call of one CUDA graph of 525 back-to-back calls] the predecessor of f73a587b360cf0f3 (E2) at K 6144, its one-launch tagged fold against the fold launch, outputs bitwise equal: per-kernel 18.34 against 19.46 µs at m 64, the fold launch's second kernel 4.43 µs against 4.39 for a one-element `zero_()`; wall time 19.29 against 18.50 µs at m 64, 26.42 against 25.07 at m 128
- S1 harness/scripts/forge_task/assets/aklr_forge_driver.py:437 `_stress_conditions`; `beside_another` at :456 gives the second call the same inputs (`twin = {**call, …}`, :457)
- S2 aklr/ops/gemm/impls/flydsl/gemm_n6144_k4096/3e2804d64ce6b4c7.json#kernel.py:26 `_SKINNY` per m; `B_LIVE` at :116; rows past the strip sent past the descriptor at :180; XCD-major index kid at :154, strip and row half at :155-160; first K tile at :187; fold at :274-312; preload at :325
- S3 aklr/ops/gemm/impls/flydsl/gemm_n6144_k6144/f73a587b360cf0f3.json#kernel.py:21 `_STRIP` and `_SPLITK` (:30); strip number at :183; 24-row B stages and the comment on reading the next stage at :140 and :156-157; last DMAs from the first waves only at :219-228; K start at :205; K slice per XCD pair at :385-389; partial stores at the default policy (`aux` 0) at :495-504, every row of the plane over `OUT_PASSES`; fold kernel in split order at :506-527, launched after the split-K kernel at :529-535; `torch.empty` workspace of the four planes per call at :552-553
