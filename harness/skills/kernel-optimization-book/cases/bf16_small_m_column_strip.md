---
title: "bf16 GEMM, N 6144 × K 4096 and K 6144 at m 1–128 — one narrow column strip of B per workgroup on a full round, its waves splitting every K tile; at K 6144, m 64–128, K split four ways across workgroups and folded in a second launch"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-32", "M 64-128"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.130–1.290; all 8 sizes at or above 1.05", gpu: MI355X, workload: "N 6144 × K 4096, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.085–1.309; all 7 sizes at or above 1.05; m 64 left out, where the measured build ran a one-launch fold", gpu: MI355X, workload: "N 6144 × K 6144, measured on the build before the fold launch at m 64, the same code at the other sizes, m 1, 2, 4, 8, 16, 32, 128", date: "2026-11-09"}
---

# bf16 GEMM at m 1–128, N 6144 × K 4096 and K 6144: one column strip of B per workgroup

Two FlyDSL kernels for `out = a @ b.T` on 48 and 72 MiB weights, one per (N, K), each built for m 1, 2, 4, …, 128.
"m" is M. N 6144 × K 4096 runs a strip builder first written for N 6144 × K 16384 and extended here; N 6144 ×
K 6144 has a strip kernel of its own and, at m 64 and 128, a split-K kernel. The band kernels of the same sizes,
which split M × N × K differently, are in [k3_bf16_band_stream.md](k3_bf16_band_stream.md); the split-K folds of the
K 7168 kernels in [k3_bf16_splitk_wide_n.md](k3_bf16_splitk_wide_n.md).

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes: every reduction these kernels run adds in a fixed order, with no atomics on data.
- N = 6144 with K = 4096 or 6144: 48 or 72 MiB of `b`, against at most 1.5 MiB of `a` at these m (estimated from the
  shapes), so each kernel gives every CU its own strip of `b`.
- The yardstick is aiter's tuned bf16 GEMM, and a size counts as won from 1.05 over it: every size of both
  kernels reaches that in a separate run, 1.081–1.316 (§ Result).

## Counter-signals

- **m 64 and 128 at K 6144 take two launches.** On this kernel a one-launch fold, in which the last of the four
  splits to count in on a launch-tagged counter adds the planes, read 18.30 µs against 19.38 for the fold launch at
  m 64, and 24.05 against 23.82 at m 128 *(field, E4)*. A timer that sums the kernels of a call gives the same order at m 64 but reads
  the fold launch's short second kernel at its floor; on the wall time of back-to-back calls the one-launch fold was
  the slower at both m, 19.29 against 18.50 µs at m 64 *(field, E5)*. Its counter word came from a `torch.empty`
  workspace the call had not written, whose contents nothing defines, so both sizes take the second
  launch. At m 64 the one-launch fold measured 1.272 over aiter *(field, E2)*.
- **The thinnest margins** are K 6144 at m 32 (1.085) and K 4096 at m 128 (1.130) *(field, E1, E2)*.
- **Fallback.** N 6144 × K 4096 sends m > 128 to `torch.matmul`, N 6144 × K 6144 any m outside its tables; no
  measured size reaches either.

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

**N 6144 × K 4096**, four waves:
- The MFMA runs over 32 columns (48 at m 64, 128). With a 24-column strip, the second B DMA of each K tile covers
  rows 16–31 of the stage; rows 24–31, past the strip, read zero through an offset past the descriptor, and those
  columns are not stored.
- Ring: 12 slots at m ≤ 16, 10 at m 32 (BK 128), 4 at BK 256 (m 64), 5 at BK 128 (m 128).
- Workgroup t takes index kid = (t mod 8)·32 + t/8. At m ≤ 32 kid is the strip; at m 64 and 128 the strip is kid / 2
  and kid mod 2 its row half, so the two halves of a 48-column strip get ids 8 apart. The workgroups whose ids are
  equal mod 8, which run on one XCD, stream one contiguous slab of B. Workgroup kid starts its K walk at tile
  (kid / G)·R mod the tile count, with (G, R) = (32, 4), (64, 7), (32, 6), (32, 6), (8, 1), (32, 3) for m ≤ 4, 8, 16,
  32, 64, 128 (`../gemm/small_m.md` §6):

  ```python
  # workgroup t runs on XCD t mod 8: number the strips XCD-major, so each XCD streams one slab of B
  kid = fx.Int32(rocdl.readfirstlane(T.i32, _as_i32(t % _XCDS * (_CUS // _XCDS) + t // _XCDS)))
  # m <= 32: kid is the strip; m 64, 128: two workgroups own the two row halves of a strip
  if const_expr(MSPLIT == 1):
      n_off = kid * SW
      m_row = 0
  else:
      m_row = kid % MSPLIT * TILE_M
      n_off = kid // MSPLIT * SW
  # strip kid starts its K walk at tile ((kid // G) * R) mod the tile count
  k0 = rocdl.readfirstlane(T.i32, _as_i32(((kid >> G_SHIFT) * fx.Int32(ROT)) & fx.Int32(K_LOOPS - 1)))
  ```
- Fold: every thread adds two adjacent columns over the four waves in order and stores them as one 4 B word. The
  three pointer arguments are preloaded into SGPRs.

  ```python
  # after the K loop: each wave has written its fp32 partials to LDS scratch [wave][row][column]
  tot = None
  for w in range_constexpr(_WAVES):  # wave 0, 1, 2, 3: the same order in every run
      off = ((fx.Index(w * TILE_M) + fx.Index(orow)) * C_STRIDE + fx.Index(ocol)) * 4
      v = lds_load(lds_at(lds_base, off), T.vec(2, T.f32), alignment=8)
      tot = v if tot is None else std_arith.AddFOp(tot, v).result
  outv = std_arith.TruncFOp(T.vec(2, T.bf16), tot).result  # round once, after the sum
  ```

  Not this: the waves adding their partials into `out` as they arrive, with an atomic add. The sum then depends
  on the arrival order, and two runs need not give the same bytes (`../gemm/correctness.md` §8).

**N 6144 × K 6144**:
- m ≤ 32, a strip kernel of eight waves, each taking an eighth of every K tile; BK 256 with 10 slots (m ≤ 8), 512 with 4
  (m 16), 256 with 4 (m 32). It differs from the K 4096 strip in three ways: workgroup b takes strip b (no
  renumbering); each B stage holds only the 24 strip rows, so MFMA columns 24–31 read the next stage's bytes and are
  never stored; and when the last DMA of a stage covers fewer than all eight waves' worth of chunks, only the first
  waves issue it. Workgroups 8r to 8r + 7 start on K tile r mod the tile count (`bid >> 3`), meant to keep the B
  stream from moving in lockstep. Each thread adds one output over the eight waves in order and stores 2 bytes.
- m 64 and 128, a split-K kernel: 256 workgroups = 64 column groups of 96 columns × 4 K slices of 1536; eight waves as
  4 (rows) × 2 (columns), so no wave splits K inside the workgroup; BK 128 with 3 slots (m 64), 64 with 5 (m 128).
  Workgroup b takes K slice (b mod 8) / 2, meant to keep a slice's workgroups on one pair of XCDs (the code calls it
  a placement hint). A workgroup gathers its waves' sub-tiles through padded LDS and writes them, at the default
  cache policy, as its fp32 partial plane to the call's `torch.empty` workspace, which holds the four planes only.
- m 64 and 128, second launch: a fold kernel adds the four planes in split order and rounds once, 16 rows × 96
  columns per workgroup (256 workgroups at m 64, 512 at m 128); every element it reads was written by the first
  launch of the same call.
- No kernel-argument preload.

## Result

The speedup is the one-sided lower bound of the aiter / kernel time ratio over 16 rounds, each timing the kernels
of CUDA-graph replays with L2 flushed before every call, against aiter's tuned `gemm_a16w16`, on one MI355X with
FlyDSL 0.3.2. N 6144 × K 6144 was measured on the build before the fold launch, whose code differs only at m 64
(§ Counter-signals). A separate run read 1.204–1.316 at K 4096 and 1.081–1.300 at
K 6144 (m 64, the fold launch, 1.117): all 16 sizes at or above 1.05.

| Kernel | Sizes | Speedup over aiter | Highest |
|---|---|---|---|
| N 6144 × K 4096 *(field, E1)* | m 1–128 | 1.130–1.290 | 1.290 at m 4 |
| N 6144 × K 6144 *(field, E2)* | m 1–32, 128 | 1.085–1.309 | 1.309 at m 128 (split-K, two launches) |

Every size passed the accuracy check. Kernel times at K 6144 ran from 12.91 µs (m 1) to 24.23 µs (m 128) *(field, E2)*.
Under stress every size gave the output of the call alone, K 6144 on the earlier build *(field, E3)*.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] N 6144 × K 4096: 1.278, 1.287, 1.290, 1.225, 1.286, 1.251, 1.218, 1.130 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E1; times are the median of a size's per-round times] N 6144 × K 6144 on the build before the fold launch, the same code but for a one-launch fold on a launch-tagged counter at m 64: 1.166, 1.106, 1.109, 1.114, 1.194, 1.085, 1.272, 1.309 at m 1, 2, 4, 8, 16, 32, 64, 128; kernel 12.91 µs at m 1, 24.23 at m 128; accuracy passed
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] N 6144 × K 4096 and the N 6144 × K 6144 build of E2, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ; a full model's decode graph was not run
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=?, the tuning run's interleaved A/B, timer not stated in the record] N 6144 × K 6144: the one-launch tagged fold with batched loads and its own quarter from registers 18.30 µs against 19.38 for the fold launch at m 64, 24.05 against 23.82 at m 128
- E5 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=10 rounds in one process, per-kernel times of CUDA-graph replays summed per call, and the wall time per call of one CUDA graph of 525 back-to-back calls] the N 6144 × K 6144 build of E2, its one-launch tagged fold against the fold launch, outputs bitwise equal: per-kernel 18.34 against 19.46 µs at m 64, the fold launch's second kernel 4.43 µs against 4.39 for a one-element `zero_()`; wall time 19.29 against 18.50 µs at m 64, 26.42 against 25.07 at m 128
