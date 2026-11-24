---
title: "bf16 GEMM, short K or large M only (Kimi-K3: N 1536 × K 128, N 163840 × K 256, N 2304 × K 1536, N 3072 × K 512) — the whole K in one tile or one workgroup's waves, persistent grids at the largest M; N 7168 and 3584 × K 512 at m 2–128, one of them a copy of aiter's FlyDSL HGEMM"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-14", "M 2-128", "M 893-16384", "M 35107-342073"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.023–1.716; 12 of 17 sizes ≥ 1.03 (not m 8, 12, 16, 32, 64); 7 of 17 in a repeat measurement", gpu: MI355X, workload: "N 1536 × K 128, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.527–1.628; 11 of 11 sizes ≥ 1.03; 11 of 11 in a repeat measurement", gpu: MI355X, workload: "N 163840 × K 256, m 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14", date: "2026-10-25"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.087–1.322; 4 of 4 sizes ≥ 1.03; 4 of 4 in a repeat measurement", gpu: MI355X, workload: "N 2304 × K 1536, m 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.113–1.294; 3 of 3 sizes ≥ 1.03; 3 of 3 in a repeat measurement", gpu: MI355X, workload: "N 3072 × K 512, m 35107, 104093, 342073", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, one-sided lower bound of the time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, per size", value: "1.092–1.353 at m 2–64; m 1 (1.022) and 128 (0.978) left out of the kernel's sizes; all 6 at or above 1.03 in a separate run", gpu: MI355X, workload: "N 7168 × K 512, the same kernel measured at m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, one-sided lower bound of the time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, per size", value: "1.181–1.303 at m 32–128; m 1–16 (1.020–1.036) left out of the kernel's sizes; all 3 at or above 1.03 in a separate run", gpu: MI355X, workload: "N 3584 × K 512, measured on a build that imported the same HGEMM from aiter 0.1.21.dev66+gdbd8bf5bd, at m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
---

# bf16 GEMM at short K or large M only (Kimi-K3): no K loop to split, a persistent grid where M is large

Four of this page's six FlyDSL kernels for `out = a @ b.T` are the Kimi-K3 set, one kernel per (N, K), each
built for a list of m. "m" is M. n163840, n2304 and n3072 below are short for N 163840 × K 256, N 2304 × K 1536
and N 3072 × K 512.
None of the four splits K across workgroups. n2304's builder and n3072's builder and tile plan also run
N 3584 × K 512 and N 28672 × K 512 at m 256–4096: [bf16_large_m_whole_k_tiles.md](bf16_large_m_whole_k_tiles.md).

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes, and the result is held to the error bound of an fp32 dot product over K.
  Every output is one fp32 sum in a fixed order.
- Shapes:
  - N 1536 × K 128 (a 384 KiB weight) at m 8–16384: K fits one LDS tile.
  - N 163840 × K 256 (80 MiB) at m 1–14: a read-once weight over a very wide N.
  - N 2304 × K 1536 at m 893–16384 and N 3072 × K 512 at m 35107–342073: only large-M sizes, compute-bound
    tiles; the second has 8 K blocks of 64.
- At every size the yardstick is aiter's tuned bf16 GEMM, and a size counts as won from 1.03 over it.
- **25 of their 35 sizes read at or above 1.03** in a repeat measurement; the count per kernel is in § Result.

## Counter-signals

- **N 1536 × K 128 at m ≤ 112 sits on the threshold.** Its 13 sizes there read 1.023–1.038 in the latest one-run
  measurement, 5 of them below 1.03 *(field, E1)*; the same code read 1.010–1.038 in an earlier run and 1.021–1.123
  in a third run's two arms *(field, E2)*. Repeat measurements move them too: 10 of the 13 read at or above 1.03
  in one, 3 (m 12, 16, 60) in a later one *(field, E17)*. These sizes run a 16-column,
  one-wave tile on 96 workgroups for 256 CUs (code read).
- **N 1536 × K 128: a retune of the three largest m buckets did not pay.** Run beside the final kernel in one
  window, it reached a lower bound of 1.02 at 14 and 12 of 17 sizes against the final
  kernel's 14 and 16 in two runs *(field, E11)*; the paired ablation bench still scores that version as the best.
- **n2304's two-tile walk requires `m % 256 == 0`** (asserted when the persistent kernel is built); m 16384 is the
  only size that takes it. n3072 sizes its persistent grid from the CU count of the current device at compile time
  and starts half of each XCD's workgroups late only up to m 200000, so m 342073 runs without
  the delay.
- **A configuration no measured size reaches:** N 1536 × K 128's table row 2731 (no size between 893 and 2732).
- Nothing these kernels compute crosses a workgroup, so there are no flags, counters or workspace: no state lives
  between calls, and no reduction depends on arrival order (code read).

## Mechanism

Shared by the four kernels:

- **MFMA `mfma_f32_16x16x32_bf16` with B as the first operand**, so a lane holds four consecutive output columns
  of one row (`../gemm/epilogue.md` §3), fp32 accumulators.
- **Kernel pointer arguments preloaded into SGPRs, except in N 3072 × K 512:** all three in N 1536 × K 128 and
  N 163840 × K 256 (`amdgpu-kernarg-preload-count` 6); in N 2304 × K 1536 only the first (2 SGPRs, though its
  count is 16: each pointer is followed by a 16-byte by-value struct) *(field, E18)* (`../gemm/small_m.md` §6).
- **A per-m configuration** compiled in a `functools.cache` function on a size's first call.

Per kernel:

**N 1536 × K 128** (one tiled builder at every m).
- The whole K = 128 is one LDS tile: one round of buffer→LDS DMAs for A and B (`buffer_load_dwordx4 … lds`,
  XOR-swizzled 16-byte chunks), one `s_waitcnt vmcnt(0)` + barrier, four MFMA K steps; no ring, no K loop
  (`../gemm/datapath.md` §2, §5).
- m ≤ 112: a tile of T × 16 with T the m bucket (16 … 112), one wave, 96 workgroups, B `nt`.
- m 893: 32×192, 4 waves, grouped XCD order (groups of 4 block rows), 224 workgroups. m 2732: 64×96, 4 waves,
  groups of 8, 688 workgroups. m 9397: 128×128, groups of 16, 888 workgroups. m 16384: 128×192, groups of 8, 1024
  workgroups.
- Stores: C staged through LDS and written as 16-byte rows with `sc1`; at m 9397 and 16384 B's rows are fetched
  permuted so a lane's two adjacent 16-column fragments are contiguous output columns, and C goes out as 16-byte
  stores straight from registers.

**N 163840 × K 256** (one skinny builder; m ≤ 16 asserted).
- One 16-row MFMA block (all m rows) by 16 or 32 columns per workgroup; the waves split K and an LDS fold sums
  their partials in wave order:
  - m 1: 4 waves, each a quarter of K, 16 columns per workgroup, 10240 workgroups;
  - m 2, 3: 8 waves, each an eighth of K, 32 columns, 5120 workgroups;
  - m 4–14: 8 waves as 2 column halves × 4 K quarters, 32 columns, 5120 workgroups.
- B goes straight from global into registers, every K step's load issued before the first MFMA; each B element
  feeds one wave, so LDS would add no reuse. A goes to registers directly at m ≤ 3 and through LDS (row stride
  padded by 8 elements) from m 4.
- Adjacent column tiles are dealt to one XCD in groups of 8, so a C row's neighbouring tiles
  finish in one L2 (`../gemm/small_m.md` §6); C leaves as 16-byte row stores from m 2.

  ```python
  _XCDS = 8
  _GROUP = 8  # adjacent column tiles per XCD: neighbouring tiles finish C rows in one L2

  def _grouped_tile(n_tiles):
      """Column tile of this workgroup: blocks are dealt to the XCDs round robin."""
      span = _XCDS * _GROUP
      assert n_tiles % span == 0
      bid = fx.block_idx.x
      # workgroups bid, bid + 8, ... run on XCD bid % 8: give them tiles next to one another
      return (bid // span) * span + (bid % _XCDS) * _GROUP + (bid // _XCDS) % _GROUP
  ```

**N 2304 × K 1536** (one builder, per-m keyword table).

| m | Tile, waves | Stages, K tile | Workgroups | Also |
|---|---|---|---|---|
| 893 | 64×128, 8 (2×4) | 3, 128 | 252 | grouped XCD order (groups of 7 block rows), fragment prefetch, K loop unrolled, C `sc1` |
| 2732 | 144×192, 12 (3×4) | 3, 64 | 228 | groups of 7, refills issued inside the compute after the first fragment reads, unrolled, C `nt` |
| 9397 | 448×192, 8 (4×2) | 2, 64 | 252, one round | C staged and stored in two passes, `sc1`; LLVM scheduler `iterative-ilp` |
| 16384 | 256×288, 8 (4×2) | 2, 64 | 256 for 512 tiles | two tiles per workgroup: the second walks K in reverse, starting on the slices the first just left in L2; tile 0's C leaves through stage 1 while stage 0 loads tile 1; scheduler `iterative-minreg`; C `sc1` |

Grids sized to whole rounds of the 256 CUs (`../gemm/grid_pricing.md` §1, §10); the scheduler
strategy goes in through `amdgpu-sched-strategy` (`../gemm/schedule_and_locality.md` §3).

**N 3072 × K 512** (one configuration for every m).
- 256×256 tiles, 8 waves (2×4), two LDS stages of K 64, grouped order in groups of 2 block rows, `waves_per_eu` 2.
- A persistent grid of G = 256 workgroups (from the device's CU count) walks head tiles pid, pid + G, … over
  the first P whole 256-row panels, then the remaining rows as 128-row tail items, handed first to the workgroups
  with one head tile fewer:
  - m 35107: P 128 panels, 1536 head tiles (6 per workgroup) and 228 tail items;
  - m 104093: P 405, 4860 head tiles (18 or 19 per workgroup) and 48 tail items;
  - m 342073: P 1337, 16044 tiles (62 or 63 per workgroup), no tail.

  ```python
  # TOTAL head tiles, T_ITEMS tail items; T_EXTRA = TOTAL % G workgroups run one head tile more
  if const_expr(TM_T):
      # This WG's first tail item; WGs with one head tile fewer come first.
      q = (fx.Int32(pid) + (G - T_EXTRA)) % G
      has_tail = q < T_ITEMS
      tail_trips = has_tail.select(
          (fx.Int32(T_ITEMS - 1) - q) // G + 1, fx.Int32(0)
      )
      tm0, tn0 = tail_mn(has_tail.select(q, fx.Int32(T_ITEMS - 1)))

  trips = (fx.Int32(TOTAL - 1) - fx.Int32(pid)) // G + 1
  for j in range(trips):  # head tiles pid, pid + G, ...
      w = fx.Int32(pid) + fx.Int32(j) * G
      wn = w + G
      in_head = wn < TOTAL
      wn = in_head.select(wn, fx.Int32(TOTAL - 1))
      m_off, n_off = tile_mn(w)
      nm_off, nn_off = tile_mn(wn)  # the item whose first two K blocks this tile fills on its way out
      if const_expr(TM_T):
          nm_off = in_head.select(nm_off, tm0)
          nn_off = in_head.select(nn_off, tn0)
      run_head(m_off, n_off, nm_off, nn_off)
  if const_expr(TM_T):
      for i in range(tail_trips):  # tail items q, q + G, ...
          t = q + fx.Int32(i) * G
          tn = t + G
          tn = (tn < T_ITEMS).select(tn, fx.Int32(T_ITEMS - 1))
          m_off, n_off = tail_mn(t)
          nm_off, nn_off = tail_mn(tn)
          run_tail(m_off, n_off, nm_off, nn_off)
  ```
- No workgroup barrier between items: each K block's barrier sits mid-block, the block after next is filled behind
  it, and each wave stages its own C rows (32 per pass) through a private LDS area and stores them `nt`.
- At m ≤ 200000, odd workgroups of each XCD sleep before their first fill, so the two halves reach their
  epilogues out of phase (`../gemm/grid_pricing.md` §10).

**What each part bought** (ablations; each kept step timed beside the version before it):

- N 1536 × K 128: the single-pass kernel with per-m tiles and preloaded pointers +0.6–4.4% at m ≤ 112 and
  +22.3–42.4% at m ≥ 893 over the starting kernel; `sc1` C stores +3.1–5.0% at m ≤ 112 and +8.3–20.3%
  above *(field, E6)*; the register-direct C store +5.9% (16384) and +3.8% (9397) *(field, E7)*.
- N 163840 × K 256: the K-split-over-waves port +18.8–41.3%; kernel-argument preload +1.6–3.0% at every size; the
  XCD grouping +0.0–1.6%; the two-half workgroup for m ≥ 4 +0.1–2.6% at m 4–14. Against the starting kernel the
  final one is +49.7–62.7% *(field, E8)*.
- N 2304 × K 1536: grouping, vector C staging and `nt` C +2.5–8.1%; BK 128 with 3 stages +6.2% at m 893; the
  unrolled K loop +1.2% / +3.2% at m 893 / 2732; the scheduler strategy +2.3% / +0.7% at m 9397 / 16384; the
  persistent two-tile walk +0.9% at m 16384. Against the starting kernel: +9.5–23.2% *(field, E9)*.
- N 3072 × K 512: the 256×256 port with grouped order and `nt` C +6.4–16.7%; the persistent loop with per-wave C
  +5.3–10.0%; tail items +2.3% at m 35107; the start delay +2.9% / +1.1% at m 35107 / 104093; the mid-block barrier
  with fragment prefetch and the fill of block kb+2 behind it, one step, +0.6–2.2%. Against the starting kernel:
  +13.8–31.5% *(field, E10)*.

## Result

The last column counts the sizes at or above 1.03 in a repeat measurement: the one-sided lower bound of the aiter /
kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, and for n2304 and n3072
earlier repeats whose ruler is not stated *(field, E17)*. The repeats ran FlyDSL 0.3.2, as the one-run
numbers below do, not their container image's 0.3.4.1: aiter dbd8bf5, installed over the image for these runs, pins
`flydsl==0.3.2` (read from the runs' setup, not from a run's log; `../pitfalls/measured_a_different_build.md`).

The other columns are one run each on MI355X, FlyDSL 0.3.2, of a lower bound of the CUDA-graph time ratio
aiter / kernel; "≥ 1.03 in one run" counts the sizes at or above 1.03 in it.

| Kernel | Sizes | Lower bound per size | ≥ 1.03 in one run | ≥ 1.03 in the repeat |
|---|---|---|---|---|
| N 1536 × K 128 *(field, E1)* | m 8–112 | 1.023–1.038 | 8 of 13 | 3 of 13 |
| | m 893, 2732, 9397, 16384 | 1.243, 1.716, 1.432, 1.435 | 4 of 4 | 4 of 4 |
| N 163840 × K 256 *(field, E3)* | m 1–14 | 1.527–1.628 | 11 of 11 | 11 of 11 |
| N 2304 × K 1536 *(field, E4)* | m 893, 2732, 9397, 16384 | 1.322, 1.165, 1.153, 1.087 | 4 of 4 | 4 of 4 |
| N 3072 × K 512 *(field, E5)* | m 35107, 104093, 342073 | 1.294, 1.248, 1.113 | 3 of 3 | 3 of 3 |

## N 7168 and 3584 × K 512 at m 2–128: all of K in one workgroup, at decode sizes

Two later kernels at small m: N 7168 × K 512 (m 2–64, 7 MiB of weight)
and N 3584 × K 512 (m 32–128, 3.5 MiB). The yardstick is aiter's tuned bf16 GEMM: all of them read at or above 1.03
over it, in a separate run too *(field, E12, E13, E20)*.
Like the kernels above, these two do not split K across workgroups; unlike n163840's, they do not split it
across waves either: each workgroup reads its rows of A and B over all of K once, and every output is one MFMA chain
in K order. N 7168 at m 1 (1.022) and m 128 (0.978) and N 3584 at
m ≤ 16 (1.020–1.036) are left out of the kernels' sizes: within about 4% of aiter or slower *(field, E12, E13)*.

**N 7168 × K 512**, one launch:
- Each workgroup DMAs all of K for its rows of A and B into LDS before computing: 8 chunks of 64, each chunk one group
  of `buffer_load_dwordx4 … lds` (64, 80 or 96 KiB of LDS). Chunk c then waits (`s_waitcnt vmcnt`, `s_barrier`)
  only until its own DMAs have landed, and its MFMAs run while the later chunks land (`../gemm/datapath.md` §5).

  ```python
  for c in range_constexpr(CHUNKS):  # every chunk's DMAs are issued before the first MFMA
      base = c * CHUNK_BYTES
      for p in range_constexpr(A_DMAS):
          issue(a_rsrc, base + p * DMA_PER_PASS, a_goff[p] + c * ROW_BYTES)
      for p in range_constexpr(B_DMAS):
          issue(b_rsrc, base + A_BYTES + p * DMA_PER_PASS, b_goff[p] + c * ROW_BYTES)

  def wait_barrier(cnt):
      llvm.InlineAsmOp(
          None, [], f"s_waitcnt vmcnt({cnt})\n\ts_barrier", "~{memory}", has_side_effects=True
      )

  for c in range_constexpr(CHUNKS):
      rocdl.sched_barrier(0)
      wait_barrier((CHUNKS - 1 - c) * PER_CHUNK)  # the loads of the chunks after c may still be in flight
      rocdl.sched_barrier(0)
      fr = [
          ([frag(b_fo[h][j], c) for j in range(N_STEPS)], [frag(a_fo[h][i], c) for i in range(M_STEPS)])
          for h in range(2)
      ]
      for h in range_constexpr(2):
          bf, af = fr[h]
          for i in range_constexpr(M_STEPS):
              for j in range_constexpr(N_STEPS):
                  acc[i * N_STEPS + j] = rocdl_d.mfma_f32_16x16x32_bf16(
                      T.f32x4, bf[j], af[i], acc[i * N_STEPS + j], 0, 0, blgp
                  ).res
  ```

  Not this: K split across workgroups to fill the CUs this grid leaves idle at the smallest m. On another
  kernel a split of the same K cost time (Counter-signals below).
- LDS rows are XOR-swizzled in 16 B slots with one key per row pair (`(row / 2) mod 8`), so that one
  `ds_read_b128` pass covers every bank (`../gemm/datapath.md` §2).
- The waves split M × N, not K: 16×64 tiles with two waves along N (m ≤ 16, 112 workgroups), 32×32 with 2 × 2 waves
  (m 32, 224), 32×64 with two waves along N and two M-blocks (m 64, 224).
- `mfma_f32_16x16x32_bf16` with B as the first operand, so a lane holds 4 adjacent output columns. At m ≤ 16 and 64,
  B's LDS rows are filled from permuted columns so that a lane's two N fragments hold 8 adjacent columns, stored as
  one 16 B write; at m 32, 8 B writes (`../gemm/epilogue.md` §3). C goes out with `sc1`; rows past m fall outside C's
  descriptor and are dropped, and A rows past m read row 0.

  ```python
  # lrow: the LDS row of B's tile this DMA fills; tcol: the column of the tile (row of b) it fetches
  if const_expr(PAIR):
      # Within each 32-row block, LDS row 16j+4q+v holds column 8q+4j+v, so a lane's two
      # N fragments hold 8 adjacent C columns.
      x = lrow % 32
      tcol = (lrow - x) + ((x // 4) % 4) * 8 + (x // 16) * 4 + x % 4
  else:
      tcol = lrow
  b_goff.append(((bn * TILE_N + tcol) * k + (dslot ^ _swz(lrow)) * 8) * 2)
  ```

**N 3584 × K 512**: aiter's FlyDSL HGEMM at split-K 1, from copies of aiter's FlyDSL files at
dbd8bf5bd kept beside the kernel, with a 39-line wrapper that chooses the tile.
- A `functools.cache` compile function calls aiter's `compile_hgemm_kernel("bf16", n, k, …)` with
  the tile for m, and the launch runs it through aiter's `_run_compiled` on the current stream. It passes a null
  pointer in the bias, semaphore and signal slots: the kernel binds the semaphore and signal only with split-K > 1
  and the bias only with `HAS_BIAS`, and the wrapper sets neither (S8).
- The copies are `splitk_hgemm.py`, `tensor_shim.py` (only the definitions the HGEMM and the launch use),
  `buffer_ops.py` and `vector.py`; they import one another relatively and nothing from aiter. Beyond the imports,
  two host-side lines differ from aiter, as they do in the copies a SiTUv2 MoE kernel carries: `buffer_ops.py` does
  not read `FLYDSL_GPU_ARCH`, and `tensor_shim.py` builds its default strides without `tolist`.
- Tiles: 32×32 with K tiles of 256 and two waves along N (m 32 and 64, 112 and 224 workgroups); 32×64 with K tiles of
  128, four waves along N, three stages and B through LDS as well (m 128, 224 workgroups). Below m 128 only A goes
  through LDS. Everything inside the kernel is aiter's code at dbd8bf5bd; its knobs are in
  `../flydsl/kernels/flydsl_knob_space.md`.

Counter-signals:
- **N 3584 keeps aiter's HGEMM as of dbd8bf5bd.** It was measured importing that code from aiter
  0.1.21.dev66+gdbd8bf5bd; the kernel now
  carries the copies, so a later aiter changes neither its import nor its kernel, and a later fix to aiter's HGEMM
  reaches it only through a new copy.
- **The baseline is aiter's own dispatch.** Its tuned `gemm_a16w16` picks its own kernel at these
  shapes; which one it ran was not recorded, so the gain is this tile choice against whatever aiter's dispatch chose.
- **Near 4.2–4.4 µs the profiled per-kernel reading stops moving with m; the N 7168 kernel does not.** N 7168 read
  4.22–4.30 µs per round at every measured m from 2 to 64 while aiter went from 4.64 to 5.77, 0.41–0.58 µs above the
  kernel at m 2–16; N 3584 read 4.16–4.32 at m 1–16 against aiter's 4.32–4.46, about a tenth of a microsecond apart
  *(field, E14, E15)*. That is the profiled timer's floor, which flattens short times: same-grid spin kernels of
  1.8–4.4 µs read
  4.52–4.57 µs. Timed back to back with B cold, the N 7168 kernel goes from 3.63 µs per call at m 2 to 4.34 at m 64,
  and aiter is 0.81–0.89 µs slower at every m from 2 to 16 *(field, E19)*, so there the gain is not aiter's time
  growing with m (`../pitfalls/profiled_timer_floor.md`).
- **No K split at K 512.** Splitting K = 512 to fill the grid cost 15–18% on another kernel (`../gemm/small_m.md`
  §4), so at m ≤ 16 N 7168 runs 112 workgroups, under half a round.

The one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed
before each call, on one MI355X: N 7168 × K 512 1.092–1.353 over aiter,
highest at m 64 *(field, E12)*; N 3584 × K 512 1.181–1.303, highest at m 64, measured on the build that imported the
HGEMM from aiter *(field, E13)*. Under stress every size gave the output of the call alone *(field, E16)*.

## Evidence

"One-run ratio bound" is a lower bound of the CUDA-graph time ratio aiter / kernel per size, from one run on
MI355X.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] N 1536 × K 128: m 8 1.025, 12 1.023, 16 1.024, 24 1.031, 32 1.023, 40 1.032, 48 1.030, 56 1.038, 60 1.037, 64 1.024, 80 1.033, 96 1.033, 112 1.032, 893 1.243, 2732 1.716, 9397 1.432, 16384 1.435
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] N 1536 × K 128 at m ≤ 112: 1.010–1.038 in the run that accepted its first version; 1.048–1.123 and 1.021–1.103 for that version and the large-m rework side by side in another run; the final ISA of the two was compared at m ≤ 893
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] N 163840 × K 256: m 1 1.628, 2 1.588, 3 1.591, 4 1.573, 5 1.575, 6 1.573, 7 1.569, 8 1.566, 10 1.562, 12 1.551, 14 1.527
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] N 2304 × K 1536: m 893 1.322, 2732 1.165, 9397 1.153, 16384 1.087
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] N 3072 × K 512: m 35107 1.294, 104093 1.248, 342073 1.113
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it, as recorded] N 1536 × K 128: single-pass kernel +2.0, 0.6, 1.2, 1.8, 1.6, 2.1, 2.1, 4.4, 3.5, 2.7, 2.8, 2.9, 3.5% at m 8–112, +22.3, 42.4, 26.4, 25.4% at 893, 2732, 9397, 16384; `sc1` C +3.1–5.0% at m ≤ 112, +10.0, 20.3, 8.7, 8.3% above
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3, paired bench of each step against the version before it] N 1536 × K 128 C from registers with permuted B rows: +5.9% at m 16384, +3.8% at 9397, +1.3% at 2732
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] N 163840 × K 256: port +18.8–41.3% (recorded); preload +1.6–3.0% (recorded); XCD grouping +0.0–1.6%; two-half workgroup +0.1–2.6% at m 4–14; all changes off +49.7–62.7%
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] N 2304 × K 1536: grouping + vector C staging + `nt` C +7.9, 8.1, 2.5, 2.6% (recorded); BK 128 +6.2% at 893; unroll +1.2 / +3.2% at 893 / 2732; scheduler strategy +2.3 / +0.7% at 9397 / 16384; persistent +0.9% at 16384; all changes off +23.2, 15.8, 15.6, 9.5% at m 893, 2732, 9397, 16384
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] N 3072 × K 512 at m 35107 / 104093 / 342073: port +9.2 / 16.7 / 6.4% (recorded); persistent loop +10.0 / 7.3 / 5.3% (recorded); tail items +2.3 / 0.1 / 0.0%; start delay +2.9 / 1.1 / 0.0%; mid-block barrier + fragment prefetch + kb+2 fill +0.6 / 2.2 / 1.1%; all changes off +28.2 / 31.5 / 13.8%
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=2, one-run ratio bound, one window per run] N 1536 × K 128, final version beside the dropped last step: 14 and 16 of 17 sizes at a lower bound of 1.02 or more against 14 and 12
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] the N 7168 × K 512 kernel, built for m 1 and 128 as well: 1.022, 1.100, 1.092, 1.108, 1.132, 1.239, 1.353, 0.978 at m 1, 2, 4, 8, 16, 32, 64, 128; m 128 judged not faster than aiter; accuracy passed
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E12; 32 rounds at m 2] an earlier build of the N 3584 × K 512 kernel (built for m 1–16 as well; its wrapper imported the HGEMM from that aiter and passed B's pointer in the bias, semaphore and signal slots, where the final kernel carries copies of aiter's files and passes a null pointer): 1.036, 1.020, 1.026, 1.030, 1.031, 1.181, 1.303, 1.286 at m 1, 2, 4, 8, 16, 32, 64, 128; m 2 judged not faster than aiter; accuracy passed
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E12] per-round times, kernel against aiter: m 1 4.22–4.28 against 4.32–4.37 µs; m 2 4.22–4.25 against 4.64–4.70, m 4 4.25–4.30 against 4.66–4.72, m 8 4.26–4.30 against 4.72–4.78, m 16 4.25–4.28 against 4.80–4.86, m 32 4.23–4.26 against 5.25–5.31, m 64 4.23–4.26 against 5.73–5.77; m 128 4.86–4.92 against 4.78–4.84
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E13] per-round times, kernel against aiter: m 1–16 4.16–4.32 against 4.32–4.46 µs; m 32–128 4.29–4.40 against 5.08–5.67
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] N 7168 × K 512 and the N 3584 × K 512 build of E13 (at m 32–128), every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E17 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of aiter dbd8bf5, installed over the image for these runs; read from the runs' setup, not from a log), N=1, repeat measurements; the latest on the ruler of E12] sizes at or above 1.03 in the latest repeat: N 1536 × K 128 7 of 17 (m 12, 16, 60, 893–16384; m 8–112 0.964–1.034), N 163840 × K 256 11 of 11; N 2304 × K 1536 4 of 4 and N 3072 × K 512 3 of 3 from earlier repeats, ruler not stated; N 1536 × K 128 in an earlier repeat: 14 of 17, m 8–112 1.018–1.037, not m 24, 48, 80
- S8 ROCm/aiter@dbd8bf5bd:aiter/ops/flydsl/kernels/splitk_hgemm.py:1094 `launch_hgemm_kernel(C, A, B, BIAS, m, semaphore, signal, stream)`; semaphore and signal bound only under `IS_SPLIT_K` at :342, the bias only under `HAS_BIAS` at :263 and :1083; without `B_TO_LDS` only A has LDS storage at :197-209
- E18 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1, one compile per kernel; `.amdhsa_user_sgpr_kernarg_preload_length` read from the final ISA and from the code object's kernel descriptor, which agree] SGPRs preloaded: N 1536 × K 128 6 and N 163840 × K 256 6, all three pointers of a 24-byte kernarg; N 2304 × K 1536 2 of a 72-byte kernarg with `amdgpu-kernarg-preload-count` 16; N 3072 × K 512 0 (no count set)
- E19 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, FlyDSL 0.3.2, aiter dbd8bf5, N=8 rounds, medians; the profiled per-kernel timer of CUDA-graph replays (L2 flushed per call) and back-to-back calls in one CUDA graph rotating 96 operand copies (704 MB of B)] N 7168 × K 512 back to back 3.625 µs at m 2 to 4.341 at m 64, 4.330–4.398 under the profiled timer; same-grid spin kernels of 1.82–4.40 µs back to back read 4.52–4.57 µs under the profiled timer; aiter `gemm_a16w16` 0.81–0.89 µs slower than the kernel back to back at m 2–16; every output within tolerance
- E20 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 7168 × K 512: 1.082, 1.089, 1.101, 1.120, 1.223, 1.349 at m 2, 4, 8, 16, 32, 64, all 6 sizes at or above 1.03; N 3584 × K 512: 1.139, 1.263, 1.237 at m 32, 64, 128, all 3 at or above 1.03; accuracy passed at each
