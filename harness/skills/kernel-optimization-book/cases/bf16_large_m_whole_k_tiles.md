---
title: "bf16 GEMM at m 256–4096, N 2048–28672 × K 512–2048 — whole-K MFMA tiles fed by LDS DMA rings on one round of 224–256 workgroups, persistent where the tiles outnumber the CUs, no split-K; N 6144 × K 1536 stops at m 1024"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 256-1024", "M 2048-4096"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.041–1.231; all 5 sizes at or above 1.03 (m 4096 1.041)", gpu: MI355X, workload: "N 2048 × K 2048, measured on the build before the current one, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.358–1.625; all 5 sizes at or above 1.03", gpu: MI355X, workload: "N 3584 × K 512, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.164–1.816; all 5 sizes at or above 1.03", gpu: MI355X, workload: "N 7168 × K 512, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.173–1.452; all 5 sizes at or above 1.03", gpu: MI355X, workload: "N 28672 × K 512, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, per size", value: "1.156–1.257 at m 256–1024; m 2048 (0.999) and 4096 (0.982) left to aiter", gpu: MI355X, workload: "N 6144 × K 1536, measured on an earlier build with the same kernel code, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
---

# bf16 GEMM at m 256–4096: whole-K tiles on one round, persistent past it

Five FlyDSL kernels for `out = a @ b.T` at m 256–4096, one per (N, K). "m" is M.
Each reuses the builder of an earlier kernel: N 2048 × K 2048 that of N 512 × K 4096
([bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md)); N 3584 × K 512 that of N 2304 × K 1536 and
N 28672 × K 512 the builder and the tile plan of N 3072 × K 512
([k3_bf16_short_k_large_m.md](k3_bf16_short_k_large_m.md)); N 7168 × K 512 the two builders of
N 7168 × K 768, and N 6144 × K 1536 the tiled builder of N 7168 × K 1536
([k3_bf16_band_stream.md](k3_bf16_band_stream.md)), aiter's async-LDS HGEMM at split-K 1.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes:
  each output is one MFMA chain over K in a fixed order (at N 2048 × K 2048, m 256, plus a fixed-order fold of four
  K waves); no split-K across workgroups, no atomics.
- Weights: N 2048 × K 2048 (8 MiB), N 3584 × K 512 (3.5 MiB), N 7168 × K 512 (7 MiB), N 28672 × K 512 (28 MiB),
  N 6144 × K 1536 (18 MiB). K is left whole: from m 256 the output tiles alone fill a round of the GPU.
- Sizes: m 256, 512, 1024, 2048, 4096 in four kernels; m 256, 512, 1024 at N 6144 × K 1536, whose m 2048 and 4096
  are left to aiter (no faster than aiter, § Counter-signals).
- The yardstick is aiter's tuned bf16 GEMM, and a size counts as won from 1.03 over it. All 23 sizes of the five
  kernels reach that in the run of § Result and in a separate run, 1.034–1.798 in the latter *(field, E10)*;
  N 2048 × K 2048 at m 4096 (§ Counter-signals) is the lowest in both.

## Counter-signals

- **N 6144 × K 1536 is not covered at m 2048 and 4096.** At m 2048 the kernel calls `torch.matmul`, which read
  0.999 against aiter, the same time per round (35.20–35.28 against 35.19–35.28 µs) *(field, E5, E6)*. No FlyDSL
  configuration built for that m beat `torch.matmul`: nine of them (six tile shapes) read 38.35–55.89 µs against
  33.97–34.07, and the best later row 36.91 against 34.08 *(field, E7)*. The m 4096 row (256×384 tiles,
  `mfma_f32_32x32x16_bf16`, C staged in two halves with each wave's first 32-row MFMA atom row stored early) read 0.982
  *(field, E5)*. Neither size is kept; the m 4096 configuration stays in the kernel's table, and no size the kernel
  is built for reaches it.
- **N 3584 at m 256 runs separate A and B ring depths, which gave wrong output under stress in an earlier build.**
  That build of the configuration (64×64, BK 64, A 5 and B 4 slots, C staged through LDS) gave an output that
  differed from the call alone on a
  stream limited to 16 CUs, and a probe of 20 runs on each of a 16-CU and a 64-CU stream gave 2 bad results, with and
  without the row guard; the symmetric ring, separate depths on another tile, or with C stored from registers, gave
  none. The cause was not found *(field, E8)*. Every stage barrier now also waits `lgkmcnt(0)`, meant so that no
  wave's read of a stage is in flight when another wave refills it. The configuration the kernel now runs (A 3, B 4
  slots at BK 128) gave
  the output of the call alone in every try at 20 tries per condition, on 16 and 64 CUs and beside a second call,
  and under the graph and concurrent checks below *(field, E9)*.

  ```python
  # every stage barrier also drains LDS reads: no wave's fragment read of a stage may still be
  # in flight when another wave's refill of that stage is issued past the barrier
  LGK = " lgkmcnt(0)"
  ...
  def barrier(vmcnt):
      llvm.InlineAsmOp(
          None,
          [],
          f"s_waitcnt vmcnt({vmcnt}){LGK}\n\ts_barrier",
          "",
          has_side_effects=True,
      )
  ```
- **Stress, every size of the five kernels:** at 20 tries each, streams limited to 16
  and 64 CUs and a second concurrent call on the same inputs; one graph replayed 24 times with
  three input sets rotated into its buffers; and two calls at once on two streams with different inputs; every try
  gave the output of the call alone *(field, E9)*. These are eager or op-level graph runs on one GPU; a full model's
  decode graph was not run.
- **N 2048 × K 2048 is thin at the largest m:** 1.071 at m 2048, 1.041 at m 4096 *(field, E1)*.
- **Leaving the round lost on these kernels both ways:** two rounds of narrower tiles, half a round of wider ones, and
  narrowing tiles to fill a partial last round (`../gemm/grid_pricing.md` §1).
- **The C store is chosen per kernel and m:** from registers or staged through LDS, `nt`, `sc1` or default; the
  winner changed with N, m and the build (`../gemm/epilogue.md` §3, §5). Moving N 28672's store into the next tile's
  K loop lost (`../gemm/epilogue.md` §4).
- **Fallback.** An m outside the tables goes to `torch.matmul` at N 2048 × K 2048, N 3584 and N 6144 × K 1536, and
  m < 256 at N 28672; no size the kernels are built for reaches it (N 6144 × K 1536 at m 2048 would).

## Mechanism

Shared:
- **One output tile per workgroup over all of K.** A and B go into LDS by `buffer_load_dwordx4 … lds` into
  XOR-swizzled rows, through rings of 2–5 stages that the workgroup's waves share with one barrier per K block
  (N 2048 × K 2048 at m 256: a private ring per wave, no barrier), and `mfma_f32_16x16x32_bf16` accumulates in fp32
  (`../gemm/datapath.md` §2, §5; `../gemm/mainloop.md` §5).
- **One round.** Every size launches 224 or 256 workgroups; only N 7168 at m 4096 (448 tiles) and N 28672 from
  m 1024 walk more tiles than that, on a persistent grid of 256 (`../gemm/grid_pricing.md` §1).
- **Tiles grouped per XCD**, so the tiles that share an XCD's L2 share A rows or B panels
  (`../gemm/schedule_and_locality.md` §2); not at N 6144 × K 1536 or N 7168 at m 256–512, which run tiles in launch
  order.
- **Kernel arguments preloaded** into SGPRs (`amdgpu-kernarg-preload-count`), except N 28672 at m 2048.

**N 2048 × K 2048**, BK 64, 256 workgroups, each XCD a box of tiles:

| m | Tile | Waves | Ring | First K tile | Loop |
|---|---|---|---|---|---|
| 256 | 32×64 | 4, one per K quarter, + 1 prefetch wave | 3 slots per wave, private | 0 | per-wave K split, no barrier in the K loop |
| 512 | 64×64 | 2 × 2 | 5 | by place in the XCD box | fenced ping-pong, 2 segments |
| 1024 | 128×64 | 4 × 2 | 5 | by place in the box | ping-pong, 2 segments |
| 2048 | 128×128 | 4 × 2 | 4 | staggered among the tiles sharing a B panel | ping-pong, 2 segments |
| 4096 | 256×128 | 4 × 1 | 3 | staggered among the tiles sharing an A panel | ping-pong, 8 segments |

- m 256 is N 512 × K 4096's per-wave K split (§ Mechanism of `bf16_k4096_splitk_hgemm.md`): four waves of 512 K
  each, partials
  added in wave order through LDS, 2-byte stores. A fifth wave touches one dword per cache line of the two K tiles
  after the prologue's in each K quarter of the workgroup's A and B panels (shared out among the box's workgroups on
  the same panel), into LDS nobody reads, then exits: meant to move only the cold misses ahead of the DMAs (the
  code's comment).

  ```python
  # The extra wave touches one dword per cache line of the PFT k tiles after the prologue,
  # split between the box workgroups sharing each panel, into LDS nobody reads: only the
  # cold misses move ahead of the DMAs.
  is_pf = _uniform(fx.Int32(tid_wg) // _WAVE) == KW * TPW // _WAVE
  if_pf = _scf.IfOp(_irv(is_pf), [], has_else=False)
  with ir.InsertionPoint(if_pf.then_block):
      kcols = [
          g * (K // KW) * 2 + t * BK * 2
          for g in range(KW)
          for t in range(STAGES - 1, min(k_tiles, STAGES - 1 + PFT))
      ]
      NT = len(kcols)
      pf_i64 = fx.Int64(fx.ptrtoint(lds.pf_lds.peek().ptr))
      for rs, nrow, step, r_own, rbase in (
          (rsrc_b, BN // xb_m, xb_m, box_x, fx.Int32(bid_y) * BN),
          (rsrc_a, BM // xb_n, xb_n, box_y, fx.Int32(bid_x) * BM),
      ):
          n_l = nrow * NT
          for j in fx.range_constexpr((n_l + _WAVE - 1) // _WAVE):
              hi = min(n_l, (j + 1) * _WAVE)
              line = lane + j * _WAVE
              line = (line < hi).select(line, fx.Int32(hi - 1))
              ti = line % NT
              col = fx.Int32(kcols[0])
              for c in range(1, NT):
                  col = (ti == c).select(fx.Int32(kcols[c]), col)
              row = rbase + r_own + (line // NT) * step
              _lds_load(rs, row * (K * 2) + fx.Int32(col), pf_i64, fx.Int32(0), "dword")
      _llvm.InlineAsmOp(None, [], "s_waitcnt vmcnt(0)\n\ts_endpgm", "", has_side_effects=True)
  ```
- m ≥ 512: tile t + 1's `ds_read`s go out with tile t's MFMAs in regions fenced with `sched_group_barrier`, and the
  refill DMAs are cut into segments between them (`../gemm/schedule_and_locality.md` §3). B's rows are permuted in the
  fill so a lane's fragments 2j and 2j + 1 hold adjacent columns: one 4 B store per pair (`../gemm/epilogue.md` §3).

  ```python
  # the source row of B that the DMA lands in LDS row r of the stage
  if fx.const_expr(perm):
      # LDS row 2W*jj + W*t + m'  <-  tile row 2W*jj + 2*m' + t
      rem = r & (2 * PW - 1)
      row = (
          ((r >> _lg2(2 * PW)) << _lg2(2 * PW))
          + ((rem & (PW - 1)) << 1)
          + (rem >> _lg2(PW))
          + row0
      )
  else:
      row = r + row0
  ```

**N 3584 × K 512**, one tile per workgroup, B first in the MFMA, an unrolled K loop with the next K
step's fragments read ahead, C stored `sc1`; each XCD takes a contiguous run of tiles, 4 (8 at m 2048) M-tiles at a
time; every stage barrier also waits `lgkmcnt(0)`:

| m | Tile, waves | BK, slots | Workgroups | C | Also |
|---|---|---|---|---|---|
| 256 | 64×64, 4 × 1 | 128; A 3, B 4 | 224 | staged through padded LDS rows | XCDs 4–7 start K half-way |
| 512 | 64×128, 2 × 2 | 64; 4 | 224 | staged through LDS | the last row group's last four MFMAs run after the next slice's barrier; the last slice lands B's first half and A first; XCDs 4–7 start K half-way |
| 1024 | 128×128, 2 × 2 | 64; 3 | 224 | from registers (B rows permuted so a lane holds 8 adjacent columns), the last slice one row group at a time | wave index runs over M first |
| 2048 | 128×224, 4 × 2 | 64; 3 | 256 | the last slice staged one row group at a time in a stage it freed, two buffers alternating | refills issued after the slice's first fragment reads, A's spread between MFMA row groups |
| 4096 | 256×224, 4 × 2 | 64; A 3, B 2 | 256 | as m 2048 | |

**N 7168 × K 512**:
- m 256, 512: 64×128 tiles with 2 × 2 waves, BK 128, 3 stages (m 256); 128×128 with 4 × 2 waves,
  BK 64, 4 stages (m 512); 224 workgroups. A first in the MFMA with B's LDS rows filled from permuted rows, so a lane's
  fragments are adjacent columns and C goes out from registers, 8 B per lane, `sc1`; the last K block runs one M step
  at a time with its stores under the next step's MFMAs. An LLVM option per m: `misched-cluster` off at m 256, the
  `max-memory-clause` scheduler at m 512.
- m ≥ 1024: eight waves (2 × 4) on 128×256 (m 1024) or 256×256 tiles, BK 64, A in 2 stages and B in
  3 (2 at m 4096), `waves_per_eu` 2, `s_setprio 1` around the MFMA groups of a K block's first step with the refill
  DMAs issued between them, `misched-cluster` off at m 1024; each XCD walks a contiguous run of tiles, 4 (m 1024) or 3
  M-tiles at a time. At m 1024
  and 2048, 224 tiles, one per workgroup, C from registers as above, `nt` at m 1024 and `sc1` at m 2048. At m 4096,
  448 tiles on 256 persistent workgroups: the next tile's first K stage loads while the current one drains, C is
  staged through per-wave LDS slices (B first in the MFMA) and stored `sc1`, and the 64 workgroups that walk one tile
  fewer start 120 `s_sleep` units late, so that their stores land mid-way through the others' second tile (the code's
  comment).

  ```python
  TILES = M_BLOCKS * N_BLOCKS
  GRID = min(_CUS, TILES)
  # One workgroup per CU walks tiles _CUS apart, loading the next tile's first K stage as this one drains.
  PERSIST_EVEN = TILES % _CUS == 0
  PERSIST_TPW = TILES // _CUS
  ...
  if const_expr(DELAY > 0 and not PERSIST_EVEN and TILES > _CUS):
      # Workgroups that walk one tile fewer start DELAY s_sleep units late, so their C stores
      # land mid-way through the others' second tile instead of in the first-round burst.
      if pid >= fx.Int32(TILES % _CUS):
          llvm.InlineAsmOp(None, [], DELAY_ASM, "", has_side_effects=True)
  ...
  if const_expr(not SINGLE):
      # Every wave is past its reads of stage 0: start the next tile's first stage.
      ldg_sts_b_async(fx.Int32(0), 0, fx.Index(bn2 * BLOCK_N))
      ldg_sts_a_async(fx.Int32(0), 0, fx.Index(bm2 * BLOCK_M))
  ```

**N 28672 × K 512**: 256-column tiles, 128 rows at m 256 and 256 rows above, BK 64, eight waves
(2 × 4), `waves_per_eu` 2; B first in the MFMA; C staged per wave through swizzled LDS and stored 16 B per lane.

| m | Work | Workgroups | Stages A / B | C |
|---|---|---|---|---|
| 256 | 224 tiles, one each | 224 | 3 / 3, A loads first, C staged in a B stage | `nt` |
| 512 | 224 tiles, one each | 224 | 2 / 3, A first, C in a B stage | `sc1` |
| 1024 | one 256×256 head tile and one 192×256 tail item each (rows 256–1023 × column blocks 48–111 are tails) | 256 | 2 / 2 | `nt` |
| 2048 | 768 head tiles over three rounds, then 256 tail items of 128×256 (rows 1536–2047 × column blocks 48–111) | 256 | 2 / 2 | `nt` |
| 4096 | 1792 tiles, seven rounds | 256 | 2 / 2 | `nt` |

- A plan picks the head/tail split; at m 1024 a fixed plan gives each XCD heads and tails on the same B panels, and
  at m 256, 512, 2048 and 4096 XCD x owns the column blocks x, x + 8, … of the heads, so its B panels stay in its L2.
- A workgroup with another item loads that item's first K blocks into the stages as the current one drains, so
  there is no barrier between items. From m 1024 the odd workgroups of each XCD first sleep 1, 2 or 3 times
  `s_sleep 32`, meant to put the XCD's C stores out of phase.
- The tail body's lane values come from an opaque copy of the thread id made after the head loop, so none of them is
  kept live across it.

**N 6144 × K 1536**, 256 workgroups in launch order, BK and stages per m, B loaded
`sc0 nt` and A `sc0`, B first in the MFMA, C staged in LDS at a padded row stride and stored 16 B with `sc1`:

| m | Tile | BK, stages | Waves | Loop options |
|---|---|---|---|---|
| 256 | 64×96 | 128, 3 | 4 × 2 | each K slice's fragments read one slice early |
| 512 | 128×96 | 64, 4 | 2 × 2 | refill DMAs after the second K slice's reads, A before B, the first slice read in MFMA order, the K offset in `soffset` |
| 1024 | 128×192 | 64, 4 | 2 × 4 | as m 512 |

## Result

The speedup is the one-sided lower bound of the aiter / kernel time ratio over 16 rounds, each timing the kernels
of CUDA-graph replays with L2 flushed before every call, against aiter's tuned `gemm_a16w16`, on one MI355X with
FlyDSL 0.3.2. A separate run, its ruler not stated in the record, read 1.034–1.798 over the 23 sizes, each passing
the accuracy check *(field, E10)*.

| Kernel | Sizes | Speedup over aiter | Highest |
|---|---|---|---|
| N 2048 × K 2048 *(field, E1)* | m 256–4096 | 1.041–1.231 | 1.231 at m 512 |
| N 3584 × K 512 *(field, E2)* | m 256–4096 | 1.358–1.625 | 1.625 at m 256 |
| N 7168 × K 512 *(field, E3)* | m 256–4096 | 1.164–1.816 | 1.816 at m 1024 |
| N 28672 × K 512 *(field, E4)* | m 256–4096 | 1.173–1.452 | 1.452 at m 2048 |
| N 6144 × K 1536 *(field, E5)* | m 256–1024 | 1.156–1.257 | 1.257 at m 256 |

Every size passed the accuracy check.

## Evidence

The E1–E6 ruler: one-sided lower bound of the aiter / kernel time ratio over 16
rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the lower-bound ruler stated above] N 2048 × K 2048 on the build before the current one, which differs only in passing the stream of the input's device: 1.214, 1.231, 1.224, 1.071, 1.041 at m 256, 512, 1024, 2048, 4096; accuracy passed
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the lower-bound ruler stated above] N 3584 × K 512: 1.625, 1.527, 1.358, 1.538, 1.372 at m 256, 512, 1024, 2048, 4096; accuracy passed
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the lower-bound ruler stated above] N 7168 × K 512: 1.164, 1.369, 1.816, 1.366, 1.323 at m 256, 512, 1024, 2048, 4096; accuracy passed
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the lower-bound ruler stated above] N 28672 × K 512: 1.339, 1.408, 1.298, 1.452, 1.173 at m 256, 512, 1024, 2048, 4096; accuracy passed
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the lower-bound ruler stated above] N 6144 × K 1536 on an earlier build with the same kernel code that also covered m 2048 and 4096: 1.257, 1.172, 1.156, 0.999, 0.982 at m 256, 512, 1024, 2048, 4096; m 2048 and 4096 judged not faster; accuracy passed
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E5] per-round times, kernel against aiter: m 2048 (`torch.matmul`) 35.20–35.28 against 35.19–35.28 µs; m 4096 67.46–67.89 against 66.40–66.65
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=3, a CUDA-graph timer, medians of 3–9 readings (first run) and of 4 (later run), `torch.matmul` re-timed in the same runs] N 6144 × K 1536 at m 2048: 256×192, 128×192, 128×96, 128×256, 256×128 and 128×128 tiles at 2–4 stages read 38.35–55.89 µs against `torch.matmul` 33.97–34.07; later the 256×192 two-stage row with A first, read order and `soffset` 36.91 against 34.08
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=20, a stress check of the call under contention against the call alone, and a 20-run probe per condition on streams limited to 16 and 64 CUs] N 3584 × K 512, m 256, an earlier build (64×64, BK 64, A 5 / B 4 slots, C staged through LDS): the stress check failed on 16 CUs, and the probe gave 2 bad runs, with and without the row guard; the symmetric ring, separate depths with C from registers, 32×112 with separate depths, and m 1024 either way gave 0; cause not found; the `lgkmcnt(0)` fence on every stage barrier was kept afterwards, and the probe was not rerun with it
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] the N 2048 × K 2048 build of E1 and the N 3584 × K 512, N 7168 × K 512, N 28672 × K 512 and N 6144 × K 1536 kernels, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ, no hang
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] at m 256, 512, 1024, 2048, 4096: N 2048 × K 2048 1.212, 1.234, 1.225, 1.059, 1.034; N 3584 × K 512 1.590, 1.519, 1.356, 1.531, 1.387; N 7168 × K 512 1.155, 1.370, 1.798, 1.351, 1.265; N 28672 × K 512 1.350, 1.481, 1.285, 1.432, 1.157; N 6144 × K 1536 at m 256, 512, 1024: 1.258, 1.175, 1.165; all 23 sizes at or above 1.03, accuracy passed at each
