---
title: "bf16 GEMM at m 256–4096, N 2048–28672 × K 512–2048 — whole-K MFMA tiles fed by LDS DMA rings on one round of 224–256 workgroups, persistent where the tiles outnumber the CUs, no split-K; N 6144 × K 1536 stops at m 1024"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 256-1024", "M 2048-4096"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1, AKLR validate measurement profile 19
measured:
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.041–1.231; 5 of 5 Points above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n2048_k2048 file 8da6be976ff855b0, measured as its predecessor, which differs only in passing the stream of the input's device; m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.358–1.625; 5 of 5 Points above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n3584_k512 file 1fbaf19b1c4fdb45, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.164–1.816; 5 of 5 Points above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n7168_k512 file 04da38be4d3af890, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.173–1.452; 5 of 5 Points above AKLR_MIN_SPEEDUP", gpu: MI355X, workload: "gemm_n28672_k512 file ce4647f84b8de43b, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.156–1.257 at m 256–1024; m 2048 (0.999) and 4096 (0.982) dropped from the file", gpu: MI355X, workload: "gemm_n6144_k1536 kernel of file f665d33c162100b7, measured as an earlier file with the same kernel, m 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/8da6be976ff855b0.json
  - aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json
  - aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/f665d33c162100b7.json
---

# bf16 GEMM at m 256–4096: whole-K tiles on one round, persistent past it

Five FlyDSL kernels for the `gemm@1.0` contract at m 256–4096, one new file per schema name. Each file embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); the function names below are in that file. "m" is the Point's M.
Each reuses a builder already on main: N 2048 × K 2048 the `_build_gemm` of `gemm_n512_k4096`
([bf16_k4096_splitk_hgemm.md](bf16_k4096_splitk_hgemm.md)); N 3584 × K 512 the `_build` of `gemm_n2304_k1536` and
N 28672 × K 512 the `_build` and `_plan` of `gemm_n3072_k512`
([k3_bf16_short_k_large_m.md](k3_bf16_short_k_large_m.md)); N 7168 × K 512 the `_build_large` and `_build_pipe` of
`gemm_n7168_k768`, and N 6144 × K 1536 the `_build_tiled` of `gemm_n7168_k1536`
([k3_bf16_band_stream.md](k3_bf16_band_stream.md)), aiter's async-LDS HGEMM at split-K 1.

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16, `trans_b`,
  no bias, bf16 `out`. The schema's `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0).
  Each output is one MFMA chain over K in a fixed order (at N 2048 × K 2048, m 256, plus a fixed-order fold of four
  K waves); no split-K across workgroups, no atomics.
- Weights: N 2048 × K 2048 (8 MiB), N 3584 × K 512 (3.5 MiB), N 7168 × K 512 (7 MiB), N 28672 × K 512 (28 MiB),
  N 6144 × K 1536 (18 MiB). K is left whole: from m 256 the output tiles alone fill a round of the GPU.
- Points: m 256, 512, 1024, 2048, 4096 in four files; m 256, 512, 1024 at N 6144 × K 1536, whose m 2048 and 4096 are
  not listed (no faster than aiter, § Counter-signals).
- No other file serves these Points, so the yardstick is aiter. Dispatch serves a Point whose CI `speedup` reaches
  `AKLR_MIN_SPEEDUP`, default 1.05 (`qualifies()` in `aklr/core/contract/spec/_workloads.py`; `aklr/ops/README.md`
  § Environment variables); all 23 listed Points of the five files reach it in CI, 1.034–1.798, and are served (#125).

## Counter-signals

- **N 6144 × K 1536 is not covered at m 2048 and 4096.** At m 2048 `run` calls `torch.matmul` (`_library`), which read
  0.999 against aiter, the same time per round (35.20–35.28 against 35.19–35.28 µs) *(field, E5, E6)*. No FlyDSL
  configuration built for that m beat `torch.matmul`: nine of them (six tile shapes) read 38.35–55.89 µs against
  33.97–34.07, and the best later row 36.91 against 34.08 *(field, E7)*. The m 4096 row (256×384 tiles,
  `mfma_f32_32x32x16_bf16`, C staged in two halves with each wave's first 32-row MFMA atom row stored early) read 0.982
  *(field, E5)*. Neither Point is listed; the m 4096 row stays in `_CONFIGS`, and no listed Point reaches it.
- **N 3584 at m 256 runs separate A and B ring depths, which gave wrong output under stress in an earlier build.**
  That build of the row (64×64, BK 64, A 5 and B 4 slots, C staged through LDS) failed the forge driver's stress on a
  stream limited to 16 CUs, and a probe of 20 runs on each of a 16-CU and a 64-CU stream gave 2 bad results, with and
  without the row guard; the symmetric ring, separate depths on another tile, or with C stored from registers, gave
  none. The cause was not found *(field, E8)*. Every stage barrier now also waits `lgkmcnt(0)`, meant so that no
  wave's read of a stage is in flight when another wave refills it. The listed row (A 3, B 4 slots at BK 128) gave
  the output of the call alone in every try at 20 tries per condition, on 16 and 64 CUs and beside a second call,
  and under the graph and concurrent checks below *(field, E9)*.
- **Stress, every listed Point of the five files:** the forge driver's conditions at 20 tries (streams limited to 16
  and 64 CUs; a second concurrent call, which the driver runs on the same inputs), one graph replayed 24 times with
  three input sets rotated into its buffers, and two calls at once on two streams with different inputs; every try
  gave the output of the call alone *(field, E9)*. These are eager or op-level graph runs on one GPU; a full model's
  decode graph was not run.
- **N 2048 × K 2048 is thin at the largest m:** 1.071 at m 2048, 1.041 at m 4096 *(field, E1)*.
- **Leaving the round lost on these kernels both ways:** two rounds of narrower tiles, half a round of wider ones, and
  narrowing tiles to fill a partial last round (`../gemm/grid_pricing.md` §1).
- **The C store is chosen per kernel and m:** from registers or staged through LDS, `nt`, `sc1` or default; the
  winner changed with N, m and the build (`../gemm/epilogue.md` §3, §5). Moving N 28672's store into the next tile's
  K loop lost (`../gemm/epilogue.md` §4).
- **Fallback.** `run` sends m outside the tables to `torch.matmul` at N 2048 × K 2048, N 3584 and N 6144 × K 1536, and
  m < 256 at N 28672; no listed Point reaches it (the unlisted N 6144 × K 1536 m 2048 would).

## Mechanism

Shared:
- **One output tile per workgroup over all of K.** A and B go into LDS by `buffer_load_dwordx4 … lds` into
  XOR-swizzled rows, through rings of 2–5 stages that the workgroup's waves share with one barrier per K block
  (N 2048 × K 2048 at m 256: a private ring per wave, no barrier), and `mfma_f32_16x16x32_bf16` accumulates in fp32
  (`../gemm/datapath.md` §2, §5; `../gemm/mainloop.md` §5).
- **One round.** Every Point launches 224 or 256 workgroups; only N 7168 at m 4096 (448 tiles) and N 28672 from
  m 1024 walk more tiles than that, on a persistent grid of 256 (`../gemm/grid_pricing.md` §1).
- **Tiles grouped per XCD**, so the tiles that share an XCD's L2 share A rows or B panels
  (`../gemm/schedule_and_locality.md` §2); not at N 6144 × K 1536 or N 7168 at m 256–512, which run tiles in launch
  order.
- **Kernel arguments preloaded** into SGPRs (`amdgpu-kernarg-preload-count`), except N 28672 at m 2048.

**N 2048 × K 2048** (`8da6be97…`), BK 64, 256 workgroups, each XCD a box of tiles:

| m | Tile | Waves | Ring | First K tile | Loop |
|---|---|---|---|---|---|
| 256 | 32×64 | 4, one per K quarter, + 1 prefetch wave | 3 slots per wave, private | 0 | per-wave K split, no barrier in the K loop |
| 512 | 64×64 | 2 × 2 | 5 | by place in the XCD box | fenced ping-pong, 2 segments |
| 1024 | 128×64 | 4 × 2 | 5 | by place in the box | ping-pong, 2 segments |
| 2048 | 128×128 | 4 × 2 | 4 | staggered among the tiles sharing a B panel | ping-pong, 2 segments |
| 4096 | 256×128 | 4 × 1 | 3 | staggered among the tiles sharing an A panel | ping-pong, 8 segments |

- m 256 is n512's per-wave K split (§ Mechanism of `bf16_k4096_splitk_hgemm.md`): four waves of 512 K each, partials
  added in wave order through LDS, 2-byte stores. A fifth wave touches one dword per cache line of the two K tiles
  after the prologue's in each K quarter of the workgroup's A and B panels (shared out among the box's workgroups on
  the same panel), into LDS nobody reads, then exits: meant to move only the cold misses ahead of the DMAs (the
  code's comment).
- m ≥ 512: tile t + 1's `ds_read`s go out with tile t's MFMAs in regions fenced with `sched_group_barrier`, and the
  refill DMAs are cut into segments between them (`../gemm/schedule_and_locality.md` §3). B's rows are permuted in the
  fill so a lane's fragments 2j and 2j + 1 hold adjacent columns: one 4 B store per pair (`../gemm/epilogue.md` §3).

**N 3584 × K 512** (`1fbaf19b…`), one tile per workgroup, B first in the MFMA, an unrolled K loop with the next K
step's fragments read ahead, C stored `sc1`; each XCD takes a contiguous run of tiles, 4 (8 at m 2048) M-tiles at a
time; every stage barrier also waits `lgkmcnt(0)`:

| m | Tile, waves | BK, slots | Workgroups | C | Also |
|---|---|---|---|---|---|
| 256 | 64×64, 4 × 1 | 128; A 3, B 4 | 224 | staged through padded LDS rows | XCDs 4–7 start K half-way |
| 512 | 64×128, 2 × 2 | 64; 4 | 224 | staged through LDS | the last row group's last four MFMAs run after the next slice's barrier; the last slice lands B's first half and A first; XCDs 4–7 start K half-way |
| 1024 | 128×128, 2 × 2 | 64; 3 | 224 | from registers (B rows permuted so a lane holds 8 adjacent columns), the last slice one row group at a time | wave index runs over M first |
| 2048 | 128×224, 4 × 2 | 64; 3 | 256 | the last slice staged one row group at a time in a stage it freed, two buffers alternating | refills issued after the slice's first fragment reads, A's spread between MFMA row groups |
| 4096 | 256×224, 4 × 2 | 64; A 3, B 2 | 256 | as m 2048 | |

**N 7168 × K 512** (`04da38be…`):
- m 256, 512, `_build_large`: 64×128 tiles with 2 × 2 waves, BK 128, 3 stages (m 256); 128×128 with 4 × 2 waves,
  BK 64, 4 stages (m 512); 224 workgroups. A first in the MFMA with B's LDS rows filled from permuted rows, so a lane's
  fragments are adjacent columns and C goes out from registers, 8 B per lane, `sc1`; the last K block runs one M step
  at a time with its stores under the next step's MFMAs. An LLVM option per m: `misched-cluster` off at m 256, the
  `max-memory-clause` scheduler at m 512.
- m ≥ 1024, `_build_pipe`: eight waves (2 × 4) on 128×256 (m 1024) or 256×256 tiles, BK 64, A in 2 stages and B in
  3 (2 at m 4096), `waves_per_eu` 2, `s_setprio 1` around the MFMA groups of a K block's first step with the refill
  DMAs issued between them, `misched-cluster` off at m 1024; each XCD walks a contiguous run of tiles, 4 (m 1024) or 3
  M-tiles at a time. At m 1024
  and 2048, 224 tiles, one per workgroup, C from registers as above, `nt` at m 1024 and `sc1` at m 2048. At m 4096,
  448 tiles on 256 persistent workgroups: the next tile's first K stage loads while the current one drains, C is
  staged through per-wave LDS slices (B first in the MFMA) and stored `sc1`, and the 64 workgroups that walk one tile
  fewer start 120 `s_sleep` units late, so that their stores land mid-way through the others' second tile (the code's
  comment).

**N 28672 × K 512** (`ce4647f8…`): 256-column tiles, 128 rows at m 256 and 256 rows above, BK 64, eight waves
(2 × 4), `waves_per_eu` 2; B first in the MFMA; C staged per wave through swizzled LDS and stored 16 B per lane.

| m | Work | Workgroups | Stages A / B | C |
|---|---|---|---|---|
| 256 | 224 tiles, one each | 224 | 3 / 3, A loads first, C staged in a B stage | `nt` |
| 512 | 224 tiles, one each | 224 | 2 / 3, A first, C in a B stage | `sc1` |
| 1024 | one 256×256 head tile and one 192×256 tail item each (rows 256–1023 × column blocks 48–111 are tails) | 256 | 2 / 2 | `nt` |
| 2048 | 768 head tiles over three rounds, then 256 tail items of 128×256 (rows 1536–2047 × column blocks 48–111) | 256 | 2 / 2 | `nt` |
| 4096 | 1792 tiles, seven rounds | 256 | 2 / 2 | `nt` |

- `_plan` picks the head/tail split; at m 1024 a fixed plan gives each XCD heads and tails on the same B panels, and
  at m 256, 512, 2048 and 4096 XCD x owns the column blocks x, x + 8, … of the heads, so its B panels stay in its L2.
- A workgroup with another item loads that item's first K blocks into the stages as the current one drains, so
  there is no barrier between items. From m 1024 the odd workgroups of each XCD first sleep 1, 2 or 3 times
  `s_sleep 32`, meant to put the XCD's C stores out of phase.
- The tail body's lane values come from an opaque copy of the thread id made after the head loop, so none of them is
  kept live across it.

**N 6144 × K 1536** (`f665d33c…`), `_build_tiled`, 256 workgroups in launch order, BK and stages per m, B loaded
`sc0 nt` and A `sc0`, B first in the MFMA, C staged in LDS at a padded row stride and stored 16 B with `sc1`:

| m | Tile | BK, stages | Waves | Loop options |
|---|---|---|---|---|
| 256 | 64×96 | 128, 3 | 4 × 2 | each K slice's fragments read one slice early |
| 512 | 128×96 | 64, 4 | 2 × 2 | refill DMAs after the second K slice's reads, A before B, the first slice read in MFMA order, the K offset in `soffset` |
| 1024 | 128×192 | 64, 4 | 2 × 4 | as m 512 |

## Result

`speedup` here is AKLR validate's one-sided lower bound of the aiter / kernel time ratio (measurement profile 19,
`MEASUREMENT_PROFILE` in `aklr/core/validation/stages/_measure.py`; aiter's tuned `gemm_a16w16` through
`gemm_baseline`), on one MI355X at main 0e02a81 with FlyDSL 0.3.2. Dispatch reads the `speedup` CI writes
(`aklr/ops/README.md` § Workflow): 1.034–1.798 over the 23 Points (#125), each passing accuracy.

| File | Points | Speedup over aiter | Highest |
|---|---|---|---|
| N 2048 × K 2048 `8da6be97…` *(field, E1)* | m 256–4096 | 1.041–1.231 | 1.231 at m 512 |
| N 3584 × K 512 `1fbaf19b…` *(field, E2)* | m 256–4096 | 1.358–1.625 | 1.625 at m 256 |
| N 7168 × K 512 `04da38be…` *(field, E3)* | m 256–4096 | 1.164–1.816 | 1.816 at m 1024 |
| N 28672 × K 512 `ce4647f8…` *(field, E4)* | m 256–4096 | 1.173–1.452 | 1.452 at m 2048 |
| N 6144 × K 1536 `f665d33c…` *(field, E5)* | m 256–1024 | 1.156–1.257 | 1.257 at m 256 |

Every Point passed accuracy.

## Reproduce

On an MI355X, from the repository root, one file at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json
```

and the same for the other paths under `sources`. The report goes to `aklr_validation/report.md`
(`aklr/ops/README.md` § Commands). Add `--base origin/main` only on a branch that adds the file: with the base set,
bench measures only schemas that are not at the base (`aklr/core/validation/stages/validation_scope.py`). The
N 6144 × K 1536 numbers were taken on a file that embeds the same kernel and also lists m 2048 and 4096 (E5). The
local result does not count; CI's is the one dispatch reads.

## Rollback

Delete the schema file (in a PR): its Points are then served by no file, and the caller runs its own path.
`AKLR_DISABLE=gemm@1.0` stops serving the whole contract, every other bf16 GEMM schema included. A Point CI measures
below the threshold stops being served without a rollback. Roll back for a wrong result CI does not catch, for
example a wrong output of N 3584 at m 256 on a stream with few free CUs (§ Counter-signals).

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`. The E1–E6 ruler:
AKLR validate at 0e02a81, measurement profile 19, one-sided lower bound of the aiter / kernel time ratio over 16
rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd.

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate profile 19] the predecessor of 8da6be976ff855b0, which differs only in passing the stream of the input's device: 1.214, 1.231, 1.224, 1.071, 1.041 at m 256, 512, 1024, 2048, 4096; accuracy passed (gemm_n2048_k2048)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate profile 19] 1fbaf19b1c4fdb45: 1.625, 1.527, 1.358, 1.538, 1.372 at m 256, 512, 1024, 2048, 4096; accuracy passed (gemm_n3584_k512)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate profile 19] 04da38be4d3af890: 1.164, 1.369, 1.816, 1.366, 1.323 at m 256, 512, 1024, 2048, 4096; accuracy passed (gemm_n7168_k512)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate profile 19] ce4647f84b8de43b: 1.339, 1.408, 1.298, 1.452, 1.173 at m 256, 512, 1024, 2048, 4096; accuracy passed (gemm_n28672_k512)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate profile 19] an earlier file with the same kernel as f665d33c162100b7: 1.257, 1.172, 1.156, 0.999, 0.982 at m 256, 512, 1024, 2048, 4096; m 2048 and 4096 demoted `not_faster`; accuracy passed (gemm_n6144_k1536)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the run of E5] per-round times, kernel against aiter: m 2048 (`torch.matmul`) 35.20–35.28 against 35.19–35.28 µs; m 4096 67.46–67.89 against 66.40–66.65 (`candidate_milliseconds`, `versus_milliseconds`)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=3, the campaign driver's graph timer, medians of 3–9 readings (first run) and of 4 (later run), `torch.matmul` re-timed in the same runs] N 6144 × K 1536 at m 2048: 256×192, 128×192, 128×96, 128×256, 256×128 and 128×128 tiles at 2–4 stages read 38.35–55.89 µs against `torch.matmul` 33.97–34.07; later the 256×192 two-stage row with A first, read order and `soffset` 36.91 against 34.08
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=20, the campaign's forge driver `--stress` and its 20-run probe per condition on streams limited to 16 and 64 CUs] N 3584 × K 512, m 256, an earlier build (64×64, BK 64, A 5 / B 4 slots, C staged through LDS): the stress check failed on 16 CUs, and the probe gave 2 bad runs, with and without the row guard; the symmetric ring, separate depths with C from registers, 32×112 with separate depths, and m 1024 either way gave 0; cause not found; the `lgkmcnt(0)` fence on every stage barrier kept in the next round, where the probe was not rerun
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] the predecessor of 8da6be976ff855b0 (E1), 1fbaf19b1c4fdb45, 04da38be4d3af890, ce4647f84b8de43b and f665d33c162100b7, every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ, no hang
- S1 aklr/ops/gemm/impls/flydsl/gemm_n2048_k2048/8da6be976ff855b0.json#kernel.py:28 `_ROUTE` per m; XCD box at :116-124; prefetch wave at :199-229; per-wave loop at :277-306; fenced ping-pong at :307-379; permuted B rows (`PAIR`) at :98 and :164-172; `torch.matmul` for other m at :458-463
- S2 aklr/ops/gemm/impls/flydsl/gemm_n3584_k512/1fbaf19b1c4fdb45.json#kernel.py:34 `_CONFIGS` per m (knobs at :185-216); `lgkmcnt(0)` on every stage barrier at :253-255; XCD runs at :382-421; K start half-way on XCDs 4–7 at :430-435; B first in the MFMA at :618; LDS-staged and direct C stores at :942 and :985
- S3 aklr/ops/gemm/impls/flydsl/gemm_n7168_k512/04da38be4d3af890.json#kernel.py:30 `_CONFIGS` and `_ROW_LLVM` (:40); permuted B rows at :126; `_build_large` store at :377-389; `_build_pipe` at :439 (grid and persistence at :473-475, XCD runs at :528-540, late start at :851-856)
- S4 aklr/ops/gemm/impls/flydsl/gemm_n28672_k512/ce4647f84b8de43b.json#kernel.py:32 `_STAGGER`, `_SPLIT_PLAN` (:45), `_STRIDED_GM` (:48); `_plan` at :67; C store policy at :243; tile maps at :298-339; staggered start at :634-637; opaque thread id for the tail at :673-683; `torch.matmul` below m 256 at :712-715
- S5 aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/f665d33c162100b7.json#kernel.py:23 `_CONFIGS` (m 256, 512, 1024, 4096); `_library` (`torch.matmul`) at :78 for any other m (:83-86); B `sc0 nt` at :135; 32×32×16 MFMA for the m 4096 row at :225; padded C stride at :175; kernarg preload at :529
