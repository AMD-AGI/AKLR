---
title: "bf16 GEMM, N 7168 × K 768–4224 and N 18432 / 20480 × K 1536 / 7168 (Kimi-K3) — one column band of B per workgroup at m ≤ 112, a tiled 8-wave grid that turns persistent at m 9397 and 16384; N 4096 × K 2048 and N 6144 × K 1536 at m 1–128 on the k1792 band kernel"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-128", "M 893-16384"]
contract: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "0.994–1.162; 5 of 17 Points ≥ 1.03 (m 80, 893, 2732, 9397, 16384); CI serves 6 of 17 as of #102", gpu: MI355X, workload: "gemm_n7168_k768, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "0.999–1.273; 14 of 17 Points ≥ 1.03; CI serves 13 of 17 as of #102", gpu: MI355X, workload: "gemm_n7168_k1536, m 8–16384 (17 Points)", date: "2026-10-26"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.256–1.634; 11 of 11 Points ≥ 1.03; CI serves 11 of 11 as of #102", gpu: MI355X, workload: "gemm_n7168_k1792, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "0.945–1.366; 16 of 17 Points ≥ 1.03; CI serves 14 of 17 as of #102", gpu: MI355X, workload: "gemm_n7168_k3584, m 8–16384 (17 Points)", date: "2026-10-26"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 19 per Point", value: "0.876–3.429; 14 of 17 Points ≥ 1.03 and served as of #102 (not m 2732, 9397, 16384)", gpu: MI355X, workload: "gemm_n7168_k4224, m 8–16384 (17 Points)", run_ref: "6d7cfec (#102, CI run 42549655045)", date: "2026-11-09"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.284–2.000 (2.000 is the bound's cap); 13 of 13 Points ≥ 1.03; CI serves 13 of 13 as of CI run 41197527705 (a29581b), unchanged by #102", gpu: MI355X, workload: "gemm_n18432_k1536, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112", date: "2026-10-25"}
  - {metric: "speedup over aiter, local validate lower bound per Point", value: "1.071–1.417; 23 of 23 Points ≥ 1.03; CI serves 23 of 23 as of #102", gpu: MI355X, workload: "gemm_n20480_k7168, m 7–112 (23 Points)", date: "2026-10-26"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.118–1.449; 8 of 8 Points at or above AKLR_MIN_SPEEDUP; CI serves 8 of 8 as of #115", gpu: MI355X, workload: "gemm_n4096_k2048 file 37c909c4b629ad59, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate profile 19, per Point", value: "1.427–1.592; 8 of 8 Points at or above AKLR_MIN_SPEEDUP; CI serves 8 of 8 as of #115", gpu: MI355X, workload: "gemm_n6144_k1536 file 1d3a1fcaf77631e1, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
sources:
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k768/0567a6d854ac7a39.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k1536/c9ad7a129bb4cfe6.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k1792/f0a48d1c7cdce51f.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k3584/251a5ceb2d416e55.json
  - aklr/ops/gemm/impls/flydsl/gemm_n7168_k4224/2d72ae169867a6d1.json
  - aklr/ops/gemm/impls/flydsl/gemm_n18432_k1536/8cffee60e0ddc139.json
  - aklr/ops/gemm/impls/flydsl/gemm_n20480_k7168/23c7fb06d509a1bf.json
  - aklr/ops/gemm/impls/flydsl/gemm_n4096_k2048/37c909c4b629ad59.json
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/1d3a1fcaf77631e1.json
---

# bf16 GEMM at N 7168, 18432 and 20480 (Kimi-K3): a column band of B per workgroup, then a tiled grid

Seven of this page's nine FlyDSL `gemm@1.0` files are the Kimi-K3 set,
one schema file per shape, written by forge-loop campaigns and reviewed. Each schema embeds its
kernel as `kernel.sources[0].content` (`kernel.py`); function names below are in that file. "m" is the Point's M.
k1536's `_build_tiled` and k768's `_build_large` and `_build_pipe` also run `gemm_n6144_k1536` and `gemm_n7168_k512`
at m 256–4096: [bf16_large_m_whole_k_tiles.md](bf16_large_m_whole_k_tiles.md).

## Applies when

- Contract `gemm@1.0`, layout variant `row_major` (`b.is_shuffled` false): `a` (m, k) and `b` (n, k) bf16,
  `trans_b`, no bias, bf16 `out`. `compare` is bitwise (`determinism: bitwise`, `max_violation_fraction` 0.0,
  the `CompareSpec` defaults), with the fp32 dot-product error bound of `_dot_k()` (`GEMM` in
  `aklr/ops/gemm/contract.py`).
- N × K = 7168 × 768, 1536, 1792, 3584, 4224 (a weight of 10.5–57.75 MiB), 18432 × 1536 (54 MiB) and
  20480 × 7168 (280 MiB): at m ≤ 112 each call streams the whole weight once
  (`../gemm/small_m.md` §2). k1792 and n18432 / n20480 have only Points at m ≤ 112; the other
  four also have m 893, 2732, 9397, 16384.
- Each schema name has only this file (no file of another language or an older kernel under the same name on
  main), so at every Point the yardstick is aiter.
- **CI serves 94 of their 115 Points** as of #102 (6d7cfec, CI run 42549655045, measurement profile 19), n18432's
  as of CI run 41197527705 (a29581b), which #102 left unchanged; the count per file is in § Result, and the current
  value of each Point is the `speedup` in its schema's `workloads`. Dispatch serves a workload only from
  `AKLR_MIN_SPEEDUP` 1.03 (`aklr/ops/README.md` § Environment variables; `aklr/core/contract/spec/_workloads.py:38`).

## Counter-signals

- **Below 1.03 in the local validate** (CI's count per file is in § Result):
  - k768 at m 8–64, 96 and 112: 0.994–1.027 *(field, E1)*. These Points run the skinny, ring and small-tile paths
    below; the commit that added the kernel notes aiter is tuned for this shape.
  - k1536 at m 32 (1.005), 893 (1.025) and 16384 (0.999) *(field, E2)*; k3584 at 16384 (0.945) *(field, E4)*;
    k4224 at 2732 (1.020) and 16384 (0.935) *(field, E5, E6)*. At m 16384 all three run the persistent
    256×256 grid below.
  - Points at m ≤ 112 this close to 1.03 cross it between runs: k768's m 32 read 1.002 in one run and 1.029 and 1.038
    in the two arms of another, all three with the same final ISA at that Point (the large-m rework was delivered
    with the ISA at m ≤ 893 unchanged) *(field, E1)*.
- **k4224 at m 56–112 folds a cross-workgroup split-K through flags whose tag does not change between calls.**
  `_sk_build`: the last split polls `MAGIC ^ tile ^ split<<16` words in a workspace allocated by `torch.empty`
  each call, at most 65536 polls with `s_sleep 1`, stores NaN if a producer has not arrived, and clears the flags
  with plain stores. The fold order is fixed (planes 0, 1, … then its own), so the result is bitwise stable, but a
  flag word is read before this call writes it: correctness depends on the previous call having cleared it.
  `aklr/ops/README.md` § The kernel section and ABI asks for scratch rewritten before it is read; zeroing
  flags on every call (`../gemm/small_m.md` §4) removes the dependency. Its docstring
  (`_sk_count_ready`) says "this call's tag".
- **A cross-workgroup split lost on k1792's grid.** A one-launch split-K 2 with a flag hand-off measured 9.89 µs at
  m 7 to 15.76 µs at m 98, worse at every Point than the single-pass kernel (6.13–11.55 µs) *(field, E18)*.
- **Stale docstring.** k768's module docstring says one launch per call; at m 9397 `_launcher` runs a head
  launch over 9216 rows and a tail launch over 181 (as do k1536, k3584 and k4224 at m 9397, whose docstrings do not
  claim one launch).
- **`run` departs from the ABI.** n20480's `run` takes `out=None`, allocates the output and returns it when no
  `out` is passed (`aklr/ops/README.md` § The kernel section and ABI: `run` never allocates or returns an output).
  AKLR passes `out`, so no Point reaches that branch.
- **Hard-coded device.** n18432 and n20480 launch exactly 256 workgroups and divide N by 256; k1792, k3584 and
  k4224 add `256 − 224` L2-prefetch workgroups; the persistent grids walk tiles 256 apart. On fewer CUs the kernels
  still compute the right result, but the one-workgroup-per-CU design no longer holds (read in the code).
- **Code no Point reaches:** k3584's `_CONFIGS` rows 128 and 2731, k4224's row 128 and k1536's row 2731 (the
  Points jump from 112 to 893 and from 893 to 2732); k1792's `m <= 48` branch of `_config` (no Point between 43
  and 48).
- **Columns computed and dropped.** n18432 and n20480 compute 96-wide (or 80-wide) tiles for 72 or 80 stored
  columns; B rows past the real ones read zero past the buffer bound or come from LDS rows no DMA writes, and
  feed only unstored columns.

## Mechanism

Shared by the seven kernels:

- **MFMA `mfma_f32_16x16x32_bf16` with B as the first operand**, so a lane's four fp32 results are four
  consecutive output columns of one row and C leaves as 8- or 16-byte stores (`../gemm/epilogue.md`
  §3); k4224's split-K path keeps A first.
- **A and B into LDS by buffer→LDS DMA** (inline `buffer_load_dwordx4 … lds`, 16 B per lane), XOR-swizzled by
  16-byte chunk, in multi-stage rings (`../gemm/datapath.md` §2, §5); k768's skinny path (m ≤ 16) is
  the exception: A staged once in LDS from registers, B straight from global into registers.
- **No K split across workgroups** except k4224 at m 56–112. Where waves of one workgroup split K, an owner wave
  adds the partials from LDS in wave order; every output is one fp32 sum in an order fixed per output.
  K-start staggers (k1792, k3584, k4224, n18432, n20480) change where a workgroup's K loop starts, the same on
  every call.
- **A per-m configuration table** compiled in a `functools.cache` function on a Point's first call.

**m ≤ 112: one workgroup per column band, the whole K inside it.**

| | k1792 (all Points), k3584 (m ≤ 112), k4224 (m ≤ 48) | k768 (m ≤ 112) | k1536 (m ≤ 112) | n18432, n20480 (all Points) |
|---|---|---|---|---|
| Function | `_kernels` (k3584's `gemm_n7168_k3584_skinny` is k1792's kernel line for line) | `_build_skinny` (m ≤ 16), `_build_ring` (24, 32), `_build_large` (40–112) | `_build_tiled` | `_build_tiled` |
| Grid | 224 bands of 32 columns (64 split by rows between two workgroups of one XCD at m ≥ 84 / 80), + 32 L2-prefetch workgroups from m 28 / 24 | 224 workgroups | 224 workgroups | 256, one per CU; workgroup i stores 64 line-aligned columns plus an 8- (n18432) or 16-column (n20480) slice of a strip region, the slices of one XCD adjacent |
| Waves, K split | 4 or 8 waves; K over 2 or 4 waves at most m, none at m 21 and 98 (k1792) | 4 waves each a quarter of K (m ≤ 16); 3 waves each a third, wave-private DMA ring of 4 step pairs, `s_setprio` around the MFMAs (24, 32); 2–4 waves, tiles 48×32, 64×32, 48×64, 64×64, 4–5 stages (40–112) | 16×32 to 64×64 tiles, 2–8 waves, 4–6 stages | tile 16–128 × 96 (80 at m ≥ 98), 2–4 waves, 5 stages (m ≤ 32) or 3 |
| Loads | separate A and B rings (depths 8 to 3 by m); B `nt` | m ≤ 16: A once into LDS, B straight from global into registers; m ≥ 24: A and B by DMA (`sc0`) | B `nt` at most Points | A `sc0`, B `sc0 nt`; each stage's DMAs spread over the first two K steps, A first; fragments read one K step ahead |
| Other | K start staggered per band (×3 mod K tiles) from m 35 / 40; partials placed clear of the last tile's ring slots so the fold needs no extra barrier; kernel pointers preloaded into SGPRs | — | operand-swapped stores | K start staggered per workgroup; waves with no rows below m skip the last A DMA at m 33–64; kernel pointers preloaded |

The L2-prefetch workgroups DMA later B tiles of their XCD's bands into LDS and discard them, so the compute
workgroups find them in L2 (`../gemm/small_m.md` §6). k4224 at m 56–112 instead splits K 3 ways
(64×96 or 96×96 tiles, 225 workgroups) or 2 ways (112×64, 224 workgroups) with the last-split fold described
under Counter-signals (`../gemm/grid_pricing.md` §2).

**m 893–16384 (k768, k1536, k3584, k4224): a tiled async-LDS HGEMM, persistent at the two largest m.**

| m | k768 | k1536 | k3584 | k4224 |
|---|---|---|---|---|
| 893 | 128×128, 8 waves, 2 stages; 392 workgroups | 128×128, 4 waves, 2 stages; 392 | 128×224, 8-wave pipeline, 3 stages; 224 | 112×224, 7 waves, 3 stages; 256 |
| 2732 | 160×128, 4 waves, 2 stages; 1008 | 160×256, 8-wave pipeline, 3 stages; 504 | 160×256, 8-wave pipeline, 3 stages; 504 | 160×256, 8-wave pipeline, 3 stages; 504 |
| 9397 | persistent 256×256 over the first 9216 rows (1008 tiles on 256 workgroups) + a 96×64 tail launch over 181 rows | same | same | same |
| 16384 | persistent 256×256, 1792 tiles, 7 per workgroup | same | same | same |
| C stores `nt` | m 893, 2732, 16384 | m 2732–16384 | every tiled Point | every tiled Point |

- **The 8-wave pipeline** interleaves the next stage's DMAs with the MFMA clusters of the first K step, reads
  each fragment one cluster ahead, raises `s_setprio` around the clusters and pins the order with a
  `sched_group_barrier` table (`../gemm/mainloop.md` §2, `../gemm/schedule_and_locality.md`
  §3).
- **The persistent walk:** 256 workgroups take tiles 256 apart; after the last K stage's barrier each loads the
  next tile's first stage, then finishes the current tile, and each wave stages its C sub-tile through a private
  LDS slice past the operand ring, so the epilogue needs no workgroup barrier
  (`../gemm/grid_pricing.md` §10).
- **Tile order:** each XCD takes a contiguous run of tiles, walked a few block rows (`GROUP_M` 3–7) at a time
  (`../gemm/schedule_and_locality.md` §2); k768 and k1536 keep the linear order at m 893.

**What each part bought** (campaign ablations; each kept round timed beside the version before it):

- k1792: the single-pass ring with a fixed-order K-wave fold +3.5% (m 98) to +50.6% (m 35) over the campaign's
  starting kernel; the K-start stagger +0.2–8.4% at m ≥ 35; the row split at m 84, 98, in one round with L2 prefetch
  and a leaner epilogue, +6.4–6.6% for the round; per-m wave layouts +12.9% at m 98; kernel-argument preload
  +2.0–4.3% at every Point; `nt` on B +0.6–3.4% up to m 84. The L2-prefetch
  workgroups measured −3.3% to +2.6% by Point *(field, E9)*.
- k3584: the skinny/tiled pair +17.2–45.7% at m ≤ 112 *(field, E10)*; the persistent grid +1.5% (m 9397) and
  +0.9% (16384) *(field, E14)*.
- k4224: the k3584 design ported +24.2–107.5% at m ≤ 893; the split-K over one M block +3.6–10.4% at m 56–112;
  an exact one-round 112×224 tile +6.7% at m 893 *(field, E11)*; the persistent grid +1.4% (9397) and +2.6% (16384)
  in a same-process paired bench *(field, E17)*.
- k768: A staged in LDS with fewer waves +6.7–46.1% at m ≤ 112; the per-wave DMA ring +6.7–13.6% at m 24–112 when
  added; small-tile deep pipelines for m 56–112 +1.9–7.7% *(field, E12)*; the 8-wave pipelined tile +5.4% at
  m 9397 and the persistent grid +6.1% (9397) and +6.5% (16384) *(field, E14)*.
- k1536: aiter's async-LDS HGEMM with a per-m table +17.0–42.9% at m ≤ 112 over the starting kernel; the 3-stage
  160×256 8-wave tile +25.2% at m 2732; the head + tail split at m 9397 +4.4% *(field, E13)*; `nt` C stores
  +1.0–2.4% at m ≥ 2732 and the persistent grid +3.5% (9397) and +3.6% (16384) *(field, E14)*.
- n18432: the one-workgroup-per-CU port +7.5–113.4%; 72-column blocks on all 256 CUs with A issued first, in one
  round that also retiled m 112, +17.2% at m 112 for the round; the line-aligned block plus strip, in one round that
  also dropped an epilogue barrier, −0.4% to +3.1% *(field, E15)*. n20480 is that kernel at K 7168, kept in one round
  *(field, E16)*.

## Result

"Served" counts the Points at or above 1.03 in CI as of #102 (6d7cfec, CI run 42549655045, measurement profile 19),
n18432 as of CI run 41197527705 (a29581b), which #102 left unchanged; the `speedup` in each schema's `workloads` is
the current value *(field, E23)*. CI runs FlyDSL 0.3.2, the version of the local numbers below, not the 0.3.4.1 of
its image: its workflow installs a `flydsl==0.3.2` pin over the image (read from the workflow, not from a run's log;
`../pitfalls/measured_a_different_build.md`).

The other columns are the campaigns' own runs of the predecessor repository's `validate` (before the rename to
AKLR) on MI355X, FlyDSL 0.3.2: a lower bound of the CUDA-graph time ratio aiter / kernel, one run each,
capped at 2.000; "≥ 1.03" counts them against the dispatch threshold (that validate itself demoted below 1.02).

| File | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 locally | Served |
|---|---|---|---|---|---|---|---|
| k768 *(field, E1)* | 0.994–1.056 | 1.092 | 1.127 | 1.162 | 1.106 | 5 of 17 | 6 of 17 |
| k1536 *(field, E2)* | 1.005–1.273 | 1.025 | 1.088 | 1.102 | 0.999 | 14 of 17 | 13 of 17 |
| k1792 *(field, E3)* | 1.256–1.634 | – | – | – | – | 11 of 11 | 11 of 11 |
| k3584 *(field, E4)* | 1.096–1.366 | 1.194 | 1.039 | 1.061 | 0.945 | 16 of 17 | 14 of 17 |
| k4224 *(field, E5, E6)* | 1.069–1.256 | 1.126 | 1.020 | 1.049 | 0.935 | 15 of 17 | 14 of 17 |
| n18432 *(field, E7)* | 1.284–2.000 | – | – | – | – | 13 of 13 | 13 of 13 |
| n20480 *(field, E8)* | 1.071–1.417 | – | – | – | – | 23 of 23 | 23 of 23 |

"–": no such Point. k4224's m ≤ 2732 values were measured on the version before the persistent grid was added for
m 9397 and 16384 (its diff to main changes those two config rows and adds the persistent path); m 9397 and 16384
are from the run on the delivered version *(field, E5, E6)*.

## N 4096 × K 2048 and N 6144 × K 1536 at m 1–128: the k1792 band kernel

Two later files, one per schema name, run edited copies of k1792's `_config` / `_kernels` builder (the first column
of the m ≤ 112 table under § Mechanism) with tables of their own: `gemm_n4096_k2048/37c909c4b629ad59.json` (16 MiB of
weight) and `gemm_n6144_k1536/1d3a1fcaf77631e1.json` (18 MiB), each listing m 1, 2, 4, …, 128. No other file serves
these Points, so the yardstick is aiter; CI serves all 8 Points of each as of #115 (`qualifies()` in
`aklr/core/contract/spec/_workloads.py`; `AKLR_MIN_SPEEDUP`, default 1.03, `aklr/ops/README.md` § Environment
variables). As in k1792, each compute workgroup streams one column band of B (`nt`) and A (cached)
by buffer→LDS DMA through an A ring and a B ring of their own depths, B is the MFMA's first operand and C leaves in
8 B stores. Where the waves split K (wk > 1), the partials sit in LDS clear of the ring slots the last tile reads, and
the owner wave of each fragment adds them in wave order. The two workgroups of an M split, each owning half the rows
of a band, get ids 8 apart, so that they share an XCD. The pointer arguments are preloaded into SGPRs.

N 4096 × K 2048: 256 workgroups of four waves at every m.

| m | Band, BK | Waves (wm, wn, wk) | A / B slots | M split | First K tile |
|---|---|---|---|---|---|
| 1–16 | 16 columns, 256 | 1, 1, 4 | 8 / 8 | no | (3·band + band/8) mod 8 |
| 32 | 32, 128 | 1, 2, 2 | 16 / 6 | 2 | (band + 2·(band/8)) mod 16 |
| 64, 128 | 32, 128 | 2, 2, 1 | 4 / 14 (m 64), 4 / 11 (m 128) | 2 | (5·band + 2·(band/8)) mod 16 |

At m 64 and 128 the waves split rows and columns only, and a tile's fragments are read from LDS before its refill
DMAs go out.

N 6144 × K 1536: BK 128 (12 K tiles), 192 band workgroups, plus 64 prefetch workgroups from m 9.

| m | Band | Waves (wm, wn, wk) | A / B slots | M split | Prefetch from K tile | First K tile | DMA order |
|---|---|---|---|---|---|---|---|
| 1–8 | 32 | 2 (1, 2, 1) | 8 / 8 | no | none | 0 | A first |
| 9–16 | 32 | 4 (1, 2, 2) | 6 / 5 | no | 9 | 0 | B first |
| 17–32 | 32 | 4 (1, 2, 2) | 6 / 5 | no | 6 | 7·(band/8) mod 12 | A first |
| 33–95 | 64 | 8 (2, 2, 2) | 4 / 4 | 2 | 5 | 7·(band/8) mod 12 | B first |
| ≥ 96 | 64 | 16 (2, 4, 2) | 3 / 4 | 2 | 4 | 11·(band/8) mod 12 | A first |

The prefetch workgroups, 8 for each group of workgroup ids that share an XCD, DMA the later K tiles of that group's
bands (cached, not `nt`) into LDS nobody reads: 18–36 loads per thread, issued before one final wait, meant to put
those tiles in L2 for the compute workgroups.

Counter-signals:
- **No K split or prefetch at m ≤ 8 of N 6144 × K 1536, by a thin margin.** Two waves over columns read 4.51, 4.648
  and 4.727 µs at m 1, 4 and 8, against 4.529, 4.68 and 4.775 for the row they replaced (0.4–1.0%); at m 1, four waves
  splitting K four ways read 4.592 and two waves splitting it two ways 4.621 *(field, E22)*. At m ≤ 8 the 64 idle CUs
  get no prefetch workgroups.
- **The thinnest margin** is N 4096 × K 2048 at m 128, 1.118 *(field, E19)*.

AKLR validate (measurement profile 19) on one MI355X at main 0e02a81, not CI: N 4096 × K 2048 1.118–1.449 over aiter,
highest at m 16 *(field, E19)*; N 6144 × K 1536 1.427–1.592, highest at m 1 *(field, E20)*. Under stress every listed
Point gave the output of the call alone *(field, E21)*. Reproduce with `python -m aklr validate` on
`aklr/ops/gemm/impls/flydsl/gemm_n4096_k2048/37c909c4b629ad59.json` and
`aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/1d3a1fcaf77631e1.json` (§ Reproduce); to roll back, delete the file: its
Points are then served by no file, and the caller runs its own path.

## Reproduce

On an MI355X with the repository's FlyDSL 0.3.2, from the repository root, one schema at a time:

```
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k768/0567a6d854ac7a39.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k1536/c9ad7a129bb4cfe6.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k1792/f0a48d1c7cdce51f.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k3584/251a5ceb2d416e55.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n7168_k4224/2d72ae169867a6d1.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n18432_k1536/8cffee60e0ddc139.json
python -m aklr validate aklr/ops/gemm/impls/flydsl/gemm_n20480_k7168/23c7fb06d509a1bf.json
```

The report goes to `aklr_validation/report.md`; a run outside CI checks the file, the served numbers are CI's
(`aklr/ops/README.md` § Workflow). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base, so on an unchanged checkout it skips the bench
(`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). To read a kernel:
`git show <rev>:<schema> | python3 -c 'import json,sys; print(json.load(sys.stdin)["kernel"]["sources"][0]["content"])'`.

## Rollback

Delete the schema file: its Points go unserved, since no other file serves them. `AKLR_DISABLE=gemm@1.0` stops serving
the whole contract, every other bf16 GEMM schema included. A Point CI measures below 1.03 is not served, and a
`failed` accuracy never is. Roll back for a wrong result CI does not catch: a NaN or stale fold from k4224's flags at
m 56–112 after an aborted or timed-out launch, or a change of FlyDSL or LLVM.

## Evidence

`<schema>.json#kernel.py:<line>` is a line of the `kernel.py` a schema embeds in `kernel.sources`; "local validate"
is the predecessor repository's `validate` (before the rename to AKLR) outside CI on MI355X, the ratio lower bound over aiter per Point.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k768 delivered: m 8 0.994, 12 1.000, 16 1.005, 24 0.994, 32 1.002, 40 1.013, 48 1.015, 56 1.011, 60 1.004, 64 1.005, 80 1.056, 96 1.027, 112 1.010, 893 1.092, 2732 1.127, 9397 1.162, 16384 1.106; in another run with the earlier version and the large-m group's module side by side, m 32 read 1.038 and 1.029, m ≤ 112 0.998–1.064 and 0.996–1.065; the delivery check compared the final ISA at m ≤ 893 (gemm_n7168_k768)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k1536: m 8 1.273, 12 1.094, 16 1.077, 24 1.077, 32 1.005, 40 1.073, 48 1.075, 56 1.052, 60 1.046, 64 1.035, 80 1.061, 96 1.033, 112 1.037, 893 1.025, 2732 1.088, 9397 1.102, 16384 0.999 (gemm_n7168_k1536)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k1792: m 7 1.517, 14 1.490, 21 1.422, 28 1.387, 35 1.634, 42 1.611, 49 1.569, 56 1.547, 70 1.508, 84 1.488, 98 1.256 (gemm_n7168_k1792)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k3584: m 8 1.366, 12 1.173, 16 1.177, 24 1.234, 32 1.260, 40 1.257, 48 1.230, 56 1.176, 60 1.146, 64 1.142, 80 1.186, 96 1.135, 112 1.096, 893 1.194, 2732 1.039, 9397 1.061, 16384 0.945 (gemm_n7168_k3584)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k4224 before the persistent grid: m 8 1.214, 12 1.199, 16 1.184, 24 1.121, 32 1.069, 40 1.191, 48 1.204, 56 1.245, 60 1.254, 64 1.249, 80 1.251, 96 1.256, 112 1.187, 893 1.126, 2732 1.020, 9397 1.041, 16384 0.923 (gemm_n7168_k4224)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] k4224 delivered: 15 of 17 Points ≥ 1.03, m 9397 1.049, m 16384 0.935 (gemm_n7168_k4224)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n18432: m 8 2.000, 12 2.000, 16 2.000 (the bound's cap), 24 1.350, 32 1.336, 40 1.380, 48 1.373, 56 1.354, 60 1.346, 64 1.284, 80 1.521, 96 1.287, 112 1.416 (gemm_n18432_k1536)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local validate] n20480: m 7 1.299, 8 1.170, 12 1.114, 14 1.146, 16 1.159, 21 1.071, 24 1.076, 28 1.078, 32 1.071, 35 1.217, 40 1.147, 42 1.171, 48 1.210, 49 1.264, 56 1.195, 60 1.172, 64 1.155, 70 1.417, 80 1.395, 84 1.147, 96 1.109, 98 1.242, 112 1.193 (gemm_n20480_k7168)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] k1792 ladder: ring + K-wave fold (recorded) +39.7, 36.8, 29.2, 24.7, 50.6, 48.2, 42.6, 38.6, 40.5, 25.8, 3.5% (m 7–98); stagger +0.2–8.4% at m 35–98; row split +6.6/+6.4% at m 84/98; wave layouts +12.9% at m 98; preload +2.0–4.3% (recorded); B `nt` +0.6–3.4% at m ≤ 84; L2 prefetch −3.3 to +2.6% (gemm_n7168_k1792)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench, recorded round] k3584 skinny/tiled pair +17.2–45.7% at m 8–112, +9.9% at 893, −7.3% at 16384 against the starting kernel (gemm_n7168_k3584)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] k4224: port +26.3–107.5% at m ≤ 112 and +24.2% at 893 (recorded); split-K over one M block (recorded) +3.6, 5.8, 6.4, 9.0, 10.4, 6.7% at m 56, 60, 64, 80, 96, 112; 112×224 tile +6.7% at 893 (gemm_n7168_k4224)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] k768: A in LDS with fewer waves +6.7–46.1% at m ≤ 112 (recorded); DMA ring +6.7–13.6% at m 24–112 (recorded); deep-pipeline small tiles +3.3, 4.6, 3.4, 1.9, 3.1, 7.7% at m 56, 60, 64, 80, 96, 112 (remeasured) (gemm_n7168_k768)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] k1536: HGEMM port +17.0–42.9% at m ≤ 112 (recorded); 3-stage 160×256 8-wave tile +25.2% at 2732; head + tail at 9397 +4.4% (gemm_n7168_k1536)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop paired bench, group campaign] persistent grid: k768 +6.1/+6.5%, k1536 +3.5/+3.6%, k3584 +1.5/+0.9% at m 9397/16384; k768 8-wave tile +5.4% at 9397; k1536 `nt` C +2.4/+1.0/+1.6% at 2732/9397/16384
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench; remeasured rounds 3 pairs] n18432: port +7.5–113.4% (recorded); 72-column blocks + A first +0.1–17.2%, +17.2% at m 112 (recorded); line-aligned block + strip −0.4 to +3.1% (gemm_n18432_k1536)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop paired bench, recorded round] n20480: the one kept round, the n18432 design at N 20480, +7.9–42.9% over the starting kernel (gemm_n20480_k7168)
- E17 [2026-10, gfx950, FlyDSL 0.3.2, N=?, same-process paired bench] k4224 persistent grid against the version before: outputs bitwise equal, +1.4% at m 9397, +2.6% at m 16384 (gemm_n7168_k4224)
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=?, campaign driver bench] k1792: split-K 2 in one launch with a flag hand-off 9.89 µs at m 7 to 15.76 µs at m 98 against 6.13–11.55 µs for the single-pass kernel (gemm_n7168_k1792)
- S1 aklr/ops/gemm/contract.py:118 `GEMM` with `_UNSHUFFLED_VARIANTS` (:116) and `_dot_k()` (:91, fp32, `max_split_k` 16); `CompareSpec` defaults bitwise at aklr/core/contract/spec/_args.py:202
- S2 aklr/ops/gemm/impls/flydsl/gemm_n7168_k1792/f0a48d1c7cdce51f.json#kernel.py:50 `_config`; L2-prefetch workgroups at :86 and :276; B `nt` at :221; B-first MFMA at :271; owner-wave fold in wave order at :357; kernel pointers preloaded at :396
- S3 aklr/ops/gemm/impls/flydsl/gemm_n7168_k3584/251a5ceb2d416e55.json#kernel.py:26 `_SKINNY`; tiled `_CONFIGS` at :419; tail launch `_TAIL_SPLITS` at :471 and `_launcher` at :475; persistent walk at :1041
- S4 aklr/ops/gemm/impls/flydsl/gemm_n7168_k4224/2d72ae169867a6d1.json#kernel.py:1102 `_SK_CONFIGS`; `_sk_count_ready` static tag at :1155; poll at :1409, NaN at :1428, flag clear at :1432, producer tag at :1443
- S5 aklr/ops/gemm/impls/flydsl/gemm_n7168_k768/0567a6d854ac7a39.json#kernel.py:28 `_CONFIGS`; `_PIPE_TAIL_SPLITS` at :46; two launches at :146; docstring "one launch per call" at :1
- S6 aklr/ops/gemm/impls/flydsl/gemm_n7168_k1536/c9ad7a129bb4cfe6.json#kernel.py:26 `_CONFIGS`; persistent walk at :625
- S7 aklr/ops/gemm/impls/flydsl/gemm_n20480_k7168/23c7fb06d509a1bf.json#kernel.py:32 `_CONFIGS`; strip slice at :219; B `sc0 nt` at :333; K stagger at :396; `run(..., out=None)` allocating at :472 (gemm_n18432_k1536 has the same lines up to :471)
- E19 [2026-11, gfx950, FlyDSL 0.3.2, N=16, AKLR validate at 0e02a81, measurement profile 19: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] 37c909c4b629ad59: 1.372, 1.407, 1.395, 1.434, 1.449, 1.362, 1.297, 1.118 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n4096_k2048)
- E20 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E19] 1d3a1fcaf77631e1: 1.592, 1.579, 1.567, 1.546, 1.491, 1.486, 1.427, 1.449 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed (gemm_n6144_k1536)
- E21 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, main 0e02a81 with each schema, each try compared byte for byte with an eager call alone] 37c909c4b629ad59 and 1d3a1fcaf77631e1, every listed Point: the forge driver's conditions (streams limited to 16 and 64 CUs; a second concurrent call on the same inputs) and one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three AKLR input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E22 [2026-11, gfx950, FlyDSL 0.3.2, N=7, the campaign's configuration probe, medians of 7 interleaved readings] N 6144 × K 1536: the submitted m ≤ 8 row, waves (1, 2, 1), 4.51, 4.648, 4.727 µs at m 1, 4, 8 against the round's canonical 4.529, 4.68, 4.775; at m 1 waves (1, 1, 4) 4.592 and (1, 1, 2) 4.621
- E23 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI perf runs; measurement profile 19 for #102] Points at or above 1.03 as of #102 (6d7cfec, CI run 42549655045): k768 6 of 17 (m 64, 80, 893–16384), k1536 13 of 17 (not m 32, 893, 2732, 16384), k1792 11 of 11, k3584 14 of 17 (not m 2732–16384), k4224 14 of 17 (not m 2732–16384), n20480 23 of 23; n18432 13 of 13 from a29581b (CI run 41197527705), which #102 left unchanged (each schema's `workloads` at 6d7cfec)
- S8 aklr/ops/gemm/impls/flydsl/gemm_n4096_k2048/37c909c4b629ad59.json#kernel.py:45 `_config` per m; M-split partners at :131-136; first K tile at :175; B `nt` at :185; fragments read before the refills at :238-243; fold at :251-298; preload count 6 at :315
- S9 aklr/ops/gemm/impls/flydsl/gemm_n6144_k1536/1d3a1fcaf77631e1.json#kernel.py:47 `_config` per m; prefetch workgroups at :73-75 and :253-283 (48-load wait at :281-283); partials placed clear of the live ring slots at :110-131; first K tile at :197-199; preload count 6 at :365
