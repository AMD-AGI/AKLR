---
title: "bf16 GEMM, N 7168 × K 768–4224 and N 18432 / 20480 × K 1536 / 7168 (Kimi-K3) — one column band of B per workgroup at m ≤ 112, a tiled 8-wave grid that turns persistent at m 9397 and 16384; N 4096 × K 2048 and N 6144 × K 1536 at m 1–128 on the k1792 band kernel"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-128", "M 893-16384"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.994–1.162; 5 of 17 sizes ≥ 1.03 (m 80, 893, 2732, 9397, 16384); 6 of 17 in a repeat measurement", gpu: MI355X, workload: "N 7168 × K 768, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112, 893, 2732, 9397, 16384", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.999–1.273; 14 of 17 sizes ≥ 1.03; 13 of 17 in a repeat measurement", gpu: MI355X, workload: "N 7168 × K 1536, m 8–16384 (17 sizes)", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.256–1.634; 11 of 11 sizes ≥ 1.03; 11 of 11 in a repeat measurement", gpu: MI355X, workload: "N 7168 × K 1792, m 7, 14, 21, 28, 35, 42, 49, 56, 70, 84, 98", date: "2026-10-25"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "0.945–1.366; 16 of 17 sizes ≥ 1.03; 14 of 17 in a repeat measurement", gpu: MI355X, workload: "N 7168 × K 3584, m 8–16384 (17 sizes)", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM in the repeat measurement of § Result, a separate run from the table's one-run values: one-sided lower bound of the time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, per size", value: "0.876–3.429; 14 of 17 sizes ≥ 1.03 (not m 2732, 9397, 16384)", gpu: MI355X, workload: "N 7168 × K 4224, m 8–16384 (17 sizes)", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.284–2.000 (2.000 is the bound's cap); 13 of 13 sizes ≥ 1.03; 13 of 13 in a repeat measurement", gpu: MI355X, workload: "N 18432 × K 1536, m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112", date: "2026-10-25"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, lower bound of the CUDA-graph time ratio, one run, per size", value: "1.071–1.417; 23 of 23 sizes ≥ 1.03; 23 of 23 in a repeat measurement", gpu: MI355X, workload: "N 20480 × K 7168, m 7–112 (23 sizes)", date: "2026-10-26"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, one-sided lower bound of the time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, per size", value: "1.118–1.449; all 8 sizes at or above 1.03, in a separate run too", gpu: MI355X, workload: "N 4096 × K 2048, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's tuned bf16 GEMM, one-sided lower bound of the time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, per size", value: "1.427–1.592; all 8 sizes at or above 1.03, in a separate run too", gpu: MI355X, workload: "N 6144 × K 1536, m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
---

# bf16 GEMM at N 7168, 18432 and 20480 (Kimi-K3): a column band of B per workgroup, then a tiled grid

Seven of this page's nine FlyDSL kernels for `out = a @ b.T` are the Kimi-K3 set, one kernel per (N, K), each
built for a list of m. "m" is M. The five N 7168 kernels are named by their K below (k768, k1536, k1792, k3584,
k4224) and the other two by their N (n18432 for N 18432 × K 1536, n20480 for N 20480 × K 7168).
k1536's tiled builder and k768's tiled and pipelined builders also run N 6144 × K 1536 and N 7168 × K 512
at m 256–4096: [bf16_large_m_whole_k_tiles.md](bf16_large_m_whole_k_tiles.md).

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`. Two runs
  must give the same bytes, and the result is held to the error bound of an fp32 dot product over K.
- N × K = 7168 × 768, 1536, 1792, 3584, 4224 (a weight of 10.5–57.75 MiB), 18432 × 1536 (54 MiB) and
  20480 × 7168 (280 MiB): at m ≤ 112 each call streams the whole weight once
  (`../gemm/small_m.md` §2). k1792 and n18432 / n20480 have only sizes at m ≤ 112; the other
  four also have m 893, 2732, 9397, 16384.
- At every size the yardstick is aiter's tuned bf16 GEMM, and a size counts as won from 1.03 over it.
- **94 of their 115 sizes read at or above 1.03** in a repeat measurement; the count per kernel is in § Result.

## Counter-signals

- **Below 1.03 in the one-run measurement** (the repeat measurement's count per kernel is in § Result):
  - k768 at m 8–64, 96 and 112: 0.994–1.027 *(field, E1)*. These sizes run the skinny, ring and small-tile paths
    below; the kernel's author notes that aiter is tuned for this shape.
  - k1536 at m 32 (1.005), 893 (1.025) and 16384 (0.999) *(field, E2)*; k3584 at 16384 (0.945) *(field, E4)*;
    k4224 at 2732 (1.020) and 16384 (0.935) *(field, E5, E6)*. At m 16384 all three run the persistent
    256×256 grid below.
  - Sizes at m ≤ 112 this close to 1.03 cross it between runs: k768's m 32 read 1.002 in one run and 1.029 and 1.038
    in the two arms of another, all three with the same final ISA at that size (the large-m rework left
    the ISA at m ≤ 893 unchanged) *(field, E1)*.
- **k4224 at m 56–112 folds a cross-workgroup split-K through flags whose tag does not change between calls.**
  The last split polls flag words that hold a fixed constant XOR the tile index XOR `split << 16`, in a workspace
  allocated by `torch.empty`
  each call, at most 65536 polls with `s_sleep 1`, stores NaN if a producer has not arrived, and clears the flags
  with plain stores. The fold order is fixed (planes 0, 1, … then its own), so the result is bitwise stable, but a
  flag word is read before this call writes it: correctness depends on the previous call having cleared it.
  A workspace from `torch.empty` holds undefined bytes, so scratch is to be written in the same call before it is
  read; zeroing flags on every call (`../gemm/small_m.md` §4) removes the dependency. The kernel's own comment
  calls the constant "this call's tag". After an aborted or timed-out launch, a later call can return NaN or a
  fold of stale planes.
- **A cross-workgroup split lost on k1792's grid.** A one-launch split-K 2 with a flag hand-off measured 9.89 µs at
  m 7 to 15.76 µs at m 98, worse at every size than the single-pass kernel (6.13–11.55 µs) *(field, E18)*.
- **A head and a tail launch.** k768's own description says one launch per call; at m 9397 it runs a head
  launch over 9216 rows and a tail launch over 181 (as do k1536, k3584 and k4224 at m 9397, whose descriptions do
  not claim one launch).
- **Hard-coded device.** n18432 and n20480 launch exactly 256 workgroups and divide N by 256; k1792, k3584 and
  k4224 add `256 − 224` L2-prefetch workgroups; the persistent grids walk tiles 256 apart. On fewer CUs the kernels
  still compute the right result, but the one-workgroup-per-CU design no longer holds (read in the code).
- **Configurations no measured size reaches:** k3584's table rows 128 and 2731, k4224's row 128 and k1536's row
  2731 (the sizes jump from 112 to 893 and from 893 to 2732); k1792's m ≤ 48 branch (no size between 43
  and 48).
- **Columns computed and dropped.** n18432 and n20480 compute 96-wide (or 80-wide) tiles for 72 or 80 stored
  columns; B rows past the real ones read zero past the buffer bound or come from LDS rows no DMA writes, and
  feed only unstored columns.

## Mechanism

Shared by the seven kernels:

- **MFMA `mfma_f32_16x16x32_bf16` with B as the first operand**, so a lane's four fp32 results are four
  consecutive output columns of one row and C leaves as 8- or 16-byte stores (`../gemm/epilogue.md`
  §3); k4224's split-K path keeps A first. In k1792:

  ```python
  row16 = lane % 16
  quad = lane // 16
  # B first: lane (row16, quad) holds row row16 of the 16×16 fragment, columns quad*4 .. quad*4 + 3
  for i in range_constexpr(rmw):
      for j in range_constexpr(rnw):
          p = i * rnw + j
          out[p] = rocdl.mfma_f32_16x16x32_bf16(
              T.vec(4, T.f32), [bv[j], av[i], out[p], 0, 0, 0]
          )

  def store(p, total):
      i = wi * rmw + p // rnw
      j = wj * rnw + p % rnw
      row = i * 16 + row16
      col = no + j * 16 + quad * 4
      # four bf16 packed in two dwords: one 8 B store per lane and fragment, no LDS round trip
      packed = vector.bitcast(
          T.vec(2, T.i32), _raw(arith.truncf(T.vec(4, T.bf16), total))
      )
      rocdl.raw_ptr_buffer_store(
          packed,
          cr,
          _raw(fx.Int32((row * n + col) * 2)),
          _raw(fx.Int32(0)),
          aux=ir.IntegerAttr.get(T.i32, 0),
      )
  ```
- **A and B into LDS by buffer→LDS DMA** (inline `buffer_load_dwordx4 … lds`, 16 B per lane), XOR-swizzled by
  16-byte chunk, in multi-stage rings (`../gemm/datapath.md` §2, §5); k768's skinny path (m ≤ 16) is
  the exception: A staged once in LDS from registers, B straight from global into registers.
- **No K split across workgroups** except k4224 at m 56–112. Where waves of one workgroup split K, an owner wave
  adds the partials from LDS in wave order; every output is one fp32 sum in an order fixed per output.
  K-start staggers (k1792, k3584, k4224, n18432, n20480) change where a workgroup's K loop starts, the same on
  every call. The fold of k1792:

  ```python
  # nloc: fragments per wave; wk: waves that split K; wq: this wave's K slot
  for p in range_constexpr(nloc):
      if const_expr(wk == 1):  # no K split inside the workgroup: store the accumulator
          store(p, acc[p])
      else:
          owner = p % wk  # the wave that folds fragment p; the others have written theirs to LDS
          if wq == owner:
              total = None
              for s in range_constexpr(wk):  # K slot 0, 1, ...: the same order in every run
                  if const_expr(s == owner):
                      v = acc[p]
                  else:
                      src = wave - wq + s
                      off = part_base + ((src * nloc + p) * 64 + lane) * 8
                      v = fx.ptr_load(
                          sp + off,
                          result_type=fx.Vector.make_type(8, fx.BFloat16),
                      )
                      v = vector.bitcast(T.vec(4, T.f32), _raw(v))
                  total = v if total is None else arith.addf(total, v)
              store(p, total)
  ```

  Not this: K split two ways across workgroups in one launch, with a flag hand-off between the two. On k1792's
  grid that was slower at every size than the single pass (§ Counter-signals).
- **A per-m configuration table** compiled in a `functools.cache` function on a size's first call.

**m ≤ 112: one workgroup per column band, the whole K inside it.**

| | k1792 (all sizes), k3584 (m ≤ 112), k4224 (m ≤ 48) | k768 (m ≤ 112) | k1536 (m ≤ 112) | n18432, n20480 (all sizes) |
|---|---|---|---|---|
| Builder | one band builder (k3584's is k1792's kernel line for line) | a skinny builder (m ≤ 16), a ring builder (24, 32), a tiled builder (40–112) | a tiled builder | a tiled builder |
| Grid | 224 bands of 32 columns (64 split by rows between two workgroups of one XCD at m ≥ 84 / 80), + 32 L2-prefetch workgroups from m 28 / 24 | 224 workgroups | 224 workgroups | 256, one per CU; workgroup i stores 64 line-aligned columns plus an 8- (n18432) or 16-column (n20480) slice of a strip region, the slices of one XCD adjacent |
| Waves, K split | 4 or 8 waves; K over 2 or 4 waves at most m, none at m 21 and 98 (k1792) | 4 waves each a quarter of K (m ≤ 16); 3 waves each a third, wave-private DMA ring of 4 step pairs, `s_setprio` around the MFMAs (24, 32); 2–4 waves, tiles 48×32, 64×32, 48×64, 64×64, 4–5 stages (40–112) | 16×32 to 64×64 tiles, 2–8 waves, 4–6 stages | tile 16–128 × 96 (80 at m ≥ 98), 2–4 waves, 5 stages (m ≤ 32) or 3 |
| Loads | separate A and B rings (depths 8 to 3 by m); B `nt` | m ≤ 16: A once into LDS, B straight from global into registers; m ≥ 24: A and B by DMA (`sc0`) | B `nt` at most sizes | A `sc0`, B `sc0 nt`; each stage's DMAs spread over the first two K steps, A first; fragments read one K step ahead |
| Other | K start staggered per band (×3 mod K tiles) from m 35 / 40; partials placed clear of the last tile's ring slots so the fold needs no extra barrier; kernel pointers preloaded into SGPRs | — | operand-swapped stores | K start staggered per workgroup; waves with no rows below m skip the last A DMA at m 33–64; kernel pointers preloaded |

The L2-prefetch workgroups DMA later B tiles of their XCD's bands into LDS and discard them, so the compute
workgroups find them in L2 (`../gemm/small_m.md` §6). In k1792:

```python
# dma: one buffer→LDS load; offsets: a band's per-thread byte offsets; tile_byte: a K tile on a band's staggered walk
if bid >= fx.Int32(compute_wgs):
    # Pull later K tiles of the B bands of this XCD into L2; the LDS copies are discarded.
    pidx = bid - compute_wgs
    xcd = pidx % _XCDS
    grp = pidx // _XCDS
    p_off = []
    p_start = []
    for jj in range_constexpr(per_pf):
        if const_expr(msplit):
            i = grp * per_pf + jj
            cbb = xcd + _XCDS * (i // 2)
            p_off.append(offsets((cbb * 2 + i % 2) * pbn, pnb))
            p_start.append(start_tile(cbb))
        else:
            cb = xcd + _XCDS * (grp * per_pf + jj)  # a band whose workgroup id is this XCD's mod 8
            p_off.append(offsets(cb * bn, nb))
            p_start.append(start_tile(cb))
    issued = 0
    for t in range_constexpr(pf_t0, tiles):  # from the first prefetched K tile to the last
        for jj in range_constexpr(per_pf):
            tb = tile_byte(t, p_start[jj])
            for j in range_constexpr(pnb):
                dma(
                    br,
                    p_off[jj][j] + tb,
                    lds_base + wave * 1024 + j * threads * 16,  # every tile lands on the same LDS bytes
                )
                issued += 1
        if const_expr(issued > pf_depth):
            wait(pf_depth)  # s_waitcnt vmcnt(48): at most 48 loads in flight
    wait(0)
```

k4224 at m 56–112 instead splits K 3 ways
(64×96 or 96×96 tiles, 225 workgroups) or 2 ways (112×64, 224 workgroups) with the last-split fold described
under Counter-signals (`../gemm/grid_pricing.md` §2).

**m 893–16384 (k768, k1536, k3584, k4224): a tiled async-LDS HGEMM, persistent at the two largest m.**

| m | k768 | k1536 | k3584 | k4224 |
|---|---|---|---|---|
| 893 | 128×128, 8 waves, 2 stages; 392 workgroups | 128×128, 4 waves, 2 stages; 392 | 128×224, 8-wave pipeline, 3 stages; 224 | 112×224, 7 waves, 3 stages; 256 |
| 2732 | 160×128, 4 waves, 2 stages; 1008 | 160×256, 8-wave pipeline, 3 stages; 504 | 160×256, 8-wave pipeline, 3 stages; 504 | 160×256, 8-wave pipeline, 3 stages; 504 |
| 9397 | persistent 256×256 over the first 9216 rows (1008 tiles on 256 workgroups) + a 96×64 tail launch over 181 rows | same | same | same |
| 16384 | persistent 256×256, 1792 tiles, 7 per workgroup | same | same | same |
| C stores `nt` | m 893, 2732, 16384 | m 2732–16384 | every tiled size | every tiled size |

- **The 8-wave pipeline** interleaves the next stage's DMAs with the MFMA clusters of the first K step, reads
  each fragment one cluster ahead, raises `s_setprio` around the clusters and pins the order with a
  `sched_group_barrier` table (`../gemm/mainloop.md` §2, `../gemm/schedule_and_locality.md`
  §3).
- **The persistent walk:** 256 workgroups take tiles 256 apart; after the last K stage's barrier each loads the
  next tile's first stage, then finishes the current tile, and each wave stages its C sub-tile through a private
  LDS slice past the operand ring, so the epilogue needs no workgroup barrier
  (`../gemm/grid_pricing.md` §10). In k1536, with a two-stage ring:

  ```python
  # pid: the workgroup; _CUS = 256
  ldg_sts_b_async(fx.Int32(0), 0)  # the first tile's first K stage
  ldg_sts_a_async(fx.Int32(0), 0)
  for t in range(n_walk):
      tile = pid + t * _CUS  # this workgroup's tiles are 256 apart
      bm, bn = tile_coords(tile)
      if const_expr(PERSIST_EVEN):  # every workgroup walks the same number of tiles
          tile_next = (t + 1 < PERSIST_TPW).select(tile + _CUS, tile)
      else:
          tile_next = (tile + _CUS < fx.Int32(TILES)).select(
              tile + _CUS, tile
          )
      bm2, bn2 = tile_coords(tile_next)
      mo = fx.Index(bm * BLOCK_M)
      no = fx.Index(bn * BLOCK_N)
      init_p = [fx.Int32(0), arith.constant(0, index=True)] + c_frags
      for bki, state in range(0, BLOCK_K_LOOPS - 1, 1, init=init_p):
          ...  # every K stage but the last: load the next stage, compute this one
      cur = fx.Index(results[1])
      cf = results[2:]
      wait_barrier(0)
      # Every wave is past its reads of stage 0: start the next tile's first stage.
      ldg_sts_b_async(fx.Int32(0), 0, fx.Index(bn2 * BLOCK_N))
      ldg_sts_a_async(fx.Int32(0), 0, fx.Index(bm2 * BLOCK_M))
      cf = ldmatrix_compute_tile_streaming(cur, cf)  # the current tile's last K stage
      epilogue_slice(cf, bm, bn)  # C through this wave's own LDS slice: no workgroup barrier
  ```
- **Tile order:** each XCD takes a contiguous run of tiles, walked a few block rows (3–7) at a time
  (`../gemm/schedule_and_locality.md` §2); k768 and k1536 keep the linear order at m 893.

**What each part bought** (ablations; each kept step timed beside the version before it):

- k1792: the single-pass ring with a fixed-order K-wave fold +3.5% (m 98) to +50.6% (m 35) over the
  starting kernel; the K-start stagger +0.2–8.4% at m ≥ 35; the row split at m 84, 98, in one step with L2 prefetch
  and a leaner epilogue, +6.4–6.6% for the step; per-m wave layouts +12.9% at m 98; kernel-argument preload
  +2.0–4.3% at every size; `nt` on B +0.6–3.4% up to m 84. The L2-prefetch
  workgroups measured −3.3% to +2.6% by size *(field, E9)*.
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
  step that also retiled m 112, +17.2% at m 112 for the step; the line-aligned block plus strip, in one step that
  also dropped an epilogue barrier, −0.4% to +3.1% *(field, E15)*. n20480 is that kernel at K 7168, kept in one step
  *(field, E16)*.

## Result

The last column counts the sizes at or above 1.03 in a repeat measurement: the one-sided lower bound of the aiter /
kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, and for n18432 an earlier
repeat whose ruler is not stated *(field, E23)*. The repeat ran FlyDSL 0.3.2, the version of the one-run numbers
below, not the 0.3.4.1 of its container image: a `flydsl==0.3.2` pin was installed over the image for the run (read
from the run's setup, not from its log; `../pitfalls/measured_a_different_build.md`).

The other columns are one run each on MI355X, FlyDSL 0.3.2, of a lower bound of the CUDA-graph time ratio
aiter / kernel, capped at 2.000; "≥ 1.03 in one run" counts the sizes at or above 1.03 in it.

| Kernel | m ≤ 112 | m 893 | m 2732 | m 9397 | m 16384 | ≥ 1.03 in one run | ≥ 1.03 in the repeat |
|---|---|---|---|---|---|---|---|
| k768 *(field, E1)* | 0.994–1.056 | 1.092 | 1.127 | 1.162 | 1.106 | 5 of 17 | 6 of 17 |
| k1536 *(field, E2)* | 1.005–1.273 | 1.025 | 1.088 | 1.102 | 0.999 | 14 of 17 | 13 of 17 |
| k1792 *(field, E3)* | 1.256–1.634 | – | – | – | – | 11 of 11 | 11 of 11 |
| k3584 *(field, E4)* | 1.096–1.366 | 1.194 | 1.039 | 1.061 | 0.945 | 16 of 17 | 14 of 17 |
| k4224 *(field, E5, E6)* | 1.069–1.256 | 1.126 | 1.020 | 1.049 | 0.935 | 15 of 17 | 14 of 17 |
| n18432 *(field, E7)* | 1.284–2.000 | – | – | – | – | 13 of 13 | 13 of 13 |
| n20480 *(field, E8)* | 1.071–1.417 | – | – | – | – | 23 of 23 | 23 of 23 |

"–": no such size. k4224's m ≤ 2732 values were measured on the version before the persistent grid was added for
m 9397 and 16384 (the final version changes those two configuration rows and adds the persistent path); m 9397 and
16384 are from the run on the final version *(field, E5, E6)*. In the repeat k4224 read 0.876–3.429 over its 17 sizes,
below 1.03 at m 2732, 9397 and 16384 *(field, E24)*.

## N 4096 × K 2048 and N 6144 × K 1536 at m 1–128: the k1792 band kernel

Two later kernels run edited copies of k1792's band builder (the first column
of the m ≤ 112 table under § Mechanism) with tables of their own: N 4096 × K 2048 (16 MiB of
weight) and N 6144 × K 1536 (18 MiB), each built for m 1, 2, 4, …, 128. The yardstick is aiter's tuned bf16 GEMM: all
8 sizes of each read at or above 1.03 over it, in a separate run too *(field, E19, E20, E25)*.
As in k1792, each compute workgroup streams one column band of B (`nt`) and A (cached)
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

The one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed
before each call, on one MI355X: N 4096 × K 2048 1.118–1.449 over aiter,
highest at m 16 *(field, E19)*; N 6144 × K 1536 1.427–1.592, highest at m 1 *(field, E20)*. Under stress every
size gave the output of the call alone *(field, E21)*.

## Evidence

"One-run ratio bound" is a lower bound of the CUDA-graph time ratio aiter / kernel per size, capped at 2.000, from
one run on MI355X.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k768, final version: m 8 0.994, 12 1.000, 16 1.005, 24 0.994, 32 1.002, 40 1.013, 48 1.015, 56 1.011, 60 1.004, 64 1.005, 80 1.056, 96 1.027, 112 1.010, 893 1.092, 2732 1.127, 9397 1.162, 16384 1.106; in another run with the earlier version and the large-m rework side by side, m 32 read 1.038 and 1.029, m ≤ 112 0.998–1.064 and 0.996–1.065; the final ISA of the two was compared at m ≤ 893
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k1536: m 8 1.273, 12 1.094, 16 1.077, 24 1.077, 32 1.005, 40 1.073, 48 1.075, 56 1.052, 60 1.046, 64 1.035, 80 1.061, 96 1.033, 112 1.037, 893 1.025, 2732 1.088, 9397 1.102, 16384 0.999
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k1792: m 7 1.517, 14 1.490, 21 1.422, 28 1.387, 35 1.634, 42 1.611, 49 1.569, 56 1.547, 70 1.508, 84 1.488, 98 1.256
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k3584: m 8 1.366, 12 1.173, 16 1.177, 24 1.234, 32 1.260, 40 1.257, 48 1.230, 56 1.176, 60 1.146, 64 1.142, 80 1.186, 96 1.135, 112 1.096, 893 1.194, 2732 1.039, 9397 1.061, 16384 0.945
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k4224 before the persistent grid: m 8 1.214, 12 1.199, 16 1.184, 24 1.121, 32 1.069, 40 1.191, 48 1.204, 56 1.245, 60 1.254, 64 1.249, 80 1.251, 96 1.256, 112 1.187, 893 1.126, 2732 1.020, 9397 1.041, 16384 0.923
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] k4224, final version: 15 of 17 sizes ≥ 1.03, m 9397 1.049, m 16384 0.935
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n18432: m 8 2.000, 12 2.000, 16 2.000 (the bound's cap), 24 1.350, 32 1.336, 40 1.380, 48 1.373, 56 1.354, 60 1.346, 64 1.284, 80 1.521, 96 1.287, 112 1.416
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one-run ratio bound] n20480: m 7 1.299, 8 1.170, 12 1.114, 14 1.146, 16 1.159, 21 1.071, 24 1.076, 28 1.078, 32 1.071, 35 1.217, 40 1.147, 42 1.171, 48 1.210, 49 1.264, 56 1.195, 60 1.172, 64 1.155, 70 1.417, 80 1.395, 84 1.147, 96 1.109, 98 1.242, 112 1.193
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] k1792 ladder: ring + K-wave fold (recorded) +39.7, 36.8, 29.2, 24.7, 50.6, 48.2, 42.6, 38.6, 40.5, 25.8, 3.5% (m 7–98); stagger +0.2–8.4% at m 35–98; row split +6.6/+6.4% at m 84/98; wave layouts +12.9% at m 98; preload +2.0–4.3% (recorded); B `nt` +0.6–3.4% at m ≤ 84; L2 prefetch −3.3 to +2.6%
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench against the starting kernel, as recorded] k3584 skinny/tiled pair +17.2–45.7% at m 8–112, +9.9% at 893, −7.3% at 16384 against the starting kernel
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] k4224: port +26.3–107.5% at m ≤ 112 and +24.2% at 893 (recorded); split-K over one M block (recorded) +3.6, 5.8, 6.4, 9.0, 10.4, 6.7% at m 56, 60, 64, 80, 96, 112; 112×224 tile +6.7% at 893
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] k768: A in LDS with fewer waves +6.7–46.1% at m ≤ 112 (recorded); DMA ring +6.7–13.6% at m 24–112 (recorded); deep-pipeline small tiles +3.3, 4.6, 3.4, 1.9, 3.1, 7.7% at m 56, 60, 64, 80, 96, 112 (remeasured)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] k1536: HGEMM port +17.0–42.9% at m ≤ 112 (recorded); 3-stage 160×256 8-wave tile +25.2% at 2732; head + tail at 9397 +4.4%
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=3, paired bench of each step against the version before it] persistent grid: k768 +6.1/+6.5%, k1536 +3.5/+3.6%, k3584 +1.5/+0.9% at m 9397/16384; k768 8-wave tile +5.4% at 9397; k1536 `nt` C +2.4/+1.0/+1.6% at 2732/9397/16384
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench of each step against the version before it; remeasured steps 3 pairs] n18432: port +7.5–113.4% (recorded); 72-column blocks + A first +0.1–17.2%, +17.2% at m 112 (recorded); line-aligned block + strip −0.4 to +3.1%
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=?, paired bench against the starting kernel, as recorded] n20480: its one kept step, the n18432 design at N 20480, +7.9–42.9% over the starting kernel
- E17 [2026-10, gfx950, FlyDSL 0.3.2, N=?, same-process paired bench] k4224 persistent grid against the version before: outputs bitwise equal, +1.4% at m 9397, +2.6% at m 16384
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the tuning run's own bench, timer not stated] k1792: split-K 2 in one launch with a flag hand-off 9.89 µs at m 7 to 15.76 µs at m 98 against 6.13–11.55 µs for the single-pass kernel
- E19 [2026-11, gfx950, FlyDSL 0.3.2, N=16, one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with L2 flushed before each call, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] N 4096 × K 2048: 1.372, 1.407, 1.395, 1.434, 1.449, 1.362, 1.297, 1.118 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed
- E20 [2026-11, gfx950, FlyDSL 0.3.2, N=16, the ruler of E19] N 6144 × K 1536: 1.592, 1.579, 1.567, 1.546, 1.491, 1.486, 1.427, 1.449 at m 1, 2, 4, 8, 16, 32, 64, 128; accuracy passed
- E21 [2026-11, gfx950, FlyDSL 0.3.2, N=20, eager and op-level CUDA-graph stress on one MI355X, each try compared byte for byte with an eager call alone] N 4096 × K 2048 and N 6144 × K 1536, every size: streams limited to 16 and 64 CUs; a second concurrent call on the same inputs; one captured graph replayed on the same inputs, 20 tries each; one graph replayed 24 times with three input sets rotated into its buffers; two calls at once on two streams with different inputs, unrestricted and both on the same 64 CUs, 20 tries each: 0 tries differ
- E22 [2026-11, gfx950, FlyDSL 0.3.2, N=7, a configuration probe, medians of 7 interleaved readings] N 6144 × K 1536: the final m ≤ 8 row, waves (1, 2, 1), 4.51, 4.648, 4.727 µs at m 1, 4, 8 against 4.529, 4.68, 4.775 for the row it replaced; at m 1 waves (1, 1, 4) 4.592 and (1, 1, 2) 4.621
- E23 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of aiter dbd8bf5, installed over the image for these runs; read from the runs' setup, not from a log), N=1, a repeat measurement; the ruler of E19 for all but n18432] sizes at or above 1.03: k768 6 of 17 (m 64, 80, 893–16384), k1536 13 of 17 (not m 32, 893, 2732, 16384), k1792 11 of 11, k3584 14 of 17 (not m 2732–16384), k4224 14 of 17 (not m 2732–16384), n20480 23 of 23; n18432 13 of 13 from an earlier repeat, ruler not stated
- E24 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the repeat measurement of E23, on the ruler of E19] k4224: 1.187, 1.216, 1.203, 3.429, 3.260, 1.196, 1.210, 1.239, 1.244, 1.272, 1.242, 1.228, 1.162 at m 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 80, 96, 112; 1.126, 0.876, 0.986, 1.009 at m 893, 2732, 9397, 16384; 14 of 17 sizes at or above 1.03; accuracy passed at each
- E25 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] N 4096 × K 2048: 1.360, 1.401, 1.392, 1.426, 1.426, 1.346, 1.284, 1.126 at m 1, 2, 4, 8, 16, 32, 64, 128; N 6144 × K 1536: 1.589, 1.589, 1.585, 1.561, 1.519, 1.481, 1.440, 1.460 at the same m; all 8 sizes of each at or above 1.03; accuracy passed at each
