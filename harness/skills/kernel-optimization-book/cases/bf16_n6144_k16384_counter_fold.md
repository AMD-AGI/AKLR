---
title: "bf16 GEMM, N 6144 × K 16384 at m 1–512 — a column strip of B per workgroup at m ≤ 16; from m 32, K split 4 or 8 ways across workgroups and folded inside the launch by the last split to count in on a counter zeroed in the call"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-16", "M 32-128", "M 256-512"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=4–6
measured:
  - {metric: "kernel time per size, the fold of this page against an owner-per-slice fold that waits, per-kernel CUDA-graph timer, arms interleaved", value: "43.90 against 37.95 µs at m 128, 33.19 against 32.54 at m 64, within 0.5% at m ≤ 32; 68.6 against 56.2 at m 256, 94.3 against 87.1 at m 512", gpu: MI355X, workload: "N 6144 × K 16384, m 1–128 on one kernel, m 256 and 512 on the other", date: "2026-11"}
  - {metric: "kernel time per size with the counters zeroed in the call against counters kept in state that outlives the call, per-kernel CUDA-graph timer, the two versions interleaved", value: "36.41 against 32.23 µs at m 32, 37.30 against 33.12 at m 64, 48.73 against 44.06 at m 128, 70.62 against 67.48 at m 256, 98.59 against 94.49 at m 512; within 0.4 µs at m ≤ 16 and m ≥ 1024", gpu: MI355X, workload: "N 6144 × K 16384, m 32, 64, 128, 256, 512", date: "2026-11"}
  - {metric: "NaN in the output of eager calls on streams limited to 64 and 128 CUs, summed over the 3 tries of a case, output against the call alone", value: "this fold none, bit for bit equal; a fold that waits 2,340,864 to 4,276,224", gpu: MI355X, workload: "N 6144 × K 16384, m 256 and 512", date: "2026-11"}
  - {metric: "kernel time at m 128 with only the split, the tile or the partial format changed, per-kernel CUDA-graph timer, arms rotated", value: "the kernel's split 8 on 128×192 tiles 47.47–47.61 µs; split 4 on 128×96 tiles, 256 workgroups, 43.20–43.32", gpu: MI355X, workload: "N 6144 × K 16384, m 128", date: "2026-11"}
---

# bf16 GEMM at N 6144 × K 16384: K split across workgroups, folded by the last split to count in

Two FlyDSL kernels for `out = a @ b.T` on a 192 MiB weight: one is written for m 1, 2, 4, …, 128, the other for
m 256 to 4096, of which 256 and 512 run FlyDSL code. "m" is M. At m ≤ 16 each workgroup streams a strip of B over
the whole K. From m 32, K is split 4 or 8 ways across workgroups and the partials are folded inside the launch: each
split publishes its packed partial and counts itself in on an atomic counter, and the split that finds all the
others already counted adds the planes in split order, so that no workgroup waits for another.
[gemm/wait_free_fold](../gemm/wait_free_fold.md) explains that fold step by step on the code of the m 1–128 kernel,
which is under § Code; `../gemm/small_m.md` §4 has the split-K rules behind it and `../gemm/small_m.md` §6 the
K-start stagger.

## Applies when

- `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, `out = a @ b.T`, no bias, bf16 `out`, with N = 6144
  and K = 16384.
- Sizes. The first kernel is written for m 1, 2, 4, 8, 16, 32, 64 and 128; its tables cover every m up to 128 (an m
  between two sizes takes the row of the next size up) and send a larger m to `torch.matmul`. The second is written
  for m 256, 512, 1024, 2048 and 4096, and has a FlyDSL kernel at 256 and 512 only (§ Counter-signals).
- Both are held to `bitwise` accuracy: two runs on the same inputs must give the same bytes. Each sum of partials
  adds in a fixed order, wave order inside a strip's workgroup and split order in the fold, and the atomics act on
  counters, not on data (`../gemm/correctness.md` §8).
- At N = 6144 and small M, 256 workgroups need `tile_n` 48 with split-K 2, 96 with 4 or 192 with 8
  (`../gemm/small_m.md` §3). These kernels take 96 with split-K 4 at m 32 and 64 and 192 with split-K 8 at m 128 and
  256; at m 512 two M-blocks of 192-column tiles with split-K 4 make 256 as well.
- The book gives no speedup of these kernels over a baseline; § Result has their times and what was compared.

## Counter-signals

- **From m 64 the fold costs more than one that waits.** Its whole read goes through one workgroup, which at m 128
  reads 7 planes, and the cost grows with the split count. Against an owner-per-slice spin fold, which waits and
  spreads its work over all splits, it read 68.6 against 56.2 µs at m 256, the widest gap
  *(field, E1)*; the other sizes are in § Result. That fold wrote NaN on streams limited to
  part of the CUs (§ Code, entry 2); a fold that waits is safe when each wait is bounded and a split that runs out
  hands its share on instead of writing (`../gemm/wait_free_fold.md` § Counter-signals).
- **No line times these kernels' split against no split.** Whether a one-launch split pays is a separate question
  from whether its fold is right: time it against split-K 1 and against a separate fixed-order reduce at each M
  (`../gemm/wait_free_fold.md` § When it applies). None of the evidence below has either arm at the sizes these
  kernels split, m 32–512; the book's lines that time a split against no split on this shape are at M 8, on other
  kernels (`../gemm/small_m.md` §2, `../patterns/low_cu_utilization.md`).
- **Zeroing the counters is one more kernel in every call from m 32.** With counters kept in state that outlives
  the call, a call read 3.1–4.7 µs less at m 32–512, the most at m 128, 44.06 against 48.73 µs
  *(field, E2)*; the other sizes are in § Result. Counters kept across calls carry a risk
  that per-call ones do not, and a launch tag in the counter word replaces the zeroing
  (`../gemm/wait_free_fold.md` § How to write it, step 5, and § Cost and risk); these kernels zero.
- **m 128 is not on the fastest row measured.** With only the split and the tile changed, split 4 on 128×96 tiles,
  still 256 workgroups, read 43.20–43.32 µs against 47.47–47.61 for the kernel's split 8 on 128×192 tiles; split 4
  on 64×192 tiles read 1.450× the kernel's time and on 128 workgroups 1.242×. Unscaled fp16 partials read 0.973× of
  it, usable only while the sums fit fp16, and bf16-rounded partials left 9 elements past the accuracy bound
  *(field, E3)*.
- **The ordering test does not see every part of the hand-off.** With the partial stores' `sc0 sc1` removed every
  launch differed, but without the `vmcnt(0)` before the counter, or without the cache bits of the fold's loads, no
  launch did *(field, E4)*: keep what the ISA requires even when the test passes
  (`../gemm/correctness.md` §3). The fold is right only as compiled; after a FlyDSL or LLVM upgrade, read the ISA
  again and run the stress test again (`../gemm/wait_free_fold.md` § Cost and risk).
- **At m 512 the address of B moved the time.** Partway through a session the unchanged kernel moved from about 93.0
  to 102.41 µs. Swapping the tensors one at a time put the move on B: one build read 101.85 µs with the bench
  harness's B and its own A, and 91.17 with its own B and the bench harness's A, in single readings. Where B was
  allocated also reversed the ranking of two K-chunk settings that were within 2% of each other
  *(field, E5)*; rank such settings over several allocation offsets (`../gemm/measurement.md`
  §6).
- **Sizes with no kernel.** Of the second kernel's sizes, m 1024 and 2048 run a fixed hipBLASLt solution where the
  installed hipBLASLt offers it for the shape and `torch.matmul` otherwise, and m 4096 runs `torch.matmul`.
- **The m 256–512 kernel is described here from its source, and its code is not shown.** Its fold is the same
  protocol in another form: the count is per wave and the arrival is a compare-and-swap (§ Mechanism).

## Mechanism

**The kernel for m 1–128.** Every size launches the GEMM once, on 256 workgroups. In both builders A and B go into
LDS by `buffer_load_dwordx4 … lds` in XOR-swizzled 16 B chunks, B with `sc0 nt` and A with `sc0`, through one ring
that every wave of the workgroup reads, with an `s_waitcnt vmcnt(N)` and an `s_barrier` per K stage, in one
inline-asm string, after which the DMAs issued refill the slot that the waves read one stage earlier
(`../gemm/datapath.md` §2, §4, §5). The MFMA is `mfma_f32_16x16x32_bf16` with A first. The K loop is a rolled loop,
an `scf.for` with `sched_vmem`, `sched_dsrd` and `sched_mfma` hints (`../gemm/schedule_and_locality.md` §3): as the
source is written, a trip waits at that barrier, issues the DMAs, then reads its stage's fragments from LDS and runs
the MFMAs, and the next trip begins with the next barrier. The compile asks for kernel arguments to be preloaded
into SGPRs: `amdgpu-kernarg-preload-count` 8 is a count of arguments (`../flydsl/gemm.md` § small_m §6), so it asks
for every argument, the column-strip builder's four pointers and the split-K builder's five, where the source's
comment reads it as eight dwords, the first four pointers. How far a preload got is read from the ISA
(`../gemm/small_m.md` §6), and the book has no such reading for these kernels.

- **m ≤ 16, column strips**: the builder that [bf16_small_m_column_strip.md](bf16_small_m_column_strip.md) extends
  for N 6144 × K 4096. Four waves per workgroup, each workgroup owning 24 output columns over all of K. K goes in
  blocks of 128 through a 12-slot ring, and each wave takes a quarter of every block. The MFMA runs over 32 columns:
  B rows 24–31 read zero through an offset past the descriptor, and their columns are not stored. After the loop the
  waves write their fp32 partials to a padded LDS scratch, and the threads add them over the four waves in order and
  round once. Strip t starts its K walk at block t mod 128. No workspace and no counter.
- **m 32, 64 and 128, MFMA tiles with K split across workgroups**, 64 K per stage:

  | m | Tile, rows × columns | Waves, rows × columns | 16×16 fragments per wave | K split (K per split) | Ring slots | Tiles × splits | K-start step |
  |---|---|---|---|---|---|---|---|
  | 17–32 | 32 × 96 | 2 × 2 | 3 | 4 (4096) | 8 | 64 × 4 | 1 |
  | 33–64 | 64 × 96 | 4 × 2 | 3 | 4 (4096) | 6 | 64 × 4 | 1 |
  | 65–128 | 128 × 192 | 4 × 2 | 12 | 8 (2048) | 4 | 32 × 8 | 3 |

  - **Placement.** Workgroup i runs on XCD i mod 8. Each K split owns two XCDs at split 4 and one at split 8, its
    tiles spread over them, "so each XCD's L2 holds one K slice of A" (the code's comment). `../gemm/small_m.md` §6
    has the rule: where an A that every block of an XCD re-reads is about the size of one XCD's L2 (4 MiB), make
    the K start depend only on the XCD, or give each XCD one K split. Tile t starts its K walk at stage (t · step)
    mod the stages of a split, "so concurrent tiles spread their B reads over the HBM channels" (the code's
    comment).
  - **Partials.** After the loop a wave packs each fragment: per lane, four fp32 rounded half away from zero to
    their top 24 bits, in three dwords, with `v_perm_b32`. A plane is 12 bytes per lane and fragment
    (`../gemm/small_m.md` §4).
  - **Fold**: steps 1–4 of `../gemm/wait_free_fold.md` § How to write it (§ Code). The planes, one per split, sit
    in a `torch.empty` workspace that the call allocates and are written before they are read. The counters, one
    64-bit word per tile on a 128 B line of its own, sit in an 8192 B `torch.zeros` buffer that the call allocates,
    the first of step 5's four ways; each call owns its counters and partials, so overlapping calls share no
    synchronization memory. The split that folds also swaps its tile's counter back to 0, and no later call relies
    on that.

**The kernel for m 256 and 512** (read from its source):

- 256×192 tiles, eight waves (4 × 2) of 24 fragments each, 256 workgroups: 32 tiles × 8 splits at m 256, 64 tiles
  × 4 splits at m 512. A goes through a 2-slot LDS ring and B through a 4-slot one, A issued first and B with
  `sc0 nt`. At m 512, on a build with 8 × 1 waves that each owned their A rows, A loaded in the MFMA lane layout or
  permuted in registers read 106.0–127.5 µs against 96.6–97.8 for A through LDS, in single readings
  *(field, E6)*.
- Split s takes K chunks s, s + split count, …: one chunk of 2048 at m 256, four of 1024 at m 512. XCD x runs the
  splits of K group x mod 2, half of K, for a quarter of the tiles, "so its L2 fetches only that share of A" (the
  code's comment). With the fold removed, two K groups read 50.26 µs against 52.08 for one group at m 256
  *(field, E7)*. At m 512 the second half of the waves runs its K steps half a step
  behind the first. As in the m 1–128 kernel, a stage begins with `s_waitcnt vmcnt(N)` and `s_barrier`, and the DMAs
  issued after it refill the A and B slots read one stage earlier. The waves that run behind read the second 32-K
  half of a stage from LDS before the next stage's barrier and run its MFMAs after it; their wait is
  `vmcnt(N) lgkmcnt(0)`, so that "the wave's carried LDS reads land before any wave refills their slot" (the code's
  comment; `../gemm/datapath.md` §4).
- **Partials**: per lane, four values as 14-bit integers on one shared power-of-two scale, in two dwords, 8 bytes
  per lane and fragment. Timed on an earlier fold in which every split summed a share of the tile, in single
  readings, that form read 91.44 against 92.97 µs at m 512 and 59.15 against 59.43 at m 256 beside four values in
  three dwords, with a peak error of 0.659 and 0.679 of the accuracy bound *(field, E8)*. The
  pack takes the largest magnitude with plain ops, not inline asm: an inline-asm `v_max3_f32` that read the
  accumulators right after the main loop got no wait states, and about 1.5% of the outputs came out wrong, seen
  once *(field, E9)*.
- **Fold.** Each wave counts in for itself, on a counter per (tile, wave) with a 128 B line of its own, in a 64 KiB
  `torch.zeros` buffer that the call allocates. After `s_waitcnt vmcnt(0)`, lane 0 runs a compare-and-swap loop on
  the counter through a buffer atomic, the other lanes addressing past the buffer; a failed swap means that another
  split's succeeded, so the retries are bounded by the split count. The wave that finds every other split counted
  swaps the counter back to 0, loads every split's plane with `sc0 sc1`, four fragments at a time, adds them in
  split order and rounds once. No barrier is in this hand-off.

## Code

### 1. Store the partial, wait, count in (m 32–128)

After the K loop of the split-K path of the m 1–128 kernel, each split stores its packed partial into its own plane
with `sc0 sc1` (`_SC01`, aux 17), which bypasses the non-coherent caches so the data meets across XCDs; waits
`vmcnt(0)` and a barrier, so that every store is acknowledged before the split counts in; then thread 0 adds 1 to
the tile's counter with an agent-scope atomic add that returns the old count, and passes the old count to its
workgroup through LDS. These are steps 1–3 of [gemm/wait_free_fold](../gemm/wait_free_fold.md) § How to write it,
with counters zeroed in the call. `bid` is the tile, `s_idx` the split and `c_frags` the wave's `NF` accumulators;
`own_off(plane, f)` is the offset of this wave's fragment `f` in a split's plane, `S` the counter buffer, with one
64-bit counter per tile, a cache line apart, and `a_lds_i64` the base of the operand ring, free once the loop is
over. `w_rsrc` is the buffer descriptor of the planes' workspace, `_pack24` the 24-bit packing of § Mechanism
(`_unpack24`, in entry 2, its inverse), and `_bstore` and `_bload` raw buffer stores and loads whose last argument
is the cache bits.

```python
counter = _global_ptr(S, bid * (_COUNTER_LINE * 4))  # one 64-bit counter per tile, a cache line apart
packed = [_pack24(fx.arith.unwrap(c_frags[f])) for f in range_constexpr(NF)]
for f in range_constexpr(NF):
    _bstore(packed[f], w_rsrc, own_off(s_idx, f), _SC01)  # step 1: own plane, _SC01 = 17 (sc0 | sc1)
# the partials bypass the non-coherent caches: once stored they are visible on every XCD
rocdl.s_waitcnt(vmcnt=0)  # step 2: every store acknowledged before the split counts in
gpu.barrier()
arrived_lds = _lds_ptr(a_lds_i64)
if tid == fx.Int32(0):  # step 3: count in; the old count goes to the workgroup through LDS
    before = llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        counter,
        fx.arith.constant(1, type=T.i64),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    )
    llvm.StoreOp(std_arith.TruncIOp(T.i32, before).result, arrived_lds)
gpu.barrier()
arrived = rocdl.readfirstlane(T.i32, llvm.LoadOp(T.i32, arrived_lds).result)
```

Not this: the partial stored with no cache bits. In an ordering test of this shape's fold, with two input sets
alternating between calls, full, 64-CU and 16-CU streams and a second launch alongside, every launch then differed,
against 0 of 340 with `sc0 sc1` *(field, E4)*.

Explained in [gemm/wait_free_fold](../gemm/wait_free_fold.md) § How to write it.

### 2. The last split to count in folds, in split order (m 32–128)

The fold of the same kernel, step 4. The split that read `SPLIT - 1`, so that every other split has counted in,
folds; the others exit. It loads every other split's plane with `sc0 sc1`, its own load pointed at `_OOB`, an offset
past the buffer descriptor where the access is dropped, `_FOLD_BATCH` (3) fragments at a time before any is
unpacked, and adds the partials in split order. Its own partial comes from its registers (`packed`), after the same
packing as the published ones, so the sum is bitwise the same whichever split folds. `arrived` is the old count of
entry 1 and `raw[f, sp]` split `sp`'s packed fragment `f`; in the kernel, thread 0 of the folding split first swaps
the counter back to 0, and `cast` is then stored to `out`, neither of which is in the excerpt.

```python
if fx.Int32(arrived) == fx.Int32(SPLIT - 1):  # step 4: every other split has counted in, so this one folds
    for f0 in range_constexpr(0, NF, _FOLD_BATCH):
        fs = list(range_constexpr(f0, min(f0 + _FOLD_BATCH, NF)))
        raw = {}
        for f in fs:
            for sp in range_constexpr(SPLIT):  # the other splits' planes; its own load is sent out of bounds
                mine = s_idx == fx.Int32(sp)
                off = arith.select(
                    mine, fx.Int32(_OOB), fx.Int32(own_off(sp, f))
                )
                raw[f, sp] = _bload(T.vec(3, T.i32), w_rsrc, off, _SC01)
        for f in fs:
            tot = None
            for sp in range_constexpr(SPLIT):  # split order, own partial from registers: the same bits whoever folds
                mine = as_ir_value(s_idx == fx.Int32(sp))
                part = std_arith.SelectOp(mine, packed[f], raw[f, sp]).result
                v = Vector(_unpack24(part))
                tot = v if tot is None else tot + v
            cast = tot.to(fx.BFloat16)
```

Not this: an owner split that polls a flag or its partners' planes until they are published. On a stream limited
to part of the CUs the partners may not be running: on this shape such a fold wrote 2,340,864 to 4,276,224 NaN over
the three tries of a case at m 256 and 512 on streams limited to 64 and 128 CUs, where the fold that does not wait
gave the output of the call alone, bit for bit *(field, E10)*.

Explained in [gemm/wait_free_fold](../gemm/wait_free_fold.md) § How to write it.

## Result

The book gives no speedup of these kernels over a baseline. What it measured are kernel times on gfx950 with FlyDSL
0.3.2, on the per-kernel CUDA-graph timer through a tuning run's bench harness with the arms interleaved
(`../gemm/measurement.md` §1: a call's time is the sum of the device durations of its kernels, in replays of a CUDA
graph with L2 emptied before each call). Per call, in µs. The three columns under E1 and E11 are one run and
compare with each other only: medians of 6 interleaved rounds at m ≤ 128 and of 4 at m 256 and 512, with the same
flat-fold times in both lines at m 64, 128, 256 and 512 (33.19, 43.90, 68.6 and 94.3 µs) *(field, E1, E11)*. That
run's flat fold kept launch-tagged counters, so it ran no zeroing kernel. E2 is another measurement, N=2 with its
two versions interleaved; its first column is the version that zeroes its counters in the call, as the kernels do.

| m | Counters zeroed in the call, as the kernels do *(field, E2)* | Counters kept in state that outlives the call *(field, E2)* | The fold of this page *(field, E1)* | An owner-per-slice fold that waits *(field, E1)* | A fixed-order pairwise tree *(field, E11)* |
|---|---|---|---|---|---|
| 32 | 36.41 | 32.23 | 32.10 *(field, E11)*, within 0.5% of the fold that waits | no time given | 33.04 |
| 64 | 37.30 | 33.12 | 33.19 | 32.54 | 33.94 |
| 128 | 48.73 | 44.06 | 43.90 | 37.95 | 43.79 |
| 256 | 70.62 | 67.48 | 68.6 | 56.2 | 70.2 |
| 512 | 98.59 | 94.49 | 94.3 | 87.1 | 95.9 |

- **Zeroing**: 3.1–4.7 µs per call at m 32–512; m ≤ 16 and m ≥ 1024, which run no fold, were within 0.4 µs
  *(field, E2)*.
- **A pairwise tree in fixed order** was no faster than the flat fold at m 128 and 1.7–2.9% slower at m 32, 64, 256
  and 512: each level waits for its own stores to be acknowledged, and the splits of a tile finish close together,
  so the last split pays every level *(field, E11)*.
- **m 128 by split and tile**: the kernel's row, split 8 on 128×192 tiles, read 47.47–47.61 µs over 6 rounds with
  the arms rotated, and split 4 on 128×96 tiles 43.20–43.32 *(field, E3)* (§ Counter-signals).
- **The K-start stagger**, on a build of 2026-10: with each workgroup's K walk starting at stage (index · R) mod the
  stage count, the m 1–128 kernel read 32.98, 33.10, 31.89, 32.71, 33.09, 33.81, 38.28 and 43.91 µs at m 1, 2, 4, 8,
  16, 32, 64 and 128, against 34.93, 36.85, 34.81, 35.50, 36.28, 35.99, 39.77 and 48.71 with every walk starting at
  stage 0, on one GPU; a second GPU gave the same sign at every size, by 2.4–5.5 µs. The HBM read counter read
  202–244 MB per call with and without it *(field, E12)*. At m 128 (A 4 MiB), on a build that ran
  every split of a tile on one XCD, a start per tile read 61.72 µs against 43.91 for one start per XCD, and 34.59
  against 37.08 with the A loads removed; at m 64 (A 2 MiB) a start per tile read 38.93 against 42.01
  *(field, E13)*. The kernel as it is places the splits differently at m 128, one K split per XCD
  (§ Mechanism): read these two lines for the direction of the stagger, not for the times of its rows.
- **Checks.** On streams limited to 64 and 128 CUs the fold gave the output of the call alone, bit for bit
  *(field, E10)*. Two more checks are on the build that first folded this way, whose counters
  were launch-tagged where these kernels zero theirs in the call. A stress test on streams limited to 16, 32, 64
  and 128 CUs, with a second launch beside the kernel, passed under every mask *(field, E14)*.
  An ordering test, with two input sets alternating between calls on full, 64-CU and 16-CU streams and a second
  launch alongside, found 0 differing launches of 340 *(field, E4)*.

Timing these kernels through Kineto on a node shared with other profiling runs lost whole measurements to a tracer
fault, which hit a plain `torch.mul` about as often: `../pitfalls/profiler_misses_kernels.md`.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=4–6, per-kernel CUDA-graph timer through the run's bench harness, arms interleaved, medians of 6 rounds at m ≤ 128 and of 4 at m 256 and 512] owner-per-slice spin fold → wait-free flat fold, its counters launch-tagged: N 6144 × K 16384 m=128 37.95 → 43.90 µs, m=64 32.54 → 33.19 µs, m≤32 within 0.5%; m=256 56.2 → 68.6 µs, m=512 87.1 → 94.3 µs (another reading of the same two kernels: `../gemm/small_m.md` E107)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=2, per-kernel CUDA-graph timer through the run's bench harness, the two versions interleaved] counters zeroed in the call vs counters kept in state that outlives the call: N 6144 × K 16384 m=32 36.41 vs 32.23 µs, m=64 37.30 vs 33.12, m=128 48.73 vs 44.06, m=256 70.62 vs 67.48, m=512 98.59 vs 94.49; m ≤ 16 and m ≥ 1024 within 0.4 µs
- E3 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=6 rounds, per-kernel CUDA-graph timer, arms rotated] m=128 on the final N 6144 × K 16384 kernel (bf16), only the split, tile or partial format changed: its split 8 (128×192) 47.47–47.61 µs; split 4, 128×96, 256 workgroups 43.20–43.32; split 4 on 64×192 1.450×, on 128 workgroups 1.242×; unscaled fp16 partials 0.973×; bf16-rounded partials 9 elements past the accuracy bound
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=340 launches, ordering test: two input sets alternating, full, 64-CU and 16-CU streams, a second launch of the kernel alongside] the arrival fold: 0 differing launches of 340; with the partial stores' `sc0 sc1` removed, every launch differed; without the `vmcnt(0)` before the counter, or without the fold loads' cache bits, no launch differed (bf16 split-K GEMM n6144 k16384)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, per-kernel CUDA-graph timer through the bench harness, and an in-process probe on copies of the inputs at chosen offsets] bf16 GEMM, N 6144, K 16384, m=512, split-K 4: partway through a session the unchanged kernel moved from about 93.0 to 102.41 µs; swapping tensors one at a time, the K-chunk-16 build read 101.85 µs with the bench harness's B and its own A, 91.17 with its own B and the bench harness's A; in that state chunk 8 92.67 against chunk 16 101.88, over 9 copied offsets 90.24–90.85 against 90.74–91.22; a later sweep with 0 / 37 / 211 MB allocated before the inputs: chunk 32 91.44 / 91.50 / 89.96 against chunk 16 90.06 / 90.05 / 90.12 (bf16 GEMM n6144 k16384)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, in-process probe, L2 flushed before each call, randn inputs, every arm against a same-process control; arms equal to the A-through-LDS build except the diagnostics] bf16, m=512, 256×192 tile, 8×1 waves (32 A rows per wave): A through LDS 96.6–97.8 µs; A in the MFMA lane layout 113–118 (2 A stages), 106.0 (3 stages); quad-contiguous loads + `ds_bpermute` 107.7–110.0; permlane/DPP register transpose 124.8 (7 spills) and 127.5 (248 VGPRs, none); contiguous-load diagnostics, wrong output, 92.7–96.1 against controls of 95.3–96.7 (bf16 GEMM n6144 k16384)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=8–10 readings per arm, per-kernel CUDA-graph timer, arms interleaved in one process against the canonical kernel, median; outputs bitwise equal] with the fold removed (wrong output on purpose), two K groups (XCD x runs the splits of group x % 2) 50.26 µs against 52.08 for one group at m=256 (256×192 tiles, split 8, A 256 × 16384 bf16 = 8 MiB); with the fold on, every split summing a share of the tile after the others arrived and its flags in the XCD-local L2 only when a tile's splits shared an XCD (device scope otherwise): 8 groups (one split per XCD) 72.18 against 57.64 µs at m=256, 66.64 / 66.71 against 57.60 / 57.72 in the previous iteration (bf16 GEMM n6144 k16384)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, the run's bench harness on the per-kernel CUDA-graph timer and its standard inputs, and an in-process probe on randn inputs with an L2 flush before each call] split-K partials as four 14-bit integers on one shared power-of-two scale (2 dwords per 4 values) against 4 values per 3 dwords, on a fold in which every split summed a share of the tile: bench harness m=512 92.97 → 91.44 µs, m=256 59.43 → 59.15; probe 99.98 → 98.19 and 65.07 → 63.76; peak error 0.656 → 0.659 (m=512) and 0.646 → 0.679 (m=256) of the accuracy bound; on the harness's standard inputs the kept kernel read 0.649 / 0.680 against torch's 0.649 / 0.666 (bf16 GEMM n6144 k16384, 256×192 and 256×96 tiles, split 4 and 8)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1, single observation, the fix session's correctness check] a packing step reading MFMA results through inline-asm `v_max3_f32` / `v_max_f32_e64`, once the pack ran straight after the main loop: about 1.5% of elements wrong, off by exact powers of two; the same max written as `math_d.absf` + `arith.MaxNumFOp` compiled to `v_max3_f32` with the wait states and passed (bf16 split-K GEMM n6144 k16384)
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=3 tries a case in one run of the stress test, eager on a CU-masked stream, output against the call alone, NaN summed over the tries] spin fold at m=256: 3,151,872 NaN on 64 CUs, 2,340,864 on 128 CUs, 35.5 and 12.4 ms per call; m=512: 4,276,224 and 3,158,016 NaN; the wait-free fold: none, bit for bit equal
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=4–6, the run of E1, its third arm] fixed-order pairwise tree vs flat fold: m=128 43.79 vs 43.90 µs, m=64 33.94 vs 33.19, m=32 33.04 vs 32.10, m=256 70.2 vs 68.6, m=512 95.9 vs 94.3
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, per-kernel CUDA-graph timer (1 GiB flush per call), one process per set with a fresh FlyDSL cache, two GPUs; placement alone moved a case about ±1 µs between processes] per-block K start `(bid·R) mod loops` against none (B `sc0 nt`): GPU 3 m=1/2/4/8/16/32/64/128 32.98/33.10/31.89/32.71/33.09/33.81/38.28/43.91 against 34.93/36.85/34.81/35.50/36.28/35.99/39.77/48.71 µs; GPU 6 the same sign on every case, 2.4–5.5 µs; HBM read 202–244 MB per call with and without it, request stalls lower with it (bf16 GEMM n6144 k16384)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, as E12] m=128 (A 128 × 16384 bf16 = 4 MiB, every split of a tile on XCD `bid mod 8`): per-block starts R=1 61.72 µs (R=3 and 5 61.8–63.4 on GPU 6) against R=16, one start per XCD, 43.91; with the A loads removed R=1 34.59 against 37.08; m=64 (A 2 MiB) R=1 38.93 < R=3 39.54 < R=16 42.01 (GPU 6); with each XCD on one K split, R=1 41.96–42.00 against R=16 42.46 (bf16 GEMM n6144 k16384)
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=3 tries a case, the tuning run's stress test: CU-masked streams of 16/32/64/128 CUs and a second launch beside the kernel, NaN summed over the tries] splits that polled flags and planes up to a cap, then wrote NaN: up to 4.7 M NaN at m=256 and m=512, calls of up to 140 ms on 16 CUs; the arrival-fold rewrite passed every mask (bf16 split-K GEMM n6144 k16384)
