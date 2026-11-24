---
title: "A split-K fold in which no workgroup waits for another: the last split to arrive folds"
kind: technique
lever: wait_free_split_k_fold
ops: [gemm, grouped_gemm]
dtypes: [bf16]
regimes: ["M 1-512", "M 512-4096 (bf16, N 64-256)", split-k]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=4
sources:
  - "../cases/k3_bf16_splitk_narrow_n.md § N 1024 and 2048 × K 7168 at m 1–128: the same in-workgroup ring, and a fold that does not wait"
  - "../cases/k3_bf16_splitk_narrow_n.md § N 384 × K 7168 at m 7–256 (DeepSeek-V4-Pro's router): N 1024's ring on narrower tiles, a tagged fold at m 256"
  - "../cases/bf16_k4096_splitk_hgemm.md § N 64–256 at m 256–4096: N 512 × K 4096's K split over waves, ported"
---

# Wait-free split-K fold

## What it does

A one-launch split-K GEMM has to add the splits' partial tiles. A fold in which an owner split polls flags, or
launch-tagged data, until its partners have published is right only while every workgroup of the launch is
resident at once: when fewer CUs are free (a stream limited to part of the GPU, another kernel holding CUs), the
waiting workgroups hold the slots their partners need, and the fold gives up or hangs. Here no workgroup ever waits.
Each split publishes its partial, counts itself in on an atomic counter for its tile, and the split that finds all
the others already counted does the whole fold; the others exit. A second form has no counter: each split tags what
it publishes with the launch and reads its partners' tags back, and whichever split finds them all folds
(§ Folding without a counter: launch-tagged slots).

## How to write it

After its main loop, each split of a one-launch split-K kernel (the code is a bf16 GEMM's, shown under the steps):

1. Each split stores its packed partial into its own plane with `sc0 sc1` (aux 17), which bypasses
   the non-coherent caches so the data meets across XCDs.
2. `s_waitcnt vmcnt(0)` and a barrier: the stores are acknowledged before the split counts in.
3. Thread 0 adds 1 to the tile's counter with an agent-scope atomic add that returns the old count, and passes the
   old count to its workgroup through LDS.
4. The split that read `SPLIT - 1` folds: it loads every other split's plane with `sc0 sc1`, its own load pointed
   out of bounds and dropped, and adds the partials in split order. Its own partial
   comes from its registers after the same packing as the published ones, so the sum is
   bitwise the same whichever split folds.
5. Make the counters valid for this call, in one of four ways (the last three tag the counter word with the launch):
   - Zero them in the call with `torch.zeros` before launching the GEMM. The split that folds also resets its tile's
     counter with an `xchg`, but no later call relies on that reset. This costs a launch per call (Cost and risk).
   - Or tag the counter word with the launch, and allocate it with `torch.empty` beside the partials. On a bf16
     GEMM with N = K = 6144, thread 0 of each split runs a compare-and-swap loop (`llvm.cmpxchg`, agent scope) on a
     64-bit word holding `tag << 2 | count`: a word with another tag counts as 0, and the split whose swap saw count
     `SPLIT - 1` under its own tag folds. The tag is a 62-bit hash of the dispatch id and the queue address
     (`dispatch.id · H1 + queue.ptr · H2` with H1 odd, shifted left past the count bits); it must not
     repeat within a queue the way a 32-bit tag does (Cost and risk). Use monotonic ordering on the swap, since
     step 2 already acknowledged the data: *(field, E11)* with acq_rel the compiler put `buffer_wbl2 sc1` and
     `buffer_inv sc1` around every swap, and m=64 took 23.30 µs against 12.87 on another bf16 GEMM (N=2048,
     K=7168). *(field, E16)* Agent-scope fences around the arrival in place of the scoped stores and loads cost
     about as much at m 1024–4096: at N=256, K=6144, m=1024 24.7 against 14.0 µs (same build otherwise); at N=64,
     K=4096, m=4096 26.18 µs (KW 4, 2 stages) against 11.92 with `sc0 sc1` (KW 2, 3 stages; partials alone read
     11.35 and 9.74 in those two configurations).
   - Or reset a stale word, then add (bf16 GEMM N=64, K=4096, m=4096, split 4). Thread 0 reads its tile's 64-bit
     word before the first DMA. After the main loop, if the word's high 56 bits are not this launch's tag (the
     dispatch id plus the queue pointer times an odd constant), it compare-and-swaps the value it read to
     `tag << 8`; a split that already counted in has changed the word, so that swap fails. Once its partial stores
     are acknowledged it adds 1, and the split that gets back `tag << 8 | S − 1` folds. *(field, E12)* With
     the reset disabled the output held 262,144 NaN within 40 calls; with the scratch prefilled with random words
     before each of 40 calls alternating two inputs, every output was bitwise equal to the clean one.
   - Or, with two splits, exchange instead of count (bf16 GEMM N=256, K=6144, m=512). Once a wave's partial stores
     are acknowledged, its lane 0 swaps the launch tag into that wave's 8-byte word of the tile's 128 B slot; the
     wave that gets this launch's tag back arrives second, swaps a sentinel back for the next launch, and adds its
     partner's partial to its own, kept in registers. *(field, E13)* Its first version, one word per tile
     swapped by the workgroup and no sentinel, read 7.95–7.97 µs against 8.36–8.39 for the compare-and-swap chain it
     replaced at m=512; the sentinel (with the partner loads issued first) and the per-wave words, added in the next
     step, read 7.963 → 7.986 and 7.970 → 7.925 µs at m=512 (7.957 in the final bench). That kernel
     tagged with the dispatch id alone and passed the stress test (§ How to confirm it took effect), but a second
     stream's dispatch ids start
     again at 0 (E7): two queues can then write the same tag, and a small-integer tag can equal a leftover word. Add
     queue bits through the hash above (Cost and risk).

   The planes sit in the call's own `torch.empty` workspace, written before they are read.

The first four steps on a bf16 GEMM with counters zeroed in the call (`bid` is the tile, `s_idx` the split,
`c_frags` the accumulators; an access at `_OOB`, an offset past the buffer descriptor, is dropped):

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

The fold, in the same kernel:

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
to part of the CUs the partners may not be running, and such a fold wrote NaN *(field, E3)*.

In FlyDSL: `../flydsl/gemm.md` § wait_free_fold How to write it.

## Folding without a counter: launch-tagged slots

Each split publishes its partial together with a tag of this launch, waits until its own stores are acknowledged,
then reads every split's tag. A split that finds all of them adds the partials in split order and writes the output.
No counter, atomic or zeroing launch is needed; the scratch is a `torch.empty` per call. The fold is
right under three conditions, each a risk to keep in view:
- The split whose stores are acknowledged last must see every other split's tag, which rests on each split reading
  only after its own `vmcnt(0)`. The stress and repeat runs (E8, E10) wrote every tile, but none forced the worst
  interleaving.
- A matching tag must mean this call's partial: a word left in the scratch that equals this launch's tag passes
  the check (Cost and risk).
- With the tag beside each element, a 16 B store must not be seen half written (new tag, old partial); that was not
  tested. The per-wave layout stores its tag only after its planes are acknowledged, so it does not rely on this.

The two layouts:

- **The tag beside every element** (bf16 GEMM N=128, K=6144, m=64 and 128, split 4). Each thread stores one 16 B
  slot per element, the fp32 partial and a 96-bit tag (the dispatch id's two words and the queue pointer's low word),
  with `sc0 sc1`; waits `vmcnt(0)`; loads that element's slot from every split; and writes C if all the tags match.
  One store and one load round per element, with no LDS or barrier in the decision. The N=384 kernel at
  m=256 packs two fp32 partials and a 64-bit tag, `dispatch.id` · H1 + `queue.ptr` · H2, into each 16 B slot, after
  its waves have added their partials through LDS.
- **One tag per (tile, wave) slot** (bf16 GEMM N=2048, K=7168, m 16–128, split 4). Each wave stores its planes with
  `sc0 sc1`, waits `vmcnt(0)`, stores the 64-bit tag `dispatch.id ^ rotl(queue.ptr, 32)` into its own slot, waits
  again, loads the slots of its wave index in every split, makes the result uniform with `readfirstlane`, and folds
  if all match. A wave reads its partners' planes only when it folds.

The per-wave layout in that kernel, from the store of a wave's summed partials on (`CPOL_SYS` is `sc0 sc1`, `S` the
split count, `fold` the function that adds the planes in split order and writes C):

```python
    buffer_store(sm, ws_rsrc, plane_off(s_idx, frags[i]), cpol=CPOL_SYS)
fx.rocdl.s_waitcnt(vmcnt=0)  # the planes are acknowledged before the tag is published

did = llvm.call_intrinsic(T.i64, "llvm.amdgcn.dispatch.id", [], [], [])
qp = llvm.call_intrinsic(ir.Type.parse("!llvm.ptr<4>"), "llvm.amdgcn.queue.ptr", [], [], [])
qpi = llvm.PtrToIntOp(T.i64, qp).result
qrot = std_arith.OrIOp(
    std_arith.ShLIOp(qpi, _i64c(32)).result, std_arith.ShRUIOp(qpi, _i64c(32)).result
).result
tag = std_arith.XOrIOp(did, qrot).result  # dispatch.id ^ rotl(queue.ptr, 32)
tag_lo = std_arith.TruncIOp(T.i32, tag).result
tag_hi = std_arith.TruncIOp(T.i32, std_arith.ShRUIOp(tag, _i64c(32)).result).result
grp_i32 = fx.Index(SLOT_BASE // 4) + (fx.Index(tile) * W + fx.Index(wid)) * SLOT_WORDS
tagv = _vector.FromElementsOp(T.vec(2, T.i32), [tag_lo, tag_hi]).result
buffer_store(tagv, ws_rsrc, grp_i32 + fx.Index(s_idx) * 2, cpol=CPOL_SYS)  # this split's slot of (tile, wave)
fx.rocdl.s_waitcnt(vmcnt=0)  # read the partners' tags only after the own tag is acknowledged
VW = min(4, 2 * S)
vals = []
for c in range_constexpr((2 * S) // VW):
    v = buffer_load(ws_rsrc, grp_i32 + c * VW, VW, T.i32, cpol=CPOL_SYS)
    for e in range_constexpr(VW):
        vals.append(v_extract(v, e))
ok = None
for sp in range_constexpr(S):  # every split's tag must be this launch's
    e_lo = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, vals[2 * sp], tag_lo).result
    e_hi = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, vals[2 * sp + 1], tag_hi).result
    e = std_arith.AndIOp(e_lo, e_hi).result
    ok = e if ok is None else std_arith.AndIOp(ok, e).result
ok_i = rocdl.readfirstlane(T.i32, std_arith.ExtUIOp(T.i32, ok).result)
guarded(std_arith.CmpIOp(std_arith.CmpIPredicate.ne, ok_i, _i32c(0)).result, fold)
```

What it costs:

- **Keep the `vmcnt(0)` between the publication and the reads:** without it no split is guaranteed to see every
  tag. *(field, E9)* On the per-element form no arm timed its removal alone: one that dropped it and also wrote C
  from every split read 6.044 µs against 4.740 at m=64, and reading first, then again behind the wait on a miss, was
  37–41% slower than the same run's canonical build at m=64 and m=128.
- **At m 64–128, most of what the fold costs is the peer reads.** *(field, E9)* On the per-element form at
  m=64 (8×16 tiles), the full fold read 4.740 µs against 4.397 with the store and its wait but no peer reads, and
  4.419 without the scratch store; at m=128 (16×16 tiles, split-major placement), 5.708 against 4.708 and 4.487. In
  the tuning run's final kernel (compact tile block per XCD, 384 threads) the store, wait and peer reads together cost about
  0.04 µs at m=64 and 0.62 µs at m=128.
- *(field, E10)* The per-wave slots beat the same kernel's launch-tagged counter (compare-and-swap) by 1.3–9.5% at
  m=32, 64 and 128 (N=2048, K=7168).
- **At m 1024–4096 the arrival costs as much as the reads** (counter forms, timing-only arms). *(field, E14)* At
  N=64, K=4096, m=4096 (split 4): partials alone 8.05 µs, with `sc0 sc1` stores, `vmcnt(0)` and barrier 8.18, with
  the counter round trip 9.38, with the fold 10.29. At N=128, K=6144, m=2048 (split 4, compare-and-swap arrival):
  10.58 → 12.76 with the arrival → 15.44 with the fold, against 13.06–13.08 unsplit. At N=256, K=6144, m=1024 a
  fixed folder with no arrival read 11.27 against 12.55–12.92 for the full fold.
- **Spacing tiles' counters apart helped one kernel's reset-then-add counter, not a plain add; one tile's
  per-wave words may share a line.** *(field, E13)* At N=64, m=4096,
  tiles' counters 8 B apart read 10.42 µs, and 64, 128 and 256 B apart 10.08, 10.10 and 10.05 (single runs; +4.3% at
  that m in the ablation of the 256 B change), so two counters in one 128 B line did as well as a line each;
  *(field, E20)* in a minimal HIP fold whose splits each add 1 to a 64-bit counter, counters 8 B apart were within 1%
  of 256 B apart. At N=256
  per-wave words 8 B apart in one 128 B slot per tile read 7.94 / 10.89 µs (m=512 / 1024) against 8.00 / 10.93 with
  a line per wave.

Where one launch with this fold paid, against the same kernel's path without a split:

- *(field, E8)* N=128, K=6144: m=64 4.732 µs against 4.940; m=128 5.299–5.314 against 5.556–5.585, but only with
  a compact block of tiles and all their splits on each XCD: split-major and
  one-run-per-XCD placements read 5.776 and 5.923 there.
- *(field, E10)* N=2048, K=7168: m=16 7.80 µs against 8.49–8.64 (the run's bench harness 8.63 → 7.91); it lost at m=8 (7.71 against
  7.00); at m=1 it read 7.65 against the whole-K kernel's 6.65, but the latter was a bench-harness reading, not a reading of
  the same probe.
- *(field, E17)* N=384, K=7168, m=256 (split 4, 32×64 tiles, each XCD a block of tiles with all their splits):
  8.48–8.50 µs against 9.44–9.47 for the kernel's whole-K path. The XCD grouping set the size of the gain: with 1 × 6
  tiles per XCD the split read 11.17 µs. At m=1 and 8 the per-wave form beat the N=2048 kernel's whole-K builder,
  which has 48 workgroups at N=384 (`../cases/k3_bf16_splitk_narrow_n.md` § N 384, its E23).
- *(field, E15)* At M 512–4096 on narrow N, at a full round of 256 workgroups, where the split bought a wider tile
  (`grid_pricing.md` §2): N=256, K=6144, m=512 (split 2: +4.8% at that m with a compare-and-swap arrival, and
  +5.6% more with the exchange arrival) and m=1024 (split 2, then 4: +2.8% and +2.2% at that m).
  These N256 changes came with new rotations/PRO and own-partial retention; the numbers are
  not isolated exchange or split effects. At N=64,
  K=4096, m=4096 (reset counter, split 4, timed with the changes made with it: +1.9% at that size but +0.22% [−0.77, +1.21]
  over five sizes, the interval crossing zero, and the ablation could not separate the fold's own cost; the
  untouched m=256 also moved by up to 2.7%); spacing the counters 256 B apart then added +4.3% at m=4096 to the
  kernel it was added to. It lost at N=128, K=6144, m=2048, N=64 at m=1024 and N=256 at m=256; at N=64, m=2048 it
  stayed within the machine's drift, neither a clear gain nor a clear loss (`grid_pricing.md` §2). A
  separate reduce launch lost at N=64, m=2048: GEMM 7.64 µs, reduce kernel 4.32 and a memset 4.04 (11.42 in the
  run's bench harness, against 8.69 for the unsplit kernel).

What was checked, and what was not, is under Cost and risk.

## When it applies

- Any kernel whose workgroups combine results through global memory: split-K GEMM folds, split-KV attention
  merges, cross-workgroup reductions. This is what correctness requires there; whether a one-launch split pays is a
  separate question, measured on bf16 GEMMs only: on N 6144 × K 16384 (30–95 µs kernels) with zeroed counters,
  which cost a per-call zero kernel *(field, E4)* and a one-workgroup fold *(field, E1)*; on N = K = 6144 with a
  launch-tagged counter, where it beat a separate fold launch at m=64 and lost at m=128 *(field, E7)*; and on
  4–14 µs kernels with launch-tagged slots, where it paid at some M and lost at others (§ Folding without a counter:
  launch-tagged slots). Time it against split-K 1 and against a separate fixed-order reduce at each M you are
  scored on.
- The result stays bitwise reproducible if the fold adds in a fixed order (split index), never in arrival order.

## Counter-signals

- A fold that waits is faster where its work spreads over all splits: the owner-per-slice spin fold beat this one
  by 15.6% at m=128 (8 splits, one workgroup now reads 7 planes) and by 22% at m=256 on a larger tile
  *(field, E1)*. A spin fold returned millions of NaN on streams limited to 64 and 128 CUs (E3). A fold that
  waits is safe when each wait is bounded and a split that runs out hands its share on instead of writing
  (`../attention/split_kv.md` §2, Bounded wait), with launch-tagged arrival words in a per-call workspace: a bf16 GEMM with
  N 3648 × K 7168 (two fold variants, one at m 40 and 48, the other at m 80–240) sums each split's share once the
  arrival mask is full, waits at most 2000 `s_memrealtime` ticks (20 µs), then hands its share by compare-and-swap
  to the split that completes the mask, which merges its own and the abandoned shares and zeroes the word; a word an
  earlier launch left is claimed by compare-and-swap, and the words, one 64-bit word per (tile, wave), are
  allocated with `torch.empty` per call. *(field, E21)* It
  gave no wrong, non-finite or nondeterministic output beside a matmul process on the same GPU, on two streams,
  with graph replays beside eager calls and with two graphs replayed at once; the records do not say whether any
  wait ran out. The kernel reads 1.146–1.870 over aiter's tuned bf16 GEMM at those sizes *(field, E22)*; no ablation
  separates the fold's share of that. Without a wait, fewer splits win back part of the speed, a cheaper partial
  format less.
  *(field, E18)* At m=128, split 4 on 128×96 tiles (still 256 workgroups) read 43.3 µs against 47.6 for the final kernel's
  split 8, but 24–45% slower on 64×192 tiles or half the workgroups; fp16 partials (only while the sums fit fp16)
  read 2.7% less, and bf16-rounded ones broke the accuracy bound.
- A pairwise tree in fixed order does not help: each level waits for its own stores to be acknowledged, and the
  splits of a tile finish close together, so the last split pays every level. It was no faster than the flat
  fold at m=128 and 1.7-2.9% slower at m=32, 64, 256 and 512 *(field, E2)*.
- A separate fold launch can still win: on a bf16 GEMM with N = K = 6144 at m=128 (split 4, 96 columns per
  workgroup), the launch-tagged counter fold read 24.66 µs against 23.87 for the fold launch, and 24.05 against
  23.82 with its loads batched; in the first version, removing the fold's loads brought it to 21.03 µs and removing
  the swap as well to 20.05 *(field, E7)*.

## How to confirm it took effect

- The final ISA has no loop around the flag or plane loads, and no `s_sleep`. A compare-and-swap loop that retries
  only because the value it expected was out of date (a stale word on the first try, or another split's swap in
  between) is not a wait: it never depends on a split that has not run. The per-element tagged slots show
  `buffer_store_dwordx4 sc0 sc1`, then `s_waitcnt vmcnt(0)`, then one `buffer_load_dwordx4 sc0 sc1` per split
  (E8).
- On streams limited to 16 and 64 CUs (`hipExtStreamCreateWithCUMask`), and beside a second call of its own on
  another stream, every workload's output is bit for bit its output alone. That is the stress test this page
  refers to. A fold that waits failed it with millions of NaN at m=256 and 512 *(field, E3)*.

## Cost and risk

- The counter fold's whole read goes through one workgroup: the cost grows with the split count (E1).
- Each call owns its counters and partials, so overlapping calls do not share synchronization memory.
- **A workspace cached per (device, stream) is shared by every CUDA graph captured on that stream.** In Triton,
  a bf16 GEMM with N 32 × K 6144 keeps its partials and counters in a `functools.cache` keyed on the device and
  the current stream, zeroes the counters once and relies on the folding split's reset to 0 (code below). Under capture
  the key is the capture stream, and torch's default capture stream is one per process, so graphs replayed at once,
  or a replay beside an eager call on the capture stream, share one set of partials and counters. *(field, E23)* At
  m 6–96, eager calls on the capture stream beside replays on another stream gave 30–33 wrong outputs of 400, and
  two graphs replayed on two streams 164–214 of 400 (relative error up to 2.2); eager calls on one or two streams,
  and beside a matmul process, none. Allocating and zeroing the workspace for each call made under capture fixed it,
  but the zeroing is captured and runs on every replay: *(field, E24)* 0.77–1.10 over aiter's tuned bf16 GEMM in graph
  replays, against 1.33–1.95 for the kernel with the cached workspace. Not tested: carving each capture's slice from a region zeroed
  once before capture, which needs no zeroing per replay because the folding split returns its counter to 0.

  ```python
  # Not this: one workspace per (device, stream), zeroed once, shared by every graph captured on the stream
  @functools.cache
  def _workspace(device, stream):
      p = torch.empty(_MAX_PART, device=device, dtype=torch.float32)
      cnt = torch.zeros(_MAX_TILES, device=device, dtype=torch.int32)
      return p, cnt

  p, cnt = _workspace(a.device, torch.cuda.current_stream(a.device).cuda_stream)
  ```

- Zeroing per-call counters adds one kernel to every split-K call: *(field, E5)* one `torch.zeros(64, int32)`
  before a bf16 GEMM (N=4096, K=2048, m=128) took the call from 8.013 to 11.909 µs. Against counters kept in
  state that outlives the call: *(field, E4)* 4.2 µs at m=32, 4.2 at m=64, 4.7 at m=128, 3.1 at
  m=256 and 4.1 at m=512.
- Counters kept across calls, zeroed once and returned to 0 by the split that folds, save that kernel and add a
  risk: a launch that aborts after some splits have counted in leaves a counter non-zero, and every later call on
  that tile starts from the wrong count. Per-call and launch-tagged counters do not carry it over.
- The fold is right only as compiled: the scoped stores, the `vmcnt(0)` before the count and the atomic's
  ordering. After a FlyDSL or LLVM upgrade, read the ISA again and re-run the stress test (§ How to confirm it took
  effect) before trusting the kernel.
- **A launch tag can replace the zeroing; a 32-bit tag is known to be unsafe, and a wider one is only as safe as no
  leftover word equalling this launch's tag** (next bullets). `dispatch.id` XOR `queue.ptr` is 64 bits; cut to 32 bits
  (as in half of a 64-bit counter word) it repeats (dispatch ids 1 and 2^32 + 1 on one queue). A stale word carrying
  this launch's tag harms each counter differently *(field, E19)*: where the count only adds (the reset-then-add
  counter), a word at `SPLIT` or more means no split sees `SPLIT - 1` and the tile is never written; where it
  wraps modulo `SPLIT` (the compare-and-swap counter), a completed word reads as no arrivals and the tile stays
  right; on both, a word left part-way can let a split fold before its partners have published, and the tile is
  written wrong.
- A dispatch-derived tag (the id itself or an odd-multiplier hash) is one to one only within its 62–64-bit domain
  on one queue, so combine it with queue-pointer bits; truncating the queue bits or an XOR does not make it unique
  across queues. Four bf16 kernels use such a tag in place of zeroing and passed that stress test: the
  launch-tagged counter (N = K = 6144, m=64; +0.66% [+0.52, +0.80] over a separate fold launch, m=64 +5.9%
  *(field, E6)*), both slot layouts above *(field, E8, E10)*, and the N=384 kernel, which also passed graph replays
  with new inputs and concurrent calls on different inputs (`../cases/k3_bf16_splitk_narrow_n.md`, its E24).
  The queue bits matter: a second stream's dispatch ids started again at 0 *(field, E7)*. Also checked: the
  dispatch id was the same in every workgroup of a launch and
  different in each launch, eager and in graph replays; a 200-round NaN and determinism check, and up to 1,500 calls
  per case with NaN-prefilled outputs, some on a second stream, plus 200 replays of a 20-call graph, gave no wrong
  output *(field, E8, E10)*. At M 2048–4096 two other tagged builds gave none either: the 56-bit reset-then-add
  kernel, with scratch prefilled with random words before each of 40 calls alternating two inputs (E12), and an
  N=128 compare-and-swap build at m=2048 that was not kept, with garbage or reused scratch and with inputs changed
  between graph replays (E14). Not covered by any test: dispatch-id wraparound, and a stale or garbage word equal to
  this launch's tag; on the per-wave form, a workspace prefilled with 0xFF and a reused workspace with alternating
  inputs (E10).
- Where the tags may live: a tag read before this call writes it can only live in scratch allocated in the call,
  and its correctness rests on no leftover word equalling this launch's tag. A workspace from `torch.empty` holds
  undefined bytes, and this protocol reads a word before the call has written it: where the caller requires scratch
  to be written before it is read, zero the counters instead.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=4, per-kernel CUDA-graph timer through the run's bench harness, arms interleaved] owner-per-slice spin fold → wait-free flat fold: N 6144 × K 16384 m=128 37.95 → 43.90 µs, m=64 32.54 → 33.19 µs, m≤32 within 0.5%; m=256 56.2 → 68.6 µs, m=512 87.1 → 94.3 µs
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=4, per-kernel CUDA-graph timer through the run's bench harness, arms interleaved] fixed-order pairwise tree vs flat fold: m=128 43.79 vs 43.90 µs, m=64 33.94 vs 33.19, m=32 33.04 vs 32.10, m=256 70.2 vs 68.6, m=512 95.9 vs 94.3
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, eager on a CU-masked stream, output against the call alone] spin fold at m=256: 3,151,872 NaN on 64 CUs, 2,340,864 on 128 CUs, 35.5 and 12.4 ms per call; m=512: 4,276,224 and 3,158,016 NaN; the wait-free fold: none, bit for bit equal
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=2, per-kernel CUDA-graph timer through the run's bench harness, the two versions interleaved] counters zeroed in the call vs counters kept in state that outlives the call: N 6144 × K 16384 m=32 36.41 vs 32.23 µs, m=64 37.30 vs 33.12, m=128 48.73 vs 44.06, m=256 70.62 vs 67.48, m=512 98.59 vs 94.49; m ≤ 16 and m ≥ 1024 within 0.4 µs
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=6–8, a probe on the per-kernel CUDA-graph timer, arms interleaved per case with the order alternating, median of 6–8 rounds] one `torch.zeros(64, int32)` launched before the GEMM: 11.909 µs against 8.013 without it (ratio 0.6728); `torch.empty` in its place 0.9993 (bf16 GEMM N=4096, K=2048, m=128, one column band of B per workgroup through LDS DMA rings)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=3, ablation ladder (remeasured), 3 pairs, per-kernel CUDA-graph timer, one-sided 95% bounds] split-K 4 at m=64 folded in the GEMM by the split that counts in last on a launch-tagged counter, against its parent's separate fixed-order fold launch: +0.66% [+0.52, +0.80], m=64 +5.87%, the other seven cases −0.24% to +0.13% (bf16 GEMM N=K=6144, m 1–128, 256 workgroups × 8 waves, 96 columns each)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=2, an interleaved A/B (timer not stated in the record), then the bench harness twice] same kernel, tagged-counter fold against the fold launch: m=64 18.30 vs 19.38 µs; m=128 24.66 / 24.65 vs 23.87 / 23.93, loads batched 24.05 vs 23.82; removal probes at m=128 (timing only): swap and sum removed 20.05, swap kept and sum loads removed 21.03; output bitwise equal to the fold-launch kernel across graph replays; the run kept the m=64 path only, `allclose: True`, the stress test passed; dispatch-id probe: two eager calls 4 and 5, a second stream 0 with a different queue address, graph replays 12, 16, 20
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=5–11, an interleaved A/B on the per-kernel CUDA-graph timer, medians of 5–11 rounds, canonical in the same run] per-element tagged slots, split 4: m=64 8×16 tiles 4.732 µs against the unsplit dot kernel's 4.940; m=128 16×16 tiles, compact block per XCD 5.299–5.314, split-major 5.776, one run per XCD 5.923, against the unsplit MFMA kernel's 5.556–5.585; ISA: `buffer_store_dwordx4 sc0 sc1`, `s_waitcnt vmcnt(0)`, four `buffer_load_dwordx4 sc0 sc1`; the stress test passed (m=64, then m=64 and m=128), a 200-round interleaved-m NaN/determinism check 0 bad outputs; a probe: dispatch id equal in every workgroup of a launch, eager 2, 5, 8, graph replays 53, 66, 79, high word 0, queue pointer low word different on two streams (bf16 GEMM N=128, K=6144)
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=5–9, an interleaved A/B on the per-kernel CUDA-graph timer, medians of 5–9 rounds, canonical in the same run; FOLD2/FOLD3 give wrong output by design] per-element tagged slots, split 4: full fold / no `vmcnt(0)` between store and reads / store and wait without peer reads / no scratch store: m=64 (8×16, 512 threads, one run per XCD; unsplit 4.958) 4.740 / 6.044 / 4.397 / 4.419 µs; m=128 (16×16, 512 threads, split-major; unsplit 5.585 / 5.580) 5.708 / — / 4.708 / 4.487; final kernel (compact block per XCD, 384 threads) full / FOLD2 / FOLD3: m=64 4.427 / 4.382 / 4.385, m=128 4.968 / 4.524 / 4.348; reads issued with no wait and a second pass behind `vmcnt(0)` on a miss: 6.073 vs 4.423 (m=64), 6.969 vs 4.954 (m=128) (bf16 GEMM N=128, K=6144)
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=2, per-kernel CUDA-graph timer with L2 flushed before every call, median of 5 readings; slot vs CAS 2 interleaved pairs, the config race one reading each; the m=1 whole-K and the final m=16 from the bench harness] per-(tile, wave) tag slots against the same kernel's launch-tagged CAS counter: m=32 9.33 / 9.16 vs 10.21 / 10.12 µs, m=64 12.12 / 12.10 vs 12.28 / 12.27, m=128 13.77 / 13.84 vs 14.23 / 14.30; one launch with split 4 against the whole-K kernel: m=16 7.80 vs 8.49 / 8.64, m=8 7.71 vs 7.00, m=1 7.65 vs 6.65 (the run's bench harness); the bench harness m=16 8.63 → 7.91; up to 1,500 calls per case with NaN-prefilled output, some concurrent on a second stream, plus 200 replays of a 20-call graph: 0 NaN, 0 mismatches, the stress test passed; "Checks not run: the 0xFF workspace prefill, alternating inputs on a reused workspace, and the ISA checks" (bf16 GEMM N=2048, K=7168)
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the run's bench harness (warmup 3, 20 iterations), one run per configuration] split-K 4 at m=64 arriving through a launch-tagged 64-bit compare-and-swap: acq_rel 23.30 µs with `buffer_wbl2 sc1` and `buffer_inv sc1` around every swap in the ISA, monotonic 12.87 with neither (bf16 GEMM N=2048, K=7168)
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=40 calls per check, eager calls checked bitwise against a clean run] reset-then-add counter, split 4 at m=4096: scratch prefilled with random words before each of 40 calls alternating two inputs, all bitwise equal to the clean result; a control build with the reset disabled gave mismatches with 262,144 NaN in iterations 37–39; ISA: no `buffer_wbl2`, `buffer_inv`, `s_sleep` or loop around the counter (bf16 GEMM N=64, K=4096)
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=5–10, interleaved in-process A/B on the per-kernel CUDA-graph timer, medians of 5–10 rounds (N=256); single bench-harness runs (N=64)] two-split exchange arrival (one `atomicrmw xchg`, the second arrival keeps its own partial in registers) against the compare-and-swap chain: incumbent 8.363 / 8.385 / 8.379 / 8.377 / 8.378, exchange 7.945 / 7.966 / 7.968 / 7.955 / 7.957 µs at m=512, `torch.equal`; next step, sentinel written back with the partner loads issued first 7.963 → 7.986 and per-wave words 7.970 → 7.925 at m=512, final bench 7.957, the stress test passed; per-wave 8 B words in one 128 B slot against one 128 B line per wave: 7.941 / 10.885 against 8.000 / 10.934 at m=512 / 1024 (bf16 GEMM N=256, K=6144); counter spacing between tiles at m=4096, HEAD 10.421 at 8 B: 64 B 10.082, 128 B 10.096 / 10.099, 256 B 10.045, later 512 / 1024 B 9.788 / 9.800; ablation of the 256 B change +1.46% [+1.31, +1.61], m=4096 +4.34% (bf16 GEMM N=64, K=4096)
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=1–3, single bench-harness runs (N=64), per-kernel CUDA-graph timer medians over 3 rounds alternated with canonical (N=128), probes on the per-kernel CUDA-graph timer (N=256); arms past the first are timing-only] N=64 m=4096 split 4: partials only 8.051, + `sc0 sc1` stores, `vmcnt(0)`, barrier 8.176, + counter round trip with the folder storing only its own partial 9.383, full fold 10.285 µs; N=128 K=6144 m=2048 split 4: planes stored, no counter 10.555, + store-acknowledge wait and barrier 10.577, + compare-and-swap arrival 12.761, + full fold 15.444, against canonical 13.060–13.071; correct with garbage and with reused scratch, and with inputs changing between graph replays; N=256 K=6144 m=1024 split 4: counter and fold removed 9.47, fixed last arriver without compare-and-swap 11.27, full-fold arms 12.55–12.92 against the unsplit incumbent's 11.55
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=3, ablation ladders, 3 pairs, the run's bench harness; the separate-launch arm by the run's bench harness and Kineto] N=64 K=4096 m=4096: whole fold-introducing step +0.22% [−0.77, +1.21] overall, +1.89% at m=4096, m=256 −0.66%; fold-deletion arm reported as not separable, no cost given; next step against a different parent, counter spacing m=4096 +4.34%, m=256 +2.72%; N=256 K=6144 (whole steps also changing rotation/PRO, exchange and own-partial retention; not isolated split effects): split m=512 with compare-and-swap arrival +1.59% [+1.31, +1.87] (m=512 +4.75%), exchange and split m=1024 +2.82% [+2.52, +3.12] (m=512 +5.62%, m=1024 +2.81%), per-wave arrival +1.16%, split 4 at m=1024 +0.86% (m=1024 +2.19%); N=64 m=2048 with fp32 partials and a reduce launch: the bench harness 11.416 µs against the run's head 8.689, Kineto GEMM 7.64, `reduce_kernel` 4.32, memset 4.04
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=1, single bench-harness run (N=64), a probe on the per-kernel CUDA-graph timer (N=256)] LLVM agent fences (release / seq_cst / acquire) around the arrival: N=64 K=4096 m=4096 (64×64, split 4) 26.177 µs (KW 4, 2 stages) against 11.922 for `sc0|sc1` buffer aux (KW 2, 3 stages); partials only, fold skipped, 11.349 (KW 4, 2 stages) and 9.743 (KW 2, 3 stages); N=256 K=6144 m=1024 (split 4, KW 2, BK 128 in both) 24.7 against 14.0 with `sc1` store/load bits
- E17 [2026-11, gfx950, FlyDSL 0.3.2, N=2–3 readings per arm, each a median of 5 readings of the per-kernel CUDA-graph timer with L2 flushed before every call, interleaved with the kept kernel] bf16 GEMM N=384, K=7168, m=256: the kept whole-K path 9.44–9.47 µs; split 4 across workgroups on 32×64 tiles, 2 ring slots, tiles and their splits grouped by XCD (2 M-tiles × 3 N-tiles per XCD): 8.48–8.50; on 3 ring slots, 2 × 3 / 1 × 6 tiles per XCD: 8.55 / 11.17
- E18 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=6 rounds, per-kernel CUDA-graph timer, arms rotated] m=128 on the final N 6144 × K 16384 kernel (bf16), only the split, tile or partial format changed: its split 8 (128×192) 47.47–47.61 µs; split 4, 128×96, 256 workgroups 43.20–43.32; split 4 on 64×192 1.450×, on 128 workgroups 1.242×; unscaled fp16 partials 0.973×; bf16-rounded partials 9 elements past the accuracy bound
- E19 [2026-11, gfx950 (MI355X), HIP 7.15, N=10 trials × 64 tiles per cell, one split held back 300 µs or none] HIP replicas of the compare-and-swap counter (the count wraps modulo `SPLIT`) and the reset-then-add counter (the count only adds), split 4, the counter word preset under this launch's tag: compare-and-swap, completed word 640/640 tiles right; reset-then-add, word at or past `SPLIT` 640/640 tiles unwritten; a word left part-way on either 640/640 wrong with a split held back, 0–64 of 640 without; other tags, zero or random words 640/640 right
- E20 [2026-11, gfx950 (MI355X), HIP 7.15 (no FlyDSL kernel), N=21 rounds, graphs of 20 calls, arms rotated, median of per-round ratios] minimal split-K fold, 4 splits per tile each adding 1 to a 64-bit counter with `global_atomic_add_x2`, 256 and 1024 workgroups: counters 8 B against 256 B apart 0.9935 and 0.9982; a tile's per-wave words in one 128 B slot against a line per wave 0.9906 and 0.9919 (probe kernel)
- E21 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=2000 calls per m beside the matmul process (1000 fresh inputs, each run twice); 400 outputs per m in each stream and graph arm (200 rounds of one call on each of two streams); each output against fp32 `a @ bᵀ` (wrong: non-finite or relative max error > 2e-2) and bit for bit against a serial call or its twin] the bounded-wait fold at m 40 and 48 (its first variant) and 80, 160 and 240 (its second): beside a second process running back-to-back bf16 8192 × 8192 matmuls on the same GPU; calls on one stream and on two streams; eager calls on the capture stream alternating with replays of a graph on another stream; two graphs, both captured on one stream, replayed on two streams: 0 wrong, 0 non-finite, 0 differing, in every arm (bf16 GEMM n3648 k7168)
- E22 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5), N=1, lower-bound speedup over aiter's tuned bf16 GEMM per size, one run on one MI355X] the bounded-wait fold at m 40, 48 (its first variant) 1.870, 1.819; at m 80, 96, 112, 120, 160, 200, 240 (its second) 1.619, 1.469, 1.269, 1.248, 1.169, 1.146, 1.158 (bf16 GEMM n3648 k7168)
- E23 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=400 outputs per m and arm (200 rounds of one call on each of two streams), 2000 calls per m beside the matmul process; each output against fp32 `a @ bᵀ` (wrong: non-finite or relative max error > 2e-2) and bit for bit against a serial call; counters zeroed before each arm] a split-K kernel with the cached workspace at m 6–96: eager calls on a capture stream alternating with replays, on another stream, of a graph captured on it: 30–33 wrong of 400; two graphs captured on that one stream, replayed on two streams: 164–214 wrong of 400, worst relative error 2.2; eager on one stream, eager on two streams, and beside a second process running bf16 8192 × 8192 matmuls on the same GPU: 0 wrong (bf16 GEMM n32 k6144, Triton)
- E24 [2026-11, gfx950 (MI355X), Triton of the sglang 0.5.22 image (no FlyDSL kernel), N=1 run per build, lower-bound speedup over aiter's tuned bf16 GEMM per size, one MI355X] the kernel of E23 at m 6–96 with the cached workspace 1.33–1.95 per size; a copy that allocates and zeroes the partials and counters for each call made under capture and keeps them for that graph's replays 0.77–1.10, with 0 wrong outputs in E23's graph arms (bf16 GEMM n32 k6144, Triton)
