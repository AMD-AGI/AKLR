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
  - aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json
  - aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json
  - aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/3af3d3de3d0f2285.json
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

In the kernel above (line numbers of `kernel.py`):

1. Each split stores its packed partial into its own plane with `sc0 sc1` (`_SC01 = 17`, line 49), which bypasses
   the non-coherent caches so the data meets across XCDs (S1).
2. `s_waitcnt vmcnt(0)` and a barrier: the stores are acknowledged before the split counts in (line 426, S2).
3. Thread 0 adds 1 to the tile's counter with an agent-scope atomic add that returns the old count, and passes the
   old count to its workgroup through LDS (line 430, S3).
4. The split that read `SPLIT - 1` folds: it loads every plane, its own included, with `sc0 sc1` and adds them in
   split order (line 440, S4). Its own partial goes through the same packing as the published ones, so the sum is
   bitwise the same whichever split folds.
5. Make the counters valid for this call, in one of four ways (the last three tag the counter word with the launch):
   - Zero them in the call with `torch.zeros` before launching the GEMM. The split that folds also resets its tile's
     counter with an `xchg` (S5), but no later call relies on that reset. This costs a launch per call (Cost and risk).
   - Or tag the counter word with the launch, and allocate it with `torch.empty` beside the partials (S8). On a bf16
     GEMM with N = K = 6144, thread 0 of each split runs a compare-and-swap loop on a 64-bit word holding
     `tag << 2 | count`: a word with another tag counts as 0, and the split whose swap saw count `SPLIT - 1` under
     its own tag folds (S6, S7). The tag is a 62-bit hash of the dispatch id and the queue address; it must not
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
     are acknowledged it adds 1, and the split that gets back `tag << 8 | S − 1` folds (S14). *(field, E12)* With
     the reset disabled the output held 262,144 NaN within 40 calls; with the scratch prefilled with random words
     before each of 40 calls alternating two inputs, every output was bitwise equal to the clean one.
   - Or, with two splits, exchange instead of count (bf16 GEMM N=256, K=6144, m=512). Once a wave's partial stores
     are acknowledged, its lane 0 swaps the launch tag into that wave's 8-byte word of the tile's 128 B slot; the
     wave that gets this launch's tag back arrives second, swaps a sentinel back for the next launch, and adds its
     partner's partial to its own, kept in registers (S15, S16). *(field, E13)* Its first version, one word per tile
     swapped by the workgroup and no sentinel, read 7.95–7.97 µs against 8.36–8.39 for the compare-and-swap chain it
     replaced at m=512; the sentinel (with the partner loads issued first) and the per-wave words, added in the next
     round, read 7.963 → 7.986 and 7.970 → 7.925 µs at m=512 (7.957 in the final bench). The campaign's kernel
     tagged with the dispatch id alone (S16) and passed `driver.py --stress`, but a second stream's dispatch ids start
     again at 0 (E7): two queues can then write the same tag, and a small-integer tag can equal a leftover word. Add
     queue bits through the hash above (Cost and risk).

   The planes sit in the call's own `torch.empty` workspace, written before they are read. AKLR no longer provides
   kernel state.

In FlyDSL: `../flydsl/gemm.md` § wait_free_fold How to write it.

## Folding without a counter: launch-tagged slots

Each split publishes its partial together with a tag of this launch, waits until its own stores are acknowledged,
then reads every split's tag. A split that finds all of them adds the partials in split order and writes the output.
No counter, atomic or zeroing launch is needed; the scratch is a `torch.empty` per call (S11, S13). The fold is
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
  with `sc0 sc1`; waits `vmcnt(0)`; loads that element's slot from every split; and writes C if all the tags match
  (S9, S10). One store and one load round per element, with no LDS or barrier in the decision. The N=384 file at
  m=256 packs two fp32 partials and a 64-bit tag, `dispatch.id` · H1 + `queue.ptr` · H2, into each 16 B slot, after
  its waves have added their partials through LDS (S17).
- **One tag per (tile, wave) slot** (bf16 GEMM N=2048, K=7168, m 16–128, split 4). Each wave stores its planes with
  `sc0 sc1`, waits `vmcnt(0)`, stores the 64-bit tag `dispatch.id ^ rotl(queue.ptr, 32)` into its own slot, waits
  again, loads the slots of its wave index in every split, makes the result uniform with `readfirstlane`, and folds
  if all match (S12). A wave reads its partners' planes only when it folds.

What it costs:

- **Keep the `vmcnt(0)` between the publication and the reads:** without it no split is guaranteed to see every
  tag. *(field, E9)* On the per-element form no arm timed its removal alone: one that dropped it and also wrote C
  from every split read 6.044 µs against 4.740 at m=64, and reading first, then again behind the wait on a miss, was
  37–41% slower than the same run's canonical build at m=64 and m=128.
- **At m 64–128, most of what the fold costs is the peer reads.** *(field, E9)* On the per-element form at
  m=64 (8×16 tiles), the full fold read 4.740 µs against 4.397 with the store and its wait but no peer reads, and
  4.419 without the scratch store; at m=128 (16×16 tiles, split-major placement), 5.708 against 4.708 and 4.487. In
  the campaign's final kernel (compact tile block per XCD, 384 threads) the store, wait and peer reads together cost about
  0.04 µs at m=64 and 0.62 µs at m=128.
- *(field, E10)* The per-wave slots beat the same kernel's launch-tagged counter (compare-and-swap) by 1.3–9.5% at
  m=32, 64 and 128 (N=2048, K=7168).
- **At m 1024–4096 the arrival costs as much as the reads** (counter forms, timing-only arms). *(field, E14)* At
  N=64, K=4096, m=4096 (split 4): partials alone 8.05 µs, with `sc0 sc1` stores, `vmcnt(0)` and barrier 8.18, with
  the counter round trip 9.38, with the fold 10.29. At N=128, K=6144, m=2048 (split 4, compare-and-swap arrival):
  10.58 → 12.76 with the arrival → 15.44 with the fold, against 13.06–13.08 unsplit. At N=256, K=6144, m=1024 a
  fixed folder with no arrival read 11.27 against 12.55–12.92 for the full fold.
- **Spread tiles' counters apart; one tile's per-wave words may share a line.** *(field, E13)* At N=64, m=4096,
  tiles' counters 8 B apart read 10.42 µs, and 64, 128 and 256 B apart 10.08, 10.10 and 10.05 (single runs; +4.3% at
  that m in the ablation of the 256 B round), so two counters in one 128 B line did as well as a line each; at N=256
  per-wave words 8 B apart in one 128 B slot per tile read 7.94 / 10.89 µs (m=512 / 1024) against 8.00 / 10.93 with
  a line per wave.

Where one launch with this fold paid, against the same kernel file's path without a split:

- *(field, E8)* N=128, K=6144: m=64 4.732 µs against 4.940; m=128 5.299–5.314 against 5.556–5.585, but only with
  a compact block of tiles and all their splits on each XCD: split-major and
  one-run-per-XCD placements read 5.776 and 5.923 there.
- *(field, E10)* N=2048, K=7168: m=16 7.80 µs against 8.49–8.64 (driver 8.63 → 7.91); it lost at m=8 (7.71 against
  7.00); at m=1 it read 7.65 against the whole-K kernel's 6.65, but the latter was a driver reading, not the same
  timer.
- *(field, E17)* N=384, K=7168, m=256 (split 4, 32×64 tiles, each XCD a block of tiles with all their splits):
  8.48–8.50 µs against 9.44–9.47 for the file's whole-K path. The XCD grouping set the size of the gain: with 1 × 6
  tiles per XCD the split read 11.17 µs. At m=1 and 8 the per-wave form beat the N=2048 file's whole-K builder,
  which has 48 workgroups at N=384 (`../cases/k3_bf16_splitk_narrow_n.md` § N 384, its E23).
- *(field, E15)* At M 512–4096 on narrow N, at a full round of 256 workgroups, where the split bought a wider tile
  (`grid_pricing.md` §2): N=256, K=6144, m=512 (split 2: +4.8% at that m with a compare-and-swap arrival, and
  +5.6% more with the exchange arrival) and m=1024 (split 2, then 4: +2.8% and +2.2% at that m).
  These N256 changes came with new rotations/PRO and own-partial retention; the numbers are
  not isolated exchange or split effects. At N=64,
  K=4096, m=4096 (reset counter, split 4, timed with the changes made with it: +1.9% at that Point but +0.22% [−0.77, +1.21]
  over five Points, the interval crossing zero, and the ablation could not separate the fold's own cost; the
  untouched m=256 also moved by up to 2.7%); spacing the counters 256 B apart then added +4.3% at m=4096 to the
  kernel it was added to. It lost at N=128, K=6144, m=2048, N=64 at m=1024 and N=256 at m=256; at N=64, m=2048 it
  stayed within the machine's drift, neither a clear gain nor a clear loss (`grid_pricing.md` §2). A
  separate reduce launch lost at N=64, m=2048: GEMM 7.64 µs, reduce kernel 4.32 and a memset 4.04 (11.42 in the
  driver, against 8.69 for the unsplit kernel).

What was checked, and what was not, is under Cost and risk.

## When it applies

- Any kernel whose workgroups combine results through global memory: split-K GEMM folds, split-KV attention
  merges, cross-workgroup reductions. This is what correctness requires there; whether a one-launch split pays is a
  separate question, measured on bf16 GEMMs only: on `gemm_n6144_k16384` (30–95 µs kernels) with zeroed counters,
  which cost a per-call zero kernel *(field, E4)* and a one-workgroup fold *(field, E1)*; on N = K = 6144 with a
  launch-tagged counter, where it beat a separate fold launch at m=64 and lost at m=128 *(field, E7)*; and on
  4–14 µs kernels with launch-tagged slots, where it paid at some M and lost at others (§ Folding without a counter:
  launch-tagged slots). Time it against split-K 1 and against a separate fixed-order reduce at each M you are
  scored on.
- The result stays bitwise reproducible if the fold adds in a fixed order (split index), never in arrival order.

## Counter-signals

- A fold that waits is faster where its work spreads over all splits: the owner-per-slice spin fold beat this one
  by 15.6% at m=128 (8 splits, one workgroup now reads 7 planes) and by 22% at m=256 on a larger tile
  *(field, E1)*. That speed is not available safely; take it from fewer splits or a cheaper partial format instead.
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
  another stream, every workload's output is bit for bit its output alone; forge_task's `driver.py --stress` runs
  this. A fold that waits failed it with millions of NaN at m=256 and 512 *(field, E3)*.

## Cost and risk

- The counter fold's whole read goes through one workgroup: the cost grows with the split count (E1).
- Each call owns its counters and partials, so overlapping calls do not share synchronization memory.
- Zeroing per-call counters adds one kernel to every split-K call: *(field, E5)* one `torch.zeros(64, int32)`
  before a bf16 GEMM (N=4096, K=2048, m=128) took the call from 8.013 to 11.909 µs. Against counters kept in
  AKLR's former kernel state (no longer offered): *(field, E4)* 4.2 µs at m=32, 4.2 at m=64, 4.7 at m=128, 3.1 at
  m=256 and 4.1 at m=512.
- **A launch tag can replace the zeroing; a 32-bit tag is known to be unsafe, and a wider one is only as safe as no
  leftover word equalling this launch's tag** (next bullets). `dispatch.id` XOR `queue.ptr` is 32 bits and repeats
  (dispatch ids 1 and 2^32 + 1 on one queue), and a stale word that reads as this launch's, already complete, means no
  split sees `SPLIT - 1` and the tile is never written.
- A dispatch-derived tag (the id itself or an odd-multiplier hash) is one to one only within its 62–64-bit domain
  on one queue, so combine it with queue-pointer bits; truncating the queue bits or an XOR does not make it unique
  across queues. Four bf16 kernels use such a tag in place of zeroing and passed `driver.py --stress`: the
  launch-tagged counter (N = K = 6144, m=64; +0.66% [+0.52, +0.80] over a separate fold launch, m=64 +5.9%
  *(field, E6)*), both slot layouts above *(field, E8, E10)*, and the N=384 file, which also passed graph replays
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
  and its correctness rests on no leftover word equalling this launch's tag. Check the scratch rules in
  `aklr/ops/README.md` § The kernel section and ABI and `../SKILL.md` § Stop before choosing this over zeroing.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=4, AKLR graph timer (driver --bench-mode), interleaved] owner-per-slice spin fold → wait-free flat fold: gemm_n6144_k16384 m=128 37.95 → 43.90 µs, m=64 32.54 → 33.19 µs, m≤32 within 0.5%; m=256 56.2 → 68.6 µs, m=512 87.1 → 94.3 µs
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=4, AKLR graph timer (driver --bench-mode), interleaved] fixed-order pairwise tree vs flat fold: m=128 43.79 vs 43.90 µs, m=64 33.94 vs 33.19, m=32 33.04 vs 32.10, m=256 70.2 vs 68.6, m=512 95.9 vs 94.3
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, eager on a CU-masked stream, output against the call alone] spin fold at m=256: 3,151,872 NaN on 64 CUs, 2,340,864 on 128 CUs, 35.5 and 12.4 ms per call; m=512: 4,276,224 and 3,158,016 NaN; the wait-free fold: none, bit for bit equal
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=2, AKLR graph timer (driver --bench-mode), the two versions interleaved] counters zeroed in the call vs counters in AKLR's state: gemm_n6144_k16384 m=32 36.41 vs 32.23 µs, m=64 37.30 vs 33.12, m=128 48.73 vs 44.06, m=256 70.62 vs 67.48, m=512 98.59 vs 94.49; m ≤ 16 and m ≥ 1024 within 0.4 µs
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=6–8, agent's probe on AKLR's graph timer, arms interleaved per case with the order alternating, median of 6–8 rounds] one `torch.zeros(64, int32)` launched before the GEMM: 11.909 µs against 8.013 without it (ratio 0.6728); `torch.empty` in its place 0.9993 (bf16 GEMM N=4096, K=2048, m=128, one column band of B per workgroup through LDS DMA rings)
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder (remeasured), 3 pairs, AKLR graph timer, one-sided 95% bounds] split-K 4 at m=64 folded in the GEMM by the split that counts in last on a launch-tagged counter, against its parent's separate fixed-order fold launch: +0.66% [+0.52, +0.80], m=64 +5.87%, the other seven cases −0.24% to +0.13% (bf16 GEMM N=K=6144, m 1–128, 256 workgroups × 8 waves, 96 columns each)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=2, agent's interleaved A/B (timer not stated in the record), then the driver bench twice] same kernel, tagged-counter fold against the fold launch: m=64 18.30 vs 19.38 µs; m=128 24.66 / 24.65 vs 23.87 / 23.93, loads batched 24.05 vs 23.82; removal probes at m=128 (timing only): swap and sum removed 20.05, swap kept and sum loads removed 21.03; output bitwise equal to the fold-launch kernel across graph replays; the run kept the m=64 path only, `allclose: True`, `--stress` PASS; dispatch-id probe: two eager calls 4 and 5, a second stream 0 with a different queue address, graph replays 12, 16, 20
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=5–11, agent's interleaved A/B on the driver's AKLR graph timer, medians of 5–11 rounds, canonical in the same run] per-element tagged slots, split 4: m=64 8×16 tiles 4.732 µs against the unsplit dot kernel's 4.940; m=128 16×16 tiles, compact block per XCD 5.299–5.314, split-major 5.776, one run per XCD 5.923, against the unsplit MFMA kernel's 5.556–5.585; ISA: `buffer_store_dwordx4 sc0 sc1`, `s_waitcnt vmcnt(0)`, four `buffer_load_dwordx4 sc0 sc1`; `--stress` PASS (m=64, then m=64 and m=128), a 200-round interleaved-m NaN/determinism check 0 bad outputs; G0 probe: dispatch id equal in every workgroup of a launch, eager 2, 5, 8, graph replays 53, 66, 79, high word 0, queue pointer low word different on two streams (bf16 GEMM N=128, K=6144)
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=5–9, agent's interleaved A/B on the driver's AKLR graph timer, medians of 5–9 rounds, canonical in the same run; FOLD2/FOLD3 give wrong output by design] per-element tagged slots, split 4: full fold / no `vmcnt(0)` between store and reads / store and wait without peer reads / no scratch store: m=64 (8×16, 512 threads, one run per XCD; unsplit 4.958) 4.740 / 6.044 / 4.397 / 4.419 µs; m=128 (16×16, 512 threads, split-major; unsplit 5.585 / 5.580) 5.708 / — / 4.708 / 4.487; final campaign kernel (compact block per XCD, 384 threads) full / FOLD2 / FOLD3: m=64 4.427 / 4.382 / 4.385, m=128 4.968 / 4.524 / 4.348; reads issued with no wait and a second pass behind `vmcnt(0)` on a miss: 6.073 vs 4.423 (m=64), 6.969 vs 4.954 (m=128) (bf16 GEMM N=128, K=6144)
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=2, AKLR `bench_cudagraph` with L2 flushed before every call, median of 5 readings; slot vs CAS 2 interleaved pairs, the config race one reading each; the m=1 whole-K and the final m=16 from the driver] per-(tile, wave) tag slots against the same kernel's launch-tagged CAS counter: m=32 9.33 / 9.16 vs 10.21 / 10.12 µs, m=64 12.12 / 12.10 vs 12.28 / 12.27, m=128 13.77 / 13.84 vs 14.23 / 14.30; one launch with split 4 against the whole-K kernel: m=16 7.80 vs 8.49 / 8.64, m=8 7.71 vs 7.00, m=1 7.65 vs 6.65 (driver); driver m=16 8.63 → 7.91; up to 1,500 calls per case with NaN-prefilled output, some concurrent on a second stream, plus 200 replays of a 20-call graph: 0 NaN, 0 mismatches, `--stress` PASS; "Checks not run: the 0xFF workspace prefill, alternating inputs on a reused workspace, and the ISA checks" (bf16 GEMM N=2048, K=7168)
- E11 [2026-11, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode (warmup 3, 20 iterations), one run per configuration] split-K 4 at m=64 arriving through a launch-tagged 64-bit compare-and-swap: acq_rel 23.30 µs with `buffer_wbl2 sc1` and `buffer_inv sc1` around every swap in the ISA, monotonic 12.87 with neither (bf16 GEMM N=2048, K=7168)
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=40 calls per check, eager calls checked bitwise against a clean run] reset-then-add counter, split 4 at m=4096: scratch prefilled with random words before each of 40 calls alternating two inputs, all bitwise equal to the clean result; a control build with the reset disabled gave mismatches with 262,144 NaN in iterations 37–39; ISA: no `buffer_wbl2`, `buffer_inv`, `s_sleep` or loop around the counter (bf16 GEMM N=64, K=4096)
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=5–10, agents' interleaved in-process A/B on AKLR's graph timer, medians of 5–10 rounds (N=256); single driver runs (N=64)] two-split exchange arrival (one `atomicrmw xchg`, the second arrival keeps its own partial in registers) against the compare-and-swap chain: incumbent 8.363 / 8.385 / 8.379 / 8.377 / 8.378, exchange 7.945 / 7.966 / 7.968 / 7.955 / 7.957 µs at m=512, `torch.equal`; next round, sentinel written back with the partner loads issued first 7.963 → 7.986 and per-wave words 7.970 → 7.925 at m=512, final bench 7.957, `--stress` PASS; per-wave 8 B words in one 128 B slot against one 128 B line per wave: 7.941 / 10.885 against 8.000 / 10.934 at m=512 / 1024 (bf16 GEMM N=256, K=6144); counter spacing between tiles at m=4096, HEAD 10.421 at 8 B: 64 B 10.082, 128 B 10.096 / 10.099, 256 B 10.045, later 512 / 1024 B 9.788 / 9.800; ablation of the 256 B round +1.46% [+1.31, +1.61], m=4096 +4.34% (bf16 GEMM N=64, K=4096)
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=1–3, single driver runs (N=64), AKLR `bench_cudagraph` medians over 3 rounds alternated with canonical (N=128), agents' probes on AKLR's graph timer (N=256); arms past the first are timing-only] N=64 m=4096 split 4: partials only 8.051, + `sc0 sc1` stores, `vmcnt(0)`, barrier 8.176, + counter round trip with the folder storing only its own partial 9.383, full fold 10.285 µs; N=128 K=6144 m=2048 split 4: planes stored, no counter 10.555, + store-acknowledge wait and barrier 10.577, + compare-and-swap arrival 12.761, + full fold 15.444, against canonical 13.060–13.071; correct with garbage and with reused scratch, and with inputs changing between graph replays; N=256 K=6144 m=1024 split 4: counter and fold removed 9.47, fixed last arriver without compare-and-swap 11.27, full-fold arms 12.55–12.92 against the unsplit incumbent's 11.55
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladders, 3 pairs, driver --bench-mode; the separate-launch arm by driver --bench-mode and Kineto] N=64 K=4096 m=4096: whole fold-introducing round +0.22% [−0.77, +1.21] overall, +1.89% at m=4096, m=256 −0.66%; fold-deletion arm reported as not separable, no cost given; next round against a different parent, counter spacing m=4096 +4.34%, m=256 +2.72%; N=256 K=6144 (whole rounds also changing rotation/PRO, exchange and own-partial retention; not isolated split effects): split m=512 with compare-and-swap arrival +1.59% [+1.31, +1.87] (m=512 +4.75%), exchange and split m=1024 +2.82% [+2.52, +3.12] (m=512 +5.62%, m=1024 +2.81%), per-wave arrival +1.16%, split 4 at m=1024 +0.86% (m=1024 +2.19%); N=64 m=2048 with fp32 partials and a reduce launch: driver 11.416 µs against HEAD 8.689, Kineto GEMM 7.64, `reduce_kernel` 4.32, memset 4.04
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=1, single driver run (N=64), agent's probe on AKLR's graph timer (N=256)] LLVM agent fences (release / seq_cst / acquire) around the arrival: N=64 K=4096 m=4096 (64×64, split 4) 26.177 µs (KW 4, 2 stages) against 11.922 for `sc0|sc1` buffer aux (KW 2, 3 stages); partials only, fold skipped, 11.349 (KW 4, 2 stages) and 9.743 (KW 2, 3 stages); N=256 K=6144 m=1024 (split 4, KW 2, BK 128 in both) 24.7 against 14.0 with `sc1` store/load bits
- E17 [2026-11, gfx950, FlyDSL 0.3.2, N=2–3 readings per arm, each a median of 5 calls of AKLR `bench_cudagraph` with L2 flushed before every call, interleaved with the kept kernel] bf16 GEMM N=384, K=7168, m=256: the kept whole-K path 9.44–9.47 µs; split 4 across workgroups on 32×64 tiles, 2 ring slots, tiles and their splits grouped by XCD (2 M-tiles × 3 N-tiles per XCD): 8.48–8.50; on 3 ring slots, 2 × 3 / 1 × 6 tiles per XCD: 8.55 / 11.17
- S1 aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json#kernel.py:49 `_SC01 = 17`, system-scope access past the non-coherent caches
- S2 aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json#kernel.py:426 `s_waitcnt vmcnt(0)` before the arrival
- S3 aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json#kernel.py:430 the agent-scope `atomicrmw add`
- S4 aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json#kernel.py:440 the last split folds in split order
- S5 aklr/ops/gemm/impls/flydsl/gemm_n6144_k16384/e8a254fd87b9a03e.json#kernel.py:443 the folding split's `xchg` of its counter back to zero
- S6 the N = K = 6144 campaign's final kernel, not committed: `_launch_tag` is `dispatch.id · H1 + queue.ptr · H2`, shifted left past the count bits (`_H1` odd)
- S7 the N = K = 6144 campaign's final kernel, not committed: the compare-and-swap loop, where a word with another tag counts as 0, `llvm.cmpxchg` monotonic, `syncscope="agent"`
- S8 the N = K = 6144 campaign's final kernel, not committed: one `torch.empty` per call holds the partials and the counter lines, not zeroed
- S9 the N=128, K=6144 campaign's final kernel, not committed: `_launch_tag` from the dispatch id's low and high words and the queue pointer's low word
- S10 the N=128, K=6144 campaign's final kernel, not committed: the 16 B slot store at aux 17, `s_waitcnt(vmcnt=0)`, one load per split, C written if every tag matches
- S11 the N=128, K=6144 campaign's final kernel, not committed: the slots in a `torch.empty` per call
- S12 aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json#kernel.py:394 planes stored, `vmcnt(0)` (:395), tag `did ^ rotl(qptr, 32)` (:403) stored to the wave's slot (:408), `vmcnt(0)` (:409), slots loaded (:413), `readfirstlane` (:422), fold if all match (:423)
- S13 aklr/ops/gemm/impls/flydsl/gemm_n2048_k7168/1f5a93c79dbd12bf.json#kernel.py:467 the planes and slots in a `torch.empty` per call
- S14 aklr/ops/gemm/impls/flydsl/gemm_n64_k4096/3af3d3de3d0f2285.json#kernel.py:304 the tile's counter word and the launch tag (`dispatch.id + queue.ptr · _TAG_MUL`, 56 bits) read before any DMA; the compare-and-swap reset of a stale word (:396-408) and the add whose old value `tag << 8 | S − 1` elects the folder (:410-418)
- S15 the N=256, K=6144 campaign's final kernel, not committed: two splits: `atomicrmw xchg` of the tag, the one that reads it back is second and swaps the sentinel back; lane 0 of each wave after its `vmcnt(0)`; own partial from registers plus the partner's
- S16 the N=256, K=6144 campaign's final kernel, not committed: one 128 B counter slot per tile, one 8 B word per wave; the tag is the dispatch id alone
- S17 aklr/ops/gemm/impls/flydsl/gemm_n384_k7168/d65ffa02dee32841.json#kernel.py:489 `slot_off`: two fp32 sums and the 64-bit tag (`dispatch.id` · H1 + `queue.ptr` · H2, :481) per 16 B slot, stored `sc0 sc1` at :501, `vmcnt(0)` at :502, the other splits' slots loaded at :509
