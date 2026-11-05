---
title: "GEMM at small M (decode, skinny) — what sets the time and what to tune"
kind: topic
lever: gemm_small_m
ops: [gemm]
gens: [gfx950]
dtypes: [bf16]
regimes: [decode, small-M, skinny, split-k]
flydsl: 0.3.2
sources:
  - aiter/ops/flydsl/kernels/splitk_hgemm.py
  - aiter/ops/flydsl/kernels/small_m_hgemm.py
  - aiter/ops/flydsl/gemm_kernels.py
  - aiter/tuned_gemm.py
  - csrc/py_itfs_cu/gemm_common.cu
  - csrc/gemm_a16w16/gemm_a16w16_tune.py
---

# GEMM at small M

> Scope: dense GEMM with M ≈ 1–256 (decode steps, speculative-decode verify), mostly bf16. MoE expert GEMMs at
> decode are in `../moe/flydsl_grouped_moe_inference.md`. Primus-Turbo's FlyDSL
> GEMMs (the exemplar elsewhere in this package) are built for training shapes and do not target small M.
> "The current kernels" are the baseline aiter's bf16 FlyDSL GEMMs behind `flydsl_hgemm`: the generic split-K
> kernel `aiter/ops/flydsl/kernels/splitk_hgemm.py` and the small-M kernel `kernels/small_m_hgemm.py`, which no
> shipped tuned row selects. Their arguments are in `../flydsl/kernels/flydsl_knob_space.md`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. The
> mechanisms carry over between kernels; the configurations do not. Each technique marked **Not in the current
> kernels** is a candidate to re-measure in your own.

## Route here when
- A decode or speculative-verify step spends visible time in dense GEMMs with M ≤ 256.
- You are racing, or reading tuned rows for, aiter's FlyDSL bf16 GEMM at small M.
- A small-M result looks good in a micro-benchmark and does not show up in the step.

## 1. Measure it the way decode runs it

- **B is cold in decode.** A step touches each weight once and streams tens of GB between visits, while a
  benchmark that replays one weight keeps it in the 256 MB MALL/L2. *(field, E1, E2)* The same kernel read 30.7 µs
  replayed and 45.1 µs in the real step, and the ranking of configurations flipped. Configurations chosen on a
  replayed weight were later measured to cost the step 3.3%.
- **Rotate enough distinct copies of B to exceed the MALL, with an explicit count.** *(field, E3)* aiter's
  `run_perftest` with the automatic rotation count picked 201 copies and read 41.56 µs for a kernel that read
  15.9–16.1 µs at every explicit count from 8 to 128.
- **Time a replayed CUDA graph, and know the floor.** *(field, E4, E5, E6)* An eager aiter call cost 16.7 µs of host time
  against kernels of 4.6–8 µs. One graph node cost 1.75–1.95 µs even for a 1-element op, so at
  M ≤ 16 on a small weight more than half the measured time is the node. *(field, E7, E8, E9)* Under AKLR's timer (L2
  flushed before each call) an empty kernel read ~4 µs and a second launch added ~4 µs (a separate split-K fold
  took M=8 from ~8 to ~11 µs): keep the fold in the GEMM launch. That ~4 µs is the timer's floor for any
  dispatch, not the kernel (`../SKILL.md` § Also in AKLR).
- **Price aiter's rows through the deployed entry** (`aiter.tuned_gemm.gemm_a16w16`, the call `gemm_baseline`
  makes), not a direct kernel call; time your own kernel through its `run` entry, as
  the benchmark calls it.
  *(field, E10)* A row that a direct call measured 0.76 µs faster was 0.34 µs slower through the entry, and in deployment
  that row alone cost about 4.2 percentage points of the step score.
- **A cold micro-benchmark shortlists; the step decides.** *(field, E11, E12, E13)* Kernel time converted to step time about
  1:1 (ratio 1.04), but 1.2% of the step score landed in other kernels, and a +10% uniformly-weighted gain was
  −1% when weighted by call counts.
- **A single-shape tuner on an idle GPU overvalues split-K.** *(field, E14)* Fifty tuned rows whose split-K only
  paid on an idle device made the step 18% slower (5742 vs 4853 µs). Deeper split-K buys parallelism from CUs
  that are idle only in the micro-benchmark.
- **Allocate fresh tensors for each repetition.** *(field, E15)* Eight allocations of one shape and configuration
  read 3.73–6.19 µs, bimodally: an address (L2-set) effect, not noise (§6, XCD placement).
- **Per-shape noise is large.** *(field, E16, E17, E18)* Per-shape drift of 3–7% while aggregates reproduced to 0.1–1.5%;
  the first arm of a probe read 1–2% slow after warm-up. Adopt a per-shape change only above that shape's
  measured spread. Within one process, run two arms as `A B B A` per rep and shuffle the order every rep
  when there are three or more (`measurement.md` §2).
- **A memory-side headroom estimate needs a measured streaming ceiling**, e.g. a plain read of the same B
  buffer, not the datasheet bandwidth. *(field, E19)* 5.8 TB/s for a cold read of a 476 MB weight.

## 2. Classify each call site first: streaming or latency-bound

The lessons below flip sign between two regimes. Decide which one a call site is in before reusing a
"what worked" list:

| | Streaming row | Latency-bound row |
|---|---|---|
| Example | o_proj / q_b / qkv projections, B tens of MB, ~100+ N-blocks | a narrow-N projection, B ≈ 1.5–3 MB |
| Bound by | HBM read of B | launch, reduction and memory latency |
| Split-K, more blocks | lose: +17–19% on an o_proj row already at an estimated ~7.7 TB/s with 192 blocks *(field, E21)* | win: −40% and −16% on two such rows, with a 4× larger grid and split-K 8 on one and split-K 16 on the other, B read cached instead of `nt` on both *(field, E22)* |
| `nt` on B | won on some long read-once streams, lost on another (§6) | loses: L2 is reducing latency, not only reusing |

## 3. Grid size at small M

- **Count workgroups against the 256-CU dispatch round before anything else.** *(field, E23, E24)* Padding one grid
  from 246 to 264 workgroups (empty blocks exit at once) made the kernel 50% slower: one round became two. Cold,
  through the deployed entry, every winning configuration sat at exactly 256 workgroups and 288 lost by 1.16×.
  `grid_pricing.md` §1 has the arithmetic.
- **"Fewer tiles than CUs" is necessary for starvation, not sufficient.** A cold, read-once B stream can be
  HBM-bound with ~200 blocks; then every way to more blocks loses (table above).
- **Cover M with one M-block: `tile_m = align_up(M, 16)`.** Each M-block re-streams all of B. *(field, E25, E26)* At M=24
  on a 38720-wide weight, two M-blocks of 16 took 127.7 µs and one block of 32 took 90.4 µs; three or more
  M-blocks were 2–4× slower than the best kernel recorded for the row. Make sure the tuner's `tile_m` filter can produce `align_up(M, 16)` for every M it
  covers.
- **On short K, race two XCD-paired M-blocks against one.** *(field, E27, E28)* At M 80–112, K=768, both M-blocks of an
  N strip on one XCD at doubled `tile_n` (same grid) won 6.7–6.9% at M 96–112 (1.7% at M=80) on one kernel and lost up to ~14% on another.
- **Odd tile widths can be the only way to reach a full round** (for N=6144 at small M, 256 workgroups need
  `tile_n` 48 with split-K 2, 96 with 4 or 192 with 8; aiter's catalog offers none of these widths), but they
  lost on other rows in deployment. Race them; do not adopt them by arithmetic.
- **A width that is no multiple of the MFMA N still builds:** compute at a wider multiple, load and store only the real
  columns (dead B rows offset past `num_records`, which reads zero). *(field, E29)* N/256 = 72 of 96: +0.5 to +7.1% at M ≥ 24, −1% at M ≤ 16; the readings at M ≤ 16, 24 and 40 sit inside that run's A/A spread.
- **Wider tiles lose at small M even when they move fewer bytes.** *(field, E30)* A tile chosen as line-cost-optimal
  by a traffic model was the slowest by 40% on a latency-bound row. Do not prune candidates with a traffic model.

In FlyDSL: `../flydsl/gemm.md` § small_m §3.

## 4. Split-K

**When it pays:** a single N-block (narrow N), where split-K is the only parallelism (34.1 → 5.0 µs at
1×32×6144 with split-K 12 *(field, E31)*), and latency-bound rows (§2). **When it does not:** streaming rows that already have
~100+ N-blocks, short K (splitting K=512 to fill a 0.44-wave grid cost 15–18% *(field, E32)*), and anything chosen on
an idle-device tuner (§1).

**The reduction is a fixed cost that dominates at M=1.**
- In the generic kernel, split 0 writes zero (or the bias) into C and raises a per-tile signal, the other splits
  spin until it is visible, and every split adds its bf16-rounded partial into C with an `atomicrmw fadd`
  (`splitk_hgemm.py:401-504, 1015, 1059-1066`). The small-M kernel lets the first split to arrive zero the tile
  (`small_m_hgemm.py:674-730`). *(field, E33)* That protocol cost 1.5–2.5 µs per launch at M=1, and its spin is the
  kind of wait the next bullet rules out.
- **No workgroup may wait for another.** A split that polls a flag, a counter or a tagged partial until another
  split has written assumes that split is already running. On a stream limited to part of the CUs, or beside another
  process, it may not be, and a capped poll then returns garbage. *(field, E106)* A split-K bf16 GEMM whose splits
  polled returned up to 4.7 M NaN at M 256–512 on a 16-CU stream, and calls of up to 140 ms; its wait-free rewrite
  passed the same test. This holds for every fold form: a designated last split, reducer workgroups appended after
  the producers, and every split summing a share of the tile all wait.
- **Fold by arrival instead.** Each split stores its partial at agent scope or wider (`sc1` or `sc0 sc1`, scope
  below), waits `s_waitcnt vmcnt(0)`, then adds one to its tile's counter with a device-scope atomic that returns the
  old value. The split that sees `s − 1` earlier arrivals reads every partial, its own through the same packing, and
  adds them in split order, so the result is bitwise the same whichever split folds; the others exit. *(field, E33)* Against zero-C + spin + atomics, per-split
  planes, a last-arriving block summing them in f32 and one counter per 128 B line took a 2048×2048, M=1, split-K 8
  kernel from 4.66 to 3.85 µs (replayed, warm weight: the cost is latency and coherence, not bandwidth).
  **Not in the current kernels.** How to write it, step by step: `wait_free_fold.md`.
- **Give each counter its own 128 B line.** *(field, E34)* Packing the per-tile counters as adjacent dwords put a whole
  launch on two cache lines and cost 1.1–2.5 µs.
- **Zero counters in per-call scratch before launching the GEMM.** AKLR kernels receive only their contract
  call keys and keep no state between calls. Allocate counters with `torch.zeros` on the input device on every
  call; the partial planes may use `torch.empty` because each is written before it is read. The zeroing adds
  one kernel to each split-K call. A launch tag in the counter word (`llvm.amdgcn.dispatch.id` xor `llvm.amdgcn.queue.ptr`) is 32
  bits and repeats: dispatch ids 1 and 2^32 on one queue give the same tag, and a stale word that reads as
  this launch's, already complete, leaves its tile unwritten.
- **Price split-K with this fold.** The last split to arrive reads every partial through one CU, and arrival skew
  leaves most tiles' fold to one split. *(field, E107)* Against folds that waited and spread the sum over all
  splits, it was 16% slower at M=128 and within 2% at M ≤ 64 on one GEMM, 24% and 8% slower at M 256 and 512 on
  another; a fixed-order pairwise tree of counters was no faster, because every level waits for its own stores to be
  acknowledged.
- A per-split marginal cost does not show the reduction: price the whole kernel.

**Constraints and correctness**
- **The current split-K is not bitwise reproducible** at split-K ≥ 3, or at split-K 2 with a bias: partials
  are added into C in arrival order. Measured, for both kernels: every one of 400 repeats differed at split-K 2
  with a bias and at 3, 4 and 8 *(field, E49)*. Where
  runs must match bitwise, use split-K 1 or 2 without a bias, or a fixed-order reduction
  (`../flydsl/kernels/flydsl_kernel_library.md` § Split-K).
- *(field, E50)* bf16 partials exceeded a 0.02 relative-error gate at split-K ≈ 12 (0.0241); f32 partials read 0.
  Gate every deep split-K row on accuracy, not only the first one tuned.
- **Race fp32 partials rounded to their top 24 bits, four packed in three dwords** (25% fewer plane bytes, error
  ≤ 2⁻¹⁶ of each partial). *(field, E51)* 4.5–4.9% at M ≈ 112, nothing at M ≤ 16; within 1 bf16 ulp unless the sum cancels.
- Coherence scope of the partials on gfx950: `sc0 sc1` is system scope (past the MALL). *(field, E52)* Using it for
  every partial was 5–9% slower than the narrowest correct scope; *(field, E53)* on another kernel agent scope (`sc1`)
  timed the same as `sc0 sc1`. Agent scope suffices across XCDs. XCD-local scope is correct only while every split
  of a tile runs on one XCD (workgroup ids `t + s·grid_x` with `grid_x % 8 == 0`), which rests on the dispatcher's
  round-robin, not on the kernel; *(field, E54, E55)* moving a tile's splits onto one XCD to use it gained nothing
  on one kernel and on another lost 2–6% at M 64–112 (within 0.5% of the canonical kernel at M ≤ 32).
- **Order the hand-off with the scoped stores and `s_waitcnt vmcnt(0)` before the counter atomic, not with fences**:
  an agent-scope release emits `buffer_wbl2 sc1`, acquire `buffer_inv sc1`; *(field, E56)* on a split-K hand-off the
  release cost ~3×, and the acquire made it slower still.
- **A fully co-resident grid hides a broken split-K hand-off.** *(field, E57)* Racy variants reported error 0 on grids
  that fit the device at once. Test the reduction on a grid larger than one dispatch round and on a stream limited to
  part of the CUs (`correctness.md` §3).
- If you pad a grid (for XCD alignment or a full round), padding blocks must exit before they touch the
  counter, workspace or swizzle.

In FlyDSL: `../flydsl/gemm.md` § small_m §4.

## 5. Tile, warps and pipeline

- **At M ≤ 16–32, a narrow `tile_n` (16 or 32) with K split inside the block and split-K 1** beat cross-block
  split-K on the rows compared; it needs a predicated C-store tail. *(field, E58)* On another shape, `tile_n` 64 was
  faster than 16 and 32, so race the width. *(field, E59)* At K=35840, cross-block split-K 8 on 256-wide tiles beat
  it by 3–8% at M=8 and M=24.
- **Splitting the K dimension across warps needs an LDS reduction; splitting N across warps does not.** The
  number of K-warps is per shape and tracks `block_k` (at most `block_k / 32`, where 32 is the K of the bf16 MFMA atom).
  *(field, E60)* Moving warps from K to N cut kernel time 1.0% and raised SNR from 50.8 to 94.8 dB.
- **Race the number of N-waves, odd counts included.** *(field, E61)* 3 N-waves (192 threads) over a 96-wide tile
  beat 2 by 5–8% at M 80–112; 6 waves against 4 moved within ±2%, at different stage counts.
- **For a row-major (N, K) B, `block_k` is the DRAM burst length of each row segment.** *(field, E62, E63, E64)* 256 was best
  by 9–40% cold on long-K rows and 512 was worse; at K=2048, 128 won because 8 iterations of 256 made the
  pipeline fill too large a share. *(field, E65)* At K=7168 with LDS capping stages, 256 × 3 stages lost 1.6% to
  128 × 5 at one paired point; race `block_k` together with the stage count it leaves room for.
- **Stages: deeper prefetch pays until LDS per workgroup crosses an occupancy step.** *(field, E66, E67)* With 2 stages
  the loop barrier drained `vmcnt(0)` every iteration; 2 → 4 stages was 1.43× at the winning tile. At a fixed
  tile, time jumped where LDS per workgroup crossed 81920 B (2 → 1 workgroup per CU). Sweep stages together
  with the tile; judging a tile at fixed 2 stages misranks it.
- **Keep B on the LDS path.** *(field, E68)* Loading B straight to registers was 2.6× slower.
  So was A at M ≈ 100: A in a 2–3-set VGPR ring was ~1.25× slower than A by LDS DMA *(field, E69)*.
- **The MFMA side is not the lever at decode M.** *(field, E70)* Deleting the whole hot-loop `ds_read` + MFMA moved an
  o_proj row by less than its noise; the time is the B stream.

## 6. Memory-side knobs

- **B's cache policy must be a raced, per-configuration axis.**
  - `nt` on B was −4.4 to −9.5% on long read-once streams *(field, E71)*.
  - On small or shared B it cost 6–15% *(field, E72, E73, E74)*.
  - It also lost on a read-once stream: `nt` or `sc1` on an 80 MB B at K=256 cost ~10–20% *(field, E75)*; `sc0` was neutral.
  - It couples with `block_k`: the same switch took one row from 98 to 159 µs at one `block_k` and another from
    153 to 83 µs at a different `block_k` *(field, E76)*.
  - A wants to stay cached: `nt` on A cost 4.5% *(field, E77)*.
  - So does C: `nt` on the C store cost up to ~2% at M 16–112 (tie at M=8), while the same GEMM's M≈900 tile gained ~2% *(field, E78)*.

  **Not in the current kernels.**
- **Race issuing each stage's A DMA before its B DMA.** *(field, E79)* 0.2–3.5% at M 21–98 on cold B, most at the largest M.
  On a second kernel it was within 0.1%; on a third, with L2 flushed per call, neutral to 1% slower at M 12–96
  (1.5–2.5% faster in a warm replay loop) and ~1% faster only on split-K tiles of ≥ 128 rows *(field, E80, E81)*.
- **Race a stagger of each block's first K-tile against none** (`k_start = (block_n · R) mod k_iters`, R coprime with `k_iters`).
  - With all blocks starting at k-tile 0, concurrent reads hit the same low address bits.
  - *(field, E82)* −2.9 to −8.6% per shape on cold read-once B.
  - *(field, E83, E84, E85, E86)* On three later kernels it was neutral or lost 1–6% (+12% at split-K 4): where blocks share an operand
    in L2, and on one `nt` B stream.
  - *(field, E87, E108)* Yet on one-M-block grids reading a `nt` B stream, dropping it cost 10.5–12% at M ≈ 112 on one
    kernel, and 4–13% at every M from 1 to 128 (split-K 1 up to M=16) on two GPUs on another, with HBM read bytes unchanged.
  - An R sharing a factor with `k_iters` (8 with 64 iterations) collapsed the blocks onto 8 starts and lost 55% against a coprime R (44% against no stagger) *(field, E88)*.
  - **The stagger is for B; an A that every block of an XCD re-reads wants the opposite.** Where that A is about the
    size of one XCD's L2 (4 MiB), make the start depend only on the XCD (`bid % 8`), so the XCD's blocks walk A
    together, or give each XCD one K split. *(field, E109)* At M=128 (A 4 MiB) per-block starts cost 41% against
    XCD-uniform ones, and won once the A loads were removed; at M=64 (A 2 MiB) per-block starts still won.
  - It interacts with the XCD band below; re-sweep both after any tile change.
  - **Not in the current kernels.**
- **When the grid leaves CUs idle, give them workgroups that load later K-tiles of B into L2** (cached, not `nt`).
  *(field, E89, E90)* Dropping them read ~6% slower at M ≈ 56–98 in single driver runs, within noise in the run's own ablation; they lost at M ≤ 21; race the first prefetched tile per M.
  *(field, E91, E92, E93)* The best start sat ~60–80% through K; a mid-K start was ~30% slower than none; on a split-K grid no start paid reliably (the best read 1.1% faster at M=112 in one run).
- **XCD placement at small grids.**
  - The hardware deals workgroup i to XCD `i % 8`, so with plain N-fastest order each XCD gets N-tiles 8 apart.
  - A band-cyclic remap (runs of adjacent N-tiles per XCD) removed the bimodal timings in §1 and was −2 to −3% on
    some rows *(field, E94, E95, E96)*, neutral or harmful on others; race it.
  - *(field, E97, E98, E99)* On a read-once B with no reuse between tiles, runs of 8 adjacent tiles per XCD beat plain order by
    0.6–4.6%; one contiguous slice per XCD, or runs of 80+, lost 2–13% against runs of 8.
  - **Not in the generic kernel.**
- **Preload the leading kernel arguments into SGPRs** (`amdgpu-kernarg-preload-count`); the dumped ISA then shows
  `.amdhsa_user_sgpr_kernarg_preload_length` > 0.
  - Preload is sequential and stops at the first aggregate (struct) argument: put the pointers and scalars first,
    and read `.amdhsa_user_sgpr_kernarg_preload_length` (in SGPRs) to see how far it got.
  - Preloading removes the `s_load` → `lgkmcnt(0)` round trip in front of the first B load.
  - `s_load`s + `s_branch` still open the ISA: a compatibility block; the body after its `.p2align 8` label reads
    the preloaded SGPRs. It does not mean the preload failed.
  - *(field, E100, E101, E102)* Kernel time −0.5 to −1.0% per shape, ~3% on every shape of a 6–10 µs kernel; the decode step got
    about 0.45% faster.
  - **Not in aiter's bf16 GEMM kernels.**
- **Fetch only the real A rows when M < 16**, but only where A is a large share of the bytes. *(field, E103)* −1.5% on
  one row, +11% on another where it made the A load wave-predicated and shortened the pipeline. Never predicate
  an LDS DMA per lane: a DMA under a partial exec mask does not land where the reader expects. **Not in the
  generic kernel.**

In FlyDSL: `../flydsl/gemm.md` § small_m §6.

## 7. aiter's tuned table at small M

The lookup (`get_GEMM_A16W16_config`, `aiter/tuned_gemm.py:130`) matches `gfx`, `cu_num`, N, K, bias, dtypes,
`scaleAB` and `bpreshuffle` exactly and tries M three times: as is, then `getPaddedM` at granularity 0, then at
granularity 1 (`csrc/py_itfs_cu/gemm_common.cu:13-40`):

| M | granularity 0 | granularity 1 |
|---|---|---|
| 1–256 | multiple of 16 | next power of two |
| 257–1024 | multiple of 32 | next power of two |
| 1025–4096 | multiple of 64 | next power of two |
| > 4096 | multiple of 128 | next power of two; 8192 when M > 8192 and N > 4096 |

What it means at small M:
- **A row serves every M that pads to it.** A table holding only an M=96 row served M 81–96 and nothing else,
  and one holding only an M=16 row served every M from 1 to 16 *(field, E105)*; by the same rule an M=128 row
  serves M 65–128. M=72 tries 72, 80 and 128, not 96; M=3 tries 3, 16, 4. *(field, E104)* An arm of four tuned
  rows cost the step 2.6–3.4%; the one row that lost was a padded M=96 row, at M=84. Validate a row
  across every M that reaches it, and add it only if it beats what the pad fallback already gives.
- **Merging model tables keeps the row with the lowest `us`** across files written by different rulers, rewrites
  the source CSVs and raises. Edit the owning file in place.
- **Tuning is compile-bound.** Compile candidates in parallel on the CPU, then time them.
- **A process does not see a table edit after its first lookup.** The lookup reads the file
  `AITER_CONFIG_GEMM_BF16` names or, by default, a merged copy under `/tmp/aiter_configs/` built once per
  process (`AITER_CONFIGS.get_config_file`, `aiter/jit/core.py`, also cached), so an edit to a source CSV shows
  only in a new process. Measured, with `AITER_CONFIG_GEMM_BF16` naming the edited file: the edit showed only
  after both `get_GEMM_A16W16_config_` and `get_GEMM_A16W16_config` were cleared. Graph-captured decode never
  sees it: capture after the table is final.

In FlyDSL: `../flydsl/gemm.md` § small_m §7.

## 8. Correctness traps specific to small M

- **N tail:** a store vector must be wholly in or out of N (keep `n % 8 == 0` for 16 B bf16 stores), and every
  launch-shape calculation must use `ceil(N / tile_n)`.
- **Wave-predicated loads change the graded `vmcnt`:** recompute the wait from the minimum per-wave issue count.
- **An LDS-size estimator that misses a field** (a 4-byte split-K flag) lets an over-limit configuration compile
  and fail at launch with `hipErrorIllegalState`.
- **An empty error string is not a measurement:** some configurations fail to compile with an empty exception;
  record them as failures, not as slow.
- Subtractive probes must keep the accumulators live and re-derive the pipeline waits, or they also remove the
  throttle they did not mean to remove (`attribution.md` §2).

## Related
`grid_pricing.md` · `epilogue.md` §5 (store cache policy) · `autotune.md` ·
`measurement.md` · `../flydsl/kernels/flydsl_kernel_library.md` ·
`../flydsl/kernels/flydsl_knob_space.md`

## Evidence

`aiter fork base <rev>` names the base commit of the campaign branch whose kernel was measured; the measured tree is
that branch at the time. A FlyDSL version is given only where the cited run, or another round in the same container and
venv, printed it (`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own
setup); otherwise `FlyDSL ?`.

- E1 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, hot = one weight replayed, cold = traced in the decode step] the same kernel read 30.7 µs replayed and 45.1 µs in the real step, and the ranking flipped (GLM-5.2 decode o_proj (8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:39)
- E2 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=2, decode step, ABBA interleaved] tables tuned on a replayed weight cost the decode step 3.3% (22.35 → 23.15 ms) (GLM-5.2 decode step, TP8 attn-DP8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:31)
- E3 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, aiter run_perftest, automatic vs explicit rotation count R] the automatic count picked 201 copies and read 41.56 µs; every explicit R from 8 to 128 read 15.86–16.14 µs (a GLM-5.2 TP4 decode GEMM with a 48 MB B, not named; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:152)
- E4 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, eager host-side call vs graph-timed kernel] aiter's eager call cost 16.7 µs median while its kernels ran 4.6–8 µs (the campaign's 63 small-M shapes, not individually named; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:109)
- E5 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, one node] one graph node cost 1.753 µs for a 1-element add_, against 2.983 µs for a 1×2048×2048 GEMM (59% of it the node); an earlier reading in the same campaign 1.766 µs (memory.md:175) (1-element add_; GEMM 1×2048×2048; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:256)
- E6 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, empty-node arm in the same 24-node graph] the per-graph-node launch cost measured 1.945 µs (1-element add in the GLM-5.2 decode GEMM graph; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:190)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3–9, AKLR bench_cudagraph (L2 flushed) in an in-process harness, median of 3–9 rounds] a fully empty kernel read 4.21 µs at m=8 (gemm_n2112_k7168 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_012.md:37)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver times, with a profiler split] a one-workgroup second kernel after the unchanged GEMM took 4.12 µs and took m=7 from 10.48 to 14.43 µs (gemm_n3584_k7168 forge run, iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3584_k7168/forge_experiments/lessons/iter_003.md:14)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph (L2 flushed) or driver --bench-mode] a separate fold kernel: m=8 11.09 against 8.08 µs, m=32 11.56 against 8.77; the fold kernel alone about 4.5 µs (gemm_n2112_k7168 forge run, iter 9; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_009.md:14)
- E10 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, cold R=16 through tuned_gemm.gemm_a16w16 vs a direct kernel call] a row a direct call measured 0.76 µs faster (17.74 → 16.98) was 0.34 µs slower through the entry (17.32 → 17.66); in deployment that row was worth about −4.2 points of score (the difference between the decode-score arms with and without it, E104) (o_proj M=96 row t96x96x128 split-K 4, GLM-5.2 TP4/EP4 decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:174)
- E11 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, decode step vs kernel trace, arm difference] Δ(all kernels) ×3 = +715 µs against Δ(step) = +742 µs, ratio 1.04 (GLM-5.2 decode step; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/goal.md:39)
- E12 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, decode step vs kernel trace] of a +742 µs step regression, +280 µs/step (1.2% of score) landed in other kernels (GLM-5.2 decode step; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/goal.md:139)
- E13 [2026-08, gfx950, FlyDSL ?, aiter fork base 1e5145d9d, N=?, cold microbench, uniform vs call-count weighting] summed uniformly over 56 shapes the microbench read +10% for FlyDSL; weighted by the step's call counts −1% (56 decode shapes; GLM-5.2 decode weights 78/78/78/21; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:47)
- E14 [2026-09, gfx950, FlyDSL ?, aiter fork 84538c1ae, N=?, whole decode step] the arm carrying 50 rows whose split-K paid only on an idle device ran 5742 µs against 4853 µs, 18% slower (GLM-5.2 decode GEMMs; aiter fork at /workspace/code/gpt_oss_docker/sync/inference/aiter; commit 84538c1ae)
- E15 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=8, graph, hot (one weight replayed)] eight independent allocations of one shape and configuration read 3.73–6.19 µs, bimodally ((4,6144,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:220)
- E16 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] per-shape cross-run drift 3–7% (256x6144x2048: 11.27, 12.12, 12.55 µs) while the aggregate bw_ratio reproduced to 0.07% (0.7454 / 0.7459) (the campaign's 63-shape score; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:175)
- E17 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=2, graph, hot (one weight replayed), another campaign's processes in the container] the same unchanged tree read 0.8153 and 0.7967 across sessions, recorded as a 1.5% span (the campaign's 63-shape score; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:210)
- E18 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, graph, cold (B rotated over >= 512 MB), 20 s burn-in] the first FlyDSL arm of a probe read 1–2% slow: 9.398/9.418/9.418 µs as arm 1 against 9.212/9.223/9.227 as the last arm (one decode GEMM configuration, three runs; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:178)
- E19 [2026-09, gfx950, torch rowsum, no FlyDSL kernel, N=?, plain read of the same B buffer] the matched HBM ceiling, a torch rowsum over the same B buffer, read 82.06 µs = 5798 GB/s for 476 MB (the 38720-wide weight of a camp23 GLM-5.2 decode point; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/memory.md:208)
- E21 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=1, graph, cold (B rotated over >= 512 MB)] split-K 2 35.450 µs (+18%), split-K 4 35.775 (+19%), tile_n 16 35.367 (+17%) against a 30.189/30.072 base (o_proj (8,6144,16384), 192 blocks; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:211)
- E22 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB), min of 7 replays] router 6.56 → 3.948 µs (−40%), gate 4.903 → 4.135 µs (−15.6%); both also switch B from nt to cached (MoE router (64,256,6144) shipped t32x64x128 split-K 8 → t16x32x128 split-K 8; gate (8,128,6144) t16x64x64 split-K 12 → t16x32x64 split-K 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/goal.md:76)
- E23 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] padding the grid from 246 to 264 workgroups (empty blocks exit at once) took the kernel from 7.14 to 10.70 µs (+50%) (2624x6144, grid_x 82 → 88; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:259)
- E24 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, cold R=16 through tuned_gemm.gemm_a16w16] every winner sat at exactly 256 CTAs; the losers at 288 CTAs measured ratio 1.159 (GLM-5.2 TP4/EP4 verify-M GEMM cells; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:144)
- E25 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] M=24 on a 38720-wide weight, tile_m 16/32/48/64: 127.71 / 90.42 / 92.36 / 97.23 µs (24,38720; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/rounds/round-03/agent.txt:19)
- E26 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] three or more M-blocks read 407.97 (84,38720 tm16), 268.02 (60,38720 tm16) and 190.87 µs (84,19360 tm16) against the frozen champions' 101.94 / 102.31 / 58.27, 2.6–4.0× off; the champions are the table's recorded us of other kernels, not one-M-block FlyDSL runs (84,38720; 60,38720; 84,19360; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/memory.md:208)
- E27 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] splitting m 65–128 rows into two XCD-paired M-blocks gained +6.65% at m=112, +6.86% at m=96 and +1.66% at m=80 (the same kept change also retiled m=2732) (gemm_n7168_k768, ladder entry 682c6e1ac (iter 5); local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k768/forge_experiments/ablation.json:189)
- E28 [2026-10, gfx950, FlyDSL 0.3.2, N=?, race.py scratch harness, flush not stated] two XCD-paired M-blocks (4,4,8,2,2) on the campaign's other (skinny) kernel read 7.409 / 7.603 / 8.105 µs against 6.509 / 7.126 / 7.531 at m=80/96/112 (+13.8%, +6.7%, +7.6%) (gemm_n7168_k768 forge run, iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k768/forge_experiments/lessons/iter_003.md:26)
- E29 [2026-10, gfx950, FlyDSL 0.3.2, N=3–6, AKLR bench_cudagraph (L2 flushed), alternating rounds in one process] base/variant time ratio of a 96-wide compute tile storing 72 columns: 0.9864–0.9942 at m ≤ 16, 1.0048–1.0706 at m ≥ 24; base against itself 0.9852–1.0099 (gemm_n18432_k1536 forge run, iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n18432_k1536/forge_experiments/lessons/iter_003.md:22)
- E30 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB), min of 7 replays] the line-cost-optimal router tile t32x64x128 (0.047 vs 0.094 for t16x32x128) was the slowest by 40% (MoE router (64,256,6144); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/goal.md:179)
- E31 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] at tile_n 32, one N-block: split-K 1 34.11 µs, split-K 8 7.24 µs, split-K 12 5.03 µs (1,32,6144; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/goal.md:197)
- E32 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, ?] split-K 2 to fill a 0.438-wave grid: +15.5% / +17.6% (kv_b draft GEMM, K=512; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:208)
- E33 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] a timing-only plain store in place of zero-C + spin + atomics: split-K 8 4.657 → 2.194 µs, split-K 4 4.180 → 2.664 (the reduction 1.5–2.5 µs); planes + last arriver with one counter per 128 B line: split-K 8 3.854 µs (2048x2048, M=1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:172)
- E34 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] per-tile counters packed as adjacent dwords put a launch on two cache lines and cost 1.1–2.5 µs (2048x2048 split-K reduction; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:175)
- E49 [2026-10, gfx950, FlyDSL 0.3.2, aiter a0e3cbea, N=400, eager calls, a side-stream matmul on alternate repeats] aiter's HGEMM and small-M split-K: with a bias at split-K 2, and at split-K 3, 4 and 8, 400/400 repeats differed from the first; split-K 1, and 2 without a bias, 0 (m,n,k = 16, 2048, 6144 (probe /workspace/code/gpt_oss_docker/work/sikl/skillenv/probes_sikl.py:165; its output is kept only in this note); local:/workspace/code/gpt_oss_docker/.claude/memory/project_sikl_flydsl_skill_port.md:30)
- E50 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] the same configuration at split-K 12: bf16 partials err_ratio 0.0241 (gate 0.02), f32 partials 0.000000 (a camp23 GLM-5.2 decode point, not named in the record; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/memory.md:181)
- E51 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver flushed bench (--bench-case, warmup 3, 20 iterations)] 24-bit packed partials: m=112 29.22 µs against 30.66 (masking alone) and 30.59 (canonical); m ≤ 16 unchanged (22.60/21.83/21.80 vs 22.56/21.83/21.81 µs); output within 1 bf16 ulp of canonical. "Unless the sum cancels" comes from a numpy simulation whose output was not kept (gemm_n8448_k7168 forge run, iters 4–5; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_004.md:32)
- E52 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, ?] unifying the split-K partials' scope with sc0|sc1 (0x1|0x10): 4.2945/4.3467/4.4319 µs, +8.8/+5.1/+4.7% slower (1,32 / 1,128 / 48,32 split-K rows; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/rounds/round-08/agent.txt:22)
- E53 [2026-10, gfx950, FlyDSL 0.3.2, N=5, AKLR bench_cudagraph, median of 5 runs] aux=16 (sc1) on scratch and flag traffic gave timings identical to sc0 sc1 at m=16, 64, 96 and 112 (gemm_n6016_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_007.md:12)
- E54 [2026-10, gfx950, FlyDSL 0.3.2, N=5, bench_cudagraph in process with the incumbent, median of 5] a tile's splits on one XCD with XCD-local scope: 18.34 vs 18.02, 18.73 vs 18.65, 21.25 vs 21.14 µs at m=48/64/96 (0.4–1.8% slower) (gemm_n6016_k7168 forge run, iter 5; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_005.md:12)
- E55 [2026-10, gfx950, FlyDSL 0.3.2, N=2, driver --bench-mode single case, two pairs] XCD-local exchange flag off vs on: m=112 29.68 vs 31.53/31.52 µs, m=64 25.98 vs 26.48 (on lost 2–6%); m=32 23.21 vs 23.04, m=16 22.15 vs 21.83 (on gained 0.7%, 1.4%), but the flag-off build was itself slower than canonical (m=16 22.16 vs 21.87, m=32 23.23 vs 22.93) (gemm_n8448_k7168 forge run, iter 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_006.md:14)
- E56 [2026-10, gfx950, FlyDSL 0.3.2, N=?, 200-call CUDA-graph harness] at m=14: both fences 30.81 µs, release fence only (acquire removed) 29.38 µs, no fences (write-through partial stores + s_waitcnt vmcnt(0), flags polled one at a time) 10.40 µs; the acquire alone, in the spin loop of the fence-free kernel, 11.22 vs 10.40 µs (gemm_n768_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards/demo_n768_12h/forge_experiments/lessons/iter_007.md:32)
- E57 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, ?] racy split-K variants reported err 0.00000 on the fully co-resident grids (1,32 / 1,128 / 48,32; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/memory.md:229)
- E58 [2026-09, gfx950, FlyDSL ?, aiter rev ?, N=?, "clean card", timer not stated] tile_n 16 / 32 added to aiter's FlyDSL catalog read 42.39 / 38.45 µs against 30.10 µs for its existing 64 (an N=6144 decode GEMM, M and K not stated; myskill b62bfc0; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/19-deployment-step-scoring.md:141)
- E59 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver graph-timer medians] cross-block split-K 8 on 32×256×64 tiles against the narrow-tile kernel: m=24 81.7 vs 88.7 µs, m=8 80.6 vs 82.9 µs (m=16 not compared) (gemm_n7168_k35840 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k35840/forge_experiments/lessons/iter_002.md:24)
- E60 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3 against late bases] warps split along N instead of K (w1x2x1): 11.257/11.320/11.449 vs 11.385/11.387 µs = −1.0%, SNR 50.8 → 94.8 dB (q_b (8,16384,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:259)
- E61 [2026-10, gfx950, FlyDSL 0.3.2, N=?, isolated times, warmup 3, 20 iterations, timer not stated] 3 against 2 N-waves over 96-wide tiles: m=80 19.73 vs 20.68, m=96 21.13 vs 22.66, m=112 22.62 vs 24.65 µs; 6 waves against 4 at m=112 22.27 vs 22.79 µs (different stage counts) (gemm_n6144_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_007.md:36)
- E62 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] raising TILE_K to 256 was worth 9% to 40% cold on the four decode shapes (GLM-5.2 decode GEMMs (8,6144,16384), (8,16384,2048), (8,2624,6144), (8,4096,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_140554/goal.md:47)
- E63 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] TILE_K 512: t16x32x512 split-K 4 44.74 µs against t16x32x256 38.79 µs ((8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_140554/goal.md:98)
- E64 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB)] STAGES 6 with BLOCK_K 128 beat STAGES 4 with BLOCK_K 256 (q_b (8,16384,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:177)
- E65 [2026-10, gfx950, FlyDSL 0.3.2, N=1, single-case driver bench, warmup 3, 20 iterations] block_k 256 × 3 stages against 128 × 5: m=16 44.3 → 45.0 µs (+1.6%, the one paired point); m=8 44.9 µs against an unpaired 43.4–44.9; m=32 did not fit LDS (gemm_n20480_k7168 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n20480_k7168/forge_experiments/lessons/iter_002.md:23)
- E66 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB)] with 2 stages the loop barrier drained vmcnt(0) every iteration; 2 → 4 stages at the winning tile 1.43× (o_proj (8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:142)
- E67 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] t16x64x128 split-K 4, stages 2–7: 42.65 / 45.79 / 52.57 / 52.32 / 52.97 / 52.81 µs, the cliff where LDS per workgroup crosses 81920 B ((8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_140554/goal.md:89)
- E68 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB)] b_to_lds=False: 76.528 vs 29.720 µs = 2.58× (o_proj (8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:223)
- E69 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver single case; a second, local CUDA-event timer] A in a 3-set VGPR ring: 68.46 µs at m=112 and 64.12 at m=98 against 53.35 and 51.32 for A by LDS DMA (1.28×, 1.25×); a 2-set ring on the local timer 1.24–1.26× (gemm_n20480_k7168 forge run, iter 4; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n20480_k7168/forge_experiments/lessons/iter_004.md:9)
- E70 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, graph, cold (B rotated over >= 512 MB)] dropping the whole hot-loop ds_read + MFMA: 30.850 µs against bases 31.044/30.927/30.745 (o_proj (8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:259)
- E71 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB), ABAB] nt on B: o_proj −9.5%, q_b −6.9%, qkv −4.4%, DSA −5.4% (the four GLM-5.2 decode shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:187)
- E72 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB), five-policy palindrome] nt on B: 4.49–4.56 µs with it, 3.955–3.967 without (MoE router (64,256,6144), B 3.1 MB; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/rounds/round-14/agent.txt:9)
- E73 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB)] the M=8 gate row, one m-block, paid 6.3% for nt (4.077 vs 3.822 µs) (gate (8,128,6144); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/rounds/round-14/agent.txt:16)
- E74 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=2, cold R=16 through tuned_gemm.gemm_a16w16, palindrome min-of-2] nt sc0 sc1 on B (cp1): +15.5% / +12.1% slower (kv_b verify cells, GLM-5.2 TP4/EP4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:208)
- E75 [2026-10, gfx950, FlyDSL 0.3.2, N=5–6, AKLR graph timer, arms interleaved in one process] B load cache bits, aux=0 / sc0 / nt / sc1: m=1 13.86 / 13.89 / 15.87 / 15.71 µs, m=8 14.41 / 14.41 / 17.17 / 17.31 µs (gemm_n163840_k256 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_012.md:15)
- E76 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] b_cpol=1 alone took 24,38720 at tile_k 64 from 98.04 to 158.68 µs and 10,38720 at tile_k 128 from 152.76 to 83.45 µs (24,38720; 10,38720; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/goal.md:235)
- E77 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB), ABAB] nt on A: 34.874/34.644 vs 33.289 µs = +4.5% (o_proj (8,6144,16384); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:187)
- E78 [2026-10, gfx950, FlyDSL 0.3.2, N=3, driver --bench-case, warmup 3, 20 iterations, 2–3 pairs per case] nt (aux=2) on the C store against aux=0: m=8 9.704 vs 9.703 (tie), m=32 11.075 vs 10.954, m=96 15.33 vs 15.12 µs; m=893 49.53–49.57 vs 50.48–50.52 µs (gemm_n7168_k3584 forge run, iter 8; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k3584/forge_experiments/lessons/iter_008.md:26)
- E79 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR bench_cudagraph (1 GiB read flush before each call), 8 input sets, median of 3–4 repeats] A DMA issued before B on top of the nt-B kernel: m=98 17.595 → 16.992 µs (−3.4%), m=21 −0.9%, m=56 −0.4%; at other m −0.038 to −0.173 µs (gemm_n3584_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3584_k7168/forge_experiments/lessons/iter_007.md:55)
- E80 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver single-case --bench-mode, 3 warmup, 20 iterations] A's DMA before B's: within 0.1% of canonical at m=16, 64 and 112 (gemm_n8448_k7168 forge run, iter 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_006.md:39)
- E81 [2026-10, gfx950, FlyDSL 0.3.2, N=3–7, AKLR bench_cudagraph (L2 flushed per call); warm = own graph replayed back to back] A before B on split rows: 1.5–2.5% faster warm; flushed, within ±0.3% at m=12–96 except m=80 20.10 vs 19.87 µs; m=893 72.58/72.75 vs 73.45/73.52, m=112 22.93/22.92 vs 23.09/23.06 µs (gemm_n6016_k7168 forge run, iter 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_006.md:12)
- E82 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB), ABAB] K-start stagger: −8.6 / −2.9 / −3.2 / −5.5% per shape (the four GLM-5.2 decode shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:203)
- E83 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph; driver --bench-mode] stagger on split-K rows, variant vs incumbent: m=32 8.88 vs 8.76 µs up to m=112 13.00 vs 12.34 (+1.4% to +5.3%); the driver read m=112 13.31 vs 12.53 (+6.2%) (gemm_n2112_k7168 forge run, iter 9; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_009.md:11)
- E84 [2026-10, gfx950, FlyDSL 0.3.2, N=?, timer not stated in that section (the record's other m=893 sections are warm L2)] split-K 4 at m=893: no rotation 46.82 µs, bid%LOOPS 52.55 µs (+12%) (gemm_n2112_k7168 forge run, iter 3; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_003.md:128)
- E85 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR bench_cudagraph (flushed), 8 input sets] with B loaded sc0 nt, stagger 0 against stagger 3 on every split-K case: −0.238 to −0.468 µs at m=28–84, +0.020 at m=98 (gemm_n3584_k7168 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3584_k7168/forge_experiments/lessons/iter_007.md:78)
- E86 [2026-10, gfx950, FlyDSL 0.3.2, N=5–11, in-process AKLR bench_cudagraph, arms interleaved, median of 5–11 readings] K-start stagger R=1 / R=5 against off at m=48–112: +0.8% to +1.3% (m=64 7.728 / 7.715 vs 7.640 µs) (gemm_n7168_k1536 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1536/forge_experiments/lessons/iter_012.md:9)
- E87 [2026-10, gfx950, FlyDSL 0.3.2, N=?, single-case driver bench, warmup 3, 20 iterations] K_STAGGER 0 at m=112: 60.7 µs against 54.1–54.5 for 3, 7 and 9 (+12%); on a fresh cache 58.80 vs 53.22 µs (+10.5%) (gemm_n20480_k7168 forge run, iters 2–3: 256 workgroups, one M-block, split-K 1, B sc0 nt; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n20480_k7168/forge_experiments/lessons/iter_002.md:13)
- E88 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=?, graph, cold (B rotated over >= 512 MB)] stagger stride kr3 30.284 µs, kr8 47.150 µs (gcd(8, 64) ≠ 1 collapses 192 blocks onto 8 starts) (o_proj (8,6144,16384), 64 k-iterations; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:199)
- E89 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations, against canonical in the same session] canonical with prefetch off at m=98: 10.04 µs against 9.47/9.49; the run's own ablation of the change was within noise (+0.20% mean, m=98 −3.3%) (gemm_n7168_k1792 forge run, iter 13; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1792/forge_experiments/lessons/iter_013.md:71)
- E90 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver single case, shared GPU] prefetch workgroups at m=21: 6.18–6.26 µs against 6.13 without (gemm_n7168_k1792 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1792/forge_experiments/lessons/iter_012.md:41)
- E91 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations] fastest prefetch start tile per M: 26 at m=24, 27 at m=32 down to 19 at m=112, of 33 K-tiles (33 = K/block_k, computed); off was best at m=8 and 16 (gemm_n7168_k4224 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k4224/forge_experiments/lessons/iter_002.md:71)
- E92 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations] at m=64, prefetch start 0 (off) 0.016432 ms, start 16 0.021571 ms (+31%) (gemm_n7168_k4224 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k4224/forge_experiments/lessons/iter_002.md:69)
- E93 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case; the driver also printed source_rejected for the hooked build] on the split-K grid, base against prefetch starts 13 / 17 / 21: m=64 13.949 vs 25.564 / 16.433 / 13.97, m=112 17.693 vs 25.499 / 21.059 / 17.5 µs (the best start within ~1% of base) (gemm_n7168_k4224 forge run, iter 5; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k4224/forge_experiments/lessons/iter_005.md:18)
- E94 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] band-cyclic remap: xb1 mean 4.90 µs (3.89..6.06) → every band 2..24 3.62–3.72 µs, deterministic ((4,6144,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_081239/memory.md:220)
- E95 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=2, graph, cold (B rotated over >= 512 MB)] every banded configuration below both base reads, ~−2.6% ((8,2624,6144); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:163)
- E96 [2026-09, gfx950, FlyDSL ?, aiter fork 2c71811b3, N=?, run_perftest rotate_args, palindromic order, same-binary control] xcd_band 2 worth −2.83% against a 0.13% noise floor (o_proj 6144x16384, M=1; aiter fork at /workspace/code/gpt_oss_docker/sync/inference/aiter; commit 2c71811b3)
- E97 [2026-10, gfx950, FlyDSL 0.3.2, N=4–6, AKLR graph timer, arms interleaved in one process] runs of 8 tiles per XCD (g8) against plain order (g1) at m=5–10: 14.55 vs 14.92 µs at m=5 … 14.51 vs 14.94 at m=10 (1.4–2.9%) (gemm_n163840_k256 forge run, iter 11; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_011.md:54)
- E98 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR graph timer, arms interleaved in one process] one contiguous slice per XCD: 16.45 vs 16.06 µs at m=14 and 16.92 vs 15.40 µs at m=12 (gemm_n163840_k256 forge run, iter 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_006.md:25)
- E99 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR graph timer; the g8 base is taken from another line of the same record] runs of 80 and 640 tiles: g=80 14.46/15.04/14.50/14.63 and g=640 15.27/16.30/15.24/15.41 µs against g8 14.45/14.64/14.23/14.39 (gemm_n163840_k256 forge run, iter 13; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_013.md:28)
- E100 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3] kernarg preload: (8,6144,16384) 29.640/29.429/29.464 vs 29.970/29.655/29.587 µs (−0.5%); (8,16384,2048) −1.0%, (8,2624,6144) −0.6%, (8,4096,2048) −0.7% (the four GLM-5.2 decode shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:247)
- E101 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder row marked recorded (the campaign's paired KEEP timing), driver --bench-mode] kernarg preload: +3.41% mean (lower bound +2.89%), +2.0% at m=98 to +4.3% at m=7 and 35, all 11 cases positive; kernel 5.9–10.8 µs (gemm_n7168_k1792, kept change 0101b4a (iter 4); local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n7168_k1792/forge_experiments/ablation.json:225)
- E102 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, decode step, min of 4 runs, three scored runs] with preload the step scored 47.0522 / 47.0345 / 47.0057 against 46.8209, about +0.45% (GLM-5.2 decode step; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:251)
- E103 [2026-08, gfx950, FlyDSL 0.3.2, aiter fork base 1e5145d9d, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3] fetching only the real A rows (mr8): qkv 8.310/8.310/8.317 vs 8.567/8.435/8.432 µs (−1.5%); q_b 13.642/13.655/13.667 vs 12.585/12.302/12.307 (+11%) (qkv (8,2624,6144), q_b (8,16384,2048); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_145853/memory.md:223)
- E104 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=2, e2e decode bench (c8/c10/c14 tok/s), two palindromes, 8 scored reads] an arm adding four tuned rows, the M=96 o_proj row among them, scored −2.63% and −3.35%; a cold probe through the entry found that row the only loser, at M=84 (o_proj 17.32 → 17.66 µs). The record states, without measuring it, that the row also served the M=72 verify batch (GLM-5.2 TP4/EP4 decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_171610/memory.md:172)
- E105 [2026-10, gfx950, FlyDSL 0.3.2, aiter in AKLR, N=?, tuned-table lookup probe through AKLR] a table holding only an M=96 row served M 81–96; one holding only an M=16 row served M 1–16 (local:/workspace/code/gpt_oss_docker/.claude/memory/project_sikl_flydsl_skill_port.md:31)
- E106 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the campaign's stress test (CU-masked streams of 16/32/64/128 CUs, a second launch of the kernel beside it, eager and graph), outputs checked for NaN and against a reference] the delivered split-K kernel, whose splits polled flags and planes up to `_MAX_POLLS` and then wrote NaN: up to 4.7 M NaN at m=256 and m=512 and calls of up to 140 ms on 16 CUs; the rewrite in which the last split to arrive at a launch-tagged counter folds: no NaN and no difference under every mask (0.71 ms on 16 CUs); the other K3 split-K GEMM's polled fold was right under the masks but once gave NaN at m=32 while another process shared the GPU (bf16 GEMM n6144 k16384, m=256 and 512; local:/shared_nfs/kyle/moe/k3six/fix/gemm_l_session.md:19; local:/shared_nfs/kyle/moe/k3six/deliver/validate_1653/report.md:36)
- E107 [2026-10, gfx950, FlyDSL 0.3.2, N=2–6 interleaved rounds, forge driver bench on AKLR's graph timer, the kernel before the wait-free commit against after] arrival fold against a fold in which every split waited and summed a share of the tile: m=1–16 within 0.1%, m=32 +0.6%, m=64 +1.8%, m=128 37.95 → 44.00 µs (+15.9%) on one GEMM; m=256 0.0551 → 0.0686 ms (+24.3%), m=512 0.0875 → 0.0947 (+8.3%) on the other; a pairwise tree of per-pair counters: m=128 43.79, m=256 0.0702, m=512 0.0959 (bf16 GEMM n6144 k16384, m=1–128 and m=256–512 lanes; local:/shared_nfs/kyle/moe/k3six/fix/gemm_s_session.md:26; local:/shared_nfs/kyle/moe/k3six/fix/gemm_l_session.md:24; :25; local:/shared_nfs/kyle/moe/k3six/fix/gemm_s_tree.md:20; local:/shared_nfs/kyle/moe/k3six/fix/gemm_l_tree.md:19)
- E108 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, AKLR bench_cudagraph (1 GiB flush per call), one process per set with a fresh FlyDSL cache, two GPUs; placement alone moved a case about ±1 µs between processes] per-block K start `(bid·R) mod loops` against none (B `sc0 nt`): GPU 3 m=1/2/4/8/16/32/64/128 32.98/33.10/31.89/32.71/33.09/33.81/38.28/43.91 against 34.93/36.85/34.81/35.50/36.28/35.99/39.77/48.71 µs; GPU 6 the same sign on every case, 2.4–5.5 µs; HBM read 202–244 MB per call with and without it, request stalls lower with it (bf16 GEMM n6144 k16384; local:/shared_nfs/kyle/moe/k3six/runs/gemm_s/workspace/forge_experiments/analysis/236ca0f0c864d119d1b61ee6ca9f60fea6b03734/generation-001/ablation/summary.md:111; :40; local:/shared_nfs/kyle/moe/k3six/runs/gemm_s/workspace/forge_experiments/analysis/236ca0f0c864d119d1b61ee6ca9f60fea6b03734/generation-001/pmc_compare.md:7)
- E109 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, as E108] m=128 (A 128 × 16384 bf16 = 4 MiB, every split of a tile on XCD `bid mod 8`): per-block starts R=1 61.72 µs (R=3 and 5 61.8–63.4 on GPU 6) against R=16, one start per XCD, 43.91; with the A loads removed R=1 34.59 against 37.08; m=64 (A 2 MiB) R=1 38.93 < R=3 39.54 < R=16 42.01 (GPU 6); with each XCD on one K split, R=1 41.96–42.00 against R=16 42.46 (bf16 GEMM n6144 k16384; local:/shared_nfs/kyle/moe/k3six/runs/gemm_s/workspace/forge_experiments/analysis/236ca0f0c864d119d1b61ee6ca9f60fea6b03734/generation-001/ablation/summary.md:129; :89; :66; local:/shared_nfs/kyle/moe/k3six/runs/gemm_s/workspace/forge_experiments/lessons/iter_003.md:32; :38)
