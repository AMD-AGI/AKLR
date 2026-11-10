---
title: "GEMM at small M (M ≤ 256, skinny) — what sets the time and what to tune"
kind: topic
lever: gemm_small_m
ops: [gemm, grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [bf16]
regimes: ["M 1-256", read-once B, small-M, skinny, split-k]
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

> Scope: dense GEMM with M ≈ 1–256 (as in decode and speculative verify), mostly bf16. MoE expert GEMMs at any
> token count start at `../moe/INDEX.md` (sorted-token rows: `../moe/grouped_sorted_tokens.md`); §1, §5 and §6
> apply to them too. Primus-Turbo's FlyDSL GEMMs (the exemplar elsewhere in this package) do not target small M.
> "The current kernels" are the baseline aiter's bf16 FlyDSL GEMMs behind `flydsl_hgemm`: the generic split-K
> kernel `aiter/ops/flydsl/kernels/splitk_hgemm.py` and the small-M kernel `kernels/small_m_hgemm.py`, which no
> shipped tuned row selects. Their arguments are in `../flydsl/kernels/flydsl_knob_space.md`.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. The
> mechanisms carry over between kernels; the configurations do not. Each technique marked **Not in the current
> kernels** is a candidate to re-measure in your own.

## Route here when
- Dense GEMMs with M ≤ 256 take visible time in a model step (as in decode or speculative verify).
- You are racing, or reading tuned rows for, aiter's FlyDSL bf16 GEMM at small M.
- A small-M result looks good in a micro-benchmark and does not show up in the step.

**Somewhere else if:** M is 256–4096 (measured on bf16): the round and split-K at a full round in `grid_pricing.md`
§1–2, the narrow-N loop in `mainloop.md` §11, the store in `epilogue.md` §5.

## 1. Measure it the way the model step runs it

- **B is cold in a step that touches each weight once** (as decode does): it streams tens of GB between visits, while a
  benchmark that replays one weight keeps it in the 256 MB MALL/L2. *(field, E1, E2)* The same kernel read 30.7 µs
  replayed and 45.1 µs in the real step, and the ranking of configurations flipped. Configurations chosen on a
  replayed weight were later measured to cost the step 3.3%.
- **Rotate enough distinct copies of B to exceed the MALL, with an explicit count.** *(field, E3)* aiter's
  `run_perftest` with the automatic rotation count picked 201 copies and read 41.56 µs for a kernel that read
  15.9–16.1 µs at every explicit count from 8 to 128.
- **Time a replayed CUDA graph, and know the floor.** *(field, E4, E5, E6)* An eager aiter call cost 16.7 µs of host time
  against kernels of 4.6–8 µs. One graph node cost 1.75–1.95 µs even for a 1-element op, so at
  M ≤ 16 on a small weight more than half the measured time is the node. *(field, E7, E8, E9)* Under AKLR's timer of
  measurement profile 18 and earlier (Kineto, `GPU_FORCE_QUEUE_PROFILING` forced, L2 flushed before each call) an
  empty kernel read ~4 µs and a second launch added ~4 µs (a separate split-K fold took M=8 from ~8 to ~11 µs): keep
  the fold in the GEMM launch, and price a per-call counter zeroing, one more kernel (`wait_free_fold.md` E4),
  against split-K 1, a separate fixed-order reduce and a launch tag in place of the zeroing (§4).
- **That ~4 µs was what the profile-18 ruler (`bench_cudagraph`, as CI timed) recorded per kernel, not a hard
  floor.** *(field, E110)* In Triton, a graph of trivial kernels (each a 64-float load and store) cost ~1.77 µs of wall
  clock per added kernel, while that ruler recorded 4.05–4.08 µs per added kernel with or without
  `GPU_FORCE_QUEUE_PROFILING`. *(field, E111, E112)* Two bf16 GEMMs read below an empty FlyDSL kernel timed the same
  way: 4.235–4.254 µs at m=1 against 4.32–4.36 (N=7168, K=512), and 4.07–4.26 µs at m 1–4 against 4.32–4.34 (N=512,
  K=4096). Current main times with its own rocprofiler-sdk tracer (`aklr_kernel_tracer`, profile 19) and no longer
  forces queue profiling, so recheck the floor there: time an empty kernel, the baseline and your candidate in pairs
  on current main before deciding a Point near 4 µs has no headroom (`../SKILL.md` § Also in AKLR;
  `../pitfalls/aklr_dispatch_floor.md`).
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
  buffer, not the datasheet bandwidth. *(field, E19)* 5.8 TB/s for a torch row sum over a 476 MB weight.

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

- **Count workgroups against the 256-CU dispatch round before anything else.** *(field, E23)* Padding one grid
  from 246 to 264 workgroups (empty blocks exit at once) made that kernel 50% slower: at an estimated one workgroup
  per CU, one round became two. That does not make exactly 256 an optimum: *(field, E24)* a 256-against-288 race
  that read 1.16× was timed on another aiter checkout than the one served, so it ranks nothing; time the grids
  around a full round on the binary you serve (`../pitfalls/measured_a_different_build.md`). `grid_pricing.md` §1
  has the arithmetic.
- **"Fewer tiles than CUs" is necessary for starvation, not sufficient.** A cold, read-once B stream can be
  HBM-bound with ~200 blocks; then every way to more blocks loses (table above).
- **On a wide N, up to ~64 rows, cover M with one M-block: `tile_m = align_up(M, 16)`** (measured at M ≤ 84 on
  N = 38720). Each M-block re-streams all of B; above ~64 rows race two XCD-paired M-blocks (the short-K bullet
  below, measured at N=7168, K=768). *(field, E25, E26)* At M=24
  on a 38720-wide weight, two M-blocks of 16 took 127.7 µs and one block of 32 took 90.4 µs; three or more
  M-blocks were 2–4× slower than the best kernel recorded for the row. Make sure the tuner's `tile_m` filter can produce `align_up(M, 16)` for every M it
  covers.
- **On a narrow N, race several small M-blocks against fewer, larger ones**, keeping the grid at one dispatch round.
  *(field, E125)* At N=1024, K=7168, m=16, two 8-row M-blocks with 8-column tiles took 6.54 µs against 7.61 for one
  16-row M-block with 4-column tiles (256 workgroups both); at m=32, four 8-row M-blocks with 16-column tiles read
  8.21 against 8.17 for two 16-row M-blocks with 8-column tiles (256 workgroups both). At m=8, 8-column tiles with a
  K-start rotation (one M-block, 128 workgroups) lost to 4-column tiles on 256 workgroups (6.64 against 5.79).
  *(field, E126)* At N=512, K=4096 (warm L2, the run's own timer) four M-blocks on 256 workgroups beat two on
  128: m=64 5.43 against 7.97 µs, m=32 4.00 against 5.27.
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
- **No result may depend on a split that is not running.** A split that polls a flag, a counter or a tagged
  partial until another split has written assumes that split is already running. On a stream limited to part of the
  CUs, or beside another process, it may not be, and a capped poll then has no correct value to write when it gives
  up, unless it hands its share to the split that arrives last, which has all the data
  (`../attention/split_kv.md` §2, Bounded wait). *(field, E106)* A split-K bf16 GEMM whose splits polled up to a cap
  and then wrote NaN returned NaN at m=256 and 512 on streams limited to part of the CUs (2.3–4.3 M NaN in the
  test on 64 and 128 CUs, up to 4.7 M in the
  run's summary) and took up to 140 ms per call on 16 CUs; its wait-free rewrite passed the same test. A designated
  last split, reducer workgroups appended after the producers, and every split summing a share of the tile all wait;
  each is correct without residency only with that hand-off, which has not been measured on a GEMM fold.
- **Fold by arrival instead.** Each split stores its partial at agent scope or wider (`sc1` or `sc0 sc1`, scope
  below), waits `s_waitcnt vmcnt(0)`, then adds one to its tile's counter with a device-scope atomic that returns the
  old value. The split that sees `s − 1` earlier arrivals reads every partial, its own through the same packing, and
  adds them in split order, so the result is bitwise the same whichever split folds; the others exit. *(field, E33)* Against zero-C + spin + atomics, per-split
  planes, a last-arriving block summing them in f32 and one counter per 128 B line took a 2048×2048, M=1, split-K 8
  kernel from 4.66 to 3.85 µs (replayed, warm weight: the cost is latency and coherence, not bandwidth).
  **Not in the current kernels.** How to write it, step by step: `wait_free_fold.md`.
- **Spread the tiles' counters apart, and race the spacing.** *(field, E34)* Packing the per-tile counters as
  adjacent dwords put a whole launch on two cache lines and cost 1.1–2.5 µs. In a counter-stride sweep two tiles'
  counters in one 128 B line did as well as a line each, and one tile's per-wave words may share its line
  (`wait_free_fold.md` E13).
- **Zero counters in per-call scratch before launching the GEMM.** AKLR kernels receive only their contract
  call keys and keep no state between calls. Allocate counters with `torch.zeros` on the input device on every
  call; the partial planes may use `torch.empty` because each is written before it is read. The zeroing adds
  one kernel to each split-K call. A launch tag in the counter word, or beside each partial, replaces the zeroing
  (`wait_free_fold.md` § How to write it, § Folding without a counter: launch-tagged slots); several bf16 kernels
  do so and pass `driver.py --stress`. Make the tag wide: `llvm.amdgcn.dispatch.id` xor `llvm.amdgcn.queue.ptr` is
  32 bits and repeats (dispatch ids 1 and 2^32 + 1 on one queue give the same tag), and a stale word that reads as
  this launch's, already complete, leaves its tile unwritten. A wider dispatch-derived tag is one to one only within
  its 62–64-bit domain on one queue; add queue-pointer bits, and note that truncated queue bits or an XOR do not make
  it unique across queues. The case no test covers is a leftover scratch word equal to this launch's tag
  (`wait_free_fold.md` § Cost and risk).
- **Price split-K with this fold.** The last split to arrive reads every partial through one CU, and arrival skew
  leaves most tiles' fold to one split. *(field, E107)* Against folds that waited and spread the sum over all
  splits, it was 16% slower at M=128 and within 2% at M ≤ 64 on one GEMM, 24% and 8% slower at M 256 and 512 on
  another; a fixed-order pairwise tree of counters was no faster, because every level waits for its own stores to be
  acknowledged.
- **Split-K in one launch on launch-tagged slots paid at m=64 and 128 on N=128, K=6144 and at m=16 on N=2048,
  K=7168 (4.9–8.6 µs cases), read 0.5–1.1% slower at m=32 on N=128 and lost at m=1 and 8 on N=2048 (m 2–4 not
  tried).** Against the run's kernel with the whole K in each workgroup at that m: *(field, E113)* N=128,
  K=6144, split-K 4: m=64 4.73 against 4.94 µs, m=128 5.30–5.31 against 5.56–5.59, but m=32 4.26–4.29 against 4.24;
  *(field, E114)* N=2048, K=7168, split 4 (one reading per configuration): m=16 7.80 against 8.49/8.64 µs, but m=8
  7.71 against 7.00 and m=1 7.65 against the driver's 6.65, so m ≤ 8 kept the whole K in one workgroup. At N=384,
  K=7168 the same split beat that file's whole-K builder at m=1 and 8, where the builder has 48 workgroups
  (`../cases/k3_bf16_splitk_narrow_n.md` § N 384, its E23): race it at each N.
- A per-split marginal cost does not show the reduction: price the whole kernel.

**Constraints and correctness**
- **The current split-K is not bitwise reproducible** at split-K ≥ 3, or at split-K 2 with a bias: partials
  are added into C in arrival order. Where
  runs must match bitwise, use split-K 1 or 2 without a bias, or a fixed-order reduction
  (`../flydsl/kernels/flydsl_kernel_library.md` § Split-K).
- *(field, E50)* bf16 partials exceeded a 0.02 relative-error gate at split-K ≈ 12 (0.0241); f32 partials read 0.
  Gate every deep split-K row on accuracy, not only the first one tuned.
- **Race fp32 partials rounded to their top 24 bits, four packed in three dwords** (25% fewer plane bytes, error
  ≤ 2⁻¹⁶ of each partial). *(field, E51)* 4.5–4.9% at M ≈ 112, nothing at M ≤ 16; within 1 bf16 ulp unless the sum cancels.
- Coherence scope of the partials on gfx950: `sc0 sc1` is system scope (past the MALL). *(field, E52)* On three
  split-K rows of one campaign, `sc0 sc1` on every partial read 4.7–8.8% slower than agent scope (`sc1`); the same
  round put the spread between readings of one arm at 7–10%. On two
  other kernels it was not slower: *(field, E53)* agent scope (`sc1`) timed the same as `sc0 sc1`; *(field, E115)* on a
  launch-tagged fold with every split of a tile on one XCD (N=128, K=6144), an arm the record calls LOCAL, XCD-local
  scope (aux 0 or 1) on the scratch traffic (which accesses it changed is not written), read 4.52 against 4.43 µs at
  m=64 and 5.18 against 4.97 at m=128, while aux 0 or 1 on the read-back loads alone, stores kept at `sc0 sc1`, tied
  (m=64 4.42 against 4.42, m=128 4.95–4.96 against 4.97; aux 0 in another run 5.28–5.29 against 5.30–5.31). Race the
  scope. Agent scope suffices across XCDs. XCD-local scope is correct only while every split
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

- **Race a narrow output tile whose workgroup reduces the whole K (split over its waves, split-K 1) against wider
  tiles and cross-block split-K**; it needs a predicated C-store tail. Which side won depended on the comparator and
  on m:
  - Against `torch.matmul` (the library kernel it dispatched was not recorded), one such launch, as the first kept
    change, gained on the mean over each run's cases on all
    14 bf16 shapes measured (N 64–28672, K 512–7168, m 1–128): +7.84% to +114.94%, every one-sided lower bound at
    least +5.25% *(field, E116)*. On 11 of them most m split the tile's K over the waves and folded the partials
    through LDS in wave order; the other three were aiter's HGEMM with split-K 1 (N=3584, K=512), a whole-K tile whose
    waves split M × N (N=7168, K=512), and a kernel whose waves split K only at m=1 and m=32 (N=28672, K=512). Three
    runs left m above 16 or 32 on `torch.matmul` with that change, and two lost to it at their largest m: N=512, K=4096, m=64
    by 7.7%; N=6144, K=4096, m=128 by 3.8% (a whole-K MFMA tile there, not the narrow one). The same loop at
    m 256–4096: `mainloop.md` §11.
  - Against cross-block split-K it lost on one kernel: *(field, E59)* at K=35840 (N=7168), cross-block split-K 8 on
    256-wide tiles beat it by 3–8% at M=8 and M=24 with each split folding its share in the launch after polling its
    peers' flags, a wait §4 rules out; with a separate fold launch instead, split-K 8 read 90.4 µs at
    M=24 against 87.8 for the narrow tile with a K stagger (the run's own timer). *(field, E58)* Separately, on an
    N=6144 decode GEMM, `tile_n` 16 and 32 added to aiter's own FlyDSL catalog read 42.39 and 38.45 µs against 30.10
    for its `tile_n` 64 with split-K 2; whether those narrow tiles split K over their waves is not recorded.
  - *(field, E117)* At N=6144, K=6144, with L2 cold, it beat a 4-way cross-workgroup split-K with a fold launch at
    m=32 (16.49 against 18.40 µs; with L2 warm the split had won, 14.42 against 15.63). At m=64 it lost to `torch.matmul`
    (21.78–22.15 against 20.81–20.85 µs) while the split-K beat `torch.matmul` by 7.3%; at m=128 the split-K beat
    `torch.matmul` by 27.4% and the narrow tile was not timed.
- **Splitting the K dimension across warps needs an LDS reduction; splitting N across warps does not.** The
  number of K-warps is per shape and tracks `block_k` (at most `block_k / 32`, where 32 is the K of the bf16 MFMA atom).
  *(field, E60)* Moving warps from K to N cut kernel time 1.0% and raised SNR from 50.8 to 94.8 dB.
- **Race the number of N-waves, odd counts included.** *(field, E61)* 3 N-waves (192 threads) over a 96-wide tile
  beat 2 by 5–8% at M 80–112; 6 waves against 4 moved within ±2%, at different stage counts.
- **At small m, race fewer waves per workgroup** on kernels that split K over their waves. *(field, E118)* At N=4096,
  K=2048, 4 waves instead of 8 were 3.9–4.4% faster at m 8–16 (2 waves 2.6–3.6%); at N=6144, K=1536, 2 waves with no
  K split and A issued first, against 4 waves splitting K 2 ways, read 0.4–1.0% faster at m ≤ 8 in a probe (that
  run's driver moved m=8 between 4.74 and 5.00 µs from run to run), and the final campaign kernel uses it; at N=2048,
  K=2048, 4 instead of 8 won at m=1 and 8 (1.0–1.2% on the driver), while 1 wave lost 2.2–4.2% and 2 waves lost
  1.9% at m=1 and tied at m=8 (its output not bitwise equal). *(field, E119)* At N=512, K=4096 (warm L2, the run's own timer)
  2 waves beat 4 and 8 at m=64 (4.45–4.53 against 5.06 and 6.08 µs) and m=32 (3.49 against 3.66 and 4.22), and
  1 wave lost (3.83 at m=32); at N=2048, K=7168 (split-K 4, one reading per configuration) the best 1- and 2-wave
  configurations beat every 4-wave one at m=16 (7.80 and 7.98 against 8.07 or more; 1 wave with a 4-slot ring read
  8.46) and m=32 (8.55 and 8.66 against 9.20 or more).
- **For a row-major (N, K) B, `block_k` is the DRAM burst length of each row segment.** *(field, E62, E63, E64)* 256 was best
  by 9–40% cold on long-K rows and 512 was worse; at K=2048, 128 won because 8 iterations of 256 made the
  pipeline fill too large a share. *(field, E65)* At K=7168 with LDS capping stages, 256 × 3 stages lost 1.6% to
  128 × 5 at one paired point; race `block_k` together with the stage count it leaves room for.
- **Stages of a ring the waves share: deeper prefetch pays until LDS per workgroup crosses an occupancy step**
  (private rings per wave held at 2–3 stages at M 256–4096, `mainloop.md` §11; the final small-m campaign kernels with
  private rings at N 1024–2048 keep 6–10 slots per wave at m ≤ 32 and 2–6 at m 64–128: race the depth per m).
  *(field, E66, E67)* With 2 stages
  the loop barrier drained `vmcnt(0)` every iteration; 2 → 4 stages was 1.43× at the winning tile. At a fixed
  tile, time jumped where LDS per workgroup crossed 81920 B (2 → 1 workgroup per CU). Sweep stages together
  with the tile; judging a tile at fixed 2 stages misranks it.
- **For a B in its (N, K) layout, race the LDS path against loading straight to VGPRs.** At m ≤ 128 every bypass
  tried lost at N 2048–6144 with K ≥ 1536. Register paths won only in narrower cases: a GEMV at N=1024, K=7168 at
  m ≤ 2 (it lost from m=4), and B in VGPRs at N=28672, K=512, m=32 (at m=64 B through LDS with A in VGPRs won); the
  final register kernels at N 64–512 had no LDS arm timed.
  - LDS: *(field, E68)* on aiter's bf16 HGEMM (N=6144, K=16384, M=8), the B-to-registers path (`b_to_lds=False`)
    read 2.6× slower than the tuned LDS path; that path only builds at 2 stages, and 2 stages alone cost the LDS path
    +55% at the deployed tile.
    *(field, E122)* At N 2048–6144, K 1536–6144, m 1–128, every bypass tried lost: A and B both straight to VGPRs
    1.13–1.70× slower on two kernels; A alone on three, 1.16–1.39× at m 32–128 but only 1.02–1.03× at m=1 and 1.13×
    at m=8 (N=6144, K=1536, deepest register lookahead tried); most of these arms streamed K through register
    stages. So was A at M ≈ 100: A in a 2–3-set VGPR ring was ~1.25× slower than A by LDS DMA *(field, E69)*.
  - VGPRs: the register winners issued their loads up front, before the first use: the GEMV at N=1024 held the whole
    K of A and B, the band kernel at N=28672 held B's whole K with A in LDS. The final campaign kernels at N=64 and 128
    (K 4096–7168, m ≤ 128) and at N=512, m ≤ 32, load A and B so (N=512 at m=64 runs its loads 10 of its 16 K windows
    ahead of the MFMAs; N=128 at m=64–128 splits K over four workgroups), with no same-kernel LDS arm timed.
    *(field, E123)* At N=1024, K=7168 a GEMV with one output column per wave doing so beat a per-wave LDS ring at m=1
    (4.48 against 5.02 µs) and m=2 (4.89 against 5.13), and lost from m=4 (7.32 against 5.13).
    *(field, E124)* At K=512 (N=28672) a band kernel holding each wave's B columns over the whole K in VGPRs, with A
    in LDS, paid +7.3% at m=32 and +10.8% at m=64 over the kernels before it; at m=64 the reverse layout, A rows in
    VGPRs and B through LDS, then beat it (7.29–7.42 against 7.83–7.94 µs), and at m=32 lost to it (7.07–7.08 against 6.41–6.45).
  - A preshuffled B (`b_shuffled` schemas) is loaded straight to VGPRs by FlyDSL's preshuffle kernels (`datapath.md`
    §7). In one 8-wave MXFP4 GEMM (M = N = 4096, K = 8192, not small M) where several waves read the same B, B
    straight to VGPRs reached 77.8% of the throughput of B through LDS (`datapath.md` E22; whether its B was
    preshuffled is not recorded): race it.
- **Compute-deletion probes are diagnostics, not an upper bound on compute's share.** *(field, E70)*
  On an M=8 o_proj (K=16384), deleting the hot-loop `ds_read` + MFMA moved the time less than the spread of its
  baseline; the probe did not keep the loads live or the schedule fixed, so it does not show that the B stream alone
  sets the time. *(field, E120)* Removing the loop's compute from N=6144/K=1536
  reduced the recorded time by 7.7% at m=128 (8.01 → 7.39 µs) and 1.5% at m=1. *(field, E121)* On K=512 kernels (N=28672, the run's own 40-call graph timer) removing the
  MFMAs took a row-owner kernel at m=128 from 10.63 to 8.43 µs and a band kernel at m=64 from 8.76 to 2.87. These
  probes give wrong output by design, and their records do not say whether the loads stayed live (§8).

## 6. Memory-side knobs

- **B's cache policy must be a raced, per-configuration axis.**
  - `nt` on B was −4.4 to −9.5% on long read-once streams *(field, E71)*.
  - On the streamed B of six small-m kernels, `nt` helped at K 4096–6144 (N=6144), was neutral at K 1536–2048
    (N 4096–6144) and hurt at K=512 (N 7168–28672); K, N and the kernel design changed together. *(field, E133)*
    Dropping `nt` from `sc0 nt` cost 3.4–9.4% at m=1, 2, 8 and 32 at K=6144 (N=6144; m=1 9.4%, m=32 5.2%), with a
    K start keyed on `bid >> 3`; before that start all five policies read the same; 2.2% at m=128 at K=4096; at K 1536–2048
    the policies were within 0.7%.
    *(field, E134)* At K=512, `nt` or `sc1` on B cost 11–19% (N=28672, m 1–16), and `nt` on the A and B DMA 17–20%
    (N=7168, m=128).
  - On small or shared B it cost 6–15% *(field, E72, E73, E74)*; `nt` on every A and B fill cost 28–34% at N=2048,
    K=2048, m 64–128 *(field, E135)*.
  - It also lost on a read-once stream: `nt` or `sc1` on an 80 MB B at K=256 cost ~10–20% *(field, E75)*; `sc0` was neutral.
  - It couples with `block_k`: the same switch took one row from 98 to 159 µs at one `block_k` and another from
    153 to 83 µs at a different `block_k` *(field, E76)*.
  - A wants to stay cached: `nt` on A cost 4.5% *(field, E77)*, 2.9% at m=32 (N=6144, K=4096) and 40% at m=64
    (N=512, K=4096) *(field, E136)*. At N=64 (A is larger than B only at m=128), `nt` on A alone read 4.19 against
    4.26 µs at m=128 and 0.4% faster on the mean over all m (a separately compiled copy of the same kernel read 0.16%
    apart in an earlier run); it was not kept *(field, E137)*.
  - So does C: `nt` on the C store cost up to ~2% at M 16–112 (tie at M=8), while the same GEMM's M≈900 tile gained ~2% *(field, E78)*.

  **Not in the current kernels.**
- **Race issuing each stage's A DMA before its B DMA.** *(field, E79)* 0.2–3.5% at M 21–98 on cold B, most at the largest M.
  On a second kernel it was within 0.1%; on a third, with L2 flushed per call, neutral to 1% slower at M 12–96
  (1.5–2.5% faster in a warm replay loop) and ~1% faster only on split-K tiles of ≥ 128 rows *(field, E80, E81)*.
- **Race three K-start forms: none, per block, and grouped.** With all blocks starting at k-tile 0, concurrent reads
  hit the same low address bits. Per block: `k_start = (block · R) mod k_iters`. Grouped: one start per group of
  blocks, `k_start = ((index >> log2 G) · R) mod k_iters`; the groups timed were runs of 8 consecutive blocks
  (`bid >> 3`, one block per XCD) or multiples of 8, the blocks of one XCD, and on one kernel runs of 4, which were
  not kept. On one kernel (N=6144, K=4096) evenly spaced starts beat closer ones at m=1 and 4 but not at m=8
  *(field, E129)*: race the number and spacing of distinct starts too.
  - The tested stagger configurations won on: *(field, E82)* four GLM-5.2 decode shapes, recorded time
    −2.9 to −8.6% against none on cold read-once B; the o_proj −8.6% also changed the band configuration,
    so it is not K_ROT alone. *(field, E87, E108)* On one-M-block `nt` streams at K=7168 and K=16384,
    removing the configured stagger increased time by 10.5–12% at M ≈ 112 on one kernel and by 4–13% across
    M1–128 on another (split-K 1 up to M16), whose HBM read counter read 202–244 MB per call with and without it.
    At M128 the R16 form does not give every workgroup a distinct start, and R1 was slower: race R per M.
    *(field, E127)* At N=1024, K=7168, m 16–64 (28 K-lines per wave, R=1): m=16 6.54 → 6.12 µs, m=32 8.17 → 7.38,
    m=64 9.91 → 8.58, within 1.4% of starts grouped by 8 blocks (6.10, 7.48, 8.54).
  - Per block lost on: *(field, E83, E84, E85, E86)* three kernels where blocks share an operand in L2, or on one
    `nt` B stream: neutral to 1–6% slower (+12% at split-K 4); *(field, E128)* the narrow-tile kernels at N 4096–6144,
    K 2048–6144, where it was timed against a grouped start: it lost or tied. At N=6144, K=4096 (a copy of the
    shipped K=16384 kernel), in runs separate from the reference, it read 10.93 against 9.01 µs at m=1 and 11.21
    against 9.58 at m=16 for starts grouped by 32 blocks; a ladder row that turned it off at m ≤ 16 together with
    longer K blocks was 13–14% faster at m=1, 4 and 16 (2% slower at m=2 and 8). At N=6144, K=6144 it beat none
    (13.23 against 14.41 µs at m=1) and lost to `bid >> 3` by 0.9–2.7% at m=1, 4, 16 and 32 (12.99 at m=1; m=32
    16.52 against 16.09), tying at m=2 and 8; at N=4096, K=2048 a per-block term alone lost ~4% at m 8–16 to the same
    term plus `(band >> 3) · 1`.
  - Grouped starts kept: *(field, E129)* at N=6144, K=4096 grouped starts (G 8–32) paid +6.15% [+6.11, +6.18] on all
    8 cases (m=128 +12.4%); with the strips later numbered XCD-major, so that a group is one XCD, R=4 (sharing
    a factor with 32; its 8 starts evenly spaced) beat R=1 and 3 at m=1 (8.74 against 8.98 and 8.94 µs) and m=4
    (8.93 against 9.26 and 9.27), but not at m=8 (R=1 9.67, R=4 9.72; best G=64, R=7, 9.63). *(field, E130)* At
    N=6144, K=1536 `((band >> 3) · 7) mod 12` at m 17–95 (R=11 at m ≥ 96);
    at N=6144, K=6144 `(bid >> 3) mod k_loops` at m ≤ 32; at N=4096, K=2048 `(s · band + u · (band >> 3)) mod tiles`
    (+2.19% [+2.04, +2.34], all 8 cases); at N=2048, K=2048 a rotation of each wave's K-lines by
    `2 · m_tile + 3 · n_group`, both taken from `pid >> 3` (ladder: +6.25% [+3.63, +8.88]; m=64 +10.9%, m=128
    +15.7%, but the same row read +2.9 to +4.7% at m ≤ 32, where the rotation is off); in that run's races a
    rotation keyed on the workgroups sharing a B tile read 2.6–3.9% slower than the run's canonical kernel and one
    keyed on `pid >> 3` 3.6–3.9% faster.
  - Other keys: *(field, E132)* at N=512, K=4096, m=64 (four M-blocks per N tile), starting the odd M-blocks half-way
    through their K windows read 4.94 µs against 5.07–5.09 for none; keyed on the N tile 5.06–5.08, on the XCD 5.19.
  - Fewer distinct starts: *(field, E88)* with per-block starts, an R sharing a factor with `k_iters` (8 with 64
    iterations) collapsed 192 blocks onto 8 starts and lost 55% against a coprime R (44% against no stagger);
    *(field, E127)* at N=1024 a grouped R of 7 over 28 lines (4 starts) lost to per block (m=16 6.63 against 6.12 µs).
  - At m ≤ 8 it did not pay on two kernels: *(field, E127, E131)* at N=1024, K=7168 no rotation beat the best
    unrotated kernel at m=4 and 8 (m=4 5.27–5.34 against 5.25 µs; m=8 5.86–6.01 against 5.79), though it paid at
    m=16 (above); at N=6144, K=1536 a grouped stagger lost at m=8 (5.12 against 5.04 µs) and also at m=16 (5.18–5.19
    against 5.14).
  - **The stagger is for B; an A that every block of an XCD re-reads wants the opposite.** Where that A is about the
    size of one XCD's L2 (4 MiB), make the start depend only on the XCD (`bid % 8`), so the XCD's blocks walk A
    together, or give each XCD one K split. *(field, E109)* At M=128 (A 4 MiB) per-block starts cost 41% against
    XCD-uniform ones, and won once the A loads were removed; at M=64 (A 2 MiB) per-block starts still won.
  - It interacts with the XCD band below; re-sweep both after any tile change.
  - **Not in the current kernels.**
- **When the grid leaves CUs idle, give them workgroups that load later K-tiles of B into L2** (cached, not `nt`).
  *(field, E89, E90)* Dropping them read ~6% slower at M ≈ 56–98 in single driver runs, within noise in the run's own ablation; they lost at M ≤ 21; race the first prefetched tile per M.
  *(field, E91, E92, E93)* On one kernel (N=7168, K=4224, 33 K-tiles) the best start sat ~60–80% through K at M 24–112,
  a mid-K start was ~30% slower than none at M=64, and on a split-K grid no start paid reliably (the best read 1.1%
  faster at M=112 in one run). *(field, E139)* On another (N=6144, K=1536, 12 K-tiles) the best start moved earlier as
  m grew: 75% through K at m=16 (starts at 50%, 42% and 33% read 5.12, 5.47 and 5.95 against 5.08, each also with a K
  stagger that alone cost ~1% there *(field, E131)*), 50% at m=32, 42% at m=64 and 33% at m=128; there a start at
  58% beat none at m=64 (7.07 against 7.57 µs).
- **XCD placement at small grids.**
  - The hardware deals workgroup i to XCD `i % 8`, so with plain N-fastest order each XCD gets N-tiles 8 apart.
  - A band-cyclic remap (runs of adjacent N-tiles per XCD) removed the bimodal timings in §1 and was −2 to −3% on
    some rows *(field, E94, E95, E96)*, neutral or harmful on others; race it.
  - *(field, E97, E98, E99)* On a read-once B with no reuse between tiles (N=163840, K=256), runs of 8 adjacent tiles
    per XCD beat plain order by 0.6–4.6%; one contiguous slice per XCD, or runs of 80+, lost 2–13% against runs of 8.
  - *(field, E129)* On a strip kernel at N=6144, K=4096, one contiguous slice per XCD lost alone (m=8 10.21 against
    9.87 µs) and paid with the K-start groups numbered by the same XCD-major index (m=8 9.63, m=1 8.74 against
    8.94): race the order together with the K start.
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
  - It was not faster on every kernel: time it per kernel and per M bucket. *(field, E138)* At N=1024, K=7168 it paid
    +2.68% [+2.46, +2.89] with only A's pointer preloaded; at N=2048, K=2048 it paid on the multi-row-tile kernels
    (m=64 5.44 → 5.18 µs, m=128 6.51 → 6.26), was slower at m=16 (4.37 → 4.40) and at m ≤ 8 the driver and an
    in-process run disagreed in sign, so that kernel preloads only the former; at N=7168, K=512 six preloaded
    pointers took m=128 from 4.87 to 4.66 µs and slowed m=16 and 32 (4.21 → 4.24, 4.19 → 4.25), and the campaign's
    final small-M kernel does not preload; at N=3584, K=512 the sign flipped between two runs (m=1 4.23 → 4.12, then
    4.22 → 4.34).
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
- **A row serves every M that pads to it.** By the table above, a table holding only an M=96 row serves M 81–96 and
  nothing else, one holding only an M=16 row every M from 1 to 16, and an M=128 row
  M 65–128. M=72 tries 72, 80 and 128, not 96; M=3 tries 3, 16, 4. *(field, E104)* An arm of four tuned
  rows cost the step 2.6–3.4%; the one row that lost was a padded M=96 row, at M=84. Validate a row
  across every M that reaches it, and add it only if it beats what the pad fallback already gives.
- **Merging model tables keeps the row with the lowest `us`** across files written by different rulers, rewrites
  the source CSVs and raises. Edit the owning file in place.
- **Tuning is compile-bound.** Compile candidates in parallel on the CPU, then time them.
- **A process does not see a table edit after its first lookup.** The lookup reads the file
  `AITER_CONFIG_GEMM_BF16` names or, by default, a merged copy under `/tmp/aiter_configs/` built once per
  process (`AITER_CONFIGS.get_config_file`, `aiter/jit/core.py`, also cached), so an edit to a source CSV shows
  only in a new process. Measured, with `AITER_CONFIG_GEMM_BF16` naming the edited file: the edit showed only
  after both `get_GEMM_A16W16_config_` and `get_GEMM_A16W16_config` were cleared. A captured graph never
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

- E1 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, hot = one weight replayed, cold = traced in the decode step] the same kernel read 30.7 µs replayed and 45.1 µs in the real step, and the ranking flipped (GLM-5.2 decode o_proj (8,6144,16384))
- E2 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=2, decode step, ABBA interleaved] tables tuned on a replayed weight cost the decode step 3.3% (22.35 → 23.15 ms) (GLM-5.2 decode step, TP8 attn-DP8)
- E3 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, aiter run_perftest, automatic vs explicit rotation count R] the automatic count picked 201 copies and read 41.56 µs; every explicit R from 8 to 128 read 15.86–16.14 µs (a GLM-5.2 TP4 decode GEMM with a 48 MB B, not named)
- E4 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, eager host-side call vs graph-timed kernel] aiter's eager call cost 16.7 µs median while its kernels ran 4.6–8 µs (the campaign's 63 small-M shapes, not individually named)
- E5 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, one node] one graph node cost 1.753 µs for a 1-element add_, against 2.983 µs for a 1×2048×2048 GEMM (59% of it the node); an earlier reading in the same campaign 1.766 µs (1-element add_; GEMM 1×2048×2048)
- E6 [2026-08, gfx950, FlyDSL 0.3.2, N=?, graph, empty-node arm in the same 24-node graph] the per-graph-node launch cost measured 1.945 µs (1-element add in the GLM-5.2 decode GEMM graph)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3–9, AKLR bench_cudagraph (L2 flushed) in an in-process harness, median of 3–9 rounds] a fully empty kernel read 4.21 µs at m=8 (gemm_n2112_k7168 forge run)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver times, with a profiler split] a one-workgroup second kernel after the unchanged GEMM took 4.12 µs and took m=7 from 10.48 to 14.43 µs (gemm_n3584_k7168 forge run)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph (L2 flushed) or driver --bench-mode] a separate fold kernel: m=8 11.09 against 8.08 µs, m=32 11.56 against 8.77; the fold kernel alone about 4.5 µs (gemm_n2112_k7168 forge run)
- E10 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, cold R=16 through tuned_gemm.gemm_a16w16 vs a direct kernel call] a row a direct call measured 0.76 µs faster (17.74 → 16.98) was 0.34 µs slower through the entry (17.32 → 17.66); in deployment that row was worth about −4.2 points of score (the difference between the decode-score arms with and without it, E104) (o_proj M=96 row t96x96x128 split-K 4, GLM-5.2 TP4/EP4 decode)
- E11 [2026-09, gfx950, FlyDSL 0.3.2, N=?, decode step vs kernel trace, arm difference] Δ(all kernels) ×3 = +715 µs against Δ(step) = +742 µs, ratio 1.04 (GLM-5.2 decode step)
- E12 [2026-09, gfx950, FlyDSL 0.3.2, N=?, decode step vs kernel trace] of a +742 µs step regression, +280 µs/step (1.2% of score) landed in other kernels (GLM-5.2 decode step)
- E13 [2026-08, gfx950, FlyDSL ?, N=?, cold microbench, uniform vs call-count weighting] summed uniformly over 56 shapes the microbench read +10% for FlyDSL; weighted by the step's call counts −1% (56 decode shapes; GLM-5.2 decode weights 78/78/78/21)
- E14 [2026-09, gfx950, FlyDSL ?, aiter fork 84538c1ae, N=?, whole decode step] the arm carrying 50 rows whose split-K paid only on an idle device ran 5742 µs against 4853 µs, 18% slower (GLM-5.2 decode GEMMs; commit 84538c1ae)
- E15 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=8, graph, hot (one weight replayed)] eight independent allocations of one shape and configuration read 3.73–6.19 µs, bimodally ((4,6144,2048))
- E16 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] per-shape cross-run drift 3–7% (256x6144x2048: 11.27, 12.12, 12.55 µs) while the aggregate bw_ratio reproduced to 0.07% (0.7454 / 0.7459) (the campaign's 63-shape score)
- E17 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=2, graph, hot (one weight replayed), another campaign's processes in the container] the same unchanged tree read 0.8153 and 0.7967 across sessions, recorded as a 1.5% span (the campaign's 63-shape score)
- E18 [2026-08, gfx950, FlyDSL 0.3.2, N=3, graph, cold (B rotated over >= 512 MB), 20 s burn-in] the first FlyDSL arm of a probe read 1–2% slow: 9.398/9.418/9.418 µs as arm 1 against 9.212/9.223/9.227 as the last arm (one decode GEMM configuration, three runs)
- E19 [2026-09, gfx950, torch rowsum, no FlyDSL kernel, N=?, plain read of the same B buffer] the matched HBM ceiling, a torch rowsum over the same B buffer, read 82.06 µs = 5798 GB/s for 476 MB (the 38720-wide weight of a GLM-5.2 decode point)
- E21 [2026-08, gfx950, FlyDSL 0.3.2, N=1, graph, cold (B rotated over >= 512 MB)] split-K 2 35.450 µs (+18%), split-K 4 35.775 (+19%), tile_n 16 35.367 (+17%) against a 30.189/30.072 base (o_proj (8,6144,16384), 192 blocks)
- E22 [2026-09, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB), min of 7 replays] router 6.56 → 3.948 µs (−40%), gate 4.903 → 4.135 µs (−15.6%); both also switch B from nt to cached (MoE router (64,256,6144) shipped t32x64x128 split-K 8 → t16x32x128 split-K 8; gate (8,128,6144) t16x64x64 split-K 12 → t16x32x64 split-K 16)
- E23 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] padding the grid from 246 to 264 workgroups (empty blocks exit at once) took the kernel from 7.14 to 10.70 µs (+50%) (2624x6144, grid_x 82 → 88)
- E24 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, the campaign's direct-call ruler of rounds 1–2, which imported another aiter checkout than the one the scored server loaded] every winner at exactly 256 CTAs, the 288-CTA losers at ratio 1.159; the next round found that ruler had timed the other checkout and re-measured on the served tree (GLM-5.2 TP4/EP4 verify-M GEMM cells)
- E25 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] M=24 on a 38720-wide weight, tile_m 16/32/48/64: 127.71 / 90.42 / 92.36 / 97.23 µs (24,38720)
- E26 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] three or more M-blocks read 407.97 (84,38720 tm16), 268.02 (60,38720 tm16) and 190.87 µs (84,19360 tm16) against the frozen champions' 101.94 / 102.31 / 58.27, 2.6–4.0× off; the champions are the table's recorded us of other kernels, not one-M-block FlyDSL runs (84,38720; 60,38720; 84,19360)
- E27 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] splitting m 65–128 rows into two XCD-paired M-blocks gained +6.65% at m=112, +6.86% at m=96 and +1.66% at m=80 (the same kept change also retiled m=2732) (gemm_n7168_k768)
- E28 [2026-10, gfx950, FlyDSL 0.3.2, N=?, ad hoc harness, flush not stated] two XCD-paired M-blocks (4,4,8,2,2) on the campaign's other (skinny) kernel read 7.409 / 7.603 / 8.105 µs against 6.509 / 7.126 / 7.531 at m=80/96/112 (+13.8%, +6.7%, +7.6%) (gemm_n7168_k768 forge run)
- E29 [2026-10, gfx950, FlyDSL 0.3.2, N=3–6, AKLR bench_cudagraph (L2 flushed), alternating rounds in one process] base/variant time ratio of a 96-wide compute tile storing 72 columns: 0.9864–0.9942 at m ≤ 16, 1.0048–1.0706 at m ≥ 24; base against itself 0.9852–1.0099 (gemm_n18432_k1536 forge run)
- E30 [2026-09, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB), min of 7 replays] the line-cost-optimal router tile t32x64x128 (0.047 vs 0.094 for t16x32x128) was the slowest by 40% (MoE router (64,256,6144))
- E31 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] at tile_n 32, one N-block: split-K 1 34.11 µs, split-K 8 7.24 µs, split-K 12 5.03 µs (1,32,6144)
- E32 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=?, ?] split-K 2 to fill a 0.438-wave grid: +15.5% / +17.6% (kv_b draft GEMM, K=512)
- E33 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] a timing-only plain store in place of zero-C + spin + atomics: split-K 8 4.657 → 2.194 µs, split-K 4 4.180 → 2.664 (the reduction 1.5–2.5 µs); planes + last arriver with one counter per 128 B line: split-K 8 3.854 µs (2048x2048, M=1)
- E34 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] per-tile counters packed as adjacent dwords put a launch on two cache lines and cost 1.1–2.5 µs (2048x2048 split-K reduction)
- E50 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] the same configuration at split-K 12: bf16 partials err_ratio 0.0241 (gate 0.02), f32 partials 0.000000 (a GLM-5.2 decode point, not named in the record)
- E51 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver flushed bench (--bench-case, warmup 3, 20 iterations)] 24-bit packed partials: m=112 29.22 µs against 30.66 (masking alone) and 30.59 (canonical); m ≤ 16 unchanged (22.60/21.83/21.80 vs 22.56/21.83/21.81 µs); output within 1 bf16 ulp of canonical. "Unless the sum cancels" comes from a numpy simulation whose output was not kept (gemm_n8448_k7168 forge run)
- E52 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, ?] unifying the split-K partials' scope with sc0|sc1 (0x1|0x10): 4.2945/4.3467/4.4319 µs, +8.8/+5.1/+4.7% slower (1,32 / 1,128 / 48,32 split-K rows)
- E53 [2026-10, gfx950, FlyDSL 0.3.2, N=5, AKLR bench_cudagraph, median of 5 runs] aux=16 (sc1) on scratch and flag traffic gave timings identical to sc0 sc1 at m=16, 64, 96 and 112 (gemm_n6016_k7168 forge run)
- E54 [2026-10, gfx950, FlyDSL 0.3.2, N=5, bench_cudagraph in process with the incumbent, median of 5] a tile's splits on one XCD with XCD-local scope: 18.34 vs 18.02, 18.73 vs 18.65, 21.25 vs 21.14 µs at m=48/64/96 (0.4–1.8% slower) (gemm_n6016_k7168 forge run)
- E55 [2026-10, gfx950, FlyDSL 0.3.2, N=2, driver --bench-mode single case, two pairs] XCD-local exchange flag off vs on: m=112 29.68 vs 31.53/31.52 µs, m=64 25.98 vs 26.48 (on lost 2–6%); m=32 23.21 vs 23.04, m=16 22.15 vs 21.83 (on gained 0.7%, 1.4%), but the flag-off build was itself slower than canonical (m=16 22.16 vs 21.87, m=32 23.23 vs 22.93) (gemm_n8448_k7168 forge run)
- E56 [2026-10, gfx950, FlyDSL 0.3.2, N=?, 200-call CUDA-graph harness] at m=14: both fences 30.81 µs, release fence only (acquire removed) 29.38 µs, no fences (write-through partial stores + s_waitcnt vmcnt(0), flags polled one at a time) 10.40 µs; the acquire alone, in the spin loop of the fence-free kernel, 11.22 vs 10.40 µs (gemm_n768_k7168 forge run)
- E57 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, ?] racy split-K variants reported err 0.00000 on the fully co-resident grids (1,32 / 1,128 / 48,32)
- E58 [2026-09, gfx950, FlyDSL ?, aiter rev ?, N=?, "clean card", timer not stated] tile_n 16 / 32 added to aiter's FlyDSL catalog read 42.39 / 38.45 µs against 30.10 µs for its existing 64, which at 96 tiles ran split_k=2 (an N=6144 decode GEMM, M and K not stated)
- E59 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver graph-timer medians; the run's own CUDA-graph replay timer for the separate-fold-kernel arm] cross-block split-K 8 on 32×256×64 tiles against the narrow-tile kernel: m=24 81.7 vs 88.7 µs, m=8 80.6 vs 82.9 µs (m=16 not compared), its fold in the launch: each split publishes a launch tag and polls flags, and the split that owns each fragment sums it; in an earlier candidate, with a separate fixed-order fp32 fold kernel, split 8 at m=24 90.4 µs against 87.8 for the narrow-tile kernel with a stagger of 1 tile per band (gemm_n7168_k35840 forge run)
- E60 [2026-09, gfx950, FlyDSL 0.3.2, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3 against late bases] warps split along N instead of K (w1x2x1): 11.257/11.320/11.449 vs 11.385/11.387 µs = −1.0%, SNR 50.8 → 94.8 dB (q_b (8,16384,2048))
- E61 [2026-10, gfx950, FlyDSL 0.3.2, N=?, isolated times, warmup 3, 20 iterations, timer not stated] 3 against 2 N-waves over 96-wide tiles: m=80 19.73 vs 20.68, m=96 21.13 vs 22.66, m=112 22.62 vs 24.65 µs; 6 waves against 4 at m=112 22.27 vs 22.79 µs (different stage counts) (gemm_n6144_k7168 forge run)
- E62 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] raising TILE_K to 256 was worth 9% to 40% cold on the four decode shapes (GLM-5.2 decode GEMMs (8,6144,16384), (8,16384,2048), (8,2624,6144), (8,4096,2048))
- E63 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] TILE_K 512: t16x32x512 split-K 4 44.74 µs against t16x32x256 38.79 µs ((8,6144,16384))
- E64 [2026-08, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB)] STAGES 6 with BLOCK_K 128 beat STAGES 4 with BLOCK_K 256 (q_b (8,16384,2048))
- E65 [2026-10, gfx950, FlyDSL 0.3.2, N=1, single-case driver bench, warmup 3, 20 iterations] block_k 256 × 3 stages against 128 × 5: m=16 44.3 → 45.0 µs (+1.6%, the one paired point); m=8 44.9 µs against an unpaired 43.4–44.9; m=32 did not fit LDS (gemm_n20480_k7168 forge run)
- E66 [2026-08, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB)] with 2 stages the loop barrier drained vmcnt(0) every iteration; 2 → 4 stages at the winning tile 1.43× (o_proj (8,6144,16384))
- E67 [2026-08, gfx950, FlyDSL ?, aiter fork rev ?, N=?, graph, cold] t16x64x128 split-K 4, stages 2–7: 42.65 / 45.79 / 52.57 / 52.32 / 52.97 / 52.81 µs, the cliff where LDS per workgroup crosses 81920 B ((8,6144,16384))
- E68 [2026-08, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB)] b_to_lds=False: 76.528 vs 29.720 µs = 2.58×; the non-b_to_lds path asserts STAGES == 2, and STAGES=2 at the deployed t16x32x256 is +55% (o_proj (8,6144,16384))
- E69 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver single case; a second, local CUDA-event timer] A in a 3-set VGPR ring: 68.46 µs at m=112 and 64.12 at m=98 against 53.35 and 51.32 for A by LDS DMA (1.28×, 1.25×); a 2-set ring on the local timer 1.24–1.26× (gemm_n20480_k7168 forge run)
- E70 [2026-09, gfx950, FlyDSL 0.3.2, N=3, graph, cold (B rotated over >= 512 MB)] dropping the whole hot-loop ds_read + MFMA: 30.850 µs against bases 31.044/30.927/30.745 (o_proj (8,6144,16384))
- E71 [2026-08, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB), ABAB] nt on B: o_proj −9.5%, q_b −6.9%, qkv −4.4%, DSA −5.4% (the four GLM-5.2 decode shapes)
- E72 [2026-09, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB), five-policy palindrome] nt on B: 4.49–4.56 µs with it, 3.955–3.967 without (MoE router (64,256,6144), B 3.1 MB)
- E73 [2026-09, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB)] the M=8 gate row, one m-block, paid 6.3% for nt (4.077 vs 3.822 µs) (gate (8,128,6144))
- E74 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=2, cold R=16 through tuned_gemm.gemm_a16w16, palindrome min-of-2] nt sc0 sc1 on B (cp1): +15.5% / +12.1% slower (kv_b verify cells, GLM-5.2 TP4/EP4)
- E75 [2026-10, gfx950, FlyDSL 0.3.2, N=5–6, AKLR graph timer, arms interleaved in one process] B load cache bits, aux=0 / sc0 / nt / sc1: m=1 13.86 / 13.89 / 15.87 / 15.71 µs, m=8 14.41 / 14.41 / 17.17 / 17.31 µs (gemm_n163840_k256 forge run)
- E76 [2026-09, gfx950, FlyDSL 0.3.2, aiter fork base 5a11152e2, N=?, aiter tuner run_perftest, rotated (cold)] b_cpol=1 alone took 24,38720 at tile_k 64 from 98.04 to 158.68 µs and 10,38720 at tile_k 128 from 152.76 to 83.45 µs (24,38720; 10,38720)
- E77 [2026-08, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB), ABAB] nt on A: 34.874/34.644 vs 33.289 µs = +4.5% (o_proj (8,6144,16384))
- E78 [2026-10, gfx950, FlyDSL 0.3.2, N=3, driver --bench-case, warmup 3, 20 iterations, 2–3 pairs per case] nt (aux=2) on the C store against aux=0: m=8 9.704 vs 9.703 (tie), m=32 11.075 vs 10.954, m=96 15.33 vs 15.12 µs; m=893 49.53–49.57 vs 50.48–50.52 µs (gemm_n7168_k3584 forge run)
- E79 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR bench_cudagraph (1 GiB read flush before each call), 8 input sets, median of 3–4 repeats] A DMA issued before B on top of the nt-B kernel: m=98 17.595 → 16.992 µs (−3.4%), m=21 −0.9%, m=56 −0.4%; at other m −0.038 to −0.173 µs (gemm_n3584_k7168 forge run)
- E80 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver single-case --bench-mode, 3 warmup, 20 iterations] A's DMA before B's: within 0.1% of canonical at m=16, 64 and 112 (gemm_n8448_k7168 forge run)
- E81 [2026-10, gfx950, FlyDSL 0.3.2, N=3–7, AKLR bench_cudagraph (L2 flushed per call); warm = own graph replayed back to back] A before B on split rows: 1.5–2.5% faster warm; flushed, within ±0.3% at m=12–96 except m=80 20.10 vs 19.87 µs; m=893 72.58/72.75 vs 73.45/73.52, m=112 22.93/22.92 vs 23.09/23.06 µs (gemm_n6016_k7168 forge run)
- E82 [2026-08, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB), ABAB] K-start stagger: −8.6 / −2.9 / −3.2 / −5.5% per shape (the four GLM-5.2 decode shapes)
- E83 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph; driver --bench-mode] stagger on split-K rows, variant vs incumbent: m=32 8.88 vs 8.76 µs up to m=112 13.00 vs 12.34 (+1.4% to +5.3%); the driver read m=112 13.31 vs 12.53 (+6.2%) (gemm_n2112_k7168 forge run)
- E84 [2026-10, gfx950, FlyDSL 0.3.2, N=?, timer not stated in that section (the record's other m=893 sections are warm L2)] split-K 4 at m=893: no rotation 46.82 µs, bid%LOOPS 52.55 µs (+12%) (gemm_n2112_k7168 forge run)
- E85 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR bench_cudagraph (flushed), 8 input sets] with B loaded sc0 nt, stagger 0 against stagger 3 on every split-K case: −0.238 to −0.468 µs at m=28–84, +0.020 at m=98 (gemm_n3584_k7168 forge run)
- E86 [2026-10, gfx950, FlyDSL 0.3.2, N=5–11, in-process AKLR bench_cudagraph, arms interleaved, median of 5–11 readings] K-start stagger R=1 / R=5 against off at m=48–112: +0.8% to +1.3% (m=64 7.728 / 7.715 vs 7.640 µs) (gemm_n7168_k1536 forge run)
- E87 [2026-10, gfx950, FlyDSL 0.3.2, N=?, single-case driver bench, warmup 3, 20 iterations] K_STAGGER 0 at m=112: 60.7 µs against 54.1–54.5 for 3, 7 and 9 (+12%); on a fresh cache 58.80 vs 53.22 µs (+10.5%) (gemm_n20480_k7168 forge run: 256 workgroups, one M-block, split-K 1, B sc0 nt)
- E88 [2026-08, gfx950, FlyDSL 0.3.2, N=?, graph, cold (B rotated over >= 512 MB)] stagger stride kr3 30.284 µs, kr8 47.150 µs (gcd(8, 64) ≠ 1 collapses 192 blocks onto 8 starts) (o_proj (8,6144,16384), 64 k-iterations)
- E89 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations, against canonical in the same session] canonical with prefetch off at m=98: 10.04 µs against 9.47/9.49; the run's own ablation of the change was within noise (+0.20% mean, m=98 −3.3%) (gemm_n7168_k1792 forge run)
- E90 [2026-10, gfx950, FlyDSL 0.3.2, N=?, driver single case, shared GPU] prefetch workgroups at m=21: 6.18–6.26 µs against 6.13 without (gemm_n7168_k1792 forge run)
- E91 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations] fastest prefetch start tile per M: 26 at m=24, 27 at m=32 down to 19 at m=112, of 33 K-tiles (33 = K/block_k, computed); off was best at m=8 and 16 (gemm_n7168_k4224 forge run)
- E92 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case, warmup 3, 20 iterations] at m=64, prefetch start 0 (off) 0.016432 ms, start 16 0.021571 ms (+31%) (gemm_n7168_k4224 forge run)
- E93 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode single case; the driver also printed source_rejected for the hooked build] on the split-K grid, base against prefetch starts 13 / 17 / 21: m=64 13.949 vs 25.564 / 16.433 / 13.97, m=112 17.693 vs 25.499 / 21.059 / 17.5 µs (the best start within ~1% of base) (gemm_n7168_k4224 forge run)
- E94 [2026-08, gfx950, FlyDSL ?, aiter fork base f464f2bd1, N=?, graph, hot (one weight replayed)] band-cyclic remap: xb1 mean 4.90 µs (3.89..6.06) → every band 2..24 3.62–3.72 µs, deterministic ((4,6144,2048))
- E95 [2026-08, gfx950, FlyDSL 0.3.2, N=2, graph, cold (B rotated over >= 512 MB)] every banded configuration below both base reads, ~−2.6% ((8,2624,6144))
- E96 [2026-09, gfx950, FlyDSL ?, aiter fork 2c71811b3, N=?, run_perftest rotate_args, palindromic order, same-binary control] xcd_band 2 worth −2.83% against a 0.13% noise floor (o_proj 6144x16384, M=1; commit 2c71811b3)
- E97 [2026-10, gfx950, FlyDSL 0.3.2, N=4–6, AKLR graph timer, arms interleaved in one process] runs of 8 tiles per XCD (g8) against plain order (g1) at m=5–10: 14.55 vs 14.92 µs at m=5 … 14.51 vs 14.94 at m=10 (1.4–2.9%) (gemm_n163840_k256 forge run)
- E98 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR graph timer, arms interleaved in one process] one contiguous slice per XCD: 16.45 vs 16.06 µs at m=14 and 16.92 vs 15.40 µs at m=12 (gemm_n163840_k256 forge run)
- E99 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR graph timer; the g8 base is taken from another line of the same record] runs of 80 and 640 tiles: g=80 14.46/15.04/14.50/14.63 and g=640 15.27/16.30/15.24/15.41 µs against g8 14.45/14.64/14.23/14.39 (gemm_n163840_k256 forge run)
- E100 [2026-09, gfx950, FlyDSL 0.3.2, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3] kernarg preload: (8,6144,16384) 29.640/29.429/29.464 vs 29.970/29.655/29.587 µs (−0.5%); (8,16384,2048) −1.0%, (8,2624,6144) −0.6%, (8,4096,2048) −0.7% (the four GLM-5.2 decode shapes)
- E101 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder row marked recorded (the campaign's paired KEEP timing), driver --bench-mode] kernarg preload: +3.41% mean (lower bound +2.89%), +2.0% at m=98 to +4.3% at m=7 and 35, all 11 cases positive; kernel 5.9–10.8 µs (gemm_n7168_k1792)
- E102 [2026-09, gfx950, FlyDSL 0.3.2, N=3, decode step, min of 4 runs, three scored runs] with preload the step scored 47.0522 / 47.0345 / 47.0057 against 46.8209, about +0.45% (GLM-5.2 decode step)
- E103 [2026-08, gfx950, FlyDSL 0.3.2, N=3, graph, cold (B rotated over >= 512 MB), ABAB×3] fetching only the real A rows (mr8): qkv 8.310/8.310/8.317 vs 8.567/8.435/8.432 µs (−1.5%); q_b 13.642/13.655/13.667 vs 12.585/12.302/12.307 (+11%) (qkv (8,2624,6144), q_b (8,16384,2048))
- E104 [2026-09, gfx950, FlyDSL ?, aiter fork base 99a91e173, N=2, e2e decode bench (c8/c10/c14 tok/s), two palindromes, 8 scored reads] an arm adding four tuned rows, the M=96 o_proj row among them, scored −2.63% and −3.35%; a cold probe through the entry found that row the only loser, at M=84 (o_proj 17.32 → 17.66 µs). The record states, without measuring it, that the row also served the M=72 verify batch (GLM-5.2 TP4/EP4 decode)
- E106 [2026-10, gfx950, FlyDSL 0.3.2, N=?, the campaign's stress test (CU-masked streams of 16/32/64/128 CUs, a second launch of the kernel beside it, eager and graph), outputs checked for NaN and against a reference] the campaign's split-K kernel, whose splits polled flags and planes up to `_MAX_POLLS` and then wrote NaN: up to 4.7 M NaN at m=256 and m=512 and calls of up to 140 ms on 16 CUs; per mask, NaN at m=256 3151872 / 2340864 and at m=512 4276224 / 3158016 on 64 / 128 CUs (STRESS FAIL), and a spin build differed from the call alone at m=256 and m=512 on 16 and 64 CUs; the rewrite in which the last split to arrive at a launch-tagged counter folds: no NaN and no difference under every mask (0.71 ms on 16 CUs); the other K3 split-K GEMM's polled fold was right under the masks but once gave NaN at m=32 while another process shared the GPU (bf16 GEMM n6144 k16384, m=256 and 512)
- E107 [2026-10, gfx950, FlyDSL 0.3.2, N=2–6 interleaved rounds, forge driver bench on AKLR's graph timer, the kernel before the wait-free commit against after] arrival fold against a fold in which every split waited and summed a share of the tile: m=1–16 within 0.1%, m=32 +0.6%, m=64 +1.8%, m=128 37.95 → 44.00 µs (+15.9%) on one GEMM; m=256 0.0551 → 0.0686 ms (+24.3%), m=512 0.0875 → 0.0947 (+8.3%) on the other; a pairwise tree of per-pair counters: m=128 43.79, m=256 0.0702, m=512 0.0959 (bf16 GEMM n6144 k16384, m=1–128 and m=256–512 lanes)
- E108 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, AKLR bench_cudagraph (1 GiB flush per call), one process per set with a fresh FlyDSL cache, two GPUs; placement alone moved a case about ±1 µs between processes] per-block K start `(bid·R) mod loops` against none (B `sc0 nt`): GPU 3 m=1/2/4/8/16/32/64/128 32.98/33.10/31.89/32.71/33.09/33.81/38.28/43.91 against 34.93/36.85/34.81/35.50/36.28/35.99/39.77/48.71 µs; GPU 6 the same sign on every case, 2.4–5.5 µs; HBM read 202–244 MB per call with and without it, request stalls lower with it (bf16 GEMM n6144 k16384)
- E109 [2026-10, gfx950, FlyDSL 0.3.2, N=5 readings per arm, as E108] m=128 (A 128 × 16384 bf16 = 4 MiB, every split of a tile on XCD `bid mod 8`): per-block starts R=1 61.72 µs (R=3 and 5 61.8–63.4 on GPU 6) against R=16, one start per XCD, 43.91; with the A loads removed R=1 34.59 against 37.08; m=64 (A 2 MiB) R=1 38.93 < R=3 39.54 < R=16 42.01 (GPU 6); with each XCD on one K split, R=1 41.96–42.00 against R=16 42.46 (bf16 GEMM n6144 k16384)
- E110 [2026-11, gfx950, Triton 3.8.0 (no FlyDSL kernel), N=4 runs, AKLR bench_cudagraph (Kineto) against the graph's wall clock; a peer's measurement] a CUDA graph of 1–16 chained trivial kernels (each a 64-float load and store): wall-clock slope ~1.77 µs per added kernel; the Kineto ruler's slope 4.05–4.08 µs per added kernel, with and without GPU_FORCE_QUEUE_PROFILING
- E111 [2026-11, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph (L2 flushed before each call), the run's own probe] an empty FlyDSL kernel, grids 1×64 to 448×256: 4.32–4.36 µs; the kernel at m=1: 4.235–4.254 µs (bf16 GEMM n7168 k512, forge run)
- E112 [2026-11, gfx950, FlyDSL 0.3.2, N=2 driver runs per build, forge driver bench (AKLR graph timer, warmup 3, 20 iterations)] an empty FlyDSL kernel (128 threads, no kernarg preload) 4.316–4.335 µs at every m; the kernel at m=1/2/4: 4.206–4.258 (iteration-4 build) and 4.067–4.104 µs (final build) (bf16 GEMM n512 k4096, forge run)
- E113 [2026-11, gfx950, FlyDSL 0.3.2, N=5–11 interleaved rounds, AKLR graph timer (driver), medians against the canonical kernel in the same run] m=64: split-K 4 on launch-tagged slots (8×16 tiles, 512 threads) 4.732 against the canonical whole-K dot kernel 4.940; m=128: split-K 4 (16×16, each XCD a block of tiles with all their splits) 5.299–5.314 against the canonical whole-K MFMA kernel 5.556–5.585; m=32: split-K 4 (8×8, 384/512 threads, placements 1/2/4) 4.264–4.288 against canon 4.243 (bf16 GEMM n128 k6144, forge run)
- E114 [2026-11, gfx950, FlyDSL 0.3.2, N=1 reading per configuration (median of 5), AKLR bench_cudagraph (L2 flushed before every call); m=1 against the driver] m=16: split 4 (1,16,32,4,6) 7.80 against whole-K (4,16,8) 8.49 and 8.64 µs; m=8: 7.71 against whole-K 7.00; m=1: 7.65 against the driver's whole-K 6.65 (bf16 GEMM n2048 k7168, forge run)
- E115 [2026-11, gfx950, FlyDSL 0.3.2, N=5–9 interleaved rounds, AKLR graph timer, medians against canon in the same run] split-K 4, every split of a tile on one XCD: the arm called LOCAL, XCD-local aux 0 / 1 (correct; which accesses it changed is not written) 4.522 / 4.522 µs at m=64 and 5.181 / 5.180 at m=128 against aux 17 (sc0 sc1) 4.427 / 4.968; store at aux 17, loads at aux 0 / 1: m=64 4.424 / 4.418 against canon 4.420, m=128 4.951 / 4.960 against 4.973; aux 0 against 17 on the read-back loads only, m=128: 5.275–5.288 against 5.299–5.314 (bf16 GEMM n128 k6144, forge run)
- E116 [2026-11, gfx950, FlyDSL 0.3.2, N=1 ablation per run, forge ablation ladder (paired, one-sided 95% bounds) on AKLR's graph timer; start = torch.matmul, dispatched library kernel not recorded] first kept round against the start, mean gain [lower bound]: n64 k4096 +84.86% [+80.56]; n2048 k2048 +49.72% [+47.32]; n28672 k512 +25.45% [+24.90] (m ≤ 32 only); n3584 k512 +7.84% [+5.25]; n4096 k2048 +52.60% [+52.48]; n6144 k1536 +48.46% [+48.25]; n6144 k4096 +40.44% [+40.41]; n6144 k6144 +9.01% [+8.99] (m ≤ 16 only); n1024 k7168 +61.63% [+61.56]; n128 k6144 +114.94% [+111.99]; n2048 k7168 +27.73% [+27.59] (m ≤ 32 only); n512 k4096 +29.91% [+29.46]; n64 k7168 +85.70% [+78.60]; n7168 k512 +17.19% [+15.78]; per m, the same first rows read n512 k4096 m=64 −7.68% and n6144 k4096 m=128 −3.78% (full-K MFMA tiles at m=128); the n28672 k512 first round ran (waves, ns, halves, a_direct, packed) (8,1,1,1,1) at m=1, (1,2,2,0,1) at m 2–16 and (2,2,4,0,1) at m=32, wave w owning K slice w % waves (kernel structure from each run's final kernel docstring)
- E117 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the run's own probe with L2 cold, same run; forge ablation ladder for m=64/128] m=32: split-K 4 with a fold launch 18.41 / 18.40 µs against the one-launch strip kernel 16.49 (torch 16.92); warm, the split had won (14.42 against 15.63); the strip at m=64 against torch 21.78–22.15 against 20.81–20.85; the ladder row adding split-K 4 + fold launch where torch had run: m=64 +7.3%, m=128 +27.4% (bf16 GEMM n6144 k6144, forge run)
- E118 [2026-11, gfx950, FlyDSL 0.3.2, N=4–8 readings per arm, in-process probes on AKLR's graph timer / bench_cudagraph; n2048 k2048 also its own 20-call graph harness without L2 flush and the driver] n4096 k2048 (canonical 8 waves; ratio canonical/arm): 4 waves m=8 1.0391, m=16 1.0437; 2 waves m=8 1.0361, m=16 1.0263; n6144 k1536, m ≤ 8 on two waves (no split, A first) against four waves splitting K 2 ways ((1,2,2)): m=1 4.51 against 4.529, m=4 4.648 against 4.68, m=8 4.727 against 4.775; that run's driver read m=8 between 4.74 and 5.00 from run to run; the final campaign kernel's m ≤ 8 row is the two-wave one; n2048 k2048 skw8 4 against 8 waves: m=8 2.503 against 2.677 (no flush), driver m=1 4.273 against 4.325, m=8 4.442 against 4.487; 2 waves m=1 +1.9%, m=8 −0.2% (output not bitwise equal), 1 wave m=1 +2.2%, m=8 +4.2%
- E119 [2026-11, gfx950, FlyDSL 0.3.2, N=? (n512: best of 5–7 replays; n2048: median of 5, one reading per configuration), n512: the run's standalone graph harness with L2 warm; n2048: AKLR bench_cudagraph, L2 flushed] n512 k4096 tiles (16,4,8), R=4, at m=64: 8 waves 6.08, 4 waves 5.06, 2 waves 4.45–4.53 µs; tiles (8,4,8), R=4, at m=32: 4.22 / 3.66 / 3.49, 1 wave 3.83; n2048 k7168 split 4: m=16 (1,16,32,4,6) 7.80, (2,16,32,4,4) 7.98, (1,16,32,4,4) 8.46 against (4,16,32,4,3) 8.07 and (4,16,32,4,4) 8.09, every 4-wave configuration 8.07–10.16; m=32 (1,32,32,4,6) 8.55, (2,32,32,4,6) 8.66 against (4,32,32,4,2/3/4) 9.20–9.40, every 4-wave configuration 9.20–11.16
- E120 [2026-11, gfx950, FlyDSL 0.3.2, N=4–6 alternating rounds, bench_cudagraph probe in one process against canonical] loop compute removed (anchor probe, wrong output; whether the loads stayed live is not recorded): m=128 8.010 → 7.392, m=64 6.409 → 6.173, m=32 5.403 → 5.292, m=1 4.514 → 4.445 µs (bf16 GEMM n6144 k1536, forge run)
- E121 [2026-11, gfx950, FlyDSL 0.3.2, N=10 replays, the run's own CUDA-graph script (40 calls per replay over 8 B tensors, median of 10 replays; up to ~0.3 µs between reruns)] band (7,1), wrong output by design: unmodified m=64 8.759, m=128 11.864; no MFMA loop 2.865 / 4.390; row-owner kernel at m=128: full 10.626, no MFMA 8.425 µs (bf16 GEMM n28672 k512, forge run)
- E122 [2026-11, gfx950, FlyDSL 0.3.2, N=4–9 readings or rounds per arm, in-process probes on AKLR's graph timer or bench_cudagraph (n6144 k6144: same-process Kineto, cold), each against the LDS kernel in the same run] A and B straight to VGPRs: n4096 k2048, kernel without LDS rings, canonical/arm 0.5866–0.8839 at m=1, 16, 32; n6144 k4096 register path at m=128 29.82 and 29.94 against 18.05 µs; A alone: n2048 k2048 +19.4% to +35.2% at m=64–128; n6144 k1536 at m=128 10.69–11.53 against DMA-path controls 8.07–8.27, at m=1 4.61 (lookahead 8) against 4.49/4.53 and at m=8 5.58 (lookahead 4) against 4.94; n6144 k6144 at m=32 18.58 against 16.08; K streamed through register stages in four of the five (n2048: A per line behind vmcnt waits; n6144 k1536: lookahead 2–8; n6144 k6144: a 3-deep ring; n6144 k4096: 5 or 8 stages), the n4096 record does not say
- E123 [2026-11, gfx950, FlyDSL 0.3.2, N=?, AKLR bench_cudagraph (L2 flushed before every call), sweeps] m=1: GEMV (A and B buffer-loaded to registers, one output column per wave) 4.48 against the per-wave LDS-ring kernel skw(4,4,8) 5.02; m=2: multi-row GEMV 4.89 against 5.13; m=4: 7.32 against 5.13 (bf16 GEMM n1024 k7168, forge run)
- E124 [2026-11, gfx950, FlyDSL 0.3.2, N=1 ablation (forge ladder); the run's own CUDA-graph script for the cross-kernel times] band kernel (each wave's B columns over the whole K in VGPRs, A staged once in LDS): m=32 +7.3% over the skinny split-K kernel, m=64 +10.8% over torch; row-owner kernel (A rows in VGPRs, B through LDS) against the band: m=64 7.287 / 7.418 against 7.830 / 7.937, m=32 7.068 / 7.084 against 6.447 / 6.414 µs (bf16 GEMM n28672 k512, forge run)
- E125 [2026-11, gfx950, FlyDSL 0.3.2, N=7 readings per configuration, AKLR bench_cudagraph (L2 flushed before each call), the run's own harness] m=16: two 8-row M-blocks, 8-column tiles (W=4, R=6) 6.540 against one 16-row M-block, 4-column tiles (4,4,8) 7.61; m=32: four 8-row M-blocks, 16-column tiles (R=6) 8.205 against two 16-row M-blocks, 8-column tiles (4,8,6) 8.167, 256 workgroups both; m=8: (4,8,6,16,8,(1,1)), one 8-row M-block, 8-column tiles, K-start rotation, 128 workgroups, 6.635 against (4,4,6) on 256 workgroups 5.793 µs; grid = ceil(m / rows per M-block) × N / tile columns (bf16 GEMM n1024 k7168, forge run)
- E126 [2026-11, gfx950, FlyDSL 0.3.2, N=? (best of 5–7 replays), the run's standalone graph harness, L2 warm] 8 waves, R=8: m=64 tiles (16,4,8) 5.43 against (32,2,8) 7.97; m=32 (8,4,8) 4.00 against (16,2,8) 5.27 µs; (TM, M-blocks, TN), grid = 512/TN × M-blocks (bf16 GEMM n512 k4096, forge run)
- E127 [2026-11, gfx950, FlyDSL 0.3.2, N=7 readings per configuration, AKLR bench_cudagraph (L2 flushed before each call), the run's own harness] K-line start of each workgroup through the DMA soffset, 28 lines per wave: per block (pid · 1) mod 28: m=16 6.115, m=32 7.376, m=64 8.579 against none 6.538 / 8.166 / 9.909; grouped ((pid // 8) · R) mod 28: m=16 6.099, m=32 7.479, m=64 8.542 (R=13); (7, 8): m=16 6.625, m=32 8.241; m=4: every rotation 5.27–5.34 against 5.246; m=8: rotations 5.86–6.01 against the best unrotated configuration 5.793 (incumbent 5.985); the final campaign kernel uses R=1 at m 16–64 (bf16 GEMM n1024 k7168, forge run)
- E128 [2026-11, gfx950, FlyDSL 0.3.2, N=1–2 runs, forge driver bench (n6144 k4096; fresh FlyDSL cache), same-process paired graph timing (n6144 k6144), AKLR graph timer ratios canonical/arm (n4096 k2048); one ablation ladder] n6144 k4096 (strip kernel copied from the shipped K=16384 kernel): G=1/R=1 m=1 10.93, m=16 11.21 (each sweep in its own process) against the reference run's 9.01 / 9.58 for G=32; the ladder row turning the per-strip stagger off at m ≤ 16 (with longer K blocks for m ≤ 64 in the same row): m=1 +14.2%, m=4 +13.6%, m=16 +13.2%, m=2 −2.3%, m=8 −2.0%; n6144 k6144 m=1: none 14.41, bid mod K 13.23, bid >> 3 12.99; bid mod K against bid >> 3: m=2 13.68 / 13.69, m=4 13.24 / 13.08, m=8 13.07 / 13.06, m=16 13.56 / 13.44, m=32: 16.52 against 16.09 µs; n4096 k2048 per-block term alone, (3,0) against (3,1): 0.9621 at m=16, 0.9634 at m=8
- E129 [2026-11, gfx950, FlyDSL 0.3.2, N=1 ablation; probes: median of 15 readings, AKLR graph timer, FlyDSL cache cleared per arm] ladder row adding grouped starts ((index >> log2 G) · R) & (loops − 1): +6.15% [+6.11, +6.18], 8 up / 0 down, m=128 +12.4%; XCD-major strips with the K-start groups still on block index at m=8: 10.212 against 9.87; both XCD-major: m=8 (G=32) R=4 9.72, R=1 9.666, G=64 R=7 9.632, m=1 (G=32) R=4 8.737, R=1 8.975, R=3 8.936 against 8.936; m=4 R=4 8.926, R=1 9.264, R=3 9.266 against 9.045 µs (bf16 GEMM n6144 k4096, forge run)
- E130 [2026-11, gfx950, FlyDSL 0.3.2, N=1 ablation per run, forge ablation ladder; final campaign kernel source] n6144 k1536: start_tile ((band // 8) · stagger) % tiles, stagger 7 at m 17–95, 11 at m ≥ 96 (kept with an earlier L2 prefetch: +3.22% [+3.01, +3.44]); n6144 k6144: kstart = (bid >> 3) % K_LOOPS (kept with a retuned table: +6.89% [+6.84, +6.94]); n4096 k2048: (band · s0 + (band >> 3) · s1) & (tiles − 1): +2.19% [+2.04, +2.34], 8 up / 0 down; n2048 k2048: K-line rotation 2 · m_tile + 3 · n_group: +6.25% [+3.63, +8.88], m=64 +10.9%, m=128 +15.7% (the row also read +2.9 to +4.7% at m ≤ 32, where the rotation is off); there, keyed on the B-tile group +3.9% / +2.6% slower and on pid >> 3 3.9% / 3.6% faster than that run's canonical kernel at m=128 / 64 (the record does not say which canonical these races used)
- E131 [2026-11, gfx950, FlyDSL 0.3.2, N=7 readings per configuration, bench_cudagraph via the run's config probe, interleaved] m=16: canonical row with stagger 5 / 7 5.19 / 5.18 against 5.14; m=8: stagger 7 with no prefetch 5.12 against 5.04 µs (bf16 GEMM n6144 k1536, forge run)
- E132 [2026-11, gfx950, FlyDSL 0.3.2, N=1–3 runs, forge driver timer (warmup 3, 20 iterations), offsets kept in the VGPR address] m=64, four M-blocks: none 5.074 / 5.087 / 5.085; (m_blk & 1) · 8 windows 4.936 / 4.936 / 4.938; n_tile % 16 5.059 / 5.077; XCD-aware 5.186 µs (bf16 GEMM n512 k4096, forge run)
- E133 [2026-11, gfx950, FlyDSL 0.3.2, N=2–9 readings or rounds, in-process probes: same-process Kineto, cold (k6144), AKLR graph timer (k4096, k2048), bench_cudagraph (k1536)] dropping nt from sc0 nt on B: n6144 k6144 m=1/2/8/32 13.98 / 14.1 / 14.4 / 17.34 against 12.78 / 13.6 / 13.92 / 16.48 (with K start bid >> 3); before that start, at (512,3,4), five policies 15.64–15.66 / 16.07–16.09; n6144 k4096 m=128: sc0 18.45, none 18.44, nt 18.05, against sc0 nt 18.05; n4096 k2048: "", sc1, sc0 sc1 against nt, ratios 0.993–1.005 at m=1, 16, 32, 64, 128; n6144 k1536 m=128: "" 8.005, sc1 8.015 against nt 7.961
- E134 [2026-11, gfx950, FlyDSL 0.3.2, N=?, forge driver graph timer (n28672); AKLR bench_cudagraph, L2 flushed, 3–6 readings (n7168)] n28672 k512 skinny-kernel B-load aux 2, 3, 16, 17, 18 against 0: m=1 7.03–7.16 against 6.315, m=2 7.36–7.39 against 6.649, m=8 7.51–7.57 against 6.387, m=16 7.71–7.74 against 6.570; n7168 k512 m=128 A and B DMA bits nt 5.99–6.07, sc0 nt 6.00–6.09 against sc0 5.06–5.10 µs
- E135 [2026-11, gfx950, FlyDSL 0.3.2, N=5 rounds, in-process AKLR bench_cudagraph] nt on every fill load (A and B DMA): +33.6% at m=128, +28.2% at m=64 (bf16 GEMM n2048 k2048, forge run)
- E136 [2026-11, gfx950, FlyDSL 0.3.2, N=2–4, AKLR graph timer (n6144, median of 9, twice); one process, 4 alternating rounds of 10 graph readings (n512)] n6144 k4096 m=32 A policy nt 11.27 against none 10.95; n512 k4096 m=64 a_aux 2 (nt) 6.87, b_aux 2 5.18, against 4.914–4.921 µs
- E137 [2026-11, gfx950, FlyDSL 0.3.2, N=6 rounds, AKLR bench_cudagraph, rounds shuffled in one process] nt on A only / B only / both, means over 8 m: 4.2117 / 4.2494 / 4.2589 against 4.2271; in an earlier run of the same iteration a separately compiled copy of c0 4.2277 against 4.2208; m=128 A only 4.189 against 4.258 µs; the round was rejected at the keep gate (bf16 GEMM n64 k4096, forge run)
- E138 [2026-11, gfx950, FlyDSL 0.3.2, N=1 ablation (n1024); in-process A/B on bench_cudagraph, 4–6 rounds (n2048, n7168, n3584)] n1024 k7168: +2.68% [+2.46, +2.89], 7 up / 1 down, kernarg_preload_length 2 (A's pointer only); n2048 k2048 no preload → preload: m=64 5.443 → 5.176, m=128 6.508 → 6.255, m=16 4.367 → 4.395; skw8 at m ≤ 8 driver 4.45–4.47 against 4.29, in-process 4.226 against 4.246; n7168 k512 pointer arguments, preload 0 → 6: m=16 4.206 → 4.242, m=32 4.189 → 4.248, m=128 4.872 → 4.660; n3584 k512 m=1 4.228 → 4.123 in run 1, 4.216 → 4.337 in run 2 (n7168 k512 final kernel, no `amdgpu-kernarg-preload-count`)
- E139 [2026-11, gfx950, FlyDSL 0.3.2, N=7 readings per configuration, bench_cudagraph via the run's config probe, interleaved; 12 K-tiles of 128] first prefetched tile: m=64 (stagger 5) pf 6 / 5 / 4 / 3 / 8: 6.72 / 6.49 / 6.67 / 6.98 / 7.00; pf 7 (stagger 0) 7.07 against off 7.57; m=128 (stagger 11) pf 6 / 5 / 4 / 3 / 2: 8.81 / 8.62 / 8.37 / 8.47 / 8.67; m=32 (stagger 7) pf 10 / 8 / 7 / 6 / 5 / 4: 5.77 / 5.69 / 5.56 / 5.50 / 5.61 / 5.82; m=16: pf 9 row 5.08, pf 6 / 5 / 4 (stagger 7) 5.12 / 5.47 / 5.95; final kernel: off at m ≤ 8, pf 9 / 6 / 5 / 4 at m 9–16 / 17–32 / 33–95 / ≥ 96 (the pf 9 row is the m=32 row timed at m=16) (bf16 GEMM n6144 k1536, forge run)
