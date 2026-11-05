---
title: "Grouped GEMM — variable-K wgrad"
kind: topic
lever: grouped_wgrad
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [wgrad, variable-k, skew, split-k, beta-accumulate]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp8_kernel.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py
---

# Variable-K wgrad

> The weight gradient of a grouped GEMM is `out[g] = A_gᵀ · B_g`, where `A_g` and `B_g` are the rows
> `[offs[g], offs[g+1])` of `A [M_total, OUT_M]` and `B [M_total, OUT_N]`. Every group has the same number of
> output tiles; the contraction length of a tile is its group's token count. This card covers what that changes:
> the loop, the K tail, MX scale timing, dispatch order, short and deep groups, split-K, beta=1 accumulation and
> the output layout. "FP8", "BF16", "MX8", "MX4" are Primus-Turbo's `grouped_gemm_{fp8,bf16,mxfp8,mxfp4}_kernel.py`
> under `primus_turbo/flydsl/grouped_gemm/` at main `ca1fdd67`. FlyDSL facts are v0.3.2.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end. Use
> them for direction and order of magnitude, not as targets.

## Route here when
- wgrad is the slowest pass of a grouped backward, or the only one that degrades under token skew.
- You are writing a K loop whose trip count is a run-time per-group value.
- One deep group, many tiny groups or empty groups dominate a skewed launch.
- wgrad accumulates into an existing gradient buffer (beta=1), or its output layout differs from its operands'.

**Somewhere else if:** XCD order and longest-first across all passes → `grouped_scheduling.md`; tile and
boundary bodies → `grouped_tile_pricing.md`; scale packing and quantization → `grouped_lowprec.md`;
race and cache mechanics → `grouped_host_autotune.md`; gates → `grouped_correctness.md`.

## 1. Why the backward's gap sits in wgrad
- Forward and dgrad parallelize over M: total M is fixed, so skew moves tile counts, not the cost of a tile.
  wgrad is the one pass whose per-tile cost follows the token distribution. *(field, E1)* With forward and dgrad
  byte-identical between two builds and robust to skew (skewed/balanced 0.92–0.99), a tensorwise fp8 wgrad ran a
  one-hot distribution at 0.34 (down projection; 0.66 on gate_up) of its balanced speed.
- *(field, E2)* On MXFP8 at 32768 total tokens, wgrad took 1.16–1.35× the time of the dense wgrad with the same FLOPs
  (+200–270 µs), the largest structural gap of the backward. Memory stalls were ≈0, occupancy, VGPR and LDS were
  the same (one workgroup per CU at 128 KB LDS); wall-averaged MFMA utilization was 45.5% against 61%. The gap is
  MFMA idle time at group boundaries (per-group prologue and epilogue, pipeline restarts), not bandwidth.
- Layout. The token axis is the contraction axis, so the tensorwise/bf16 operands `[M_total, OUT]` are K-row on
  both sides and both go through transpose reads, which cost more LDS instructions per MAC than K-contiguous
  reads (`../gemm/datapath.md` §9). MX wgrad instead takes column-wise quantized operands
  `[OUT, M_total]`, because the E8M0 blocks run along the tokens; its loads are K-contiguous (MX8:1459-1460), at
  the price of §3.

## 2. A K loop with a run-time trip count
Two forms are in use.
- **Runtime chunk loop.** An outer loop over chunks with a run-time trip count, each chunk the unrolled
  distance-2, 4-buffer pipeline body. Keep CHUNK even so the cur/next buffers are back in their starting roles at
  every chunk boundary (MX8:1428 asserts it; BF16 uses CHUNK=4, BF16:277-343; FP8's masked kernel 8,
  FP8:2264-2300). The remainder either runs on zeros from the descriptor (§3) or, in MX8, as an unrolled tail whose
  steps sit under a wave-uniform `k_abs < k_iters` guard (MX8:1574-1615). *(field, E3)* Chunk sizes 8, 4 and 2 were
  all neutral on a tensorwise masked wgrad: rounding up to a chunk is not where skew costs.
- **Whole loop with a run-time count.** One inline-asm region (`../gemm/mainloop.md` §3) takes the
  trip count in an SGPR. FP8's 4-wave wgrad runs the largest multiple of 6 K blocks (one pass of its pool
  depths) and the 0–5 remainder as a tail fused into the same asm block; a group shorter than one pass runs only
  the tail (`nval_can_be_zero`, FP8:2797-2798, 3434-3437). MX4 passes the raw 256-block count (MX4:1887).

In FlyDSL: `../flydsl/moe.md` § grouped_wgrad §2.

## 3. K tails: let the descriptor zero them, except where it cannot
- **Token-outer operands (tensorwise, bf16).** Fold `m_start · OUT` into the 64-bit descriptor base and bound
  `num_records` to `max(M_g · OUT − tile_offset, 0)`: reads past the group's last row return zero, so the last
  chunk or tail needs no predicate (`_wgrad_rebase`, FP8:2072-2085; FP8:3443-3451). The clamp at zero is what
  keeps an empty group legal (`grouped_group_mapping.md` §7–§8).
- **Token-inner operands (MX, `[OUT, M_total]`).** A record count bounds the whole row band, not a row's token
  range, so a tile that runs past its group reads the next group's real tokens and accumulates them. Pad each
  group to the K block (128 tokens in MX8) so the step count is exact (MX8:1481), and guard the last chunk at
  run time (`k_abs < k_iters`). Only an unbalanced gate sees a missing guard: tile-aligned groups never over-run
  (`grouped_correctness.md` §1).
- Over-reading past K is legal only when every term it adds is exactly zero: `../gemm/mainloop.md` §6.

## 4. MX: the lever is when the scales are loaded
- *(field, E4)* MXFP8 variable-K wgrad loaded the A/B scales in the same step as the MFMAs that use them, which put
  the scale `ds_read` latency on the MFMA path. A change that prefetched them one step ahead, as the dense NT kernel
  does, took four shapes from 930/1047/530/307 to 851/936/478/274 µs (about 9%). The change also reworked how the
  accumulators are held, which alone gained nothing; the prefetch gave about all of it (`_wgrad_ssa_chunk`, MX8:166).
- Scale loads must be few and wide. *(field, E5)* A whole-loop MXFP8 wgrad that fetched scales with 16 `buffer_load_ubyte`
  per phase into 16 scattered VGPRs ran 811 µs against 479 µs for the chunked kernel. From an 807 µs reading,
  removing those loads alone gave 448 µs; cutting the scale registers from 16 to 2 gave 506 µs. Byte loads are
  issue-bound and prefetching does not hide them. Pack four E8M0 per dword, per group: `grouped_lowprec.md` §3.
- The scaled MFMA instruction itself is free (`grouped_lowprec.md` §4).

In FlyDSL: `../flydsl/moe.md` § grouped_wgrad §4.

## 5. Dispatch order: wgrad tiles are not equivalent
In NT every tile of a launch costs the same and skew changes how many tiles a group has. In wgrad every group has
the same tiles and skew changes what each costs. The mechanics of XCD orders are in
`grouped_scheduling.md`; what follows is specific to wgrad.
- **Never slice the tile list contiguously across XCDs.** That hands whole groups to one XCD
  (`grouped_scheduling.md` §1).
- **Keep the hardware round-robin (`bid % 8`, `num_xcd = 1`)**, which spreads every group evenly over the XCDs,
  and take L2 reuse from a `group_m`/`group_n` band inside a group. Turbo's tensorwise candidates are all
  `num_xcd = 1` ("xcd=1 keeps group-major LPT order", FP8:4654-4656); its XCD-affine variant keeps the hardware
  split and sizes the super-block to one group "so every XCD holds the same fraction of every group's K"
  (FP8:3897-3902). *(field, E6)* On a tensorwise wgrad at G=32, `num_xcd=2` was 6.94% faster on a balanced load and
  22.07% slower under skew; `num_xcd=8` +3.72% and −42.84%.
- **Group-major order is longest-first only if the token histogram decreases with the group index.** Choose the
  walk direction or rank the groups on the device from the group table (`grouped_scheduling.md` §5).
- **Group-interleaved band order** (band b of every group, then band b+1) is a different mechanism from the XCD
  band-cyclic remap. It helped under skew at the hardware XCD order (`num_xcd = 1`) and caused the skew collapse
  on top of a contiguous XCD remap (`grouped_scheduling.md` §3).
- **Do not copy NT dispatch parameters or defaults to wgrad** (`grouped_scheduling.md` §1). Each path gets
  its own candidates and constants.

## 6. Short contractions turn store-bound
- The output `[G, OUT_M, OUT_N]` has a size set by G alone: an 8-token group writes as many bytes as an
  8192-token one. *(field, E7)* MXFP4 wgrad, G=32, constant total tokens: from balanced to a 159:1 skew, the ratio to a
  large-M reference fell from 0.94 to 0.22 (~3800 → ~870 TF/s); forward and dgrad were immune.
- Switch blocking for short contractions and store C non-temporally (`_GMXFP4_WGRAD_CFG_SHORT`, MX4:1124).
  *(field, E8)* gate_up at 159:1 skew 916 → 3969 TF/s, at 36:1 1288 → 4002, at 10:1 2133 → 3971; the reference cell
  rose too.
- **The crossover scales with tiles per group.** *(field, E9)* Sweeping 2048–32768 tokens per group at G=32, the
  short/regular crossover was 24576–32768 for a 276-tile-per-group shape and 12288–16384 for a 144-tile one: about
  85–120 tokens per output tile in both (one reading per point; the sweep brackets the crossover). Gate on
  `M_total / G > c · tiles_per_group`, not on a token count (MX4 uses a fitted `_GMXFP4_WGRAD_SHORT_MG = 8192`,
  MX4:1152, selector MX4:2083-2091).
- **The skew tax that remains.** *(field, E10)* Tensorwise wgrad, G=32, 131072 tokens (4096 per expert on average),
  one expert with 97–99.8% of them, split-K active: the main kernel still paid +159 µs (gate_up) and +114 µs
  (down) over balanced; skewed time stayed 1.13–1.20× the balanced time. It comes from 31 eight-token groups
  that each compute a full output, plus one group that eight pieces cannot spread over 256 CUs.

## 7. Two K-loop bodies: choose by per-group M, not by M_total
- *(field, E11)* Tensorwise wgrad with a masked chunked body and an exact-trip body (2-stage prefetch, runs exactly
  `M_g`): at `M_g = 512` the exact body was 60% faster (856 → 1369 TF/s); it won by 3–20% at `M_g ≤ 1024`, the
  two were within ±4% between 1024 and 2048, and masked won by 5–15% at `M_g ≥ 2048`.
- Gate on `M_total / G`. *(field, E12)* A gate on `M_total` sent G=8, `M_g`=512 (`M_total` 4096) to the masked body,
  where the exact body was 13% faster.
- A single whole-loop body with a run-time count and a fused tail (§2) removes the choice; FP8 uses one 4-wave body
  for every length and keeps the masked kernel as the fallback for 64-bit traversal and for a
  candidate that fails to build.

## 8. Four waves beat eight on per-tensor fp8 wgrad
- *(field, E13)* Tensorwise fp8 grouped, 28 points (4 token counts × 7 shapes): the 4-wave whole loop (256 threads, one
  wave per SIMD, 256×256 tile, AGPR accumulators) won 28/28 on wgrad (+3–17%), 6/28 on dgrad and 1/28 on the
  forward NT. At 1024 tokens per group it still beat the exact-trip 8-wave body on 5 of 7 shapes and the masked
  one on 7 of 7: keep short contractions in the 4-wave race.
- Why: both operands are transpose-read, and a 128×128 block per wave issues fewer reads per MFMA. *(field, E14)* 2647
  against 2204 TF/s on one shape, the 8-wave kernel at two waves per SIMD; bank conflicts were 2.4% and memory
  stalls 0.05%: the limit is transpose-read feed per MFMA, which more waves on one LDS read port make worse.
- Flat or negative on that kernel *(field, E15)*: more waves per SIMD, smaller tiles, store-defer (−18.6% once it
  cost LDS), plain `ds_read_b64` instead of transpose reads (0), wider epilogue stores (−2 to −7.7%), cross-tile
  overlap (−5%), epilogue double buffering (−1 to −3.5%), swizzle × XCD × `vmcnt` sweeps (<2%), *(field, E16)* persistent vs
  one tile per workgroup on balanced loads (±2%; under heavy skew one tile per workgroup won,
  `grouped_scheduling.md` §6).
- What paid *(field, E17)*: both B pools three buffers deep instead of only one of them, +2.1–3.8%. The partial drain
  must count the in-flight fills of every deep pool (`../gemm/correctness.md` §4). Also the wave
  bank swizzle (`../gemm/datapath.md` §2) and the lgkm-side partial drain
  (`../gemm/schedule_and_locality.md` §4).

## 9. Deep groups: deterministic split-K
Under skew one group's contraction outlasts the rest and the launch ends with one window running while the other
CUs idle. The device is full, so the dense wave-quantization gate (`../gemm/grid_pricing.md` §2) never
fires; the trigger is the deepest group. Determinism by fixed-order reduce is the same idea as
the in-launch split-K reduce of `../gemm/grid_pricing.md` §2 and of
`../attention/flydsl_attn_backward.md` §1.
- **Decide on the device, from `group_offs`.** FP8 fixes the piece count from the static shape (tiles per group
  against CUs; `_wgrad_split_geom`, FP8:3688-3704) and cuts a group only if its own contraction clears a bar
  (`_wgrad_is_cut`, FP8:3747). A variant chose one window and its slice count S at run time. *(field, E18)* That
  variant gained 7.57% on a six-cell score (two projections × balanced, moderate and heavy skew) of a tensorwise
  wgrad; the dispatch layer it added had to win its host cost back (`grouped_host_autotune.md` §2).
- **Layout.** The piece that starts at K=0 writes C; every later piece writes a scratch band at C's pitch, so the
  store only swaps the descriptor base (FP8:3425-3433). Swapping only the base keeps C's cache bits on the band;
  give the band its own: *(field, E19, E20)* on a dense split-K, `nt` partials cost 3–6% and `sc1` paid when another XCD
  reduced them (`../gemm/epilogue.md` §5). A reduce kernel launched by the same host launcher, on
  the same stream, folds the bands into C in ascending chunk order, in fp32, with no atomics: bit-reproducible
  (`kernel_grouped_tn_wgrad_reduce`, FP8:4450; launch FP8:4617). Its grid is the compile-time tile space, and a
  launch that cut nothing exits at once. *(field, E21)* ~4 µs on a balanced load, 46–75 µs when it folds.
- **Dense split-K forms worth racing here:** the reduce in the same launch, done by the last split to arrive on a
  per-tile counter, never by a workgroup that waits for another (`../gemm/grid_pricing.md` §2); counters
  in per-call scratch, zeroed before the GEMM launch, and fp32 partials packed to 24 bits
  (`../gemm/small_m.md` §4); and each XCD on a contiguous run of one split's tiles
  (`../gemm/schedule_and_locality.md` §2).
- **Scratch for CUDA graphs.** Keep it persistent per (shape, device) and size it for the deepest geometry any
  candidate can pick, so replay sees fixed addresses (`_wgrad_split_ws`, FP8:3758-3775).
- **What it buys.** *(field, E22)* One expert with 99.8% of the tokens: gate_up 3024 → 2389 µs (21% less time), down 1381
  → 1298 µs (6%); balanced unchanged (down) or 2.7% faster (gate_up). SNR of the weight gradient (BGrad) 52.8 dB
  with the reduce live.
- **Every output format keeps the split.** Tight C and an fp32 accumulator need bands at C's pitch and dtype and a
  reduce vector the store can issue: 8 fp32 lanes are 256 bits, beyond `buffer_store_dwordx4`, so FP8 halves the
  vector for fp32 (FP8:3989).
- **Granularity.** *(field, E23)* Eight pieces were best; 16 were slower (down 1289 → 1428 µs); piece counts that are not
  powers of two break the power-of-two chunk geometry. Eight reduce workgroups per tile instead of four: ~0.3% of
  profiler-isolated time; *(field, E31)* folding them back to four read flat on a score over random token distributions.
- **A run-time divisor needs full coverage.** When the slice count is chosen at run time and the division uses a
  precomputed reciprocal table, the table must cover every S the policy can reach, and its exactness check must use
  the true dividend range. Assert the coverage at compile time; the gate that catches a gap is in
  `grouped_correctness.md` §9.

In FlyDSL: `../flydsl/moe.md` § grouped_wgrad §9.

## 10. beta = 1: accumulating into the gradient buffer
- **The read-back is a physical cost and scales with the accumulator dtype.** *(field, E24)* Tensorwise 4-wave wgrad,
  one square tight-C shape, time beta=0 → beta=1: fp32 0.842 → 1.020 ms (21% more time), bf16 0.854 → 0.933 ms
  (9% more), fp16 0.847 → 0.907 ms (7% more). An fp32 accumulator is read at twice the bytes of a 16-bit one.
- **Do not move the read-back into the main loop at one wave per SIMD.** One quadrant's read-back is 64 fp32
  VGPRs held across the loop. *(field, E25)* Hoisting one and then two quadrants raised the fp32 read-back cost from
  21% to 115% and 177% more time, bf16 from 9% to 37% and 142%: spills. Pipeline it in the epilogue
  (`../gemm/epilogue.md` §7).
- **Fused still beats beta=0 plus a separate add.** *(field, E24, E26)* The add alone took 0.79 ms (fp32) and 0.27 ms (bf16),
  against 0.18 ms and 0.08 ms of read-back inside the fused kernel.
- **Ranking: two measured facts.** *(field, E27)* Timing every candidate at both betas on a tensorwise wgrad (two
  projections × G = 24 and G = 32), the beta=0 winner differed from the beta=1 winner in two of four cases, both
  at G = 24 (1.1% and 0.5% lost at beta=1). A race at beta=0 picked the worse config in 4 of 6 runs on one case;
  ranking at the caller's beta picked the better one 6 of 6, at one extra compile per surviving candidate (cold
  autotune about doubled). The other side: ranking at the caller's beta lets a beta=0 caller and a beta=1 caller
  run different configurations of one shape. Which a race uses is the owner's choice. FP8 ranks at the caller's
  beta and keeps its numeric gate at beta=0 (FP8:4934-4945); race mechanics are in
  `grouped_host_autotune.md` §6.
- beta can change code paths beyond the epilogue: the empty-group zero pass exists only at beta=0 (§11).

## 11. Empty groups
- Store zeros and skip the fetch pipeline. BF16 runs its prologue only when `group_tokens > 0` and then stores the
  zero accumulators (BF16:256-275); in MX8 a zero step count leaves the loop empty and the epilogue stores zeros.
- With the split active at beta=0, FP8 stops dispatching empty groups (FP8:4408) and zero-fills them flat in the
  reduce pass (`_zero_fill`, FP8:4558; `_ZERO = _SPLIT and not beta_is_one`, FP8:4001). As a GEMM tile body the
  zero store has no MFMA to hide under; Turbo's note puts that burst at ~0.27 µs/MB against a 0.17 µs/MB write
  roofline (FP8:3996-4000).
- At beta=1 an empty group's output is left as it was, so FP8 keeps its tiles in the GEMM. Descriptor clamps for
  empty groups: `grouped_group_mapping.md` §8; gates: `grouped_correctness.md`.

## 12. Output layout
- **Decouple C's pitch from the operands'.** Operands stay padded for the fill addressing; C gets the real
  extents (FP8's `m_real` / `n_real` / `c_tight`, FP8:3851-3853). Every C addressing site moves together: the
  store's row pitch, its row bound `(g + 1) · C_M`, the split band row, the reduce's element offset and column
  predicate, the C and scratch descriptors, the host allocation of output and scratch, and the cache key. A
  fallback that can only write the padded pitch must refuse a tight C (FP8:4826-4828). *(field, E28)* Missing one
  site, the split band row, gave a relative error of 1.42 and non-reproducible output.
- **Pad free dims in the producer.** A free dim that is 64 mod 128 elements leaves every other 1-byte operand row
  straddling two 128 B lines (`../gemm/datapath.md` §10) and the last output tile ragged. Padding
  that dim in the quantizer and slicing the result is byte-correct for any pad value, since the pad lands in
  sliced-off output columns, and lets the compile prove the per-column store bound dead (`_COL_SAFE`,
  FP8:3949). *(field, E29)* +3%.
- **`trans_c` is a different store path.** *(field, E30)* bf16 grouped wgrad writing `[G, N, K]` through the
  transposed store (BF16:350-369) instead of `[G, K, N]`: 1293 → 1010 TF/s (22% less throughput) on gate_up; on
  down the transposed store matched a vendor library that ran both layouts at one speed. Benchmark the layout
  each caller requests.

## Verify
- The gate includes skewed, one-hot and empty-group distributions and a large-M shape that forces the deepest
  split branch, compared with an fp32 reference, not only bitwise run-to-run.
- Dispatch order and `num_xcd` were judged on skewed cells as well as balanced ones.
- The MX last chunk is guarded.
- beta=1 was timed at the accumulator dtype the caller uses, with the read-back in the epilogue.
- A tight-C or fp32 output reaches the same split path as the default output.

In FlyDSL: `../flydsl/moe.md` § grouped_wgrad Verify.

## Related
`grouped_scheduling.md` · `grouped_tile_pricing.md` · `grouped_lowprec.md` ·
`grouped_host_autotune.md` · `grouped_correctness.md` · `grouped_measurement.md` ·
`../gemm/mainloop.md` · `../gemm/grid_pricing.md` · `../gemm/epilogue.md` ·
`../flydsl/debug-flydsl-kernel.md` · `../attention/flydsl_attn_backward.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. "Turbo" is Primus-Turbo, and `Turbo <rev>` the base, HEAD or frozen commit the record names. A line whose
only reference is a myskill or memory note gives the month of that note ("recorded by"); such notes rarely state N.
Secondary references in a line sit in the directory of its first `local:` path unless they carry their own.

- E1 [2026-08, gfx950, FlyDSL ?, Turbo 527e62c7 (frozen build), N=?, skewed/balanced throughput ratio per op, same-machine A/B of two builds through the public entry, smci355 GPU4] forward and dgrad byte-identical between the two builds and robust to skew (0.92–0.99); the frozen tensorwise fp8 wgrad ran a one-hot distribution at 0.336 (down) and 0.656 (gate_up) of its balanced throughput (gpt-oss-20b, G=32, M_total=131072, H=I=2880; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_wgrad_skew_campaign.md:84-86)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, per-component times dense vs grouped at total_M=32768; rocprofv3 PMC] MXFP8 grouped wgrad 1.16–1.35× the time of the dense wgrad (+200–270 µs absolute); var-K vs dense NT wgrad at 4096×7168: MemStall ≈0, same occupancy (LDS 128 KB, one workgroup per CU), VGPR and LDS nearly equal, MFMA utilization 45.5% vs 61% (Turbo MXFP8 grouped GEMM, MoE training shapes, not named per cell; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:8; :11)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, ?] chunk size 8 → 4 → 2 on the masked chunked wgrad: all neutral (Turbo grouped fp8 tensorwise TN wgrad, masked body; shapes not named; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:130)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, ?] MXFP8 variable-K wgrad with chunk-local SSA accumulators and per-step scale prefetch: 930/1047/530/307 → 851/936/478/274 µs (~9%); SSA accumulators alone 0, scale prefetch alone ≈ all of it (Turbo MXFP8 grouped wgrad, four shapes not named; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:15-16)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, cache cleared] whole-loop (4-wave bare-asm) MXFP8 grouped wgrad with scales: 811 µs (another reading 817) against 479 µs for the chunked production kernel; without the 16 `buffer_load_ubyte` per phase 807 → 448 µs; scale VGPRs 16 → 2: 807 → 506 µs (B=8, M=4096, N×K=4096×4096; the whole-loop kernel was deleted on 2026-07-05 and never committed; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:196-201; /workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:30)
- E6 [2026-08, gfx950, FlyDSL ?, N=?, in-process interleaved throughput ratio deploy (G=32, 4096 tokens per expert) / large-M reference (G=4), same M_total; single-variable arms, byte-identical output] gate_up `num_xcd=2` +6.94% on the balanced load (ratio 0.8922 → 0.9542), `num_xcd=8` +3.72%; the same arms −22.07% / −42.84% on the skew load (gpt-oss-20b tensorwise fp8 grouped wgrad, gate_up N=5760; campaign 20260810_104512, optimize round 1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_104512/goal.md:308-310)
- E7 [2026-08, gfx950, FlyDSL ?, Turbo a8079ca7, N=3, median of 3 bench repeats, cache cleared, spread 0.8%; deploy/large-M throughput ratio interleaved in one process] MXFP4 grouped wgrad ratio balanced .94, canon (10×) .52, moderate (36×) .31, heavy (159×) .22 (gate_up); deploy TF/s ~3800 balanced to ~870 heavy while the reference held ~3800–4100; forward and dgrad .83–1.01 (gpt-oss-20b MXFP4 grouped GEMM, G=32, M_total=131072; campaign 20260811_114814 baseline; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/memory.md:57; :62; :73)
- E8 [2026-08, gfx950, FlyDSL ?, Turbo a8079ca7 base, N=3, bench repeats (per-cell TF/s from the run log)] short-contraction wgrad config (XCD de-interleave off, N band, `nt` C store) vs the baseline: gate_up heavy 915.8 → 3968.7 TF/s, moderate 1287.5 → 4001.5, canon 2133.0 → 3970.8; the large-M reference cell 4137.5 → 4363.2 (gpt-oss-20b MXFP4 grouped wgrad, G=32; campaign 20260811_114814 rounds 0 and 2, 2026-08-11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/run.jsonl:4; run.jsonl:8; /workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:1396-1397)
- E9 [2026-08, gfx950, FlyDSL ?, N=1, each per-group token count measured once per forced config (SHORT / SPAN / CFG)] G=32, 2048–32768 tokens per group: the SHORT → regular crossover at 24576–32768 for gate_up (276 tiles per group) and 12288–16384 for down (144 tiles per group); the upper ends are ≈119 and ≈114 tokens per tile (gpt-oss-20b MXFP4 grouped wgrad; recorded 2026-08-12, myskill 8360f19; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:1400-1417)
- E10 [2026-08, gfx950, FlyDSL ?, Turbo 1d66cc27 plus a local split-K fix, N=?, profiler-isolated wgrad kernel time] one expert holding 97–99.8% of the tokens, split-K on: the main kernel still paid +159 µs (gate_up) and +114 µs (down) over balanced; skewed/balanced time 1.127× (gate_up) and 1.201× (down); attributed to 31 eight-token groups each computing a full output (gpt-oss-20b tensorwise fp8 grouped wgrad, G=32, M_total=131072, K=2880; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_wgrad_skew_padk.md:28; :74-83; :89)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, ?] masked chunked body vs exact-trip body (2-stage prefetch): M_g=512 856 → 1369 TF/s (+60%); exact body +3–20% at M_g ≤ 1024, ±4% between 1024 and 2048, masked +5–15% at M_g ≥ 2048 (Turbo grouped fp8 tensorwise wgrad; shapes not named; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/08-splitk-crosslane-lowprec-wgrad.md:51; :56-58)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] the old gate `m_total <= 2048` sent G=8, M_g=512, m_total=4096 to the masked body; the exact-trip body measured +13% there (Turbo grouped fp8 tensorwise wgrad; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/06-autotune-dispatch.md:25)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, direct-compile harness (`_ow.py`) and probes, per-shape TF/s] 4-wave vs 8-wave over 28 points (per-group m 1024/2048/4096/8192 × 7 production shapes): wgrad 28/28 (+3–17%), dgrad 6/28, forward NT 1/28; at m=1024 the 4-wave beat the persistent 8-wave on 5/7 shapes and the masked 8-wave on 7/7 (Turbo grouped fp8 tensorwise, branch feat/kyle/grouped-wgrad-4wave; 2026-07-05; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_grouped_nn_4wave_build.md:16-26)
- E14 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC on repeated launches (`_op.py`)] tensorwise wgrad 4-wave vs 8-wave (two waves per SIMD): 2647 vs 2204 TF/s, MfmaUtil 60% vs 46%; 4-wave LDS bank conflicts 2.4%, MemUnitStalled 0.05% (deepseek up projection, m=4096, OUT 7168×4096, G=8; 2026-07-01, chi2810 `/opt/venv-tw`; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_occ_feed_bound.md:3; :13-21)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, ?] on the 4-wave tensorwise wgrad: store-defer ×2 neutral, ×3 −18.6%; plain `ds_read_b64` instead of `ds_read_b64_tr_b8` 0; epilogue write128 / tr16 −2 to −7.7%; cross-tile overlap −5%; epilogue double buffering −1 to −3.5%; swizzle × xcd × vmcnt on the worst shape <2% (Turbo grouped fp8 tensorwise TN wgrad, 4 waves; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:103-113; /workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_occ_feed_bound.md:98; :111)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, `_ow.py` with persistence gated off, cold compile, best of 3 autotune configs per arm] persistent vs one tile per workgroup, balanced loads, 7 shapes × m2048/m8192: within ±2% (e.g. deepseek-up m8192 2903 vs 2959, gpt_oss-up m2048 2113 vs 2067 TF/s) (Turbo grouped fp8 tensorwise 4-wave wgrad; 2026-07-02, chi2810 GPU0; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_occ_feed_bound.md:149-166)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, `_ow.py` full sweep, best per shape, same GPU] both B pools three buffers deep against the production build with one: +2.1–3.8% on all 7 shapes at m2048 (Turbo grouped fp8 tensorwise 4-wave wgrad; recorded by 2026-07-14, myskill 241cb37; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_occ_feed_bound.md:1402-1415)
- E18 [2026-08, gfx950, FlyDSL ?, Turbo bc5e4fdf, N=3, raw scores of 3 repeats; in-process interleaved ratio wgrad / frozen forward (`_bench_wgrad_vs_fwd.py`), gm over 6 configs] runtime single-window split-K with a narrow reduce: +7.57% on the score over gate_up/down × balanced/moderate/heavy; same-day base 0.9113 → 0.9970 (gpt-oss tensorwise fp8 grouped wgrad, 4 local experts; campaign 20260803_111756 round 3, chi2798 `mlperf_gptoss` `/opt/venv-syncv3`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_111756/memory.md:119-120)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=4, own A/B races on AKLR bench_cudagraph (rep=20) over AKLR's 8 timing input sets, median; HEAD against HEAD 0.996–1.006] dense split-K plane stores with aux 18/19 (`nt`): 0.947–0.967 of the incumbent aux 17 (dense bf16 GEMM gemm_n144_k7168 forge run, iter 10; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n144_k7168/forge_experiments/lessons/iter_010.md:50)
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder, 3 pairs, driver --bench-mode] dense split-K workspace partials stored with `sc1` (aux=16), K splits spread across XCDs: m=35–98 +2.0% to +3.6% (dense bf16 GEMM gemm_n768_k7168 forge run, kept change 23cee34; AKLR 2a4328229; local:/workspace/code/gpt_oss_docker/work/sikl/kb_evidence/sources/distill_12h/ablation.json:228)
- E21 [2026-08, gfx950, FlyDSL ?, N=?, profiler-isolated kernel times] the reduce kernel ~4 µs on balanced loads (nothing cut); 46 µs (down, extreme skew), 70 µs (gate_up, extreme), 75 µs (down, four hot experts) when it folds (gpt-oss-20b tensorwise fp8 grouped wgrad, G=32, 4096 tokens per expert, K=2880, tight fp32 C; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_wgrad_skew_padk.md:121)
- E22 [2026-08, gfx950, FlyDSL ?, Turbo 1d66cc27 plus the split-K fix, N=?, profiler-isolated wgrad time, unfixed and fixed builds] split-K enabled for tight / fp32 C, one expert with 99.8% of the tokens: gate_up 3024.4 → 2389.4 µs (−21.0%), down 1381.2 → 1298.0 µs (−6.0%); balanced down 1088.7 → 1081.3 µs, gate_up 2180.2 → 2120.8 µs (−2.7%); BGrad SNR 52.80 dB with the reduce firing (gpt-oss-20b tensorwise fp8 grouped wgrad, G=32, 4096 tokens per expert; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_wgrad_skew_padk.md:61; :74-83)
- E23 [2026-08, gfx950, FlyDSL ?, N=?, profiler-isolated, extreme skew] split pieces NB=8 best; NB=16 slower (down 1289 → 1428 µs, gate_up 2370 → 2503 µs); NB=12/24 broke (power-of-two geometry required); `_WGRAD_RED_WPT` 4 → 8 about 0.3% in total (reduce down 91 → 80 µs, gate_up 81 → 76 µs) (same workload as E22; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_wgrad_skew_padk.md:87-88)
- E24 [2026-08, gfx950, FlyDSL ?, N=50, min of 50, tight N=K=2880, GPU4] tensorwise 4-wave wgrad time beta=0 → beta=1: fp32 0.842 → 1.020 ms (+21.15%), bf16 0.854 → 0.933 ms (+9.15%), fp16 0.847 → 0.907 ms (+7.07%) (gpt-oss-20b down projection, padK-on-main branch, syncv3; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_fused_accum_beta1_perf_equals_beta0.md:20)
- E25 [2026-08, gfx950, FlyDSL ?, N=?, env-gated A/B (`PT_RB_HOIST=0/1/2`), JIT cache cleared per arm] beta=1 read-back hoisted into the main loop for one and two quadrants: fp32 penalty +21% → +115% → +177%, bf16 +9% → +37% → +142% (same shape as E24; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_fused_accum_beta1_perf_equals_beta0.md:22)
- E26 [2026-08, gfx950, FlyDSL ?, N=?, same harness as E24] host `main_grad.add_` alone 0.79 ms (fp32) / 0.27 ms (bf16); unfused total 1.71 / 1.14 ms; fused faster by 13–37% (same shape as E24; local:/workspace/code/gpt_oss_docker/.claude/memory/feedback_fused_accum_beta1_perf_equals_beta0.md:20)
- E27 [2026-09, gfx950, FlyDSL ?, N=6, six race runs on the down G=24 case; every candidate timed at both betas (TF/s)] beta=0 winner vs beta=1 winner: down G=24 (1,4) vs (4,2), 1.1% lost at beta=1; gate_up G=24 (1,4) vs (1,3), 0.5% lost; G=32 the same winner; the beta=0 race chose the worse config in 4 of 6 runs, ranking at the caller's beta 6 of 6; cold autotune about doubled (~87 s per race before) (gpt-oss-20b tensorwise fp8 grouped wgrad, B=24 campaign branch; local:/workspace/code/gpt_oss_docker/.claude/memory/project_b24_pad2944_merge.md:96-104; :120; :132)
- E28 [2026-08, gfx950, FlyDSL ?, N=4, relative error against the reference; determinism over 4 runs] tight-C wgrad with the split-K `band_row` still at `OUT_M`: rel 1.42, det 0/4 (gpt-oss-20b down projection, tensorwise fp8 grouped wgrad; campaign 20260817_065611 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/memory.md:157)
- E29 [2026-08, gfx950, FlyDSL ?, N=?, campaign score (wgrad / forward ratio, gm)] the quantizer pads the wgrad's 1-byte operand free dim 2880 → 2944 and the output is sliced back: +3% gm (gpt-oss tensorwise fp8 grouped wgrad, 4 local experts; built into the baseline of campaign 20260803_111756; local:/workspace/code/gpt_oss_docker/.claude/memory/project_wgrad_reach_fwd_campaign.md:27-31; /workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_111756/goal.md:685)
- E30 [2026-08, gfx950, FlyDSL ?, Turbo 840a0801, N=?, TF/s geomean over 2 shapes × 4 distributions, second of two passes] bf16 grouped wgrad with `trans_c=True` ([G,N,K]) vs False: gate_up 1293 → 1010 TF/s (−22%); a vendor library (hipBLASLt) 1275 / 1258 for the two layouts; on down the transposed path was at parity with it (1.01–1.04) (gpt-oss shapes, branch kyle/dev/bf16_grouped_gemm; local:/workspace/code/gpt_oss_docker/.claude/memory/project_bf16_grouped_vs_hipblaslt.md:41-42; :53)
- E31 [2026-08, gfx950, FlyDSL ?, Turbo 1d66cc27 (frozen ruler), N=1, one bench run of the arm against the kept arm's mean of 4 runs; in-process interleaved TF ratio against the frozen kernel, gm over random in-sample token distributions × {gate_up, down}, JIT cache cleared] folding `_WGRAD_RED_WPT` 8 → 4 on top of the kept boundary-body change: gm 1.0227 against 1.0234 (flat, reverted) (gpt-oss-20b tensorwise fp8 grouped wgrad, G=32, 131072 tokens; campaign 20260824_154607 round 5, `/opt/venv-syncv3`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260824_154607/memory.md:233; goal.md:478)
