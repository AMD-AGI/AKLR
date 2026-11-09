---
title: "Attention gradients — dQ/dK/dV accumulation, determinism, workspace, masking and fusion"
kind: topic
lever: attn_gradients
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [dQ dK dV, split-k workspace, causal, sliding-window, gqa]
flydsl: 0.3.2
---

# Attention gradients: dQ, dK and dV

> This card is what measured dQ/dK/dV kernels taught: a KV-outer dK/dV body that emits dQ as split-K partials,
> a Q-outer dq kernel, a delta (`dO·O`) kernel, and the reduce that folds the partials.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You decide how dQ, dK and dV are accumulated, and whether the result must be bit-reproducible.
- The dQ workspace, its reduce, or an atomic alternative dominates the time of the dQ/dK/dV kernels.
- You consider fusing dq with dK/dV, fusing the delta kernel, or splitting a kernel.
- A causal or windowed dQ/dK/dV kernel is slower than its FLOPs suggest.

**Somewhere else if:** grid order, XCD placement, GQA merging → `grid_dispatch.md`;
occupancy and spills → `registers_occupancy.md`; gates → `correctness.md`.

## 1. Determinism by construction
- One workgroup owns each output tile, and no float atomic touches an output.
- Reductions across workgroups go through a split-K workspace (one slot per split) summed in fp32 in a
  fixed order (ascending band). The kernel is then bitwise reproducible from run to run on any grid; a different
  band or split partition gives different, still deterministic, bits.
- Determinism also closes "relax it to drop MFMAs". Whether a non-deterministic path is acceptable is the
  owner's decision, not a performance call (§3).

## 2. The D/A law: workspace bytes are set by the register file
Let `A` be the accumulator floats one workgroup holds.
- KV-outer (workgroup holds dK/dV, dQ leaves as partials): `A = BLOCK_KV · D · 2`; partial bytes
  `(Skv / BLOCK_KV) · |dQ| / 2 = B·Sq·Skv·Hq·D² / A`.
- Q-outer (workgroup holds dQ, dK/dV leave as partials): `A = BLOCK_M · G · D` (G = GQA sharers covered);
  partial bytes are the same `B·Sq·Skv·Hq·D² / A`, and dK/dV partials usually need fp32, so ×2.
- So partial bytes per FLOP are `D / A`: no loop order removes them, KV-outer is the cheaper side, and at a
  fixed register file they grow linearly with head dim.
- Only three moves change `A`: a larger accumulator (the 512-register pool is the hardware limit), another
  storage level (LDS is usually already spent: at D128 and `BLOCK_KV = 256` dV alone is 128 KB), or recomputation
  (*(field, E1)* splitting dV across sibling workgroups made each recompute S/dP: +20% MFMA; it read 0.97 ms slower,
  a gross cost near 1.6 ms against a byte saving near 0.6 ms).
- *(field, E2)* Measured on a fused body: the D128 compute phase took no more than 2× the D64 one (4.18 ms
  against 2 × 2.12 ms); the D128 excess over 2× D64 sat in the dQ workspace path: partial store 1.82 vs 1.11 ms, reduce
  0.63 vs 0.47 ms, 0.88 ms of a 0.96 ms gap.

## 3. Atomic dQ
- **fp32 atomics on dQ lose by construction.** Each dQ element receives ~`Skv / (2·BLOCK_KV)` contributions (32 at Skv
  = 8192, BLOCK_KV = 128). fp32 read-modify-write moves 2× the bytes of bf16 partials plus their fold, and the MFMA C
  layout gives one lane 8 floats of one q row, so a wave touches 32 cache lines per instruction. *(field, E3)* 105.5 ms
  against 6.34 ms, to replace a 1.229 ms split-K round trip. Packing a wave's lanes onto 4 cache lines measured
  slower still, on a probe that kept the band-indexed workspace; a larger `BLOCK_KV` (fewer contributions) was 17%
  (D128) and 65% (D64) slower.
- **Packed bf16 atomics (`buffer_atomic_pk_add_bf16`) into a band-less dQ image** remove the workspace and
  the fold. Costs and conditions:
  - dQ is no longer bit-reproducible (dK/dV still are). *(field, E4)* ~1.1 dB lower dQ SNR. This changes
    behaviour for every caller, so it needs the owner's consent; keep it behind an explicit
    `deterministic=False` route with the split-K path as the deterministic one.
  - The 64 lanes of one instruction must cover 256 contiguous bytes (two 128 B lines, four 64 B write
    requests). *(field, E5)* Issued with the store's address layout, one instruction spread over 16 such 64 B
    segments, and its atomics cost about 10× as much (10.5 ms against ~1.0–1.1 ms with coalesced addresses; whole
    backward 13.3 ms).
  - Issue it with the default cache policy. *(field, E6)* With SC0 set it lost three of four updates.
  - Price the exposed fold first. *(field, E7, E8)* D128: fold 0.414 ms = 12.9%, atomics together with the 256-row
    band they made affordable +2.11%, and they made a
    later +12.29% change possible; *(field, E9)* on another kernel the atomic path took D64 962.9 → 1023.8 and D128
    863.5 → 1008.8 TF/s and cut the dQ workspace from 4.8 to 0.8 GiB. Where the fold is already hidden
    (an idempotent store overlapped with the body), atomics are a pure loss.
- **dK/dV atomic epilogue loses.** The MFMA C layout spreads a wave's lanes over many lines, so each atomic
  degrades to per-line read-modify-write. *(field, E10)* D128 0.419 → 0.554 ms (32% slower), D64 12% slower, in the same
  kernel where coalesced dQ atomics won. Gate an atomic on "does one wave's address set coalesce", not on
  dispatches saved.

## 4. Pricing the workspace round trip
- **Subtract in layers.** A "skip the partial store" probe also lets LLVM delete the GEMM that fed it; run
  the traffic-only, store-and-consumer and next-stage arms together (`attribution.md` §5).
- **Price bytes with an alias probe, not a roofline.** *(field, E11)* Aliasing all bands onto one slab saved
  0.071 ms/GB against a roofline 0.161 ms/GB: the roofline overstated the gain 2.3×.
- **Time an auxiliary kernel alone** (`measurement.md` §9). *(field, E12)* Widening a fold's tile
  (+10.7–12.9% in the kernel) moved wall an estimated 0.07–0.14%.
- **Re-price the fold after every body win.** *(field, E13)* Removing it (diagnostic) cut 4.9% (0.289 ms), and 5.84%
  (0.333 ms) after the body got 4% faster with the same bytes.
- **Each q-split carries a fixed cost.** *(field, E14)* Fixing three chunks, a 1/8 tail and a 1/4 tail both cost
  8%: the tail was ≤ 0.1%, the rest was doubled dK/dV slots and per-band K/V staging.
- **A workspace capacity cap is not free.** *(field, E15)* Capping at 3 GiB instead of 16 gave 876 against 990 TF/s
  (D64) and cost D128 22%: grouping bands to fit the cap trades throughput 1:1 for capacity. Check that a
  tighter cap does not switch off pair folding (§5).

## 5. Workspace layout and residency
- **Interleave bands by concurrent writer.** Descending q order makes many bands write the same q rows at
  once; in `[band][B][Sq][Hq][D]` those writes land on as many DRAM pages. `[band/ILV][B][Sq][Hq][band%ILV][D]`
  puts them on one page. D128 (64 concurrent bands) +1.3%, flat from ILV = 4 *(field, E16)*; D64 (8 bands, page
  already full) flat *(field, E17)*. A head-major workspace `[band][B][Hq][Sq][D]` was −5.8% *(field, E16)*.
- **Opposite cache policies on the two ends.** Read the partials non-temporal (they are dead after the
  read; caching them evicts the body's Q/dO); write them cached when one store covers less than a 128 B
  line (two 64 B stores merge in L2). Judge by DRAM transactions (UMC counters), not by rate. *(field, E18)*
  NT writes added ~1.3 GB of DRAM traffic and were 0.45% slower; cached reads cost 0.1 J per pass with DRAM
  traffic unchanged.
- **MALL residency is a step.** *(field, E19)* Clamping the fold's read footprint: 268 MB 7.66 J, 1.07 GB 8.10 J,
  4.29 GB 8.13 J. Above ~1 GB everything prices as a full miss; only a footprint within the 256 MB MALL
  drops. A slightly smaller working set buys nothing.
- **Size every buffer slice under 4 GiB.** `num_records` is 32-bit; a larger slice silently drops the
  out-of-range writes. *(field, E20)* An interleave sweep at ILV = 64 built an 8.6 GB slice and read a fake +18%.
  See `../gemm/datapath.md` §8.
- **Causal pair folding.** Under causal, band group `g` never writes the first `g/ng` of its slab, and group
  `ng − g` needs exactly that many rows, so the two share one slab: `ng → ng/2 + 1` slabs, no extra bytes,
  the minimum for this layout (`Sq · (ng + 1) / 2`). Remove the unpaired branch at compile time.
- **Window: size the workspace as a ring** of `ceil(W / BLOCK_KV) + 1` bands. It is a capacity lever, not
  a speed one. *(field, E21)* 16 → 1.25 GiB, and the two kernels read 1.4% faster.

## 6. Reducing in the producer: the last-arriving workgroup
- Folding the partials in the last dK/dV workgroup to finish a q tile, instead of in a trailing reduce kernel,
  was correct but lost: *(field, E22)* on a D128 backward 1.5–1.8× the time of the separate fold (4.67–5.89
  against 3.19 ms). The separate reduce wins on workgroup count: its thousands of small workgroups keep
  thousands of reads in flight, where a few fat body workgroups in their epilogue keep tens.
- If built, keep it deterministic: an integer counter (`global_atomic_add_u32`) with a release, no float atomics,
  and the sum taken in ascending band order regardless of who arrives last.
- `s_waitcnt` covers only the issuing wave. After a multi-wave store phase, the order is `s_waitcnt(0)` in
  every wave → workgroup barrier → the one atomic announcement.
- With few query rows per launch (as in decode) the same shape also serializes the reduce; see
  `split_kv.md` §3.

## 7. Fusion
- **Drop removable GEMMs first.** *(field, E23)* Removing a redundant GEMM and the global rho/R renormalization
  from dq: +9.8% (accuracy cost ~0.2 dB).
- **Fuse the delta `rowsum(dO·O)` into a kernel that already holds dO.** *(field, E24)* A FlyDSL delta kernel in
  place of framework ops, measured together with an `s_setprio` change in dq: +5.3%, largest at long Sq;
  *(field, E25)* fused into a Q-outer dq kernel: +1.13% for ~6 µs of extra O reads. The cross-lane sum's XOR masks
  must be the lane distances between lanes holding the same q row (16 and 32 for a 16-row MFMA tile); a mask of
  64 or more wraps onto the lane itself, doubles the value, and dropped dQ to 1.45 dB. Cast O and dO to
  the input dtype before the fused kernel reads them.
- **dq + dK/dV fusion: the form decides.**
  - Q-outer fusion reduces dK/dV over q, which takes float atomics (not deterministic) or a dK/dV partial
    workspace, which §2 shows saves no bytes.
  - A KV-outer single pass computes S, dP and the softmax once: 5 GEMM-equivalents instead of 7. It needs a
    wide KV band per workgroup, because `A` sets the dQ partial bytes (§2): the working form used
    `BLOCK_KV = 256` at D64 and 128 at D128, where the dK/dV accumulators cap it.
    *(field, E26)* At `BLOCK_KV = 64` the same fusion was 1.95–2.55× slower (small-tile overhead plus ~2 GB of
    dQ workspace outweighed the 5-instead-of-7 GEMM saving). *(field, E27)* The wide form started below the split
    pair (690–785 vs 840 TF/s) and passed it only after workspace and scheduling work (879, later ~964).
  - The crossover is real: *(field, E28)* at 64k context, D128, fused 226.9–228.2 ms against 192.6–193.2 ms
    split.
  - Before calling a fused path slow, A/B its host heuristics (pipeline gate, band width, q-split).
    *(field, E29)* A reported 3.7× loss (Hq=8) came back to parity, and, on sub-1024 D128 shapes, 1.31× and 1.30× to 1.05× and 0.98×, after fixing
    two host constants.
- **Splitting a kernel usually loses.** *(field, E30)* dK/dV split into dV-only and dK-only kernels to reach
  occupancy 3: −36% (Q/dO read twice). *(field, E31)* Overlapping the reduce with the next band group's body: both
  fought for DRAM, wall 226.7 → 237.4 ms.
- **Price an auxiliary kernel by the main kernel's disturbed time.** An occupancy-1 body that fills the
  machine has no idle slots, so a "hidden" kernel only renames CU time.
  - *(field, E32)* A delta kernel's per-call time beside the body spread over 25× (`measurement.md` §9); not
    chunking it at all tied (+0.03%).
  - *(field, E33)* Aligning chunks to whole fills still left the D128 body 18% slower and the co-resident reduce
    3× slower: the binding term was the body's register headroom (498/512 → 18% tax; 449/512 → 3.6%).
  - Split a knob's effect into body and eviction by running it with and without the co-resident kernel.
    *(field, E34)* A "ring depth regression" of 0.6 ms was 0.15 ms body and 0.46 ms evicted reduce.
  - The co-residency tax scales with fold bytes divided by the body work beside them. *(field, E35)* Spreading the
    same fold over 3 unequal body chunks instead of 2: 1.8% faster at D64 (1.9% slower at D128); slicing the burst 8
    ways read 1.35% faster in one round, flat from 4 to 16 slices and worse from 32, but the next round found 8 slices
    about as costly as a single-shot fold, and another campaign read 8 slices +0.9%.

## 8. Causal and window masking in the q-loop
- **Price diagonal blocks by time, not by visit share.** *(field, E36)* In a causal KV-outer body the diagonal
  q-block was 6.06% of visits; skipping it in every wave (wrong results) bought only 0.37%.
- **Audit the masked q-loop separately**; it is a second, more expensive copy of the body.
  - *(field, E37)* Folding `-log2e · lse` into GEMM1's C-init in the masked loop too: 0.98% faster (masked loop
    6338 → 5402 instructions, hazard `s_nop` 779 → 222 cycles per trip) from 6% of the trips.
  - With `ROWS_PER_WAVE_KV == BLOCK_Q` on square shapes each wave of a diagonal block is entirely below,
    on, or above the diagonal. Classify per wave: below needs no mask; on the diagonal skip the 16-tile
    blocks with `nt > mt`; above publishes zeros. Waves wait for each other at the barrier, so the block
    costs the slowest wave. *(field, E38)* Diagonal MFMAs 128 → 88, critical path 675 → 560, 0.59% faster,
    bit-identical; *(field, E39)* the dq version 0.34% faster.
  - Skip with a wave-uniform branch. An MFMA ignores EXEC, so lane masks skip nothing.
  - Write the arms as sequential wave-uniform branches, not nested ones (`registers_occupancy.md` §6).
  - Count the values a branch must carry. *(field, E39)* In a fully unrolled dK/dV head loop the branch carried
    40 values: spill 0 → 5, 0.17% slower. Check spill before counting saved work.
- **Sliding window.**
  - Bound the q-loop by the window under a compile-time guard so the full-causal binary is byte-identical.
  - *(field, E40)* With that bound already in place, window-gated host configuration on the split pair
    (q_split 4 → 1, retuned dq and dK/dV tiles and wave counts) and kernel changes took a windowed backward from
    216 to 298 effective TF/s.
  - Take the band from the window, `min(cap, pow2_floor(W))` with the cap set by the K/V staging, and scale the
    waves with it. *(field, E41)* With the waves fixed, the full-causal band was the worst choice under a window:
    6.6–9.0× slower than the split path, 1.04–1.41× one step narrower; a narrow band also let interleaved
    waves share Q/dO reads. The round that took the band from the window, which also set flat_wg 256 and
    windowed the reduce-pipeline gates, gained 17.79% *(field, E42)*.
  - Keep out-of-window kv bands in the grid: their rows still owe zero dK/dV. Give their K/V read descriptor a zero
    `num_records` so the hardware drops those loads. *(field, E44)* On a rectangular windowed backward that cut the
    body 3.8% (wall 1.44%); dispatching the live bands first instead was 16% slower, because the dead bands' zero
    stores had been covering the live bands' uneven trip counts. Keep q-split at 1.

## 9. Head dim
Retune per head dim. *(field, E43)* A GEMM2 prefetch-ring depth of 2 was the D128 setting and 1.9% slower at
D64; band interleave paid at D128 only (§5); the D128 gap is the workspace (§2).

## Verify
- dQ/dK/dV are bitwise identical over repeated runs, or the non-deterministic route is explicit and approved.
- Every buffer slice is under 4 GiB, and the workspace capacity and cache policy were measured at both ends.
- The masked q-loop and the window bounds have their own ISA census and accuracy gate.

## Related
`grid_dispatch.md` · `split_kv.md` · `registers_occupancy.md` ·
`attribution.md` · `correctness.md` · `../gemm/datapath.md` §8 ·
`../gemm/epilogue.md` §5 · `../gemm/attribution.md` §2 · `../hardware/gfx950/memory_and_sync.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. A version named in a round agent's prose or a prompt is not counted. The month is the run's; for a line
backed only by a note (a myskill card), the line says which date it is. "Campaign score" is the campaign's own
scored ruler. "Primus-Turbo" commits are those of the local repository
`/workspace/code/gpt_oss_docker/sync/tensorwise/Primus-Turbo` unless the line names another; their messages carry
the numbers.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, campaign score convTF(D128 fused) / convTF(D64 fused) with both arms interleaved in one process; arms read as wall ms] sibling role split at a two-band group (role K holds dK and dQ, role V holds dV, each recomputing GEMM1, which the plan priced at +20% flops): role K alone 4.83 ms, role V alone 2.81, both dispatched 8.02, with q_pref 7.37 against the reference's 6.40 (−0.97 ms); the round reads the bill as a gross tax near 1.6 ms against a byte benefit near 0.6 ms (gpt-oss D128 fused backward, B=2 Hq64/Hkv8 S=8192 causal; campaign 20260813_154001 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_154001/memory.md:120; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_154001/memory.md:122; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_154001/memory.md:38; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_154001/memory.md:13)
- E2 [2026-08, gfx950, FlyDSL ?, N=?, layered subtraction probes on the fused body (partial stores sent to an out-of-range descriptor, then GEMM3 removed, then the reduce removed), wall ms] D128 6.78 → 5.70 → 4.91 → 4.18 ms against D64's compute phase doubled, 4.25 ms; D128 against 2× D64: partial store 1.82 vs 1.11 ms, reduce 0.63 vs 0.47, everything else 4.18 vs 4.25, wall 6.78 vs 5.82 (+0.96) (gpt-oss fused backward, B=2 Hq64/Hkv8 S=8192 THD causal, D128 against D64; campaign 20260810_114422 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:112)
- E3 [2026-08, gfx950, FlyDSL ?, N=?, `_probe_bwdcfg.py`, each cell in its own process, min of 40] a real fp32-atomic dQ path (512 `buffer_atomic_add_f32`, no scratch): 105.48 against 6.34 ms (+1566%), against a whole dQ split-K round trip of 1.229 ms; band 256 → 512 at D64 9.218 against 5.601 ms (+65%); at D128 a 256-row band (with BLOCK_Q 32) 7.428 against 6.341 ms (+17%) (round 4). An earlier probe that kept writing into the band-indexed workspace read 27.54 ms for atomics against 6.342, and 30.27 / 30.55 ms with a wave's 64 lanes packed onto 4 cache lines (round 1); round 4 records that such a probe does not price the band-less buffer's line-touch pattern. No transposed epilogue was timed on the band-less buffer (llama-70B-shaped backward d128_70b_b2 = B2 Hq64/Hkv8 S8192 D128 causal, and D64; campaign 20260822_152617 rounds 1 and 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/memory.md:164; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/rounds/round-04/agent.txt:7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/memory.md:168; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/goal.md:133; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/memory.md:131; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/memory.md:167)
- E4 [2026-08, gfx950, FlyDSL ?, N=?, ?] bf16 packed-atomic dQ (a16) costs ~1.1 dB of dQ SNR; deployed dq 47.8 / dk 49.0 / dv 49.4 dB (llama D128 backward, a16 path; stated in the goal of campaign 20260826_010443, local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/goal.md:166; myskill local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/15-attention-optimize.md:210)
- E5 [2026-08, gfx950, FlyDSL ?, N=?, ?] a16 atomics issued with the store's own address layout (16 lines per instruction where the coalesced image covers 4): whole backward 13.289 ms with its atomics 10.5 ms, against 1.05–1.11 ms of atomics (whole backward 3.84–3.90 ms) for the coalesced-address probe and 0.781 ms for the native coalesced image; the "8.6x" in the campaign's goal and code comment is a ratio of RMW line transactions, not a timing (llama D128 backward l70b, B=1; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:12; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:13; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:111; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/rounds/round-16/diff.txt:195)
- E6 [2026-08, gfx950, FlyDSL ?, N=?, ?] `buffer_atomic_pk_add_bf16` with sc0 set (aux 1/17) lost 3/4 of the updates, and the performance arm raised a memory-aperture violation (llama D128 backward, a16 path; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:62; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:63)
- E7 [2026-08, gfx950, FlyDSL ?, N=?, min of N, deployed base against one read with the fold deleted (`nored`)] exposed deterministic dQ fold 0.414 ms = 12.9%: base 3.2008 ms (3.19–3.23 in the next campaign's table), `nored` 2.7868 ms (llama D128 backward l70b = 1/64/8/8192/128; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:9; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:10; local:/workspace/code/gpt_oss_docker/sync/goal_a16.md:16; the same number in local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260825_152128/memory.md:25; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260825_152128/memory.md:19)
- E8 [2026-08, gfx950, FlyDSL ?, N=3–4, campaign score `llama`, each arm in its own process, warm, min of N, in-batch controls] a16 turned on in production form, together with block_kv 256, block_q 32 and q_split 4: +2.11% (bench 1.01337 / 1.01611 / 1.01958); the round: at block_kv 128 the atomic leg loses (+0.79 ms against the fold's 0.41), and the win is that a16 removes the co-resident fold, which makes the 256-row band affordable; then a tied-operand MFMA on the dK chain plus q_pref, which the a16 allocation made affordable: +12.29% (1.0183 → 1.1438 / 1.13302 / 1.15356 / 1.15175) (llama D128 backward, l70b and l8b cells; campaign 20260826_010443 rounds 16 and 20; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:392; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:387; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:391; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:434; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:433)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, each cell in its own process, 5 warm-up calls, min of 40, JIT cache cleared per block] the a16 path ported to another tree: d64_gptoss_b4 962.9 → 1023.8 TF/s with the dQ workspace 4.83 → 0.83 GiB; l70b 863.5 → 1008.8 (gpt-oss D64 backward B4 4/64/8/8192/64 and llama-70B D128 backward 1/64/8/8192/128; Primus-Turbo 2067760b in local:/workspace/code/gpt_oss_docker/sync/syncv4/Primus-Turbo)
- E10 [2026-09, gfx950, FlyDSL ?, N=?, campaign score (speedup × accuracy, `bench.sh`, 3-run range 0.62%) and the scored kernel time] dK/dV bf16 atomic fold in place of the slot reduce: d128 0.419 → 0.554 ms (+32%), d64 1.577 ms (+12%), reverted (dualwave_swp backward, D=128 llama-8b and D=64 gpt-oss cells; campaign 20260919_155238 round 6; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/memory.md:105)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, one probe, wall ms] aliasing the partial writes of all 64 bands onto one slab (dQ wrong, diagnostic) saved 8.5 GB for 0.60 ms = 0.071 ms/GB against a roofline 0.161 ms/GB, 2.3× (gpt-oss D128 fused backward, B=2 Hq64/Hkv8 S=8192; campaign 20260810_114422 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:112; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:114)
- E12 [2026-09, gfx950, FlyDSL 0.2.4, N=?, the fold kernel launched alone (12 shapes, warm-up, min of 7×25 launches)] widest-first dK/dV slot-fold tiles: b1 15.39 → 13.63 µs (+12.9%), b2 18.12 → 16.37 µs (+10.7%) in the kernel; the round converts these to b1 0.14% / b2 0.07% of wall by its own accounting (gpt-oss D64 backward Hq64/Hkv8 S=8192 causal, B=1 and B=2; campaign 20260911_114527 round 12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:430; version printed at local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-01/deep_raw.jsonl:2536)
- E13 [2026-08, gfx950, FlyDSL ?, N=4, `nored` subtraction probe, 4 of 4 pairs, spread 0.5%] removing the fold (diagnostic): −5.84% (0.333 ms) after the body got 4% faster; the campaign's first pricing read −4.9% (0.289 ms) (gpt-oss D64 fused backward, B=4 Hq64/Hkv8 S=8192 causal; campaign 20260819_065627 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/rounds/round-08/agent.txt:70)
- E14 [2026-08, gfx950, FlyDSL ?, N=4, two-factor experiment, 0 of 4 pairs faster] q_split 8 with three fixed chunks: tail 1/8 (`[1,6,1]`) +8.01%, tail 1/4 (`[2,4,2]`) +8.11% (gpt-oss D64 fused backward, B=4 S=8192; campaign 20260819_065627 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/rounds/round-08/agent.txt:81)
- E15 [2026-08, gfx950, FlyDSL ?, N=?, ?] workspace cap 3 GiB against the uncapped (16 GiB) axis: D64 causal B4 876 against 990 TF/s; llama-70B D128 gives up 22% (gpt-oss D64 and llama D128 backward; Primus-Turbo 9c58dc31; Primus-Turbo@9c58dc31:primus_turbo/flydsl/attention/flash_attn_bwd.py:85)
- E16 [2026-08, gfx950, FlyDSL ?, N=4, campaign score convTF(D128 fused) / convTF(D64 fused), one interleaved process; the ILV sweep read once per point] band interleave: ILV 1/2/4/8/16 = 813.2/811.5/820.1/818.6/819.7 TF/s, flat from 4; the kept change +1.3% (four final benches all above the base); head-major `[band][B][Hq][Sq][D]` 7.1686 ms / 767.0 TF, −5.8% (gpt-oss D128 fused backward, B=2 Hq64/Hkv8 S=8192, 64 concurrent bands; campaign 20260810_114422 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:112; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:18; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:116)
- E17 [2026-08, gfx950, FlyDSL ?, N=3, ?] the same interleave on D64 full causal read flat three times (−0.14%; 5.9721 against 5.9479; 5.7127 ms / 8.046 J against 5.7035 / 8.023) (gpt-oss D64 fused backward; myskill pitfalls/13 §⑪, myskill commit 6f9fc83 of 2026-08-28; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1183; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1184)
- E18 [2026-08, gfx950, FlyDSL ?, N=2, `_pw_probe.py bwd`, each read in its own process, arms alternating; energy per pass and UMC activity points] partial writes non-temporal against cached: 5.7364 / 5.7545 against 5.7021 / 5.7278 ms, UMC 25 → 27 (+1.3 GB), wall +0.45%; partial reads cached against non-temporal: 0.1 J more per pass, UMC unchanged (gpt-oss D64 fused backward, B=4 S=8192 Hq64/Hkv8; myskill pitfalls/13 §⑩, measured 2026-08-20, no run record found; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1168; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1175; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1170; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1171; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1160)
- E19 [2026-08, gfx950, FlyDSL ?, N=?, `band_ring` clamps the fold's read footprint (dQ wrong, the same loads issued); energy per pass and UMC points] 268 MB 7.662 J / 17 UMC; 1.07 GB 8.101 / 26; 2.14 GB 8.132 / 27; 4.29 GB 8.127 / 27 (gpt-oss D64 fused backward; myskill pitfalls/13 §⑩, 2026-08; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1192; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1193)
- E20 [2026-08, gfx950, FlyDSL ?, N=?, campaign score, one reading] ILV=64: 5.6965 ms, "+18.7%", an artefact: the 8.6 GB slice overflowed the 32-bit num_records and the partial writes were dropped (gpt-oss D128 fused backward; campaign 20260810_114422 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/memory.md:112; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:20)
- E21 [2026-08, gfx950, FlyDSL ?, N=2, two sessions per arm] ring opened on the window: workspace 16 GiB → 1.25 GiB; dkdv 0.4214 → 0.4137, dqred 0.1332 → 0.1328 ms, recorded as +1.4% over two sessions, dQ bit-identical; the same change ported to another tree, together with band interleave turned off, read 0.6168 → 0.6053 ms on the windowed B4 shape (gpt-oss D64 sliding-window (W=128) backward; campaign 20260818_070555 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_070555/memory.md:157; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_070555/memory.md:158; the port: Primus-Turbo 9c58dc31)
- E22 [2026-08, gfx950, FlyDSL ?, N=?, the campaign's own-process warm min-of-N probe against the deployed base] in-body last-arriver fold (q_split 1 body, a per-(batch, q-block, kv-head) i32 counter, one lane announcing after the last head with `s_waitcnt(0)` as the release, the last arriver folding the block in its own epilogue in ascending-band fp32): correct (dQ 48.2 dB, deterministic) but 5.886 ms at fold depth 8 against the deployed base's 3.19 ms; with the band interleave turned off 4.67 ms, "still 1.5x base"; the round puts the loss on workgroup count: thousands of small reduce workgroups keep thousands of requests in flight, a handful of fat body workgroups in their epilogue keep tens (llama D128 backward l70b = 1/64/8/8192/128; campaign 20260826_010443 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:254; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_010443/memory.md:256)
- E23 [2026-07, gfx950, FlyDSL ?, N=3, campaign score: mean conv TF/s over four Sq on the fast-exp path, gates SNR ≥ 34 and determinism] dropping the B-GEMM and its rho/R correction from dq: 704.2 → 772.4 (+9.7%), scored +9.81% (770.7 / 765.0 / 773.0 against 703.9); accuracy cost ~0.2 dB (Meta hd64 varlen backward, dq kernel; campaign 20260715_142909 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:79; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:84; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:80)
- E24 [2026-07, gfx950, FlyDSL ?, N=3, campaign score: mean conv TF/s over four Sq] the delta computed by a FlyDSL odo kernel in place of the framework's `-(dO*O).sum(-1)` (O and dO cast to the input dtype), together with `s_setprio(1)/(0)` around dq's GEMM2 MFMA chain: +5.5% (660 → ~696, largest at Sq 16384: 645 → ~699), scored +5.34% (695.2 / 690.8 / 688.1 against 660.0) (Meta hd64 varlen backward; campaign 20260715_142909 rounds 2–3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:18; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:19; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:23; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:36)
- E25 [2026-07, gfx950, FlyDSL ?, N=2, campaign score (whole-backward conv TF, S=8192, median), same-process ABBA, 2 of 2 non-overlapping] delta fused into the Q-outer dq kernel and the odo launch deleted: 840.8 / 839.8 against 830.1 / 831.8 (+1.13%), outputs bit-identical; dq itself 1362.7 → 1344.7 µs, the note putting the extra O reads at ~6 µs; butterfly XOR masks of 64/128 paired each lane with itself and dropped dQ SNR to 1.45 dB (gpt-oss D64 backward, B=2 Hq64/Hkv8 S=8192 causal; campaign 20260728_123346 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:183; myskill local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:352; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:353)
- E26 [2026-07, gfx950, FlyDSL ?, N=?, campaign score: mean conv TF/s over four Sq] a fused KV-outer backward (dK/dV plus a transient dQ) at BLOCK_KV 64: 1.95–2.55× slower; its deterministic dQ workspace was sized at ~2 GB by arithmetic when the round scoped it (Meta hd64 varlen backward, Hq 128 / Hkv 16; campaign 20260715_142909 round 11, scoped in round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:127; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:130; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:31; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:119)
- E27 [2026-08, gfx950, FlyDSL ?, N=3, campaign score: full-causal effective TF/s at B=3 S=8192 Hq64/Hkv8 D64, three reads] the fused KV-outer backward against the 839.6 TF/s split pair: 690.1 (round 4), 785.7 (round 7; reads 779.1 / 785.7 / 782.1); after further campaigns 879.2 (reads 876.2 / 879.2 / 877.3) and a best of 964.6 (gpt-oss D64 backward; campaigns 20260803_091701, 20260804_003925 round 9, 20260804_132242; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_091701/memory.md:42; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_091701/memory.md:81; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:119; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_132242/run.log:234)
- E28 [2026-08, gfx950, FlyDSL ?, N=3, campaign score cell `full_d128_64k`, split and fused wall ms in one bench process, three reads] 64k context, D128: fused 226.9–228.2 ms against split 192.6–193.2 ms (226.8517 against 192.6252 in one read); the commit that later deleted the split path gives dQ through the partials 112 ms against 77 ms for the Q-outer dq kernel at 64k (gpt-oss D128 backward, 64k context; campaign 20260816_161801 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260816_161801/run.jsonl:22; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260816_161801/run.jsonl:23; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260816_161801/run.jsonl:24; Primus-Turbo 324965e3)
- E29 [2026-08, gfx950, FlyDSL ?, N=3 for the gate (three interleaved pairs per shape, same process), N=? for the band, fused/split through the deployment entry] after the dQ-pipeline area gate was corrected, the fused path at Hq=8, which had read 3.7× the split pair's time, read parity; after the band choice was corrected, D128 768² 1.31 → 1.05 and 1000² 1.30 → 0.98 (D128 backward, small Hq and sub-1024 shapes; Primus-Turbo a1569ce9; Primus-Turbo b9de6db0)
- E30 [2026-07, gfx950, FlyDSL ?, N=?, campaign score: mean conv TF/s over four Sq] dK/dV split into dV-only and dK-only kernels, reaching occupancy 3: −36% (Meta hd64 varlen backward; campaign 20260715_142909, measured in round 14 and recorded in round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:181; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260715_142909/memory.md:184)
- E31 [2026-08, gfx950, FlyDSL ?, N=?, the round's own probe, wall ms] band groups double-buffered so the reduce overlaps the next group's body: 237.4 against 226.7 ms, reverted; the round reads the 64k cell as already memory-saturated, the overlap adding only contention (gpt-oss D128 backward, 64k context, cell `full_d128_64k`; campaign 20260816_161801 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260816_161801/rounds/round-05/agent.txt:16)
- E32 [2026-09, gfx950, FlyDSL 0.2.4, N=2, rocprofv3 `--kernel-trace --stats`; the campaign's same-process paired ABBA cell ruler, two reads against the session's zero point] the same odo kernel running beside the dK/dV body at b4: per call 59.8 → 1527 µs (25×), bytes identical; b4 not chunked (`_A16_MIN_BAT` 4 → 8): 0.9992 / 1.0068, mean 1.0030 against a zero point of 1.0027 (+0.03%) (gpt-oss D64 backward, Hq64/Hkv8 S=8192, B=4; campaign 20260911_114527 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-09/agent.txt:17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-09/agent.txt:25; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-01/deep_raw.jsonl:2536)
- E33 [2026-08, gfx950, FlyDSL ?, N=?, per-kernel times from the campaign's probes] D128 dQ pipeline cut into whole fills (512 workgroups = 2 fills): wall 3.0837 → 3.0521 ms (−1.0%), body 2.2022 → 2.5281 (+18%), reduce 0.6569 → 1.9798 (3.0×); the round puts the binding term on the body's 498/512 registers. On a D64 cell whose body holds 449: body 2.027 against 1.956 ms with the reduce co-resident, a 71 µs tax (3.6%) (gpt-oss sliding-window and rectangular backward cells; campaign 20260814_082030 rounds 18, 3 and 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/rounds/round-18/agent.txt:19; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:287; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:106; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:231)
- E34 [2026-08, gfx950, FlyDSL ?, N=?, full minus `nored`, each candidate run with and without the co-resident reduce] `g2d=1` read as a 0.6 ms scheduling regression and was 0.15 ms body plus 0.46 ms evicted reduce (gpt-oss D128 fused backward, B=2 Hq64/Hkv8 S=8192; campaign 20260813_154001 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260813_154001/memory.md:136)
- E35 [2026-08, gfx950, FlyDSL ?, N=6–17, each arm in its own process, min of 40 per read, palindromic order] the hidden fold delivered as 8 slices (batch × q): mean 5.8513 ms (n=10) against 5.9315 (n=17), −1.35%; 4–16 slices a plateau, 32 and more worse; chunk plan `[1,2,1]` against equal width: 5.7563 against 5.8608 ms (6 reads per arm), which the round reads as −1.8% at D64 and +1.9% at D128. The next round's granularity re-sweep found the deployed 8 slices about as costly as a true single-shot fold, and a later campaign read 8 slices +0.9% against its default (gpt-oss D64 fused backward, B=4 Hq64/Hkv8 S=8192; campaign 20260819_065627 rounds 2 and 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:196; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:208; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:209; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:190; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260822_152617/memory.md:132)
- E36 [2026-08, gfx950, FlyDSL ?, N=?, campaign probes; the record writes "3/11"] the diagonal q-block is 6.06% of q-block visits in a causal KV-outer body; skipping it in every wave (wrong results, diagnostic): +0.37% (gpt-oss D64 fused backward, B=3 Hq64/Hkv8 S=8192 causal; campaign 20260804_003925 round 8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:102; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:104)
- E37 [2026-08, gfx950, FlyDSL ?, N=14, 14 palindromic pairs over 3 sessions, each arm in its own process] folding −log2e·lse into GEMM1a's C-init in the masked q-loop too: masked loop 6338 → 5402 instructions, `s_nop` wait 779 → 222 cycles per eight-head trip, unmasked loop byte-identical, mean −0.98% (14 of 14 the same sign); myskill puts the diagonal block at 6% of trips (gpt-oss D64 fused backward, B=4 S=8192; campaign 20260819_065627 round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:219; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:220; myskill local:/workspace/code/gpt_oss_docker/myskill/optimizer/00-decision-index.md:287)
- E38 [2026-08, gfx950, FlyDSL ?, N=18, 18 palindromic pairs over 3 sessions] per-wave below/diagonal/dead classification of the masked q-block: diagonal-arm critical path 675 → 560, pooled −0.59%, dq/dk/dv `torch.equal` to the old path; MFMA 128 → 88 in myskill's summary of the same change (gpt-oss D64 fused backward, B=4 S=8192; campaign 20260819_065627 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:232; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:236; myskill local:/workspace/code/gpt_oss_docker/myskill/optimizer/00-decision-index.md:288)
- E39 [2026-07, gfx950, FlyDSL ?, N=2, campaign score (whole-backward conv TF), same-process ABBA, two reads per arm] dq per-wave causal classification: 842.7 / 842.7 against 839.0 / 840.6 (+0.34%); the same classification in the fully unrolled dK/dV head loop carried 40 values through the branch: VGPR 243 → 256, spill 0 → 5, 840.7 / 842.8 against 843.0 / 843.2 (−0.17%) (gpt-oss D64 backward, Hq64/Hkv8 S=8192; campaign 20260728_123346 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:183)
- E40 [2026-08, gfx950, FlyDSL ?, N=3, campaign score: SWA(128) effective TF/s at B=3 S=8192 Hq64/Hkv8 D64, three reads] 216.5 → 294.6 (+36.09%, round 3) → 298.1 (round 4), from window-gated host configuration (q_split 4 → 1 alone with dq BLOCK_M 192 → 128: 215.58 → 241.37; then dq BLOCK_M and BLOCK_KV, dK/dV BLOCK_KV, flat_wg and waves_per_eu) and per-kernel changes on the split pair, the full-causal path bitwise unchanged; round 1 found the window trim already implemented in both kernels (gpt-oss D64 sliding-window backward; campaign 20260808_153438 rounds 1–4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:59; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:71; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:37; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:45; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260808_153438/memory.md:53)
- E41 [2026-08, gfx950, FlyDSL ?, N=?, fused/split through the deployment entry with the fused window path switched on] under a window, the commit that took the band one step narrower than full causal's (D64 256 → 128, D128 128 → 64) changed only the band and lists B3 8192² D64 W128 1.24, W2048 1.04, B4 4096² D128 W2047 1.41, B4 1024×16384 D128 W2047 1.26, against 6.6 to 9.0 at the old band (gpt-oss sliding-window backward, D64 and D128; Primus-Turbo 17f683584)
- E42 [2026-08, gfx950, FlyDSL ?, N=3, campaign score `sp_over_fu` (split time / fused time), three reads] the band taken from the window, `min(cap, pow2_floor(W))` floored at 32 with the cap set by the K/V LDS staging (D128 128, D64 256), flat_wg doubled to 256 for bands of 128 and more so the per-thread dK/dV accumulators stay unchanged, and windowed reduce-pipeline gates: +17.79%; flat_wg alone 0.8451, the cap used unconditionally 0.9555 (the W=128 cell 0.691 → 0.803 ms); on cells with W ≫ BLOCK_KV the rule widened the band against the narrower window choice and halved the dQ partial bands; Q/dO sharing between interleaved waves (FQ_PAIR, D64 narrow band) came in round 3 (gpt-oss sliding-window backward cells, D64 and D128; campaign 20260814_082030 rounds 2–3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:97; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:91; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:92; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:96; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:104)
- E43 [2026-08, gfx950, FlyDSL ?, N=6, six palindromic pairs] `dkdv_g2d=2` (GEMM2's dt prefetch ring one group deeper) at D64: 26 fewer `s_waitcnt` in the unmasked loop, wall +1.9% (6 of 6, single pairs +1.2% to +2.7%); depth 2 is the deployed D128 value (gpt-oss D64 fused backward, B=4 S=8192; campaign 20260819_065627 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:234; myskill local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1099)
- E44 [2026-08, gfx950, FlyDSL ?, N=9, cross-process interleaved 9 v 9 (3 reads per process, 3 processes per arm), body and wall time] dead kv bands of a rectangular bottom-right windowed backward kept in the grid with a zero-`num_records` K/V read: body 0.8445 → 0.8122 (−3.8%, distributions not overlapping), wall 1.2732 → 1.2549 (−1.44%); the band axis rotated so live bands dispatch first: body 0.808 → 0.938 (+16%), at Skv=4096 0.755 → 0.902 (+19%) (gpt-oss D64 windowed fused backward, Sq=1024; campaign 20260814_082030 round 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260814_082030/memory.md:231)
