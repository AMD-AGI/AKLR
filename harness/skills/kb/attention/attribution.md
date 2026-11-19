---
title: "Attention — attribution: the true clock, which counter ranks which change, stall pools and probes"
kind: method
lever: attn_attribution
ops: [attention, mla]
gens: [gfx950]
dtypes: [bf16]
regimes: [many query rows, dQ dK dV, few query rows, occupancy-1, power-capped]
flydsl: 0.3.2
---

# Attributing an attention kernel's time

> Counters propose; probes decide. This card adds what attention bodies taught on top of
> `../gemm/attribution.md` (compute-additivity and byte-rate probes, drop/alias/identity/illegal
> arms, additive margins, the wave-cycle budget, readings you cannot trust), which it does not repeat. Trace
> capture is in `../profile.md`. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed
> under Evidence at the end. Use them for direction and order of magnitude, not as targets.

## Route here when
- You are about to rank candidates for an attention body and need the counter that can see the change.
- A counter reads "LDS-bound", "latency-bound" or "X% waiting" and you are about to spend a round on it.
- A probe priced a pool and the built change did not deliver it.

## 1. Measure the real clock first
- Derive sclk from `GRBM_GUI_ACTIVE / 8 XCDs / dispatch duration` (`../gemm/attribution.md` §4).
  A `rocm-smi` sample taken while the benchmark runs can land after a cached kernel has already finished and
  report the idle clock, 158 MHz *(field, E1)*.
- Profiling did not lower the clock on a power-capped MFMA microbenchmark *(field, E30)*: under `rocprofv3 --pmc`
  the in-kernel clock read about 1% higher than without it, and `GRBM_GUI_ACTIVE / 8 / dispatch time` matched it
  within 0.6%. The first dispatches after idle read low (1.60–1.90 against ~2.02 GHz sustained) either way, so
  ~1.82 GHz under counter collection against ~2.07 GHz sustained *(field, E2, E3)* does not show a profiler effect.
  Warm the card before collecting, and compare arms under the same collection mode and run length.
- A roofline needs the true clock. The bf16 MFMA roofline is `256 CU × 4 SIMD × 1024 FLOP/cycle × sclk`
  (`v_mfma_f32_16x16x32_bf16`: 16384 FLOP in 16 cycles). Utilisation computed at the nominal clock reads ~20%
  low *(field, E4)* and turns "energy-bound" into "room left".

## 2. Which counter ranks which change

| Change | Rank it with | Blind counters |
|---|---|---|
| grid, dispatch order, XCD placement, bytes | wall time; `SQ_WAIT_ANY` to cross-check | L2 hit rate; `SQ_VALU_MFMA_COEXEC_CYCLES` under a remap |
| schedule of an occupancy-1 body (anchors, fences, hoists) | `SQ_WAVE_CYCLES`; `SQ_WAIT_INST_LDS` to explain, not to rank | MFMA busy |
| the same instructions in a new order | wall time, with sclk at the power cap (§8); `GRBM_GUI_ACTIVE` to cross-check | `SQ_VALU_MFMA_COEXEC_CYCLES` (§8), `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_ACTIVE_INST_VALU` |
| MFMA atom at the power cap | wall, sclk and power together | MFMA busy, MFMA utilisation |
| an `iglp_opt` strategy or a software-pipelining choice | wall time; `SQ_WAVE_CYCLES` or `SQ_WAIT_ANY` to cross-check | instruction and register counts |

- **Grid and placement changes: rank by wall time.** Across XCD maps, coexec / MFMA busy stayed flat and the L2 hit
  rate ranged 74–96% with no wall difference *(field, E31)*; on an hd64 backward's dq kernel a grid order cut
  `SQ_WAIT_ANY` 4.1% and ran 1.98% slower *(field, E43)*.
- **`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` is a constant**, the pipe cycles of the instruction mix (16.000
  for `v_mfma_f32_16x16x32_bf16`). Nine arms whose wall time spanned 76% read 16.000 bit-identically *(field, E5)*.
  It cannot see the atom's energy, the instruction order, or how well the MFMAs are covered.
- **An atom swap at the cap moves the clock, which MFMA busy cannot see.** In a Triton attention forward,
  16x16x32 against 32x32x16 read MFMA busy bit-identically and `MfmaUtil` implied 21% slower; the wall was 9.2%
  slower with sclk 10.6% higher *(field, E32)*.
- **Rulers for occupancy-1 body scheduling.** *(field, E6, E7)* `SQ_WAVE_CYCLES` repeated within 0.20–0.64% and
  `SQ_WAIT_INST_LDS` within 0.1%, against 0.62% for the benchmark score (2.0% for one cell's candidate time), all
  A-vs-A on an unchanged tree; both counters exclude the host and the neighbours, but a wall median of 210 calls
  repeated within 0.34% too *(field, E33)*. Rank with `SQ_WAVE_CYCLES`, every arm profiled in the same position of
  one procedure; `SQ_WAIT_INST_LDS` fell 4% on a move 3% slower *(field, E33)*: use it to explain, not to rank.
  Close with the benchmark.
- **Co-execution is not a safe ruler for `iglp_opt` strategies.** It ranked three of one dkdv body in wall order
  *(field, E8)*, but put a microbenchmark pair 3.3% apart the wrong way round while `SQ_WAVE_CYCLES` and
  `SQ_WAIT_ANY` ranked all four in wall order *(field, E34)*; −351 instructions elsewhere on the dkdv body was worth
  0.0% *(field, E9)*. Rank strategies by wall time.
- **Normalise before comparing.** MFMA busy and coexec count cycles, `SQ_ACTIVE_INST_VALU` and `SQ_WAVE_CYCLES`
  quad-cycles; all four are summed over the SIMDs, and all but coexec over the waves of a SIMD too *(field, E35)*.
  Ratios within one unit, such as coexec / MFMA busy, compare directly; multiply the quad-cycle counters by 4 before
  setting them beside MFMA busy or coexec.
- **Do not copy a co-execution target from another kernel.** *(field, E10)* A dkdv body read coexec / MFMA busy
  19.2% and a forward on the same machine 48.1%, but `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES` (a resident-wave proxy)
  was 7.7 against 31.5: most of the forward's overlap came from sibling waves. Derive the reachable value from
  the pipe arithmetic of your body.

## 3. The stall pool
MFMA busy and VALU busy are fixed work. Split the timeline into `busy_union = MFMA% + VALU% − coexec%` and
`neither = 1 − busy_union`. For a target speed-up X with co-execution unchanged:
```
stall' = 1/X − (MFMA% + VALU% − coexec%)    # as a share of today's timeline
```
- *(field, E11)* A dkdv body at 44.0 / 36.4 / 8.5 (neither 28.1%) needed stall cut to 14.7% for 1.155×, with no
  co-execution gain; doubling co-execution alone still required stall to fall 17%. With stall unchanged,
  co-execution would have had to rise about 2.6×, to about half of MFMA busy (estimated from the same three
  shares). Price the stall pool and the reachable co-execution before choosing a route.
- The overlap headroom to report is `busy_union` shrinking to the larger pipe with waits unchanged (an estimated 1.39× on
  that body), not `(MFMA% + VALU%) / max` (1.83× there), which assumes zero waiting and no existing
  overlap.

## 4. Wait counters are not the bound
- `SQ_WAIT_ANY` counts wave-cycles, not idle SIMD cycles: at ≥2 waves per SIMD another wave issues while one
  waits. Use the `WAIT_*` family to name the dependency, never as a recoverable percentage; `MfmaUtil` 66% is not
  "34% left" either.
  - *(field, E12)* A forward read `SQ_WAIT_ANY` 22%; deleting every `vmcnt` wait, every `lgkmcnt(0)` or every
    `s_barrier`, one class per arm, moved wall time ≤2.6% in total.
- `SQ_VALU_MFMA_BUSY_CYCLES` (summed over the 4 SIMDs) against `SQ_LDS_IDX_ACTIVE` (one count per CU per active
  cycle) does not clear the LDS port (`../gemm/attribution.md` §6): 4× is only parity per SIMD, and a HIP
  read → wait → MFMA loop at 6× still ran 1.25× faster with its LDS reads halved *(field, E36)*. Price the LDS
  with a halve-the-reads arm before dismissing `SQ_WAIT_INST_LDS`.

## 5. Probe discipline
Subtractive arms give a ceiling; additive arms give a marginal price; the two can disagree in sign
(`../gemm/attribution.md` §2–§3). What attention bodies add:
- **Do all three layers in one go**: instructions issued with addresses pointed out of bounds (memory only) →
  instructions and their consumers removed → the next stage removed. *(field, E13)* A "skip the partial store" arm
  also removed the GEMM whose accumulator the store read (dead-code elimination): 6.78 → 5.70 → 4.91 → 4.18 ms.
  The middle arm alone would have charged the GEMM to bytes and overstated them by 72%.
- **Keep the consumer chain alive.** *(field, E14)* Deleting 32 exps also deleted 4 MFMAs and 16 converts and inflated
  the reading from +8% to +21.6%. Check an opcode histogram for every subtractive arm.
- **A deletion ceiling is not a hiding gain.** *(field, E15)* Skipping the per-tile K/V DMA showed 8.7%; the built
  double-buffered prefetch lost 4.3–4.7%, because the deleted cost had a side effect (LDS write/read contention)
  the prefetch kept. Split a cost into throughput and latency before building a hider: a deletion probe prices both
  together, and only latency can be hidden.
- **Illegal subtraction overstates.** *(field, E16)* Replacing every barrier with a scheduling fence read +3.77% at
  B=1 (+3.36% at B=2, +0.20% at B=4); opening one redundant barrier priced 0.10% each; the legal reduction from 16
  to 9 barriers lost 1.2–1.7%, because it moved an LDS write burst into a transposed-read burst
  (`lds.md` §9). The illegal arm also bought freedom from wave drift, which no legal version gets.
- **Price a synchronisation family at N, N/2 and 0.** *(field, E17)* Two barriers per iteration → one gained
  0.12–0.18%, → zero lost 0.25%: the remaining barrier pinned the dual-wave phase. Measuring only 0 hides both
  the sign and the optimum. Both 2 → 1 arms broke a real write-after-read dependency (SNR 51.7 → 23.0 dB): they
  are bounds, not edits; the legal form doubles the LDS buffers (2 → 4) so one rendezvous covers two KV tiles.
- **A price expires when the code around it changes.** *(field, E18, E17)* Deleting one barrier priced 0.97% while
  `sched_barrier(0)` fences wrapped the barriers; after they were relaxed the whole family priced within ±0.25%
  (above). Re-measure rendezvous prices on the current tree.
- **Pick a probe direction that does not spill.** *(field, E19)* Six arms that removed or replaced a row-sum MFMA run
  all spilled; running it twice on the same accumulator was register-neutral and priced it at 4.4% of wall,
  not the 11% its pipe share suggested.

## 6. The stall census
One arm per wait source: remove exactly one source, keep the consumer graph and the MFMA and exp counts, pass an
ISA gate (opcode histogram, spill, VGPR) before benchmarking, bracket the arms with same-binary controls.
- Pools need not add, in either direction *(field, E37)*: deleting the P exps and the K/V loads of a Triton forward
  together saved 0.95× the sum of the two apart at the power cap and 1.14× below it. Measure the combined arm before
  summing pools.
- *(field, E20, E17)* One forward ranked: exp/transcendental 7.8% > LDS reads 5.8% > K/V DMA 3.95% > per-workgroup fixed
  cost 1.4% > grid and tail ≤0.2% > rendezvous ≤0.25%; no counter had suggested that order.
- An arm whose whole ISA, operands and kernel descriptor included, is identical to the base's is the same kernel:
  drop it without benchmarking. An opcode-only hash is not enough: in a HIP microbenchmark, arms that changed only a
  shift immediate or `s_waitcnt` counts hashed identically to the base and ran 11.8× and 2.8× slower *(field, E38)*.

## 7. Cycles to wall
- **The conversion rate is not fixed.** *(field, E21)* A controlled pair that added 4.04% cycles cost 3.69%
  wall (91%); *(field, E22)* a change that removed 1.4% cycles without removing work returned 0.12–0.24% wall (8–17%). These
  are one change each, of different kinds, so direction is not separated from kind. Measure
  the rate in the improving direction on your own kernel before pricing a cycle-only candidate.
- Do not apply a cycle→wall discount to a number already measured in wall time. *(field, E23)* That double discount
  turned an 8.7% pool into 2.5% and closed the largest axis of a kernel.

## 8. At the power cap
- **At ≥99% TBP, wall ≈ total energy / TBP.** A pure reorder can still change the energy: the arm with more cycles
  clocks higher, but the clock made up only part of the added cycles. *(field, E24)* Five
  reorder arms, four with more interleave and one with the exps isolated, all fell 1.46 to 3.39% below the deployed
  order, which sat at the top of the band. One instruction-identical pair: co-execution −9.86%, cycles +4.39%,
  sclk +3.6%, 1.87% slower.
- **Three price classes, each read from a few arms:** on a forward whose co-execution covered 44% of SIMD cycles,
  52% of MFMA busy *(field, E24)*, deleting work (the 64 exps) gained 8.7% *(field, E23)*, deleting 1.4% of the
  cycles without work returned 8–17% of them *(field, E22)*, and reorders read up to 3.4% slower than the deployed
  order *(field, E24)*. Low co-execution does not fix a reorder's price either: on a backward at 17% coexec / MFMA
  busy, reorders read from 0.97% slower to 1.33% faster, the largest gain from moving eight dQ atomics, which changes
  when a memory burst arrives *(field, E25, E29)*; on a microbenchmark, from 4.8% slower to 8.8% faster than the
  lowest-coexec order (16–32%) *(field, E40)*. Report coexec / MFMA busy with the class, measure a reorder rather
  than assume its class's price, and name the class before proposing a change.
- **A slower arm clocks higher.** *(field, E19)* Running a row-sum MFMA twice added 11.1% MFMA-pipe work: 4.4% slower,
  sclk +3.9%. Count a change as an energy win only when wall and clock improve together; a clock rise alone is
  not one.
- **At the cap, work under MFMAs still costs, but exp is not the expensive instruction.** *(field, E23)* Removing
  the 64 `v_exp_f32` of a loop gained 8.7% wall with sclk +3.1%; on a power-capped HIP microbenchmark, per pipe
  cycle an exposed exp cost 0.74× and an exp in an MFMA's shadow 0.32× the wall of an MFMA *(field, E41)*. Rank
  candidates by measured wall at the cap, not by trans count or MFMA cycles alone; whether a byte change
  is a cycle or an energy change must be measured (`grid_dispatch.md` §9).
- **At the cap an occupancy gain is paid back as clock, partly or in full.** *(field, E26)* A round that raised
  occupancy 2.985 → 3.978 waves per SIMD while also changing the K/V LDS reads ran at unchanged power, sclk −3.3%,
  0.5% faster. In a microbenchmark where only the LDS size set the occupancy, four of six one-wave steps at the cap
  gave back 24–51% of the cycle saving and two all of it, 0.3–1.1% slower *(field, E42)*. Report sclk next to every
  occupancy candidate and judge it by wall at the cap; below the cap, measure whether the cycle gain survives before
  assuming it does.

## 9. Per-kernel additive probes
Run the compute-additivity probe (`../gemm/attribution.md` §1) on each kernel of an operator, not once for
the operator. *(field, E27)* Adding MFMA work to a sparse-MLA kernel with many query rows (a prefill) changed wall
time by 0.1% (fully hidden); quadrupling the PV MFMAs in the same family's split-KV producer with few query rows (a decode) made it
7% slower (~6 wall cycles per 16-cycle `16x16x32` MFMA, about a third of
its pipe time). One kernel of the family is fabric-bound, the other is not.

## 10. Budget arithmetic for "hide it under the MFMAs"
- Pipe cycles, issue block and VALU shadow per MFMA atom are in `mfma_layout.md` §1; exp and VALU
  rates in `softmax.md` §1. In short: 8 shadow cycles per 16x16x32 and 24 per 32x32x16, shared by
  every wave on the SIMD; an exp uses them twice as fast as a multiply and never runs beside other VALU work.
- Check a per-class model against `SQ_ACTIVE_INST_VALU` before trusting a VALU budget: *(field, E28)* only the
  8-cycle exp fit a backward body's counter within 1%.

## Related
`../gemm/attribution.md` · `workflow.md` · `measurement.md` ·
`schedule_sync.md` · `../profile.md` § kernel-trace-analysis

## Evidence

Every workload below is an attention kernel unless the line names a microbenchmark. A FlyDSL version is given only where the cited run, or another round in the
same container and venv, printed it (`flydsl.__version__`, an import path with its version, or a `pip install flydsl==`
in the run's own setup); otherwise `FlyDSL ?`. Months are those of the round timestamps, or, where
the number is unverified, of the date it was reported. A sign on a wall-time change is the record's own. Lines E30 to E42 are
standalone runs on MI355X, not campaign rounds; the month and toolchain in their brackets are those of the run.

- E1 [2026-09, gfx950, FlyDSL 0.2.4, N=?, `rocm-smi` sampled while a short benchmark ran] `rocm-smi` read the idle 158 MHz because the JIT-cached kernel had already finished; `GRBM_GUI_ACTIVE ÷ 8 XCD ÷ dispatch time` gave ~1823 MHz (gpt-oss D=64 attention backward, Hq=64, Hkv=8, S=8192, causal, cell b4)
- E2 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 `--pmc`, clock derived from `GRBM_GUI_ACTIVE`] 40,318,962 cycles per dispatch ÷ 8 XCD = 5,039,870 cycles in 2.764 ms ⇒ ~1823 MHz under counter collection (same kernel and round)
- E3 [2026-09, gfx950, FlyDSL 0.2.4, N=8, sustained load (20000 / 80000 calls), sclk and socket power sampled 8 times, no profiler] b4 2070 ± 10 MHz at 1399 ± 2 W, b1 2072 ± 4 MHz at 1400 ± 2 W; the record calls the 1823 MHz figure a profiling artefact (same kernel, cells b4 and b1)
- E4 [2026-07, gfx950, FlyDSL 0.2.2, N=?, sclk and power read under sustained load] under load 1713 MHz at 1346 W (96% of the 1400 W limit): the on-clock bf16 peak is 1796 TF, not the nominal 2177 TF, so 1132 TF is 63% utilisation, not 51% (hd64 attention forward, Sq=Skv=16384, B=1, full causal)
- E5 [2026-09, gfx950, FlyDSL ?, N=?, rocprofv3 `--pmc`] `SQ_VALU_MFMA_BUSY_CYCLES` / `SQ_INSTS_MFMA` = 6.7686e9 / 4.2303e8 = 16.000, bit-identical in nine arms (`g2_hs`, `g3d`, `g3_sb` settings) whose wall spanned 76% (FlyDSL dualwave_swp attention backward, D=128 cell, bf16, causal)
- E6 [2026-09, gfx950, FlyDSL ?, N=3–5, A-vs-A on an unchanged tree: 5 PMC samples, 3 bench runs] `SQ_WAVE_CYCLES` 0.20–0.64%, `SQ_WAIT_INST_LDS` <0.1%, the bench score 0.62% (3-run range) (same campaign)
- E7 [2026-09, gfx950, FlyDSL ?, N=3, three same-session reads] the d128 cell's candidate time (`cand_ms`) spread 2.0% on an unchanged tree; unverified (same campaign)
- E8 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, three builds differing only in the `iglp_opt` id] `MFMA_BUSY` bit-identical; COEXEC / MFMA busy 19.24 / 15.55 / 14.60% in the known wall order, while the wall separated them by 0.85% against a 0.8% noise floor (d64 attention backward with fp8 operands, dkdv kernel)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench, single-variable arms] −351 instructions / −234 shuttle / −8 dwords / spill 1→0 = 0.0% (same kernel)
- E10 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, one trace] dkdv COEXEC / MFMA busy 19.2% against 48.1% for the forward kernel in the same trace; `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES` 31.5 for the forward against 7.7 for dkdv, a 4-wave body at occupancy 1: a proxy for comparing the two kernels, not a count of resident waves (same kernel and round as E8)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] dkdv MFMA busy 44.0%, VALU busy 36.4%, coexec 8.5% ⇒ busy-union 71.9%, 28.1% in pure wait, perfect overlap 1.39× (computed). The stall each route to 1.155× needs (5.2924 → 4.5813 ms; 0.281 → 0.147 of the timeline with coexec unchanged; with coexec doubled to 0.170 the stall must still fall 17%) is arithmetic on this triple (same kernel and round as E8)
- E12 [2026-09, gfx950, FlyDSL ?, N=?, subtractive arms bracketed by same-binary controls (drift d64 +0.00%, d128 −1.06%)] `SQ_WAIT_ANY` 22%; deleting all `vmcnt` waits −0.27% / −0.32%, all `lgkmcnt` +0.68% / +0.10%, all `s_barrier` −2.32% / −0.27% (d128 / d64): all waits together ≤2.6% of wall (FlyDSL dualwave attention forward, D=128 and D=64 cells)
- E13 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench (~75 s)] fused body 6.78 ms; stores aimed at an out-of-bounds SRD 5.70; stores and GEMM3 removed 4.91; the reduce also removed 4.18 ms; a single-layer probe books 72% of the compute time as bytes (gpt-oss fused attention backward, D=128, Hq64/Hkv8, Sq=Skv=8192, B=2, THD, full causal)
- E14 [2026-07, gfx950, FlyDSL ?, N=?, ?] deleting 32 exps also deleted 4 MFMAs and 16 cvts and inflated the probe from +8% to +21.6%; unverified (hd64 attention forward)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, ?] skipping the per-tile K/V DMA (tile reused) +8.7%; the built double-buffered prefetch (LDS 16→32 KB, occupancy unchanged) −4.3~4.7%; unverified (Meta hd64 attention backward, dq kernel, fast-exp path)
- E16 [2026-09, gfx950, FlyDSL 0.2.4, N=?, campaign per-cell ruler against the session zero point] all barriers replaced by `sched_barrier(0)` (wrong result) +3.77%, the B=1 cell of a probe that read ×1.0377 / ×1.0336 / ×1.0020 at B=1 / 2 / 4; 8 barriers added (`HS_WAR_BAR`, everything else unchanged) b1 1.0160 / 1.0151 = −0.78 / −0.87% ⇒ 0.10% per barrier; legal `qdo_ring` 16→9 barriers (registers held, SNR 48.7–49.2 dB) b1 −1.73% / b2 −1.63% / b4 −1.20% (gpt-oss D=64 attention backward, a16 dQ path, occupancy 1, 4 waves)
- E17 [2026-09, gfx950, FlyDSL ?, N=2, sustained b4 ruler (≥14 s, 26 samples, 4 burn arms, same-binary controls bracketing)] barriers per iteration 2→1: +0.1236% / +0.1841% (two sites); 2→0: −0.2472%; deleting either single barrier dropped `snr_fwd` 51.7 → 23.0 dB; the legal form doubles the LDS buffers 2 → 4 (68096 B/WG) (gpt-oss D=64 attention forward)
- E18 [2026-08, gfx950, FlyDSL ?, N=3, palindromic, one process per arm, ISA-gated] once the `sched_barrier(0)` fences around `s_barrier` were gone, deleting both hot-loop barriers read +0.10% (3/3), against +0.97% per barrier recorded on the same kernel before (round 5) (gpt-oss D=64 attention forward)
- E19 [2026-09, gfx950, FlyDSL ?, N=2, replicated in reversed order at a pinned 1400 W] `dup` (each row-sum MFMA run twice on the same accumulator, register-neutral): MFMA pipe 1152 → 1280 cycles (+11.1%), 1.9182 ms (−4.4%) at 1684 MHz (+3.9%); six rounds of removal arms all spilled (93/94/101/118/198/217) (gpt-oss D=64 attention forward)
- E20 [2026-08, gfx950, FlyDSL ?, N=3, palindromic arms bracketed by controls, one process per arm, ISA-gated] stall census: K/V DMA +3.95% (3/3), LDS reads 5.8% (V 3.6 / head-K 1.54 / tail-K 0.64), exp/trans 7.8%, fixed cost 1.4%, grid ≤0.2%, rendezvous 0.10% (same kernel)
- E21 [2026-09, gfx950, FlyDSL ?, N=?, controlled pair, PMC, sclk and power beside the wall] a duplicated K/V DMA (`buffer_load` 4 → 8, ISA otherwise identical): `SQ_WAVE_CYCLES` +4.04%, wall 1.8353 → 1.9030 ms (−3.69%), sclk 1625 → 1620 MHz at 1400 W, MFMA busy bit-identical (same kernel)
- E22 [2026-09, gfx950, FlyDSL ?, N=?, scored ruler with bracketing controls; sustained probe] `kh1,nf1`: `GRBM_GUI_ACTIVE` −1.435%, wall +0.12% (scored) / +0.24% (sustained) ⇒ 8–17% paid back (gpt-oss D=64 attention forward)
- E23 [2026-09, gfx950, FlyDSL ?, N=2, replicated at a pinned 1400 W] `noexp` (the 64 `v_exp_f32` removed, consumer graph intact): 1.6881 ms = 1304 TF/s (+8.7%) at 1671 MHz (+3.1%); a cycles→wall factor applied to that wall number had demoted the pool from 8.7% to 2.5% (gpt-oss D=64 attention forward)
- E24 [2026-09, gfx950, FlyDSL ?, N=3, instruction-identical pair, three batches reproduced to 0.01%, PMC and sclk] `sa` (only the instruction order changes): coexec −9.86% (43.64% → 37.68% of SIMD cycles; 52% of MFMA busy before, computed from the two counters), cycles +4.39%, sclk +3.6%, wall −1.87%; of the round's 14 arms the reorder arms were four that added interleave, −1.46 / −1.55 / −2.44 / −3.39%, and `sa`, which isolated the exps, −1.87%, with the deployed order at the top (also summarised as all 14 arms inside ±3.4%); classes 100% (`noexp` +8.7%) / ~15% / 0 ± 2% (gpt-oss D=64 attention forward)
- E25 [2026-09, gfx950, FlyDSL ?, N=?, campaign ruler] COEXEC / MFMA busy 16.88% on the backward; pure-reorder arms −1.33% (eight dQ atomics moved out of the window before a barrier) / −0.84…−0.99% / −0.34% of wall, all faster; in the same family four `ds_write` and their `vmcnt` waits moved into MFMA shadows read 0.0% against a byte-identical control leg (gpt-oss D=64 attention backward)
- E26 [2026-07, gfx950, FlyDSL 0.2.2, N=3, interleaved A/B, sclk and power on both arms] occupancy 2.985 → 3.978 waves/SIMD: 1352 → 1353 W, sclk 1710 → 1654 MHz (−3.3%), wall +0.50% = cycles saved +3.9% less the clock given back (hd64 attention forward, Sq=Skv=16384, B=1)
- E27 [2026-09, gfx950, FlyDSL 0.3.2, N=3, compute-additivity probes, three decode shapes] prefill: MFMA +76% → +0.1% wall; decode producer: PV's 16 `mfma_f32_16x16x32` ×4 → +7.03 / +7.75 / +6.79%, about 6 wall cycles per MFMA (about a third of its pipe time) (aiter's FlyDSL sparse-MLA prefill and decode kernels, topk 2048)
- E28 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 PMC, per-class model] `SQ_ACTIVE_INST_VALU` modelled with exp at 2 quad-cycles (8 cycles): 3,218 against 3,246 measured (0.9%); a quarter rate predicts 4,246, full rate 2,710 (gpt-oss D=64 attention backward, b4)
- E29 [2026-09, gfx950, FlyDSL ?, N=6–16, in-process paired arms in Thue-Morse order on the B=4 cell (and B=2 for `soft_ilv`), each against a byte-identical control leg] two reorders that set a batch of VALU work against a run of back-to-back MFMAs: `g2_wlead` (a d-tile's four dV MFMAs issued as one group ahead of the dS block, instruction count unchanged) +0.674% against a +0.028% control leg = +0.65% slower at B=4 (1 of 6 pairs faster); `soft_ilv=4` (the P packs issued as a batch after the exp chain) +0.755% at B=4 and +0.966% at B=2 slower (0 of 16 pairs faster each); the record sums up the round's batch-type reorders as all at +0.65 to +0.97%, none a gain (gpt-oss D=64 attention backward, the kernel of E25)
- E30 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=5, in-kernel `s_memtime` clock, each arm its own process after 5 s idle] clock under `rocprofv3 --pmc` 1.009–1.011× that without; `GRBM_GUI_ACTIVE / 8 / dispatch time` within 0.6% of it; dispatches 1–5 after idle 1.60–1.90 GHz with or without the profiler, ~2.02 GHz sustained (power-capped MFMA microbenchmark)
- E31 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=9 rounds per arm in one process, A-vs-A 0.36%] six XCD maps of one binary: at Sq=4096 `SQ_WAIT_ANY` in wall order on all five separated pairs, coexec / MFMA busy 0.1507 on every arm; at Sq=2048 no separated pair while the L2 hit rate ranged 74.4–95.8% (attention-forward-shaped kernel, D=128, bf16, occupancy 1)
- E32 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=6, wall and sysfs sclk at full load; counters under `rocprofv3 --pmc`] `matrix_instr_nonkdim` 16 against 32: wall ×1.092, sclk ×1.106, 1298 against 1372 W of a 1400 W cap; `SQ_VALU_MFMA_BUSY_CYCLES` equal; `MfmaUtil` implied ×1.214 (Triton attention forward, non-causal, D=64)
- E33 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=6 process pairs (timing, then `rocprofv3 --pmc`), arm order rotated] the unchanged arm: wall (median of 210 calls) 0.34%, `SQ_WAVE_CYCLES` 0.52%, within 0.03% when profiled in the same position; fences with the next tile's global prefetch moved after the softmax: wall +3.0%, `SQ_WAVE_CYCLES` +2.0%, `SQ_WAIT_INST_LDS` −4.1% (attention-forward-shaped kernel, D=128, bf16, occupancy 1)
- E34 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=6 rounds at full load, shuffled] `__builtin_amdgcn_iglp_opt(0/1/2/3)` on one source, same work, all six pairs separated: coexec / MFMA busy ranked iglp 3 (0.315) ahead of iglp 0 (0.270), which was 3.3% faster; `SQ_WAVE_CYCLES` and `SQ_WAIT_ANY` in wall order on all six pairs (microbenchmark, K reads, MFMAs, exps and V reads, 2 waves/SIMD)
- E35 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=3, placement from `HW_ID`] one workgroup on one CU, against 1 wave: 4 waves on 4 SIMDs ×4.000 for MFMA busy, coexec, `SQ_ACTIVE_INST_VALU` and `SQ_WAVE_CYCLES`; 2 waves per SIMD ×8.0 for MFMA busy, `SQ_ACTIVE_INST_VALU` and `SQ_WAVE_CYCLES`; cycles: MFMA busy 16.0 per `v_mfma_f32_16x16x32_bf16`, coexec 4.0 per MFMA beside a 4-cycle `v_mul_f32`; quad-cycles: `SQ_WAVE_CYCLES` 0.2505 of the in-kernel cycles, `SQ_ACTIVE_INST_VALU` 1 per `v_mul_f32` (microbenchmark)
- E36 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=5, loop span in `s_memtime` cycles] a per-wave LDS read → `s_waitcnt lgkmcnt(0)` → MFMA loop at 4 waves/SIMD: `SQ_VALU_MFMA_BUSY_CYCLES / SQ_LDS_IDX_ACTIVE` 5.99, the loop 1.25× faster with its reads halved (8 → 4); `SQ_LDS_IDX_ACTIVE` one count per CU per cycle on a read-only loop, MFMA busy the sum over the 4 SIMDs (microbenchmark)
- E37 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=6, one process, four arms shuffled per round, median per-call time; sysfs power] removing the `exp2` of P and skipping the K/V loads together saved 0.947× the sum of the two apart on the full card at 1389 W of 1400 W, 1.14× on 16 CUs below the cap (Triton attention forward, non-causal, bf16, D=64)
- E38 [2026-11, gfx950 (MI355X), HIP, N=10, one process, hipEvent, median of round medians] an opcode-only hash (`grep -o "^\s*[a-z_0-9]*" | md5sum`) matched the base for one `v_lshlrev_b32` shift immediate 2 → 7 (×11.76) and eight `s_waitcnt lgkmcnt(15)` → `lgkmcnt(0)` (×2.83), each slower in 10 of 10 rounds (microbenchmark, 8 `ds_read_b32` per iteration)
- E40 [2026-11, gfx950 (MI355X), HIP, rocprofv3 1.3.5, N=6 rounds at full load, at the cap by the amd-smi PPT share] per wave a dependent L2 read, then 16 MFMAs, 8 `v_exp_f32` and 8 `v_fma_f32` in six orders, same instructions: against the lowest-coexec order, 0.39–8.79% faster at 2 waves/SIMD (lowest coexec / MFMA busy 0.161), 0.26–4.81% slower at 3 (0.261), 0.3–5.6% faster at 4 (0.324) (microbenchmark)
- E41 [2026-11, gfx950 (MI355X), HIP, N=6 rounds × 6 s per arm at full load, shuffled] 1 wave/SIMD, base 1370–1373 W of a 1400 W cap, prices from arms that added or removed MFMAs or exps: per pipe cycle (MFMA 16, exp 8) an exposed exp 0.744 and an exp in MFMA shadows 0.320 of an MFMA (microbenchmark, `v_mfma_f32_16x16x32_bf16`)
- E42 [2026-11, gfx950 (MI355X), HIP, N=6 rounds × 5 s per arm at full load, shuffled, sysfs sclk and power] 2–6 waves/SIMD set only by the dynamic LDS size, same work; of six one-wave steps with both arms at the cap, four gave back 24–51% of the cycle saving, two 104% and 119% (0.3% and 1.1% slower); on 16 CUs below the cap wall followed cycles (microbenchmark, dependent L2 reads then MFMAs, exps and FMAs)
- E43 [2026-07, gfx950, FlyDSL 0.1.1.dev409, N=2, the dq kernel timed alone at full load, arms interleaved, median of 9 blocks of 40 calls; L2 and `SQ_WAIT_ANY` from a PMC A/B] grid linearised with the q tile varying fastest instead of kv-head-major: +1.98% wall, every block slower than every base block; L2 hit rate 94.51 → 95.02%, L2 misses −9.3%, `SQ_WAIT_ANY` −4.1% (hd64 attention backward, dq kernel, B=1, Hq=128, Hkv=16, Sq=Skv=16384)
