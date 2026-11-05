---
title: "Attention — attribution: the true clock, which counter ranks which change, stall pools and probes"
kind: method
lever: attn_attribution
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, decode, occupancy-1, power-capped]
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
- Profiling changes the clock it measures: *(field, E2, E3)* ~1.82 GHz derived under counter collection against ~2.07 GHz
  under sustained load without it. Compare arms under the same collection mode.
- A roofline needs the true clock. The bf16 MFMA roofline is `256 CU × 4 SIMD × 1024 FLOP/cycle × sclk`
  (`v_mfma_f32_16x16x32_bf16`: 16384 FLOP in 16 cycles). Utilisation computed at the nominal clock reads ~20%
  low *(field, E4)* and turns "energy-bound" into "room left".

## 2. Which counter ranks which change

| Change | Rank it with | Blind counters |
|---|---|---|
| grid, dispatch order, XCD placement, bytes | `SQ_WAIT_ANY`, `SQ_VALU_MFMA_COEXEC_CYCLES` | L2 hit rate (can rise with zero wall effect) |
| schedule of an occupancy-1 body (anchors, fences, hoists) | `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_LDS` | MFMA busy |
| the same instructions in a new order | `SQ_VALU_MFMA_COEXEC_CYCLES` with `GRBM_GUI_ACTIVE` | `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_ACTIVE_INST_VALU` |
| MFMA atom at the power cap | wall, sclk and power together | MFMA busy, MFMA utilisation |
| an `iglp_opt` strategy or a software-pipelining choice | co-execution | instruction and register counts |

- **`SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` is a constant**, the pipe cycles of the instruction mix (16.000
  for `v_mfma_f32_16x16x32_bf16`). Nine arms whose wall time spanned 76% read 16.000 bit-identically *(field, E5)*.
  It cannot see the atom's energy, the instruction order, or how well the MFMAs are covered.
- **Rulers for occupancy-1 body scheduling.** *(field, E6, E7)* `SQ_WAVE_CYCLES` repeated within 0.20–0.64% and
  `SQ_WAIT_INST_LDS` within 0.1%, against 0.62% for the benchmark score (2.0% for one cell's candidate time), all
  A-vs-A on an unchanged tree; both counters exclude the host and the neighbours. Rank with them, then close with
  the benchmark.
- **Co-execution is the only counter that ranked three `iglp_opt` strategies in wall order** *(field, E8)*: MFMA
  busy was bit-identical across them, and −351 instructions elsewhere on the same body was worth 0.0% *(field, E9)*.
- **Normalise before comparing.** `SQ_VALU_MFMA_BUSY_CYCLES` and `SQ_VALU_MFMA_COEXEC_CYCLES` are summed over
  the 4 SIMDs, `SQ_ACTIVE_INST_VALU` and `SQ_WAVE_CYCLES` are not: ratios such as coexec / MFMA busy compare
  directly, absolute shares need ÷4.
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
  co-execution gain; doubling co-execution alone still required stall to fall 17%. Every route crosses the
  stall pool; price it first.
- The overlap headroom to report is `busy_union` shrinking to the larger pipe with waits unchanged (an estimated 1.39× on
  that body), not `(MFMA% + VALU%) / max` (1.83× there), which assumes zero waiting and no existing
  overlap.

## 4. Wait counters are not the bound
- `SQ_WAIT_ANY` counts wave-cycles, not idle SIMD cycles: at ≥2 waves per SIMD another wave issues while one
  waits. Use the `WAIT_*` family to name the dependency, never as a recoverable percentage; `MfmaUtil` 66% is not
  "34% left" either.
  - *(field, E12)* A forward read `SQ_WAIT_ANY` 22%; deleting every `vmcnt` wait, every `lgkmcnt(0)` or every
    `s_barrier`, one class per arm, moved wall time ≤2.6% in total.
- `SQ_VALU_MFMA_BUSY_CYCLES` at least 4–5× `SQ_LDS_IDX_ACTIVE` means the LDS port is not the limit, whatever
  `SQ_WAIT_INST_LDS` says (`../gemm/attribution.md` §6).

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
- The pools overlap; do not add them.
- *(field, E20, E17)* One forward ranked: exp/transcendental 7.8% > LDS reads 5.8% > K/V DMA 3.95% > per-workgroup fixed
  cost 1.4% > grid and tail ≤0.2% > rendezvous ≤0.25%; no counter had suggested that order.
- An arm whose opcode sequence hashes identical to the base is inert: drop it without benchmarking.

## 7. Cycles to wall
- **The conversion rate depends on direction.** *(field, E21)* A controlled pair that added 4.04% cycles cost 3.69%
  wall (91%); *(field, E22)* a change that removed 1.4% cycles without removing work returned 0.12–0.24% wall (8–17%). Measure
  the rate in the improving direction on your own kernel before pricing a cycle-only candidate.
- Do not apply a cycle→wall discount to a number already measured in wall time. *(field, E23)* That double discount
  turned an 8.7% pool into 2.5% and closed the largest axis of a kernel.

## 8. At the power cap
- **At ≥99% TBP, wall ≈ total energy / TBP.** A pure reorder does not change the energy; it trades power per cycle
  against cycles, so sclk and `GRBM_GUI_ACTIVE` move in opposite directions and wall barely moves. *(field, E24)* Five
  reorder arms, four with more interleave and one with the exps isolated, all fell 1.46 to 3.39% below the deployed
  order, which sat at the top of the band. One instruction-identical pair: co-execution −9.86%, cycles +4.39%,
  sclk +3.6%, 1.87% slower.
- **Three price classes:** deleting work pays ~100%; deleting cycles without work pays ~15%; reordering pays
  0 ± 2% on a forward whose co-execution covered 44% of SIMD cycles, 52% of MFMA busy *(field, E24)*; on a backward at
  17% coexec / MFMA busy pure reorders still gained 0.3–1.3% *(field, E25)*, while two reorders that set a batch of
  VALU work against a run of back-to-back MFMAs lost 0.65–0.97% on the B=4 and B=2 cells *(field, E29)*. Report
  coexec / MFMA busy with the class, and name the class before proposing a change.
- **A slower arm clocks higher.** *(field, E19)* Running a row-sum MFMA twice added 11.1% MFMA-pipe work: 4.4% slower,
  sclk +3.9%. Count a change as an energy win only when wall and clock improve together; a clock rise alone is
  not one.
- **Transcendentals are the power-heavy instructions on these bodies.** *(field, E23)* Removing the 64 `v_exp_f32` of a
  loop gained 8.7% wall with sclk +3.1%. Rank by trans instruction count, not by MFMA cycles; whether a byte change
  is a cycle or an energy change must be measured (`grid_dispatch.md` §9).
- **An occupancy gain is partly paid back as clock.** *(field, E26)* 2.985 → 3.978 waves per SIMD: power unchanged,
  sclk −3.3%, 0.5% faster from 3.9% fewer cycles. Report sclk next to every occupancy candidate; below the
  cap, measure whether the cycle gain survives before assuming it does.

## 9. Per-kernel additive probes
Run the compute-additivity probe (`../gemm/attribution.md` §1) on each kernel of an operator, not once for
the operator. *(field, E27)* Adding MFMA work to a sparse-MLA prefill changed wall time by 0.1% (fully hidden); quadrupling
the PV MFMAs in its decode producer made it 7% slower (~6 wall cycles per 16-cycle `16x16x32` MFMA, about a third of
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

Every workload below is an attention kernel. A FlyDSL version is given only where the cited run, or another round in the
same container and venv, printed it (`flydsl.__version__`, an import path with its version, or a `pip install flydsl==`
in the run's own setup); otherwise `FlyDSL ?`. Months are those of the round timestamps (`run.jsonl`), or of the myskill
commit where only myskill records the number. A sign on a wall-time change is the record's own.

- E1 [2026-09, gfx950, FlyDSL 0.2.4, N=?, `rocm-smi` sampled while a short benchmark ran] `rocm-smi` read the idle 158 MHz because the JIT-cached kernel had already finished; `GRBM_GUI_ACTIVE ÷ 8 XCD ÷ dispatch time` gave ~1823 MHz (gpt-oss D=64 attention backward, Hq=64, Hkv=8, S=8192, causal, cell b4; campaign 20260911_114527 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:376)
- E2 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 `--pmc`, clock derived from `GRBM_GUI_ACTIVE`] 40,318,962 cycles per dispatch ÷ 8 XCD = 5,039,870 cycles in 2.764 ms ⇒ ~1823 MHz under counter collection (same kernel and round; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:364)
- E3 [2026-09, gfx950, FlyDSL 0.2.4, N=8, sustained load (20000 / 80000 calls), sclk and socket power sampled 8 times, no profiler] b4 2070 ± 10 MHz at 1399 ± 2 W, b1 2072 ± 4 MHz at 1400 ± 2 W; the record calls the 1823 MHz figure a profiling artefact (same kernel, cells b4 and b1, the round-8 replan; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/goal.md:162)
- E4 [2026-07, gfx950, FlyDSL 0.2.2, N=?, sclk and power read under sustained load] under load 1713 MHz at 1346 W (96% of the 1400 W limit): the on-clock bf16 peak is 1796 TF, not the nominal 2177 TF, so 1132 TF is 63% utilisation, not 51% (hd64 attention forward, Sq=Skv=16384, B=1, full causal; campaign 20260728_082734 round 11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-11/agent.txt:3)
- E5 [2026-09, gfx950, FlyDSL ?, N=?, rocprofv3 `--pmc`] `SQ_VALU_MFMA_BUSY_CYCLES` / `SQ_INSTS_MFMA` = 6.7686e9 / 4.2303e8 = 16.000, bit-identical in nine arms (`g2_hs`, `g3d`, `g3_sb` settings) whose wall spanned 76% (FlyDSL dualwave_swp attention backward, D=128 cell, bf16, causal; campaign 20260919_155238; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/goal.md:43)
- E6 [2026-09, gfx950, FlyDSL ?, N=3–5, A-vs-A on an unchanged tree: 5 PMC samples, 3 bench runs] `SQ_WAVE_CYCLES` 0.20–0.64%, `SQ_WAIT_INST_LDS` <0.1%, the bench score 0.62% (3-run range) (same campaign; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260919_155238/goal.md:28)
- E7 [2026-09, gfx950, FlyDSL ?, N=3, three same-session reads] the d128 cell's candidate time (`cand_ms`) spread 2.0% on an unchanged tree; recorded only in myskill, commit fb1084e4 (same campaign; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:2707)
- E8 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, three builds differing only in the `iglp_opt` id] `MFMA_BUSY` bit-identical; COEXEC / MFMA busy 19.24 / 15.55 / 14.60% in the known wall order, while the wall separated them by 0.85% against a 0.8% noise floor (d64 attention backward with fp8 operands, dkdv kernel; campaign 20260828_090905 round 17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-17/agent.txt:6)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench, single-variable arms] −351 instructions / −234 shuttle / −8 dwords / spill 1→0 = 0.0% (same kernel; campaign 20260828_090905 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-09/agent.txt:12)
- E10 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, one trace] dkdv COEXEC / MFMA busy 19.2% against 48.1% for the forward kernel in the same trace; `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES` 31.5 resident waves per CU for the forward against 7.7 for dkdv (same kernel and round as E8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-17/agent.txt:27)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] dkdv MFMA busy 44.0%, VALU busy 36.4%, coexec 8.5% ⇒ busy-union 71.9%, 28.1% in pure wait, perfect overlap 1.39× (computed). The stall each route to 1.155× needs (5.2924 → 4.5813 ms; 0.281 → 0.147 of the timeline with coexec unchanged; with coexec doubled to 0.170 the stall must still fall 17%) is arithmetic on this triple, in local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/goal.md:346 (same kernel and round as E8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-17/agent.txt:35)
- E12 [2026-09, gfx950, FlyDSL ?, N=?, subtractive arms bracketed by same-binary controls (drift d64 +0.00%, d128 −1.06%)] `SQ_WAIT_ANY` 22%; deleting all `vmcnt` waits −0.27% / −0.32%, all `lgkmcnt` +0.68% / +0.10%, all `s_barrier` −2.32% / −0.27% (d128 / d64): all waits together ≤2.6% of wall (FlyDSL dualwave attention forward, D=128 and D=64 cells; campaign 20260924_001030 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:229)
- E13 [2026-08, gfx950, FlyDSL ?, N=?, campaign bench (~75 s)] fused body 6.78 ms; stores aimed at an out-of-bounds SRD 5.70; stores and GEMM3 removed 4.91; the reduce also removed 4.18 ms; a single-layer probe books 72% of the compute time as bytes (gpt-oss fused attention backward, D=128, Hq64/Hkv8, Sq=Skv=8192, B=2, THD, full causal; campaign 20260810_114422 round 3; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260810_114422/rounds/round-03/agent.txt:18)
- E14 [2026-07, gfx950, FlyDSL ?, N=?, ?] deleting 32 exps also deleted 4 MFMAs and 16 cvts and inflated the probe from +8% to +21.6%; no run record was found; month of the myskill commit 587b6182 (hd64 attention forward; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:736)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, ?] skipping the per-tile K/V DMA (tile reused) +8.7%; the built double-buffered prefetch (LDS 16→32 KB, occupancy unchanged) −4.3~4.7%; no run record was found; the entry is dated 2026-07-21 (Meta hd64 attention backward, dq kernel, fast-exp path; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:158)
- E16 [2026-09, gfx950, FlyDSL 0.2.4, N=?, campaign per-cell ruler against the session zero point] all barriers replaced by `sched_barrier(0)` (wrong result) +3.77%, the B=1 cell of a probe that read ×1.0377 / ×1.0336 / ×1.0020 at B=1 / 2 / 4 (local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/goal.md:116); 8 barriers added (`HS_WAR_BAR`, everything else unchanged) b1 1.0160 / 1.0151 = −0.78 / −0.87% ⇒ 0.10% per barrier; legal `qdo_ring` 16→9 barriers (registers held, SNR 48.7–49.2 dB) b1 −1.73% / b2 −1.63% / b4 −1.20% (gpt-oss D=64 attention backward, a16 dQ path, occupancy 1, 4 waves; campaign 20260911_114527 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-09/agent.txt:15)
- E17 [2026-09, gfx950, FlyDSL ?, N=2, sustained b4 ruler (≥14 s, 26 samples, 4 burn arms, same-binary controls bracketing)] barriers per iteration 2→1: +0.1236% / +0.1841% (two sites); 2→0: −0.2472%; deleting either single barrier dropped `snr_fwd` 51.7 → 23.0 dB; the legal form doubles the LDS buffers 2 → 4 (68096 B/WG) (gpt-oss D=64 attention forward; campaign 20260901_123110 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260901_123110/rounds/round-13/agent.txt:18)
- E18 [2026-08, gfx950, FlyDSL ?, N=3, palindromic, one process per arm, ISA-gated] once the `sched_barrier(0)` fences around `s_barrier` were gone, deleting both hot-loop barriers read +0.10% (3/3), against +0.97% per barrier recorded on the same kernel before (round 5) (gpt-oss D=64 attention forward; campaign 20260831_084030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:279)
- E19 [2026-09, gfx950, FlyDSL ?, N=2, replicated in reversed order at a pinned 1400 W] `dup` (each row-sum MFMA run twice on the same accumulator, register-neutral): MFMA pipe 1152 → 1280 cycles (+11.1%), 1.9182 ms (−4.4%) at 1684 MHz (+3.9%); six rounds of removal arms all spilled (93/94/101/118/198/217) (gpt-oss D=64 attention forward; campaign 20260831_084030 round 17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:319)
- E20 [2026-08, gfx950, FlyDSL ?, N=3, palindromic arms bracketed by controls, one process per arm, ISA-gated] stall census: K/V DMA +3.95% (3/3), LDS reads 5.8% (V 3.6 / head-K 1.54 / tail-K 0.64), exp/trans 7.8%, fixed cost 1.4%, grid ≤0.2%, rendezvous 0.10% (same kernel; campaign 20260831_084030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:281)
- E21 [2026-09, gfx950, FlyDSL ?, N=?, controlled pair, PMC, sclk and power beside the wall] a duplicated K/V DMA (`buffer_load` 4 → 8, ISA otherwise identical): `SQ_WAVE_CYCLES` +4.04%, wall 1.8353 → 1.9030 ms (−3.69%), sclk 1625 → 1620 MHz at 1400 W, MFMA busy bit-identical (same kernel; campaign 20260831_084030 round 18; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:333; table in local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1590)
- E22 [2026-09, gfx950, FlyDSL ?, N=?, scored ruler with bracketing controls; sustained probe] `kh1,nf1`: `GRBM_GUI_ACTIVE` −1.435%, wall +0.12% (scored) / +0.24% (sustained) ⇒ 8–17% paid back (gpt-oss D=64 attention forward; campaign 20260901_123110 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260901_123110/rounds/round-02/agent.txt:24)
- E23 [2026-09, gfx950, FlyDSL ?, N=2, replicated at a pinned 1400 W] `noexp` (the 64 `v_exp_f32` removed, consumer graph intact): 1.6881 ms = 1304 TF/s (+8.7%) at 1671 MHz (+3.1%); a cycles→wall factor applied to that wall number had demoted the pool from 8.7% to 2.5% (gpt-oss D=64 attention forward; campaign 20260831_084030 round 17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:319)
- E24 [2026-09, gfx950, FlyDSL ?, N=3, instruction-identical pair, three batches reproduced to 0.01%, PMC and sclk] `sa` (only the instruction order changes): coexec −9.86% (43.64% → 37.68% of SIMD cycles; 52% of MFMA busy before, computed from the two counters), cycles +4.39%, sclk +3.6%, wall −1.87%; of the round's 14 arms the reorder arms were four that added interleave, −1.46 / −1.55 / −2.44 / −3.39%, and `sa`, which isolated the exps, −1.87%, with the deployed order at the top (myskill sums this up as all 14 arms inside ±3.4%); classes 100% (`noexp` +8.7%) / ~15% / 0 ± 2% (gpt-oss D=64 attention forward; campaign 20260901_123110 round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260901_123110/rounds/round-02/agent.txt:24; table in local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/03-profiling-utilization.md:768)
- E25 [2026-09, gfx950, FlyDSL ?, N=?, campaign ruler] COEXEC / MFMA busy 16.88% on the backward; pure-reorder arms −1.33% / −0.84…−0.99% / −0.34% of wall, all faster (gpt-oss D=64 attention backward; local:/workspace/code/gpt_oss_docker/myskill/optimizer/00-decision-index.md:339, which cites campaign 20260911 round 12; run records for 16.88% local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:175 and for −1.33% local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/memory.md:184; the −0.84…−0.99% was not found in a run record)
- E26 [2026-07, gfx950, FlyDSL 0.2.2, N=3, interleaved A/B, sclk and power on both arms] occupancy 2.985 → 3.978 waves/SIMD: 1352 → 1353 W, sclk 1710 → 1654 MHz (−3.3%), wall +0.50% = cycles saved +3.9% less the clock given back (hd64 attention forward, Sq=Skv=16384, B=1; campaign 20260728_082734 round 14; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:249)
- E27 [2026-09, gfx950, FlyDSL 0.3.2, N=3, compute-additivity probes, three decode shapes] prefill: MFMA +76% → +0.1% wall; decode producer: PV's 16 `mfma_f32_16x16x32` ×4 → +7.03 / +7.75 / +6.79%, about 6 wall cycles per MFMA (about a third of its pipe time) (aiter's FlyDSL sparse-MLA prefill and decode kernels, topk 2048; campaign 20260911_143318 rounds 9–10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-10/agent.txt:22)
- E28 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 PMC, per-class model] `SQ_ACTIVE_INST_VALU` modelled with exp at 2 quad-cycles (8 cycles): 3,218 against 3,246 measured (0.9%); a quarter rate predicts 4,246, full rate 2,710 (gpt-oss D=64 attention backward, b4; campaign 20260911_114527 round 7; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:363)
- E29 [2026-09, gfx950, FlyDSL ?, N=6–16, in-process paired arms in Thue-Morse order on the B=4 cell (and B=2 for `soft_ilv`), each against a byte-identical control leg] two reorders that set a batch of VALU work against a run of back-to-back MFMAs: `g2_wlead` (a d-tile's four dV MFMAs issued as one group ahead of the dS block, instruction count unchanged) +0.674% against a +0.028% control leg = +0.65% slower at B=4 (1 of 6 pairs faster); `soft_ilv=4` (the P packs issued as a batch after the exp chain) +0.755% at B=4 and +0.966% at B=2 slower (0 of 16 pairs faster each); the record sums up the round's batch-type reorders as all at +0.65 to +0.97%, none a gain (gpt-oss D=64 attention backward, the kernel of E25; campaign 20260910_112555 round 10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/rounds/round-10/agent.txt:35; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/rounds/round-10/agent.txt:20; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/rounds/round-10/agent.txt:22)
