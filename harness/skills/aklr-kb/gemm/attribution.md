---
title: "GEMM — attribution: price where the time goes before changing anything"
kind: method
lever: gemm_attribution
ops: [gemm]
gens: [gfx950]
dtypes: [fp8, mxfp4, mxfp8, bf16]
regimes: [compute-bound, feed-bound, power-capped]
flydsl: 0.3.2
---

# Attributing a GEMM's time

> Counters propose; probes decide. Numbers marked `*(field, E<n>)*` are measured values; their conditions are
> listed under Evidence at the end. Use them for direction and order of magnitude, not as targets.
> Trace capture and ATT tooling are in `../profile.md`; this card is about turning readings into a price.

## Route here when
- You need to know which part of a kernel a change could save before building it.
- A counter says "LDS-bound" or "latency-bound" and you are about to act on it.
- A change moved wall time but not the counter you expected, or the other way round.

Start from the trace: `../profile.md` § capture-kernel-trace and `../profile.md` § kernel-trace-analysis
capture and read an ATT trace, and FlyDSL's GEMM guide maps the common GEMM stalls to first actions and counts
the hot loop (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §8.4, §9)). This card is for when those readings are
ambiguous: probes that remove or repeat a part and read the time.

## 1. Two probes that classify the regime without a profiler

**Compute additivity.** Add compute with no extra memory traffic and see whether wall time moves. For
example, run the MFMA loop twice over the same operands (the result is wrong; this probe is for timing
only). Check in the ISA that the instructions really were added and not removed as dead code.
- Wall time does not move → compute is fully hidden. Drop every "fewer instructions / less VALU / fewer
  LDS issues / different occupancy" candidate at once. *(field, E1)* On an attention kernel, +76% MFMAs cost
  +0.1%.
- Wall time grows in proportion → compute is on the critical path, and removing instructions is a real
  lever.
- Do it per kernel; two kernels of one operator can sit in different regimes.
- The same idea for synchronization: add two `gpu.barrier()` per tile. If that is free, barrier count is
  not the problem; look at what the barrier protects.

**Apparent byte rate.** Divide the kernel's bytes by its wall time and compare with the stream bandwidth
you measured on the same box. *(field, E2)* torch's `copy_` 4.98 TB/s, `add(out=)` 6.25 TB/s, `sum` 3.74,
`amax` 2.01 (a reduction is not a bandwidth ruler). A rate above stream bandwidth means caches are serving
part of the traffic; the kernel is not at the HBM limit.

## 2. Subtractive probes: remove a part, read the time

Build each arm as a compile-time switch on the real kernel. Prove each arm compiled in (an output checksum
that changes). Keep probe switches out of the delivered code.

| Arm | What it does | What it prices |
|---|---|---|
| **drop** | do not issue the fills (or reads, stores) of the main body | the whole cost of that stream |
| **alias** | issue the same instructions, but point every address at one block that stays in L2 | instruction issue + LDS write, without the data movement |
| **identity** | replace a new cross-lane helper with an identity that keeps the same return shape | the request-path gain of a change, without its VALU cost |
| **illegal** | remove a wait or a step entirely (wrong results) | the ceiling of any change that shortens it (`grid_pricing.md` §7) |

- Touch only the main loop body. Keep the prologue and K-tail as they are, so LDS always holds legal data
  and the kernel cannot hang.
- *(field, E3)* On a grouped GEMM, dropping the main body's fills saved 155 µs (18.6%); aliasing them saved
  roughly half of that. The other half was issue and LDS-write cost, which deeper prefetch cannot reach.
  Without the alias arm, prefetch would have been credited with twice what it could deliver.
- **A probe gives a bound, not an achievable gain, and the bound can be wrong in either direction.**
  *(field, E4)* On an attention kernel, a "store is 30–47% of wall" ceiling turned into net losses when built as
  DMA, triple buffering or query blocking, because the removed part was doing other work (cross-wave sharing,
  latency hiding). Other changes beat their bound because they also removed a serial dependency or a spill.
  Decide only on the real implementation.
- **Instruction counts predict exactly; time does not.** *(field, E5, E6)* On a grouped GEMM, a paper count of
  1.043478× MFMAs matched `SQ_INSTS_MFMA` to five digits, while the time saved was 50–60% of the proportional
  estimate.
- **An isolated micro-benchmark can flip the sign.** *(field, E7)* In an attention backward pass, a reduce
  kernel's best workgroup shape in isolation lost 0.4% in the full pass. Use micro-benchmarks to discard bad
  options, and pick the winner in the full pipeline.

## 3. Additive probes: do it twice, read the margin

When a part cannot be removed safely, run it one extra time and fold the result back exactly.
- Fold back with an exact operation (e.g. `× 1/N` with N a power of two after a cross-wave sum) so the
  correctness gate still passes.
- Write the extra copy to its own scratch so it adds no write-after-read hazard, and place it after the
  original so the original's waits are already satisfied.
- Make the probe a compile-time parameter that appears in the kernel name, and time it against the base
  in the same process. *(field, E8, E9)* On an attention kernel, cross-process spread reached 12.9%, which
  hides a 1–4% margin.
- **The margin is an upper bound on the saving only when the kernel has no idle issue slots.**
  *(field, E9, E10)* On an attention kernel, a rendezvous that cost +1–4% as an extra copy was *slower* to
  remove (+1.2–5.9%), because its reads were covering another read's latency.
- The same trick prices a whole launch. *(field, E11)* `rocprofv3 --kernel-trace` start→end overstated a small
  kernel of a grouped GEMM 2–3.2× against its marginal cost in the stream.

## 4. At the power cap, price in watts, not instructions

A dense GEMM at the power cap trades cycles for clock. Two changes with the same cycle saving can be
worth very different wall time.
- **Build an all-removed arm and sample the clock under sustained load.** *(field, E12)* With every load, LDS
  read, barrier and scale operand removed, wall time fell 25.7%, but cycles fell only 12.6%: the clock rose
  from 1702 to 2034 MHz. The kernel was energy-bound, not issue-bound.
- **In an energy-bound kernel, the savings of separate instruction groups add up.** *(field, E13)* Removing the A
  DMA, the B loads and the A LDS reads separately saved 8.09 + 5.50 + 6.16 = 19.75%; together, 20.98%. In an
  issue-bound kernel the savings would overlap.
- **Rank candidates by what an instruction moves, not by how many you delete.** *(field, E14, E15)* Deleting
  7.4% of the hot loop's instructions, all SALU and `s_nop`, gained 0%; cutting global bytes per MAC by 20%
  gained 5.4%.
- **Cycles convert to wall at a shape-dependent rate.** *(field, E16, E17)* On a grouped GEMM, 5.3% fewer cycles
  gave 2.2% wall on one shape (42%), and 7.1% fewer gave 1.4% on another (20%). Discount cycle-only gains
  accordingly.
- **Read the clock to classify a win.** If wall time and clock improved together, the change saved energy;
  if wall time moved and the clock did not, the change moved cycles at constant energy per cycle (stalls or work
  added or removed); below the power cap every change reads this way. A clock rise alone is not an energy
  saving: a slower arm can clock higher (`../attention/attribution.md` §8).
- **Sample the clock correctly.** Use one shape under sustained load for at least 45 s and average only
  the samples taken under load. The mode of a whole benchmark run is the idle clock. Alternatively use
  `GRBM_GUI_ACTIVE / 8 / dispatch time`: the counter is summed over the 8 XCDs, and only long dispatches
  give sane values.

## 5. The wave-cycle budget

`SQ_ACTIVE_INST_ANY + SQ_WAIT_INST_ANY + SQ_WAIT_ANY = SQ_WAVE_CYCLES` (all in quad-cycles): issuing,
wanting to issue but blocked by a busy pipe, and waiting on `s_waitcnt` or `s_barrier`.
- In an MFMA-bound loop a high `SQ_WAIT_INST_ANY` is healthy. Idle time is `SQ_WAIT_ANY`; read it next to
  MFMA busy. If `SQ_WAIT_ANY` rises while MFMA busy is unchanged, the change is slower. *(field, E18)* On a
  grouped GEMM, 31.7% → 50.0% came with −26%. The sign is reliable; the slope is not.
- **This budget is the only locator for changes that keep bytes and instructions identical**, such as a
  different wave → tile assignment. *(field, E18)* A wave-grid change on a grouped GEMM lost 26% with every
  memory, LDS and MFMA counter bit-identical.
- **`SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` gates "remove instructions" candidates, per kernel.**
  *(field, E19, E20, E21)* 19% on a two-waves-per-SIMD grouped GEMM left 5× headroom, so removing instructions
  was worth nothing there; 57–60% on one occupancy-1 kernel made it the top lever; another occupancy-1 kernel,
  a grouped GEMM, read 36–41%. Measure it on your kernel. Region markers (`s_setprio`, `sched_barrier`) do not
  count as instructions saved.
- `SQ_WAIT_INST_LDS` near 1% of wave cycles rules out the whole "tune `ds_read` waits / LDS phase" family.
  It says nothing about LDS energy (§4).

## 6. LDS: latency, bandwidth, or neither

`SQ_WAIT_INST_LDS` counts dependent waits. Co-resident waves can hide them, so a high value is not proof that
LDS bounds the kernel. *(field, E22)* An attention kernel with 81% of its cycle budget in LDS waits gained 0.0%
in each of three probes that cut one LDS read stream by 17–33%. Decide with a halve-the-reads probe and ISA
counts. Compare `SQ_LDS_IDX_ACTIVE` with `SQ_VALU_MFMA_BUSY_CYCLES` to see whether the LDS port is anywhere
near busy.

## 7. Per-tile fixed cost

- **Same FLOPs, different tile counts.** Swap N and K (e.g. 8192×12288×3072 and 8192×3072×12288). MFMA
  counters come out bit-identical, so the whole time difference is the extra tiles' fixed cost, read from
  one dispatch each. *(field, E23)* That gave about 16k cycles per tile, an epilogue-dominated serial VALU
  stream.
- **Trip-count additive probe.** Add 2 and 4 iterations to the loop count in the same process, with the
  rest of the ISA unchanged. Three points give the per-phase cost P and the fixed cost F directly.
- **A K-sweep fit (`grid_pricing.md` §6) has five systematic errors:**
  1. Using the nominal clock instead of the measured one.
  2. Counting a neighbour kernel whose time also grows with K into P. Take the target kernel's row from a
     kernel trace.
  3. Different cells running at different clocks at the cap. F is latency, so it is constant in time,
     not in cycles.
  4. Tile count or tiles per workgroup changing with K.
  5. F is a residual and an order of magnitude more fragile than P.

## 8. Upper bounds for "hide it under the MFMAs"

- **A `v_mfma_f32_16x16x32_bf16` offers 8 cycles of shadow on gfx950.**
  `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` reads 16 cycles for it, and a microbenchmark shows it blocks VALU
  issue on its SIMD for 8 of them: one `v_exp_f32` or two `v_mul_f32` fit per MFMA *(field, E40)*. The same
  microbenchmark gives `v_mfma_f32_32x32x16_bf16` 32 cycles with a 24-cycle shadow; the 12-cycle shadow belongs
  to `v_mfma_f32_16x16x16_bf16` (64-bit A/B, 16 cycles, half the FLOP rate). *(field, E24, E25, E40)* The same
  16 was read on two attention kernels and by the microbenchmark; an A6W4 GEMM issuing the f8f6f4 atom on
  fp6 × fp4 operands also read 16 *(field, E26)*. Read the ratio for the MFMA
  and operand format your kernel issues: `v_mfma_f32_16x16x128_f8f6f4` with fp8 A and B reads 32 on a grouped
  GEMM *(field, E27)* (`grid_pricing.md` §5). Its fp4 form is shorter — LLVM's gfx950 hazard model
  gives it fewer wait states (`correctness.md` §7) — so read its ratio rather than assuming 32. The
  ceiling for interleaving work is the pipe cycles less the issue block, per MFMA in the phase: `8 × MFMAs` for
  the bf16 atom above.
- **The register allocator is the constraint.** *(field, E28)* An epilogue interleaved into the last chunk's MFMAs
  did interleave (one MFMA per 29 instructions), but spills went 0 → 16 and wall time lost 3.6% (peeling the last
  chunk alone: spills 0 → 5, −2.2%). Dump the ISA and check spills before timing.
- **At two or more waves per SIMD, moving VALU work between clusters rarely pays.** Other waves fill most gaps.
  *(field, E29)* On an attention kernel, moving a row sum into the P·V cluster cost 3.1% and into a memory cluster
  0.1%; deleting it gained 6.4%. The clusters are not interchangeable, though: count what each cluster's MFMAs
  already cover before moving work into it (`../attention/registers_occupancy.md` §8).
- **Price a bare VALU chain with a deletion probe before covering it.** *(field, E30, E31)* On an attention
  kernel, a chain that looked like a bubble was on the critical path: covering it with MFMAs lost 1.5%, while
  deleting the scaling step's 32 packed FMAs saved 11.25%. Count `v_exp_f32` at 8 cycles in any per-pipe
  census, as an inline-asm microbenchmark measured it *(field, E32)*.
- `v_exp_f32` (also `v_log_f32`, `v_rcp_f32`, `v_sqrt_f32`) takes 8 cycles per wave64 instruction on gfx950,
  against 4 for `v_mul_f32` from one wave; at two or more waves per SIMD plain VALU reaches 2 cycles per SIMD
  and exp stays at 8, in an inline-asm microbenchmark *(field, E32)*
  (`../attention/softmax.md` §1). Check a per-class model against `SQ_ACTIVE_INST_VALU`.

## 9. Readings you cannot trust

- rocprof's `VGPR_Count` reports half the true value, and `Accum_VGPR_Count` reports 0 *(field, E33)*. Read
  registers from the ISA (`mainloop.md` §10).
- `MfmaUtil` (`SQ_VALU_MFMA_BUSY_CYCLES` over `GRBM_GUI_ACTIVE`·SIMDs) is not reliably comparable across wave
  counts. On grouped GEMMs, a 1-wave-per-SIMD kernel read 32.2% against 63.8% for a 2-wave-per-SIMD kernel with
  the same MFMA count and wall time 4% apart *(field, E34)*, while one kernel built at 1 and 2 waves per SIMD
  reported identical MFMA busy cycles for the same work (MfmaUtil 57.3% → 48.2%) *(field, E41)*. Before comparing
  kernels with different waves per SIMD, check busy cycles against `SQ_INSTS_MFMA`, or compare time per K
  iteration.
- `SQ_LEVEL_WAVES` and `SQ_ACCUM_PREV_HIRES` read 0 in rocprofv3 on gfx950 (`SQ_ACCUM_PREV_HIRES` on an
  attention kernel) *(field, E35, E36)*. Wait counters cannot separate `vmcnt` waits from barrier waits there;
  use probes.
- In an ATT trace, an MFMA waiting for its operand is charged to the MFMA, not to an `s_waitcnt`. *(field, E37)*
  1.8% VMEM wait next to 89.5% MFMA stall was hidden memory latency.
- **Static ISA size is nearly free; a spill slot is not.** *(field, E38)* On a grouped GEMM, halving a whole-loop
  kernel's static instructions gained 0.2%, and the kernel that gained one spill slot lost 0.30–0.41% instead of
  gaining (about 0.45% per slot, estimated). Removing code can also create a spill, because the allocator's
  input changed.
- **Do not trust an arm's first read at a single repetition.** *(field, E39)* On an attention kernel such a read was
  +15% while three normal runs read ±0.2%; that run also dumped its IR, so the two causes are not separated.

In FlyDSL: `../flydsl/gemm.md` § attribution §9.

## Related
`measurement.md` (rulers, null controls, A/B design) · `grid_pricing.md` (rounds,
fixed cost, illegal arm) · `mainloop.md` · `../profile.md` § kernel-trace-analysis

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. Kernels are the FlyDSL GEMM, grouped GEMM and attention kernels of the campaigns named on each line;
"attention" and "grouped GEMM" in a claim mark the numbers the card states for another kind of kernel.

- E1 [2026-09, gfx950, FlyDSL 0.3.2, N=?, in-process probe (73 s, 0.39% spread), arm checked in the ISA] adding PV MFMAs (21 → 37 per tile, +76%) with no extra memory traffic read 740.6 µs, +0.1% against the control (sparse-MLA prefill attention kernel, T=4096, topk 2048, inference2 aiter fork; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-09/agent.txt:44)
- E2 [2026-09, gfx950, torch (no FlyDSL kernel), N=?, torch stream ops on the measuring box] `copy_` 4.98, `add(out=)` 6.25, `sum` 3.74, `amax` 2.01 TB/s (torch on one large buffer, measured in the sparse-MLA campaign; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-09/agent.txt:80)
- E3 [2026-08, gfx950, FlyDSL ?, N=6, isolated probe, PROBE_ROUNDS=6, palindrome A/B, median] dropping the four main-loop g2s issues: 865.1 → 703.9 µs (18.58%, 155 µs); aliasing them to k-block 0 with the same instructions: 774.0 µs (10.58%), split as 84 µs data movement and 71 µs issue, addressing and LDS write (gpt-oss-20b down-projection padN dgrad NN, grouped tensorwise fp8, G=32, M=131072, K=N=2944; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/remote_data.md:12)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, subtractive probes against built implementations] SKIPST priced the exposed store at 30% (cr128) to 47% (flash cr4) of wall, and the DMA, triple-buffer and query-blocking builds were net losses (DeepSeek-V4 sparse-MLA attention forward; KB notes, no run record on this machine; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/03-profiling-utilization.md:588)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] `SQ_INSTS_MFMA` mx/tw = 36,175,872 / 34,668,544 = 1.04348, the paper count 12/11.5 (gpt-oss MXFP8 grouped GEMM NT against the tensorwise fp8 grouped GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-05/agent.txt:11)
- E6 [2026-07, gfx950, FlyDSL ?, N=?, campaign score across rounds] the boundary-quadrant skip realized ~50–60% of its proportional time saving in three earlier rounds (same grouped GEMMs; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-05/agent.txt:54)
- E7 [2026-08, gfx950, FlyDSL ?, N=2, isolated sweep against the full-backward harness] the dQ reduce kernel's workgroup (1024,8) won the isolated sweep at 557.2 µs and read 874.94 / 876.02 TF/s in the full backward, −0.4% against (512,2) (gpt-oss fused flash-attention backward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:114)
- E8 [2026-09, gfx950, FlyDSL ?, N=?, cross-process repeats of one configuration] one configuration ranged 12.9% across processes; the campaign trusts only same-process ABBA (sparse-MLA decode and prefill ruler; ruler note, the runs are not recorded; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/memory.md:40)
- E9 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-process ABBA, probe as a compile-time parameter in the kernel name] one extra cross-wave rsum rendezvous per tile: seq48 +0.207 µs (+1.37%), seq60 +0.169 µs (+1.01%), seq84 +0.887 µs (+4.00%) (GLM-5.2 TP4/EP4 sparse-MLA decode producer; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-02/agent.txt:21)
- E10 [2026-09, gfx950, FlyDSL ?, N=?, ?] deferring the rsum fold, which drops the four `ds_read_b32` after the barrier, was 1.2% slower at seq 48, 2.0% at seq 60 and 5.9% at seq 84 (same decode producer; source comment, aiter fork /workspace/code/gpt_oss_docker/sync/inference2/aiter commit 4e80e6619, aiter/ops/flydsl/kernels/sparse_mla_decode.py:433)
- E11 [2026-09, gfx950, FlyDSL 0.2.4, N=9, same-process additive probe (0/1/2/4 extra launches, 9 alternations, median slope) against rocprofv3 --kernel-trace Start→End] an early-returning output-tail kernel: 1.579 / 2.006 µs in the stream against 5.10 / 5.08 µs Start→End (3.2×); the scale-preshuffle kernel `kern_0`: ≈10.2 µs against 20.32 µs (≈2×) (gpt-oss MXFP4 grouped MoE MLP, fc2_fwd and fc1_dgrad; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/memory.md:331)
- E12 [2026-10, gfx950, FlyDSL ?, N=?, clock sampled after 75 s of back-to-back launches] shipped 580.3 µs at 1702 MHz / 1400 W; with every load, LDS read, barrier and scale operand removed 431.1 µs at 2034 MHz / 1380 W: wall −25.7%, cycles 987.7k → 876.9k (−12.6%) (A6W4 forward GEMM, linear1 training shape; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-04/agent.txt:80)
- E13 [2026-10, gfx950, FlyDSL ?, N=15, same-process palindrome pairs, A/A floor ±0.03%] removing the A DMA −8.09%, the B loads −5.50%, the A LDS reads −6.16% (sum 19.75%); the three together 20.98% (A6W4 forward GEMM, linear1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-04/agent.txt:79)
- E14 [2026-10, gfx950, FlyDSL ?, N=3, campaign score (training-time-weighted TF/s over six shapes), three reads per arm] soffset/imm_offset addressing: hot loop 949 → 879 instructions (SALU 98 → 52, `s_nop` 42 → 18; ds_read, vmem and s_waitcnt unchanged); score 3964.08 / 3954.31 / 3951.09 against 3950.12 / 3952.65 / 3956.84, 0% (A6W4 forward GEMM, six training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:178)
- E15 [2026-10, gfx950, FlyDSL ?, N=?, campaign score] tile (256,384) on four shapes: global bytes per MAC 0.625 → 0.500 (−20%), score 4167.19 (+5.43%) (same A6W4 forward GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:182)
- E16 [2026-07, gfx950, FlyDSL ?, N=2, power and clock probe, palindrome mx/tw/tw/mx, 14 s sustained drive] cycle ratio tw/mx 1.0315 → 1.0858 (+5.3%) against wall 1.0172 → 1.0392 (+2.2%), ~1397 W (gpt-oss MXFP8 grouped GEMM, fwd down balanced; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:216)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, same probe] mx 2905 against tw 3110 cycles (7.1%), wall 1.6014 against 1.6241 (1.42%), sclk 1814 against 1915 MHz, both at 1400 W (same grouped GEMM, dgrad gate_up heavy; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/memory.md:248)
- E18 [2026-09, gfx950, FlyDSL 0.2.2, N=?, rocprofv3 PMC and wall] a `wave_id → (wave_m, wave_n)` remap: wall −25.9% (−26.5% on the shipped 16-wide kernel); `SQ_WAIT_ANY/SQ_WAVE_CYCLES` 31.7% → 50.0%; TCP/TCC request, hit, miss and DRAM counters, `SQ_INSTS_MFMA` and MFMA busy bit-identical (≤0.3%), and per the KB note the LDS counters too (gpt-oss-20b grouped per-tensor FP8 GEMM, B=24, 2 waves/SIMD; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-13/agent.txt:33)
- E19 [2026-09, gfx950, FlyDSL 0.2.2, N=?, rocprofv3 PMC] `SQ_ACTIVE_INST_ANY/SQ_WAVE_CYCLES` = 19.2% at 2 waves/SIMD (same grouped FP8 GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-13/agent.txt:17)
- E20 [2026-09, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] at occupancy 1: 56.70% (FC1_fwd) and 59.48% (out_proj_wgrad) (MXFP4 dense GEMM, Llama training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/rounds/round-08/agent.txt:25)
- E21 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 PMC] at occupancy 1: ACTIVE 35.58% (fc2_fwd) and 41.22% (fc1_dgrad); `SQ_WAIT_INST_LDS` 0.99% and 0.78% (gpt-oss MXFP4 grouped MoE MLP, NT `kern_1`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/rounds/round-12/agent.txt:37)
- E22 [2026-08, gfx950, FlyDSL ?, N=7, interleaved A/B] `SQ_WAIT_INST_LDS` ≈ 81% of the CU-cycle budget (memory.md:30); cutting the lse/delta `ds_read_b128` 384 → 320, the GEMM1 A fragments 384 → 256 or the GEMM2 transpose reads 1536 → 1280 each read 0.0% (3/7, 3/7, 2/7 faster) (gpt-oss fused flash-attention backward, dkdv; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260804_003925/memory.md:51)
- E23 [2026-10, gfx950, FlyDSL ?, N=?, rocprofv3 PMC, one dispatch each] fc1 8192×12288×3072 against fc2 8192×3072×12288: MFMA busy cycles and `SQ_INSTS_MFMA` bit-identical, 148.2 against 123.6 µs, ≈16.3k cycles per extra tile, split ACTIVE +27.5% / WAIT_INST +60.2% / WAIT_ANY +12.3% (A6W4 forward GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/rounds/round-12/agent.txt:9)
- E24 [2026-09, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] `SQ_VALU_MFMA_BUSY_CYCLES/SQ_INSTS_MFMA` = 16.00 (gpt-oss D64 fused attention backward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_112555/memory.md:354)
- E25 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 PMC] MFMA pipe depth 16.00 cycles per instruction (gpt-oss D64 attention backward, second campaign; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:364)
- E26 [2026-10, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] `SQ_VALU_MFMA_BUSY_CYCLES/SQ_INSTS_MFMA` = 16.00 (A6W4 forward GEMM with a fused GELU and quantization epilogue; it issues the scaled f8f6f4 MFMA on fp6 × fp4 operands, not the bf16 atom; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/memory.md:346)
- E27 [2026-09, gfx950, FlyDSL 0.2.2, N=?, rocprofv3 PMC] `v_mfma_f32_16x16x128_f8f6f4` with fp8 A and B: `MFMA_BUSY/INSTS_MFMA` = 32.00 on both kernel skeletons (gpt-oss-20b grouped per-tensor FP8 GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151102/rounds/round-14/agent.txt:19)
- E28 [2026-10, gfx950, FlyDSL ?, N=1, campaign score of the fused GEMM + GELU + quantization chain] peeling the last chunk: 2607.84 (−2.2%), spills 0 → 5; adding the 1:1 accumulator-quad MFMA interleave: 2568.97 (−3.6%), spills 16; the last 200 MFMAs sat 29.1 instructions apart (A6W4 forward GEMM with fused epilogue; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261012_143244/memory.md:313)
- E29 [2026-07, gfx950, FlyDSL ?, N=?, ?] at 3 waves/SIMD, moving the row sum out of the QK cluster: into the PV cluster −3.08%, into the memory cluster −0.10%; deleting it +6.4% (hd64 flash-attention forward; KB note, no run record found; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/03-profiling-utilization.md:731)
- E30 [2026-09, gfx950, FlyDSL ?, N=2, probe mean] splitting the bare 16-PKFMA chain around an MFMA32 with `sched_barrier(0)`: d64 2.10848 / 2.10980 → 2.14013 / 2.14087 ms (−1.5%) (gpt-oss D64 attention forward, 3 waves/SIMD; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/rounds/round-18/agent.txt:17)
- E31 [2026-09, gfx950, FlyDSL ?, N=?, probe mean, identity replacement] `noscale` (the score-scaling chain, 32 `v_pk_fma_f32`, replaced by an identity): d64 2.10726 → 1.87011 ms (−11.25%), spills 22 → 13 (same kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:451)
- E32 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] `v_exp_f32` (and log, rcp, sqrt) 8.000 cycles per wave64 instruction, `v_mul_f32` 4.000; at 2 and 4 waves/SIMD exp stays at 8.0 per SIMD while mul reaches 2.0 (microbenchmark; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:23)
- E33 [2026-07, gfx950, rocprofv3, N=?, rocprofv3 kernel CSV against the ISA] `VGPR_Count` read 128 for an 8-wave whole-loop GEMM with 256 VGPRs, and half the true value on an attention backward (2026-07-27); `Accum_VGPR_Count` always 0 (KB note; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/03-profiling-utilization.md:347)
- E34 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] MfmaUtil 32.2% for a 4-wave (1 wave/SIMD) kernel against 63.8% for an 8-wave (2 waves/SIMD) kernel with equal `SQ_INSTS_MFMA` and wall 4% apart (gpt-oss tensorwise fp8 and MXFP8 grouped GEMM wgrad; KB note; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/06-register-wholeloop-emit.md:131)
- E35 [2026-09, gfx950, FlyDSL ?, rocprofv3, N=?, rocprofv3 PMC] `SQ_LEVEL_WAVES` always read 0 (MXFP4 dense GEMM, Llama training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/rounds/round-11/agent.txt:116)
- E36 [2026-09, gfx950, FlyDSL 0.3.2, rocprofv3, N=?, rocprofv3 PMC] `SQ_LEVEL_WAVES` and `SQ_ACCUM_PREV_HIRES` always read 0 on that build (GLM-5.2 TP4/EP4 sparse-MLA decode attention; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260912_134949/rounds/round-02/agent.txt:30)
- E37 [2026-06, gfx950, FlyDSL ?, N=1, ATT trace of one dispatch (rocprofv3, dispatch 538)] share of stall cycles: MFMA 89.5%, VMEM load 2.7%, barrier 2.5%, VMEM wait (`s_waitcnt vmcnt`) 1.8%, VMEM store 1.1%, LDS 1.0% (MXFP4 4-wave whole-loop dense GEMM, M=N=8192, K=28672, BN256 BK256; trace report of 2026-06-17, myskill /workspace/code/gpt_oss_docker/myskill commit 2fef875, mxfp4-4wave-optim/08-att-root-cause.md:28)
- E38 [2026-09, gfx950, FlyDSL 0.2.2, N=7, palindrome, two arms] removing the duplicate tile bodies that never run (ISA −49%): gate_up +0.23% / +0.15% (7/7 draws), down −0.41% / −0.30% (0/7) with `vgpr_spill_count` 0 → 1; the round derives "+0.2% per halving, ≈ −0.45% per spill slot" (gpt-oss fp8 per-tensor grouped wgrad, occupancy 1, 4 waves; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_153457/rounds/round-15/agent.txt:15)
- E39 [2026-09, gfx950, FlyDSL 0.3.2, N=3, same-process probe] an arm's first read under `FLYDSL_DUMP_IR=1` and `REPS=1`: +15.44%; three normal re-runs ±0.2%; the read had both the IR dump on and a single repetition, so it does not tell which of the two caused the gap (sparse-MLA decode producer; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-10/agent.txt:15)
- E40 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] `v_mfma_f32_16x16x32_bf16` 16.000 and `v_mfma_f32_32x32x16_bf16` 32.000 cycles per instruction, dependent through srcC or independent; `v_mfma_f32_16x16x16_bf16` 16.0 at half the FLOP rate of 16x16x32; with k VALU after each MFMA, cycles = max(pipe, X + 8·k_exp + 4·k_mul), X = 8 for 16x16x32 and 32x32x16 and 4 for 16x16x16, so the shadow is 8 cycles (one exp or two mul), 24 and 12 (microbenchmark; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:82)
- E41 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC (`SQ_VALU_MFMA_BUSY_CYCLES`, `GRBM_GUI_ACTIVE`, `SQ_WAVES`)] one whole-loop kernel built with 4 waves (1 wave/SIMD) and 8 waves (2 waves/SIMD): MFMA busy cycles identical (612.0B), waves 20.2M → 40.5M, `GRBM_GUI_ACTIVE` 10.68B → 12.69B, MFMA util 57.3% → 48.2%, wall +19% (MXFP4 grouped GEMM `kern_1`, GEMM only, M=N=K=4096; KB note, no run record on this machine; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_grouped_fp4vs8_occ.md:46)
