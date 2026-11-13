---
title: "GEMM — the main loop: 2-stage ping-pong, 8-wave pipeline, 4-wave designs"
kind: topic
lever: gemm_mainloop
ops: [gemm, grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [fp8, int8, mxfp4, mxfp8, bf16, fp16]
regimes: [large-M, large-K, "M 256-4096 (bf16)"]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/gemm/gemm_mxfp4_kernel.py
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/gemm_a16w16_gfx950.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_8wave.py
  - ROCm/FlyDSL@dd8373349:kernels/gemm/fp8_gemm_4wave.py
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
---

# The GEMM main loop

> Three loop designs are in use for GEMMs on gfx950: FlyDSL's 2-stage LDS ping-pong
> (`preshuffle_gemm`, explained step by step in `../flydsl/guides.md` § gemm-optimization (FlyDSL's §2 and §4–§6)), the 8-wave
> pipeline (FlyDSL's `fp8_gemm_8wave.py`; Primus-Turbo's fp8 and MXFP8 GEMMs) and the 4-wave designs (FlyDSL's
> `fp8_gemm_4wave.py` and `fp4_gemm_4wave.py`; Primus-Turbo's whole-loop inline asm). Primus-Turbo's kernels are
> built for large M (small M is in `small_m.md`). "FP8" below means Turbo's
> `gemm/gemm_fp8_kernel.py`, "FP4" means `gemm/gemm_mxfp4_kernel.py` and "GH" means `utils/gemm_helper.py`, all
> under `primus_turbo/flydsl/` at main `7615a5a2`. All of them run on FlyDSL v0.3.2. What to know when calling
> FlyDSL's kernels is in `../flydsl/gemm.md` § mainloop §8.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are choosing or writing the K loop of a GEMM: wave count, LDS buffering, where the waits go,
  and whether the loop should be one inline-asm region.
- An inline-asm MFMA or loop compiles and runs but produces wrong numbers on some shapes.
- You are deciding between AGPR and VGPR accumulators, or between MFMA atoms.

Grid, tile and split-K pricing are in `grid_pricing.md`. Wait counts, priority and tile order
are in `schedule_and_locality.md`. LDS layout, global→LDS DMA and M0 are in
`datapath.md`. The C store is in `epilogue.md`.

## The baseline: FlyDSL's 2-stage ping-pong (`preshuffle_gemm`)

FlyDSL's guide walks through this design in detail (`../flydsl/guides.md` § gemm-optimization); the essentials:
- **Four waves split `tile_n`; B never touches LDS.** B is preshuffled on the host into the MFMA register
  layout and loaded straight to VGPRs (guide §4.2, §12). LDS holds only A, in two buffers (guide §2).
- **Each loop iteration covers two K tiles, one per buffer.** For each: load the next A tile into the idle
  buffer and the next B tile into registers, run the MFMAs on the current buffer, then `s_waitcnt` + barrier
  (guide §2.3, §4.4).
- **Scheduling-group hints spread the loads over the MFMAs**, one group per instruction class (VMEM reads, LDS
  reads, LDS writes, MFMAs), with counts that come from this tile (guide §5; `schedule_and_locality.md` §3).
- **Async copy (gfx950).** Moving A global→LDS by DMA frees A's staging VGPRs (guide §11).
  *(field, E1)* Staging A or B through VGPRs + `ds_write` instead of DMA lost 10–15% on a bf16 8-wave loop at M=16384 (timing-only builds; their output was not checked).
- Budget it with the guide's formulas: MFMAs per tile (§1.5), LDS per stage (§2.4), VGPRs (§10).

The 8-wave and 4-wave designs below put both operands through LDS and use larger tiles.

In FlyDSL: `../flydsl/gemm.md` § mainloop The baseline: FlyDSL's 2-stage ping-pong (`preshuffle_gemm`).

## 1. Two large-tile loop designs — race them, do not pick by rule

| | 8-wave software pipeline | 4-wave whole-loop |
|---|---|---|
| Threads / wave grid | 512, waves 2×4 (`wave_m = wid//4`, `wave_n = wid%4`) | 256, waves 2×2 |
| Occupancy | 2 waves per SIMD | 1 wave per SIMD (the whole 512-register file is one wave's) |
| K loop | compiler-scheduled, fully unrolled (`K_ITERS-2` iterations) | one inline-asm region with a hardware loop |
| LDS | 8 buffers: A/B × cur/next × two halves | operand pools of mixed depth (A 2-deep, B 3-deep) |
| Where it wins | short K, where the epilogue is a large share of the time and the second wave per SIMD overlaps it | long K, where the steady state amortizes an epilogue that nothing overlaps |
| Turbo source | FP8 `_compile_dense_nt` (:85) | FP8 `_dense_nt_wave4_asm` (:2187), `_dense_tn_wave4_asm` (:1401); FP4 `MfmaScaleFp4` |

A 4-wave design can also keep a compiled K loop: the same 2×2 wave split (a 128×128 block per wave, quadrants
c00/c01/c10/c11) with an 8-buffer, depth-2 LDS ping-pong, and one inline-asm block per MFMA to keep the
accumulator in AGPRs. Turbo's 4-wave design runs all of K as one asm region (§3). The LDS argument below holds
for every 2×2 split.

**Why the 4-wave tile wins on long K.** A 2×2-wave split of a 256×256 tile gives each wave a 128×128
block. That is 64 MACs per operand element per K step, against 42.7 for the 8-wave's 128×64 wave blocks,
so the 4-wave tile issues 1.5× fewer LDS reads per MAC. *(field, E2, E3)* On an MXFP4 GEMM at M=N=8192 (Turbo's
4-wave whole loop against its 8-wave intrinsic build), the 8-wave/4-wave LDS instruction ratio was 1.49×. The 4-wave lead grew with K, from 6.4% at K=8192 to 10.5% at K=28672.
Turbo's own comment gives the mechanism on a power-capped part: "the whole-loop moves a third less LDS
traffic per MFMA, which buys clock" (FP8:3200). Sharing an operand between two such wave groups does not buy that
back: *(field, E25)* on a fused-MoE fp8 × MXFP4 stage 1, two 4-wave 64×256 groups sharing B through LDS (8 waves,
128 VGPRs, 44 spilled) issued 1.5–1.6× the LDS instructions of one 4-wave group at 234 VGPRs, with the same MFMA busy
cycles, and ran 14–18% slower; with two waves per group (256 VGPRs, no spill), still 2–5% slower.

**Why it loses on short K.** At one wave per SIMD nothing overlaps the epilogue. Turbo therefore only
offers the whole loop when `ceil(K/128) >= 16` (`_NT4_MIN_K_ITERS`, FP8:3203). Above that it compiles
both designs and races them on the first call for each shape (`_autotune_nt_dispatch`, FP8:3210; see
`autotune.md`). Do the same in your own kernel: the crossover depends on the shape and on the
epilogue, so no fixed rule works.

In FlyDSL: `../flydsl/gemm.md` § mainloop §1.

## 2. The 8-wave pipeline (FP8 `_compile_dense_nt`)

- **LDS.** Eight LDS buffers: A/B × cur/next × two halves (FP8:134-142).
  A 256×256×128 fp8 tile uses 128 KiB.
- **Quadrants.** Each wave owns four accumulator quadrants, c00/c01/c10/c11. One iteration interleaves
  four S2R fragment loads, the next tiles' G2S fills and four MFMA clusters (FP8:264-310):
  ```
  b0 = S2R(b_cur0); a0 = S2R(a_cur0); G2S(a_next1, k+1); s_barrier
  s_setprio(1); c00 = mfma(a0, b0); s_setprio(0); s_barrier
  b1 = S2R(b_cur1); G2S(b_cur0, k+2); s_barrier
  ... c01, c10 ...
  G2S(b_cur1, k+2); wait_barrier(2*nA + nB)          # s_waitcnt vmcnt(n); s_barrier
  s_setprio(1); c11 = mfma(a1, b1); s_setprio(0); s_barrier
  s_waitcnt vmcnt(3)                                   # end-of-iteration drain
  swap cur/next names in Python
  ```
- **Every barrier is load-bearing.** The kernel docstring says "each guards a compiler-reorder race".
  *(field, E4)* On Turbo's MXFP4 4-wave whole loop, the one barrier cut checked for correctness raced, even with the `vmcnt` drain fully emptied; with a barrier in every phase it did not: the cross-wave cause is a missing
  LDS rendezvous, not out-of-order `vmcnt`. One barrier per phase can be enough if every refill still comes after all waves' reads of its slot and every read after all waves' DMAs into it *(field, E39)*.
- **The end-of-iteration drain is a correctness knob, not a tuning knob.** `nt_vmcnt=3` is documented
  at FP8:92: N=3 gave bitwise-repeatable output at ≤1.1% cost, N≥4 raced, N<3 was slower *(field, E5)*.
  Only part of this held on other tiles of the same kernel *(field, E48)*: a drain at or above the loop's own
  `wait_barrier` count is a no-op, the race was rare and tile-dependent (3 in 8000 repeats at BLOCK_M=128, none at
  BLOCK_M=256 even with no drain), and only N=0–1 cost measurable time.
  - If you build this pipeline, sweep the drain with a bitwise repetition gate (`correctness.md` §3),
    never with SNR alone.
- **`s_setprio(1)` … `s_setprio(0)` around each MFMA cluster.** At two waves per SIMD, with both waves' clusters
  wrapped as Turbo wraps them, the raise does not let the wave that reached its MFMAs first keep the matrix pipe: the
  two are at equal priority and the older wave takes it, as without the raise *(field, E40)*; what it changes in a
  loop with LDS and VMEM traffic was not isolated. Always drop priority afterwards. The whole-loop kernels at
  one wave per SIMD use none.
  - *(field, E6, E7, E8, E9)* Added alone to three other loops at two waves per SIMD it did not pay:
    ~4% slower on an 8-wave DMA loop without per-cluster barriers, neutral to ~4% slower on compiler- or
    hint-scheduled 2-stage loops with one barrier per K step. Race it; do not port it.
  - *(field, E32)* Nor remove it unraced: on an 8-wave bf16 pipe at two waves per SIMD (N=7168, K=512), dropping the
    per-cluster `s_setprio` it came with cost 2.9% at m=2048 and 2.0% at m=4096; at m=1024 it read 0.6% faster in
    one race and 10.56–10.71 against 10.59–10.76 µs in a 4-process confirmation. Priority added to four other bf16
    kernels at m 256–4096 read 0.2–4% slower or within noise.
- **If a third full stage overflows LDS, give one operand a third stage.** *(field, E10)* ~11% on a bf16
  192×256×64 tile at M=9397, against a ~13% ceiling with the loop-top wait removed; deepening A or B gained the same.
  *(field, E11)* On 224–352-row bf16 tiles of another kernel either operand's third stage was neutral to 4% slower: race it.
  *(field, E33)* At K=512 (8 K-blocks per tile) on bf16 GEMMs: at N=28672 one-round launches read faster with three
  stages for both operands and C staged in a B stage the last K-block freed (m=256, 11.17 → 10.88 µs in-process) and
  with A 2 / B 3 (m=512, driver 18.60 → 17.91–17.97; in-process 15.46 against 15.49), but a separate in-process run
  read both slower (12.09 against 11.79, 16.99 against 16.92); the round, with computed load waits, was recorded at
  +19.2% and +9.0% at those m. At N=7168, m=2048 a bundle of A 2 / B 3, A-before-B loads, vmcnt(4), out-of-K B-load
  zeroing and C in A's stage 0 read 1.2–1.8% faster in the campaign, and *(field, E36)* 0.970% faster (CI −1.015% to
  −0.925%) in paired sessions of the same two sources rebuilt; B's third stage was not timed alone. *(field, E33)* At
  m=1024 a third A stage read slower in one run and was not timed again: race it rather than rule it out.
  At N=3584, m=4096 A 3 / B 2 paid 2.5%, and A 5 / B 4 at m=256 failed the 16-CU stress check, cause not found.
  Do not halve the K tile to make room for more stages (`datapath.md`).
- **Edge cases the prologue must cover.** The prologue primes k=0 and k=1. When `K_ITERS == 2` the main
  loop is skipped, so the prologue's own wait must drain what the epilogue reads (FP8:257-262).
- **Code size grows with K** because the loop is fully unrolled. That is acceptable for an 8-wave tile
  at moderate K. It is one reason the whole loop exists.

In FlyDSL: `../flydsl/gemm.md` § mainloop §2.

## 3. The 4-wave whole loop: one inline-asm region for all of K

The K loop is one inline-asm region. The loop label and back edge are part of the asm string
(`s_add_u32 cnt; s_cmp_lt_u32; s_cbranch_scc1 1b`, FP8:2355-2359).

What this buys:
- No compiler-generated code between iterations, so the compiler cannot insert a conservative `vmcnt(0)` or
  move instructions across the phases. Every wait is the one you wrote.
- Code size does not grow with K. The trip count can be a compile-time constant or a runtime SGPR.
- A fully unrolled K loop can outgrow the instruction cache: a source comment puts one 4-wave MXFP4 loop at
  "full unroll -> 111KB > 32KB I-cache" (FlyDSL's `kernels/gemm/fp4_gemm_4wave.py:651`).
- *(field, E12)* On Turbo's MXFP4 4-wave dense kernel, the whole-loop rewrite was a key lever.

How Turbo structures it (FP8:1133-1230, 2187-2412):
- **Operand pools.** A macro tile is described as `(side, width, nbuf)` pools. The NT square tile is
  `((A,128,2),(A,128,2),(B,128,3),(B,128,3))` (FP8:2049). The B pools are 3 deep so their loads stay in
  flight across a phase drain. Deep pools are issued last, so their loads are the youngest in flight: a
  partial `vmcnt(N)` drain waits for the shallow pools and leaves the deep pools' loads outstanding
  (assert, FP8:2323).
- **Passes and phases.** One pass is `lcm(nbuf)` phases, here 6 K-blocks, so every pass ends with every
  pool back on buffer 0 and the loop body is identical each pass (`_tn4_phases`, FP8:1187; assert
  FP8:2324).
- **Inside a phase.** MFMAs run in a code-generated order: srcA pool outermost, walking a diagonal
  (`_tn4_mfma_order`, FP8:2165). Each operand fragment is re-read from LDS right after its last
  consuming MFMA. The G2S fills for a later phase are spread evenly over the MFMA slots that have no
  refill (FP8:2306). An `s_nop 0` goes between instruction groups (`_tn4_pad`, FP8:1175): at one wave per
  SIMD, memory issue arriving in bursts back-pressures the matrix pipe.
- **Phase boundary.** `s_waitcnt vmcnt(n_flight) lgkmcnt(drain_lgkm)` then `s_barrier`. `n_flight`
  counts only the fills nothing reads before a later phase. The barrier is required because a wave's
  `ds_read`s consume chunks that every wave's DMA wrote.
- **Peeled first and last phases.** The first pass is peeled so its opening MFMAs take `src2=0` and
  write the accumulators directly. That removes the zero-fill: one VALU per accumulator dword, 256 per
  tile in FP4. The closing phase is handed back as SSA values, so the epilogue can be scheduled between
  its MFMAs (FP8:2366-2368, 2703-2708).
- **Persistent carry.** When a workgroup walks several tiles, the closing phases prefetch the next tile's
  first K-blocks into buffers the next tile does not re-prime (`_nt4_carry_bufs`, FP8:2512).
- **Buffer select is free.** Each pool keeps one set of per-lane addresses. The buffer and tile deltas
  go into the 16-bit `ds_read` offset field (`_Tn4Pool` asserts the sum stays below 65536, FP8:1208).

*(field, E13)* In one 4-wave whole loop built this way, on a grouped fp8 GEMM (its weight-gradient pass), the
emit order was a sharp local optimum. Every perturbation lost:

| Perturbation | Effect on the weight-gradient group's geomean speed ratio against a frozen MXFP8 kernel (0.9696 before) |
|---|---|
| Refill two MFMAs later (to avoid a read-after-write) | −0.9 pp |
| All G2S fills at the head of the phase | −15% (a run of `buffer_load … lds` stalls the MFMA stream) |
| Spread fills by instruction count instead of by MFMA slot | −8% |
| Separate each `m0` write from its `buffer_load` by one slot | −0.7 pp |

*(field, E14)* At two waves per SIMD (an 8-wave bf16 loop) the head burst was fine: spreading the fills over the MFMAs lost 0–4%.
*(field, E24)* Also at two waves per SIMD, on a fused-MoE stage 1, moving the last 4 of a step's 12 VMEM loads behind
the first MFMA group gained 1.1%, while spreading all 12 over five groups lost 0.6% at one of two sizes: race head splits too.

After the order is right, the next gain comes from changing the work (tile shape, quadrants, boundary
bodies), not from re-ordering.

In FlyDSL: `../flydsl/gemm.md` § mainloop §3.

## 4. Register plan and inline-asm constraints

**The plan.** Accumulators are `"=a"` outputs in AGPRs, up to 64 `vector<4xf32>` (256 AGPRs, FP8:1123).
Any beyond that go to a pinned arch-VGPR window. Operand fragments sit in a pinned VGPR window starting
at `v8` (FP8:1122). The loop counter and the per-pool soffsets are early-clobber SGPR outputs.

**Constraint rules.** Breaking rule 2, 5 or 6 has produced silently wrong output and breaking rule 4 a memory fault, none with a compile error; rules 1 and 3 rest on Turbo's source comments:
1. **A zero accumulator is a literal `src2=0`, never a VALU-zeroed register tied in.** The hazard
   recognizer cannot see an MFMA inside inline asm, so it cannot pad the two wait states a VALU write
   needs before an MFMA reads it. Turbo's `asm_mma_do` checks for a zero-fill vector and emits `0`
   (GH:1597-1635; the comment cites CDNA4 ISA §7.6, Table 38).
2. **An output that lives in arch VGPRs must be early-clobber (`=&v` / `=&{v[a:b]}`).** Without `&` the
   allocator may place an input (a `ds_read` base, a voffset) inside the output range. The asm then
   overwrites it mid-loop. *(field, E15)* Moving accumulators from AGPR to VGPR without `&` passed 3 of 10
   shapes and collapsed the other 7 to 0–2 dB; a failing and a passing shape had the same K. The ISA check, run on a
   passing shape, stayed green. Adding `&` made all 12 shapes then tested bitwise equal to the base build. `&` and a
   tied operand (`"0"`) can be combined (Turbo's `asm_mma_do` uses `"=&v,v,v,0"`, GH:1627) *(field, E41)*: the tied
   output shares its input's register, and `&` keeps every other input, even one holding the same value, out of it
   (`correctness.md` §7).
3. **An unwritten `=&s` output is a register-allocation hazard** (FP8:1430). Write every declared output.
4. **A loop inside the asm needs `~{scc}` in the clobbers.** Without it LLVM can keep an outer loop's
   exit condition in SCC across the blob. *(field, E16)* The back edge became a bare `s_cbranch_scc1` with no
   `s_cmp`, giving a wrong trip count and a memory fault. Fully unrolled callers looked fine, so the bug
   stayed latent until a loop was rolled.
5. **Waits that order an intrinsic DMA must be side-effecting asm (`has_side_effects=True`, as Turbo's `wait_barrier`, GH:484-491) or carry `~{memory}`** *(field, E42)*: a wait with neither was scheduled after the LDS read it guarded. Asm `ds_read`s must drain their own
   `lgkmcnt` before their consumer (`datapath.md` §4).
6. **An untied "pin" (`"=a,v"`) deletes the operand.** See `correctness.md` §7.

**AGPR or VGPR accumulators.** Judge by the shuttle traffic (`v_accvgpr_read/write/mov` counts in the
hot loop) and by wall time, not by `accum_offset` or register totals.
- A scaled MFMA needs `dst` and `src2` in the same file but takes srcA/srcB from either. FP4 therefore
  swaps register files: the accumulator slices live in arch VGPRs and the A/B fragments in AGPRs, so the
  C store packs straight out of arch VGPRs with no shuttle (`_MXFP4_ARCH_ACC`, FP4:578-582).
- The same trade on the fp8 NT whole loop "measured a wash" (FP8:2045-2047): an MFMA writing an arch
  VGPR that a VALU then reads needs wait states that the single wave per SIMD has nothing to fill.
- *(field, E17)* At unchanged occupancy, freed registers are not a gain by themselves. In one attention kernel, a change that freed 72
  VGPRs and 72 AGPRs by re-reading an operand from LDS (256 more reads, still one wave per SIMD) cost −2.05%. Name the specific gain the headroom is for before building it.

**Setting `amdgpu-agpr-alloc` and friends.** Without the attribute the compiler may give an inline-asm AGPR
accumulator fewer AGPRs than it holds *(field, E43)*: in a HIP kernel at `waves_per_eu=2` a 128-AGPR accumulator got 4 and ~99% of outputs were wrong; at `waves_per_eu=1` a 64-AGPR one got its 64. Set it to the accumulator's AGPR count, and declare every AGPR the asm string names: the attribute does not cover undeclared ones *(field, E46)*. After setting any kernel attribute,
read `.agpr_count`, `.vgpr_count` and the waves-per-eu fields from the ISA (§10).

In FlyDSL: `../flydsl/gemm.md` § mainloop §4.

## 5. MFMA atom choice

- FlyDSL's guide lists the atom per dtype (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §6.2)).
- fp8 and MX formats on gfx950 use `v_mfma_f32_16x16x128_f8f6f4`, or its scaled form for MX. The
  operand formats come from `cbsz` (A) and `blgp` (B); the operand tuple width must match the format
  (fp4 takes 4 dwords).
- **bf16 MFMA atoms: race 16x16x32 and 32x32x16 on the target kernel and ruler, including at the power cap.**
  In the recorded swap the accumulator count, operand bytes and `ds_read` count stayed unchanged; fewer
  accumulator register-file bytes per MAC is a proposed mechanism, not an isolated energy measurement.
  *(field, E18)* In one bf16 grouped GEMM with 256×256 tiles at the power cap, the
  swap raised the clock from 1.37–1.40 GHz (a recorded baseline) to 1.667 GHz and cut the Down projection's
  dispatch time 10.8% (the scored gain over both projections was +9.2%), while cycles per dispatch rose 8% and
  MFMA utilization fell from 84% to 77%.
  `SQ_VALU_MFMA_BUSY_CYCLES` was bit-identical across the swap. The gain came with the clock, which fits a power
  effect, but register-file energy was not measured: race both atoms rather than prefer one.
  - **Rank atoms on wall time and energy, never on utilization counters.**

In FlyDSL: `../flydsl/gemm.md` § mainloop §5.

## 6. K tails inside the kernel

Turbo's fp8 kernel accepts any K ≥ 129 without a padding copy; its MXFP4 kernel needs K % 64 == 0, and
its MXFP8 kernel needs K % 128 == 0 and K ≥ 256.
An over-read past K is legal only when every term it contributes has a factor that is exactly zero by
construction. The three ways to get that zero:
- **K-contiguous operands** (stored `[·, K]`, as in NT). Run `ceil(K/128)` iterations. A partial last block over-reads into the
  next row, not past the descriptor, so zero A's out-of-range K bytes in registers (`mask_a_tail`,
  GH:1551-1577; FP8:362, 384, 2712-2717). This relies on quantized operands containing no NaN encodings.
- **K-row operands** (stored `[K, ·]`: B in NN, A and B in TN). The buffer descriptor's `num_records` bound makes out-of-range loads return
  zero.
- **MX formats.** The packed E8M0 scale is zero past K, so the tail terms are scaled by 2⁻¹²⁷ twice and
  underflow to exactly zero, provided the tail bytes hold no NaN or Inf encoding: FP4 and FP6 have none, FP8
  and BF8 do (`../hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md`, Table 30). Drop a 128-K sub-step that is
  entirely padding rather than computing it.
- Row strides that are not a multiple of the 128 B cache line cost bandwidth even when the tail is
  correct. See `datapath.md` §10.

In FlyDSL: `../flydsl/gemm.md` § mainloop §6.

## 7. Occupancy and allocation knobs

- 8-wave candidates race `waves_per_eu=2` with different AGPR budgets: Turbo's NT table lists
  32, 48 and 64 (`_NT_CANDIDATES`, FP8:3193).
- **Reach a register tier by shortening A's live range, not by a hint; race the extra wave, it can cost more than it gives.** When A's registers set the occupancy tier,
  read A from LDS in 16-row groups one to three ahead of their MFMAs instead of a whole K tile ahead. On fused-MoE
  fp8 × MXFP4 expert GEMMs: 202 → 168 VGPRs (2 → 3 waves per SIMD) and 7.6–7.9% off the call, where
  `waves_per_eu=3` on the old loop spilled 210 VGPRs and ran 2.4× slower *(field, E23)*; 202 → 148 VGPRs and
  −6.1/−6.8% MoE time *(field, E22)*; 146 → 122 VGPRs, after which, with the change extended to the last K tile, `waves_per_eu` 4 built at 128 VGPRs with no spill
  where it had spilled 19–43 *(field, E21)*. In a HIP bf16 GEMM the same change reached 3 waves per SIMD with no spill yet ran 16–17% slower than at 2 *(field, E44)*.
- **Fully unrolling a short K loop needs VGPR headroom; headroom alone does not make it pay.** *(field, E19)* At 12–24 K steps:
  ~1–5% faster on ~100-VGPR tiles, ~40% slower on one 256-VGPR tile (spills 9 → 59) and about 1% slower on another (its ISA was not read). In a HIP bf16 GEMM without LDS, unrolling into spills ran 1.67–1.95× slower, without spills it gained 0.1–0.8% *(field, E45)*.
  *(field, E34)* At m 256–4096 on bf16: at N=128, K=6144 unrolling together with a new K-start rotation took m=1024
  from 10.21 to 10.01 µs, and unrolling alone tied at m=512 and lost 0.7% at m=2048 (13.95 against 13.85); at
  N=3584, K=512, m=4096 a 244-VGPR tile without spills gained only with the next fragments prefetched (16.70 against
  16.79 µs; 22.05 unrolled without the prefetch); at N=2048, K=2048, m=256 it lost (8.46 against 8.13).
- **Two loop bodies in one launch can spill at the seam.** *(field, E35)* In a persistent bf16 kernel (N=28672,
  K=512, m=2048) the lane values a tail body derived from the thread id were computed before the head loop and kept
  live across it: 256 VGPRs, 13 spilled. Building the tail body after the head loop, from an opaque copy of the
  thread id, gave 238 VGPRs, no spill, and 5.7% at that m. The copy is what counts *(field, E47)*: built after the head loop without it, the tail body still compiled to 256 VGPRs and 13 spills; with the copy, before or after the head loop, 238 and none.
- The 4-wave whole loop runs at `waves_per_eu=1` and uses the full register file (FP8:1992). FP4 also sets
  `amdgpu-agpr-alloc=256` (FP4:3095).
- Arch VGPRs and AGPRs share one 512-entry file per SIMD. Allocation is in granules of 8, and the
  occupancy tiers are 64/72/80/96/128/168/256 total registers (8 down to 2 waves per SIMD).
  Arch registers are rounded up to 4 before the AGPRs are added: 101 arch + 21 AGPR is 125 total.
  FlyDSL's guide has the same model with a worked VGPR estimate for a `preshuffle_gemm` tile
  (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §10)).

In FlyDSL: `../flydsl/gemm.md` § mainloop §7.

## 8. FlyDSL's own GEMM kernels (v0.3.2): what to know when calling or tuning them

In FlyDSL: `../flydsl/gemm.md` § mainloop §8.

## 9. Throughput arithmetic

```
TFLOP/s = 2·M·N·K / t
bytes   = M·K·a_bytes + N·K·b_bytes + M·N·c_bytes  (+ scales)
```

The ridge is `peak / 8 TB/s`: about 312 FLOP/B for bf16 and 625 for fp8 (vendor theoretical dense peaks at
MI355X clocks: 2.5 PFLOP/s bf16/fp16, 5 PFLOP/s fp8, 10 PFLOP/s MXFP4/MXFP6; 8 TB/s HBM). With N and K much larger than M, a GEMM's intensity is about
M FLOP/B (bf16) or 2M (fp8). Compare against the best kernel on the same box, never against the
datasheet peak. Dense fp8 on realistic data runs at the power cap, so a reduction in register-file or LDS
traffic can show up as clock rather than as fewer cycles (§1, §5).

FlyDSL's guide gives the same formulas with per-dtype byte counts, including MX scales
(`../flydsl/guides.md` § gemm-optimization (FlyDSL's §8.1–§8.2)), and the rule of thumb "M ≤ 512 is memory-bound" (§8.3). The ridge
above puts the crossover nearer M ≈ 300 on gfx950 for both bf16 and fp8, and only when N and K are much larger
than M; compute the intensity for your shape rather than using either number.

## 10. Reading the ISA

- Count instructions inside the hot loop's label range only. Compare them with the plan: MFMAs per phase,
  `ds_read`s per fragment, `buffer_load … lds` per fill, and one `s_waitcnt`/`s_barrier` per phase.
  FlyDSL's guide gives the expected per-tile counts for `preshuffle_gemm` and target ratios such as "MFMA ≥ 40%
  of the loop" (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §9)). Use the ratios to spot a loop body far from its plan, not
  to rank changes: *(field, E20)* deleting 7.4% of a hot loop's instructions, all SALU and `s_nop`, gained 0% on one A6W4 GEMM (the same edit laid A out wave-major in LDS; LDS, VMEM-load and wait counts were unchanged)
  (`attribution.md` §4).
- Check `.agpr_count`, `.vgpr_count`, `.amdhsa_next_free_vgpr` and the scratch size after every
  register-plan or attribute change (§4).
- For a whole-loop kernel, check that the back edge has its `s_cmp` and that no compiler `s_waitcnt`
  sits between the asm phases.

In FlyDSL (`FLYDSL_DUMP_IR=1` dumps the IR and the final ISA): `../flydsl/gemm.md` § mainloop §10.

## 11. Narrow N at M 256–4096: K over waves, each with a private LDS ring

Measured on three bf16 GEMMs: N=64/K=4096 at M256–4096, N=128/K=6144 at M256–2048,
and N=256/K=6144 at M256–1024; each campaign started from a `torch.matmul` call, and the library kernel it
dispatched was not recorded. A one-round grid
(`grid_pricing.md` §1) leaves each workgroup a small output tile and the whole K. The port gives each
workgroup one tile and splits K over its KW waves: each wave streams its slice through its private LDS-DMA ring,
waits on its own `vmcnt` and runs no `s_barrier` in the K loop. The fp32 partials meet in LDS,
are added in wave order and rounded once, so the result repeated bitwise in the configurations tested. *(field, E26)*
The port from AKLR's `gemm_n512_k4096` kernel (`../cases/bf16_k4096_splitk_hgemm.md`) and its accompanying
tuning gained +29.9%, +26.9% and +11.9%, respectively, over that `torch.matmul` starting kernel, not over AKLR's
baseline. No arm isolated K-over-waves alone; it fell with M at N=128 (m=256
+54.7%, m=2048 +8.0%) and N=256 (m=1024 +1.5%) but held at 25–34% over every m at N=64.

What held inside it:
- **BK = 128 (256 B per bf16 row).** *(field, E27)* At N=64 on the ported tiles it beat BK 64 at m 256–2048 (m=256
  5.94 against 6.17 µs, m=2048 8.33 against 8.61) and tied at m=4096 (11.71 against 11.68); at N=128, m=2048
  14.46–14.52 against 15.95 or more for BK 64 and 17.41 for BK 256. The final campaign kernels keep BK 64 at N=64's
  m=2048 (a 32×16 tile) and m=4096 fold, and at N=256's m=1024.
- **Race the ring depth for each row.** The kept rows use two or three stages per wave, except N=64's
  four-stage BK-64 row at m=2048 and its separate m=4096 fold path. Each wave has its whole ring in flight
  with no barrier to drain, unlike the shared ring of `small_m.md` §5. *(field, E28)* In the campaigns N=64, BK 64
  with 6–8 stages read as with 4–5 (within 0.3% at m 256–2048), and m=4096 at 3 or 5 stages 11.73 against 11.65 µs;
  at N=128, m=256, 3 stages with UNROLL 6.00 against 5.73 for 2, and at m=2048 BK 64 with 5–6 stages 13.17–13.35
  against 12.99 for BK 128 with 3; at N=256, m=256 and 512 run 2 stages (+2.25% together with new K starts).
  In paired sessions on rebuilt configurations, at two of these points: *(field, E37)* at N=128,
  m=256, 3 stages with UNROLL were 4.025% slower than 2 stages (CI +3.980% to +4.070%), while UNROLL at 2 stages was
  within ±0.3% of 2 stages alone (−0.011%, CI −0.049% to +0.027%): UNROLL by itself cost nothing there.
  *(field, E38)* At N=64, m=4096, on the private-ring kernel without the fold path that row later took, 3 and 5
  stages were 0.994% and 0.835% slower than 4 (CIs +0.949% to +1.039% and +0.787% to +0.884%). The best depth moved
  with the row: race it, there is no fixed two- or three-stage rule.
- **More K waves than the kept count lost.** *(field, E29)* N=128, m=256: 12 waves 5.98–6.04 against 5.52–5.54
  µs for 8; N=64, m=2048: 8 waves 8.01–8.46 against 7.24 for 4; N=64, m=1024: 16 waves at BK 64 5.77 against 5.37
  for 8 at BK 128. Fewer did not win either: at N=64, m 256–1024 8 waves tied with the port's 4 and were kept, and at
  N=128, m=512 8 beat 4 (7.82 against 7.94 µs).
- **Spread the fold over every thread.** *(field, E30)* At N=256, all 256 threads folding 1–4 columns each in wave
  order, with packed stores, in place of wave 0 folding, in a round that also built the C descriptor in the prologue
  (not timed apart): +2.15%; a padded row-major fold tile +0.61% more.
- Race, do not adopt *(field, E31)*: issuing the next tile's DMA before the wait (`lgkmcnt(0)`, issue,
  `vmcnt((S − 1) · loads)`) came with BK 128 in a +9.3% round at N=256 (not timed apart), read up to 5% slower at N=128
  (m=256) and was not shown correct at N=64; giving wave w the K tiles w, w + KW, … in place of a contiguous
  slice paid 1.1% at N=256, m 512–1024 and lost at N=64 and N=128; refilling a slot as soon as its fragments were
  read lost at N=128 and N=256.

At these M the next levers were a K split across workgroups as well (`grid_pricing.md` §2) and
kernel-argument preload (`grid_pricing.md` §6). At N 1024–2048 a K split
did not decide large M (`../cases/bf16_k4096_splitk_hgemm.md` § Counter-signals).

## Related
`grid_pricing.md` · `schedule_and_locality.md` · `datapath.md` ·
`epilogue.md` · `autotune.md` · `correctness.md` ·
`../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. The month is the run's where the record dates it; where it
does not, the line says which date it is. "Turbo" is Primus-Turbo.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, per-arm timer not stated (the run's baseline is the driver bench, warmup 3, 20 iterations); timing-only builds, output not checked] global→VGPR→`ds_write` staging after compute at m=16384: A 1.1660, B 1.1804, both 1.2080 ms, against 1.0529 ms for the DMA loop in the same session (+10.7% to +14.7%) (gemm_n6144_k7168 forge run; 256×256 tile, 2×4 waves, 2 stages, BK 64)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, rocprofv3 PMC] SQ_INSTS_LDS of the 8-wave intrinsic build against the 4-wave whole loop: 1.49× (Turbo MXFP4 dense GEMM, M=N=8192; unverified)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, ruler not recorded] 4-wave whole loop against the 8-wave intrinsic build: +6.4% at K=8192, +10.5% at K=28672 (Turbo MXFP4 dense GEMM, M=N=8192; unverified)
- E4 [2026-06, gfx950, FlyDSL ?, N=?, run-to-run determinism] on the MXFP4 4-wave whole loop, cutting barriers (`LEANBAR`) triggered an operand race; a race table: with the vmcnt drain fully emptied the race remained, with a barrier in every phase it was gone (Turbo MXFP4 dense GEMM; unverified; the race table's kernel is not stated)
- E5 [2026-06, gfx950, FlyDSL ?, N=?, ?] Turbo's source comment on `nt_vmcnt`: an end-of-iteration vmcnt(3) gave det=0 at ≤1.1% cost, 4 and above raced, below 3 cost more; unverified, and the month is that of the commit that added the comment (Turbo FP8 dense NT 8-wave kernel `_compile_dense_nt`; comment added in Primus-Turbo e7ecaf26 (Primus-Turbo#356); Primus-Turbo 7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py:92)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode --bench-case, warmup 3, 10 iterations; baseline a full driver bench, 20 iterations] `s_setprio` around the MFMA groups at m=16384: 1.1077 against 1.0602 ms (+4.5%) (gemm_n6144_k7168 forge run; 256×256 tile, 2×4 waves, 2 stages)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver bench, warmup 3, 20 iterations, one run] `s_setprio` around each MFMA cluster of the tiled path: m=893 0.084402 against 0.080896 ms (+4.3%), m=2732 0.190357 against 0.187004 (+1.8%), m=9397 0.642948 against 0.637774 (+0.8%), m=16384 1.170480 against 1.148148 (+1.9%) (gemm_n6288_k7168 forge run; 8 waves per workgroup, 2- and 3-stage rows)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, own harness: L2 flushed by zeroing 512 MB, CUDA events, median of 20–60 calls, 3–4 alternating rounds] the same change, ranges against canonical: m=893 103.0–104.8 against 102.8–107.2 µs, m=2732 197.6–212.1 against 193.4–239.0, m=9397 638.1–676.1 against 629.3–729.3, m=16384 1115–1134 against 1097–1185 (gemm_n6288_k7168 forge run)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=4, script: 10 launches per CUDA graph, 512 MB flush before each replay, median of 20 replays per repeat, 4 interleaved repeats] `s_setprio` around each MFMA cluster: m=9397 904 / 907 → 916 µs, m=16384 1616 → 1632 µs (about +1%) (gemm_n8448_k7168 forge run; 320×256 and 416×192 tiles, 8 waves, 2 stages)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, harness alternating the kernel and the variants on AKLR bench_cudagraph, 8 timing sets] a third LDS stage for A or for B at m=9397: 132.77 and 133.20 µs against canonical 150.03 (−11.5%, −11.2%); canonical with the loop-top vmcnt wait removed (wrong output) 130.47 against 150.04 (−13.0%) (gemm_n1280_k7168 forge run; 192×256 tile, BK 64, 2×4 waves, 2 stages)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=5, bench_cudagraph in process with the incumbent, median of 5] A×2 + B×3: m=16384 1068.40 against 1054.24 µs, m=9397 624.52 against 624.95, m=2732 183.66 against 183.31; A×3 + B×2: m=16384 1077.02 against 1035.32, m=9397 635.73 against 625.21 (−0.1% to +4.0%) (gemm_n6016_k7168 forge run; tiles 352×192, 224×256, 256×256, 2×4 waves)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] the whole-loop bare-asm rewrite is reported as the key lever of Turbo's MXFP4 4-wave kernel, from 3583 (intrinsic) to 5401, stated as +16%; the two throughputs give +51%, and neither the shape nor the run is stated; unverified, and the month is the record's (Turbo MXFP4 4-wave dense GEMM)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, campaign ruler: geomean of t_mx / t_tw against a frozen MXFP8 kernel, arms interleaved in one process, median] emit-order perturbations of the 4-wave whole loop, as the wgrad group's ratio: baseline 0.9696; refill two MFMAs later 0.9604; all G2S at the head of the phase −15% (overall 0.93161 against 0.98644); fills spaced by instruction count −8%; each `m0` write one slot ahead of its load 0.9629 (gpt-oss tensorwise fp8 grouped GEMM, weight-gradient pass (TN, variable K), `kernel_grouped_tn_wgrad_4wave`; the round's harness step timed out, so these are the round agent's own readings)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=?, per-arm timer not stated (the run's baseline is the driver bench, warmup 3, 20 iterations)] G2S fills spread into the MFMA stream: m=16384 spread 1/2/4 1.0563/1.0631/1.0953 ms against 1.0529 unspread (+0.3% to +4.0%); spread 1 at m=893 0.08205 against 0.08207, m=2732 0.18101 against 0.17989, m=9397 0.62490 against 0.61983 (gemm_n6144_k7168 forge run; 2×4 waves)
- E15 [2026-09, gfx950, FlyDSL ?, N=1, the round's correctness gate: bitwise equality with the base build plus SNR, 10 shapes] traded accumulators in arch VGPRs with `"={v[b:b+3]}"`: 7 of 10 shapes neither bitwise equal nor above 1.77 dB (0.03–1.77 dB); `out_proj_fwd` 32768×4096×4096 passed while `KI16_wide` 1024×2048×4096, same K and K iterations, failed; with `=&` 12 of 12 shapes bitwise equal and the ISA byte-identical (Turbo MXFP4 dense GEMM on the Llama training shapes)
- E16 [2026-09, gfx950, FlyDSL ?, N=1, one run and its ISA dump] a scf.for-rolled persistent tile loop around the whole-loop asm, without `~{scc}`: GPU memory fault on the first run, back edge a bare `s_cbranch_scc1` with no `s_cmp`; with the clobber the shipped, unrolled path stayed byte-identical (4696 instructions, 496 VGPRs) (Flux MXFP4 dense GEMM, `call_mxfp4_wholeloop`)
- E17 [2026-09, gfx950, FlyDSL 0.2.4, N=1, campaign score: geomean of reference/current time over three batch sizes times a guard term, one scored run] on an attention kernel, `g3_kreg=False` (256 more LDS reads) freed 72 VGPRs and 72 AGPRs (479 → 407, 223 → 151, spill 0, still one wave per SIMD) and scored 0.9795 (−2.05%) (gpt-oss D=64 attention backward)
- E18 [2026-08, gfx950, FlyDSL ?, N=1, rocprofv3 PMC run of 10 dispatches against a recorded baseline; the campaign score for the whole round, 3 reads] Mfma32x32x16 → Mfma16x16x32 in the 8-wave 256×256 forward NT tile: the Down projection's dispatch 1803 → 1608.9 µs (−10.8%), sclk 1.40 → 1.667 GHz at a pinned 1400 W (the same record also gives the baseline as 1.37 and as 1.37–1.40 GHz), cycles per dispatch up ~8%, MFMA utilization 84% → 77%, SQ_VALU_MFMA_BUSY_CYCLES 2.12337e9 on both; campaign score +9.17% (1283.9 → 1401.7 TF/s geomean over 8 configurations) (bf16 grouped GEMM forward, gpt-oss-20b MoE: 32 experts, M=131072, K=2880, N=5760 and 2880; Primus-Turbo ed8d7af4 (#486))
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=2, temporary CUDA-graph bench, median of 10 replays of 50 calls, two readings per arm against canonical; VGPR and spill counts from the distill probe's ISA dumps] fully unrolled K loop: m=893 (12 K steps, 102 VGPRs) 10.30 and 10.29 µs against 10.94 and 10.90; m=2732 (24 K steps, 104 VGPRs) 22.97 and 22.68 against 23.48 and 23.69; m=9397 (24 K steps, 256 VGPRs, spills 9 → 59) 79.65 and 79.32 against 56.16 and 56.23 (+41%); m=16384 (256 VGPRs, ISA of the unrolled build not read) 93.10 and 94.59 against 92.81 and 93.50; the run's ablation of the kept rows (m=893, m=2732) +1.09%, one-sided 95% bounds +1.03% … +1.14% (gemm_n2304_k1536 forge run)
- E20 [2026-10, gfx950, FlyDSL ?, N=3, campaign score: training-time-weighted harmonic mean TF/s of six shapes, three reads per arm] A's K-tile offset moved into `soffset` and A's LDS laid out wave-major: hot loop 949 → 879 instructions (SALU 98 → 52, `s_nop` 42 → 18), ds_read, VMEM-load and s_waitcnt counts unchanged; score 3964.08 / 3954.31 / 3951.09 against 3950.12 / 3952.65 / 3956.84, 0% (Turbo A6W4 dense forward GEMM, six training shapes)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=?, ranges over repeated reads, stage-2 kernel time in one process and ISA resources] row-group-major MFMAs with each 16-row group's A `ds_read` issued one group ahead (two groups live instead of four, BM 64): 122 against 146 VGPRs, no spills, 175.4–176.3 against 180.9–181.6 µs, output bit-identical; the same extended to the last K tile with `compile_hints["waves_per_eu"]` 4: 128 VGPRs, 0 spills, no scratch, kept at +0.29% (+0.25…+0.34%); the hint without it: 128 VGPRs with 43 spilled (0.5498 against 0.4411 ms) and 19 spilled (fused_moe situv2 fp8 activations × MXFP4 down weights, 896 experts, top-16, n_tokens 2732, stage 2 BM64 BN256 BK128)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2, stage 1: driver bench (warmup 3, 20 iterations), one run of the base and two of the change; stage 2: torch-profiler kernel time, 10 calls, cache cleared] stage 1, whole A tiles preloaded from LDS into VGPRs and carried across the ping-pong halves (aiter's `prefetch_full_a_from_lds`, `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:1312`) at 202 VGPRs: 1.1423 / 1.7838 ms; A read per 16-row slice one phase ahead at 148 VGPRs: 1.0728–1.0737 / 1.6618–1.6629 ms at 9397 / 16384 tokens; the run's ablation of that round +7.53% (bounds +7.20% … +7.86%); stage 2, the same change: 146 → 122 VGPRs, 418.6 → 384.9 / 586.1 → 547.4 µs (fused MoE, fp8 activations × MXFP4 experts, 896 experts, top-k 16, inter 384, model dim 3584, stage 1 tile 64×128×256)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=2, forge driver --bench-mode (graph timer, warmup 3, 20 iterations), whole call at n_tokens 9397 / 16384; registers from the ISA] A streamed from LDS in 16-row × 128-K sub-tiles, 3 in flight, against the whole A tile read from LDS one step ahead: 0.796 / 1.247 (re-run 0.797 / 1.246) against 0.864 / 1.350 ms, main kernel 202 → 40 + 128 = 168 VGPRs, scratch 0; forge ablation, 3 pairs: the version without it +8.50% [+8.26, +8.75]; `compile_hints={"waves_per_eu": 3}` on the old loop: 168 VGPRs, 210 spilled, 380 B private segment, 2.059 / 3.217 against 0.861 / 1.352 ms (fused MoE SiTUv2 EP8 stage 1, 64×128×256 tile, LDS 50,176 B)
- E24 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm (the kept arm: forge ablation, 3 pairs), forge driver --bench-mode, whole call at n_tokens 9397 / 16384; ISA not read] 12 VMEM loads per K step (A scales, w1 scales, w1 k0, w1 k1) at 240 VGPRs: all at the step head 0.7198 / 1.1069 ms; 8 at the head and 4 after MFMA group 0: 0.7165 / 1.0912; (4,2,2,2,2,0) over five groups: 0.7244 / 1.1052; ablation of the kept split: +1.12% [+1.05, +1.19] (fused MoE SiTUv2 EP8 stage 1, 64×256×256 tile)
- E25 [2026-10, gfx950, FlyDSL 0.3.2, N=2 sequential rounds per comparison on one GPU, torch.profiler stage-1 kernel time with L2 flushed; rocprofv3 PMC on one call per build] the pair tile (two 64-row groups sharing W1 through LDS, 8 waves, 128 VGPRs, 44 spilled, 180 B scratch, 81,920 B LDS) against the 4-wave 64×256×256 tile (234 VGPRs, 50,176 B LDS): 549 / 793–810 against 479–481 / 684–689 µs at n_tokens 9397 / 16384, and 553 / 814 against 472 / 697 in a separate measurement; pair ÷ 4-wave: MFMA instructions and busy cycles 1.00, GRBM_GUI_ACTIVE 1.20 / 1.22, LDS instructions 1.59 / 1.51, SQ_WAIT_INST_LDS 2.80 / 2.14; the pair with 2 waves per group (`waves_per_eu=2`, 256 VGPRs, no scratch): 536.4 / 749.9 and 531.2 / 740.8 against 513.5 / 738.0 and 504.8 / 729.4 µs (fused MoE SiTUv2, fp8 activations × MXFP4 weights, 112 of 896 experts local, top-16, model dim 3584, inter 3072)
- E26 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladders (recorded), 3 pairs, driver --bench-mode; started from torch.matmul] port of the `gemm_n512_k4096` schema kernel (per-wave K split with private LDS rings) together with tuning, measured over each round's parent, the campaign's starting `torch.matmul` kernel (the library kernel it dispatched was not recorded), not isolated K-over-waves; N64/N256 iteration 1 and N128 iteration 2 (iteration 1 reverted): N=64 K=4096 +29.87% [+29.60, +30.14] (m=256 +29.26%, m=1024 +34.41%, m=2048 +25.03%, m=4096 +29.53%); N=128 K=6144 +26.91% [+26.71, +27.12] (m=256 +54.70%, m=512 +34.81%, m=1024 +10.09%, m=2048 +8.04%); N=256 K=6144 +11.94% [+11.39, +12.48] (m=256 +20.34%, m=512 +13.96%, m=1024 +1.51%); the final campaign kernels' docstrings: waves split K, each with a private LDS ring and no barrier in its K loop, partials folded in a fixed order
- E27 [2026-11, gfx950, FlyDSL 0.3.2, N=1–3, one driver --bench-mode run per configuration] N=64 K=4096: BK 128, KW 4 5.943 / 5.987 / 6.145 / 8.330 / 11.706 against the port's BK 64 6.172 / 6.242 / 6.365 / 8.608 / 11.678 µs at m=256…4096; N=128 K=6144 m=2048: BK 128 KW 3 S3 14.49 / 14.52 / 14.50 (14.46 without the XCD box), BK 64 15.95–16.12, BK 256 KW 2 S2 17.41; final BK per m: N=64 128 / 128 / 128 / 64 (m=2048 a 32×16 tile at BK 64) and BK 64 in the m=4096 fold, N=128 128 throughout, N=256 128 / 128 / 64
- E28 [2026-11, gfx950, FlyDSL 0.3.2, N=1–5, one driver run per arm (N=64, N=128 m=256), AKLR `bench_cudagraph` medians over 5 rounds (N=128 m=2048); ablation 3 pairs (N=256)] N=64 BK 64 with 8 stages at m 256–1024 and 6 at m=2048: 6.160 / 6.222 / 6.357 / 8.602 against 6.172 / 6.242 / 6.365 / 8.608 at 4–5 stages; m=4096 5 stages 11.736, 3 stages 11.732 against HEAD 11.650; N=128 m=256 3 stages with UNROLL 6.000 / 5.999 against about 5.73; m=2048 BK 64 with 5–6 stages 13.172–13.345 against 12.986 (BK 128, 3 stages); N=256 two stages at m=256 and 512 with new K-walk rotations: +2.25% [+2.05, +2.44]
- E29 [2026-11, gfx950, FlyDSL 0.3.2, N=1–3, AKLR `bench_cudagraph` over 3 rounds against the current kernel (N=128), one driver run (N=64)] N=128 m=256: 12 K waves 5.975–6.037 against 5.522 / 5.535 for 8; N=64 m=2048: 8 waves 8.46 at 3 stages and 8.01 at 2 against 7.24 for 4; m=1024 16 waves at BK 64 5.77 against 5.37 (8 waves, BK 128); the port at m=256 / 512 / 1024 with 8 waves 6.163 / 6.208 / 6.364 against 6.172 / 6.242 / 6.365 with 4; N=128 m=512 BK 128 8 waves 2 stages 7.82, 4 waves 4 stages 7.94
- E30 [2026-11, gfx950, FlyDSL 0.3.2, N=3, forge-loop ablation ladder (remeasured), 3 pairs, driver --bench-mode] N=256 K=6144: the `kw == 0` fold replaced by all 256 threads folding G = 1, 2 or 4 columns in partial order, with packed stores and the C descriptor built early in the same kept round, not timed apart: +2.15% [+1.88, +2.41]; a padded row-major fold tile with vector LDS reads at m=512 / 1024: +0.61% [+0.21, +1.01] (m=1024 11.568 / 11.574 against 11.701 / 11.702 µs in the probe)
- E31 [2026-11, gfx950, FlyDSL 0.3.2, N=1–4, driver --bench-mode runs and agents' interleaved probes on AKLR's graph timer; ablation 3 pairs] issue-before-wait: N=256 BK 128 with `ISSUE_FIRST` +9.29% [+9.13, +9.44] (m=1024 +12.58%; no BK 128 arm without it), N=128 EARLY=1 6.016 / 7.249 / 9.762 / 13.639 against 5.719 / 7.145 / 9.759 / 13.622, N=64 m=512 4.60 against 4.40 with `allclose: False` (`source_rejected` in the same runs; cause not isolated); K tiles interleaved over waves: N=256 9.542–9.572 / 12.355–12.373 against 9.645 / 12.499 (m=512 / 1024), N=64 slower at m=256, 1024, 2048, 4096, N=128 slower at every m (m=256 5.863 against 5.741); early slot release: N=128 m=256 6.082 / 6.069 against 5.938 / 5.932 and 5.712 against 5.554, m=1024 9.379 against 9.322; N=256 m=1024 11.723 against 11.582
- E32 [2026-11, gfx950, FlyDSL 0.3.2, N=3–7, agents' interleaved in-process probes on AKLR's graph timer, medians of 5–7 readings] N=7168 K=512 pipe, `s_setprio` mode 0 (the incumbent's placement) against mode 2 (none): m=1024 10.632–10.651 against 10.565–10.587, m=2048 16.311–16.448 against 16.842–16.864, m=4096 29.054–29.180 against 29.658–29.748 µs (3 processes); a 4-process confirmation at m=1024: no `s_setprio` 10.558–10.713 against 10.586–10.758; added priority: N=2048 m=2048 21.059 against 21.009; N=6144 m=1024 21.81 against 21.33–21.34; N=128 within about 0.02 µs; N=3584 around every K step's MFMAs +0.302 / +0.659 µs at m=2048 / 4096
- E33 [2026-11, gfx950, FlyDSL 0.3.2, N=1–5, in-process runs and driver --bench-mode (N=28672), interleaved probes on AKLR's graph timer (N=7168, N=3584); ablation 3 pairs] N=28672: m=256 3 + 3 stages with C aliased onto a B stage 11.17 → 10.88 (11.34 → 10.90 generalized), m=512 A 2 / B 3 driver 18.60 → 17.91 / 17.97, in-process 15.46 against 15.49; a separate in-process run (3 repetitions) read candidate against canonical m=256 12.09 against 11.79 and m=512 16.99 against 16.92; ablation (with the computed load waits) +5.66% [+5.52, +5.79], m=256 +19.17%, m=512 +9.01%; N=7168 m=2048 A2/B3 + A-before-B loads + vmcnt(4) + out-of-K B-load zeroing + C-in-A-stage-0 bundle +1.68 / +1.15 / +1.76% (not a B-stage-only ablation; original-source/configuration recompile in E36); A 3-deep at m=1024 10.641 against 10.577 (one run; that build's source was not kept); N=3584 m=4096 A 3 / B 2 −0.473 µs over 5 pairs, ablation +0.63%, m=4096 +2.48%; m=256 with separate A and B depths failed the stress check on 16 CUs (2 bad runs of 20), cause not found
- E34 [2026-11, gfx950, FlyDSL 0.3.2, N=3, driver --bench-mode (N=128, N=3584), an in-process paired probe (N=2048)] N=128 K=6144: m=1024 UNROLL with the K-start rotation changed from (3,1) to (3,3) 10.02 / 10.01 / 10.01 against 10.21 / 10.22 / 10.19, UNROLL alone at m=512 7.33–7.35 against 7.34–7.35 and at m=2048 13.947–13.953 against 13.851–13.858; N=3584 K=512 m=4096: UNR=1 without fragment prefetch 22.053, with it 16.698 / 16.715 / 16.687 against 16.780 / 16.795, 244 VGPRs, no spills; N=2048 K=2048 m=256 unrolled 8.464 against 8.132
- E35 [2026-11, gfx950, FlyDSL 0.3.2, N=3, ISA dumps and an in-process probe; ablation 3 pairs] N=28672 K=512 m=2048, persistent head loop plus tail body: 256 VGPRs, `vgpr_spill_count` 13, 26 `scratch_` instructions, reloads after the head loop's back edge; tail body built after the head loop from a `v_mov_b32` copy of tid (inline asm with side effects): 238 VGPRs, 0 spills; ablation +1.10% [+0.79, +1.42], m=2048 +5.67%
- E36 [2026-11, gfx950 MI355X, FlyDSL 0.3.2, N=24 sequential sessions × 3 cyclic readings; cold-L2 official `bench_cudagraph`, rep=20, 90 s GPU ramp; session-mean latency ratios, t(23) Bonferroni 95% familywise CI over candidate + AA] N=7168 K=512 M=2048, original canonical vs candidate sources/configurations recompiled: A2/B3 + A-first + vmcnt(4) + out-of-K B-load zeroing + C-in-A-stage-0 bundle −0.970% latency [−1.015, −0.925]; AA −0.012% [−0.033, +0.009], inside the ±0.15% gate. GPU ISA: both 163840 B LDS, 204 VGPRs, no spills/scratch; bundle vmcnt(4) is present. Three input seeds pass the official float64-reference DotProductBound comparison and are cross-arm bitwise equal. The bundle as a whole, at this M only; B's third stage not timed alone. The interval treats the sequential sessions as independent; one process, no restarts.
- E37 [2026-11, gfx950 MI355X, FlyDSL 0.3.2, N=24 sequential sessions × 3 cyclic readings; E36's cold-L2 ruler and session analysis, Bonferroni family candidate contrasts + AA, practical equivalence margin ±0.3%] N=128 K=6144 M=256, original campaign builder, no new SCHED/UROT/PRO options and original unconditional scheduling: recorded S3+UNROLL vs S2 +4.025% latency [+3.980, +4.070]; diagnostic S2+UNROLL vs S2 −0.011% [−0.049, +0.027], equivalent, not a standalone UNROLL regression. GPU ISA: S2/S3+UNROLL/S2+UNROLL LDS 98304/147456/98304 B, VGPRs 44/42/42, no spills/scratch; three seeds pass official float64 comparison and cross-arm bitwise equality. Configurations rebuilt on the original builder; the campaign's discarded build was not available, and its absolute 6.000/5.999 vs 5.734 µs were not re-timed. This M only.
- E38 [2026-11, gfx950 MI355X, FlyDSL 0.3.2, N=24 sequential sessions × 3 cyclic readings; E36's cold-L2 ruler and session analysis, Bonferroni family candidate contrasts + AA] N=64 K=4096 M=4096, original campaign builder `_build_gemm_impl`, private-wave, no rotation, A-first, only STAGES changed: S3 vs S4 +0.994% latency [+0.949, +1.039], S5 vs S4 +0.835% [+0.787, +0.884]. GPU ISA: S3/S4/S5 LDS 98304/131072/163840 B, VGPRs 72/90/110, no spills/scratch; three seeds pass official float64 comparison and cross-arm bitwise equality. Configurations rebuilt on the original builder; the campaign's discarded build was not available, and its absolute 11.732/11.736 vs 11.650 µs were not re-timed; the fold path the m=4096 row later took was not in the comparison. This M only.
- E39 [2026-11, gfx950 (MI355X), HIP 7.15, N=100 launches per arm, per-thread checksum compared bitwise with a global-memory reference, `s_sleep` skew] 4 waves, a two-slot LDS ring filled by buffer-load-to-LDS, each wave reading another's share: one barrier per phase, after the DMA wait (refilling the slot read a phase earlier) or before the in-place refill (each DMA drained a phase ahead, one more barrier before the loop), 0 of 100 wrong; the latter barrier removed, 100 of 100 (HIP microbenchmark)
- E40 [2026-11, gfx950 (MI355X), HIP 7.15, N=100 launches × 8 rounds per configuration, cluster time from `s_memtime`] two waves on one SIMD, each cluster 256 independent `v_mfma_f32_16x16x32_bf16` wrapped in `s_setprio 1` / `s_setprio 0` on both waves, the other wave starting about 800 ticks in: the older wave ran its cluster uninterrupted either way, a younger one that started first took 8208 ticks against 4152–4160 alone (1.97×); every reading the same within 4 ticks without the raise (HIP microbenchmark)
- E41 [2026-11, gfx950 (MI355X), HIP 7.15, N=50 runs per arm, every output compared exactly with a host reference; registers read from the ISA] `"=&v,v,v,0"` around one `v_mfma_f32_16x16x32_bf16` and `"=&v,v,v,v,v,0"` around two compiled, 0 wrong in 50 of 50 runs, every MFMA with src2 = dst and none with dst on srcA or srcB; an untied output without `&` put dst on an input in 3 of 8 MFMAs, every run wrong (HIP microbenchmark, K=2048)
- E42 [2026-11, gfx950 (MI355X), HIP 7.15 (only the wait's two asm flags edited in the LLVM IR), N=200 launches per arm and code shape, per-thread checksum compared bitwise with a global-memory reference; wait placement from the ISA] buffer-load-to-LDS DMA, asm `s_waitcnt vmcnt(0)` + `s_barrier`, then a compiler-visible LDS read of another wave's share: the wait side-effecting (Turbo's `wait_barrier`), with `~{memory}`, or both: every read after it, 0 of 200 wrong; neither: the wait scheduled after the reads, 200 of 200 wrong (HIP microbenchmark, 4 waves, 2048 workgroups)
- E43 [2026-11, gfx950 (MI355X), HIP 7.15, N=10 repetitions per process, outputs compared exactly with a CPU reference; registers and wait states from the ISA] inline-asm `v_mfma_f32_16x16x32_bf16` with `"=a"` accumulators tied in, no `amdgpu-agpr-alloc` given: the compiler inferred `"4"`; `waves_per_eu=1`, 16 accumulators: 64 AGPRs, 0 wrong; `waves_per_eu=2`, 32 accumulators (128 AGPRs): 4 AGPRs, copies through arch VGPRs around the asm MFMAs with as few as 1 wait state, 1038455–1041250 of 1048576 outputs wrong; the attribute at 128: 0 wrong (HIP microbenchmark)
- E44 [2026-11, gfx950 (MI355X), HIP 7.15, N=9 rounds, hipEvent, median of 30 calls per arm per round, arms interleaved; registers from the ISA] bf16 GEMM with A through LDS in K tiles of 128, M=512, N=98304, K=1024, outputs bitwise equal: A read one to three 16-row groups ahead instead of a whole K tile, 182 → 142–150 VGPRs (2 → 3 waves per SIMD), no scratch, 0.2699 → 0.3144–0.3153 ms, 1.159–1.172× in every round (HIP microbenchmark, 256 threads, 64 × 64 per wave)
- E45 [2026-11, gfx950 (MI355X), HIP 7.15, N=12 paired rounds, hipEvent, median of 7 batches of 20 launches per arm per round; spills from the ISA] bf16 GEMM, 4096 × 4096 output, operands from global memory into registers (no LDS), fully unrolled against rolled loops of 12 or 24 K steps, outputs identical: without spills unrolling gained 0.12–0.78%, every 95% CI above 0; unrolled into spills 1.67–1.95× slower (HIP microbenchmark)
- E46 [2026-11, gfx950 (MI355X), HIP 7.15 (only `amdgpu-agpr-alloc` edited in the LLVM IR), N=10 runs per build, outputs compared exactly with an integer reference; registers from the ISA] one `v_mfma_f32_16x16x32_bf16` accumulator per wave plus extra live values, the attribute inferred, removed or set to `"4,4"`: AGPRs declared as `"+a"`, tied `"=a"` or clobbers, 84 of 84 builds exact; `a[0:3]` named in the asm string but undeclared, 17 of 28 exact, 4 wrong and 7 memory faults, among them one wrong and two faulting with the attribute set (HIP microbenchmark, K=4096)
- E47 [2026-11, gfx950, FlyDSL 0.3.2, N=1, compile only, `.vgpr_count` and `.vgpr_spill_count` of the final ISA, variants differing in one statement] E35's kernel at m=2048: tail body built before the head loop 256 VGPRs, 13 spilled; built after it without copying `tid`, the same; with the inline-asm `v_mov_b32` copy, before or after the head loop, 238 and none (AKLR persistent bf16 GEMM N=28672, K=512)
- E48 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=2000 repeats per arm, 8 back-to-back outputs compared bitwise with the arm's first; timing: median of per-round ratios over 7 rounds of 50 event-timed calls, threshold twice a repeat timing's drift] only `nt_vmcnt` changed (none, 0–6), each drain right after the loop's `wait_barrier` `vmcnt(6)` with no LDS DMA between (`vmcnt(4)` at BLOCK_M=128, by the source); repeats differing from the first: every arm 0 of 2000 at BLOCK_M=256, at BLOCK_M=128 N=5 2 and N=6 1, the rest 0; slower than N=3 beyond the threshold only N=0 (1.020 at 8192×3072×1536, threshold 1.54%; 1.120 at BLOCK_M=128, 8.04%) and N=1 (1.081 at BLOCK_M=128) (Primus-Turbo 7615a5a2 FP8 dense NT 8-wave kernel at 4096×4096×2048 and 8192×3072×1536 with BLOCK_M=256, 4096×4096×2048 with BLOCK_M=128)
