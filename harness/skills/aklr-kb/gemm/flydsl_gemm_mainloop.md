---
title: "FlyDSL GEMM — the main loop: 2-stage ping-pong, 8-wave pipeline, 4-wave designs"
kind: topic
lever: flydsl_gemm_mainloop
ops: [gemm]
gens: [gfx950]
dtypes: [fp8, int8, mxfp4, mxfp8, bf16, fp16]
regimes: [prefill, training, large-K]
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

# The FlyDSL GEMM main loop

> Three loop designs are in use for FlyDSL GEMMs on gfx950: FlyDSL's 2-stage LDS ping-pong
> (`preshuffle_gemm`, explained step by step in `../flydsl/guides.md` § gemm-optimization (FlyDSL's §2 and §4–§6)), the 8-wave
> pipeline (FlyDSL's `fp8_gemm_8wave.py`; Primus-Turbo's fp8 and MXFP8 GEMMs) and the 4-wave designs (FlyDSL's
> `fp8_gemm_4wave.py` and `fp4_gemm_4wave.py`; Primus-Turbo's whole-loop inline asm). Primus-Turbo's kernels are
> built for training shapes (large M; small M is in `flydsl_gemm_small_m.md`). "FP8" below means Turbo's
> `gemm/gemm_fp8_kernel.py`, "FP4" means `gemm/gemm_mxfp4_kernel.py` and "GH" means `utils/gemm_helper.py`, all
> under `primus_turbo/flydsl/` at main `7615a5a2`; "PS" is FlyDSL's `kernels/gemm/preshuffle_gemm.py`. All of
> them run on FlyDSL v0.3.2. What to know when calling FlyDSL's kernels is in §8.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are choosing or writing the K loop of a FlyDSL GEMM: wave count, LDS buffering, where the waits go,
  and whether the loop should be one inline-asm region.
- An inline-asm MFMA or loop compiles and runs but produces wrong numbers on some shapes.
- You are deciding between AGPR and VGPR accumulators, or between MFMA atoms.

Grid, tile and split-K pricing are in `flydsl_gemm_grid_pricing.md`. Wait counts, priority and tile order
are in `flydsl_gemm_schedule_and_locality.md`. LDS layout, global→LDS DMA and M0 are in
`flydsl_gemm_datapath.md`. The C store is in `flydsl_gemm_epilogue.md`.

## The baseline: FlyDSL's 2-stage ping-pong (`preshuffle_gemm`)

FlyDSL's guide walks through this design in detail (`../flydsl/guides.md` § gemm-optimization); the essentials:
- **Four waves split `tile_n`; B never touches LDS.** B is preshuffled on the host into the MFMA register
  layout and loaded straight to VGPRs (guide §4.2, §12). LDS holds only A, in two buffers (guide §2).
- **Each loop iteration covers two K tiles, one per buffer.** For each: load the next A tile into the idle
  buffer and the next B tile into registers, run the MFMAs on the current buffer, then `s_waitcnt` + barrier
  (guide §2.3, §4.4).
- **`hot_loop_scheduler` spreads the loads over the MFMAs** with `sched_vmem` / `sched_dsrd` / `sched_dswr`
  / `sched_mfma` groups whose counts come from this tile (guide §5; `flydsl_gemm_schedule_and_locality.md` §3).
- **Async copy (gfx950).** `use_async_copy` moves A global→LDS by DMA and frees A's staging VGPRs (guide §11).
  *(field, E1)* Staging A or B through VGPRs + `ds_write` instead of DMA lost 10–15% on a bf16 8-wave loop at M=16384 (timing-only builds; their output was not checked).
- Budget it with the guide's formulas: MFMAs per tile (§1.5), LDS per stage (§2.4), VGPRs (§10).

The 8-wave and 4-wave designs below put both operands through LDS and use larger tiles (FlyDSL's defaults are
256×256).

## 1. Two large-tile loop designs — race them, do not pick by rule

| | 8-wave software pipeline | 4-wave whole-loop |
|---|---|---|
| Threads / wave grid | 512, waves 2×4 (`wave_m = wid//4`, `wave_n = wid%4`) | 256, waves 2×2 |
| Occupancy | 2 waves per SIMD | 1 wave per SIMD (the whole 512-register file is one wave's) |
| K loop | FlyDSL code, `range_constexpr(K_ITERS-2)` | one inline-asm region with a hardware loop |
| LDS | 8 buffers: A/B × cur/next × two halves | operand pools of mixed depth (A 2-deep, B 3-deep) |
| Where it wins | short K, where the epilogue is a large share of the time and the second wave per SIMD overlaps it | long K, where the steady state amortizes an epilogue that nothing overlaps |
| Turbo source | FP8 `_compile_dense_nt` (:85) | FP8 `_dense_nt_wave4_asm` (:2187), `_dense_tn_wave4_asm` (:1401); FP4 `MfmaScaleFp4` |

FlyDSL's `fp8_gemm_8wave.py` has the same 8-wave structure (512 threads, eight LDS buffers). FlyDSL's
`fp8_gemm_4wave.py` and `fp4_gemm_4wave.py` use the same 2×2 wave split (a 128×128 block per wave, quadrants
c00/c01/c10/c11) but keep the K loop in FlyDSL code with an 8-buffer, depth-2 ping-pong, and use one
inline-asm block per MFMA to keep the accumulator in AGPRs. Turbo's 4-wave design runs all of K as one asm
region (§3). The LDS argument below holds for every 2×2 split.

**Why the 4-wave tile wins on long K.** A 2×2-wave split of a 256×256 tile gives each wave a 128×128
block. That is 64 MACs per operand element per K step, against 42.7 for the 8-wave's 128×64 wave blocks,
so the 4-wave tile issues 1.5× fewer LDS reads per MAC. *(field, E2, E3)* On an MXFP4 GEMM at M=N=8192 (Turbo's
4-wave whole loop against its 8-wave intrinsic build), the 8-wave/4-wave LDS instruction ratio was 1.49× at K=28672. The 4-wave lead grew with K, from 6.4% at K=8192 to 10.5% at K=28672.
Turbo's own comment gives the mechanism on a power-capped part: "the whole-loop moves a third less LDS
traffic per MFMA, which buys clock" (FP8:3200). Sharing an operand between two such wave groups does not buy that
back: *(field, E25)* on a fused-MoE fp8 × MXFP4 stage 1, two 4-wave 64×256 groups sharing B through LDS (8 waves,
128 VGPRs, 44 spilled) issued 1.5–1.6× the LDS instructions of one 4-wave group at 234 VGPRs, with the same MFMA busy
cycles, and ran 14–18% slower; with two waves per group (256 VGPRs, no spill), still 2–5% slower.

**Why it loses on short K.** At one wave per SIMD nothing overlaps the epilogue. Turbo therefore only
offers the whole loop when `ceil(K/128) >= 16` (`_NT4_MIN_K_ITERS`, FP8:3203). Above that it compiles
both designs and races them on the first call for each shape (`_autotune_nt_dispatch`, FP8:3210; see
`flydsl_gemm_autotune.md`). Do the same in your own kernel: the crossover depends on the shape and on the
epilogue, so no fixed rule works.

## 2. The 8-wave pipeline (FP8 `_compile_dense_nt`)

- **LDS.** One `@fx.struct` with eight `fx.Array` fields: A/B × cur/next × two halves (FP8:134-142).
  A 256×256×128 fp8 tile uses 128 KiB. Allocate it with `fx.SharedAllocator().allocate(...).peek()`.
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
  *(field, E4)* On Turbo's MXFP4 4-wave whole loop, every attempt to drop a barrier produced an operand race; the cross-wave cause is a missing
  LDS rendezvous, not out-of-order `vmcnt`.
- **The end-of-iteration drain is a correctness knob, not a tuning knob.** `nt_vmcnt=3` is documented
  at FP8:92: N=3 gave bitwise-repeatable output at ≤1.1% cost, N≥4 raced, N<3 was slower *(field, E5)*.
  - If you build this pipeline, sweep the drain with a bitwise repetition gate (`flydsl_gemm_correctness.md` §3),
    never with SNR alone.
- **`s_setprio(1)` … `s_setprio(0)` around each MFMA cluster.** At two waves per SIMD this lets the wave
  that reached its MFMAs keep the matrix pipe. Always drop priority afterwards. The whole-loop kernels at
  one wave per SIMD use none.
  - *(field, E6, E7, E8, E9)* Added alone to three other loops at two waves per SIMD it did not pay:
    ~4% slower on an 8-wave DMA loop without per-cluster barriers, neutral to ~4% slower on compiler- or
    hint-scheduled 2-stage loops with one barrier per K step. Race it; do not port it.
- **If a third full stage overflows LDS, give one operand a third stage.** *(field, E10)* ~11% on a bf16
  192×256×64 tile at M=9397, against a ~13% ceiling with the loop-top wait removed; deepening A or B gained the same.
  *(field, E11)* On 224–352-row bf16 tiles of another kernel either operand's third stage was neutral to 4% slower: race it.
  Do not halve the K tile to make room for more stages (`flydsl_gemm_datapath.md`).
- **Edge cases the prologue must cover.** The prologue primes k=0 and k=1. When `K_ITERS == 2` the main
  loop is skipped, so the prologue's own wait must drain what the epilogue reads (FP8:257-262).
- **Code size grows with K** because the loop is `range_constexpr`. That is acceptable for an 8-wave tile
  at moderate K. It is one reason the whole loop exists.

## 3. The 4-wave whole loop: one inline-asm region for all of K

The K loop is a Python-generated assembly string passed to one `_llvm.inline_asm` call. The loop label
and back edge are part of the string (`s_add_u32 cnt; s_cmp_lt_u32; s_cbranch_scc1 1b`, FP8:2355-2359).
Results come back through `llvm.extractvalue` on a struct return type.

What this buys:
- No FlyDSL/LLVM boundary per iteration, so the compiler cannot insert a conservative `vmcnt(0)` or
  move instructions across the phases. Every wait is the one you wrote.
- Code size does not grow with K. The trip count can be a compile-time constant or a runtime SGPR.
- FlyDSL's own fp4 kernel avoids full unrolling for that reason: "full unroll -> 111KB > 32KB I-cache"
  (`kernels/gemm/fp4_gemm_4wave.py:651`).
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

## 4. Register plan and inline-asm constraints

**The plan.** Accumulators are `"=a"` outputs in AGPRs, up to 64 `vector<4xf32>` (256 AGPRs, FP8:1123).
Any beyond that go to a pinned arch-VGPR window. Operand fragments sit in a pinned VGPR window starting
at `v8` (FP8:1122). The loop counter and the per-pool soffsets are early-clobber SGPR outputs.

**Constraint rules.** Each of these has produced silently wrong output:
1. **A zero accumulator is a literal `src2=0`, never a VALU-zeroed register tied in.** The hazard
   recognizer cannot see an MFMA inside inline asm, so it cannot pad the two wait states a VALU write
   needs before an MFMA reads it. Turbo's `asm_mma_do` checks for a zero-fill vector and emits `0`
   (GH:1597-1635; the comment cites CDNA4 ISA §7.6, Table 38).
2. **An output that lives in arch VGPRs must be early-clobber (`=&v` / `=&{v[a:b]}`).** Without `&` the
   allocator may place an input (a `ds_read` base, a voffset) inside the output range. The asm then
   overwrites it mid-loop. *(field, E15)* Moving accumulators from AGPR to VGPR without `&` passed 3 of 10
   shapes and collapsed the other 7 to 0–2 dB, depending on tile count, not on K. Every ISA check stayed
   green. Adding `&` fixed all of them and changed no ISA byte. `&` and a tied operand (`"0"`) are
   mutually exclusive: a tied output shares its input's register by definition.
3. **An unwritten `=&s` output is a register-allocation hazard** (FP8:1430). Write every declared output.
4. **A loop inside the asm needs `~{scc}` in the clobbers.** Without it LLVM can keep an outer loop's
   exit condition in SCC across the blob. *(field, E16)* The back edge became a bare `s_cbranch_scc1` with no
   `s_cmp`, giving a wrong trip count and a memory fault. Fully unrolled callers looked fine, so the bug
   stayed latent until a loop was rolled.
5. **Waits that order an intrinsic DMA need `~{memory}`**, and asm `ds_read`s must drain their own
   `lgkmcnt` before their consumer (`flydsl_gemm_datapath.md` §4).
6. **An untied "pin" (`"=a,v"`) deletes the operand.** See `flydsl_gemm_correctness.md` §7.

**AGPR or VGPR accumulators.** Judge by the shuttle traffic (`v_accvgpr_read/write/mov` counts in the
hot loop) and by wall time, not by `accum_offset` or register totals.
- A scaled MFMA needs `dst` and `src2` in the same file but takes srcA/srcB from either. FP4 therefore
  swaps register files: the accumulator slices live in arch VGPRs and the A/B fragments in AGPRs, so the
  C store packs straight out of arch VGPRs with no shuttle (`_MXFP4_ARCH_ACC`, FP4:578-582).
- The same trade on the fp8 NT whole loop "measured a wash" (FP8:2045-2047): an MFMA writing an arch
  VGPR that a VALU then reads needs wait states that the single wave per SIMD has nothing to fill.
- *(field, E17)* At unchanged occupancy, freed registers buy nothing on their own. In one attention kernel, releasing 72
  VGPRs and 72 AGPRs cost −2.05%. Name the specific gain the headroom is for before building it.

**Setting `amdgpu-agpr-alloc` and friends.** Turbo passes these through `value_attrs`
(`make_value_attrs`, GH:1580). Put them under `"llvm.passthrough": [...]`. A bare `"passthrough": [...]`
key is silently dropped at `convert_gpu_to_rocdl`. A dropped attribute gives no error. An inline-asm AGPR
accumulator then gets the default allocation, and the result is wrong. After setting any kernel attribute,
read `.agpr_count`, `.vgpr_count` and the waves-per-eu fields from the ISA (`FLYDSL_DUMP_IR=1`, §10).

## 5. MFMA atom choice

- FlyDSL's guide lists the atom per dtype (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §6.2)). On gfx950, PS switches
  fp8 to the 16x16x128 form when `tile_k % 128 == 0` (`use_mfma_scale_128`, PS:159).
- fp8 and MX formats on gfx950 use `v_mfma_f32_16x16x128_f8f6f4`, or its scaled form for MX. The
  operand formats come from `cbsz` (A) and `blgp` (B); the operand tuple width must match the format
  (fp4 takes 4 dwords).
- **bf16: prefer 16x16x32 to 32x32x16.** Both cover the same block with the same accumulators, operand
  bytes and `ds_read` count, but each MFMA rewrites its whole accumulator tile, so the smaller atom moves
  fewer register-file bytes per MAC. *(field, E18)* In one bf16 grouped GEMM with 256×256 tiles at the power cap, the
  swap raised the clock from 1.37–1.40 GHz (a recorded baseline) to 1.667 GHz and cut the Down projection's
  dispatch time 10.8% (the scored gain over both projections was +9.2%), while cycles per dispatch rose 8% and
  MFMA utilization fell from 84% to 77%.
  `SQ_VALU_MFMA_BUSY_CYCLES` was bit-identical across the swap.
  - **Rank atoms on wall time and energy, never on utilization counters.**

## 6. K tails inside the kernel

Turbo's fp8 kernel accepts any K ≥ 129 without a padding copy; its MXFP4 kernel needs K % 64 == 0, and
its MXFP8 kernel needs K % 128 == 0 and K ≥ 256. FlyDSL's shipped kernels require K to be a multiple of
their K tile.
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
  correct. See `flydsl_gemm_datapath.md` §10.

## 7. Occupancy and allocation knobs

- 8-wave candidates race `waves_per_eu=2` with different AGPR budgets: Turbo's NT table lists
  32, 48 and 64 (`_NT_CANDIDATES`, FP8:3193).
- **Reach a register tier by shortening A's live range, not by a hint.** When A's registers set the occupancy tier,
  read A from LDS in 16-row groups one to three ahead of their MFMAs instead of a whole K tile ahead. On fused-MoE
  fp8 × MXFP4 expert GEMMs: 202 → 168 VGPRs (2 → 3 waves per SIMD) and 7.6–7.9% off the call, where
  `waves_per_eu=3` on the old loop spilled 210 VGPRs and ran 2.4× slower *(field, E23)*; 202 → 148 VGPRs and
  −6.1/−6.8% MoE time *(field, E22)*; 146 → 122 VGPRs, after which `waves_per_eu` 4 built at 128 VGPRs with no spill
  where it had spilled 19–43 *(field, E21)*.
- **Fully unrolling a short K loop (`range_constexpr`) pays only with VGPR headroom.** *(field, E19)* At 12–24 K steps:
  ~1–5% faster on ~100-VGPR tiles, ~40% slower on one 256-VGPR tile (spills 9 → 59) and about 1% slower on another (its ISA was not read).
- The 4-wave whole loop runs at `waves_per_eu=1` and uses the full register file
  (`make_value_attrs(1, 0, "256,256")`, FP8:1992). FP4 sets `amdgpu-agpr-alloc=256` through
  `llvm.passthrough` (FP4:3095).
- Arch VGPRs and AGPRs share one 512-entry file per SIMD. Allocation is in granules of 8, and the
  occupancy tiers are 64/72/80/96/128/168/256 total registers (8 down to 2 waves per SIMD).
  Arch registers are rounded up to 4 before the AGPRs are added: 101 arch + 21 AGPR is 125 total.
  FlyDSL's guide has the same model with a worked VGPR estimate for a `preshuffle_gemm` tile
  (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §10)).

## 8. FlyDSL's own GEMM kernels (v0.3.2): what to know when calling or tuning them

"PS" is `kernels/gemm/preshuffle_gemm.py` and "A16" is `kernels/gemm/gemm_a16w16_gfx950.py`. They handle no
K tail (K must be a multiple of the K tile), and the caller picks the configuration. PS and the MXFP4 kernels need B pre-shuffled on the host; the fp8 4-/8-wave kernels take either
layout (`b_preshuffled`).

| Kernel | dtypes | Loop | Notes |
|---|---|---|---|
| PS `compile_preshuffle_gemm` | fp8, int8, fp16, bf16 | 4 waves along N; A through LDS, pre-shuffled B straight to VGPR | per-row × per-col scales; `use_async_copy` is gfx950-only |
| A16 `gemm_a16w16` | bf16, fp16 | `m×n×k` waves (default 2×4×1), async DMA for A and B, `stages` deep | split-K by atomics (not bitwise reproducible), `group_m` XCD swizzle, half-tile interleave |
| `fp8_gemm_8wave.py`, `fp8_gemm_4wave.py` | fp8 (row-wise scales) | the §2 pipeline / a 4-wave per-MFMA-asm loop | NT only, `K % 128 == 0` |
| `fp4_gemm_4wave.py` | MXFP4 | 2×2 waves, 8-buffer LDS ping-pong, scales through LDS | host-side scale and weight preshuffle |
| `mxfp4_preshuffle.py` | A MXFP4/MXFP6/MXFP8 × B MXFP4/MXFP8 | 4 waves along N; A by DMA into a 2-buffer LDS ring; preshuffled B and the scales loaded straight to registers | host-side scale and weight preshuffle |

Facts about PS and A16 that matter when tuning them:
- **PS tile algebra.** `tile_K_perm = 128 if mfma_scale else (64 if 8-bit else 32)` (PS:171). Four waves
  split N. The A tile must be a multiple of 4096 B, or its tail is never loaded (PS:190).
- **PS waits.** `rocdl.s_waitcnt(num_b_loads)` passes a raw bitfield, not a `vmcnt`. `s_waitcnt(8)`
  emits `vmcnt(8) expcnt(0) lgkmcnt(0)`. In your own code write `rocdl.s_waitcnt(vmcnt=n)`.
- **PS scheduler hints.** `sched_group_barrier` groups spread VMEM reads (mask `0x020`), DS reads and DS
  writes over the MFMAs. Without `preload=`, 8-bit tiles on gfx950 look up `_TILE_PRELOAD_TABLE`
  (PS:19), and every other dtype or tile gets `(0, 0)`.
- **PS epilogue.** Ragged M is bounded by the descriptor (`max_size=False, num_records_bytes=...`), not
  by a predicate.
- **A16 waits.** A16 waits with `rocdl.asyncmark()` / `rocdl.wait_asyncmark(stages-2)` around
  `cdna4.BufferLoadAsyncLDS128b`, which counts DMA groups instead of `vmcnt`. This is the in-DSL
  alternative to hand-written waits.
- **Configuration space.** The arguments are the entry-point signatures in `INDEX.md`'s kernel map; aiter's
  tuned rows for its own bf16 GEMM are in `flydsl_gemm_small_m.md` §7.

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

`FLYDSL_DUMP_IR=1` writes every pass to `<FLYDSL_DUMP_DIR or ~/.flydsl/debug>/<kernel>/`. It skips the
disk-cache lookup, so the dump reflects the current source. Do not time a run with it on.
`ARCH=gfx950 COMPILE_ONLY=1` compiles without launching.
- Count instructions inside the hot loop's label range only. Compare them with the plan: MFMAs per phase,
  `ds_read`s per fragment, `buffer_load … lds` per fill, and one `s_waitcnt`/`s_barrier` per phase.
  FlyDSL's guide gives the expected per-tile counts for `preshuffle_gemm` and target ratios such as "MFMA ≥ 40%
  of the loop" (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §9)). Use the ratios to spot a loop body far from its plan, not
  to rank changes: *(field, E20)* deleting 7.4% of a hot loop's instructions, all SALU and `s_nop`, gained 0% on one A6W4 GEMM (the same edit laid A out wave-major in LDS; LDS, VMEM-load and wait counts were unchanged)
  (`flydsl_gemm_attribution.md` §4).
- Check `.agpr_count`, `.vgpr_count`, `.amdhsa_next_free_vgpr` and the scratch size after every
  register-plan or attribute change (§4).
- For a whole-loop kernel, check that the back edge has its `s_cmp` and that no compiler `s_waitcnt`
  sits between the asm phases.

## Related
`flydsl_gemm_grid_pricing.md` · `flydsl_gemm_schedule_and_locality.md` · `flydsl_gemm_datapath.md` ·
`flydsl_gemm_epilogue.md` · `flydsl_gemm_autotune.md` · `flydsl_gemm_correctness.md` ·
`../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The Kimi-K3 forge runs installed `flydsl==0.3.2`. The month is the run's where the record dates it; where it
does not, the line says which date it is. "Turbo" is Primus-Turbo.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, per-arm timer not stated (the run's baseline is the driver bench, warmup 3, 20 iterations); timing-only builds, output not checked] global→VGPR→`ds_write` staging after compute at m=16384: A 1.1660, B 1.1804, both 1.2080 ms, against 1.0529 ms for the DMA loop in the same session (+10.7% to +14.7%) (gemm_n6144_k7168 forge run, iter 5; 256×256 tile, 2×4 waves, 2 stages, BK 64; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_005.md:28)
- E2 [2026-07, gfx950, FlyDSL ?, N=1, rocprofv3 PMC, one profiled run per K] SQ_INSTS_LDS at K=28672: 2.202e7 for the 8-wave intrinsic build against 1.481e7 for the 4-wave whole loop, 1.49× (K=16384: 1.258e7 against 8.52e6) (Turbo MXFP4 dense GEMM, M=N=8192; record dated by its file time, 2026-07-01; local:/workspace/code/gpt_oss_docker/.claude/memory/diag_4w_vs_8w.md:124; summarized in myskill methodology/06-register-wholeloop-emit.md:256)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, same-session best-min TF/s with the median in parentheses, every arm SNR 55.6 dB and deterministic] 4-wave whole loop against the 8-wave intrinsic build: K=8192 4977.8 (4808) against 4676.3 (4537), +6.4%; K=16384 +9.0%; K=28672 5573.0 (5534) against 5042.0 (5003), +10.5% (Turbo MXFP4 dense GEMM, M=N=8192; record dated by its file time, 2026-07-01; local:/workspace/code/gpt_oss_docker/.claude/memory/diag_4w_vs_8w.md:86; myskill methodology/06-register-wholeloop-emit.md:257)
- E4 [2026-07, gfx950, FlyDSL ?, N=?, the kernel test's SNR and run-to-run determinism output] on the MXFP4 4-wave whole loop, merging the phase's two barriers into one (`FP4_INPLACE_1BAR=1`) raced: K=8192 SNR 55.6 dB but det 214684 (≠ 0), K=28672 SNR 46.4 dB and det ≠ 0; myskill's race table: with the vmcnt drain fully emptied the race remained, with a barrier in every phase it was gone (Turbo MXFP4 dense GEMM, 8192×8192×{8192, 28672}; local:/workspace/code/gpt_oss_docker/.claude/memory/mxfp4_4w_ceiling.md:47; myskill methodology/02-correctness-race-cache.md:105-108, whose kernel and date are not stated)
- E5 [2026-06, gfx950, FlyDSL ?, N=?, ?] Turbo's source comment on `nt_vmcnt`: an end-of-iteration vmcnt(3) gave det=0 at ≤1.1% cost, 4 and above raced, below 3 cost more; no run record is kept, and the month is that of the commit that added the comment (Turbo FP8 dense NT 8-wave kernel `_compile_dense_nt`; comment added in e7ecaf26 (#356); Primus-Turbo 7615a5a2:primus_turbo/flydsl/gemm/gemm_fp8_kernel.py:92)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver --bench-mode --bench-case, warmup 3, 10 iterations; baseline a full driver bench, 20 iterations] `s_setprio` around the MFMA groups at m=16384: 1.1077 against 1.0602 ms (+4.5%) (gemm_n6144_k7168 forge run, iter 4; 256×256 tile, 2×4 waves, 2 stages; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_004.md:15)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, driver bench, warmup 3, 20 iterations, one run] `s_setprio` around each MFMA cluster of the tiled path: m=893 0.084402 against 0.080896 ms (+4.3%), m=2732 0.190357 against 0.187004 (+1.8%), m=9397 0.642948 against 0.637774 (+0.8%), m=16384 1.170480 against 1.148148 (+1.9%) (gemm_n6288_k7168 forge run, iter 6; 8 waves per workgroup, 2- and 3-stage rows; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6288_k7168/forge_experiments/lessons/iter_006.md:21)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, own harness: L2 flushed by zeroing 512 MB, CUDA events, median of 20–60 calls, 3–4 alternating rounds] the same change, ranges against canonical: m=893 103.0–104.8 against 102.8–107.2 µs, m=2732 197.6–212.1 against 193.4–239.0, m=9397 638.1–676.1 against 629.3–729.3, m=16384 1115–1134 against 1097–1185 (gemm_n6288_k7168 forge run, iter 10; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6288_k7168/forge_experiments/lessons/iter_010.md:53)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=4, scratch script: 10 launches per CUDA graph, 512 MB flush before each replay, median of 20 replays per repeat, 4 interleaved repeats] `s_setprio` around each MFMA cluster: m=9397 904 / 907 → 916 µs, m=16384 1616 → 1632 µs (about +1%) (gemm_n8448_k7168 forge run, iter 8; 320×256 and 416×192 tiles, 8 waves, 2 stages; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n8448_k7168/forge_experiments/lessons/iter_008.md:10)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=?, scratch harness alternating the kernel and the variants on AKLR bench_cudagraph, 8 timing sets] a third LDS stage for A or for B at m=9397: 132.77 and 133.20 µs against canonical 150.03 (−11.5%, −11.2%); canonical with the loop-top vmcnt wait removed (wrong output) 130.47 against 150.04 (−13.0%) (gemm_n1280_k7168 forge run, iter 13; 192×256 tile, BK 64, 2×4 waves, 2 stages; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n1280_k7168/forge_experiments/lessons/iter_013.md:17)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=5, bench_cudagraph in process with the incumbent, median of 5] A×2 + B×3: m=16384 1068.40 against 1054.24 µs, m=9397 624.52 against 624.95, m=2732 183.66 against 183.31; A×3 + B×2: m=16384 1077.02 against 1035.32, m=9397 635.73 against 625.21 (−0.1% to +4.0%) (gemm_n6016_k7168 forge run, iter 5; tiles 352×192, 224×256, 256×256, 2×4 waves; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6016_k7168/forge_experiments/lessons/iter_005.md:23)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, ?] myskill records the whole-loop bare-asm rewrite as the key lever of Turbo's MXFP4 4-wave kernel, from 3583 (intrinsic) to 5401, which it calls +16%; the two throughputs give +51%, and the line names neither the shape nor the run; the month is that of the myskill commit that recorded it, 99db6eec (Turbo MXFP4 4-wave dense GEMM; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/06-register-wholeloop-emit.md:11)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, campaign ruler `_bench_tw_vs_mx.py`: geomean of t_mx / t_tw against a frozen MXFP8 kernel, arms interleaved in one process, median] emit-order perturbations of the 4-wave whole loop, as the wgrad group's ratio: baseline 0.9696; refill two MFMAs later 0.9604; all G2S at the head of the phase −15% (overall 0.93161 against 0.98644); fills spaced by instruction count −8%; each `m0` write one slot ahead of its load 0.9629 (gpt-oss tensorwise fp8 grouped GEMM, weight-gradient pass (TN, variable K), `kernel_grouped_tn_wgrad_4wave`; the round's harness step timed out, so these are the round agent's own readings; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260731_073404/rounds/round-10/agent.txt:14)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=?, per-arm timer not stated (the run's baseline is the driver bench, warmup 3, 20 iterations)] G2S fills spread into the MFMA stream: m=16384 spread 1/2/4 1.0563/1.0631/1.0953 ms against 1.0529 unspread (+0.3% to +4.0%); spread 1 at m=893 0.08205 against 0.08207, m=2732 0.18101 against 0.17989, m=9397 0.62490 against 0.61983 (gemm_n6144_k7168 forge run, iter 5; 2×4 waves; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n6144_k7168/forge_experiments/lessons/iter_005.md:26)
- E15 [2026-09, gfx950, FlyDSL ?, N=1, the round's correctness gate: bitwise equality with the base build plus SNR, 10 shapes] traded accumulators in arch VGPRs with `"={v[b:b+3]}"`: 7 of 10 shapes neither bitwise equal nor above 1.77 dB (0.03–1.77 dB); `out_proj_fwd` 32768×4096×4096 passed while `KI16_wide` 1024×2048×4096, same K and K iterations, failed; with `=&` 12 of 12 shapes bitwise equal and the ISA byte-identical (Turbo MXFP4 dense GEMM on the Llama training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/memory.md:227)
- E16 [2026-09, gfx950, FlyDSL ?, N=1, one run and its ISA dump] a scf.for-rolled persistent tile loop around the whole-loop asm, without `~{scc}`: GPU memory fault on the first run, back edge a bare `s_cbranch_scc1` with no `s_cmp`; with the clobber the shipped, unrolled path stayed byte-identical (4696 instructions, 496 VGPRs) (Flux MXFP4 dense GEMM, `call_mxfp4_wholeloop`; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260927_035256/rounds/round-03/agent.txt:33)
- E17 [2026-09, gfx950, FlyDSL 0.2.4, N=1, campaign score: geomean of reference/current time over three batch sizes times a guard term, one scored run] on an attention kernel, `g3_kreg=False` freed 72 VGPRs and 72 AGPRs (479 → 407, 223 → 151, spill 0, still one wave per SIMD) and scored 0.9795 (−2.05%) (gpt-oss D=64 attention backward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:176)
- E18 [2026-08, gfx950, FlyDSL ?, N=1, rocprofv3 PMC run of 10 dispatches against a recorded baseline; the campaign score for the whole round, 3 reads] Mfma32x32x16 → Mfma16x16x32 in the 8-wave 256×256 forward NT tile: the Down projection's dispatch 1803 → 1608.9 µs (−10.8%), sclk 1.40 → 1.667 GHz at a pinned 1400 W (the same record also gives the baseline as 1.37 and as 1.37–1.40 GHz), cycles per dispatch up ~8%, MFMA utilization 84% → 77%, SQ_VALU_MFMA_BUSY_CYCLES 2.12337e9 on both; campaign score +9.17% (1283.9 → 1401.7 TF/s geomean over 8 configurations) (bf16 grouped GEMM forward, gpt-oss-20b MoE: 32 experts, M=131072, K=2880, N=5760 and 2880; Primus-Turbo ed8d7af4 (#486); local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_063921/rounds/round-15/agent.txt:3)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=2, temporary CUDA-graph bench, median of 10 replays of 50 calls, two readings per arm against canonical; VGPR and spill counts from the distill probe's ISA dumps] fully unrolled K loop: m=893 (12 K steps, 102 VGPRs) 10.30 and 10.29 µs against 10.94 and 10.90; m=2732 (24 K steps, 104 VGPRs) 22.97 and 22.68 against 23.48 and 23.69; m=9397 (24 K steps, 256 VGPRs, spills 9 → 59) 79.65 and 79.32 against 56.16 and 56.23 (+41%); m=16384 (256 VGPRs, ISA of the unrolled build not read) 93.10 and 94.59 against 92.81 and 93.50; the run's ablation of the kept rows (m=893, m=2732) +1.09%, one-sided 95% bounds +1.03% … +1.14% (gemm_n2304_k1536 forge run, iter 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2304_k1536/forge_experiments/lessons/iter_007.md:12)
- E20 [2026-10, gfx950, FlyDSL ?, N=3, campaign score: training-time-weighted harmonic mean TF/s of six shapes, three reads per arm] A's K-tile offset moved into `soffset` and A's LDS laid out wave-major: hot loop 949 → 879 instructions (SALU 98 → 52, `s_nop` 42 → 18), ds_read, VMEM-load and s_waitcnt counts unchanged; score 3964.08 / 3954.31 / 3951.09 against 3950.12 / 3952.65 / 3956.84, 0% (Turbo A6W4 dense forward GEMM, six training shapes; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20261010_100503/memory.md:178)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=?, ranges over repeated reads, stage-2 kernel time in one process and ISA resources] row-group-major MFMAs with each 16-row group's A `ds_read` issued one group ahead (two groups live instead of four, BM 64): 122 against 146 VGPRs, no spills, 175.4–176.3 against 180.9–181.6 µs, output bit-identical (local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_l/workspace/forge_experiments/lessons/iter_012.md:29); the same extended to the last K tile with `compile_hints["waves_per_eu"]` 4: 128 VGPRs, 0 spills, no scratch (iter_018.md:10 in the same folder), kept at +0.29% (+0.25…+0.34%, local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_l/workspace/forge_experiments/ablation.md:23); the hint without it: 128 VGPRs with 43 spilled (iter_008.md:21, 0.5498 against 0.4411 ms) and 19 spilled (iter_017.md:14) (fused_moe situv2 fp8 activations × MXFP4 down weights, 896 experts, top-16, n_tokens 2732, stage 2 BM64 BN256 BK128)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2, stage 1: driver bench (warmup 3, 20 iterations), one run of the base and two of the change; stage 2: torch-profiler kernel time, 10 calls, cache cleared] stage 1, whole A tiles preloaded from LDS into VGPRs and carried across the ping-pong halves (aiter's `prefetch_full_a_from_lds`, `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:1312`) at 202 VGPRs: 1.1423 / 1.7838 ms; A read per 16-row slice one phase ahead at 148 VGPRs: 1.0728–1.0737 / 1.6618–1.6629 ms at 9397 / 16384 tokens; the run's ablation of that round +7.53% (bounds +7.20% … +7.86%); stage 2, the same change: 146 → 122 VGPRs, 418.6 → 384.9 / 586.1 → 547.4 µs (fused MoE, fp8 activations × MXFP4 experts, 896 experts, top-k 16, inter 384, model dim 3584, stage 1 tile 64×128×256; local:/shared_nfs/kyle/moe/kf/runs/tp8_cmp/workspace/forge_experiments/lessons/iter_002.md:9; lessons/iter_002.md:21; lessons/iter_005.md:18; ../ablation.md:14)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=2, forge driver --bench-mode (graph timer, warmup 3, 20 iterations), whole call at n_tokens 9397 / 16384; registers from the ISA] A streamed from LDS in 16-row × 128-K sub-tiles, 3 in flight, against the whole A tile read from LDS one step ahead: 0.796 / 1.247 (re-run 0.797 / 1.246) against 0.864 / 1.350 ms, main kernel 202 → 40 + 128 = 168 VGPRs, scratch 0; forge ablation, 3 pairs: the version without it +8.50% [+8.26, +8.75]; `compile_hints={"waves_per_eu": 3}` on the old loop: 168 VGPRs, 210 spilled, 380 B private segment, 2.059 / 3.217 against 0.861 / 1.352 ms (fused MoE SiTUv2 EP8 stage 1, 64×128×256 tile, LDS 50,176 B; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_002.md:3; :18; :21; iter_001.md:13; :17; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/ablation.md, row f417a2c38b53)
- E24 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm (the kept arm: forge ablation, 3 pairs), forge driver --bench-mode, whole call at n_tokens 9397 / 16384; ISA not read] 12 VMEM loads per K step (A scales, w1 scales, w1 k0, w1 k1) at 240 VGPRs: all at the step head 0.7198 / 1.1069 ms; 8 at the head and 4 after MFMA group 0: 0.7165 / 1.0912; (4,2,2,2,2,0) over five groups: 0.7244 / 1.1052; ablation of the kept split: +1.12% [+1.05, +1.19] (fused MoE SiTUv2 EP8 stage 1, 64×256×256 tile; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_015.md:13; :14; :15; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/ablation.md, row 58fc121f7e81)
- E25 [2026-10, gfx950, FlyDSL 0.3.2, N=2 sequential rounds per comparison on one GPU, torch.profiler stage-1 kernel time with L2 flushed; rocprofv3 PMC on one call per build] the pair tile (two 64-row groups sharing W1 through LDS, 8 waves, 128 VGPRs, 44 spilled, 180 B scratch, 81,920 B LDS) against the 4-wave 64×256×256 tile (234 VGPRs, 50,176 B LDS): 549 / 793–810 against 479–481 / 684–689 µs at n_tokens 9397 / 16384, and 553 / 814 against 472 / 697 in another session; pair ÷ 4-wave: MFMA instructions and busy cycles 1.00, GRBM_GUI_ACTIVE 1.20 / 1.22, LDS instructions 1.59 / 1.51, SQ_WAIT_INST_LDS 2.80 / 2.14; the pair with 2 waves per group (`waves_per_eu=2`, 256 VGPRs, no scratch): 536.4 / 749.9 and 531.2 / 740.8 against 513.5 / 738.0 and 504.8 / 729.4 µs (fused MoE SiTUv2, fp8 activations × MXFP4 weights, 112 of 896 experts local, top-16, model dim 3584, inter 3072; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_021.md:6; :10; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_019.md:10; :35; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_022.md:15)
