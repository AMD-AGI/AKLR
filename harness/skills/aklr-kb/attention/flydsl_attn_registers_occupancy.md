---
title: "FlyDSL attention — register budget, occupancy steps and MFMA–VALU overlap"
kind: topic
lever: flydsl_attn_registers_occupancy
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, gqa]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:73
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:427
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1036
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:1614
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:52
---

# Registers, occupancy and MFMA–VALU overlap in a FlyDSL attention kernel

> Attention bodies carry an O accumulator, Q packs, a score tile and softmax state at once, so the
> register file sets the occupancy and the occupancy decides which overlap tricks can pay. The register
> model is in `../gemm/mainloop.md` §7 (one 512-entry pool per SIMD shared by arch VGPRs and
> AGPRs, granule 8, arch rounded to 4 before AGPRs are added) and workgroup-quantized occupancy in
> `../gemm/grid_pricing.md` §4.
> This card adds the attention rules.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You want one more wave per SIMD, or a change moved `.vgpr_count`, spills or the scratch size.
- You plan dual-wave, ping-pong, warp specialization or "hide exp under the MFMAs".
- Freed registers or AGPR pinning did not pay.

## 1. Read the budget from the ISA
- Waves per SIMD = `min(8, floor(512 / roundup8(align4(arch) + agpr)))`, then rounded down to whole workgroups
  (steps: `../gemm/grid_pricing.md` §4). In these cards *occupancy N* means N co-resident
  workgroups per CU, which is what decides whether sibling waves can cover each other. A wave
  addresses at most 256 arch VGPRs; AGPRs come out of the same pool.
- Read `.vgpr_count`, `.agpr_count`, `vgpr_spill_count` and `private_segment_fixed_size` from the
  `FLYDSL_DUMP_IR=1` dump. rocprof's register columns are wrong (`../gemm/attribution.md` §9).
  *(field, E1)* A sparse-MLA backward dQ kernel reported as 172 VGPR / 0 AGPR was 256 + 81 = 337: one wave per SIMD.
- For an 8-wave workgroup only 2/4/6/8 waves per SIMD are reachable. *(field, E2)* An 8-wave forward at 4
  waves/SIMD had to stay at ≤ 128 registers; at 3 waves/SIMD the second workgroup does not fit, so the
  third wave slot buys nothing.
- FlyDSL v0.3.2: the occupancy target is the `waves_per_eu` compile hint (`maxnreg` also exists).
  FlyDSL's gfx950 forward defaults to `waves_per_eu=2`; its D=128 causal kernel compiles to 246 VGPRs,
  0 AGPRs, no spills. Kernel attributes through `value_attrs` go under `"llvm.passthrough"`
  (`../flydsl/gemm.md` § INDEX FlyDSL version).

## 2. Force occupancy with `waves_per_eu` only when the code already nearly fits
- *(field, E3)* Allocation 130 → 128 registers with 2 spills: +8.5%, the deployed setting. Beyond that the allocator
  meets the target by spilling. *(field, E4)* Forward, 3 → 4 waves/SIMD: 128 registers, spills 46 → 213, hot loop
  244 → 343 instructions, 5.47× slower. *(field, E5)* A sparse-MLA backward dQ kernel forced to `waves_per_eu` 2: MFMA
  utilization 18% → 9%, 2× slower; to 3: 4× slower.
- To climb a step, delete live values. The reverse trade is often cheap: *(field, E6, E4)* dropping a forward from
  3 to 2 waves/SIMD to clear all spills cost 3.2–3.9%.

## 3. Price occupancy in steps
- Price a step before building for it. *(field, E3)* +8.5% on a forward whose code already nearly fit the next step
  (130 → 128 registers, 2 spills) when only `waves_per_eu` changed (§2). Padding LDS with a dead buffer to force one
  workgroup per CU, everything else unchanged: *(field, E2)* −9.9%. *(field, E7)* The same dead-LDS probe on a
  sparse-MLA decode producer that halved its occupancy cost 0.1–0.2%, which closed a whole register-squeeze family
  without writing it.
- At the power cap part of an occupancy gain returns as clock. *(field, E8)* 3 → 4 waves/SIMD at 96.6% of TBP:
  sclk −3.3%, wall 0.5% faster, an estimated 3.9% fewer cycles (from wall and sclk). Report sclk with every
  occupancy arm (`flydsl_attn_measurement.md`).
- A change that drops a step loses even when it removes traffic: GQA merging past the step is in
  `flydsl_attn_grid_dispatch.md` §9.
- **Registers saved without crossing a step are worth 0 or less.** *(field, E9)* At occupancy 1, dropping an
  AGPR tie that freed 14 registers: −0.80% at batch 4 (batch 1 inside noise). *(field, E10)* Sharing VOP3P splat
  constants (20 → 4 VGPRs, `.vgpr_count` unchanged): −0.9% at 3 waves/SIMD, −0.61% at 2. The 72+72-register
  example is in `../gemm/mainloop.md` §4.
- Exception: while spills or a private segment exist, moving a value out of the loop carry pays without a
  step. *(field, E11)* A 16-dword loop-carried softmax seed rebuilt by scalar broadcast at each use: private
  segment 188 → 16 B, +0.40%; LICM hoists the broadcast, and the saving lands in prologue/epilogue spills.

## 4. Before squeezing, find where the peak is
- Print the highest VGPR index referenced in each basic block. *(field, E12)* Nine blocks peaked at 142–149
  with `.vgpr_count` 150; of three liveness cuts, two left the allocation unchanged and one raised it
  to 154. Only
  a block clearly above the others is worth local work; otherwise change the resident set.
- Liveness follows the scheduled order, not the text. *(field, E13)* Reordering to end the coexistence of two
  score tiles (64 registers together) freed 4 registers.
- The budget is the shape of the live intervals, not a dword count. *(field, E14)* Freeing 16 loop-carried
  dwords did not make room for a knob that needed 8 (still 12 spills).
- MFMAs that consume a value anchor its allocation in place. *(field, E15)* Deleting 8 row-sum MFMAs: spill
  118; replacing half with VALU: 93; *(field, E14)* moving 2 of them: spill 4 → 8.

## 5. Spill scale
- *(field, E16)* At 128 VGPRs, 4 waves/SIMD: 5 spills −0.3%, 10 spills −0.2%, 35 spills −19%. Single-digit
  spills cost 0.2–0.4%; 30+ is a cliff.
- Spill 0 is not a goal. *(field, E14)* Reaching 0 by dropping a K prefetch that covered the first QK MFMAs:
  −1.1%. Re-sweep "keep a little more resident" knobs at the current spill level.
- Gate every arm on the ISA spill count before timing; a dump takes seconds, a bench round does not.

## 6. Code shapes that spill
- **A loop remainder is a third body instance.** Register pressure scales with body instances. *(field, E17)* 2× unroll
  of a four-deep K/V ring plus a tail loop: spill 2 → 35, −19% (spill 30 with a 2-deep ring); removing the tail:
  35 → 7 (10 → 0 with a shallower K prefetch). Write the remainder as a predicated second body: a runtime
  `if j + 1 < end:` on a workgroup-uniform condition (the rewriter lowers it to `scf_if_dispatch`), yielding the
  accumulators from both branches. Wrapping one body instance in a workgroup-uniform `if` whose `else` yields the
  accumulators is harmless; a branch holding its own body copy is not (below); *(field, E18)* a conditional barrier
  inside a branch, with K held fully resident, spilled 97.
- **Specialize with sequential ifs, not nested ones.** Each runtime `if` returns its accumulators and the `else`
  yields them unchanged. *(field, E19)* Nested: 476 registers (arch + 219 AGPR), above the 464 that kept a co-resident
  reduce kernel resident; sequential: 460,
  spill 0, 468 fewer instructions in the masked loop.
- **`scf.if` around a large region keeps loop invariants (Q packs, LDS bases) live across it.** *(field, E20)*
  A per-wave phase-reversal vehicle was 10.6% slower by itself; duplicating the main loop per branch, 27% slower.
- Runtime conditional syntax: `../flydsl/debug-flydsl-kernel.md` § Runtime versus compile-time
  conditionals.

## 7. AGPRs are not free space
- *(field, E21)* Forcing `amdgpu-agpr-alloc` to 32/64 to absorb 31 spilled dwords raised spills to 52/77 (the
  second also added 26 `v_accvgpr` moves).
- Pinning a value in AGPRs (tied asm `"=a,0"`) is zero-sum when the register budget is already spent: the allocator
  evicts another AGPR resident instead of growing the AGPR count. *(field, E22)* Pinning a 32-dword pack: `.agpr_count`
  223 → 219, another operand group pushed out, shuttle traffic up, −0.49%. Check that `.agpr_count` rose before
  trusting a pin.
- Pin only values an MFMA reads directly and that stay fixed for a kv block. *(field, E23)* Two such operand
  groups: +0.79% and +0.53%; two per-head-step groups: −3.4% and −2.6%.
- VALU cannot read AGPRs, so every VALU use costs a `v_accvgpr_read`. Judge pins by the
  `v_accvgpr_*` count and wall time, not `accum_offset`; untied pins delete the operand
  (`../gemm/correctness.md` §7).

## 8. MFMA–VALU co-execution
What the rules below rest on (`flydsl_attn_mfma_layout.md` §1, `flydsl_attn_softmax.md` §1): *(field, E24)* a
`16x16x32` bf16 MFMA leaves an 8-cycle VALU shadow and a `32x32x16` 24, shared by every wave on the SIMD; an exp uses
it twice as fast as a multiply and gains nothing over other VALU work from MFMA overlap.
- Read coexec / MFMA busy and the stall pool as in `flydsl_attn_attribution.md` §2–§3.

**Two co-resident workgroups per CU (backward): the overlap is already free.** Two independent workgroups sharing each
SIMD fill each other's barrier and drain bubbles with no barrier between them. In-workgroup overlap trades that for a
barrier tax. *(field, E25)* 8-wave single workgroup: MFMA utilization 53 → 40%, −11%; adding a phase stagger recovered
only to 1029, in a build with a WAR race (NaN output), against the 4-wave baseline's 1116; *(field, E26)*
2-producer/2-consumer role split: spill 80 and 254 TF at occupancy 2, 291 TF at occupancy 1 with each GEMM on half the
waves, against ~1100 for the baseline; *(field, E27)* cross-head double buffering: 31% and 42% slower.
Check whether the baseline already co-hosts two workgroups per CU before building any of these.

**Hiding exp under MFMAs with two co-resident workgroups is worth ~0.** *(field, E28)* A `sched_barrier(0)`-forced
interleave of the pack/dS VALU 0.00%; *(field, E29)* counted `sched_group_barrier` MFMA:TRANS 1:2, 2:4, 4:8: +0.26%,
+0.04%, +0.05%, all inside noise. Only removing instructions moves wall there (`../gemm/attribution.md`
§8).

**One workgroup per CU (forward) is the only regime where in-workgroup dual-wave scheduling can pay.** FlyDSL v0.3.2's
gfx950 forward is the reference: 8 waves, split into two groups by `wave_id // 4`; group B issues one extra
`s_barrier` at entry and group A one at exit, so the groups run one cluster apart, one loading while the other
computes. Each loop iteration is 8 clusters separated by `sched_barrier(0); s_barrier; sched_barrier(0)`, alternating
memory and compute, with compute clusters wrapped in `s_setprio(1)`/`s_setprio(0)`. It is one answer, not the only
one: *(field, E30, E31)* on 8-wave forwards running two workgroups per CU (4 waves/SIMD), turning the stagger on was
4.8% and 6.3% slower; price it on your own kernel at one workgroup per CU. *(field, E37)* A 12-wave MLA decode
alone on its CU (3 waves per SIMD) lost 2.5–15% in all nine spill-free forms of two halves run half a step apart.

**At one workgroup per CU a role split can beat the stagger.** *(field, E35)* On a dense fp8 MLA decode, two
staggered groups of three waves ran at 0.70–0.99× the speed in five builds; three QK + softmax waves with five P·V
waves a step behind (a three-tile ring, one barrier per step) were 9% faster, together with each K read shared by two
head tiles, and moving the fp8 conversion onto the P·V waves added 4–6.5% *(field, E36)*.

**Count what the shadow already hides before moving VALU into it.** *(field, E32)* Moving 16 more exps into a P·V
cluster whose 8 MFMAs already covered 16 `ds_read_b128` and 2 DMAs cost 2.3%; moving the whole softmax there wove
perfectly in the ISA and still lost 0.33%; moving that cluster's own 16 exps to the QK cluster, which had free VALU
slots, was neutral (−0.27%). At two or more waves per SIMD the clusters are still not interchangeable.

**Independent MFMAs placed in a VALU window get hoisted out.** *(field, E33)* Splitting a GEMM so 16 independent MFMAs
faced the softmax window: coexec 19.2% → 15.9%, −3.0%. Pin the region with `sched_barrier(0)` placed after the
`gpu.barrier()` that owns the LDS write and outside `range_constexpr` loops (`flydsl_attn_schedule_sync.md`).

**Static shadow accounting is a zero-GPU prefilter.** Price each instruction class against the shadow budget above
to discard candidates; rank survivors by measurement. *(field, E34)* Coverage distance, waitcnt and `s_nop` counts, MFMA
run length and pipe/slack models each failed to rank arms on at least one body.

## Verify
- `.vgpr_count`, `.agpr_count`, spills and private segment read from the ISA for every arm.
- The occupancy target is a reachable workgroup multiple, and the step was priced with a dead-LDS or
  `waves_per_eu` probe.
- Overlap work starts from the ISA and the measured residency: co-resident waves and no bare VALU between MFMAs →
  remove instructions; otherwise → schedule.

## Related
`flydsl_attn_schedule_sync.md` · `flydsl_attn_lds.md` · `flydsl_attn_attribution.md` · `flydsl_attn_softmax.md` ·
`flydsl_attn_backward.md` · `../gemm/mainloop.md` · `../gemm/grid_pricing.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the round's; "KB note" marks a line whose only record is a myskill or auto-memory note.

- E1 [2026-07, gfx950, FlyDSL ?, N=1, ISA dump (`num_vgpr`/`num_agpr`) of the isolated kernel against the rocprofv3 kernel CSV] rocprofv3 reported VGPR 172 / AGPR 0 for the dQ kernel; the ISA gave `num_vgpr=256` (arch cap) + `num_agpr=81` = 337, one workgroup per CU, no spill (DeepSeek-V4 sparse-MLA backward dQ, topk 128/192; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260709_144443/rounds/round-13/agent.txt:15; the 172/0 reading: local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_interm_register_transpose.md:40)
- E2 [2026-08, gfx950, FlyDSL ?, N=2, campaign bench (`bench.sh`, ms of the scored cell), two readings per arm in one session] pure-occupancy probe, LDS padded 34048 → 90112 B to force one workgroup per CU (vgpr 134 / spill 0 / 237 instructions, MFMA and exp counts unchanged): 2.0223 / 2.0401 against 1.8423 / 1.8466 ms, −9.9%; deleting the `rocdl.waves_per_eu` request: vgpr 132 / spill 0, the same one-workgroup-per-CU cliff, rejected at the ISA gate; the deployed kernel runs 8-wave CTAs at vgpr 128, two workgroups per CU (gpt-oss D=64 attention forward, B=4 Hq=64 Hkv=8 S=8192 causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:289)
- E3 [2026-07, gfx950, FlyDSL 0.2.2, N=?, campaign score: conv TF/s of `_bench_campaign_fwd16384.py`] the deployed code with only `waves_per_eu` changed: 4 (alloc 128, spill 2) 1213.7 against 3 (alloc 130, spill 0) 1118.6, +8.50% per occupancy step (Primus-Turbo hd64 flash-attention forward, Sq=Skv=16384, B=1, full causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-25/agent.txt:36; the alloc and spill per arm: local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:915)
- E4 [2026-09, gfx950, FlyDSL ?, N=?, ISA dump and the campaign's wall-clock timer] `waves_per_eu` 3 → 4: 128 VGPRs, spill 46 → 213, hot loop 244 → 343 instructions, 5.47× slower (an earlier round read 10.198 against 1.885 ms); 3 → 2: 204 VGPRs, spill 0, −3.9% (gpt-oss D=64 attention forward, dualwave, B=4 Hq=64 Hkv=8 S=8192 causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_082555/rounds/round-04/agent.txt:32)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, ?] the dQ kernel forced to `waves_per_eu` 2: VGPR 128, occupancy 3.82 → 7.5, scratch 340, MfmaUtil 18% → 9%, pro cr0 0.604 → 1.219 ms (2×); `waves_per_eu` 3: spill 736, 4× slower (DeepSeek-V4 sparse-MLA backward dQ; KB note dated 20260709 in its text, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dsv4_interm_register_transpose.md:42)
- E6 [2026-09, gfx950, FlyDSL ?, N=?, ISA gate and the campaign's wall-clock timer] `waves_per_eu` 3 → 2: vgpr 196, spill 31 → 0, −3.2% (gpt-oss attention forward, dualwave, d64 cell; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:360)
- E7 [2026-09, gfx950, FlyDSL 0.3.2, N=?, same-process probe, cold and warm, both ABBA orders] a dead LDS block raising `group_segment_fixed_size` 44800 → 82048 B (resident CTAs halved, checked in the ISA), instruction stream unchanged: 0.1–0.2% at seq96 (GLM-5.2 sparse-MLA decode producer; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-10/agent.txt:13)
- E8 [2026-07, gfx950, FlyDSL 0.2.2, N=?, campaign score (conv TF/s) with PMC occupancy, power and sclk] occupancy 2.985 → 3.978 waves/SIMD at 1352 → 1353 W (96.6% of TBP): sclk 1710 → 1654 MHz (−3.3%), wall +0.50% (Primus-Turbo hd64 flash-attention forward, Sq=Skv=16384, B=1; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-14/agent.txt:5)
- E9 [2026-09, gfx950, FlyDSL 0.2.4, N=2, campaign score against a same-session zero point] `mfma_tie=1` (vgpr 495 → 481, spill 0): b4 1.0010 / 0.9937 against the zero point 1.0054, −0.80%; b1 +0.48%, inside the 0.67–1.41% residual (gpt-oss D=64 attention backward, one workgroup per CU; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:414)
- E10 [2026-09, gfx950, FlyDSL ?, N=2, single-process timer (40 s warm-up, 15 s sampling), ABAB, mean, output signature checked] the two splat constants of 16 `v_pk_fma_f32` shared, 20 → 4 VGPRs, bit-identical output: d64 2.10848 / 2.12349 → 2.13452 / 2.13499 ms (−0.9%), d128 0.46203 / 0.46230 → 0.46489 / 0.46513 ms (−0.61%); `vgpr_count` 168 unchanged, spill 22 → 23; the d128 cell runs at 256 VGPRs (gpt-oss attention forward, dualwave; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:484; d128 registers: memory.md:333)
- E11 [2026-09, gfx950, FlyDSL ?, N=2, same-session ABBA after a 90 s warm-up; A/A control −0.07%; a 25 s warm-up read +1.43%] the 16-dword `-reference` seed taken out of the loop carry and broadcast from a scalar at each use: private segment 188 → 16 B, spill 46 → ~0, hot loop 244 → 242, MFMA and VALU pipe cycles unchanged, `vgpr_count` 168; d64 +0.40% (A 1.897 / 1.890 against B 1.885 / 1.883 ms) (gpt-oss D=64 attention forward, dualwave; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_082555/rounds/round-04/agent.txt:44)
- E12 [2026-07, gfx950, FlyDSL 0.2.2, N=1, ISA dump: highest VGPR index referenced per basic block] nine blocks at 142–149 (entry 142 … LBB0_12 149) against `.vgpr_count` 150; a K split (peak 142 → 132, allocation unchanged), a V split, and an `l_row` carry reduction that freed 3 carried registers and raised the allocation to 154 (Primus-Turbo hd64 flash-attention forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-13/agent.txt:25)
- E13 [2026-07, gfx950, FlyDSL 0.2.2, N=?, ISA dump and campaign score (conv TF/s)] QK MFMAs scheduled after `cast_p`, so the new score tile's 32 registers no longer live beside the carried P's 32 f32: `.vgpr_count` 166 → 162, spill 0, bit-identical output, wall 1116.1 against 1116.3; a hard fence on top: still 162, −0.90% (Primus-Turbo hd64 flash-attention forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-08/agent.txt:12)
- E14 [2026-08, gfx950, FlyDSL ?, N=2, campaign bench (`bench.sh`, ms), every arm ISA-gated first] `K_HEAD=0`: spill 0, 1.8590 / 1.8553 against 1.8362 ms, −1.1%; `V_HEAD=2`: spill 11; both (16 dwords freed, 8 needed): spill 12, −2.1%; two of the 8 row-sum MFMAs deferred past the QK run: spill 4 → 8, −0.3% (gpt-oss D=64 attention forward, B=4 Hq=64 Hkv=8 S=8192 causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:253)
- E15 [2026-08, gfx950, FlyDSL ?, N=1, ISA gate] row-sum MFMAs: 4 instead of 8 (one pack's row sum folded on the VALU) spill 93; removed with no replacement spill 118 (dot2 94, f32 VALU 198–217) (same kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:267)
- E16 [2026-07, gfx950, FlyDSL ?, N=?, campaign score (conv TF/s)] `.vgpr_spill_count` against score at vgpr 128 / 4 waves per SIMD: 0 → 1217.9; 5 → 1214.1 (−0.3%); 10 → 1215.3 (−0.2%); 35 → 981 (−19%) (Primus-Turbo hd64 flash-attention forward; KB note, the round's own record is empty; written in myskill commit 587b6182; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:872)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, campaign score (conv TF/s) and ISA] K/V ring four deep + 2× unroll with a tail loop: spill 2 → 35, 981 TF (−19%); 2-deep with the same unroll: spill 30; tail loop removed: spill 35 → 7 (`K_HEAD=2`), 10 → 0 (`K_HEAD=1`); remainder as a predicated second body: spill 0, vgpr 128 (same kernel; KB note, the round's own record is empty; myskill commit 587b6182; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:862)
- E18 [2026-07, gfx950, FlyDSL 0.2.2, N=1, ISA gate] K held fully resident plus a conditional `scf.if` barrier: spill 97 dwords (Primus-Turbo hd64 flash-attention forward, 8-wave merged path; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/memory.md:312)
- E19 [2026-08, gfx950, FlyDSL ?, N=1, ISA dump] three-way specialisation of the masked loop: nested ifs 219 AGPR, vgpr 476, above the 464 line that keeps the co-resident dQ reduce kernel resident; three sequential wave-uniform ifs with identity `else`: vgpr 460, spill 0, 468 fewer instructions in the masked loop (gpt-oss D=64 attention backward, B=4 Sq=Skv=8192 Hq=64 Hkv=8, one workgroup per CU; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/memory.md:235; AGPR and spill: local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:1088)
- E20 [2026-08, gfx950, FlyDSL ?, N=2, campaign bench (`bench.sh`, ms), two readings per arm] per-wave phase split: the vehicle with both branches issuing the same code 2.0361 / 2.0414 ms, −10.6% (phase inversion itself −1.5%); `scf.if` inside a cluster spill 4 → 103; only the main loop duplicated per branch: spill 63, 8 `scratch_load_dwordx4` in the hot loop, −27% (gpt-oss D=64 attention forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:289)
- E21 [2026-09, gfx950, FlyDSL ?, N=1, ISA gate] `amdgpu-agpr-alloc` forced to `32,32` / `64,64`: spill 31 → 52 / 77, the second with 26 more `v_accvgpr` instructions (VALU pipe 1400) (gpt-oss D=64 attention forward, dualwave; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:365)
- E22 [2026-09, gfx950, FlyDSL 0.2.4, N=?, campaign score with the guard used as zero control] the 32-dword V pack pinned with a tied asm: `agpr` 223 → 219, GEMM3's B operand pushed out of AGPRs (B=a MFMAs 112 → 256), shuttle up, −0.49% after the zero-point correction (gpt-oss D=64 attention backward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:164)
- E23 [2026-08, gfx950, FlyDSL ?, N=?, ?] tied `"=a,0"` pins: the two kv-block-invariant MFMA operands +0.79% and +0.53%; the two per-head-step ones (P's pack, dO's ring) −3.4% and −2.6%, each 12 dwords fatter (gpt-oss D=64 attention backward with fp8 operands in HBM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-16/agent.txt:17)
- E24 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] with k VALU after each MFMA, cycles = max(pipe, X + 8·k_exp + 4·k_mul), leaving an 8-cycle shadow per `v_mfma_f32_16x16x32_bf16` (one exp or two mul) and 24 per `v_mfma_f32_32x32x16_bf16`; a wave streaming exps on the SIMD of an MFMA wave lost ~7.5 cycles per MFMA (microbenchmark; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:82)
- E25 [2026-07, gfx950, FlyDSL ?, N=?, campaign score (conv TF/s), same session] 8-wave single workgroup (BLOCK_KV=256): 994.8 (MfmaUtil 53 → 40, −11%, reproducing an earlier round); with a phase stagger 1029.8, with NaN output (a WAR race on the shared Q/dO tile); 4-wave baseline 1116.7 (dK/dV backward kernel, hw exp; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260720_143502/memory.md:105)
- E26 [2026-07, gfx950, FlyDSL ?, N=?, ?] 2-producer/2-consumer role split: barrier-serial ~396 TF; software-pipelined at occupancy 2 spill 80, 254 TF; at occupancy 1 vgpr 291, spill 0, 291 TF, against a baseline of ~1100 (dK/dV backward kernel, hw exp; KB note, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dkdv_hwexp_overlap.md:209)
- E27 [2026-07, gfx950, FlyDSL ?, N=?, same-session A/B] cross-GQA-head LDS double buffer (N_BUF=2): dkdv 7.91 → 10.35 ms at 16384 (+31%), 1.96 → 2.78 ms at 8192 (+42%) (dK/dV backward kernel, hw exp; KB note, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dkdv_hwexp_overlap.md:123)
- E28 [2026-07, gfx950, FlyDSL ?, N=?, campaign score (conv TF/s)] pack/dS VALU forced between the MFMAs with `sched_barrier(0)`, interleave checked in the ISA (56 runs of exactly 4): wall 0.00% (Primus-Turbo hd64 flash-attention backward, dK/dV, Hq=64 Hkv=8 D=64 Sq=Skv=8192 B=2 causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:137)
- E29 [2026-07, gfx950, FlyDSL ?, N=?, campaign score (conv TF/s)] `sched_group_barrier` MFMA:TRANS 1:2 / 2:4 / 4:8, policy confirmed in the ISA (`s_nop` cycles 222 → 85): 816.6 / 814.8 / 814.9 against the 814.5 base, +0.26 / +0.04 / +0.05%, all neutral (the round's summary gives +0.26 / −0.04 / −0.07%) (same kernel; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/memory.md:147; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_123346/rounds/round-08/agent.txt:13)
- E30 [2026-07, gfx950, FlyDSL ?, N=?, same-process interleaved A/B (`_ab_fwd_cfg.py`)] `stagger=True` −4.8% at VGPR 128 (4 waves per SIMD, two 8-wave workgroups per CU) (gpt-oss SBHD attention forward, Hq=64 Hkv=8 D=64 B=2 causal, S=8192 and 16384; KB note, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_fwd16384_campaign.md:199)
- E31 [2026-08, gfx950, FlyDSL ?, N=2, campaign bench (`bench.sh`, ms), palindromic, one process per arm] `stagger=True` 1.9565 / 1.9657 ms, −6.3% (gpt-oss D=64 attention forward, vgpr 128, 8-wave CTA; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:265)
- E32 [2026-07, gfx950, FlyDSL 0.2.2, N=?, campaign score (conv TF/s)] exps moved into the P·V cluster (registers unchanged) 1187.4, −2.3%; the P·V cluster's exps moved to the QK cluster head 1211.5, −0.27%; the P·V cluster's 8 MFMAs already cover 16 `ds_read_b128` and 2 DMA issues; a variant that moved the whole softmax into that cluster wove perfectly in the ISA and read −0.33% (Primus-Turbo hd64 flash-attention forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-26/agent.txt:22; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-25/agent.txt:34)
- E33 [2026-08, gfx950, FlyDSL ?, N=?, rocprofv3 PMC and wall, MFMA count unchanged] GEMM3 split into one pass per q-half: aimed at the softmax window coexec 19.24% → 15.93%, wall −3.0%; aimed away 16.08%, −2.2% (gpt-oss D=64 attention backward with fp8 operands in HBM, dK/dV; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260828_090905/rounds/round-17/agent.txt:25)
- E34 [2026-09, gfx950, FlyDSL ?, N=?, campaign score] coverage distance 17.8 → 23.4 with VGPR and spill unchanged read d128 −0.65% (gpt-oss attention forward, dualwave; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/memory.md:175); on a D=64 attention backward, a schedule that lengthened MFMA runs 1.56 → 1.83 and cut `s_nop` 499 → 415 cycles per trip and 54 instructions read +0.80% slower (KB note; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/09-isa-profiling.md:33)
- E35 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per build, CUDA-graph probe on random inputs, ratio to the kept kernel, outputs bit-identical] six-wave kernel split into two phase-staggered groups of three waves, three LDS tiles, two barriers per step: five timed builds (two groupings, tile-offset arithmetic, read look-ahead 2 and 3) at 0.700–0.989× on batch 8 / 32 / 112, worst 0.700× at batch 112, best 0.961 / 0.989 / 0.968× (dense fp8 MLA decode, 96 heads, query length 1, page size 1, one workgroup per CU; local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace/forge_experiments/lessons/iter_008.md:7)
- E36 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR graph timer, paired timing the round was kept on (one-sided 95% bounds); the conversion move also by CUDA-graph probe] 8-wave role split (3 QK + softmax waves of two head tiles each sharing every K read, 5 P·V waves a step behind, 227 VGPRs, 0 spills) against the 6-wave kernel: +9.05% kept, +8.99% re-read (+8.96 … +9.03%), 11 of 11 workloads faster; fp8 conversion and gather moved from the QK waves to the P·V waves: probe 1.043 / 1.065 / 1.054× at batch 8 / 40 / 112, kept together with softmax VALU trims at +13.2% (same decode; local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace/forge_experiments/ablation.md:21; local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace/forge_experiments/lessons/iter_010.md:10; local:/shared_nfs/kyle/aklr_mla/runs/L3/workspace/forge_experiments/ablation.md:22)
- E37 [2026-10, gfx950, FlyDSL 0.3.2, N=? (CUDA-event, then CUDA-graph replay of 20 calls per round, median of interleaved per-round ratios; a self-against-self null read 1.000 / 0.994 / 1.003), outputs bitwise identical] two 6-wave halves of a 12-wave workgroup half a step apart on a double- or triple-buffered KV tile, against the single-phase kernel: five forms in one round 1.032–1.147 (b8 / b112: 1.073 / 1.114, 1.092 / 1.086, without `s_setprio` 1.086 / 1.084, halves interleaved per SIMD 1.147 / 1.130, triple buffer 1.032 / 1.051), four in a later round 1.025–1.070 (with and without `s_setprio`, with a carried gather); a spilling first form 3.2–3.7×; an `s_getreg HW_ID` probe placed the 12 waves 3 per SIMD (dense MLA decode, 96 heads, one query, fp8 KV, one workgroup per CU, 256 CTAs; MI355X; local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace/forge_experiments/lessons/iter_011.md:21-35; local:/shared_nfs/kyle/aklr_mla/runs/L4/workspace/forge_experiments/lessons/iter_013.md:15-17)
