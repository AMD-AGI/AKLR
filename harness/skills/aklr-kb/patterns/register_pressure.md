---
title: "Register pressure: VGPR / AGPR spills, or occupancy below the plan"
kind: pattern
signals: ["vgpr_spill_count > 0", "scratch_ loads and stores in the hot loop", "v_accvgpr_read / v_accvgpr_write in the hot loop",
  ".vgpr_count + .agpr_count above a reachable occupancy tier", ".private_segment_fixed_size > 0", "one wave per SIMD"]
ops: [gemm, grouped_gemm, fused_moe, attention]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# Register pressure: spills, or occupancy below the plan

## Symptom

- The dumped ISA shows `vgpr_spill_count` above 0, or `scratch_` and `v_accvgpr_read/write` in the hot loop (S1, S2).
  Read `.agpr_count`, `.vgpr_count`, `.amdhsa_next_free_vgpr` and the scratch size after every register change (S3).
- The total lands above the tier the plan needs. Arch VGPRs and AGPRs share one 512-entry file per SIMD, allocated
  in granules of 8; arch registers round up to 4 before the AGPRs are added (S4, S5).
- Hand-written partial `vmcnt` waits stop meaning what they count: scratch traffic shares `vmcnt` with the DMA (S6).

## Likely causes

1. **Accumulators past the AGPR file.** A wide tile can need more than 256 accumulator dwords per lane (S7).
2. **A fully unrolled short K loop** without VGPR headroom (S8; `gemm/mainloop.md` §7).
3. **A prefetch** that pushes the kernel over a register tier (S9).
4. **The scheduler lengthening live ranges.** A relaxed fence lets VALU intermediates hoist across it (S10); reading
   the next K step's fragments early spilled on tiles already at 256 VGPRs (S11).
5. **Work interleaved into MFMA shadows:** the register allocator is the constraint (S12).
6. **A dropped attribute.** A bare `"passthrough"` key is dropped silently, so an inline-asm AGPR accumulator gets the
   default allocation (S13, S30).

## Rule out first

- **The wrong reading.** rocprof's `VGPR_Count` reads half and `Accum_VGPR_Count` 0: read registers from the ISA
  (S14). Identical counts across variants that should differ usually mean one cached binary (S15).
- **An unreachable tier.** `; Occupancy:` ignores workgroup quantization (S16): an 8-wave workgroup reaches only
  2/4/6/8 waves per SIMD (S17). At unchanged occupancy, freed registers buy nothing on their own (S18).
  `analysis/budget.py --isa <NN_final_isa.s> --waves W` reads the registers and LDS from the dump and lists the ceiling
  of each reachable tier; `analysis/isa_diff.py` shows which spills and instructions two builds differ by.
- **Static code size** is nearly free; a spill slot is not, and removing code can create one (S19).

## Candidate techniques

- Place accumulators yourself, part AGPR and part VGPR — `gemm/schedule_and_locality.md` §7 — for cause
  1; check the `v_accvgpr_*` and `scratch_` counts in the hot loop (S1).
- AGPR or VGPR accumulators — `gemm/mainloop.md` §4 — judged by shuttle traffic and wall time, not
  register totals (S20).
- `amdgpu-agpr-alloc` under `"llvm.passthrough"`, `waves_per_eu` raced with AGPR budgets — `gemm/mainloop.md`
  §4, §7, `flydsl/gemm.md` § mainloop §4 — then read the counts back from the ISA (S21, S22, S3).
- Aim a register cut at a reachable tier — `gemm/grid_pricing.md` §4 — before chasing it (S23).
- Keep fences pinning the arithmetic (S10) and race LLVM's scheduling strategy per tile (S24) —
  `gemm/schedule_and_locality.md` §3 — gate any mask change on `vgpr_spill_count` before timing (S25).
- Move a `ds_write`, do not insert work in front of its wait — `gemm/schedule_and_locality.md` §5 — when
  VGPR headroom is small (S26).
- One base VGPR for a fragment's four transpose reads — `gemm/datapath.md` §9 — avoids an
  address-register spill (S27).
- Async copy for A — `../flydsl/guides.md` § gemm-optimization (FlyDSL's §11) — A goes global → LDS by DMA and saves arch VGPRs (S28).
- Not a register cap to make room — `../flydsl/language/limits.md` — in FlyDSL v0.3.2 `maxnreg` does not cap registers, and
  a cap through the `amdgpu-num-vgpr` attribute turns into spills (S29).
- Other kernel families: attention's occupancy steps, spill scale, code shapes that spill and AGPRs —
  `attention/registers_occupancy.md` §3, §5–7; grouped GEMM's register wall and waves per SIMD per layout —
  `moe/grouped_tile_pricing.md` §1–2.

## Evidence

- S1 gemm/schedule_and_locality.md:261 count `v_accvgpr_read/write` and `scratch_` in the hot loop
- S2 gemm/schedule_and_locality.md:157 the ISA's `vgpr_spill_count`
- S3 gemm/mainloop.md:348 check .agpr_count, .vgpr_count, .amdhsa_next_free_vgpr, scratch after changes
- S4 gemm/mainloop.md:310 one 512-entry file per SIMD shared by arch VGPRs and AGPRs, granule 8
- S5 gemm/mainloop.md:312 arch registers rounded up to 4 before the AGPRs are added
- S6 gemm/correctness.md:109 spill ≠ 0 voids every hand-written partial vmcnt; scratch shares vmcnt
- S7 gemm/schedule_and_locality.md:257 a wide tile can need more than 256 accumulator dwords per lane
- S8 gemm/mainloop.md:298 fully unrolling a short K loop pays only with VGPR headroom
- S9 gemm/datapath.md:214 a prefetch that pushes the kernel over a register tier
- S10 gemm/schedule_and_locality.md:142 the arithmetic must stay pinned; a relaxed fence hoisted VALU
- S11 gemm/schedule_and_locality.md:187 early fragment reads lost on tiles already at 256 VGPRs (spilled)
- S12 gemm/attribution.md:174 the register allocator is the constraint on interleaving
- S13 flydsl/gemm.md:155 a bare "passthrough" key is silently dropped, with no error
- S14 gemm/attribution.md:192 rocprof VGPR_Count half, Accum_VGPR_Count 0; read registers from the ISA
- S15 ../flydsl/debug-flydsl-kernel.md:83 identical VGPR_Count / Accum_VGPR_Count across variants: one cached binary
- S16 gemm/grid_pricing.md:255 the `; Occupancy:` comment does not apply workgroup quantization
- S17 gemm/grid_pricing.md:244 8-wave workgroup: only 2/4/6/8 waves per SIMD reachable
- S18 gemm/mainloop.md:240 at unchanged occupancy freed registers buy nothing on their own
- S19 gemm/attribution.md:205 static ISA size nearly free; a spill slot is not; removing code can spill
- S20 gemm/mainloop.md:233 judge AGPR vs VGPR accumulators by shuttle traffic and wall time
- S21 flydsl/gemm.md:155 put kernel attributes under "llvm.passthrough"
- S22 gemm/mainloop.md:290 8-wave candidates race waves_per_eu=2 with different AGPR budgets
- S23 gemm/grid_pricing.md:245 align a register-reduction target to a reachable tier
- S24 gemm/schedule_and_locality.md:160 race LLVM's scheduler per tile (amdgpu-sched-strategy)
- S25 gemm/schedule_and_locality.md:157 gate any mask change on vgpr_spill_count before benchmarking
- S26 gemm/schedule_and_locality.md:225 with small VGPR headroom, inserting work lengthens a live range
- S27 gemm/datapath.md:245 one address VGPR instead of four avoids an address-register spill
- S28 ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md:623 async copy saves arch_vgpr: A bypasses VGPRs
- S29 ../flydsl/language/limits.md:82 maxnreg leaves the binary unchanged; the amdgpu-num-vgpr attribute spills
- S30 gemm/mainloop.md:243 without the attribute an inline-asm AGPR accumulator gets the default allocation, and the result is wrong
