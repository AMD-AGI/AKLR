---
title: "Register budget on gfx942: the combined VGPR pool and occupancy"
kind: reference
gens: [gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/prefetch-data-load/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/SKILL.md
---

# Register budget on gfx942

Copied from FlyDSL's own skills at v0.3.2; not re-measured.

## From FlyDSL's prefetch-data-load skill

### Register Budget

**Always check register headroom before adding prefetch buffers:**

On CDNA3 (gfx942 MI300X/MI308), VGPRs are tracked as two **physical** files that
share **one combined 512-entry occupancy budget** per SIMD:
- **arch_vgpr** (up to 256 per SIMD): used by VALU, VMEM loads, LDS ops, and prefetch buffers
- **accum_vgpr / AGPR** (up to 256 per SIMD): used by MFMA result writeback

Prefetch buffers physically live in **arch_vgpr** and MFMA accumulators in
**accum_vgpr**, but occupancy is governed by their **sum** (`arch_vgpr +
accum_vgpr`), so growing prefetch buffers *does* compete with MFMA accumulators
for the shared 512 budget and can cost occupancy.

```python
# Estimate arch_vgpr cost of prefetch buffers:
#   - Each buffer_load_dwordx4 = 4 arch_vgpr per load
#   - 8 K-cache loads = 8 x 4 = 32 arch_vgpr for one buffer set
#   - Double-buffering = 2 x 32 = 64 arch_vgpr (but one set is reused)
#   - Net additional arch_vgpr ~ 32 (the "next" buffer)
#
# On MI300X (gfx942): arch_vgpr + accum_vgpr share ONE combined 512 budget/SIMD
# Occupancy = 512 / (arch_vgpr_alloc + accum_vgpr_alloc) waves per SIMD
# (combined-pool model — NOT 256/max)
#
# Example: arch=148, accum=148 -> combined 296 -> 512//296 = 1 wave
# Adding 32 arch_vgpr -> combined 328 -> still 1 wave (safe)
# To reach 2 waves you need combined (arch+accum) <= 256
# arch+accum > 512 -> SPILL (exceeds the combined per-SIMD budget)
```

**Critical thresholds (gfx942, combined arch+accum budget):**
| Combined arch_vgpr + accum_vgpr | Max Waves/SIMD | Impact |
|--------------|---------------|--------|
| <= 128 | 4 | High occupancy |
| <= 168 | 3 | Good occupancy |
| <= 256 | 2 | Moderate occupancy |
| <= 512 | 1 | Minimum occupancy |
| > 512 | **SPILL** | Register overflow -> severe perf regression |

From its "Don't" list:
- **Don't assume occupancy can increase**: on MI308 with 512 max VGPRs, adding prefetch buffers that push total VGPRs above 256 will drop occupancy from 2 to 1 wave/SIMD. This may or may not be acceptable — profile both configurations.

## From FlyDSL's gemm-optimization skill (§10.2 Occupancy Impact)

On gfx942, `arch_vgpr` and `accum_vgpr` are two physical files that
share **one combined 512-entry occupancy budget per SIMD** — they *do* compete:

| arch_vgpr + accum_vgpr | Waves/SIMD | Assessment |
|------------------------|------------|------------|
| <= 64 | 8 | Maximum (the per-SIMD cap) |
| <= 72 | 7 | |
| <= 96 | 5 | |
| <= 128 | 4 | High occupancy |
| <= 168 | 3 | Good |
| <= 256 | 2 | Moderate |
| <= 512 | 1 | Minimum |

Occupancy is `min(8, 512 / alignTo(arch_vgpr + accum_vgpr, 8))`. Both constants
come from LLVM for GFX90AInsts targets: `getMaxWavesPerEU` returns 8 and
`getVGPRAllocGranule` returns 8 (`Utils/AMDGPUBaseInfo.cpp`). The granule matters
at the boundaries -- a combined 170 rounds up to 176 and yields **2** waves, not
the 3 that `512/170` alone suggests. MFMA accumulators live in `accum_vgpr` and prefetch
buffers / A / B tiles in `arch_vgpr`, but growing either one costs occupancy.

The separate 256-per-file model (`max(arch, accum)`) does not apply to
gfx942. In LLVM the count is `getTotalNumVGPRs(has90AInsts, NumAGPR, NumVGPR)`
(`llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp`), which returns
`alignTo(NumVGPR, 4) + NumAGPR` on gfx942, a GFX90AInsts target.

## From FlyDSL's kernel-trace-analysis skill (Register pressure check)

**VGPR is a combined 512-entry pool on gfx942.** The arch (256) and accum (256)
VGPR files share one 512 budget per SIMD. Occupancy from VGPR is
`512 / (arch + accum)` — NOT `256 / max(arch, accum)`.

| Property | CDNA3 (gfx942) |
|---|---|
| VGPR pool | 512 combined (256 arch + 256 accum, unified budget) |
| Occupancy formula (VGPR) | `512 / (arch_alloc + accum_alloc)` |
| Alloc granularity | 8 VGPRs |
| LDS size | 64 KB |
| LDS alloc block | 256 bytes |
| VMCNT width | 6 bits (max 63 in-flight) |
| LGKMCNT width | 4 bits (max 15 in-flight) |

Worked example (PA decode, gfx942): arch 144 + accum 136 = 280 combined → `512//280 = 1`
wave/SIMD, VGPR-bound (LDS allows 5, SGPR allows 7). Reaching 2 waves needs
combined ≤ 256, e.g. freeing ~24 VGPRs.
