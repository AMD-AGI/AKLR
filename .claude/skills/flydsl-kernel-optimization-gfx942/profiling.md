---
title: "Profiling on gfx942: trace-analysis examples, the PMC hang, MFMA latency rows"
kind: reference
gens: [gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py
  - ROCm/FlyDSL@dd8373349:.claude/skills/capture-kernel-trace/SKILL.md
---

# Profiling on gfx942

Copied from FlyDSL's own skills at v0.3.2; not re-measured.

## From FlyDSL's kernel-trace-analysis skill

### How `hotspot_analyzer.py` labels a gfx942 trace

**Auto-detection**: absence of `v_mfma_scale_f32_*`, `v_mfma_f32_16x16x128_*` and `v_mfma_f32_32x32x64_*`
instructions indicates CDNA3. The analyzer then reports the architecture as `gfx942 (CDNA3)` and its LDS
limiter uses `lds_total = 65536` (64KB CDNA3).

### L2 / HBM efficiency analysis (PMC, not ATT)

**Worked example (PA decode, gfx942, bs=16, ctx=131072, batch=256):**
L2 hit 1.7%, 32B 0%, over-fetch 1.04x, 2.85 TB/s = 54% peak. Conclusion: the
memory subsystem is clean; there is **no KV-load optimization left** — verified
by also testing block_size 16→64 (regressed +7.8%) and confirming dwordx8
doesn't exist on CDNA3 (dwordx4 / 16B is the max single vector load).

**Counter-capture caveat**: keep each PMC job to ≤ ~4 TCC counters (single
hardware pass). Multi-pass collection has triggered a GPU Hang on gfx942 — see
capture-kernel-trace.

### MFMA latency reference (cycles = pipeline depth)

| Instruction | Variant | Cycles | Notes |
|---|---|---|---|
| `v_mfma_f32_*_fp8_fp8` | 16x16x32 | 16 | CDNA3 |
| `v_mfma_f32_*_fp8_fp8` | 32x32x16 | 32 | CDNA3 |

## From FlyDSL's capture-kernel-trace skill (PMC Mode)

**CRITICAL — keep each job to a single hardware pass (≤ ~4 TCC counters).**
Packing many counters into one job forces multi-pass collection, which on
gfx942 has been observed to trigger a **GPU Hang (HW Exception)**. Split into
multiple single-pass jobs (as above) instead of one big counter list.
