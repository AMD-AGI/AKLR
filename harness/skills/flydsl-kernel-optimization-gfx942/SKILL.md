---
name: flydsl-kernel-optimization-gfx942
description: >
  FlyDSL kernel optimization facts for gfx942 (MI300X/MI308): LDS banks and size, the
  register budget, the synchronous-copy GEMM schedule, MFMA and DMA forms, and
  trace-analysis examples. Copied from FlyDSL's own skills at v0.3.2, not re-measured
  here. Use when optimizing a FlyDSL kernel for gfx942.
gens: [gfx942]
arch-names: [gfx942, CDNA3, MI300, MI308, MI325]
allowed-tools: Read Edit Bash Grep Glob Agent
---

# FlyDSL kernel optimization on gfx942

This skill covers gfx942 (CDNA3: MI300X, MI308) only. Its text is copied from FlyDSL's own skills at v0.3.2
(`ROCm/FlyDSL@dd8373349:.claude/skills/`: lds-optimization, gemm-optimization, prefetch-data-load,
kernel-trace-analysis, capture-kernel-trace) and was not re-measured here; `gemm.md` also carries three notes
written in this repository from FlyDSL v0.3.2's source, marked as such. Each page names the files it came from.

| You need | Read |
|---|---|
| LDS size, banks and bank index, conflicting strides, the XOR swizzle mask, padding, LDS limits | `lds.md` |
| The combined VGPR budget, occupancy thresholds, a worked register example | `registers.md` |
| GEMM on gfx942: LDS limits, the synchronous-copy schedule, BF16/FP16 MFMA forms, peak figures, DMA width; the K=16 path of FlyDSL's bf16-activation MoE | `gemm.md` |
| Trace analysis: the PMC worked example, the multi-pass counter hang, MFMA latency rows, how the analyzer labels a gfx942 trace | `profiling.md` |
