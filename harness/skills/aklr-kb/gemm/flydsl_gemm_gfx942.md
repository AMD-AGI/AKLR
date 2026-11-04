---
title: "GEMM on gfx942 (MI300X): where FlyDSL's own skills cover it, and notes from FlyDSL v0.3.2's source"
kind: reference
gens: [gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/prefetch-data-load/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/kernel-trace-analysis/SKILL.md
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/
---

# GEMM on gfx942

gfx942 (CDNA3: MI300X, MI308) only. FlyDSL's own skills at v0.3.2 cover it next to the newer generation; they are
linked here, not copied, and were not re-measured. The last section is this repository's, written from FlyDSL
v0.3.2's source.

## In FlyDSL's own skills

| You need | Read (FlyDSL v0.3.2) |
|---|---|
| Tile sizes and the LDS budget (64 KB per CU) | gemm-optimization [§1.4](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#14-tile-size-constraints), [§2.4](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#24-lds-size-budget), [§14](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#14-worked-example-optimizing-a-512051208320-fp8-gemm) step 3 |
| A through LDS with the synchronous-copy schedule | gemm-optimization [§4.1](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#41-a-matrix-global--lds), [§5.3](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#53-standard-schedule-pattern-gfx942-sync-copy), [§5.4](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#54-key-scheduling-insights) |
| BF16/FP16 MFMA forms and peak figures | gemm-optimization [§6.2](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#62-mfma-instruction-selection), [§8.1](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#81-tflops-calculation) |
| Async copy and its trade-offs | gemm-optimization [§11.2](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#112-implementation), [§11.3](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#113-trade-offs); on gfx942 see the notes below |
| LDS size, 32 banks and the bank index, conflicting strides, the XOR swizzle mask, padding, LDS limits | lds-optimization [LDS Architecture on CDNA3](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#lds-architecture-on-cdna3-gfx942), [Key Differences](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#key-differences-from-gfx942), [Swizzle Layout](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#optimization-method-1-swizzle-layout), [Choosing Swizzle Parameters](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#choosing-swizzle-parameters), [Padding](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#optimization-method-2-padding), [Verification Checklist](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#verification-checklist) |
| Bank conflicts in a trace | lds-optimization [Diagnosing LDS Bottlenecks from Trace](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/lds-optimization/SKILL.md#diagnosing-lds-bottlenecks-from-trace) |
| The combined VGPR budget (arch + accum share 512 per SIMD) and occupancy | prefetch-data-load [Register Budget](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/prefetch-data-load/SKILL.md#register-budget), gemm-optimization [§10.2](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/gemm-optimization/SKILL.md#102-occupancy-impact), kernel-trace-analysis [Register pressure check](https://github.com/ROCm/FlyDSL/blob/v0.3.2/.claude/skills/kernel-trace-analysis/SKILL.md#register-pressure-check-architecture-aware) |
| Trace analysis on gfx942 | `../profile.md` § gfx942 (MI300X) |

## This repository's notes

Written in this repository from FlyDSL v0.3.2's source (`PS` is `kernels/gemm/preshuffle_gemm.py`, `A16` is
`kernels/moe/moe_2stage_a16wmix/`); not re-measured.

**No async copy on gfx942 in this GEMM.** `preshuffle_gemm` raises `ValueError` for `use_async_copy=True` on
gfx942 (PS:154-157). The DMA instruction itself also lowers on gfx942, at 4 B per op.

**K=16 MFMA forms for BF16/FP16.** The K=16 forms (`mfma_f32_16x16x16bf16_1k`, `mfma_f32_16x16x16f16`) are what
`preshuffle_gemm` uses on gfx942.

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile. On
gfx942 it uses the sync-copy pattern of FlyDSL's guide: a prologue places the first LDS reads, then each step is
one VMEM load, a group of MFMAs, one LDS read and a second group of MFMAs, with the next tile's LDS writes at the
tail so they land before the barrier (gemm-optimization §5.3–5.4; PS:402-438).

Global → LDS DMA and transpose reads:

| | gfx942 |
|---|---|
| `buffer_load_dword … lds` | yes |
| `buffer_load_dwordx4 … lds`, `global_load_lds_dwordx4` | no |
| `ds_read_b64_tr_b16` and the other transpose reads | no |

**FlyDSL's bf16-activation MoE takes a K=16 path on gfx942.** `moe_2stage_a16wmix` picks it in
`a16wmix_use_k16` (A16 `utils.py:26-38`): gfx942 has no `mfma_f32_16x16x32_bf16` and no `v_cvt_pk_bf16_f32`, so
each K32 step is two 16x16x16 MFMAs into one accumulator (A16 `gemm1.py:507-509`) and the int4 dequant converts to
bf16 one value at a time (A16 `utils.py:132-138`). A reaches LDS through VGPRs, a 16 B `buffer_load` then a 16 B
`ds_write`, because the 16 B direct-to-LDS form in the table above does not exist here (A16 `gemm1.py:282-291`).
