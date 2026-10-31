---
title: "FlyDSL GEMM on gfx942: LDS limits, the sync-copy schedule, MFMA forms, peaks and DMA width"
kind: reference
gens: [gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/
---

# FlyDSL GEMM on gfx942

Copied from FlyDSL's own skills at v0.3.2, except the last section, which this repository wrote from FlyDSL
v0.3.2's source; not re-measured.

## From FlyDSL's gemm-optimization skill

The numbers below are that skill's section numbers.

### 1.4 Tile Size Constraints

- `tile_m * tile_k * elem_bytes` should fit comfortably in LDS (64KB on gfx942)

### 2.4 LDS Size Budget

Limits: 64 KB on gfx942.

### 4.1 A Matrix: Global → LDS

In this preshuffle GEMM, v0.3.2 raises `ValueError` for `use_async_copy=True` on gfx942
(`kernels/gemm/preshuffle_gemm.py:154-157`). The DMA instruction itself also lowers on gfx942, at 4 B per op.

### 5.3 Standard Schedule Pattern (gfx942, sync copy)

```python
def hot_loop_scheduler():
    mfma_group = num_acc_n
    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
    mfma_per_iter = 2 * mfma_group
    sche_iters = mfma_total // mfma_per_iter

    # Prologue: pre-load first 2 LDS packs, interleave with first few MFMAs
    rocdl.sched_dsrd(2)                    # 2 ds_read for a0_prefetch
    rocdl.sched_mfma(1)
    rocdl.sched_mfma(1)

    # Main schedule: each iteration = 1 VMEM + mfma_group MFMAs + 1 ds_read + mfma_group MFMAs
    dswr_tail = num_a_loads
    dswr_start = max(sche_iters - dswr_tail - 2, 0)
    for sche_i in range_constexpr(sche_iters):
        rocdl.sched_vmem(1)                # 1 global load (B tile or A tile)
        rocdl.sched_mfma(mfma_group)       # N MFMA instructions
        rocdl.sched_dsrd(1)                # 1 LDS read (A data)
        rocdl.sched_mfma(mfma_group)       # N more MFMAs
        if sche_i >= dswr_start - 1:
            rocdl.sched_dswr(1)            # LDS write (next A tile, tail end)

    rocdl.sched_barrier(0)                 # fence
```

### 5.4 Key Scheduling Insights

1. **MFMA instructions dominate**: they form the backbone of the schedule
2. **LDS reads (ds_read) interleave with MFMAs**: one ds_read per 2×mfma_group MFMAs
3. **Global loads (VMEM) interleave**: one buffer_load per scheduler iteration
4. **LDS writes (ds_write) go at the tail**: they overlap with the last MFMAs
   of the current tile, landing before the `gpu.barrier()` at iteration boundary
5. **dswr_start** ensures LDS writes are scheduled early enough to complete
   before the barrier, but late enough to not interfere with compute

### 6.2 MFMA Instruction Selection

For BF16/FP16, the K=16 forms (`mfma_f32_16x16x16bf16_1k`, `mfma_f32_16x16x16f16`) are what v0.3.2's
`kernels/gemm/preshuffle_gemm.py` uses on gfx942.

### 8.1 TFLOPS Calculation

```python
# Peak references (gfx942 MI300X, single GCD):
#   FP8:  ~653 TFLOPS peak (mfma_f32_16x16x32_fp8)
#   BF16: ~326 TFLOPS peak
#   INT8: ~653 TOPS peak
```

### 11.2 Implementation (Async Copy)

```python
# Direct global → LDS DMA
rocdl.raw_ptr_buffer_load_lds(
    a_rsrc, lds_ptr, size_i32, global_offset,
    soffset, offset_imm, aux,
)
# gfx942: 4 bytes per DMA op
```

### 11.3 Trade-offs

| Aspect | Sync Copy | Async Copy |
|--------|----------|------------|
| gfx942 granularity | 16B (dwordx4) | 4B (1 dword per DMA) |

### 14. Worked Example, Step 3: Estimate LDS usage

```
lds_tile = 64 × 128 = 8 KB
2-stage = 16 KB (well within 64 KB limit)
```

## From this repository's GEMM notes

Not from a FlyDSL skill: these three notes were written in this repository from FlyDSL v0.3.2's source
(`PS` is `kernels/gemm/preshuffle_gemm.py`, `A16` is `kernels/moe/moe_2stage_a16wmix/`); not re-measured.

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile. On
gfx942 it uses the sync-copy pattern of FlyDSL's guide: a prologue places the first LDS reads, then each step is
one VMEM load, a group of MFMAs, one LDS read and a second group of MFMAs, with the next tile's LDS writes at the
tail so they land before the barrier (5.3–5.4 above; PS:402-438).

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
