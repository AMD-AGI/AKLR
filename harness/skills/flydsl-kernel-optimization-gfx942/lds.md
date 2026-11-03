---
title: "LDS on gfx942: size, banks, conflict strides, swizzle mask, padding and limits"
kind: reference
gens: [gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md
---

# LDS on gfx942

Copied from FlyDSL's own skills at v0.3.2; not re-measured.

## From FlyDSL's lds-optimization skill

### LDS Architecture on CDNA3 (gfx942)

#### Hardware Facts

- LDS size: **64 KB per CU** (workgroup-shared)
- LDS is organized into **32 banks**, each **4 bytes wide**
- Bank index = `(byte_address / 4) % 32`
- **Bank conflict**: when 2+ threads in the same wavefront access **different addresses** in the **same bank** in the same cycle, accesses are serialized
- **Broadcast**: when 2+ threads access the **same address** in the same bank, hardware broadcasts (no conflict)
- LDS throughput: **128 bytes/cycle** (peak, no conflicts)
- LDS latency: **~20-40 cycles** (async, hidden if enough work between write and read)
- **VGPR context**: LDS ops use **arch_vgpr** (for addresses/data), not accum_vgpr. But occupancy on CDNA3 is governed by the **combined** 512-entry budget `arch_vgpr + accum_vgpr` (the two physical files share one occupancy budget — see `/kernel-trace-analysis`). So LDS-addressing logic that grows arch_vgpr *can* still cost occupancy even though it never touches the MFMA accumulators. Keep LDS-address VGPR pressure low when the kernel is near a 2-wave boundary.

### Summary (from "Key Differences")

| Aspect | gfx942 (CDNA3) |
|--------|----------------|
| LDS size per CU | 64 KB |
| Number of banks | 32 |
| Bank index formula | `(addr/4) % 32` |
| Peak throughput | 128 bytes/cycle |
| LDS allocation granularity | 256 bytes |
| Max LDS per workgroup | 64 KB |
| MFMA Transpose Load | Not available |

### Bank conflicts in a trace (from "Diagnosing LDS Bottlenecks from Trace", Type A)

- `ds_read2_b64` / `ds_write2_b32` with offsets that are multiples of the bank count:
  - **gfx942**: multiples of 32 (= same bank, 32-bank LDS)

### Strides that conflict (from "Optimization Method 1: Swizzle Layout")

- **gfx942 (32 banks)**: stride multiple of 128 bytes causes full conflict

```
# gfx942 (32 banks): stride=128 -> full conflict
Thread 0: addr = base + 0*128  -> bank (0*128/4)%32 = 0
Thread 1: addr = base + 1*128  -> bank (1*128/4)%32 = 0  <- CONFLICT
Thread 2: addr = base + 2*128  -> bank (2*128/4)%32 = 0  <- CONFLICT
```

### XOR mask (from "Choosing Swizzle Parameters")

For XOR mask:
- **gfx942 (32 banks)**: use `32 / (vec * element_size / 4) - 1` to ensure full bank coverage

### Padding (from "Optimization Method 2: Padding")

```python
# gfx942 (32 banks):
# Without padding: row stride = HEAD_SIZE (e.g., 128)
# Bank stride = 128 * 2 / 4 = 64 -> 64 % 32 = 0 -> ALL rows hit same bank column
# With padding: row stride = HEAD_SIZE + 2 (e.g., 130)
# Bank stride = 130 * 2 / 4 = 65 -> odd dword stride -> conflict-free
# (+1 element gives 64.5 dwords: lanes 2k and 2k+1 share a bank)
```

The cost of padding is extra LDS usage:
- each padding element per row: `KV_BLOCK_SIZE * element_size` extra bytes
- Must ensure total LDS usage stays within **64 KB** per CU (gfx942)

| Aspect | Swizzle | Padding |
|--------|---------|---------|
| LDS limit | N/A | gfx942: 64 KB |

### LDS usage (from "Verification Checklist")

```python
# Estimate: sum of all @fx.struct fx.Array field byte sizes
# gfx942: Must be <= 65536 bytes (64 KB) per workgroup
```
