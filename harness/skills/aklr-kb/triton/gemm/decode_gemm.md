# Decode GEMM in Triton (bf16, M ≤ 64, gfx950)

Measured with `python -m aklr validate` on MI355X (graph replay, 1 GiB flush per call, 8 rotated input
sets, lower bound of 32 rounds) for `gemm@1.0`, `trans_b`, no bias: n = 6144, k = 2048 (a 25 MB
weight) and n = 2624, k = 6144 (32 MB). Ratios are lower bounds against AITER's `gemm_a16w16`.
"Kept, not separated" marks a technique the winning kernels use whose own effect was not measured.

## 1. It is a weight stream: read B once

At M ≤ 64 the call is bound by reading the weight from HBM (a 25 MB weight took ~6 µs, about
4.2 TB/s). Shape the kernel so B is read exactly once:

- One M tile covers all of A: `BLOCK_M = 16` for M ≤ 16, then 32 or 64.
- Each workgroup owns one N tile across all rows of A, walks all of K, accumulates in fp32
  registers and stores once. No split-K, no atomics, so the result is bitwise reproducible.
- The shipped kernels, which also use §2 – §4: n = 6144 / k = 2048 at M = 2 – 64: 1.09 – 1.31;
  n = 2624 / k = 6144 at M = 2, 4, 8: 1.085 – 1.106.

## 2. Load B in the layout `tl.dot` wants

For a row-major (N, K) weight (`trans_b`), build the `[BLOCK_K, BLOCK_N]` tile directly:
`b_offs = offs_k[:, None] + offs_n[None, :] * K`, then `tl.dot(a, b)` with no `tl.trans`. Load B with
`cache_modifier=".cg"`, since every byte is read once (kept, not separated).

## 3. Order the tiles per XCD

Workgroup `i` runs on XCD `i % 8`. Remap the program id so each XCD streams a contiguous run of N
tiles. The remap must be a permutation of the grid for every grid size, including one not divisible
by 8 (164 tiles = 8 × 20 + 4):

```python
@triton.jit
def _xcd_remap(pid, GRID: tl.constexpr, NXCD: tl.constexpr):
    per: tl.constexpr = (GRID + NXCD - 1) // NXCD
    tall: tl.constexpr = GRID % NXCD if GRID % NXCD else NXCD
    xcd = pid % NXCD
    local = pid // NXCD
    if xcd < tall:
        return xcd * per + local
    return tall * per + (xcd - tall) * (per - 1) + local
```

A remap that is not a permutation skips some tiles and computes others twice; a timing-only
comparison can show it as a win. Check bijectivity for each grid size used (kept, not separated).

## 4. Start each workgroup's K walk at its own phase

Workgroups sharing an XCD read the same A lines in lockstep. Rotate the K walk per workgroup:
`kphase = ((pid_raw // NXCD) * KROT) % n_kb`, then `kb = ((i + kphase) % n_kb) * BLOCK_K`. The best
configurations used `KROT = 1` for k = 2048 at M = 32 and 64 and for k = 6144 at M ≤ 16, and `KROT = 0`
for k = 6144 at M = 64 (kept, not separated).

## 5. Long K, small M: fewer, longer workgroups

For k = 6144 at M ≤ 8, `BLOCK_N = 16`, `BLOCK_K = 512`, one K pass per workgroup (164 workgroups on
256 CUs) gave lower bounds of 1.085 – 1.106 at M = 2, 4, 8 (`BLOCK_K` and `KROT` were chosen
together). Fewer workgroups than CUs is fine when each streams a long K.

## 6. Split-K under a bitwise contract

`gemm@1.0` is `determinism: bitwise`: split-K must reduce in a fixed order. Write each split's fp32
partial tile to scratch (`torch.empty`, sized in the compile function) and sum the partials in a
second kernel in split order. The second kernel is one more graph node, so it pays only when the
split saves more than that node costs:

| k = 6144 | best deterministic choice | vs AITER |
|---|---|---|
| M = 64 | 64 × 64 tiles, split 6, fixed-order reduce | 1.236 |
| M = 32 | split, fixed-order reduce | 0.97 – 0.98 (sweep) |
| M = 16 | no split | 0.957 |

## 7. Leave M = 1 to a GEMV

A 16-row MFMA tile carrying one row lost to dedicated M = 1 paths: 0.964 – 0.984 for k = 2048 and
0.949 for k = 6144 against FlyDSL kernels that use a dot / vector path at M = 1. The same Triton tiles
won at M = 2 against a dot path (1.083).

## 8. Re-tune under AKLR's measurement

A configuration taken from elsewhere did not carry. The source served k = 2048 at M = 64 with AITER and
tuned its `BLOCK_M = 64` tier (`BLOCK_K = 128`, 8 warps, no K phase) for M = 33 – 48 with the weight hot
in the MALL (one flush, then many replays). Used at M = 64 under AKLR's measurement it gave 1.022;
re-tuned there, `BLOCK_K = 256`, 4 warps, `KROT = 1` gave 1.105 (the three changed together). Settle
choices worth 1 – 2 % with `python -m aklr validate`, not with a single-input sweep.

## Kept, not separated

`waves_per_eu = 2`, `matrix_instr_nonkdim = 16`, `num_stages = 3`; at M = 32 a 32-row tile with
`BLOCK_K = 64` unrolled 4× and `schedule_hint="memory-bound-attention"`.
