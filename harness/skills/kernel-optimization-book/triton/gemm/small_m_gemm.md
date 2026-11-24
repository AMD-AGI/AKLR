---
title: "Small-M GEMM in plain Triton: bf16, M ≤ 64, the weight read once (gfx950)"
kind: topic
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-64", weight-stream, split-k]
gens: [gfx950]
triton: 3.8.0
---

# Small-M GEMM in Triton (bf16, M ≤ 64, gfx950)

How the two plain-Triton bf16 GEMM kernels of the library (154 Triton and Gluon kernels for gfx950: [../kernels/kernel_library.md](../kernels/kernel_library.md) §1) read their weight once per call, and what was measured to move them, on MI355X
with Triton 3.8.0. Both compute `out = a @ b.T` with `a` (m, k) and `b` (n, k) bf16, row-major and not preshuffled, no
bias, bf16 `out`: the N 6144 × K 2048 kernel (a 25 MB weight) and the N 2624 × K 6144 kernel (32 MB), always named by both
numbers. Times are the ruler's timer ([../tuning.md](../tuning.md) §1) unless a sentence names another timer; it read the
kernel durations from Kineto unless a sentence names rocprofiler-sdk. Each Evidence line names its timer.
The two kernels' speedups against aiter, the ruler's stored values, are in [../gemm.md](../gemm.md) §1 and §9.

The configurations were tuned in runs whose Triton version was not recorded; those runs, on an earlier form of the ruler, recorded lower
bounds against aiter of 1.09–1.31 (N 6144 × K 2048, M = 2–64), 1.085–1.106 (N 2624 × K 6144, M = 2, 4, 8) and 1.236 (N 2624 × K 6144, M = 64),
repeat count unknown *(field, E12)*.

## Route here when

- You write or tune a 16-bit GEMM with M ≤ 64 in plain `tl.*` (`tl.dot`): tile, weight view, tile
  order, K phase, split-K when every run must give the same bytes, the M = 1 case.
- A configuration tuned elsewhere loses under the timer that judges the kernel (§8).

Elsewhere: what Triton 3.8.0 compiles these two kernels to, and the fp8 Gluon GEMMs, [../gemm.md](../gemm.md) §9; the same
regime measured on FlyDSL kernels, [../../gemm/small_m.md](../../gemm/small_m.md); what one more launch costs under each
timer, [../authoring.md](../authoring.md) §7; how to A/B a change, [../tuning.md](../tuning.md) §2. "Kept, not
separated" marks a technique the two kernels use whose own effect was not measured.

## 1. It is a weight stream: read B once

At small M the call ran close to a plain read of its weight under the same timer: the N 6144 × K 2048 kernel (25.17 MB) took 5.94–6.38 µs
at M = 2–16 (3.95–4.23 TB/s) against 5.53–5.61 µs for the read, within 1.2×; the N 2624 × K 6144 kernel (32.24 MB) took 7.98–8.40 µs at
M = 2–8 against 6.90–7.02 µs, within about 1.22×, over 4 runs *(field, E1)*. The N 2624 × K 6144 kernel was not measured at M = 16 or 32.
Above that weight reading alone no longer sets time: the N 6144 × K 2048 kernel took 7.16–7.22 µs at M = 32; at M = 64 it took
8.53–8.56 µs and the N 2624 × K 6144 kernel 12.49–12.65 µs, 1.52–1.54× and 1.78–1.83× their same-run reads *(field, E1)*.
Reading B once is the source construction, not a measured HBM-traffic count:

- One M tile covers all of A: `BLOCK_M = 16` for M ≤ 16, then 32 or 64. On a synthetic plain-Triton bf16 GEMM
  (N 6144 × K 2048), two M tiles took 1.426–1.428× one tile at M = 32 (10.43–10.48 against 7.30–7.35 µs)
  and 1.358–1.379× at M = 64 (12.43–12.50 against 9.03–9.15 µs), 4 runs, 5 alternating rounds per arm with
  disjoint ranges in each run *(field, E2)*. This tests the combined tile-shape change (MFMA, VGPRs and program count
  change too), not isolated HBM traffic or the two kernels of this page; M ≤ 16 was not A/B tested.
- Each workgroup owns one N tile across all rows of A, walks all of K, accumulates in fp32 registers and stores once
  (the N 2624 × K 6144 kernel at M = 64 splits K instead, §6). Neither kernel contains an atomic, and each of the 9 sizes that
  run without a split (N 6144 × K 2048 at M = 2 – 64, N 2624 × K 6144 at M = 2, 4, 8) gave one output over 50 runs, inside the accuracy
  bound *(field, E8)*.

## 2. Load B in the layout `tl.dot` wants

For a row-major (N, K) weight (`out = a @ b.T`), build the `[BLOCK_K, BLOCK_N]` tile directly:
`b_offs = offs_k[:, None] + offs_n[None, :] * K`, then `tl.dot(a, b)` with no `tl.trans`:

```python
# the N 2624 × K 6144 kernel: one N tile (pid_n) per workgroup over all rows of A; pid_k is the K split (§6), kphase §4
offs_m = tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
row_ok = offs_m[:, None] < M
a_offs = offs_m[:, None] * K + offs_k[None, :] + pid_k * k_chunk
b_offs = offs_k[:, None] + offs_n[None, :] * K + pid_k * k_chunk  # [BLOCK_K, BLOCK_N], as tl.dot takes it
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for i in tl.range(0, n_kb):
    kb = ((i + kphase) % n_kb) * BLOCK_K
    if EVEN_M:
        a = tl.load((A + kb) + a_offs)
    else:
        a = tl.load((A + kb) + a_offs, mask=row_ok, other=0.0)
    b = tl.load((B + kb) + b_offs, cache_modifier=".cg")
    acc = tl.dot(a, b, acc=acc)
```

Not this: loading `[BLOCK_N, BLOCK_K]` and calling `tl.trans`. In the N 2624 × K 6144 kernel that added `ds_bpermute_b32`
shuffles (12 and 16 against 0 in the
M = 2, 4, 8 and M = 64 tiles), kept at M = 2, 4, 8 the K loop the direct tile fully unrolls (48 against 192 MFMAs)
*(field, E16)*, and took 1.09 – 1.17× the time at M = 2, 4, 8 and 1.01× at M = 64 in two runs, one on the per-kernel
timer with rocprofiler-sdk *(field, E3, E16)*.
Both kernels load B with `cache_modifier=".cg"`. Without it both took 1.1 – 3.4 % more time with the caches flushed
before each call (no stable difference at N 6144 × K 2048, M = 32) but 0.58 – 0.95× the time when the next calls re-read the
same weight, 1 run *(field, E17)*; what the hint compiles to and bought elsewhere: [../gluon/memory.md](../gluon/memory.md) §6.

## 3. Order the tiles per XCD

On the tested grids, launched on the default stream, workgroup `i` ran on XCD `i % 8` *(field, E4)*; on a new stream
the absolute XCD index rotates by a constant and workgroups congruent mod 8 still share an XCD
([../gluon/memory.md](../gluon/memory.md) §14). The placements below assume that grouping. Both kernels remap the
program id so each XCD gets N tiles in contiguous runs: one run per XCD in the N 2624 × K 6144 kernel at M ≤ 16 (at M = 64 a run of
split-major tile numbers, which crosses a K split on five XCDs), runs of 8 or 24 tiles in the N 6144 × K 2048 kernel.
In program order (`pid = raw`) they took up to 1.8 % more time with the caches flushed before each call, but
5 – 16 % less at N 6144 × K 2048, M ≤ 16 when the next calls re-read the same weight (warm cache or back-to-back graph
calls), 1 run *(field, E18)*; what XCD order measured on other kernels:
[../tuning.md](../tuning.md) §12. A remap must be a permutation of the grid for every grid
size launched, including one not divisible by 8 (164 tiles = 8 × 20 + 4). The N 2624 × K 6144 kernel's gives `per` tiles to the first
`tall` XCDs and `per - 1` to the rest, one contiguous run per XCD, and is a permutation for every grid of 1 – 4096
tiles, enumerated from the formula:

```python
# pid: the launch's program id; the result is the N tile (times the K split at M = 64) this workgroup computes
@triton.jit
def _xcd_remap(pid, GRID: tl.constexpr, NXCD: tl.constexpr):
    per: tl.constexpr = (GRID + NXCD - 1) // NXCD
    tall: tl.constexpr = GRID % NXCD if GRID % NXCD else NXCD
    xcd = pid % NXCD
    local = pid // NXCD
    if xcd < tall:
        tile = xcd * per + local
    else:
        tile = tall * per + (xcd - tall) * (per - 1) + local
    return tile
```

The N 6144 × K 2048 kernel deals runs of `CHUNK = 8` tiles per XCD at M = 32 and 64 (3 runs per XCD over its 192 tiles) and one run of 24
per XCD at M ≤ 16:

```python
# the kernel function launched at M <= 16: one run of N_TILES // NXCD tiles per XCD
pid = tl.program_id(0)
pid = (pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)
```

```python
# the kernel function launched at M = 32 and 64: runs of CHUNK tiles per XCD
pid = tl.program_id(0)
# ... (the K phase of §4, taken from the unmapped pid)
blk: tl.constexpr = NXCD * CHUNK
local = pid // NXCD
pid = (local // CHUNK) * blk + (pid % NXCD) * CHUNK + (local % CHUNK)
```

Neither of these two remaps is a permutation at 164 tiles (enumerated from the formulas). A remap that is not a permutation
leaves tiles unwritten, and timing does not flag it: the N 2624 × K 6144 kernel at M = 8 with `xcd * (GRID // 8) + local` on its 164 tiles
skipped tiles 160 – 163 (columns 2560 – 2623 stayed NaN) and computed tiles 20, 40, 60, 80 twice, yet timed 0.999× and
0.993× the kernel with its own remap in two runs (back-to-back calls, no flush), within the run spread *(field, E5)*. The
`CHUNK` remap on 164 tiles instead misses 12 tiles and sends 12 programs past the grid. Check bijectivity for each
grid size used.

## 4. Start each workgroup's K walk at its own phase

Without a K phase every workgroup walks K in the same order. The kernels rotate each workgroup's K walk:
`kphase = ((pid_raw // NXCD) * KROT) % n_kb`, then `kb = ((i + kphase) % n_kb) * BLOCK_K` (the N 6144 × K 2048 kernel writes
`& (n_kb - 1)`, the same for its power-of-two `n_kb`). At the N 6144 × K 2048 kernel's M = 64 launch (n = 6144, k = 2048; 64 × 32
tiles, `BLOCK_K = 256`, 192 workgroups, 4 warps), `KROT = 0` took 9.14 – 9.18 µs against 8.56 – 8.61 µs with
`KROT = 1`, 6.5 – 7.2 % longer in each of 4 runs *(field, E6)*. A K-phase timing benefit was also observed
in a separate private-A construction (no two workgroups read the same A line): `KROT = 0`
cost 11.6 – 11.8 % (15.05 – 15.08 against 13.47 – 13.50 µs, 4 runs) *(field, E6)*. The two kernels use
`KROT = 1` for N 6144 × K 2048 at M = 32 and 64 and for N 2624 × K 6144 at M ≤ 16, and `KROT = 0` for N 2624 × K 6144 at M = 64.
Flipped one at a time with the output checked, on the ruler's timer (rocprofiler-sdk), `KROT = 0` took 11.6 % more time at
N 6144 × K 2048, M = 64, 1.1 % more at M = 32 and 1.6 – 2.4 % more at N 2624 × K 6144, M = 2, 4, 8, while `KROT = 1` at N 2624 × K 6144,
M = 64 took 0.4 % more, inside that size's A/A threshold, 1 run *(field, E19)*; M = 1 and 16 were not timed.

## 5. Long K, small M: fewer, longer workgroups

At M ≤ 16, the N 2624 × K 6144 kernel uses 16 × 16 tiles with `BLOCK_K = 512` and no K split, so each workgroup walks all of
K: 164 workgroups, fewer than the 256 CUs *(field, E4)*. At M = 2, 4, 8 a call took 8.01 – 8.03, 8.09 – 8.12 and
8.35 – 8.39 µs over 2 runs, 1.14 – 1.20× a plain read of its 32 MB weight *(field, E1)*. This is this kernel's
configuration, not a rule that fewer workgroups than CUs suffice for long K (§6).

## 6. Split-K under a bitwise contract

Fewer workgroups than CUs is not a general rule for long K. At (M, N, K) = (64, 2624, 12288) an 82-workgroup kernel
with no split (`BLOCK_N` 32) took 48.27 µs against 22.70 µs for split 6 with `BLOCK_N` 64, split / no split 0.470
(24 balanced pairs, one run) *(field, E13)*. At M = 16, the N 2624 × K 6144 kernel's M ≤ 16 launch (`BLOCK_K` 512,
no split) beat split 6: split / no split 1.240 and 1.244 for two input seeds, 16 balanced pairs each, under the
ruler's timer (Kineto) *(field, E13)*. That holds under the ruler's timer only: under the device-graph event timer
(48 balanced pairs per seed, two seeds) split / no split at M = 16 was 0.983 and 0.980 with intervals spanning 1, and at
M = 2, 4, 8 no split was faster (split / no split 1.037–1.104, two of the six intervals spanning 1) *(field, E13)*.

Both kernels must be bitwise reproducible: every run's output byte-identical to the first's. So split-K must
reduce in a fixed order: write each split's fp32 partial tile to scratch (`torch.empty`, sized by the cached host
helper and allocated in the call) and sum the partials in a second kernel in split order:

```python
# the end of the GEMM kernel. One split (M ≤ 16): round and store to the output, which the host passes as W.
# SPLIT_K > 1 (M = 64): split pid_k stores its fp32 partial tile into plane pid_k of the scratch W
if SPLIT_K == 1:
    c_ptrs = W + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(W.dtype.element_ty), mask=row_ok)
else:
    w_ptrs = W + pid_k * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_ptrs, acc, mask=row_ok)
```

```python
# second kernel: adds the planes in split order 0, 1, ..., SPLIT_K - 1, then rounds once
@triton.jit
def _reduce_kernel(W, C, TOTAL, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < TOTAL
    acc = tl.load(W + offs, mask=ok, other=0.0, cache_modifier=".cg")
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(W + s * TOTAL + offs, mask=ok, other=0.0, cache_modifier=".cg")
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=ok)
```

```python
# host, the per-shape helper's M = 64 arm (cached): split_k = 6, total = m * n; it returns the scratch size with the launch
def launch(a, b, out, ws):
    _gemm_kernel[(n_tiles * split_k,)](
        a,
        b,
        ws,
        # ... (m, the constexprs and the launch options)
    )
    _reduce_kernel[(triton.cdiv(total, 1024),)](
        ws, out, total, SPLIT_K=split_k, BLOCK=1024, num_warps=4
    )

return launch, split_k * total
```

```python
# host entry, per call: ws_elems is the helper's scratch size, 0 (no scratch) where K is not split
ws = (
    torch.empty(ws_elems, dtype=torch.float32, device=a.device)
    if ws_elems
    else None
)
launch(a, b, out, ws)
```

Not this: each split adding its partial into `out` with a float atomic. On a test GEMM with the N 2624 × K 6144 kernel's
M = 64 shapes (6 fp32 partial planes), the fixed-order reduce gave one output in 100 runs, bit-equal to
a sequential sum in split order (the reverse order differed in 90,596 values), while a float-atomic K split gave 100
different outputs in 100 runs; the N 2624 × K 6144 kernel itself at M = 64 gave one output in 50 runs, and it ran and was timed
in a captured CUDA graph with its per-call scratch *(field, E8)*. The second kernel is one more graph node (what a node costs under the ruler's timer and in a serving graph:
[../authoring.md](../authoring.md) §7):

| N 2624 × K 6144 | launch | time per call |
|---|---|---|
| M = 64 | 64 × 64 tiles, split 6 (246 workgroups), fixed-order reduce | 12.49 – 12.62 µs, 2 runs *(field, E1)*; 4.2 µs more than its GEMM kernel alone under the ruler's timer (Kineto), the timer's cost per kernel rather than the reduce kernel's own time *(field, E9)*; 1.87 µs more under the device-graph timer below *(field, E11)* |
| M = 32 | none: the kernel was not tuned or measured at M = 32 | — |
| M ≤ 16 | 16 × 16 tiles, no split (164 workgroups) | 8.01 – 8.39 µs at M = 2 – 8 *(field, E1)* |

The tuning runs (Triton version not recorded) recorded 1.236 against aiter at M = 64
*(field, E12)*; the ruler's stored value is 1.174 ([../gemm.md](../gemm.md) §1).

**Split count at M = 64.** One run at (M, N, K) = (64, 2624, 6144) *(field, E11)*, with `BLOCK_M=64`,
`BLOCK_K=128`, `NXCD=8`, `KROT=0`, 8 warps, 3 stages, `waves_per_eu=2` and `matrix_instr_nonkdim=16`, swept split ∈
{1, 2, 3, 4, 6, 8, 12} × `BLOCK_N` ∈ {16, 32, 64}. All 21 configurations met the accuracy bound against CPU fp64 /
fp32 references on 3 input seeds and gave byte-identical output over 100 repeats per configuration and seed.

A 16-pair pilot picked one `BLOCK_N` per split (32 for splits 1, 2, 3; 64 for 4, 6, 8, 12); each comparison then
took 32 balanced, randomized AB/BA pairs with A/A pairs interleaved, on one input seed. Split 6 / `BLOCK_N` 64 beat
splits 1 / 2 / 3 / 8 / 12 (their time over split 6's: 1.744 / 1.242 / 1.085 / 1.215 / 1.178). Against split 4 /
`BLOCK_N` 64 the ratio was 1.095 (family-corrected 99 % interval 1.068–1.123), which did not clear that run's A/A
noise floor plus the AB/BA order gap *(field, E11)*. Re-measured on two other input seeds, split 4 took 1.11× split 6's
time on a device-graph event timer and 1.15× on the ruler's timer (rocprofiler-sdk), slower in every round *(field, E21)*,
so split 6 / `BLOCK_N` 64 was the fastest of the seven splits in this sweep. Other N, K and tile choices were not searched.

The timer measured one preallocated device-graph call, with a 1 GiB flush outside each event window: whole = GEMM +
fixed-order reduce, partial = GEMM only. Split 6's whole / partial means were 16.140 / 14.273 µs; the paired
difference, 1.868 µs (interval 1.466–2.269), estimates the extra device-graph cost, not the reduce kernel's own time
*(field, E11)*. Allocation, Python dispatch, JIT, capture and ramp-up were excluded. Do not compare these absolute
times with those of the ruler's timer above, which is another timer.

## 7. M = 1: a 16-row MFMA tile carrying one row

At M = 1 both source paths select a 16-row tile and compile to `v_mfma_f32_16x16x32_bf16`, no `v_dot2`. Called at M = 1,
Triton 3.8.0 turns M into the constant 1 (S15): the N 6144 × K 2048 kernel has 64 MFMAs and the N 2624 × K 6144 kernel 192, its K loop fully unrolled
*(field, E20)*; with M kept a runtime i32 the N 2624 × K 6144 kernel has 48 in a loop *(field, E15)* and took 1.08× the time on the
ruler's timer (rocprofiler-sdk) *(field, E20)*. FlyDSL kernels for
these two shapes take a dot / vector path at M = 1, with `fdot2` and no MFMA. Neither Triton kernel was tuned for
M = 1. In one A/B on the ruler's timer (4 alternated rounds) of the Triton kernels run at M = 1 against those
FlyDSL kernels, FlyDSL took 0.975× the Triton time at the N 6144 × K 2048 shape (5.86 against 6.01 µs) and 0.933× at the N 2624 × K 6144 shape
(7.27 against 7.79 µs), lower in every round; at M = 2 on the N 6144 × K 2048 shape, where FlyDSL also takes its dot path, the
Triton tile took 0.917× FlyDSL's time, and 0.40× at the N 2624 × K 6144 shape (8.0 against 20.0 µs) *(field, E14)*. A plain-Triton
GEMV (`tl.sum` over K, no `tl.dot`, six tile sizes) beat the 16-row tile on 1 of 5 shapes (N = 6144, K = 2048: 0.972×)
*(field, E14)*. Gluon kernels fill the spare rows instead ([../gluon/mfma.md](../gluon/mfma.md) §8).

## 8. Re-tune under the timer that judges the kernel

A configuration taken from elsewhere did not carry. At N 6144 × K 2048, M = 64 the `BLOCK_M = 64` tier taken from the source
kernel (`BLOCK_K = 128`, 8 warps, no K phase) took 1.05 – 1.08× the time of the re-tuned one (`BLOCK_K = 256`,
4 warps, `KROT = 1`; the three changed together): about 9.0 – 9.3 µs against 8.53 – 8.78 µs under the ruler's
timer (graph replays, flushed before each call), 4 runs *(field, E7)*. A hot cache does not explain the gap:
timed with the weight hot (one flush, then replays of a 50-call graph, where the re-tuned kernel took 6.99 – 7.05 µs
against 8.53 – 8.78 µs flushed), the source tier took 1.08 – 1.13× the re-tuned one's time, against 1.05 – 1.08×
flushed; the hot ratio was 0.02 – 0.05 higher in each of 4 runs *(field, E7)*. Only M = 64 was timed. How to A/B such
a choice: [../tuning.md](../tuning.md) §1–§2.

## 9. Kept, not separated

`waves_per_eu = 2` and `num_stages = 3` on every GEMM launch; `matrix_instr_nonkdim = 16` on all but the N 6144 × K 2048 kernel's
M ≤ 16 launch, which passes none; and on the N 6144 × K 2048 kernel's M = 32 launch
`schedule_hint="memory-bound-attention"`:

```python
# the N 6144 × K 2048 kernel's launch for 16 < m <= 32; grid = (n // 32,)
_gemm_xcd_kernel[grid](
    a,
    b,
    out,
    m,
    N=n,
    K=k,
    BLOCK_M=32,
    BLOCK_N=32,
    BLOCK_K=64,
    NXCD=8,
    CHUNK=8,
    EVEN_M=even_m,
    UNROLL=4,
    KROT=1,
    num_warps=4,
    num_stages=3,
    waves_per_eu=2,
    matrix_instr_nonkdim=16,
    schedule_hint="memory-bound-attention",
)
```

Triton 3.8.0 reads the hint nowhere: its only mention is the
`HIPOptions` field ("Experimental; right now no effect", S5), and the N 6144 × K 2048 kernel's M = 32 kernel compiled to byte-identical
AMDGCN with and without it *(field, E15)*. Triton 3.7.0 turns the hint into the `iterative-ilp` scheduler (S6):
that kernel's M = 32 LLIR then differs from a build without it only in `"amdgpu-sched-strategy"="iterative-ilp"`, the
TTGIR is identical, and the AMDGCN changes *(field, E15)*. To keep that scheduler on Triton 3.8.0, pass
`llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` (S14), which gave the same attribute on 3.8.0 and was
dropped by 3.7.0 *(field, E15)*, and have the host refuse to build the kernel on a Triton older than 3.8
([../tuning.md](../tuning.md) §9).

At M = 32 the N 6144 × K 2048 kernel uses a 32-row tile with `BLOCK_K = 64` and `UNROLL = 4`, the `loop_unroll_factor` of its
K loop's `tl.range`. Compiled at k = 2048
(Triton 3.8.0), the K loop is fully unrolled (64 MFMAs, no loop) with `UNROLL = 4` or 1 alike, so the 4× request
changes neither the MFMA count nor the absence of a K loop in this build *(field, E10)*.

## Evidence

Measured on one MI355X with Triton 3.8.0 in the `lmsysorg/sglang:v0.5.22-rocm10-mi35x` image (2026-11); E12 is an
earlier record whose Triton version was not recorded.
E16–E21 name their own torch build in their lines. "The ruler" and "the ruler's timer" are [../tuning.md](../tuning.md) §1's.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the ruler's timer (Kineto), in runs of the ruler for the GEMMs; a Gluon read-and-sum of the same bytes under the same timer for the plain read] over 4 runs: N 6144 × K 2048 (25.17 MB weight) at m = 2–16 5.94–6.38 µs (3.95–4.23 TB/s), m = 32 7.16–7.22, m = 64 8.53–8.56 µs, plain read 5.53–5.61 µs (4.48–4.55 TB/s); N 2624 × K 6144 (32.24 MB) at m = 2–8 7.98–8.40 µs, m = 64 12.49–12.65 µs, plain read 6.90–7.02 µs; m = 64 over the same run's read 1.52–1.54 (N 6144 × K 2048) and 1.78–1.83 (N 2624 × K 6144). Two of the runs per m: N 6144 × K 2048 m = 2, 4, 8, 16, 32, 64 5.95 / 5.98, 5.98 / 5.98, 6.05 / 6.07, 6.36 / 6.34, 7.21 / 7.19, 8.56 / 8.54 µs, read 5.61 / 5.59; N 2624 × K 6144 m = 2, 4, 8, 64 8.03 / 8.01, 8.12 / 8.09, 8.39 / 8.35, 12.49 / 12.62 µs, read 7.02 / 6.97
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the ruler's timer (Kineto), median of 5 alternated rounds] a plain-Triton bf16 test GEMM, N 6144 × K 2048, `BLOCK_N` 32, `BLOCK_K` 256, 4 warps: one M tile against two that each re-read B, m = 32: 7.30–7.35 against 10.43–10.48 µs (1.426–1.428×), m = 64: 9.03–9.15 against 12.43–12.50 µs (1.358–1.379×), round ranges disjoint in each run, max relative error 0.25–0.26 % (test kernel)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, A/B on the ruler's timer (Kineto), 4 alternated rounds; ISA of a stand-alone `tl.dot` test kernel with the two tiers' tile sizes, not of the N 2624 × K 6144 kernel] the N 2624 × K 6144 kernel with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against its direct tile: time ratio 1.172, 1.163, 1.121, 1.009 at m = 2, 4, 8, 64; the test kernel with B loaded the same two ways, 16 × 16 × 512 tile, 2 warps: 509 against 350 instructions (16 against 0 `ds_bpermute_b32`); 64 × 64 × 128 tile, 8 warps: 331 against 261 (6 against 0)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, XCC and CU ids read from the hardware in each workgroup, launched on the default stream] XCD = workgroup id % 8 with no mismatch on 1-D grids of 8, 100, 2048 and 4096 workgroups and on 6 multi-axis grids (3 launches each), and with offset 0 in 5 further launches of 4096, in each of 3 runs; 256 CUs (`multi_processor_count`, and 32 distinct CU ids on each of 8 XCCs) in each run; the N 2624 × K 6144 kernel's m ≤ 16 grid read from its source: 164
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, CUDA events over 20 back-to-back calls (no flush), median of 9, 5 alternated rounds] the N 2624 × K 6144 kernel at m = 8 with the remap `xcd * (GRID // 8) + local` on 164 tiles: columns 2560 – 2623 NaN in all 8 rows, bitwise equal elsewhere; broken remap / the kernel's own remap 11.93 / 11.94 µs (0.999×; round ranges 11.90 – 12.05 and 11.88 – 12.53), the other run 12.59 / 12.69 µs (0.993×; 12.53 – 12.73 and 12.55 – 12.72)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the ruler's timer (Kineto), median of 5 alternated rounds] the N 6144 × K 2048 kernel's m = 64 kernel launched directly with its own options (64 × 32 tiles, `BLOCK_K` 256, 192 workgroups, 4 warps), `KROT` 1 against 0: A shared by all workgroups 8.562 – 8.609 against 9.136 – 9.181 µs, `KROT` 0 / 1 = 1.065 – 1.072 (per run 8.581 / 9.136, 8.609 / 9.166, 8.562 / 9.181, 8.562 / 9.155 µs); A private to each workgroup (A of 192 × 64 rows, each workgroup offset to its own 64 rows by its remapped tile index) 13.47 – 13.50 against 15.05 – 15.08 µs, 1.116 – 1.118 (per run 13.47 / 15.07, 13.50 / 15.07, 13.48 / 15.05, 13.49 / 15.08 µs); output not checked
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, flushed: the ruler's timer (Kineto); hot: one flush and one replay of a 50-call graph, then the next replay timed, median of 15; median of 3 rounds each] the N 6144 × K 2048 kernel at m = 64, the source's tier (`BLOCK_K` 128, 8 warps, `KROT` 0) against the re-tuned launch, one input: flushed tier / re-tuned 1.054 – 1.082 (re-tuned 8.527 – 8.78 µs, tier ≈ 9.04 – 9.27 µs), hot 1.080 – 1.128 (re-tuned 6.989 – 7.046 µs), hot ratio minus flushed ratio 0.021 – 0.052; per run flushed 1.054× (8.78 µs) and hot 1.107× (7.031 µs), 1.060× (8.527 µs) and 1.080× (7.046 µs), 1.082× (8.563 µs) and 1.128× (6.99 µs), 1.076× (8.594 µs) and 1.119× (6.989 µs)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, repeated calls compared bitwise] the N 6144 × K 2048 kernel at m = 2 – 64 and the N 2624 × K 6144 kernel at m = 2, 4, 8: one distinct output in 50 runs at each of the 9 sizes, inside the accuracy bound, 0 atomics in the sources; a test GEMM with 6 fp32 partial planes [6, 64, 2624] summed in split order by a second kernel: one output in 100 runs, bit-equal to the sequential sum (reverse order differs in 90,596 values), a float-atomic K split 100 distinct outputs in 100 runs; the N 2624 × K 6144 kernel itself at m = 64: one output in 50 runs; captured into a CUDA graph and timed there at m = 64 with its per-call scratch, no finding other than speed
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler's timer (Kineto), median of 3; Kineto trace of a graph replay for the kernel list] the N 2624 × K 6144 kernel at m = 64: whole call 12.57 µs, its GEMM kernel alone 8.374 µs, difference 4.2 µs; the replay runs the GEMM kernel and the reduce kernel at m = 64, the GEMM kernel only at m = 2
- E10 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, compiled-kernel census of an offline build] the N 6144 × K 2048 kernel's m = 32 kernel (`BLOCK_M` 32, `BLOCK_N` 32, `BLOCK_K` 64, k 2048, 4 warps) built with `UNROLL` 4 and 1: 64 `v_mfma` in each, no backward branch in either
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one GPU run, one size, bf16, preallocated device graphs timed by events, 1 GiB flush outside each event window; confirmation by 99 % family-corrected paired-log Student-t intervals against the larger of the local and global A/A floors plus the AB/BA order gap] (M, N, K) = (64, 2624, 6144), `BLOCK_M` 64, `BLOCK_K` 128, `KROT` 0, 8 warps, 3 stages, `waves_per_eu` 2: all 21 split × `BLOCK_N` configurations accurate on 3 seeds and byte-identical over 100 repeats; competitor / split 6 (`BLOCK_N` 64) time 1.744 / 1.242 / 1.085 / 1.215 / 1.178 for splits 1 / 2 / 3 / 8 / 12, 1.095 (1.068–1.123) for split 4, unresolved; split 6 whole / partial 16.140 / 14.273 µs, difference 1.868 µs (1.466–2.269); clocks and package versions not recorded (plain-Triton copies of the N 2624 × K 6144 kernel)
- E12 [2026-10, gfx950 (MI355X), Triton version not recorded, N=?, an earlier form of the ruler: lower bound of the aiter / kernel time ratio over 32 rounds, 8 rotated input sets, graph replay, 1 GiB flush per call] lower bounds against aiter's `gemm_a16w16`: the N 6144 × K 2048 kernel at m = 2–64 1.09–1.31; the N 2624 × K 6144 kernel at m = 2, 4, 8 1.085–1.106, at m = 64 1.236; recorded as ranges only, raw reports not kept
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, balanced AB/BA pairs: device-graph calls timed by events (1 GiB flush per call at K = 12288); the ruler's timer (Kineto, 1 GiB flush per call) at M = 16] (64, 2624, 12288): no split, `BLOCK_N` 32, 82 workgroups 48.27 µs against split 6, `BLOCK_N` 64 22.70 µs, split / no split 0.4701 (99 % interval 0.464–0.476), 24 pairs; M = 16, N = 2624, K = 6144: the N 2624 × K 6144 kernel's m ≤ 16 launch (`BLOCK_K` 512, no split) 9.74 and 9.67 µs against split 6 12.09 and 12.04 µs, split / no split 1.240 (1.234–1.247) and 1.244 (1.242–1.247) for two seeds, 16 pairs each; the same launch against split 6 (`BLOCK_N` 64, `BLOCK_K` 128) at N = 2624, K = 6144 in device-graph event timing, 48 pairs per seed, two seeds, 99.875 % intervals, accurate before and after timing: split / no split at M = 16 0.9831 (0.9607–1.0059) and 0.9799 (0.9556–1.0048), 16.18 and 16.23 µs no split; M = 2 1.0949 (1.0768–1.1132) and 1.0867 (1.0702–1.1035); M = 4 1.0371 (0.9849–1.0920) and 1.0638 (1.0398–1.0883); M = 8 1.0371 (1.0089–1.0661) and 1.1042 (0.8659–1.4081) (plain-Triton copies of the N 2624 × K 6144 kernel)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler's timer (Kineto), 4 alternated rounds per arm; GEMV against MFMA under the same timer, 5 alternated rounds] the N 6144 × K 2048 and N 2624 × K 6144 Triton kernels run at m = 1 and 2 against FlyDSL kernels for N 6144 × K 2048 and N 2624 × K 6144 run at m = 1, 2: FlyDSL / Triton time at m = 1 0.9754 (5.86 against 6.01 µs; pairs 0.973–0.979) and 0.9332 (7.27 against 7.79 µs; pairs 0.926–0.944), FlyDSL lower in every round; at m = 2 Triton / FlyDSL 0.9167 at the N 6144 × K 2048 shape (5.97 against 6.51 µs) and 0.40 at the N 2624 × K 6144 shape (8.03–8.05 against 19.98–20.09 µs per round); a plain-Triton GEMV (fp32 `tl.sum` over all K, `.cg` on B, six (BN, BK) tiles) against the 16-row MFMA tile at m = 1 on five shapes: best GEMV / best MFMA 0.972 at N 6144 × K 2048, 1.174 at (2624, 6144), 1.049 at (4096, 2048), 1.090 at (2048, 7168), 1.019 at (6144, 4096), max error 3.5e-03
- E15 [2026-11, gfx950 (offline), Triton 3.8.0 and 3.7.0, N=4, offline compiles of the two kernels] the m = 1 tiles (`BLOCK_M` 16) compiled with M a runtime i32 and `matrix_instr_nonkdim` 16 on both, not as the kernels' launches call them at m = 1 (E20): 64 (N 6144 × K 2048) and 48 (N 2624 × K 6144) `v_mfma_f32_16x16x32_bf16`, 0 `v_dot2`, in each of 4 runs; the N 6144 × K 2048 kernel's m = 32 kernel on 3.8.0 with and without `schedule_hint="memory-bound-attention"`: byte-identical AMDGCN; on 3.7.0 (3.7.0+gitb4e20bbe.rocm7.14.0a20260608) the hinted LLIR differs only in `"amdgpu-sched-strategy"="iterative-ilp"`, the TTGIR is identical and the AMDGCN differs; `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` gives that attribute on 3.8.0 and none on 3.7.0
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the ruler's timer (rocprofiler-sdk), 30 random-order rounds; ISA of the kernels the N 2624 × K 6144 kernel launches] the N 2624 × K 6144 kernel with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against its direct tile: time ratio 1.155, 1.148, 1.087, 1.009 at m = 2, 4, 8, 64; `ds_bpermute_b32` 12 (m = 2, 4, 8 tile) and 16 (m = 64 tile) against 0; at m = 2, 4, 8, 48 against 192 `v_mfma_f32_16x16x32_bf16` and one loop back-branch against none
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, CUDA events, median of 40 calls × 15 rounds; cold: a 1 GiB flush before each call; warm: one call after the flush, then a spin kernel before each call; graph: 20 back-to-back calls] the two kernels without `cache_modifier=".cg"` on the B load, time over the kernels with it, N 6144 × K 2048 m = 2–64 and N 2624 × K 6144 m = 2, 4, 8, 64: cold 1.011–1.034 (N 6144 × K 2048 m = 32 1.000, not stable), warm 0.728–0.947, graph 0.585–0.875
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the three timers of E17] the two kernels with program order (`pid = raw`) in place of the XCD remap, time over the remapped kernels: cold at most 1.018 (N 6144 × K 2048 m = 2–64, N 2624 × K 6144 m = 2, 4, 8, 64); N 6144 × K 2048 m = 2, 4, 8, 16 warm 0.913–0.949, graph 0.838–0.910
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, the ruler's timer (rocprofiler-sdk), 30 random-order rounds] each kernel's `KROT` flipped, outputs within the accuracy bound, flipped over unflipped time: N 6144 × K 2048, 1 → 0: 1.116 at m = 64, 1.011 at m = 32; N 2624 × K 6144, 1 → 0: 1.016, 1.024, 1.019 at m = 2, 4, 8; m = 64, 0 → 1: 1.004, under that size's A/A threshold of 1.7 %
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=2, the `.amdgcn` of the JIT entry each call compiled; the ruler's timer (rocprofiler-sdk), 30 random-order rounds] the kernels' launches at m = 1 compile M as the constant 1: the N 2624 × K 6144 kernel 192 `v_mfma_f32_16x16x32_bf16` with no loop back-branch, the N 6144 × K 2048 kernel 64, 0 `v_dot2` in both; the N 2624 × K 6144 kernel with `do_not_specialize=['M']`: 48 with one back-branch, outputs bitwise equal, 1.077× the specialized time
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, torch 2.11.0+rocm10.0.0, N=1, two timers: E11's device-graph event timer with a 1 GiB flush (800 rounds) and the ruler's timer (rocprofiler-sdk, 236 rounds); input seeds 3301 and 7717] (M, N, K) = (64, 2624, 6144) at E11's configuration, `BLOCK_N` 64, split 4 over split 6 time: 1.1081 on the event timer and 1.1505 on the ruler's timer, above 1 in every round (the N 2624 × K 6144 kernel's GEMM and reduce kernels)
- S5 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:104 `schedule_hint`: "Experimental; right now no effect", the field (:105) the only mention of `schedule_hint` in Triton 3.8.0's Python
- S6 triton/backends/amd/compiler.py:393 in the Triton 3.7.0 wheel (3.7.0+gitb4e20bbe.rocm7.14.0a20260608): `memory-bound-attention` in `schedule_hint` adds `amdgpu-sched-strategy` `iterative-ilp` (:394)
- S14 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:110 `llvm_fn_attrs`, set on the kernel function at :479-481
- S15 triton-lang/triton@c01b677:python/src/specialize.cc:274 an integer argument equal to 1 is passed as a constexpr when its parameter is specialized; `do_not_specialize` turns that off (python/triton/runtime/jit.py:412)
