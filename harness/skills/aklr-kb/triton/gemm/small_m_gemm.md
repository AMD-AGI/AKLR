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

How AKLR's two plain-Triton `gemm@1.0` kernels (`trans_b`, no bias: n = 6144, k = 2048, a 25 MB weight, `f806`; n = 2624,
k = 6144, 32 MB, `b88c`; S13) read their weight once per call, and what was measured to move them, on MI355X with Triton
3.8.0. Times from AKLR's validate and `bench_cudagraph` were taken with its Kineto timer, before #33 replaced it with
AKLR's rocprofiler-sdk collector; both sum a call's per-kernel device durations. Each Evidence line names its timer.
CI's stored speedups against aiter, validate lower bounds, are in [../gemm.md](../gemm.md) §1 and §9.

The configurations were tuned before AKLR's CI (Triton version not recorded), where the source campaign recorded lower
bounds against aiter of 1.09–1.31 (`f806`, M = 2–64), 1.085–1.106 (`b88c`, M = 2, 4, 8) and 1.236 (`b88c`, M = 64),
repeat count unknown *(field, E12)*.

## Route here when

- You write or tune a 16-bit GEMM with M ≤ 64 in plain `tl.*` (`tl.dot`) for an AKLR schema: tile, weight view, tile
  order, K phase, split-K under a `bitwise` contract, the M = 1 case.
- A configuration tuned elsewhere loses under AKLR's measurement (§8).

Elsewhere: what Triton 3.8.0 compiles these two files to, and the fp8 Gluon GEMMs, [../gemm.md](../gemm.md) §9; the same
regime measured on FlyDSL kernels, [../../gemm/small_m.md](../../gemm/small_m.md); what one more launch costs under each
ruler, [../authoring.md](../authoring.md) §7; how to A/B a change, [../tuning.md](../tuning.md) §2. "Kept, not
separated" marks a technique the committed kernels use whose own effect was not measured.

## 1. It is a weight stream: read B once

At small M the call ran close to a plain read of its weight under the same timer: `f806` (25.17 MB) took 5.94–6.38 µs
at M = 2–16 (3.95–4.23 TB/s) against 5.53–5.61 µs for the read, within 1.2×; `b88c` (32.24 MB) took 7.98–8.40 µs at
M = 2–8 against 6.90–7.02 µs, within about 1.22×, over 4 runs *(field, E1)*. `b88c` has no M = 16 or 32 Point.
Above that weight reading alone no longer sets time: `f806` took 7.16–7.22 µs at M = 32; at M = 64 `f806` took
8.53–8.56 µs and `b88c` 12.49–12.65 µs, 1.52–1.54× and 1.78–1.83× their same-run reads *(field, E1)*.
Reading B once is the source construction, not a measured HBM-traffic count:

- One M tile covers all of A: `BLOCK_M = 16` for M ≤ 16, then 32 or 64 (S3, S4). On a synthetic plain-Triton bf16 GEMM
  (N 6144, K 2048), two M tiles took 1.426–1.428× one tile at M = 32 (10.43–10.48 against 7.30–7.35 µs)
  and 1.358–1.379× at M = 64 (12.43–12.50 against 9.03–9.15 µs), 4 runs, 5 alternating rounds per arm with
  disjoint ranges in each run *(field, E2)*. This tests the combined tile-shape change (MFMA, VGPRs and program count
  change too), not isolated HBM traffic or AKLR kernels; M ≤ 16 was not A/B tested.
- Each workgroup owns one N tile across all rows of A, walks all of K, accumulates in fp32 registers and stores once
  (S3, S4; `b88c` at M = 64 splits K instead, §6). Neither file contains an atomic, and each of the 9 Points served
  without a split (`f806` at M = 2 – 64, `b88c` at M = 2, 4, 8) gave one output over 50 runs, inside `gemm@1.0`'s
  bound *(field, E8)*.

## 2. Load B in the layout `tl.dot` wants

For a row-major (N, K) weight (`trans_b`), build the `[BLOCK_K, BLOCK_N]` tile directly:
`b_offs = offs_k[:, None] + offs_n[None, :] * K`, then `tl.dot(a, b)` with no `tl.trans` (S2, S4). In `b88c`, loading
`[BLOCK_N, BLOCK_K]` and calling `tl.trans` instead added `ds_bpermute_b32` shuffles (509 against 350 instructions for
the 16 × 16 × 512 tile, 331 against 261 for the 64 × 64 × 128 tile) and took 1.12 – 1.17× the time at M = 2, 4, 8
and 1.01× at M = 64, in one run *(field, E3)*. Both kernels load B with `cache_modifier=".cg"` (S2, S4; kept, not separated: its
effect on them is untested; what the hint compiles to and bought elsewhere: [../gluon/memory.md](../gluon/memory.md)
§6).

## 3. Order the tiles per XCD

On the tested grids, launched on the default stream, workgroup `i` ran on XCD `i % 8` *(field, E4)*; on a new stream
the absolute XCD index rotates by a constant and workgroups congruent mod 8 still share an XCD
([../gluon/memory.md](../gluon/memory.md) §14). The placements below assume that grouping. Both kernels remap the
program id so each XCD gets N tiles in contiguous runs: one run per XCD in `b88c`, runs of 8 or 24
tiles in `f806` (S1, S4, S9; kept, not separated: the remap's effect on their time is untested; what XCD order
measured on other kernels: [../tuning.md](../tuning.md) §12). A remap must be a permutation of the grid for every grid
size launched, including one not divisible by 8 (164 tiles = 8 × 20 + 4). `b88c`'s gives `per` tiles to the first
`tall` XCDs and `per - 1` to the rest, one contiguous run per XCD, and is a permutation for every grid of 1 – 4096
tiles (S1):

```python
# aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:25-35
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

`f806` deals runs of `CHUNK = 8` tiles per XCD at M = 32 and 64 (3 runs per XCD over its 192 tiles) and one run of 24
per XCD at M ≤ 16 (S4, S9). Neither of its remaps is a permutation at 164 tiles (S9). A remap that is not a permutation
leaves tiles unwritten, and timing does not flag it: `b88c` at M = 8 with `xcd * (GRID // 8) + local` on its 164 tiles
skipped tiles 160 – 163 (columns 2560 – 2623 stayed NaN) and computed tiles 20, 40, 60, 80 twice, yet timed 0.999× and
0.993× the committed kernel in two runs (back-to-back calls, no flush), within the run spread *(field, E5)*. `f806`'s
`CHUNK` remap on 164 tiles instead misses 12 tiles and sends 12 programs past the grid (S9). Check bijectivity for each
grid size used.

## 4. Start each workgroup's K walk at its own phase

Without a K phase every workgroup walks K in the same order. The kernels rotate each workgroup's K walk:
`kphase = ((pid_raw // NXCD) * KROT) % n_kb`, then `kb = ((i + kphase) % n_kb) * BLOCK_K` (S2; `f806` writes
`& (n_kb - 1)`, the same for its power-of-two `n_kb`, S4). At `f806`'s M = 64 launch (n = 6144, k = 2048; 64 × 32
tiles, `BLOCK_K = 256`, 192 workgroups, 4 warps), `KROT = 0` took 9.14 – 9.18 µs against 8.56 – 8.61 µs with
`KROT = 1`, 6.5 – 7.2 % longer in each of 4 runs *(field, E6)*; no other Point was measured. A K-phase timing benefit was also observed
in a separate private-A construction (no two workgroups read the same A line): `KROT = 0`
cost 11.6 – 11.8 % (15.05 – 15.08 against 13.47 – 13.50 µs, 4 runs) *(field, E6)*. The committed configurations use
`KROT = 1` for k = 2048 at M = 32 and 64 and for k = 6144 at M ≤ 16, and `KROT = 0` for k = 6144 at M = 64 (S3, S4).
Only k = 2048 at M = 64 was timed, and without checking the output *(field, E6)*; whether the other three choices
are best is untested.

## 5. Long K, small M: fewer, longer workgroups

For k = 6144 at M ≤ 16, `b88c` uses 16 × 16 tiles with `BLOCK_K = 512` and no K split, so each workgroup walks all of
K: 164 workgroups (S3), fewer than the 256 CUs *(field, E4)*. At M = 2, 4, 8 a call took 8.01 – 8.03, 8.09 – 8.12 and
8.35 – 8.39 µs over 2 runs, 1.14 – 1.20× a plain read of its 32 MB weight *(field, E1)*. This is the source's
configuration, not a rule that fewer workgroups than CUs suffice for long K (§6).

## 6. Split-K under a bitwise contract

Fewer workgroups than CUs is not a general rule for long K. At (M, N, K) = (64, 2624, 12288) an 82-workgroup kernel
with no split (`BLOCK_N` 32) took 48.27 µs against 22.70 µs for split 6 with `BLOCK_N` 64, split / no split 0.470
(24 balanced pairs, one run) *(field, E13)*. At M = 16, K = 6144, `b88c`'s committed M ≤ 16 launch (`BLOCK_K` 512,
no split) beat split 6: split / no split 1.240 and 1.244 for two input seeds, 16 balanced pairs each, under AKLR's
bench timer (Kineto, before #33) *(field, E13)*. That holds under AKLR's timer only: under the device-graph event timer
(48 balanced pairs per seed, two seeds) split / no split at M = 16 was 0.983 and 0.980 with intervals spanning 1, and at
M = 2, 4, 8 no split was faster (split / no split 1.037–1.104, two of the six intervals spanning 1) *(field, E13)*.

`gemm@1.0` is `determinism: bitwise`: every run's output must be byte-identical to the first's (S13). So split-K must
reduce in a fixed order: write each split's fp32 partial tile to scratch (`torch.empty`, sized by the cached host helper:
`_launch_for` returns the size and `run` allocates it) and sum the partials in a second kernel in split order
(S7). On a test GEMM with `b88c`'s M = 64 shapes (6 fp32 partial planes), that gave one output in 100 runs, bit-equal to
a sequential sum in split order (the reverse order differed in 90,596 values), while a float-atomic K split gave 100
different outputs in 100 runs; the shipped `b88c` at M = 64 gave one output in 50 runs, and AKLR's audit and
graph-captured bench accept its scratch *(field, E8)*. The second kernel is one more graph node (what a node costs under AKLR's timer and in a serving graph:
[../authoring.md](../authoring.md) §7):

| k = 6144 | committed launch (S3) | time per call |
|---|---|---|
| M = 64 | 64 × 64 tiles, split 6 (246 workgroups), fixed-order reduce | 12.49 – 12.62 µs, 2 runs *(field, E1)*; 4.2 µs more than its GEMM kernel alone under the Kineto timer (before #33), the timer's cost per kernel rather than the reduce kernel's own time *(field, E9)*; 1.87 µs more under the device-graph timer below *(field, E11)* |
| M = 32 | none served: the committed file lists no M = 32 Point (S8) | — |
| M ≤ 16 | 16 × 16 tiles, no split (164 workgroups) | 8.01 – 8.39 µs at M = 2 – 8 *(field, E1)* |

Before AKLR's CI (Triton version not recorded) the source campaign recorded 1.236 against aiter at M = 64
*(field, E12)*; AKLR's CI stored 1.174 at 43e112e ([../gemm.md](../gemm.md) §1).

**Split count at M = 64.** One run at (M, N, K) = (64, 2624, 6144) *(field, E11)*, with `BLOCK_M=64`,
`BLOCK_K=128`, `NXCD=8`, `KROT=0`, 8 warps, 3 stages, `waves_per_eu=2` and `matrix_instr_nonkdim=16`, swept split ∈
{1, 2, 3, 4, 6, 8, 12} × `BLOCK_N` ∈ {16, 32, 64}. All 21 configurations met the accuracy bound against CPU fp64 /
fp32 references on 3 input seeds and gave byte-identical output over 100 repeats per configuration and seed.

A 16-pair pilot picked one `BLOCK_N` per split (32 for splits 1, 2, 3; 64 for 4, 6, 8, 12); each comparison then
took 32 balanced, randomized AB/BA pairs with A/A pairs interleaved, on one input seed. Split 6 / `BLOCK_N` 64 beat
splits 1 / 2 / 3 / 8 / 12 (their time over split 6's: 1.744 / 1.242 / 1.085 / 1.215 / 1.178). Against split 4 /
`BLOCK_N` 64 the ratio was 1.095 (family-corrected 99 % interval 1.068–1.123), but that did not clear the A/A noise
floor plus the AB/BA order gap, so split 4 against 6 is unresolved, and split 6 is not shown best even within this
sweep *(field, E11)*. Other N, K and tile choices were not searched.

The timer measured one preallocated device-graph call, with a 1 GiB flush outside each event window: whole = GEMM +
fixed-order reduce, partial = GEMM only. Split 6's whole / partial means were 16.140 / 14.273 µs; the paired
difference, 1.868 µs (interval 1.466–2.269), estimates the extra device-graph cost, not the reduce kernel's own time
*(field, E11)*. Allocation, Python dispatch, JIT, capture and ramp-up were excluded. Do not compare these absolute
times with the validate times above, which use another timer.

## 7. M = 1: a 16-row MFMA tile carrying one row

At M = 1 both source paths select a 16-row tile: with M as a runtime i32 and `matrix_instr_nonkdim=16`, both compile
to `v_mfma_f32_16x16x32_bf16` (64 in `f806`, 48 in `b88c`) and no `v_dot2` *(field, E15)*. AKLR's FlyDSL kernels at
these two shapes take a dot / vector path at M = 1, with `fdot2` and no MFMA (S11). Neither Triton file lists an
M = 1 Point (S8). In one validate A/B (4 alternated rounds) of Triton copies given an M = 1 Point against those
FlyDSL kernels, FlyDSL took 0.975× the Triton time at `f806`'s shape (5.86 against 6.01 µs) and 0.933× at `b88c`'s
(7.27 against 7.79 µs), lower in every round; at M = 2 on `f806`'s shape, where FlyDSL also takes its dot path, the
Triton tile took 0.917× FlyDSL's time, and 0.40× at `b88c`'s (8.0 against 20.0 µs) *(field, E14)*. A plain-Triton
GEMV (`tl.sum` over K, no `tl.dot`, six tile sizes) beat the 16-row tile on 1 of 5 shapes (N = 6144, K = 2048: 0.972×)
*(field, E14)*. Gluon kernels fill the spare rows instead ([../gluon/mfma.md](../gluon/mfma.md) §8).

## 8. Re-tune under AKLR's measurement

A configuration taken from elsewhere did not carry. At k = 2048, M = 64 the `BLOCK_M = 64` tier taken from the source
kernel (`BLOCK_K = 128`, 8 warps, no K phase) took 1.05 – 1.08× the time of the re-tuned one (`BLOCK_K = 256`,
4 warps, `KROT = 1`; the three changed together): about 9.0 – 9.3 µs against 8.53 – 8.78 µs under AKLR's flushed graph
timer, 4 runs *(field, E7)*. The tier's speedup against aiter was not measured. A hot cache does not explain the gap:
timed with the weight hot (one flush, then replays of a 50-call graph, where the re-tuned kernel took 6.99 – 7.05 µs
against 8.53 – 8.78 µs flushed), the source tier took 1.08 – 1.13× the re-tuned one's time, against 1.05 – 1.08×
flushed; the hot ratio was 0.02 – 0.05 higher in each of 4 runs *(field, E7)*. Only M = 64 was timed. How to A/B such
a choice: [../tuning.md](../tuning.md) §1–§2.

## 9. Kept, not separated

`waves_per_eu = 2` and `num_stages = 3` on every GEMM launch; `matrix_instr_nonkdim = 16` on all but `f806`'s M ≤ 16
kernel (`_gemm_m16_kernel`), which passes none (S3, S4, S12); and on `f806`'s M = 32 launch
`schedule_hint="memory-bound-attention"` (S4). Triton 3.8.0 reads the hint nowhere: its only mention is the
`HIPOptions` field ("Experimental; right now no effect", S5), and `f806`'s M = 32 kernel compiled to byte-identical
AMDGCN with and without it *(field, E15)*. Triton 3.7.0 turns the hint into the `iterative-ilp` scheduler (S6):
`f806`'s M = 32 LLIR then differs from a build without it only in `"amdgpu-sched-strategy"="iterative-ilp"`, the
TTGIR is identical, and the AMDGCN changes *(field, E15)*. To keep that scheduler on Triton 3.8.0, pass
`llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` (S14), which gave the same attribute on 3.8.0 and was
dropped by 3.7.0 *(field, E15)*, and `"requires": {"triton": ">=3.8"}`, which AKLR checks before it builds the kernel
(S10; [../tuning.md](../tuning.md) §9).

At M = 32 `f806` uses a 32-row tile with `BLOCK_K = 64` and `UNROLL = 4` in the source (S4). Compiled at k = 2048
(Triton 3.8.0), the K loop is fully unrolled (64 MFMAs, no loop) with `UNROLL = 4` or 1 alike, so the 4× request
changes neither the MFMA count nor the absence of a K loop in this build *(field, E10)*.

## Evidence

Measured on one MI355X with Triton 3.8.0 in the `lmsysorg/sglang@sha256:1438dd8b…` image (2026-11): E1–E10 on AKLR
d908d01 with this skill's branch; E11 and E13–E15 are recorded in #89; E12 is a pre-CI record, published in AKLR's initial commit 9a29616.
AKLR paths and line numbers are on main `0e02a81` unless a line names another commit; the two Triton kernel sources
are the same at d908d01, 43e112e and 0e02a81.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, AKLR validate (graph replay, 1 GiB flush per call, Kineto timer, before #33) for the GEMMs; a Gluon read-and-sum of the same bytes under AKLR's `bench_cudagraph` for the plain read] over 4 runs: `f806` (25.17 MB weight) at m = 2–16 5.94–6.38 µs (3.95–4.23 TB/s), m = 32 7.16–7.22, m = 64 8.53–8.56 µs, plain read 5.53–5.61 µs (4.48–4.55 TB/s); `b88c` (32.24 MB) at m = 2–8 7.98–8.40 µs, m = 64 12.49–12.65 µs, plain read 6.90–7.02 µs; m = 64 over the same run's read 1.52–1.54 (`f806`) and 1.78–1.83 (`b88c`). Two of the runs per m: `f806` m = 2, 4, 8, 16, 32, 64 5.95 / 5.98, 5.98 / 5.98, 6.05 / 6.07, 6.36 / 6.34, 7.21 / 7.19, 8.56 / 8.54 µs, read 5.61 / 5.59; `b88c` m = 2, 4, 8, 64 8.03 / 8.01, 8.12 / 8.09, 8.39 / 8.35, 12.49 / 12.62 µs, read 7.02 / 6.97 (f806ad2b086d757d, b88c8d5f28d86777; AKLR@d908d01)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, AKLR `bench_cudagraph` (Kineto timer, before #33), median of 5 alternated rounds] a plain-Triton bf16 test GEMM, N 6144, K 2048, `BLOCK_N` 32, `BLOCK_K` 256, 4 warps: one M tile against two that each re-read B, m = 32: 7.30–7.35 against 10.43–10.48 µs (1.426–1.428×), m = 64: 9.03–9.15 against 12.43–12.50 µs (1.358–1.379×), round ranges disjoint in each run, max relative error 0.25–0.26 % (test kernel; AKLR@d908d01)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR validate A/B (Kineto timer, before #33), 4 alternated rounds; ISA of the two tiers] `b88c` with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against the committed direct tile: time ratio 1.172, 1.163, 1.121, 1.009 at m = 2, 4, 8, 64; 16 × 16 × 512 tile, 2 warps: 509 against 350 instructions (16 against 0 `ds_bpermute_b32`); 64 × 64 × 128 tile, 8 warps: 331 against 261 (6 against 0) (b88c8d5f28d86777; AKLR@d908d01)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, XCC and CU ids read from the hardware in each workgroup, launched on the default stream] XCD = workgroup id % 8 with no mismatch on 1-D grids of 8, 100, 2048 and 4096 workgroups and on 6 multi-axis grids (3 launches each), and with offset 0 in 5 further launches of 4096, in each of 3 runs; 256 CUs (`multi_processor_count`, and 32 distinct CU ids on each of 8 XCCs) in each run; `b88c`'s m ≤ 16 grid read from its source: 164 (b88c8d5f28d86777; AKLR@d908d01)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, CUDA events over 20 back-to-back calls (no flush), median of 9, 5 alternated rounds] `b88c` at m = 8 with the remap `xcd * (GRID // 8) + local` on 164 tiles: columns 2560 – 2623 NaN in all 8 rows, bitwise equal elsewhere; broken / committed 11.93 / 11.94 µs (0.999×; round ranges 11.90 – 12.05 and 11.88 – 12.53), the other run 12.59 / 12.69 µs (0.993×; 12.53 – 12.73 and 12.55 – 12.72) (b88c8d5f28d86777; AKLR@d908d01)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, AKLR `bench_cudagraph` (graph replay, flush per call, Kineto timer, before #33), median of 5 alternated rounds] `f806`'s m = 64 kernel launched directly with the committed options (64 × 32 tiles, `BLOCK_K` 256, 192 workgroups, 4 warps), `KROT` 1 against 0: A shared by all workgroups 8.562 – 8.609 against 9.136 – 9.181 µs, `KROT` 0 / 1 = 1.065 – 1.072 (per run 8.581 / 9.136, 8.609 / 9.166, 8.562 / 9.181, 8.562 / 9.155 µs); A private to each workgroup (A of 192 × 64 rows, each workgroup offset to its own 64 rows by its remapped tile index) 13.47 – 13.50 against 15.05 – 15.08 µs, 1.116 – 1.118 (per run 13.47 / 15.07, 13.50 / 15.07, 13.48 / 15.05, 13.49 / 15.08 µs); output not checked (f806ad2b086d757d; AKLR@d908d01)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, flushed: AKLR `bench_cudagraph` (Kineto timer, before #33); hot: one flush and one replay of a 50-call graph, then the next replay timed, median of 15; median of 3 rounds each] `f806` at m = 64 (k 2048), the source's tier (`BLOCK_K` 128, 8 warps, `KROT` 0) against the committed launch, one input: flushed tier / committed 1.054 – 1.082 (committed 8.527 – 8.78 µs, tier ≈ 9.04 – 9.27 µs), hot 1.080 – 1.128 (committed 6.989 – 7.046 µs), hot ratio minus flushed ratio 0.021 – 0.052; per run flushed 1.054× (8.78 µs) and hot 1.107× (7.031 µs), 1.060× (8.527 µs) and 1.080× (7.046 µs), 1.082× (8.563 µs) and 1.128× (6.99 µs), 1.076× (8.594 µs) and 1.119× (6.989 µs) (f806ad2b086d757d; AKLR@d908d01)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, repeated calls compared bitwise] `f806` at m = 2 – 64 and `b88c` at m = 2, 4, 8: one distinct output in 50 runs at each of the 9 Points, inside the accuracy bound, 0 atomics in the sources; a test GEMM with 6 fp32 partial planes [6, 64, 2624] summed in split order by a second kernel: one output in 100 runs, bit-equal to the sequential sum (reverse order differs in 90,596 values), a float-atomic K split 100 distinct outputs in 100 runs; the shipped `b88c` at m = 64: one output in 50 runs; `aklr validate --stages audit,bench` of `b88c`: audit passed with 0 findings, no bench finding other than speed, m = 64 timed (f806ad2b086d757d, b88c8d5f28d86777; AKLR@d908d01)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR `bench_cudagraph` (Kineto timer, before #33), median of 3; Kineto trace of a graph replay for the kernel list] `b88c` at m = 64: whole call 12.57 µs, its `_gemm_kernel` alone 8.374 µs, difference 4.2 µs; the replay runs `_gemm_kernel` and `_reduce_kernel` at m = 64, `_gemm_kernel` only at m = 2 (b88c8d5f28d86777; AKLR@d908d01)
- E10 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, compiled-kernel census of an offline build] `f806`'s `_gemm_xcd_kernel` at m = 32 (`BLOCK_M` 32, `BLOCK_N` 32, `BLOCK_K` 64, k 2048, 4 warps) built with `UNROLL` 4 and 1: 64 `v_mfma` in each, no backward branch in either (f806ad2b086d757d; AKLR@d908d01)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one GPU run, one Point, bf16, preallocated device graphs timed by events, 1 GiB flush outside each event window; confirmation by 99 % family-corrected paired-log Student-t intervals against the larger of the local and global A/A floors plus the AB/BA order gap] (M, N, K) = (64, 2624, 6144), `BLOCK_M` 64, `BLOCK_K` 128, `KROT` 0, 8 warps, 3 stages, `waves_per_eu` 2: all 21 split × `BLOCK_N` configurations accurate on 3 seeds and byte-identical over 100 repeats; competitor / split 6 (`BLOCK_N` 64) time 1.744 / 1.242 / 1.085 / 1.215 / 1.178 for splits 1 / 2 / 3 / 8 / 12, 1.095 (1.068–1.123) for split 4, unresolved; split 6 whole / partial 16.140 / 14.273 µs, difference 1.868 µs (1.466–2.269); clocks and package versions not recorded (plain-Triton copies of b88c8d5f28d86777; #89)
- E12 [2026-10, gfx950 (MI355X), Triton version not recorded, N=?, AKLR validate lower bound of 32 rounds, 8 rotated input sets, graph replay, 1 GiB flush per call] lower bounds against aiter's `gemm_a16w16`: `f806` at m = 2–64 1.09–1.31; `b88c` at m = 2, 4, 8 1.085–1.106, at m = 64 1.236; published as ranges only, raw reports not kept (first in AKLR@9a29616, 2026-10-31; AKLR@a12866b:harness/skills/aklr-kb/triton/gemm/small_m_gemm.md §1, §5, §6)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, balanced AB/BA pairs: device-graph calls timed by events (1 GiB flush per call at K = 12288); AKLR `bench_cudagraph` (Kineto timer, before #33, 1 GiB flush per call) at M = 16] (64, 2624, 12288): no split, `BLOCK_N` 32, 82 workgroups 48.27 µs against split 6, `BLOCK_N` 64 22.70 µs, split / no split 0.4701 (99 % interval 0.464–0.476), 24 pairs; M = 16, N = 2624, K = 6144: `b88c`'s m ≤ 16 launch (`BLOCK_K` 512, no split) 9.74 and 9.67 µs against split 6 12.09 and 12.04 µs, split / no split 1.240 (1.234–1.247) and 1.244 (1.242–1.247) for two seeds, 16 pairs each; the same launch against split 6 (`BLOCK_N` 64, `BLOCK_K` 128) at N = 2624, K = 6144 in device-graph event timing, 48 pairs per seed, two seeds, 99.875 % intervals, accurate before and after timing: split / no split at M = 16 0.9831 (0.9607–1.0059) and 0.9799 (0.9556–1.0048), 16.18 and 16.23 µs no split; M = 2 1.0949 (1.0768–1.1132) and 1.0867 (1.0702–1.1035); M = 4 1.0371 (0.9849–1.0920) and 1.0638 (1.0398–1.0883); M = 8 1.0371 (1.0089–1.0661) and 1.1042 (0.8659–1.4081) (plain-Triton copies of b88c8d5f28d86777; #89)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR validate (Kineto timer, before #33), 4 alternated rounds per arm; GEMV against MFMA under AKLR's `bench_cudagraph`, 5 alternated rounds] Triton copies of `f806` and `b88c` given m = 1 and 2 Points against AKLR's FlyDSL `gemm_n6144_k2048` and `gemm_n2624_k6144` cut to m = 1, 2: FlyDSL / Triton time at m = 1 0.9754 (5.86 against 6.01 µs; pairs 0.973–0.979) and 0.9332 (7.27 against 7.79 µs; pairs 0.926–0.944), FlyDSL lower in every round; at m = 2 Triton / FlyDSL 0.9167 at `f806`'s shape (5.97 against 6.51 µs) and 0.40 at `b88c`'s (8.03–8.05 against 19.98–20.09 µs per round); a plain-Triton GEMV (fp32 `tl.sum` over all K, `.cg` on B, six (BN, BK) tiles) against the 16-row MFMA tile at m = 1 on five shapes: best GEMV / best MFMA 0.972 at N 6144, K 2048, 1.174 at (2624, 6144), 1.049 at (4096, 2048), 1.090 at (2048, 7168), 1.019 at (6144, 4096), max error 3.5e-03 (f806ad2b086d757d, b88c8d5f28d86777, FlyDSL eb749c07b94008ab and 4001df477e6c67f6; #89)
- E15 [2026-11, gfx950 (offline), Triton 3.8.0 and 3.7.0, N=4, offline compiles of the committed launches] the m = 1 tiles (`BLOCK_M` 16, M a runtime i32, `matrix_instr_nonkdim` 16 on both): 64 (`f806`) and 48 (`b88c`) `v_mfma_f32_16x16x32_bf16`, 0 `v_dot2`, in each of 4 runs; `f806`'s m = 32 kernel on 3.8.0 with and without `schedule_hint="memory-bound-attention"`: byte-identical AMDGCN; on 3.7.0 (3.7.0+gitb4e20bbe.rocm7.14.0a20260608) the hinted LLIR differs only in `"amdgpu-sched-strategy"="iterative-ilp"`, the TTGIR is identical and the AMDGCN differs; `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` gives that attribute on 3.8.0 and none on 3.7.0 (f806ad2b086d757d, b88c8d5f28d86777; #89)
- S1 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:26 `_xcd_remap`: `per` tiles to the first `tall` XCDs, `per - 1` to the rest (to :35); a permutation with one run per XCD for every grid of 1 – 4096 (164: `per` 21, `tall` 4; 246: `per` 31, `tall` 6), enumerated from the formula
- S2 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:62 `kphase = ((raw // NXCD) * KROT) % n_kb`; `b_offs` as `[BLOCK_K, BLOCK_N]` (:68); `kb = ((i + kphase) % n_kb) * BLOCK_K` (:71); B loaded with `.cg` and `tl.dot(a, b, acc=acc)` (:76-77)
- S3 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:100 m ≤ 16: `n // 16` tiles (:101), `BLOCK_M=16`, `BLOCK_N=16`, `BLOCK_K=512`, `SPLIT_K=1` (:113-116), `KROT=1` (:120); m = 64: 64 × 64 tiles, `BLOCK_K=128`, split 6, `KROT=0`, 8 warps (:128-149), then the reduce (:154)
- S4 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:145 m ≤ 32: `BLOCK_M=32`, `BLOCK_K=64`, `UNROLL=4`, `KROT=1`, 4 warps, `schedule_hint="memory-bound-attention"` (:156-168); m = 64: `BLOCK_M=64`, `BLOCK_K=256`, `KROT=1`, 4 warps (:182-190); m ≤ 16: `BLOCK_M=16` (:133), no `matrix_instr_nonkdim` (:139-141), remap `(pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)` (:45); `kphase = ((pid // NXCD) * KROT) & (n_kb - 1)` (:90), the `CHUNK` remap (:93); `b_offs` (:99), `kb = ((i + kphase) & (n_kb - 1)) * BLOCK_K` (:102), `.cg` and `tl.dot` (:107-108)
- S5 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:104 `schedule_hint`: "Experimental; right now no effect", the field (:105) the only mention of `schedule_hint` in Triton 3.8.0's Python
- S6 triton/backends/amd/compiler.py:393 in the Triton 3.7.0 wheel (3.7.0+gitb4e20bbe.rocm7.14.0a20260608): `memory-bound-attention` in `schedule_hint` adds `amdgpu-sched-strategy` `iterative-ilp` (:394)
- S7 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:87 `_reduce_kernel` adds the split planes in order, `tl.static_range(1, SPLIT_K)` (:91-92)
- S8 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json:61 `workloads`: Points m = 2, 4, 8 and 64 (:64, :71, :78, :85), none at 1 or 32; `f806`'s (`gemm_n6144_k2048/f806ad2b086d757d.json:61`) m = 2, 4, 8, 16, 32, 64
- S9 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:45 the m ≤ 16 remap `(pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)`; the `CHUNK` remap at :91-93; enumerated from these formulas: at 192 tiles the `CHUNK` remap gives 3 runs of 8 per XCD and the m ≤ 16 remap one run of 24, both permutations; at 164 tiles the m ≤ 16 remap computes 20, 40, 60, 80 twice and misses 160 – 163, the `CHUNK` remap misses 12 tiles and reaches tile 187
- S10 aklr/core/platform/toolchain.py:24 `unmet_requirement`: why the installed toolchain misses a kernel's `requires`, e.g. `triton 3.7.0 is not >=3.8` (:31-32); dispatch skips such a schema before building it (aklr/core/runtime/dispatch/_registry.py:127-129) and validate fails it with `NO_MATCHING_TOOLCHAIN` (aklr/core/validation/stages/_bench/_evaluation/_candidate.py:170-172)
- S11 aklr/ops/gemm/impls/flydsl/gemm_n6144_k2048/eb749c07b94008ab.json#kernel.py:818 `DOT_MAX_M` 2 at (n, k) = (6144, 2048): m ≤ 2 take `_dot_gemm` (:25), whose products are `fdot2_f32_bf16` (:59); aklr/ops/gemm/impls/flydsl/gemm_n2624_k6144/4001df477e6c67f6.json#kernel.py:854 `VECTOR_GEMV` 1 at m = 1: `_vector_gemv` (:58), `fdot2_f32_bf16` (:96); neither body issues an MFMA
- S12 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:121 `num_stages=3`, `waves_per_eu=2`, `matrix_instr_nonkdim=16` on the m ≤ 16 launch (:121-124) and the m = 64 launch (:149-152); gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:139 the m ≤ 16 launch with `num_stages=3`, `waves_per_eu=2` and no `matrix_instr_nonkdim` (:139-141), the m ≤ 32 (:164-168) and m = 64 (:190-193) launches with it
- S13 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json:4 `gemm@1.0`, `trans_b` true and no bias (:24-25), bf16 operands (:33, :44), `"determinism": "bitwise"` (:107); gemm_n6144_k2048/f806ad2b086d757d.json:121 the same determinism; these two are the only Triton `gemm@1.0` schemas at 43e112e and on main; weights 2624 × 6144 and 6144 × 2048 bf16, 32,243,712 and 25,165,824 bytes (computed); validate's bitwise rule, every run's output byte-identical to the first's: aklr/core/contract/semantics/_compare.py:142
- S14 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:110 `llvm_fn_attrs`, set on the kernel function at :479-481
