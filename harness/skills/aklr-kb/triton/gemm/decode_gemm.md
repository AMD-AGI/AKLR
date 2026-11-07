---
title: "Decode GEMM in plain Triton: bf16, M ≤ 64, the weight read once (gfx950)"
kind: topic
ops: [gemm]
dtypes: [bf16]
regimes: [decode]
gens: [gfx950]
triton: 3.8.0
---

# Decode GEMM in Triton (bf16, M ≤ 64, gfx950)

How AKLR's two plain-Triton `gemm@1.0` kernels (`trans_b`, no bias: n = 6144, k = 2048, a 25 MB weight, `f806`; n = 2624,
k = 6144, 32 MB, `b88c`) read their weight once per call. Measurements below are finite MI355X
examples with Triton 3.8.0; distinguish AKLR's flushed Kineto-duration timer from a device-graph event timer.
CI's stored lower-bound speedups are in [../gemm.md](../gemm.md) §1. Source references are at the end.

## Route here when

- You write or tune a 16-bit GEMM with M ≤ 64 in plain `tl.*` (`tl.dot`) for an AKLR schema: tile, weight view, tile
  order, K phase, split-K under a `bitwise` contract, the M = 1 case.
- A configuration tuned elsewhere loses under AKLR's measurement (§8).

Elsewhere: what Triton 3.8.0 compiles these two files to, and the fp8 Gluon GEMMs, [../gemm.md](../gemm.md) §9; the same
regime measured on FlyDSL kernels, [../../gemm/small_m.md](../../gemm/small_m.md); what one more launch costs under each
ruler, [../authoring.md](../authoring.md) §7; how to A/B a change, [../tuning.md](../tuning.md) §2. "Kept, not
separated" marks a technique the committed kernels use whose own effect was not measured.

## 1. It is a weight stream: read B once

At the tested small M, f806 M2–16 ran within 1.2× of a plain read; b88c M2–8 ran within about 1.22× under the same timer: `f806` (25.17 MB)
took 5.94–6.38 µs at M = 2–16 (3.95–4.23 TB/s) against 5.53–5.61 µs for the read; `b88c` (32.24 MB) took
7.98–8.40 µs at M = 2–8 against 6.90–7.02 µs. `b88c` M = 16/32 was not tested in.
Above that weight reading alone no longer sets time: `f806` took 7.16–7.22 µs at M = 32; at M = 64 `f806` took
8.53–8.56 µs and `b88c` 12.49–12.65 µs, 1.52–1.54× and 1.78–1.83× their same-run reads.
Reading B once is the source construction, not a measured HBM-traffic count:

- One M tile covers all of A: `BLOCK_M = 16` for M ≤ 16, then 32 or 64 (S3, S4). On a synthetic plain-Triton bf16 GEMM
  (N 6144, K 2048), two M tiles took 1.426–1.428× one tile at M = 32 (10.43–10.48 against 7.30–7.35 µs)
  and 1.358–1.379× at M = 64 (12.43–12.50 against 9.03–9.15 µs) with
  disjoint ranges in each run. This tests the combined tile-shape change (MFMA, VGPRs and program count
  change too), not isolated HBM traffic or AKLR kernels; M ≤ 16 was not A/B tested.
- Each workgroup owns one N tile across all rows of A, walks all of K, accumulates in fp32 registers and stores once
  (S3, S4; `b88c` at M = 64 splits K instead, §6). Neither file contains an atomic, and each of the 9 Points served
  without a split (`f806` at M = 2 – 64, `b88c` at M = 2, 4, 8) gave one output, inside `gemm@1.0`'s
  bound.

## 2. Load B in the layout `tl.dot` wants

For a row-major (N, K) weight (`trans_b`), build the `[BLOCK_K, BLOCK_N]` tile directly:
`b_offs = offs_k[:, None] + offs_n[None, :] * K`, then `tl.dot(a, b)` with no `tl.trans` (S2, S4). In `b88c`, loading
`[BLOCK_N, BLOCK_K]` and calling `tl.trans` instead added `ds_bpermute_b32` shuffles (509 against 350 instructions for
the 16 × 16 × 512 tile, 331 against 261 for the 64 × 64 × 128 tile) and took 1.12 – 1.17× the time at M = 2, 4, 8
and 1.01× at M = 64, . Both kernels load B with `cache_modifier=".cg"` (S2, S4; kept, not separated: its
effect on them is untested; what the hint compiles to and bought elsewhere: [../gluon/memory.md](../gluon/memory.md)
§6).

## 3. Order the tiles per XCD

In  tested grids on this MI355X, workgroup `i` ran on XCD `i % 8`; the following placement descriptions assume that mapping. Both kernels remap the program id so each XCD gets N tiles in
contiguous runs:  per XCD in `b88c`, runs of 8 or 24 tiles in `f806` (S1, S4; kept, not separated: the remap's effect on their time is untested; what XCD order measured on other
kernels: [../tuning.md](../tuning.md) §12). A remap must be a permutation of the grid for every grid size launched,
including one not divisible by 8 (164 tiles = 8 × 20 + 4). `b88c`'s gives `per` tiles to the first `tall` XCDs and
`per - 1` to the rest, one contiguous run per XCD, and is a permutation for every grid of 1 – 4096 tiles (S1):

```python
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

`f806` deals runs of `CHUNK = 8` tiles per XCD at M = 32 and 64 (3 runs per XCD over its 192 tiles) and  of 24
per XCD at M ≤ 16 (S4). Neither of its remaps is a permutation at 164 tiles. A remap that is not a permutation
leaves tiles unwritten, and timing does not flag it: `b88c` at M = 8 with `xcd * (GRID // 8) + local` on its 164 tiles
skipped tiles 160 – 163 (columns 2560 – 2623 stayed NaN) and computed tiles 20, 40, 60, 80 twice, yet timed 0.999× and
0.993× the committed kernel in two runs (back-to-back calls, no flush), within the run spread. `f806`'s
`CHUNK` remap on 164 tiles instead misses 12 tiles and sends 12 programs past the grid. Check bijectivity for each
grid size used.

## 4. Start each workgroup's K walk at its own phase

Without a K phase every workgroup walks K in the same order. The kernels rotate each workgroup's K walk:
`kphase = ((pid_raw // NXCD) * KROT) % n_kb`, then `kb = ((i + kphase) % n_kb) * BLOCK_K` (S2; `f806` writes
`& (n_kb - 1)`, the same for its power-of-two `n_kb`, S4). At `f806`'s M = 64 launch (n = 6144, k = 2048; 64 × 32
tiles, `BLOCK_K = 256`, 192 workgroups, 4 warps), `KROT = 0` took 9.14 – 9.18 µs against 8.56 – 8.61 µs with
`KROT = 1`, 6.5 – 7.2 % longer in each of; no other Point was measured. A K-phase timing benefit was also observed
in a separate private-A construction (no two workgroups read the same A line): `KROT = 0`
cost 11.6 – 11.8 % (15.05 – 15.08 against 13.47 – 13.50 µs). The committed configurations use
`KROT = 1` for k = 2048 at M = 32 and 64 and for k = 6144 at M ≤ 16, and `KROT = 0` for k = 6144 at M = 64 (S3, S4).
Only k = 2048 at M = 64 was A/B timed above; those shared/private KROT timing arms did not check numeric output. The timings alone are not a correctness-and-speed comparison, and whether the other three choices are best is untested.

## 5. Long K, small M: fewer, longer workgroups

For k = 6144 at M ≤ 16, `b88c` uses 16 × 16 tiles with `BLOCK_K = 512` and no K split, so each workgroup walks all of
K: 164 workgroups (S3), fewer than the 256 CUs. At M = 2, 4, 8 a call took 8.01 – 8.03, 8.09 – 8.12 and
8.35 – 8.39 µs, 1.14 – 1.20× a plain read of its 32 MB weight. This source configuration is not a rule that fewer workgroups than CUs suffice for long K;
compare full-call split variants at each Point, including their merge cost.

## 6. Split-K under a bitwise contract

Fewer workgroups than CUs is not a general long-K performance rule: a legal no-split control at
M64/N2624/K12288 was about 2.1× slower than split6/BN64 under a cold device-graph timer.
This compares complete configurations, not a universal occupancy threshold.

`gemm@1.0` is`gemm@1.0` is `determinism: bitwise`: every run's output must be byte-identical to the first's. So split-K must
reduce in a fixed order: write each split's fp32 partial tile to scratch (`torch.empty`, sized by the cached host helper:
`_launch_for` returns the size and `run` allocates it) and sum the partials in a second kernel in split order
(S7). A fixed-order split/reduce implementation was repeatable on the tested inputs, while an arrival-order
float-atomic split produced different bits. Repeatability does not mean different split configurations must
match one another bitwise; each must meet the numeric bound and reproduce its own output.

| k = 6144 | Committed launch (S3) | Why |
|---|---|---|
| M = 64 | 64 × 64 tiles, split 6, 246 workgroups, fixed-order reduce | more independent work, at the cost of scratch and a second launch |
| M = 2/4/8 | 16 × 16 tiles, no split, 164 workgroups | long K walk in each workgroup; benchmark split neighbours rather than assuming this is optimal |
| M = 32 | no listed Point (S8) | a host branch is not measured/admitted coverage |

At M64/N2624/K6144, a finite cold-cache sweep of split1/2/3/4/6/8/12 with BN16/32/64 and fixed
BM64/BK128, 8 warps, 3 stages and `waves_per_eu=2` favored split6 over the selected split1/2/3/8/12
competitors. Split6 versus split4 remained unresolved after local A/A and order controls: neither equality
nor optimality follows. Broader tile/backend searches and other N/K remain separate tuning tasks.
At M16 the original BK512 no-split path was faster than split6 in the measured cold-call comparisons;
there is no universal split threshold or claim that splitting only pays at M64.

**Price the whole call.** Include GEMM, initialization, scratch allocation where the interface pays it, and reduce.
At the M64 Point above, preallocated event-timed device graphs gave an extra whole-minus-partial cost of about
1.9 µs; AKLR's separate Kineto-duration measurement gave about 4.2 µs. Neither difference is the intrinsic
reduce-kernel duration, and the two absolute times cannot be compared as if they used one ruler.

## 7. M = 1: a 16-row MFMA tile carrying one row

At M = 1 both source paths select a 16-row tile. A compile probe with M as runtime i32 and matrix_instr_nonkdim=16 on both kernels emits
`v_mfma_f32_16x16x32_bf16` and no `v_dot2`. AKLR's FlyDSL kernels at these two shapes take a dot / vector path
at M = 1, with `fdot2` and no MFMA. Neither Triton file lists an M = 1 Point (S8), and how the 16-row tile
compares with those paths at M = 1 is untested. Gluon kernels fill the spare rows instead
([../gluon/mfma.md](../gluon/mfma.md) §8).

## 8. Re-tune under AKLR's measurement

A configuration taken from elsewhere did not carry. At k = 2048, M = 64 the `BLOCK_M = 64` tier taken from the source
kernel (`BLOCK_K = 128`, 8 warps, no K phase) took 1.05 – 1.08× the time of the re-tuned one (`BLOCK_K = 256`,
4 warps, `KROT = 1`; the three changed together): about 9.0 – 9.3 µs against 8.53 – 8.78 µs under AKLR's flushed graph
timer. The tier's speedup against aiter was not measured. A hot cache does not explain the gap:
timed with the weight hot (one flush, then replays of a 50-call graph, where the re-tuned kernel took 6.99 – 7.05 µs
against 8.53 – 8.78 µs flushed), the source tier took 1.08 – 1.13× the re-tuned one's time, against 1.05 – 1.08×
flushed; the hot ratio was 0.02 – 0.05 higher in each of. Only M = 64 was timed. How to A/B such
a choice: [../tuning.md](../tuning.md) §1–§2.

## 9. Kept, not separated

`waves_per_eu = 2` and `num_stages = 3` on every GEMM launch; `matrix_instr_nonkdim = 16` on all but `f806`'s M ≤ 16
kernel (`_gemm_m16_kernel`), which passes none (S3, S4); and on `f806`'s M = 32 launch
`schedule_hint="memory-bound-attention"` (S4). Triton 3.8.0 reads the hint nowhere: its only
mention is the `HIPOptions` field ("Experimental; right now no effect"), and `f806`'s M = 32 kernel compiles to
byte-identical AMDGCN with and without it (S5). Older Triton builds used a different scheduler interpretation. To request `iterative-ilp` explicitly on
Triton 3.8.0, pass `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` and declare
`"requires": {"triton": ">=3.8"}` ([../tuning.md](../tuning.md) §9). Treat that as a measured tuning candidate,
not a guaranteed benefit.

At M = 32At M = 32 `f806` uses a 32-row tile with `BLOCK_K = 64` and `UNROLL = 4` in the source (S4). Compiled at k = 2048
(Triton 3.8.0), the K loop is fully unrolled (64 MFMAs, no loop) with `UNROLL = 4` or 1 alike, so the 4× request does not change
the MFMA count or the absence of a K loop in this build .

## Evidence

AKLR source paths refer to main `43e112e` unless another commit is stated.
Triton compiler paths refer to `triton-lang/triton@c01b677` (v3.8.0).
Measurements in the text are gfx950 examples, not universal performance guarantees.

- S1 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:26 `_xcd_remap`: `per` tiles to the first `tall` XCDs, `per - 1` to the rest (to :35); a permutation with  per XCD for every grid of 1 – 4096 (164: `per` 21, `tall` 4; 246: `per` 31, `tall` 6)
- S2 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:62 `kphase = ((raw // NXCD) * KROT) % n_kb`; `b_offs` as `[BLOCK_K, BLOCK_N]` (:68); `kb = ((i + kphase) % n_kb) * BLOCK_K` (:71); B loaded with `.cg` and `tl.dot(a, b, acc=acc)` (:76-77)
- S3 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:100 m ≤ 16: `n // 16` tiles (:101), `BLOCK_M=16`, `BLOCK_N=16`, `BLOCK_K=512`, `SPLIT_K=1` (:113-116), `KROT=1` (:120); m = 64: 64 × 64 tiles, `BLOCK_K=128`, split 6, `KROT=0`, 8 warps (:128-149), then the reduce (:154)
- S4 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:145 m ≤ 32: `BLOCK_M=32`, `BLOCK_K=64`, `UNROLL=4`, `KROT=1`, 4 warps, `schedule_hint="memory-bound-attention"` (:156-168); m = 64: `BLOCK_M=64`, `BLOCK_K=256`, `KROT=1`, 4 warps (:182-190); m ≤ 16: `BLOCK_M=16` (:133), no `matrix_instr_nonkdim` (:139-141), remap `(pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)` (:45); `kphase = ((pid // NXCD) * KROT) & (n_kb - 1)` (:90), the `CHUNK` remap (:93); `b_offs` (:99), `kb = ((i + kphase) & (n_kb - 1)) * BLOCK_K` (:102), `.cg` and `tl.dot` (:107-108)
- S5 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:104 `schedule_hint`: "Experimental; right now no effect", the field (:105) the only mention of `schedule_hint` in Triton 3.8.0's Python; `f806`'s m = 32 AMDGCN byte-identical with and without the hint
- S7 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:87 `_reduce_kernel` adds the split planes in order, `tl.static_range(1, SPLIT_K)` (:91-92)
- S8 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json:61 `workloads`: Points m = 2, 4, 8 and 64 (:64, :71, :78, :85), none at 1 or 32; `f806`'s (`gemm_n6144_k2048/f806ad2b086d757d.json:61`) m = 2, 4, 8, 16, 32, 64
