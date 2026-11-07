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
k = 6144, 32 MB, `b88c`; S13) read their weight once per call, and what was measured to move them. Times are finite per-call observations on MI355X. E1–E10 record Triton 3.8.0; each Evidence entry states its timer,
including AKLR validate/bench_cudagraph where specified. E11 uses its separately described device-graph ruler; CI's stored speedups against aiter are in [../gemm.md](../gemm.md) §1. The page-local [Evidence](#evidence) records finite measurements and original-kernel or compiler sources;
unverified scope is stated separately.

Historical Triton 3.7 lower bounds 1.177/1.182/1.313/1.138/1.091/1.106 and 1.088/1.085/1.106 are not bound to located original raw reports. Separate Triton 3.8 metadata at 90b9d50 records f806 M2/4/8/16/32/64 = 1.116/1.129/1.292/1.138/1.064/1.116 and b88c M2/4/8/64 = 1.048/1.070/1.062/1.174; these are stored lower bounds, not a replacement for the old measurements. Case coverage is finite; an INDEX entry is not certification of every original claim.

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
7.98–8.40 µs at M = 2–8 against 6.90–7.02 µs, over 4 runs *(field, E1)*. `b88c` M = 16/32 was not tested in E1.
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

In E4's tested grids on this MI355X, workgroup `i` ran on XCD `i % 8`; the following placement descriptions assume that mapping *(field, E4)*. Both kernels remap the program id so each XCD gets N tiles in
contiguous runs: one run per XCD in `b88c`, runs of 8 or 24 tiles in `f806` (S1, S4, S9; kept, not separated: the remap's effect on their time is untested; what XCD order measured on other
kernels: [../tuning.md](../tuning.md) §12). A remap must be a permutation of the grid for every grid size launched,
including one not divisible by 8 (164 tiles = 8 × 20 + 4). `b88c`'s gives `per` tiles to the first `tall` XCDs and
`per - 1` to the rest, one contiguous run per XCD, and is a permutation for every grid of 1 – 4096 tiles (S1):

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
Only k = 2048 at M = 64 was A/B timed above; those shared/private KROT timing arms did not check numeric output (E6). The timings alone are not a correctness-and-speed comparison, and whether the other three choices are best is untested.

## 5. Long K, small M: fewer, longer workgroups

For k = 6144 at M ≤ 16, `b88c` uses 16 × 16 tiles with `BLOCK_K = 512` and no K split, so each workgroup walks all of
K: 164 workgroups (S3), fewer than the 256 CUs *(field, E4)*. At M = 2, 4, 8 a call took 8.01 – 8.03, 8.09 – 8.12 and
8.35 – 8.39 µs over 2 runs, 1.14 – 1.20× a plain read of its 32 MB weight *(field, E1)*. Finite M2/4/8 split comparisons now exist under a separate ruler (G01_original_unsplit_confirm_summary.json);
they do not establish the historical AITER ratios or a global optimum; this source configuration is not a validated rule that fewer
workgroups than CUs are sufficient for long K.

## 6. Split-K under a bitwise contract

The b88c source uses BN16/BK512/split1 at M2/4/8, giving 164 workgroups. The historical 1.085–1.106 AITER lower bounds remain unverified; source configuration and finite correctness do not supply a historical timing denominator.

Fewer workgroups than CUs is not a general long-K performance rule. At (M,N,K)=(64,2624,12288), the legal 82-workgroup BN32 no-split control took about 2.13 times the split6/BN64 time (24 balanced pairs; split/no-split=0.4701). This does not define the original word “fine” or establish a universal threshold.

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
| M = 64 | 64 × 64 tiles, split 6 (246 workgroups), fixed-order reduce | 12.49 – 12.62 µs, 2 runs *(field, E1)*; historical whole-minus-GEMM graph difference ≈ 4.2 µs, not intrinsic reduce time *(field, E9)*; independent final measurement below uses a different ruler (E11) |
| M = 32 | none served: the committed file lists no M = 32 Point (S8) | — |
| M ≤ 16 | 16 × 16 tiles, no split (164 workgroups) | 8.01 – 8.39 µs at M = 2 – 8 *(field, E1)* |

The b88c M64 source uses 64×64 tiles, split6 and a fixed-order reduce. The historical 1.236 AITER ratio has no located raw support; 90b9d50 instead stores 1.174 under its own measurement record.

Independent final check: one bf16 Point `(M,N,K) = (64,2624,6144)` on gfx950, N=1 full GPU run (E11).
With `BLOCK_M=64`, `BLOCK_K=128`, `NXCD=8`, `KROT=0`, 8 warps, 3 stages, `waves_per_eu=2` and
`matrix_instr_nonkdim=16`, the finite sweep tested split ∈ {1,2,3,4,6,8,12} × `BLOCK_N` ∈ {16,32,64}.
All 21 configurations met the accuracy bound against independent CPU fp64/fp32 references on 3 input seeds
and gave byte-identical output over 100 repeats per configuration per seed. These inputs/repeats are not extra full runs.

For each split, a 16-pair pilot selected one `BLOCK_N` (32 for split 1/2/3, 64 for split 4/6/8/12), not a proven
best configuration. Independent confirmation used 32 balanced randomized AB/BA pairs per comparison with interleaved
A/A calibration; performance used only seed 12011. Split 6 / BN64 was faster than the pilot-selected split 1/2/3/8/12
(competitor / split-6 time ratios 1.744/1.242/1.085/1.215/1.178). Against split 4 / BN64, the ratio was 1.095
with family-corrected CI [1.068,1.123], but did not clear the local A/A floor plus order-effect gate:
`unresolved_not_equivalent`. This is neither equivalence nor proof that split 6 is optimal, even in this finite sweep.

The timer measured one preallocated device-graph call, with a 1 GiB flush outside each event window:
whole = GEMM + fixed-order reduce, partial = GEMM only. Split 6 whole/partial means were 16.140/14.273 µs;
the paired difference 1.868 µs (CI [1.466,2.269]) is an extra device-graph cost estimate, not intrinsic reduce time
or an independently calibrated direction claim. Allocation, Python dispatch, JIT/capture and ramp-up were excluded.
Do not compare these absolute times directly with E1/E9's different timer.

**Unverified scope** (case `triton/checks/d_bf16_split_whole_call.py`, check `.scope`; E11):
- At (M,N,K)=(64,2624,12288), the tested legal 82-workgroup no-split control was slower than split6/BN64 (split/no=0.4701, 24 balanced pairs). Long K alone does not establish efficiency (`decode_gemm-20`).
- Finite M2/4/8/16 original no-split versus split6 comparisons have executed under their recorded rulers. At M16, the original BK512 no-split was faster in two cold-call comparisons (split/no=1.240235 and 1.244363, 16 balanced pairs each). This is not global optimality or the historical 0.957 AITER ratio (`decode_gemm-27`). A separate explicit legal vendor split1 six-M comparison also exists; it is not proof of the historical only-M64 claim.
- Other N/K Points, broader tile/backend searches and general optimality remain untested (`decode_gemm-25`).

## 7. M = 1: a 16-row MFMA tile carrying one row

At M = 1 both source paths select a 16-row tile. A compile probe with M as runtime i32 and matrix_instr_nonkdim=16 on both kernels emits
`v_mfma_f32_16x16x32_bf16` and no `v_dot2` (S10). AKLR's FlyDSL kernels at these two shapes take a dot / vector path
at M = 1, with `fdot2` and no MFMA (S11). Neither Triton file lists an M = 1 Point (S8), and how the 16-row tile
compares with those paths at M = 1 is untested. Gluon kernels fill the spare rows instead
([../gluon/mfma.md](../gluon/mfma.md) §8).

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
`schedule_hint="memory-bound-attention"` (S4). Triton 3.8.0 reads the hint nowhere: its only
mention is the `HIPOptions` field ("Experimental; right now no effect"), and `f806`'s M = 32 kernel compiles to
byte-identical AMDGCN with and without it (S5). Triton 3.7.0 turns the hint into the `iterative-ilp` scheduler:
`f806`'s M = 32 LLIR gains `"amdgpu-sched-strategy"="iterative-ilp"`; the recorded check found differences only on LLVM attribute-definition lines, and its AMDGCN changes (S6). To
keep that scheduler on Triton 3.8.0, pass `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]` and
`"requires": {"triton": ">=3.8"}` (S6; [../tuning.md](../tuning.md) §9).

At M = 32 `f806` uses a 32-row tile with `BLOCK_K = 64` and `UNROLL = 4` in the source (S4). Compiled at k = 2048
(Triton 3.8.0), the K loop is fully unrolled (64 MFMAs, no loop) with `UNROLL = 4` or 1 alike, so the 4× request does not change
the MFMA count or the absence of a K loop in this build (S15; *(field, E10)*, one run).

## Evidence

E1–E10 are finite measurements on MI355X with Triton 3.8.0 (image
`lmsysorg/sglang@sha256:1438dd8b…`, AKLR `d908d01` with this skill's branch); E11 states its own final-log scope.
Kernel citations (`aklr/…`) are at AKLR main `43e112e`, `triton/checks/` citations are in this skill.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, AKLR validate (graph replay, flush per call) for the GEMMs; a Gluon read-and-sum of the same bytes under AKLR's bench_cudagraph for the plain read] `f806` (25.17 MB weight) at m = 2, 4, 8, 16, 32, 64: 5.95, 5.98, 6.05, 6.36, 7.21, 8.56 µs (4.23, 4.21, 4.16, 3.96, 3.49, 2.94 TB/s), plain read 5.61 µs (4.48 TB/s); `b88c` (32.24 MB) at m = 2, 4, 8, 64: 8.03, 8.12, 8.39, 12.49 µs, plain read 7.02 µs; earlier run `f806` 5.98, 5.98, 6.07, 6.34, 7.19, 8.54 µs, read 5.59 µs, `b88c` 8.01, 8.09, 8.35, 12.62 µs, read 6.97 µs (case triton/checks/bench_roofline_weight_stream.py (09, 10); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_roofline_weight_stream_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_roofline_weight_stream.log) Four-run evidence (200/800/880/870, this case N=4; preceding values retain their stated historical scope and other-case counts): 09: The bf16 n6144 call at M2–16 took 5.94–6.38 µs versus 5.53–5.61 µs for a plain read; n2624 at M2–8 took 7.98–8.40 versus 6.90–7.02 µs (4 runs). n6144 M32 took 7.16–7.22 µs; at M64 n6144/n2624 took 8.53–8.56/12.49–12.65 µs, 1.52–1.54/1.78–1.83× the same-run reads, so weight reading alone no longer sets time. n2624 M16/32 was not tested. Fourth-run raw: `CHECK bench_roofline_weight_stream.09 FAIL worst_t_over_stream=1.832 bf16_n6144_k2048={"stream_us":5.53,"stream_TBps":4.55,"t_us":{"2":5.94,"4":5.95,"8":6.04,"16":6.31,"32":7.16,"64":8.53},"TBps":{"2":4.23,"4":4.23,"8":4.16,"16":3.99,"32":3.51,"64":2.95},"max_t_over_stream":1.54,"max_over_min":1.43} bf16_n2624_k6144={"stream_us":6.9,"stream_TBps":4.67,"t_us":{"2":7.98,"4":8.07,"8":8.32,"64":12.65},"TBps":{"2":4.04,"4":4.0,"8":3.88,"64":2.55},"max_t_over_stream":1.83,"max_over_min":1.58} fp8_n3072_k1536={"stream_us":4.52,"stream_TBps":1.04,"t_us":{"2":4.36,"4":4.47,"8":4.25,"16":4.35,"32":4.5,"64":4.43},"TBps":{"2":1.08,"4":1.06,"8":1.11,"16":1.09,"32":1.05,"64":1.06},"max_t_over_stream":1.0,"max_over_min":1.06} fp8_n7168_k2048={"stream_us":4.64,"stream_TBps":3.16,"t_us":{"2":4.5,"4":4.52,"8":4.28,"16":4.52,"32":4.64,"64":6.27},"TBps":{"2":3.26,"4":3.25,"8":3.43,"16":3.25,"32":3.16,"64":2.34},"max_t_over_stream":1.35,"max_over_min":1.47}`; all four valid runs: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_roofline_weight_stream.log FAIL; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_roofline_weight_stream_rep2.log FAIL; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_roofline_weight_stream_rep3.log FAIL; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_roofline_weight_stream_cleanup.log FAIL | 10: The 25.17 MB weight (n6144,k2048) took 5.94–6.38 µs at M2–16, 3.95–4.23 TB/s (4 runs); M32 took 7.16–7.22 and M64 8.53–8.56 µs. Fourth-run raw: `CHECK bench_roofline_weight_stream.10 PASS weight_MB=25.17 us_m2=5.944 us_m4=5.948 us_m8=6.044 us_m16=6.312 TBps_m2=4.234 TBps_m4=4.231 TBps_m8=4.164 TBps_m16=3.987`; all four valid runs: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_roofline_weight_stream.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_roofline_weight_stream_rep2.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_roofline_weight_stream_rep3.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_roofline_weight_stream_cleanup.log PASS. No paired cleanup/original comparison or significance evidence assigns changes to cleanup or pure noise.
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, AKLR bench_cudagraph, median of 5 alternated rounds] plain-Triton bf16 test GEMM, N 6144, K 2048, `BLOCK_N` 32, `BLOCK_K` 256, 4 warps: one M tile against two that each re-read B, m = 32: 7.30 against 10.43 µs (1.428×), m = 64: 9.15 against 12.43 µs (1.358×), round ranges disjoint, max relative error 0.25 %; earlier run 7.33 against 10.45 µs (1.426×) and 9.09 against 12.50 µs (1.375×) (case triton/checks/bench_roofline_weight_stream.py (11); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_roofline_weight_stream_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_roofline_weight_stream.log) Four-run evidence (200/800/880/870, this case N=4; preceding values retain their stated historical scope and other-case counts): 11: On a synthetic plain-Triton bf16 GEMM (N6144,K2048), two M tiles took 1.426–1.428× one tile at M32 and 1.358–1.379× at M64 (4 runs, 5 alternating rounds/arm, disjoint ranges each run). This tests the combined tile-shape change, not isolated HBM traffic or AKLR kernels. Fourth-run raw: `CHECK bench_roofline_weight_stream.11 PASS M64={"t_A_us":9.03,"t_B_us":12.44,"B_over_A":1.379,"A_range":[9.01,9.06],"B_range":[12.41,12.46],"need":1.05,"max_rel_err":0.00243} M32={"t_A_us":7.32,"t_B_us":10.44,"B_over_A":1.426,"A_range":[7.29,7.32],"B_range":[10.43,10.47],"need":1.03,"max_rel_err":0.00262}`; all four valid runs: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_roofline_weight_stream.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_roofline_weight_stream_rep2.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_roofline_weight_stream_rep3.log PASS; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_roofline_weight_stream_cleanup.log PASS. No paired cleanup/original comparison or significance evidence assigns changes to cleanup or pure noise.
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, aklr validate A/B, 4 alternated rounds; ISA of the two tiers] `b88c` with B loaded as `[BLOCK_N, BLOCK_K]` and `tl.trans` against the committed direct tile: time ratio 1.172, 1.163, 1.121, 1.009 at m = 2, 4, 8, 64; 16 × 16 × 512 tile, 2 warps: 509 against 350 instructions (16 against 0 `ds_bpermute_b32`); 64 × 64 × 128 tile, 8 warps: 331 against 261 (6 against 0) (case triton/checks/lower_lds_operand_reads.py (04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_lds_operand_reads.log)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, XCC and CU ids read from the hardware in each workgroup] XCD = workgroup id % 8 with no mismatch on 1-D grids of 8, 100, 2048 and 4096 workgroups and on 6 multi-axis grids (3 launches each), and with offset 0 in 5 further launches of 4096, in each of 3 runs; 256 CUs (`multi_processor_count`, and 32 distinct CU ids on each of 8 XCCs) in each run; `b88c`'s m ≤ 16 grid read from its source: 164 (case triton/checks/bench_xcd_l2_reuse.py (01, 04, 06, 07, 07.grid, 11); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_xcd_l2_reuse_rep3.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_xcd_l2_reuse_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_xcd_l2_reuse.log)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, CUDA events over 20 back-to-back calls (no flush), median of 9, 5 alternated rounds] `b88c` at m = 8 with the remap `xcd * (GRID // 8) + local` on 164 tiles: columns 2560 – 2623 NaN in all 8 rows, bitwise equal elsewhere; broken / committed 11.93 / 11.94 µs (0.999×; round ranges 11.90 – 12.05 and 11.88 – 12.53), earlier run 12.59 / 12.69 µs (0.993×; 12.53 – 12.73 and 12.55 – 12.72) (case triton/checks/census_xcd_remap_permutation.py (18); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/550_census_xcd_remap_permutation_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_census_xcd_remap_permutation.log)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, AKLR bench_cudagraph (graph replay, flush per call), median of 5 alternated rounds] `f806`'s m = 64 kernel launched directly with the committed options (64 × 32 tiles, `BLOCK_K` 256, 192 workgroups, 4 warps), `KROT` 1 against 0, in 4 runs: A shared by all workgroups 8.562 – 8.609 against 9.136 – 9.181 µs, `KROT` 0 / 1 = 1.065 – 1.072 (per run 8.581 / 9.136, 8.609 / 9.166, 8.562 / 9.181, 8.562 / 9.155 µs); A private to each workgroup (A of 192 × 64 rows, each workgroup offset to its own 64 rows by its remapped tile index) 13.47 – 13.50 against 15.05 – 15.08 µs, 1.116 – 1.118 (per run 13.47 / 15.07, 13.50 / 15.07, 13.48 / 15.05, 13.49 / 15.08 µs); output not checked; check 07 reports FAIL in each run because it requires the private ratio within 1 % of 1 (case triton/checks/ab_bf16_decode_gemm_design.py (07); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_ab_bf16_decode_gemm_design_rep3.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_ab_bf16_decode_gemm_design_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/450_ab_bf16_decode_gemm_design_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_bf16_decode_gemm_design.log)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, flushed: AKLR bench_cudagraph; hot: one flush and one replay of a 50-call graph, then the next replay timed, median of 15; median of 3 rounds each] `f806` at m = 64 (k 2048), the source's tier (`BLOCK_K` 128, 8 warps, `KROT` 0) against the committed launch, one input, in 4 runs: flushed tier / committed 1.054 – 1.082 (committed 8.527 – 8.78 µs, tier ≈ 9.04 – 9.27 µs), hot 1.080 – 1.128 (committed 6.989 – 7.046 µs), hot ratio minus flushed ratio 0.021 – 0.052; per run flushed 1.054× (8.78 µs) and hot 1.107× (7.031 µs), 1.060× (8.527 µs) and 1.080× (7.046 µs), 1.082× (8.563 µs) and 1.128× (6.99 µs), 1.076× (8.594 µs) and 1.119× (6.989 µs); check 19 reports FAIL in each run because it requires the hot ratio at least 0.03 below the flushed one (case triton/checks/ab_bf16_decode_gemm_design.py (19); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_ab_bf16_decode_gemm_design_rep3.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_ab_bf16_decode_gemm_design_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/450_ab_bf16_decode_gemm_design_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_bf16_decode_gemm_design.log)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, repeated calls compared bitwise] `f806` at m = 2 – 64 and `b88c` at m = 2, 4, 8: one distinct output in 50 runs at each of the 9 Points, inside the accuracy bound, 0 atomics in the sources; a test GEMM with 6 fp32 partial planes [6, 64, 2624] summed in split order by a second kernel: one output in 100 runs, bit-equal to the sequential sum (reverse order differs in 90,596 values), float-atomic K split 100 distinct outputs in 100 runs; shipped `b88c` at m = 64: one output in 50 runs; `aklr validate --stages audit,bench` of `b88c`: audit passed with 0 findings, no bench finding other than speed, m = 64 timed (case triton/checks/num_bitwise_determinism.py (07, 09, 10, 11); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_bitwise_determinism_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_num_bitwise_determinism.log)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR bench_cudagraph, median of 3; Kineto trace of a graph replay for the kernel list] `b88c` at m = 64: whole call 12.57 µs, its `_gemm_kernel` alone 8.374 µs, difference 4.2 µs; the replay runs `_gemm_kernel` and `_reduce_kernel` at m = 64, `_gemm_kernel` only at m = 2 (case triton/checks/bench_launch_cost.py (23, 27, 32); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_launch_cost.log)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled-kernel census, offline build of the kernel (check 19 reports FAIL by design: it looks for the K loop and finds none)] `f806`'s `_gemm_xcd_kernel` at m = 32 (`BLOCK_M` 32, `BLOCK_N` 32, `BLOCK_K` 64, k 2048, 4 warps) built with `UNROLL` 4 and 1: 64 `v_mfma` in each, no backward branch in either (case triton/checks/rt_census_compiled_kernels.py (19); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log)
- E11 [2026-11, gfx950, N=1 complete GPU run, one Point (64,2624,6144), bf16, preallocated cold-cache device graphs; 1 GiB flush outside each event window] case `triton/checks/d_bf16_split_whole_call.py`, checks `.accuracy_determinism`, `.whole_vs_partial`, `.confirm_s6_vs_s1/2/3/4/8/12`, `.own_best_split_comparison`, `.scope`; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_d_bf16_split_whole_call.log (lines 1–34: source/contract/compile; 34–160: actual JIT/numeric checks for 3 seeds × 21 configurations and 100 bitwise repeats; 214: split-6 whole-minus-partial estimate; 219–234: completed cost/confirmation phases). 33 CPU and 165 GPU checks passed, 0 skips; `CASE PASS` means measurement completed, not the historical claims held. Confirmation uses 99% family-corrected paired-log Student-t CIs, the larger of local/global A/A floors plus the AB/BA order gap; split 4 remains unresolved. CPU checks and within-run samples are not independent full runs. The log does not separately record device model, clocks, load or package versions; do not infer complete environment provenance from the gfx950 gate or absence of a version warning.
- S1 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:26 `_xcd_remap`: `per` tiles to the first `tall` XCDs, `per - 1` to the rest (to :35); a permutation with one run per XCD for every grid of 1 – 4096 (164: `per` 21, `tall` 4; 246: `per` 31, `tall` 6), case triton/checks/census_xcd_remap_permutation.py (16)
- S2 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:62 `kphase = ((raw // NXCD) * KROT) % n_kb`; `b_offs` as `[BLOCK_K, BLOCK_N]` (:68); `kb = ((i + kphase) % n_kb) * BLOCK_K` (:71); B loaded with `.cg` and `tl.dot(a, b, acc=acc)` (:76-77)
- S3 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:100 m ≤ 16: `n // 16` tiles (:101), `BLOCK_M=16`, `BLOCK_N=16`, `BLOCK_K=512`, `SPLIT_K=1` (:113-116), `KROT=1` (:120); m = 64: 64 × 64 tiles, `BLOCK_K=128`, split 6, `KROT=0`, 8 warps (:128-149), then the reduce (:154)
- S4 aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:145 m ≤ 32: `BLOCK_M=32`, `BLOCK_K=64`, `UNROLL=4`, `KROT=1`, 4 warps, `schedule_hint="memory-bound-attention"` (:156-168); m = 64: `BLOCK_M=64`, `BLOCK_K=256`, `KROT=1`, 4 warps (:182-190); m ≤ 16: `BLOCK_M=16` (:133), no `matrix_instr_nonkdim` (:139-141), remap `(pid % NXCD) * (N_TILES // NXCD) + (pid // NXCD)` (:45); `kphase = ((pid // NXCD) * KROT) & (n_kb - 1)` (:90), the `CHUNK` remap (:93); `b_offs` (:99), `kb = ((i + kphase) & (n_kb - 1)) * BLOCK_K` (:102), `.cg` and `tl.dot` (:107-108)
- S5 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:104 `schedule_hint`: "Experimental; right now no effect", the field (:105) the only mention of `schedule_hint` in Triton 3.8.0's Python; `f806`'s m = 32 AMDGCN byte-identical with and without the hint, case triton/checks/lower_gluon_ignores_options.py (17)
- S6 local:/opt/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py:393 Triton 3.7.0 (3.7.0+gitb4e20bbe.rocm7.14.0a20260608): `memory-bound-attention` in `schedule_hint` adds `amdgpu-sched-strategy` `iterative-ilp` (:394); `f806`'s m = 32 LLIR differs from the plain build only in that function attribute, TTGIR identical, AMDGCN differs; on 3.8.0 `llvm_fn_attrs` gives the same attribute, 3.7.0 drops it, and AKLR's `unmet_requirement({"triton": ">=3.8"})` rejects 3.7.0, case triton/checks/lower_gluon_ignores_options.py (16, 18)
- S7 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:87 `_reduce_kernel` adds the split planes in order, `tl.static_range(1, SPLIT_K)` (:91-92)
- S8 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json:61 `workloads`: Points m = 2, 4, 8 and 64 (:64, :71, :78, :85), none at 1 or 32; `f806`'s (`gemm_n6144_k2048/f806ad2b086d757d.json:61`) m = 2, 4, 8, 16, 32, 64
- S9 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/census_xcd_remap_permutation.py:661 check 17 on `f806`'s remaps: at 192 tiles the `CHUNK` remap gives 3 runs of 8 per XCD and the m ≤ 16 remap one run of 24, both permutations; at 164 tiles the m ≤ 16 remap computes 20, 40, 60, 80 twice and misses 160 – 163, the `CHUNK` remap misses 12 tiles and reaches tile 187
- S10 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/ab_bf16_decode_gemm_design.py:406 check 16: the m = 1 tiles (`BLOCK_M` 16, `M` passed as a runtime i32, `matrix_instr_nonkdim` 16 on both) compile to 64 (`f806`) and 48 (`b88c`) `v_mfma_f32_16x16x32_bf16` and 0 `v_dot2`, the same in each of 4 runs (Triton 3.8.0, gfx950; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_ab_bf16_decode_gemm_design_rep3.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_ab_bf16_decode_gemm_design_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/450_ab_bf16_decode_gemm_design_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_bf16_decode_gemm_design.log)
- S11 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/ab_bf16_decode_gemm_design.py:369 check 15: AKLR's FlyDSL `gemm_n6144_k2048` takes `_dot_gemm` at m = 1 (`DOT_MAX_M` default 2) and `gemm_n2624_k6144` `_vector_gemv` (`VECTOR_GEMV` default 1 at m = 1); both bodies call `fdot2_f32_bf16` and contain no MFMA (a source census, PASS in each of 4 runs, logs as S10)
- S12 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/census_family_structure.py:1020 check 07 over the 10 GEMM launches (reduce excluded): `waves_per_eu` 2 and `num_stages` 3 on all; `matrix_instr_nonkdim` 16 on `b88c` m = 2, 4, 8, 64 and `f806` m = 32, 64, none on `f806` m = 2 – 16 (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_census_family_structure.log)
- S13 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/census_family_table.py:378 checks 14 – 16: the Triton `gemm@1.0` schemas are exactly `b88c` (n 2624, k 6144) and `f806` (n 6144, k 2048), `trans_b`, no bias, bf16, `@triton.jit` only; weights 32,243,712 and 25,165,824 bytes (:381); both `determinism: bitwise` (`b88c8d5f28d86777.json:107`, `f806ad2b086d757d.json:121`), and validate's bitwise rule is every run's output byte-identical to the first's (:391)
- S14 harness/skills/aklr-kb/triton/gemm/decode_gemm.md:1 page-local Evidence retains the source and measurement conditions.
- S15 local:/tmp/aklr_triton/harness/skills/aklr-kb/triton/checks/rt_census_compiled_kernels.py:1430 check 19 (fails by design: it looks for the K loop): `f806`'s `_gemm_xcd_kernel` at m = 32 (`BLOCK_M` 32, `BLOCK_N` 32, `BLOCK_K` 64, k 2048, 4 warps) built with `UNROLL` 4 and 1: 64 `v_mfma` in each, no backward branch in either (Triton 3.8.0, gfx950; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log)
