---
title: "Attention — measuring a change: steady state, rulers, zero points, FLOP accounting, host-bound shapes"
kind: method
lever: attn_measurement
ops: [attention, mla]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [many query rows, dQ dK dV, few query rows, launch-bound, kernel-A/B]
flydsl: 0.3.2
---

# Measuring an attention kernel change

> The shared protocol — ruler choice, arm order, warm-up and first-arm bias, noise floors, statistics, closing a
> series of changes — is in `../gemm/measurement.md`. This card is what attention adds: small
> workloads, multi-cell scores with guards, workspace-heavy dQ/dK/dV cells, FLOP conventions, and launch-bound
> kernels with few query rows. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed
> under Evidence at the end. Use them for direction and order of magnitude.

## Route here when
- Setting up a benchmark or scorer for an attention kernel, or reading one you inherited.
- A delta is under ~1%, a cell is noisy, or two rulers disagree.
- Reporting TF/s for a causal, windowed, fp8 or sparse kernel.

## 1. Hot steady state
- **Never sleep between repetitions on a small workload.**
  - *(field, E1)* A B=1 backward kernel read 1024 TF/s cold, 727–969 with `sleep(0.3)` between calls, and a
    different value again under continuous load: a 40% swing that buried a 5–14% gap.
- Warm with back-to-back calls and no sleep until the clock settles (at least 90 s from an idle card,
  `../gemm/measurement.md` §3), then take the median of ≥7 back-to-back repetitions.
- Decide on a cell large enough to hold the clock steady, and isolate sweeps (§4).

## 2. The bench mirrors the deployment
- **Each arm at its own best configuration** (`../gemm/measurement.md` §7). The attention form of the
  trap: a bench that calls the kernel builder with its defaults while the deployed launcher passes other values
  (`block_kv`, `waves_per_eu`, LSE folding). *(field, E3)* The two rulers differed by 1.3–1.4% on the same code,
  and the bench was optimising a configuration that did not ship.
- Make the builder's defaults equal the deployed values instead of relying on every bench to pass them. When a
  builder signature changes, update every bench that calls the builder directly.
- A ruler fix moves the score; report it separately from kernel speed-ups so history stays comparable.

## 3. Zero points and guards
A multi-cell score usually carries guard cells: cells the candidate does not change, scored as `min(ratio, 1)`.
- **The guard geometric mean is the session's zero point; use it.** *(field, E4)* Seven runs of one scorer on
  byte-identical guard code read +0.52 / +0.35 / +0.87 / −0.26 / −0.94 / −0.20 / +0.15%: a 1.8% swing, same sign
  across three independent shapes, so a per-session slot bias that also hits the target cells. Decide on
  `raw / guard_geomean`; it reproduced an independent re-measurement within 0.26% where the raw score ranked two
  configurations backwards.
- The corrected value carries about ±0.5% *(field, E5)* of its own noise: a difference under ~0.8% needs two independent
  sessions agreeing in sign before you keep it.
- **A hard cap is a one-sided tax.** *(field, E6)* One attention guard cell spread 3.2% on identical code; capped at
  1.0 it could only subtract, and caught no real regression in 17 rounds, in which no edit could reach it (every
  edit carried a `D == 64` term). Set the cap tolerance at or above the measured spread of the guard cell.
- **Check for two modes before choosing an estimator.** *(field, E7)* A 0.57 ms sliding-window cell, 24 processes on one
  binary: 0.567–0.574 or 0.634–0.642 ms, nothing between (the high mode 11.4% slower). For a bimodal cell use the
  minimum over ≥12 processes; a mean or median moves with the fraction of runs that land high, and a guard in the high
  mode costs the score ~1.5% by itself.

## 4. Isolation and arm order
- **One process per workspace-heavy cell.** Each dQ/dK/dV cell allocates gigabytes of dQ workspace, and allocator
  fragmentation sets the next cell's time. *(field, E8, E9)* The same cell read 11.23 and 6.64 ms depending on
  what ran before it, and `empty_cache()` did not help; forking a process per cell cut the cell swing from up to 70%
  to 0.29–2.50%.
- Arm order and when a palindrome applies: `../gemm/measurement.md` §2 (palindromes only for two
  module instances interleaved in one process; one process per arm runs A B).
- **Ratio rulers between two arms that share a body do not credit body changes.** For `bf16_ms / fp8_ms` with one
  shared backward body, an optimisation of that body speeds up both arms and the score does not move. Only
  changes that act on one arm count; fork the body per arm (separate cache-key entries) when that is intended.
- **Expect an in-process kernel ruler to read larger than the end-to-end score.** *(field, E10)* +1.3% in-process
  against +0.8% on the full bench for one forward change; the in-process ruler omits fixed per-call costs.

## 5. FLOP and byte accounting
- **The numerator is the mathematically required work, not what the hardware issues.** O (QK and P·V):
  `4·B·Hq·Sq·Skv·D × attended fraction`. dQ, dK and dV: 2.5× that (5 GEMMs). For square causal,
  `10·B·Hq·S²·D·(1 − (S−1)/(2S))`; for a `(W, 0)` window the attended fraction is about `(W+1)/S`. Rectangular
  and windowed shapes need the real attended fraction, not the square-causal factor.
- Work an implementation issues beyond the convention is not credited: a deterministic two-kernel backward
  issues 7 GEMM-equivalents for the convention's 5 (~29% of MACs). Use the gap as the redundancy
  ratio of `workflow.md` Step 3, not as throughput.
- **Lower precision shows up as a smaller denominator with the same numerator.** Changing the numerator for an fp8
  arm is cheating. Produce every quantised byte inside the timed region: *(field, E11)* with Q/K quantisation moved
  into the clock, an fp8 backward that had "won" three times read 0.58× of bf16.
- **A gather-bound kernel is priced in bytes, not FLOPs.** Count logical bytes (rows × row width), not HBM traffic
  after cache hits, and compare against measured bandwidth for the same access pattern, not against FLOP peak or a
  stream copy; a same-pattern reference that the kernel beats is not a ceiling.

## 6. Small shapes, the host and the launcher
- **Every placeholder tensor built in the launcher is a kernel launch.** *(field, E14)* A `torch.zeros(1)` placeholder
  that the dense path did not read cost a 4.6–5.6 µs fill kernel per backward call, ~0.4% of a 1.22 ms call.
  Look for `fill_`/`zero_` entries that match no kernel of yours in the profiler table.
- **For kernels of 20–200 µs, a `synchronize()`-bracketed wall time is not a kernel ruler.** *(field, E15)* ~27 µs
  of constant host time per call turned a 16.8% kernel gap between FlyDSL and Triton into 38% on wall, and the
  wall ruler produced three false scaling conclusions. Rulers for launch-bound kernels, worst to best:
  1. `synchronize()`-bracketed wall time;
  2. event timing without a graph — measures the Python launch rate (a 21 µs kernel read 33.9 µs and
     two arms read 0.06% apart whatever their GPU work) *(field, E16)*;
  3. the same configuration repeated across processes (12.9% range) *(field, E17)*;
  4. ~20 calls captured in one HIP graph, timed with events, arms interleaved ABBA in one process, after
     75 s of warm-up (under 0.3% spread across repetitions) *(field, E18)*; or kernel time from
     `rocprofv3 --kernel-trace`.

In FlyDSL: `../flydsl/attention.md` § measurement §6.

## 7. Cache hygiene
- Clear or isolate the JIT and autotune cache when comparing builds: a cache carried across code versions skews
  the readings *(field, E19)*.
- The first pass in a fresh cache directory includes compilation and any autotuning; discard it.

In FlyDSL: `../flydsl/attention.md` § measurement §7.

## 8. Coverage traps
- **A single scored shape gets traded against the others, silently.** *(field, E20)* With only full causal
  scored, one round bought +1.8% there and −12% on the same shape with a sliding window; at a 3:1 window to
  full-causal deployment mix that round lost. Close every series of changes on the deployment mix: all shapes × all mask
  modes, weighted by the mix. Before gating a regime-specific change, align the other levers for that regime
  (`workflow.md` § Rules that save rounds); gate on the trait only if it still loses there.
- **Weight cells by their share of step time.** *(field, E21)* Full causal was 92% of a backward step and the window
  cells 8%; an equal-weight geometric mean let a 12% cut in window time lift the score while step time did not move.
- **A shared file is not a shared gain.** *(field, E22)* 40 rounds on the D=128 path gained 19.3%; the D=64 path in the
  same file went from 997.2 to 990.0 TF/s, because four `D == 128` gates kept the work out. Measure every unscored
  variant before and after; a zero delta means look for a gate (opening one of them, the gate on
  the whole a16 scheme, with two D=64 re-prices of its companion knobs, was worth 6.36% in one round).
- **Audit the dispatch per cell before claiming "no regression".** Wrap the dispatcher and log the path each cell
  takes. A window covering the whole sequence is normalised to full causal; a framework's default backend may not
  be your kernel, so the probe never reaches the change.
  - A command that pins the backend and one that resolves it automatically are different rulers.

## 9. Helper kernels in concurrent traces
- **Do not divide a helper kernel's trace duration by its bytes when it runs beside the main body.** *(field, E24)* A
  reduction recorded as 670 MB in 317 µs (2.1 TB/s, "4× from peak") ran at 5.4–6.2 TB/s when launched alone, against
  a 5.95–6.2 TB/s stream floor on that capped card; the 317 µs was its time competing for CUs.
- Read per-call min and max first. *(field, E25)* The same delta kernel ran 28.1 µs serially (7.2 TB/s, already above the
  read stream floor, so not slow) and 59.8–1527 µs beside an occupancy-1 body: the inflation was CU starvation, not
  extra work.
- **Kernel-time share is not wall share.** *(field, E26)* A helper was 14.95% of the summed kernel time of a B=4
  backward, while all exposed helper time (that helper, lset and the tail fold) was 1.9% of wall at B=4 and 3.9% at
  B=1 on the timeline. Rank helper-kernel work by exposed time; fusion pricing is in `gradients.md` §7.

## Related
`../gemm/measurement.md` · `attribution.md` · `correctness.md` ·
`split_kv.md` §8 · `../flydsl/debug-flydsl-kernel.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the run's where the record dates it; for a note kept outside the run the line says which date
it is. Every workload below is an attention kernel or one of its helper kernels.

- E1 [2026-07, gfx950, FlyDSL ?, N=?, B=1 bench `_bench_square.py`: cold start, `sleep(0.3)` between calls, continuous load] the same kernel read 1024 TF/s cold, 727–969 half-cold with the sleep, and a value between them under continuous load: a 40% spread that buried a 5–14% gap; the card names no repeat count and no run record (hd64 deterministic attention backward at B=1, Hq=128, Hkv=16, D=64, full causal; dated 2026-07-21 in the card; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/02-measurement-noise.md:99; bench defaults local:/workspace/code/gpt_oss_docker/sync/meta-attn/meta_aiter_attn/flydsl/_bench_square.py:14)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, campaign score: full-backward conv TF/s from `_bench_campaign_bwd16384.py`] making the scoring bench build dkdv with the deployed `fold_lse` default moved the unchanged kernels from 856.8 to 868.5 TF/s (+1.37%), kept as a ruler correction; myskill gives the bench-against-deployment gap (dq `block_kv`/`waves_per_eu`, dkdv `fold_lse`) as 1.4% (hd64 attention backward, Primus-Turbo `flash_attn_bwd.py`, Sq=Skv=16384, B=1, hw-exp path; campaign 20260726_152547 round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:259; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:184)
- E4 [2026-09, gfx950, FlyDSL 0.2.4, N=7, `_bench_attn_bwd2.py`: same-process ABBA against a frozen copy of the kernel, guard cells scored `min(ratio, 1)`] seven benches on byte-identical guard code read guard geomeans of +0.52 / +0.35 / +0.87 / −0.26 / −0.94 / −0.20 / +0.15% (1.8% swing), the three guard shapes shifting in the same direction; corrected values 1.0183 and 1.0117 against the orchestrator's independent re-measurements in other sessions, 1.0195 and 1.0143 (within 0.26%), while the raw scores 1.0158 / 1.0170 ranked the two configurations backwards (gpt-oss D=64 attention backward, Hq=64, Hkv=8, S=8192, causal, target cells B=4/2/1, guards one D=64 and two D=128 shapes; campaign 20260911_114527 round 7, filed in myskill as r19; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/goal.md:23; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:368; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:2309)
- E5 [2026-09, gfx950, FlyDSL 0.2.4, N=?, as E4] the corrected value carries about ±0.5% of its own noise (the three guards' resid 0.4–2.1%), so the keep threshold became two independent sessions agreeing in sign for any difference under 0.8% (same campaign and round; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-07/agent.txt:69; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:2312)
- E6 [2026-08, gfx950, FlyDSL ?, N=8, campaign score with the guard legs capped at 1.0] the `l70b` guard leg spanned 2.6087–2.6917 (3.2%) across eight scored runs on a byte-identical D=128 code path; in the 17 rounds of the campaign it guarded it caught no real regression, every edit there carrying a `D == 64` term (attention backward, guard `l70b` = B=1, Hq=64, Hkv=8, S=8192, D=128, causal, in a gpt-oss D=64 backward campaign; campaigns 20260829_150647 and 20260830_154052; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/goal.md:250; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260830_154052/memory.md:63)
- E7 [2026-08, gfx950, FlyDSL ?, N=24, 24 independent processes on one binary] the 0.57 ms windowed cell split into a low mode at 0.567–0.574 ms and a high mode at 0.634–0.642 ms with no reading between (+11.4% between the modes); the campaign adopted min-of-N with N ≥ 12 for it; a guard in the high mode costs the score about 1.5% (gpt-oss-20b attention backward cell `gptoss_swa`: B=4, Sq=Skv=8192, Hq=64, Hkv=8, D=64, window 128; campaign 20260819_065627 round 5; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/rounds/round-05/agent.txt:11; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260819_065627/goal.md:411; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/02-measurement-noise.md:227)
- E8 [2026-08, gfx950, FlyDSL ?, N=3, the campaign bench: eight cells, min of 40 per cell] with the cells sharing one process `d128_hq64_b2` read 11.23 and 6.64 ms on identical code and `gptoss_swa` 0.551 ms alone against 0.590 ms after `gptoss_full`, and `empty_cache()` did not undo it; with one process per cell the per-cell spread over three runs was 0.29–2.50% (gpt-oss attention backward cells, D=128 Hq=64 B=2 S=8192 and the D=64 windowed cell; campaign 20260818_155723; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:126)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, ?] with the cells sharing one process, the allocator state a cell inherited moved its time by up to 70%; one process per cell took the per-cell spread to 0.29–2.50% over three runs (gpt-oss-20b attention backward cells of the same campaign; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:125; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:128)
- E10 [2026-09, gfx950, FlyDSL ?, N=3–5, in-process lab ruler with a same-binary A-vs-A control in each of 5 sessions, against the campaign bench run 3 times before and 3 times after] one change read +1.52 / +1.28 / +1.21 / +1.30 / +1.45% in-process and +0.82% on the bench (`cand_ms` 2.04458 → 2.02786) (gpt-oss D=64 attention forward, `K_HEAD` 3→2 with 10 scheduling anchors; campaign 20260924_001030 round 13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260924_001030/rounds/round-13/agent.txt:44; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/15-attention-optimize.md:381)
- E11 [2026-09, gfx950, FlyDSL ?, N=?, `geomean(bf16_ms / fp8_ms)` over two cells, both arms alternated in one process from bf16 inputs, every E4M3 byte produced inside the timed region] zero point 0.5804 (5.2008 / 8.8679 = 0.5865 and 2.5842 / 4.4999 = 0.5743 ms); on the earlier ruler, with the Q/K quantisation outside the clock, three campaigns had netted about +0.7% (gpt-oss D=64 attention backward, fp8 against bf16, cells 4/64/8/8192/64 and 2/64/8/8192/64; campaign 20260902_093653; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260902_093653/goal.md:41; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260902_093653/memory.md:15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260902_093653/memory.md:118)
- E14 [2026-09, gfx950, FlyDSL 0.2.4, N=?, kernel trace and torch profiler for the dispatch; wall share from the campaign's paired ruler] a `torch.zeros(1)` placeholder cost a 4.6–5.6 µs `vectorized_elementwise_kernel` per backward call (profiler `aten::zeros/zero_/fill_` 2.0 µs); removing it was worth 0.40% of wall at B=1 (1.22 ms) and 0.19% at B=2 (gpt-oss D=64 attention backward, Hq=64, Hkv=8, S=8192; campaign 20260911_114527 round 12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:431; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/13-meta-hd64-flash-bwd.md:2614)
- E15 [2026-09, gfx950, FlyDSL ?, N=?, `synchronize()`-bracketed `perf_counter` wall against GPU time] on decode a 16.8% gap in GPU time between FlyDSL and Triton read 38% on wall, through ~27 µs of constant host time per call; three false scaling conclusions came from the wall readings (sparse-MLA decode, FlyDSL against Triton, GLM-5.2 TP4: H=16, topk=2048, fp8 KV; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_130431/goal.md:33; probe docstrings tools/_bench_sparse_mla.py:21 and tools/_ab_decode_variants.py:4 in local:/workspace/code/gpt_oss_docker/scratch/archive/inference2_cleanup_20261022.tar.gz)
- E16 [2026-09, gfx950, FlyDSL ?, N=?, CUDA-event timing without a graph] a 21 µs kernel read 33.9 µs and two arms read 0.06% apart whatever their GPU work (as E15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_130431/goal.md:35; tools/_ab_decode_variants.py:39 in local:/workspace/code/gpt_oss_docker/scratch/archive/inference2_cleanup_20261022.tar.gz)
- E17 [2026-09, gfx950, FlyDSL ?, N=?, the same configuration repeated across processes] range 12.9% (as E15; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_130431/goal.md:37; tools/_ab_decode_variants.py:8 in local:/workspace/code/gpt_oss_docker/scratch/archive/inference2_cleanup_20261022.tar.gz)
- E18 [2026-09, gfx950, FlyDSL ?, N=?, 20 calls captured in one HIP graph, event timing, ABBA in one process, 75 s warm-up] spread under 0.3 percentage points ("paired ABBA, HIP-graph timed, <0.3pt spread") on the sparse-MLA decode producer, seq 48–96, topk 2048 (aiter f01257989; probe defaults tools/_ab_decode_variants.py:123 in local:/workspace/code/gpt_oss_docker/scratch/archive/inference2_cleanup_20261022.tar.gz)
- E19 [2026-08, gfx950, FlyDSL ?, N=?, ?] clearing the FlyDSL JIT/autotune cache carried across code versions changed the readings; the campaign's own notes record the effect without a figure (gpt-oss-20b attention backward cells; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:140)
- E20 [2026-07, gfx950, FlyDSL ?, N=?, ?] the GQA sharer-merge round of a campaign scored only on full causal bought full causal +1.8% and cost the same shape with a window −12%; the shipping mix is 3 windowed : 1 full causal, on which that round lost (hd64 attention forward; dated 2026-07-30 in the card; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/15-attention-optimize.md:175; the 3 : 1 mix in Primus-Turbo e93fc062)
- E21 [2026-08, gfx950, FlyDSL ?, N=?, `step_ms = 12 × (gptoss_full + gptoss_swa)`, min of 40 per cell, each cell in its own process] full causal 72.15 ms (92%) and the windowed cell 6.61 ms (8%) of a 78.76 ms step; the previous run's equal-weight geomean read 1.059 for a step that had not moved (window −12% time, full +1.3%, sum +0.1%) (gpt-oss-20b attention backward, B=4, Sq=Skv=8192, Hq=64, Hkv=8, D=64, window 128; campaign 20260818_155723; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:13; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260818_155723/memory.md:44)
- E22 [2026-08, gfx950, FlyDSL ?, N=?, the campaign bench, each cell in its own process] a 40-round campaign took the D=128 path of `flash_attn_bwd.py` from 857 to 1022 TF/s (+19.30%) while `d64_gptoss_b4` went from 997.2 to 990.0 TF/s; opening the `D == 128` gate of the a16 scheme to D=64 (with two D=64 re-prices of its companion knobs) then scored +6.36% in one round (gpt-oss attention backward, `d64_gptoss_b4` = B=4, Hq=64, Hkv=8, S=8192, D=64; campaign 20260829_150647, round 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/memory.md:58; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260829_150647/memory.md:169)
- E24 [2026-09, gfx950, FlyDSL 0.2.4, N=?, kernel trace of the backward (concurrent) against the fold kernel launched alone] the 4-slot dK/dV fold recorded 670 MB in 317 µs = 2.1 TB/s in the trace; launched alone it ran 5.40–5.55 TB/s (deployed shape) and 6.15–6.16 TB/s (widest tile), against a measured HBM stream floor of 5.95–6.2 TB/s (gpt-oss D=64 attention backward, `_reduce_dkdv_slots`, B=4; campaign 20260911_114527 rounds 11–12; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:419; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:441)
- E25 [2026-09, gfx950, FlyDSL 0.2.4, N=?, rocprofv3 `--kernel-trace --stats`, per-call min and max] the delta (odo) kernel moved 203 MB in 28.1 µs run serially (7.2 TB/s) at B=1/2, and took 59.8 µs (min) to 1527 µs (max) per call beside the dkdv body at B=4, with the same bytes (gpt-oss D=64 attention backward; campaign 20260911_114527 round 9; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/rounds/round-09/agent.txt:17)
- E26 [2026-09, gfx950, FlyDSL 0.2.4, N=?, kernel trace: share of the summed kernel time against exposed time on the timeline] the odo kernel was 14.95% of the summed kernel time at B=4; all exposed helper time (odo, lset and the tail fold) was 3.9% of wall at B=1 (1.22 ms) and 1.9% at B=4 (gpt-oss D=64 attention backward; campaign 20260911_114527, the R8 table and the round-12 re-count; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/goal.md:62; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_114527/memory.md:442)
