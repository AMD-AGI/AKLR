---
title: "Profiled AKLR timing can mask differences between short kernels"
kind: pitfall
scope: [aklr validate and AKLR CI timing, measurement profiles 17 through 19, any kernel or baseline shorter than about 4.5 µs]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=3
sources:
  - "AMD-AGI/AKLR@a29581b2d:aklr/core/validation/stages/_bench/_execution/_runner.py:475"
---

# AKLR's dispatch floor

## Symptom

Under AKLR's timer, a kernel and its baseline that differ a lot read about the same, both between 4.0 and 4.6 µs,
on measurement profiles up to 18 and on profile 19 (§ Root cause). A 1.7 µs bmm against
a 2.6 µs baseline read 4.09 against 4.10 µs in `aklr validate` and was demoted as `not_faster`
*(field, E2)*. An empty kernel (a one-element `zero_`) read 4.19 µs on AKLR's graph timer and 1.70 µs replayed
back to back in a graph without profiling *(field, E1)*. A kernel several times longer than the floor is not held
to it.

## Root cause

Up to measurement profile 18, AKLR read the kernel times of graph replays from Kineto, and its bench runner set
`GPU_FORCE_QUEUE_PROFILING=1` in every runner process (S1). The empty kernel's 4.19 µs came from that Kineto graph
timer, called in a process of its own outside the runner *(field, E1)*. The bmm's 4.09 against 4.10 µs came from
validate already switched to CUDA-event timing of back-to-back replays with the runner still forcing queue
profiling, which held it at 4.1–4.3 µs, one kernel per call *(field, E3)*; with the runner not forcing it, the same
bmm read 1.70 against 2.59 µs *(field, E2)*. The 4 µs was those rulers' reading, not a cost a deployment pays per
kernel.
Current main (profile 19) reads kernel times from AKLR's own rocprofiler-sdk tracer, `aklr_kernel_tracer`, and no
longer forces queue profiling (`aklr/core/validation/stages/_measure.py`, `aklr/core/utils/testing.py`), and short
kernels still sit at a floor under it: spin kernels of 1.8–3.5 µs back to back read 4.40–4.56 µs, from about 5.4 µs
up within 0.08 µs of their back-to-back time *(field, E4)*. The bmm against aiter's baseline read 1.13× where
back-to-back timing gives 1.50× *(field, E5)*.

## How to detect

- Read `baseline.json` (forge_task) or a validate report: a baseline at 4–5 µs timed under profile 18 or earlier may
  have been at that ruler's floor.
- On current main, time an empty kernel through `bench_cudagraph` beside the baseline and the candidate: a side
  that reads about the same as the empty kernel is at the ruler's floor, and the candidate is the one a short kernel
  puts there first.
- Time the call outside AKLR, in a process without `GPU_FORCE_QUEUE_PROFILING`: replay it back to back in one CUDA
  graph, with CUDA events, the call count taken from a short graph rather than an eager call. If both sides drop
  well below AKLR's reading, the gap is the ruler's.

## Fix or workaround

- Before a campaign, recheck every baseline near 4–5 µs on current main: time the candidate and the baseline in
  pairs there, with an empty kernel as the floor reference, before deciding a Point has no headroom.
- Report an external-timer gain with that ruler, not as an AKLR speedup unless AKLR measured it.

## Scope

AKLR validate and CI from `AMD-AGI/AKLR@a29581b2d` (measurement profile 17) through profile 19 (current main), on
MI355X. Kernels and baselines longer than about 5 µs are timed as before. What a deployment's graph pays per call is
the back-to-back time, not AKLR's reading.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AKLR graph timer vs back-to-back graph replay with CUDA events] K3 top-k start kernel 4.27 vs 3.44 µs, aiter top-k 12.05 vs 10.54 µs, one-element `zero_` 4.19 vs 1.70 µs, at n_tokens=8
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=3, aklr validate] bmm_n128_k512 m=8: 4.09 vs 4.10 µs with the runner forcing queue profiling; 1.70 vs 2.59 µs, ratio lower bound 1.480, with the runner not forcing it
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, aklr validate with back-to-back timing, kernels of the timed graph from Kineto] one `bmm_n128_k512_tiled_0` per call, the readings still 4.1-4.3 µs
- E4 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, Triton 3.8.0, N=3 GPUs, per kernel 8 back-to-back and 80 profile-19 readings, main's `bench_cudagraph` (profile 19) against back-to-back graph replay with CUDA events in a process without queue profiling] a 256-workgroup spin kernel: 1.8–3.5 µs back to back read 4.40–4.56 µs; from about 5.4 µs back to back up to 21.6 µs, within 0.08 µs
- E5 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, FlyDSL 0.3.2, aiter dbd8bf5, N=3 processes × 5 readings, main's `bench_cudagraph` (profile 19) against back-to-back graph replay with CUDA events in a clean process] bmm_n128_k512 at batch 12, m = 8: candidate 1.76 → 4.30 µs, aiter CK baseline 2.64 → 4.85 µs, ratio 1.50 → 1.13; empty kernel 1.61 → 4.42 µs
- S1 AMD-AGI/AKLR@a29581b2d:aklr/core/validation/stages/_bench/_execution/_runner.py:475 `GPU_FORCE_QUEUE_PROFILING="1"`
