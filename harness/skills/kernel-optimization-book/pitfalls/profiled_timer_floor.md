---
title: "Timing a kernel under profiling puts a floor of about 4 µs under short kernels"
kind: pitfall
scope: [three timers measured on MI355X, a Kineto graph timer with queue profiling forced, CUDA-event timing of back-to-back graph replays in a process that forces queue profiling, and one kernel tracer built on rocprofiler-sdk, any kernel or baseline shorter than about 4.5 µs]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=3
---

# The profiled timer's floor

## Symptom

Timed in a process that profiles the GPU queue, a kernel and its baseline that differ a lot read about the same,
both between 4.0 and 4.6 µs. A 1.7 µs bmm against a 2.6 µs baseline read 4.09 against 4.10 µs by CUDA events in a
process that forced queue profiling, and was judged not faster *(field, E2)*. An empty kernel (a one-element
`zero_`) read 4.19 µs on a Kineto graph timer and 1.70 µs replayed back to back in a graph without profiling
*(field, E1)*. A kernel several times longer than the floor is not held to it.

## Root cause

With kernel times read from Kineto and `GPU_FORCE_QUEUE_PROFILING=1` set in the measuring process, the empty kernel
read 4.19 µs *(field, E1)*. CUDA-event timing of back-to-back replays in a process that still forced queue
profiling held the bmm at 4.1–4.3 µs, one kernel per call *(field, E3)*; with queue profiling not forced, the same
bmm read 1.70 against 2.59 µs *(field, E2)*. The 4 µs is those rulers' reading, not a cost a deployment pays per
kernel.
One kernel tracer built on rocprofiler-sdk, which does not force queue profiling, still has a floor: spin kernels of
1.8–3.5 µs back to back read 4.40–4.56 µs on it, and from about 5.4 µs up within 0.08 µs of their back-to-back time
*(field, E4)*. The bmm against aiter's baseline read 1.13× on it where back-to-back timing gives 1.50×
*(field, E5)*.

## How to detect

- A baseline or a candidate that reads 4–5 µs on one of these timers may be at that ruler's floor.
- Time an empty kernel on the same timer beside the baseline and the candidate: a side that reads about the same as
  the empty kernel is at the ruler's floor, and the candidate is the one a short kernel puts there first.
- Time the call in a process without `GPU_FORCE_QUEUE_PROFILING`: replay it back to back in one CUDA graph, with
  CUDA events, the call count taken from a short graph rather than an eager call. If both sides drop well below the
  profiled reading, the gap is the ruler's.

## Fix or workaround

- Before deciding that a size has no headroom, time the candidate and the baseline in pairs on the ruler that will
  judge them, with an empty kernel as the floor reference.
- Report a gain with the ruler that measured it: a back-to-back gain is not a gain on the profiled timer.

## Scope

The three timers measured here, on MI355X: a Kineto graph timer with queue profiling forced; CUDA-event timing of
back-to-back graph replays in a process that forces queue profiling; one kernel tracer built on rocprofiler-sdk,
without it. Another profiler-based timer may have another floor or none: time an empty kernel on it. Kernels and
baselines longer than about 5 µs read as they do back to back. What a deployment's graph pays per call is the
back-to-back time, not the profiled reading.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3, Kineto graph timer with queue profiling forced vs back-to-back graph replay with CUDA events] a Kimi-K3 top-k start kernel 4.27 vs 3.44 µs, aiter top-k 12.05 vs 10.54 µs, one-element `zero_` 4.19 vs 1.70 µs, at n_tokens=8
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=3, CUDA-event timing of back-to-back graph replays] a bmm of N 128 × K 512 at m=8: 4.09 vs 4.10 µs with the measuring process forcing queue profiling; 1.70 vs 2.59 µs, ratio lower bound 1.480, with it not forcing it
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=3, back-to-back timing, kernels of the timed graph from Kineto] one bmm kernel per call, the readings still 4.1-4.3 µs
- E4 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, Triton 3.8.0, N=3 GPUs, per kernel 8 back-to-back and 80 tracer readings, one kernel tracer built on rocprofiler-sdk over graph replays against back-to-back graph replay with CUDA events in a process without queue profiling] a 256-workgroup spin kernel: 1.8–3.5 µs back to back read 4.40–4.56 µs; from about 5.4 µs back to back up to 21.6 µs, within 0.08 µs
- E5 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, FlyDSL 0.3.2, aiter dbd8bf5, N=3 processes × 5 readings, the tracer of E4 against back-to-back graph replay with CUDA events in a clean process] a bmm of N 128 × K 512 at batch 12, m = 8: candidate 1.76 → 4.30 µs, aiter CK baseline 2.64 → 4.85 µs, ratio 1.50 → 1.13; empty kernel 1.61 → 4.42 µs
