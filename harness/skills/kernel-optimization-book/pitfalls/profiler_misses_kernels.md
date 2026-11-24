---
title: "Kineto can drop the kernels a HIP graph launches; a timer that reads kernel times from it finds no records at all, a serving profile a partial trace"
kind: pitfall
scope: [torch profiler (Kineto) traces of HIP-graph replays, the per-kernel CUDA-graph timer read through Kineto and harnesses that call it, sglang /start_profile with CUDA graphs on]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
---

# Kineto misses graph-launched kernels

## Symptom

Two forms, both silent about which kernels went missing.

- **In the per-kernel CUDA-graph timer read through Kineto (`../gemm/measurement.md` §1): no records at all.** `RuntimeError: Kineto recorded no kernels
  for a single call`, raised while sizing a measurement (S1). A harness that does not catch it loses the arm's measurement:
  in one tuning run on a bf16 GEMM, 3 of 14 paired timings were lost this way, two on the
  candidate arm and one on the unchanged incumbent, and all three were recorded as performance reverts
  *(field, E1)*. In another, a change was reverted after two correctness invocations aborted with it on all three
  N 2112 rows while one invocation on the base passed *(field, E2)*.
- **In a serving profile: a partial trace that loads fine.** With CUDA graphs on, sglang's `/start_profile`
  recorded, for one request, 6846 kernels summing 61.2 ms over 30 steps of 28.4 ms, about 2 ms per step, with no
  MoE kernel among them; for four requests the same profiler recorded kernel time equal to the step period.
  Both traces hold 60 graph launches *(field, E3)*.

## Root cause

- A fix's diagnosis traces it to Kineto's roctracer backend, in the torch 2.11 build with Kineto `pytorch/kineto@7a731b6`
  and a torch 2.10 build: stopping a profiling window waits only for
  the correlation ids of direct kernel launches, for at most 50 ms; `hipGraphLaunch` reports none, so a window of
  graph replays stops at once while records may still arrive. Records that arrive after the stop are lost; a
  window whose records all arrive late is "no kernels for a single call". Under full CPU load with eight GPUs
  timing in parallel it hit about 0.5% of processes *(field, E4)*; on a node where other processes were profiling
  and benchmarking at the same time it hit a plain `torch.mul` on every try for 10 minutes *(field, E8)*. Upstream
  Kineto has since replaced that backend with rocprofiler-sdk, whose flush is synchronous.
- The timer's sizing window held one L2-flush kernel and one call, captured in a graph and
  replayed once and then six times (S1). The flush kernel was in the graph, so zero records meant the tracer
  recorded nothing, not that the kernel under test launched nothing. Reading the durations through
  a rocprofiler-sdk tracer that raises when records of the replayed
  kernel dispatches are dropped or missing takes Kineto's stop out of the measurement.
- The partial serving trace is not explained by the record: whether that torch 2.9.1 build uses the roctracer
  backend was not checked.

## How to detect

- **Tell a tracer fault from a kernel fault by what is missing.** Put a kernel of your own in every captured call,
  as that timer's L2 flush is, and count it per replay. No device record at all ("Kineto recorded no kernels for a
  single call", no flush kernel in a replay) is the tracer: the flush is in the graph, so the kernel under test
  cannot have caused it. A count that is short but not zero can be a partly late window as well as a failing
  kernel: one paired timing stopped on "expected 56 L2 flushes per replay, got 54" *(field, E6)*. Measure again
  before calling it either.
- **In a profile you read for kernel shares**, check that the trace is complete before reading it: kernel time
  per step against the step's wall period, the count of a kernel you know runs once per graph replay, and the
  names of the kernels you expect.

## Fix or workaround

- A measurement that stopped on a tracer fault (above): catch it, measure again once in place, then
  again in a fresh process, and do not turn a tracer fault into a revert or a keep.
  Whether a kernel can make the fault more likely is not established: it recurred on one change twice running
  (E2), and a script that called the timer 473 times in nine fresh processes on another kernel saw no
  fault *(field, E5)*.
- **With Kineto, on a node where several processes profile at once, retry after a
  wait, not at once.** The tracer can then lose every call for minutes, and retries back to back all fail: a
  paired timing that tried 3 fresh processes in a row
  failed in all three, in three runs; with the sizing step of S1 retried after waits of 15, 30, 60 and 120 s, both
  reruns completed *(field, E7)*.
- **Before blaming the kernel, check a plain op.** The sizing step of S1 on a plain `torch.mul` faulted about as often
  as on a FlyDSL GEMM in the same checks *(field, E8)*: there, a high fault rate was no sign of a kernel bug.
- For kernel shares from a serving trace, use a trace whose kernel time matches the step period, or time the
  kernel's exact call in a microbenchmark. `rocprofv3 --pid` attached to that server wrote nothing.
- A trace taken with graphs off, to see every kernel, bills each GPU's wait for the others to the
  collectives: `nograph_trace_inflates_collectives.md`.

## Scope

- Kineto traces of HIP-graph replays on MI355X: torch 2.10 and 2.11 builds with the roctracer backend;
  the serving trace on torch 2.9.1+rocm7.2.0 (sglang v0.5.19 MI35x image of 2026-09-16).
- Which Kineto backend the torch of the image `lmsysorg/sglang:v0.5.22-rocm10-mi35x` (2026-10)
  uses is not recorded here.
- Not covered: eager launches, and timers that use HIP events instead of Kineto.

## Evidence

- E1 [2026-10, gfx950, FlyDSL ?, N=14, paired candidate / incumbent timing of a tuning run whose harness called the per-kernel CUDA-graph timer (Kineto) directly] timings 7 and 11 (candidate arm) and 14 (incumbent arm) of 14 paired timings crashed with "Kineto recorded no kernels for a single call" and were recorded as performance reverts (bf16 GEMM, N 768 × K 7168)
- E2 [2026-10, gfx950, FlyDSL ?, N=3, the harness's correctness invocations] with the change, 2 of 2 invocations stopped on "Kineto recorded no kernels for a single call" on all three N 2112 rows; on the base, 1 of 1 passed all 12 rows (Kimi-K3 bf16 GEMMs, small-N group)
- E3 [2026-10, gfx950, torch 2.9.1+rocm7.2.0, N=1, sglang /start_profile over 30 scheduler steps, CUDA graphs on] 1 request: 6846 kernels, 61.2 ms of kernel time over a 875.8 ms span, no kernel with "moe" in its name, step period 28.4 ms; 4 requests: 89007 kernels, 1205.2 ms over a 1253.2 ms span, 13800 MoE kernels; 60 graph launches in each trace (Kimi-K3 DSPARK decode, TP0 trace; counts recomputed from the traces)
- E4 [2026-10, gfx950, torch with Kineto's roctracer backend, N=?, 8 GPUs timing in parallel with every core busy] "Kineto recorded no kernels for a single call" in about 0.5% of processes with the patched timer and with a variant that made the stop wait, by the count recorded with the patch (the per-kernel CUDA-graph timer, through Kineto, on bf16 GEMMs)
- E5 [2026-10, gfx950, FlyDSL ?, N=9, a script calling the per-kernel CUDA-graph timer (Kineto) directly with no retry, nine fresh processes] 473 calls, 0 faults (bf16 GEMM, N 768 × K 7168, the candidate of E1's timing 7)
- E6 [2026-10, gfx950, FlyDSL ?, N=?, paired timing of a tuning run on the per-kernel CUDA-graph timer (Kineto)] a timing stopped on "expected 56 L2 flushes per replay, got 54" (a bf16 GEMM tuning run)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3 runs without the waits and 2 with them, the closing paired timing of a tuning run on the per-kernel CUDA-graph timer (Kineto), nodes shared with other runs' timing] without waits, every run ended "the reviewed kernel could not be timed … BENCH CRASHED" with "Kineto recorded no kernels for a single call" and "(failed in each of 3 fresh processes)": twice on one GEMM (incumbent arm), once on another (candidate arm); with the sizing step of S1 retried after waits of 0, 15, 30, 60 and 120 s, both closing timings completed on the node where one had just failed (bf16 split-K GEMMs, N 6144 × K 16384, a small-M and a large-M run)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=? (about 15 checks of 5–60 tries), the sizing step of S1 or the whole per-kernel CUDA-graph timer (Kineto) in fresh processes, a node shared with other tuning runs] in a loop over the whole timer a plain `x.mul_(1.0)` faulted in 13 of 40 readings and a FlyDSL GEMM in 49 of 168; the sizing step on `torch.mul` faulted in 0 of 30 to 14 of 20 tries per check, on the GEMM in 7 of 20, 11 of 25 and 10 of 25; in one paired check `torch.mul` 13 of 20 against the GEMM 16 of 20; later `torch.mul` passed in none of ~400 polls of 5–10 tries over 10 minutes while 16–26 other processes profiled or benchmarked on the host (bf16 GEMM, N 6144 × K 16384, m 1–128)
- S1 the per-kernel CUDA-graph timer read through Kineto, as it stood in 2026-10: its sizing step captures a flush and one call in a graph, replays it once and then 6 times under the profiler, and raises when no device record came back
