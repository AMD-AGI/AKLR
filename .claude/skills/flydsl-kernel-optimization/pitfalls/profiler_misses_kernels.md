---
title: "Kineto can drop the kernels a HIP graph launches; AKLR reports the all-dropped case as a platform failure"
kind: pitfall
scope: [torch profiler (Kineto) traces of HIP-graph replays, AKLR's bench_cudagraph timer and harnesses that call it, sglang /start_profile with CUDA graphs on]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/core/utils/testing.py:85"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/core/validation/stages/_measure.py:239"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/core/validation/stages/_bench/_stage.py:319"
  - "AMD-AGI/AKLR@db3f6cda3 commit message (PR #54)"
---

# Kineto misses graph-launched kernels

## Symptom

Two forms, both silent about which kernels went missing.

- **In AKLR's timer: no records at all.** `RuntimeError: Kineto recorded no kernels for a single call`, raised
  while sizing a measurement (S1). Through `python -m aklr validate` it ends as a `platform_failure` finding only
  after AKLR's own retries (below). A harness that calls `bench_cudagraph` directly loses the arm's measurement:
  in one optimization run on a bf16 GEMM, 3 of 14 measured rounds lost their paired timing this way, two on the
  candidate arm and one on the unchanged incumbent, and all three were recorded as performance reverts
  *(field, E1)*. In another, a change was reverted after two correctness invocations aborted with it on all three
  n2112 rows while one invocation on the base passed *(field, E2)*.
- **In a serving profile: a partial trace that loads fine.** With CUDA graphs on, sglang's `/start_profile`
  recorded, for one request, 6846 kernels summing 61.2 ms over 30 steps of 28.4 ms, about 2 ms per step, with no
  MoE kernel among them; for four requests the same profiler recorded kernel time equal to the step period.
  Both traces hold 60 graph launches *(field, E3)*.

## Root cause

- Commit `db3f6cda3` (PR #54) traces it to Kineto's roctracer backend, in the torch 2.11 build it names (Kineto
  `7a731b6`) and the torch 2.10 that AKLR's CI image was then built on: stopping a profiling window waits only for
  the correlation ids of direct kernel launches, for at most 50 ms; `hipGraphLaunch` reports none, so a window of
  graph replays stops at once while records may still arrive. Records that arrive after the stop are lost; a
  window whose records all arrive late is "no kernels for a single call". Under full CPU load with eight GPUs
  timing in parallel it hit about 0.5% of processes *(field, E4)*. Upstream Kineto has since replaced that
  backend with rocprofiler-sdk, whose flush is synchronous.
- The window AKLR checks holds one L2-flush kernel and one call, captured in a graph and replayed once and then
  six times (S1). The flush kernel is in the graph, so zero records means the tracer recorded nothing, not that
  your kernel launched nothing.
- The partial serving trace is not explained by the record: whether that torch 2.9.1 build uses the roctracer
  backend was not checked.

## How to detect

- **In AKLR, read the message, not the stage.** Four messages count as tracer faults: "Kineto recorded no kernels
  for a single call", "L2 flushes per replay, got 0", `std::bad_alloc` and `std::length_error` (S2). AKLR measures
  again once in place (S3), then reruns the whole task once in a fresh process (S4), and reports
  `platform_failure` only when both fail. The other trace checks of the timer ("did not record every graph
  launch", "expected N L2 flushes per replay, got M" with M above 0, "no target kernels", "incomplete or
  inconsistent kernels", S5) are not on that list: they fail the task as a candidate crash (S6), although a
  partly late window can produce them as well: one harness run crashed on "expected 56 L2 flushes per replay, got
  54" *(field, E6)*.
- **In a profile you read for kernel shares**, check that the trace is complete before reading it: kernel time
  per step against the step's wall period, the count of a kernel you know runs once per graph replay, and the
  names of the kernels you expect. AKLR's timer does this with its flush kernel: exactly one per call in every
  replay, and the same kernel counts in every replay (S5).

## Fix or workaround

- `platform_failure` with one of the four messages in `aklr validate`: rerun (`../SKILL.md` § Also in AKLR).
- A harness that calls `bench_cudagraph` directly: catch the tracer-fault messages, measure again once, then
  again in a fresh process, as AKLR does (S3, S4), and do not turn a tracer fault into a revert or a keep.
  Whether a kernel can make the fault more likely is not established: it recurred on one change twice running
  (E2), and a script that called `bench_cudagraph` 473 times in nine fresh processes on another kernel saw no
  fault *(field, E5)*.
- For kernel shares from a serving trace, use a trace whose kernel time matches the step period, or time the
  kernel's exact call in a microbenchmark. `rocprofv3 --pid` attached to that server wrote nothing.

## Scope

- Kineto traces of HIP-graph replays on MI355X: torch 2.10 and 2.11 builds with the roctracer backend per
  `db3f6cda3`; the serving trace on torch 2.9.1+rocm7.2.0 (sglang v0.5.19 MI35x image of 2026-09-16).
- AKLR's CI image moved to `lmsysorg/sglang:v0.5.22-rocm10-mi35x` in 2026-10; which Kineto backend its torch
  uses is not recorded here.
- Not covered: eager launches, and timers that use HIP events instead of Kineto.

## Evidence

- E1 [2026-10, gfx950, FlyDSL ?, N=14, KernelForge paired timing; the harness driver called AKLR bench_cudagraph directly] rounds 7 and 11 (candidate arm) and 14 (incumbent arm) of 14 measured rounds crashed with "Kineto recorded no kernels for a single call" and were recorded REVERT_PERF (gemm_n768_k7168 bf16 GEMM; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards/demo_n768_12h/forge_experiments/candidates/iter_014/validation.txt:5; iter_007/meta.json:18; iter_011/meta.json:18)
- E2 [2026-10, gfx950, FlyDSL ?, N=3, the harness driver's correctness invocations] with the change, 2 of 2 invocations returned `platform_failure: Kineto recorded no kernels for a single call` on all three n2112 rows; on the base, 1 of 1 passed all 12 rows (Kimi-K3 bf16 GEMMs, small-N group; local:/workspace/code/gpt_oss_docker/work/sikl/k3_forge/r2_lessons/r2_small_n_k7168/iter_001.md:11)
- E3 [2026-10, gfx950, torch 2.9.1+rocm7.2.0, N=1, sglang /start_profile over 30 scheduler steps, CUDA graphs on] 1 request: 6846 kernels, 61.2 ms of kernel time over a 875.8 ms span, no kernel with "moe" in its name, step period 28.4 ms; 4 requests: 89007 kernels, 1205.2 ms over a 1253.2 ms span, 13800 MoE kernels; 60 graph launches in each trace (Kimi-K3 DSPARK decode, TP0 trace; counts recomputed from local:/workspace/code/gpt_oss_docker/scratch/k3cb/results/smci_dspark_decode_prof/prof_bs1/TP-0.trace.json.gz and prof_bs4; step period and verdict local:/workspace/code/gpt_oss_docker/scratch/k3cb/BASELINE.md:165)
- E4 [2026-10, gfx950, torch with Kineto's roctracer backend, N=?, 8 GPUs timing in parallel with every core busy] "Kineto recorded no kernels for a single call" in about 0.5% of processes with the patched timer and with a variant that made the stop wait, by the commit's count; the KB note of the same experiment gives 2 of 96 processes on the previous timer and 4 of 480 on the patched one (AKLR bench_cudagraph on bf16 GEMMs; db3f6cda3, #54; local:/workspace/code/gpt_oss_docker/.claude/memory/project_hyperloom_sikl_path.md:108)
- E5 [2026-10, gfx950, FlyDSL ?, N=9, a script calling bench_cudagraph directly with no retry, nine fresh processes] 473 calls, 0 faults (gemm_n768_k7168 candidate of round 7; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards/demo_n768_12h/forge_experiments/lessons/iter_008.md:32)
- E6 [2026-10, gfx950, FlyDSL ?, N=?, KernelForge paired timing over AKLR bench_cudagraph] a timing crash read "expected 56 L2 flushes per replay, got 54" (gemm_n768_k7168 bf16 GEMM, demo run of 2026-10-22; KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_forge_loop_sikl_upgrade.md:31)
- S1 aklr/core/utils/testing.py:85 at 0cef4f2e3: `_records_per_call` captures a flush and one call, replays 1 then 6 times under a profiler, raises at :110 when no device record came back
- S2 aklr/core/validation/stages/_measure.py:298 at 0cef4f2e3: `_TRACER_FAULT_MESSAGES`
- S3 aklr/core/validation/stages/_measure.py:239 at 0cef4f2e3: `Clock.graph_reading` measures again once on a tracer fault
- S4 aklr/core/validation/stages/_bench/_stage.py:319 at 0cef4f2e3: a tracer fault on the first attempt reruns the task in a fresh process
- S5 aklr/core/utils/testing.py:114 at 0cef4f2e3: the per-replay checks on graph launches, flush counts and kernel counts (:124-158)
- S6 aklr/core/validation/stages/_bench/_stage.py:448 at 0cef4f2e3: a task error is `platform_failure` only when it is a tracer fault, else `candidate_crash`
