---
kind: index
---

# Pitfalls

Silent wrong results, FlyDSL version traps and measurement artifacts.

<!-- generated:start (maintenance/kb_index.py) -->
| Page | Scope | Gens | FlyDSL | Basis |
|---|---|---|---|---|
| [The baseline you time against can return wrong values, and a benchmark that checks only the candidate does not see it](baseline_silently_wrong.md) | aiter gemm_a4w4 asm path at K % 256 != 0, hipBLASLt MX GEMMs through torch._scaled_mm, any baseline a kernel is timed against | gfx950 | 0.3.2 | single-observation |
| [Benchmark sizes with degenerate inputs let a kernel win on a path serving does not take](degenerate_points.md) | benchmark sizes and the distributions that draw their inputs, kernels with a fast path keyed on input values (per-request lengths, pool sizes), case pages that cite a speedup measured on such inputs | gfx950 | 0.3.2 | single-observation |
| [A speedup timed in CUDA-graph replays leaves out the host time an eager call pays](eager_call_host_time.md) | speedups timed in CUDA-graph replays, kernels called outside a CUDA graph, launchers that run Python on every call (FlyDSL `@flyc.jit`) | gfx950 | 0.3.2 | single-observation |
| [A GPU memory access fault names the kernel that faulted, not the one that wrote out of bounds](fault_in_the_wrong_kernel.md) | runs of many kernels (a server, an e2e step) with FlyDSL kernels whose buffer descriptors are wider than their buffers | gfx950 | 0.3.2 | single-observation |
| [The code you timed is not the code you edited](measured_a_different_build.md) | remote or synced bench harnesses, servers and benches that import another checkout, FlyDSL's compile cache, several venvs or FlyDSL versions, layers that run a stored copy of a kernel or fall back to another implementation | gfx950 | 0.3.2 | single-observation |
| [A collective's trace time includes each GPU's wait for the others; with CUDA graphs off the wait is expected to grow](nograph_trace_inflates_collectives.md) | per-kernel tables from multi-GPU traces, aiter's custom allreduce kernels and the allreduce fused with RMSNorm | gfx950 | 0.3.2 | source-semantics |
| [Timing a kernel under profiling puts a floor of about 4 µs under short kernels](profiled_timer_floor.md) | three timers measured on MI355X, a Kineto graph timer with queue profiling forced, CUDA-event timing of back-to-back graph replays in a process that forces queue profiling, and one kernel tracer built on rocprofiler-sdk, any kernel or baseline shorter than about 4.5 µs | gfx950 | 0.3.2 | measured N=3 |
| [Kineto can drop the kernels a HIP graph launches; a timer that reads kernel times from it finds no records at all, a serving profile a partial trace](profiler_misses_kernels.md) | torch profiler (Kineto) traces of HIP-graph replays, the per-kernel CUDA-graph timer read through Kineto and harnesses that call it, sglang /start_profile with CUDA graphs on | gfx950 | 0.3.2 | single-observation |
| [A speedup measured by another harness does not predict the per-size speedup on the ruler that judges the kernel](speedup_from_another_ruler.md) | any kernel whose speedup is quoted from another ruler than the one that judges it, scores from agent tuning runs (AgentKernelArena and others), ad hoc timing probes | gfx950 | 0.3.2 | single-observation |
| [State kept between calls lets a kernel pass the checks it should fail, or time faster than it runs in serving](state_kept_across_calls.md) | launch code and kernels in FlyDSL, Triton / Gluon or HIP, kernels written by an agent tuning run, harnesses that time or check repeated calls on one set of buffers, ops that update state in place | gfx950 | 0.3.2 | single-observation |
<!-- generated:end -->
