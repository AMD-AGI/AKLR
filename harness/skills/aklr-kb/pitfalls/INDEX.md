---
kind: index
---

# Pitfalls

Silent wrong results, FlyDSL version traps and measurement artifacts.

<!-- generated:start (maintenance/kb_index.py) -->
| Page | Scope | Gens | FlyDSL | Basis |
|---|---|---|---|---|
| [AKLR reads every dispatch at about 4.1 µs or more: shorter kernels all look the same](aklr_dispatch_floor.md) | aklr validate and AKLR CI timing, any kernel or baseline shorter than about 4 µs | gfx950 | 0.3.2 | measured N=3 |
| [The baseline you time against can return wrong values, and AKLR's bench does not check it](baseline_silently_wrong.md) | aiter gemm_a4w4 asm path at K % 256 != 0, hipBLASLt MX GEMMs through torch._scaled_mm, any baseline a AKLR Point is timed against | gfx950 | 0.3.2 | single-observation |
| [A GPU memory access fault names the kernel that faulted, not the one that wrote out of bounds](fault_in_the_wrong_kernel.md) | runs of many kernels (a server, an e2e step) with FlyDSL kernels whose buffer descriptors are wider than their buffers | gfx950 | 0.3.2 | single-observation |
| [The code you timed is not the code you edited](measured_a_different_build.md) | remote or synced campaign harnesses, servers and benches that import another checkout, FlyDSL's compile cache, several venvs or FlyDSL versions, AKLR validate and e2e runs | gfx950 | 0.3.2 | single-observation |
| [Kineto can drop the kernels a HIP graph launches; AKLR reports the all-dropped case as a platform failure](profiler_misses_kernels.md) | torch profiler (Kineto) traces of HIP-graph replays, AKLR's bench_cudagraph timer and harnesses that call it, sglang /start_profile with CUDA graphs on | gfx950 | 0.3.2 | single-observation |
| [A speedup measured by another harness does not predict the per-Point speedup AKLR CI writes](speedup_from_another_ruler.md) | any kernel headed for AKLR, scores from agent optimization loops (KernelForge, AgentKernelArena), ad hoc timing probes | gfx950 | 0.3.2 | single-observation |
| [State kept between calls lets a kernel pass the checks it should fail](state_kept_across_calls.md) | launch code and kernels in any AKLR language, kernels written by an optimization loop, harnesses that time or check repeated calls on one set of buffers | gfx950 | 0.3.2 | single-observation |
<!-- generated:end -->
