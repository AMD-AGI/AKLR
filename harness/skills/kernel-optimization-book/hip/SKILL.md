---
name: kernel-optimization-book/hip
description: >
  Making HIP kernels fast and AITER-exact on gfx950 (MI355X). Covers memory-bound
  kernels (normalization, elementwise): the stores' cache policy, which schedule changes matter,
  reproducing AITER's arithmetic bit for bit from its built ISA, disassembling a built library,
  and ablating one change at a time. Use when writing or tuning a HIP kernel.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# HIP Kernel Optimization (gfx950)

## Start here

| Task | Read |
|---|---|
| A memory-bound kernel (norm, elementwise) against AITER | `memory_bound.md` |
| Cache bits, waits and hazards of an instruction | `../hardware/gfx950/memory_and_sync.md` |
| What makes a kernel fast whatever its language (op packages, symptoms, cases) | `../SKILL.md` |

## Build and timing

- A HIP kernel built through `torch.utils.cpp_extension` with `-O3` only has no `-ffast-math`, and the
  fp32 denormal mode of the build differs from AITER's fast-math one. Write what you rely on in the
  source.
- The per-kernel CUDA-graph timer (`../gemm/measurement.md` §1), read through Kineto with
  `GPU_FORCE_QUEUE_PROFILING` forced, holds a kernel below ~4 µs at its floor: two kernels that both run under it
  read about the same whatever their real gap, and a gain read where only the faster one is under it is
  smaller than the real gain. Read through a rocprofiler-sdk tracer, with queue
  profiling not forced, it still has that floor (`../pitfalls/profiled_timer_floor.md`).
