---
name: aklr-kb/hip
description: >
  Making HIP kernels fast and AITER-exact on gfx950 (MI355X) inside AKLR. Covers memory-bound
  kernels (normalization, elementwise): the stores' cache policy, which schedule changes matter,
  reproducing AITER's arithmetic bit for bit from its built ISA, disassembling a built library,
  and ablating one change at a time. Use when writing or tuning a hip kernel for a AKLR schema.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# HIP Kernel Optimization (gfx950, AKLR)

## Start here

| Task | Read |
|---|---|
| A memory-bound kernel (norm, elementwise) against AITER | `memory_bound.md` |
| The ABI and audit of a hip schema kernel (`TORCH_LIBRARY_EXPAND`, SH0 – SH5) | `aklr/ops/README.md` (hip kernels) |
| Cache bits, waits and hazards of an instruction | `../hardware/gfx950/memory_and_sync.md` |
| How AKLR times a Point | `harness/skills/aklr-read-validation/references/ruler.md` |
| What makes a kernel fast whatever its language (op packages, symptoms, cases, AKLR's rules) | `../SKILL.md` |

## In AKLR

- AKLR builds a hip kernel with `torch.utils.cpp_extension` and `-O3` only
  (`aklr/core/jit/hip_extension.py`): no `-ffast-math`, and the fp32 denormal mode of the build
  differs from AITER's fast-math one. Write what you rely on in the source.
- `hip_extension.load_library(name, sources, root, arch)` keys its build directory on the toolchain
  and arch only: building a second package under the same `root` finds the first one's marker and
  loads a library that does not exist. Give each package its own `root` when building variants
  yourself; AKLR's builder already does.
- Below ~4 µs a kernel sits on the graph-node floor of AKLR's measurement: a gain there is a fixed
  latency difference and may not carry to another node.
