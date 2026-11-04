---
name: aklr-kb/flydsl
description: >
  Writing FlyDSL kernels for AKLR on gfx950 (MI350/MI355X): FlyDSL v0.3.2 limits, compile errors and version
  differences, debugging wrong or non-compiling kernels and the compile cache, aiter's FlyDSL kernels and their
  knobs, authoring a kernel structure first, and links to FlyDSL's own skills (GEMM walkthrough, LDS, prefetch,
  authoring, code cleanup) with what differs on gfx950. Read with aklr-kb's SKILL.md when optimizing a FlyDSL kernel.
  Not for Triton, Gluon or HIP kernels.
allowed-tools: Read Edit Bash Grep Glob Agent
gens: [gfx950]
---

# FlyDSL kernels

How to write in FlyDSL what `../SKILL.md` says to do; check a plan against its Stop table first. Paths
below are relative to this folder. FlyDSL's own skills are linked at `v0.3.2`, not copied: `guides.md` gives each
link and what differs on gfx950 or what this repository adds. Read one with `curl -sSfL <link>` (it needs network);
without network, `guides.md` still has every correction.

Every API statement here is for FlyDSL v0.3.2; AKLR requires `flydsl>=0.3.2` (`pyproject.toml`). CI measures on the
FlyDSL that the baseline aiter installs (`AKLR_BASELINE_AITER` in `.github/workflows/ci.yaml`; its `setup.py` pins
0.3.2 today), so a new baseline can change it: check it before trusting a version difference
(`../pitfalls/measured_a_different_build.md`).

## By task

| Task | Read |
|---|---|
| Your first FlyDSL GEMM (FlyDSL's walkthrough), before `../gemm/INDEX.md` | `guides.md` § gemm-optimization |
| The language: layouts, `@flyc.kernel` / `@flyc.jit`, copy atoms, MFMA, loops | `guides.md` § flydsl-kernel-authoring |
| Author and optimize any other kernel, structure first | `kernels/flydsl_authoring_method.md` |
| Read aiter's FlyDSL kernels (dense GEMM, MoE, attention and the rest) and their knobs | `kernels/flydsl_kernel_library.md` → `kernels/flydsl_knob_space.md` |
| LDS bank conflicts, double buffering, swizzle | `guides.md` § lds-optimization |
| Overlap loads with compute, prefetch, software pipelining; MFMAs wait on global loads | `guides.md` § prefetch-data-load |
| Modernize or clean up an existing kernel's code | `guides.md` § kernel-code-cleanup |
| Wrong output, NaN, won't compile, a fix that didn't take | `debug-flydsl-kernel.md` (clear the cache first); an error message: `language/compile_errors.md` |
| Code from another FlyDSL version, or FlyDSL doing what you did not expect | `language/version_notes.md`, `language/limits.md` |
| Run or extend FlyDSL's own tests and benchmarks, inside a FlyDSL checkout | `flydsl_repo_tests.md` |

## Rules that hold everywhere

- **Clear the compile cache** (`rm -rf "${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}"`) before believing
  any result; `debug-flydsl-kernel.md` § What the cache key sees tells you when a result was stale.
