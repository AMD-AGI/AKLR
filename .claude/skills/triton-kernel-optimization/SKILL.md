---
name: triton-kernel-optimization
description: >
  Making Triton kernels fast on gfx950 (MI355X) inside AKLR. Covers small-M (decode) dense bf16
  GEMMs that stream their weight once: tile shape, weight view, XCD order, K phase, split-K under a
  bitwise contract, the M = 1 case, and re-tuning for AKLR's cold-cache measurement. Use when
  writing or tuning a Triton kernel for a AKLR schema, especially a decode GEMM.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Triton Kernel Optimization (gfx950, AKLR)

## Start here

| Task | Read |
|---|---|
| A dense GEMM with M ≤ 64 (decode) | `gemm/decode_gemm.md` |
| How AKLR times a Point, and why a result moved | `harness/skills/aklr-read-validation` (`references/ruler.md`) |
| The kernel's rules: audit, compile functions, scratch | `aklr/ops/README.md` (the kernel section and ABI) |

## In AKLR

- A `@triton.jit` body reads no module-level value: pass it as a `tl.constexpr` argument. Choose the
  configuration in a `@functools.cache` compile function of shapes and dtypes. `@triton.autotune`
  or `@triton.heuristics` above `@triton.jit` makes the def host code, audited by SA3, SA7 and SA8;
  choose the configuration in the compile function instead.
- The dispatcher passes contiguous row-major tensors, so strides can be `tl.constexpr` values
  folded from the shape, and an `EVEN_M`-style constexpr can drop row masks.
- Every Point is timed with a 1 GiB flush before each call: weights come from HBM, never from the
  256 MB MALL. Tune under that; configurations found hot do not carry.
- CI's image may carry another Triton than a local container (CI has run 3.8.0 where a container
  had 3.7.0): re-time a Triton kernel on CI's image before trusting a small margin.
