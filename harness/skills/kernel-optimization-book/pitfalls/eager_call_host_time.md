---
title: "A speedup timed in CUDA-graph replays leaves out the host time an eager call pays"
kind: pitfall
scope: [speedups timed in CUDA-graph replays, kernels called outside a CUDA graph, launchers that run Python on every call (FlyDSL `@flyc.jit`)]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1358"
---

# Host time outside the graph ruler

## Symptom

A kernel recorded well above its baseline makes a step slower where it is called eagerly. A Kimi-K3
biased top-16-of-896 router kernel was recorded at 2.98–8.53× over aiter's `biased_grouped_topk`
*(field, E2)*. In graph replays it was 2.84× faster at 2732 tokens and 7.10× at 16384; called eagerly back to back it
took 85.8 and 96.1 µs per call against aiter's 15.5 and 65.5, 0.18× and 0.68× *(field, E1)*.

## Root cause

- **The recorded speedup is GPU time in CUDA-graph replays** (a lower-bound speedup, `../gemm/measurement.md` §1;
  `../flydsl/gemm.md` § autotune §6); a replay runs no Python,
  so a launcher's host work is outside it.
- **A `@flyc.jit` launcher does host work on every call:** it checks captured globals, binds the signature and
  rebuilds the cache key before it reaches the cached call state (`JitFunction.__call__`, S2). `flyc.compile[hints]`
  applied to a launcher with no other arguments returns the launcher itself, so one wrapped that way still pays it
  (S2). The router kernel's entry point calls such a launcher with 15 arguments: 85.8–96.1 µs per eager call, close at both token
  counts, against 5.8–9.5 µs of GPU time per replayed call *(field, E1)*. The cost on launchers with other argument
  counts is measured in `../flydsl/attention.md` § measurement §6 and `../flydsl/gemm.md` § autotune §6.
- The eager readings were taken with nothing queued ahead of the calls, so the host set the pace; behind enough
  queued GPU work the host time is hidden.

The router kernel's launch path, the shape to look for:

```python
@flyc.jit
def _launch(
    logits: fx.Pointer,
    bias: fx.Pointer,
    n_tokens: fx.Int32,
    # ... the other arguments, `stream` last
):
    # ... builds the kernel and launches it
    ).launch(grid=(n_blocks, 1, 1), block=(64 * WPB, 1, 1), stream=stream)

# hints only, no other arguments: this returns `_launch` itself, still a JitFunction
_launch = flyc.compile[
    {"llvm_options": {"amdgpu-kernarg-preload-count": 6, "slp-vectorize-hor": False}}
](_launch)

def run(*, router_logits, correction_bias, topk_weights, topk_ids, num_expert_group, topk_group, renormalize, routed_scaling_factor):
    n_tokens = router_logits.shape[0]
    wpb, prune, n_blocks, top2, ts, tiered = _config(n_tokens)
    _launch(  # every eager call re-checks globals, binds the signature and rebuilds the cache key
        flyc.from_c_void_p(fx.Int32, router_logits.data_ptr()),
        flyc.from_c_void_p(fx.Int32, correction_bias.data_ptr()),
        n_tokens,
        # ... 11 more arguments
        stream=torch.cuda.current_stream(),
    )
```

## How to detect

- Ask how the call site runs: in a captured graph, or eagerly. For an eager call site, time eager calls back to back
  against the baseline called the same way, as well as the graph ruler (E1's eager reading).
- Read the launch path: a `@flyc.jit` function called directly, or wrapped by `flyc.compile[hints]` with no other
  arguments, re-binds and rebuilds its cache key on every call (S2).

## Fix or workaround

- Launch through the object that `flyc.compile` returns when given the launcher and its arguments, built once per
  configuration outside any capture (`../flydsl/gemm.md` § autotune §6, `../flydsl/moe.md` § grouped_host_autotune
  §2); not measured on the router kernel.
- For a kernel whose call site is eager, report the eager reading beside the graph-replay speedup.

## Scope

- Any launcher that runs Python on every call; observed on one FlyDSL kernel with FlyDSL 0.3.2.
- Call sites that replay a CUDA graph do not pay it. The kernel's GPU-side lead held with and without a 1 GiB flush
  before each call (E1).
- Not this trap: the floor that timing under profiling puts under kernels shorter than about 4.5 µs (`profiled_timer_floor.md`).

## Evidence

- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=30 rounds (flush row) and 50 calls in one window (replay and eager rows), kernel and baseline in one process, CUDA-event time per call, medians; rows: one eager call after a 1 GiB write and a GPU spin (GPU time); graph replays without a flush; eager calls back to back with nothing queued (host-bound); both checked against a float64 reference (ids exact, weights 1e-4): the kernel passed at both n_tokens, aiter failed at 16384 (ids differing in 8.8e-5 of entries, weights in 7.6e-6)] the router kernel against aiter's `biased_grouped_topk`, kernel / aiter at n_tokens 2732: flush 7.16 / 20.04 µs (2.80×), replay 5.83 / 16.57 (2.84×), eager 85.79 / 15.49 (0.18×); at n_tokens 16384: flush 15.42 / 71.60 (4.64×), replay 9.45 / 67.04 (7.10×), eager 96.13 / 65.48 (0.68×) (FlyDSL top-k, 896 experts, top 16)
- E2 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5), N=1, lower-bound speedup over aiter's `biased_grouped_topk` per size, one run] the router kernel at n_tokens 893, 2732, 9397, 16384: 2.984, 3.725, 6.718, 8.533
- S2 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1358 `JitFunction.__call__`: the globals-drift check, `sig.bind`, `_build_full_cache_key`, then the call-state cache (:1377-1400); `CompileCallable.__call__` with no arguments returns the function itself (:1715-1717); `CompiledFunction` at :1613
