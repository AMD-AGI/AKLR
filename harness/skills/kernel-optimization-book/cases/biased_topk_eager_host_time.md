---
title: "Biased top-16 of 896 experts per token (aiter's `biased_grouped_topk`, every group kept), bf16 logits and bias — fast fp32 keys bound the top 16 and the candidates are ranked by the reference's key, in one launch; a `@flyc.jit` launcher that runs Python on every call, slower than aiter when called eagerly"
kind: case
ops: [topk]
dtypes: [bf16, fp32]
regimes: ["tokens 893", "tokens 2732-16384"]
accuracy: bitwise
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=30 (rounds, the reading after a flush; the graph-replay and eager readings are 50 calls in one window)
measured:
  - {metric: "lower-bound speedup over aiter's `biased_grouped_topk`, per size, GPU time in CUDA-graph replays, one run", value: "2.984, 3.725, 6.718, 8.533", gpu: MI355X, workload: "893, 2732, 9397, 16384 tokens; 896 experts, top 16; aiter failed the timing run's float64 check at 16384", date: "2026-11"}
  - {metric: "time per call, kernel / aiter's `biased_grouped_topk`, eager calls back to back with nothing queued, median of 50 calls", value: "85.79 / 15.49 µs (0.18×) at 2732 tokens, 96.13 / 65.48 µs (0.68×) at 16384", gpu: MI355X, workload: "2732 and 16384 tokens; 896 experts, top 16; aiter failed the timing run's float64 check at 16384", date: "2026-11"}
  - {metric: "time per call, kernel / aiter's `biased_grouped_topk`, CUDA-graph replays without a flush, median of 50 calls", value: "5.83 / 16.57 µs (2.84×) at 2732 tokens, 9.45 / 67.04 µs (7.10×) at 16384", gpu: MI355X, workload: "2732 and 16384 tokens; 896 experts, top 16; aiter failed the timing run's float64 check at 16384", date: "2026-11"}
  - {metric: "time per call, kernel / aiter's `biased_grouped_topk`, one eager call after a 1 GiB write and a GPU spin, median of 30 rounds", value: "7.16 / 20.04 µs (2.80×) at 2732 tokens, 15.42 / 71.60 µs (4.64×) at 16384", gpu: MI355X, workload: "2732 and 16384 tokens; 896 experts, top 16; aiter failed the timing run's float64 check at 16384", date: "2026-11"}
---

# Biased top-16 of 896 experts (Kimi-K3's router): fast in graph replays, slower than aiter when called eagerly

One FlyDSL kernel for aiter's `biased_grouped_topk` at the shape of Kimi-K3's router: per token, the 16 of 896
experts with the highest sigmoid(logit) + bias, each weighed by its sigmoid score renormalized over the 16. It was
recorded at 2.98–8.53× over aiter in CUDA-graph replays *(field, E1)* (8.53 at 16384 tokens, where aiter failed the
float64 check of the run that timed the eager calls: § Counter-signals), and called eagerly it was slower than
aiter, 0.18× and 0.68× at 2732 and 16384 tokens *(field, E2)*: its launcher runs Python on every call. The trap is
[pitfalls/eager_call_host_time](../pitfalls/eager_call_host_time.md); the launch path that avoids it is in
[flydsl/gemm](../flydsl/gemm.md) § autotune §6 and [flydsl/moe](../flydsl/moe.md) § grouped_host_autotune §2.

## Applies when

- **Op.** aiter's `biased_grouped_topk` with every expert group kept, so that the grouping selects nothing: per
  token, the 16 experts of 896 with the highest key, sigmoid(logit) + bias, in descending order of the key, the lower
  expert first on a tie. A chosen expert's weight is its sigmoid score without the bias, divided by the sum of the
  16 when `renormalize` is set, times `routed_scaling_factor`.
- **Operands.** `router_logits` bf16 (tokens, 896) and `correction_bias` bf16 (896); outputs `topk_weights` fp32
  (tokens, 16) and `topk_ids` int32 (tokens, 16). The kernel takes `num_expert_group` and `topk_group` and does not
  read them.
- **Sizes.** 893, 2732, 9397 and 16384 tokens; the form changes with the token count (§ Mechanism).
- **Baseline.** aiter's `biased_grouped_topk`.
- **Accuracy.** The kernel is held to `bitwise` accuracy: every run on the same inputs gives the same bytes. It has
  no atomics, and each token's outputs are written by the wave that owns the token.

## Counter-signals

- **Called eagerly it is slower than aiter.** Called back to back with nothing queued it took 85.8 and 96.1 µs per
  call at 2732 and 16384 tokens against aiter's 15.5 and 65.5, 0.18× and 0.68×
  *(field, E2)*. Its entry point calls a `@flyc.jit` launcher, which does host work
  on every call: 85.8–96.1 µs per eager call, close at both token counts, against 5.8–9.5 µs of GPU time per
  replayed call.
- **The recorded speedup does not see it.** It is GPU time in CUDA-graph replays, and a replay runs no Python, so a
  launcher's host work is outside it.
- **Where the host time shows.** The eager readings were taken with nothing queued ahead of the calls, so the host
  set the pace; behind enough queued GPU work the host time is hidden, and a call site that replays a CUDA graph
  does not pay it.
- **The fix was not measured on this kernel.** It is to launch through the object that `flyc.compile` returns when
  given the launcher and its arguments (§ Code). That object takes positional arguments only and has the launcher's
  `Constexpr` arguments baked in at compile time (`../flydsl/attention.md` § measurement §6); this launcher has
  seven, five of them chosen from the token count, so it is one object per configuration.
- **Eager readings exist at two of the four sizes,** 2732 and 16384 tokens. 893 and 9397 tokens have the
  graph-replay speedup only; the smallest of the four speedups is 2.984, at 893 tokens
  *(field, E1)*.
- **At 16384 tokens the baseline failed the check of the timing run** (§ Result). Against a float64 reference (ids
  exact, weights within 1e-4) the kernel passed at both token counts and aiter failed at 16384, its ids differing in
  8.8e-5 of the entries and its weights in 7.6e-6 *(field, E2)*. A baseline's
  failure is reported with the size (`../pitfalls/baseline_silently_wrong.md`).

## Mechanism

The selection is one launch. By the source's account it is exact against the fp32 reference: fast fp32 keys bound
the top 16 from below, and the candidates above the bound are ranked by the reference's own key.

- **Bound.** The fast key is sigmoid(x) + b in fp32 through `v_exp_f32` and `v_rcp_f32`. The threshold is L, the
  16th largest of the lanes' best fast keys (above 12288 tokens, of each lane pair's two best), which at least 16
  experts reach, lowered by 2^-16 · max(1, |L|). The experts at or above it are the candidates; they are compacted
  into LDS, a lane's slot counted from wave ballots (`rocdl.ballot`, `rocdl.mbcnt_lo` and `mbcnt_hi`), each entry
  holding the expert id and the bf16 bits of its logit and bias.
- **Rank.** Each lane takes one candidate and counts the candidates that rank above it. A lane with fewer than 16
  above it stores its id and weight at (token, rank); every other lane's store gets an offset past the output's
  buffer descriptor, which drops it. The exact rank word is the reference's key, `1 / (1 + exp(-x))` with an accurate
  `exp` and an IEEE divide plus the bias, as an order-preserving integer above the reversed expert id, so the lower
  expert wins a tie. The weight's denominator, the sum of the chosen experts' sigmoid scores, is a lane reduction
  (DPP steps, then lane swaps), and the division is a `v_rcp_f32`.
- **Up to 2048 tokens (893 of the measured sizes): one wave per token.** Each lane loads 14 logits of the token's
  row and their 14 biases, the wave holds up to 64 candidates, and every candidate is ranked by the exact word.
- **Above 2048 tokens (2732, 9397, 16384): a bias prune, then two tokens per wave,** with 8 waves per workgroup, 4
  from 4097 to 12288 tokens. Wave 0 of each workgroup builds from the bias alone the list of experts that can reach
  the top 16, while the other waves wait at a barrier: with T′ the 16th largest of the lanes' bias maxima, a key is
  at most fl(1 + b) and at least 16 keys are T′ or more, so an expert with fl(1 + b) < T′ is left out. Each wave
  copies its two rows into LDS (`rocdl.buffer_load_to_lds`, 16 B per lane) and scores one token on lanes 0–31 and
  the other on lanes 32–63, up to 5 list entries per lane and 32 candidates per token. It ranks by 32-bit words
  built from the fast keys and goes to the exact words when two neighbouring ranks' keys are within
  2^-21 · max(1, |k|) of each other, or when a candidate's key lies so far above the threshold, 2^26 − 1 or more
  steps of the keys' order-preserving integer, that its 32-bit word clamps.
- **Fallbacks.** A token with more candidates than its lanes hold is redone on all 64 lanes of its wave and, above
  64 candidates, by 16 rounds of an arg-max of the reference's key over its whole row; a workgroup whose list has
  more than 160 entries scores each token on 64 lanes.

**Launch.** `run` takes the token count's configuration from a cached function and calls the launcher with 15
arguments: the four tensors as `fx.Pointer` values made from `data_ptr()`, the token count, the scale, the workgroup
count, seven `Constexpr` switches and the stream. The launcher is a `@flyc.jit` function that launches the kernel on
`n_blocks` workgroups of `64 * WPB` threads. It is wrapped by `flyc.compile[...]` with two LLVM options and no other
argument: `amdgpu-kernarg-preload-count` 6, which asks for the kernel's 6 run-time arguments in SGPRs
(`../gemm/small_m.md` §6), and `slp-vectorize-hor` off, which by the source's comment keeps the straight-line rank
counts as compare-and-add. Wrapped that way the launcher is still a `JitFunction`
(S1), and every call pays its host work (§ Code).

## Code

### 1. A `@flyc.jit` launcher behind `flyc.compile[hints]`, called on every call

The kernel's launch path, the one that pays the host time: `_launch`, a `@flyc.jit` function that launches the
kernel on `n_blocks` workgroups of `64 * WPB` threads (`WPB`: the waves per workgroup); a `flyc.compile[...]` that
gives it two LLVM options and no other argument; and the entry point `run`, which takes the token count's
configuration from `_config` and calls `_launch` with 15 arguments. Applied with hints only, `flyc.compile` returns
`_launch` itself, still a `JitFunction` (S1), so every eager call checks captured
globals, binds the signature and rebuilds the cache key before it reaches the cached call state. The right form
launches through the object that `flyc.compile` returns when it is given the launcher and its arguments, built once
per configuration outside any capture ([flydsl/gemm](../flydsl/gemm.md) § autotune §6,
[flydsl/moe](../flydsl/moe.md) § grouped_host_autotune §2); it was not measured on this kernel.

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

Not this: a `@flyc.jit` launcher called directly, or wrapped by `flyc.compile[hints]` with no other arguments, at a
call site that runs eagerly: it re-binds and rebuilds its cache key on every call
(S1). On this kernel an eager call took 85.8–96.1 µs, close at both token counts,
against 5.8–9.5 µs of GPU time per replayed call *(field, E2)*; the cost on
launchers with other argument counts is in `../flydsl/attention.md` § measurement §6 and `../flydsl/gemm.md`
§ autotune §6.

Explained in [pitfalls/eager_call_host_time](../pitfalls/eager_call_host_time.md) § Root cause.

## Result

Two rulers, both against aiter's `biased_grouped_topk`, on FlyDSL 0.3.2.

**The recorded speedup** is a lower-bound speedup (`../gemm/measurement.md` §1): GPU time in CUDA-graph replays,
per size, one run.

**The timing run** has the kernel and aiter in one process, on an MI355X, and reads the median CUDA-event time per
call three ways: one eager call after a 1 GiB write and a GPU spin, which is GPU time (30 rounds); CUDA-graph
replays without a flush; and eager calls back to back with nothing queued, where the host sets the pace (50 calls in
one window each).

| Tokens | Lower-bound speedup in graph replays *(field, E1)* | One call after a flush, kernel / aiter *(field, E2)* | Graph replays, kernel / aiter *(field, E2)* | Eager back to back, kernel / aiter *(field, E2)* |
|---|---|---|---|---|
| 893 | 2.984 | — | — | — |
| 2732 | 3.725 | 7.16 / 20.04 µs, 2.80× | 5.83 / 16.57 µs, 2.84× | 85.79 / 15.49 µs, 0.18× |
| 9397 | 6.718 | — | — | — |
| 16384 | 8.533 | 15.42 / 71.60 µs, 4.64× | 9.45 / 67.04 µs, 7.10× | 96.13 / 65.48 µs, 0.68× |

The kernel's GPU-side lead held with and without a flush before each call; called eagerly it was slower than aiter
at both token counts *(field, E2)*. In the same run both were checked against a
float64 reference (ids exact, weights within 1e-4): the kernel passed at both token counts and aiter failed at 16384
(§ Counter-signals).

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5), N=1, lower-bound speedup over aiter's `biased_grouped_topk` per size, one run] the router kernel at n_tokens 893, 2732, 9397, 16384: 2.984, 3.725, 6.718, 8.533
- E2 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=30 rounds (flush row) and 50 calls in one window (replay and eager rows), kernel and baseline in one process, CUDA-event time per call, medians; rows: one eager call after a 1 GiB write and a GPU spin (GPU time); graph replays without a flush; eager calls back to back with nothing queued (host-bound); both checked against a float64 reference (ids exact, weights 1e-4): the kernel passed at both n_tokens, aiter failed at 16384 (ids differing in 8.8e-5 of entries, weights in 7.6e-6)] the router kernel against aiter's `biased_grouped_topk`, kernel / aiter at n_tokens 2732: flush 7.16 / 20.04 µs (2.80×), replay 5.83 / 16.57 (2.84×), eager 85.79 / 15.49 (0.18×); at n_tokens 16384: flush 15.42 / 71.60 (4.64×), replay 9.45 / 67.04 (7.10×), eager 96.13 / 65.48 (0.68×) (FlyDSL top-k, 896 experts, top 16)
- S1 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1358 `JitFunction.__call__`: the globals-drift check, `sig.bind`, `_build_full_cache_key`, then the call-state cache (:1377-1400); `CompileCallable.__call__` with no arguments returns the function itself (:1715-1717); `CompiledFunction` at :1613
