---
title: "Small-M launch floor: a decode row bound by launch, reduction and latency, not by reading B"
kind: pattern
signals: ["M ≤ 256 on a row bound by launch, reduction and memory latency", "a large per-tile intercept F in a K-sweep fit",
  "one CUDA-graph node or an empty kernel costs a large share of the measured time",
  "s_load then lgkmcnt(0) in front of the first B load", ".amdhsa_user_sgpr_kernarg_preload_length = 0"]
ops: [gemm]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
---

# Small-M launch floor: bound by launch, reduction and latency

## Symptom

- M ≤ 256 (decode, speculative verify) on a row bound by launch, reduction and memory latency rather than by the HBM
  read of B (S10); a K-sweep at fixed M and N fits a large per-tile intercept `F` (S18).
- One graph node has a floor even for a 1-element op; at the smallest M on a small weight the node can be more than
  half of the measured time (S3).
- The ISA's first B load waits behind the `s_load` → `lgkmcnt(0)` round trip for the kernel arguments (S4).

## Likely causes

1. **The launch itself.** Under AKLR's timer an empty kernel and each extra launch have a floor of their own, so a
   split-K fold in a separate launch adds one (S5).
2. **The host path.** A `@flyc.jit` call re-binds the signature, rebuilds the cache key and re-checks globals on every
   launch (S6); a dispatch layer that calls the `@flyc.jit` closure pays that cost on every call (S7).
3. **A per-tile fixed cost `F`:** prologue, epilogue, tail peel, address and descriptor setup and drain, paid once
   per tile; its share grows as K shrinks (S8).
4. **The split-K reduction,** a fixed cost that dominates at M = 1 (S9).
5. **A latency-bound row:** bound by launch, reduction and memory latency rather than the HBM read of B (S10).

## Rule out first

- **The host in the window.** Time R consecutive calls inside one event pair; time flat across sizes means the
  host is being measured (S11). One call per CUDA graph measures the launch floor (S12).
- **The wrong span.** `rocprofv3 --kernel-trace` start→end overstated a small kernel against its marginal cost in
  the stream (S13). Price aiter's rows through the deployed entry and your kernel through its `run` entry (S14).
- **The grid.** Count workgroups against the 256-CU round first (S15); see `low_cu_utilization.md`.
- **A bad `F` fit.** A K-sweep fit has five systematic errors, among them the nominal clock and a tile count that
  changes with K (S16).

## Candidate techniques

- Classify the call site as streaming or latency-bound — `gemm/small_m.md` §2 — the small-M lessons
  flip sign between the two (S17).
- Measure `F` — `gemm/grid_pricing.md` §6, `gemm/attribution.md` §7 — a K-sweep fit (S18),
  a trip-count additive probe (S19), or N and K swapped at equal FLOPs (S20).
- Kernel-argument preload — `gemm/small_m.md` §6 — trims `F` at any M (S21); pass operands as
  `fx.Pointer` (`flydsl/gemm.md` § small_m §6, S22) and confirm `.amdhsa_user_sgpr_kernarg_preload_length` > 0 in the ISA (S23).
- Keep the split-K fold in the GEMM launch, in a fixed order, done by the last split to arrive on an atomic counter
  — `gemm/small_m.md` §4 — so that no workgroup waits for another (S24).
- Split-K on latency-bound rows and single N-blocks — `gemm/small_m.md` §4 — not on streaming rows or
  short K (S26, S27).
- Launch the `flyc.compile` object, not the `@flyc.jit` function, except under graph capture, where the raw
  `@flyc.jit` closure is the one to launch — `flydsl/gemm.md` § autotune §6 (S28, S29).
- Fewer launches by fusion — `../flydsl/kernels/flydsl_authoring_method.md` — for latency-bound small shapes (S30).
- Other kernel families: drop a graph node by folding a small kernel into its consumer — `moe/few_tokens.md`
  §1–2; a decode attention's combine is a fixed cost, and when merging it into the producer pays —
  `attention/split_kv.md` §1–2; the small kernels around MoE expert GEMMs —
  `moe/grouped_sorted_tokens.md` §7.

## Evidence

- S3 gemm/small_m.md:50 one graph node's floor; at small M on a small weight more than half is the node
- S4 gemm/small_m.md:399 preload removes the s_load → lgkmcnt(0) round trip before the first B load
- S5 gemm/small_m.md:51 AKLR timer: empty kernel and a second launch each cost; keep the fold in the launch
- S6 flydsl/gemm.md:469 the jit call re-binds the signature, rebuilds the key, re-checks globals
- S7 flydsl/gemm.md:476 a dispatch layer calling the @flyc.jit closure pays host cost per call
- S8 gemm/grid_pricing.md:294 F is prologue, epilogue, tail peel, setup, drain; share grows as K shrinks
- S9 gemm/small_m.md:139 the reduction is a fixed cost that dominates at M=1
- S10 gemm/small_m.md:93 streaming rows bound by HBM read of B; latency-bound by launch/reduction/latency
- S11 gemm/measurement.md:126 keep the host out of the GPU window; flat time across sizes = host
- S12 gemm/measurement.md:129 one call per CUDA/HIP graph measures the launch floor
- S13 gemm/attribution.md:89 rocprofv3 kernel-trace start→end overstated a small kernel
- S14 gemm/small_m.md:65 price aiter through the deployed entry; time your kernel through `run`
- S15 gemm/small_m.md:99 count workgroups against the 256-CU dispatch round before anything else
- S16 gemm/attribution.md:151 a K-sweep fit has five systematic errors
- S17 gemm/small_m.md:87 the lessons flip sign between streaming and latency-bound
- S18 gemm/grid_pricing.md:288 at fixed M and N, sweep K and fit t_tile = a · n_kblocks + F
- S19 gemm/attribution.md:149 trip-count additive probe gives P and F
- S20 gemm/attribution.md:145 same FLOPs, different tile counts: swap N and K
- S21 gemm/grid_pricing.md:301 kernel-argument preload trims F at any M
- S22 flydsl/gemm.md:385 pass operands as fx.Pointer; a raw torch.Tensor stops the preload
- S23 gemm/small_m.md:395 the dumped ISA shows .amdhsa_user_sgpr_kernarg_preload_length > 0
- S24 gemm/small_m.md:145 no result may depend on a split that is not running; the last split to arrive folds
- S26 gemm/small_m.md:134 split-K pays on a single N-block and on latency-bound rows
- S27 gemm/small_m.md:135 split-K does not pay on streaming rows or short K
- S28 flydsl/gemm.md:469 call the flyc.compile object, not the @flyc.jit function
- S29 flydsl/gemm.md:474 under CUDA-graph capture call the raw closure
- S30 ../flydsl/kernels/flydsl_authoring_method.md:50 latency-bound (small shapes): reduce launch count, fusion
