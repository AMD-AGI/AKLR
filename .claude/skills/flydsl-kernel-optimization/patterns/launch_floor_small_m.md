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

- Classify the call site as streaming or latency-bound — `gemm/flydsl_gemm_small_m.md` §2 — the small-M lessons
  flip sign between the two (S17).
- Measure `F` — `gemm/flydsl_gemm_grid_pricing.md` §6, `gemm/flydsl_gemm_attribution.md` §7 — a K-sweep fit (S18),
  a trip-count additive probe (S19), or N and K swapped at equal FLOPs (S20).
- Kernel-argument preload — `gemm/flydsl_gemm_small_m.md` §6 — trims `F` at any M (S21); pass operands as
  `fx.Pointer` (S22) and confirm `.amdhsa_user_sgpr_kernarg_preload_length` > 0 in the ISA (S23).
- Keep the split-K fold in the GEMM launch, in a fixed order — `gemm/flydsl_gemm_small_m.md` §4 — race its form per
  M (S24); touch the fold's workspace lines with one used load before the K loop (S25).
- Split-K on latency-bound rows and single N-blocks — `gemm/flydsl_gemm_small_m.md` §4 — not on streaming rows or
  short K (S26, S27).
- Launch the `flyc.compile` object, not the `@flyc.jit` function, except under graph capture, where the raw
  `@flyc.jit` closure is the one to launch — `gemm/flydsl_gemm_autotune.md` §6 (S28, S29).
- Fewer launches by fusion — `optimize/flydsl_levers/flydsl_authoring_method.md` — for latency-bound small shapes (S30).
- Other kernel families: drop a graph node by folding a small kernel into its consumer — `moe/flydsl_moe_decode.md`
  §1–2; a decode attention's combine is a fixed cost, and when merging it into the producer pays —
  `attention/flydsl_attn_decode_split_kv.md` §1–2; the small kernels around MoE expert GEMMs —
  `grouped_gemm/flydsl_grouped_moe_inference.md` §7.

## Evidence

- S3 gemm/flydsl_gemm_small_m.md:46 one graph node's floor; at small M on a small weight more than half is the node
- S4 gemm/flydsl_gemm_small_m.md:233 preload removes the s_load → lgkmcnt(0) round trip before the first B load
- S5 gemm/flydsl_gemm_small_m.md:48 AKLR timer: empty kernel and a second launch each cost; keep the fold in the launch
- S6 gemm/flydsl_gemm_autotune.md:161 the jit call re-binds the signature, rebuilds the key, re-checks globals
- S7 gemm/flydsl_gemm_autotune.md:174 a dispatch layer calling the @flyc.jit closure pays host cost per call
- S8 gemm/flydsl_gemm_grid_pricing.md:232 F is prologue, epilogue, tail peel, setup, drain; share grows as K shrinks
- S9 gemm/flydsl_gemm_small_m.md:113 the reduction is a fixed cost that dominates at M=1
- S10 gemm/flydsl_gemm_small_m.md:78 streaming rows bound by HBM read of B; latency-bound by launch/reduction/latency
- S11 gemm/flydsl_gemm_measurement.md:126 keep the host out of the GPU window; flat time across sizes = host
- S12 gemm/flydsl_gemm_measurement.md:129 one call per CUDA/HIP graph measures the launch floor
- S13 gemm/flydsl_gemm_attribution.md:89 rocprofv3 kernel-trace start→end overstated a small kernel
- S14 gemm/flydsl_gemm_small_m.md:50 price aiter through the deployed entry; time your kernel through `run`
- S15 gemm/flydsl_gemm_small_m.md:84 count workgroups against the 256-CU dispatch round before anything else
- S16 gemm/flydsl_gemm_attribution.md:151 a K-sweep fit has five systematic errors
- S17 gemm/flydsl_gemm_small_m.md:72 the lessons flip sign between streaming and latency-bound
- S18 gemm/flydsl_gemm_grid_pricing.md:226 at fixed M and N, sweep K and fit t_tile = a · n_kblocks + F
- S19 gemm/flydsl_gemm_attribution.md:149 trip-count additive probe gives P and F
- S20 gemm/flydsl_gemm_attribution.md:145 same FLOPs, different tile counts: swap N and K
- S21 gemm/flydsl_gemm_grid_pricing.md:239 kernel-argument preload trims F at any M
- S22 gemm/flydsl_gemm_small_m.md:231 pass operands as fx.Pointer; a raw torch.Tensor stops the preload
- S23 gemm/flydsl_gemm_small_m.md:230 the dumped ISA shows .amdhsa_user_sgpr_kernarg_preload_length > 0
- S24 gemm/flydsl_gemm_small_m.md:123 race the in-launch fixed-order fold's form per M; no form won everywhere
- S25 gemm/flydsl_gemm_small_m.md:136 touch the fold's workspace lines with one used load before the K loop
- S26 gemm/flydsl_gemm_small_m.md:108 split-K pays on a single N-block and on latency-bound rows
- S27 gemm/flydsl_gemm_small_m.md:109 split-K does not pay on streaming rows or short K
- S28 gemm/flydsl_gemm_autotune.md:161 call the flyc.compile object, not the @flyc.jit function
- S29 gemm/flydsl_gemm_autotune.md:169 under CUDA-graph capture call the raw closure
- S30 optimize/flydsl_levers/flydsl_authoring_method.md:50 latency-bound (small shapes): reduce launch count, fusion
