---
title: "Small-M launch floor: an M ≤ 256 row bound by launch, reduction and latency, not by reading B"
kind: pattern
signals: ["M ≤ 256 on a row bound by launch, reduction and memory latency", "a large per-tile intercept F in a K-sweep fit",
  "one CUDA-graph node or an empty kernel costs a large share of the measured time",
  "s_load then lgkmcnt(0) in front of the first B load", ".amdhsa_user_sgpr_kernarg_preload_length = 0"]
ops: [gemm, grouped_gemm, fused_moe, attention]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
---

# Small-M launch floor: bound by launch, reduction and latency

## Symptom

- M ≤ 256 (as in decode or speculative verify) on a row bound by launch, reduction and memory latency rather than by the HBM
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
4. **The split-K reduction,** a cost set by its protocol (S9): a zeroing node or a second reduce launch adds about one
   graph-node floor per call, a fifth to under half of an M = 1 call in a Triton split-K GEMM, while atomics into C with
   no zeroing node in the call add 0.1–0.3 µs *(field, E1)*.
5. **A latency-bound row:** bound by launch, reduction and memory latency rather than the HBM read of B (S10).

## Rule out first

- **The host in the window.** Time R consecutive calls inside one event pair; time flat across sizes can mean the
  host or a launch floor is being measured (S11), or a grid that stays the same across sizes (S31). One call per
  CUDA graph reads the launch floor, not the kernel, for a kernel well under that floor (S12).
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
- Kernel-argument preload — `gemm/small_m.md` §6 — can trim `F` past small M (S21), but not on every kernel or at
  every M: it was slower at some M on other kernels (S32); time it per M bucket. Pass operands as
  `fx.Pointer` (`flydsl/gemm.md` § small_m §6, S22) and confirm `.amdhsa_user_sgpr_kernarg_preload_length` > 0 in the ISA (S23).
- Keep the split-K fold in the GEMM launch, in a fixed order, done by the last split to arrive on an atomic counter
  — `gemm/small_m.md` §4 — so that no workgroup waits for another (S24); a bounded wait with a hand-off on timeout
  is another safe form (`attention/split_kv.md` §2, Bounded wait), not measured on a GEMM fold.
- Split-K on latency-bound rows and single N-blocks (S26) — `gemm/small_m.md` §4 — not on a row whose split-1 kernel
  already reads B near HBM bandwidth, which its shape alone does not tell (S27, `low_cu_utilization.md` E1, E2); on short K, race it with the zeroing
  or reduction launch counted (E2 there).
- Launch the `flyc.compile` object, not the `@flyc.jit` function (S28). Under graph capture launch the raw closure or a
  `flyc.compile` object built before the capture, not one first built inside it — `flydsl/gemm.md` § autotune §6,
  `flydsl/moe.md` § grouped_host_autotune §2 (S29).
- Fewer launches by fusion — `../flydsl/kernels/flydsl_authoring_method.md` — for latency-bound small shapes (S30).
- Other kernel families: drop a graph node by folding a small kernel into its consumer — `moe/few_tokens.md`
  §1–2; a split-KV attention's combine is a fixed cost, and when merging it into the producer pays —
  `attention/split_kv.md` §1–2; the small kernels around MoE expert GEMMs —
  `moe/grouped_sorted_tokens.md` §7.

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=11, CUDA graph of ≥ 64 calls per timing, arms shuffled every round] reduction cost over an arm in which each split stores its own fp32 partial and nothing is reduced, at M = 1, split-K 4 and 8, three N×K: atomics into C with no zeroing node 0.12–0.29 µs, a `C.zero_()` node before the atomic GEMM 1.64–2.19 µs (21–43% of the call), a second launch summing the partials 1.73–2.32 µs (23–45%) (Triton split-K GEMM, bf16 in, fp32 accumulate)
- S3 gemm/small_m.md:50 one graph node's floor; at M ≤ 16 on a small weight about a third to a half of the time, more on a GEMM tuned to ~3 µs
- S4 gemm/small_m.md:398 preload removes the s_load → lgkmcnt(0) round trip before the first B load
- S5 gemm/small_m.md:53 AKLR's timer up to profile 18: an empty kernel and a second launch each cost; keep the fold in the launch
- S6 flydsl/gemm.md:477 the jit call re-binds the signature, rebuilds the key, re-checks globals
- S7 flydsl/gemm.md:486 a dispatch layer calling the @flyc.jit closure pays host cost per call
- S8 gemm/grid_pricing.md:301 F is prologue, epilogue, tail peel, setup, drain; share grows as K shrinks
- S9 gemm/small_m.md:143 one FlyDSL kernel's zero-C + spin + atomic protocol cost 1.5–2.5 µs per launch at M=1
- S10 gemm/small_m.md:93 streaming rows bound by HBM read of B; latency-bound by launch/reduction/latency
- S11 gemm/measurement.md:137 keep the host out of the GPU window; flat time across sizes can be the host or a launch floor
- S12 gemm/measurement.md:141 one call per CUDA/HIP graph reads the launch floor for a kernel well under it
- S13 gemm/attribution.md:93 rocprofv3 kernel-trace start→end overstated a small kernel
- S14 gemm/small_m.md:65 price aiter through the deployed entry; time your kernel through `run`
- S15 gemm/small_m.md:99 count workgroups against the 256-CU dispatch round before anything else
- S16 gemm/attribution.md:157 a K-sweep fit has five systematic errors
- S17 gemm/small_m.md:87 the lessons flip sign between streaming and latency-bound
- S18 gemm/grid_pricing.md:295 at fixed M and N, sweep K and fit t_tile = a · n_kblocks + F
- S19 gemm/attribution.md:155 trip-count additive probe gives P and F
- S20 gemm/attribution.md:151 same FLOPs, different tile counts: swap N and K
- S21 gemm/grid_pricing.md:308 kernel-argument preload can trim F past small M: −0.2 to −0.6% at M ≈ 900–16k on one kernel
- S22 flydsl/gemm.md:393 pass operands as fx.Pointer; a raw torch.Tensor stops the preload
- S23 gemm/small_m.md:394 the dumped ISA shows .amdhsa_user_sgpr_kernarg_preload_length > 0
- S24 gemm/small_m.md:145 no result may depend on a split that is not running; the last split to arrive folds
- S26 gemm/small_m.md:134 split-K pays on a single N-block and on latency-bound rows
- S27 gemm/small_m.md:135 split-K does not pay on streaming rows or short K
- S28 flydsl/gemm.md:477 call the flyc.compile object, not the @flyc.jit function
- S29 flydsl/gemm.md:482 Turbo launches the raw closure under CUDA-graph capture: "a flyc.compile'd object regresses under CUDA-graph capture"; built before the capture on FlyDSL v0.3.2 it did not
- S30 ../flydsl/kernels/flydsl_authoring_method.md:50 latency-bound (small shapes): reduce launch count, fusion
- S31 gemm/grid_pricing.md:113 a small-M example: 128×128 tiles with split-K 1 launch the same 47 workgroups for every M ≤ 128
- S32 gemm/small_m.md:403 preload was not faster on every kernel: slower at m=16 (and 32) on two kernels, sign flipped between runs on a third
