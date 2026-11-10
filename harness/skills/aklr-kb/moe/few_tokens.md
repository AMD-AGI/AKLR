---
title: "MXFP4 fused MoE at 8–64 tokens per call: count the graph nodes, fold the routing sort into stage 1"
kind: technique
lever: moe_few_tokens
ops: [moe, fused_moe]
dtypes: [mxfp4]
regimes: ["tokens 8-64", small-m, launch-bound]
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json
---

# MXFP4 fused MoE at 8–64 tokens: one graph node less

The kernel this page reads from is the schema file `4fbd9d02b4c053c7.json` of
`fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256`: `fused_moe@1.0` with MXFP4 weights and
activations (E8M0 scale per 32), 257 experts, top-9, model dim 6144, inter dim 256, SiLU with a separate gate, at 8,
16, 32 and 64 tokens. Its sources are `kernel.py` (the entry), `g1_fold.py` (stage 1) and `g2_port.py` (stage 2);
line numbers below are in those files. "Tokens" (M in formulas) is the Point's `n_tokens`. Where it is served, and what the other
kernel of the same schema name does at 1–2 tokens: [fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md).
aiter's MoE kernels, their names and knobs: `../flydsl/kernels/flydsl_kernel_library.md` (MoE).

## What it does

A MoE call of 8–64 tokens (as a decode step makes) lasts a few tens of µs, and every launch in it costs a fixed
amount under graph replay. The
kernel drops the routing-sort launch: every stage-1 workgroup rebuilds, in LDS, the part of the sort it needs, and
the zeroing of the output moves into the same launch. Two graph nodes instead of three.

### 1. Count the graph nodes

- The unfolded source runs three nodes at these sizes: aiter's MXFP4 routing sort (per expert, padded to 16-row
  blocks; it also zeroes the output buffer), stage 1 (gate/up GEMM, activation, quantization of the intermediate)
  and stage 2 (down GEMM, weighted accumulation into the output) (S1, S2; read from the source and the commit, not
  from a trace).
- A call took 39.1–102.0 µs with three nodes and 34.4–95.7 µs folded *(field, E1)*.
- Under AKLR's timer up to measurement profile 18 (Kineto, queue profiling forced) a node that did almost no work
  still read about 4 µs: a fully empty kernel timed in place of a dense GEMM read 4.21 µs at m = 8 *(field, E2)*, and
  an add + RMSNorm kernel (a HIP kernel, not a MoE) read 3.93–4.04 µs at 1–256 rows of 6144 *(field, E3)*. Recheck
  it under profile 19 on current main; more on that floor: `../gemm/small_m.md` §1.
- Removing the sort node saved 3.5–6.4 µs per call, 6–12% of the call *(field, E1)*.

### 2. Fold the routing sort into stage 1

Two nodes instead of three:

- Every stage-1 workgroup (one per 16-row expert block × 256-column gate/up block) rebuilds the whole padded sort of
  the M × 9 routed items from `topk_ids` in LDS: a histogram of the experts, a wave scan of the per-expert block
  counts, the rank of each item of its own block's expert. The work is repeated in every workgroup and needs no grid
  barrier and no communication between workgroups (S3).
- The workgroup with n-block 0 of each expert block publishes that block's sorted token ids, routing weights and
  expert id for stage 2 (block 0 also the valid count), and the whole grid zeroes the output in stripes, since
  stage 2 adds into it with atomics (S4).
- Stage 1 quantizes its 16 gathered hidden rows to MXFP4 inline instead of reading a pre-quantized copy (S5).

Measured with the same inputs, one median per arm *(field, E1)*:

| tokens | 3 nodes (µs) | folded (µs) | vs aiter, 3 nodes | vs aiter, folded |
|---|---|---|---|---|
| 8 | 39.09 | 34.38 | 1.007 | 1.146 |
| 16 | 53.20 | 49.68 | 1.015 | 1.088 |
| 32 | 84.28 | 79.19 | 1.058 | 1.132 |
| 64 | 102.04 | 95.67 | 1.043 | 1.099 |

Lower bounds of the folded kernel under AKLR's validator outside CI, on FlyDSL 0.3.2: 1.122, 1.079, 1.099, 1.081
*(field, E4)*. The predecessor repository's CI (2026-10), measured on the CI image's FlyDSL 0.3.4.1 rather than the pinned 0.3.2
(`../pitfalls/measured_a_different_build.md`): 1.129, 1.087, 1.117, 1.090 *(field, E5)*; the file's current values differ.

## How to write it

The head is in `gemm1_kernel_fold` (`g1_fold.py:1870`), before and just after the bound check:

| Step | What | `g1_fold.py` |
|---|---|---|
| a | zero `out` as int32 words striped over the whole grid, surplus workgroups included | :1944–1950 |
| b | load the workgroup's `topk_ids` items and their routing weights into registers; padding items read −1 | :1938–1943, :1957–1959 |
| c | zero a 272-word LDS histogram while those loads fly, then one LDS atomic add per item | :1951–1956, :1965–1968 |
| d | per expert `ceil(count / 16)` blocks; an inclusive DPP scan per wave (`_dpp_scan_wave`), the four wave totals through LDS, an exclusive block offset per expert; expert 256 (the 257th) is placed last | :2010–2035, :252–261 |
| e | the thread whose expert owns this m-block writes the expert id and the block's index within the expert to LDS | :2108–2129 |
| f | rank this expert's items in item order (a DPP scan of `item == expert`), keep ranks [lo, lo + 16) as the block's rows: token ids, `(slot << 24) \| token` ids and routing weights | :2137–2176 |
| g | n-block 0 publishes the block's sorted ids, weights and expert id; block 0 also `num_valid_ids` | :2178–2209 |

- The device then computes `bound = total_m_blocks × 2` n-blocks (:2048); a workgroup at or past it exits, and at 32
  tokens the tile id is remapped per XCD only after that test (:2091–2097;
  `grouped_sorted_tokens.md` §2).
- The first two weight and weight-scale prologue stages are issued as soon as step e has the expert id, before the
  LDS-only steps f and g, so the head runs under the weight stream (`_BEF`, :219, :966–980). Its effect is not
  measured here.
- The token count is a compile-time constant (`M_CONST`, :1830): the sort loops unroll to ⌈9M / 256⌉ passes of 256
  threads (1–3 here), and the grid is the upper bound on 16-row blocks times 2 n-blocks (`gemm1_grid`, :314–324):
  144, 288, 518 and 554 workgroups at 8, 16, 32 and 64 tokens.
- Stage 1 asserts tile-M 16 with inline quantization for the fold (:1829).

In FlyDSL: `../flydsl/moe.md` § few_tokens How to write it.

## When it applies

- Token counts where a call lasts tens of µs and the sort is a node of its own: 8–64 tokens here, 34.4–95.7
  µs folded *(field, E1)*.
- The sort fits one 256-thread workgroup: 9M items (72–576 here); one thread per expert in the scan, with expert
  256 counted apart (:2011–2014), so up to 256 + 1 experts as written.
- Tile-M 16 and inline activation quantization in stage 1 (:1829). One compile per token count, which suits AKLR's
  discrete Points; a server calling arbitrary token counts would compile per count.
- A bounded contract: stage 2 adds the weighted rows into `out` with bf16x2 atomics (`g2_port.py:899–915`).
  `fused_moe@1.0` is bounded (`aklr/ops/fused_moe/contract.py`, its `compare`).

## Counter-signals

- **1 and 2 tokens.** The kernel is not built there (`kernel.py:61`). The other file of the schema name,
  `1fee59821837c26f.json`, runs two launches with no sort at all, one set of workgroups per route, and was served at
  2.472 and 1.990 in the predecessor repository's CI (2026-10) *(field, E6)*.
- **4 tokens, and 128 or more.** Not built folded, so not measured. The source had no folded build at 4 tokens, and
  its unfolded path calls aiter's C++ MXFP4 sort, which an AKLR kernel does not call (S2). At 128 or more the
  replicated sort grows with 9M items, and it runs in every workgroup of a grid that grows with M (estimated).
- **A sort node that is already one small kernel.** *(field, E8)* On an fp8 × MXFP4 MoE (896 experts, top-16)
  whose sort, token quantization, scale scatter and output zeroing ran as one 3.9–5.4 µs kernel, folding lost
  0.2–2.3% at 8–32 tokens; it paid (0.4–4.1% at 8–16 tokens) only once stage 1 loaded and quantized just its
  block's live rows.
- **A wide stage 1.** On an a8w4 MoE with 112 of 896 experts local and inter 3072 (48–96 n-blocks per row) the fold
  lost: 80.7 against 45.7 µs at 8 tokens; on the exact m-block grid, 43.4 against 42.6 for sort + stage 1 *(field, E10)*.
- **A separate counting sort.** A counting-sort kernel in place of the MXFP4 sort node is untested; aiter's FlyDSL one
  (`aiter/ops/flydsl/moe_sorting.py`) is a drop-in for `moe_sorting_fwd`, not for `aiter.mxfp4_moe_sort` (S2).
- **What aiter runs there.** The baseline aiter's table picks its inline-quantizing `f16in` stage 1 for 4–128
  tokens of this shape and a pre-quantized-activation stage 1 at 1 and 2 (S6).
- **A bitwise contract.** The atomic stage 2 adds a token's 9 rows in arrival order. Use a fixed-order stage 2 there,
  for example one wave per route and an LDS sum in route order, as `1fee59821837c26f.json` does at 1–2 tokens
  (`grouped_sorted_tokens.md` §6, `../gemm/correctness.md` §8).

## How to confirm it took effect

- A kernel trace of one call shows two kernels, `gemm1_a4w4_fold_…_fold<M>` and
  `gemm2_a4w4_port_…_atomic_g2bs2_pro125` (names built at `g1_fold.py:1766–1831`, `g2_port.py:131–135`), and no
  `mxfp4_moe_sort` or `moe_sorting` kernel.
- With `AKLR_EXPORT=1`, the records of calls to the op's api at these Points carry an empty `fallback_reason`
  (`../pitfalls/measured_a_different_build.md` § How to detect).

### 3. Equivalence under atomics

Stage 2's atomic accumulation makes the output differ run to run (`determinism: bounded`, whole-output SQNR bound
13 dB). Call a port numerically equivalent to its source when their cross-SQNR lies within the port's own run-to-run
SQNR. Measured on 3 input sets per token count *(field, E7)*: folded port vs aiter 45.9–49.9 dB, the source vs aiter
45.8–51.9 dB, port vs source 48.5–51.5 dB, port run 2 vs run 1 48.4–51.9 dB.

## Cost and risk

- **Repeated work.** Every stage-1 workgroup, those past the device bound included, loads the items, builds the
  histogram and runs the scans: 144–554 workgroups at 8–64 tokens. The head's own cost is not measured here; on the
  896-expert fold above, running the sort chain twice added 0.5–1.5%, and a barrier-free per-wave sort did not gain *(field, E9)*.
- **LDS.** (352 + 256 · ⌈9M / 256⌉) words of sort scratch, 2.4–4.4 KiB, beside the GEMM's stages (`_sort_words`,
  `g1_fold.py:238–240`).
- **Idle HBM during the head**, until step e names the expert; the early weight issue above is the mitigation.
- **Output bits differ run to run** (§3); bounded contracts only.
- **Compiles.** One stage-1 compile per token count; stage 2 is compiled once for all four (`kernel.py:42–55`).

## Evidence

`kernel.py`, `g1_fold.py` and `g2_port.py` are sources of the schema in `sources`; their line numbers are the file
contents' lines.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, bench_cudagraph: CUDA-graph replay with a 1 GiB read flush before each call, Kineto kernel time, median; one run per arm, aiter re-timed in each arm] the source with the fold off at these counts (three nodes) against the fold on: 8 tokens 39.086 vs 34.376 µs, 16 53.201 vs 49.683, 32 84.284 vs 79.193, 64 102.04 vs 95.673; aiter / candidate 1.007 vs 1.146, 1.015 vs 1.088, 1.058 vs 1.132, 1.043 vs 1.099 (257 experts, inter 256, top-9, the bench's input sets)
- E2 [2026-10, gfx950, FlyDSL ?, N=?, AKLR bench_cudagraph in an in-process harness] a fully empty kernel, timed in place of the GEMM, read 4.21 µs at m = 8 (a dense bf16 GEMM forge run, gemm_n2112_k7168)
- E3 [2026-10, gfx950, HIP, N=5, bench_cudagraph: graph replay, 1 GiB flush per call, median of 5 rounds] an add + RMSNorm HIP kernel read 3.93–4.04 µs at m 1, 2, 4, 8, 12, 16, 32, 64, 128, 256 with hidden 6144; aiter's 4.04–4.46 µs (add + RMSNorm)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR validator bench stage outside CI: one-sided lower bound over graph-replay rounds, L2 flushed per call] 1.1220, 1.0786, 1.0987, 1.0805 at 8, 16, 32, 64 tokens (an earlier schema, the same sources before the `author` field)
- E5 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, the predecessor repository's CI, 2026-10; the file's current values differ] speedup over aiter 1.129, 1.087, 1.117, 1.090 at 8, 16, 32, 64 tokens (fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7)
- E6 [2026-10, gfx950, FlyDSL 0.3.4.1, N=1, the predecessor repository's CI, 2026-10; the file's current values differ] speedup over aiter 2.472 at 1 token and 1.990 at 2 (fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/1fee59821837c26f)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the bench's input sets per token count, SQNR of the bf16 outputs] port vs aiter 45.88–49.89 dB, source vs aiter 45.85–51.88, port vs source 48.49–51.53, port run 2 vs run 1 48.35–51.87 at 8, 16, 32, 64 tokens
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, graph-timer driver bench (warmup 3, 20 iterations), arms back to back in one session; prelude time from Kineto] separate prelude 3.9 / 4.7 / 4.7 / 5.1 / 5.4 µs at 8 / 12 / 16 / 24 / 32 tokens; folded with every row of the block quantized: 54.37 / 72.75 / 87.59 / 116.84 / 141.47 against 53.40 / 72.03 / 87.39 / 115.39 / 138.34 µs; folded with only rows 0–15 loaded and quantized into LDS (plus a swizzle at 8 tokens and one weight tile in flight at 12): 51.29 / 71.14 / 87.70 against 53.46 / 72.72 / 88.07 µs at 8 / 12 / 16 tokens; all 32 rows quantized into LDS read 58.62 / 84.02 / 93.88 against 51.82 / 72.08 / 89.46 µs (Kineto); the kept fold +1.07% of the score in a paired ablation (fp8-activation × MXFP4 SiTUv2 fused MoE, 896 experts, top-16, inter 384, model dim 3584)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2, graph-timer driver bench (warmup 3, 20 iterations), one session] the sort chain (mask clear to placement barrier) run twice per stage-1 workgroup: +0.64 / +0.94 / +0.39 / +1.60 / +1.09 µs over 49.86 / 67.78 / 83.09 / 109.90 / 135.88 µs at 8 / 12 / 16 / 24 / 32 tokens; a per-wave sort with ballot, DPP prefix and no barrier on the load path: 50.15 / 67.67 / 83.38 / 110.92 / 136.38 µs, the same in a second bench (same kernel as E8)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer (bench_cudagraph, L2 flushed per replay), median of the readings; per-kernel times from Kineto after a flush] sort, token quant and `out` zeroing folded into stage 1, grid y = min(upper-bound m-blocks, 112): 80.66 / 83.84 / 110.13 / 128.70 / 175.01 µs against 45.66 / 62.58 / 97.07 / 119.84 / 169.06 at 8 / 12 / 16 / 24 / 32 tokens; stage 1 at 8 tokens 78.9 µs folded, 43.4 with an exact host-computed grid, against 38.0 + 4.6 µs sort (a8w4 SiTUv2 MoE, 112 of 896 experts local, top-16, model dim 3584, inter 3072)
- S1 g1_fold.py:131 the source's note: the fold does, inside stage 1's block head, the work of a separate `moe_sort_quant` graph node, the 257-expert padded sort and the zero-init of the [M, D_HIDDEN] bf16 `moe_buf`; two FlyDSL kernels instead of aiter's sort + stage 1 + stage 2
- S2 the AgentX deployment's fused-MoE seam (not public) folds only at M in {6, 12, …, 42} and otherwise calls aiter's `_fused_moe_impl`, whose sort is `aiter.mxfp4_moe_sort` (`aiter/fused_moe.py:161` of the baseline); the baseline's `aiter/ops/flydsl/moe_sorting.py:4-8` is a drop-in for `moe_sorting_fwd`, with the same signature
- S3 g1_fold.py:1938 head steps b–f; histogram atomics at :1968, scans at :2018 and :2146
- S4 g1_fold.py:1944 output zeroed over the grid (`_NDW` words, :1855–1856); publish at :2179–2209; stage 2 atomics at g2_port.py:908
- S5 g1_fold.py:768 `_inline_quant_core_batch`: amax over 8 bf16 values, DPP quad max, E8M0 round-up, `cvt_scalef32_pk_fp4_bf16` into the swizzled LDS A tile (:807–819)
- S6 aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv:52 tokens 4–128 at model_dim 6144, inter_dim 256, 257 experts: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` (rows 52–57); tokens 1 and 2: `flydsl_moe1_afp4_wfp4_bf16_…_fp4` (rows 50–51)

## Related

`grouped_sorted_tokens.md` (how MoE expert GEMMs map work; cache policy; stage 2) ·
`../flydsl/kernels/flydsl_kernel_library.md` § AKLR's own FlyDSL MoE kernels ·
[fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md) · `INDEX.md`
