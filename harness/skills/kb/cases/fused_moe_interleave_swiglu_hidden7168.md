---
title: "MXFP4 fused MoE with clamped SwiGLU on gate/up-interleaved weights, hidden 7168, top-6 (384 experts at inter 384 or 768, or 96 local of 384 at inter 3072) — two FlyDSL launches on MX-fp8 activations with the routes taken one per workgroup row at 1–4 tokens and grouped by expert from 8 tokens (inter 384) or 16"
kind: case
ops: [moe]
dtypes: [mxfp4, bf16, fp8]
regimes: ["tokens 1-4", "tokens 8-64"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=4 (on the predecessors; the shipped files are not measured yet)
measured:
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point, of the predecessor (the GPU code of 7856e34c973b92a8 at m 16, 32; a fixed-scale fp8 intermediate at m 1, 2)", value: "1.129–5.140; 4 of 4 Points ≥ 1.03 (m 1, 2, 16, 32)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384, m 1, 2, 16, 32", run_ref: "base 0e02a81", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point, of the predecessor (the GPU code of e18b982b504d1626 at m 1–64)", value: "1.081–3.808; 7 of 7 Points ≥ 1.03 (m 1, 2, 4, 8, 16, 32, 64); also 1.030 at m 128, which e18b982b504d1626 does not list", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384, m 1, 2, 4, 8, 16, 32, 64, 128", run_ref: "base 0e02a81", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point, of the predecessor (the GPU code of 93aeaecbfa4c7956)", value: "1.151–3.295; 6 of 6 Points ≥ 1.03 (m 1, 2, 4, 16, 32, 64)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768, m 1, 2, 4, 16, 32, 64", run_ref: "base 0e02a81", date: "2026-11-09"}
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts96_topk6_inter_dim3072_global_experts384/7856e34c973b92a8.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/e18b982b504d1626.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim768/93aeaecbfa4c7956.json
---

# MXFP4 fused MoE with clamped SwiGLU, gate/up-interleaved weights, hidden 7168, top-6

DeepSeek-V4-Pro's MoE (hidden 7168, 384 routed experts, top-6: `aklr/benchmarks/models/deepseekv4/README.md`)
under three schema names, one kernel file each: all 384 experts local at inter dim 384 (`e18b982b…`) and 768
(`93aeaecb…`), and expert parallel with 96 local experts of 384 at inter dim 3072 (`7856e34c…`). Each embeds one
`kernel.py`; the function names below are in it. "m" is the Point's `n_tokens`: the tokens in the call, the input
batch, not the rows an expert receives. A route is one (token, expert) pair, 6m per call.

Each file has two predecessors. The later ones (of `7856e34c…`, `e18b982b…` and
`93aeaecb…`) allocated their scratch in every call, as the shipped files do; the earlier ones
before them allocated it once per shape (Counter-signals). `e18b982b…` and `93aeaecb…` run the GPU code
of their predecessors at every Point they list; `7856e34c…` runs that of its predecessor at m 16 and 32, and at m 1 and
2 it scales the intermediate per 32 values where that predecessor used one scale derived from the limit
(Counter-signals). The shipped files are not measured yet: the page cites runs of the predecessors as such, and the
gains of single changes come from their optimization ladders.

The weight layout, gate mode, clamp, comparison and baseline are those of DeepSeek-V4-Flash's hidden-4096 kernels
([fused_moe_interleave_swiglu.md](fused_moe_interleave_swiglu.md) § Applies when); hidden 7168 has its own schema
names and kernels.

## Applies when

- Contract `fused_moe@1.0`, layout variant `gate_up_interleaved`: `w1` (experts, 2 × inter, 7168) with gate and up
  rows interleaved in 16-row tiles, `w2` (experts, 7168, inter), fp4 e2m1 with E8M0 scales per 1×32 stored 2-D
  (`aklr/ops/fused_moe/contract.py`). `gate_mode` interleave with `swiglu_limit`, bf16 `hidden_states` and `out`,
  fp32 `topk_weights`, int32 `topk_ids`, routing weight in stage 2. Every workload passes `swiglu_limit` 10.0.
- Expert parallel (`7856e34c…`): `expert_mask` over 384 global experts, 96 local; ids are global. No kernel here
  takes `num_local_tokens`: the `has_num_local_tokens` names of this shape (m 1024 and 2048) have no kernel.
- A token's expert ids are distinct (the `topk_ids` contract), and `93aeaecb…` relies on it when it ORs a token's bit
  into each of its experts' bitmaps.
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, whole-output SQNR ≥ 13 dB. The reference keeps the
  activations unrounded in this gate mode (`aklr/ops/fused_moe/reference.py`, its docstring); all three kernels
  quantize them to MX-fp8 (Counter-signals).
- Points listed, one kernel file per name: m 1, 2, 4, 8, 16, 32, 64 (inter 384, `e18b982b…`); m 1, 2, 4, 16, 32,
  64 (inter 768, `93aeaecb…`); m 1, 2, 16, 32 (96 local, `7856e34c…`). They are served once CI records a speedup
  ≥ 1.03 for them (Result): the files carry no `speedup` yet, so dispatch serves none of them before that. Each
  name's other file (`5587cccd…`, `3cd667a8…`, `1a61a10a…`) has empty `sources`, a definition that is not served
  (`aklr/ops/README.md` § Schema fields), so the yardstick is aiter; it defines the same Points, and `5587cccd…`
  also m 128 and 256. Not covered: m 128 and 256 at inter 384 (Counter-signals), the expert-parallel name with 48
  local experts (`a6c17379…`, m 1–128) and the `has_num_local_tokens` names, which have no kernel.
- The baseline is aiter's `fused_moe` with `AITER_BF16_FP8_MOE_BOUND=16` (`aklr/ops/fused_moe/baseline.py`):
  bf16 activations below 16 rows, fp8 from 16.

## Counter-signals

- **m 128 and 256 at inter 384 are not listed.** `e18b982b…` takes at most 64 tokens. Its predecessor
  listed m 128 and read 1.030 there, at the serving threshold *(field, E2)*.
- **Output bits vary where stage 2 adds with atomics.** `93aeaecb…` adds into `out` with packed bf16 atomics at m 1,
  2 and 16–64. `7856e34c…` and `e18b982b…` at every Point and `93aeaecb…` at m 4 sum each token's routes in a fixed
  order. Under a bitwise contract the atomic Points need a fixed-order stage 2 (`../moe/grouped_sorted_tokens.md`
  §6).
- **Activations and the intermediate pass through fp8.** All three quantize the token rows to fp8 e4m3 with an E8M0
  scale per 32 values and run stage 1 on the scaled MFMA. Stage 1 stores SiLU(gate)·up in bf16 (`93aeaecb…` at 1 and
  2 tokens and `e18b982b…` at 2 and 4 store fp32 gate and up partials that stage 2 finishes); `e18b982b…` and
  `93aeaecb…` at every Point, and `7856e34c…` at 1 and 2 tokens, requantize it to MX-fp8 in stage 2, while
  `7856e34c…` at 16 and 32 tokens keeps it bf16 (bf16 MFMAs on w2 converted from fp4). Each E8M0 scale comes from
  the absolute maximum of its own 32 values, so none depends on `swiglu_limit`, which the contract allows from 7 up
  to float32's largest value. The predecessor of `7856e34c…` instead stored the intermediate at 1 and 2 tokens as fp8
  under one scale per call derived from the limit (|y| ≤ limit · max(limit, 0.28)), for a limit in (0, 65536], which
  rounds a value below about limit² · 2⁻¹⁷ to zero: at limit 10, gate and up of 1/64 everywhere (y ≈ 1.2e-4) give an
  all-zero intermediate, and at limit 1000 most values from the bench's own inputs round to zero (estimated). The
  predecessors passed every listed Point with a worst violation fraction of 0 in four runs, all at limit 10, the
  only limit the workloads pass *(field, E1, E2, E3, E4, E9, E10)*; the SQNR itself is not recorded.
- **Scratch cached per shape is shared by concurrent calls.** `aklr/ops/README.md:180–181` allows scratch allocated
  once per shape in a compile function if every call rewrites it before reading it; that is safe only while one
  stream calls a Point at a time, since the buffer belongs to every call of that shape and two calls at once on two
  streams overwrite each other's intermediates. The earlier predecessors did this: two concurrent calls with
  different inputs gave wrong output in most tries from 2 to 8 tokens up, depending on the file and on whether the
  streams shared CUs *(field, E11)*, while a second call with the same inputs, as KernelForge's stress check runs it
  (`harness/scripts/forge_task/assets/aklr_forge_driver.py:456–465`), writes the same values and passed *(field,
  E11)*. The shipped files allocate their scratch in every call (`7856e34c…`: local slots, `Y` and the stage-2 token
  order; `e18b982b…`: `Y` and the split-K partials; `93aeaecb…`: `Y` or its fp32 partials and the group table);
  `_build` keeps only the compiled launch and its shape constants. Their predecessors,
  which allocate scratch the same way, passed with no differing try and no hang: the driver's
  three conditions (a stream limited to 16 and to 64 CUs, a second call with the same inputs; `:380–381`,
  `:437–469`) plus replay of one CUDA graph, 20 tries each per Point; one CUDA graph replayed with three input sets
  copied into its buffers in turn, 24 replays per Point; and two calls with different inputs at once on two
  streams, unrestricted and both on the same first 64 CUs, 20 tries each per Point *(field, E8)*. Not tested: the
  shipped files themselves, a full-model decode graph, more than two concurrent streams, and the host cost of the
  per-call allocations in eager mode (the timer sums GPU kernel time and replays a graph, so it does not see them).
- **Group ranks follow arrival order in `93aeaecb…` at 32 and 64 tokens.** `_groups` ranks each route on its expert
  with an LDS counter, so which routes share a group changes between runs; each route keeps its own MFMA row, and
  stage 2 adds with atomics anyway. At 16 tokens the persistent stage 1 ranks from per-expert token bitmaps instead.
- **Token counts outside the Points.** `7856e34c…` takes up to 32 tokens (it asserts at most 256 routes and 128
  local experts on its tile path), `e18b982b…` up to 64, `93aeaecb…` 1, 2, 3, 4, 16, 32 and 64 tokens without a
  mask; `run` raises on any other call. m 3 of `93aeaecb…` and the counts between Points run at no Point and are not
  validated.

## Mechanism

The gains this page gives for single changes (E5–E7) are steps along the optimization ladders that produced the
predecessors, every version timed against the version before it; they are not leave-one-out ablations of the final
kernel. E5–E7 say which steps were timed a second time.

What the three share:

- **MX-fp8 activations on the scaled MFMA.** The token row is quantized to fp8 e4m3 with one E8M0 scale per 32 values,
  chosen so the group's absolute maximum stays inside e4m3, by the lanes that hold it (`cvt_scalef32_pk_fp8_bf16` or
  a multiply by the inverse scale), and stage 1 multiplies it with the fp4 weights on
  `mfma_scale_f32_16x16x128_f8f6f4`, both E8M0 scales as operands (`../gemm/mainloop.md` §5).
- **The interleaved weight pair in one load pair.** Each 16-column tile of stage 1 reads the gate and the up tile of
  the interleaved `w1` with one 16-byte load per lane each, and one scale dword holds the gate and up bytes of two
  K chunks, as in the hidden-4096 kernels ([fused_moe_interleave_swiglu.md](fused_moe_interleave_swiglu.md) §
  Mechanism). The clamp and SiLU(gate)·up (`exp2`, `rcp`) run in stage 1's epilogue, or in stage 2 where stage 1
  leaves fp32 partials.
- **Two launches** up to 64 tokens, with no routing launch but `93aeaecb…`'s group table at 32 and 64: the expert of
  a route is read from `topk_ids` (under expert parallelism its local slot counted from the mask with ballots and
  `ctpop`, `_local_expert`). Weight loads are non-temporal in part: `7856e34c…` stage 1's w1 from 2 tokens, its
  stage 2 reading w2 under the default policy; `e18b982b…` stage 1's from 2 tokens (grids of more than 256
  workgroups) and stage 2's at m 2, 4, 16 and 32 only; `93aeaecb…` both stages' at every Point. Every launch takes its
  leading kernel arguments preloaded in SGPRs (`amdgpu-kernarg-preload-count` 16, or `llvm.inreg` argument
  attributes in `7856e34c…`; `../gemm/small_m.md` §6), and launch constants are computed in `_build`, not in the
  `@flyc.jit` launcher. Moving `7856e34c…`'s stage-1 grid size out of the launcher is worth +28.9% at m 1 and +8.4%
  at m 2 *(field, E5)*: a step that was not re-timed, and what made it that large is not established.

**1–4 tokens: one route per workgroup row, K split over waves or workgroups.**

- `e18b982b…` (inter 384), m 1, 2, 4. Stage 1: workgroup (16-column tile, route), 4 waves splitting K = 7168; at m 2
  and 4 each route's last 10 or 4 tiles run as two K halves in separate workgroups, leaving fp32 gate and up sums for
  stage 2 to fold, meant (per the code comment) to keep some CUs from running one tile more than the rest; the
  load-ahead depth is set per grid size. Stage 2: workgroup (64 or, at m 4, 128 output columns, token), one wave per
  route; the token's SwiGLU intermediate rows (`Y`, one per route) are quantized to MX-fp8 in LDS while every wave's w2 loads are in flight, and the routes'
  weighted sums are added in route order. Staging the token row first, with the load-ahead depth set per grid size,
  is worth +16.8, +20.8 and +10.7% at m 1, 2, 4 *(field, E6)*.
- `93aeaecb…` (inter 768), m 1 and 2: stage 1 splits K evenly over the workgroups of a token (each runs the K tail of
  one tile and the head of the next as two MFMA rows) into fp32 gate/up partials, and zeroes `out`; stage 2,
  workgroup (columns, route), stages the route's SwiGLU of those partials in LDS as MX-fp8 once for all its waves,
  and each wave adds its routing-weighted tiles into `out` with packed bf16 atomics. m 4: stage 1 per (route,
  16-column tile) with 7 waves splitting K; stage 2 per token, one wave per route, each staging its row in LDS as
  MX-fp8, the weighted partials added in route order. Running both GEMMs on the scaled MX-fp8 × MXFP4 MFMA is worth
  +9.5, +6.1 and +3.1% at m 1, 2, 4, and the even K split with the atomic stage 2 +7.6 and +2.1% at m 1, 2
  *(field, E7)*.
- `7856e34c…` (96 local of 384, inter 3072), m 1 and 2: stage 1 packs the live routes first (ballot over the local
  slots), one workgroup row per live route, 4 waves splitting K, each wave quantizing its slice of the token row into
  LDS; rows past the live count exit. A workgroup writes 16 columns of a route's `Y` in bf16, half of a 32-value MX
  block, so the block's scale is taken where the whole block is loaded, in stage 2. Stage 2 (`_stage2_tm`): one
  workgroup per 16 output columns walks the live routes of all tokens with 12 waves, so every workgroup carries the
  same work however the routes fall across tokens; it is compiled with a `waves_per_eu` hint of 6, set (per its
  comment) for two such workgroups per CU, an occupancy not measured. Each wave loads its K range of every live
  route's `Y`, 16 values per lane, quantizes it to MX-fp8 in LDS with `_quant_fp8` (a block's absolute maximum taken
  over two lanes, its E8M0 scale stored beside the data) and runs scaled MFMAs on it, each lane passing its block's
  scale. Giving each stage-2 wave a K range of every route with `Y` in LDS, in a per-token stage 2, is worth +5.9
  and +20.4% at m 1 and 2, and one stage-2 workgroup per column block over every token a further +0.9 and +4.6%
  *(field, E5)*; the per-32 scales are not measured yet.

**8–64 tokens: the routes of one expert grouped, so its weight tile is read once per group.**

- `e18b982b…`, m 8–64: from 48 routes a route leads a group when its rank among its expert's routes is a multiple
  of 4, and its workgroup runs up to 4 routes of that expert as MFMA rows (the others' workgroups exit); stage 2 as
  above with 32-column workgroups, in an XCD-grouped order at 32 and 64 tokens.
- `93aeaecb…`, m 16, 32, 64: groups of 2 (16 tokens) or 4 routes of one expert. At 32 and 64 a one-workgroup launch
  writes the group table; at 16 a persistent stage 1 on twice the CU count builds the same table in LDS in every
  workgroup from per-expert token bitmaps and also writes it for stage 2. Stage 1 runs each (16-column tile, group)
  with MFMA row g on the group's route g; stage 2 runs (columns, group) and adds into `out` with packed bf16 atomics.
  Moving m 32 and 64 from aiter to the grouped kernels is worth +7.5 and +8.0% there *(field, E7)*.
- `7856e34c…`, m 16 and 32: stage-1 workgroup row = one live local expert, its routes in groups of 16 over four
  16-column tiles, K split over 2 waves at 16 tokens; the token rows go through an LDS ring padded to keep
  `ds_read_b128` free of bank conflicts. Stage 1 also writes a stage-2 token order: tokens sorted by the first token
  that shares one of their experts, which brings some tokens sharing an expert next to each other but does not group
  them strictly; its comment's intent is that the second read of that expert's w2 hits in cache. The order is worth
  +2.4% at m 32 *(field, E5)*. Stage 2 (`_stage2_cr`): workgroup (two 16-column blocks, token); wave w walks its K
  range of every live route with `Y` in LDS, the routes summed in order in each wave and the waves in wave order.
  Folding the routing into stage 1 and moving m 16 from aiter onto this path is worth +12.3% at m 16 *(field, E5)*.

## Result

`python -m aklr validate` of the three predecessors on an MI355X at main
0e02a81, measurement profile 19: the published `speedup` is a one-sided Student-t lower bound of the mean log ratio of
aiter's time to the kernel's (α = 0.025 per look), read at the first look, 16 or 32 rounds, that decides every verdict
(in this run every Point at 16), of CUDA-graph replays with 1 GiB read before each call to flush L2
(`MEASUREMENT_PROFILE` and `TimingRecord.speedup_bounds` in `aklr/core/validation/stages/_measure.py`). A call's time
is `gpu_duration_sum`: the summed durations of the GPU kernels it launches, not wall time. FlyDSL 0.3.2, aiter
0.1.21.dev66+gdbd8bf5bd, torch 2.11.0+rocm10.0.0; audit and bench passed with no finding. The shipped files are not
measured yet: `e18b982b…` and `93aeaecb…` differ from these predecessors only in host code, and `7856e34c…` keeps
its predecessor's GPU code at m 16 and 32 but not at m 1 and 2. CI writes `speedup` into the shipped files once they
merge; dispatch serves a Point from 1.03 of that number (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment
variables).

| Predecessor (of the shipped file) | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 96 local of 384, inter 3072 (`7856e34c…`) *(field, E1)* | 5.140 | 1.890 | – | – | 1.235 | 1.129 | – |
| 384 experts, inter 384 (`e18b982b…`) *(field, E2)* | 3.808 | 3.155 | 2.537 | 1.868 | 1.578 | 1.205 | 1.081 |
| 384 experts, inter 768 (`93aeaecb…`) *(field, E3)* | 3.295 | 2.391 | 1.871 | – | 1.240 | 1.151 | 1.172 |

"–": no workload of that name. Every Point passed accuracy, at limit 10. At m 1 and 2 the predecessor of `7856e34c…` ran its fixed-scale
fp8 intermediate (Counter-signals), so these are not `7856e34c…`'s numbers; the predecessor of `e18b982b…` also read 1.030 at m 128, which
`e18b982b…` does not list. The earlier predecessors, which differ from these only in where the scratch is allocated
(once per shape there, in every call here; the GPU code is the same), were measured three more times, in two
profile-19 runs *(field, E10, E9)* and a profile-18 run at main a12866b1e *(field, E4)*, and every listed Point
reached 1.03 in each. Across the four runs a Point's bound moved by up to 5.6% (inter 768, m 1: 3.295 here, 3.331,
3.155 and 3.154 in the earlier runs; inter 384, m 32: 1.205, 1.204, 1.162, 1.201), and at the smallest margin by at
most 0.01 (inter 384, m 64: 1.081, 1.086, 1.091, 1.087). At one token the predecessors' GPU time is 10.69, 9.55 and
12.13 µs against aiter's 55.09, 36.55 and 40.01 µs (median of the rounds, in the table's order) *(field, E1, E2, E3)*.
The gate of the bench clamps each round's ratio at 2 (`MAX_ABS_LOG_RATIO`), so the report's `graph/baseline` column
shows 2.0000 where the published, unclamped bound is above 2.

## Reproduce

On an MI355X, from the repository root, one schema file at a time:

```
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim7168_n_experts384_topk6_inter_dim384/e18b982b504d1626.json
```

and the same for the other two paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup, and `report.json` beside it with every round
(`aklr/ops/README.md` § Commands). Add `--base origin/main` only on a branch that edits the file: with the base set,
bench measures only schemas that are not at the base (`aklr/core/validation/stages/validation_scope.py`,
`of_working_tree`). The local result does not count; CI's measurement is the one dispatch reads (`aklr/ops/README.md`
§ Workflow).

## Rollback

Delete a schema file (in a PR): its Points go unserved, since the other file of each name has no kernel.
`AKLR_DISABLE=fused_moe@1.0` stops serving the whole contract, every other MoE kernel included. Roll back for a wrong
result CI does not catch, for example after a FlyDSL or LLVM upgrade. If the contract becomes bitwise, drop
`93aeaecb…`'s atomic Points (m 1, 2, 16–64); if its error bound no longer admits fp8 activations, all three.

## Evidence

`<uuid>.json#kernel.py:<line>` is a line of the `kernel.py` that schema file embeds (its path is under `sources`).

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] the predecessor of 7856e34c973b92a8 (scratch allocated in every call; at m 1 and 2 an fp8 intermediate under one limit-derived scale): speedup over aiter 5.140, 1.890, 1.235, 1.129 at m 1, 2, 16, 32; median kernel 10.69, 36.65, 134.96, 169.35 µs, aiter 55.09, 69.59, 166.77, 191.37 µs; audit and bench passed with 0 findings, worst violation fraction 0 (…_n_experts96_topk6_inter_dim3072_global_experts384, aiter 0.1.21.dev66+gdbd8bf5bd, base 0e02a81)
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] the predecessor of e18b982b504d1626 (the same GPU code at m 1–64; it also listed m 128): speedup over aiter 3.808, 3.155, 2.537, 1.868, 1.578, 1.205, 1.081, 1.030 at m 1, 2, 4, 8, 16, 32, 64, 128; median kernel 9.55, 12.55, 19.99, 34.96, 63.14, 102.30, 173.49, 242.68 µs, aiter 36.55, 39.82, 50.84, 65.36, 99.68, 123.59, 188.44, 250.27 µs; audit and bench passed with 0 findings, worst violation fraction 0 (…_n_experts384_topk6_inter_dim384, aiter 0.1.21.dev66+gdbd8bf5bd, base 0e02a81)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] the predecessor of 93aeaecbfa4c7956 (the same GPU code): speedup over aiter 3.295, 2.391, 1.871, 1.240, 1.151, 1.172 at m 1, 2, 4, 16, 32, 64; median kernel 12.13, 19.87, 35.31, 119.18, 199.38, 325.28 µs, aiter 40.01, 47.71, 66.30, 147.99, 229.82, 383.28 µs; audit and bench passed with 0 findings, worst violation fraction 0 (…_n_experts384_topk6_inter_dim768, aiter 0.1.21.dev66+gdbd8bf5bd, base 0e02a81)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 18] the earlier predecessors (scratch allocated once per shape, otherwise the GPU code of E1–E3), published speedup over aiter per Point: the earlier predecessor of 7856e34c973b92a8 5.103, 1.879, 1.236, 1.129 at m 1, 2, 16, 32; the earlier predecessor of e18b982b504d1626, measured as another file with the same kernel content, 3.797, 3.173, 2.444, 1.863, 1.539, 1.201, 1.087, 1.030, 1.041 at m 1, 2, 4, 8, 16, 32, 64, 128, 256; the earlier predecessor of 93aeaecbfa4c7956 3.154, 2.321, 1.839, 1.234, 1.146, 1.151 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every Point (base a12866b1e; `schemas[].points[].reconstructed_published_speedup` of the `pro` group)
- E5 [2026-11, gfx950, FlyDSL ?, N=?, KernelForge ablation ladder: each version on the ladder timed beside the version before it, the earlier version's time over the later one's minus one; recorded = timed when the version joined the ladder, remeasured = timed again by the ablation in 3 pairs] the ladder that produced the earlier predecessor of 7856e34c973b92a8: stage-1 grid size set in `_build` instead of in the jit launcher +28.9, +8.4% at m 1, 2 (recorded); stage 2 splitting each token's routes into per-wave K ranges with `Y` in LDS +5.9, +20.4% at m 1, 2 (recorded); routing folded into the per-expert stage 1 and m 16 moved from aiter to that path +12.3% at m 16 (recorded); one stage-2 workgroup per column block covering every token +0.9, +4.6% at m 1, 2 (remeasured); stage-2 token order by the first token sharing an expert +2.4% at m 32 (remeasured)
- E6 [2026-11, gfx950, FlyDSL ?, N=?, KernelForge ablation ladder, as E5] the ladder that produced the earlier predecessor of e18b982b504d1626: token row staged first and stage-1 load-ahead depth set per grid size +16.8, +20.8, +10.7, +8.8% at m 1, 2, 4, 8 (recorded)
- E7 [2026-11, gfx950, FlyDSL ?, N=?, KernelForge ablation ladder, as E5] the ladder that produced the earlier predecessor of 93aeaecbfa4c7956: m 32 and 64 moved from aiter to kernels that deduplicate experts in both stages +7.5, +8.0% at m 32, 64 (recorded); both GEMMs on scaled MX-fp8 × MXFP4 MFMAs with activations quantized in LDS +9.5, +6.1, +3.1% at m 1, 2, 4 (recorded); stage 1 split evenly over workgroups with SwiGLU finished in a per-route atomic stage 2 +7.6, +2.1% at m 1, 2 (remeasured)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd, N=20 tries per Point and condition (24 replays for the graph with new inputs), stress, graph-replay and concurrent-call checks on main 0e02a81] the three predecessors of E1–E3 (scratch allocated in every call): the driver's conditions (a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream; `harness/scripts/forge_task/assets/aklr_forge_driver.py:380–381`, `:437–469`) plus replay of one captured CUDA graph, each try against the call alone; one CUDA graph captured on input set 0 and replayed with three AKLR input sets copied into its buffers in turn, against an eager call on the same inputs; two calls at once on two streams, each on its own input set and into its own outputs, unrestricted and both on the same first 64 CUs, against an eager call alone: 0 differing tries, no hang
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds (32 at inter 384, m 128), AKLR validate, measurement profile 19] a second run of the earlier predecessors (scratch allocated once per shape): published speedup over aiter: the earlier predecessor of 7856e34c973b92a8 5.164, 1.865, 1.240, 1.128 at m 1, 2, 16, 32; the earlier predecessor of e18b982b504d1626 3.792, 3.074, 2.500, 1.816, 1.555, 1.162, 1.091, 1.033 at m 1, 2, 4, 8, 16, 32, 64, 128, at m 128 median kernel 243.91 µs against aiter's 251.72, decided at 32 rounds; the earlier predecessor of 93aeaecbfa4c7956 3.155, 2.364, 1.861, 1.232, 1.138, 1.157 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every Point (base 0e02a81)
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] a first run of the earlier predecessors (scratch allocated once per shape): speedup over aiter: the earlier predecessor of 7856e34c973b92a8 5.165, 1.866, 1.244, 1.124 at m 1, 2, 16, 32; the earlier predecessor of e18b982b504d1626, measured as another file with the same kernel content, which also lists m 256, 3.756, 3.149, 2.545, 1.872, 1.578, 1.204, 1.086, 1.030, 1.020 at m 1, 2, 4, 8, 16, 32, 64, 128, 256, median kernel 243.15 and 276.42 µs against aiter's 250.84 and 282.84 at m 128 and 256; the earlier predecessor of 93aeaecbfa4c7956 3.331, 2.321, 1.849, 1.235, 1.151, 1.176 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every Point (base 0e02a81)
- E11 [2026-11, gfx950, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd, N=20 tries per Point and condition, concurrent-call check on main 0e02a81] the earlier predecessors under two calls at once on two streams, each on its own input set, against an eager call alone; differing tries of 20 with unrestricted streams, then with both streams on the same first 64 CUs: the earlier predecessor of 7856e34c973b92a8 m 1 0, 0; m 2 18, 20; m 16 and 32 20, 20; the earlier predecessor of e18b982b504d1626 m 1 0, 1; m 2 0, 19; m 4 0, 20; m 8, 16, 32 and 64 20, 20; m 128 0, 0; the earlier predecessor of 93aeaecbfa4c7956 m 1 0, 0; m 2 0, 20; m 4 16, 20; m 16, 32 and 64 20, 20 (the same predecessors under the driver's same-input second call and the other driver conditions, 20 tries each: 0 differing)
- S1 7856e34c973b92a8.json#kernel.py:38 `_local_expert` (ballots over the mask); `_quant32` at :86; `_quant_fp8` (MX-fp8 of `Y`, a block over two lanes) at :116; `_mfma_sc` (`mfma_scale_f32_16x16x128_f8f6f4`) at :147; `_stage1` at :240 with the padded LDS ring at :278 and the stage-2 token order (comment at :323–324) at :322–347; `_stage2_cr` at :459 (w2 loads under the default policy at :543); `_stage2_tm` at :587 (w2 loads at :677, `Y` quantized into LDS with its scales at :681–688, each lane's block scale into the scaled MFMA at :702–703); `_build` (functools.cache) at :724: `DIRECT = R <= 64` at :732, K split at :735, non-temporal stage-1 w1 `NT1` at :741, grid constants moved out of the launcher at :742, scratch sizes at :755 (comment at :753–754), `waves_per_eu` hint for `_stage2_tm` at :783–786, `llvm.inreg` arguments at :798, scratch allocated in each call at :812; `run` at :825, at most 32 tokens at :846–847
- S2 e18b982b504d1626.json#kernel.py:16 `_FLYDSL_MAX_TOKENS` 64; `_mfma_mx` at :42 and `_mxfp8` at :72; `_stage1` at :141; `_stage2` (route order, comment at :328–329) at :309; `_build` (functools.cache) at :446: K halves (comment at :453) `NTT1` at :454, load-ahead `PF1` at :460, non-temporal stage 1 `NT1` at :461, groups of `NR1` 4 routes from 48 routes at :463, stage-2 widths at :464, XCD order at :466, non-temporal stage 2 `NT2` at :469, scratch sizes at :470–472, kernarg preload at :500, scratch allocated in each call at :529–530; `run` at :540, at most 64 tokens at :562–563
- S3 93aeaecbfa4c7956.json#kernel.py:871 `_TOKENS`; `_STREAM` (non-temporal weight loads) at :18, in stage 1 at :195 and :357 and in stage 2 at :678, :757 and :831; `_KW1` 7 at :873, `_WV2S` at :875, `_G` at :877, `_PERS1` at :879; `_groups` with the LDS-counter ranks at :103 and :111; `_quant_fp8` at :155; `_stage1s` (even K split, zeroes `out`) at :284, :299 and :338; `_stage1p` (persistent, bitmaps from distinct ids) at :506, :526 and :558; `_stage2` (route order) at :639 and :653; `_stage2r` and `_stage2g` with `BufferAtomicPkAdd` at :776 and :858; `_build` (functools.cache) at :886 with scratch sizes at :903–909, kernarg preload at :977 and scratch allocated in each call at :982–983; `run` at :992, its `_TOKENS` test at :1014
