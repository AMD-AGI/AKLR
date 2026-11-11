---
title: "MXFP4 fused MoE with clamped SwiGLU on gate/up-interleaved weights, hidden 4096, top-6 (256 experts, or 32 / 64 local of 256) — one workgroup per route and bf16 MFMAs at 1–8 tokens; routes sorted into expert blocks inside stage 1, MX-fp8 activations and a mostly atomic stage 2 at 16–256 tokens"
kind: case
ops: [moe]
dtypes: [mxfp4, bf16, fp8]
regimes: ["tokens 1-2 (expert parallel)", "tokens 1-8 (all experts local)", "tokens 16-256"]
contract: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter, AKLR CI measurement profile 18, per Point", value: "3.251–3.448; 2 of 2 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256 file df5c661e4ce848dc, m 1, 2", run_ref: "d908d01 (CI run 42000899019)", date: "2026-11-05"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 18, per Point", value: "2.878–4.094; 2 of 2 Points ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256 file 5db89e7db4f729ff, m 1, 2", run_ref: "d908d01 (CI run 42000899019)", date: "2026-11-05"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 18, per Point", value: "2.006–2.920; 4 of 4 Points ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256 file a5a343437ee191a7, m 1, 2, 4, 8", run_ref: "d908d01 (CI run 42000899019)", date: "2026-11-05"}
  - {metric: "speedup over aiter, AKLR CI measurement profile 18, per Point", value: "1.503–2.894; 4 of 4 Points ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512 file 9dcb4d796b597d0b, m 1, 2, 4, 8", run_ref: "d908d01 (CI run 42000899019)", date: "2026-11-05"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.339–1.622; 5 of 5 listed Points ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256 file 640d56e2cf48884d, measured as its earlier version with the same GPU code, m 16, 32, 64, 128, 256", run_ref: "base 0e02a81", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.241–1.406; 5 of 5 listed Points ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256 file 3279ada84a65bbd9, measured as its earlier version with the same GPU code, m 16, 32, 64, 128, 256", run_ref: "base 0e02a81", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.358–1.962; 5 of 5 listed Points ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256 file b9ae6161bd156dc2, measured as its earlier version with the same GPU code, m 16, 32, 64, 128, 256", run_ref: "base 0e02a81", date: "2026-11-09"}
  - {metric: "speedup over aiter, AKLR validate measurement profile 19, per Point", value: "1.250–1.662; 5 of 5 listed Points ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512 file eb8af99c7e8ad333, measured as its earlier version with the same GPU code, m 16, 32, 64, 128, 256", run_ref: "base 0e02a81", date: "2026-11-09"}
sources:
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/df5c661e4ce848dc.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/5db89e7db4f729ff.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/a5a343437ee191a7.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/9dcb4d796b597d0b.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts32_topk6_inter_dim2048_global_experts256/640d56e2cf48884d.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_expert_mask_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts64_topk6_inter_dim2048_global_experts256/3279ada84a65bbd9.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim256/b9ae6161bd156dc2.json
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/eb8af99c7e8ad333.json
---

# MXFP4 fused MoE with clamped SwiGLU, gate/up-interleaved weights, hidden 4096, top-6

DeepSeek-V4-Flash's MoE (hidden 4096, 256 routed experts, top-6:
`aklr/benchmarks/models/deepseekv4/README.md`) under four schema names: expert parallel with 32 or 64 local experts
of 256 and inter dim 2048, and all 256 experts local at inter dim 256 or 512. Each name has two kernel files, split
by token count:

- **1–8 tokens**, on main: `df5c661e…`, `5db89e7d…`, `a5a34343…` and `9dcb4d79…`, which embed the same `kernel.py`
  (identical `kernel` sections). `kernel.py:<line>` and bare function names in the 1–8-token parts below are in it.
- **16–256 tokens**, measured by CI as of #125: `640d56e2…` (32 local), `3279ada8…` (64 local), `b9ae6161…`
  (inter 256) and `eb8af99c…` (inter 512), one kernel each. The three built on aiter's FlyDSL code carry copies of
  the aiter modules they use and import no aiter at run time.

"m" is the Point's `n_tokens`: the tokens in the call, the input batch, not the rows an expert receives. A route
is one (token, expert) pair, 6m per call; how many land on one expert depends on the routing.

## Applies when

- The same layout at hidden 7168 (DeepSeek-V4-Pro) has its own kernels:
  [fused_moe_interleave_swiglu_hidden7168.md](fused_moe_interleave_swiglu_hidden7168.md).
- Contract `fused_moe@1.0`, layout variant `gate_up_interleaved`: `w1` (experts, 2 × inter, 4096) with gate and up rows
  interleaved in 16-row tiles and fp4 e2m1 in 16 × 64-byte blocks, `w2` (experts, 4096, inter), E8M0 scales per 1×32
  stored 2-D (`aklr/ops/fused_moe/contract.py`, the `gate_up_interleaved` variant). `gate_mode` interleave with
  `swiglu_limit` (gate clamped to at most the limit, up to ± the limit, then SiLU(gate)·up). bf16 `hidden_states`,
  fp32 `topk_weights` (asserted by the 1–8-token kernel and `b9ae6161…`), int32 `topk_ids` (the schemas' dtype),
  routing weight in stage 2, bf16 `out`.
- Activations: the reference keeps them unrounded in this gate mode (`aklr/ops/fused_moe/reference.py`, its
  docstring), and `act_quant`'s description counts aiter's fp8 activations past `AITER_BF16_FP8_MOE_BOUND` rows
  within the bound. The 1–8-token kernel keeps them bf16; the 16–256-token kernels quantize them to MX-fp8
  (Counter-signals).
- Expert parallel names: `expert_mask` over 256 global experts, 32 or 64 local; ids are global. The other two names
  have all 256 experts local and no mask. A token's expert ids are distinct (the `topk_ids` contract);
  `640d56e2…` sizes its expert blocks on it. No kernel here takes `num_local_tokens`: the
  `has_num_local_tokens` names of this shape (m 1024 and 2048) have no kernel.
- Every workload passes `swiglu_limit` 10.0. The contract's interval starts at 7.0, but its comment validates the
  interleave gate mode from DeepSeek-V4's 10 only, since aiter's CK-Tile decode path does not clamp.
- Compare: `determinism: bounded`, `max_violation_fraction` 1e-3, whole-output SQNR ≥ 13 dB (`SqnrBound`). The
  1–8-token kernel reduces in a fixed order with no atomics; the 16–256-token kernels add into `out` with atomics,
  except `b9ae6161…` at 16 and 32 tokens.
- Points. Served on main: m 1, 2 (expert parallel) and m 1, 2, 4, 8 (all local), by the 1–8-token files
  *(field, E1, E17)*, and, as of #125, m 16, 32, 64, 128 and 256 of every name, by the 16–256-token files, whose CI
  `speedup` is 1.235 or more; a workload without a `speedup` of 1.03 or more is not served
  (`AKLR_MIN_SPEEDUP`, `aklr/ops/README.md` § Environment variables; S13). Each name also has a kernel-less file
  (`550200e0…`, `58ad3640…`, `af9db8f6…`, `ff293e2a…`) defining m 16–256, which is not served (`aklr/ops/README.md`
  § Schema fields); no other Point of these names has a kernel.
- The baseline is aiter's `fused_moe` with `AITER_BF16_FP8_MOE_BOUND=16`, sglang's DeepSeek-V4 setting
  (`aklr/ops/fused_moe/baseline.py`): below 16 rows aiter keeps bf16 activations, from 16 it runs fp8. One kernel
  file measures each Point, so the yardstick is aiter.

## Counter-signals

**1–8 tokens (the per-route kernel)**

- **The kernel stops at 8 tokens.** Its path for more than 64 routes (`_prep`, `_route`, routes binned per local
  expert into 16-row tiles, four launches) is reached by no Point: the largest route count it serves is 48 (m 8, all
  local). m 16–256 are the 16–256-token files' Points.
- **15 of 16 MFMA rows are discarded.** Each route sits alone in its 16-row MFMA tile in both stages (every A row is
  the same token or route; only row 0 is stored). What that costs at these sizes is not measured.
- **Expert-parallel grids carry routes that are not local.** Stage 1 launches a workgroup row for every route; a
  route to an expert off this rank finds no local slot and its workgroups exit. Stage 2 takes a data-dependent branch:
  a token with exactly one local route spreads that route's K over all 6 waves, otherwise each local route's wave
  walks its own K. Which branch the bench's inputs exercise is not recorded.
- **The baseline does not clamp.** aiter's CK-Tile decode path does not apply `swiglu_limit` (`contract.py`, the
  comment at `swiglu_limit`); the interleave gate mode is validated from 10 only so that a limit binding on the
  bench's inputs does not leave the comparison without a baseline. Whether the clamp binds on the bench's draws at 10
  is not established.
- **Stale docstring.** It describes routes binned per local expert into 16-row tiles as the main path and says rows at
  or past `num_local_tokens` contribute nothing; at every Point the per-route path runs, and `run` takes no
  `num_local_tokens` (it passes `HAS_NLT=False`), so those parts are reached by no Point.
- **Per-shape scratch, shared by concurrent calls.** `_build` (a `functools.cache` compile function) allocates `C`,
  `SLOT`, `LOC`, `ROUTES`, `MAP` and `Y` once per shape (S7). On the per-route path each call writes `LOC` for every
  route and `Y` for every local route before stage 2 reads them; the other buffers serve only the unreached path. Two
  calls of one Point at once on two streams share them: with different inputs (20 tries per Point, on two
  unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone),
  `df5c661e…` gave wrong output on the shared CUs at m 1 in 2 and m 2 in 20 of 20 tries, `a5a34343…` there at m 2,
  4 and 8 in 1, 3 and 20 of 20, and `9dcb4d79…` there at m 2–8 in 15–20 of 20 and at m 8 in 15 of 20 on unrestricted
  streams; `5db89e7d…`, the same `kernel.py`, in none *(field, E14)*. That is consistent with the shared scratch, not
  shown by a per-call-allocation control on these files, and does not explain why `5db89e7d…` did not fail.

**16–256 tokens (the sorted-block kernels)**

- **Output bits vary between runs.** Stage 2 adds each block's weighted rows into `out` with bf16 atomics in arrival
  order: all four files at every Point, except `b9ae6161…` at 16 and 32 tokens, which adds a token's routes in LDS in
  route order. Under a bitwise contract they need a fixed-order stage 2 (`../moe/grouped_sorted_tokens.md` §6).
- **Activations and the intermediate pass through fp8.** The token rows are quantized to fp8 e4m3 with an E8M0
  scale per 32 values, and SiLU(gate)·up again before stage 2, where the reference keeps both unrounded. Every Point
  passed with a worst violation fraction of 0 *(field, E3, E4, E5, E6)*; the SQNR itself is not recorded. aiter's
  path from 16 rows also runs fp8 activations. Values that differ below fp8's resolution within a group lose the
  difference: the review's input, 1.015625 against 1.0 around a gate that cancels them, reads about 0 dB
  (estimated); aiter's fp8 path rounds it the same way, and the contract's `act_quant` description counts that
  rounding as within the bound.
- **At 16 tokens several workgroups store the same activation rows.** In `640d56e2…` every stage-1 workgroup of an
  expert block quantizes the block's token rows into the shared `a1` and `a1_scale` (`part2`, `kernel.py`:746, the
  row offset at :805), waits for its own stores (`part2_wait`, :828) and reads back only rows it stored. Two
  workgroups that store one row store the same bytes, the same bf16 row through the same quantization, so no reader
  can see a mixed row (code read).
- **Built on copies of aiter's FlyDSL code.** `640d56e2…`, `3279ada8…` and `eb8af99c…` carry the aiter modules
  they use among their sources, copied from aiter dbd8bf5bd, and import no aiter at run time. A copy differs from
  aiter's file in its imports, in environment reads replaced by their defaults, in nested imports of modules the
  file does not carry replaced by `raise NotImplementedError`, in a few host-side edits the audit requires, and in
  the launch-argument builders a listed Point reaches taking the stream of their input's device; `mx_types.py`
  keeps only aiter's int enums, and `fused_moe_v2_stage2.py` only the three functions of aiter's `fused_moe.py`
  that `eb8af99c…` calls (S14). `640d56e2…` runs aiter's stage-1 and stage-2 builders with its own source edits,
  one edited copy per configuration a listed Point uses: eight in `edited_builders.py`, about 1.1 MB of its 1.7 MB
  schema file (S9).
  `eb8af99c…` takes stage-1 settings by aiter's tuned kernel names (`get_flydsl_kernel_params`) and carries aiter's
  v2 stage-2 launcher; `3279ada8…` has its own builders and copies aiter's helpers. A later aiter neither changes
  nor fixes them: a fix reaches them only by copying it in.
- **Shapes are fixed in the code.** `b9ae6161…` asserts hidden 4096 and inter 256; `eb8af99c…` hard-codes 256
  experts, top-6, hidden 4096 and inter 512; `640d56e2…` assumes 32 local of 256 experts, hidden 4096, inter 2048 and
  at most 256 tokens, and has builders for the five listed token counts only; `3279ada8…` asserts hidden = 16 ×
  global experts and at most 64 local experts. None of them runs DeepSeek-V4-Pro's hidden 7168.
- **Scratch has to be allocated in each call.** The four files allocate their routing buffers, intermediates and
  scratch inside the call (`b9ae6161…`: S11), and their earlier versions (Result) gave 0 differing tries at every
  Point under three checks: the forge driver's conditions (CU-limited streams, a second call with the same inputs,
  graph replay), one CUDA graph replayed with three input sets, and two calls at once with different inputs as above
  *(field, E12)*. The same GPU code as `b9ae6161…` with its scratch allocated once per shape in `_build` passed the
  first two and gave wrong output under the third in 20 of 20 tries at 32–256 tokens, and at 16 tokens on the shared
  CUs *(field, E13)*: the same-input conditions cannot show scratch that only receives plain stores of
  input-determined values, as this kernel's does; scratch that each call zeroes and then accumulates into with
  atomics can show under them, but enqueued back to back only in 1–2 of 200 tries on a toy HIP op *(field, E15)*.
- **Code no Point reaches.** `b9ae6161…` asserts at most 1536 routes and no expert mask, which every listed Point
  meets (S11). `eb8af99c…`'s unfolded path below 256 tokens (`_pf_depth` 2, `_early_tail`, the K-stagger off) and its
  copy of aiter's v2 stage 2 above 256 tokens, `3279ada8…`'s default configuration for token counts outside `_CFG`,
  and the copied aiter modules' paths for other dtypes, layouts and epilogues run at no Point.

## Mechanism

**1–8 tokens: one workgroup per route and 16 columns, two launches.** Stage 1 then stage 2 at every served Point
(`_build`: `DIRECT = R <= 64`); no routing kernel, no atomics, no workspace counters.

Stage 1 (`_stage1`, per-route path). Grid (inter / 16, routes): one workgroup per (route, 16 intermediate columns),
4 waves; 96–1536 workgroups (inter 256: 96, 192, 384, 768 at m 1–8; inter 512: 192–1536; inter 2048 with 6 or 12
routes: 768, 1536).
- The expert: all local, the route's id; expert parallel, `_local_expert` builds the mask's four 64-expert words
  with wave `ballot`s and counts the set bits below the id with `ctpop`, which gives the local slot (or −1: the
  workgroup exits). The workgroup at column block 0 writes the slot to `LOC` for stage 2.
- K = 4096 is split over the 4 waves, 1024 each (8 chunks of 128). Each wave stages its own slice of the token's row
  in LDS with its own lanes, so no barrier; then issues every scale and weight load of its 8 chunks before the first
  MFMA, in the order row, scales, weights, pinned with `sched_barrier` so `vmcnt` retires them in that order.
- Weights: one 16-byte load per lane per chunk for the gate tile and one for the up tile of the interleaved pair; each
  dword of 8 fp4 values becomes 8 bf16 with the block scale applied by `cvt_scalef32_pk_bf16_fp4`; one scale dword
  holds the gate and up bytes of two chunks. Non-temporal loads under expert parallelism, cached otherwise
  (`../moe/grouped_sorted_tokens.md` §4). The code's note says routes that share an expert then reuse it from L2; at
  8 tokens with L2 flushed, non-temporal loads were 9.5–13% slower in five of six cases where 6 or 8 experts took all
  48 routes and within 2.4% under random routing, with a HIP copy's L2 hit rates at 7–26% either way
  *(field, E19, E20)*, so little of that reuse is in L2.
- `mfma_f32_16x16x32_bf16`, 4 per chunk for gate and 4 for up, into fp32 (`../gemm/mainloop.md` §5).
- `_reduce_waves`: the 4 waves' partials through LDS, summed in wave order (`../gemm/small_m.md` §4).
  Wave 0 clamps, applies SiLU with `exp2` and `rcp`, rounds to bf16 and stores the route's 16 values to `Y`.

Stage 2 (`_stage2_tp`, `_stage2_ep`). Grid (4096 / 16, m): one workgroup per (token, 16 output columns), 6 waves,
one per route; 256–2048 workgroups.
- Each wave walks its route's K = inter (2, 4 or 16 chunks of 128) with `mfma_f32_16x16x32_bf16` on `w2` converted the
  same way. The unrolled walk (`_w2_walk`) takes chunks in blocks of 4 whose `Y`, weight and scale loads all issue
  before the block's first MFMA. Expert parallel, it always runs; all local, it runs when m × inter / 128 ≤ 8, and a
  runtime loop of one chunk at a time runs above (inter 256 at m 8, inter 512 at m 4, 8).
- Row 0 of each wave's tile goes to LDS; 16 threads add the 6 routes × routing weight in route order, skipping routes
  that are not local, and store bf16 (`../gemm/correctness.md` §8).

No sort and one route per workgroup set, as the 1–2-token kernels of
[fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md); `../moe/few_tokens.md` §1 on why the launch count
matters at these sizes.

**16–256 tokens: routes sorted into expert blocks, MX-fp8 activations on scaled MFMAs, stage 2 adding into
`out`.** The four kernels share one plan and differ in where the routing runs and how the blocks are cut.

- **The routing runs inside stage 1 up to 128 tokens.** Every stage-1 workgroup rebuilds the sort of the call's routes
  for its own expert block(s) in LDS from `topk_ids` (ballot ranks, or an LDS histogram and a prefix scan), so no
  workgroup waits on another and no routing launch runs; the sorted ids and routing weights of each block are
  published for stage 2, and the grid zeroes `out` in stripes for stage 2's adds (`../moe/few_tokens.md` §2). Under expert
  parallelism the local slot comes from the mask by ballot, and routes to other ranks join no block. At 256 tokens
  `640d56e2…`, `3279ada8…` and `eb8af99c…` run one launch before stage 1 that sorts and quantizes, and in `3279ada8…`
  and `eb8af99c…` also zeroes `out` (`640d56e2…` zeroes it inside stage 1, `_make_zero`);
  `640d56e2…`'s launch adds 256 workgroups that request the first half of the start k-tile of each local expert's W1
  from the XCD that stage 1 runs it on. That made stage 1 about 2% faster at m 256, but not through L2 hits: a probe
  with the same requests and placement saw none in the next launch *(field, E21)*. `b9ae6161…` has no sort: at 16
  tokens (96 routes) stage 1
  runs one workgroup per (route, 16 columns) as the 1–8-token kernel does; from 32 tokens one workgroup per (expert,
  32 columns) collects its expert's routes by ballots in route order and runs them 16 per pass, and its column block 0
  publishes the list for stage 2. One FlyDSL launch for sort, quantization and zeroing in place of aiter's, paired
  against the version before it: about +12.7% at m 16 down to +4.9% at m 256 in `3279ada8…`'s campaign, +14.2% to
  +6.3% in `eb8af99c…`'s *(field, E8)*. That launch folded into stage 1, paired the same way: +5.2%, +1.4% and +0.8%
  at m 16, 32 and 64 (`eb8af99c…`); +2.8% at m 16, then +3.8% and +3.3% at m 16 and 32 when the fold reached 32
  tokens with a one-wave routing head and the quantization before the W1 request (`640d56e2…`); +0.8% and +1.3% at
  m 16 and 64 (`3279ada8…`) *(field, E9)*.
- **Activations as MX-fp8.** Token rows are quantized to fp8 e4m3 with one E8M0 scale per 32 values from the group's
  absolute maximum, reduced across lanes. The scale is applied by `cvt_scalef32_pk_fp8_bf16` in `b9ae6161…`, in the
  stage 1 of `3279ada8…` and `eb8af99c…` up to 128 tokens, and in `640d56e2…`'s A loader at 32–128 tokens; the
  separate routing launches at 256 tokens and `640d56e2…`'s fold at 16 tokens multiply by the power of two themselves
  and convert with `cvt_pk_fp8_f32`. Both stages multiply fp8 activations with the fp4 weights on the scaled MFMA,
  both E8M0 scales as operands (`mfma_scale_f32_16x16x128_f8f6f4` in the three kernels that carry their GEMM code;
  `640d56e2…` runs its edited copies of aiter's fp8 × fp4 builders; `../gemm/mainloop.md` §5). `640d56e2…` at
  32–128 tokens quantizes inside stage 1's A loader, straight into the LDS A tile, so the fp8 copy never goes to
  memory. Stage 1's epilogue clamps, applies SiLU(gate)·up and stores the intermediate as MX-fp8 (`b9ae6161…`: bf16,
  requantized per 32 in stage 2).
- **Blocks cut to the routes.** `3279ada8…` sorts into 16-row expert blocks; `eb8af99c…` puts at most 8 routes
  (16 at 256 tokens) into each 32-row block and skips the MFMAs of rows 16–31; `640d56e2…` bounds its 32-row blocks
  by the routes (at most 32 blocks up to 32 tokens, where every expert's routes fit one block because a token's
  expert ids are distinct). Padding rows read past the end of the activations, which returns zeros without a memory
  request (`640d56e2…`; `3279ada8…` at 16, 64 and 128 tokens).
- **Weights streamed once, K starts staggered.** Weight loads are non-temporal in most stages
  (`../moe/grouped_sorted_tokens.md` §4); `b9ae6161…`'s per-route stage 1 at 16 tokens reads w1 cached. The routes of
  one expert read each 16-column w1 tile from one L2 (placement below) whether the read is cached or non-temporal
  (equal L2 hits on a HIP copy of its load pattern), and with L2 flushed the cached read was 2–4% faster only when
  many routes shared an expert *(field, E19, E20)*. Workgroups start their K loop at different K tiles and wrap:
  `640d56e2…` by expert block in both GEMMs, `3279ada8…` per stage-2 workgroup, `eb8af99c…` per stage-1 column tile and
  block. Experts streamed at once then do not hit the same address bits together (`640d56e2…`'s comment at
  `_TIERS`); each block's accumulation order stays fixed (`../gemm/small_m.md` §6 on racing a K-start stagger). The
  rotation in `640d56e2…`, paired against the version before it: +2.8% to +7.7% per Point *(field, E10)*.
- **Stage 2 adds into `out`.** Each block's routing-weighted rows go to `out` with packed bf16 atomics
  (`../moe/grouped_sorted_tokens.md` §6). `b9ae6161…` does so from 64 tokens (384 routes) with one workgroup per
  (expert, 256 or, at 256 tokens, 512 output columns); at 16 and 32 tokens it runs one workgroup per (token, 32
  columns), one wave per route, and adds the routes in LDS in route order. The per-expert stage 1 with the
  expert-major stage 2, paired against the version before it (per-route at 32 tokens, aiter above): +10.4% to +21.5%
  at m 32–256; atomics in place of a combine kernel: +4.0% to +5.6% at m 64–256 *(field, E11)*.
- **Launch arguments, occupancy and placement.** `b9ae6161…` and `eb8af99c…` preload 16 kernel arguments into SGPRs
  (`../gemm/small_m.md` §6); `3279ada8…` pads stage 1's LDS so that at most 7 (16 tokens), 10 (32 tokens) or 8
  workgroups share a CU; `b9ae6161…` places the per-route stage-1 workgroups so that blocks whose index agrees mod 8
  share an XCD (which XCD is rotated on some launches) and the routes of one expert read each 16-column w1 tile from
  one L2 *(field, E16)* (`../gemm/schedule_and_locality.md` §2).

Per Point:

| File | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `640d56e2…` 32 local | routing in stage 1 (one-wave head); 64-column stage-2 tiles | routing in stage 1; activations quantized in the A loader | the same, LDS-histogram routing head | the same, two blocks per stage-1 workgroup | separate routing launch that also warms W1 |
| `3279ada8…` 64 local | routing in stage 1; 2-wave stage 2, K tile 128 | the same, 10 workgroups per CU | routing in stage 1; tiles dealt to XCDs in contiguous ranges | the same | separate routing launch; stage-1 K tile 1024 |
| `b9ae6161…` inter 256 | per-route stage 1; per-token stage 2 | per-expert stage 1; per-token stage 2 | per-expert stage 1; expert-major atomic stage 2 | the same | the same; stage-2 weights issued before the route count is read |
| `eb8af99c…` inter 512 | routing in stage 1, 8-route blocks | the same | the same | the same | separate routing launch, 16-route blocks |

## Result

Both sets of numbers use the ruler of `MEASUREMENT_PROFILE` in `aklr/core/validation/stages/_measure.py`, a one-sided
Student-t lower bound of aiter's time over the kernel's after 16 or 32 rounds of CUDA-graph replays with L2 flushed
before each call; the readings in the tables below are of its version 19, which collects kernel times with AKLR's own
rocprofiler-sdk tracer (#33).

**1–8 tokens, in CI** (CI run 42740594241, #125, for `df5c661e…`, 42549655045 for `5db89e7d…` and `9dcb4d79…`,
42599763780 for `a5a34343…`), the speedups dispatch reads as of #125 *(field, E1, E17)*. Bold: 1.03 or above.

| File | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 32 local of 256, inter 2048 `df5c661e…` | **3.297** | **3.343** | – | – |
| 64 local of 256, inter 2048 `5db89e7d…` | **2.805** | **4.053** | – | – |
| 256 local, inter 256 `a5a34343…` | **2.941** | **2.697** | **2.440** | **2.021** |
| 256 local, inter 512 `9dcb4d79…` | **2.991** | **2.535** | **1.853** | **1.479** |

"–": no workload in that file. Every workload reads `accuracy: passed`. Across the profile-18 CI runs that rewrote
each file, a Point moved by up to 0.19 (`df5c661e…` m 2: 3.255–3.448; `5db89e7d…` m 2: 3.929–4.120), and none came
near 1.03 *(field, E2)*.

**16–256 tokens, the new files**, measured as their earlier versions
(`python -m aklr validate` at main 0e02a81 on an MI355X, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd,
torch 2.11.0+rocm10.0.0). Those imported aiter at run time; the current files run the same GPU code from their own
copies of aiter's modules (S14). CI measured the current files at 1.235–1.959, every Point served and passing
accuracy (#125). Bold: 1.03 or above.

| File | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 32 local of 256, inter 2048 `640d56e2…` *(field, E3)* | **1.622** | **1.485** | **1.432** | **1.339** | **1.350** |
| 64 local of 256, inter 2048 `3279ada8…` *(field, E4)* | **1.406** | **1.363** | **1.285** | **1.243** | **1.241** |
| 256 local, inter 256 `b9ae6161…` *(field, E5)* | **1.962** | **1.692** | **1.414** | **1.358** | **1.362** |
| 256 local, inter 512 `eb8af99c…` *(field, E6)* | **1.662** | **1.432** | **1.318** | **1.263** | **1.250** |

All 20 Points passed accuracy with a worst violation fraction of 0. A second run for `640d56e2…`, `3279ada8…` and
`eb8af99c…` read 1.330–1.646, 1.242–1.407 and 1.256–1.660 *(field, E7)*; `b9ae6161…`'s GPU code read 1.333–1.948 and
1.342–1.970 in two runs with its scratch allocated once per shape *(field, E5)*. Median call times at 16 and 256
tokens: 25.9 and 78.3 µs against aiter's 42.0 and 105.9 (`640d56e2…`); 58.2 and 140.3 against 81.8 and 174.4
(`3279ada8…`); 25.4 and 76.5 against 50.0 and 104.3 (`b9ae6161…`); 45.7 and 140.5 against 76.0 and 176.0
(`eb8af99c…`) *(field, E3, E4, E5, E6)*.

## Reproduce

On an MI355X, from the repository root, one schema file at a time:

```
python -m aklr validate aklr/ops/fused_moe/impls/flydsl/fused_moe_interleave_has_swiglu_limit_gate_up_interleaved_model_dim4096_n_experts256_topk6_inter_dim512/eb8af99c7e8ad333.json
```

and the same for the other paths under `sources`. Each runs the audit and bench stages and writes
`aklr_validation/report.md` with the per-Point speedup (`aklr/ops/README.md` § Commands); its `graph/baseline` column
is the gate's bound, and `report.json` beside it holds every round. Add `--base origin/main` only on a branch
that edits the file: with the base set, bench measures only schemas that are not at the base, so on an unchanged
checkout it skips the bench (`aklr/core/validation/stages/validation_scope.py`, `of_working_tree`). The local result
does not count; CI's measurement is the one dispatch reads (`aklr/ops/README.md` § Workflow).

## Rollback

Delete a schema file (in a PR): a 1–8-token file's Points go unserved, since no other file of its name holds them; a
16–256-token file's listed Points are left unserved, since the kernel-less file of its name has no kernel.
`AKLR_DISABLE=fused_moe@1.0` stops serving the whole contract, every other MoE kernel included. Roll back for a wrong
result CI does not catch, for example after a FlyDSL or LLVM upgrade. Remove the 16–256-token files if the
contract becomes bitwise (atomic stage 2) or its error bound no longer admits fp8 activations.

## Evidence

`kernel.py:<line>` is a line of the `kernel.py` the four 1–8-token files embed (byte-identical `kernel` sections);
`<uuid>.json#<file>:<line>` is a line of that source of the 16–256-token file (its path is under `sources`).

- E1 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of baseline aiter dbd8bf5, read from the workflow), N=1, AKLR CI, measurement profile 19] speedup over aiter in the committed file as of #125: df5c661e4ce848dc 3.297, 3.343 at m 1, 2 (3.251, 3.448 in #71 under profile 18, E2); accuracy passed at both Points (da38d32, #125, CI run 42740594241, measured at d7c0a3e)
- E2 [2026-11, gfx950, FlyDSL ?, N=6, AKLR CI, measurement profile 18; N=3 for a5a343437ee191a7] range over the profile-18 CI runs that rewrote each file: df5c661e4ce848dc m 1 3.190–3.364, m 2 3.255–3.448; 5db89e7db4f729ff m 1 2.756–2.886, m 2 3.929–4.120; a5a343437ee191a7 m 1 2.894–2.957, m 2 2.702–2.723, m 4 2.366–2.429, m 8 1.984–2.015; 9dcb4d796b597d0b m 1 2.894–3.051, m 2 2.467–2.592, m 4 1.831–1.880, m 8 1.496–1.512 (3fb9033, CI run 41812750710; 5f63e8a, CI run 41956295934; 93eba6d, CI run 41964469977; 90b9d50, CI run 41976547645; 43e112e, CI run 41991630592; d908d01, CI run 42000899019)
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] speedup over aiter 1.622, 1.485, 1.432, 1.339, 1.350 at m 16, 32, 64, 128, 256; median kernel 25.87, 41.57, 55.63, 74.18, 78.33 µs, aiter 41.99, 61.91, 79.69, 99.51, 105.93 µs; accuracy passed, worst violation fraction 0 (…_n_experts32_topk6_inter_dim2048_global_experts256, an earlier version of file 640d56e2cf48884d, base 0e02a81)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] speedup over aiter 1.406, 1.363, 1.285, 1.243, 1.241 at m 16, 32, 64, 128, 256; median kernel 58.16, 78.43, 110.41, 133.22, 140.32 µs, aiter 81.81, 106.91, 141.91, 166.05, 174.35 µs; accuracy passed, worst violation fraction 0 (…_n_experts64_topk6_inter_dim2048_global_experts256, an earlier version of file 3279ada84a65bbd9, base 0e02a81)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] speedup over aiter 1.962, 1.692, 1.414, 1.358, 1.362 at m 16, 32, 64, 128, 256; median kernel 25.42, 39.65, 57.21, 68.24, 76.50 µs, aiter 49.95, 67.09, 81.28, 92.76, 104.30 µs; accuracy passed, worst violation fraction 0 (…_n_experts256_topk6_inter_dim256, an earlier version of file b9ae6161bd156dc2, base 0e02a81); the same GPU code with its scratch allocated once per shape read 1.948, 1.669, 1.420, 1.333, 1.356 and 1.970, 1.687, 1.422, 1.352, 1.342
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19] speedup over aiter 1.662, 1.432, 1.318, 1.263, 1.250 at m 16, 32, 64, 128, 256; median kernel 45.67, 73.46, 108.29, 128.16, 140.47 µs, aiter 75.98, 105.59, 142.96, 163.51, 176.03 µs; accuracy passed, worst violation fraction 0 (…_n_experts256_topk6_inter_dim512, an earlier version of file eb8af99c7e8ad333, base 0e02a81)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, AKLR validate, measurement profile 19, a second run of the same files] speedup over aiter at m 16, 32, 64, 128, 256, earlier versions of: 640d56e2cf48884d 1.646, 1.503, 1.436, 1.330, 1.352; 3279ada84a65bbd9 1.407, 1.371, 1.269, 1.242, 1.255; eb8af99c7e8ad333 1.660, 1.425, 1.318, 1.275, 1.256; accuracy passed, worst violation fraction 0 (base 0e02a81)
- E8 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] one FlyDSL launch for sort, activation quantization and zeroing of `out` in place of aiter's: 3279ada84a65bbd9's campaign +12.7, +7.6, +5.5, +5.2, +4.9% at m 16, 32, 64, 128, 256; eb8af99c7e8ad333's +14.2, +9.4, +7.4, +6.5, +6.3%
- E9 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] that launch folded into stage 1: eb8af99c7e8ad333 up to 64 tokens +5.2, +1.4, +0.8% at m 16, 32, 64; 640d56e2cf48884d at 16 tokens +2.8% at m 16, then up to 32 tokens with a one-wave routing head and the quantization before the W1 request +3.8, +3.3% at m 16, 32; 3279ada84a65bbd9 at 16 and 64 tokens, dropping its prologue launch, +0.8, +1.3% at m 16, 64
- E10 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] 640d56e2cf48884d's campaign, each block's K loop started at a rotated k-tile in both GEMMs: +2.8, +5.8, +7.7, +4.2, +4.4% at m 16, 32, 64, 128, 256
- E11 [2026-11, gfx950, FlyDSL ?, N=?, campaign paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] the campaign that produced b9ae6161bd156dc2's GPU code (as a version with its scratch allocated once per shape): per-expert stage 1 with in-kernel routing and an expert-major stage 2 +10.4, +21.5, +14.1, +14.6% at m 32, 64, 128, 256; bf16 atomics into `out` in place of the combine kernel +4.0, +4.8, +5.6% at m 64, 128, 256
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and Point (24 graph replays), stress checks on MI355X at main 0e02a81] 0 differing tries for the earlier versions of 640d56e2cf48884d, 3279ada84a65bbd9, eb8af99c7e8ad333 and b9ae6161bd156dc2 at every Point under (a) the forge driver's conditions: a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream, replays of one captured graph; (b) one CUDA graph replayed 24 times with three input sets copied into its buffers, each replay compared with an eager call on the same inputs; (c) two calls at once on two streams, each with its own input set and outputs, on unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and Point, the checks of E12] b9ae6161bd156dc2's GPU code with its scratch allocated once per shape in `_build`: under (c) wrong output at m 16 in 0 of 20 tries on unrestricted streams and 20 of 20 on the shared 64 CUs, at m 32, 64, 128 and 256 in 20 of 20 under both; under (a) and (b) 0 differing tries
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and Point, check (c) of E12 on main's unmodified files] wrong output, tries out of 20 on unrestricted streams / on the shared 64 CUs: df5c661e4ce848dc m 1 0/2, m 2 0/20; a5a343437ee191a7 m 1 0/0, m 2 0/1, m 4 0/3, m 8 0/20; 9dcb4d796b597d0b m 1 0/0, m 2 0/16, m 4 0/15, m 8 15/20; 5db89e7db4f729ff 0 at every Point
- E15 [2026-11, gfx950 (MI355X), HIP 7.15, N=200 tries per condition, two calls on two streams sharing scratch, same inputs, exact compare] scratch given plain stores of input-determined values: 0 of 200 wrong; scratch zeroed then added into with atomics: 1–2 of 200 enqueued back to back, 200 of 200 with both calls behind one event (a toy HIP routed op, not an AKLR kernel)
- E16 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=224 launches, `HW_REG_XCC_ID` recorded per block] block L of a launch on XCD (L + c) mod 8, one c per launch: c = 0 in 104 of 224 launches, 5–7 in the rest; blocks of one (expert, 16-column w1 tile) on more than one XCD in 0 of 32,256 groups (b9ae6161bd156dc2's per-route `_stage1`, m 16, 8 and 13)
- E17 [2026-11, gfx950, FlyDSL 0.3.2, N=1, AKLR CI, measurement profile 19] speedup over aiter in the files on main: 5db89e7db4f729ff 2.805, 4.053 at m 1, 2 and 9dcb4d796b597d0b 2.991, 2.535, 1.853, 1.479 at m 1, 2, 4, 8 (6d7cfec, #102, CI run 42549655045 at 0e02a81); a5a343437ee191a7 2.941, 2.697, 2.440, 2.021 at m 1, 2, 4, 8 (227aa66, #115, CI run 42599763780); accuracy passed at every Point
- E19 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=15 rounds, CUDA events, a 1 GiB read before each call, median non-temporal / cached time] stage 1 with only its w1 loads' cache bit changed: df5c661e4ce848dc's at m 8 (48 routes) 1.095–1.132 in 5 of 6 cases with 6 or 8 experts (0.966 with the same 6 at 256 local, inter 2048), 0.988–1.024 random; b9ae6161bd156dc2's per-route one at m 16 and 8 1.021–1.039 with 6 or 8 experts, 0.993–1.001 random
- E20 [2026-11, gfx950 (MI355X), HIP 7.15, N=5 dispatches per arm, rocprofv3 TCC hits and misses, cache flushed before each] L2 hit rate cached / non-temporal: b9ae6161bd156dc2's pattern at m 16 0.889 / 0.889 (6 experts) and 0.871 / 0.871 (8); df5c661e4ce848dc's at m 8, 6 or 8 experts, 0.070–0.258 / 0.202–0.264 (a HIP kernel with only stage 1's w1 loads, not an AKLR kernel)
- E21 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=8 rounds, CUDA graph, AKLR's tracer, a 1 GiB read before each call; probe HIP 7.15, N=9 rounds] 640d56e2cf48884d at m 256: stage 1 1.021× as fast with its routing launch's W1 requests (8 of 8 rounds); a HIP probe with the same requests and XCD placement: the next launch's L2 misses 133,296 with and without them
- S1 kernel.py:532 `DIRECT = R <= 64`; launches at :560–574: `_prep` and `_route` only when not `DIRECT`
- S2 kernel.py:55 `_local_expert`: `ballot` at :70, `ctpop` at :71 and :74; `LOC` written by column block 0 at :255–256
- S3 kernel.py:251 `XL` (the token row staged in LDS by each wave's own lanes, :282–302); loads issued row, scales, weights between `sched_barrier`s at :287–297; `PF1` = all of a wave's chunks at :535
- S4 kernel.py:86 `_fp4x8` with `cvt_scalef32_pk_bf16_fp4` (:91); `_mfma` `mfma_f32_16x16x32_bf16` at :96–97; non-temporal under expert parallelism `NT1` at :540 (comment :538–539)
- S5 kernel.py:210 `_reduce_waves` sums in wave order (:222–223); clamp, SiLU and the row-0 store at :311–325
- S6 kernel.py:401 `_stage2_tp`, route sum in route order with the `LOC >= 0` select at :448–455; `_stage2_ep` at :462, one-local-route split at :488–497 (`_split_chunks`, :333); `UNROLL2` at :547
- S7 kernel.py:1 the docstring (binning into 16-row tiles, `num_local_tokens` at :11–12); `run` passes `HAS_NLT=False` at :596–597; scratch `ws` per shape at :548–552; fp32 `topk_weights` asserted at :526
- S8 aklr/ops/fused_moe/baseline.py:44 `AITER_BF16_FP8_MOE_BOUND=16` for gate_mode interleave; aklr/ops/fused_moe/contract.py:249 the comment (:249–252): the interleave limit validated from 10 because aiter's CK-Tile decode path does not clamp; interval from 7.0 at :253; reference.py:71 activations kept unrounded under interleave (:90–93)
- S9 640d56e2cf48884d.json#kernel.py:392 `_TIERS` by route count (fold at m ≤ 128, comment on the K rotation at :388–391); the routing heads `part1s` (:431) at m ≤ 32 and `part1w` (:564) above, chosen at :946; activations quantized in stage 1's A loader (comment at :401–403); the prelude with W1-warming workgroups at :104 and :148; fp8 conversion by `cvt_pk_fp8_f32` in the prelude (:250) and the fold's quantization (:803), by `cvt_scalef32_pk_fp8_bf16` in the A loader (:896); the edited builder of each configuration in `_S1_BUILDERS` (:983) and `_S2_BUILDERS` (:990), loaded by `_s1_compile` (:998) and `_s2_compile` (:1006); buffers per call at :1075–1080; stage 2 `mode="atomic"` at :1146. 640d56e2cf48884d.json#edited_builders.py:1 the docstring listing the edits; the 16-token stage-1 builder at :67, its K rotation at :686, fold head at :582 and padding rows past the activations at :801; the A-loader quantization of the 32-token builder at :6772; stage 2's late `num_valid_ids` read at :3803 and K rotation at :4858; `_make_zero` in the 256-token builder at :20753
- S10 3279ada84a65bbd9.json#kernel.py:51 `_CFG` (16-row blocks, fused at m ≤ 128, comments at :42–50); `_prologue` at :95, its MX-fp8 scaling by hand and `cvt_pk_fp8_f32` at :198–221; the fused stage 1's `CvtScaleF32PkFp8Bf16Op` at :2246–2247; `_fused_route_head` at :287; stage-2 builder copy at :471, `raw_ptr_buffer_atomic_fadd` at :1370; LDS padded to cap workgroups per CU at :1610; `_plan` asserting hidden = 16 × global experts and ≤ 64 local at :3184; buffers per call at :3309
- S11 b9ae6161bd156dc2.json#kernel.py:16 route thresholds (`DIRECT_MAX_ROUTES` 96, `FLY_MAX_ROUTES` 1536, expert-major stage 2 from 384, `SPEC` from 1536); `_stage1` per route with the XCD placement at :175–176 and w1 read cached (cache policy 0) at :203; `_cvt_fp8` with `cvt_scalef32_pk_fp8_bf16` at :266 and `mfma_scale_f32_16x16x128_f8f6f4` at :273; `_stage1_ex` with `_route_list` ballots at :478 and the zeroing of `out` at :395; `_atom_pk` (`BufferAtomicPkAdd`) at :78, used at :633 and :643; `_stage2_tp` route-order sum at :738; `_build` with fp32 `topk_weights` asserted at :753, scratch sizes at :772 and the scratch allocated in each call in `launch` at :863; kernarg preload at :858; `run` asserting at most `FLY_MAX_ROUTES` routes and no expert mask at :895
- S12 eb8af99c7e8ad333.json#kernel.py:53 `_E`, `_TOPK`, `_D`, `_INTER` fixed; `_kn1` (aiter kernel names) at :72; `_k_stagger` at :85 and `_fold_k_stagger` at :92; `_kernarg_preload` at :107; `_fp8_pack` with `cvt_pk_fp8_f32` at :237 and :246, used by the prep kernel at :386; `_fp8_pack_bf16` with `cvt_scalef32_pk_fp8_bf16` at :302, used by the folded stage 1 at :2429; `_rows16` at :3824; own stage 2 over the routed rows at :3831; the copy of aiter's v2 stage 2 past `_rows16` at :4350 (eb8af99c7e8ad333.json#fused_moe_v2_stage2.py:25); `_fold` (m ≤ 128) at :4388 and `_setup_fold` (8-route blocks) at :4394; buffers per call at :4445
- S13 aklr/core/contract/spec/_workloads.py:38 `qualifies`: a workload without `speedup` does not reach `AKLR_MIN_SPEEDUP` (:41); aklr/ops/README.md:338 `AKLR_MIN_SPEEDUP`, default 1.03
- S14 640d56e2cf48884d.json#aiter_stubs.py:1 stand-ins for aiter's `dtypes`, `fp8_e8m0`, `logger`, `ActivationType` and `get_cu_num`, the same file in 3279ada84a65bbd9 and eb8af99c7e8ad333; in the copies, a nested import of a module the file does not carry raising at 640d56e2cf48884d.json#moe_kernels.py:643, environment defaults at 640d56e2cf48884d.json#tensor_shim.py:24 and :27 and moe_kernels.py:2062, the argument-keyed pad cache turned off at moe_kernels.py:1177 and :1182 (its `global` is rejected by SA1), `.tolist()` replaced at tensor_shim.py:175 (SA3), and the launch-argument builders a listed Point reaches take the stream of their input's device (`dev`) at moe_kernels.py:838 and :930 (eb8af99c7e8ad333.json#moe_kernels.py:838, at 256 tokens); the copies' other `torch.cuda.current_stream()` calls, 18 in 640d56e2cf48884d and 20 in eb8af99c7e8ad333, are reached at no listed Point, and 3279ada84a65bbd9's copies launch nothing; 3279ada84a65bbd9.json#mx_types.py:1 keeps aiter's int enums only; eb8af99c7e8ad333.json#fused_moe_v2_stage2.py:1 the three functions copied from aiter's `fused_moe.py`
