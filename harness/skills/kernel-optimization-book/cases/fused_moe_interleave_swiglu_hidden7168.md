---
title: "MXFP4 fused MoE with clamped SwiGLU on gate/up-interleaved weights, hidden 7168, top-6 (384 experts at inter 384 or 768, or 96 local of 384 at inter 3072) — two FlyDSL launches on MX-fp8 activations with the routes taken one per workgroup row at 1–4 tokens and grouped by expert from 8 tokens (inter 384) or 16"
kind: case
ops: [moe]
dtypes: [mxfp4, bf16, fp8]
regimes: ["tokens 1-4", "tokens 8-64"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=4 (on earlier builds; the kernels as described here are not measured yet)
measured:
  - {metric: "speedup over aiter's fused MoE, per token count, of the measured build (the kernel's GPU code at m 16, 32; a fixed-scale fp8 intermediate at m 1, 2): one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays, L2 flushed before each call", value: "1.129–5.140; 4 of 4 token counts ≥ 1.03 (m 1, 2, 16, 32)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts, gate/up interleaved, SwiGLU limit 10, hidden 7168, top-6, 96 local of 384 experts under an expert mask, intermediate 3072; m 1, 2, 16, 32", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count, of the measured build (the kernel's GPU code at m 1–64), the same ruler", value: "1.081–3.808; 7 of 7 token counts ≥ 1.03 (m 1, 2, 4, 8, 16, 32, 64); also 1.030 at m 128, which the kernel does not take", gpu: MI355X, workload: "the same operand formats, hidden 7168, top-6, all 384 experts local, intermediate 384; m 1, 2, 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count, of the measured build (the kernel's GPU code), the same ruler", value: "1.151–3.295; 6 of 6 token counts ≥ 1.03 (m 1, 2, 4, 16, 32, 64)", gpu: MI355X, workload: "the same operand formats, hidden 7168, top-6, all 384 experts local, intermediate 768; m 1, 2, 4, 16, 32, 64", date: "2026-11-09"}
---

# MXFP4 fused MoE with clamped SwiGLU, gate/up-interleaved weights, hidden 7168, top-6

DeepSeek-V4-Pro's MoE (hidden 7168, 384 routed experts, top-6: `config.json` of `deepseek-ai/DeepSeek-V4-Pro`) in three
layer shapes, one kernel each: all 384 experts local at inter dim 384 (the inter-384 kernel) and 768 (the inter-768
kernel), and expert parallel with 96 local experts of 384 at inter dim 3072 (the 96-local kernel). "m" is the number of
tokens in the call, the input batch, not the rows an expert receives. A route is one (token, expert) pair, 6m per call.

Each kernel has two earlier builds. The later ones, called the measured builds below, allocated their scratch in every
call, as the kernels described here do; the ones before them, the per-shape-scratch builds, allocated it once per shape
(Counter-signals). The inter-384 and inter-768 kernels run the GPU code of their measured builds at every token count
they cover; the 96-local kernel runs that of its measured build at m 16 and 32, and at m 1 and 2 it scales the
intermediate per 32 values where that build used one scale derived from the limit (Counter-signals). The kernels as
described are not measured yet: the page cites runs of the earlier builds as such, and the gains of single changes come
from their optimization ladders.

The weight layout, gate mode, clamp, comparison and baseline are those of DeepSeek-V4-Flash's hidden-4096 kernels
([fused_moe_interleave_swiglu.md](fused_moe_interleave_swiglu.md) § Applies when); hidden 7168 has its own kernels.

## Applies when

- MXFP4 weights in aiter's gate/up-interleaved preshuffle: `w1` (experts, 2 × inter, 7168) with gate and up rows
  interleaved in 16-row tiles, `w2` (experts, 7168, inter), fp4 e2m1 with E8M0 scales per 1×32 stored 2-D. `gate_mode`
  interleave with `swiglu_limit`, bf16 `hidden_states` and `out`, fp32 `topk_weights`, int32 `topk_ids`, routing weight
  in stage 2. Every workload passes `swiglu_limit` 10.0.
- Expert parallel (the 96-local kernel): `expert_mask` over 384 global experts, 96 local; ids are global. No kernel here
  takes `num_local_tokens`: calls of this shape that pass it (m 1024 and 2048) have no kernel here.
- A token's expert ids are distinct (a precondition on `topk_ids`), and the inter-768 kernel relies on it when it ORs a
  token's bit into each of its experts' bitmaps.
- Accuracy: bounded, whole-output SQNR ≥ 13 dB against the reference. The reference keeps the activations unrounded in
  this gate mode; all three kernels quantize them to MX-fp8 (Counter-signals).
- Token counts covered: m 1, 2, 4, 8, 16, 32, 64 (inter 384); m 1, 2, 4, 16, 32, 64 (inter 768); m 1, 2, 16, 32 (96
  local). A token count counts as won from a speedup of 1.03 over the baseline. Not covered: m 128 and 256 at
  inter 384 (Counter-signals), expert parallel with 48 local experts (m 1–128) and calls that pass `num_local_tokens`,
  which have no kernel here.
- The baseline is aiter's `fused_moe` with `AITER_BF16_FP8_MOE_BOUND=16`: bf16 activations below 16 rows, fp8 from 16.

## Counter-signals

- **m 128 and 256 at inter 384 are not covered.** The inter-384 kernel takes at most 64 tokens. Its measured build also
  ran m 128 and read only 1.030 there *(field, E2)*.
- **Output bits vary where stage 2 adds with atomics.** The inter-768 kernel adds into `out` with packed bf16 atomics at
  m 1, 2 and 16–64. The 96-local and inter-384 kernels at every token count and the inter-768 kernel at m 4 sum each
  token's routes in a fixed order. Where two runs must give the same bytes the atomic token counts need a fixed-order
  stage 2 (`../moe/grouped_sorted_tokens.md` §6).
- **Activations and the intermediate pass through fp8.** All three quantize the token rows to fp8 e4m3 with an E8M0
  scale per 32 values and run stage 1 on the scaled MFMA. Stage 1 stores SiLU(gate)·up in bf16 (the inter-768 kernel at
  1 and 2 tokens and the inter-384 kernel at 2 and 4 store fp32 gate and up partials that stage 2 finishes); the
  inter-384 and inter-768 kernels at every token count, and the 96-local kernel at 1 and 2 tokens, requantize it to
  MX-fp8 in stage 2, while the 96-local kernel at 16 and 32 tokens keeps it bf16 (bf16 MFMAs on w2 converted from fp4).
  Each E8M0 scale comes from the absolute maximum of its own 32 values, so none depends on `swiglu_limit`, which may be
  anything from 7 up to float32's largest value. The measured build of the 96-local kernel instead stored the
  intermediate at 1 and 2 tokens as fp8 under one scale per call derived from the limit (|y| ≤ limit · max(limit,
  0.28)), for a limit in (0, 65536], which rounds a value below about limit² · 2⁻¹⁷ to zero: at limit 10, gate and up of
  1/64 everywhere (y ≈ 1.2e-4) give an all-zero intermediate, and at limit 1000 most values from the measured inputs
  round to zero (estimated). The earlier builds passed every token count with a worst violation fraction of 0 in four
  runs, all at limit 10, the only limit the workloads pass *(field, E1, E2, E3, E4, E9, E10)*; the SQNR itself is not
  recorded.
- **Scratch cached per shape is shared by concurrent calls.** Scratch allocated once per shape in a cached compile
  function, which every call rewrites before reading it, is safe only while one stream calls a shape at a time, since
  the buffer belongs to every call of that shape and two calls at once on two streams overwrite each other's
  intermediates. The per-shape-scratch builds did this: two concurrent calls with different inputs gave wrong output in
  most tries from 2 to 8 tokens up, depending on the kernel and on whether the streams shared CUs *(field, E11)*, while
  a second call with the same inputs writes the same values and passed *(field, E11)*. The kernels described here
  allocate their scratch in every call (96-local: local slots, the intermediate and the stage-2 token order; inter-384:
  the intermediate and the split-K partials; inter-768: the intermediate or its fp32 partials and the group table); the
  cached compile function keeps only the compiled launch and its shape constants. The measured builds, which allocate
  scratch the same way, passed with no differing try and no hang: three same-input conditions (a stream limited to 16
  and to 64 CUs, a second call with the same inputs) plus replay of one CUDA graph, 20 tries each per token count; one
  CUDA graph replayed with three input sets copied into its buffers in turn, 24 replays per token count; and two calls
  with different inputs at once on two streams, unrestricted and both on the same first 64 CUs, 20 tries each per token
  count *(field, E8)*. Not tested: the kernels as described here, a full-model decode graph, more than two concurrent
  streams, and the host cost of the per-call allocations in eager mode (the timer sums GPU kernel time and replays a
  graph, so it does not see them).
- **Group ranks follow arrival order in the inter-768 kernel at 32 and 64 tokens.** Its group-table launch ranks each
  route on its expert with an LDS counter, so which routes share a group changes between runs; each route keeps its own
  MFMA row, and stage 2 adds with atomics anyway. At 16 tokens the persistent stage 1 ranks from per-expert token
  bitmaps instead.
- **Token counts outside the measured ones.** The 96-local kernel takes up to 32 tokens (it asserts at most 256 routes
  and 128 local experts on its tile path), the inter-384 kernel up to 64, the inter-768 kernel 1, 2, 3, 4, 16, 32 and 64
  tokens without a mask; the entry point raises on any other call. m 3 of the inter-768 kernel and the counts between
  the measured ones were neither measured nor checked for accuracy.

## Mechanism

The gains this page gives for single changes (E5–E7) are steps along the optimization ladders that produced the
per-shape-scratch builds, every version timed against the version before it; they are not leave-one-out ablations of the
final kernel. E5–E7 say which steps were timed a second time.

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
- **Two launches** up to 64 tokens, with no routing launch but the inter-768 kernel's group table at 32 and 64: the
  expert of a route is read from `topk_ids` (under expert parallelism its local slot counted from the mask with ballots
  and `ctpop`). Weight loads are non-temporal in part: in the 96-local kernel stage 1's w1 from 2 tokens, its stage 2
  reading w2 under the default policy; in the inter-384 kernel stage 1's from 2 tokens (grids of more than 256
  workgroups) and stage 2's at m 2, 4, 16 and 32 only; in the inter-768 kernel both stages' at every token count. Every
  launch takes its leading kernel arguments preloaded in SGPRs (`amdgpu-kernarg-preload-count` 16, or `llvm.inreg`
  argument attributes in the 96-local kernel; `../gemm/small_m.md` §6), and launch constants are computed in the cached
  compile function, not in the `@flyc.jit` launcher. Moving the 96-local kernel's stage-1 grid size out of the launcher
  is worth +28.9% at m 1 and +8.4% at m 2 *(field, E5)*: a step that was not re-timed, and what made it that large is
  not established.

  ```python
  # in the cached compile function of the 96-local kernel: launch sizes as Python constants
  # A ternary on these inside the jit launcher made each stage-1 launch slower.
  GX1 = I // 16 // (4 // KW1 if DIRECT else 4)
  GY1 = R if DIRECT else min(R, E)
  NB1 = 256 if DIRECT else 256 * KW1

  @flyc.jit
  def dispatch(
      X: fx.Pointer,
      ...
      execute: fx.Int32,
      stream: fx.Stream,
  ):
      if execute != 0:
          _stage1(
              X, W1, WS1, Y, LIM, IDS, MASK, LOC, RW, PRM, D, I, R, TOP, KW1, G, DIRECT, PF1, NT1
          ).launch(grid=(GX1, GY1), block=(NB1,), stream=stream)
          ...
          # Arguments marked inreg arrive preloaded in user SGPRs, so the first loads need not wait
          # on the kernarg segment; stage 1 has room for its leading pointers and LIM only.
          for op in CompilationContext.get_current().gpu_module_body.operations:
              if op.OPERATION_NAME == "gpu.func":
                  nargs = len(op.regions[0].blocks[0].arguments)
                  npre = 8 if "_stage1" in str(op.attributes["sym_name"]) else nargs
                  dicts = [
                      ir.DictAttr.get({"llvm.inreg": ir.UnitAttr.get()})
                      if i < npre
                      else ir.DictAttr.get({})
                      for i in range(nargs)
                  ]
                  op.attributes["arg_attrs"] = ir.ArrayAttr.get(dicts)
  ```

  Not this: the same ternaries evaluated inside the `@flyc.jit` launcher. Each stage-1 launch was then slower, which is
  the step measured above.

**1–4 tokens: one route per workgroup row, K split over waves or workgroups.**

- The inter-384 kernel, m 1, 2, 4. Stage 1: workgroup (16-column tile, route), 4 waves splitting K = 7168; at m 2 and 4
  each route's last 10 or 4 tiles run as two K halves in separate workgroups, leaving fp32 gate and up sums for stage 2
  to fold, meant to keep some CUs from running one tile more than the rest; the load-ahead depth is set per grid size.
  Stage 2: workgroup (64 or, at m 4, 128 output columns, token), one wave per route; the token's SwiGLU intermediate
  rows (one per route) are quantized to MX-fp8 in LDS while every wave's w2 loads are in flight, and the routes'
  weighted sums are added in route order. Staging the token row first, with the load-ahead depth set per grid size, is
  worth +16.8, +20.8 and +10.7% at m 1, 2, 4 *(field, E6)*.
- The inter-768 kernel, m 1 and 2: stage 1 splits K evenly over the workgroups of a token (each runs the K tail of one
  tile and the head of the next as two MFMA rows) into fp32 gate/up partials, and zeroes `out`; stage 2, workgroup
  (columns, route), stages the route's SwiGLU of those partials in LDS as MX-fp8 once for all its waves, and each wave
  adds its routing-weighted tiles into `out` with packed bf16 atomics. m 4: stage 1 per (route, 16-column tile) with 7
  waves splitting K; stage 2 per token, one wave per route, each staging its row in LDS as MX-fp8, the weighted partials
  added in route order. Running both GEMMs on the scaled MX-fp8 × MXFP4 MFMA is worth +9.5, +6.1 and +3.1% at m 1, 2, 4,
  and the even K split with the atomic stage 2 +7.6 and +2.1% at m 1, 2 *(field, E7)*.
- The 96-local kernel (96 local of 384, inter 3072), m 1 and 2: stage 1 packs the live routes first (ballot over the
  local slots), one workgroup row per live route, 4 waves splitting K, each wave quantizing its slice of the token row
  into LDS; rows past the live count exit. A workgroup writes 16 columns of a route's intermediate in bf16, half of a
  32-value MX block, so the block's scale is taken where the whole block is loaded, in stage 2. Stage 2: one workgroup
  per 16 output columns walks the live routes of all tokens with 12 waves, so every workgroup carries the same work
  however the routes fall across tokens; it is compiled with a `waves_per_eu` hint of 6, set for two such workgroups per
  CU, an occupancy not measured. Each wave loads its K range of every live route's intermediate, 16 values per lane,
  quantizes it to MX-fp8 in LDS (a block's absolute maximum taken over two lanes, its E8M0 scale stored beside the data)
  and runs scaled MFMAs on it, each lane passing its block's scale. Giving each stage-2 wave a K range of every route
  with the intermediate in LDS, in a per-token stage 2, is worth +5.9 and +20.4% at m 1 and 2, and one stage-2 workgroup
  per column block over every token a further +0.9 and +4.6% *(field, E5)*; the per-32 scales are not measured yet. The
  quantization, on the dwords of bf16 pairs a lane holds:

  ```python
  def _quant_fp8(words, LG=2):
      # An MX block over LG lanes to fp8 e4m3, under the E8M0 scale that keeps amax below e4m3's top binade.
      lo = [(w << 16) for w in words]  # each bf16 half as the fp32 with the same value
      hi = [w & fx.Int32(-65536) for w in words]
      am = fx.Int32(0)
      for v in lo + hi:
          va = v & fx.Int32(0x7FFFFFFF)
          am = (va > am).select(va, am)
      d = 1
      while d < LG:  # the block's other lane(s)
          o = fx.Int32(am.shuffle_xor(d, 64))
          am = (o > am).select(o, am)
          d *= 2
      ex = am >> 23
      ex = (ex < 8).select(fx.Int32(8), ex)
      sbyte = ex - 7
      inv = ((fx.Int32(254) - sbyte) << 23).bitcast(fx.Float32)  # 1 / scale
      q = []
      for k in range_constexpr(len(words) // 2):
          p = fx.Int32(0)
          for h in range_constexpr(2):
              w = 2 * k + h
              p = fx.Int32(
                  rocdl.cvt_pk_fp8_f32(
                      T.i32, lo[w].bitcast(fx.Float32) * inv, hi[w].bitcast(fx.Float32) * inv, p, h == 1
                  )
              )
          q.append(p)
      return sbyte, q
  ```

**8–64 tokens: the routes of one expert grouped, so its weight tile is read once per group.**

- The inter-384 kernel, m 8–64: from 48 routes a route leads a group when its rank among its expert's routes is a
  multiple of 4, and its workgroup runs up to 4 routes of that expert as MFMA rows (the others' workgroups exit); stage
  2 as above with 32-column workgroups, in an XCD-grouped order at 32 and 64 tokens. Every stage-1 workgroup works its
  rank and its group's members out by itself from `topk_ids` (`ex` is its route's expert, `tile` its route, `NR` the
  group size; lane l of `ids_v[q]` holds the id of route l + 64 q):

  ```python
  # This route's rank among expert ex's routes; a block leads a group when NR divides its rank.
  msk = []
  hits = []
  for q in range_constexpr(len(ids_v)):
      h = ids_v[q] == ex
      if const_expr(64 * (q + 1) > R):
          h = h & (lane + 64 * q < R)
      hits.append(h)
      msk.append(fx.Int64(rocdl.ballot(T.i64, _raw(h))))
  lane_below = (fx.Int64(1) << fx.Int64(lane)) - 1
  rank = fx.Int32(0)
  total = fx.Int32(0)
  rks = []
  for q in range_constexpr(len(msk)):
      rks.append(total + fx.Int32(fx.ctpop(msk[q] & lane_below)))
      rel_t = tile - 64 * q
      sh = fx.Int64((rel_t > 0).select(rel_t, fx.Int32(0)) & 63)
      low = (rel_t >= 64).select(fx.Int64(-1), (rel_t > 0).select((fx.Int64(1) << sh) - 1, fx.Int64(0)))
      rank = rank + fx.Int32(fx.ctpop(msk[q] & low))
      total = total + fx.Int32(fx.ctpop(msk[q]))
  rank = _uniform(rank)
  total = _uniform(total)
  live = live & ((rank & (NR - 1)) == 0)  # the other routes' workgroups exit
  for j in range_constexpr(1, NR):  # the routes of rank + 1 .. rank + NR - 1 become MFMA rows 1 .. NR - 1
      mj = fx.Int32(-1)
      for q in range_constexpr(len(msk)):
          oh = fx.Int64(rocdl.ballot(T.i64, _raw(hits[q] & (rks[q] == rank + j))))
          mj = (oh != 0).select(fx.Int32(fx.ctpop(oh - 1)) + 64 * q, mj)
      members.append(_uniform(mj))
  ```
- The inter-768 kernel, m 16, 32, 64: groups of 2 (16 tokens) or 4 routes of one expert. At 32 and 64 a one-workgroup
  launch writes the group table; at 16 a persistent stage 1 on twice the CU count builds the same table in LDS in every
  workgroup from per-expert token bitmaps and also writes it for stage 2. Stage 1 runs each (16-column tile, group) with
  MFMA row g on the group's route g; stage 2 runs (columns, group) and adds into `out` with packed bf16 atomics. Moving
  m 32 and 64 from aiter to the grouped kernels is worth +7.5 and +8.0% there *(field, E7)*.
- The 96-local kernel, m 16 and 32: stage-1 workgroup row = one live local expert, its routes in groups of 16 over four
  16-column tiles, K split over 2 waves at 16 tokens; the token rows go through an LDS ring padded to keep
  `ds_read_b128` free of bank conflicts. Stage 1 also writes a stage-2 token order: tokens sorted by the first token
  that shares one of their experts, which brings some tokens sharing an expert next to each other but does not group
  them strictly; the intent is that the second read of that expert's w2 hits in cache. The order is worth +2.4% at m 32
  *(field, E5)*. Stage 2: workgroup (two 16-column blocks, token); wave w walks its K range of every live route with the
  intermediate in LDS, the routes summed in order in each wave and the waves in wave order. Folding the routing into
  stage 1 and moving m 16 from aiter onto this path is worth +12.3% at m 16 *(field, E5)*.

## Result

The three measured builds on an MI355X: the speedup is a one-sided Student-t lower bound of the mean log ratio of
aiter's time to the kernel's (α = 0.025 per look), read at the first look, 16 or 32 rounds, that decides every verdict
(in this run every token count at 16), of CUDA-graph replays with 1 GiB read before each call to flush L2. A call's time
is the summed durations of the GPU kernels it launches, not wall time. FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd,
torch 2.11.0+rocm10.0.0. The kernels as described here are not measured yet: the inter-384 and inter-768 kernels differ
from these builds only in host code, and the 96-local kernel keeps its measured build's GPU code at m 16 and 32 but not
at m 1 and 2.

| Measured build of | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 96 local of 384, inter 3072 *(field, E1)* | 5.140 | 1.890 | – | – | 1.235 | 1.129 | – |
| 384 experts, inter 384 *(field, E2)* | 3.808 | 3.155 | 2.537 | 1.868 | 1.578 | 1.205 | 1.081 |
| 384 experts, inter 768 *(field, E3)* | 3.295 | 2.391 | 1.871 | – | 1.240 | 1.151 | 1.172 |

"–": not measured for that shape. Every token count passed accuracy, at limit 10. At m 1 and 2 the measured build of the
96-local kernel ran its fixed-scale fp8 intermediate (Counter-signals), so these are not the numbers of the kernel
described here; the measured build of the inter-384 kernel also read 1.030 at m 128, which the kernel does not take. The
per-shape-scratch builds, which differ from these only in where the scratch is allocated (once per shape there, in every
call here; the GPU code is the same), were measured three more times, in two runs of the same ruler *(field, E10, E9)*
and one of its earlier version *(field, E4)*, and every token count the kernels cover (§ Applies when) reached 1.03 in
each; inter 384 at m 256, which the kernel does not take, read 1.020 in one of them *(field, E10)*. Across the four
runs a token count's bound moved by up to 5.6% (inter 768, m 1: 3.295 here, 3.331, 3.155 and 3.154 in the earlier
runs; inter 384, m 32: 1.205, 1.204, 1.162, 1.201), and at the smallest margin by at most 0.01 (inter 384, m 64:
1.081, 1.086, 1.091, 1.087). At one token the measured builds' GPU time is 10.69, 9.55 and 12.13 µs against aiter's 55.09, 36.55 and 40.01 µs
(median of the rounds, in the table's order) *(field, E1, E2, E3)*.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, one-sided Student-t lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays with 1 GiB read before each call to flush L2, a call's time the summed durations of its GPU kernels, one MI355X, aiter 0.1.21.dev66+gdbd8bf5bd] the measured build of the 96-local kernel (96 local of 384 experts, inter 3072; scratch allocated in every call; at m 1 and 2 an fp8 intermediate under one limit-derived scale): speedup over aiter 5.140, 1.890, 1.235, 1.129 at m 1, 2, 16, 32; median kernel 10.69, 36.65, 134.96, 169.35 µs, aiter 55.09, 69.59, 166.77, 191.37 µs; accuracy passed, worst violation fraction 0
- E2 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1] the measured build of the inter-384 kernel (384 experts, inter 384; the same GPU code at m 1–64; it also ran m 128): speedup over aiter 3.808, 3.155, 2.537, 1.868, 1.578, 1.205, 1.081, 1.030 at m 1, 2, 4, 8, 16, 32, 64, 128; median kernel 9.55, 12.55, 19.99, 34.96, 63.14, 102.30, 173.49, 242.68 µs, aiter 36.55, 39.82, 50.84, 65.36, 99.68, 123.59, 188.44, 250.27 µs; accuracy passed, worst violation fraction 0
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1] the measured build of the inter-768 kernel (384 experts, inter 768; the same GPU code): speedup over aiter 3.295, 2.391, 1.871, 1.240, 1.151, 1.172 at m 1, 2, 4, 16, 32, 64; median kernel 12.13, 19.87, 35.31, 119.18, 199.38, 325.28 µs, aiter 40.01, 47.71, 66.30, 147.99, 229.82, 383.28 µs; accuracy passed, worst violation fraction 0
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the earlier version of the ruler of E1] the per-shape-scratch builds (scratch allocated once per shape, otherwise the GPU code of E1–E3), speedup over aiter per token count: of the 96-local kernel 5.103, 1.879, 1.236, 1.129 at m 1, 2, 16, 32; of the inter-384 kernel, measured as a copy with the same kernel code, 3.797, 3.173, 2.444, 1.863, 1.539, 1.201, 1.087, 1.030, 1.041 at m 1, 2, 4, 8, 16, 32, 64, 128, 256; of the inter-768 kernel 3.154, 2.321, 1.839, 1.234, 1.146, 1.151 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every token count
- E5 [2026-11, gfx950, FlyDSL ?, N=?, ablation ladder: each version on the ladder timed beside the version before it, the earlier version's time over the later one's minus one; recorded = timed when the version joined the ladder, remeasured = timed again by the ablation in 3 pairs] the ladder that produced the per-shape-scratch build of the 96-local kernel: stage-1 grid size set in the cached compile function instead of in the jit launcher +28.9, +8.4% at m 1, 2 (recorded); stage 2 splitting each token's routes into per-wave K ranges with the intermediate in LDS +5.9, +20.4% at m 1, 2 (recorded); routing folded into the per-expert stage 1 and m 16 moved from aiter to that path +12.3% at m 16 (recorded); one stage-2 workgroup per column block covering every token +0.9, +4.6% at m 1, 2 (remeasured); stage-2 token order by the first token sharing an expert +2.4% at m 32 (remeasured)
- E6 [2026-11, gfx950, FlyDSL ?, N=?, ablation ladder, as E5] the ladder that produced the per-shape-scratch build of the inter-384 kernel: token row staged first and stage-1 load-ahead depth set per grid size +16.8, +20.8, +10.7, +8.8% at m 1, 2, 4, 8 (recorded)
- E7 [2026-11, gfx950, FlyDSL ?, N=?, ablation ladder, as E5] the ladder that produced the per-shape-scratch build of the inter-768 kernel: m 32 and 64 moved from aiter to kernels that deduplicate experts in both stages +7.5, +8.0% at m 32, 64 (recorded); both GEMMs on scaled MX-fp8 × MXFP4 MFMAs with activations quantized in LDS +9.5, +6.1, +3.1% at m 1, 2, 4 (recorded); stage 1 split evenly over workgroups with SwiGLU finished in a per-route atomic stage 2 +7.6, +2.1% at m 1, 2 (remeasured)
- E8 [2026-11, gfx950, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd, N=20 tries per token count and condition (24 replays for the graph with new inputs), stress, graph-replay and concurrent-call checks] the three measured builds of E1–E3 (scratch allocated in every call): same-input conditions (a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream) plus replay of one captured CUDA graph, each try against the call alone; one CUDA graph captured on input set 0 and replayed with three input sets copied into its buffers in turn, against an eager call on the same inputs; two calls at once on two streams, each on its own input set and into its own outputs, unrestricted and both on the same first 64 CUs, against an eager call alone: 0 differing tries, no hang
- E9 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds (32 at inter 384, m 128), the ruler of E1] a second run of the per-shape-scratch builds (scratch allocated once per shape): speedup over aiter: of the 96-local kernel 5.164, 1.865, 1.240, 1.128 at m 1, 2, 16, 32; of the inter-384 kernel 3.792, 3.074, 2.500, 1.816, 1.555, 1.162, 1.091, 1.033 at m 1, 2, 4, 8, 16, 32, 64, 128, at m 128 median kernel 243.91 µs against aiter's 251.72, decided at 32 rounds; of the inter-768 kernel 3.155, 2.364, 1.861, 1.232, 1.138, 1.157 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every token count
- E10 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1] a first run of the per-shape-scratch builds (scratch allocated once per shape): speedup over aiter: of the 96-local kernel 5.165, 1.866, 1.244, 1.124 at m 1, 2, 16, 32; of the inter-384 kernel, measured as a copy with the same kernel code, which also runs m 256, 3.756, 3.149, 2.545, 1.872, 1.578, 1.204, 1.086, 1.030, 1.020 at m 1, 2, 4, 8, 16, 32, 64, 128, 256, median kernel 243.15 and 276.42 µs against aiter's 250.84 and 282.84 at m 128 and 256; of the inter-768 kernel 3.331, 2.321, 1.849, 1.235, 1.151, 1.176 at m 1, 2, 4, 16, 32, 64; accuracy passed, worst violation fraction 0 at every token count
- E11 [2026-11, gfx950, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd, N=20 tries per token count and condition, concurrent-call check] the per-shape-scratch builds under two calls at once on two streams, each on its own input set, against an eager call alone; differing tries of 20 with unrestricted streams, then with both streams on the same first 64 CUs: of the 96-local kernel m 1 0, 0; m 2 18, 20; m 16 and 32 20, 20; of the inter-384 kernel m 1 0, 1; m 2 0, 19; m 4 0, 20; m 8, 16, 32 and 64 20, 20; m 128 0, 0; of the inter-768 kernel m 1 0, 0; m 2 0, 20; m 4 16, 20; m 16, 32 and 64 20, 20 (the same builds under a same-input second call and the other same-input conditions of E8, 20 tries each: 0 differing)
