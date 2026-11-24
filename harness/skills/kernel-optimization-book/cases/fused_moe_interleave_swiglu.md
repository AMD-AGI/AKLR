---
title: "MXFP4 fused MoE with clamped SwiGLU on gate/up-interleaved weights, hidden 4096, top-6 (256 experts, or 32 / 64 local of 256) — one workgroup per route and bf16 MFMAs at 1–8 tokens; routes sorted into expert blocks inside stage 1, MX-fp8 activations and a mostly atomic stage 2 at 16–256 tokens"
kind: case
ops: [moe]
dtypes: [mxfp4, bf16, fp8]
regimes: ["tokens 1-2 (expert parallel)", "tokens 1-8 (all experts local)", "tokens 16-256"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's fused MoE, per token count, on the earlier version of the ruler of § Result", value: "3.251–3.448; 2 of 2 token counts ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "the per-route kernel; bf16 tokens × MXFP4 experts, gate/up interleaved, SwiGLU limit 10, hidden 4096, top-6, 32 local of 256 experts under an expert mask, intermediate 2048; m 1, 2", date: "2026-11-05"}
  - {metric: "speedup over aiter's fused MoE, per token count, on the earlier version of the ruler of § Result", value: "2.878–4.094; 2 of 2 token counts ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "the per-route kernel; the same operand formats, hidden 4096, top-6, 64 local of 256 experts under an expert mask, intermediate 2048; m 1, 2", date: "2026-11-05"}
  - {metric: "speedup over aiter's fused MoE, per token count, on the earlier version of the ruler of § Result", value: "2.006–2.920; 4 of 4 token counts ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "the per-route kernel; the same operand formats, hidden 4096, top-6, all 256 experts local, intermediate 256; m 1, 2, 4, 8", date: "2026-11-05"}
  - {metric: "speedup over aiter's fused MoE, per token count, on the earlier version of the ruler of § Result", value: "1.503–2.894; 4 of 4 token counts ≥ 1.03 (m 1, 2, 4, 8)", gpu: MI355X, workload: "the per-route kernel; the same operand formats, hidden 4096, top-6, all 256 experts local, intermediate 512; m 1, 2, 4, 8", date: "2026-11-05"}
  - {metric: "speedup over aiter's fused MoE, per token count: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays, L2 flushed before each call", value: "1.339–1.622; 5 of 5 token counts ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "the sorted-block kernel for 32 local of 256 experts under an expert mask, intermediate 2048 (bf16 tokens × MXFP4 experts, gate/up interleaved, SwiGLU limit 10, hidden 4096, top-6), measured on an earlier build with the same GPU code; m 16, 32, 64, 128, 256", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays, L2 flushed before each call", value: "1.241–1.406; 5 of 5 token counts ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "the sorted-block kernel for 64 local of 256 experts under an expert mask, intermediate 2048 (the same operand formats, hidden 4096, top-6), measured on an earlier build with the same GPU code; m 16, 32, 64, 128, 256", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays, L2 flushed before each call", value: "1.358–1.962; 5 of 5 token counts ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "the sorted-block kernel for all 256 experts local, intermediate 256 (the same operand formats, hidden 4096, top-6), measured on an earlier build with the same GPU code; m 16, 32, 64, 128, 256", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count: one-sided lower bound of the aiter / kernel time ratio over 16 rounds of CUDA-graph replays, L2 flushed before each call", value: "1.250–1.662; 5 of 5 token counts ≥ 1.03 (m 16, 32, 64, 128, 256)", gpu: MI355X, workload: "the sorted-block kernel for all 256 experts local, intermediate 512 (the same operand formats, hidden 4096, top-6), measured on an earlier build with the same GPU code; m 16, 32, 64, 128, 256", date: "2026-11-09"}
---

# MXFP4 fused MoE with clamped SwiGLU, gate/up-interleaved weights, hidden 4096, top-6

DeepSeek-V4-Flash's MoE (hidden 4096, 256 routed experts, top-6: `config.json` of `deepseek-ai/DeepSeek-V4-Flash`) in
four layer shapes: expert parallel with 32 or 64 local experts of 256 and inter dim 2048, and all 256 experts local at
inter dim 256 or 512. Each shape has two kernels, split by token count:

- **1–8 tokens**: one kernel source for the four shapes, called the per-route kernel below; "the 32-local shape", "the
  64-local shape", "the inter-256 shape" and "the inter-512 shape" are that kernel on each of them.
- **16–256 tokens**: one kernel per shape, called below the 32-local kernel, the 64-local kernel, the inter-256 kernel
  and the inter-512 kernel, and together the sorted-block kernels. The three built on aiter's FlyDSL code carry copies
  of the aiter modules they use and import no aiter at run time.

"m" is the number of tokens in the call, the input batch, not the rows an expert receives. A route is one (token,
expert) pair, 6m per call; how many land on one expert depends on the routing.

## Applies when

- The same layout at hidden 7168 (DeepSeek-V4-Pro) has its own kernels:
  [fused_moe_interleave_swiglu_hidden7168.md](fused_moe_interleave_swiglu_hidden7168.md).
- MXFP4 weights in aiter's gate/up-interleaved preshuffle: `w1` (experts, 2 × inter, 4096) with gate and up rows
  interleaved in 16-row tiles and fp4 e2m1 in 16 × 64-byte blocks, `w2` (experts, 4096, inter), E8M0 scales per 1×32
  stored 2-D. `gate_mode` interleave with `swiglu_limit` (gate clamped to at most the limit, up to ± the limit, then
  SiLU(gate)·up). bf16 `hidden_states`, fp32 `topk_weights` (asserted by the per-route kernel and the inter-256 kernel),
  int32 `topk_ids`, routing weight in stage 2, bf16 `out`.
- Activations: the reference the output is compared with keeps them unrounded in this gate mode, and aiter's fp8
  activations past `AITER_BF16_FP8_MOE_BOUND` rows count as within the bound. The per-route kernel keeps them bf16; the
  sorted-block kernels quantize them to MX-fp8 (Counter-signals).
- Expert parallel shapes: `expert_mask` over 256 global experts, 32 or 64 local; ids are global. The other two shapes
  have all 256 experts local and no mask. A token's expert ids are distinct (a precondition on `topk_ids`); the 32-local
  kernel sizes its expert blocks on it. No kernel here takes `num_local_tokens`: calls of this shape that pass it (m
  1024 and 2048) have no kernel here.
- Every workload passes `swiglu_limit` 10.0. Limits from 7.0 are admitted, but the interleave gate mode is checked from
  DeepSeek-V4's 10 only, since aiter's CK-Tile decode path does not clamp.
- Accuracy: bounded, whole-output SQNR ≥ 13 dB against the reference. The per-route kernel reduces in a fixed order with
  no atomics; the sorted-block kernels add into `out` with atomics, except the inter-256 kernel at 16 and 32 tokens.
- Token counts: m 1, 2 (expert parallel) and m 1, 2, 4, 8 (all local) for the per-route kernel *(field, E1, E17)*; m 16,
  32, 64, 128 and 256 of every shape for the sorted-block kernels, which a separate run read at 1.235 or more
  *(field, E22)*. A token count counts as won from 1.03 over the baseline. No kernel here covers another token count
  of these shapes.
- The baseline is aiter's `fused_moe` with `AITER_BF16_FP8_MOE_BOUND=16`, sglang's DeepSeek-V4 setting: below 16 rows
  aiter keeps bf16 activations, from 16 it runs fp8. It is the yardstick at every token count.

## Counter-signals

**1–8 tokens (the per-route kernel)**

- **The kernel stops at 8 tokens.** Its path for more than 64 routes (a mask-scan launch, a routing launch, routes
  binned per local expert into 16-row tiles, four launches) is reached by no measured token count: the largest route
  count it runs is 48 (m 8, all local). m 16–256 belong to the sorted-block kernels.
- **15 of 16 MFMA rows are discarded.** Each route sits alone in its 16-row MFMA tile in both stages (every A row is
  the same token or route; only row 0 is stored). What that costs at these sizes is not measured.
- **Expert-parallel grids carry routes that are not local.** Stage 1 launches a workgroup row for every route; a route
  to an expert off this rank finds no local slot and its workgroups exit. Stage 2 takes a data-dependent branch: a token
  with exactly one local route spreads that route's K over all 6 waves, otherwise each local route's wave walks its own
  K. Which branch the measured inputs exercise is not recorded.
- **The baseline does not clamp.** aiter's CK-Tile decode path does not apply `swiglu_limit`; the interleave gate mode
  is checked from 10 only so that a limit binding on the measured inputs does not leave the comparison without a
  baseline. Whether the clamp binds on the measured draws at 10 is not established.
- **Per-shape scratch, shared by concurrent calls.** The compile function, cached per shape with `functools.cache`,
  allocates the kernel's six scratch buffers once per shape: each route's local expert slot, the intermediate that stage
  1 stores, and four for the unreached path. On the per-route path each call writes the slot for every route and the
  intermediate for every local route before stage 2 reads them; the other buffers serve only the unreached path. Two
  calls of one shape and token count at once on two streams share them: with different inputs (20 tries per token count,
  on two unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call
  alone), the 32-local shape gave wrong output on the shared CUs at m 1 in 2 and m 2 in 20 of 20 tries, the inter-256
  shape there at m 2, 4 and 8 in 1, 3 and 20 of 20, and the inter-512 shape there at m 2–8 in 15–20 of 20 and at m 8 in
  15 of 20 on unrestricted streams; the 64-local shape, the same kernel source, in none *(field, E14)*. That is
  consistent with the shared scratch, not shown by a per-call-allocation control on this kernel, and does not explain
  why the 64-local shape did not fail.

**16–256 tokens (the sorted-block kernels)**

- **Output bits vary between runs.** Stage 2 adds each block's weighted rows into `out` with bf16 atomics in arrival
  order: all four kernels at every token count, except the inter-256 kernel at 16 and 32 tokens, which adds a token's
  routes in LDS in route order. Where two runs must give the same bytes they need a fixed-order stage 2
  (`../moe/grouped_sorted_tokens.md` §6).
- **Activations and the intermediate pass through fp8.** The token rows are quantized to fp8 e4m3 with an E8M0 scale per
  32 values, and SiLU(gate)·up again before stage 2, where the reference keeps both unrounded. Every token count passed
  with a worst violation fraction of 0 *(field, E3, E4, E5, E6)*; the SQNR itself is not recorded. aiter's path from 16
  rows also runs fp8 activations. Values that differ below fp8's resolution within a group lose the difference: an input
  of 1.015625 against 1.0 around a gate that cancels them reads about 0 dB (estimated); aiter's fp8 path rounds it the
  same way, and that rounding counts as within the bound.
- **At 16 tokens several workgroups store the same activation rows.** In the 32-local kernel every stage-1 workgroup of
  an expert block quantizes the block's token rows into the shared fp8 activation buffer and its scale buffer, waits for
  its own stores and reads back only rows it stored. Two workgroups that store one row store the same bytes, the same
  bf16 row through the same quantization, so no reader can see a mixed row (code read).
- **Built on copies of aiter's FlyDSL code.** The 32-local, 64-local and inter-512 kernels carry the aiter modules they
  use among their sources, copied from aiter dbd8bf5bd, and import no aiter at run time. A copy differs from aiter's
  file in its imports, in environment reads replaced by their defaults, in nested imports of modules the kernel does not
  carry replaced by `raise NotImplementedError`, in a few host-side edits (an argument-keyed pad cache turned off, a
  `.tolist()` replaced), and in the launch-argument builders a measured token count reaches taking the stream of their
  input's device; the copy of aiter's MX type module keeps only its int enums, and one more module holds only the three
  functions of aiter's `fused_moe.py` that the inter-512 kernel calls. The 32-local kernel runs aiter's stage-1 and
  stage-2 builders with its own source edits, one edited copy per configuration a measured token count uses: eight,
  about 1.1 MB of the kernel's 1.7 MB of source. The inter-512 kernel takes stage-1 settings by aiter's tuned kernel
  names (`get_flydsl_kernel_params`) and carries aiter's v2 stage-2 launcher; the 64-local kernel has its own builders
  and copies aiter's helpers. A later aiter neither changes nor fixes them: a fix reaches them only by copying it in.
- **Shapes are fixed in the code.** The inter-256 kernel asserts hidden 4096 and inter 256; the inter-512 kernel
  hard-codes 256 experts, top-6, hidden 4096 and inter 512; the 32-local kernel assumes 32 local of 256 experts, hidden
  4096, inter 2048 and at most 256 tokens, and has builders for the five measured token counts only; the 64-local kernel
  asserts hidden = 16 × global experts and at most 64 local experts. None of them runs DeepSeek-V4-Pro's hidden 7168.
- **Scratch has to be allocated in each call.** The four kernels allocate their routing buffers, intermediates and
  scratch inside the call, and their earlier builds (Result) gave 0 differing tries at every token count under three
  checks: same-input conditions (CU-limited streams, a second call with the same inputs, graph replay), one CUDA graph
  replayed with three input sets, and two calls at once with different inputs as above *(field, E12)*. The same GPU code
  as the inter-256 kernel with its scratch allocated once per shape in the cached compile function passed the first two
  and gave wrong output under the third in 20 of 20 tries at 32–256 tokens, and at 16 tokens on the shared CUs
  *(field, E13)*: the same-input conditions cannot show scratch that only receives plain stores of input-determined
  values, as this kernel's does; scratch that each call zeroes and then accumulates into with atomics can show under
  them, but enqueued back to back only in 1–2 of 200 tries on a toy HIP op *(field, E15)*. The inter-256 kernel's
  compile function keeps only the sizes; the launch closure allocates:

  ```python
  @functools.cache
  def _build(x_shape, ids_shape, w1_shape, ids_dtype, rw_dtype, device):
      ...
      # scratch, allocated by each call so that calls on two streams never share it
      sizes = ((R * I, torch.bfloat16), (E * R, torch.int32), (E, torch.int32))
      ...
      def launch(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, swiglu_limit, out):
          ws = tuple(torch.empty((n,), dtype=dtype, device=hidden_states.device) for n, dtype in sizes)
          inputs = (hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, *ws, out)
          limit = 3.0e38 if swiglu_limit is None else swiglu_limit
          compiled(
              *(_ptr(t) for t in inputs), limit, 1, fx.Stream(torch.cuda.current_stream(hidden_states.device))
          )

      return launch
  ```

  Not this: the `ws = tuple(torch.empty(…) …)` line in the cached compile function, outside `launch`. Every call of that
  shape then gets the same buffers, and two calls at once with different inputs read each other's stores (the wrong
  outputs above).
- **Code no measured token count reaches.** The inter-256 kernel asserts at most 1536 routes and no expert mask, which
  every measured token count meets. The inter-512 kernel's unfolded path below 256 tokens (w1 loads 2 K tiles ahead,
  stage 1's prologue tiles issued early, the K-stagger off) and its copy of aiter's v2 stage 2 above 256 tokens, the
  64-local kernel's default configuration for token counts outside its table, and the copied aiter modules' paths for
  other dtypes, layouts and epilogues run at no measured token count.

## Mechanism

**1–8 tokens: one workgroup per route and 16 columns, two launches.** Stage 1 then stage 2 at every measured token count
(the per-route path runs up to 64 routes); no routing kernel, no atomics, no workspace counters.

Stage 1 (per-route path). Grid (inter / 16, routes): one workgroup per (route, 16 intermediate columns), 4 waves;
96–1536 workgroups (inter 256: 96, 192, 384, 768 at m 1–8; inter 512: 192–1536; inter 2048 with 6 or 12 routes: 768,
1536).
- The expert: all local, the route's id; expert parallel, the workgroup builds the mask's four 64-expert words with wave
  `ballot`s and counts the set bits below the id with `ctpop`, which gives the local slot (or −1: the workgroup exits).
  The workgroup at column block 0 writes the slot to a per-route table for stage 2.

  ```python
  # gid: the route's global expert id; G: global experts; every lane of the wave runs this (ballots)
  inside = (gid >= 0) & (gid < max(G, E))
  # Bit g of word g // 64 is set when global expert g lives on this rank; the local
  # slot of gid is the number of set bits below it.
  safe = inside.select(gid, fx.Int32(0))
  q = safe >> 6
  below = fx.Int32(0)
  word = fx.Int64(0)
  for k in range_constexpr((G + 63) // 64):
      g = lane + k * 64
      held = (_p(MASK, fx.Int32)[(g < G).select(g, fx.Int32(0))] != 0) & (g < G)
      bits = fx.Int64(rocdl.ballot(T.i64, _raw(held)))
      below = below + (q > k).select(fx.Int32(fx.ctpop(bits)), fx.Int32(0))
      word = (q == k).select(bits, word)
  b = fx.Int64(safe & 63)
  below = below + fx.Int32(fx.ctpop(word & ((fx.Int64(1) << b) - 1)))
  e = (inside & (((word >> b) & 1) != 0)).select(below, fx.Int32(-1))
  ```
- K = 4096 is split over the 4 waves, 1024 each (8 chunks of 128). Each wave stages its own slice of the token's row
  in LDS with its own lanes, so no barrier; then issues every scale and weight load of its 8 chunks before the first
  MFMA, in the order row, scales, weights, pinned with `sched_barrier` so `vmcnt` retires them in that order.
- Weights: one 16-byte load per lane per chunk for the gate tile and one for the up tile of the interleaved pair; each
  dword of 8 fp4 values becomes 8 bf16 with the block scale applied by `cvt_scalef32_pk_bf16_fp4`; one scale dword holds
  the gate and up bytes of two chunks. Non-temporal loads under expert parallelism, cached otherwise
  (`../moe/grouped_sorted_tokens.md` §4). The cached loads are meant to let routes that share an expert reuse it from
  L2; at 8 tokens with L2 flushed, non-temporal loads were 9.5–13% slower in five of six cases where 6 or 8 experts took
  all 48 routes and within 2.4% under random routing, with a HIP copy's L2 hit rates at 7–26% either way
  *(field, E19, E20)*, so little of that reuse is in L2.

  ```python
  def _e8m0(byte):  # E8M0 scale byte -> the fp32 power of two
      return (fx.Int32(byte) << 23).bitcast(fx.Float32)

  def _fp4x8(word, scale):
      # Eight fp4 (byte b: elements 2b low nibble, 2b + 1 high) times scale, as v8bf16.
      v2bf16 = ir.Type.parse("vector<2xbf16>")
      words = []
      for sel in range_constexpr(4):
          pair = rocdl.cvt_scalef32_pk_bf16_fp4(v2bf16, _raw(word), _raw(scale), sel)
          words.append(fx.Vector(pair).bitcast(fx.Int32)[0])
      return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)

  # in the K walk, chunk i: bg, bu are the gate and up 16-byte loads, word the scale dword of chunks 2j, 2j + 1
  sh = (i % 2) * 16
  sg = _e8m0((word >> sh) & 255)
  su = _e8m0((word >> (sh + 8)) & 255)
  gf = [_fp4x8(bg[k], sg) for k in range(4)]
  uf = [_fp4x8(bu[k], su) for k in range(4)]
  ```
- `mfma_f32_16x16x32_bf16`, 4 per chunk for gate and 4 for up, into fp32 (`../gemm/mainloop.md` §5).
- Wave reduction: the 4 waves' partials through LDS, summed in wave order (`../gemm/small_m.md` §4). Wave 0 clamps,
  applies SiLU with `exp2` and `rcp`, rounds to bf16 and stores the route's 16 values to the intermediate buffer.

Stage 2. Grid (4096 / 16, m): one workgroup per (token, 16 output columns), 6 waves, one per route; 256–2048 workgroups.
- Each wave walks its route's K = inter (2, 4 or 16 chunks of 128) with `mfma_f32_16x16x32_bf16` on `w2` converted the
  same way. The unrolled walk takes chunks in blocks of 4 whose intermediate, weight and scale loads all issue before
  the block's first MFMA. Expert parallel, it always runs; all local, it runs when m × inter / 128 ≤ 8, and a runtime
  loop of one chunk at a time runs above (inter 256 at m 8, inter 512 at m 4, 8).
- Row 0 of each wave's tile goes to LDS; 16 threads add the 6 routes × routing weight in route order, skipping routes
  that are not local, and store bf16 (`../gemm/correctness.md` §8).

No sort and one route per workgroup set, as the 1–2-token kernels of
[fused_moe_shuffled_16x16.md](../cases/fused_moe_shuffled_16x16.md); `../moe/few_tokens.md` §1 on why the launch count
matters at these sizes.

**16–256 tokens: routes sorted into expert blocks, MX-fp8 activations on scaled MFMAs, stage 2 adding into
`out`.** The four kernels share one plan and differ in where the routing runs and how the blocks are cut.

- **The routing runs inside stage 1 up to 128 tokens.** Every stage-1 workgroup rebuilds the sort of the call's routes
  for its own expert block(s) in LDS from `topk_ids` (ballot ranks, or an LDS histogram and a prefix scan), so no
  workgroup waits on another and no routing launch runs; the sorted ids and routing weights of each block are published
  for stage 2, and the grid zeroes `out` in stripes for stage 2's adds (`../moe/few_tokens.md` §2). Under expert
  parallelism the local slot comes from the mask by ballot, and routes to other ranks join no block. At 256 tokens the
  32-local, 64-local and inter-512 kernels run one launch before stage 1 that sorts and quantizes, and in the 64-local
  and inter-512 kernels also zeroes `out` (the 32-local kernel zeroes it inside stage 1); the 32-local kernel's launch
  adds 256 workgroups that request the first half of the start k-tile of each local expert's W1 from the XCD that stage
  1 runs it on. That made stage 1 about 2% faster at m 256, but not through L2 hits: a probe with the same requests and
  placement saw none in the next launch *(field, E21)*. The inter-256 kernel has no sort: at 16 tokens (96 routes) stage
  1 runs one workgroup per (route, 16 columns) as the per-route kernel does; from 32 tokens one workgroup per (expert,
  32 columns) collects its expert's routes by ballots in route order and runs them 16 per pass, and its column block 0
  publishes the list for stage 2. One FlyDSL launch for sort, quantization and zeroing in place of aiter's, paired
  against the version before it: about +12.7% at m 16 down to +4.9% at m 256 on the 64-local kernel, +14.2% to +6.3% on
  the inter-512 kernel *(field, E8)*. That launch folded into stage 1, paired the same way: +5.2%, +1.4% and +0.8% at m
  16, 32 and 64 (inter-512 kernel); +2.8% at m 16, then +3.8% and +3.3% at m 16 and 32 when the fold reached 32 tokens
  with a one-wave routing head and the quantization before the W1 request (32-local kernel); +0.8% and +1.3% at m 16 and
  64 (64-local kernel) *(field, E9)*. The inter-256 kernel's route list, built by every workgroup of expert `e` from the
  ids of all `R` routes (lane l of pass k holds the id of route l + 64 k):

  ```python
  # Expert e's routes in route order into lst (other lanes write dead slot R); barrier before reading.
  NB = (R + 63) // 64
  below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
  count = fx.Int32(0)
  for k in range_constexpr(NB):
      hit = gids[k] == e
      if R % 64 != 0 and k == NB - 1:
          hit = hit & (lane + k * 64 < R)
      bits = fx.Int64(rocdl.ballot(T.i64, _raw(hit)))
      pos = count + fx.Int32(fx.ctpop(bits & below))  # this route's rank among the expert's routes
      slot = (hit & (wave == 0)).select(pos, fx.Int32(R))
      lst[slot] = lane + k * 64
      count = count + fx.Int32(fx.ctpop(bits))
  return _uniform(count)
  ```
- **Activations as MX-fp8.** Token rows are quantized to fp8 e4m3 with one E8M0 scale per 32 values from the group's
  absolute maximum, reduced across lanes. The scale is applied by `cvt_scalef32_pk_fp8_bf16` in the inter-256 kernel, in
  the stage 1 of the 64-local and inter-512 kernels up to 128 tokens, and in the 32-local kernel's A loader at 32–128
  tokens; the separate routing launches at 256 tokens and the 32-local kernel's fold at 16 tokens multiply by the power
  of two themselves and convert with `cvt_pk_fp8_f32`. Both stages multiply fp8 activations with the fp4 weights on the
  scaled MFMA, both E8M0 scales as operands (`mfma_scale_f32_16x16x128_f8f6f4` in the three kernels that carry their
  GEMM code; the 32-local kernel runs its edited copies of aiter's fp8 × fp4 builders; `../gemm/mainloop.md` §5). The
  32-local kernel at 32–128 tokens quantizes inside stage 1's A loader, straight into the LDS A tile, so the fp8 copy
  never goes to memory. Stage 1's epilogue clamps, applies SiLU(gate)·up and stores the intermediate as MX-fp8
  (inter-256 kernel: bf16, requantized per 32 in stage 2).
- **Blocks cut to the routes.** The 64-local kernel sorts into 16-row expert blocks; the inter-512 kernel puts at most 8
  routes (16 at 256 tokens) into each 32-row block and skips the MFMAs of rows 16–31; the 32-local kernel bounds its
  32-row blocks by the routes (at most 32 blocks up to 32 tokens, where every expert's routes fit one block because a
  token's expert ids are distinct). Padding rows read past the end of the activations, which returns zeros without a
  memory request (the 32-local kernel; the 64-local kernel at 16, 64 and 128 tokens).
- **Weights streamed once, K starts staggered.** Weight loads are non-temporal in most stages
  (`../moe/grouped_sorted_tokens.md` §4); the inter-256 kernel's per-route stage 1 at 16 tokens reads w1 cached. The
  routes of one expert read each 16-column w1 tile from one L2 (placement below) whether the read is cached or
  non-temporal (equal L2 hits on a HIP copy of its load pattern), and with L2 flushed the cached read was 2–4% faster
  only when many routes shared an expert *(field, E19, E20)*. Workgroups start their K loop at different K tiles and
  wrap: the 32-local kernel by expert block in both GEMMs, the 64-local kernel per stage-2 workgroup, the inter-512
  kernel per stage-1 column tile and block. Experts streamed at once then do not hit the same address bits together (the
  32-local kernel's stated reason); each block's accumulation order stays fixed (`../gemm/small_m.md` §6 on racing a
  K-start stagger). The rotation in the 32-local kernel, paired against the version before it: +2.8% to +7.7% per token
  count *(field, E10)*.
- **Stage 2 adds into `out`.** Each block's routing-weighted rows go to `out` with packed bf16 atomics
  (`../moe/grouped_sorted_tokens.md` §6). The inter-256 kernel does so from 64 tokens (384 routes) with one workgroup
  per (expert, 256 or, at 256 tokens, 512 output columns); at 16 and 32 tokens it runs one workgroup per (token, 32
  columns), one wave per route, and adds the routes in LDS in route order. The per-expert stage 1 with the expert-major
  stage 2, paired against the version before it (per-route at 32 tokens, aiter above): +10.4% to +21.5% at m 32–256;
  atomics in place of a combine kernel: +4.0% to +5.6% at m 64–256 *(field, E11)*.

  ```python
  # from 64 tokens: two adjacent bf16 outputs added into `out` by one packed atomic
  def _atom_pk(p, dword, v0, v1):
      buf = _buf(p, fx.BFloat16, 2)
      reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.BFloat16)
      reg.store(fx.Vector.from_elements([fx.BFloat16(v0), fx.BFloat16(v1)], fx.BFloat16))
      atom = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16)
      fx.copy(atom, reg, fx.slice(buf, (None, dword)))

  # at 16 and 32 tokens: each wave leaves its route's weighted row in LDS, then one fixed-order sum and a plain store
  fx.gpu.barrier()
  if tid < BN:
      total = fx.Float32(0.0)
      col0 = bxx * BN
      for j in range_constexpr(TOP):
          total = total + partials[j * BN + tid]
      _p(OUT, fx.BFloat16)[token * D + col0 + tid] = fx.BFloat16(total)
  ```

  Not the atomic form where two runs must give the same bytes: its adds land in arrival order (Counter-signals).
- **Launch arguments, occupancy and placement.** The inter-256 and inter-512 kernels preload 16 kernel arguments into
  SGPRs (`../gemm/small_m.md` §6); the 64-local kernel pads stage 1's LDS so that at most 7 (16 tokens), 10 (32 tokens)
  or 8 workgroups share a CU; the inter-256 kernel places the per-route stage-1 workgroups so that blocks whose index
  agrees mod 8 share an XCD (which XCD is rotated on some launches) and the routes of one expert read each 16-column w1
  tile from one L2 *(field, E16)* (`../gemm/schedule_and_locality.md` §2).

Per token count:

| Kernel | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 32 local | routing in stage 1 (one-wave head); 64-column stage-2 tiles | routing in stage 1; activations quantized in the A loader | the same, LDS-histogram routing head | the same, two blocks per stage-1 workgroup | separate routing launch that also warms W1 |
| 64 local | routing in stage 1; 2-wave stage 2, K tile 128 | the same, 10 workgroups per CU | routing in stage 1; tiles dealt to XCDs in contiguous ranges | the same | separate routing launch; stage-1 K tile 1024 |
| inter 256 | per-route stage 1; per-token stage 2 | per-expert stage 1; per-token stage 2 | per-expert stage 1; expert-major atomic stage 2 | the same | the same; stage-2 weights issued before the route count is read |
| inter 512 | routing in stage 1, 8-route blocks | the same | the same | the same | separate routing launch, 16-route blocks |

## Result

Both sets of numbers use one ruler: a one-sided Student-t lower bound of aiter's time over the kernel's after 16 or 32
rounds of CUDA-graph replays with L2 flushed before each call; the readings in the tables below are of the version that
collects kernel times with a rocprofiler-sdk tracer.

**1–8 tokens**, from three runs of the ruler on an MI355X (the 32-local shape; the 64-local and inter-512 shapes; the
inter-256 shape) *(field, E1, E17)*. Bold: 1.03 or above.

| Shape | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 32 local of 256, inter 2048 | **3.297** | **3.343** | – | – |
| 64 local of 256, inter 2048 | **2.805** | **4.053** | – | – |
| 256 local, inter 256 | **2.941** | **2.697** | **2.440** | **2.021** |
| 256 local, inter 512 | **2.991** | **2.535** | **1.853** | **1.479** |

"–": not measured for that shape. Every workload passed the accuracy check. Across the runs of the ruler's earlier
version, a token count moved by up to 0.19 (32 local, m 2: 3.255–3.448; 64 local, m 2: 3.929–4.120), and none came near
1.03 *(field, E2)*.

**16–256 tokens**, measured on earlier builds of the four kernels (on an MI355X, FlyDSL 0.3.2, aiter
0.1.21.dev66+gdbd8bf5bd, torch 2.11.0+rocm10.0.0). Those imported aiter at run time; the current kernels run the same
GPU code from their own copies of aiter's modules. A separate run, its ruler not stated in the record, measured the
current kernels at 1.235–1.959, every token count at or above 1.03 and passing accuracy *(field, E22)*. Bold: 1.03 or
above.

| Kernel | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 32 local of 256, inter 2048 *(field, E3)* | **1.622** | **1.485** | **1.432** | **1.339** | **1.350** |
| 64 local of 256, inter 2048 *(field, E4)* | **1.406** | **1.363** | **1.285** | **1.243** | **1.241** |
| 256 local, inter 256 *(field, E5)* | **1.962** | **1.692** | **1.414** | **1.358** | **1.362** |
| 256 local, inter 512 *(field, E6)* | **1.662** | **1.432** | **1.318** | **1.263** | **1.250** |

All 20 token counts passed accuracy with a worst violation fraction of 0. A second run for the 32-local, 64-local and
inter-512 kernels read 1.330–1.646, 1.242–1.407 and 1.256–1.660 *(field, E7)*; the inter-256 kernel's GPU code read
1.333–1.948 and 1.342–1.970 in two runs with its scratch allocated once per shape *(field, E5)*. Median call times at 16
and 256 tokens: 25.9 and 78.3 µs against aiter's 42.0 and 105.9 (32 local); 58.2 and 140.3 against 81.8 and 174.4 (64
local); 25.4 and 76.5 against 50.0 and 104.3 (inter 256); 45.7 and 140.5 against 76.0 and 176.0 (inter 512)
*(field, E3, E4, E5, E6)*.

## Evidence

- E1 [2026-11, gfx950, FlyDSL 0.3.2 (the pin of aiter dbd8bf5, installed over the image for the run), N=1, one-sided Student-t lower bound of the aiter / kernel time ratio after 16 or 32 rounds of CUDA-graph replays with L2 flushed before each call, kernel times from a rocprofiler-sdk tracer, one MI355X] speedup over aiter, the per-route kernel on 32 local of 256 experts, inter 2048: 3.297, 3.343 at m 1, 2 (3.251, 3.448 on the ruler's earlier version, E2); accuracy passed at both token counts
- E2 [2026-11, gfx950, FlyDSL ?, N=6, the earlier version of the ruler of E1; N=3 for 256 local, inter 256] range over the runs, the per-route kernel: 32 local of 256, inter 2048 m 1 3.190–3.364, m 2 3.255–3.448; 64 local of 256, inter 2048 m 1 2.756–2.886, m 2 3.929–4.120; 256 local, inter 256 m 1 2.894–2.957, m 2 2.702–2.723, m 4 2.366–2.429, m 8 1.984–2.015; 256 local, inter 512 m 1 2.894–3.051, m 2 2.467–2.592, m 4 1.831–1.880, m 8 1.496–1.512
- E3 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1 on one MI355X] speedup over aiter 1.622, 1.485, 1.432, 1.339, 1.350 at m 16, 32, 64, 128, 256; median kernel 25.87, 41.57, 55.63, 74.18, 78.33 µs, aiter 41.99, 61.91, 79.69, 99.51, 105.93 µs; accuracy passed, worst violation fraction 0 (32 local of 256 experts, inter 2048: an earlier build of the 32-local kernel)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1 on one MI355X] speedup over aiter 1.406, 1.363, 1.285, 1.243, 1.241 at m 16, 32, 64, 128, 256; median kernel 58.16, 78.43, 110.41, 133.22, 140.32 µs, aiter 81.81, 106.91, 141.91, 166.05, 174.35 µs; accuracy passed, worst violation fraction 0 (64 local of 256 experts, inter 2048: an earlier build of the 64-local kernel)
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1 on one MI355X] speedup over aiter 1.962, 1.692, 1.414, 1.358, 1.362 at m 16, 32, 64, 128, 256; median kernel 25.42, 39.65, 57.21, 68.24, 76.50 µs, aiter 49.95, 67.09, 81.28, 92.76, 104.30 µs; accuracy passed, worst violation fraction 0 (256 experts local, inter 256: an earlier build of the inter-256 kernel); the same GPU code with its scratch allocated once per shape read 1.948, 1.669, 1.420, 1.333, 1.356 and 1.970, 1.687, 1.422, 1.352, 1.342
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1 on one MI355X] speedup over aiter 1.662, 1.432, 1.318, 1.263, 1.250 at m 16, 32, 64, 128, 256; median kernel 45.67, 73.46, 108.29, 128.16, 140.47 µs, aiter 75.98, 105.59, 142.96, 163.51, 176.03 µs; accuracy passed, worst violation fraction 0 (256 experts local, inter 512: an earlier build of the inter-512 kernel)
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=16 rounds, the ruler of E1 on one MI355X, a second run of the same builds] speedup over aiter at m 16, 32, 64, 128, 256, earlier builds of: the 32-local kernel 1.646, 1.503, 1.436, 1.330, 1.352; the 64-local kernel 1.407, 1.371, 1.269, 1.242, 1.255; the inter-512 kernel 1.660, 1.425, 1.318, 1.275, 1.256; accuracy passed, worst violation fraction 0
- E8 [2026-11, gfx950, FlyDSL ?, N=?, paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] one FlyDSL launch for sort, activation quantization and zeroing of `out` in place of aiter's: the 64-local kernel +12.7, +7.6, +5.5, +5.2, +4.9% at m 16, 32, 64, 128, 256; the inter-512 kernel +14.2, +9.4, +7.4, +6.5, +6.3%
- E9 [2026-11, gfx950, FlyDSL ?, N=?, paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] that launch folded into stage 1: the inter-512 kernel up to 64 tokens +5.2, +1.4, +0.8% at m 16, 32, 64; the 32-local kernel at 16 tokens +2.8% at m 16, then up to 32 tokens with a one-wave routing head and the quantization before the W1 request +3.8, +3.3% at m 16, 32; the 64-local kernel at 16 and 64 tokens, dropping its prologue launch, +0.8, +1.3% at m 16, 64
- E10 [2026-11, gfx950, FlyDSL ?, N=?, paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] the 32-local kernel, each block's K loop started at a rotated k-tile in both GEMMs: +2.8, +5.8, +7.7, +4.2, +4.4% at m 16, 32, 64, 128, 256
- E11 [2026-11, gfx950, FlyDSL ?, N=?, paired timing: the version before the change over the version with it, minus one; a step may bundle more than the change named] the inter-256 kernel's GPU code (as a build with its scratch allocated once per shape): per-expert stage 1 with in-kernel routing and an expert-major stage 2 +10.4, +21.5, +14.1, +14.6% at m 32, 64, 128, 256; bf16 atomics into `out` in place of the combine kernel +4.0, +4.8, +5.6% at m 64, 128, 256
- E12 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count (24 graph replays), stress checks on MI355X] 0 differing tries for the earlier builds of the 32-local, 64-local, inter-512 and inter-256 kernels at every token count under (a) same-input conditions: a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream, replays of one captured graph; (b) one CUDA graph replayed 24 times with three input sets copied into its buffers, each replay compared with an eager call on the same inputs; (c) two calls at once on two streams, each with its own input set and outputs, on unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone
- E13 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count, the checks of E12] the inter-256 kernel's GPU code with its scratch allocated once per shape in the cached compile function: under (c) wrong output at m 16 in 0 of 20 tries on unrestricted streams and 20 of 20 on the shared 64 CUs, at m 32, 64, 128 and 256 in 20 of 20 under both; under (a) and (b) 0 differing tries
- E14 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count, check (c) of E12 on the per-route kernel] wrong output, tries out of 20 on unrestricted streams / on the shared 64 CUs: 32 local of 256, inter 2048 m 1 0/2, m 2 0/20; 256 local, inter 256 m 1 0/0, m 2 0/1, m 4 0/3, m 8 0/20; 256 local, inter 512 m 1 0/0, m 2 0/16, m 4 0/15, m 8 15/20; 64 local of 256, inter 2048 0 at every token count
- E15 [2026-11, gfx950 (MI355X), HIP 7.15, N=200 tries per condition, two calls on two streams sharing scratch, same inputs, exact compare] scratch given plain stores of input-determined values: 0 of 200 wrong; scratch zeroed then added into with atomics: 1–2 of 200 enqueued back to back, 200 of 200 with both calls behind one event (a toy HIP routed op, not one of these kernels)
- E16 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=224 launches, `HW_REG_XCC_ID` recorded per block] block L of a launch on XCD (L + c) mod 8, one c per launch: c = 0 in 104 of 224 launches, 5–7 in the rest; blocks of one (expert, 16-column w1 tile) on more than one XCD in 0 of 32,256 groups (the inter-256 kernel's per-route stage 1, m 16, 8 and 13)
- E17 [2026-11, gfx950, FlyDSL 0.3.2, N=1, the ruler of E1, two more runs] speedup over aiter, the per-route kernel: 64 local of 256, inter 2048 2.805, 4.053 at m 1, 2 and 256 local, inter 512 2.991, 2.535, 1.853, 1.479 at m 1, 2, 4, 8 (one run); 256 local, inter 256 2.941, 2.697, 2.440, 2.021 at m 1, 2, 4, 8 (another run); accuracy passed at every token count
- E19 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=15 rounds, CUDA events, a 1 GiB read before each call, median non-temporal / cached time] stage 1 with only its w1 loads' cache bit changed: the per-route kernel's (32 local of 256, inter 2048) at m 8 (48 routes) 1.095–1.132 in 5 of 6 cases with 6 or 8 experts (0.966 with the same 6 at 256 local, inter 2048), 0.988–1.024 random; the inter-256 kernel's per-route one at m 16 and 8 1.021–1.039 with 6 or 8 experts, 0.993–1.001 random
- E20 [2026-11, gfx950 (MI355X), HIP 7.15, N=5 dispatches per arm, rocprofv3 TCC hits and misses, cache flushed before each] L2 hit rate cached / non-temporal: the inter-256 kernel's pattern at m 16 0.889 / 0.889 (6 experts) and 0.871 / 0.871 (8); the per-route kernel's (32 local of 256, inter 2048) at m 8, 6 or 8 experts, 0.070–0.258 / 0.202–0.264 (a HIP kernel with only stage 1's w1 loads, not one of these kernels)
- E21 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=8 rounds, CUDA graph, kernel times from a rocprofiler-sdk tracer, a 1 GiB read before each call; probe HIP 7.15, N=9 rounds] the 32-local kernel at m 256: stage 1 1.021× as fast with its routing launch's W1 requests (8 of 8 rounds); a HIP probe with the same requests and XCD placement: the next launch's L2 misses 133,296 with and without them
- E22 [2026-11, gfx950, FlyDSL 0.3.2, N=?, a separate run; ruler not stated in the record] the current sorted-block kernels, speedup over aiter at m 16, 32, 64, 128, 256: 32 local of 256, inter 2048 1.624, 1.510, 1.446, 1.338, 1.351; 64 local of 256, inter 2048 1.412, 1.389, 1.293, 1.268, 1.235; 256 local, inter 256 1.959, 1.701, 1.437, 1.374, 1.345; 256 local, inter 512 1.687, 1.439, 1.319, 1.277, 1.253; all 20 token counts at or above 1.03, accuracy passed at each
