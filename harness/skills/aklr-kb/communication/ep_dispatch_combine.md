---
title: "EP against TP for a MoE layer at a few tokens per GPU: what EP's dispatch costs, and what removed part of it"
kind: topic
ops: [moe]
dtypes: [mxfp4, fp8]
regimes: ["few tokens per GPU", "tokens per GPU 1-96"]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - https://github.com/ROCm/aiter/issues/5322
  - ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/
  - xiaobochen-amd/aiter@40ae5ad7f (branch kyle/megamoe-ep8-campaign-20260910)
---

# EP dispatch and combine at a few tokens per GPU

## Route here when

- Choosing expert parallelism (EP) or tensor parallelism (TP) for a MoE layer at a few tokens per GPU (decode
  batch sizes, for instance).
- An EP MoE layer is slower than its GEMMs explain.
- Writing or changing a dispatch between GPUs. `MegaMoEV2`'s stage 1 runs producers and consumers in one launch
  (workgroups wait on flags that other workgroups of the launch write; whether those waits are bounded was not
  checked), and the `mega_moe` op's FlyDSL kernels in AKLR, taken from `MegaMoEV2`, still wait that way inside
  one launch (S18). Such a wait hangs or reads garbage when the launch's workgroups are not all resident
  (`../gemm/correctness.md` §4), so a new AKLR kernel bounds it with hand-off (`../SKILL.md` § Stop) or uses a
  separate launch or a last-arriver count on an atomic counter; either way what carries over from §2 is the layout
  (fixed regions, parity halves) and where the acquire goes.

The kernels themselves (files, entry points, tuning tables) are in `../flydsl/kernels/flydsl_kernel_library.md`;
the MoE GEMMs in `../moe/INDEX.md`.

## Where this holds

- One node of eight MI355X; GLM-5.2 (hidden size 6144, 256 routed experts, top-8, one shared expert, MXFP4 weights)
  served by sglang with TP8 attention; EAGLE speculative decoding (5 steps, 6 draft tokens) at a simulated acceptance
  length of 3.61, so a speculative cycle (five draft forwards and one verify) gives each request about 3.61 output
  tokens. A verify forward runs 75 MoE layers (from the reproduction's report), and the draft's MoE layer
  runs five times per cycle, likely through `MegaMoEV2` too.
- 14 concurrent requests: a verify forward has 84 tokens, and attention holds all of them on every GPU. Under EP8,
  sglang reduce-scatters attention's output before each MoE layer, so `MegaMoEV2` gets about 84 / 8 ≈ 11 of them per
  GPU (read from sglang `b96766116`, S10; not checked at run time). Under TP8 every GPU's MoE gets all 84.
- With 32 prompts of fixed length, concurrency 14 runs as about 14 + 14 + 4 requests, so E1's and E2's gaps mix 14
  requests in flight with a tail of 4; 8 and 16 divide the 32 evenly. That third wave is likely the issue's "cliff" at
  14 between 8 and 16 (estimated). TP8 was not measured at 8 or 16.
- Not measured: other concurrencies, calls of many tokens per GPU (a long prompt's prefill), DP attention, EP
  across nodes, and dispatch kernels other than
  `MegaMoEV2`'s.

## 1. What EP costs against TP

- **The layouts differ by more than the all-to-all.** TP splits every expert across the GPUs along its intermediate
  dimension: each GPU computes its slice of every route, and an allreduce joins the slices. EP gives each GPU whole
  experts: a token's activations go to the GPUs that hold its experts (dispatch) and the results come back
  (combine). Both hold the same routed-expert bytes per GPU. In the measurements below the EP8 arm also:
  - reduce-scattered attention's output before each MoE layer and all-gathered it before the next attention, in
    place of TP's allreduce fused with RMSNorm (S10);
  - ran the shared expert unfused (`--disable-shared-experts-fusion`), most likely on the same stream before
    `MegaMoEV2`: the two-stream path needs `alt_stream`, which ROCm creates only with
    `SGLANG_ROCM_USE_MULTI_STREAM`, not set in E1's or E8's launch (S11);
  - kept the shared expert whole on every GPU (S15): each GPU reads all of it in every layer, eight times TP8's
    share. How much of the gap that is was not measured.
- *(field, E1)* EP8 with `MegaMoEV2` served 20.7% fewer output tokens per second than TP8/EP1, with a 29.1% longer
  time per output token (TPOT). *(field, E2)* Reproduced at 22.5% and 30.9%: 2.22 ms per output token, about 8 ms
  per cycle (estimated).
- **The GEMM shapes were not the difference.** *(field, E3)* On one GPU, aiter's FlyDSL two-stage MoE kernel (the
  `mixed_moe_gemm_2stage` family that TP8's arm runs, here with FP4 activations and the bench's own tiles; not
  `MegaMoEV2`'s fused FP8 × FP4 stage 1) took 8112 µs in the EP8 shape (32 whole experts per GPU) and 7946 µs in the
  TP8 shape (256 experts, intermediate split eight ways): 2.1% apart, EP8 the slower. Each time is 75 layers at the
  draft's token count plus 75 at the verify's, not one cycle's MoE. How `MegaMoEV2`'s own GEMMs compare was not
  measured.
- **Most of the gap is not in an isolated timing of the dispatch.** Before §2's changes, the reproduction's report
  put the dispatch timed on its own at about 30–40 µs per layer, about 2.6 ms over 75 layers (estimated), against a gap of about 8 ms per cycle. The two commits of §2 brought
  EP8's TPOT to 8.27 ms *(field, E11)*, about 3.9 ms per cycle behind TP8 (estimated; the two arms were timed in
  different sessions). Only r4's dispatch was timed on its own: about 18 µs per layer less, about 1.4 ms per cycle
  (estimated from E9), of the 1.5–2 ms that r4 gained end to end (estimated from its 4.72%). The token count of
  these operator timings was not recorded.
- **`MegaMoEV2`'s time grows far slower than its tokens.** *(field, E4)* The whole operator (dispatch, both GEMMs,
  combine) took 2.2× as long at 96 tokens per GPU as at 1, for 96× the tokens. The experts its GEMMs read grow only
  from about 7 to all 32 per GPU (estimated, uniform routing), and a fixed per-launch cost weighs at 1 token
  (inferred). The dispatch's own growth was not measured, nor whether EP's layout wins at larger batches.
- **What is left in EP8's cycle after §2.** *(field, E11)* Cutting GEMM1's K from 24 steps to 2 raised EP8 from 1167
  to 1308 tok/s (TPOT 8.27 → 7.29 ms), about 3.5 ms per cycle (estimated), against about 4.5 ms to read the 29.4 GB of
  weights removed at 6.49 TB/s (estimated): most of GEMM1's weight read is not overlapped with anything. The probe
  would speed TP8 too; it sizes GEMM1 within EP8's cycle and does not explain the remaining gap.

## 2. What the dispatch protocol pays for

Four states of the code, read at the commits named:

| State | Layout for GLM-5.2's decode | Barrier between the GPUs at each launch | Consumer's system-scope acquire |
|---|---|---|---|
| Baseline `dbd8bf5` (its stage 1 is the measured `70f50bb10`'s, line for line) | compact | yes | per tile, in every wave, on the compact path |
| Campaign r3 `a40fd4980` | direct fixed slots | yes | per wave, after the workgroup barrier |
| Campaign r4 `40ae5ad7f` | direct fixed slots, two parity halves | compact layout only | once per workgroup, before the workgroup barrier |
| aiter main `790fae18f` | compact (fixed slots only at eight GPUs with 48 experts per GPU, S4) | none on the compact path GLM-5.2 takes; the fixed-slot path only (S4); how main's compact path avoids overwriting a peer's rows was not read | not read here |

The campaign branch continues past r4; the numbers here stop at r4. Neither r3 nor r4 is upstream. The baseline is
not an ancestor of the fork the issue measured: the two share `mega_moe_stage1.py`, but sglang `b96766116` calls
the fork's `MegaMoEV2.set_weights`, which the baseline lacks.

- **A packed layout waits for every GPU's counts.** In the compact layout a GPU knows where its rows go in a peer's
  buffer only once every GPU's counts are in. With a fixed region per expert, a producer claims a row with one
  system-scope atomic on the destination's counter (S5) and writes without waiting for the counts; at the baseline
  it still waits at the launch barrier below. `MegaMoEV2` uses fixed slots only when at most 255 tokens per GPU are
  reserved (S1; the reserve is a power of two, so 128 or fewer), and its direct fixed-slot dispatch only at eight
  GPUs with 48 experts per GPU (S2). GLM-5.2's decode reserved 256 tokens per GPU (E1's launch) and has 32 experts
  per GPU, so it ran the compact layout.
  - *(field, E8)* The commit that opened the direct fixed-slot path to GLM-5.2 gained 4.87% in output throughput. It
    also retuned the 64- and 128-token configurations (stage-1 tile rows, `num_dispatch_cu`, stage-2 tiles) and how
    producers split the destinations; the share of each was not measured. At the ~11 tokens per GPU of a verify
    forward the configuration bucket is 8 (84 tokens pick 64; S16), so the retuned 64- and 128-token configurations
    were likely not selected in E8's serving run.
  - The same commit fixed defects of the direct path that do not depend on the expert count: the capacity check
    used the compact layout's row count; `group_done`, the count of this GPU's finished producer workgroups, had one
    element while each launch's reset writes one per GPU; and the consumers took only an agent-scope acquire (S6).
    It allowed fixed slots above 255 reserved tokens at eight GPUs and up to 64 experts per GPU, while experts × GPUs
    × tokens per GPU × row bytes ≤ 512 MiB, a bound that trades buffer size against handshakes (S17).
  - Cost: fixed slots reserve room for the worst case on every destination, twice with the parity halves below
    (for GLM-5.2's decode about 384 MiB per half, estimated); at token counts far above the 256 reserved per GPU
    there (a long prompt's prefill) it does not fit.
- **A barrier between the eight GPUs opens every launch.** At each stage-1 launch one owner workgroup posts its
  epoch to every peer and spins until theirs arrive, and the launch's other workgroups wait at its gate (S3), so that
  the launch does not overwrite rows a peer is still reading from the previous one. Two parity halves remove the
  need: a launch writes the half its parity selects (S7), and the cross-GPU completion count (`count_done`) already
  keeps the GPUs within one launch of each other, so the half being written is never the one being read (S8).
  Without that bound, dropping the barrier is a data race.
- **A system-scope acquire per wave.** r4 also moved the consumer's system-scope acquire from once per wave to once
  per workgroup: 2048 to 256 per layer for 256 workgroups of 8 waves (estimated from the configuration). The fixed
  slots are complete once the plan has landed, so the thread that waits for the plan's flag issues the acquire, and
  the workgroup barrier after it releases every wave (S9). The order is the point: wait, acquire, then barrier. One
  wave issuing it after the barrier leaves the other waves free to read stale lines; every wave issuing it is the
  per-wave cost again.
  - *(field, E9)* With the barrier gone and the acquire per workgroup, the dispatch went from 37.4 to 19.4 µs per
    layer and output throughput rose 4.72%. The two were measured together.
- *(field, E10)* With both commits EP8 trailed TP8 by 15% instead of 22.5%.

## 3. Changes that did not move it

- **More CUs for the dispatch.** *(field, E5)* The best `num_dispatch_cu` from the fork's
  `tune_mega_moe_v2.py --sweep dispatch-cu` (not in the baseline aiter) was 1.1% faster than the default. The
  dispatching workgroups and GEMM1's share one launch (S12), and the dispatching ones join GEMM1 when they are done
  (S13): more dispatch CUs shorten only the dispatch phase.
- **Launch geometry from the tuning table.** `MegaMoEV2` does not read the EP8 table's dispatch rules: its dispatch
  runs inside stage 1 on `num_dispatch_cu` workgroups (S12), and only the standalone
  `FlyDSLDispatchCombineIntraNodeOp.dispatch()` looks the rules up (S14). *(field, E6)* Six (block count, warps per
  block) rules added for hidden size 6144 gave stage-1 times within 0.31% of each other, no different from no rule
  and inside the 0.64% noise of that timing.
- **Chunking the dispatch payload.** *(field, E7)* `payload_chunk_rows`, which only the compact layout has, made
  stage 1 65% slower; why was not analyzed.

## 4. Measuring it

- Time the operator at the tokens per GPU it gets in the deployment: about 11 per verify forward under EP8 above,
  not the 84 that attention sees (estimated, S10). The token count picks the configuration bucket (11 tokens pick 8,
  84 pick 64; S16), the time grows 2.2×
  from 1 to 96 tokens (§1), and the best `num_dispatch_cu` was not the same at every token count (E5). The operator
  timings here (E5–E7, E9) did not record the token count; the noise figure is from 84.
- Judge EP against TP end to end, at the deployment's concurrency. An operator's gain is diluted by its share of
  the step: *(field, E12)* `MegaMoEV2` 4.6–11.5% faster on its own gave 2.7% and 3.5% more output tokens at
  concurrency 8 and 16.
- A collective's time in a trace includes waiting for the other GPUs:
  `../pitfalls/nograph_trace_inflates_collectives.md`.

## Evidence

- E1 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2 per aiter's pin, N=3 (median of three after a discarded warm-up; the first EP8 repeat was a cold outlier), sglang serving bench, 32 prompts, 1024 in / 256 out, concurrency 14, EAGLE 5 steps / 6 draft tokens, simulated acceptance length 3.61, decode reserve 256 tokens per GPU] TP8/EP1 1372.68 output tok/s, TPOT 7.13 ms; tuned EP8 `MegaMoEV2` 1089.03 tok/s, 9.21 ms (https://github.com/ROCm/aiter/issues/5322)
- E2 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=3, E1's protocol, aiter 70f50bb10 plus the environment commit 4ba4205c6 on the campaign branch, sglang b96766116] TP8/EP1 1372.10 tok/s, 7.18 ms; EP8 1063.75 tok/s, 9.40 ms
- E3 [2026-09, gfx950, FlyDSL 0.3.2, N=?, one GPU, a MoE minibench: aiter's `flydsl_moe_stage1`/`stage2` with FP4 activations and FP4 weights, replayed from a CUDA graph, 75 layers at M = concurrency and at M = 6 × concurrency, concurrency not recorded (the tool's default is 10)] EP8 shape 8112 µs, TP8 shape 7946 µs; the report's bandwidth column does not follow from its bytes and times and is not used
- E4 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=?, operator timing of the whole operator per layer; the tool is not named in the cited lines (the issue's validation runs `test_mega_moe_v2.py --network glm52 --max-tok-per-rank 256`)] 1 → 96 tokens per GPU: 2.2× the time
- E5 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=?, `tune_mega_moe_v2.py --sweep dispatch-cu`, aiter version and token counts not recorded] best `num_dispatch_cu` 1.1% faster than the default; at 32 tokens per GPU the default was best
- E6 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=?, `test_mega_moe_v2.py` operator timing, token count not recorded; that timing's noise 0.64% at 84 tokens] six dispatch geometries for hidden size 6144: stage-1 times within 0.31% of each other, no different from no rule
- E7 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=?, operator timing, token count not recorded] `payload_chunk_rows`: stage 1 65% slower
- E8 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=1 (one bench per candidate, scored against the best recorded in an earlier batch of runs, not a paired arm; the same commit measured 1053.33 and 1063.08 in two sessions), the campaign ruler (E1's workload; three warm-ups, median of 3, of the middle 3 of 5 when the spread exceeds 3%; spread 0.92%), the version r3 and r4 ran not recorded] direct fixed-slot dispatch for GLM-5.2 and the commit's other changes: 1063.08 → 1114.86 tok/s, +4.87% (xiaobochen-amd/aiter@a40fd4980, its message and diff)
- E9 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=1 (one bench per candidate), E8's ruler; the per-layer dispatch time from operator timing, N=?] parity halves without the launch barrier, acquire per workgroup: 1114.86 → 1167.52 tok/s, +4.72%; dispatch 37.4 → 19.4 µs per layer (xiaobochen-amd/aiter@40ae5ad7f)
- E10 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=1 (one bench per candidate), E8's ruler; TP8/EP1 from E2, a separate measurement and protocol] EP8 1063.08 → 1167.52 tok/s against TP8/EP1's 1372.10
- E11 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2, N=1 (one bench per arm), E8's ruler, a subtractive probe] GEMM1 K 24 → 2 steps: 1167 → 1308 tok/s, TPOT 8.27 → 7.29 ms
- E12 [2026-09, gfx950 (MI355X ×8), FlyDSL 0.3.2 per aiter's pin, N=?, E1's workload at concurrency 8 and 16; in isolation, the operator's cross-rank maximum per token bucket] tuned against untuned `MegaMoEV2`: 4.6–11.5% faster in isolation; 898.08 → 922.45 output tok/s at concurrency 8, 1491.59 → 1543.57 at 16 (https://github.com/ROCm/aiter/issues/5322)
- S1 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py:52 `compact = self.mtpr > FIXED_SLOT_MAX_MTPR`, with `FIXED_SLOT_MAX_MTPR = 255` at `mega_moe_config.py:27`; the reserve must be a power of two (`mega_moe_v2.py:37-38`)
- S2 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:42 `_use_direct_fixed_slot` is true only with fixed slots enabled and `npes == 8 and experts_per_rank == 48 and cap == required_cap`
- S3 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:251-252 the owner workgroup's first `npes` threads store this launch's epoch into each peer's `launch_ready` slot, then wait until the peer's has arrived; the other workgroups wait at the owner's gate (:277)
- S4 ROCm/aiter@790fae18f:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:59 fixed slots other than eight GPUs with 48 experts per GPU raise; `launch_ready` is set up (:327-329) and the barrier run (:364-394) only on the fixed-slot path
- S5 xiaobochen-amd/aiter@40ae5ad7f:aiter/ops/flydsl/kernels/mega_moe/dispatch.py:288-292 a producer claims its row with `atomic_add_system` on the destination's per-expert counter
- S6 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py:127 `group_done` has one element, while the direct path's reset writes `fz_npes` of them (`mega_moe_stage1.py:263-264`); its consumers take `fence_agent_acquire` (:402), and the per-tile `fence_system_acquire` runs only off the direct path (:450-451); fixed in xiaobochen-amd/aiter@a40fd4980
- S7 xiaobochen-amd/aiter@40ae5ad7f:aiter/ops/flydsl/kernels/mega_moe/dispatch.py:260-261 "Parity mirrors the slot region, so this launch never overwrites rows the destination is still consuming for the previous launch."
- S8 xiaobochen-amd/aiter@40ae5ad7f:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:152-154 "Direct fixed slots mirror the payload region per epoch parity and the count_done handshake already bounds the launch skew to one epoch, so only the single-buffered compact layout still needs the cross-rank launch barrier" (`launch_barrier = not direct_fixed_slot`, :155)
- S9 xiaobochen-amd/aiter@40ae5ad7f:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:412-424 thread 0 waits for the plan's flag, then `fence_system_acquire()` ("acquire once per block instead of per work item", :418-420), then `fx.barrier()`
- S10 xiaobochen-amd/sglang@b96766116:python/sglang/srt/layers/communicator.py:386-394 a sparse layer with an all-to-all backend runs its MoE `SCATTERED`; `_scatter_hidden_states_and_residual` (:1182-1195) reduce-scatters attention's output to `1/attn_tp_size` per GPU before it, and `_scattered_to_tp_attn_full` (:931, :944) gathers it back
- S11 xiaobochen-amd/sglang@b96766116:python/sglang/srt/layers/moe/mega_moe.py:137-151 the shared expert and the MegaMoE call run on two streams only under graph capture, with the shared expert unfused and `moe.alt_stream` set; `python/sglang/srt/models/deepseek_v2.py:2638-2646` creates `alt_stream` on ROCm only with `SGLANG_ROCM_USE_MULTI_STREAM`
- S12 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:106 `launch_grid_x = planner_blocks + dispatch_blocks + grid_x`
- S13 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:425 "Control CTAs join the work pool after dispatch."
- S14 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py:1100 `dispatch()` resolves its block count and warps per block from the table (`_resolve_launch_geometry("dispatch", …)`)
- S15 xiaobochen-amd/sglang@b96766116:python/sglang/srt/models/deepseek_v2.py:737-757 under the megamoe all-to-all backend the shared expert is built with `tp_size=1`
- S16 xiaobochen-amd/aiter@70f50bb10:aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py:90-100 `nearest_token_bucket` picks the nearer configuration bucket
- S17 xiaobochen-amd/aiter@a40fd4980:aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py:31 `FIXED_SLOT_MAX_RESERVED_BYTES = 512 << 20`, used by `use_fixed_slot_dispatch()` (:110-124)
- S18 aklr/ops/mega_moe/impls/flydsl/mega_moe_ep_size8_gate_up_interleaved_model_dim7168_n_experts48_topk6_inter_dim3072/1f7d5b1b02359d25.json#mega_moe_large_m.py:344 the owner workgroup waits for `group_done` to reach the producer count (raised at :572), the rest wait at its gate (:1511), all in the one launch of :1901; `86ac3e8237e1bc1f.json` and `eafbde5d34ed4034.json` the same at `mega_moe_small_m.py`:1041, one launch (:1367)
