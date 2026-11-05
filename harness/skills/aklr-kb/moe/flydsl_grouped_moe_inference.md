---
title: "FlyDSL grouped GEMM — inference MoE expert GEMMs: sorted-token mapping, persistence, cache policy, stage 2"
kind: topic
lever: flydsl_grouped_moe_inference
ops: [grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [bf16, fp8, int8, int4, mxfp4]
regimes: [moe-inference, decode, prefill, skew]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_sorting_kernel.py:3-21
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_sorting_kernel.py:43
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:62-72
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:394-423
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:518-535
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:384
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:434-443
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:595-619
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/moe_reduce.py:164-203
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/__init__.py:4-17
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:59-67
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:120-122
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:748-774
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:105
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:230-289
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:315-319
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:21-48
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:110
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:125-138
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:202-209
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:34-41
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py:502-505
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:495-505
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:517
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:630-641
---

# FlyDSL MoE expert GEMMs for inference

> The one card on inference MoE expert GEMMs: tokens are sorted by expert into blocks of tile-M rows, and each
> workgroup runs one (M-block, N-block) of one expert — stage 1 is gate + up with the activation, stage 2 the down
> projection with the routing weights. FlyDSL facts are for v0.3.2 (`kernels/moe/`); this repository does not
> contain that tree, and the copies the baseline aiter ships differ in places (§1). "BM" in FlyDSL's code is tile-M.
> Numbers marked `*(field, E<n>)*` are measured values from MoE kernels of this family at decode batch sizes (a4w4,
> and a8w4 for the comm-fused stage 2 and E24–E29's 112-of-896-expert MoE), except where a sentence names another kernel; their conditions are listed
> under Evidence at the end. Use them for direction and order of magnitude. aiter's FlyDSL
> MoE families, their entry points and knobs, and AKLR's own MoE kernels are in
> `../flydsl/kernels/flydsl_kernel_library.md`; a worked AKLR case is `../moe/flydsl_moe_decode.md`. Training
> grouped GEMMs (variable-K wgrad, group offsets) are the rest of this package (`INDEX.md`).

## Route here when
- You are writing, calling or tuning a FlyDSL MoE stage 1 or stage 2, or the token sort in front of them.
- You are choosing tile-M, tile-N, grid order, the weights' cache policy, atomics or a reduce, or a persistent
  stage 2.
- A MoE GEMM is fast at one batch size and slow at another.

## 1. What FlyDSL v0.3.2 ships

| Path | Operands | Stage 2 | Grid |
|---|---|---|---|
| `moe_gemm_2stage/` | fp8 E4M3, int8, int8smooth, int4 W (W4A8) | atomics, or partials + reduce | 2D: x = N, y = M-block |
| `mxfp_moe/` | a4w4, a8w4 (fp8 A × MXFP4 W), gfx950 | atomic, non-atomic, CShuffle, MXFP4-/fp8-out | 1D, N fastest |
| `moe_2stage_a16wmix/` | bf16 A × MXFP4, int4 or bf16 W | atomic scatter, optional persistent | 1D, N fastest |
| `moe_sorting_kernel.py` | token sort by expert | — | — |

`moe_gemm_2stage` has f16/bf16/f32 outputs. `mxfp_moe` stage 1 fuses gate+up, SiLU
and an fp4 re-quantization of the intermediate; its MXFP4-out stage 2 is persistent. `moe_2stage_a16wmix` issues
one K=32 MFMA (`mfma_f32_16x16x32_bf16`) per K step (`moe_2stage_a16wmix/utils.py:26-38`).

`mxfp_moe` uses its own sorting contract (`cumsum` plus `m_indices`, `__init__.py:4-17`). CShuffle is the
epilogue that stages C through LDS to reshuffle it into wide, coalesced stores. FlyDSL also ships a fused
dispatch + GEMMs + combine kernel (`kernels/mega_moe/`); it is not covered here.

**In the baseline aiter** (`aiter/ops/flydsl/kernels/`): the FlyDSL code that aiter's `fused_moe`, AKLR's
baseline, may dispatch to by tuned row (other rows run CK or HIP kernels):

| FlyDSL v0.3.2 | baseline aiter | How they relate |
|---|---|---|
| `mxfp_moe/` | `mxfp4_gemm1.py` (`compile_gemm1_a4w4_port` :775), `mxfp4_gemm2.py` (`compile_gemm2_a4w4_port` :86); hosts `aiter/ops/flydsl/mxfp4_gemm{1,2}_kernels.py` | one pipeline, "ported back from aiter" into FlyDSL (`mxfp_moe/__init__.py:6`); the copies have diverged since |
| `moe_2stage_a16wmix/` | `moe_2stage_a16wmix/` (`compile_gemm1_a16w4_port` `gemm1.py:375`, `compile_gemm2_a16w4_port` `gemm2.py:304`) | the same design (persistent above `4 · NUM_CU`, `b_cache_mod=2`); line numbers differ |
| `moe_gemm_2stage/` | no copy; only `moe_reduce.py` corresponds (`compile_moe_reduction` :206) | aiter's `flydsl_moe1_*` and non-layout `flydsl_moe2_*` rows run the parametric `mixed_moe_gemm_2stage_common.py` instead, which v0.3.2 no longer ships (`mxfp_moe/__init__.py:16`) |
| `moe_sorting_kernel.py` | `moe_sorting_kernel.py` (`moe_sorting_flydsl` :1934), `UNIT_SIZE = 32` | the same algorithm; aiter sorts with Opus unless `AITER_USE_FLYDSL_MOE_SORTING=1` (`aiter/fused_moe.py:64-66`), and its MXFP4 path with the HIP `aiter.mxfp4_moe_sort` (`aiter/fused_moe.py:161`) |
| — | v2 stage 2: `mxmoe_dispatcher.py` (`mxfp4_moe_gemm2` :616), `mxmoe_gemm_v2.py` (:221) | aiter only |

Most FlyDSL rows in aiter's tuned fused-MoE tables are `flydsl_moe1_*` and `flydsl_moe2_*` (the mixed builders, and
the v2 stage 2 for `flydsl_moe2_layout_*`); `flydsl_mxmoe_*` rows are a small minority. Where aiter's
`mxfp4_gemm{1,2}` differ from what this card cites: stage 2 is persistent for the plain non-atomic and the MXFP4-out
epilogues, not CShuffle (`_persistent`, `mxfp4_gemm2.py:113`); the non-persistent atomic path keeps the hardware XCD
order; the host takes `use_nt` and `xcd_swizzle` from the tuned row's kernel name (`aiter/fused_moe.py:1646`,
`:1833`) instead of the automatic switch of §4; and the hosts build a4w4 only, with no fp8-out epilogue
(`_SUPPORTED` in both host files).

## 2. How the work is mapped

- **The sort is the contract.** FlyDSL's sorting kernel is a counting sort in LDS (histogram, prefix sum,
  scatter) with three paths by token count: one kernel for small batches, two kernels (two phases fused) up to
  2048 tokens, four beyond. Token ids are packed as `(slot << 24) | token`; padding is `(topk << 24) | M`. The
  unit is the GEMM's tile-M (`UNIT_SIZE = 32`, "block_size in CK") (`moe_sorting_kernel.py:3-21`, `:43`). It
  produces sorted token ids, one expert id per M-block, and the number of valid ids.
- **One expert per M-block, read once and wave-uniform.** `moe_gemm_2stage` reads `expert_ids[e_idx]` for the
  block's BM sorted rows (`gemm1.py:399-403`), so the sort unit must equal its tile-M. `mxfp_moe` broadcasts
  the expert id with `readfirstlane` (`gemm1.py:120-122`).
- **Over-launch, exit on the device.** Grids come from an upper bound on M-blocks: in `mxfp_moe`,
  `(tokens·topk + active·(BM − 1) + BM − 1) // BM` with `active = min(tokens·topk, experts)` (all experts at
  BM 128), since each active expert pads at most BM − 1 rows (`gemm1.py:59-67`). Surplus workgroups exit when
  `e_idx · BM` reaches the valid count (`moe_gemm_2stage/gemm2.py:384`, `gemm1.py:394`), or when `bx` reaches
  `bound = cumsum[0] / BM · n_blocks` computed on the device (`mxfp_moe/gemm1.py:748-750`). No host read of the
  counts.
- **Under expert parallelism, sort and size by the local experts.** aiter pads the sort buffers, and so stage 1's
  grid, for every global expert: it passes `expert_mask.numel()` (`aiter/fused_moe.py:767-769`, `:983`, padding
  `:221`; grid `aiter/ops/flydsl/moe_kernels.py:1625`). With 112 of 896 experts local, 9% of 45,696 stage-1
  workgroups had work at 112 tokens, and a local sort cut the sort 24.5 → 9.5 µs and stage 1 174.4 → 160.6 µs
  *(field, E30)*; at 893 / 2732 tokens local sizing cut a call 2.0–3.7% and a local-only sort 3.0–3.4% more *(field, E33)*.
  One way to sort locally: a global → local table in LDS, −1 dropping a route to another rank's expert; at 8–16 tokens
  it cut 11–16 µs per call *(field, E25)*.
- **Under expert parallelism, cap that grid.** The bound counts every route and every global expert's padding, so
  most rows exit. *(field, E37)* On an fp8 × MXFP4 prefill stage 1, 1.25 × the expected local M-blocks + one per local
  expert, with an 8-row tail launch striding over the rest, cut 4.3–4.9%; the same cap as a stride loop in one
  kernel (202 → 264 VGPRs) ran 17–20% slower.
- **Sorted ids staged in LDS, used twice.** `moe_gemm_2stage` loads the block's sorted ids into LDS once and
  reads them for the A gather and again for the output scatter. The gather and scatter layouts read up to
  `256 / (tile_k / 16)` rows, more than BM at decode, so every readable slot is first seeded with the
  out-of-range id M — the buffer bound drops those accesses — before the real ids are written; an unseeded slot
  decodes to token 0 and piles garbage onto it (`gemm1.py:412-423`).
- **Exit on the raw id, then remap.** `mxfp_moe` tests `bx < bound` first; only then does it map `bx` through a
  bijection over `[0, bound)` that gives each of the 8 XCDs a contiguous slice of the tile order, followed by an
  M-group swizzle of `xcd_swizzle` blocks. Stage 1 applies this only when `xcd_swizzle > 0` (default 0,
  `gemm1.py:752-774`, `host.py:110`); stage 2 always applies the XCD part (`gemm2.py:230-254`, `:273-289`), aiter's
  copy only on its persistent path. Remapping before the exit would put all dead ids on one XCD
  (`flydsl_grouped_group_mapping.md` §4). The contiguous slice is not measured here; *(field, E1, E2)* on a dense
  read-once stream, one contiguous slice per XCD lost 2–10% where runs of 8 tiles gained
  (`../gemm/small_m.md` §6): race the remap against the hardware order.
- **aiter's mixed builders remap tiles across XCDs before the validity test, and not one-to-one.** The remap runs
  over the padded bound, not `num_valid_ids` (`mixed_moe_gemm_2stage_common.py:540-574`, stage 2 :3741), the test
  `bx_m < num_valid` comes after it (:727, :3966), and with `num_workgroups // 8` per XCD the last `grid % 8` tiles
  get no workgroup. *(field, E27)* Stage-1 `xcd_swizzle` 2–8 ran 1.3–3.4× slower; *(field, E22)* a bijection over the
  valid count instead took 6–12% off stage 1 at 9–16k tokens.
- **Persistent stage 2, capped at the CU count.** `mxfp_moe` (MXFP4-out epilogue only, `gemm2.py:105`; aiter's copy
  also for the plain non-atomic one) and `moe_2stage_a16wmix` (optional) launch `min(work, NUM_CU)` workgroups when
  the padded upper-bound work exceeds `4 · NUM_CU`, and stride through `[0, bound)` by the grid
  (`mxfp_moe/gemm2.py:315-319`, `a16wmix/gemm2.py:495-505`); with many experts the padded grid would otherwise
  over-launch mostly empty workgroups (`a16wmix/gemm2.py:34-36`). A barrier at the top of the loop keeps the previous
  tile's epilogue LDS apart from the next tile's A DMA (`a16wmix/gemm2.py:630-641`). *(field, E3)* A persistent loop
  with no barrier between tiles, C staged per wave, beat one tile per workgroup by 3–5% on a dense GEMM
  (`../gemm/grid_pricing.md` §10): race that form too.
- **The non-persistent atomic stage 2 is HBM-bound**: each tile reads a full fp4 weight column block with about 4%
  L2 reuse; FlyDSL's `mxfp_moe` applies the XCD remap above to it to balance HBM channels (`gemm2.py:262-289`).
- Per-expert 64-bit bases: `flydsl_grouped_group_mapping.md` §7.
- **A guard inside a helper needs `@flyc.jit`.** `a16wmix`'s atomic epilogue helper is a `@flyc.jit` function so
  that `if token_id < M` is rewritten into a device branch. As a plain Python function the `if` runs at trace
  time: on a `Numeric` value it raises, but on a raw `ArithValue` (e.g. an `llvm.load` result) it is always true,
  so the guard disappears silently, the atomic scatter fires on padded rows, and stage 2 ran ~13× slower at a
  very large expert count (`gemm2.py:39-41`; `Numeric.__bool__`, `python/flydsl/expr/numeric.py:502-505`).

## 3. Grid shape at decode sizes

- **At decode M the N blocks are the parallelism.** With tile-M at the sort unit, a decode batch fills only a few
  M-blocks. *(field, E4)* On an a4w4 atomic stage 2 with tile-M 32, narrowing tile-N from 256 to 128 to 64 helped at
  each step (64 against 128: +0.84%, +0.88%, +0.96% in three pairs); 32, buildable only with an in-block K
  split, was slower than 64. Race tile-N downward until the per-block fixed cost wins, and count the workgroups
  against the 256-CU dispatch round as you go (`../gemm/small_m.md` §3, §5).
- **Wider channel blocks pay until the grid runs out.** FlyDSL's fp8/int8 stage 1 uses a 128-channel block (half
  the blocks and waves, less barrier and LDS-sync overhead) but caps it at 128 so tile-N 256 stays two blocks per
  row, because one 256-channel block made a 256-wide intermediate ~40% slower (grid too small, register pressure)
  (`moe_gemm_2stage/gemm1.py:62-72`).
- **Split K inside a block only when K is long.** *(field, E5)* Splitting K across the waves of a stage-2 block,
  where K is the intermediate size, was neutral to negative: the split landed on launch and epilogue fixed costs.
- **Keep the grid N-fastest.** *(field, E6)* Swapping to M-fastest cost 2.4% on stage 1 and 1.5% on stage 2 (whose
  relative L2 error also rose from 0.0003 to 0.0017). N-fastest keeps the N-blocks that share one A tile running
  together, so the A tile stays in L2 and MALL (the last-level Infinity Cache). Both FlyDSL grids are N-fastest
  (`moe_gemm_2stage/gemm1.py:518-535`, `mxfp_moe/gemm1.py:120-121`).
- **Keep an M-block's N tiles on one XCD, dealing M-blocks round-robin over XCDs.** *(field, E21)* On a fused-MoE stage 1 this cut
  the call 511 → 475 µs against the hardware order; aiter's contiguous-chunk + `GROUP_M` remap (`xcd_swizzle` 4–16) took 574–582 µs.
- *(field, E7)* Halving stage 2's A-tile LDS (packing fp4 at its real width) was neutral: that stage 2 was not limited
  by LDS occupancy.
- **Send padding rows' A loads out of range where each reads a distinct row.** Out-of-range buffer reads return
  zero (`../hardware/gfx950/memory_and_sync.md` §2.3; the bounded descriptor, `../flydsl/language/limits.md`). A stage 2 that reads A by
  sorted row gained: +1.0%, 0.9–1.0 µs at 8 and 12 tokens *(field, E26)*; 58.7 → 57.0 µs at 40 tokens *(field, E31)*.
  A stage 1 that maps padding to token 0 gained nothing: every padding row already hits one cached row.
- **aiter's v2 stage 2: stage C in fp32.** With `g2_kstatic` the bf16 default leaves 2 A slots and a barrier per K tile
  (`mxmoe_dispatcher.py:187–201`, `mxmoe_gemm_v2.py:734`); *(field, E29)* `g2_bf16_lds=False` won every pair, +0.37%.

## 4. Cache policy on the weights

- **FlyDSL's switch.** `mxfp_moe` stage 1 (its BM-32 variant) streams B non-temporally at small M, where no tile
  reuses another's weights, and switches to cached loads once total M-blocks ≥ experts (on average at least one
  padded block per expert); consecutive M-blocks of one expert reuse weight columns only past tile-M rows per
  expert (§8). Its note: L2 hit ~34% streaming against ~59% cached, ~22% faster at an effective M of 8192
  (`host.py:125-138`). The `a16wmix` stage 2 defaults its B loads to the same non-temporal cache modifier (`b_cache_mod=2`,
  `gemm2.py:517`). aiter's host has no such switch: `use_nt` comes from the tuned row (§1). On dense small-M GEMMs
  the B policy also coupled with the K tile (`../gemm/small_m.md` §6).
- **Non-temporal is a property of each load, not of the operand.** *(field, E8)* Choosing per M-block — cached when a
  neighbour in sorted order has the same expert (`sorted_expert_ids[blk ± 1] == expert`), non-temporal otherwise —
  gained 1.82% over all-non-temporal on an a4w4 decode score. Conditions:
  - the reuse exists only if same-expert neighbours share an XCD: with the hardware XCD order (no remap),
    adjacent M-blocks of an N-fastest grid are `grid.x` workgroups apart, so `grid.x` must be a multiple of 8;
    a contiguous XCD remap such as `mxfp_moe` stage 2's keeps them on one XCD except where a slice ends;
  - one neighbour on each side is enough: in a run of two or more same-expert blocks, every block has one;
  - compile the branch only when the host knows a repeat is possible (tokens > block M): the if/else duplicates
    the GEMM body, and *(field, E9)* small batches that could not repeat a block paid up to 2.1% for it.
- **A skip-the-block bound underestimates reuse.** *(field, E10)* A probe that skipped the repeated blocks bounded the
  gain at 2.11% of the score and 1.0 µs at one batch size; at that batch size the cached version saved 1.82 µs,
  beyond the bound, because it also relieved L2 contention with the other streams.
- **Keep scale loads cached.** *(field, E11)* Marking the B-scale loads non-temporal on single-read blocks lost 1.7%:
  4-byte loads gain nothing from streaming and lose line merging. A and its scales stay cached as well — all the
  N-blocks of a row share the A tile.
- **Dead alternatives.** *(field, E12)* One workgroup walking two M-blocks of an expert (turning cross-workgroup reuse
  into same-workgroup reuse) halved the workgroup count and scored 8.5% lower than the all-non-temporal
  baseline; an XCD swizzle that grouped same-expert blocks scored 16.6% lower. Bypassing L1 (`sc0`/GLC) on
  the cached blocks scored the same as plain cached: the reuse is across CUs. *(field, E38)* On an fp8 × MXFP4 prefill
  stage 1, a software L2 prefetch of the weights (a one-dword touch load per lane, 2–3 K steps ahead) lost 7.5–9.4%.
- **Warming stage 2's weights from an earlier launch.** *(field, E28)* Spare stage-1 workgroups reading w2 slowed stage 1
  more than stage 2 sped up at every share of w2 tried (¼: +7.6 against −2.6 µs at 8 tokens).

## 5. Tile-M equals the sort unit

- Each sorted block holds one expert's rows. A tile-M smaller than the block splits it into several M-tiles, and
  every M-tile streams the expert's weight slab again: weight traffic ∝ M-tiles × N-tiles × (tile-N · K bytes).
  *(field, E13)* A comm-fused MoE stage 2 (a8w4, the down GEMM and its cross-rank collective in one kernel) running
  tile-M 16 against a 32-row sort unit read every weight byte twice; tile-M 32 cut its time by 16–34% at the larger
  batches.
- **Tile-M, the B cache policy and the K tile move together.** *(field, E14)* Non-temporal B lost on tile-M 16 (three
  of four batch sizes much slower) and paid on tile-M 32; a 256-deep K tile lost on its own and paid on tile-M 32
  with non-temporal B. Sweep them as one axis.
- A tile-M larger than the sort unit is not available when the unit must be a multiple of tile-M; to try one,
  change the sort unit with it.
- **Skip the padding rows a capped sort leaves.** *(field, E19)* On an fp8 × MXFP4 decode MoE whose sort capped
  each 32-row block at 16 routes, 16-row stage-2 tiles strided 32 rows cut stage 2 by 0.8–2.4% (unstrided:
  +68–73%), and skipping stage 1's MFMAs on the padding rows, occupancy re-raced, gained 1.6%.

## 6. Stage 2: atomics or split + reduce

- **`moe_gemm_2stage`** (`gemm2.py:595-619`): `accumulate=True` adds into the output with atomics — f16 half2
  atomics are fast but can overflow to ±inf, bf16 uses buffer atomics, f32 scalar atomics are slower and
  avoid the overflow. `accumulate=False` writes per-(token, slot) partials (f16/bf16 only) for
  `compile_moe_reduction`, launched over `(tokens, model_dim blocks)` (`moe_reduce.py:164-203`).
- **Count the local routes per token before choosing.** With all 16 of a token's routes local (TP, 896 experts, 9–16k
  tokens), the atomic stage 2 made the whole MoE 31–42% slower than split + reduce *(field, E23)*. With about 2 local
  routes per token (EP8, 112 of 896 experts local, 893 and 2732 tokens), fp8 route rows plus a reduce kernel took
  123 / 153 µs where the atomic stage 2 took 110 / 125–127 *(field, E36)*. That contention per output element decides
  it is estimated, not measured.
- **Where the compare block bounds output SQNR, the route rows can be MXFP4.** Stage 2 packs each row to e2m1 with
  an E8M0 scale per 32 (`rocdl.cvt_scalef32_pk_fp4_bf16`), the reduce unpacks it (`cvt_scalef32_pk_f32_fp4`). With all
  16 routes per token local, rows shrank 3712 → 1920 B, the reduce 190 → 104 µs at 16384 tokens, and the call gained
  10.35%; the output read 17.97 dB SQNR against the fp8-row kernel's, against a 13 dB bound *(field, E39)*.
- **Keep the reduce its own launch on the same stream.** Folding it into stage 2, the last workgroup of a token to
  arrive on an atomic counter summing its rows, took stage 2 from ~305 to 526–547 µs against a 62 µs reduce; the
  reduce on a second stream, overlapping stage 2, made the call 0.6–1.8% slower *(field, E40)*.
- **Keep packed atomics coalesced.** FlyDSL's stage 2 keeps the CShuffle layout channel-major (32 lanes walk
  contiguous channels of one row) and `e_vec = 2` on the atomic path, so the 32 lanes cover 64 contiguous
  channels; its note puts the wider form at 4.25× the atomic traffic and 4.8× slower (`gemm2.py:434-443`).
- **Atomics are not bit-reproducible**; gate on a tolerance against an independent reference
  (`../gemm/correctness.md` §8). AKLR's `fused_moe` contracts are `determinism: bounded`, gated on
  output SQNR ≥ 13 dB with `max_violation_fraction` 1e-3 (`aklr/ops/fused_moe/contract.py`), so atomics are allowed
  there, unlike AKLR's GEMM contracts, all bitwise but `gemm_a8w8_blockwise_scaled`; a port's equivalence is judged
  by SQNR (`../moe/flydsl_moe_decode.md` §3). Four of AKLR's five MoE kernels still sum the routes in a fixed order
  without atomics (`../flydsl/kernels/flydsl_kernel_library.md` § AKLR's own FlyDSL MoE kernels).
- **`mxfp_moe` stage 2 variants** (`host.py:21-48`): atomic at BM 16/32/64 with or without non-temporal B,
  non-atomic, CShuffle, MXFP4-out and fp8-out at BM 128 (CShuffle also at 32/64). FlyDSL's host defaults
  `xcd_swizzle = 3` for the non-atomic and fp8-out epilogues and routes BM-128 non-atomic to CShuffle
  (`host.py:202-209`); aiter's takes these from the tuned row (§1).

## 7. The small kernels around the GEMMs

- **Price sorting and quantization kernels by launching them again** (`flydsl_grouped_measurement.md` §6).
  *(field, E15)* One sorting phase cost 2.3–2.4 µs per extra launch under graph replay; shortening the loops inside it
  moved the score by 0.03%, inside the noise; merging two sorting phases into one kernel gained 1.08%. Removing a
  whole kernel pays most; trimming inside one pays only above the ~4 µs node floor, so price each phase by deleting
  it. One wave per expert instead of one workgroup each (936 → 152 workgroups) took a sort kernel from 7.3–7.8 to
  6.5 µs, +0.54% of the MoE call *(field, E20)*; deleting phases took another from 7.65 to 5.32 µs against a 4.4 µs
  floor *(field, E32)*. FlyDSL v0.3.2's mid-size sorting path runs two kernels (`moe_sorting_kernel.py:12-14`).
  *(field, E16, E17)* Under AKLR's timer a graph node costs ~4 µs (measured on an empty kernel and on add + RMSNorm;
  that is the timer's floor for any dispatch, since its bench runner forces `GPU_FORCE_QUEUE_PROFILING`,
  `aklr/core/validation/stages/_bench/_execution/_runner.py:475`), and folding the sort into stage 1 removed one (`../moe/flydsl_moe_decode.md` §1–§2).
- **Or fold the per-token kernels into the sort launch.** *(field, E24)* Token quantization and the sorted-scale scatter
  as extra sort blocks, one per token, removed two nodes: 6.9–8.0 µs per call at 8–32 tokens.
- **Merging kernels turns an inter-kernel order into an intra-kernel race.** One phase seeded a table with −1
  and the next overwrote the local experts' entries; the kernel boundary ordered the two. Once merged,
  `gpu.barrier()` is not guaranteed to order the global stores (it synchronizes the waves of one workgroup).
  Make the two writes disjoint (seed only the entries the scatter never writes). Replacing a kernel boundary with
  a flag needs the polling rules of `../gemm/correctness.md` §4.

## 8. Measuring at decode sizes

- Decode reads cold weights, runs under graph replay and pays a per-node floor: rotate weight copies past the
  MALL, time a replayed graph, and price through the deployed entry (`../gemm/small_m.md` §1). AKLR's
  bench already replays a graph with a cache flush per call, and times aiter's MoE through `fused_moe_baseline`
  (`../flydsl/kernels/flydsl_kernel_library.md` § Other FlyDSL families in the baseline aiter).
- *(field, E18)* A MoE decode scorer's in-batch noise was ~0.2%, but identical code drifted 0.3–0.4% between batches.
  Decide on interleaved pairs within one batch; use cross-batch values for direction only. *(field, E35)* Unchanged
  code also moved 9% within one session, and a null change was recorded at +6%: even direction needs a fresh pair.
- Cover several batch sizes, from a few tokens (single M-block per expert, non-temporal B) to above
  `experts · tile-M / top-k` tokens (over tile-M rows per expert: repeated blocks, cached B); the policy flips
  between them. Total M-blocks ≥ experts already holds at one block per expert, where *(field, E34)* cached B
  cost a stage 2 ~15%. At three to five blocks per expert non-temporal weights lost: the call took 26% longer with
  W1 non-temporal in stage 1 and 6–9% longer with W2 non-temporal in stage 2 *(field, E41)*.

## Verify
- [ ] The sort unit equals tile-M (or is a multiple of it), and the tile-M sweep includes the unit itself.
- [ ] The grid is N-fastest, exits on the raw id before any XCD rotation, and bounds its work on the device.
- [ ] The weights' cache policy was raced per batch size, with the repeated-block condition and its host gate.
- [ ] Atomic stage 2 is gated on a tolerance, and the atomic layout keeps 32 lanes on contiguous channels.
- [ ] Small kernels were priced by extra launches, and merged kernels write disjoint regions.

## Related
`../moe/flydsl_moe_decode.md` · `../flydsl/kernels/flydsl_kernel_library.md` ·
`../gemm/small_m.md` · `../gemm/grid_pricing.md` · `flydsl_grouped_scheduling.md` ·
`flydsl_grouped_measurement.md` · `INDEX.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. `aiter fork rev ?`: the GLM-5.2 campaigns ran on the aiter fork AKLR grew out of, at a commit their
records do not give. The Kimi-K3 forge runs installed FlyDSL 0.3.2.

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=3–4, AKLR graph timer, arms interleaved in one process] one contiguous band of tiles per XCD: 16.45 against 16.06 µs at m=14 and 16.92 against 15.40 µs at m=12 (dense bf16 GEMM n=163840, k=256, Kimi-K3 forge run, iter 6; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_006.md:25)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=4–6, AKLR graph timer, arms interleaved in one process] runs of 8 tiles per XCD against plain order at m=5–10: 14.55 against 14.92 µs at m=5 … 14.51 against 14.94 at m=10 (1.4–2.9%) (dense bf16 GEMM n=163840, k=256, Kimi-K3 forge run, iter 11; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n163840_k256/forge_experiments/lessons/iter_011.md:54)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, own CUDA-graph harness, all arms in one process, 10 launches per graph, median of 10 replays] the same 256×256 body as a persistent 256-workgroup grid against one tile per workgroup: 0.1143 against 0.1200, 0.3019 against 0.3183, 0.9738 against 1.0022 ms at m = 35107 / 104093 / 342073 (−2.8% to −5.2%) (dense bf16 GEMM n=3072, k=512, Kimi-K3 forge run, iter 2; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n3072_k512/forge_experiments/lessons/iter_002.md:22)
- E4 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=3 pairs, same-batch interleaved A/B, MoE decode score over 8 / 16 / 64 / 128 tokens under graph replay] stage-2 tile-N 64 against 128: 1.09594 / 1.09836 / 1.09857 against 1.08680 / 1.08881 / 1.08810, +0.84 / +0.88 / +0.96% per pair; tile-N 256 → 128 → 64 read 1.05688 → 1.0915 → 1.1020; tile-N 32 with a two-way in-block K split 1.0883 against 1.0997 at 64 (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:155)
- E5 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, one batch, MoE decode score] splitting K across the waves of a stage-2 block (`k_wave` 2 / 4): 1.0992 / 1.0790 against 1.0997 unsplit at tile-N 64, 1.0890 against 1.0925 at tile-N 128; stage 2's K (inter_dim) is 2048 (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:146)
- E6 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, same-batch interleaved A/B, MoE decode score] the grid swapped to M-fastest: −2.4% on stage 1, −1.5% on stage 2, whose rel_l2 rose from 0.0003 to 0.0017 (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:175)
- E7 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=1 pair per tile-N, same batch, MoE decode score] stage 2's fp4 A tile packed at its real width (16 → 8 KB of LDS per workgroup): 1.10187 against 1.10077 at tile-N 64, 1.09246 against 1.09149 at tile-N 128 (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:139)
- E8 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=3 pairs, same-batch interleaved A/B, MoE decode score] B cached on M-blocks with a same-expert neighbour and non-temporal elsewhere, against all non-temporal: 1.11075 → 1.13092 (+1.82%), every pair positive (A 1.11089 / 1.10991 / 1.11146, B 1.12989 / 1.13224 / 1.13064) (GLM-5.2 TP4/EP4 a4w4 MoE decode, stage 1 and stage 2; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:198)
- E9 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, MoE decode score, before and after the host gate] compiling the branch for every batch size cost +1.76 µs (+2.1%) at 16 tokens and +0.35 µs at 8; with the host gate both returned to the baseline (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:196)
- E10 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, MoE decode score] a probe that skipped the repeated blocks scored 1.13649, a bound of +2.11% (64 tokens −1.0 µs, 128 tokens −9.9 µs); the cached version moved 64 tokens 176.13 → 174.31 µs and 128 tokens 211.37 → 204.71 µs (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:192)
- E11 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, MoE decode score] B-scale loads non-temporal on single-read blocks and cached on repeated ones: 1.10968, −1.7% (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/rounds/round-13/agent.txt:26)
- E12 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, MoE decode score against the all-non-temporal 1.11075] one workgroup walking two M-blocks (`persist_m=2`) 1.01627 (−8.5%); an XCD swizzle grouping same-expert blocks (`xcd_swizzle=1`) 0.92692 (−16.6%); sc0/GLC on the cached blocks 1.13120 against 1.13137 for plain cached (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:200)
- E13 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, 4-rank graph ruler with the real runner] at 8 / 16 / 64 / 128 tokens, tile-M 16 ran 97.25 / 94.87 / 122.5 / 204.2 µs and tile-M 32 92.43 / 92.02 / 102.3 / 135.4 µs; tile-M 16 splits each 32-row sort block in two and streams the w2 slab twice (805 against 402 MB) (comm-fused MoE stage-2 megakernel, a8w4, GLM-5.2 TP4/EP4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_144151/goal.md:135)
- E14 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, 4-rank graph ruler] non-temporal B on tile-M 16: 127.54 / 119.35 / 147.56 / 190.64 µs against 97.25 / 94.87 / 122.5 / 204.2; tile_k 256 alone 104.63 against 102.53 µs at 64 tokens (round 14; that base matches tile-M 32's 102.3 µs at 64 tokens rather than tile-M 16's 122.5, while round 15 reads the loss as a tile-M 16 result); on tile-M 32 with non-temporal B, tile_k 256 took 64 tokens 96.4 → 94.7 µs and 128 tokens 127.8 → 115.5 µs (round 15) (comm-fused MoE stage-2 megakernel, a8w4, GLM-5.2 TP4/EP4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_144151/goal.md:138)
- E15 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=3 pairs for the merge, same-batch interleaved, MoE decode score under graph replay] launching one sorting phase once more cost +2.29 µs at 64 tokens and +2.41 µs at 128; subtractive probes that shortened its loops read +0.03% each; merging the two phases into one kernel took the score 1.0982 → 1.1124 (+1.08%), every pair positive (GLM-5.2 TP4/EP4 a4w4 MoE decode sorting; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/rounds/round-11/agent.txt:7)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=3–9, AKLR bench_cudagraph (L2 flushed) in an in-process harness, median of 3–9 rounds] a fully empty kernel read 4.21 µs at m=8 (gemm_n2112_k7168 forge run, iter 12; local:/workspace/code/gpt_oss_docker/work/sikl/dashboards_k3/gemm_n2112_k7168/forge_experiments/lessons/iter_012.md:37)
- E17 [2026-10, gfx950, FlyDSL ?, N=?, AKLR validation (graph replay, cache flushed per call)] add + RMSNorm at hidden 6144 took 3.9–4.5 µs from 1 to 256 tokens, for both kernels compared (AKLR's ruler notes; 0cef4f2e3 harness/skills/aklr-read-validation/references/ruler.md:42)
- E18 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=? (two reads at each end of a round), MoE decode score] in-batch noise 0.2%; code-identical trees read 1.1019 / 1.1017 at a round's start and 1.0977 / 1.0980 at its end (−0.4%), and 1.09795 against 1.10117 in two batches of the next round (+0.29%) (GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/memory.md:161)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm for stage 2, 3 pairs for stage 1, Kineto per-kernel time with L2 flushed (stage 2); paired ablation of the graph-timed score (stage 1)] stage 2 BM 16, m_stride 32, BN 256: 21.79 / 29.63 / 33.65 / 49.26 / 61.22 against BM 32 22.18 / 30.25 / 34.47 / 50.42 / 61.69 µs at 8 / 12 / 16 / 24 / 32 tokens; BM 16 unstrided 37.20 and 106.85 µs at 8 and 32 tokens; stage 1 with the MFMAs of rows 16–31 skipped (122 VGPRs at 8/16/24/32 tokens, 108 at 12), its waves per EU re-raced: +1.57% (one-sided 95% bounds +1.55 … +1.59%, 5 of 5 token counts faster) (fp8-activation × MXFP4 SiTUv2 fused MoE, 896 experts, top-16, inter 384, model dim 3584, 8–32 tokens; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_s/workspace/forge_experiments/lessons/iter_007.md:12; :28; :31; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_s/workspace/forge_experiments/lessons/iter_009.md:14; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_s/workspace/forge_experiments/ablation.md:20)
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=1 (median of 30 calls per arm), per-kernel Kineto time, 1 GiB read flush before each call] a fused MoE sort + MXFP8 quant kernel, one 512-thread workgroup per expert against one wave per expert (8 per workgroup): 936 → 152 workgroups at 40 tokens, 7.26–7.78 → 6.52 µs; 1008 → 224 at 112 tokens, 8.78 → 7.88 µs; that round, which also moved the output zeroing to the kernel start, gained +0.54% (one-sided 95% bounds +0.53…+0.54%, 8 of 8 token counts up) on the whole call under the campaign driver's graph timer (896 experts, top-16, model dim 3584, inter 384; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_m/workspace/forge_experiments/lessons/iter_005.md:18; :23; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_m/workspace/forge_experiments/ablation.md:15)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=3 captured graphs per setting, flushed CUDA graphs, mean per-call total; outputs bit-identical] stage-1 tile order only: block L → M-block 8s + L % 8, N tile (L mod 8·grid_x) // 8 (trailing partial run unmapped) 475.0 µs; no remap 511.3; `xcd_swizzle` 4 / 8 / 16 574.2 / 579.9 / 582.3 (fused_moe situv2 fp8 activations × MXFP4 weights, 896 experts, top-16, model 3584, inter 384 gate/up interleaved, n_tokens 2732, stage 1 t64x128x256, 6 N tiles; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_l/workspace/forge_experiments/lessons/iter_010.md:30; the same order against no remap in the driver: local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_l/workspace/forge_experiments/lessons/iter_014.md:28)
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=1, torch-profiler stage-1 median over 10 calls; driver bench for the op; the run's ablation, paired] stage 1 765.6 → 673.9 µs at 9397 tokens and 1007.3 → 946.5 µs at 16384; op 1.229 → 1.150 / 1.835 → 1.790 ms; ablation +4.71% (bounds +4.48% … +4.94%); the bijection `x·(n//8) + min(x, n%8) + id//8` over `ceil(num_valid_ids / BM)` M-blocks, the tail keeping its id and exiting (fused MoE, fp8 activations × MXFP4 experts, 896 experts, top-k 16, inter 384, model dim 3584, stage 1 tile 64×128×256, `xcd_swizzle` 4; local:/shared_nfs/kyle/moe/kf/runs/tp8_cmp/workspace/forge_experiments/lessons/iter_001.md:17; ../ablation.md:13)
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=2 rounds, one driver bench (warmup 3, 20 iterations) per arm per round] the v2 stage 2 with its `atomic` epilogue against `reduce` + `moe_reduce`: 1.4022 / 2.3576 against 1.0717 / 1.6639 ms (round 3), 1.388 / 2.308 against 1.0556 / 1.6441 ms (round 5) at 9397 / 16384 tokens (same MoE as E19, stage 2 64×256×128; local:/shared_nfs/kyle/moe/kf/runs/tp8_cmp/workspace/forge_experiments/lessons/iter_003.md:13; lessons/iter_003.md:3; lessons/iter_005.md:12; lessons/iter_005.md:5)

E24–E29: a8w4 SiTUv2 fused MoE, 112 of 896 experts local, top-16, model dim 3584, inter 3072, at 8 / 12 / 16 / 24 / 32
tokens; the ruler is AKLR's graph timer (bench_cudagraph, L2 flushed per replay), median of the readings.

- E24 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer] token quant and sorted-scale scatter folded into the one-shot sort launch: 55.4 / 72.0 / 107.0 / 131.0 / 180.0 → 48.5 / 64.5 / 99.4 / 123.0 / 172.1 µs (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/lessons/iter_002.md:8; :26)
- E25 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer] sort over the 112 local experts against over 896 with the mask: 70.1 / 86.9 / 126.7 → 58.9 / 73.9 / 110.9 µs at 8 / 12 / 16 tokens, one-shot sort on both sides; 156.2 → 138.6 and 207.3 → 189.7 at 24 / 32, where it also replaced the two-kernel sort (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/lessons/iter_001.md:7; :13–17)
- E26 [2026-10, gfx950, FlyDSL 0.3.2, N=? pairs, AKLR graph timer, the kept version beside the one before it, remeasured] stage-2 refills of padding rows pointed past the A buffer (prologue preload unchanged): +1.02% [+0.92, +1.13], 5 of 5 cases up; 44.23 / 59.79 → 43.34 / 58.77 µs at 8 / 12 tokens, level at 16–32 (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/ablation.md:21; lessons/iter_013.md:3; :9)
- E27 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer] stage-1 `xcd_swizzle` {4, 2, 8, 4, 2}: 147.13 / 156.26 / 174.18 / 186.33 / 214.19 against 43.36 / 58.77 / 93.70 / 116.59 / 167.39 µs; that the live tiles land on one XCD is estimated from the source, not measured (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/lessons/iter_014.md:53; :61)
- E28 [2026-10, gfx950, FlyDSL 0.3.2, N=1, probe with random inputs, torch profiler, cache flushed per call] w2 DMA'd from 512 extra stage-1 workgroups: stage 1 / stage 2 at 8 / 16 / 32 tokens 36.9 / 19.0, 71.1 / 37.0, 114.7 / 60.0 µs without; with ¼ of w2 44.5 / 16.4, 82.6 / 30.1, 137.9 / 44.9; with all of it 61.4 / 12.7, 127.6 / 24.2, 203.9 / 41.6 (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/lessons/iter_014.md:21–29)
- E29 [2026-10, gfx950, FlyDSL 0.3.2, N=? pairs, AKLR graph timer, the kept version beside the one before it, remeasured] `g2_bf16_lds=False` at BM 32, BK 256, BN 128/256: +0.37% [+0.31, +0.44], 5 of 5 cases up; faster by 0.15–0.8 µs in every interleaved pair (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/ablation.md:17; lessons/iter_005.md:44–46)
- E30 [2026-10, gfx950, FlyDSL 0.3.2, N=1, Kineto kernel times over 20 calls, staging-only arms] sort sized by 896 global experts: padded rows 30,448, stage-1 grid 48 × 952 = 45,696 workgroups, 4,224 (9%) with a valid M-block; topk_ids remapped to the 112 local ids (remap outside the timed window): sort 24.48 → 9.47 µs, stage 1 174.38 → 160.64 µs, kernel sum 285.53 → 256.97 µs (−10.0%) at 112 tokens; −11.3% at 40; stage-1 grid alone cut to the valid blocks: 175.7 → 157.9 µs (fused MoE fp8 × MXFP4, SiTUv2, EP8: 112 of 896 experts, top-16, model 3584, inter 3072; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/analysis/84daab2233e5d344ebf8e0242e221f58ba95feb9/generation-001/cases/n_tokens_112-115301ba/analysis.md:12; :18; :21; :32)
- E31 [2026-10, gfx950, FlyDSL 0.3.2, N=3 pairs, Kineto median over 40 calls, L2 flushed per call] stage 2 (BM 32, BN 128, fp8 A read by sorted row, ~4.5% of 32-row tile rows real) with the padding rows' A DMA sent out of range: 57.0 against 58.7 µs at 40 tokens; 79.6 against 79.9 at 112 with BN 256. Stage 1 (padding gathers token 0) with padding rows pointed out of range, graph medians of 3: 151.9 against 152.8 µs at 112 tokens, equal at 40, 60, 96 (EP8 fused MoE as E19; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/lessons/iter_004.md:31; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/analysis/84daab2233e5d344ebf8e0242e221f58ba95feb9/generation-001/cases/n_tokens_40-52bd03b6/analysis.md:9; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/lessons/iter_005.md:20)
- E32 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AKLR-style graph median, L2 flushed per call] one-launch sort + token-quant kernel at 112 tokens: 7.65 µs; its column scan deleted 5.96; only loads, quant and zeroing kept (wrong outputs) 4.4; after replacing the scan with per-expert bitmasks (LDS atomic OR, popcount), one packed prefix sum, batched lookups and loads hoisted before stores: 5.32 µs (6.29 → 5.04 at 40 tokens); that round, with a 0.24–0.56 µs stage-1 change, paid +0.97% (one-sided 95% +0.90 … +1.03%, 8 of 8 token counts) (EP8 fused MoE as E19; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/lessons/iter_005.md:7; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_m/workspace/forge_experiments/ablation.md:17)
- E33 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, the campaign's graph timer (`driver.py --bench-mode --warmup 3 --iters 20`)] sort buffers and GEMM grids sized for the 112 local experts instead of 896: 0.3689 → 0.3616 ms at 893 tokens, 0.4600 → 0.4432 ms at 2732; then the sort run over the 112 local experts (a global → local table in LDS, non-local routes dropped) instead of the masked 896-expert sort: 0.3608 → 0.3484 ms, 0.4333 → 0.4203 ms (fused MoE SiTUv2, fp8 activations × MXFP4 weights, model dim 3584, inter 3072, top-16, 112 of 896 experts local; forge campaign ep8_bw_l; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_002.md:11; :15; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_003.md:8; :24)
- E34 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, Kineto per-kernel median after a read flush; and the campaign's graph timer, paired with the run just before] stage-2 B loads cached instead of non-temporal: 126.3 against 110.0–110.2 µs at 893 tokens (112 M-blocks of 32 rows), 144.2 against 125.3 µs at 2732 (115 M-blocks of 64 rows), over 112 local experts; per call 0.3459 / 0.4024 against 0.3289 / 0.3837 ms (fused MoE SiTUv2, fp8 activations × MXFP4 weights, model dim 3584, inter 3072, top-16, 112 of 896 experts local; forge campaign ep8_bw_l; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_008.md:80; :85; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_022.md:19; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_021.md:43)
- E35 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2 per reading, the campaign's graph timer] the same build read 0.3513 ms at 2732 tokens at a session's start and 0.3837 later in it, 0.3833 after a cache clear; in another session 2732 timed alone read 0.3832 ms and in the full suite about 0.353; a change kept at +6.04% over the recorded best re-measured at −0.02% (−0.08 … +0.04%) beside it (fused MoE SiTUv2, fp8 activations × MXFP4 weights, model dim 3584, inter 3072, top-16, 112 of 896 experts local; forge campaign ep8_bw_l; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_021.md:7; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_016.md:10; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/ablation.md:24)
- E36 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, Kineto per-kernel median after a read flush] stage 2 writing fp8 per-route rows plus a reduce kernel: 112.7 + 10.6 µs at 893 tokens, 129.2 + 24.0 µs at 2732; the atomic bf16 stage 2 110.0–110.2 and 125.3–126.5 µs in the same session (fused MoE SiTUv2, fp8 activations × MXFP4 weights, model dim 3584, inter 3072, top-16, 112 of 896 experts local; forge campaign ep8_bw_l; local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_l/workspace/forge_experiments/lessons/iter_008.md:90-92; :12; :85)
- E37 [2026-10, gfx950, FlyDSL 0.3.2, N=1–3 per arm, forge driver --bench-mode (the AKLR-style graph timer), whole call, n_tokens 9397 / 16384] stage-1 grid capped at 480 / 752 M-rows (from 3245 / 4992) with an 8-row tail launch: 0.863 / 1.345 ms (re-run 0.862 / 1.352) against 0.902 / 1.414 uncapped; the same cap as `scf.for` over the blocks inside the kernel: 1.054 / 1.693 ms, 202 → 264 VGPRs, 0 spills; folding the tail into the capped kernel later lost again (stage 1 679.6 / 987.8 against 584.3 / 899.6 µs; 292 VGPRs, 0.856 / 1.328 against 0.750 / 1.166 ms) (fused MoE SiTUv2, fp8 × MXFP4, 112 of 896 experts local, top-16, model_dim 3584, inter 3072, BM 64; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_001.md:9; :10; :13; iter_005.md:31; iter_008.md:28; grid sizes from local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/kernel.py:269; :333)
- E38 [2026-10, gfx950, FlyDSL 0.3.2, N=1–2 per arm, forge driver --bench-mode, whole call, n_tokens 9397 / 16384] w1 touched two steps ahead: 0.817 / 1.253 against 0.7497 / 1.1655 ms; one LDS-DMA dword per lane three steps ahead into a dummy slot: 0.8205 / 1.2639 (with sc1 0.8213 / 1.2631) against 0.7500 / 1.1664; one line per wave (diagnostic) 0.7534 / 1.1726; a B-hot oracle bounded the possible gain at 0.6708 / 1.0933 (same fused MoE as E19, stage 1 at 64×256, two waves per SIMD; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_006.md:20; iter_008.md:5; :12; :18)
- E39 [2026-10, gfx950, FlyDSL 0.3.2, N=? pairs, AKLR graph timer (forge ablation, the kept version beside the one before it, as recorded); per-kernel torch.profiler medians, one session] stage-2 route rows packed to e2m1 with an E8M0 scale per 32 (`cvt_scalef32_pk_fp4_bf16`) and unpacked in the reduce, in place of fp8 rows: +10.35% (one-sided 95% +10.30 … +10.40%, 2 of 2 cases); rows 3712 → 1920 B; reduce 107.7 / 189.5 → 57.7 / 104.1 µs and stage 2 312.4 / 497.7 → 277.9 / 447.1 µs at 9397 / 16384 tokens; output 17.97 dB SQNR against the fp8-row kernel's, AKLR validate passed; a torch emulation put the noise e2m1 rows add at 18.15 dB with each scale's exponent that of the block amax minus 2 (15.85 dB with minus 1), against the definition's 13 dB bound (fused MoE SiTUv2, fp8 activations × MXFP4 weights, TP: 896 experts, top-16, model dim 3584, inter 384; commit 20fe49315948; local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/workspace/forge_experiments/ablation.md:13; local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/workspace/forge_experiments/lessons/iter_007.md:3; :49–58)
- E40 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, torch.profiler kernel times (fold, 9397 tokens only) and per-call sums (streams)] fold: incumbent stage 2 300–309 µs plus a 61.5 µs reduce; stage 2 with a per-(token, N-tile) agent-scope atomic counter whose last arriver sums the 16 fp4 rows in slot order 526–547 µs, counters and sc1 stores without the sum 364, counters alone 345; two streams (stage 2 and the reduce, overlap seen in the trace): 818.0 → 833.1 µs at 9397 and 1267.5 → 1275.1–1278.4 µs at 16384, stage 2 about 508 µs during the overlap against 425 alone (same MoE as E39; local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/workspace/forge_experiments/lessons/iter_027.md:15–19; local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/workspace/forge_experiments/lessons/iter_029.md:11–15)
- E41 [2026-10, gfx950, FlyDSL 0.3.2, N=6 pairs per arm, AKLR graph timer, arms interleaved in one process, ratio base / candidate; the two A/Bs ran at the same time on one GPU] W1 buffer loads non-temporal (cache modifier 2) in stage 1: 0.791 / 0.795; W2 non-temporal in stage 2 (`use_nt=True`): 0.942 / 0.914; W1 with modifier 1: 0.9984 / 0.9978; at n_tokens 9397 / 16384, 3 / 5 64-row blocks per expert for 110 / 109 of the 112 local experts (fused MoE SiTUv2, fp8 × MXFP4, 112 of 896 experts local, top-16, model dim 3584, inter 3072, stage 1 64×256×256, stage 2 64×256×128; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_012.md:12–14; block counts local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_019.md:21)
