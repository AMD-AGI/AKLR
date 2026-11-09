Direction: replace-the-torch-path-with-a-fused-flyd

# Plan for iteration 3: remove the torch glue around the inlined FlyDSL stage kernels

## 0. Where we are and what this plan targets

**State at commit 6469848.** `run()` makes 2 FlyDSL launches, the aiter stage-1 and stage-2 builders. They sit inside torch glue:

- a torch sort (`kernel.py:8705-8722`);
- `_quant(x)` plus `_sorted_scales` (8725-8726);
- stage 1 writing **bf16** y;
- `_quant(y)` plus `_sorted_scales` (8738-8739);
- `out.zero_()`.

The score is 0.179× aiter on every Point.

**Measured (iter_002 §1).** aiter's stage kernels take about 158 µs at n=256, against a 0.879 ms call for us. At n=32768 they take 2.19 ms, against our 24.4 ms.

**Inferred, not measured.** Almost all of the gap is glue:

- At S the cost is node count. Estimates of about 70–145 nodes and about 4–9 µs per node all point the same way.
- At L the cost is bandwidth: `_quant` makes about 20 or more fp32/int64 passes over x and y.

No per-kernel trace exists for our build. The first thing this iteration does is capture one at n=256 and n=32768, so the split between glue and stage kernels is measured, not assumed.

All three specialists agree on what to remove, and I checked their key source claims myself. The plan below is their work merged into one ordered sequence. Each step must pass validate on all 25 Points and be committed before the next step starts. Edit only `kernel.py`, and run `git status` after every step; iteration 1 was reverted for adding files.

## 1. Facts checked in source that the plan depends on

- **Stage 1 already has an MXFP4 output path; it is switched off.**
  - `need_fp4 = out_dtype == "fp4"` and `need_sort = need_quant` are set at `kernel.py:3150-3153`.
  - With `v2_output_layout=False`, code rows are written at `ts_idx = t*topk + s` (5339-5342), with a 384-byte row stride (5317-5325). That is the route-order `(R, 384)` layout stage 2 already reads as `yq`.
  - E8M0 bytes go to `arg_out_scale_sorted` at sorted row `row`, at offset `d0*n32_sort + d3*256 + d5*64 + d2*4 + d4*2 + d1` (5585-5614). That is the same permutation as `_sorted_scales`.
  - `arg_out_scale_sorted` is already in the launcher signature (6267). Today it is fed `empty_u8`.
- **Keep `v2_output_layout=False`.** This settles the earlier dispute over the flag. Setting it to True would move the codes into sorted-row order while stage 2 still reads route order. Do not change it on either side.
- **Epilogue E8M0 rule.** It is `max(((bits(amax)+0x400000)&0xFF800000)>>23 − 2, 0)` (5441-5444). It differs from torch's strict `> 0.75` only on an exact mantissa tie. The epilogue also quantizes fp32 values, not bf16-rounded ones. Both behaviours are aiter's own.
- **Unresolved; check before Step 1.** `n32_sort = sorted_scale_cols_i32 * 32` (5414). Find where `sorted_scale_cols_i32` is defined and confirm it is 24 (768/32, padded to 8). Also confirm that the `sorted_scale_rsrc` buffer bound covers `round_up(n_sorted,256)*24` bytes.

## 2. Step 1: stage 1 writes MXFP4 output (host change plus one compile flag)

**Changes in `_compiled` and `run`:**

- Compile stage 1 with `out_dtype="fp4"`. Leave everything else unchanged.
- Replace the bf16 `y` with `yq = torch.empty((R, 384), uint8)` and pass it as `arg_out`.
- Allocate `ys_sorted = torch.zeros(round_up(n_sorted,256)*24, uint8)` and pass it in the `arg_out_scale_sorted` slot, which today receives `empty_u8`. Use `zeros`, not `empty`, for this first landing: padding rows get no scale writes, and `_sorted_scales` zero-fills today.
- Delete `_quant(y)` and `_sorted_scales(ys)`. Stage 2's arguments stay `yq` and `ys_sorted`.

**One-off parity check (scratch, outside `kernel.py`).** At n=256, compare the epilogue's `yq` and `ys_sorted` bytes with `_sorted_scales(_quant(y_bf16))`. A mismatch of about 1e-3 or less is expected. A large mismatch means a layout or `n32_sort` bug; fix that before running validate.

**Expected (argued).** About 20–45 fewer nodes. Roughly 0.88 → 0.65 ms at S, and about 5–9 ms less at n=32768.

**Stop rule.** If validate fails and the parity check does not explain it, revert to the bf16 y and go on to Step 2. Step 2 does not depend on Step 1.

## 3. Step 2: one FlyDSL x-quant kernel that also scatters sorted scales and zeroes `out`

This replaces `_quant(x)`, `grid_t`, `xs[r//top]`, `_sorted_scales(xs)` and `out.zero_()`. It is the biggest win at L.

**Layout.** I chose the memory specialist's layout, which coalesces better than one lane per block:

- Grid = M workgroups of 256 threads. Each workgroup owns one token row: 6144 bf16 = 768 chunks of 16 B, so each thread handles 3 chunks of 8 elements.
- A 32-element block covers 4 adjacent lanes. Take amax over `shuffle_xor` 1 and 2.
- **E8M0 rule:** use the one that already passed validate, the torch strict rule: `e = ((bits>>23)&0xFF) − 2 + ((bits & 0x7FFFFF) > 0x400000)`, clamped to [0, 254].
  - Do not copy the epilogue's `+0x400000` rounding here. With bf16's 7-bit mantissa, exact ties occur in about 1/128 of blocks.
  - Trying the aiter rule instead is an optional later A/B, only if validate margins are tight.
- **Codes:** copy the epilogue's `cvt_scalef32_pk_fp4_f32(old, a, b, native_scale, k)` pattern exactly, with `native_scale = (e8m0<<23).bitcast(f32)` (5449-5472). Convert bf16 to f32 first. Each lane stores 4 bytes, so each wave writes a contiguous 256 B, in **token order** `(M, 3072)` as `xq` is today.
- **Scales:** each lane with `lane%4==0` writes its E8M0 byte to the 4 sorted positions `pos[4t+k]`, using the shuffled offset with `n32_sort = 192*32`.
  - Until Step 3 lands, `pos` is the existing torch `pos` cast to int32.
  - `xs_sorted` starts as `torch.zeros(round_up(n_sorted,256)*192)`.
- **Zero `out`:** the same workgroup writes zeros over `out[t,:]`, 12 KB as 3 × 16 B per thread. This is safe because stage 2 runs in a later launch.
- Compile the kernel in `_compiled` and launch it through the existing `_launch` path. Do not read the environment, synchronize or print in `run`.

**Fallback.** If the scattered 1-byte scale stores are measurably slow at L: write E8M0 in token order and add a small scale-sort kernel that emits coalesced 256 B chunks. This is aiter's split and costs one more node.

**Expected (argued).**

- S: about 0.35–0.45 ms, now limited by the torch sort.
- 32768: about 2.8–3.3 ms, because about 0.93 GB of traffic replaces the roughly 10–14 ms torch quant.
- Mean score roughly 0.45–0.55.

## 4. Step 3: one FlyDSL routing-sort kernel in a single workgroup

This replaces the roughly 20–45-node torch sort. It is what brings S toward 1.0×.

**Shape.** 1024 threads, with three 128 × i32 LDS arrays: `cnt`, `start` and `cur`. Use LDS atomics through `llvm.AtomicRMWOp` on address space 3, the same API used at 5624.

1. Zero the counters.
2. For each route, `atomic_add(cnt[e])`.
3. One wave computes the exclusive prefix of `roundup(cnt, sort_block)`, giving `start[e]` and the total.
4. For each route, take `rank = atomic_add(cur[e])` and write:
   - `sorted_ids[start+rank] = (k<<24)|t`;
   - `sorted_w`, read directly from `topk_weights`;
   - `pos[4t+k]`.
5. For every slot below `m_blocks*sort_block` that is not valid, write `(top<<24)|M` and weight 0. Padding slots and valid slots never overlap.
6. Compute `sorted_eids[b]` by binary search over `start`, with `searchsorted(right=True).clamp_max(E-1)` semantics.
7. Write `num_valid = [total, M]`.

**Determinism.** The order of ranks within an expert changes between runs. No value changes:

- each row's GEMM depends only on its own A row and scale;
- stage 2 already accumulates with atomics in arrival order, which `bounded` determinism allows.

**Optional source.** If adapting the deterministic `_route_expert` in KB case `ef15e878` to the packed `sorted_ids` encoding is faster than writing this kernel, lift **only the route kernel**. Do not lift its fp8 stage 2.

**Cost at L.** An argued 30–60 µs for 131k routes, against aiter's 77 µs. If one workgroup turns out slower than the torch sort at L, keep the torch sort for n > 1024. `_compiled` can choose per n. Multi-workgroup sort comes later.

**End state.** 4 launches: sort → x-quant+zero → stage 1 (fp4 out) → stage 2, plus up to 2 zero-fill memsets. aiter uses 6.

**Micro-step afterwards.** Switch `ys_sorted` and `xs_sorted` to `torch.empty` and re-validate. The masked padding rows should make this safe. If validate shows NaN, have the sort kernel zero the padding-row scale bytes instead. This saves 2 nodes, about 2–4% at S.

**Expected (argued).** S at about 0.17–0.22 ms against aiter's 0.179 ms, and mean score roughly 0.85–1.0. Once the glue is gone, the remaining unknown is our FlyDSL stage 1 against aiter's **asm** `t32x128x256` stage 1, which takes 100.5 µs. Capture a per-kernel trace at n=256 to measure it.

## 5. Step 4, only after Steps 2–3 are committed: tile_m for the 5 large Points

This was the compute lane. A tile_m change is buried in noise while glue dominates, so do not race it earlier. The one exception is a cheap BM=128 swap at 3112/3816, which can be slotted in if Step 3 stalls.

- **Stage 2 does not need to share stage 1's tile_m.** Its `sort_block_m` parameter (6343-6347) allows any stage-2 tile_m that divides the sort block. The rule is therefore: sort block = stage-1 tile_m, and stage-2 tile_m divides the sort block.
  - When the two differ, pass `sort_block_m=<sort block>` and count stage 2's `i32_size_expert_ids_in` in stage-2 tiles: `m_blocks*sort_bm//s2_tm`.
  - The sort kernel takes the sort block as a constexpr.
- **3112 and 3816: use sort block = stage-1 tile_m = 128.** This halves the weight sweep, from about 1.9 to about 0.96 GB at 3112, and matches aiter's t128. For stage 2, race tile_m 128 against 64.
- **10000, 13152 and 32768:** race three stage-1 configs:
  - BM=128 with `waves_per_eu` 2;
  - BM=64 with `persist_m` 2 or 4;
  - aiter parity, BM=64.
  
  Stage-2 persistence is already active, so the only extra knob is `cu_num_mul` 1 against 2. Do not use BM=256, because of accumulator VGPR spills.
- **Before crediting any timing change:**
  - Confirm from the dumped kernel descriptor (`FLYDSL_DUMP_IR=1`) that `waves_per_eu` reaches the ISA. The launcher sets it only under `heterogeneous_b` (6190-6194).
  - Confirm that BM=128 has no scratch spill.
- **n ≤ 640 stays exactly as it is now** (32/128/3/128). Choose the config from n inside `_compiled`, from a table raced offline. Clear the FlyDSL cache before timing.

## 6. Rejected or deferred

- **`v2_output_layout=True`:** rejected; see §1.
- **Merging sort and x-quant into one launch** (last-arriving-workgroup counter, as in ef15e878): about −1 node, deferred until Step 3 is committed.
- **Deferred as separate lanes after this sequence:**
  - stage-2 combine A/B (atomics against fixed-order reduce), the largest L lever since aiter's stage 2 is 54% of its call at 32768;
  - XCD ordering;
  - cache policy;
  - S-regime tile_n.
- **Stage-1 tile_n=256 for wider fp4 stores:** y codes are about 1% of traffic at L, so this goes into the tuning race only.
- **Quantizing x in sorted-row order:** rejected, because it reads x 4×.
- **A probe-only re-time of the current source:** no value, because no knobs exist.

## 7. Acceptance and session-time policy

- A step counts only if driver validate passes on all 25 Points, `git status` shows only `kernel.py`, there are no runtime aiter imports, and SA1–SA4 hold:
  - scratch is `torch.empty` or `torch.zeros` per call;
  - nothing persists across calls;
  - the entry docstring is free of history.
- Commit after **every** validated step. The incumbent is 0.179, so Step 1 alone is a KEEP.
- If the session is running short, the priority order is Step 1 → Step 2 → Step 3. Never leave the file in an unvalidated state.
- Record `KB: matched …` / `KB: no-match …` lines in the lesson. Before Step 3, read `moe/INDEX.md` and `cases/fused_moe_shuffled_16x16.md`.
