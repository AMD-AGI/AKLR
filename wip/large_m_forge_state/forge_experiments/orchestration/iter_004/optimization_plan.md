Direction: new | dispatch / routing sort | parallel routing sort for n > 1024 (first), then per-regime-tile-m-equal-to-the-sort-unit (a time-boxed race)

# Iteration 4 plan: close the gap at large n without touching small n

## Starting point

- **Current score.** The incumbent `ccff3ae8` has a mean of 0.988 (driver bench, `lessons/iter_003.md` §4).
  - The 20 Points with n ≤ 640 run at 1.024–1.052×.
  - The five large Points run at 0.786× (3112), 0.791× (3816), 0.830× (10000), 0.779× (13152) and 0.724× (32768). They account for almost all of the remaining loss.
- **Per-kernel times in µs.** Ours are measured (iter_003 §3); aiter's are as cited from iter_002 §1:

| n | sort | x-quant | stage 1 | stage 2 | aiter stage 1 / stage 2 |
|---|---|---|---|---|---|
| 256 | 8.9 | 4.7 | 100.9 | 59.6 | — |
| 3816 | 51.4 | 27.9 | 247 | 167 | 162 (tile t128x128x256, kernel type not recorded) / 182 |
| 32768 | 392 | 227 | 1497 | 1346 | 870 (t64) / 1315 |

- **Two costs grow with n:**
  1. **The routing sort runs in one workgroup** (`kernel.py:8893`, `grid=(1,1,1)`). It takes about 42, 51, 120, 158 and 392 µs across the five large Points (3112 → 32768). Removing it does not depend on any open question.
  2. **Stage 1 is slower than aiter's.** It runs at tile_m=64; at 3112/3816 that means about two w1 passes per expert. Whether tile_m=128 helps in *this* builder is an open race (Step 2).
- **Stage 2 is not the problem.** It is at or better than aiter's, so it is not restructured.
- **x-quant is not the lever.** Source lines 8684–8711 show it already uses 16-byte coalesced loads and stores with one workgroup per token. At L it moves about 930 MB in 227 µs (~4.1 TB/s), so at most about 60 µs is recoverable.
- **No gain figure below is measured.** No probe could vary a constant: `_config` reads no environment, and SA3 forbids adding that. Race configurations in scratch copies, never in `run`.

## Step 1 (first): a multi-workgroup routing sort for n > 1024

This step comes first because its gain is close to certain and independent of the stage-1 occupancy question.

**Dispatch.**
- Use the new sort only for n > 1024.
- Keep `_build_sort` exactly as it is for n ≤ 1024. The new path adds a launch, and the S Points (1.024–1.052×) cannot afford 2–4 µs.

**K1 `sort_hist`** (grid G = ceil(n/256), 256 threads):
1. Each thread loads its token's 4 routes with one dwordx4 load. `topk_ids` is read as i32 today (8820); assert that it is contiguous.
2. Zero an LDS `hist[128]`, then barrier.
3. LDS `atomic_add` for each route, then barrier.
4. Threads with tid < 128 store all 128 entries to `hist_g[g*128+tid]`, zeros included. Because every entry is written, `hist_g` can be `torch.empty((G,128), int32)`.

**K2 `sort_scatter`** (grid G, 256 threads):
1. **Read the histogram.** Thread e < 128 loops over all G rows of `hist_g` and accumulates two values:
   - `tot[e]`, over all rows;
   - `pre[e]`, over rows g' < g.

   Write this loop with `range_constexpr`; G is a compile-time value per n. Never use a Python `while`, which broke compilation in iter_003 §3.
2. **Expert starts.** Compute the exclusive scan of `padded(tot)` over the 128 experts in LDS to get `start[e]`. Set `cur[e] = start[e] + pre[e]`, then barrier.
3. **Scatter routes.** Each route takes `p = LDS atomic_add(cur[e], 1)` and stores:
   - `sids[p] = (slot<<24)|tok`;
   - `sw[p] = w`;
   - `pos[i] = p`.
4. **Workgroup 0 only** writes:
   - the per-expert padding rows, `(topk<<24)|n` with weight 0, spread across threads rather than a serial loop per expert;
   - tail rows from `total` to `n_sorted`;
   - `sorted_eids` by the existing binary search, with the clamp to `experts-1` kept exactly;
   - `num_valid = [total, n_tokens]`.
5. **Rules:**
   - no global atomics, no counters, no polling;
   - parameterize the sort by `sort_m`, so Step 2 can reuse it at 64 or 128.

**Validation.**
- Re-run iter_003's exactness check at 3112, 13152 and 32768: each route's placement, the padding, the per-block expert ids and the valid-row count.
- Order within one expert is still set by LDS atomics. That was already true before, and `bounded` determinism covers it.
- Run the full driver correctness check on all five large Points.

**Timing.** Time K1+K2 on their own in the per-kernel trace; the estimate is 6–12 µs, unmeasured.

**Commit** once it validates and benches above the incumbent. Expected effect: about +0.07–0.09 on each large Point and a mean of about 0.988 → ~1.00 (argued).

## Step 2: a time-boxed tile_m=128 race for stage 1 at n > 1024

**The correction from review: BM=128 halves stage-1 occupancy relative to today.**
- With fp4 output, `cshuffle_elem_bytes = 4` (3179). The builder formulas (3180–3194) give the following LDS budget at BM=128, tile_n=128:

| Buffer | Formula | Size |
|---|---|---|
| x region | 128×256 | 32 KB |
| cshuffle out | 4×128×128 | 64 KB |
| tid | 128×4 | 512 B |
| pong | max(x, out) + tid | ≈ 64.5 KB |
| ping | x region | 32 KB |
| **Total** | pong + ping | **≈ 97 KB** |

- 97 KB is more than half of 160 KB, so only **1 workgroup per CU fits: 4 waves, 1 wave per SIMD**.
- `waves_per_eu` padding (3224–3231) adds nothing here.
- Today's BM=64 uses about 49 KB with `wpe=3`, so 3 workgroups per CU fit (12 waves).
- BM=128 therefore trades a 3× occupancy drop for half the B traffic, and the sign of the result is unknown.
- `tile_n=64` does not help: LDS fits 2 workgroups, but `num_waves = tile_n//32 = 2` (3063), so it is still 4 waves per CU.
- **aiter's 162 µs at 3816 is not a target for this builder.** iter_002 does not say which kernel produced it, and aiter is already about 1.7× faster per tile than our builder at the same BM=64 at L.

**Budget.**
- One build of the configuration below.
- Race `s1_wpe ∈ {2, None}` at 3816 and 32768.
- **Stop rule:** if the per-kernel trace at 3816 does not show stage 1 at ≤ 220 µs (at least 10% better than 247), revert `_config` to today's values for that regime and stop.
- At n > 8192, keep BM=128 only for Points where it wins the race. A table chosen from the shape is allowed.

**Configuration.** `_config` returns `(sort_m, s1_tile_m, s1_tn, s1_wpe, s2_tile_m, s2_tn, s2_sort_block_m)`:
- **n ≤ 1024:** `(32, 32, 128, 3, 32, 128, 0)`, byte-identical to today.
- **Racing candidate for n > 1024:** `(128, 128, 128, 2 or None, 64, 128, 128)`.
- **Fallback:** today's `(64, 64, 128, 3, 64, 128, 0)`.

**Plumbing.**
- The following all use `sort_m`:
  - `max_sorted`, `m_blocks`, `n_sorted`;
  - the xs/ys round-up;
  - the Step 1 sort's padding unit.
- Stage 1 is built with `tile_m = sort_m`, since it hard-codes `sort_block_m = tile_m` (3062).
- `s1_blocks = min(m_blocks, R)` is in 128-row units. The scale bound `round_up(size*128, 256)*24` (3580–3596) stays covered by xs/ys.
- **Stage 2 is settled from source:** `tile_m=64`, `sort_block_m=128` (allowed, 6347–6351).
  - `size_expert_ids_in` is counted in **stage-2 tile_m units**. The sorted-id/weight bound is `size*tile_m*4` (6786–6793). Then `bx_m = bx*tile_m` and `sort_blk = bx_m/_sort_block_m` (6851–6858). The expert-id bound is `ceil(size*tile_m/sort_block_m)` (6798–6803).
  - So pass `s2_blocks = n_sorted // s2_tile_m`, which is `2*m_blocks` here.
  - Compute the `persist_m` choice (`-1 if s2_blocks > 256`) from that count.
  - In persistent mode the tile count comes from `num_valid` (6811–6824). An undersized `size_expert_ids_in` would only shrink the buffer bound: rows past it read 0, their weight becomes 0, and those routes would be silently dropped. The full correctness run must therefore include 3112, 3816 and 32768.
  - No stage-2 BM=128 fallback is needed.

**ISA and descriptor check** (`FLYDSL_DUMP_IR=1`):
- `.group_segment_fixed_size` is about 97 KB, which confirms 1 workgroup per CU;
- `.vgpr_count + .agpr_count ≤ 256`;
- no `scratch_` instructions;
- the hot loop still issues 32 MFMAs per K-step.

**Padding at 10000/13152.** An expert with about 312 routes pads to 384 rows at sort_m=128 instead of 320. Stage 1 then makes 3 passes instead of 5, but stage 2 at tile 64 gets one extra, entirely padding, tile per expert. Record the stage-1 **and** stage-2 per-kernel times at those two Points next to each other.

Do not try BM=256: it forces split-LDS and one wave per SIMD, and no evidence supports it.

**Commit** only Points that win; otherwise revert. A neutral or negative result here is an expected outcome, not a failure of the route.

## Step 3 (only if time remains): sort micro-work at S

Keep the single-workgroup sort and reshape only its work. The outputs stay identical.
- Replace the O(E²) prefix loop (8824–8830) with a two-wave LDS/DPP scan.
- Spread the padding loop (8853) across all threads.
- Keep `ids`/`weights` in registers between the two passes.

**Keep it only if** a measured driver bench at n=256 and n=640 shows the sort at least 2 µs faster with no regression. Otherwise drop it, so the 1.03 serving gate is not put at risk.

## Rejected or deferred

- **Folding x-quant into the sort.** It needs `pos` from the global scan, so it could only run inside K2, on 13–128 workgroups: an estimated ≥44 vs 28 µs at 3816 and about 450 vs 227 µs at L.
- **x-quant rewrite or "scale pull".** Deferred. Only revisit it if timing `moe_xquant` with the `if_w` scatter compiled out (a diagnostic only, since the output becomes wrong) saves ≥40 µs.
- **Removing the `out` zeroing / a non-atomic combine.** Every alternative moves more bytes, and stage 2 is already at parity with aiter.
- **Stage-2 `b_nt`.** It is a no-op because of `del b_nt` (6346).
- **Stage-1 `persist_m=2`.** The B panel is evicted from L2 between passes.
- **BM=256.**
- **Nonzero `xcd_swizzle` or `use_async_copy` on stage 1.** Next iteration's levers for per-tile efficiency at L. Race them only after reading the mapping; they are not part of this plan.

## Guardrails

- **Error at L.** The error at 32768 is already 6.5–7.0e-4, against a 0.001 violation fraction. Run the full driver correctness check on all five large Points after each step. If the margin narrows visibly, keep that Point's stage-2 configuration as it is today.
- **S path.** The n ≤ 1024 path stays byte-identical in Steps 1–2. Confirm in the driver bench that the n ≤ 640 times did not regress.
- **Files.** Everything stays in `kernel.py`: no new files and no runtime aiter import. Copied code keeps its SPDX header and an AMD modifications line.
- **Compilation.** Every configuration choice is made inside the `@functools.cache` `_compiled` function. `run` does not sync, read the environment or time anything.

## Expected outcome (argued, not measured)

| n | now | after Step 1 (sort) | if Step 2 wins |
|---|---|---|---|
| 3112 | 0.786 | ~0.85 | ~0.90+ |
| 3816 | 0.791 | ~0.87 | ~0.93+ |
| 10000 | 0.830 | ~0.92 | race |
| 13152 | 0.779 | ~0.87 | race |
| 32768 | 0.724 | ~0.81 | race |

- **Conservative outcome:** Step 1 alone takes the mean from 0.988 to about 1.00.
- **If Step 2 clears its stop rule:** about 1.01–1.02.
- **What remains for next iteration:** 32768 stays below 1.0 either way. The remaining gap is stage 1's per-tile efficiency: 1497 vs aiter's 870 µs at the same BM=64. Record that in the session notes.
