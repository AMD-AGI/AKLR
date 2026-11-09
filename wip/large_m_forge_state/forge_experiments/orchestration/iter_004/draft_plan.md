Direction: per-regime-tile-m-equal-to-the-sort-unit (primary), followed by new | dispatch / routing sort | parallel routing sort for n > 1024

# Plan for iteration 4: close the large-n gap without touching the small-n path

## Where we are

- **Score and split.** The incumbent `ccff3ae8` scores a mean of 0.988.
  - The 20 Points with n ≤ 640 run at 1.024–1.052×.
  - The five large Points run at 0.786× (3112), 0.791× (3816), 0.830× (10000), 0.779× (13152) and 0.724× (32768). These numbers are the driver bench in `lessons/iter_003.md` §4.
  - Almost all of the remaining loss to the mean is on those five Points.
- **Per-kernel times** (iter_003 §3, measured) next to aiter's (iter_002 §1, as cited by the compute specialist):

| n | sort | x-quant | stage 1 | stage 2 | aiter stage 1 / stage 2 |
|---|---|---|---|---|---|
| 256 | 8.9 | 4.7 | 100.9 | 59.6 | — |
| 3816 | 51.4 | 27.9 | **247** | 167 | **162** (t128) / 182 |
| 32768 | **392** | 227 | **1497** | 1346 | 870 (t64) / 1315 |

- **What the table shows.** The large-n gap comes from two places, and they stack:
  1. **Stage 1 at tile_m=64.** At 3112/3816 each expert's w1 (~4.7 MB) is read twice per expert. Stage 1 then moves about 4.9 TB/s, which is near the memory roof. aiter uses tile_m=128 at 3816.
  2. **The single-workgroup routing sort.** `_build_sort` launches `grid=(1,1,1)` (kernel.py:8893). Its cost grows linearly with n: about 42, 51, 120, 158 and 392 µs across the five large Points.
- **Stage 2 is fine.** It is already at or better than aiter's (167 vs 182 µs at 3816, within 2% at 32768). Do not restructure it.
- **The x-quant kernel is not the lever.** I checked the source (8684–8711): it already uses 16-byte coalesced loads and stores with one workgroup per token. The separate zero-fill is already gone. Its 227 µs at L is about 4.1 TB/s once the 403 MB zeroing of `out` is counted. At most about 60 µs (~2%) can be recovered there.

No specialist probe could vary a constant (`_config` reads no environment variable, and SA3 forbids adding that). So every gain number below is argued. Race configurations in scratch copies of the file, not in `run`.

## Step 1: tile_m=128 for the sort and stage 1 at n > 1024 (config-level, do this first)

This is the cheapest change and has the largest expected gain per Point.

**Change `_config`** (kernel.py:8619) to return `(sort_m, s1_tile_m, s1_tn, s1_wpe, s2_tile_m, s2_tn, s2_sort_block_m)`:
- **n ≤ 1024:** `(32, 32, 128, 3, 32, 128, 0)`, exactly as today. The S path must not change: those Points sit near the 1.03 serving gate and have nothing to gain from this.
- **3112 and 3816 (1024 < n ≤ 8192):** `(128, 128, 128, 2, 64, 128, 128)`.
- **n > 8192:** race the same tuple against today's `(64, 64, 128, 3, 64, 128, 0)` and keep whichever wins per Point. A table chosen from the shape is allowed. aiter itself stays at t64 for 32768, so BM=128 is not assumed to win there.

**Plumbing in `_compiled` and `run`:**
- `max_sorted`, `m_blocks`, `n_sorted`, the xs/ys round-up and `_build_sort(tile_m=…)` all use `sort_m`.
- Stage 1 is built with `tile_m=sort_m`, since stage 1 hard-codes `sort_block_m = tile_m` (3062). `s1_blocks = min(m_blocks, R)` is then in 128-row units.
- The scale-buffer bound `round_up(size_expert_ids*128, 256)*24` (3580–3596) is still covered by `ys_sorted` and `xs_sorted`.
- **Stage 2:** pass `tile_m=64, sort_block_m=128`. The builder accepts this: it requires `sort_block_m % tile_m == 0` (6347–6351).
  - **Before choosing `s2_blocks`, verify the unit of `size_expert_ids_in`.** `gy` is computed as `ceil(size_expert_ids_in / persist_m)` (8470–8474). Find where `_sort_block_m` maps a block id to an `expert_ids` index and a row base, and decide whether the count is in 64-row tiles (then use `2*m_blocks`) or 128-row sort blocks (then use `m_blocks`).
  - A wrong count silently skips or duplicates rows. Confirm it with the full correctness run at 3112, 3816 and 32768.
  - **If the mapping is unclear, fall back to stage-2 `tile_m=128`, `sort_block_m=0`.** That needs no unit question, and you can race it.

**Registers and LDS.**
- At BM=128, stage 1 has 128 accumulator VGPRs and about 81 KB of LDS. That fits 2 workgroups per CU with `waves_per_eu=2`, but not with 3.
- Race `s1_wpe ∈ {2, None}`. Confirm in the `FLYDSL_DUMP_IR=1` ISA:
  - `.vgpr_count + .agpr_count ≤ 256`;
  - no `scratch_` instructions;
  - the hot loop still issues 32 MFMAs per K-step.
- Do not try BM=256: it forces split-LDS and one wave per SIMD, and no evidence supports it.

**Sort at tile_m=128.** The current single-workgroup sort compiles a 127-iteration unrolled padding loop. That works, but it is serial per expert; Step 2 replaces it at this size anyway.

**Expected (argued).** Stage 1 at 3112/3816 goes from about 247 to about 165 µs, taking those Points from ~0.79 to ~0.93–0.95. At 10000/13152/32768, BM=128 halves B loads per MFMA. If it wins the race, stage 1 gains 10–30% there.

**Commit** once Step 1 validates and benches above the incumbent.

## Step 2: a multi-workgroup routing sort for n > 1024

Use the algorithm specialist's two-launch design. The memory specialist's version is the same.

**K1 `sort_hist`:** grid G = ceil(n/256), 256 threads, one dwordx4 load of 4 routes per thread.
- Zero an LDS `hist[128]` and barrier.
- Do an LDS `atomic_add` per route and barrier.
- Threads with tid < 128 store all 128 entries to `hist_g[g*128+tid]`. Because every entry is written, `hist_g` can be `torch.empty`.

**K2 `sort_scatter`:** grid G, 256 threads.
- Thread e < 128 loops over all G chunks of `hist_g` and accumulates `tot[e]` and `pre[e]` (chunks before its own). Write this loop with `range_constexpr`; never use a Python `while`, which broke compilation in iter_003 §3.
- Compute the exclusive scan of `padded(tot)` in LDS to get `start[e]`, then set `cur[e] = start[e] + pre[e]` and barrier.
- Each route does an LDS atomic on `cur[e]` to get `p`, then writes `sids[p] = (slot<<24)|tok`, `sw[p] = w` and `pos[i] = p`.
- **Workgroup 0 only:**
  - padding rows, spread across threads rather than a per-expert serial loop;
  - tail rows from `total` to `n_sorted`;
  - `sorted_eids` by the existing binary search, keeping the clamp to `experts-1` exactly;
  - `num_valid = [total, n_tokens]`.

**Rules:**
- No global atomics, no counters, no polling.
- Make G and the tile size compile-time values per n inside `_compiled`.
- Write `sort_m` generically so the sort serves both 64 and 128.

**Dispatch.** Use K1/K2 only when n > 1024; keep `_build_sort` at n ≤ 1024. The new path adds a launch, and the S Points cannot afford 2–4 µs.

**Validation.** Re-run iter_003's exactness check (route placements, padding, per-block expert ids, valid-row count) at 3112, 13152 and 32768. Within-expert order is decided by LDS atomics. That was already true before and is covered by `bounded` determinism.

**Expected (argued).** The sort drops from 42–392 µs to roughly 6–12 µs. That is about +0.07–0.09 on each large Point. Time K1+K2 with the per-kernel trace; the 6–12 µs figure is unmeasured.

**Commit** after Step 2.

## Step 3 (only if time remains): sort micro-work at S, measured per kernel

This step keeps the single-workgroup sort and only reshapes its work:
- Replace the O(E²) per-thread prefix loop (8824–8830) with a two-wave LDS/DPP scan.
- Spread the 31-iteration padding loop (8853) across all 1024 threads.
- Keep `ids`/`weights` in registers between the two passes (`r_iters=1` at S).

The outputs stay identical. The upside is a few µs off 175 µs on 20 Points, roughly +1–2% on the mean, but it is only argued. **Keep it only if the per-kernel trace at n=256 and n=640 shows the sort dropping by at least 2 µs.** Otherwise drop it: a noise-level change at S risks the 1.03 gate for nothing.

## Rejected or deferred, and why

- **Folding x-quant into the sort launch.** x-quant's scale scatter needs `pos`, which needs the global scan, and no workgroup may wait on another. Putting it into K2 leaves x-quant on only 13–128 workgroups, an estimated ≥44 µs vs 28 at 3816 and ~450 vs 227 at L. Keep it as its own token-parallel launch.
- **Rewriting or widening the x-quant kernel.** It is already vectorized with a full grid; the floor is about 155–170 µs vs 227. The "scale pull" redesign (token-order scales plus a gather kernel) is deferred. If someone wants it later, first run the memory specialist's diagnostic: time `moe_xquant` with the `if_w` scatter compiled out. Only pursue the redesign if that saves ≥40 µs.
- **Removing the `out` zeroing / non-atomic stage-2 combine.** Every alternative moves more bytes, and stage 2 is already at parity with aiter.
- **Stage-2 cache policy (`b_nt`).** The ported stage-2 builder executes `del b_nt` (6346), so the knob does nothing. Changing it means editing the vendored B-load for about +1–2%. Not now.
- **Stage-1 `persist_m=2`.** The B panel is evicted from L2 between passes. It is not a substitute for BM=128.
- **BM=256, stage-2 restructuring, XCD tile order as a new mechanism.** `xcd_swizzle` already exists in both builders and is 0 today. You may race nonzero values for stage 1 at M/L as a cheap knob after Steps 1–2, but only after reading its mapping and checking that it keeps a block's 6 N-tiles together. It is not part of the committed plan.

## Correctness and acceptance guardrails

- **Error at L is tight.** The error at 32768 is already 6.5–7.0e-4, attributed to stage-2 atomic order; the contract limit is a violation fraction of 0.001. Any stage-2 grid or tile change (Step 1's fallback, BM choices at L) changes that order.
  - Run the full driver correctness check on all five large Points after each step.
  - If the margin narrows visibly, keep stage 2 at today's configuration for that Point.
- **The S path stays byte-identical** in Steps 1–2. Confirm that the n ≤ 640 times did not regress in the driver bench, since those Points are at 1.024–1.052×.
- **Everything stays in `kernel.py`.** No new files, and no runtime aiter import. Any new FlyDSL code that is copied keeps its SPDX header.
- **Compile choices are made in the `@functools.cache` `_compiled` function.** `run` does no syncs, no environment reads and no timing.

## Expected result (argued, not measured)

| n | now | after Step 1 | after Steps 1+2 |
|---|---|---|---|
| 3112 | 0.786 | ~0.90 | ~0.97 |
| 3816 | 0.791 | ~0.93 | ~1.00 |
| 10000 | 0.830 | ~0.88 (if BM=128 wins) | ~0.97 |
| 13152 | 0.779 | ~0.83 | ~0.92 |
| 32768 | 0.724 | ~0.76–0.80 | ~0.85–0.90 |

The mean would rise from 0.988 to roughly 1.02–1.03.

32768 will still be below 1.0 after all of this. The remaining gap there is stage 1's per-tile efficiency (aiter reaches 870 µs at the same BM=64, against our 1497). That is the next iteration's target: `use_async_copy`, `xcd_swizzle`, or the ledger's compute-leaning main loop. Note it in the session record whether or not Step 1's BM=128 wins at L.
