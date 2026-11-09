Direction: weight-load-cache-policy-per-regime-and-

# Iteration 5 plan: faster small Points first, then a bit-identical XCD remap for the large ones

## Summary

- **Where the score is.** The 20 Points with n ≤ 640 make up 80% of the equal-weight mean. In `run_state.json` they are at 1.0275–1.055×, and n=256 is below the 1.03 serving gate at 1.0275×. The five large Points have more margin: 3112 is at 1.052×, 3816 at 1.050×, 10000 at 1.137×, 13152 at 1.055× and 32768 at 1.070×. The plan therefore puts the small Points first.
- **Stage 1 has no room left at small n.** It moves 642 MB in 100.9 µs, about 6.36 TB/s, which is the roof (iter_003 §3). Every stage-1 setting tried at tile_m=32 lost (iter_004 §2).
- **Two places at small n do have room.** These are estimates from the iter_003 per-kernel split:
  - Stage 2 reaches about 5.45 TB/s, roughly 8 µs short of the roof.
  - The sort plus x-quant take 13.6 µs while HBM sits idle.
- **Large Points.** The stage-1 XCD remap already exists in the code but is turned off. It gives bit-identical output, but the current mapping has to be fixed before it can win (Step 4). Step 5 is optional.

## Measurement rules for every step

- Clear the FlyDSL cache (`rm -rf "${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}"`) before every timing.
- Measure each step on its own. Keep it only if it passes the gate below.

**Gate (every step):**
- **(a)** The driver correctness run reports `allclose: True`.
- **(b)** No Point that is at ≥ 1.03× today drops below 1.03×.
- **(c)** No Point gets more than 0.5% slower than the incumbent. If one seems to, rerun the driver bench for that Point before treating it as a real regression.
- **Secondary goal, not a gate:** lift n=256 above 1.03×. Step 1 is the most likely to do this.

---

## Step 1: non-temporal stage-2 B loads (`b_nt=2`) at n ≤ 1024

This has the best gain for its cost, and the arithmetic does not change.

**Evidence from the source:**
- The stage-2 builder throws its `b_nt` argument away (`del b_nt`, `kernel.py:6348`).
- Its routed W2 `load_cell` (`7146-7155`) passes no `cache_modifier`; stage 1 does (`3877`).
- `_compiled` hard-codes `b_nt=0` for stage 2 (`9099`).
- At small n, `s2_tm == tile_m == 32` (`8626`). Each expert then has exactly one block, so W2 is read once, as a stream.
- In stage 1, changing `b_nt` from 2 to 0 or 1 cost 14–15 µs (measured, iter_004 §2). aiter's small-n stage 2 also uses a `_bnt2` variant (report §4.3).

**Edits:**
1. Delete `del b_nt` at `6348`.
2. Add `cache_modifier=b_nt` to the routed W2 `_buffer_load_vec` in `load_cell` (`7146`).
   - At small n, `use_async_copy=False` and `r139_xdma_first` is off (it needs async and tile_m=64, `6355-6363`), so `load_cell` is the only routed-W2 path that matters here.
   - Still search the builder for other `_buffer_load_vec` calls on the W2 resource, and add the modifier to any routed-W2 load you find.
   - Leave the W2-scale loads and the A (yq) loads cached.
3. Add an `s2_bnt` field to `_config`: 2 at n ≤ 1024, 0 above. Pass it at `9099` in place of the literal.
   - Do not use non-temporal loads at large n. B is reused across M-tiles there, and stage-1 `b_nt=2` was 1.75× slower at 32768 (measured).
4. Add a `_bnt{b_nt}` tag to the stage-2 kernel name so the cache cannot serve a stale binary.

**Race and keep rule:**
- Time stage 2 alone at n=256 and n=640 with `b_nt` ∈ {0, 1, 2}. Then run the driver bench on all 25 Points.
- Keep it if stage 2 is at least 3 µs faster at both sizes and the gate passes.
- **Expected (estimate):** stage 2 goes from 59.6 µs to about 51–54 µs, which is +3–5% on each small Point.

## Step 2: issue the x-quant loads before the stores, only if the ISA shows they are serialized

`xquant` (`8689-8731`) runs `iters = 768/256 = 3` iterations (`8642`). Each iteration does load, convert, then store to `xq`, `out` and the scales.

1. **Dump the current ISA first** (`FLYDSL_DUMP_IR=1`).
   - If the three `buffer_load_dwordx4` already sit under one `s_waitcnt vmcnt`, skip this step. Record that the ISA already hoists the loads; it is not a measured negative.
2. Otherwise, issue the `pos` load and all three x loads first, keep them in a list, then run the compute-and-store loop.
   - Confirm in the new ISA that there is one `vmcnt` wait before the first conversion.
   - The outputs must be byte-identical.
3. **Expected (estimate):** about 1–2 µs at small n, and a few µs at 3112/3816, where x-quant runs at about 3.8 TB/s.

## Step 3: fold the routing sort into x-quant at n ≤ 1024 (biggest structural gain, with a stop rule)

The single-workgroup sort takes 8.9 µs on one CU, and x-quant only needs `pos` from it. **Remove the sort launch on the n ≤ 1024 path.**

Each x-quant workgroup (one per token, 256 threads) rebuilds the routing it needs by itself. This is legal: no workgroup waits on another, and every buffer is rewritten on every call.

**Per workgroup:**
1. **Loads first, together with the Step 2 loads:**
   - its own `topk_ids[4t..4t+3]` and weights as dwordx4 (assert `topk_ids` is contiguous);
   - a strided slice of all R ids, bounds-checked because R is not always a multiple of 256 (for example 1088).
2. **Histogram:** an LDS-atomic histogram `cnt[128]`.
3. **Stable rank:** `rk[k]` = the number of routes with index j < 4t+k whose id equals `own_e[k]`. Count per thread, reduce per wave, then add into LDS.
4. **Prefix sum** of `padded(cnt)` with a **wave scan**, not the 128-iteration loop (`8826-8832`). Then `p_k = ends[e_k] − padded(cnt[e_k]) + rk[k]`.
   - Lanes 0–3 write `sorted_ids[p_k] = (k<<24)|t` and `sorted_w[p_k]`.
   - The scale scatter uses `p_k` in place of `pos_s`.
5. **Expert duties, in workgroups t < 128 (with e = t):**
   - fill that expert's padding rows with the existing fill value `(topk<<24)|n_tokens` and weight 0 (`8752`, `8858`);
   - write `sorted_eids` for that expert's blocks;
   - assert n_tokens ≥ 128 on the host.
6. **Tail:** a grid-stride fill of rows `[total, n_sorted)`. Clamp the tail eids to `experts−1`, as the current code does (`8880`).
7. **Valid count:** workgroup 0, lane 0 writes `num_valid = [total, n_tokens]`.

`pos32` is no longer needed on this path. Stage 1, stage 2 and the whole n > 1024 path stay unchanged.

**Validation:**
- **Reference.** The order within each expert changes from atomic arrival order to token order. Row values do not change; only the stage-2 atomic-add order does.
- At n=256, 272 and 640, compare against the old sort:
  - per-expert-block multisets of `(sorted_ids, sorted_w)`;
  - `sorted_eids`, which must match exactly;
  - `num_valid`, which must match exactly.
- Then run the full driver correctness check, explicitly covering n=640, where the atomic order changes the most.

**Stop rule:**
- Each workgroup now runs a serial chain: loads, histogram, barrier, rank, barrier, prefix, barrier, stores. At n=640 the grid takes about 2.5 rounds over the CUs, so the chain is paid about 3 times.
- Measure x-quant at n=640 against the post-Step-2 version.
- **If it is more than 6 µs slower,** or the result is not correct after one build plus one debug cycle, stop and take **fallback 3b**.
- **Fallback 3b:** keep a separate sort, but
  - load ids and weights once and keep them in registers (they are loaded twice today);
  - use a wave scan for the prefix;
  - spread the padding fill over all 1024 threads.

  Estimate: 8.9 µs → about 4–5 µs.
- **Expected (estimate):** net 5–7 µs per small Point, about +3–4%.

## Step 4: stage-1 XCD remap at n > 1024, with a new mapping

**Why the current remap needs changing (from source):**
- `_compiled` passes `xcd_swizzle=0` (`9090`). With gx=6, a workgroup's XCD is (6y+x) mod 8, so each expert's W1 tiles are fetched into several L2s.
- The existing remap (`3440-3447`) has two problems:
  - **It is not a bijection.** It is one only when the workgroup count is divisible by 8. At 32768, 6906 mod 8 = 2, so some tiles would be skipped and others computed twice.
  - **It gives each XCD a contiguous range of tiles.** `grid_y = s1_blocks = m_blocks` is an upper bound based on `R + E·(tile_m−1)`. The empty blocks past `num_valid` are therefore all at the end, roughly 25–30% of the grid at 3112, 14% at 10000 and 5% at 32768 (estimate). With contiguous ranges they all land on the last one or two XCDs, which would make stage 1 slower, not faster.

**New mapping.** Replace the body at `3440-3460`, keeping the `const_expr(xcd_swizzle > 0)` gate:
```
xcd     = linear_id % 8
slot    = linear_id // 8
wpg     = group_m * grid_x                  # one super-group = group_m m-blocks × all N tiles
full    = (num_workgroups // (8*wpg)) * (8*wpg)
if linear_id < full:                         # remap only the full rounds
    g          = (slot // wpg) * 8 + xcd     # super-group g runs on XCD g mod 8
    in_g       = slot % wpg
    bx_persist = g*group_m + in_g % group_m
    by         = in_g // group_m
else:                                        # tail: identity
    bx_persist = linear_id // grid_x
    by         = linear_id % grid_x
```

Why this works:
- `full` is a multiple of `grid_x`, so the tail covers exactly the m-blocks from `full/grid_x` upward. The mapping is therefore a bijection for every n.
- Each expert's m-blocks stay together on one XCD.
- The empty tail, both the trailing super-groups and the identity tail, is spread evenly over all eight XCDs.
- The mapping assumes workgroups are dispatched round-robin to XCDs in `linear_id` order.

**Race:**
- Try `group_m` ∈ {0 (off), 1, 2, 4, 8} at 3112, 3816, 10000, 13152 and 32768, timing stage 1 alone. Record every result.
- Pick the value per regime in `_config`. Keep n ≤ 1024 at 0.
- Check that yq is byte-identical to swizzle=0 at 32768 and 3112. Each stage-1 tile is computed independently with no atomics, so any difference is a mapping bug.
- **Drop rule:** if no setting is at least 3% faster on stage 1 at 32768, or any setting slows stage 1 at any large Point, leave swizzle at 0 for that regime. If it wins nowhere, drop the step.
- **Expected (estimate):** 0–7% of stage 1, which is up to about 2% per large Point.

## Step 5 (optional, unlikely to be reached this iteration): two stage-1 workgroups per CU at t128

- At t128, stage 1 uses about 97 KB of LDS, so only 1 workgroup per CU fits (1 wave per SIMD).
- Forcing the existing `split_lds_out` path (`3199-3213`, `3484-3497`) would bring it to about 65 KB.
- **Before racing**, read `.group_segment_fixed_size`, `.vgpr_count` and `.agpr_count` from the kernel descriptor. Two waves per SIMD needs arch+acc VGPRs ≤ 256.
- That path may never have run with fp4-quant output. Run the full correctness check, and compare the error at 32768 with iter_004's 7–8e-4.
- **Stop rule:** if stage 1 at 3816 does not drop from 171 µs to ≤ 155 µs, revert.

---

## Rejected, and why

| Idea | Reason |
|---|---|
| Stage-1 tile_n=64 at small n | Stage 1 is already at the roof, and the grid already fills exactly 3 workgroups per CU. |
| Per-M-block cache policy | No target: at small n each expert has one block (Step 1 covers it), and at large n B is reused. |
| Staging cshuffle in bf16 | Changes the numerics while the error is already at 7–8e-4, and how the epilogue rounds is unconfirmed. |
| 32x32x64 MFMA rewrite; stage-2 main-loop or s_setprio work | `s_setprio` and the B→VGPR pipeline already exist. Stage 2 has only 3 k-iterations. High effort, no shown gain. |
| Stage-2 XCD swizzle | Changes the atomic order on `out`, so the error moves. Its interaction with `persist_m=-1` has not been read. |
| Kernarg preload, expert-id-before-A, skipping padding MFMAs | ≤ 1% (estimate): the small-n GEMMs are limited by B bandwidth. |
| Folding the scatter into x-quant at 3112/3816 | About 49 MB of extra L2 reads to save a launch of about 4 µs. |
| MALL prefetch of w1 | Speculative, and the idle window it would fill mostly disappears after Step 3. |

## Expected outcome (estimates, not measured)

| Steps done | Projected mean |
|---|---|
| Incumbent | 1.048 |
| **Conservative: Step 1 alone delivers, Step 3 neutral or 3b** | **about 1.07** |
| Steps 1–3 all deliver | about 1.09–1.10 |
| Step 4 | +0.2–0.4% on top |

A neutral Step 3 or Step 4 does not mean this route failed.

## What to put in the session notes

- Stage-2 per-kernel times at n=256 and n=640 for each `b_nt` value, and whether n=256 crossed 1.03×.
- The x-quant ISA wait counts before the Step 2 edit, and after it if it was made.
- The sort-equivalence results, and the x-quant time at n=640 compared with the 6 µs stop rule.
- Stage-1 times for each `group_m` value at all five large Points, plus the yq bitwise checks.
- The Step 5 descriptor numbers, if Step 5 was run.
- `KB:` lines, as the knowledge-base skill requires.
