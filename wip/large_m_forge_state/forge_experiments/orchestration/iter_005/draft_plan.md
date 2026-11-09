Direction: weight-load-cache-policy-per-regime-and-

# Iteration 5 plan: speed up the 20 small Points first, then make a cheap, bit-identical change for the large ones

## The decision in brief

- **The score depends mostly on the 20 small Points (n ≤ 640).** They are 80% of the equal-weight mean, and some sit only just above the 1.03 serving gate. The smallest case speedup in iter_004 was 1.027×.
- **Stage 1 has no room left at small n.** It reads 642 MB in 100.9 µs, about 6.36 TB/s, which is the roof. Every stage-1 knob tried at tile_m=32 lost (iter_004 §2).
- **Two places at small n still have room** (argued from the iter_003 per-kernel split):
  - Stage 2 reaches only about 5.45 TB/s, roughly 8 µs from the roof.
  - The sort and x-quant together take 13.6 µs, and HBM is idle during that time.
- **The large Points come after that.** The one cheap, bit-identical change available is to turn on the stage-1 XCD swizzle that is already in the code. An occupancy change for stage 1 at t128 is optional and must stop early if it does not pay.

The plan has five steps, in order. Each one is measured on its own and kept only if it passes its gate. Clear the FlyDSL cache (`rm -rf "${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}"`) before every timing.

**Gate for every step:**
- the driver correctness run reports `allclose: True`;
- no Point gets slower than the incumbent;
- every Point with n ≤ 640 stays at ≥ 1.03×.

---

## Step 1: stage-2 B loads non-temporal (`b_nt=2`) at n ≤ 1024

This has the best expected gain for its cost and does not change the arithmetic.

**Why:**
- The stage-2 builder accepts `b_nt` and then throws it away (`del b_nt`, `kernel.py:6348`). Its routed W2 load at `load_cell` (`7146-7155`) passes no `cache_modifier`.
- Stage 1 does pass it (`3877`).
- At small n each expert has exactly one M-block, so every W2 byte is read once. That is the same streaming pattern that made `b_nt=2` worth about 13% in stage 1. Moving stage 1 from `b_nt=2` to 0 or 1 cost 14–15 µs (measured, iter_004 §2).
- aiter's small-n stage 2 is also an `_atomic_bnt2` variant (report §4.3).

**Edits:**
1. Remove `del b_nt` at `6348`.
2. Add `cache_modifier=b_nt` to the routed W2 `_buffer_load_vec` in `load_cell` (`7146`).
   - Also add it to every other routed-W2 B load site in the stage-2 builder. Check the `use_async_copy` branch and any preload or prologue B-load helper.
   - Leave the W2-scale loads and the A loads (yq) cached.
3. Add an `s2_bnt` field to `_config`: 2 at n ≤ 1024 and 0 above. Pass it at `9099` instead of the literal `b_nt=0`.
   - **Do not use non-temporal loads at large n.** There each B tile is reused across several M-tiles, and stage-1 `b_nt=2` cost 1.75× at 32768 (measured).
4. Add a `_bnt{b_nt}` tag to the stage-2 kernel/module name so a stale cached binary cannot be timed by mistake.

**Race and keep rule:**
- Time stage 2 alone at n=256 and n=640 with `b_nt` ∈ {0, 1, 2}, then run the driver bench on all 25 Points.
- **Expected (argued): stage 2 goes from 59.6 µs to about 51–54 µs, +3–5% on each of the 20 small Points.**
- Keep it if stage 2 improves by at least 3 µs at both sizes.

## Step 2: issue all x-quant loads before any store (all 25 Points)

**What to change:**
- `xquant` (`8689-8731`) runs `iters = 768/256 = 3` iterations. The algorithm specialist said 6; the source shows 3. Each iteration loads, converts, then stores to `xq`, `out` and the scales.
- The stores go to other buffer resources, and LLVM probably cannot reorder the next iteration's load ahead of them, so the three loads likely run one after another.
- Fix: issue the `pos` load and all three x `buffer_load`s first, keeping them in a list, then do the compute and stores loop.

**Check, gain and risk:**
- **Check:** with `FLYDSL_DUMP_IR=1`, the ISA should show one `s_waitcnt vmcnt` before the first conversion, not one per iteration.
- **Expected (argued):** about 1–2 µs at small n, and a few µs at 3112/3816, where x-quant runs at about 3.8 TB/s.
- **Risk:** the outputs must be byte-identical. Commit this together with Step 1, or right after it.

## Step 3: fold the routing sort into x-quant at n ≤ 1024 (biggest structural gain, time-boxed)

The single-workgroup sort takes 8.9 µs on one CU while about 255 CUs sit idle, and x-quant needs only `pos` from it. **Remove the sort launch on the n ≤ 1024 path.** Each x-quant workgroup (one per token) rebuilds the routing it needs on its own. This is legal: no workgroup waits on another, and every output buffer is rewritten on every call.

**What each workgroup computes:**
1. **Loads first, together with Step 2's loads:**
   - its own `topk_ids[4t..4t+3]` and weights, as dwordx4 (check that `topk_ids` is contiguous);
   - its strided slice of all R ids, about 4–10 per thread, bounds-checked because R is not a multiple of 256 (for example R=1088).
2. **Histogram:** an LDS-atomic histogram `cnt[128]`.
3. **Stable rank** `rk[k]` = the number of routes j < 4t+k whose id is `own_e[k]`. Count per thread, reduce per wave, then add into LDS.
   - This gives a deterministic order: token order within each expert, where the current sort uses atomic arrival order.
   - Row values do not change; only the stage-2 atomic-add order changes, and that order is already arbitrary.
4. **Prefix sum** of `padded(cnt)` into `ends[]`, then `p_k = ends[e_k] − padded(cnt[e_k]) + rk[k]`.
   - Lanes 0–3 write `sorted_ids[p_k] = (k<<24)|t` and `sorted_w[p_k]`.
   - The scale scatter uses `p_k` in place of `pos_s`.
5. **Per-expert duties, in workgroup t < 128 with e = t:**
   - fill that expert's padding rows with exactly the fill value and weight the current `_build_sort` writes (copy them from it);
   - write `sorted_eids` for that expert's blocks.
   - Add a host-side assert that n_tokens ≥ 128. All small Points are ≥ 256.
6. **Tail:** a grid-stride fill of rows `[total, n_sorted)`, plus the tail entries of `sorted_eids`. Clamp the tail eids to `experts−1`, matching the current binary-search clamp.
7. **Valid count:** workgroup 0, lane 0 writes `num_valid = [total, n_tokens]`.

`pos32` is then no longer needed on this path. Stage 1 and stage 2 are **unchanged**, and the n > 1024 path stays byte-identical.

**Validation before the driver run:**
- At n=256, 272 and 640, compare the new buffers with the old sort's: per-expert-block multisets of `(sorted_ids, sorted_w)`, `sorted_eids` exactly, and `num_valid` exactly.
- Then run the full driver correctness check.

**Expected gain and limits:**
- **Expected (argued):** the 8.9 µs launch goes away and x-quant grows by about 2–3 µs, for a net 5–7 µs per small Point, about +3–4%.
- **Unknown cost:** LDS-atomic contention at n=640, where each workgroup handles R=2560. If it shows up, add a tokens-per-workgroup knob in {1, 2}.
- **Time box:** one build plus one debug cycle. If it is not correct by then, take **fallback 3b**: keep a separate sort, but
  - load ids and weights once and keep them in registers (they are loaded twice today);
  - replace the 128-iteration prefix loop with a wave scan;
  - spread the padding fill over all 1024 threads.

  Argued: 8.9 µs → about 4–5 µs, low risk.

## Step 4: stage-1 XCD swizzle at n > 1024, with a remainder guard (5 large Points, bit-identical)

**Why:**
- `_compiled` passes `xcd_swizzle=0` (`9090`). With gx=6, workgroups land on XCD (6y+x) mod 8, so each expert's W1 tiles are fetched into several XCD L2s.
- The remap already exists (`3425-3460`).

**Mandatory correctness fix (confirmed in source):**
- `workgroups_per_xcd = num_workgroups // 8` (`3444`) gives a one-to-one mapping only when 6·grid_y ≡ 0 (mod 8).
- At n=32768, grid_y=1151 and 6906 mod 8 = 2. Unguarded, some tiles would be skipped and others computed twice, giving silently wrong yq.
- **Fix:** apply the XCD remap only when `linear_id < 8·(num_workgroups//8)`, and use the identity mapping for the remainder.
- Prefer this over padding `s1_blocks`. Padding would push buffer bounds past the torch allocations of `sorted_ids`, `sorted_w` and `sorted_eids`.
- Leave stage 2 unswizzled: its gx=48 already pins each N-column to one XCD.

**Race and expected gain:**
- Try `xcd_swizzle` ∈ {0, 1, 2, 4, 8} at 3112, 3816, 10000, 13152 and 32768, timing stage 1 alone. Choose per regime in `_config`, and keep n ≤ 1024 at 0.
- Each stage-1 tile is computed independently with no atomics, so the output must stay bitwise equal. Verify yq bytes against swizzle=0 at 32768.
- **Expected (argued):** 0–7% of stage 1, i.e. up to about 2% per large Point.

## Step 5 (optional, only if budget remains): two stage-1 workgroups per CU at t128

The critic's LDS figure holds: about 97 KB, so 1 workgroup per CU, i.e. 1 wave per SIMD. The cheapest way to get 2 workgroups per CU without changing numerics:

- **Force the existing `split_lds_out` path at t128** (`3199-3213`, `3484-3497`), which comes to about 65 KB.
- **Before racing**, read `.group_segment_fixed_size`, `.vgpr_count` and `.agpr_count` from the descriptor. Two waves per SIMD needs arch+acc VGPRs ≤ 256.
- That path may never have run with fp4-quant output. Run the full correctness check, with the error at 32768 compared against iter_004's 7–8e-4.
- **Stop rule:** if stage 1 at 3816 does not drop from 171 to ≤ 155 µs, revert.
- If the descriptor shows VGPR headroom but split-LDS fails, a two-k-tile B lookahead is the only fallback worth one build.

---

## Rejected, and why

| Idea | Reason |
|---|---|
| Stage-1 tile_n=64 at small n | Stage 1 is already at the roof, and the grid is already exactly 3 workgroups per CU. |
| Per-M-block cache policy | No target: at small n each expert has one block (Step 1's uniform `b_nt` covers it), and at large n B is reused. |
| Staging cshuffle in bf16 to save LDS | Changes the numerics with an error budget already at 7–8e-4, and the bf16-rounding behaviour of the epilogue is unconfirmed. |
| 32x32x64 MFMA rewrite; stage-2 main-loop or s_setprio work | `s_setprio` and a 2-stage B→VGPR pipeline already exist. Stage 2 has only 3 k-iterations, so its cost is in the gather and epilogue, not MFMA. High effort, no shown gain. |
| Stage-2 XCD swizzle | It would change the cross-XCD atomic order on `out` and move the error. Its interaction with `persist_m=-1` is unread. |
| Kernarg preload, expert-id-before-A, skipping padding MFMAs | ≤ 1% (argued). Small-n GEMMs are limited by B bandwidth, and no tile is all padding below `num_valid`. |
| Folding the scatter into x-quant at 3112/3816 | About 49 MB of extra L2 reads to save a ~4 µs launch. |
| MALL prefetch of w1 during the sort window | Speculative (cache retention under `b_nt=2` is unknown). The window mostly disappears after Step 3 anyway. |

## Expected outcome (argued, not measured)

| Steps | Projected mean |
|---|---|
| Incumbent | 1.048 |
| Steps 1–2 | about 1.07 |
| Step 3 added | about 1.09–1.10 |
| Step 4 added | +0.2–0.4% |

The gain comes almost entirely from the 20 small Points. If Step 3 has to fall back to 3b, the projection drops by about 1%.

## Record in the session notes

- Stage-2 per-kernel times at 256 and 640 for each `b_nt` value.
- The x-quant ISA wait count before and after Step 2.
- The sort-equivalence check results.
- Stage-1 times for each swizzle value at all 5 large Points, plus the yq bitwise check.
- If Step 5 was run, its descriptor numbers.

Log `KB:` lines as the knowledge-base skill requires.
