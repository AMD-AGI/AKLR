# Review of the iteration-4 plan: per-regime tile-M, then a multi-workgroup sort

VERDICT: REVISE

**Bottom line:** the route is right, but Step 1's feasibility claim is wrong and its gain estimate does not hold. Fix those, settle the open stage-2 question from the source (below), and change the order before executing.

## Route check (no replacement needed)

- **The loss is in the five large Points.** The 20 Points with n ≤ 640 are at 1.024–1.052×. The five large ones are at 0.72–0.83× (`lessons/iter_003.md` §4).
- **Two costs grow with n there:**
  - the routing sort runs in one workgroup (`kernel.py:8893`, `grid=(1,1,1)`): 392 µs at n=32768 and 51 µs at n=3816 (iter_003 §3);
  - stage 1 is far slower than aiter's: 247 vs 162 µs at 3816, and 1497 vs 870 µs at 32768 (iter_002 §1).
- **The rejections are sound, and I checked the ones that rest on source:**
  - x-quant already uses 16-byte loads and stores with one workgroup per token (`8684–8711`).
  - Stage 2 throws away its cache-policy knob: `del b_nt` (`6346`).
  - Folding x-quant into the sort would make it wait on `pos`, and it would run on too few workgroups.
- **No alternative backend dominates.** Running aiter is not allowed (operator guidance), so I see no case for a challenger.

## Problems that change the plan

### 1. The LDS figure for Step 1 is wrong: BM=128 fits only one workgroup per CU, not two

The plan, following the compute specialist, says BM=128 uses about 81 KB of LDS and fits 2 workgroups per CU at `waves_per_eu=2`. The stage-1 builder's own formulas give a different answer (`kernel.py:3179-3231`). With fp4 output, `cshuffle_elem_bytes = 4` (`3179`):

| Buffer | Formula | Bytes |
|---|---|---|
| x region | 128 × 256 | 32 KB |
| cshuffle out | 4 × 128 × 128 | 64 KB |
| tid | 128 × 4 | 512 B |
| pong | max(x, out) + tid | ≈ 64.5 KB |
| ping | x region | 32 KB |
| **Total** | pong + ping | **≈ 97 KB** |

- 97 KB is more than half of 160 KB, so only **1 workgroup per CU** fits: 4 waves, which is 1 wave per SIMD.
- `waves_per_eu=2` adds no padding here, because the minimum is about 54.6 KB and the kernel already uses more.
- Today's BM=64 uses about 49 KB with `wpe=3`, giving 3 workgroups per CU, or 12 waves.
- So BM=128 trades a **3× drop in occupancy** for half the B traffic. That is not obviously a win.
- `tile_n=64` at BM=128 does not help: LDS drops to about 65 KB and 2 workgroups fit, but `num_waves = tile_n//32 = 2` (`3063`), so it is still 4 waves per CU.

**The 247 → ~165 µs estimate leans on aiter's 162 µs, which iter_002 §1 does not show came from this builder.** Treat that number as untransferred:
- iter_002 §1 tags aiter's stage-1 kernel as asm at n=256 (`t32x128x256`). At 3816 it records only the tile, `t128x128x256`, and does not say what kind of kernel it is.
- The draft itself notes that aiter is 1.7× faster per tile than our FlyDSL stage 1 at the same BM=64 at L. So aiter's numbers do not tell us what this builder will do at BM=128.

**Required revision:**
- State that the occupancy at BM=128 is 4 waves per CU.
- Confirm it from the kernel descriptor (`.group_segment_fixed_size`) next to the VGPR and AGPR counts.
- Treat the gain as an open race, not as an expected ~+0.15 per Point.
- Add a stop rule: if the per-kernel trace at 3816 does not show stage 1 at ≤ about 220 µs (at least 10% better), keep BM=64 there and move on.

### 2. The stage-2 unit question is answered by the source: count 64-row tiles

In stage 2, `size_expert_ids_in` is counted in **`tile_m` units**:
- The sorted-id and sorted-weight buffer bound is `size_expert_ids_in * tile_m * 4` (`6786-6793`).
- The block's first row is `bx_m = bx * tile_m`, and its expert is `sort_blk = bx_m / _sort_block_m` (`6851-6858`).
- The expert-id bound is `ceil(size*tile_m / sort_block_m)` (`6798-6803`).

So with `tile_m=64, sort_block_m=128`, pass `s2_blocks = n_sorted // 64`, which is `2*m_blocks`. **The plan's fallback to stage-2 BM=128 is not needed.**

- At n > 1024 stage 2 is persistent (`persist_m=-1`). The tile count then comes from `num_valid` (`6811-6824`), so `size_expert_ids_in` only sets the buffer bounds.
- If you pass `m_blocks` by mistake, rows past the halfway point are read out of bounds. The hardware returns 0 for those reads, so the weight is 0 and those routes are **silently dropped** rather than duplicated.

### 3. Do the sort (Step 2) first, or limit Step 1 to a fixed time budget

- **Step 2's gain is nearly certain.** It removes about 40–380 µs of single-CU time per large Point, and it does not depend on the stage-1 occupancy question.
- **Step 1 is now a real race with an uncertain sign.** Running it first, with its plumbing and the per-Point table, risks leaving the sure gain undone if the race drags on.
- **Either order works, provided:** Step 1 gets a fixed budget, for example one build plus a `{wpe=2, None}` race at 3816 and 32768. If it does not clear the stop rule, revert `_config` and go straight to Step 2.

### 4. Padding waste at 10000 and 13152

At sort_m=128, an expert with about 312 routes pads to 384 rows instead of 320:
- stage 1 does 3 sweeps instead of 5;
- stage 2 at tile 64 runs one extra, entirely padding, tile per expert.

I did not confirm whether stage 2 skips work on such tiles. The per-Point race already covers this, but the session record should state the stage-2 time at those two Points next to the stage-1 time.

## Smaller points (do not block execution)

- **K2 design.** It is legal: no polling, no counter, and `hist_g` is fully written, so `torch.empty` is fine. Still:
  - `topk_ids` is read as i32 (`8820`), so the dwordx4 load of four routes per token is valid. Check that `topk_ids` is contiguous.
  - At G=128, the unrolled 128-iteration `range_constexpr` loop over `hist_g` is fine. Time K1+K2 on their own, since 6–12 µs is an estimate.
- **Step 3 (S-regime sort micro-optimizations).** The ≥2 µs keep threshold is right. Let it ride only on a measured driver bench at n=256 and n=640, so that the 1.03 serving gate is not put at risk.
- **Correctness guard.** The plan's rule of a full driver correctness run on all five large Points after each step is correct. The 32768 error (6.5–7.0e-4) sits close to the limit, and the stage-2 grid will change.
- **Expected mean.** 1.02–1.03 assumed Step 1's gain. Without it, Step 2 alone gives roughly 0.988 → about 1.00 (algorithm specialist's figure). Record this as the conservative outcome, so that a neutral Step 1 is not read as the route failing.

## What to change before executing

1. Replace the "≈81 KB, 2 WG/CU" claim with about 97 KB, 1 WG/CU (4 waves). Add the descriptor check and the stage-1 ≤ 220 µs stop rule at 3816.
2. Set `s2_blocks = n_sorted // s2_tile_m` and drop the stage-2 BM=128 fallback.
3. Put Step 2 first, or give Step 1 an explicit time budget and revert path.
4. Lower the Step 1 expectations to "race, sign unknown". Keep the S path byte-identical, as the plan already says.
