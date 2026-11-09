# Review of the iteration-5 plan: stage-2 non-temporal B loads, x-quant load hoist, fused sort, stage-1 XCD swizzle

VERDICT: REVISE

**Bottom line:** the plan aims at the right Points and the right levers, so the route stands. Two things need fixing before execution:
- The per-step gate fails on the incumbent itself, so as written it blocks every step.
- Step 4 has a load-balance flaw that the plan does not see.

Step 1 can start as soon as the gate is reworded.

## The route is right

- **The plan works on the Points that weigh most.** From `run_state.json` (`baseline_case_times` / `best_case_times`):
  - The 20 Points with n ≤ 640 are at 1.0275–1.055×.
  - The large Points have more margin than the dispatch implies: 3112 is at 1.052×, 3816 at 1.050×, 10000 at 1.137×, 13152 at 1.055× and 32768 at 1.070×.
  - So in EXPLOIT mode, putting the small Points first is correct.
- **The stage-1 roof argument holds.** Stage 1 moves 642 MB in 100.9 µs (iter_003 §3). Every stage-1 knob tried at tile_m=32 lost (`lessons/iter_004.md` §2).
- **Step 1's premise is confirmed in source:**
  - The stage-2 builder does `del b_nt` (`kernel.py:6348`).
  - Its routed `load_cell` (`7146-7155`) passes no `cache_modifier`. Stage 1 does (`3877`).
  - `_compiled` hard-codes `b_nt=0` for stage 2 (`9099`).
  - At small n, `s2_tm == tile_m == 32` (`8626`), so each expert has one block and W2 is streamed once. The plan's argument from stage 1 (2 vs 0 was worth 14 µs) and from aiter's `_bnt2` choice carries over.
  - This is the best ratio of gain to effort in the plan.
- **Step 2's iteration count is right.** `iters = 768/256 = 3` (`8642`). The plan correctly overrides the specialist's figure of 6.

## Problems that change the plan

### 1. The gate rejects the incumbent itself

The gate says "every Point with n ≤ 640 stays at ≥ 1.03×". The incumbent fails that today:
- n=256 is at 0.17918 / 0.174381 = **1.0275×**.
- n=272 is at 1.0305×, and n=320, 336 and 640 are at about 1.033–1.034×.

Read literally, the gate cannot pass unless a step also lifts n=256, and the wording is ambiguous.

Replace it with:
- **(a)** no Point that is ≥ 1.03× today drops below 1.03×;
- **(b)** no Point regresses by more than the measurement noise (state a tolerance, for example 0.5%, or require a rerun before calling a regression);
- **(c)** `allclose: True`.

Lifting n=256 above 1.03 is an explicit secondary goal of Step 1.

### 2. Step 4: the existing XCD remap gives contiguous ranges, and the stage-1 grid is an upper bound

The remap (`3440-3447`) gives XCD k the contiguous range `[k·wpx, (k+1)·wpx)` of `workgroup_id`. Its `grid_y` is `size_expert_ids_in`, which is `s1_blocks = m_blocks`. That count comes from `max_sorted = R + E·(tile_m−1)` (`9091-9093`), not from the real block count.

At t128, the trailing blocks past `num_valid` are a large share of the grid:

| n | m_blocks | Real blocks (estimate) | Empty tail |
|---|---|---|---|
| 3112 | 225 | about 160 | about 25–30% |
| 10000 | 440 | about 377 | about 14% |
| 32768 | 1151 | about 1088 | about 5% |

These figures are argued: about 64 padding rows per expert on average.

- **With the contiguous mapping, all those early-exit workgroups land on the last one or two XCDs.** The real work then crowds onto the remaining XCDs. Hardware dispatch to XCDs is static round-robin, so no XCD can take work from another.
- **The likely result is a slower stage 1:**
  - about 1.06× at 32768;
  - about 1.15–1.4× at 3112, 3816 and 10000.

  Either would wipe out the 0–7% L2 gain the plan expects. The race would catch it, but only after the build and timing time is spent.
- **"Bit-identical" makes the step numerically safe, not performance-safe.**

**Required revision:**
- Change the mapping, not only guard the remainder. Distribute GROUP_M super-groups round-robin across XCDs: super-group g goes to XCD g mod 8. That keeps an expert's m-blocks together and spreads the empty tail evenly.
- Keep the remainder guard for the bijection; the plan's `linear_id < 8·(nwg//8)` fix is correct.
- Race against swizzle=0 at all five large Points, timing stage 1 alone, and record the per-swizzle times.
- If no setting gains at least 3% at 32768, drop the step.

### 3. Step 3: correct by construction, but name the conditions that end it

The step is legal by the ABI table: no inter-workgroup waits, and every buffer is rewritten on each call. The fill value `(topk<<24)|n_tokens` with weight 0 (`8752`, `8858`) and the tail clamp to `experts−1` (`8880`) are reproduced correctly. Two additions:
- **Make the time box concrete.** If x-quant at n=640 grows by more than about 6 µs, the net gain is under about 3 µs. In that case take fallback 3b instead of debugging further.
  - Per-workgroup latency now sits on a serial chain: loads, then histogram, barrier, rank reduction, barrier, prefix, barrier, then stores.
  - At n=640 the grid runs about 2.5 rounds over 256 CUs, so that chain is paid 3 times.
  - Use the wave scan for the prefix, not the 128-iteration loop (`8826-8832`).
- **State the reference for the equivalence check.** The order within each expert changes from atomic arrival order to token order. The multiset check per block is the right test, and the plan has it. The driver run also needs `allclose` at n=640, because the stage-2 atomic order changes.

### 4. Step 2: dump the ISA before editing

The serialization claim is a guess ("LLVM probably cannot reorder"). Dump the current x-quant ISA first; it is cheap. If the three `buffer_load_dwordx4` are already grouped under one `vmcnt`, skip the edit and do not log it as a measured negative.

## Smaller points

- **Step 1 B-load paths.** At small n, `use_async_copy=False` and `r139_xdma_first` is off (it needs async and tile_m 64, `6355-6363`), so `load_cell` is the only routed-W2 path that matters here. Still grep the builder for other `_buffer_load_vec` calls on `w_rsrc`.
- **Step 1 cache key.** Adding `_bnt{b_nt}` to the stage-2 kernel name is correct. Clear the cache before every timing, as the plan says.
- **Projection.** The projected 1.09–1.10 adds three argued gains together. If Step 1 alone delivers (about +3%), the conservative outcome is about 1.07. Record that, so that a neutral Step 3 is not read as a failure of the route.
- **Step 5.** Leave it optional, with the descriptor check first. Reaching it this iteration is unlikely.
- **Rejections.** The rejected list is sound, notably stage-1 tile_n=64 at small n, and stage-2 XCD swizzle because of the atomic order and the error budget.

## What to change before executing

1. Reword the gate as in problem 1, with a noise tolerance, so it does not depend on n=256 already being above 1.03.
2. In Step 4, change the XCD mapping so super-groups go round-robin across XCDs (or justify the contiguous ranges given the upper-bound grid), and set the ≥ 3%-at-32768 drop rule.
3. In Step 3, set the stop rule: if x-quant grows by more than 6 µs at n=640, take fallback 3b.
4. In Step 2, dump the ISA before editing.
