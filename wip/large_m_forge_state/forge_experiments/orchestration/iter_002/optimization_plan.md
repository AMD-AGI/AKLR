Direction: replace-the-torch-path-with-a-fused-flyd

# Iteration 2 plan: land an inline fused FlyDSL a4w4 MoE in `kernel.py`

## 0. Ground truth this plan is built on

1. **The incumbent is the torch path. Any correct landing on all 25 Points is a KEEP.**
   - In `run_state.json`, `best_case_times` are the torch times: 16.1–16.8 ms at S, 22.1–22.9 ms at M, 31.0–62.7 ms at L.
   - `best.mean_case_speedup` is `null` and `cumulative.kept` is 0. The baseline (aiter) is 0.179–2.457 ms.
   - The score today is about 0.0145, so landing correctly comes before tuning.
   - Beating aiter (mean above 1.0, and 1.03 per Point to be served) is the goal for later iterations. It is not this KEEP's bar.
2. **Only `kernel.py` may change.**
   - Iteration 1 was reverted because it added `_moe_*.py` files (`lessons/iter_001.md:8-9`). Those files no longer exist, so the port is rebuilt from aiter's source and inlined.
   - Before finishing, `git status --short` must list only `kernel.py`. Keep scratch scripts and backups in `/tmp`.
   - Import nothing from aiter at runtime.
3. **Iteration 1's "passed correctness" proves less than it seems.**
   - The lesson file was machine-written and is truncated. It does not say whether iteration 1's sort and quant were aiter's runtime-JIT modules (`module_moe_sorting_opus`, `module_quant`).
   - At most, it shows that aiter's stage-1/stage-2 FlyDSL kernels can pass this contract's bound.
   - Treat the sort, quant and handoff layout as unverified.
4. **What the current source actually contains** (checked in `kernel.py`):
   - `_mxfp4` (`kernel.py:62-80`) returns dequantized bf16 (`(codes*scale).to(x.dtype)`). It does not return packed e2m1 bytes and E8M0 exponents, so it cannot feed the fp4 MFMA.
   - The torch sort at `kernel.py:113-118` (`scatter_add`, stable `argsort`, `cumsum`, rank) is already sync-free. It is the fastest starting point for routing.
5. **`swiglu_limit` arrives as a Python float.** aiter's `_s1_args_fp4` passes `float(swiglu_limit)` as a runtime scalar (`aiter/ops/flydsl/moe_kernels.py:800-868`, as read by the specialist). The clamped-SwiGLU epilogue is a port, not new code. No `.item()`, and the value is not part of the compile key.

## 1. Step 0: read before writing (source reading only, no timing)

These reads decide the handoff spec, which is where iteration 1's 17% error came from.

1. **Read the a4w4 FlyDSL branch of aiter's `fused_moe.py`** and record:
   - the value it passes for `v2_output_layout`:
     - `False` (the default, `moe_kernels.py:1459`): fp4 y is token-major `(token_num, topk, inter/2)` (`:1588-1591`).
     - `True`: y is in sorted-row order `(_sorted_rows, inter/2)` (`:1576-1587`).
   - which kernel quantizes x and lays out its scales in sorted order;
   - how the x-scale buffer is sized (`ceil(max_sorted/32)·32·(D/32)·2` bytes per the memory specialist, `fused_moe.py:1610-1614`).

   Port stage 1 and stage 2 under that same `v2_output_layout` value. Remove the other branch only after confirming it is unused. The likely cause of the iteration-1 17% error is that stage 1 wrote with one setting while stage 2 read with the other (inference).
2. **Write the handoff spec as a comment block in `kernel.py`** and check every read in the ported stage kernels against it:
   - `sorted_ids`: packed `(slot<<24)|token`, with padding entries `(topk<<24)|M`.
   - `sorted_weights`.
   - one `sorted_expert_ids` entry per sort block.
   - `num_valid_ids`: the padded total row count.
   - x codes: token order, read by stage 1 through `sorted_ids`. Confirm this against stage 1's A addressing.
   - x scales: sorted-row order, in the shuffled `[rows/32][K/256][4][16][2][2]` E8M0 layout, padded to (256, 8).
   - y and its `out_scale_sorted` layout under the chosen `v2_output_layout`.
   - The sort block, stage-1 `tile_m` and stage-2 `tile_m` relationship (see §3).
3. **Open the AKLR prior art for this contract family** (`cases/fused_moe_shuffled_16x16.md`). These are single-file FlyDSL `kernel.py` implementations of `fused_moe@1.0` with the same `shuffled_16x16` layout, model_dim 6144, MXFP4, `act_quant` 4:

   | Source | What to lift |
   |---|---|
   | `ef15e878…` (33 experts) | `_route_expert`: one workgroup per expert compacts its routes in input order (deterministic) while the rest of the grid quantizes x in the same launch. Its MXFP4 quant routine (lane amax → E8M0 → `cvt_scalef32_pk_fp4_*`). `_chiplet_order` for later. |
   | `1bf41ccb…` (65 experts) | `collect` / `compact`: ballot ranks in input order and expert-major padded blocks. |
   | `4fbd9d02…` `g2_port.py` | An aiter `mxfp4_gemm2` port with kernarg preload and the expert id read before the A loads. Reference only, this iteration. |

   Path: `/home/ruibzhan/AKLR-wt-minimax-moe/aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json` and its sibling directories.

   Do **not** lift:
   - their stage-2 fp8 rows, bf16 truncation (`bits >> 16`) or reduce kernels. Those passed an SQNR ≥ 13 dB bound and will not pass this contract's `BaselineRelativeBound`;
   - their module-level `_RUN_CACHE` dict, their scratch allocated on `"cuda"`, or `Path(__file__).read_bytes()`. These break the AKLR ABI. Use a `functools.cache` compile function and allocate on the argument's device.

   The stage GEMMs stay the aiter port, because that is what reproduces aiter's rounding.

## 2. Checkpoint A: the fastest correct landing

Build in this order. After each piece, run `validate` on a few Points (256, 640, 3816, 32768) before moving on.

1. **Sort: torch, sync-free.** Adapt `kernel.py:113-118` to aiter's encoding:
   - Fixed upper bound `max_sorted = R + E·(sort_block − 1)`, so no size ever leaves the device.
   - Padded per-expert counts, then `cumsum`, then each route's sorted position.
   - Fill `sorted_ids` with the padding sentinel, then scatter the real entries.
   - `sorted_expert_ids` per block via `searchsorted` over the block ends (as `kernel.py:135` does).
   - `num_valid_ids`, plus the inverse map `pos[token, slot]`.
   - No `bincount`, `.item()`, `nonzero` or other host sync.
   - Kernels launch on the upper-bound grid and exit past `num_valid_ids`.
2. **One FlyDSL MXFP4 quant kernel, used twice:**
   - for x, reading bf16 rows and writing fp4 codes in token order;
   - for y, from the bf16-out stage 1, writing codes in the layout stage 2 reads.

   Each use writes the E8M0 scales straight into the sorted, shuffled, padded scale layout, using `pos` for x. Rules:
   - Lift the quant routine from `ef15e878…` and check its exponent and rounding against aiter's quant (Step 0.1). Match aiter's rounding; do not "improve" it.
   - Write every padding scale byte (e.g. 127) and every padding row. `torch.empty` scratch may hold 0xFF, which E8M0 decodes as NaN.
   - Send padding A rows out of range so the buffer bound returns zero.
3. **Stage 1:** ported `compile_mixed_moe_gemm1_common`, a4w4, `gate_mode` separated, scaled-fp4 MFMA 16x16x128 straight on the contract's preshuffled w1 and w1_scale (no repack). Use **bf16 out** for this checkpoint.
4. **Stage 2:** ported `compile_mixed_moe_gemm2_common`, atomic, with the routing weight applied in stage 2. Write into `out`, which is zeroed in the same call with `out.zero_()` for now.
   - Atomic is legal at every Point: 32768·6144·2 B < 4 GiB (`requires_flydsl_stage2_reduce`, `moe_kernels.py:112-116`).
   - Determinism is `bounded`, so bf16 atomics are allowed.
5. **Strip** the fp8, a16w4, split-K, gate-up-interleave and bias branches, and the unused `v2_output_layout` branch (once Step 0.1 confirms it is unused). This keeps first-call compile time manageable.
6. **Configs:** aiter's heuristics, chosen from `n_tokens` in a module-level `@functools.cache` compile function. Use `xcd_swizzle = 0` everywhere.

   | Points | Stage 1 | Stage 2 |
   |---|---|---|
   | n ≤ 640 | `t32x128x256_w2` | atomic, b_nt = 2 |
   | 3112 / 3816 | `t128x128x256` | atomic |
   | 10000–32768 | `t64x128x256_w4` | atomic |

   The stage-2 tile_m must equal the sort block unless Step 0.2 shows otherwise (§3).

**A is done when all of these hold:**
- `validate` passes all 25 Points: violation fraction ≤ 0.001 plus the bounded run-to-run check.
- `git status` lists only `kernel.py`.
- There is no runtime aiter import, and no sync, env read or timing in `run`.

Then **immediately** copy this `kernel.py` to `/tmp`, run the driver and record per-case times. This is the version to submit if anything later fails. Iteration 1's round ended in `sdk_error`, so do not leave the landing unsaved.

**Debugging aids** (they do not decide acceptance):
- On one Point, in a scratch script under `/tmp`, dequantize x codes and scales through stage 1's indexing, and y and its scales through stage 2's indexing. Compare against a torch reference.
- Bit-parity with a codes-emitting variant of `_mxfp4`.
- Switch to a different rounding path only if the primary gate fails.

## 3. Sort block and tile_m are coupled

One sort serves both stages, so the sort block, stage-1 `tile_m` and stage-2 `tile_m` are not independent. Use the same value for all three unless the ported stage-2 source shows that a stage-2 `tile_m` dividing the sort block decodes `sorted_expert_ids` and `num_valid_ids` correctly.

Any race in §5 must therefore vary the pair (s1 tm, s2 tm) jointly under one sort block. It must count stage 2's cost at the forced tile_m, e.g. tm = 128 at L if stage 1 moves to 128.

## 4. Checkpoint B: remove glue and nodes (only after A is saved)

Each step is kept only if `validate` still passes all 25 Points and S time improves. **Stop rule:** if a step breaks `validate` after one check against the handoff spec, revert to A and move on.

1. **K0, a FlyDSL router, replaces the torch sort.** Either adapt `ef15e878…`'s `_route_expert` (one workgroup per expert, input-order compaction, with the rest of the grid quantizing x in the same launch), or write a single 1024-thread workgroup that does an LDS histogram, a padded exclusive scan and an LDS-atomic rank scatter.
   - **Switch rule:** if adapting the lifted router to aiter's packed encoding is taking longer than writing the single-workgroup version, write the single-workgroup version.
   - Either way:
     - counters live in LDS or are rewritten every call;
     - no workgroup waits on another;
     - the x-scale scatter to sorted positions must not depend on another workgroup's output in the same launch. If quant and route share a launch, the routing workgroup recomputes the E8M0 from `hidden_states` (amax is cheap), or the scale scatter goes in the next launch.
2. **Fold the `out` zeroing into the quant (or route) launch**, writing to memory disjoint from everything else that launch writes.
3. **Fused fp4-out stage-1 epilogue** (aiter's `out_dtype="fp4"` path) under the Step 0.1 `v2_output_layout` value, writing scales in the spec's padded sorted layout. This removes the y-quant node.
   - If it does not validate after one check against the spec, keep bf16-out plus the FlyDSL y quant. At M/L that split costs about 1–3% (argued).

Target: 4 launches (route+quant, zero folded in, stage 1, stage 2), or 5 if the scale scatter needs its own launch. Making stage 1 read token-order x scales so route and quant fully merge changes the main loop's A-scale load; that waits for a later iteration.

## 5. Checkpoint C: measure aiter, then race offline (only after B is saved)

1. **Kineto on the baseline, in a scratch script:**
   - aiter's kernel names and configs at 3112, 10000 and 13152;
   - aiter's per-kernel split at 256, 640, 3816 and 32768.

   This tells you how many nodes aiter runs at S and which stage dominates at L.
2. **Offline race, then a hard-coded per-case table** in the compile function. Use only the port's knobs: stage-1 tile_n ∈ {32, 64, 128} at tm = 32 and {64, 128} otherwise; wpe 1–4; b_nt {0, 2}; k_wave only at tm = 32; stage-2 tile_n {128, 256}, tile_k {128, 256}, persist. Tile_m is raced jointly per §3. Priority:
   - **L:** the (128, 128) tile_m pair against aiter's 64. The argued mechanism is halving weight sweeps (about 15.9 → 8 GB at 32768); it is not measured. Check spills and VGPR/AGPR use with `FLYDSL_DUMP_IR=1` before trusting a result. Also race b_nt = 0 and the stage-2 `persist_async_w4_cumul3` variant.
   - **S:** stage-1 {tn128, kw1} against {tn64, kw2/kw4} × wpe, with b_nt = 2 (parity with aiter). Expect 0–3%.
   - **M:** wpe, b_nt and stage-2 tile_n only. Expect about 1.0–1.1×.

   Split configs per case. Never average a config that wins on one L Point and loses on another.
3. **Stop rule:** if S is below about 0.97× aiter, stop tuning L and compare per-kernel times against aiter at n = 512 first (node count, b_nt, stage-2 tile).

## 6. Rejected or deferred

| Idea | Decision | Why |
|---|---|---|
| Torch `_mxfp4` as the A quant | Reject | It emits bf16, not codes and scales, and costs about 0.8 ms at S and 23.6 ms at L (report §4.1). |
| Stepping stones (faster dequant or torch quant) | Reject | Capped well below the port. |
| `90eaf5c4` / bf16-dequant MFMA | Reject | Numerically wrong at M ≥ 256 and slower at M/L (report §4.3). |
| Prior art's fp8 or truncated stage-2 rows and reduce | Reject | They fit an SQNR bound, not `BaselineRelativeBound`. |
| Fixed-order reduce instead of atomics | Reject | Extra node and traffic; atomics are allowed. |
| Fusing stage 1 and stage 2 into one kernel | Reject | Needs cross-workgroup waiting, which AKLR forbids. |
| `xcd_swizzle = 4`, `_chiplet_order` | Defer | Remap and validity-order risk; A/B on all M/L Points after C. |
| BM > 128, 32x32x64 MFMA, LDS ring, `s_setprio`, stage-2 whole-K B preload | Defer | New main-loop kernels, not knobs of the port. |

## 7. Risks

- **Compile time:** about 6–8 configs compile on first call. Strip unused branches.
- **Concurrent streams under `--stress`:** per-call `torch.empty` scratch, no module-level tensors, no state across calls.
- **One lost workload rejects the candidate:** re-validate all 25 Points after every layout or config change.

## 8. Expected outcome (argued, not measured)

| After | Mean vs aiter | Note |
|---|---|---|
| A (torch sort, FlyDSL quant ×2, bf16-out stage 1, `zero_`) | about 0.6–0.9 at S (several torch sort launches plus 2 extra nodes against a ~180 µs budget); about 0.85–1.0 at M/L | Already about 40–60× over the incumbent. This is the KEEP. |
| B (FlyDSL route, folded zeroing, fp4-out epilogue) | about 0.95–1.01 | |
| C (Kineto plus joint race) | about 1.0–1.07, mostly from L | Upside only. Never risk the saved A or B for it. |

If A ends up with leftover torch glue on some path, it is still worth submitting. A result anywhere from about 0.15× aiter upward far exceeds today's 0.0145.
