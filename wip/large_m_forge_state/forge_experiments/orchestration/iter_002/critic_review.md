# Critique of the iteration-2 plan: fused FlyDSL a4w4 MoE ported inline from aiter

**VERDICT: REVISE**

The route is right. Porting aiter's afp4_wfp4 stage kernels inline into `kernel.py` is the only direction with first-order impact (report §1, §6.1). The draft also fixes the last critic's main error. Several concrete defects would still cost this iteration time or mislead it. All of them can be fixed without leaving the route.

## What the draft gets right (checked)

- **The incumbent really is the torch path.** In `run_state.json`:
  - `best_case_times` are 16.1–62.7 ms, the torch times.
  - `best.mean_case_speedup` is `null`.
  - `cumulative.kept` is 0.
  - `baseline_case_times` are the aiter times (0.179–2.457 ms).

  The previous critique's §1 was wrong, and the draft's step 0.1 is correct. Any correct landing on all 25 Points is a KEEP.
- **Iteration 1 was reverted for adding files, not for wrong output.** `lessons/iter_001.md:8-9`. Keeping everything inside `kernel.py` and checking `git status` is the right constraint. The iteration-1 files are gone; `workspace/_moe_kernels_common.py` does not exist. So the port has to be rebuilt from aiter source.
- The gate is the validate bound, not bit-parity, and atomics are allowed because determinism is bounded. Rejecting the stepping stones, `90eaf5c4`, and BM > 128 is correct.

## Defects that change what gets executed

### 1. In Checkpoint A, the torch `_mxfp4` cannot feed the fp4 MFMA, and the expected numbers are wrong

- `_mxfp4` (`kernel.py:62-80`) returns **dequantized bf16** (`(codes*scale).to(x.dtype)`). It does not return packed e2m1 bytes and E8M0 exponents. Using it as A's x quant means first writing a different torch routine that emits codes and scales.
- Torch quant is also expensive:
  - About 0.8 ms at n=256: 5% of 16.1 ms (report §4.1).
  - About 23.6 ms at n=32768: 37.6% of 62.75 ms.
  - A torch y-quant after a bf16-out stage 1 adds a similar cost at L.
- With torch quant, Checkpoint A lands near **0.15–0.2× aiter at S and about 0.1× at L**, not the 0.8–1.0 in the §4 table.

**Fix:**
- Put one FlyDSL MXFP4 quant kernel into Checkpoint A. It should write codes plus E8M0 into the sorted, shuffled scale layout. Use it for both x and the bf16-out y. It is an elementwise kernel, no harder than a correct torch codes/scales emitter.
- Correct the §4 expectation table.
- B then reduces to fusing the `out` zeroing and moving to the fp4-out epilogue.

### 2. Existing single-file AKLR prior art for this contract family was not searched

The KB case `cases/fused_moe_shuffled_16x16.md` lists FlyDSL kernels for `fused_moe@1.0` with the same `shuffled_16x16` layout, model_dim 6144, MXFP4 and act_quant 4. Each one embeds the whole pipeline in a single `kernel.py`, which is exactly the form the workspace gate requires.

- **`ef15e878…`** (33 experts): served at m 64–4096 at 1.03–1.61× over aiter.
  - At m ≥ 64, `_route_expert` compacts each expert's routes in input order, which is deterministic. The rest of the grid quantizes the input in the same launch: the fused sort+quant that the draft defers.
  - At m 2–32, `_quantize_route_fused` takes ballot ranks.
  - It also provides `_chiplet_order`.
- **`1bf41ccb…`** (65 experts): has `collect`/`compact` with LDS-atomic ranks.
- **`4fbd9d02…` `g2_port.py`**: an aiter `mxfp4_gemm2` port with kernarg preload and the expert id read before the A loads.

The path exists: `/home/ruibzhan/AKLR-wt-minimax-moe/aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts33_topk9_inter_dim2048/ef15e87894a8ae06.json`.

**What to change:** before writing K0 and K1 from scratch, lift their route and quant kernels, then adapt the output to aiter's packed `sorted_ids` encoding and sorted-scale layout.

**What not to lift:** their stage-2 fp8 rows and bf16 truncation. Those passed an SQNR ≥ 13 dB bound. They will not pass this contract's `BaselineRelativeBound` (report §4.4).

The stage kernels should stay the aiter port, since that is what reproduces aiter's rounding.

### 3. The y-layout dispute depends on a flag the draft strips

In `aiter/ops/flydsl/moe_kernels.py`:
- With `v2_output_layout=False`, the default at `:1459`, fp4 y is token-major `(token_num, topk, inter/2)` (`:1588-1591`).
- With `v2_output_layout=True`, y is **sorted-row** `(_sorted_rows, inter/2)` (`:1576-1587`), which is the memory specialist's version.

The iteration-1 "17% error, y layout differs" is very plausibly a mismatch of this flag between the stage-1 output and the stage-2 read.

**Fix:**
- Read what aiter's `fused_moe.py` a4w4 branch passes for `v2_output_layout` and for the x-quant/scale-sort kernel.
- Port stage 1 and stage 2 under that same value.
- Write the chosen value into the handoff spec.

"Strip v2 layout" is only safe once this is confirmed.

### 4. Iteration 1's "passed correctness" is weaker evidence than the draft treats it

The lesson file is machine-written and truncated ("No summarizer session was available"). It does not say whether iteration 1's sort and quant came from aiter's runtime-JIT `module_moe_sorting_opus` / `module_quant`. Those are the modules report §2 says the baseline builds. If they did, the pass validated the stage kernels only, not a pipeline without aiter. Do not count the sort/quant handoff as de-risked.

### 5. The Checkpoint C race contradicts the draft's own handoff rule

- §1 says "Stage 2 must use the same sort block (= stage-1 `tile_m`)".
- Checkpoint C races stage-2 tile_m over {16, 32, 64, 128} independently of stage 1.
- One sort serves both stages, so (s1 tm, s2 tm) must be raced jointly under one sort block. The alternative is to show from the stage-2 source that a sub-multiple tile_m is legal.

This also limits the "L: stage-1 tm=128" lever. It forces stage 2 to tm=128 at L, and that cost has to be included in the race.

### 6. Session-time risk: order the work so the landing comes first

- Iteration 1's whole round was about 888 s and ended in `sdk_error` (`run_state.json` `round_costs`).
- Any correct landing is a KEEP, so the fastest correct path should come first.
- Recommended order for A:
  1. The sync-free torch sort already in `kernel.py:113-118`, adapted to aiter's encoding.
  2. The FlyDSL quant kernel.
  3. The inline stage-1/stage-2 port.
- Promote K0 in B. This reverses the draft's "K0 first, torch after a quarter-session".
- Commit or save the A version as soon as it validates on all 25 Points.

## Required success, stop and switch conditions

- **A succeeds when:** validate passes all 25 Points (violation fraction ≤ 0.001, bounded determinism), `git status` shows only `kernel.py`, and there are no runtime aiter imports. Expect about 0.15–1.0× aiter depending on how much glue remains.
- **Stop B** if any fused step breaks validate after one layout-spec check. Keep A.
- **Run C only after B is banked.** If S is below 0.97× aiter, investigate S before tuning L (the draft already says this).
- **Switch rule for K0/K1:** if adapting the lifted `ef15e878` route/quant to aiter's encoding takes longer than writing the simple single-workgroup K0, use K0. Torch glue is acceptable only inside A.

Smaller points that are fine as written: atomic legality at 32768, `xcd_swizzle=0`, `swiglu_limit` passed as a float scalar argument, and the NaN-padding handling.
