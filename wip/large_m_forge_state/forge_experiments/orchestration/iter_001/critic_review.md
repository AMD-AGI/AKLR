# Critique: fused native-MXFP4 FlyDSL MoE plan (Direction: replace-the-torch-path-with-a-fused-flyd)

**VERDICT: REVISE**

The route is correct: a fused FlyDSL a4w4 MoE built on aiter's afp4_wfp4 stage kernels. Alternatives were checked properly. A bf16-dequant design loses at M and L (report §4.3). The stepping stones cap out below 0.05× (report §1, §6.5). The prior art `90eaf5c4` is numerically wrong at M ≥ 256 (report §4.3). The choice of `mfma_scale 16x16x128` is causally tied to the scale shuffle at `kernel.py:43-44`. The real gaps are the success gate, an expected gain that is wrong, and missing stop and switch conditions. Each of these would change what gets executed this iteration.

## 1. The success gate assumes the wrong incumbent (this changes the plan)

The plan frames the win as "0.016 → 0.8–1.1". But `run_state.json` says otherwise:
- `best_case_times` equals `baseline_case_times` exactly.
- `analysis_evidence.current_mean_case_speedup` and `mean_case_speedup_at_collection` are both 1.0.

So the incumbent the KEEP gate compares against looks like aiter's times, not the 16–63 ms torch path. If so, a first landing at 0.8–1.0 mean is **not kept**. The same goes for the "torch routing glue" fallback (about 0.4× at S, per the plan's own estimate).

What the plan must add:
- Confirm which time the KEEP gate divides by.
- State that the landing target is mean > 1.0 against `best_case_times`.

The arithmetic: S is 20 of 25 Points. If S lands at about 0.97, the 5 M/L Points must average at least about 1.12 to clear 1.0. The plan has to show where that comes from.

## 2. The claimed M-regime gain from "BM=128 at first landing" is mostly already in the baseline

- Report §4.3 says aiter already runs `t128x128x256` at 3816, under token key 4096. 3112 most likely falls in the same bucket.
- So the plan's line "BM=128 sweeps 1.15 GB against 2.11 GB at BM=64 (3816)" compares against a config the baseline does not run. A straight port at BM=128 is about 1.0× at M, not 1.3–1.6×.
- The only first-landing tile gain is at L. At 32768 aiter runs `t64x128x256_w4`. The configs at 10000 and 13152 are not recorded.
- Before claiming a gain anywhere, Step 0 must record **aiter's chosen config at every M/L Point**.

## 3. aiter's builder limits the "port, then race" knob space

`get_flydsl_stage1_kernels` (`/opt/venv/.../aiter/ops/flydsl/moe_kernels.py:200-214`) sets these limits:
- With fp4 A, `tile_m ∈ {32, 64, 128}`.
- `tile_n ∈ {32, 64, 128}` at tm=32, and `{64, 128}` otherwise.
- Stage 2 also caps at tm=128 (`:299`).

What this means for the plan:
- BM=160, 192 or 256 (Phase B, and the "race 256 later" at L) is **new main-loop code**, not a knob on the port.
- The algorithm specialist's "stage-1 tile_n 256 is legal" is not available from aiter for fp4 A.

The plan should say which Phase B items are knobs of the ported kernel and which are new kernels, and cost them separately. The ported kernel's knobs are `k_wave`, `waves_per_eu`, `b_nt`, `tile_n`, `persist`, and `atomic` vs `reduce`. The cheap way to win at M and L this round is an offline race over these knobs. Where aiter was on an untuned heuristic (report §4.3), this race is the realistic source of the 1.12 M/L average from §1.

## 4. The correctness gate is stricter than the contract and risks burning the iteration

- Step 1 requires the dequantized x to **equal** `kernel._mxfp4(x)`, with integer RNE as the fallback. But the `out` bound is `BaselineRelativeBound` (report §4.4): it scales with aiter's own error, and aiter quantizes with the hardware cvt path.
- Bit-parity with `_mxfp4` is a good diagnostic. It should not block landing.
- **Primary gate:** the `validate` violation fraction (≤ 0.001) on all 25 Points, plus the run-to-run bounded determinism check.
- **Secondary gate:** a bit-exact x/y match.
- Do not switch to the slower integer RNE path unless the primary gate fails. Otherwise the plan can pay a doubled stage-1 epilogue for no scoring reason.

## 5. Missing stop and route-switch conditions

These need to be written down explicitly:
- **Step 0 result: no `has_swiglu_limit` / 1.702 / `+1` epilogue in aiter's stage 1.** Then the epilogue is new code. Budget it, and say whether stage 1 stays bf16-out with a separate y-quant node or gets the fused epilogue.
- **K0 (the custom 128-WG redundant-scan router) is the most novel part.** It is not a port: aiter's sort is the HIP/opus JIT `module_moe_sorting_opus` (report §2), not FlyDSL.
  - Time-box it.
  - If it slips, the documented switch should be a simpler single-pass FlyDSL histogram/scan kernel that is correct but slower. It should not be torch glue, which cannot be kept under §1.
  - Also keep the encoding contract (packed topk slot, `num_valid_ids`, and the sorted, shuffled x-scale layout that `moe_mxfp4_sort` produces) as a written spec checked against aiter's stage-kernel reads. This handoff is where silent bugs live.
- **KEEP and stop criteria after landing:**
  - Mean > 1.0 against `best_case_times`, all 25 Points correct.
  - If the mean is below 1.0 with S below 0.95, stop tuning M/L and inspect the S node count and stream rate first.
  - Get aiter's per-kernel Kineto split at n=256 and 640. The plan defers this to Phase C, but it is cheap and decides how the S margin is spent, so it belongs in Step 0.

## 6. Smaller points

- In `run`, `swiglu_limit` arrives as a scalar keyword. Step 0 item 6 is right; just pass it as a kernel scalar argument, with no `.item()`.
- At 32768, an atomic `out` is 402 MB, below the 32-bit offset limit (`requires_flydsl_stage2_reduce`, `moe_kernels.py:112-116`). The atomic path is valid at every Point.
- The plan rightly keeps `xcd_swizzle=0` until the remap and validity-test order is fixed.
- The plan rightly rejects fixed-order reduce at S.

## What the revised plan should contain

1. A KEEP target that matches the actual incumbent.
2. Step 0 extended with aiter's config at every M/L Point, its per-kernel split at S, and whether the swiglu-limit epilogue exists.
3. A landing built from a faithful port plus a knob race inside aiter's knob space where aiter was untuned.
4. A gate on the validate bound, not on `_mxfp4` bit-parity.
5. A time-boxed K0 with a FlyDSL fallback, not a torch one.
6. BM > 128 and the new main loop deferred, and labelled as new kernels.
