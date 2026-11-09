Direction: replace-the-torch-path-with-a-fused-flyd

# Plan for iteration 2: land the inlined fused FlyDSL a4w4 MoE, then tune it inside the port's knob space

## 0. Decisions that change last round's framing

1. **The incumbent to beat is the torch path, not aiter.**
   - The critic's §1 assumed that `best_case_times` equals the baseline times. `run_state.json` says otherwise:
     - `best_case_times` holds the torch times (16.1–16.8 ms at S, 22–23 ms at M, 31–63 ms at L).
     - `best.mean_case_speedup` is `null`, and `cumulative.kept` is 0.
   - So the score today is about 0.0145 (baseline over candidate). A correct port that lands at about 0.8–1.0× of aiter would raise it roughly 55–70×.
   - **This iteration's top priority is to land a correct port on all 25 Points.** The knob race comes second.
   - Beating 1.0 against aiter, and the 1.03 per-Point serving bar, remain the goal for later. They are not the bar for this KEEP.
   - In practice: do not drop a correct, mostly-glue landing because S comes in at 0.9× of aiter.
2. **Everything goes inside `kernel.py`.**
   - Iteration 1's port passed the driver's correctness suite. It was reverted only because it added `_moe_*.py` files (`lessons/iter_001.md:8-9,18`).
   - Before finishing, `git status --short` must show only `kernel.py` modified. Use scratch scripts outside the workspace (e.g. `/tmp`), or delete them before finishing, as iteration 1 did with `forge_experiments/probe`.
3. **The swiglu-limit epilogue already exists in aiter's stage 1.** `_s1_args_fp4` passes `float(swiglu_limit)` as a runtime scalar (`aiter/ops/flydsl/moe_kernels.py:800-868`, per the algorithm specialist). The epilogue is a port, not new code. Pass `swiglu_limit` as a Python float kernel argument: no `.item()`, and its value is not part of the compile key.

## 1. Landing sequence

Work in order. Get each checkpoint to validate before starting the next. Keep a copy of the last validated `kernel.py` outside the workspace so you can roll back.

### Checkpoint A: a correct end-to-end port

Port these from aiter: `compile_mixed_moe_gemm1_common` and `compile_mixed_moe_gemm2_common` (a4w4, gate_mode separated, scaled-fp4 MFMA 16x16x128 on the contract's 16x16-preshuffled w1/w2 and shuffled E8M0 scales, no repack).
- Strip the branches this contract never reaches: fp8, a16w4, split-K, gate-up interleave, bias, v2 layout. This keeps the file and the first-call compile time manageable.
- Do not import aiter at runtime.

Pipeline for checkpoint A:

| Step | What runs |
|---|---|
| Sort | The FlyDSL single-workgroup router K0 (see below). If it is not working within its time box, use sync-free torch glue (see below). |
| x quant | The existing torch `_mxfp4` in `kernel.py` is acceptable here, followed by a scatter of the x-scales into the sorted, shuffled layout. |
| Stage 1 | **bf16 out** (the configuration iteration 1 verified), then a separate y→MXFP4 quant. |
| Stage 2 | Atomic, routing weight applied in stage 2, into `out` zeroed in the same call. Atomic is legal at every Point: 32768·6144·2 B < 4 GiB. |

**K0, the sort.** One workgroup of 1024 threads.
- Build an LDS histogram over the 128 experts.
- Exclusive scan of the counts padded to `tile_m`.
- Scatter pass with LDS-atomic ranks.
- Outputs:
  - `sorted_ids`, packed `(slot<<24)|token`; a padding entry is `(topk<<24)|M`
  - `sorted_weights`
  - one `sorted_expert_ids` entry per block
  - `num_valid_ids`
  - the inverse map `pos[token, slot]`

Why this design:
- It is the simplest design that meets the stop table: no cross-workgroup waiting, and the counters live in LDS, so nothing carries across calls.
- Arrival-order ranks only permute rows inside an expert tile, so the result is bounded-deterministic.
- Estimated cost (not measured) is about 2–3 µs at S and 10–20 µs at 32768. The 128-workgroup redundant-scan router is not worth building.

**Torch fallback for the sort** (time box: if K0 is not producing correct `sorted_ids` after about a quarter of the session, switch to this):
- Stable argsort of the flattened `topk_ids`.
- Counts via `scatter_add_` into a zeroed tensor.
- `cumsum` of the padded counts, and index arithmetic.
- Do **not** use `torch.bincount` or anything else that syncs or calls `.item()`.

At S this costs latency, but it is still a huge gain over the incumbent, so it is an acceptable landing.

**The handoff spec: write it down and check every read against it.** This is where iteration 1's 17% error came from. The two specialists disagree on where stage 1 puts y:
- The memory specialist assumed y is in sorted-row order.
- The algorithm specialist cites aiter's code (`moe_kernels.py:1588-1591, 1634-1645`): y is **token-major `(token_num, topk, inter/2)` fp4x2**, while `out_scale_sorted` is in **sorted-row order, rows padded to 256 and columns padded to 8**.
- **Trust the source-cited version, and confirm it against the ported stage 2's own address math.**

Other handoff rules:
- Stage 2 must use the same sort block (= stage-1 `tile_m`) and the same padded scale pitch.
- Size the x-scale buffer as aiter does: `ceil(max_sorted/32)·32·(D/32)·2` bytes, copying the ×2 slack.

**Padding hazard.** Scratch from `torch.empty` can contain E8M0 0xFF, which decodes to NaN.
- Write every scale-padding byte before it is read (e.g. 127).
- Send padding A rows out of range so the buffer bound returns zero.
- Do not rely on NaN staying confined to rows that are skipped.

**Correctness gate.**
- **Primary (decides acceptance):** `validate` violation fraction ≤ 0.001 on all 25 Points, plus the bounded run-to-run determinism check.
- **Debugging aids only:** bit-parity with `_mxfp4`, and a scratch comparison of each intermediate (x bytes and scales, y, `out_scale_sorted`) against a torch reference dequantized through the stage-2 indexing on one Point. Use these to find a layout bug, not to decide acceptance. Do not switch to a slower rounding path unless the primary gate fails.

**Starting configs.** Use aiter's heuristics (report §4.3), so checkpoint A is about parity with aiter:

| Points | Stage 1 | Stage 2 |
|---|---|---|
| n ≤ 640 | `t32x128x256_w2` | atomic, b_nt = 2 |
| 3112 / 3816 | `t128x128x256` | |
| 10000–32768 | `t64x128x256_w4` | |

Further rules:
- Keep `xcd_swizzle = 0`.
- Select configs from `n_tokens` inside a module-level `@functools.cache` compile function. No env reads, timing or sync inside `run`.
- Call through the compiled handles, per shape.

At checkpoint A, run the driver, record per-case times, and make sure this version is what you would submit if nothing later works.

### Checkpoint B: replace the glue on the S-critical path

At S (80% of the mean weight), every remaining torch launch costs about 4+ µs against a budget of about 180 µs. In this order, keeping each step only if `validate` still passes and S time improves:
1. **K1: fused x→MXFP4 quant, x-scale scatter to the sorted positions via `pos`, and zeroing of `out`, in one FlyDSL launch.** Copy aiter's quant rounding; do not "improve" it. This replaces the ~20-pass torch `_mxfp4` and a separate `zero_`.
2. **Fused fp4-out stage-1 epilogue** (aiter's `out_dtype="fp4"` path), removing the separate y-quant node. Time box: if it does not validate quickly once the handoff spec above is enforced, keep bf16-out plus the y quant. At M/L that split costs only about 1–3%.

Target: 4 nodes (K0, K1, stage 1, stage 2). Folding K0 and K1 into a single launch needs stage 1 to read x-scales in token order through `sorted_ids`, which changes the main loop's A-scale load. That belongs to a later iteration.

### Checkpoint C: measure aiter, then race offline (only after B is banked)

1. **Step 0, using Kineto on the baseline (scratch only):**
   - aiter's kernel names and configs at 3112, 10000 and 13152;
   - aiter's per-kernel split at n = 256, 640, 3816 and 32768.

   This tells you how many nodes aiter runs at S, and whether stage 1 or stage 2 dominates at L.
2. **Race offline, then hard-code a per-case table** in the compile function. Only the port's real knobs are in play:
   - stage 1: tile_m ∈ {32, 64, 128}; tile_n ∈ {32, 64, 128} at tm = 32 and {64, 128} otherwise; wpe 1–4; b_nt {0, 2}; k_wave only at tm = 32;
   - stage 2: tile_m {16, 32, 64, 128}, tile_n {128, 256}, tile_k {128, 256}, persist, atomic.

   Priority order:
   - **L, stage-1 tile_m = 128** (aiter uses 64 at 32768). The argued mechanism is halving the weight sweeps, about 15.9 → 8 GB at 32768 (not measured). Expect 1.15–1.4× at 32768 and less at 10000 because of padding. Check the kernel descriptor for spills and VGPR/AGPR use (`FLYDSL_DUMP_IR=1`) before trusting a result: a 128×128 tile with gate and up accumulators may force wpe ≤ 2.
   - **b_nt per regime:**
     - S: keep b_nt = 2 (aiter already does this; it is parity, not a win).
     - L: race b_nt = 0 (cached), since several M-blocks share an expert.
     - Keep scale and A loads cached.
   - **S, stage 1:** {tn128, kw1} against {tn64, kw2/kw4} × wpe. Expected 0–3%.
   - **Stage 2:** tile_n 128 vs 256, plus the persist variant (`t64x128x256_atomic_persist_async_w4_cumul3`) at L.
   - **M:** wpe and b_nt only. Expect about 1.0–1.1×.

   A config that wins at one L Point and loses at another is split per case, never averaged.
3. **Stop rule:** if S lands below about 0.97× of aiter after B, stop tuning L. Compare per-kernel times against aiter at n = 512 first; likely causes are node count, b_nt, or the stage-2 tile.

## 2. Rejected or deferred, and why

| Idea | Decision | Reason |
|---|---|---|
| Stepping stones (FlyDSL dequant, faster torch `_mxfp4`) | Reject | At most 3–5×; the port is about 60× and already proven to validate. |
| Prior art `90eaf5c4` / bf16-dequant MFMA | Reject | Numerically wrong at M ≥ 256 and below 1 at M/L (report §4.3). |
| 128-workgroup redundant-scan router | Reject | The single-workgroup sort is simpler, compliant, and costs under 1% at L. |
| Fixed-order reduce instead of atomics | Reject | It adds a node and partial-row traffic. Atomics are allowed under bounded determinism. |
| One fused kernel for both stages | Reject | Needs cross-workgroup waiting (forbidden) or serial w2 streaming. |
| Folding the x quant into stage 1's A-load | Reject for now | About 12× redundant x reads at S. |
| `xcd_swizzle = 4` | Defer | Remap and validity-test ordering are unverified, and field data shows it slower in stage 1. A/B it on all 5 M/L Points only after C. |
| Stage-2 whole-K B preload; BM > 128; 32x32x64 MFMA; LDS ring; `s_setprio` | Defer | These are new main-loop kernels, not knobs of the port. Next iteration, once a correct inlined port is banked. |
| Skipping padding MFMAs / out-of-range padding A loads | Optional in B | Do the out-of-range A loads anyway for NaN safety (§1). Skipping MFMAs is an extra 0.5–1.5% only if cheap. |

## 3. Risks to watch

- **Compile time:** about 6–8 configs compiled on first call. Strip unused branches so this fits the driver's budget.
- **Stress / concurrent streams:** use per-call `torch.empty` scratch, LDS-only sort counters, and no module-level tensors.
- **Validate on all 25 Points every time a layout or config changes.** Losing one workload rejects the candidate.

## 4. Expected outcome (argued, not measured)

| After | Mean vs aiter | Over today's incumbent |
|---|---|---|
| Checkpoint A | ~0.8–1.0 (lower end if torch glue remains) | ~55–70× |
| Checkpoint B | ~0.97–1.01 | |
| Checkpoint C | ~1.02–1.07, gain mostly at L | |

The KEEP comes from A and B. C is upside, so do not put the banked landing at risk for it.
