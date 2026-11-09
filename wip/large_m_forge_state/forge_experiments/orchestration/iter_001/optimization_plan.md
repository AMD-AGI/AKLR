Direction: replace-the-torch-path-with-a-fused-flyd

# Plan: fused native-MXFP4 FlyDSL MoE, landed against the aiter incumbent

## 1. Decision and success gate

**Route.** Keep the fused FlyDSL a4w4 MoE: native `mfma_scale_f32_16x16x128_f8f6f4` on the 16×16-preshuffled weights, with no bf16 weights built anywhere. Build it from aiter's own afp4_wfp4 stage-1 and stage-2 FlyDSL kernels, plus a FlyDSL front end that replaces aiter's HIP sort and quant.

The alternatives are dominated:
- The stepping stones stay below 0.05× even with the whole dequant deleted (report §1, §6.5).
- A bf16-dequant design has a practical floor at or above the baseline from 3816 up (report §4.3).
- The prior art `90eaf5c4` is numerically wrong at M ≥ 256 (report §4.3).

**The incumbent is aiter, not the torch path.** I checked `run_state.json`:
- `best_case_times` equals `baseline_case_times` value for value.
- `best.mean_case_speedup` is null.
- The evidence mean speedup is 1.0.

So a KEEP has to beat aiter's per-Point times. A first landing that only reaches 0.8–1.0 is **not kept**, and neither is any interim landing that uses torch glue. Before the first submission, confirm in the driver output which times the KEEP gate divides by.

**Landing target for this iteration:** all 25 Points correct, and a mean speedup over `best_case_times` above 1.0.

**Arithmetic.** With the 20 S Points averaging s and the 5 M/L Points averaging m, the mean is above 1.0 only when m > 5 − 4s.
- s = 1.00 needs m > 1.00.
- s = 0.97 needs m > 1.12.
- s = 0.95 needs m > 1.20.

**Where the margin can come from (argued, not measured; §5 states how each is checked):**
1. **L, measured lever.** At 32768 aiter runs `t64x128x256_w4`, which sweeps about 15.9 GB of weights through L2/MALL (report §4.3, §6.3). Stage-1 tile_m = 128 is inside aiter's own knob space and about halves those sweeps. aiter's configs at 10000 and 13152 are not recorded yet (Step 0).
2. **Untuned configs everywhere.** aiter reports "no tuned FlyDSL config" at every Point. An offline race inside aiter's knob space (§5) is the realistic source of most of the M/L margin and of any S margin.
3. **Fewer graph nodes at S.** Each removed node is worth roughly 2% at S (estimated at about 4 µs per node against 180–190 µs calls). This can come from folding x-quant and `out` zeroing into the routing launch, and from a stage-1 variant that writes fp4 y directly if one exists (Step 0).

**M regime correction.** aiter already runs `t128x128x256` at 3816 (token key 4096), and 3112 very likely falls in the same bucket. Moving to BM=128 at M therefore gains nothing over the baseline. A faithful port is about 1.0× at M, and any M gain must come from the knob race and node savings.

## 2. What is measured and what is not

**Measured** (report §4.3–4.4, source map):
- the baseline times and aiter's chosen configs at ≤ 640, 3816 and 32768;
- one weight sweep at BM = 32 for every S Point (the largest expert has ≤ 31 routes);
- the `BaselineRelativeBound` on `out`;
- aiter's knob space, which I read in `moe_kernels.py:188-338`.

**Not measured:** everything else. No probes were spent and PMC is unavailable on this image. All gains and costs below are estimates until the race in §5 measures them.

## 3. Architecture

| Topic | Decision | Reason |
|---|---|---|
| Stage kernels | Port aiter's afp4_wfp4 stage 1 and stage 2 from `/opt/venv/lib/python3.12/site-packages/aiter/ops/flydsl/` (`kernels/mxfp4_gemm1*`, `mxfp4_gemm2*`, `mixed_moe_gemm_2stage_common.py`, the naming and params logic in `moe_kernels.py`). Remove the `tensor_shim`/`dtypes` imports; AKLR may not import aiter at runtime. | This is the code the baseline times, already working on this layout, so a faithful port starts at about 1.0×. |
| MFMA | 16x16x128 scaled fp4 in both stages at every regime. | The scale shuffle (`kernel.py:43-44`) matches the 16x16x128 lane map. 32x32x64 would need per-lane byte permutes and cross-lane moves inside the loop. |
| Front end | **K0** (FlyDSL): route + x → MXFP4 + zero `out`. It writes metadata **in exactly the encoding aiter's stage kernels read**: sorted_token_ids with the packed top-k slot, sorted_expert_ids, num_valid_ids, and the x-scale layout (token-major, or the sorted/shuffled form `moe_mxfp4_sort` produces). | aiter's sort is the HIP/opus `module_moe_sorting_opus` (report §2), not FlyDSL, so it cannot be ported as-is. Matching its encoding lets the stage kernels run unchanged. |
| Routing algorithm | 128 routing workgroups, one per expert. Each scans all R ids, builds a 128-bin histogram in LDS, takes its padded exclusive prefix, and compacts its own expert's routes in input order with ballot prefix sums. | No workgroup waits on another, the result is deterministic, and it scales to R = 131072. |
| Combine | bf16x2 atomics into `out`, which K0 zeroes. | Allowed by `determinism: bounded`, and it is aiter's own `atomic` path. At 32768, `out` is 402 MB, below the 32-bit offset limit (`requires_flydsl_stage2_reduce`, `moe_kernels.py:112-116`), so atomics are valid at every Point. Fixed-order reduce costs about 8% at S and about 0.5 ms at L: rejected. |
| Configuration | One entry per `n_tokens` in a module-level `functools.cache` compile function, filled from the offline race (§5). Values unknown to the table use a regime default: S → BM = 32, M → 128, L → 128. | AKLR Stop table: no timing in launch code. |
| `swiglu_limit` | Pass it to the kernel as a scalar argument. | It arrives as a scalar keyword. Never call `.item()` in `run`. |
| XCD remap | `xcd_swizzle = 0` unless Step 0 shows the remap runs **after** the validity test. | When the remap ran first, it measured 1.3–3.4× slower (inference card §2). |

## 4. Implementation sequence

### Step 0: read and measure before writing code (short, decides the rest)

1. **aiter's chosen config at all 25 Points**, both stages: kernel name, plus `waves_per_eu`, `b_nt`, `k_wave` and `persist`.
   - 10000 and 13152 are unknown today, and 3112's bucket is assumed.
   - Read them from the Kineto names in `evidence/prior_art/probe_*`, or from one profiled baseline call per Point.
2. **aiter's per-kernel Kineto split at n = 256 and 640**, plus 3816 and 32768: sort, quant, any mxfp4 sort, stage 1, any y-quant, stage 2, any memset.
   - This gives the node count and how much of each call is stream time and how much is fixed cost. It decides how the S margin is spent.
3. **The front-end encoding spec, written down.**
   - The sorted_token_ids bit layout and padding sentinel.
   - The block→expert map, `num_valid_ids`, and how the device-side tile bound is read.
   - The x-scale layout stage 1 reads.
   - Whether sorted routing weights are passed separately.
   - Keep this spec as the test oracle in §4 Step 1.
4. **The stage-1 output path.**
   - Does the baseline's `…_bf16_…` stage 1 write bf16 y and run a separate y-quant node?
   - Does a registered `_fp4` output variant exist (the suffix regex in `moe_kernels.py:54-56` accepts `_fp4`), and does it work on this layout?
5. **The clamped-SwiGLU epilogue.** Does aiter's stage 1 implement it exactly?
   - gate = `min(g, L)` and up = `clamp(u, −L, L)`;
   - y = `g·sigmoid(1.702·g)·(u+1)`;
   - gate is w1 rows [0, 768) and up is rows [768, 1536);
   - rounding goes fp32 → bf16 → MXFP4.
6. **The order of XCD remap and validity test**, and the semantics of stage-1 `k_batch` (split-K): how its partials are reduced before the nonlinear activation.

**Switch conditions from Step 0:**
- **No swiglu-limit epilogue in the FlyDSL stage 1.** It is new code: about 10 fp32 VALU per element added to the ported epilogue. Budget it before Step 3. Keep bf16 y with a separate FlyDSL y-quant launch at first, and fuse later only if Step 0 item 4 shows the fp4-output path exists.
- **aiter already uses BM = 128 at 10000/13152.** Then the L-regime target rests on 32768 plus the race, and the m > 5 − 4s budget must be rechecked.

### Step 1: K0, routing + x quant + zero `out` (time-boxed)

- **Routing:** as in §3.
- **x quant:** one lane per 32-value block.
  1. Four 16-byte loads, then an in-lane amax on |bf16|.
  2. E8M0 = floor(log2 amax) − 2, plus 1 when the frexp mantissa is > 0.75, clamped to ±127.
  3. Convert with hardware `cvt_scalef32_pk_fp4_bf16`.
  4. One 16-byte store plus the scale bytes, in the layout from Step 0 item 3.
- **Zeroing:** the same workgroups zero `out` row by row.
- **Oracle:** an offline probe script, which `kernel.py` must not import, runs aiter's sort and quant on the timing inputs and compares field by field: sorted ids, expert ids, `num_valid_ids`, x codes and x scales.
  - Route maps must match exactly (both are input-order stable).
  - x codes differing from aiter or `_mxfp4` are a **diagnostic**, not a blocker (see the gates in Step 4).
- **Time box.** If K0's route maps are not matching after about a third of this iteration's implementation budget, switch to the fallback.
- **Fallback (FlyDSL, never torch):** a single-workgroup FlyDSL sort that loops over all routes in chunks of 1024: a histogram in LDS, a scan, then an input-order compaction. Run x-quant + zeroing as a separate FlyDSL launch.
  - It is correct at every R.
  - Estimated cost: a few µs at S, and tens of µs at 32768 (about 1–4% there).
  - It adds one node at S, so revisit K0 in a later round.

### Step 2: stage 2 (ported)

- Feed it y from the reference path in a dev harness only.
- Port `mxfp4_gemm2` with routing weight × output and bf16x2 atomics.
- The `token < M` padding guard must sit in a `@flyc.jit` helper, or it is silently dropped (inference card §2).
- **Check:** `out` within the validate bound.

### Step 3: stage 1 (ported)

- Gate and up for the same inter columns stay in the same wave.
- Epilogue order: clamp-SwiGLU in fp32 → bf16 → MXFP4. The y scales must use exactly the layout stage 2's A-scale load reads. **This is the most likely silent bug:** check y codes and scales against an offline `_mxfp4(bf16 y)` as a diagnostic.

### Step 4: land, with gates

**Primary gate (blocks landing):**
- `validate` violation fraction ≤ 0.001 on **all 25 Points**;
- the bounded-determinism run-to-run check;
- the driver's `--stress` (concurrent streams).

**Secondary diagnostics (do not block):**
- bit-exact x and y against `_mxfp4`;
- x codes against aiter's quant.

**Integer-RNE fallback** (about 8–10 VALU per element, roughly doubling the stage-1 epilogue cost): switch to it **only if the primary gate fails** and the diagnostics trace the failure to the hardware cvt's rounding or scale direction.

**ABI:**
- Scratch is allocated per call with `torch.empty`, or once per shape and fully rewritten every call. No counters persist across calls.
- No sync, `.item()`, printing, clock reads or environment reads in `run`.
- `run` writes into `out` and returns nothing.
- The docstring describes the kernel, with no history or speedups.
- Keep the code in `kernel.py`, the only listed editable source, unless the harness accepts sibling modules.

**Structural check:** Kineto shows only FlyDSL kernels per call (no `Cijk`, `direct_copy` or `elementwise`), with the node count from Step 0 item 2 or fewer. Clear the FlyDSL cache before trusting any number.

## 5. Offline knob race (required before the first submission)

**Method:**
- Race in a probe script, against aiter, with **interleaved A/B pairs in one session** (about 3% session drift exists).
- Bake the winners into the compile-function table as constants keyed by `n_tokens`.
- Only options aiter registers are in scope. For fp4 A, this is what `moe_kernels.py:188-338` registers:

**Stage 1:**
- tile_m ∈ {32, 64, 128}.
- tile_n ∈ {32, 64, 128} at tm = 32, and {64, 128} otherwise. tile_n = 256 is **not** available for fp4 A.
- tile_k = 256.
- waves_per_eu ∈ {1, 2, 3, 4}.
- k_wave ∈ {1, 2, 4}, at tm = 32 only, with at most 8 waves.
- b_nt ∈ {0, 2}.
- xcd ∈ {0, 4}, the latter only if the remap order is fixed.
- k_batch ∈ {2, 4, 7, 14}, at wpe = 3 and tm = 32 only. Use it only if Step 0 item 6 shows its partials are reduced before the activation, and the result passes the primary gate. Exclude `_go` (mock gate-only).

**Stage 2:**
- tile_m ∈ {16, 32, 64, 128}. It must equal the sort block unless an `_sbm`/v2-layout variant is used.
- tile_n ∈ {128, 256}; tile_k ∈ {128, 256}.
- atomic only.
- b_nt ∈ {0, 2}; persist; xcd ∈ {0, 4}.
- The production `t64x128x256_atomic_persist_async_w4_cumul3`.
- v2-layout `flydsl_moe2_layout_…` variants (nt, persist).

**Priority by regime:**

| Regime | First races | Expectation (argued) |
|---|---|---|
| L (10000, 13152, 32768) | Stage-1 tm 128 vs aiter's choice, with b_nt = 0 (cached, since weights are reused across M-blocks), waves_per_eu, tile_n 64 vs 128. Stage-2 tm 128, persist and production variants. | The largest lever, from halving the weight sweeps at 32768. Expected about 1.2–1.6× there; the other two depend on Step 0 item 1. |
| M (3112, 3816) | Holding tm = 128: waves_per_eu, b_nt, tile_n 64 (more workgroups), stage-2 persist and production variants. | About 1.0–1.15×. |
| S (256–640) | Holding tm = 32: tile_n {32, 64, 128} × k_wave {1, 2, 4} × wpe, b_nt = 2 (non-temporal) on weights; stage-2 tile_n 128 vs 256, persist, v2 layout. | About 0.98–1.05×, plus about 2% per node removed relative to aiter. |

## 6. KEEP, stop and switch criteria

| Condition | Action |
|---|---|
| All 25 correct and mean > 1.0 against `best_case_times` | KEEP. Then Phase B/C (§7). |
| All correct, mean ≤ 1.0, S average < 0.95 | Stop tuning M/L. Compare our S node count and per-kernel stream rate (bytes from the shapes ÷ kernel time) with aiter's split from Step 0 item 2. Fix the S front end (node count, K0 cost) first. |
| All correct, mean ≤ 1.0, S ≥ 0.97, M/L average short of 5 − 4s | Do not submit. Widen the L race: tile_n, waves_per_eu, persist on both stages. Then start Phase B's first new-kernel item at 32768 only. |
| Any Point fails the primary gate | No submission. Diagnose with the secondary checks: layout handoff first, rounding second. |
| K0 over its time box | FlyDSL single-workgroup sort fallback (§4 Step 1). Never torch glue. |

## 7. After the KEEP (new kernels, costed separately)

**Phase B: new main-loop code, M/L only. These are not knobs of the port.**
- **BM ∈ {160, 192, 256}** for stage 1 and stage 2 (aiter caps both at 128). This is new tiling and scale-layout code (BM must stay a multiple of 32). Highest value at 13152 and 32768.
- **New stage-1 main loop:** 8 waves as 2×4, 2 waves/SIMD, gathered A through a 3-stage XOR-swizzled LDS ring, B straight to VGPRs prefetched 1–2 K-steps ahead with loads spread by `sched_group_barrier`, and `s_setprio` raced.
- Confirm VGPR/AGPR counts and spills with `FLYDSL_DUMP_IR=1`.
- **GROUP_M + XCD-aware order** after the raw-id bound test, at L only.
- Estimated at several days of work and about 1.3–2.2× on the L Points (argued). Start with 32768.

**Phase C: S fixed cost (20 Points).**
1. Kernarg preload, and reading the expert id before the A loads.
2. A fused fp4 epilogue in stage 1, if Step 0 showed a separate y-quant node.
3. The S-only 2-launch fold: each stage-1 workgroup issues its first B loads, then scans `topk_ids`, quantizes its ≤ 32 gathered rows inline, and zeroes `out`. Keep it only if it wins the same-session interleaved A/B; the redundant L2 reads of x (about 190–300 MB) are its risk.

**Phase D:**
- A persistent stage 2 at S, if the race did not already cover it.
- A final per-shape table refresh.

## 8. Rejected

- **Stepping-stone torch kernels:** they cannot reach 0.1×.
- **bf16-dequant MFMA:** it loses at M and L.
- **fp8 or truncated stage-2 partials:** they violate `BaselineRelativeBound`.
- **32x32x64 MFMA:** it needs scale permutes inside the main loop.
- **Fixed-order reduce:** about 8% at S and about 0.5 ms at L.
- **XCD remap and GROUP_M at S:** there is no weight reuse there.
- **Software L2 prefetch of w2:** it lost in the field, and w2 exceeds the MALL.
- **Any torch fallback as a landed path:** it cannot beat the aiter incumbent.

## 9. Risks

1. **Silent layout bugs** in the front-end encoding and in the y-scale handoff between stage 1 and stage 2. Mitigation: the field-by-field oracle against aiter's sort and quant.
2. **The S margin is inside the noise.** It is 1.00–1.08× in practice, with about 3% session drift. Decide only on interleaved same-session pairs and per Point, never on the mean alone.
3. **Hardware cvt rounding** differing from `_mxfp4`. It only matters if the primary gate fails.
4. **The L lever depends on aiter's untracked configs** at 10000 and 13152 (Step 0 item 1). Recheck the m > 5 − 4s budget as soon as they are known.
