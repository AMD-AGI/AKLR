Direction: cut-graph-nodes-fold-x-quantization-and-

# Plan for iteration 6: merge the sort, x-quant and `out` zero-fill into one kernel at n ≤ 1024, then two cheap config races

## Decision

- **Main work: n ≤ 1024.** The 20 small-n Points (256–640) carry 80% of the mean. The report (§4.1/§4.2) measured 16–24 µs per call of serial prologue plus launch gaps there:
  - a one-workgroup sort of 7.6–9.1 µs on a single CU;
  - x-quant plus zero-fill of 4.6–5.4 µs;
  - 4–9 µs of graph time minus busy time.

  This is the largest lever left on those Points. The algorithm specialist's design removes it without touching stage 1: each x-quant workgroup computes its own token's 4 sorted positions. This is the main work this iteration.
- **Two cheap races follow**, each about one edit plus one A/B. They are independent of the main work and of each other:
  - **Stage 2 `tile_n=256` at n ≤ 1024.** This is the memory specialist's gate before any persistent stage-2 work.
  - **Large-n stage-2 LDS trim plus `waves_per_eu=5`.** This is the compute specialist's corrected version of the occupancy direction.
- **Deferred or rejected:** the persistent stage-2 N-sweep, the token-order scale layout, the stage-1 XCD remap, and stage-1 at 2 WG/CU (reasons at the end).

Incumbent: mean 1.0674, every Point ≥ 1.051 (iter 5). The KB route for the implementer to read before Step 1 is `moe/flydsl_moe_decode.md` §1–§2 (graph-node cost; E2 measured 3.5–6.4 µs from removing a sort node) and `moe/flydsl_grouped_moe_inference.md` §7.

---

## Step 0: A/B harness (do this first, it costs little)

- Write one process that builds the canonical and the candidate executables. Clear `${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}` first.
- Measure at n = 256, 512, 640, 3112, 10000 and 32768:
  - per-kernel Kineto times;
  - graph time.
- Every keep/drop decision below uses this harness. Confirm the final file with the driver's validate run.

## Step 1: fused route + x-quant kernel for n ≤ 1024 (main deliverable)

### Why this works (checked in source)

- x-quant uses the sort only for `pos32`: the scale scatter at `kernel.py:8688` and `8726-8732`.
- Stage 1 does not read `pos32`: it is not in the exe1 arguments at `9160-9163`.
- At n ≤ 640, R ≤ 2560 ids, about 10 KB. Every workgroup can read all of them and compute its own ranks, so no workgroup waits on another.

### Kernel `moe_route_xquant_fp4_k6144_m{n}_t{tile_m}`

- Grid `(n_tokens,)`, 256 threads.
- Arguments: x, xq, out, topk_ids, topk_w, sorted_ids, sorted_w, sorted_eids, num_valid, xs_sorted.
- LDS: `cnt[128]`, `ends[128]`, `rank[4]`, about 1.1 KB.
- Use a `SmemAllocator` with a new `global_sym_name`, and finalize it in the `@flyc.jit` wrapper the same way `launch_sort` does (`8892-8895`).

Per workgroup t:

1. **Issue the routing-independent memory work first.**
   - The 3 × 16 B x loads (`iters = 3`), kept in registers.
   - The 3 `out` zero stores (`zero_v4`).
   - The token's own 4 ids and 4 weights, as vec4 loads at `t*4`.
   - Threads with tid < 128 zero `cnt`; tid < 4 zero `rank`. Then barrier.
2. **Histogram.**
   - Load ids as vec4 (R/4 vectors, at most 3 per thread, guarded by `i4 < R/4`).
   - For each id e at route index i: `atomic_add(cnt[e], 1)`.
   - For each k in 0..3: if `e == e_k && i < 4t`, then `atomic_add(rank[k], 1)`. Matches are rare (≤ 31 per expert), so contention is negligible and this is simpler than shuffle reductions.
   - Barrier.
3. **Expert ranges.** For tid < 128, compute `ends` with the existing padded prefix loop, copied verbatim from `8828-8834`, then barrier.
   - Swap in a wave scan only if step 1's budget below is missed.
4. **Sorted positions.** `pos_k = ends[e_k] − padded(cnt[e_k]) + rank[k] + #{k' < k : e_k' == e_k}`.
   - The last term is a uniform guard. Top-k ids should be distinct, so it should always be 0.
   - Lanes 0–3 write `sorted_ids[pos_k] = (k<<24)|t` and `sorted_w[pos_k] = w_k`.
5. **Quantize.** Run the current quant body (`8691-8733`) on the preloaded words, with `pos_s[k] = pos_k`.
   - Keep the E8M0 bytes, the `0x3FFFFF` rounding constant and the shuffled `[rows/32][cols/8][4][16][2][2]` scatter exactly as they are.
6. **Padding, tail rows and block experts.**
   - Cover rows `s = t*256 + tid + it*n_tokens*256`, with a static `fill_iters = ceil(n_sorted/(n_tokens*256))`, which is 1 at every small-n Point.
   - Copy the body of `_build_sort_mw` verbatim (`9045-9061`): binary search over `ends`, fill rows with `s ≥ first_pad`, and write `sorted_eids` when `s & (tile_m−1) == 0`.
7. **`num_valid`.** If t == 0 and tid == 0, write `num_valid = [ends[127], n_tokens]`.

### Host changes

- `_compiled` builds the fused kernel when n ≤ 1024 and keeps sort_mw plus xquant above that. The large-n path is unchanged.
- `run()` does not allocate `pos32` on the small-n path, and launches the fused kernel in place of sort + x-quant.

### Correctness checks

The new order within each expert is token order (a stable sort), so an exact reference exists:

- Build expected `sorted_ids`, `sorted_w`, `sorted_eids` and `num_valid` in torch with a stable argsort of `topk_ids.flatten()` plus per-expert padding to 32. Check them **bitwise** at 256, 512 and 640.
- Check `xq` and `xs_sorted` at valid rows bitwise against the old pipeline, with rows remapped through `pos`.
- Run driver `allclose` and `--stress`.
- Kineto must show **3 kernels** at n ≤ 640.

### Budget and stop rules (gains are argued, not measured)

- **Target:** the fused kernel at ≤ ~8 µs at n=640, against today's 9.1 + 5.4 µs. Graph time should drop by 6–10 µs per call, about +3.5–5.5% on every small-n Point.
- **If the fused kernel is above ~11 µs at 640:**
  - First replace the serial prefix with a wave scan.
  - If it is still too slow, take the fallback below. Do not keep debugging.
- **Fallback:** keep x-quant as it is today and replace only the one-workgroup sort with the same redundant-histogram, token-order-rank logic, spread over `ceil(R/256)` workgroups, still writing `pos32`. Expected about −4 to −6 µs.

## Step 2: race stage-2 `tile_n=256` at n ≤ 1024 (config only)

- In `_config` (`8628`), set s2 `tile_n=256`, keeping `tile_m=32` and `s2_bnt=2`.
- Check VGPRs and residency in the `FLYDSL_DUMP_IR=1` ISA.
- **Keep only if** stage 2 gains ≥ 1.5 µs at 256, 512 and 640, and no small-n Point regresses by more than 0.5%.
- Expected 0 to −2 µs. The memory specialist's two-point fit says stage 2 is not slower per byte than stage 1, so the persistent rewrite is expected to gain little.
- A positive result here is the only thing that would justify a persistent N-sweep with cross-tile B prefetch in a later iteration.

## Step 3: large-n stage 2: trim the wasted A LDS, then try 5 WG/CU

The premise needs correcting first. The direction assumed 2 WG/CU with LDS room for 4. In fact stage 2 at 124 VGPRs and 33,280 B LDS already runs **4 WG/CU**; `cu_num_mul` only changes the persistent M-split. 5 WG/CU needs VGPRs ≤ 96 **and** LDS ≤ 32 KB.

**3a. LDS trim (verified).**
- On the async fp4 path, A rows use `eff_lds_stride = lds_stride//2` (`6616-6621`).
- But three places use the full stride:
  - `lds_x_bytes` (`6534`);
  - `lds_x_b` (`6688`);
  - the pong base `lds_base_nxt = tile_m*lds_stride` (`7679-7681`).
- Change all three to `eff_lds_stride` when `use_async_copy and a_elem_vec_pack > 1`. LDS goes from 33,280 B to 16,896 B.
- The pong offset **must** change with the allocation, or ping and pong overlap the c-shuffle region incorrectly.
- Verify the kernel descriptor's LDS size and run `allclose` at 3112 and 32768.

**3b. `s2_wpe=5`** for the n > 1024 config (`8629`).
- Dump the ISA. Require 0 scratch/spills and VGPRs ≤ 96. If it spills, drop this step; the hand-scheduled E21/E22 A-read pipelining is out of scope this iteration.
- Race at 3112, 3816, 10000, 13152 and 32768.
- **Keep if** stage 2 improves ≥ 3% at 32768 and no M or L Point regresses by more than 0.5%.
- Re-check the violation fraction at n ≥ 10000. Today it is 4–8e-5 against the 1e-3 allowance.
- If 3b is dropped, revert 3a as well: it gains nothing alone, and its value is the room it leaves for later cross-tile prefetch.

## Step 4 (optional, only if time remains)

- **Contiguous same-expert M-chunks in the persistent stage-2 loop** (an index change).
  - Lower confidence. In the current stride order, `(n, y)` and `(n, y+1)` are 48 linear ids apart (a multiple of the 8 XCDs). If dispatch is x-fastest, concurrent same-expert blocks already share an XCD's L2 (inferred). So the gain may be zero.
  - Race it only if 3b is kept or found neutral.
- **Stage-1 `tile_m=64` for 3112/3816** (config race; the sort unit and stage-2 `sort_block_m` follow it).
  - 3816 loses 10.5 µs to aiter in stage 1 at 3.59 rounds.
  - Small weight in the mean (2 Points).

## Gate for every step

Each step must meet all of the following (as the critic required):

1. No Point currently ≥ 1.03× drops below 1.03×.
2. No Point regresses by more than ~0.5%; rerun before calling a regression.
3. `allclose: True` and `--stress` pass.

Use the same-process A/B with the cache cleared, then the driver.

## Not this iteration

| Idea | Reason |
|---|---|
| Token-order scales with a stage-1 gather by token | The shuffled scale word packs rows r and r+16, which belong to different tokens. That forces byte loads into the stage-1 hot path, which is already at its roof. |
| Fold the sort into stage 1 | Stage-1 workgroups would wait on other workgroups' x-quant rows, which AKLR forbids. |
| Persistent small-n stage-2 N-sweep | Argued gain is only 0.5–1.5%, at high effort. Revisit only if Step 2 shows a response. |
| Non-atomic stage 2 | Report §4.3 nets it at 0–3%. |
| `b_nt=2` at large n | Measured negative in iter 5. |
| Stage-2 `tile_m=128` at L | Loses the tuned t64 paths; lower value than Step 3. |
| Stage-1 2 WG/CU at L | Needs both about 68 fewer VGPRs and LDS under 80 KB; too large for this iteration. |
| Stage-1 XCD remap | The critic found the contiguous mapping piles the early-exit tail onto 1–2 XCDs; it needs a redesign first. |

## Expected outcome (argued)

| Result | Mean |
|---|---|
| Step 1 alone | about 1.067 → 1.095–1.10 |
| Plus Steps 2 and 3 | +0–1% more |
| Conservative, if only the Step 1 fallback lands | about 1.08 |

Record the per-kernel Kineto times for each step, so that a neutral race is logged as a measured result rather than a failure of the route.
