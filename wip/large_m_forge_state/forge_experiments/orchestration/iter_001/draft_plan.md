Direction: replace-the-torch-path-with-a-fused-flyd

# Plan: fused native-MXFP4 FlyDSL MoE for all 25 Points

## 1. Decision

The canonical `kernel.py` is still the pure-torch reference. It takes 16–63 ms against aiter's 0.179–2.457 ms, a mean speedup of about 0.016. Most of that time goes to three things: dequantizing all 128 experts to bf16 on every call (12.5 ms, the same at every n), about 20 torch passes per `_mxfp4` call, and per-slab weight copies (report §1, §4.1).

All three specialists agree, and the report backs them: only a fused FlyDSL a4w4 MoE moves the score. It must use native scaled-fp4 MFMA on the 16×16-preshuffled weights and never build bf16 weights. Every other ledger direction is a refinement of that kernel. The stepping-stone direction is rejected: even deleting the whole dequant leaves 0.045× (report §1, §6.5).

**This round has one target.** Land a correct fused path on all 25 Points, built by porting aiter's own a4w4 stage-1 and stage-2 FlyDSL kernels (what the baseline runs), with a new front end. That takes the score from about 0.016 to about 0.8–1.1 per case (report §6.1; this is argued, not measured). Per-regime tuning comes afterwards and is ordered below.

Nothing in the specialist reports is measured. No probes were spent, and PMC is unavailable on this image. Every constant below is argued and has to be raced once the kernel exists. The measured anchors are:
- baseline times and the aiter configs it picked (report §4.3);
- the correctness bound, which is `BaselineRelativeBound` (report §4.4);
- expert route counts, max ≤31 per expert at n ≤ 640 (report §4.3).

## 2. Target architecture (resolving the specialists' disagreements)

| Topic | Decision | Why / what was rejected |
|---|---|---|
| Starting code | Port aiter's a4w4 stage-1/stage-2 mainloops from `/opt/venv/lib/python3.12/site-packages/aiter/ops/flydsl/` (`moe_kernels.py`, `kernels/mxfp4_gemm1*`, `mxfp4_gemm2*`, `mixed_moe_gemm_2stage_common.py`). Remove the aiter imports (`tensor_shim`, `dtypes`). AKLR may not import aiter at runtime. | This is the code the baseline runs, already working on this layout. The prior art `90eaf5c4` (bf16 MFMA, BM=16, unrounded activations) is 9% slower and numerically wrong at M ≥ 256. Reuse only its clamped-SwiGLU math. The KB template `ef15e878` stores fp8 stage-2 rows, which breaks this contract's bound. Copy its routing ideas, not its stage 2. |
| MFMA | `mfma_scale_f32_16x16x128_f8f6f4`, fp4 on both operands, in both stages and at every regime. | The compute specialist's argument holds. The scale shuffle (`kernel.py:43-44`: `[N/32,K/256,4,16,2,2]`) matches the 16x16x128 lane map. Under 32x32x64 it needs per-lane byte permutes and cross-lane moves inside the loop. This rejects the algorithm specialist's 32x32x64-at-L suggestion. 32x32x64 can come back later as a race for stage 2 only. |
| Launches (first landing) | Up to 4: **K0** route + x quant + zero `out` → (an optional quant-in-sorted-order launch, see §3 step 1) → **K1** stage 1 → **K2** stage 2. | A 3-launch prep design (algorithm specialist) works for every regime and quantizes x once. The compute specialist notes that inline quantization at L would re-read and re-quantize x six times. The memory specialist's 2-launch fold is an S-only race for later (Phase C). |
| Routing | K0 starts 128 routing workgroups, one per expert. Each one scans all R ids, builds the 128-bin histogram in LDS, takes its padded exclusive prefix, and compacts its own expert's routes **in input order** with ballot prefix sums (the `_route_expert` pattern). It writes sorted ids (padding sentinel = M), the expert of each block, and the total block count. | No workgroup waits on another, the result is deterministic, and it scales to R=131072, unlike the prior art's single-CTA route. The cost at R=131k is an estimated 3–4 µs, overlapped with the quant workgroups. Write the metadata **in the encoding aiter's stage kernels expect** (sorted_token_ids with the packed topk slot, sorted_expert_ids, num_valid_ids) so the ported mainloops run unchanged. |
| Stage-2 combine | bf16x2 atomics into `out`, which K0 zeroes. | All three specialists agree. `determinism: bounded` allows atomics, and aiter's own `atomic_bnt2` path, whose error the bound scales from, works the same way. Fixed-order route rows plus a reduce launch cost about 8% at S and about 0.5 ms at 32768: rejected. |
| Tile M per regime (compile function keyed on `n_tokens`) | S (≤640): BM=32. M (3112/3816): BM=128. L (≥10000): BM=128 at first, race 256 later. | BM=32 gives one weight sweep at S. BM=128 sweeps 1.15 GB against 2.11 GB at BM=64 (3816). aiter's BM=64 at 32768 sweeps 15.9 GB. Taking BM=128 at L when the port first lands is the cheapest part of the per-regime tile-M direction. |
| Cache policy at S | w1/w2 non-temporal. Scales, A, route ids and y cached. Do not compile the per-block cached/NT branch at n ≤ 640. | At S no workgroup reuses another's weights (one M-block per expert). Field data: non-temporal scales cost −1.7%, and the dead branch cost small batches up to 2.1%. |

## 3. Implementation sequence

Each step is checked bit for bit before the next one starts. Keep the torch path in a private helper for development A/B only. It must not be the landed path for any Point.

**Step 0: read before writing (no code yet).** In aiter's `moe_kernels.py`, trace the a4w4 call chain for the configs the baseline uses (`t32x128x256_w2` + `atomic_bnt2`, `t128x128x256`) and record:
1. The sorted_token_ids encoding and padding sentinel, the block→expert map, and how the device-side tile bound is read.
2. **The x-scale layout stage 1 reads:** token-major, or sorted and shuffled (as in aiter's `moe_mxfp4_sort`).
3. Whether stage 1 writes fp4 y plus shuffled scales in its epilogue, or writes bf16 y followed by a separate quant kernel. The `_bf16` in the kernel name makes this unclear.
4. Whether the `has_swiglu_limit` epilogue exists, and that it matches the reference:
   - gate = `min(g, L)`;
   - up = `clamp(u, −L, L)`;
   - y = `g·sigmoid(1.702g)·(u+1)`;
   - gate is w1 rows [0, 768), up is rows [768, 1536);
   - the result is rounded fp32 → bf16 → MXFP4, in that order.
5. Whether the XCD remap runs before the validity test. If it does, keep `xcd_swizzle=0`; the KB measured that order 1.3–3.4× slower.
6. How `swiglu_limit` arrives. If it is a device tensor, read it inside the kernel. Never call `.item()` in `run`.

**Step 1: K0, route + x quant + zero `out`.**
- **Routing:** as in §2.
- **x quant:** one lane per 32-element block.
  - Four 16-byte loads, then an in-lane amax on |bf16|.
  - Exponent: E8M0 = floor(log2 amax) − 2, plus 1 when the frexp mantissa is > 0.75, clamped to ±127.
  - Convert with `cvt_scalef32_pk_fp4_bf16`, then one 16-byte store plus the scale bytes.
- **Zeroing:** the workgroups that read a token's x row also zero that token's `out` row.
- **If stage 1 expects sorted/shuffled x scales**, there are two options. Prefer (a), the token-major layout the compute specialist proposes (`[token][K/256][4][2]`, one u16 per lane per 2 K-steps, byte picked by opsel), if aiter's A-scale load is easy to modify. Otherwise (b) add a quant-in-sorted-order launch after K0, which matches aiter's node count.
- **Gate:** dequantized x must **equal** `kernel._mxfp4(x)` exactly. Test RNE ties, the 0.75-mantissa boundary, all-zero blocks, the −127 clamp and saturation. If the hardware cvt differs (multiply vs divide by the scale, rounding mode), fall back to integer RNE on the e2m1 grid, about 8–10 VALU per element.
- **Gate:** the route maps must equal the ones derived from the stable argsort.

**Step 2: K2, stage 2**, at first fed y from torch.
- Port aiter's `mxfp4_gemm2`: multiply by the routing weight, then bf16x2 atomics.
- The padding guard (`token < M`) must sit in a `@flyc.jit` helper, or it is silently dropped (inference card §2).
- This removes the 4 ms w2 dequant and the K=4 matmul combine. Check `out` against the reference within the bound.

**Step 3: K1, stage 1.**
- Port aiter's stage 1. Gate and up for the same inter columns must sit in the same wave, so SwiGLU stays in-lane.
- Epilogue, in order:
  1. Clamp in fp32, then SwiGLU.
  2. `cvt_pk_bf16_f32`.
  3. Stage the bf16 tile through LDS.
  4. One lane per row × 32 columns runs the same quantizer as x.
  5. Store 16-byte fp4 y in sorted-row order, with scales in exactly the layout stage 2's A-scale load reads: the weight-scale shuffle `[rows/32][3][4][16][2][2]`.
- **The scale layout is the most likely silent bug.** Check y codes and scales bit-exact against `_mxfp4(bf16 y)`.
- If aiter used a separate y-quant kernel, fusing it here also removes a node.

**Step 4: land.**
- `run` looks up the config from `n_tokens` in a module-level `functools.cache` compile function.
- Scratch (sorted ids, expert map, block total, x and y codes and scales) is allocated with `torch.empty` per call, or once per shape and fully rewritten every call. No counters live across calls; the driver's `--stress` test checks this.
- No sync, `.item()`, printing or environment reads in `run`. Update the docstring to describe the kernel, with no history or speedups.
- Keep everything in `kernel.py`, the only listed editable source, unless sibling modules are allowed.
- Clear the FlyDSL cache before trusting any result.
- **Check:** Kineto shows 3–4 kernels per call and no `Cijk`/`direct_copy`/`elementwise` kernels. At n ≤ 640 the stage kernels move ≥5 TB/s (bytes from the shapes ÷ kernel time).
- **Fallback if K0 routing slips:** a fused GEMM path with torch routing glue (about 0.25 ms) still lifts the score far above 0.016. It loses S badly (about 0.4×), so treat it only as an interim landing.

## 4. After the first landing (ordered by expected value)

**Phase B: per-regime main loop (M/L, 5 Points, ~1.3–2.2× each, argued).**
- **Stage 1 tile:** 8 waves as 2(M)×4(N), 2 waves/SIMD, a 128-column gate + 128-column up tile.
- **A path:** gathered A rows go through a 3-stage XOR-swizzled LDS ring (64 B/row/step; swizzle by `row&3`, or pad rows to 80 B). B goes straight to VGPRs, prefetched 1–2 K-steps ahead, with loads interleaved via `sched_group_barrier`.
- **Race:**
  - BM {128, 160/192, 256}. BM must stay a multiple of 32 for the scale layout.
  - BNi {64, 128}.
  - `s_setprio` on/off.
  - 2 vs 3 waves/SIMD at BM=128.
  - At L, B direct-to-VGPR vs B through LDS.
- **Register check:** confirm VGPR/AGPR counts and spills with `FLYDSL_DUMP_IR=1`, not just by time.
- **Order at L:** GROUP_M plus XCD-aware order, applied **after** the raw-id bound test, so an expert's M-tiles share L2.
- **Stage 2:** load the full K=768 A panel into LDS once, so the MFMA loop has no barrier.

**Phase C: node and fixed-cost cuts at S (20 Points, about 2% per removed node).**
- First get aiter's per-kernel Kineto split at n=256 and 640. It is the most decision-relevant missing measurement.
- Then, in order:
  1. Kernarg preload, and reading the expert id before the A loads in both stages.
  2. Race stage-2 BN 256 vs 512.
  3. Race the S-only 2-launch fold: each stage-1 workgroup scans `topk_ids` itself, issues its first 2 K-steps of B loads **before** the scan, quantizes its ≤32 gathered rows inline into LDS, zeroes `out`, and publishes the route list for stage 2.
- Keep the fold only if it beats the 3-node version in interleaved same-session A/B. The redundant x re-reads (about 190–300 MB from L2) are its risk.

**Phase D: medium risk, then tuning.**
- Persistent or capped stage-2 grid, issuing the next tile's w2 loads before the current tile's atomics. Exit on the raw id before any remap.
- Per-shape config table raced offline, last.

## 5. Rejected or deferred, with reasons

- **Stepping-stone torch dequant/quant kernels:** each is a 3–5× gain against a 50–90× gap, and no score improvement.
- **bf16-dequant MFMA:** its practical floor is at or above the baseline from 3816 up.
- **fp8 or truncated stage-2 partials:** they violate `BaselineRelativeBound`.
- **XCD remap and GROUP_M at S:** there is no weight reuse to localize there, and the KB records field losses.
- **Software L2 prefetch of w2 during stage 1:** it lost in the field, and w2 is larger than the MALL.
- **Fixed-order reduce at S and L:** it costs about 8% at S and about 0.5 ms at L.

## 6. Risks to watch

1. **Bit-exact x and y rounding.** It is the top correctness risk, and the gate comes before any timing.
2. **Sorted-row and scale layout handoffs** between K0, K1 and K2.
3. **Noise at S.** Margins are 1.00–1.08 with about 3% session drift. Judge S changes by interleaved A/B pairs in one session, and never on the mean alone. A candidate that loses any Point's correctness is rejected outright.
