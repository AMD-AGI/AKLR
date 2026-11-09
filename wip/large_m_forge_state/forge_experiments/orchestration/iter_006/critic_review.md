# Review of the iteration-6 plan: fused route + x-quant at n ≤ 1024, then stage-2 `tile_n=256` and the large-n LDS trim with `wpe=5`

VERDICT: ACCEPT

**Bottom line:** the plan goes after the largest measured lever left on the Points that make up 80% of the mean. The source supports its premises, and it fixes both problems the last review raised. The issues below should be applied while executing; none of them changes the route.

## The route holds up against the evidence

- **The bottleneck is measured, not assumed.** Report §4.1 and §4.2 measure the prologue plus launch gaps at the small-n Points:
  - n=256: 16.2 µs (9.7%) of a 168 µs call;
  - n=640: 23.8 µs (13.1%).

  Every remaining stage-1 and stage-2 lever there is a few µs (report §5–6.2). The memory specialist's two-point fit makes this stronger: stage 2 streams at the same per-byte rate as stage 1, so persistence is worth about 1–3 µs. The plan was right to make the prologue Step 1 and to reduce the persistent rewrite to a cheap `tile_n=256` gate.
- **Step 1's main premise checks out in source.**
  - x-quant reads the sort only through `pos_v` and `pos_s` (`kernel.py:8688-8689`), and uses them only in the scale scatter (`8726-8732`).
  - The sort's outputs are fully specified in `8815-8887`: arrival-order rank (`8843`), packed id `((i&3)<<24)|(i>>2)` (`8847`), padding and tail fill, a binary search for block experts clamped to `experts−1` (`8882`), and `num_valid = [total, n_tokens]`.
  - The fused design reproduces each of these, and the token-order rank makes the result deterministic.
  - No workgroup reads another workgroup's writes, so the ABI rule against workgroups waiting on each other is met.
  - It runs in one round: about 23 VGPRs and about 1.1 KB LDS gives roughly 8 WGs per CU, and the largest grid is 640 WGs.
- **The incumbent figures are correct.** From `run_state.json`, the lowest Point is n=256 at 0.179183 / 0.170514 = 1.0508. The gate's three conditions match what the iter-5 review asked for.
- **The stage-1 XCD remap is gone.** It was the flawed item last round, and the plan drops it.
- **The Step 3 premise correction checks out.**
  - `lds_x_bytes` (`6534`) and `lds_x_b` (`6688`) both use the full `lds_stride`, while the layout uses `eff_lds_stride` (`6616-6625`).
  - `lds_base_nxt = tile_m*lds_stride` (`7679-7681`).
  - The total is `max(lds_x, lds_out) + tid + tw` (`6538`), which comes to 16,896 B after the trim.
  - The specialist's 4-WG/CU arithmetic is right: a stage-2 grid of 24,576 WGs against 1,024 slots, so `cu_num_mul` only shapes the tail. That explains why iter 4 measured only −0.6%.

## Corrections to apply while executing

1. **Step 1 needs a minimum gain to keep it.** The gate only blocks regressions, so a fused kernel that comes out neutral would still be kept, and it adds a lot of code. Keep it only if graph time (not Kineto) at n = 256, 512 and 640 drops by at least about 2.5 µs per call, roughly 3× the ~0.5% noise. Below that, take the fallback, or drop Step 1 and log the measured times. Graph time is the right measure because part of the gain is a removed launch gap, which Kineto does not show.
2. **Step 1 prefix: decide by measurement, not up front.** Copying the serial padded prefix verbatim (`8828-8834`) is reasonable as a first build: it is 128 broadcast LDS reads per thread. But every WG now pays it on its critical path. Record the fused kernel's Kineto time at n=640 with the serial prefix. If it is over ~8 µs, swap in the wave scan before deciding between the ~11 µs threshold and the fallback.
3. **Step 1: an explicit barrier before the scale scatter.** Every thread needs the final `rank[k]`, `cnt` and `ends` to form `pos_k` (`8726-8732` runs on all threads with `c&3==0`, not just lanes 0–3). Make sure there is a barrier after the rank atomics and the prefix, not only after the histogram.
4. **Step 3a: grep before changing three lines.**
   - Grep the stage-2 builder for every use of `lds_stride` and `lds_tile_elems` (DMA destination offsets around `7380`, and the ping/pong swaps in the hot loop and the tail). Fix them all consistently.
   - After the trim, the pong buffer (8–16 KB) lies inside the c-shuffle alias region `lds_out` (0–16 KB), as ping already does. Confirm that a `gpu.barrier()` separates the tail's last A `ds_read` from the first c-shuffle write.
   - Confirm that `r139_xdma_first` (`6355-6363`, active for exactly this t64 async fp4 build, `7805-7806`) issues the next tile's DMA only after the previous tile's epilogue barrier.
   - Run `allclose` and `--stress` several times. A race here would not show up reliably in one run.
5. **Step 2 may not compile.** `tile_n=256` was compiled only on the async t64 path (iter 4). The small-n build is non-async t32 (`lessons/iter_005.md` HELD-FIXED). If it fails to compile or spills, log that and move on. Do not debug the builder for a gain argued at 0–2 µs.

## Opportunity cost and what to record

Step 1 is the right single bet: about +3–5% argued across 20 Points. It is low risk numerically, because the E8M0 bytes and the scatter layout are unchanged and there is a bitwise reference from a stable argsort. Steps 2 and 3 are bounded races with explicit keep and drop rules. Step 4 is properly optional.

If time runs short, cut Step 3 before Step 1's validation. Record per-kernel Kineto times plus graph time for every step, including the ones dropped, so that the persistent stage-2 and stage-1 occupancy directions get measured evidence for or against them next iteration.
