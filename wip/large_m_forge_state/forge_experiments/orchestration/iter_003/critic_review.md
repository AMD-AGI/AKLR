# Review of the iteration-3 plan: remove the torch glue around the inlined FlyDSL stage kernels

VERDICT: ACCEPT

**Bottom line:** the plan is worth executing as written, with the five small notes at the end folded in. None of them requires re-planning.

## Is this the right route?

Yes. The incumbent's two stage launches are aiter's own FlyDSL builders, inlined. The 0.179× score comes from the torch code around them. That code is visible at `kernel.py:8701-8740`:
- the sort;
- `_quant` run twice, at about 20 elementwise passes each with int64 temporaries;
- `_sorted_scales` run twice;
- `out.zero_()`.

The numbers in `lessons/iter_002.md` §1 put almost all of the gap in that code:
- **n=256:** aiter's stage kernels take about 158 µs; our whole call takes 0.879 ms.
- **n=32768:** aiter's stage kernels take 2.19 ms; our whole call takes 24.4 ms.

Removing the glue is the only change that matters at first order. Every alternative lane in the ledger (tile-M, XCD order, combine, cache policy) is lost in noise until the glue is gone. Keeping the stage kernels as the aiter port is also correct: it reproduces aiter's rounding. No alternative backend dominates this route, so I see no case for a challenger.

## Source claims I checked

- **Code-row addressing is route order when `v2_output_layout=False`.** `kernel.py:5339-5342` sets `payload_row_idx = ts_idx` (`t*topk+s`). The row stride is `inter_dim//2` (384 B) when `need_fp4` (5317-5325). That is the same `(R,384)` layout stage 2 reads today from `_quant(y.view(R,768))`. Keeping the flag False on both sides is right, which closes the earlier dispute over this flag.
- **fp4 output switches on the sorted scales.** `need_fp4 = out_dtype=="fp4"` and `need_sort = need_quant` (3150-3153).
- **Scale offsets match `_sorted_scales`.** The offsets at 5594-5607 are `d0*n32_sort + d3*256 + d5*64 + d2*4 + d4*2 + d1`. That is the same permutation as `view(...,2,16,cols/8,2,4).permute(0,3,5,2,4,1)` (8689).
- **The plan's open question is answered: the column count is 24.** `sorted_scale_cols = ((768//32)+7)//8*8 = 24` (3576). So `n32_sort = 768`, which matches `cols_pad` in `_sorted_scales`.
- **The scale-buffer bound fits the allocation.** The kernel's bound is `round_up(size_expert_ids_in*sort_block_m, 256) * 24` (3580-3596). The host passes `size_expert_ids_in = s1_blocks = min(m_blocks, R)`. So `ys_sorted = round_up(m_blocks*tile_m, 256)*24` is never smaller than the bound.
- **The x-quant scale rule is the validated one.** The epilogue's E8M0 is `((bits+0x400000)&0xFF800000)>>23 − 2`, clamped at 0 (5441-5444). For x, the plan's `F − 2 + (mant > 0x400000)` is exactly the torch rule that passed validate: frexp exponent `e = F − 126`, and `m > 0.75` is the same test as mantissa bits `> 0x400000`. Copying the epilogue's `cvt_scalef32_pk_fp4_f32` with `native_scale = 2^(e8m0−127)` reproduces the torch codes, including the 6.0 cap.
- **aiter has no separate y-quant kernel at S.** The iter_002 profile shows only x-quant (7.2 µs) and scale-sort (4.6 µs) at n=256. That is consistent with aiter's stage 1 writing fp4 itself. Step 1 therefore moves y numerics *toward* the baseline, not away from it.

## Sequencing, risk and stop rules

- **Order and safety are right.** Step 1 is a compile flag plus two buffers, and it is a guaranteed KEEP once it validates. Steps 2 and 3 are independent of Step 1. Committing after each step and re-running `git status` addresses the iteration-1 revert. Starting `ys_sorted`/`xs_sorted` as `torch.zeros` before micro-optimising to `empty` is the right risk order.
- **Previous critic's points are resolved:**
  - the y-layout flag, from source;
  - joint tile_m: stage 2 takes any divisor of the sort block via `sort_block_m`;
  - torch `_mxfp4` cannot feed MFMA: the plan uses codes plus E8M0 throughout;
  - the ordering of the steps.
- **Lifting `ef15e878`'s route kernel is left as an option.** That is acceptable, since the single-workgroup LDS-atomic sort is small. Its varying within-expert order is covered by `bounded` determinism, and stage 2 already accumulates atomically.

## Notes to fold in during execution (not blocking)

1. **Compile keys for the new kernels.** Make M, R and `n_sorted` runtime arguments, not constexprs, so that 25 Points do not cause 25 separate compiles of the quant and sort kernels. Route them through the same `_compiled` / `_launch` path; `_compiled` is cached per n.
2. **Make the per-kernel trace a decision point, not just a measurement.** If, after Steps 1–3, our FlyDSL stage 1 at n=256 is more than about 10% slower than aiter's asm `t32x128x256` (100.5 µs), say so explicitly. In that case S cannot reach 1.03× from glue removal alone, and the next iteration should target S stage-1 config (tile_n, grid fill, kernarg preload), not L tile_m.
3. **`pos` dtype.** Until Step 3 lands, `pos` is int64 (8713). The `.int()` cast adds one node, which is fine. Step 3's kernel should emit int32 directly.
4. **Sort-kernel workgroup size.** Use 256 threads unless FlyDSL's 1024-thread launch bound has been confirmed. Put a barrier after zeroing the LDS counters and after the prefix phase. At L, if the single-workgroup sort takes more than about 150 µs, keep the torch sort for n > 1024, as the plan already allows.
5. **Expected scores.** The plan's 0.85–1.0 is argued, not measured. The KEEP gate is against the 0.179 incumbent, so any validated step banks. Do not treat failing to reach ≥1.03× on S Points this iteration as a failure of the route.
