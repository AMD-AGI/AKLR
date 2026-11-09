# Iteration 3 record

All timings below come from a scratch CUDA-graph timer in `/tmp/w3/cmp.py`, which compares against the iter-2 kernel, unless marked as driver results.

## 1. Stage 1 writes MXFP4 directly (completed, kept)

I compiled stage 1 with `out_dtype="fp4"` and passed `ys_sorted` in the `arg_out_scale_sorted` slot. The torch `_quant(y)` and `_sorted_scales` calls were removed.

- **As first built, the output was wrong.** Relative error against iter 2 was 0.170 at n_tokens=256, 3816 and 32768, with 85% of elements outside tolerance. Speed was 734→580 µs at 256, 3115→2307 µs at 3816 and 24306→17193 µs at 32768.
- **Diagnosis.** The sorted scales matched the torch path on 99.997% of bytes. The fp4 codes matched on only 83.9%. The dominant mismatch was code 6→5 at the clamp value 56.
- **Fixes.** Two changes were made in the epilogue:
  - round each value to bf16 before quantizing it;
  - change the E8M0 rounding constant from `+0x400000` to `+0x3FFFFF`, so the exponent rounds up only strictly above 1.5×.
- **After the fixes.** Scales matched on 100% of bytes and codes on 99.997% (3 bytes differed, 0 vs 8, i.e. +0 vs −0). Relative error was 9.3e-4, 3.0e-4 and 2.2e-4. Speed was 581, 2320 and 17186 µs.
- **Profile after this step.** A call launched 100 kernels at n=256 (655 µs in total) and 107 at n=32768 (17231 µs).

## 2. FlyDSL x-quant kernel that also scatters the scales and zeroes `out` (completed, kept)

The kernel worked on the first build.

- Speed was 340, 647 and 3336 µs, against 738, 3129 and 24287 µs for iter 2.
- The full driver correctness run reported `allclose: True`.

## 3. FlyDSL routing sort in a single workgroup (completed, kept)

- **First build failed.** The FlyDSL compile raised `TypeError: Invalid attribute value ... arith.constant (std::bad_cast)`. The cause was a Python `while` loop that the DSL converted into an scf while-loop. Replacing it with a precomputed tuple of search steps fixed it.
- **Exactness.** At n=256, 3112 and 32768, every sort output matched what the torch sort implies: each route's placement, the padding, the per-block expert ids and the valid-row count.
- **Per-kernel profile, µs per call:**

| n_tokens | sort | x-quant | stage 1 | stage 2 | zero-fill |
|---|---|---|---|---|---|
| 256 | 8.9 | 4.7 | 100.9 | 59.6 | 7.4 |
| 3816 | 51.4 | 27.9 | 247.3 | 166.8 | 9.2 |
| 32768 | 392.4 | 227.4 | 1497 | 1346 | 11.0 |

- **Error at n=32768 rose** from 2.2e-4 to 6.5–7.0e-4. Because the sort is exact, this is attributed (not measured) to the order of bf16 atomic adds in stage 2.

## 4. Scale buffers allocated with `torch.empty` instead of `zeros` (completed, kept)

- Speed was 180.3, 181.1, 434.5 and 3471 µs at n_tokens=256, 272, 3112 and 32768.
- The driver correctness run reported `allclose: True`.
- **Driver bench** (`--warmup 3 --iters 20`): the mean speedup, computed against the baseline times, was 0.988×. The 20 workloads with n_tokens ≤ 640 ran at 1.024–1.052×. The other five ran at 0.786× (3112), 0.791× (3816), 0.830× (10000), 0.779× (13152) and 0.724× (32768).

## 5. Final cleanup (completed)

I removed `empty_u8` and rewrote the module docstring. The driver correctness run reported `allclose: True`. This exact file was not re-benchmarked; a smoke test at n=256 measured 178.8 µs.

HELD-FIXED: tile_m=32 (≤1024) / 64, s1 tile_n=128, tile_k=256, waves_per_eu=3, b_nt=2, v2_output_layout=False; s2 tile_n=128, atomic, persist_m=-1 when blocks>256; cases n_tokens=256, 3816, 32768

NEGATIVES: stage-1 fp4 epilogue as first built (fp32 values not rounded to bf16, `+0x400000` E8M0 rounding) gave wrong output (rel err 0.170 at n_tokens=256, 3816, 32768); fixed within the session

DISPROOF: none

KB: none

SCOPE: measured on n_tokens=10000, n_tokens=13152, n_tokens=256, n_tokens=272, n_tokens=288, n_tokens=304, n_tokens=3112, n_tokens=320, n_tokens=32768, n_tokens=336, n_tokens=352, n_tokens=368, n_tokens=3816, n_tokens=384, n_tokens=400, n_tokens=416, n_tokens=424, n_tokens=432, n_tokens=448, n_tokens=464, n_tokens=480, n_tokens=496, n_tokens=512, n_tokens=576, n_tokens=640 | held fixed tile_m=32 (≤1024) / 64, tile_n=128, tile_k=256, waves_per_eu=3, b_nt=2, v2_output_layout=False, persist_m=-1 when blocks>256, n_tokens=256 | carries a measured negative | no feasibility claim

OUTCOME: KEEP | mean case speedup 0.987974x vs best 0.987974x | wall 10.5277 ms vs best 10.5277 ms | session ended: candidate_submitted
