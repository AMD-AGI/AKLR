# Iteration 4 record

Unless marked as driver results, timings are per-kernel µs from the PyTorch profiler (10 calls) in `/tmp/w4/sweep.py`. A tuple means `_config` = (tile_m, s1_tn, s1_wpe, s1_async, s1_bnt[, s2_tm, s2_tn, s2_wpe, s2_async, s2_cumul]).

**0. Starting point and aiter profile (diagnostic, no edits).**
- Iter-3 driver bench median: 10.519 ms.
- aiter stage 1 / stage 2 time: 144/154 µs at n=3112 (t128) and 870/1324 µs at n=32768 (t64). Its stage 1 runs with `use_async_copy=True` and `b_nt=0`.

**1. Stage-1 knobs at tile_m=64** (completed). Stage-1 time at n=3112 / n=32768:

| Configuration | n=3112 | n=32768 |
|---|---|---|
| (3,F,2), the iter-3 settings | 220.5 | 1500.7 |
| (4,T,0) | 203.2 | 860.1 |
| (3,T,0) | 204.7 | 859.5 |
| (4,F,0) | 205.1 | 857.4 |
| (4,T,2) | 219.8 | 1495.7 |

**2. The same knobs at small n** (completed, worse, not adopted). Stage-1 time at n=256 / n=640:

| Configuration | n=256 | n=640 |
|---|---|---|
| (3,F,2), the iter-3 settings | 100.9 | 104.8 |
| b_nt=0 | 115.0 | 120.1 |
| b_nt=1 | 115.7 | 120.5 |
| async + b_nt=0 | 114.8 | 119.5 |
| wpe=4 + b_nt=0 | 115.4 | 120.2 |

HELD-FIXED: tile_m=32, s1_tn=128, s2 tile_m=32, s2_tn=128, s2 wpe=None, s2 async=False

**3. tile_m=128 in both stages** (completed). At n=3112 / n=3816:
- Stage 1 improved from 206/228 to 152/171 µs.
- Stage 2 got worse, from 146.7/167.6 to 154.9/187.2 µs.
- wpe=1 and wpe=2 measured the same; async=False measured the same.

HELD-FIXED: s1_tn=128, s1_bnt=0, s2_tn=128, s2 wpe=None, s2 async=False

**4. Stage-1 t128 with stage-2 t64 (sort_block_m=128)** (adopted). Total µs against (64,128,3,T,0,64,128):

| n_tokens | t128 / t64 split | (64,…,64) |
|---|---|---|
| 3112 | 364.7 | 419.8 |
| 3816 | 426.9 | 473.2 |
| 10000 | 981.5 | 1052.8 |
| 13152 | 1272.9 | 1290.9 |
| 32768 | 2754.3 | 2783.0 |

- Error against iter 3 rose at n ≥ 10000: rel 7.2–7.9e-4, against 3.3–4.2e-4.
- At n=10000, the largest per-token error was 3.6e-3, against 3.5e-3 for a rerun of the tile_m=64 config. No token exceeded 1e-2.
- Two other tile_m=64 variants: (64,128,4,T,0) measured 1282.4 at 13152 and 2793.7 at 32768.

**5. Multi-workgroup routing sort for n > 1024** (adopted).
- It is a histogram kernel plus a scatter kernel.
- Its output matched the torch sort exactly at n=3112, 13152 and 32768 on the first build.
- Sort time (histogram + scatter, µs) by chunk size:

| Chunk | n=3112 | n=13152 | n=32768 |
|---|---|---|---|
| 512 | 11.8 | 12.7 | 16.1 |
| 1024 | 14.2 | 14.2 | 15.7 |
| 2048 | 21.1 | 20.4 | 23.9 |
| 4096 | 33.8 | 32.8 | 31.2 |

- One earlier timing ran while a driver correctness run was in progress, so it is contaminated.

**6. Stage-2 variants** (stage-2 time, µs):

| Variant | n=3112 | n=32768 |
|---|---|---|
| base | 146.4 | 1304.5 |
| async (adopted) | 137.8 | 1299.6 |
| wpe=4 + async | 137.4 | 1303.7 |
| tile_n=256 | 147.4 | 1356.0 (rel error 9.0e-4) |

HELD-FIXED: (128,128,2,T,0), s2_tm=64, sort_block_m=128, chunk 2048, cu_num_mul=1

**7. Stage-2 cu_num_mul** (stage-2 time, µs):

| cu_num_mul | n=3816 | n=32768 |
|---|---|---|
| 1 | 167.0 | 1312.0 |
| 2 (adopted) | 166.0 | 1285.9 |
| 3 | 170.0 | 1278.5 |
| 3 + wpe=4 | 170.1 | 1280.0 |

HELD-FIXED: (128,128,2,T,0,64,128,None,T), chunk 512 for n ≤ 16384 and 1024 above

**8. Final file.**
- Driver bench: mean 1.049×, which I computed from the case times against the baseline times. Every case is ≥ 1.027×.
- Driver correctness run on the final file: `allclose: True`.
- The docstring was also edited.

NEGATIVES: b_nt=0, b_nt=1 and async+b_nt=0 at tile_m=32 (n=256, 640); stage-2 tile_m=128 (n=3112, 3816); stage-2 tile_n=256 (n=3112, 32768); sort chunk 2048 and 4096 compared with 512 or 1024; cu_num_mul=3 at n=3816; tile_m=128 raised the error at n ≥ 10000; (64,128,4,T,0) at n=32768.
DISPROOF: none
KB: none

SCOPE: measured on n_tokens=10000, n_tokens=13152, n_tokens=256, n_tokens=272, n_tokens=288, n_tokens=304, n_tokens=3112, n_tokens=320, n_tokens=32768, n_tokens=336, n_tokens=352, n_tokens=368, n_tokens=3816, n_tokens=384, n_tokens=400, n_tokens=416, n_tokens=424, n_tokens=432, n_tokens=448, n_tokens=464, n_tokens=480, n_tokens=496, n_tokens=512, n_tokens=576, n_tokens=640 | held fixed tile_m=32, s1_tn=128, s2_tn=128, wpe=None, async=False, s1_bnt=0, s2_tm=64, sort_block_m=128, cu_num_mul=1 | carries a measured negative | no feasibility claim

OUTCOME: KEEP | mean case speedup 1.047726x vs best 1.047726x | wall 8.4918 ms vs best 8.4918 ms | session ended: candidate_submitted
