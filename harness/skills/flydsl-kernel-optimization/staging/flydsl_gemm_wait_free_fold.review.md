# Review: flydsl_gemm_wait_free_fold

- How to write it: every step read in the delivered kernel (`gemm_s_fixwork` workspace at its HEAD, the source of
  `e8a254fd87b9a03e.json` on `kyle/k3-six`), lines cited S1-S5.
- E1, E2: the fix sessions' interleaved timing tables, driver `--bench-mode` (AKLR's graph timer); figures in the
  text recomputed from the tables (22% = 68.6 / 56.2, 1.7-2.9% = tree over flat).
- E3: stress outputs read line by line; the spin kernel is the campaign's delivered-then-withdrawn version, the
  wait-free one is the kernel delivered on `kyle/k3-six`; also `driver.py --stress` on the new forge_task branch.
- E4: the two versions' interleaved `--bench-mode` table, two rounds each, on the cards that ran stress.
- Historical validation before the state interface was removed passed both wait-free kernels with counters in state
  (validate_ci_gemm_state_0823: gemm m 1-128 1.23-1.71x, m 256/512/1024 1.11x/1.15x/1.04x per Point against aiter,
  the faster at those Points; m 2048/4096 1.016x/0.999x, not served, and untouched by the change).
- Duplicates: `gemm/flydsl_gemm_small_m.md` §4 covers split-K choice and race guidance; it is being edited in this
  change so as not to recommend a fold that waits. This page is the one lever, linked from there once approved.

The rebase onto stateless AKLR initializes counters on every call. The historical state-based timings above
do not validate the revised entry points; their performance must be measured again by GPU CI.
