# Review: flydsl_gemm_wait_free_fold

- How to write it: every step read in the delivered kernel (`gemm_s_fixwork` workspace at its HEAD, the source of
  `6bf403875a978f7e.json` on `kyle/k3-six`), lines cited S1-S5.
- E1, E2: the fix sessions' interleaved timing tables, driver `--bench-mode` (AKLR's graph timer); figures in the
  text recomputed from the tables (22% = 68.6 / 56.2, 1.7-2.9% = tree over flat).
- E3: stress outputs read line by line; the spin kernel is the campaign's delivered-then-withdrawn version, the
  wait-free one is the kernel delivered on `kyle/k3-six`; also `driver.py --stress` on the new forge_task branch.
- AKLR validate with main's timer passes both wait-free kernels (validate_ci_0207: gemm m 1-128 1.26-1.72x,
  m 256/512 1.12x/1.16x per Point against aiter, the faster at those Points).
- Duplicates: `gemm/flydsl_gemm_small_m.md` §4 covers split-K choice and race guidance; it is being edited in this
  change so as not to recommend a fold that waits. This page is the one lever, linked from there once approved.
