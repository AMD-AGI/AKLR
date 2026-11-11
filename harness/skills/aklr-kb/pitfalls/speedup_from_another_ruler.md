---
title: "A speedup measured by another harness does not predict the per-Point speedup AKLR CI writes"
kind: pitfall
scope: [any kernel headed for AKLR, scores from agent optimization loops (KernelForge, AgentKernelArena), ad hoc timing probes]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "aklr/core/validation/stages/_bench/_report.py:85"
  - "aklr/core/utils/testing.py:164"
---

# A speedup from another ruler

## Symptom

The speedup in a PR description or a loop's summary and the per-Point `speedup` that CI writes into the schema
disagree, in both directions, on the same device code. On the FlyDSL blockwise-scaled fp8 GEMMs of one PR (15
kernels, 13 Points each):

| Kernel | PR description, mean over its 13 cases *(field, E3)* | AKLR CI, per Point |
|---|---|---|
| `b_shuffled_row_major_a_scale_n512_k4096` | 1.7498× | 4.374× at m=1, 5.400× at m=32, 1.492× at m=4096 *(field, E1)* |
| `b_shuffled_n4096_k1024` | 1.0377× | 0.807× (m=128) to 1.113× (m=4096); 7 of 13 Points below 1.0 *(field, E2)* |
| `b_shuffled_n4096_k512` | 1.1744× | 0.989× to 1.508×; m=1 to m=256, 9 of 13 Points, below 1.03 *(field, E2)* |

The descriptions give one number per kernel, 1.0377× to 1.7498× *(field, E3)*; CI's per-Point values for the 15
kernels run from 0.807× to 5.400× (E1, E2). In all 15 the decorated kernel and launcher functions are textually
identical to the PR's. In 14 nothing else that reaches the device changed: the output buffer is passed in rather
than allocated, and `run` takes keywords (S1). In `n4096_k4096` the output row stride those functions capture also
lost its padding.

The same holds inside one optimization loop moved between machines. In a KernelForge run on a fused MoE resumed on
another MI355X, two rounds scored 1.5287 and 1.5266 against a best of 1.4451 scored on the first machine (+5.8%,
+5.6%); timed beside that best on the same card they were 1.0006 and 1.0013 faster, under the round's keep bar,
and both were reverted *(field, E4)*.

## Root cause

The two numbers come from different rulers, and the description's number is a mean over shapes.

- **AKLR CI's `speedup` for a Point** is the one-sided 95% Student-t lower bound of the mean log ratio of baseline
  to candidate time over 32 rounds (S2). Each round's reading is the median of 10 graph replays of the per-call
  GPU kernel time recorded by Kineto, every call preceded by a 1 GiB read that flushes L2 and MALL (S3); the arms
  run in a seeded random order each round (S4).
- **The baseline is the contract's `<contract>_baseline`**, here `gemm_a8w8_blockwise_scaled_baseline`, which calls
  aiter's `gemm_a8w8_blockscale_bpreshuffle`; for the `b_shuffled_row_major_a_scale` variant, the timed baseline
  call also relayouts `a_scale` (S5).
- **The CI of these runs used FlyDSL 0.3.2** (in the repository before this one): the GPU job installed the
  checkout with `--no-deps` (S6), but that editable install ran the checkout's `setup.py`, which replaced the
  image's FlyDSL (0.3.4.1 for one CI run, 0.3.1 for another) with its pin 0.3.2 (S7; read from
  the workflow and `setup.py`, not checked in the runs' logs; `../pitfalls/measured_a_different_build.md`). The
  PR's numbers were taken on FlyDSL 0.3.2 too.
- **The PR's ruler** (AgentKernelArena) alternated baseline and candidate three times on one MI355X, took per-case
  medians and averaged 13 cases (E3); its baseline call, cache state and input handling for these kernels are not
  recorded. For another task of the same pipeline, the timing is described as one CUDA graph
  captured over one set of input buffers and replayed.
- Which of these differences makes the gap at each Point is not established.
- **A loop's "vs best"** divides the round's score by a best scored earlier, on whatever card the loop ran then.
  After a move to another card the ratio mixes two machines; the loop's keep decision, which times the round beside
  the best on one card, does not.

## How to detect

- Read your result per Point from AKLR's ruler before you trust it: `bench_cudagraph` (S3), arms in random order,
  against the contract's `<contract>_baseline`. `python -m aklr validate --base origin/main` runs that ruler
  locally; the local result does not count, CI decides (`aklr/ops/README.md` § Workflow).
- A number that is a mean over shapes says nothing about any one Point; compare Point by Point
  (`../CONTRIBUTING.md` § Evidence).
- Check that your probe and CI time the same build and the same baseline: `../pitfalls/measured_a_different_build.md` and
  `../pitfalls/baseline_silently_wrong.md`.

## Fix or workaround

- Report speedups per Point, from AKLR's ruler, against the faster of the kernel AKLR serves at that Point and
  aiter (`../SKILL.md`, the table under "Stop: what a AKLR kernel cannot do").
- After a loop moves to another machine, read its rounds by the same-card comparison it keeps on, not by "vs best",
  and re-measure the best there before quoting any ratio.
- Use another harness's number for direction only. How to build a ruler that resolves a given effect is in
  `../gemm/measurement.md` §1–2 and §4; pricing the competitor and the operating point in §7.

## Scope

- Any kernel whose speedup was measured outside AKLR's bench; the cases above are FlyDSL fp8 blockwise-scaled
  GEMMs on MI355X.
- CI's per-Point values of `b_shuffled_row_major_a_scale_n512_k4096` at m=32 and m=4096 stayed within 4% over
  four runs (E1), so that kernel's disagreement is not CI noise. Other Points moved more across the same four
  runs, which also changed the image and the measurement profile: the 15 kernels' 195 Points by a median 6.4% and
  up to 36%, and by up to 9.5% between the two runs on one image *(field, E1)*. A gap of a few percent at one Point
  can be CI's own spread.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=4, AKLR CI bench (32 rounds, median of 10 graph replays per reading, cold L2, one-sided t lower bound; the earliest of the four runs predates measurement profile 17; their FlyDSL: the pin each run's `setup.py` installs (S7), not read from the runs' logs)] `b_shuffled_row_major_a_scale_n512_k4096` per Point in one CI run: m=1 4.374, m=2 5.234, m=4 5.315, m=8 5.285, m=16 5.354, m=32 5.400, m=64 4.662, m=128 4.328, m=256 2.999, m=512 3.070, m=1024 2.577, m=2048 1.863, m=4096 1.492; over four CI runs m=32 read 5.278 / 5.264 / 5.194 / 5.400 and m=4096 1.479 / 1.511 / 1.475 / 1.492; over the same four runs, the kernels' sources unchanged, the max/min of the 15 kernels' 195 Points had a median of 1.064 and a maximum of 1.363 (`b_shuffled_n4096_k4096` m=1, 1.666 to 2.271), and between the last two runs, on the same image, a median of 1.009 and a maximum of 1.095 (FlyDSL fp8 blockwise-scaled GEMMs, N=512 K=4096 and the other 14 kernels of the same PR)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the same CI bench] `b_shuffled_n4096_k1024`: 0.980, 0.988, 1.057, 1.047, 1.044, 1.030, 0.955, 0.807, 0.868, 0.990, 0.919, 1.105, 1.113 at m=1…4096; `b_shuffled_n4096_k512`: 1.019, 0.993, 0.996, 0.997, 1.001, 0.994, 0.989, 0.992, 0.995, 1.065, 1.176, 1.508, 1.348; the 15 kernels' Points span 0.807–5.400 (FlyDSL fp8 blockwise-scaled GEMMs)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AgentKernelArena: baseline and candidate alternating three times on one MI355X, per-case medians, mean over 13 cases] described speedups 1.7498 (aslogical n512_k4096), 1.0377 (asraw n4096_k1024), 1.1744 (asraw n4096_k512); 1.0377–1.7498 over the 15 a8w8 kernels (FlyDSL fp8 blockwise-scaled GEMMs)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=3–5 pairs, KernelForge paired keep (round and best timed alternately on one card, AKLR graph timer)] after the run moved from one MI355X node to another, rounds 15 and 16 scored 1.5287 and 1.5266 against the best 1.4451 of round 13; paired against it: mean 1.0006 (n_tokens 893 1.0000, 2732 1.0013) and 1.0013 (893 1.0002, 2732 1.0024), lower bounds 1.0004 and 1.0007 under the keep bar 1.0011; both REVERT_PERF (fp8 × MXFP4 SiTUv2 fused MoE, EP8, 112 local experts, inter 3072, n_tokens 893 and 2732)
- S1 the predecessor repository's `b_shuffled_row_major_a_scale_n512_k4096` schema file (2026-10): against the PR's version only `launch` takes `out` and `run` takes keyword arguments; the sources have not changed since main's kernels were imported unedited, and the four later commits to that schema file wrote measurements only
- S2 aklr/core/validation/stages/_bench/_report.py:85: `speedup` is the graph ratio lower bound against the baseline, `ratio_lower_bound` at aklr/core/validation/stages/_measure.py:144, 32 rounds
- S3 aklr/core/utils/testing.py:164: `bench_cudagraph`, the flush at :70-75, the median of 10 replays
- S4 aklr/core/validation/stages/_measure.py:286: a seeded shuffle of the participants every round
- S5 aklr/ops/gemm/baseline.py:58 `gemm_a8w8_blockwise_scaled_baseline` relays out a row-major `a_scale` (`a_scale.T.contiguous().T`, :59) inside the timed call
- S6 `pip install --no-build-isolation --no-deps -e .` in the image `AKLR_CI_IMAGE`
- S7 setup.py, in the repository before this one: an editable install (`is_develop_mode`) without `AITER_TRITON_ONLY` runs `pip install flydsl==0.3.2` (`FLYDSL_VERSION`) whenever another FlyDSL version is installed; the parents of the four runs of E1 carry the same block and pin and install the checkout as S6 does
