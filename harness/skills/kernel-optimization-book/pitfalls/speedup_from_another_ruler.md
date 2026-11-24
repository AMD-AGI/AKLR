---
title: "A speedup measured by another harness does not predict the per-size speedup on the ruler that judges the kernel"
kind: pitfall
scope: [any kernel whose speedup is quoted from another ruler than the one that judges it, scores from agent tuning runs (AgentKernelArena and others), ad hoc timing probes]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
---

# A speedup from another ruler

## Symptom

The speedup in a submission's description or a tuning run's summary and the per-size speedup on the ruler that
judges the kernel disagree, in both directions, on the same device code. On the FlyDSL blockwise-scaled fp8 GEMMs
of one submission (15 kernels, 13 sizes each), judged by their lower-bound speedup (`../gemm/measurement.md` §1):

| Kernel | Submission's description, mean over its 13 cases *(field, E3)* | Lower-bound speedup, per size |
|---|---|---|
| N 512 × K 4096, `b` preshuffled, `a_scale` row-major | 1.7498× | 4.374× at m=1, 5.400× at m=32, 1.492× at m=4096 *(field, E1)* |
| N 4096 × K 1024, `b` preshuffled | 1.0377× | 0.807× (m=128) to 1.113× (m=4096); 7 of 13 sizes below 1.0 *(field, E2)* |
| N 4096 × K 512, `b` preshuffled | 1.1744× | 0.989× to 1.508×; m=1 to m=256, 9 of 13 sizes, below 1.03 *(field, E2)* |

The descriptions give one number per kernel, 1.0377× to 1.7498× *(field, E3)*; the lower-bound speedups of the 15
kernels' sizes run from 0.807× to 5.400× (E1, E2). In all 15 the decorated kernel and launcher functions are textually
identical under both rulers. In 14 nothing else that reaches the device changed: the output buffer is passed in rather
than allocated, and the entry point takes keywords. In the N 4096 × K 4096 kernel the output row stride those functions capture also
lost its padding.

The same holds inside one tuning run moved between machines. In an agent tuning run on a fused MoE resumed on
another MI355X, two candidates scored 1.5287 and 1.5266 against a best of 1.4451 scored on the first machine (+5.8%,
+5.6%); timed beside that best on the same card they were 1.0006 and 1.0013 faster, under the run's keep bar,
and both were reverted *(field, E4)*.

## Root cause

The two numbers come from different rulers, and the description's number is a mean over shapes.

- **The lower-bound speedup for a size** is the one-sided 95% Student-t lower bound of the mean log ratio of baseline
  to candidate time, here over 32 rounds. Each round's reading is the median of 10 graph replays of the per-call
  GPU kernel time recorded by Kineto, every call preceded by a 1 GiB read that flushes L2 and MALL; the arms
  run in a seeded random order each round.
- **Its baseline** is
  aiter's `gemm_a8w8_blockscale_bpreshuffle`; for the kernels whose `a_scale` is row-major, the timed baseline
  call also relayouts `a_scale` (`a_scale.T.contiguous().T`).
- **Both rulers ran FlyDSL 0.3.2**, so the version does not make the gap. The install step of the lower-bound
  runs ran a copy of aiter's `setup.py`, which replaced the
  image's FlyDSL (0.3.4.1 for one run, 0.3.1 for another) with its pin 0.3.2 (read from
  the build steps and `setup.py`, not checked in the runs' logs; `../pitfalls/measured_a_different_build.md`, S12). The
  description's numbers were taken on FlyDSL 0.3.2 too.
- **The description's ruler** (AgentKernelArena) alternated baseline and candidate three times on one MI355X, took per-case
  medians and averaged 13 cases (E3); its baseline call, cache state and input handling for these kernels are not
  recorded. For another task of the same pipeline, the timing is described as one CUDA graph
  captured over one set of input buffers and replayed.
- Which of these differences makes the gap at each size is not established.
- **A tuning run's "vs best"** divides a candidate's score by a best scored earlier, on whatever card the run used then.
  After a move to another card the ratio mixes two machines; the run's keep decision, which times the candidate beside
  the best on one card, does not.

## How to detect

- Read your result per size from the ruler that judges the kernel before you trust it: here the lower-bound
  speedup, with its flush before every call and its arms in random order,
  against the baseline that ruler names.
- A number that is a mean over shapes says nothing about any one size; compare size by size
  (`../CONTRIBUTING.md` § Evidence).
- Check that both rulers time the same build and the same baseline: `../pitfalls/measured_a_different_build.md` and
  `../pitfalls/baseline_silently_wrong.md`.

## Fix or workaround

- Report speedups per size, from the ruler that judges the kernel, against the faster of the kernel in use at that size and
  aiter.
- After a tuning run moves to another machine, read its candidates by the same-card comparison it keeps on, not by "vs best",
  and re-measure the best there before quoting any ratio.
- Use another harness's number for direction only. How to build a ruler that resolves a given effect is in
  `../gemm/measurement.md` §1–2 and §4; pricing the competitor and the operating point in §7.

## Scope

- Any kernel whose speedup was measured on another ruler than the one that judges it; the cases above are FlyDSL fp8 blockwise-scaled
  GEMMs on MI355X.
- The lower-bound speedups of the N 512 × K 4096 kernel at m=32 and m=4096 stayed within 4% over
  four runs (E1), so that kernel's disagreement is not the ruler's noise. Other sizes moved more across the same four
  runs, which also changed the image and the ruler's settings: the 15 kernels' 195 sizes by a median 6.4% and
  up to 36%, and by up to 9.5% between the two runs on one image *(field, E1)*. A gap of a few percent at one size
  can be the ruler's own spread.

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=4, lower-bound speedup (Kineto; 32 rounds, median of 10 graph replays per reading; the earliest of the four runs ran earlier settings of that ruler; their FlyDSL: the pin each run's install step leaves, not read from the runs' logs)] the N 512 × K 4096 kernel (`b` preshuffled, `a_scale` row-major) per size in one run: m=1 4.374, m=2 5.234, m=4 5.315, m=8 5.285, m=16 5.354, m=32 5.400, m=64 4.662, m=128 4.328, m=256 2.999, m=512 3.070, m=1024 2.577, m=2048 1.863, m=4096 1.492; over four runs m=32 read 5.278 / 5.264 / 5.194 / 5.400 and m=4096 1.479 / 1.511 / 1.475 / 1.492; over the same four runs, the kernels' sources unchanged, the max/min of the 15 kernels' 195 sizes had a median of 1.064 and a maximum of 1.363 (the N 4096 × K 4096 kernel with `b` preshuffled, m=1, 1.666 to 2.271), and between the last two runs, on the same image, a median of 1.009 and a maximum of 1.095 (FlyDSL fp8 blockwise-scaled GEMMs, N=512 K=4096 and the other 14 kernels of the same submission)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, the same ruler] N 4096 × K 1024 with `b` preshuffled: 0.980, 0.988, 1.057, 1.047, 1.044, 1.030, 0.955, 0.807, 0.868, 0.990, 0.919, 1.105, 1.113 at m=1…4096; N 4096 × K 512 with `b` preshuffled: 1.019, 0.993, 0.996, 0.997, 1.001, 0.994, 0.989, 0.992, 0.995, 1.065, 1.176, 1.508, 1.348; the 15 kernels' sizes span 0.807–5.400 (FlyDSL fp8 blockwise-scaled GEMMs)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AgentKernelArena: baseline and candidate alternating three times on one MI355X, per-case medians, mean over 13 cases] described speedups 1.7498 (N 512 × K 4096, `a_scale` row-major), 1.0377 (N 4096 × K 1024), 1.1744 (N 4096 × K 512); 1.0377–1.7498 over the 15 a8w8 kernels (FlyDSL fp8 blockwise-scaled GEMMs)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=3–5 pairs, an agent tuning run's paired keep test (candidate and best timed alternately on one card, on the per-kernel CUDA-graph timer)] after the run moved from one MI355X node to another, two candidates scored 1.5287 and 1.5266 against the best 1.4451 scored before the move; paired against it: mean 1.0006 (n_tokens 893 1.0000, 2732 1.0013) and 1.0013 (893 1.0002, 2732 1.0024), lower bounds 1.0004 and 1.0007 under the keep bar 1.0011; both reverted (fp8 × MXFP4 SiTUv2 fused MoE, EP8, 112 local experts, inter 3072, n_tokens 893 and 2732)
