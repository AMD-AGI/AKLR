---
title: "MXFP4 fused MoE, hidden 6144, top-9, 16×16-shuffled weights (257 experts at inter 256 or 512, 65 and 33 experts) — no sort and two launches at 1–2 tokens; the sort folded into stage 1 or one prelude launch, mostly with an atomic stage 2, at 4–128; from 256 tokens sorted row tiles whose bf16, fp8 or MXFP4 route rows a separate launch sums in a fixed order, or an atomic stage 2"
kind: case
ops: [moe]
dtypes: [mxfp4]
regimes: ["tokens 1-2", "tokens 4-128", "tokens 256-4096"]
accuracy: bounded
gens: [gfx950]
flydsl: 0.3.2
basis: measured N=1
measured:
  - {metric: "speedup over aiter's fused MoE, per token count", value: "0.137–2.489; 2 of 13 token counts ≥ 1.03 (m 1, 2)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 256; the all-m kernel, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.079–1.113; 4 of 4 token counts ≥ 1.03 (m 8, 16, 32, 64)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 256; the fold kernel, m 8, 16, 32, 64", date: "2026-11-05"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "0.973–1.670; 10 of 13 token counts ≥ 1.03 (all but m 8, 16, 32)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 33 experts, top-9, inter 2048; m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "0.811–1.316; 4 of 13 token counts ≥ 1.03 (m 1, 2, 4, 1024)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 65 experts, top-9, inter 2048; the all-m kernel, m 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.246–1.800; 4 of 4 token counts ≥ 1.03 (m 512, 1024, 2048, 4096)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 256; the large-m kernel, measured on its earlier build (same GPU code, not re-measured), m 512, 1024, 2048, 4096", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.049; 1 of 1 token count ≥ 1.03 (m 32)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 512; the fold kernel, measured on its earlier build (same GPU code, not re-measured), m 32; measured as the same kernel in a build that also ran m 64", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.069–1.668; 4 of 4 token counts ≥ 1.03 (m 512, 1024, 2048, 4096)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 257 experts, top-9, inter 512; the large-m kernel, measured on its earlier build (same GPU code but for one buffer bound, not re-measured), m 512, 1024, 2048, 4096; measured as the same kernel in a build that also ran m 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.123–1.193; 6 of 6 token counts ≥ 1.03 (m 4, 8, 16, 32, 64, 128)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 65 experts, top-9, inter 2048; the mid-m kernel, measured on its earlier build (same GPU code, not re-measured), m 4, 8, 16, 32, 64, 128", date: "2026-11-09"}
  - {metric: "speedup over aiter's fused MoE, per token count", value: "1.230–1.365; 3 of 3 token counts ≥ 1.03 (m 256, 2048, 4096)", gpu: MI355X, workload: "bf16 tokens × MXFP4 experts in aiter's 16×16 preshuffle, MXFP4 activations, hidden 6144, 65 experts, top-9, inter 2048; the large-m kernel, measured on its earlier build (same GPU code, not re-measured), m 256, 2048, 4096", date: "2026-11-09"}
---

# MXFP4 fused MoE with 16×16-shuffled weights, hidden 6144, top-9

Nine FlyDSL kernels for a fused MoE layer of four shapes: 257 experts with inter dim 256 and with inter dim 512, 65
experts and 33 experts with inter dim 2048. Each is named below by its layer (experts / inter dim, or the expert count
alone at inter dim 2048) and by the token counts it is built for:

| Kernel | Layer | Built for m | Source |
|---|---|---|---|
| 257/256 all-m | 257 experts, inter 256 | 1–4096 | one file |
| 257/256 fold | 257 experts, inter 256 | 8, 16, 32, 64 | eight files |
| 257/256 large-m | 257 experts, inter 256 | 512–4096 | its own file beside copies of aiter modules, 16 files in all |
| 257/512 fold | 257 experts, inter 512 | 32 | its own file beside copies of aiter modules, 10 files in all |
| 257/512 large-m | 257 experts, inter 512 | 512–4096 | its own file beside copies of aiter modules, 20 files in all |
| 65-expert all-m | 65 experts, inter 2048 | 1–4096 | one file |
| 65-expert mid-m | 65 experts, inter 2048 | 4–128 | its own file beside copies of aiter modules, 12 files in all |
| 65-expert large-m | 65 experts, inter 2048 | 256, 2048, 4096 | one file |
| 33-expert | 33 experts, inter 2048 | 1–4096 | one file |

The 257/256 all-m, 33-expert and 65-expert all-m kernels were measured in two or more runs and the 257/256 fold kernel
in one (the first table of § Result); the other five are newer and, but for one separate reading of the 257/512 fold
kernel, have this page's runs only. "m" is the tokens in the call, the input batch, not the rows an expert receives. A
route is one (token, expert) pair, 9m per call.

The five newer kernels replace earlier builds that this page's runs measured and that imported aiter at run time, one
each. Their own kernel builders are unchanged, the aiter modules the earlier builds imported are carried as copies from
the aiter commit they were measured with (dbd8bf5bd), and the code paths that none of their token counts reached are
removed. So their GPU code is the measured builds', except that the 257/512 large-m kernel's head launch bounds its
sorted-id reads and its stage 1 at 1024 tokens re-sorts the activation scales (Counter-signals). The measurements below
are of the earlier builds, but for E18 on the 257/512 large-m kernel.

## Applies when

- `w1` (experts, 2 × inter, 6144) and `w2` (experts, 6144, inter) as fp4 e2m1 in aiter's 16×16 preshuffle, one E8M0
  scale per 1×32, shuffled and padded to (256, 8); bf16 `hidden_states`, fp32 `topk_weights`, int32 `topk_ids`, no
  expert mask; SiLU on separate gate and up rows, routing weight applied in stage 2, activations quantized to MXFP4;
  bf16 `out`. A token's 9 expert ids are distinct (a precondition on `topk_ids`); none of the four layers has an expert
  mask or a count of local tokens.
- Accuracy is bounded, by one bound on the whole output: SQNR ≥ 13 dB against a reference computation of the layer in
  bf16, with no per-element bound. Two runs need not give the same bytes, so atomic accumulation is allowed.
- The model: the 257/256 fold kernel was written for GLM-5.2's MXFP4 MoE (256 routed experts plus the fused shared one,
  top-9) at decode sizes. The four shapes fit that layer (hidden 6144, MoE intermediate 2048, `config.json` of
  `zai-org/GLM-5.2`): inter 256 and 512 are 2048 split 8 and 4 ways, 65 and 33 experts are 64 or 32 routed experts plus the shared one. The other kernels' records
  name no model.
- Token counts. The yardstick is aiter's `fused_moe`, judged per token count, not as a mean over them; a token count
  counts as won from 1.03 over it, and where two kernels of a layer cover one token count, the faster is the yardstick
  for the other. The four kernels of § Result's first table, the three all-m kernels by their latest run and the
  257/256 fold kernel by its one run *(field, E1)*:

  | Layer | At or above 1.03 | Below 1.03, or no kernel measured |
  |---|---|---|
  | 257 experts, inter 256 | m 1, 2 (all-m); m 8–64 (fold) | m 4, 128–4096 |
  | 257 experts, inter 512 | none (the layer has only its two newer kernels, which that run did not measure) | all 13 token counts |
  | 65 experts, inter 2048 | m 1, 2, 4, 1024 (all-m) | m 8–512, 2048, 4096 |
  | 33 experts, inter 2048 | m 1, 2, 4, 64–4096 | m 8, 16, 32 |

  The five newer kernels are built for these token counts (this page's runs of their earlier builds
  *(field, E4, E5, E6, E7, E8)*); they raise on any other token count:

  | Layer | Built for, by the five newer kernels |
  |---|---|
  | 257 experts, inter 256 | m 512–4096 (large-m) |
  | 257 experts, inter 512 | m 32 (fold; 1.043 in a separate run *(field, E20)*); m 512–4096 (large-m) |
  | 65 experts, inter 2048 | m 4–128 (mid-m); m 256, 2048, 4096 (large-m) |

  At every one of those token counts but one the layer has no other kernel with a measurement there, or that kernel
  reads below 1.03 *(field, E1)*, so the yardstick there is aiter. The exception is m 4 of the 65 experts, which the
  mid-m kernel covers and where the all-m kernel read 1.031 in its latest run: in a run where the all-m kernel reaches
  1.03 at m 4, it, not aiter, is the yardstick there.

## Counter-signals

**Every regime**

- **Output bits vary where stage 2 adds with atomics.** The 257/256 fold kernel (m 8–64), the 257/512 fold kernel (m
  32), the 65-expert mid-m kernel (m 4–128) and the 257/512 large-m kernel at m 512 add the weighted rows into `out`
  with bf16 atomics in arrival order; the 257/256 fold kernel's output read a run-to-run SQNR of 48.4–51.9 dB
  *(field, E3)*, the others' is not measured. Where two runs must give the same bytes they need a fixed-order stage 2
  (`../moe/few_tokens.md`, Counter-signals). The other kernels and token counts sum each token's routes in a fixed
  order.
- **Precision is spent where the SQNR bound allows it.** The 33-expert kernel stores its stage-2 rows as fp8 e4m3 with
  no scale at m 16, 32, 64 and 128–4096 (`cvt_pk_fp8_f32`) before the route sum; the 65-expert all-m kernel truncates,
  rather than rounds, its routing-weighted bf16 rows and its bf16 output at every m (`bits >> 16`); the 257/256 all-m
  kernel at m ≥ 512, where it is below 1.03, stores int8 rows with one fp32 scale per 32 values. The 257/256 large-m
  kernel (every token count), the 257/512 large-m kernel (m 1024–4096) and the 65-expert large-m kernel (every token
  count) store each route's stage-2 row as MXFP4, e2m1 with an E8M0 scale per 32, before the route sum. The runs record
  that the accuracy check passed, not the SQNR. Against a bound relative to aiter's own error, which is not the 13 dB
  bound these kernels are held to (§ Applies when), the 33-expert kernel's fp8 rows fail at 31.4–31.5 dB against
  aiter's 46.9–47.3 dB, and its fp32 or bf16 rows pass; the 65-expert all-m kernel fails at 26.5–30.3 dB, and rounding
  to nearest even instead of truncating moves it by under 0.1 dB, so rounding is not what limits it *(field, E19)*. The
  MXFP4 route rows are not measured against such a bound.
- **Launches and scratch kept per shape in the three all-m kernels.** The 257/256 all-m, 33-expert and 65-expert all-m
  kernels keep built launches in a module-level dict filled from the entry function, not in a `functools.cache` compile
  function, allocate their per-shape scratch on device `"cuda"` rather than an argument's device, and the 257/256 all-m
  kernel hashes its own source file (`Path(__file__).read_bytes()`) on a shape's first call: file I/O in launch code.
  Their scratch is written by every call before it is read (the 257/256 all-m kernel zeroes its expert counters in its
  input-quantization launch at m ≥ 32; the others store their counts per call). The other six kernels use
  `functools.cache` and allocate their workspace per call.
- **Concurrent calls of the three all-m kernels give wrong output.** Their per-shape scratch sits in the launch that the
  module-level dict keeps, so two calls of one shape at once on two streams share it. With different inputs (20 tries
  per token count, on two unrestricted streams and on two streams limited to the same first 64 CUs, each compared with
  an eager call alone), the 33-expert kernel gave wrong output at every token count in 16–20 of 20 tries, the 65-expert
  all-m kernel at every token count from m 2 in 18–20 of 20 and at m 1 on the shared CUs, and the 257/256 all-m kernel
  at m 2 (3 of 20) and m 8 (20 of 20) on the shared CUs and at m 4 and from m 16 up in 19–20 of 20; at m 4096 a GPU
  memory access fault ended its process after the unrestricted tries. The 257/256 fold kernel and the earlier builds of
  the five newer kernels gave 0 of 20 at every token count *(field, E16, E15)*.
- **No aiter at run time in the five newer kernels.** None of the five imports aiter. The 257/512 fold, 65-expert mid-m,
  257/256 large-m and 257/512 large-m kernels carry the aiter modules they use as copies from aiter dbd8bf5bd: whole
  files, except three host functions of aiter's `fused_moe.py` in the 257/512 large-m kernel, changed only in imports
  and host code: environment knobs fixed at their default values, a cross-call cache removed, a `.tolist()` written as a
  list, an argument named `input` renamed, streams taken from the input's device, a few package-level names from a small
  stub module, and imports of modules not copied, in code none of their token counts reaches, replaced by a
  `NotImplementedError`. The measured runs' environment is not recorded; if it set one of those knobs, what they ran
  differs from the newer kernels (code read). Each kernel's configuration per m is a table in its own source, and a
  token count outside it raises instead of falling back to aiter. The 257/256 large-m kernel fixes the block sizes and
  stage-1 kernels that its earlier build took from aiter's tuned table (`glm5_fp4_tuned_fmoe.csv` at dbd8bf5bd) where
  its own tables named none; the 257/512 large-m kernel fixes the plan that its earlier build checked against aiter's
  configuration, runs its own driver where that ran aiter's `fused_moe_2stages`, and carries aiter's stage-1 builder as
  three edited copies where the earlier build edited aiter's source at run time. Installing another aiter does not
  change what runs.
- **The 257/512 large-m kernel's head launch bounds its sorted-id reads.** At m 2048 and 4096 the last 32-row tile of
  the head launch's scale re-sort reads sorted ids up to index 51,327 and 86,207 (9 past the end) from a buffer of
  51,319 and 86,199. Its earlier build read them through an unbounded buffer descriptor; the kernel gives the descriptor
  the buffer's size, so those reads return 0. The rows they feed lie past the valid count and their scales were already
  stored as 0, so the output does not change (code read).
- **At 1024 tokens the 257/512 large-m kernel's stage 1 does not read the activation scales in token order.** With that
  read (the build before), two calls with different inputs on two streams gave a wrong output in about 0.35% of outputs:
  a few tokens each with one route row wrong, by up to the output's mean magnitude. A single call beside an unrelated
  GEMM did too, and stage 2 rerun alone on the captured stage-1 output reproduced the wrong rows, so the error arises in
  stage 1 or before it; the cause inside that path is not found. The kernel re-sorts the scales at 1024 as at 2048 and
  4096: 0 wrong outputs in 1000 at the same speed, 1.161 at m 1024 *(field, E18)*. m 512 keeps the token-order read: 0
  of 1200 concurrent outputs had a row off by more than 4, where the atomic stage 2 moves rows by up to 2.5 alone
  *(field, E18)*.
- **Code no measured token count reaches:** in the 33-expert kernel, two routing kernels that take route ranks with
  atomics, a routing kernel in which each wave owns one expert at a time, a stand-alone input quantization and a
  stand-alone activation of split stage-1 partials; in the 65-expert all-m kernel, the torch sort fallback, a
  stand-alone quantization kernel and a kernel that combines and quantizes split stage-1 partials; in the 257/256 fold
  kernel, the split-K exchange, n-fusion, item-direct head and k-rotation of its stage 1 and the non-atomic epilogues of
  its stage 2; in the 257/256 large-m kernel, the fp8-row, atomic and persistent branches of its stage-2 GEMM copy,
  which its constants turn off at every token count; in the four kernels with copied aiter modules, most of those
  modules' functions (those that raise `NotImplementedError` included). No measurement here covers them.

**1–2 tokens**

- **65 experts at m 2 and m 4 sit near 1.03.** The all-m kernel reads 1.085 at m 2 and 1.031 at m 4 in its latest run
  *(field, E1)*, 1.050 and 0.979 in the run before *(field, E17)*, and 0.794, 1.049, 0.989, 1.014, 1.024 at m 2 across
  five earlier runs of the unchanged kernel *(field, E2)*: which run is read decides whether m 2 and m 4 count as won.

**4–128 tokens**

- **257 experts, inter 256, at m 4 and 128: no kernel reaches 1.03.** The fold kernel is not built for them and the
  all-m kernel reads 0.137 and 0.314 *(field, E1)*. At m 4, 16, 128, 512 and 1024 the all-m kernel repacks `w1`, `w2`
  and their scales into a record layout on every call (two extra launches, one per weight with its scales): about 0.64
  GB read and as much written per call (estimated from the tensor sizes). Those five token counts read 0.137–0.431, its
  other token counts below 1.03 0.741–0.925 *(field, E1)*. A kernel that keeps nothing derived from its arguments
  between calls cannot keep a repacked weight, and the 16×16-preshuffled layout is the one its stages read without
  repacking.
- **At m 8–64 two 257-expert inter-256 kernels cover the same token counts:** the fold kernel reads 1.079–1.113; the
  all-m kernel reads 0.180–0.925 there *(field, E1)*, so the fold kernel is 1.20–6.0× faster (estimated from the two
  ratios).
- **257 experts, inter 512: m 64 and m 128 have no kernel.** The fold kernel, as its earlier build, read 1.025 at m 64
  (median 181.7 µs against aiter's 186.2) and the large-m kernel 1.014 at m 128 (203.9 against 207.0 µs), both below
  1.03 *(field, E5, E6)*; both kernels raise at those token counts.
- **The 257/512 fold kernel's m 32 is the token count closest to 1.03.** Its earlier build's lower bound read 1.049 and
  1.040 in two runs, the second 0.010 above 1.03; a run below 1.03 leaves m 32 without a winning kernel, since the layer
  has no other kernel there *(field, E5)*. The 257/512 large-m kernel's m 512 read 1.069 and 1.067 *(field, E6)*. The
  33- and 65-expert runs below show how far such margins move between runs.
- **33 experts: m 8, 16 and 32 are below 1.03; m 4 and 64 sit near it.** In the latest run: m 4 1.052, m 8 1.021, m 16
  0.973, m 32 1.022, m 64 1.053 *(field, E1)*; in the run before: 1.031, 1.024, 0.967, 1.018, 1.031 *(field, E17)*.
  Across five earlier runs of the unchanged kernel: m 8 0.994, 1.031, 1.009, 1.012, 1.019; m 32 0.974, 1.024, 1.003,
  1.002, 0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; m 2 1.011, 1.039, 1.033, 1.046, 1.049; m 16 0.983, 1.045, 1.036,
  1.043, 1.039 *(field, E2)*. Which of these token counts count as won depends on the run that is read. m 1 read 0.921
  in the first of the five and 1.258–1.279 in the other four.
- **The 65-expert mid-m kernel splits stage 1's last round across workgroups at m 32–128.** Expert blocks from the 64th
  on run as 6 K parts in separate workgroups; each stores its fp32 accumulators to a workspace with coherent stores and
  counts in on a per-tile counter the prelude zeroed, and the last to arrive adds the parts in part order and runs the
  epilogue. No workgroup waits for another, so it needs no co-residency; a stress run on CU-limited streams and beside a
  second call agreed with the call alone *(field, E14)*.

**256–4096 tokens**

- **257 experts, inter 256, at m 256: no kernel reaches 1.03.** The large-m kernel starts at 512 and the all-m kernel
  reads 0.855 *(field, E1)*.
- **65 experts at m 512: neither newer kernel is built for it, and the all-m kernel crosses 1.03 between runs.** It
  reads 1.020 in its latest run *(field, E1)*, 1.007 in the run before *(field, E17)* and 1.012–1.147 across the five
  earlier runs *(field, E2)*; neither the mid-m nor the large-m kernel takes m 512. At the other token counts those two
  cover it reads 0.811–1.012, and 1.031 at m 4 *(field, E1)*.
- **The 65-expert large-m kernel reuses constants from another load point at 256 and 2048 tokens.** It is the 33-expert
  kernel's pipeline with branches written for 65 experts (non-temporal stage-1 weights at 256 tokens, no groups of row
  tiles in either GEMM, a pipelined stage-1 panel fill that reads the routed rows itself, a software-pipelined stage 2,
  tiles derived from the expert counts, short-tile copies of the tile body, non-temporal w2 at 256 tokens), and it
  applies the constants of the 33-expert layer's 2048-token row at all three token counts. Only at 4096 tokens do the
  routes per expert match that row (9m divided by the experts: about 567 against 558); at 2048 tokens they are about 284
  and at 256 about 35 (estimated).

## Mechanism

What the nine kernels share:

- **Scaled fp4 MFMA.** `mfma_scale_f32_16x16x128_f8f6f4` with fp4 on both operands and the E8M0 scale bytes as operands
  (the 33-expert kernel also its 32×32×64 form in stage 2 at m 64, 256, 512), fp32 accumulation (`../gemm/mainloop.md`
  §5). The preshuffled weights go from global straight into MFMA operand registers with 16-byte loads
  (`../gemm/datapath.md` §10, §11).
- **Activations quantized twice on the GPU:** the input rows to MXFP4 (amax of each 32-value group across lanes, an E8M0
  exponent from it, `cvt_scalef32_pk_fp4_*`), and the SiLU(gate)·up intermediate again before stage 2: in stage 1's
  epilogue, or, in the 257/256 all-m kernel, at the start of stage 2 at m 1 and 2 and in a separate activation launch at
  m 4–32. The reference the accuracy bound is taken against rounds both to MXFP4 too.
- **Configuration per m** from constants in the source, compiled on a shape's first call. In the 257/256 large-m kernel
  they include the block sizes and stage-1 kernels its earlier build took from aiter's tuned table, in the 257/512
  large-m kernel the stage wrappers and stage-1 output format its earlier build took from aiter's configuration.

**1–2 tokens: no sort, one route per workgroup set.**

- **At m = 1 the three all-m kernels run two launches and no routing sort.** Each route gets its own workgroups, so the
  expert is read straight from `topk_ids`; the token row is quantized inside each stage-1 workgroup; stage 1 splits K
  over waves or workgroups and adds the parts in a fixed order; stage 2 runs one wave per route and adds the 9 weighted
  rows of a token in LDS in route order. Every row of the 16-row MFMA tile is the same token and only row 0 is kept.
  This is the few-M-block shape of `../moe/grouped_sorted_tokens.md` §3 (the N blocks are the parallelism) taken to one
  route per workgroup.
- The 257/256 all-m kernel, m 1 and 2, two launches. Stage 1: 8 column blocks of 64 × 9m routes × 4 K parts (288 and 576
  workgroups of 4 waves); the workgroup quantizes its token's 1536-value K quarter into LDS once, then each wave runs 12
  scaled MFMAs on 16 gate or up columns and stores row 0 as an fp32 partial per K part. Stage 2: 192 blocks of 32 output
  columns × m tokens, 9 waves, one per route; each wave adds its route's 4 partials in part order, applies SiLU·up,
  rounds to bf16, quantizes 4 values per lane and moves the packed words into MFMA operand order with `ds_bpermute`, so
  the intermediate never goes to memory; 32 threads then add the 9 routes × routing weight in route order and store
  bf16:

  ```python
  # stage 2, one workgroup per (token, BN output columns), one wave per route; inside the K loop:
  if FUSE_QUANT:
      # One wave owns the route's full quantization groups. Gather their
      # packed dwords into the MFMA lane layout without a global store.
      stride = 2 if FUSED_QUANT4 and I <= 256 else 1
      av = fx.Vector.from_elements([
          fx.Int32(rocdl.ds_bpermute(T.i32, _raw((k * 4 + j) * (4 * stride)), _raw(packed)))
          for j in range_constexpr(4)], fx.Int32)
      sa = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(k * (16 * stride)), _raw(exponent)))
  # after the K loop: row 0 of each wave's tile to LDS, then the token's routes added in route order
  if lane < 16:
      for b in range_constexpr(BN // 16):
          partials[wave * BN + b * 16 + lane] = accs[b][0]
  fx.gpu.barrier()
  if tid < BN:
      total = fx.Float32(0.0)
      for j in range_constexpr(TOP):
          v = partials[j * BN + tid]
          weight = _p(RW, fx.Float32)[token * TOP + j]
          total = total + v * weight
      _p(OUT, fx.BFloat16)[token * D + fx.Int32(fx.gpu.block_idx.x) * BN + tid] = fx.BFloat16(total)
  ```
- The 65-expert all-m kernel, m 1, two launches: stage 1 with 9 routes × 64 tiles of 32 gate + 32 up columns (576
  workgroups), each workgroup quantizing the token row into LDS and its 4 waves taking a K quarter each; the quarters
  are added in LDS in quarter order before SiLU·up and MXFP4. Stage 2: 192 workgroups of 32 columns, 9 waves, one per
  route, K = 2048, the weighted rows added in LDS in route order. At m 2 it also runs per route without a sort, stage 1
  without the K split and stage 2's rows added by a separate launch in route order (the per-route path runs up to 2
  tokens, the route sum inside stage 2 at m 1 only).
- The 33-expert kernel, m 1 and 2: the first two rows of its table below.

**4–128 tokens: sorted expert blocks, the sort inside stage 1 or in one prelude launch.**

- The 257/256 fold kernel, m 8, 16, 32, 64: two launches with the routing sort folded into stage 1:
  `../moe/few_tokens.md` describes the fold.
  - Stage 1: 16-row expert blocks × 2 blocks of 128 gate + 128 up columns, 4 waves of 4 16-column tiles; K = 6144 in 24
    tiles of 256. Weights non-temporal into a 2-stage register ring; the 16 hidden rows gathered by token id two loop
    bodies ahead, quantized inline into a 3-stage XOR-swizzled LDS ring; 8 scaled MFMAs per tile and wave at raised
    priority (`s_setprio`); accumulators through LDS, SiLU·up, MXFP4 with shuffled E8M0 scales in sorted-row order. At m
    = 32 the tile order is grouped per XCD after the bound test (`../gemm/schedule_and_locality.md` §2). 144–554
    workgroups.
  - Stage 2 (aiter's `mxfp4_gemm2.py` with a CK-style tile remap, the expert id read before the A loads, and
    `amdgpu-kernarg-preload-count` 16, `../gemm/small_m.md` §6): per 16-row block, 24 workgroups of 256 output columns,
    K = 256 in one tile, the MXFP4 intermediate by buffer→LDS DMA, weights non-temporal into registers, × routing
    weight, bf16x2 `atomicrmw fadd` into `out`. Upper-bound grids of 1728–6648 workgroups; those past the device bound
    exit (`../moe/grouped_sorted_tokens.md` §2, §6). The epilogue's adds:

    ```python
    # per row and pair of columns: the fp32 result × the route's routing weight, packed as 2 × bf16
    pk = Vec.from_elements(
        [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
    ).to(fx.BFloat16)
    # then every row of the block adds its pairs into its token's row of `out`
    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(4):
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(_pk_all[mr][s]),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
    ```

    Not this where two runs must give the same bytes: a token's 9 routes sit in different expert blocks, so their adds
    land in arrival order (Counter-signals, Every regime). The 1–2-token stage 2 above adds them in LDS in route order
    instead.
- The 257/512 fold kernel, m 32: the same fold in two launches of its own. Every stage-1 workgroup sorts all routes
  itself in LDS (no grid-wide barrier) into 16-row expert blocks, starts its weight stream before the sort's rank and
  publish steps since it needs only the expert id, loads the gathered hidden rows two k-tiles ahead of their MXFP4
  quantization in the loop, and runs 256-column gate/up tiles; the grid zeroes `out`. Stage 2 walks (m-block, n-block)
  tiles in interleaved groups, loads each row's sorted id and routing weight before the GEMM and adds pairs of bf16
  columns into `out` atomically. At m 32 stage 1 takes XCD-contiguous tiles and stage 2 an XCD-major block order; both
  preload 16 kernel arguments. Paired against the version before the fold was added, about +3.9% at m 32 *(field, E11)*.
- The 65-expert mid-m kernel, m 4–128. Stage 1 is aiter's builder copied (32-row tiles of 128 gate/up columns), with w1
  loaded two K tiles ahead and SiLU·up quantized to MXFP4 in its epilogue with 4-byte stores; stage 2 is aiter's copied
  too, adding into `out` with bf16 atomics, each expert block's K split over two workgroups (the atomics sum them too),
  w2 non-temporal in 64-column tiles, the routing loads issued at kernel start.
  - m 4, 8, 16: no prelude. Stage 1 sorts the routes, quantizes its A rows straight into LDS and zeroes `out`
    itself; below 32 tokens every expert has at most one 32-row block, since a token's expert ids are distinct.
  - m 32, 64, 128: a prelude launch sorts (route counts by LDS atomic adds, a one-wave scan), quantizes the input to
    MXFP4, zeroes `out` and the split-tail counters; stage 1's last, mostly empty round of blocks is split six ways in K
    (Counter-signals):

    ```python
    # buffer cache policy sc0 | sc1: system coherent (L2 write-through on stores, miss on loads)
    _TAIL_CPOL = 0x11
    # a K part of a tail block: store this part's fp32 accumulators to the workspace, [slot][part][v][thread][4]
    for v in range_constexpr(n_v):
        buffer_ops.buffer_store(vecs[v], ws_r, ws_off(v, part32), cache_modifier=_TAIL_CPOL)
    rocdl.s_waitcnt(0)
    gpu.barrier()
    # count in on the tile's counter, which the prelude zeroed; thread 0 passes the old value on through LDS
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        cnt_ptr,
        _fc(1),
        llvm.AtomicOrdering.monotonic,
    ).result
    vector.store(
        vector.from_elements(T.vec(1, T.i32), [old]), lds_tail_flag, [arith.index(0)]
    )
    # `arrived` is that value read back from LDS: the last part to arrive sums all parts in part order
    if_last = scf.IfOp(arith.cmpi(CmpIPredicate.eq, arrived, _fc(tail_parts - 1)))
    sums = None
    for p_py in range_constexpr(tail_parts):
        ld = [
            buffer_ops.buffer_load(
                ws_r, ws_off(v, _fc(p_py)), vec_width=4, dtype=T.f32, cache_modifier=_TAIL_CPOL
            )
            for v in range_constexpr(n_v)
        ]
        sums = ld if sums is None else [arith.addf(a, b) for a, b in zip(sums, ld)]
    ```
  - Paired against the version before each was added: the one FlyDSL prelude in place of aiter's sort, quantization and
    zeroing about +3.1% to +3.8% per token count, the non-temporal w2 with 64-column stage-2 tiles +2.3% to +3.1%
    *(field, E12)*.
- The 33-expert kernel, at or above 1.03 at m 4, 64 and 128: the rows m 4–32 and 64–1024 of its table below. The
  65-expert all-m kernel runs m 4–128 on sorted 16-row tiles (32 at m 128) in four launches and is at or above 1.03 only
  at m 4.

**256–4096 tokens: sorted row tiles whose route rows (bf16, fp8 or MXFP4) a separate launch sums in a fixed order,
or an atomic stage 2.**

- The 257/256 large-m kernel, m 512–4096: a routing launch, stage 1, stage 2 and a route sum.
  - A routing launch sorts the routes per expert and, in extra workgroups, quantizes the activations to MXFP4 in token
    order (bit-exact with aiter's unshuffled MX quantization, by the code's comment); at 512 and 1024 tokens more
    workgroups read the first and last experts' W1 and W1 scales while HBM is otherwise idle, so stage 1's first
    workgroups find them in the memory-side cache. At 2048 and 4096, when stage 1's kernel is XCD-swizzled, the sort
    deals the valid blocks evenly over the 8 XCD slices of its grid.
  - Stage 1 is aiter's FlyDSL builder (a vendored copy) building its sorted A-scale tiles in LDS from the token-order
    scales, so no scale-shuffle launch runs.
  - Stage 2 (a port of aiter's a4w4 down GEMM) writes each route's row as MXFP4; 256-column tiles let an expert's M
    tiles share w2, kept in L2 (not non-temporal) at 2048 and 4096 with an output-tile order of group 8; a wave whose
    rows are all padding skips its passes.
  - The route sum: one workgroup per token decodes its 9 rows, weights them and adds them in route order in fp32, stores
    bf16.
  - Paired against the version before each was added (a step may bundle more than the change named): MXFP4 route rows
    with the FlyDSL route sum, replacing fp8 rows and aiter's reduction, about +5.0%, +6.1% and +7.8% at 1024, 2048 and
    4096; the in-LDS scale tiles, replacing a separate scale-shuffle launch, +2.2% to +4.2% per token count
    *(field, E9)*.
- The 257/512 large-m kernel, m 512–4096: its own driver around launches of its own and copies of aiter's FlyDSL stage
  kernels.
  - The routing kernel, one launch: the sort into block-padded expert ranges, the activations' MXFP4 quantization in
    token order, and at m 512 the zeroing of `out`.
  - Stage 1 is aiter's FlyDSL kernel, named per m in a table, built by one of three edited copies of aiter's stage-1
    builder (at 1024–4096 its XCD tile remap covers the valid blocks only; at 512 it reads the activation scales in
    token order, so no scale re-sort runs). At 1024–4096 it runs as two launches, the first sort blocks by a head
    kernel, fused with the scale re-sort, and the rest in whole dispatch rounds; 4096 uses 192-row tiles.
  - Stage 2 at m 512: aiter's v2 atomic kernel into `out` (copied). At 1024–4096 a kernel of its own, persistent on 512
    workgroups and expert-stationary: a workgroup holds one panel of w2 (an expert × 1/24 of the output columns) in
    registers while the next panel's loads are in flight and streams the expert's sorted rows 32 at a time through an
    LDS double buffer, writing unweighted MXFP4 route rows; at 2048 and 4096 it takes the experts in descending
    route-count order. A route-sum kernel then adds each token's rows, weighted, in route order in fp32.
  - The persistent expert-stationary stage 2, paired against the version before it was added: about +1.5%, +6.1%
    and +4.7% at 1024, 2048 and 4096 *(field, E10)*.
- The 65-expert large-m kernel, m 256, 2048, 4096: the 33-expert kernel's pipeline with the constants of its 2048-token
  row (96-row tiles) and branches of its own for 65 experts. One launch routes per expert and quantizes the input in
  separate workgroups; both GEMMs derive their tiles from the expert counts, so no tile-list launch runs. Stage 1 fills
  its LDS panels straight from the routed rows of the quantized input, the next panel prefetched into registers in the
  last slice of each K step into double-buffered LDS; experts' short last tiles run 32- and 64-row copies of the tile
  body. Stage 2 is software-pipelined, the next activation slab and weights in flight during the current step's MFMAs, 8
  waves of 32 columns per 256-column tile, and writes MXFP4 route rows that a route-sum launch adds in route order in
  fp32. At 256 tokens both weights are streamed non-temporally. The direct copy of the activations into LDS with the
  first weight slice loaded a step ahead, paired against the version before it was added: about +7.0% and +7.2% at 2048
  and 4096 *(field, E13)*.
- The 65-expert all-m kernel, m 512 and 1024 (at or above 1.03 at 1024), five launches: a collect launch (per expert,
  ballot ranks in input order; extra workgroups quantize the input) and a compact launch (expert-major 64-row blocks
  padded with a sentinel row); stage 1 on 64-row tiles, 4 waves of 32 + 32 columns sharing the tile's activation rows
  through two alternating LDS pages, the next K chunk loaded through a loop-carried `range(..., init=...)`, SiLU·up
  through LDS and one lane per row quantizing its 32 values; stage 2 on 64-row tiles, 4 waves × 128 columns, with the
  tile's whole K = 2048 activation panel loaded into LDS once so the MFMA loop has no cross-wave barrier, MFMA operands
  swapped so a lane holds 4 adjacent columns, 8-byte stores of the weighted rows; a route-sum launch adds each token's 9
  rows in route order. No atomics on data at any token count at or above 1.03 (the m 4–128 routing takes route ranks
  with LDS atomics; results stay indexed by route).
- The 33-expert kernel changes structure with m across all three regimes. Every GEMM maps its workgroup id to a
  contiguous slice of the tile order per XCD (`../gemm/schedule_and_locality.md` §2):

  ```python
  # workgroup pid runs on XCD pid % 8: XCD c takes one contiguous run of the `bound` tiles
  def _chiplet_order(pid, bound):
      chiplet = pid % 8
      quotient, remainder = bound // 8, bound % 8
      return chiplet * quotient + (chiplet < remainder).select(chiplet, remainder) + pid // 8
  ```

  One aligned scale dword serves two 16-column halves and two K slices; the intermediate is rounded to bf16 before its
  MXFP4 quantization. Stage-1 weights load non-temporal at m 2–256, cached at m 1 and m ≥ 512. Per m:

| m | Launches | Routing | Stage 1 | Stage 2 rows and route sum |
|---|---|---|---|---|
| 1 | 2 | none | 144 workgroups of 16 waves, 8 per K half; each quantizes the token row into LDS; halves added in order, SiLU, MXFP4 | a wave per route, the weighted rows added in LDS in route order |
| 2 | 4 | one launch: the grid quantizes the input; workgroup 0 ranks each expert's routes with ballots in input order and writes the 16-row tile table | 16-row tiles, 2 K halves × 4 waves | fp32 rows; a route-sum launch |
| 4, 8, 16, 32 | 4 | the same | 16-row tiles gathered from the quantized input by route; 64 + 64 columns and 4 waves (m 4), 128 + 128 and 8 waves (8, 16), 4 waves with K steps in pairs (32) | fp32 rows and a route-sum launch at m 4, 8; fp8 rows through LDS and a route sum of fp8 rows at 16, 32 |
| 64–1024 | 5 | one launch: one workgroup per expert compacts its routes in input order while the rest of the grid quantizes the input | a gather launch (a coalesced XOR-swizzled LDS transpose from m 256) copies each tile's rows in tile-major order; stage 1 with 32-, 48- or 96-row tiles (m 64, 128, 256–1024) staged through LDS two K steps at a time, the next weights prefetched at m 64, 128, the quantized output staged in LDS for 16-byte stores except at m 128; groups of 2–3 row tiles of one expert per column sweep at m 512, 1024 | fp8 rows, a route sum of fp8 rows; 32×32×64 MFMA at m 64, 256, 512 |
| 2048, 4096 | 6 | the same plus a tile-list launch | 96- and 128-row tiles, in groups of 3 and 9 | the same, 16×16 MFMA |

## Result

Both sets of measurements use one ruler: a one-sided Student-t lower bound of aiter's time over the kernel's after 16 or
32 rounds of CUDA-graph replays with L2 flushed before each call, on an MI355X. The latest run of the three all-m
kernels and this page's runs of the five newer kernels collected kernel times with a rocprofiler-sdk tracer; the 257/256
fold kernel's numbers and the run before the latest collected them through Kineto.

**The three all-m kernels**, by their latest run, **and the 257/256 fold kernel**, by its one run, an earlier one
*(field, E1)*.
Bold: 1.03 or above.

| Kernel | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 257 experts, inter 256, all-m | **2.489** | **2.016** | 0.137 | 0.925 | 0.180 | 0.820 | 0.813 | 0.314 | 0.855 | 0.372 | 0.431 | 0.793 | 0.741 |
| 257 experts, inter 256, fold | – | – | – | **1.113** | **1.079** | **1.112** | **1.094** | – | – | – | – | – | – |
| 33 experts | **1.272** | **1.109** | **1.052** | 1.021 | 0.973 | 1.022 | **1.053** | **1.186** | **1.237** | **1.285** | **1.670** | **1.090** | **1.072** |
| 65 experts, all-m | **1.316** | **1.085** | **1.031** | 0.984 | 0.965 | 0.998 | 1.012 | 0.898 | 0.958 | 1.020 | **1.238** | 0.868 | 0.811 |

"–": the kernel is not built for that token count. Every measured token count passed the accuracy check. Earlier runs of
the 33- and 65-expert kernels moved the token counts near 1.03 across it *(field, E2, E17)*: the 65-expert all-m
kernel's m 4 read 0.979 in the run before the latest.

**The five newer kernels' earlier builds** (the same ruler on an MI355X, FlyDSL 0.3.2, aiter 0.1.21.dev66+gdbd8bf5bd,
torch 2.11.0+rocm10.0.0). The newer kernels keep their GPU code (the 257/512 large-m kernel but for its sorted-id bound
and its scale path at m 1024, which E18 measured: 1.061, 1.161, 1.425, 1.649). Bold: 1.03 or above. m 64 (1.025) and m
128 (1.014) are below 1.03, so the kernels are not built for them. Each row is the first of two runs; the second is in
the evidence line.

| Kernel | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 257 experts, inter 256, large-m *(field, E4)* | – | – | – | – | – | – | – | **1.246** | **1.505** | **1.587** | **1.800** |
| 257 experts, inter 512, fold *(field, E5)* | – | – | – | **1.049** | 1.025 | – | – | – | – | – | – |
| 257 experts, inter 512, large-m *(field, E6)* | – | – | – | – | – | 1.014 | – | **1.069** | **1.161** | **1.409** | **1.668** |
| 65 experts, mid-m *(field, E7)* | **1.193** | **1.135** | **1.123** | **1.156** | **1.179** | **1.152** | – | – | – | – | – |
| 65 experts, large-m *(field, E8)* | – | – | – | – | – | – | **1.230** | – | – | **1.365** | **1.276** |

Every measured token count passed the accuracy check with a worst violation fraction of 0. Median call times: the
257/256 large-m kernel 120.9 µs at 512 and 247.2 at 4096 against aiter's 150.8 and 445.1; the 257/512 large-m kernel
225.8 and 363.6 against 241.4 and 607.0; the 65-expert mid-m kernel 96.7 at 4 and 212.9 at 128 against 115.6 and 245.3;
the 65-expert large-m kernel 234.8 at 256 and 855.4 at 4096 against 289.0 and 1093.5; the 257/512 fold kernel 132.9 at
32 against 139.9 *(field, E4, E5, E6, E7, E8)*.

## Evidence

E4–E15 measured the earlier builds of the five newer kernels (257/256 large-m, 257/512 fold, 257/512 large-m,
65-expert mid-m, 65-expert large-m); the newer kernels keep their GPU code, the 257/512 large-m kernel but for its
head launch's sorted-id bound and its scale path at m 1024 (E18).

- E1 [2026-11, gfx950, FlyDSL ?, N=1, one-sided Student-t lower bound of aiter's time over the kernel's after 16 or 32 rounds of CUDA-graph replays with L2 flushed before each call, kernel times from a rocprofiler-sdk tracer (the 257/256 fold kernel: through Kineto), one MI355X] speedup over aiter in the latest run, per token count m 1–4096: 257/256 all-m 2.489, 2.016, 0.137, 0.925, 0.180, 0.820, 0.813, 0.314, 0.855, 0.372, 0.431, 0.793, 0.741; 33-expert 1.272, 1.109, 1.052, 1.021, 0.973, 1.022, 1.053, 1.186, 1.237, 1.285, 1.670, 1.090, 1.072; 65-expert all-m 1.316, 1.085, 1.031, 0.984, 0.965, 0.998, 1.012, 0.898, 0.958, 1.020, 1.238, 0.868, 0.811; 257/256 fold at m 8, 16, 32, 64: 1.113, 1.079, 1.112, 1.094, from an earlier run and not re-measured in the latest
- E2 [2026-10, gfx950, FlyDSL 0.3.2 (a pin installed over the image's FlyDSL, not read from the runs' logs; `../pitfalls/measured_a_different_build.md`), N=5, five separate runs made before the runs of E1 and E17; ruler not stated in the record] the five measurements of an unchanged kernel, in order: 33 experts m 1 0.921, 1.279, 1.261, 1.278, 1.258; m 2 1.011, 1.039, 1.033, 1.046, 1.049; m 8 0.994, 1.031, 1.009, 1.012, 1.019; m 16 0.983, 1.045, 1.036, 1.043, 1.039; m 32 0.974, 1.024, 1.003, 1.002, 0.991; m 64 1.008, 1.033, 1.025, 1.022, 1.030; 65 experts m 2 0.794, 1.049, 0.989, 1.014, 1.024; m 256 1.034, 0.968, 0.917, 1.026, 1.023; m 512 1.147, 1.092, 1.012, 1.107, 1.113 (the 33-expert and 65-expert all-m kernels)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=3, three of the ruler's input sets per token count, SQNR of the bf16 output between two runs] 48.35–51.87 dB at m 8, 16, 32, 64 (the 257/256 fold kernel's sources, byte-identical)
- E4 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two runs of the ruler of E1 with the rocprofiler-sdk tracer, 16 timing rounds each] speedup over aiter 1.246, 1.505, 1.587, 1.800 at m 512, 1024, 2048, 4096; median kernel 120.85, 134.01, 179.83, 247.16 µs, aiter 150.82, 201.94, 285.59, 445.12 µs; accuracy passed, worst violation fraction 0 (257 experts, inter 256, the earlier build of the large-m kernel); a second run of the same build read 1.248, 1.473, 1.592, 1.812
- E5 [2026-11, gfx950, FlyDSL 0.3.2, N=2 at m 32, two runs of the ruler of E1 with the rocprofiler-sdk tracer, 16 timing rounds each; N=1 at m 64, 32 timing rounds] speedup over aiter 1.049 at m 32, 1.025 at m 64; median kernel 132.86 and 181.68 µs, aiter 139.94 and 186.19 µs; accuracy passed, worst violation fraction 0 (257 experts, inter 512, the kernel of the earlier build of the fold kernel, measured as a build that also ran m 64); a second run of that earlier build read 1.040 at m 32
- E6 [2026-11, gfx950, FlyDSL 0.3.2, N=2 at m 512–4096, two runs of the ruler of E1 with the rocprofiler-sdk tracer, 16 timing rounds each; N=1 at m 128, 16 timing rounds] speedup over aiter 1.014, 1.069, 1.161, 1.409, 1.668 at m 128, 512, 1024, 2048, 4096 (m 128 recorded as not faster than aiter); median kernel 203.92, 225.77, 247.51, 290.50, 363.60 µs, aiter 206.97, 241.43, 287.56, 410.69, 607.03 µs; accuracy passed, worst violation fraction 0 (257 experts, inter 512, the kernel of the earlier build of the large-m kernel, measured as a build that also ran m 128); a second run of that earlier build read 1.067, 1.147, 1.393, 1.662
- E7 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two runs of the ruler of E1 with the rocprofiler-sdk tracer, 16 timing rounds each] speedup over aiter 1.193, 1.135, 1.123, 1.156, 1.179, 1.152 at m 4, 8, 16, 32, 64, 128; median kernel 96.73, 139.97, 179.29, 203.78, 206.31, 212.87 µs, aiter 115.55, 158.94, 201.43, 235.75, 243.53, 245.31 µs; accuracy passed, worst violation fraction 0 (65 experts, the earlier build of the mid-m kernel); a second run of the same build read 1.182, 1.137, 1.117, 1.151, 1.173, 1.150
- E8 [2026-11, gfx950, FlyDSL 0.3.2, N=2, two runs of the ruler of E1 with the rocprofiler-sdk tracer, 16 timing rounds each] speedup over aiter 1.230, 1.365, 1.276 at m 256, 2048, 4096; median kernel 234.80, 487.23, 855.38 µs, aiter 288.96, 665.60, 1093.51 µs; accuracy passed, worst violation fraction 0 (65 experts, the earlier build of the large-m kernel); a second run of the same build read 1.183, 1.382, 1.274
- E9 [2026-11, gfx950, FlyDSL ?, N=?, paired timing while the kernel was tuned: the version without the change over the version with it, minus one] the earlier build of the 257/256 large-m kernel: fp8 route rows and aiter's reduction in place of MXFP4 route rows and the FlyDSL route sum +0.0, +5.0, +6.1, +7.8% at m 512, 1024, 2048, 4096; a separate scale-shuffle launch in place of the sorted A-scale tiles built in LDS +4.2, +3.4, +3.3, +2.2%
- E10 [2026-11, gfx950, FlyDSL ?, N=?, paired timing, as E9] the earlier build of the 257/512 large-m kernel: the version before the persistent expert-stationary stage 2 with next-panel prefetch +1.5, +6.1, +4.7% at m 1024, 2048, 4096
- E11 [2026-11, gfx950, FlyDSL ?, N=?, paired timing, as E9] the earlier build of the 257/512 fold kernel: the routing sort and zeroing of `out` outside stage 1 +3.9% at m 32
- E12 [2026-11, gfx950, FlyDSL ?, N=?, paired timing, as E9] the earlier build of the 65-expert mid-m kernel: aiter's sort, quantization and zeroing in place of one FlyDSL prelude +3.8, +3.1, +3.1, +3.3, +3.3, +3.3% at m 4, 8, 16, 32, 64, 128; cached stage-2 weights without the 64-wide tiles +2.8, +2.5, +2.6, +2.8, +3.1, +2.3%
- E13 [2026-11, gfx950, FlyDSL ?, N=?, paired timing, as E9] the earlier build of the 65-expert large-m kernel: stage-1 activations not copied straight into LDS and the first weight slice not loaded a step ahead +7.0, +7.2% at m 2048, 4096
- E14 [2026-11, gfx950, FlyDSL ?, N=3, stress check, 3 tries per condition] every token count, including m 64 and m 128, on a stream limited to 16 and to 64 CUs and beside a second call on another stream agreed with the call alone within the accuracy bound: passed for the five kernels, byte-identical to the sources of the newer kernels' earlier builds
- E15 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count (24 graph replays), stress checks on MI355X] 0 differing tries for the earlier builds of the five newer kernels at every token count they are built for under (a) a stream limited to 16 CUs, one limited to 64 CUs, a second call with the same inputs on another stream, replays of one captured graph; (b) one CUDA graph replayed 24 times with three input sets copied into its buffers, each replay compared with an eager call on the same inputs; (c) two calls at once on two streams, each with its own input set and outputs, on unrestricted streams and on two streams limited to the same first 64 CUs, each compared with an eager call alone
- E16 [2026-11, gfx950, FlyDSL 0.3.2, N=20 tries per condition and token count, check (c) of E15 on the unmodified kernels] wrong output, tries out of 20 on unrestricted streams / on the shared 64 CUs: 33-expert m 1 16/20, m 2 19/20, m 4 18/20, m 8 19/20, m 16 19/20, m 32–4096 20/20; 65-expert all-m m 1 0/20, m 2 18/20, m 4–32 19/20, m 64–4096 20/20; 257/256 all-m m 1 0/0, m 2 0/3, m 4 20/20, m 8 0/20, m 16 20/20, m 32 19/20, m 64–128 20/20, m 256 19/20, m 512–2048 20/20, m 4096 20 of 20 unrestricted, then a GPU memory access fault in one of its grouped GEMM kernels; 257/256 fold 0 at every token count
- E17 [2026-11, gfx950, FlyDSL ?, N=1, the ruler of E1 with kernel times through Kineto, one MI355X] speedup over aiter in the run before the latest, per token count m 1–4096: 257/256 all-m 2.460, 2.024, 0.136, 0.909, 0.178, 0.827, 0.828, 0.311, 0.861, 0.372, 0.426, 0.808, 0.758; 33-expert 1.217, 1.090, 1.031, 1.024, 0.967, 1.018, 1.031, 1.192, 1.170, 1.263, 1.720, 1.073, 1.077; 65-expert all-m 1.282, 1.050, 0.979, 0.960, 0.941, 0.967, 0.969, 0.863, 0.954, 1.007, 1.221, 0.867, 0.803
- E18 [2026-11, gfx950, FlyDSL 0.3.2, N=500 tries per run (two streams, different inputs, the timing ruler's inputs), each output compared with the call alone, bitwise at m 1024, rows off by more than 4 at m 512] the 257/512 large-m kernel's build before at m 1024: 8, 6 and 8 wrong outputs of 2000 in three runs, one route row wrong per affected token, up to 45.75 off; stage 2 rerun alone on the captured stage-1 output reproduced them; the same without the split stage 1 at 1024: 2 of 1000; the kernel 0 of 1000, and the ruler of E1 (rocprofiler-sdk tracer, 16 rounds) 1.061, 1.161, 1.425, 1.649 at m 512, 1024, 2048, 4096, accuracy passed; the kernel at m 512: 0 of 1200 outputs with a row off by more than 4, worst 3.0, alone repeats up to 2.5
- E19 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, aiter dbd8bf5, N=3 input sets per token count; SQNR of the bf16 output against the reference rounded to bf16, and a bound relative to the error of aiter's `fused_moe`] the 33-expert kernel with its fp8 stage-2 rows: 31.44–31.52 dB at m 16, 64, 512, 4096, the bound violated; with fp32 or bf16 rows instead, within the bound; aiter 46.93–47.34 dB. The 65-expert all-m kernel: 26.46–30.33 dB at m 1, 16, 512, violated; its truncations rounded to nearest even change SQNR by −0.053 to +0.078 dB and still fail. Every reading passes the 13 dB SQNR bound
- E20 [2026-11, gfx950, FlyDSL ?, N=1, a separate run; ruler not stated in the record] the 257/512 fold kernel at m 32: 1.043, accuracy passed
