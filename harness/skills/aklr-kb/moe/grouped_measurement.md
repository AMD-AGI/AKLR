---
title: "Grouped GEMM — measuring: distributions, the deployment's operating point, trace FLOPs, per-cell noise"
kind: method
lever: grouped_measurement
ops: [grouped_gemm]
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, skew, kernel-A/B]
flydsl: 0.3.2
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:benchmark/ops/training/config.py:487-497
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2453-2467
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2483-2484
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2622
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4772-4777
---

# Measuring a grouped GEMM

> The shared protocol — ruler choice, arm order, warm-up, noise floors, statistics, closing a series of changes —
> is in `../gemm/measurement.md`; probes and power-cap pricing are in
> `../gemm/attribution.md`. This card is what the group dimension adds: the token distribution as an
> axis of the table, the deployment's operating point, FLOP accounting from training traces, and floors that
> differ per cell. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence
> at the end. Use them for direction and order of magnitude.

## Route here when
- Building or inheriting a grouped-GEMM benchmark, scorer or autotune race.
- A kernel-only gain does not show up in the training step, or the bench and a trace disagree.
- A delta sits near 1% on skewed cells, or two runs of identical code disagree.

## 1. The table: geometries × projections × passes × distributions

- **Several model geometries, both projections, all three passes.** Use the gate_up projection
  (N = 2 · intermediate, K = hidden) and the down projection (N = hidden, K = intermediate) of at least three
  models, for forward, dgrad and wgrad, at the per-call tokens per group the deployment runs plus one size on each
  side. Score the geometric mean across geometries: a single-geometry table lets NT changes overfit without any
  warning (`grouped_workflow.md` §8). Take the dimensions from the model configurations; this card fixes
  none.
- **Every distribution is a column.** Per cell:
  - balanced;
  - mild random skew (Turbo's generator: `grouped_correctness.md` §2);
  - heavy skew, with one group holding about half of the tokens;
  - one-hot or near-degenerate;
  - empty groups (`grouped_correctness.md` §2);
  - the deployment's measured histogram.
- **The extreme cells are real.** *(field, E1)* A one-hot bench cell reproduced the slowest wgrad call of a training
  trace within 1.4%.
- Report each pass's skew-to-balanced ratio and every cell, not only the geometric mean
  (`../gemm/measurement.md` §4). The heavy-skew cells are usually the minimum, and they are where a
  skew lever is accepted or rejected.

## 2. Lock the deployment's operating point

A clean ruler at the wrong operating point optimizes a kernel the training run never launches. Copy these from
the deployment, not from a previous bench:

| Condition | What goes wrong when it drifts |
|---|---|
| Groups per call (local experts = experts / EP, the expert-parallel degree) | bands, tiles per group and skew change with G; tunings do not transfer |
| Per-call tokens per group | the step aggregate is several calls; a bench at the aggregate size reads high |
| Histogram of tokens per group | balanced-only benches miss every skew effect |
| Which axis is padded, and to what | an unpadded contraction can take a slow path the deployment never runs |
| beta and the accumulator dtype | the C read-back of beta=1 scales with the accumulator width |
| Eager or graph replay | host costs are present in one and hidden in the other |
| Every argument the call site passes | one optional argument can select another body |

- **Copy the call site's arguments, not only its shapes.** *(field, E2)* A bench passed a CU count and ran the
  persistent body; the deployment's call site passed none and ran the non-persistent one. Over a long series
  of changes, every change the bench scored as a win was a loss on the deployed body; with two smaller causes (a
  bench bucket the deployment does not reach, and communication), the bench's 5.2% time saving was 0.84% in
  training.
- **Per call, not per step** (`../gemm/measurement.md` §7). *(field, E3)* A step ran four
  micro-batches, so each call saw a quarter of the step's tokens per group; at that size the wide projection
  reached 23% (forward) and 36% (wgrad) less throughput than at the aggregate size. Confirm the size by
  comparing each kernel's launched grid in a trace with the bench's: the grid fields match exactly when the size
  is right, which is a harder check than durations.
- **Pad the contraction axis the way the deployment does** (`grouped_lowprec.md` §6: padding only the
  output axis left a forward on its misaligned slow path). Count useful FLOPs from the unpadded sizes.
- **Accumulator dtype.** *(field, E4)* A wgrad timed with an fp32 accumulator where the deployment used bf16 took
  8–14% more time on one shape (beta=0 and beta=1). The beta=1 read-back itself is priced in
  `grouped_wgrad.md` §10.
- **Graph replay hides host enqueue, not kernel boundaries.** *(field, E5)* On an inference MoE decode path, one extra
  small kernel on the dependency chain still cost a full launch on every call under graph replay
  (`grouped_moe_inference.md` §7). Measure launch-count changes on the scoring bench, in the mode the
  deployment runs.
- **Screen with kernel-only, decide in the step.** *(field, E6)* Kernel time saved converted to step time at 81% over
  one series of changes; a probe 0.5% faster on the deployed grid was 0.71% slower in training.

## 3. Reading a training trace: per-call sizes and FLOPs

- **Read sizes, do not reconstruct them.** Pair each GPU kernel with its CPU op through the trace's
  `External id` and read the op's `Input Dims`: `A = [M_total, K]` and `B = [G, N, K]` give G and the mean tokens
  per group (`M_total / G`) exactly; the per-group histogram is in the `group_lens` contents, not in the dims.
  Rebuilding them from batch size, sequence length and EP is how operating points
  got mixed up.
- **Do not divide by the layer count.** Every token is routed again in every layer: assignments per layer are
  tokens × top-k. A FLOP count divided by the layers reads an order of magnitude low.
- **One kernel name, several logical GEMMs.** The gate_up and down projections of a pass launch the same kernel,
  so a per-name aggregate merges two GEMMs with different N. Split the calls by per-call duration clusters, compute
  each GEMM's rate, and check that the parts sum back to the aggregate.
- **Training traces carry contention.** *(field, E7)* The median of one profiled step read 5–9% low from overlap with
  other work, and *(field, E8)* the long-tail calls were the ones overlapping collectives. Compare with a bench using
  the minimum or the calls without overlap.
- A trace's per-kernel rate must land on the bench at the same operating point. A miss of an order of magnitude
  is FLOP accounting, not the kernel.

## 4. Timing hygiene

- **Keep compilation out of the timed region**: compile each configuration once, before timing it.
- **Keep the host out of the GPU window** (`../gemm/measurement.md` §6), and assert it: time the host
  enqueue of the same R calls with a wall clock and require `t_host < 0.9 · t_gpu`. A slow wrapper looks normal
  until it is compared with a fast one.
- **Give beta=1 a fresh accumulator every repetition.** *(field, E9)* Reusing one bf16 accumulator for 30 repetitions
  overflowed it and read 3200 µs for a 1814 µs kernel.
- **Freeze the denominator.** When a score compares two operating points of the same kernel (two group counts,
  or a wgrad against its own forward), a live denominator rewards slowing it down. Use a copy of the base kernel
  that the work cannot touch, measured in the same interleaved window, plus a live guard on the other point.
  Set the guard's tolerance from its measured spread: *(field, E10)* a min-of-four guard at 0.99 failed the baseline
  itself (its own spread reached 0.984); a geometric mean at 0.98 passed.
- **Version A/B in one process** (`../gemm/measurement.md` §1): load the old kernel as its own module,
  so a cache keyed on names cannot hand both arms one compiled kernel. Bit-identical outputs with a ratio of ~1.0
  mean "suspect one binary served both arms", not "no effect".
- **A probe tree goes stale when the tree under test is refactored.** *(field, E11)* The wgrad half of a change,
  +4.8% on a probe tree copied before a dispatch refactor, read ~0 on the table's geomean with one of three changed
  files synced and +0.34% (worst cell +1.05%) with all three: re-measure on the tree that will land, with every
  file it changed.

In FlyDSL: `../flydsl/moe.md` § grouped_measurement §4.

## 5. Noise per cell and arm order

- **The floor belongs to the cell.** *(field, E12)* In one run of one binary, the A-vs-A reading was −0.02% on a
  balanced cell and +1.39% on a skewed cell, whose histogram was redrawn every run so its tile counts and
  imbalance changed. Set acceptance thresholds per cell; to resolve under ~1.5% on a skewed cell, add samples or
  detrend, and keep the draw identical across arms when ranking a kernel change.
- **Single cells swing more than aggregates.** *(field, E13)* One wgrad cell moved ±2.5% run to run on identical code.
  Decide on paired differences, not on one cell's reading.
- **Detrend with an in-run control arm.** *(field, E14)* A control arm drifted 1.8% over six back-to-back runs; an
  order that looks balanced does not cancel a monotone drift. Normalize each run by the control arm of that run.
- **Arm order** follows `../gemm/measurement.md` §2: one process per arm runs `A B` (or `A B A B`);
  palindromes are only for two module instances interleaved inside one process. The first arm of a batch runs
  cold (`../gemm/measurement.md` §3). *(field, E15)* On grouped tables, cells a knob could not touch (the
  NT cells under a wgrad-only change) read ~0.5 pp low in whichever arm ran first. Use the untouched pass's
  cells as the batch's drift ruler (`../gemm/measurement.md` §4) next to an inert arm (§2 there).

## 6. Small kernels and ISA statistics

- **Price small kernels additively** — group-offset kernels, scale preshuffle, tail kernels, the split-K reduce —
  by launching them 0, 1, 2 and 4 extra times in one process, alternating arm order, and fitting the slope
  (`../gemm/attribution.md` §3). *(field, E16)* `rocprofv3` start→end gave an idle tail kernel 5.10 µs
  against a marginal 1.58 µs, and a working preshuffle 20.3 µs against ~10.2; a candidate priced at +0.374% from
  the trace was worth an estimated +0.121% (+0.107% measured).
- **Split ISA dumps per kernel.** A module's ISA dump holds all of its kernels back to back. *(field, E17)* Four
  stores read as "the GEMM's prologue draining" belonged to the preshuffle kernel in front of it. Cut at the
  `.amdhsa_kernel` and label lines first.
- **Count over a steady-state window.** Boundary-body variants copy the tile body in the binary while each
  execution runs one copy. *(field, E18)* A wgrad with four boundary variants read 6.6 instructions per MFMA over the
  whole main-loop region and 3.2 inside an MFMA steady-state window; its NT sibling read 4.1 and 3.4.
- **Say which MFMA utilization you report**, wall or steady state (`grouped_workflow.md` §5).

## 7. Power, clocks, and comparisons across precisions

- **Read power per pass before ranking cycle-saving levers.** At ≥99% of board power, part of a cycle saving is lost
  to a lower clock (`../gemm/attribution.md` §4, whose 20% conversion case was the skewed worst cell of a
  grouped MXFP8 NT kernel *(field, E19)*). *(field, E20)* A variable-K wgrad ran at its maximum clock (2400 MHz, about
  266 W), far below the cap, so cycle savings there are not handed back as clock. Prefer levers that remove bytes
  or requests on capped passes and skewed cells.
- **Compare precisions on equal terms** (`grouped_lowprec.md` §8; quantization inside the timed region:
  `../attention/measurement.md` §5).

## 8. The race is a ruler too

Turbo's fp8 NT/NN race scores candidates on balanced synthetic offsets at two canonical sizes, 1024 and 8192
tokens per group (`grouped_gemm_fp8_kernel.py:2453-2467`, `:2483-2484`), and its wgrad race on a balanced load
only (`:4772-4777`). *(field, E21)* An NN dgrad arm that won at the deployed size could not pass the race's 1.5%
adoption margin (`_NP_RACE_MARGIN`, `:2484`, `:2622`) at the two race points
(`../gemm/autotune.md` §2). Check what the race picks at the deployed size and under skew; the
remedies are in `grouped_host_autotune.md`.

## Verify
- [ ] The table covers ≥3 geometries × both projections × three passes × balanced, heavy-skew, one-hot and
      empty-group columns.
- [ ] G, per-call tokens per group, the histogram, padding, beta, accumulator dtype, graph mode and the call
      site's arguments are copied from the deployment, and the launched grids match a trace.
- [ ] beta=1 timing uses a fresh accumulator per repetition.
- [ ] Each cell has its own A-vs-A floor from this session; the decision uses paired differences.
- [ ] Small-kernel prices come from additive launches, not from trace durations.

In FlyDSL: `../flydsl/moe.md` § grouped_measurement Verify.

## Related
`../gemm/measurement.md` · `../gemm/attribution.md` · `grouped_correctness.md` ·
`grouped_host_autotune.md` · `../attention/measurement.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. `Turbo fork <rev>` names the Primus-Turbo fork commit that ran, where the record gives one.

- E1 [2026-08, gfx950, FlyDSL ?, Turbo fork 527e62c7, N=?, bench one-hot cell against the slowest trace call] the frozen wgrad's down projection read 808 TF/s on the bench's one-hot cell; the slowest wgrad down call of the training trace read 797 TF/s (gpt-oss-20b tensorwise fp8, G=32; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_wgrad_skew_campaign.md:88)
- E2 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=? (two training A/B pairs), training step wall; per commit the same shapes timed on both grids] the bench scored the campaign −5.2% (spd 1.0453), training −0.84% (750.90 → 744.60 ms); on the deployed grid (`num_cu` not passed) its commits read 1.0107 / 1.0156 / 1.0165 of the base time, on the bench's `num_cu=256` grid 0.9945 / 0.9782 / 0.9656 (gpt-oss-20b fp8 grouped MLP step; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_stepfuse_campaign.md:137)
- E3 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, kernel-only bench at 8192 and 32768 tokens per expert; the trace's grids nt[3036] / nn[1584] / tn[1008] matched at 8192] forward gate_up 2083 against 2722 TF/s (−23%), wgrad gate_up 1890 against 2945 (−36%), down −5% (gpt-oss-20b EP8, four micro-batches per step; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_e2e_trace_grouped_flops.md:14)
- E4 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, ?] gate_up wgrad at B=32: beta=0 bf16 1676 µs against fp32 1808 µs (+7.9%), beta=1 bf16 1814 µs against fp32 2067 µs (+13.9%) (gpt-oss tensorwise fp8 wgrad, 4096 tokens per group; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_b24_wgrad_campaign.md:39)
- E5 [2026-09, gfx950, FlyDSL ?, aiter fork rev ?, N=?, MoE decode score under graph replay] launching a MoE sorting kernel once more cost +2.29 µs at 64 tokens and +2.41 µs at 128; the batch sizes that do not run it did not move (a MoE sorting kernel, GLM-5.2 TP4/EP4 a4w4 MoE decode; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_155754/rounds/round-11/agent.txt:7)
- E6 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=? (two training A/B pairs), training step wall] compute kernels −7.78 ms per step gave −6.31 ms of wall (81%); with the grouped GEMM files reverted, the deployed-grid single-GPU probe read 0.5% faster and training 0.71% slower (734.29 → 739.52 ms) (gpt-oss-20b; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_stepfuse_campaign.md:164)
- E7 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=1 profiled step (24 calls per shape), training trace] the per-shape median read 5–9% low from concurrent work; the minimum, a call without contention, gives the kernel's speed (gpt-oss-20b fp8 training trace fp8-49457, EP1, G=32; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_teacher_trace_fp8_49457.md:27)
- E8 [2026-08, gfx950, FlyDSL ?, Turbo fork 527e62c7, N=? (one rank-0 trace, steps 30–32), training trace, kernels paired with ops by External id] in every layout the long-tail calls (duration above 1.3× the median) overlapped NCCL kernels 192–200%, the other calls 0–3% (gpt-oss-20b training trace; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_wgrad_skew_campaign.md:82)
- E9 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=30 repetitions on one accumulator, ?] the probe read 3200 µs for a kernel whose time was 1814 µs (gate_up wgrad at B=32, beta=1, bf16 accumulator; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_b24_wgrad_campaign.md:37)
- E10 [2026-09, gfx950, FlyDSL ?, Turbo fork rev ?, N=4 probes per guard reading, a live B=32 guard beside a frozen kernel copy] a min-of-four guard at 0.99 failed the baseline, the leg's own spread reaching 0.984; a geometric-mean guard at 0.98 passed (gpt-oss tensorwise fp8 wgrad, B=24 against B=32; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_b24_wgrad_campaign.md:49)
- E11 [2026-08, gfx950, FlyDSL ?, Turbo fork 16ce4959 with commit 702d9881, N=3 reads, the campaign's scoring script] the wgrad half of a band remap, +4.8% on a probe tree copied before a refactor, read 1118.24 / 1115.72 / 1119.97 with one of three changed files synced; with all three, gm 1161.93 → 1165.93 (+0.34%), min 1031.2 → 1042.0 (+1.05%) (gpt-oss bf16 grouped GEMM; local:/workspace/code/gpt_oss_docker/.claude/memory/project_bf16_flydsl_grouped_campaign.md:402)
- E12 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=4+4 samples, `--diag` gemm-only arms in one run, A-vs-A on a code-identical control arm] the control arm read −0.02% on the balanced cell and +1.39% on the skewed cell (gpt-oss fused GLU/dGLU grouped GEMMs, fc2_dglu; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:227)
- E13 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=5 bench runs, identical code] wgrad down balanced read 1.563 / 1.606 / 1.644 / 1.650 / 1.657 (±2.5%), the NT cells ±0.4% (gpt-oss MXFP4 grouped GEMM, campaign bench; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_155024/memory.md:171)
- E14 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=6 back-to-back runs, code-identical control arm] the control arm drifted from 1.2731 to 1.2505 (−1.8%) (gpt-oss fused GLU/dGLU grouped GEMMs; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_120547/memory.md:238)
- E15 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=5 base→candidate and 3 candidate→base pairs] gate_up wgrad canon +1.70 pp in one order and +2.34 pp in the other; the forward cell the wgrad-only knob cannot touch read ~0.5 pp low in whichever arm ran first (gpt-oss MXFP4 grouped GEMM; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260811_114814/rounds/round-13/agent.txt:24)
- E16 [2026-09, gfx950, FlyDSL 0.2.4, Turbo fork rev ?, N=4–5 samples per arm (ABBA) for the measured difference; additive probe of 0/1/2/4 extra launches in one process] an idle tail kernel read 5.10 µs from rocprofv3 Start→End against a marginal 1.579 µs; the preshuffle kernel 20.32 against ≈10.2 µs; a candidate priced at gm +0.374% from rocprof was worth +0.121% at the probe's price and +0.107% in the measured difference (gpt-oss MXFP4 grouped MoE MLP GEMMs, fc2_fwd; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_090609/memory.md:333)
- E17 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, stage ISA dump `21_final_isa.s`] four stores counted as the GEMM's prologue draining belonged to the preshuffle kernel placed before it in the dump (MXFP4 grouped GEMM; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/09-isa-profiling.md:98)
- E18 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, static ISA histograms] whole main-loop region: wgrad 7581 instructions / 1152 MFMA = 6.6 per MFMA, NT 4546 / 1104 = 4.1; steady-state window `mfma[8..40]`: wgrad 106 / 33 = 3.2, NT 111 / 33 = 3.4 (MXFP8 grouped wgrad with four boundary variants, NT with two; local:/workspace/code/gpt_oss_docker/myskill/optimizer/pitfalls/05-mxfp4-mxfp8-deadends.md:1037)
- E19 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, interleaved mx/tw probe with sclk and power sampled, both arms at the 1400 W cap] dgrad gate_up heavy: 2905 against 3110 cycles (7.1% fewer) but 1.6014 against 1.6241 ms of wall (1.42%), sclk 1814 against 1915 MHz: 20% of the cycle saving reached the wall (gpt-oss MXFP8 grouped NT against tensorwise fp8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260729_093030/rounds/round-13/agent.txt:39)
- E20 [2026-07, gfx950, FlyDSL ?, Turbo fork rev ?, N=?, ?] the grouped wgrad ran at the full 2400 MHz at about 266 W, far below the power cap (fp8 grouped variable-K wgrad; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/01-optimize-loop-benchmark.md:206)
- E21 [2026-08, gfx950, FlyDSL ?, Turbo fork rev ?, N=5 palindrome rounds, isolated kernel] the `(num_xcd, group_m) = (8, 4)` arm read +2.5–2.9% at the deployed 4096 tokens per expert and −0.94% / +0.54% at the race's 1024 and 8192 tokens (geomean ≈ +0.2%) (gpt-oss-20b down-projection padN dgrad NN, tensorwise fp8; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260817_065611/rounds/round-06/agent.txt:21)
