---
title: "GEMM — measuring a change so the number means something"
kind: method
lever: gemm_measurement
ops: [gemm, grouped_gemm, fused_moe]
gens: [gfx950]
dtypes: [fp8, bf16, mxfp4, mxfp8]
regimes: [kernel-A/B, autotune, campaign]
flydsl: 0.3.2
---

# Measuring a GEMM change

> The effects worth chasing on a tuned GEMM are 0.3–3%; the apparatus moves by that much on its own.
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- Comparing two builds, configs or kernels; deciding whether a delta is real.
- Writing a probe, a micro-ruler or an autotune race.
- Closing a series of changes: which of them actually paid.

## 1. Pick the ruler for the effect size

| Ruler | What it cancels | Use for |
|---|---|---|
| **One process, two module instances, arms interleaved per rep** (load the frozen base file and the candidate file as two modules; median of 6–8 groups) | drift, clocks, allocator state — both arms see them | effects of 0.1–2% that live in the loaded file. *(field, E1, E2)* A-vs-A read −0.009% (geomean) here with the machine idle (−0.098% with neighbours on it), while the same code read 0.61% apart over five runs of the one-process-per-arm scorer |
| **One process per arm, full score, A B (A B A B)** | nothing within the run; relies on per-arm warmup | changes outside the loaded file (toolchain, other modules, env), final confirmation |
| Isolated single-op loop | nothing about the surrounding workload | ranking arms of equal occupancy and equal cache behaviour only |

- The in-process ruler only works when every change lives in the file you load twice: both instances
  share the rest of the package. Prove the arms are two binaries (different ISA hash or kernel name)
  before reading the result.
- Cross-process comparisons of effects under ~3% are unreliable: *(field, E3, E4)* the same comparison read
  +1.66% in one pair of processes and −2.2% in another; a bit-identical control cell of a grouped GEMM
  drifted 2.6% in an hour. Bracket each candidate with a fresh baseline (A B A) instead of reusing one from
  earlier.

## 2. Arm order

- **One process per arm (whole-score or kernel A/B): `A B`, or `A B A B` for more confidence. No
  palindromes** (`A B B A`): each arm already carries its own warmup and median, so a palindrome only
  doubles machine time.
- **One process, two module instances interleaved per rep (§1): `A B B A` within each rep.** The
  palindrome cancels a linear drift inside the rep; take the median of the per-rep ratios
  (`autotune.md` §2).
- **One process, three or more arms: shuffle the arm order randomly every rep.** On a power-capped card
  the slot an arm runs in changes its clock: *(field, E5)* on a grouped GEMM the last slot read +1.3% on one cell
  in three independent processes (−0.24% to +1.39% on the other cells); reversing the order moved the "gain"
  onto the unmodified arm. With three or more arms a fixed order — a palindrome or a cyclic rotation — pins
  that term to arm order instead of cancelling it, because it keeps who-follows-whom. Never park the candidate
  last by habit, and discard the first pass.
  - Shuffling (on a grouped GEMM's bench the A-vs-A pair read +2.2% / −1.9% with stable signs until the order was
    shuffled *(field, E6)*) removes it.
- **Add an inert arm** — the candidate switch compiled to the baseline's exact configuration — and report
  every gain as "candidate minus inert". *(field, E8)* On a grouped GEMM, an identical pair read +1.11% in the
  same batch where a candidate read +0.78%: the real effect was −0.33%.

## 3. Warmup, clocks and load

- **An idle card needs a minute or two of sustained real-shape load to reach its working clock** (warm for
  at least 90 s). *(field, E9)* The same config climbed from 3984 to 4910 TF/s while each reading looked
  stable (0.26% spread). Short warmups make every absolute number ~20% low.
- **On a power-capped kernel the first arm of a batch runs slow** while the clock policy settles
  (on an attention kernel 0.9–1.7% slower at 3–4% lower sclk *(field, E10)*). Burn a throwaway arm first, or
  use the end-of-batch baseline.
- **Record sclk per arm**, derived as `GRBM_GUI_ACTIVE / 8 XCDs / duration` rather than sampled with
  `rocm-smi` between batches (which catches idle windows). Two arms at different clocks are two
  machines, not two kernels.
- **Check for another tenant before timing** (`rocm-smi --showuse`, `--showpids`). *(field, E11)* With the GPU
  90–96% busy from another process, readings were up to 4× slow (223 against ~56 µs) and scattered up to 9×
  within one cell (7.72–68.83 µs): discard them, do not average them.
- **Measure CU-fill levers under the real load mix.** *(field, E12)* An isolated loop let the under-filled
  arm boost to 1699 MHz and made a tail split look negative; under the real mix both arms ran at
  ~1605 MHz and the split won by 21.5% on the scored table.
- **Cache, residency, swizzle and placement levers do not transfer from isolated back-to-back loops.**
  *(field, E13, E14)* On quantization (cast) kernels, a write-placement change read +0.4…6.5% isolated on 10
  of 11 casts (no gain on the eleventh) and −0.45% on the scored bench; an `nt` hint read −2.1% isolated and
  was worth +2.4% inside the step. Levers that delete bytes or requests do transfer (the isolated reading is
  then a lower bound).

## 4. Noise floors and null controls

- The floor belongs to a ruler and a cell, and it moves between sessions. Measure it every session
  with a same-binary arm; never estimate it from two reads (on a grouped GEMM two reads of one arm differed by
  0.13%; three reads of another arm spread ~0.5% *(field, E15)*).
- **Use cells the change cannot touch as a free drift ruler.** Prove they are untouched (identical ISA
  hash), then subtract their mean movement. *(field, E16, E17)* Untouched cells moved +0.21…0.37% in one
  direction and erased three "wins"; with the inert cells as reference an estimated +0.29% gm effect on a
  grouped GEMM came out of three runs whose raw gm readings spread 0.26 points (100.99–101.25), about as large
  as the effect.
- Report per cell, not only the geometric mean: an 18-cell gm can sit inside 0.02 pp while one cell
  moved 3% *(field, E18)*.

## 5. Statistics

- Compare means with their standard error (Welch t, non-overlap of the arm distributions), not with
  the range or the per-read spread; the range grows as you add reads.
- Small-n permutation tests have a floor (2 vs 2: p ≥ 0.167; 3 vs 3: 0.05; 4 vs 4: 0.014). Hitting the
  floor is the best that design can say, not a refutation.
- A large t does not fix a systematic bias in the apparatus (§6).
- `min` of N defends against in-process outliers; a paired win count defends against shifts between
  processes; a cell with two modes needs the minimum over many processes.
- **Under graph replay the mode can be fixed per capture**: recapture each arm several times. *(field, E32)* On a
  fused-MoE stage 1, one binary ran 224–226 or 267–271 µs per captured graph, in three sessions.
- Paired win-count trial counts that held up on a fused attention backward kernel: |Δ| > 2% — 9 trials;
  0.5–2% — at least 11; < 0.5% — at least 11 in each of two sessions *(field, E30)*.

## 6. Apparatus traps

- **Swapping arms by copying files biases ~0.4%.** *(field, E19)* An arm whose file was never imported read
  +0.39%: a different file size meant a different idle time before the run, so a different
  temperature. Equalize the gap (sleep ~20 s after each swap) or let a compile separate the arms.
- **Operands made outside the real path change GEMM time.** *(field, E20)* All-zero operands ran 155 µs
  against 205 µs on real data, same build and shape. The values cannot account for that gap: data entropy alone is
  put at about 10%. The likelier cause is allocation history and page placement. Time on tensors from the real path;
  SNR-gate every probe.
- **Output-buffer placement moves the same binary 10–15%** on a quantization kernel *(field, E21)*. Share
  buffers across arms and pre-warm all of them; per-arm allocation in a tuning sweep biases the winner.
- **Input placement moves a GEMM too.** *(field, E33)* Where B was allocated alone moved one bf16 GEMM binary from 91
  to 102 µs and reversed a K-chunk ranking. Rank address-sensitive settings on the mean over several allocation
  offsets (`small_m.md` §1).
- **Do not trust an arm's first read at a single repetition.** On an attention kernel such a read was +15% on a ±0.2%
  ruler *(field, E22)*; that run also dumped its IR, so the two causes are not separated.
- **Keep the host out of the GPU window.** Time R consecutive calls inside one event pair; a
  `synchronize` per call puts Python into the measurement (on a grouped GEMM, 650 TF/s read for ~1150 real
  *(field, E23)*). If time is flat across sizes, you are measuring the host.
- **One call per CUDA/HIP graph measures the launch floor**, and `torch.cuda.current_stream()` must be
  read inside the captured closure.
- Give every arm an identity fingerprint — config, grid, kernel name, error signature, opcode counts —
  plus one arm that must lose (e.g. an injected `s_sleep`). Arms silently collapsing into one binary
  is the most common way a probe lies.

In FlyDSL: `../flydsl/gemm.md` § measurement §6.

## 7. The operating point and the denominator

- **Benchmark the deployment's per-call shape:** M per call (not per step) and the contraction
  padding. Check the launched grid against a trace.
- **Each arm at its own best configuration.** A configuration pinned for both arms can be one kernel's best and
  the other's worst.
- **Re-measure the competitor column yourself**, with its own calling convention, and confirm it took
  its fast path; *(field, E25)* re-measured in two runs, the competitor reached only 0.966–0.990 of its own
  frozen column (1.0–3.4% below it), which turned a 10/12 result into 12/12.
- **Aggregate with weights of the right dimension.** Time-share weights take an arithmetic mean of
  times; harmonic means are for FLOP-weighted rates. *(field, E26)* A harmonic mean over time-share weights
  reported +8.0% for a true +6.9%.
- Use measured ceilings with the same read/write ratio as the kernel, never the datasheet or your own
  fastest kernel.
- Report TF/s and bytes moved with the per-dtype formulas in FlyDSL's guide, including MX scale bytes
  (`../flydsl/guides.md` § gemm-optimization (FlyDSL's §8.1–§8.2)). gfx950's vendor theoretical dense peaks at MI355X clocks
  are 2.5 PFLOP/s bf16/fp16, 5 PFLOP/s fp8 and 10 PFLOP/s MXFP4/MXFP6, with 8 TB/s
  HBM.

## 8. Autotune races

A per-shape race is a measurement too, with its own noise and order bias. How to build one that cannot
regress is in `autotune.md`.

## 9. Closing a series of changes

- **Ablate every component on the full table, including an all-off arm**: BASE → each one off → all
  off → BASE again. Turning off one at a time makes two compensating components look like two
  winners. *(field, E27)* Removing two tile pickers one at a time cost 1.31% and 0.32% (1.6% summed); removing both
  while keeping the machinery they enabled lost 3.0% — more than reverting the whole change (2.1%).
  The machinery was net negative without its lever.
- Keep a change only if its price exceeds the BASE-to-BASE drift and its sign agrees across cells.
  Re-verify any causal claim made from a single shape on the whole table (a narrow-tile rule
  credited with +1.9% on one shape was +0.09% better removed *(field, E28, E29)*).
- The best score banked across many attempts is the maximum of a noise distribution; re-measure it.
- Equivalence by ISA hash only after confirming the same tree gives a stable hash (in-process
  autotuning can change the ISA); never drop outliers or let sample counts differ between arms.

## Related
`attribution.md` (pricing a change before and after) · `correctness.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. Kernels are the FlyDSL GEMM, grouped GEMM, quantization and attention kernels of the campaigns named on each
line; "grouped GEMM", "quantization kernel" and "attention kernel"
in a sentence mark the numbers this page states for another kind of kernel.

- E1 [2026-09, gfx950, FlyDSL ?, N=?, paired in-process ruler (one process, the frozen base and the candidate loaded as two module instances)] A-vs-A (the same code in both arms) read −0.009% on the geomean with the machine idle; with neighbours on the machine −0.098% (MXFP4 dense GEMM, 12 Llama training shapes; unverified)
- E2 [2026-09, gfx950, FlyDSL ?, N=5, one process per arm, scored bench (12-row geomean against frozen reference values)] five reads of the same code spread 0.61% (MXFP4 dense GEMM, 12 Llama training shapes)
- E3 [2026-07, gfx950, FlyDSL ?, N=2, one process per arm, ruler not recorded] the same comparison read +1.66% in one pair of processes and −2.2% in another (kernel and shapes not recorded; the run behind it is not recorded)
- E4 [2026-09, gfx950, FlyDSL 0.2.4, N=8, scored bench, every arm a fresh session with the runtime cache cleared] a control cell with zero opcode differences read 98.74 → 101.33 gm within an hour (+2.6%), while A-vs-A inside one session was ±0.5% (gpt-oss-20b MoE FC1 forward, grouped MXFP4 GEMM with fused GLU and quantization epilogue `grouped_gemm_fp4_glu_impl`)
- E5 [2026-09, gfx950, FlyDSL 0.2.2, N=3, one process, three arms in a fixed palindromic order every rep, 6 warmup and 9 timed iterations, median] the arm in the last slot (`aux1`) read +1.33 / +1.45 / +1.29% on Down/bal in three processes, and +1.3 / +1.04 / +1.39% on Down/unbal, +0.34% on GateUP/bal (first process only) and +0.34 / +1.11 / −0.24% on GateUP/unbal; with the order reversed the unmodified `ship` arm, now last, read +2.23 / −1.69 / −0.05 / +3.71% on the two Down cells (gpt-oss-20b grouped per-tensor FP8 GEMM forward, B=24, Down and GateUP cells under balanced and unbalanced routing)
- E6 [2026-09, gfx950, FlyDSL ?, N=?, one process, eight cells each with an A-vs-A twin arm, fixed arm order every rep] the A-vs-A pairs read +2.2% / −1.9% with stable signs, and rotating the order did not lower that floor (gpt-oss-20b MoE fused GLU and dGLU grouped MXFP4 GEMMs, eight cells)
- E8 [2026-09, gfx950, FlyDSL 0.2.2, N=1, one in-process palindromic probe batch, outputs bit-identical] an identical arm pair read +1.72 / +1.18 / +1.49 / +0.07% on four cells (mean +1.11%); the candidate in the same batch read +0.78%, −0.33% net of the pair (gpt-oss-20b grouped per-tensor FP8 GEMM, B=24)
- E9 [2026-09, gfx950, FlyDSL ?, N=?, config, data and process held fixed] one shape climbed 3984 → 4910 TF/s (+23%) from accumulated load; the cold readings were stable, spread 0.26% (MXFP4 dense GEMM, Llama training shapes)
- E10 [2026-09, gfx950, FlyDSL ?, N=3, one process, arms back to back for at least 14 s each at the 1400 W cap, sclk sampled] the first base arm of three batches read 1.8657 / 1.8623 / 1.8533 ms at 1546 / 1533 / 1554 MHz against 1.8343 / 1.8387 / 1.8363 ms at 1594 / 1602 / 1617 MHz for the same tree at the end of each batch, summarized as ~1.5% slow at ~4% lower sclk; per batch 1.71 / 1.28 / 0.93% more time at 3.0 / 4.3 / 3.9% lower sclk (gpt-oss D=64 attention forward)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=3, AKLR `bench_cudagraph` with cold L2 (median of 5) and the driver bench, on a GPU another process kept 90–96% busy] at 96% busy m=16 readings ranged 7.72 to 68.83 µs and m=24 read 84.41 against 8.16 µs for the other arm; at 90% busy a batch ran at 115–132 µs, which the run's distill note sets against ~56 µs; at 92% busy m=2732 read 223 µs against 56.05 for the canonical build at the start of the preceding session (Kimi-K3 bf16 GEMM gemm_n1280_k7168 forge run, three candidates)
- E12 [2026-09, gfx950, FlyDSL ?, N=2, scored bench (12-shape table, 90 s warmup, two passes) against a 60 s isolated loop of the one shape] isolated, plain QKV_wgrad read 375.9 µs at 1699 MHz and beat both split arms; under the 12-shape mix all arms ran at ~1605 MHz and the tail split won +17.3%; on the scored table QKV_wgrad went 4417 → 5392 TF/s (+21.5%) in two clean reads (MXFP4 dense GEMM, Llama QKV_wgrad 6144×4096×32768)
- E13 [2026-09, gfx950, FlyDSL ?, N=3, three isolated palindromic single-op sweeps with bit-identical outputs, then one read per arm of the scored step in one session] per-shape best pid-placement tables gained +1.1 / +1.9 / +2.4 / +0.4 / +6.4 / +6.5 / +4.2 / +6.2 / +2.8 / +1.9% isolated (one more cast had no gain); in the step the table read 105.96 against 106.44 gm without it (−0.45%), QKV −1.42 (MXFP8 dual-cast quantization kernel, scored in one linear's whole training step over five Llama shapes)
- E14 [2026-09, gfx950, FlyDSL 0.2.4, N=1, isolated back-to-back A/B of the cast (A-vs-A 0.26–0.73%) against one read per arm of the scored step `gptoss_q`] the `nt` source-load hint read −2.10% isolated on the 295 MB cast; in the step, turning it off for that cast (band edge 268 → 335 MB) took `gptoss_q` from 1.2301 to 1.2001 (−2.44%), out_proj_fwd −5.0% (MXFP4 dual-cast quantization kernel inside the gpt-oss dense MXFP4 GEMM step)
- E15 [2026-08, gfx950, FlyDSL ?, N=2+3, scored bench gm] one arm read 1.51151 / 1.50954 (0.13% apart); the next arm read 1.51587 / 1.51773 / 1.51058, a ~0.5% spread (gpt-oss MXFP4 grouped GEMM forward, dgrad and wgrad)
- E16 [2026-09, gfx950, FlyDSL ?, N=8, chained five-cell ruler in one process, median of 21 per cell, 8-run palindrome, band pinned] cells the change left byte-identical moved +0.227% (nt_mlpdn), +0.214% (tn_lin1) and +0.366% (nn_mlpup) with the changed cells (+0.274%, +0.173%); net of that batch bias the round's three candidate "wins" fell to about zero (dense per-tensor FP8 GEMM, Llama training shapes)
- E17 [2026-09, gfx950, FlyDSL 0.2.4, N=3, scored bench against frozen denominators, four changed and four inert cells per run] raw gm read 101.25 / 101.24 / 100.99 against 100.81 for the reference; changed-cell mean minus inert-cell mean read +0.46 / +0.50 / +0.67 against −0.05, which the round turns into +0.59 on four cells, about +0.29 gm (gpt-oss MXFP4 grouped MoE MLP GEMMs, wgrad cells)
- E18 [2026-09, gfx950, FlyDSL 0.2.4, N=2, 18-cell scored bench, the same code twice (one comment apart), a neighbour on another GPU] the total read 1.0109 / 1.0107 (0.02 pp); per-cell differences had a median of 0.38% and a maximum of 3.36% (gptoss_QKV_wgrad) (MXFP4 dense GEMM, 6 gpt-oss and 12 Llama shapes; unverified)
- E19 [2026-10, gfx950, FlyDSL ?, N=4, arms swapped by `cp` between benches, ABBA, four reads per arm] with the swapped file never imported, B read 4823.45 against 4804.64 TF/s (+0.391%, t 2.35, permutation p 0.043); with the file live +0.331% (MXFP4 dense GEMM linear2_dgrad 16384×15360×3072; unverified)
- E20 [2026-09, gfx950, FlyDSL ?, N=?, probes on one build and shape] all-zero operands (`torch.zeros`) read 155 µs, real-data operands 205 µs, 32%; data entropy alone put at 10%, the rest on allocation history and page placement (MXFP4 dense GEMM, Llama-2 70B training shapes; the probe runs are not recorded)
- E21 [2026-09, gfx950, FlyDSL ?, N=?, one compiled binary, fixed inputs and build knobs, only the output planes' byte offsets moved] 16384×21504 read 262.8–301.2 µs (14.6%), 16384×15360 191.6–220.6 µs (15.1%), three other shapes 10–11% (MXFP8 dual-cast quantization kernel)
- E22 [2026-09, gfx950, FlyDSL 0.3.2, N=3, same-process probe] an arm's first read under `FLYDSL_DUMP_IR=1` and `REPS=1`: +15.44%; three normal re-runs ±0.2%; the read had both the IR dump on and a single repetition, so it does not tell which of the two caused the gap (GLM-5.2 sparse-MLA decode attention producer)
- E23 [2026-08, gfx950, FlyDSL ?, N=?, one CUDA event pair and a `synchronize` per call] a wrapper that rebuilt its launch on every call (2.52 ms of host time per call) made the forward read 650 TF/s, ~1150 once fixed (gpt-oss-20b MoE grouped bf16 GEMM forward, G=32)
- E25 [2026-09, gfx950, FlyDSL ?, N=2, the competitor's own API and calling convention in the same harness, 200 s warmup, 60 iterations in one event pair, best of two passes] AITER `gemm_a4w4` read 1.0–3.4% below the frozen reference column on the 12 rows over two runs, i.e. 0.966–0.990 of it (also reported as 0.966–0.992); the one per-row table kept reads 0.968–0.990; against the frozen column 10/12 rows were ahead, against measured AITER 12/12 (MXFP4 dense GEMM, 12 Llama training shapes; unverified)
- E26 [2026-10, gfx950, FlyDSL ?, N=5, both arms on one GPU, each run one process covering all eighteen rows, 10 s warmup, median of five reads of 200 iterations] 18 training GEMMs weighted by % train time went 4626.9 → 4947.4 TF/s, +6.93% as a time-weighted arithmetic mean; the time-weighted harmonic form read +8.02% (the eighteen training GEMMs of one model, 6 A6W4 forward and 12 MXFP4 rows, Primus-Turbo b484f490 against c417c70f)
- E27 [2026-10, gfx950, FlyDSL ?, N=?, closing ablation, arms swapped by `cp` with a 20 s sleep after each swap] `_mxfp4_pick_block_n` off −1.31%, `_mxfp4_pick_block_m` off −0.32%, both pinned to 256 with the rest of the change kept −3.00%, the whole change reverted −2.11% (MXFP4 dense backward GEMM, 12 training shapes, ablation dated 2026-10-11; unverified)
- E28 [2026-09, gfx950, FlyDSL 0.2.4, N=3, scored reads of one cell] the M-axis narrow tile (BLOCK_M 192) took Llama QKV_wgrad 6144×4096×32768 from 5410 to 5515 TF/s (MXFP4 dense GEMM; docstring in the round's diff)
- E29 [2026-09, gfx950, FlyDSL ?, N=?, 18-cell scored table, BASE read before and after, floor 0.16%] narrow N axis only 1.0123 against both axes narrow 1.0114: removing the M-axis rule gained 0.09% (MXFP4 dense GEMM, 6 gpt-oss and 12 Llama shapes; commit message of a4bbbc8c3, carried into Primus-Turbo #514 (4b5833334))
- E30 [2026-08, gfx950, FlyDSL ?, N=3, one process, the two arms interleaved, paired win count over 9, 11 or 15 trials per session, three independent sessions] two candidates that won 8/9 (+0.49%) and 7/9 (+0.31%) in one session read 6/11 and 3/15, and 2/11 and 5/15, in the next two (both judged noise); a +0.53% change kept its sign at 8/9, 7/9 (+0.23%) and 11/15; a −2.8% change read 9/9, and another −2.9% / −2.8% at 11/11 in each of two sessions; the rule written from these: |Δ| > 2% one 9-trial session, 0.5–2% at least 11 trials, < 0.5% at least 11 trials in each of two sessions with the same sign (fused FlyDSL flash-attention backward, dQ folded into the dK/dV kernel, D=64, scored at B=3 S=8192 Hq64 Hkv8 causal)
- E32 [2026-10, gfx950, FlyDSL 0.3.2, N=3 sessions, per-kernel time inside CUDA graphs with the L2 flushed per call] stage 1 about 226 or 268 µs, fixed for each captured graph; about 224 or 267 in a second session; 224.8 and 271.2 µs in two identical captures in a third. Not removed by one scratch arena, shared outputs or 1–8 dummy workgroups launched first; XCC ids read round-robin in fast and slow graphs alike (fused_moe situv2 fp8 × MXFP4, 896 experts, top-16, n_tokens 2732, stage 1 t64x128x256)
- E33 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, AKLR graph timer through the driver, and an in-process probe on copies of the inputs at chosen offsets] bf16 GEMM, N 6144, K 16384, m=512, split-K 4: partway through a session the unchanged kernel moved from about 93.0 to 102.41 µs; swapping tensors one at a time, the K-chunk-16 build read 101.85 µs with the driver's B and its own A, 91.17 with its own B and the driver's A; in that state chunk 8 92.67 against chunk 16 101.88, over 9 copied offsets 90.24–90.85 against 90.74–91.22; a later sweep with 0 / 37 / 211 MB allocated before the inputs: chunk 32 91.44 / 91.50 / 89.96 against chunk 16 90.06 / 90.05 / 90.12 (bf16 GEMM n6144 k16384)
