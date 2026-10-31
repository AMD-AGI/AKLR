---
title: "FlyDSL attention — correctness gates: spike inputs, paths the gate must reach, shared LSE, seeds and mask conventions"
kind: method
lever: flydsl_attn_correctness
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [prefill, backward, causal, sliding-window, rectangular]
flydsl: 0.3.2
---

# Correctness for a FlyDSL attention kernel

> Most cases marked `*(field, E<n>)*` below passed a gate that reported clean. Use them to design the gate before
> trusting it. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at
> the end. Use them for direction and order of magnitude.
> The general GEMM rules — gates that reach the changed path, calling twice, races found by bitwise repetition,
> waits and barriers, inline-asm hazards, a third reference after a reduction-order change — are in
> `../gemm/flydsl_gemm_correctness.md`. Debugging a kernel already known to be wrong starts in
> `../bottleneck/debug-flydsl-kernel.md` (its §1 covers `-inf − -inf` and divide-by-zero NaNs).

## Route here when
- You change the softmax shift, the LSE, a scale, an accumulator seed, a mask, a barrier or an anchor.
- A gate passes and you have not shown it runs the path you changed, at the dynamic range that breaks it.
- One output is wrong and the others are right, or a metric reads exactly 1.000.

## 1. What a gate measures
For each output — O, LSE, dQ, dK, dV — against an fp32 reference:
- **SNR**, to catch structural errors. *(field, E1, E2)* Correct bf16 kernels read ~50–53 dB on O and the
  gradients; a mask with one missing edge key reads 22–38 dB.
- **Norm ratio `|got| / |ref|`**, to catch "finite but biased", which SNR and finiteness checks miss. *(field, E3)* dV
  fell to 0.94 at D=64 (0.95 at D=128) at spike 96 with no non-finite value anywhere.
- **Non-finite count** on every output, LSE included.
- **Bitwise determinism**: two runs, `torch.equal`. In synchronisation changes it usually fails before SNR does.
- Score accuracy by the **worst** cell, not the mean: an average lets one row that is 5% off hide behind twenty
  correct rows.
- Check dQ, dK and dV separately; a change to a dQ scaling constant is invisible in dV.

## 2. Inputs that reach the numerics
- **Independent Gaussian inputs cannot exercise the softmax range.** The row maximum then grows slowly with the
  logit scale and stays near the first tile; trained attention is concentrated, and the maximum is often far
  from it. Build spike inputs: a few keys with large logits, placed away from the first KV tile, at several
  spike levels.
- **Push the spike to failure, not to "enough".** *(field, E4)* A gate that stopped at spike 24 reported clean; the
  same kernel produced its first non-finite values at spike 36, growing with the spike.
  - A gate that never fires is not a gate.
- **Prove the gate on known-bad code.** Run it on a build known to fail and see `ok=false`; passing on good code
  proves nothing about the gate. *(field, E5)* Run on a build known to overflow, the extended gate failed it at
  spike 36.
- **Prove the probe on known-good code.** *(field, E6)* A gradient probe passed the LSE as `[B·Sq, Hq]` to an entry
  that expected `[B, Hq, Sq]`, gave every row another row's normaliser and reported gradients 7.5× too large; the
  same reading on the unmodified baseline exposed the probe.

## 3. The gate must reach the path under test
- **Defaults choose paths per head_dim, causality and window.** A kernel whose softmax path differs by head_dim
  can pass every D=128 check while the D=64 path never runs. List, per gated cell, which path ran
  (`flydsl_attn_measurement.md` §8).
- **Every configuration sweep carries the SNR gate.** *(field, E7)* Several `block_kv` values computed dQ silently
  wrong (8–14 dB) and one read 854 TF/s, exactly on target, because it skipped work.
- **Synchronisation, anchor and fence changes need a grid of configurations, with determinism.** *(field, E8)* Two
  faster-but-wrong variants (+2.53% and +0.31% on the dK/dV kernel's time) failed only at `q_split=4, BLOCK_KV=128`
  and at `BLOCK_KV=64`; the main shape passed. Cover at least the split and tile values the dispatcher can pick.
- **Each path that applies a window needs its own test.** A window bound that admitted W+2 keys sat in a
  fold path that the D=64 tests did not take, while the mask-style path they did take was right.
- Call at least twice and compare from the second call (`../gemm/flydsl_gemm_correctness.md` §2).

## 4. Poison workspaces
A `torch.empty` workspace that is read before it is written returns whatever the allocator left there; unit
tests pass, full runs fail. Fill workspaces with NaN in the gate. *(field, E10)* With a window narrower than the
q-block (W=32 against `BLOCK_Q=64`), a reduction read half-written bands; poisoning reproduced it exactly
(NaN on rows with `row % 64 ≥ 32`, only at W=32).

## 5. Forward and backward share state
- **Gate end to end.** The backward consumes the forward's O and LSE, so an edit to the LSE's layout or meaning
  keeps the forward's own SNR perfect and breaks the backward silently. Drive the backward with the optimised
  forward's own outputs, not the reference's:
  ```python
  o, lse = forward(q, k, v)                   # the side being optimised
  dq, dk, dv = backward(do, q, k, v, o, lse)  # its own o and lse
  ```
  *(field, E11)* Forward 51.7 dB, end-to-end gradients 48.8–49.3 dB; a 45 dB gate leaves normal variation alone.
- **Recompute P in the backward from the same rounded operand as the forward.** If the forward rounds
  `q·(sm·log2e)` to bf16 and the backward rounds `k·(sm·log2e)` instead, the two scores differ by δ ∝ |s|, P is
  multiplied by `2^δ`, and since `E[2^δ] > 1` the bias is systematic and grows with concentration. *(field, E12)* dV
  +4.6% (D=64) to +6.4% (D=128) at spike 96, read with the P clamp removed. *(field, E13)* Computing the backward
  scale in fp32 ("more exact") was worse in a CPU replay that tracked the GPU readings to within 3% (1.041 at
  D=64); mirroring the forward's rounding gave 1.0000 / 0.9999 there.
- **Keep one owner for each convention.** A pre-scaled tensor may be read elsewhere (the same LDS K feeding two
  GEMMs); a second path may reduce dQ. Publish the scale from the builder and have every consumer read it. A
  gradient off by a constant factor names the wrong constant: log2e (1.443), ln 2 (0.693), 1/√D, or
  1/(sm·log2e) (7.84 at D=128).
- The LSE pre-scaling format must match the exp mode on both sides (`flydsl_attn_softmax.md`).

## 6. Accumulator seeds
When the softmax scale c is applied to the score accumulator rather than folded into Q, everything seeded into
that accumulator must already be divided by c, or the scale applies twice.
- Enumerate every source of the accumulator's initial value (host LSE seeds, the reference written by a rebase
  branch) and audit the set closed. *(field, E14)* A missed second source broke only D=64 at high spike (dq ratio
  0.036), because the branch that produced it was enabled only for D ≤ 64 and fired only when a tile overran the
  reference. Find such a bug by intersecting the trigger conditions, not by reading line by line.
- **Dividing a masked-row floor by c overflows fp32.** A floor of −3.0e38 divided by c = 0.1275 (D=128) is
  −2.35e39. It fires only where a whole tile is masked for a row (a left window) and no finite bound replaced the
  floor first. **LSE breaks and O stays correct**, because O is normalised: *(field, E15)* 15360–29696 of 32768 LSE
  values non-finite with O clean. Seed zero and fold the reference into the scaling FMA (`fma(acc, c, −ref)`);
  gate LSE, not only O.

## 7. Locating layout errors
- **Use position-encoded inputs, not permutation-invariant checks.** Row sums of P equal to 1, a constant C, or
  matching value multisets all survive a wrong element-to-row map. *(field, E16)* A multiset comparison of two byte
  operand streams overlapped 89.1% and decided nothing.
- **Ramp-in-GEMM:** make one operand encode position (`K[kv, 0] = kv`, `Q[:, 0] = 1`, so S = kv) and read the map
  back element by element; a q-ramp checks the other axis. To locate a read, inject values constant along the
  contraction axis and varying along M: every lane must then return one constant, which separates a wrong value,
  a wrong M map and an unwritten byte.

## 8. Reading a difference
- **Compute `max|d|` with `torch.where(valid, d, 0)`, never `d * mask`.** Unwritten causal tails hold `-inf`;
  `-inf * 0` is NaN, and Python's `max(0.0, nan)` returns 0.0, so a whole table reads 0.0e+00 and looks
  bit-identical.
- **A metric at exactly 1.000 can be two errors cancelling.** *(field, E12)* A clamp turned a +6% dV bias into
  −6% (D=128; +4.6% into −7.6% at D=64); a dQ loss stayed hidden under a positive P bias until that bias was fixed.
  Confirm against an independent reference, or switch each known error source off and see whether the metric moves.
- **A "no-op" whose removal changes the result is doing something.** A clamp that shifts the result by 13% when
  removed is firing often; one that never fires on real data but drops SNR when removed is carrying a hazard.
  - *(field, E17)* A `min(s, 0)` clamp did not fire (largest argument −5.7), yet removing it took SNR to 10.5 dB,
    against 50.5 dB for a correct build: it was the only compiler-visible VALU read of an MFMA result feeding an
    inline-asm exp (`../gemm/flydsl_gemm_correctness.md` §7).

## 9. Mask conventions
- **`window_size = (W, 0)` keeps W+1 keys**, `kv ∈ [q + off − W, q + off]`, diagonal included. A reference or a
  kernel site that keeps W leaves out one edge key: *(field, E2)* ~22 dB at W=128 (measured at D=64); *(field, E1)*
  ~37 dB at W=1024, for D=64 and D=128 alike. Check the convention of the reference before blaming the kernel.
- **SDPA's `is_causal` is top-left aligned.** It equals the kernel's bottom-right causal mask only when Sq == Skv
  (a rectangular shape read SNR 0.25 *(field, E18)*) and cannot express a left window. Give rectangular and windowed
  references an explicit bottom-right mask.
- A masked SDPA reference falls back to the math backend, materialises `[B, H, Sq, Skv]` and keeps it for its own
  backward: put the reference forward and backward inside the same out-of-memory guard.

## Related
`../gemm/flydsl_gemm_correctness.md` · `../bottleneck/debug-flydsl-kernel.md` · `flydsl_attn_softmax.md` ·
`flydsl_attn_backward.md` · `flydsl_attn_measurement.md` · `flydsl_attn_schedule_sync.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the run's where the record dates it; where it does not, the line says which date it is. E13 is
a CPU replay, not a GPU run.

- E1 [2026-08, gfx950, FlyDSL ?, N=?, fp32 autograd reference called at the impl layer with either window convention] a reference keeping W keys read ~37 dB at D=64 (36.5/36.7/38.0) and ~38 dB at D=128 at W=1024; against the kernel's W+1 convention D=128 read O 53.2 / dq 52.0 / dk 50.4 / dv 50.6 dB and D=64 53.1 / 52.0 / 50.5 / 50.5 (FlyDSL attention backward with a 1024-key sliding window, D=64 and D=128; record dated 2026-08-02; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_flash_d128.md:25)
- E2 [2026-07, gfx950, FlyDSL ?, N=?, probe against an fp32 SDPA reference] backward masks that kept W keys instead of W+1 read ~22 dB on dq/dk/dv at window_left=128; after the fix 52.3/50.6/50.6 dB (FlyDSL attention backward, D=64, Hq64/Hkv8 and Hq128/Hkv16, dense and varlen, with and without sink; record dated by its file time, 2026-07-31; fix in Primus-Turbo 8eaab045; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_flash_sink.md:22)
- E3 [2026-09, gfx950, FlyDSL ?, N=?, gradient norm ratios against an fp32 reference at key spikes 1/24/48/96] dv 0.9412 at D=64 and 0.9515 at D=128 at spike 96 (0.9860 / 0.9889 at spike 24), with no non-finite value in any output; the commit gives the same D=128 values (dv 0.95148 at spike 96) and no non-finite output up to spike 96 (FlyDSL online-softmax forward with its backward, D=64 and D=128; Primus-Turbo e3e3008c; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_online_softmax_fast.md:322)
- E4 [2026-09, gfx950, FlyDSL ?, N=?, non-finite counts of the outputs over key spikes 24 to 48; the campaign's own gradient gate stopped at spike 24] the forward delivered by a campaign whose gate was green at spike 24 gave 0 non-finite values at spikes 24/28/32 and its first at spike 36 (128 at D=128, 64 at D=64), growing to 5029 / 4643 at spike 48 (FlyDSL attention forward 09cff04d, D=128 and D=64; gate record local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260917_153128/goal.md:50; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_online_softmax_fast.md:247)
- E5 [2026-09, gfx950, FlyDSL ?, N=?, NaN gate over spikes 1/8/24/36/48/96 at D=128 and D=64, failing on any non-finite O/LSE/dq/dk/dv] gate self-check: the pre-campaign base c2209644 ok=true; the earlier campaign's winner 09cff04d ok=false (D=128 spike 36: 128 non-finite in O, 262144 in dk) (FlyDSL attention forward and backward; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_online_softmax_fast.md:304)
- E6 [2026-09, gfx950, FlyDSL ?, N=?, a gradient probe calling the backward op directly] the forward's LSE passed as [B·Sq, Hq] to an entry expecting [B, Hq, Sq] read gradients 7.5× too large; the unmodified base read the same −17 dB, which exposed the probe (FlyDSL attention backward, shape not stated; record dated 2026-09-17; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_online_softmax_fast.md:20)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, conv TF/s sweep over the dq kernel's block_kv, then an SNR probe] block_kv 80/112/128/160 compute dQ silently wrong (8–14 dB); during the sweep block_kv 80 read 854 conv TF/s, on target (hd64 attention dq kernel, B=1 square causal; record dated by its file time, 2026-07-26; restated in local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/memory.md:113; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dq_inloop_l2_dma.md:26)
- E8 [2026-07, gfx950, FlyDSL ?, N=2, kernel-scoped interleaved A/B, median of 9, two rounds; SNR and bitwise-determinism grid over Sq, seed and block_kv] coarser hazard anchors: dK/dV kernel 7.3061 → 7.1463 ms (−2.53%) with dk/dv 16.89/17.85 dB and det=False at Sq 4096 and 8192 (q_split=4, BLOCK_KV=128); a min3 group 7.2472 → 7.2245 ms (−0.31%) with dk/dv 19.7 dB and det=False at BLOCK_KV=64 (hd64 attention backward, B=1 square causal, Sq to 16384; campaign 20260726_152547 rounds 16–17; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/rounds/round-16/agent.txt:18; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260726_152547/rounds/round-17/agent.txt:27)
- E10 [2026-08, gfx950, FlyDSL ?, N=?, the attention test suite, then a NaN-poisoned dQ workspace] at window_left=32, narrower than BLOCK_Q=64, the fused backward returned NaN dQ in a suite run and passed standalone; poisoning the workspace reproduced it: NaN in exactly the rows with row % 64 >= 32, at window 32 and at no other width (gpt-oss attention backward, fused dQ path; Primus-Turbo c34001fc; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_attn_bwd_fuse4_anyshape.md:124)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, the campaign bench's two 45 dB gates against fp32 references, run every round] the deployed forward read snr_fwd 51.7 dB; its own O and LSE driven through the backward gave dq/dK/dV 49.3 / 48.8 / 49.0 dB (gpt-oss D=64 attention forward, 4/64/8/8192/64; campaign 20260831_084030; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/memory.md:47)
- E12 [2026-09, gfx950, FlyDSL ?, N=?, dv norm ratio against an fp32 reference at key spike 96] with the backward pre-scaling K while the forward pre-scales Q: 0.937 (D=128) / 0.924 (D=64) with the P clamp, 1.064 / 1.046 without it; the clamp turns the positive bias negative (FlyDSL attention backward; record dated 2026-09-19; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_fwd_bwd_prescale_rounding_mismatch.md:24)
- E13 [2026-09, gfx950, FlyDSL ?, N=?, CPU replay built from the GPU probe's inputs at key spike 96, not a GPU run] dv norm ratio with fp32 scaling in the backward 0.991 (D=128) / 1.041 (D=64); mirroring the forward's Q pre-scaling 1.0000 / 0.9999; for the two arms the GPU read as 0.937/0.924 and 1.064/1.046 the replay gave 0.946/0.931 and 1.093/1.074 (FlyDSL attention backward; record dated 2026-09-19; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_fwd_bwd_prescale_rounding_mismatch.md:26)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, gradient norm ratios against an fp32 autograd reference at key spikes 1/8/32/96, D=64 and D=128] with the softmax scale on the score accumulator and the rebase branch's seed not pre-divided: dq 0.03612 (dk 0.7105, dv 0.34819) at D=64 spike 96, normal at D=128 and at low spikes; after the fix all within 7e-4 of 1.0 (FlyDSL attention forward and backward; Primus-Turbo 438692bc; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_score_accumulator_seed_compensation.md:22)
- E15 [2026-09, gfx950, FlyDSL ?, N=?, non-finite count of the forward's outputs] with the reference pre-divided by c and no score bound, under a left window: 23552 / 29696 / 15360 of 32768 LSE values non-finite at window_left 512 / 128 / 64, O clean (FlyDSL attention forward, S=2048, D=128, at Primus-Turbo 438692bc; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_masked_tile_floor_overflow.md:25)
- E16 [2026-09, gfx950, FlyDSL ?, N=?, LDS read probe comparing the operand multisets of two read paths] the multisets of the two byte operand streams overlapped 89.1%, which settled nothing (fp8 operand assembly for the scaled 16x16x128 MFMA of a gpt-oss attention backward; record dated 2026-09-03; local:/workspace/code/gpt_oss_docker/.claude/memory/reference_ds_read_tr8_b64_lane_map.md:66)
- E17 [2026-07, gfx950, FlyDSL ?, N=?, host model of the kernel's softmax arguments; SNR against an fp32 reference] largest softmax argument −5.742, 0 of 2.5e8 slots above 0; without the clamp 10.5 dB at every margin from 0 to 2.0 folded into the LSE; the no-clamp build with enable-post-misched=False read 50.5 dB, det=True, and was 1.7% slower (1157 vs 1177) (hd64 attention dK/dV kernel, fold path; record dated by its file time, 2026-07-26; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dkdv_clamp_hazard_truth.md:14)
- E18 [2026-08, gfx950, FlyDSL ?, N=?, attention test against an SDPA reference using is_causal] a rectangular shape (Sq ≠ Skv) read SNR 0.25 against the top-left aligned is_causal reference (FlyDSL attention, test suite; record dated 2026-08-17; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_attn_bwd_fuse4_anyshape.md:158)
