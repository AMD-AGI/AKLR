---
title: "Attention — correctness gates: spike inputs, paths the gate must reach, shared LSE, seeds and mask conventions"
kind: method
lever: attn_correctness
ops: [attention]
gens: [gfx950]
dtypes: [bf16]
regimes: [many query rows, dQ dK dV, causal, sliding-window, rectangular]
flydsl: 0.3.2
---

# Correctness for an attention kernel

> Several cases marked `*(field, E<n>)*` below passed a gate that reported clean. Use them to design the gate before
> trusting it. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at
> the end. Use them for direction and order of magnitude.
> The general GEMM rules — gates that reach the changed path, calling twice, races found by bitwise repetition,
> waits and barriers, inline-asm hazards, a third reference after a reduction-order change — are in
> `../gemm/correctness.md`. Debugging a kernel already known to be wrong starts in
> `../flydsl/debug-flydsl-kernel.md` (its §1 covers `-inf − -inf` and divide-by-zero NaNs).

## Route here when
- You change the softmax shift, the LSE, a scale, an accumulator seed, a mask, a barrier or an anchor.
- A gate passes and you have not shown it runs the path you changed, at the dynamic range that breaks it.
- One output is wrong and the others are right, or a metric reads exactly 1.000.

## 1. What a gate measures
For each output — O, LSE, dQ, dK, dV — against an fp32 reference:
- **SNR**, to catch structural errors. *(field, E19)* With independent Gaussian inputs correct bf16 kernels
  read ~50–54 dB on O and the gradients (48.8–49.3 dB end to end *(field, E11)*), but a correct dQ read 42–46 dB
  with the spiked keys of §2: take the threshold from a known-correct kernel on the same inputs, not a fixed 50 dB.
  A mask missing one edge key reads ~15 dB at W=32 but ~41 dB at W=2048.
- **Norm ratio `|got| / |ref|`**, to catch "finite but biased", which SNR and finiteness checks miss. *(field, E3)* dV
  fell to 0.94 at D=64 (0.95 at D=128) at spike 96 with no non-finite value anywhere.
- **Non-finite count** on every output, LSE included.
- **Bitwise determinism**: two runs, `torch.equal`. When a synchronisation change breaks a kernel outright, SNR and
  determinism fail together *(field, E8, E20)*; the bitwise check earns its place on rare races that leave SNR
  clean (`../gemm/correctness.md` §3).
- Score accuracy by the **worst** cell, not the mean: an average lets one row that is 5% off hide behind twenty
  correct rows.
- Check dQ, dK and dV separately; a change to a dQ scaling constant is invisible in dV.

## 2. Inputs that reach the numerics
- **Independent Gaussian inputs barely exercise the softmax range.** The row maximum then grows slowly with the
  logit scale and stays near the first tile; concentrated attention can put the maximum far
  from it. Build spike inputs: a few keys with large logits, placed away from the first KV tile, at several
  spike levels.
- **Push the spike to failure, not to "enough".** *(field, E4)* A gate that stopped at spike 24 reported clean; the
  same kernel overflowed beyond the gate's range.
  - A gate that never fires is not a gate.
- **Prove the gate on known-bad code.** Run it on a build known to fail and see `ok=false`; passing on good code
  proves nothing about the gate.
- **Prove the probe on known-good code**: a reading that the unmodified baseline reproduces belongs to the probe,
  not to the change.

## 3. The gate must reach the path under test
- **Defaults choose paths per head_dim, causality and window.** A kernel whose softmax path differs by head_dim
  can pass every D=128 check while the D=64 path never runs. List, per gated cell, which path ran
  (`measurement.md` §8).
- **Every configuration sweep carries the SNR gate.** *(field, E7)* Several `block_kv` values computed dQ silently
  wrong (8–14 dB) and one read 854 TF/s, exactly on target, because it skipped work.
- **Synchronisation, anchor and fence changes need a grid of configurations, with determinism.** *(field, E8)* Two
  faster-but-wrong variants (+2.53% and +0.31% on the dK/dV kernel's time) failed only at `q_split=4, BLOCK_KV=128`
  and at `BLOCK_KV=64`; the main shape passed. Cover at least the split and tile values the dispatcher can pick.
- **Each path that applies a window needs its own test.** A window bound that admitted W+2 keys sat in a
  fold path that the D=64 tests did not take, while the mask-style path they did take was right.
- Call at least twice and compare from the second call (`../gemm/correctness.md` §2).

## 4. Poison workspaces
A `torch.empty` workspace that is read before it is written returns whatever the allocator left there; unit
tests pass, full runs fail. Fill workspaces with NaN in the gate. *(field, E10)* With a window narrower than the
q-block (W=32 against `BLOCK_Q=64`), a reduction read half-written bands; poisoning reproduced it exactly
(NaN on rows with `row % 64 ≥ 32`, only at W=32).

## 5. The O kernel and the gradient kernels share state
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
  about +6% at spike 96, read with the P clamp removed. *(field, E13)* Computing the backward scale in fp32 ("more
  exact") still left D=64 4% off; mirroring the forward's rounding brought dV to 1.000.
- **Keep one owner for each convention.** A pre-scaled tensor may be read elsewhere (the same LDS K feeding two
  GEMMs); a second path may reduce dQ. Publish the scale from the builder and have every consumer read it. A
  gradient off by a constant factor names the wrong constant: log2e (1.443), ln 2 (0.693), 1/√D, or
  1/(sm·log2e) (7.84 at D=128).
- The LSE pre-scaling format must match the exp mode on both sides (`softmax.md`).

## 6. Accumulator seeds
When the softmax scale c is applied to the score accumulator rather than folded into Q, everything seeded into
that accumulator must already be divided by c, or the scale applies twice.
- Enumerate every source of the accumulator's initial value (host LSE seeds, the reference written by a rebase
  branch) and audit the set closed. *(field, E14)* A missed second source broke only D=64 at high spike (dq ratio
  0.036), because the branch that produced it was enabled only for D ≤ 64 and fired only when a tile overran the
  reference. Find such a bug by intersecting the trigger conditions, not by reading line by line.
- **Dividing a masked-row floor by c overflows fp32.** A floor of −3.0e38 divided by c = 0.1275 (D=128) is
  −2.35e39. It fires only where a whole tile is masked for a row (a left window) and no finite bound replaced the
  floor first. **The LSE breaks** *(field, E15)*. Seed zero and fold the reference into the scaling FMA
  (`fma(acc, c, −ref)`); gate LSE, not only O.

## 7. Locating layout errors
- **Use position-encoded inputs, not permutation-invariant checks.** Row sums of P equal to 1, a constant C, or
  matching value multisets all survive a wrong element-to-row map.
- **Ramp-in-GEMM:** make one operand encode position (`K[kv, 0] = kv`, `Q[:, 0] = 1`, so S = kv) and read the map
  back element by element; a q-ramp checks the other axis. To locate a read, inject values constant along the
  contraction axis and varying along M: every lane must then return one constant, which separates a wrong value,
  a wrong M map and an unwritten byte.

## 8. Reading a difference
- **Compute `max|d|` with `torch.where(valid, d, 0)`, never `d * mask`.** Unwritten causal tails hold `-inf`;
  `-inf * 0` is NaN, and Python's `max(0.0, nan)` returns 0.0, so a whole table reads 0.0e+00 and looks
  bit-identical.
- **A metric at exactly 1.000 can be two errors cancelling.** *(field, E12)* A clamp turned a +6% dV bias into
  −6%; a dQ loss stayed hidden under a positive P bias until that bias was fixed.
  Confirm against an independent reference, or switch each known error source off and see whether the metric moves.
- **A "no-op" whose removal changes the result is doing something.** A clamp that shifts the result by 13% when
  removed is firing on values that carry the result, possibly only a few large ones; one that never fires on real
  data but drops SNR when removed is carrying a hazard.

## 9. Mask conventions
- **`window_size = (W, 0)` keeps W+1 keys**, `kv ∈ [q + off − W, q + off]`, diagonal included. A reference or a
  kernel site that keeps W leaves out one edge key (S1). Check the convention of the reference before blaming the
  kernel.
- **SDPA's `is_causal` is top-left aligned.** It equals the kernel's bottom-right causal mask only when Sq == Skv
  and cannot express a left window. Give rectangular and windowed references an explicit bottom-right mask.
- In PyTorch 2.11 on ROCm, a masked SDPA reference with grouped KV heads (`enable_gqa=True`) falls back to the math
  backend, materialises an fp32 `[B, Hq, Sq, Skv]` and keeps it for its own backward: put the reference forward and
  backward inside the same out-of-memory guard. *(field, E21)* At Hq=64, S=4096 the backward peaked at 17.4 GB;
  with equal head counts the same mask ran on the efficient backend.

## Related
`../gemm/correctness.md` · `../flydsl/debug-flydsl-kernel.md` · `softmax.md` ·
`gradients.md` · `measurement.md` · `schedule_sync.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. The month is the run's where the record dates it; where it does not, the line says which date it is.

- E3 [2026-09, gfx950, FlyDSL ?, N=?, gradient norm ratios against an fp32 reference at key spikes 1/24/48/96] dv 0.9412 at D=64 and 0.9515 at D=128 at spike 96 (0.9860 / 0.9889 at spike 24), with no non-finite value in any output; the commit gives the same D=128 values (dv 0.95148 at spike 96) and no non-finite output up to spike 96 (FlyDSL online-softmax forward with its backward, D=64 and D=128; Primus-Turbo e3e3008c; its child 86be8d0a gives dV before its change as 0.986 / 0.941 at D=64 and 0.989 / 0.951 at D=128, spikes 24 / 96; D=64 spike 96 dv 0.941169)
- E4 [2026-09, gfx950, FlyDSL ?, N=?, the campaign's own gradient gate stopped at spike 24] the forward delivered by a campaign whose gate was green at spike 24 detected an outrun reference from the row sum, which reports only after P has overflowed; unrolling that detection 4 → 1 moved the failure point by zero (FlyDSL attention forward, D=128 and D=64; Primus-Turbo e3e3008c)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, conv TF/s sweep over the dq kernel's block_kv, then an SNR probe] block_kv 80/112/128/160 compute dQ silently wrong (8–14 dB); during the sweep block_kv 80 read 854 conv TF/s, on target (hd64 attention dq kernel, B=1 square causal)
- E8 [2026-07, gfx950, FlyDSL ?, N=2, kernel-scoped interleaved A/B, median of 9, two rounds; SNR and bitwise-determinism grid over Sq, seed and block_kv] coarser hazard anchors: dK/dV kernel 7.3061 → 7.1463 ms (−2.53%) with dk/dv 16.89/17.85 dB and det=False at Sq 4096 and 8192 (q_split=4, BLOCK_KV=128); a min3 group 7.2472 → 7.2245 ms (−0.31%) with dk/dv 19.7 dB and det=False at BLOCK_KV=64 (hd64 attention backward, B=1 square causal, Sq to 16384)
- E10 [2026-08, gfx950, FlyDSL ?, N=?, the attention test suite, then a NaN-poisoned dQ workspace] at window_left=32, narrower than BLOCK_Q=64, the fused backward returned NaN dQ in a suite run and passed standalone; poisoning the workspace reproduced it: NaN in exactly the rows with row % 64 >= 32, at window 32 and at no other width (gpt-oss attention backward, fused dQ path; Primus-Turbo c34001fc)
- E11 [2026-08, gfx950, FlyDSL ?, N=?, the campaign bench's two 45 dB gates against fp32 references, run every round] the deployed forward read snr_fwd 51.7 dB; its own O and LSE driven through the backward gave dq/dK/dV 49.3 / 48.8 / 49.0 dB (gpt-oss D=64 attention forward, 4/64/8/8192/64)
- E12 [2026-09, gfx950, FlyDSL ?, N=?, dv norm ratio against an fp32 reference at key spike 96] with the backward pre-scaling K while the forward pre-scales Q: 0.937 (D=128) / 0.924 (D=64) with the P clamp; the clamp turned the bug's +6% into −6%, and removing it moved dV by 13% (FlyDSL attention backward; Primus-Turbo 86be8d0a)
- E13 [2026-09, gfx950, FlyDSL ?, N=?, gradient norm ratios against an fp32 autograd reference at key spikes 1/24/96/192] the backward mirroring the forward's Q pre-scaling: dV 1.000 at every spike at D=128 and D=64; the commit adds that scaling the fp32 accumulator exactly instead "still leaves D=64 4% off" (FlyDSL attention backward; Primus-Turbo 86be8d0a)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, gradient norm ratios against an fp32 autograd reference at key spikes 1/8/32/96, D=64 and D=128] with the softmax scale on the score accumulator and the rebase branch's seed not pre-divided: dq 0.03612 at D=64 spike 96, normal at D=128 and at low spikes; after the fix all within 7e-4 of 1.0 (FlyDSL attention forward and backward; Primus-Turbo 438692bc)
- E15 [2026-09, gfx950, FlyDSL ?, N=?, ?] with the reference pre-divided by c, a fully-masked prologue tile under a left window reached the −3.0e38 floor through the division, overflowed fp32 and left the LSE non-finite (FlyDSL attention forward, at Primus-Turbo 438692bc; Primus-Turbo 0a8e4d17)
- E19 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, aiter dbd8bf5, FlyDSL 0.3.2, N=2 seeds per cell, SNR against an fp32 reference] correct bf16 kernels, independent Gaussian inputs: O, dQ, dK, dV 50.5–54.0 dB; 64 keys scaled by 8 or 24: dQ 42.5–46.0 dB; one missing edge key (W keys against the W+1-key reference): W=32 14.5–15.0, W=2048 40.9–42.0 dB (SDPA, aiter fmha, FlyDSL forward; D=64 and 128, S=1024 and 4096)
- E20 [2026-11, gfx950 (MI355X), HIP 7.15, N=100 runs per cell as 50 pairs, SNR fails below 45 dB] an LDS-tiled bf16 MFMA GEMM with LDS barriers replaced by a scheduling barrier, six variants on three shapes: 15 of 18 cells broke the result, each failing determinism and SNR together, none determinism alone
- E21 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0 with AOTriton, N=1 per configuration, backend from the profiler's aten ops] SDPA with an explicit bool mask, Hq=64, Hkv=8, `enable_gqa=True`: math backend, an fp32 [1, 64, Sq, Skv] saved for the backward, at Sq=Skv=4096 4.50 GB retained after the forward and a 17.4 GB backward peak; Hq=Hkv=8: efficient backend (B=1, D=64)
- S1 Primus-Turbo@8eaab045:primus_turbo/flydsl/attention/flash_attn_bwd.py:930 the backward keeps `kv >= q+off-W` (W+1 keys), matching the forward; the commit moved four backward mask sites from W to W+1 keys
