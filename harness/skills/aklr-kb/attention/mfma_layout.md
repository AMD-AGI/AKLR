---
title: "Attention — MFMA timing, atom choice, fragment layouts and fp8 MFMAs on gfx950"
kind: topic
lever: attn_mfma_layout
ops: [attention]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [prefill, decode, backward, causal]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3500
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3732
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:182
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3199
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:5560
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:520
---

# MFMA atoms and fragment layouts in an attention kernel

> Attention chains GEMMs through registers: S (or dP) leaves one MFMA as a C fragment and enters the next one
> as an operand. The atom therefore fixes the softmax lane map, the reductions, the store and the register
> footprint, not only the MAC rate. GEMM-side atom facts are in `../gemm/mainloop.md` §5, per-lane
> register counts per atom in `../hardware/gfx950/matrix_core.md`, transposed-read lane maps in
> `lds.md` §3.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- You are choosing the atom of an attention GEMM, or porting a kernel from one atom to another.
- You need the lane ↔ (row, column) map of an operand or accumulator to write a mask, a reduction, a pack or
  a store.
- You are counting how much softmax work fits under the MFMAs.
- You are considering fp8 for one of the GEMMs.

## 1. MFMA timing on gfx950 (measured)
The timings of this section were measured with HIP inline-asm microbenchmarks, not inside a kernel.

| Instruction | Pipe cycles | Issue blocked | VALU shadow per MFMA |
|---|---|---|---|
| `v_mfma_f32_16x16x32_bf16` | 16 | 8 | 8: one `v_exp_f32` or two `v_mul_f32` *(field, E1, E2)* |
| `v_mfma_f32_32x32x16_bf16` | 32 | 8 | 24: three exps or six multiplies *(field, E1, E2)* |
| `v_mfma_f32_16x16x16_bf16` | 16 | 4 | 12: one exp or three multiplies *(field, E1, E2)* |

- An MFMA followed by k VALU ops takes `max(pipe cycles, X + 8·k_exp + 4·k_mul)` cycles in one wave, with
  X = 8 for the two 128-bit-operand atoms and 4 for 16x16x16; every microbenchmark point fit exactly *(field, E2)*. Each exp or
  multiply beyond the shadow adds its full cost.
- A chain dependent through srcC (srcC = previous vdst) runs at the pipe rate: the compiler inserts no `s_nop`,
  and the chain costs nothing extra. Two or four independent accumulators run at the same 16 or 32 cycles per
  instruction *(field, E1)*. Splitting an accumulation into independent chains buys no latency ILP.
- One MFMA pipe per SIMD: MFMA streams of two waves on one SIMD run one after the other, and the shadow is
  shared by every wave on that SIMD *(field, E3)*. exp and mul draw on the same shadow (`softmax.md` §1).
- `16x16x16_bf16` takes as long as `16x16x32` for half the K, i.e. half the MAC rate. The "4-cycle issue,
  12-cycle shadow" figure belongs to it, not to 16x16x32 *(field, E1, E2)*.
- 16x16x32 and 32x32x16 have the same MAC rate (512 MAC per cycle), but per MAC the 32x32x16 atom leaves 1.5×
  the VALU shadow (24 of 32 cycles against 8 of 16).
- **Softmax budget.** A 32 kv × 32 q tile at head dim D needs D/8 `32x32x16` MFMAs for QK and P·V together,
  i.e. 3·D shadow cycles, against 16 `v_exp_f32` per wave (128 cycles) plus the max, shift, pack and sum. At
  D = 64 the exps alone take two thirds of the shadow; at D = 128 one third. FlyDSL's forward groups one
  `32x32x16` with three exps (`_sched_barrier_exp_pairs(traits, 6, 3, …)`), exactly one shadow.
- These are pipe facts. At the power cap rank atoms and schedules by wall time and sclk
  (`attribution.md` §8), and use the shadow table only to prefilter
  (`registers_occupancy.md` §8).

## 2. An MFMA ignores EXEC
- On gfx950 an MFMA computes and writes all 64 lanes whatever EXEC holds. In a HIP inline-asm microbenchmark, with
  EXEC = 0, one lane or half the lanes, `v_mfma_f32_16x16x32_bf16` still wrote its full result to every lane, and
  the operands of masked-off lanes contributed to it; a `v_mov_b32` in the same region honoured EXEC *(field, E4)*.
- A lane mask or a divergent `if` neither skips an MFMA nor protects its destination. Skip MFMA work with a
  wave-uniform branch, and mask scores with `-inf` (exp2 then gives exactly 0). Per-wave classification of
  diagonal blocks: `backward.md` §8.

## 3. Fragment layouts (bf16, wave64)
`D[m][n] = Σ_k A[m][k]·B[k][n] + C[m][n]`; s indexes the 8 bf16 values of an operand register group, t and e
the f32 accumulator elements of a lane.

| Atom | A | B | C / D |
|---|---|---|---|
| 16x16x32 | m = lane%16, k = (lane//16)·8 + s | n = lane%16, k = (lane//16)·8 + s | n = lane%16, m = (lane//16)·4 + t, t < 4 |
| 32x32x16 | m = lane%32, k = (lane//32)·8 + s | n = lane%32, k = (lane//32)·8 + s | n = lane%32, m = (e//4)·8 + (lane//32)·4 + e%4, e < 16 |

- In C the column n is the lane and the row m runs over the lane's elements: a lane owns part of one column.
  16x16x32 gives a lane 4 consecutive m; 32x32x16 gives it four runs of 4 consecutive m, the neighbouring runs
  sitting in lane ^ 32.
- Prove a map element by element with a position-encoded input (a ramp in one operand, ones in the other).
  Keep the ramp exactly representable in bf16 (integers ≤ 256). Row-sum and sum-over-k checks are
  permutation-invariant. *(field, E5)* A 32x32x16 port passed `rowsum(P) = 1`
  and probes that varied C along one index only, while its operand orientation and pack order were wrong.

## 4. Orientation: compute Sᵀ and keep P inside the lane
- **Sᵀ.** Issue QK with A = K and B = Q. C is then Sᵀ: n = q is the lane, m = kv runs over the elements, so a
  lane holds the scores of one q row (16 kv per 32x32x16 tile, the other 16 in lane ^ 32). Row max and row sum
  are in-lane folds plus one `permlane32_swap`, and per-row values (running max, l, LSE, delta) are one per
  lane, with no gather. FlyDSL's gfx950 forward computes QK this way (`qk`) and reduces with one
  `permlane32_swap` (`_reduction_pair`).
- Both QK operands stay contiguous in d (`[kv, d]` and `[q, d]` in LDS); the only transposed read is Vᵀ for
  P·V (`lds.md`).
- **P or dS feeding the next GEMM is locked to the previous GEMM's C layout.** Orient the previous GEMM so the
  next GEMM's contraction index is C's element index m, never the lane index n. P then packs inside the lane:
  FlyDSL's forward passes P as B of P·V with A = Vᵀ. With the contraction index on the lane, every tile needs a
  cross-lane transpose.
- The k order is free as long as both operands of an MFMA agree. Address the other operand's transposed reads
  (two `ds_read_b64_tr_b16` per 8-wide fragment) so they produce C's kv order, `(e//4)·8 + (lane//32)·4 + e%4`
  for 32x32x16; the 16 C elements then pack straight into two B registers. *(field, E6)* A pack that added a
  `permlane32_swap` on top of matching orders exchanged kv 4–7 with 8–11 and halved dQ. For 16x16x32, two
  adjacent 16-kv C tiles give a lane the 8 kv of one B register.
- **Store.** A 32x32x16 O fragment holds runs of 4 consecutive d per lane; FlyDSL's forward swaps halves with
  `permlane32_swap` to assemble 8 consecutive bf16 (16 B) per lane per store (`_store_permlane_o`). A
  16x16x32 fragment's 4 consecutive values go out directly as one 8-byte store.

## 5. Choosing the atom per GEMM
- Dump the atom of every GEMM in both passes before reasoning: the forward and the backward of one code base
  can use different atoms. Changing the atom is a port — fragment layout, transposed-read addressing,
  register footprint and softmax lane map all change. Read the ISA delta (`v_mfma` count, VGPR, spill,
  `ds_read` mix) before timing.
- **Backward: 16x16x32.** *(field, E7)* Rewriting a dK/dV kernel from 32x32x16 cut its time 1415 → 1168 µs and the whole
  backward gained 8.7% (`SQ_WAIT_ANY` −47%; a full rewrite, the cause was not isolated); *(field, E8)* a dq body ran 1204 TF/s
  against 1088–1093, and even without exps the correct 32x32x16 version (1198) stayed below 16x16x32 (1220); *(field, E9)* on a D=128
  backward, one GEMM switched back to 32x32x16 alone cost 12%. What does hold either way:
  - the C fragment is 4 f32 per lane instead of 16, so the S, dP and dS of a tile live in small pieces and the
    transient register peak is lower (a 32x32x16 dq port spilled until restructured to one kv tile at a
    time) *(field, E10)*;
  - its 4 consecutive values per lane store without a permlane;
  - the accumulator registers for a given output block are the same for both atoms.
  Independent-chain latency ILP is not a reason (§1).
- **Forward: 32x32x16.** FlyDSL's gfx950 forward uses `v_mfma_f32_32x32x16_bf16` (`…_f16` for f16 inputs) for QK
  and P·V (`flash_attn_utils.py:3500`). It leaves 1.5× the VALU shadow per MAC (§1), needs one swap per row
  reduction (§4) and keeps one q row per lane.
  *(field, E11)* A D=64 forward ported to 16x16x32, spill-free and numerically exact, ran 8.4% slower;
  *(field, E12)* a spill-free arm with part of the O accumulator removed read 2.4% slower on the scorer and 1.2%
  faster in a sustained loop, where it clocked higher.
- **Decode P·V: `16x16x32`, not `16x16x16bf16_1k`** (half the MAC rate, §1). On an MLA decode producer the switch
  alone cut an estimated 8–14% of producer time, with the operand orientation unchanged.
- The GEMM finding that a smaller atom moves fewer register-file bytes per MAC and can clock higher at the power
  cap (`../gemm/mainloop.md` §5) applies here too: rank atoms on wall time, sclk and power, never on
  MFMA utilization.

In FlyDSL: `../flydsl/attention.md` § mfma_layout §5.

## 6. Row sums on the matrix pipe
- The softmax row sum can be an MFMA with a constant all-ones A over the packed P. Use 16x16x32: it needs half
  the pipe cycles of a 32x32x16 row sum; by the measured kernel's own accounting a 32x32x16 row sum would cost
  more than the VALU adds it replaces.
- Layout, with packed P as B of a 32x32x16 P·V (n = q = lane%32, k = kv): read the same registers as B of a
  16x16x32. Lane groups 0 and 2 then hold the two kv halves of q = n, groups 1 and 3 those of q = n + 16. Fill
  A with bf16 1.0 (`0x3F803F80` per dword) in lanes where `lane%4 == 0 and (lane//4)%2 == (lane//16)%2`, zero
  elsewhere. Element 0 of each lane's D is the sum for its own q = lane%32, half-wave partner included: the adds
  and the `permlane32_swap` disappear.
- *(field, E13)* Per iteration `v_add_f32` 68 → 0 and `permlane32_swap` 2 → 0 for 8 new 16x16x32 MFMAs; VGPR
  162 → 150; +1.96%; SNR 51.26 → 51.61 dB because numerator and denominator share the bf16 P.
- These MFMAs anchor register allocation: removing or replacing them raised spills sharply
  (`registers_occupancy.md` §4). *(field, E14)* Restoring an MFMA row sum in a different online forward
  was within noise; price it per kernel.
- An MFMA that reads P straight from an inline-asm pack needs padding the compiler cannot add
  (`schedule_sync.md`). *(field, E15)* Symptom: non-deterministic results with errors on half of the q
  rows.

## 7. More rows per wave
- Two M tiles per wave sharing every K/V fragment halve the operand reads per MAC but double the output state:
  O is `ROWS·D/64` and Q `ROWS·D/128` dwords per lane. Two q heads or two row groups cost the same registers.
- *(field, E16)* D=64 forward, rows per wave doubled: non-MFMA instructions per unit of work 194 → 151.5, sclk +9.2%
  at the power cap, but loop-carried VGPRs 100 → 192, occupancy 4 → 2 waves per SIMD, cycles +11.5%: net
  −0.86%. Price the occupancy step first (`registers_occupancy.md` §3).

## 8. fp8 MFMAs
- Only the wide-K f8f6f4 forms raise throughput: 16x16x128 and 32x32x64 do 4× the K of the bf16 atoms in 2×
  the cycles. The legacy fp8 forms (`16x16x32_fp8_fp8`, `32x32x16_fp8_fp8`) take the bf16 cycle counts for the
  same K (32 cycles per `32x32x16_fp8_fp8`, the same as bf16) *(field, E17)*. Name the shape whenever an fp8 speed-up
  is quoted. *(field, E17)* Moving a forward's P·V to the legacy fp8 form bought no MFMA throughput.
- The scaled forms (`v_mfma_scale_f32_16x16x128_f8f6f4`, `…_32x32x64_…`) take E8M0 scales (bias 127): scale bytes
  of 127 (×1.0) give the unscaled product (`../hardware/gfx950/matrix_core.md` §2).
- In LLVM the wide-K forms exist only as the scale intrinsics `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4` and
  `llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4` (`IntrinsicsAMDGPU.td:3732-3733`); when both scale operands are the
  constant 0, the backend selects the unscaled `v_mfma_f32_16x16x128_f8f6f4` / `…_32x32x64_…` encoding
  (`UnscaledMFMAOptimizationPat`, `SIInstrInfo.td:316-327`; LLVM `e2a39f50`, the one FlyDSL v0.3.2 pins).
- A GEMM qualifies only with a contraction ≥ 128. At D = 64, QK and dP (contraction D) and dV and dK
  (contraction BLOCK_Q = 64) do not; dQ (contraction BLOCK_KV) can. *(field, E18)* Storing only the dQ partials in
  fp8 saved bytes, not MFMA time: +3.5–4.1% with dq SNR 47.9 → 31.3 dB.
- A scale must be constant along the contraction axis to leave the MFMA. A tensor contracted along different
  axes in two GEMMs — dO along d in dP = dO·Vᵀ and along q in dV = Pᵀ·dO; dS along q and along kv — needs a
  per-tensor scale or two quantized copies. That is why fp8 attention defaults to per-tensor scales.
- fp8 operands do not shrink the f32 accumulators; the S/dP register peak stays where it was.
- Converting e4m3 to bf16 for a bf16 MFMA: `v_cvt_scalef32_pk_bf16_fp8` with scale 1.0 is one bit-exact
  instruction per pair, half the VALU of `v_cvt_pk_f32_fp8` plus `v_cvt_pk_bf16_f32` *(field, E21)*.
- Treat accuracy as a budget: error powers of independent quantizations add. *(field, E19)* In a PyTorch model of
  the D=64 backward, Q and K dominate (an S error is amplified by the exponential), P and dS are the cheapest;
  finer scales recover only ~0.5–0.9 dB because the e4m3 mantissa dominates; *(field, E20)* in a PyTorch model of
  the forward's P, e5m2 is ~6 dB worse than e4m3; fp8 P at D = 64 is estimated near 27.6 dB, below the forward's
  48 dB gate.
- fp8 P also constrains the softmax shift (`softmax.md` §6.1, §6.5).

In FlyDSL: `../flydsl/attention.md` § mfma_layout §8.

## Verify
- The atom of every GEMM in both passes is read from the ISA, not assumed.
- Lane maps and k orders are proven element by element with position-encoded inputs.
- No MFMA relies on EXEC to be masked or skipped.
- An atom or rows-per-wave change reports VGPR, spill, occupancy and sclk next to the wall time.

## Related
`softmax.md` · `lds.md` · `registers_occupancy.md` ·
`schedule_sync.md` · `backward.md` · `attribution.md` ·
`../gemm/mainloop.md` · `../hardware/gfx950/matrix_core.md`

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. E1–E4 are HIP inline-asm microbenchmarks and E19–E20 torch models, not FlyDSL kernels. The month is the
run's; for a KB note without a run record, the date the note gives.

- E1 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] `v_mfma_f32_16x16x32_bf16` 16.000 and `v_mfma_f32_32x32x16_bf16` 32.000 cycles per instruction with 1, 2 or 4 accumulator chains, the single chain dependent through srcC (the compiler emits no `s_nop` in it); `v_mfma_f32_16x16x16_bf16` 16.0 cycles, half the FLOP rate of 16x16x32 (microbenchmark, one wave; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:107)
- E2 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] one wave issuing an MFMA followed by k `v_exp_f32` or k `v_mul_f32`: cycles per unit = max(pipe cycles, X + 8·k_exp + 4·k_mul) at every measured point, X = 8 for 16x16x32 and 32x32x16 and 4 for 16x16x16, i.e. VALU shadows of 8, 24 and 12 cycles (microbenchmark; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:82)
- E3 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=?, per-SIMD window max(t1) − min(t0), median of the launches (repeat count not recorded; the probe's default is 5)] two waves on one SIMD: two 16x16x32 MFMA streams 272.1 and 527.1 cycles per iteration; a wave streaming `v_exp_f32` or `v_mul_f32` beside a 16x16x32 stream 404.7 against 284.1 cycles alone, about 7.5 cycles per MFMA (microbenchmark `p1c_dualwave.hip`; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:67; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/raw/r_p1c_v5.csv:14)
- E4 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=1, one launch per EXEC mask, all 64 lanes read back] `v_mfma_f32_16x16x32_bf16` with A = B = bf16 1.0 and C = 0 under EXEC = all lanes, 0x1, the low 32 lanes, 0xFFFF0000FFFF0000 and 0: destination written in 64 of 64 lanes and 256 of 256 elements equal to 32.0 in every case; a `v_mov_b32` in the same region wrote 64, 1, 32, 32 and 0 lanes (microbenchmark `p4_mfma_exec.hip`; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:177; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/raw/r_p4.txt:15)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, debug probes in the dq bench `_bench_dwpipe.py`] a 32x32x16 dq port read `rowsum(P)` = 1 and matched probes with C = kv, C = q and C = 1 while its GEMM1 orientation was reversed and its pack exchanged kv 4–7 with 8–11; only C = P, varying in kv and q, exposed the pack (SNR 5.9 dB, dQ ratio ~0.47) (hd64 attention backward dq kernel; KB note, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dq_dualwave_hwexp.md:150)
- E6 [2026-07, gfx950, FlyDSL ?, N=?, debug probes in the dq bench `_bench_dwpipe.py`] a `permlane32_swap` added to the C → B-operand pack, whose k orders already matched, exchanged kv 4–7 with kv 8–11: C = P gave SNR 5.9 dB and a dQ ratio of ~0.47; packing in lane order gave 55.6 dB, real dq 52.5 dB (hd64 attention backward dq kernel, 32x32x16 port; KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dq_dualwave_hwexp.md:160)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, campaign harness bench (backward TF/s) and `_prof_dkdv_occ.py` PMC] dK/dV kernel rewritten from 32x32x16 to 16x16x32: kernel 1415 → 1168 µs, `SQ_WAIT_ANY` 7.2e8 → 3.8e8 (−47%), backward 581.6 → 632 TF/s (+8.7%) (gpt-oss attention backward dkdv kernel, bench shape b1 s8192; KB note dated 2026-07-10, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_dkdv_bwd_16x16.md:16)
- E8 [2026-07, gfx950, FlyDSL ?, N=?, dq bench `_bench_dwpipe.py`] dq body: 16x16x32 1204 TF/s, the correct 32x32x16 port 1088–1093; without exps 32x32x16 1198 against 16x16x32 1220 (hd64 attention backward dq kernel; KB note, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dq_dualwave_hwexp.md:164)
- E9 [2026-08, gfx950, FlyDSL ?, N=?, ?] GEMM2 switched from 16x16x32 to 32x32x16 (`g2_w32`): +12% (llama D=128 attention backward, cell not stated; KB note, no run record found; a campaign goal of 2026-08-24 already lists it as history; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_attn_bwd_1000tf_push.md:113)
- E10 [2026-07, gfx950, FlyDSL ?, N=?, register counts from a system trace and the dq bench] first 32x32x16 dq port: VGPR 128 with 308 scratch (spill); computing S, dP and C one kv tile at a time: scratch 308 → 16; loading K per kv tile as well: scratch 0 (hd64 attention backward dq kernel; KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_dq_dualwave_hwexp.md:138)
- E11 [2026-08, gfx950, FlyDSL ?, N=1, campaign scorer `fwd64` (WARMS=5, REPS=40, min of 40, five fresh processes)] the spill-free, numerically exact 16x16x32 port (vgpr 128, spill 0, SNR bit-identical): `fwd64` 0.9234, 2.0407 ms = 1077.6 TF/s, −8.4% (gpt-oss D=64 attention forward, B=4 Hq=64 Hkv=8 S=8192 causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/rounds/round-05/NOTES.md:40)
- E12 [2026-08, gfx950, FlyDSL ?, N=2, palindromic min-of-200, two reads per arm; a 30000-rep sustained loop with rocm-smi] a spill-free 16x16x32 pricing arm with 16 O accumulators removed (`DBG_O=4`, vgpr 126): 1.894 / 1.897 ms against control 1.848 / 1.855 (−2.4%); sustained 1.871 against 1.8945 ms (+1.2%) at 1877 MHz / 1375 W against 1668 MHz / 1400 W (same forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260831_084030/rounds/round-04/agent.txt:10)
- E13 [2026-07, gfx950, FlyDSL 0.2.2, N=3, interleaved A/B in one session, three reads per arm, scorer `_bench_campaign_fwd16384.py`] online-softmax row sum as `v_mfma_f32_16x16x32_bf16` with an all-ones A: 1159.4 / 1159.1 / 1159.9 against 1137.0 / 1137.0 / 1137.2 TF/s (+1.96%); per iteration `v_add_f32` 68 → 0, `v_permlane32_swap` 2 → 0, 8 new 16x16x32 MFMAs; `.vgpr_count` 162 → 150; SNR 51.26 → 51.61 dB (Primus-Turbo hd64 flash-attention forward, B=1, Sq=Skv=16384, full causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-12/agent.txt:12)
- E14 [2026-09, gfx950, FlyDSL ?, N=?, online-path overhead against the fixed-reference path] the MFMA row sum restored on the online path: overhead +15.97% → +15.31%, inside the noise (llama-8B D=128 attention forward, online softmax; KB note dated 2026-09-17, no run record found; local:/workspace/code/gpt_oss_docker/.claude/memory/project_flydsl_online_softmax_fast.md:67)
- E15 [2026-07, gfx950, FlyDSL 0.2.2, N=2, the round's gate (SNR and run-to-run determinism) and a per-q-row residual probe] the row-sum MFMA reading P right after the inline-asm `cvt_pk_bf16_f32`: SNR 25.01 and 24.99 dB, det=False, bad rows exactly `(q//4)%2==0` with residuals 0.06–0.09 against 0.0027 on good rows; `sched_barrier(0)` + `s_nop` before the MFMAs fixed it (SNR 51.61 dB, det=True) (same hd64 forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-12/agent.txt:12)
- E16 [2026-09, gfx950, FlyDSL ?, N=2, campaign scorer `fwd64` for the arm (control N=9, spread 0.65%); sclk and power from a sustained run] `ROWS_PER_WAVE` 32 → 64: non-MFMA instructions per unit of work 194 → 151.5, sclk 1632 → 1783 MHz (+9.22%) at a 1394 W cap, cycles +11.51%, loop-carried VGPRs 100 → 192 (vgpr 240), 4 → 2 waves/SIMD; best arm 0.99595 against control 1.00456 (−0.8571%) (gpt-oss D=64 attention forward, B=4 Hq=64 Hkv=8 S=8192; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260901_123110/rounds/round-12/agent.txt:51)
- E17 [2026-07, gfx950, FlyDSL 0.2.2, N=1, rocprofv3 PMC (MFMA busy / MFMA instructions) and the scorer, one run per arm; operands bit-reinterpreted, SNR not gated] P·V on legacy `v_mfma_f32_32x32x16_fp8_fp8`: MFMA-busy cycles identical to the bf16 arm, 4.87e9 / 1.69e8 = 28.8 cycles per MFMA averaged over the loop's 32x32x16 and 16x16x32 MFMAs, the bf16 model (32×32 + 8×16)/40, so the fp8 form takes bf16's 32 cycles; wall +1.42% from halving operand width, spill 2 → 20; native `v_mfma_f32_32x32x64_f8f6f4` cut MFMA busy 22.2% but spilled 120 dwords (439.8 TF/s) (same hd64 forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-23/agent.txt:22)
- E18 [2026-08, gfx950, FlyDSL ?, N=3, campaign bench, three reads of the score cell; the round's gate read two (+3.51% against the previous best 978.30)] dQ split-K partials stored as E4M3: 1015.7 / 1015.4 / 1015.2 TF/s against a ~975 clean-tree median (+4.1%); dq SNR 31.32–31.39 dB at D64 against 47.9 dB in bf16, dk/dv unchanged; the campaign's total over its four kept changes 970.5 → 1012.65 (+4.34%) (gpt-oss D=64 attention backward, B=4 Hq=64 Hkv=8 S=8192 full causal; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260826_153437/memory.md:181)
- E19 [2026-09, gfx950, torch (no FlyDSL kernel), N=?, torch simulation of the backward against fp32] per-tensor E4M3 on one tensor at a time: k → dq 28.4 dB, q → dk 28.7 dB, any other tensor 31.3–32.3 dB; all tensors 23.2 dB; 32-element MX scales: all tensors 24.1 dB (+0.9), the four operands 25.2 against 24.7 dB (+0.5); error powers add (D=64 attention backward model; KB note of a manual pricing session, not a campaign; local:/workspace/code/gpt_oss_docker/.claude/memory/project_gptoss_fp8_attn_bwd_onclock_campaign.md:111)
- E20 [2026-07, gfx950, torch (no FlyDSL kernel), N=1, PyTorch SNR model of fp8 P, kernel unchanged] bf16 P 57.47 dB, e4m3 with per-row and with per-tile scale 33.41 dB, e5m2 with per-row scale 27.40 dB (6.0 dB below e4m3); scaled to the kernel's measured 51.61 dB, e4m3 P ≈ 27.6 dB, which the round sets against a 48 dB gate (same hd64 forward; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260728_082734/rounds/round-19/agent.txt:8)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-and-run probe, ISA dump, output compared bitwise with torch's e4m3 → bf16] 16 fp8 bytes per lane: `cvt_pk_f32_fp8` (lo, hi) then `.to(BFloat16)` emit 8 `v_cvt_pk_f32_fp8` + 8 `v_cvt_pk_bf16_f32`; `cvt_scalef32_pk_bf16_fp8` with scale 1.0 emits 8 `v_cvt_scalef32_pk_bf16_fp8`; both bit-equal to torch on all 254 non-NaN e4m3 codes (±0 included) (local:/shared_nfs/kyle/aklr_mla/runs/L3/knowledge_patch_v2/probe/cvt.py); instruction semantics in `../hardware/gfx950/cdna4/ch12_11_vop3a_vop3b_instructions.md` § V_CVT_SCALEF32_PK_BF16_FP8
