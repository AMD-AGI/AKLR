---
title: "Attention — exp2, scale and shift, row reductions and the online-softmax rescale"
kind: topic
lever: attn_softmax
ops: [attention, mla]
gens: [gfx950]
dtypes: [bf16, fp8]
regimes: [many query rows, few query rows, dQ dK dV, causal]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:432
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3317
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3710
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3766
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3840
---

# Softmax and exp2 in an attention kernel

> Everything between the QK MFMAs and the P·V MFMAs: the exp, the scale and shift, the row max and row sum,
> and the online rescale. Guards for `-inf − -inf` and empty partitions are in
> `../flydsl/debug-flydsl-kernel.md` §1; carrying the running max and sum through a FlyDSL loop is in
> `../flydsl/guides.md` § prefetch-data-load; how many exps fit under the MFMAs is in `mfma_layout.md` §1
> and `registers_occupancy.md` §8.
>
> Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.
> Use them for direction and order of magnitude, not as targets.

## Route here when
- `v_exp_f32` is the largest VALU class, or `v_ldexp_f32` / an `__ocml_exp2_f32` call sits next to it.
- You are deciding where the softmax scale, the max shift or a clamp is applied.
- A fixed reference max, a skipped or lazy rescale, or a cheaper exp is proposed.
- A gradient output (dQ, dK or dV) is all NaN, or an accuracy gate fails for one exp mode only.

## 1. What one exp costs on gfx950
- *(field, E1)* On a HIP inline-asm microbenchmark, `v_exp_f32` (also `v_log_f32`, `v_rcp_f32`, `v_sqrt_f32`) takes
  8 cycles per wave64 instruction; `v_mul_f32`, `v_add_f32` and `v_fmac_f32` take 4 from one wave and reach 2 per
  SIMD once two or more waves issue them. The exp stays at 8, so at SIMD level one exp costs as much VALU time as
  four multiplies.
- *(field, E1)* A transcendental never runs beside other VALU work: on the same microbenchmark exp + mul in one wave
  take 12 cycles, and
  while one wave streams exps no other wave on that SIMD issues VALU. It draws on the same MFMA shadow as every
  other VALU op, twice as fast as a multiply.
- It is a trans op (CDNA4 ISA Table 12): a non-trans VALU consumer of its result needs one wait state. The
  compiler pads that for an inline-asm `v_exp_f32` too (AMD clang 23 puts a generic `s_nop 0` between an inline-asm
  result and its reader), but no more: an MFMA reading an inline-asm result, or inline asm reading an MFMA result, is
  left short of the wait states it needs (`schedule_sync.md` §7).
- To place exps between MFMAs, use the transcendental class `0x400` in `sched_group_barrier`; the VALU class
  `0x002` does not match `v_exp_f32` (`schedule_sync.md`).
- The tile fixes the count: one `v_exp_f32` per 64 scores. Only fewer scores (causal tile skipping, tile
  geometry) lowers it.
- *(field, E2)* Deleting every softmax exp of a D=64 forward loop (wrong results, diagnostic only) bought ~8%,
  deleting half of them about half of that. *(field, E4, E5)* In backward bodies hardware exp cost
  4–6% against a no-exp body and a Schraudolph exp ~3%. Price your own with a subtractive arm
  that keeps the consumers alive (`attribution.md` §5).

## 2. exp2: one `v_exp_f32` is enough
- `__ocml_exp2_f32` and `llvm.exp2.f32` keep f32 denormal results while the function's f32 denormal mode is IEEE
  (LLVM's default): about 8 VALU (a compare with −126, a +64 bias, `v_exp_f32`, `v_ldexp_f32`). The AMDGPU backend
  chooses that sequence by the f32 denormal mode, not by fast-math flags: `afn` or `fast` on `llvm.exp2.f32` leaves
  it in place (LLVM `AMDGPUTargetLowering::lowerFEXP2`, `needsDenormHandlingF32`). One `v_exp_f32` comes from the
  `llvm.amdgcn.exp2.f32` intrinsic, or from `llvm.exp2.f32` in a function whose f32 input denormals are flushed
  (`preserve-sign`). Which of these a language's exp2 reaches depends on the language: read it from the ISA
  (FlyDSL rewrites an `afn` exp2 to the intrinsic itself: `../flydsl/attention.md` § softmax §2).
- The range reduction of `__ocml_exp2_f32` only exists to return denormal results; `v_exp_f32` flushes denormals to zero (ISA).
  After the max shift every argument is ≤ 0 and every result lies in (0, 1], where a denormal is worthless
  next to bf16 P, so the single instruction is enough. *(field, E5, E7)* Replacing the range-reduced exp with one
  `v_exp_f32` bought +20% and +25% in two backward kernels.
- `v_exp_f32(-inf) = 0` exactly (ISA): write `-inf` into masked scores and no select is needed after the exp.
- Prefer the `llvm.amdgcn.exp2.f32` intrinsic to inline-asm `v_exp_f32`: it gives the same instruction and stays
  visible to the scheduler and to the hazard recognizer. *(field, E43)* In an hd64 attention backward it took 0.79%
  off the dq kernel but added 1.06% to dkdv (more `s_nop` around its exps) until the dP MFMAs were issued ahead of
  the exps, or one q-tile at a time between them: then 0.3–0.5% less than inline asm. Measure it per kernel.
- **Reference pattern** — FlyDSL's gfx950 forward: Q is multiplied once by `rsqrt(D)·log2e` in f32 and truncated
  back to bf16 (`scale_all`), so scores and the running max are in log2 units; every softmax exp is
  the exp2 intrinsic (the D=128 causal ISA has 197 `v_exp_f32`, no `v_ldexp_f32`, no OCML call); the LSE is converted
  back to natural log at the store.

In FlyDSL: `../flydsl/attention.md` § softmax §2.

## 3. Scale, shift, clamp and pack at zero instruction cost
- **Shift through the MFMA C operand.** MFMA computes A·B + C, so seeding C with the negative shift (`-ref` in
  the forward, `-log2e·lse` in the backward) makes the subtraction free. Build the seed inside the QK step from
  a scalar; a seed vector materialized outside the loop stays resident (16 VGPRs for a 32x32 C fragment). The
  measured price of carrying it is in `registers_occupancy.md` §3; fold it in the masked loop too
  (`gradients.md` §8).
- **Choose where the scale goes.** Pre-scaling Q costs nothing but rounds `q·sm·log2e` to bf16; the backward must
  then reproduce that rounded operand (`correctness.md` §5). Scaling the accumulator leaves no residual
  (it removed a dQ correction pass that cost up to 8.5× when triggered, for an 8–13% slower forward, 4.1 points
  of it at D=64 from holding the pre-divided seed) *(field, E8)*, but every value seeded into it must be pre-divided
  by the scale; one missed seed site applies the scale twice, and a `-3e38` floor divided by a scale < 1 overflows
  fp32. If you must scale the accumulator, do the
  shift inside the scaling FMA (`fma(acc, c, -ref)`) and seed zero. Gates: `correctness.md`.
- **A seed must not cost the accumulator its precision.** The Schraudolph-prescaled LSE (~1e9) cannot be seeded: f32
  spacing at 1e9 is 64, the trick needs the low mantissa bits exact.
  Keep that FMA as the last per-element step.
- **Clamp in the exp's modifier.** `min(max(exp2(x), 0), 1)` (max first) folds into the VOP3 clamp bit:
  one `v_exp_f32_e64 … clamp`, no extra VALU. `max(min(e, 1), 0)` does not fold (`min(NaN, 1) = 1`, the clamp
  gives 0) and costs two VALU. A clamp turns an overflow into silent mass loss: use it as a backstop on a path
  whose arguments are already provably ≤ 0, never as the fix (§6.1).
- **Pack natively.** An f32→bf16 `fptrunc` of a 2-element vector lowers to one `v_cvt_pk_bf16_f32`. The same
  instruction written as inline asm is invisible to the hazard recognizer, so an MFMA that reads that P directly
  needs explicit padding (`schedule_sync.md` §7).

In FlyDSL: `../flydsl/attention.md` § softmax §3.

## 4. Precision tiers and the LSE contract
| exp | per element | accuracy *(field, E5, E10, E11)* |
|---|---|---|
| Schraudolph bit trick (FMA into the exponent field, convert to int, reinterpret) | one convert; the FMA exists anyway | ~35 dB: a linear mantissa |
| `v_exp_f32` | one 8-cycle trans op | ~50.5–52.6 dB, the ceiling set by packing P/dS to bf16 |
| polynomial `2^n·P(f)` | round, subtract, convert, shift-add, 2–3 FMAs | degree 3 equal to hardware, degree 2 ~1 dB lower; not higher in that test |

- An accurate software exp2 needs six or more VALU instructions against one `v_exp_f32` worth two plain VALU
  ops from one wave, four with ≥ 2 waves (the §1 microbenchmark) *(field, E1)*. It cannot win on gfx950.
  *(field, E12)* Degree 2: −13%; degree 3: −17%. `v_exp_f16` is a trans op too and the ISA has no packed transcendental; the f32↔f16 round
  trip only adds VALU (−39%) *(field, E13)*.
- Pick the exp by the accuracy gate. The bit trick is ~3% faster than hardware exp *(field, E5)* and passes
  only gates below ~35 dB *(field, E10)*.
- **The LSE prescale belongs to the exp mode.** Hardware exp takes `-log2e·lse`; the bit trick takes
  `-log2e·lse·2^23 + (127·2^23 − 486411)`. Fed the other mode's LSE, exp2 overflows and the output is all NaN.
  Derive both from one function keyed by the mode. When one output of a kernel is all NaN and the others are
  fine, check that output's input preprocessing first.
- **R ≡ 1.** With the forward's true LSE, every row of `exp2(s − lse₂)` sums to 1. A backward that carries a row
  sum and renormalizes by it does dead work. *(field, E15)* Removing it from a dq kernel with the Schraudolph exp:
  about +4%, SNR unchanged within 0.01 dB (the Schraudolph errors cancel along the row to bf16 precision).
- **The backward's exp arguments are ≤ 0 only because lse ≥ the row max.** A forward shortcut that loses mass
  (a clamp, a dropped tile) puts lse below the max, and the backward's `exp2(s − lse)` overflows. *(field, E17)* A
  forward with clean outputs produced ~8·10⁵ non-finite gradients. Gate forward and backward together, and make
  the backward reproduce the forward's scale rounding (`correctness.md`).

## 5. Row max and row sum
- Orientation sets the cross-lane cost. With S computed as Sᵀ a lane holds the scores of one q row, and one
  `permlane32_swap` finishes each reduction (`mfma_layout.md` §4).
- Max: LLVM folds a linear max chain over a lane's scores into `v_max3_f32` (16 for 32 values). A ternary tree of
  `max(max(a, b), c)` groups keeps one `v_max3_f32` per triple and cuts the depth from 16 to 4 *(field, E18)*. gfx950
  has no `v_pk_max_f32` or `v_pk_min_f32`; packed f32 exists for add, mul, fma and mov only.
- Sum: dependency depth did not matter (four accumulators, depth 32 → 8, 0.00%) *(field, E20)*; packing it into
  `v_pk_add_f32` removed 40 instructions at equal VGPRs and lost 1.45% *(field, E21)*. The matrix-pipe row sum
  (`mfma_layout.md` §6) removes the adds and the cross-lane step.
- Sum the same bf16-rounded P that feeds P·V, so numerator and denominator share one rounding
  (51.26 → 51.61 dB) *(field, E22)*.

## 6. Online softmax
### 6.1 A fixed reference max is not a general lever
- With a constant reference r, P = 2^(s − r). Scores more than 128 log2 above r overflow; a clamp P ≤ 1 stops
  that but silently drops the mass above r; scores ~126 log2 below r flush to 0, and a row entirely below gives
  l = 0 and 0/0. A margin only splits those ~126 binades between headroom and tail: any fixed r is exact
  inside a data-dependent window and silently wrong outside it.
- *(field, E23, E24)* Reference = first-block row max + 96 log2: exact only inside that window and silently
  degraded beyond it, where true online softmax kept 44.8 dB on a row whose max sat 158.7 log2 above the first
  block. Training diverged after some steps. *(field, E25)* iid Gaussian inputs only reached the window's edge
  (gap ≈ 97 log2 at a max logit of 140): trained attention is concentrated, random data is not.
- Gate every softmax shortcut with spike inputs (a few keys with large logits far from the first block),
  raise the spike until the gate fails, and check norm ratios of O, LSE, dq, dk, dv, not `isfinite` counts.
- A max over a fully masked block (sliding-window start, first causal pair) is `-inf`, and
  `exp2(s − (−inf)) = +inf` → NaN once a real score meets it. Floor it before use: a row max applied only to its
  own masked scores can sit at `-3e38` (FlyDSL's forward, `floor_masked_max`); a bound reused as the reference
  for later tiles needs a floor those scores cannot overflow against.
- With P packed to fp8 e4m3 (max 448) for the P·V MFMA, a fixed reference cannot replace the rescale: without a
  clamp P overflows the format, with one it loses mass.
- The exact path can be made nearly as fast. *(field, E26)* A tuned online forward reached ~99–100% of the
  fixed-reference speed (D=128 and D=64) with the rules below.

### 6.2 What the rescale must touch
- Per tile: `m_new = max(m_old, m_tile)` before the tile's exps, `corr = exp2(m_old − m_new)`, then O and l
  are multiplied by `corr`. The running max must be loop-carried. *(field, E27)* Fingerprint of a reset max: correct
  at short S, the row-sum ratio falling with S (0.80 at 1024, 0.53 at 2048).
- Rescale O and l, not P that is already packed. When tile j's packed P is still in flight as tile j+1 moves the
  max, let the P·V steps consume it on the old reference, then scale O and l once: every term and the sum share
  that reference. *(field, E28)* Dropping the unpack-multiply-repack of the carried P removed 5 pp of a 21% online
  overhead.
- The cost centre is the per-tile shift and multiply of the whole O accumulator, not the max. *(field, E29)* D=128
  forward, ~16% online overhead: by ablation it divides into the shift (197 subtracts) and the O rescale (138 packed
  multiplies); the max reduction is about 2.8 pp.
- Do not seed the shift with the previous iteration's max to save the subtractions: P loses its ≤ 1 bound
  (it is bounded only by `2^(m_new − m_old)`) and a narrower window comes back (peak 6·10¹⁵ at one input scale,
  inf at the next) *(field, E30)*.

### 6.3 Skipping rescales
- A skip branch must be wave-uniform, and one wave holds 32 rows.
- A runtime branch in a hand-pipelined loop costs by itself (~1.3 pp per branch point) *(field, E32)*. Count branch points;
  the instructions moved into a rarely taken branch are nearly free, and static counts misprice both.
- Lazy rescale — keep the old reference while the max moved less than a threshold — pays only where its branch
  breaks no pipeline. FlyDSL's forward does it per wave: skip when every lane has `m_tile − m_row ≤ 8.0` (log2),
  tested as `ballot == exec`, so P may reach 2⁸. *(field, E33)* Ported into a sparse-MLA forward, the same branch
  was 31% slower on one path (it broke a store pipeline). On dense fp8 MLA decodes
  (D_v 512, bf16 P, the rescale in a branch on a wave-uniform `ballot`) it paid in all four kernels that tried it:
  2.7% at threshold 8 *(field, E37)*, 4–6% at threshold 0 *(field, E39)*, and at threshold 8 in single runs 8%
  *(field, E40)* and 7–13% *(field, E42)*.
- The threshold need only keep P, O and l finite: with bf16 P, threshold + log2(keys × max|V|) < 128; fp8 P caps it
  near 8. On those decodes a larger threshold paid again *(field, E39)*: 0 → 8 1–3%, 8 → 64 4%; *(field, E41)* 8 → 80
  log2, re-based 48 above the new max: 2.6–6.2% less time (logits spread over tens of log2).
- Check that a skip stayed a branch: the backend if-converts a short body (2–3 scaled vectors in a probe) into
  `v_pk_mul_f32` plus `v_cndmask` on every trip; an empty side-effecting inline asm in the body keeps the
  branch *(field, E38)*.

In FlyDSL: `../flydsl/attention.md` § softmax §6.3.

### 6.4 Rare rebase without a window
- Keep the reference loop-carried and seeded (§3). Per tile, decide whether any score can exceed it; only then
  compute the exact tile max, raise the reference, multiply O and l by `2^−Δ` and recompute the tile.
- Decide before exp2. A test on the row sum is one step late: the sum is built from exp2 outputs, so it sees an
  overflow after it happened, at any check frequency (the failure point did not move whether checked once per
  8 tiles or 4× as often) *(field, E34)*. A clamp under such a test only makes the error silent.
- A test that cannot miss: `‖q_i‖·max_j‖k_j‖` (q already scaled by `sm_scale·log2e`) bounds every score
  of row i (Cauchy–Schwarz). With the key norms from a pass over K, one comparison per row decides whether a rebase
  is reachable: if not, the max is skipped; if so, every tile's max is reduced exactly; the reference still updates
  online. *(field, E35)* NaN-free to spike 96 and bit-identical to the kernel without the skip; arming it cost
  2.0 pp at D=128, so that kernel arms the test at D=64 only.

### 6.5 fp8 P with a fresh accumulator per tile: fold the rescale into the P·V MFMA
- If each tile's P·V goes into a fresh accumulator merged as `O·alpha + tile·beta`, exponentiate the tile
  against `max(m_running, m_tile)` (beta then sits inside P and its sum) and pass `O·alpha` as the P·V MFMA's C.
  *(field, E36)* On a sparse-MLA decode producer: 16 `v_pk_mul_f32` + 16 `v_pk_add_f32` fewer per tile and wave
  (−10% instructions), producer 1.2–3.8% faster, and lower error (relL2 0.0262 → 0.0249): beta had been applied
  after fp8 quantization, scaling the quantization error up.

## Verify
- ISA: one `v_exp_f32` per score per lane, no `v_ldexp_f32` or `__ocml` in the hot loop, clamps as `clamp`
  modifiers rather than `v_min`/`v_max`.
- A spike-input gate raised until it fails, with norm ratios on every output; forward and backward gated
  together.
- The LSE prescale and the exp mode come from one place.
- Every value seeded into a score accumulator is accounted for: all seed sites, the scaling, the floor.

## Related
`mfma_layout.md` · `registers_occupancy.md` · `schedule_sync.md` ·
`correctness.md` · `gradients.md` · `split_kv.md` ·
`attribution.md` · `../flydsl/debug-flydsl-kernel.md` · `../flydsl/guides.md` § prefetch-data-load

## Evidence

A FlyDSL version is given only where the cited run, or another round in the same container and venv, printed it
(`flydsl.__version__`, an import path with its version, or a `pip install flydsl==` in the run's own setup); otherwise
`FlyDSL ?`. "Primus-Turbo" commits are those of the
Primus-Turbo repository; their
messages carry the numbers.

- E1 [2026-10, gfx950, hipcc inline asm (no FlyDSL kernel), N=7, s_memtime slope method, median of 7 launches] `v_exp_f32` (and log, rcp, sqrt) 8.000 cycles per wave64 instruction, `v_mul_f32`/`v_add_f32`/`v_fmac_f32` 4.000 from one wave; at 2 and 4 waves/SIMD exp stays at 8.0 per SIMD while mul reaches 2.0; exp + mul in one wave 12.0 cycles per pair (microbenchmark)
- E2 [2026-08, gfx950, FlyDSL ?, N=2–4, palindromic arms, one process per arm, JIT and debug caches cleared per arm, ISA gate on vgpr/agpr/spill/LDS] softmax `v_exp_f32` pool priced by subtraction: 64 exps (deployed) 1.8363 / 1.8502 ms; 32 exps 1.7704 / 1.7685 ms (+4.3%); 0 exps 1.7041 / 1.6953 / 1.7109 / 1.7107 ms (+7.8%, 1293.8 TF/s), vgpr/agpr/spill/LDS bit-identical (gpt-oss D=64 attention forward, scored cell `d64_gptoss_b4`)
- E4 [2026-08, gfx950, FlyDSL ?, N=3, palindrome, three rounds per arm] exp2 replaced by an identity: 5.5701 against 5.8051 ms (−4.05%) (gpt-oss D=64 full-causal fused attention backward, B=4, Sq=Skv=8192, Hq64/Hkv8)
- E5 [2026-07, gfx950, FlyDSL ?, N=?, conv TF/s] dkdv: the range-reduced `ArithValue.exp2()` 974 → one bare `v_exp_f32` 1165 TF/s (+20%), dk/dv SNR 52.6 dB, deterministic; Schraudolph exp 1202; exp replaced by an identity 1237 (hd64 deterministic attention backward, dkdv, 16x16x32)
- E7 [2026-07, gfx950, FlyDSL ?, N=?, conv TF/s] dq: `ArithValue.exp2()` (v_exp + v_ldexp) 897 → one bare `v_exp_f32` 1118 TF/s (+25%), MfmaUtil 47 → 52.6, SNR 52.5 dB (hd64 deterministic attention backward, dq)
- E8 [2026-09, gfx950, FlyDSL ?, N=?, ?] softmax scale moved from Q to the score accumulator: backward at a spike that fires the dQ untwist gate 3.55683 → 0.40137 ms at D=128 and 12.04823 → 1.41074 ms at D=64 (8.52× and 8.53× → 1.00× and 0.99×); forward 0.5238 → 0.5935 ms at D=64 (+13.3%) and 0.1310 → 0.1415 ms at D=128 (+8.0%); of the D=64 +13.3%, +8.9% is the fold and +4.1% the rebase seed compensation, priced by an uncompensated build (0.5705 ms on d64_b4_s4096) (FlyDSL attention forward and backward, Primus-Turbo; Primus-Turbo 438692bc, 4099eaea)
- E10 [2026-07, gfx950, FlyDSL ?, N=?, SNR against a reference] Schraudolph exp2: dq/dk/dv SNR 34.8 / 34.9 / 34.9 dB; hardware exp 52.5 / 50.6 / 50.5, described as the bf16-pack ceiling; both deterministic (hd64 deterministic attention backward)
- E11 [2026-07, gfx950, FlyDSL ?, N=?, dk/dv SNR against a reference, all arms deterministic] hardware exp 52.6 / 52.6 dB; degree-3 polynomial 52.6 / 52.6; degree-2 polynomial 51.6 / 51.5 (≈ −1 dB) (hd64 deterministic attention backward, dkdv, B4 Hq128 Hkv16 D64 Skv16384)
- E12 [2026-07, gfx950, FlyDSL ?, N=?, campaign bench conv TF/s] hardware exp 1113–1115 TF/s, degree-3 polynomial 925.5 (−16.7%), degree-2 polynomial 965.1 (−13.3%) (same dkdv)
- E13 [2026-07, gfx950, FlyDSL ?, N=?, ?] `v_exp_f16` with the f32↔f16 round trip against `v_exp_f32`: 1136 → 689 TF/s (−39%), SNR 49.7 dB, deterministic (hd64 deterministic attention backward, dkdv; the kernel's comment on its gated-off probe)
- E15 [2026-07, gfx950, FlyDSL ?, N=?, conv TF/s per shape; SNR over 4 seeds/shapes] Schraudolph dq without the row sum and renorm: 1202 → 1245 TF/s at 8192, 1179 → 1229 at 16384 (+4%), SNR Δ < 0.01 dB (34.79 → 34.79), deterministic (hd64 deterministic attention backward, dq)
- E17 [2026-09, gfx950, FlyDSL ?, N=?, ?] with the clamped fixed-reference forward entirely finite, its backward returned 815693 non-finite gradients at D=128 and 491459 at D=64 at an input scale of 8, 9.6e6 at 16 (FlyDSL attention forward and backward, B1 S4096 Hq32 Hkv8; Primus-Turbo d1f276a9)
- E18 [2026-09, gfx950, FlyDSL ?, N=?, ISA of the build] online forward: the per-lane max fold grouped three at a time "still emits one three-operand max per triple at a quarter of the chain's dependency depth" (FlyDSL attention forward, online softmax; Primus-Turbo c4a0db7f)
- E20 [2026-07, gfx950, FlyDSL 0.2.2, N=1, campaign scorer conv TF/s, one read per arm] row-sum fold with 4 independent accumulators (depth 32 → 8): 1128.1 against 1127.8 TF/s (Primus-Turbo hd64 attention forward, B=1, Sq=Skv=16384, full causal)
- E21 [2026-07, gfx950, FlyDSL ?, N=?, ?] row sum packed into `v_pk_add_f32`: 46 `v_pk_add_f32`, `v_add_f32` 86 → 0 (40 fewer instructions), vgpr 166 → 166, SNR 51.26 dB deterministic, wall −1.45% (same hd64 forward; unverified)
- E22 [2026-07, gfx950, FlyDSL 0.2.2, N=3, interleaved A/B in one session, three reads per arm, scorer] online-softmax row sum as `v_mfma_f32_16x16x32_bf16` with an all-ones A over the packed P: 1159.4 / 1159.1 / 1159.9 against 1137.0 / 1137.0 / 1137.2 TF/s (+1.96%); SNR 51.26 → 51.61 dB (same hd64 forward)
- E23 [2026-09, gfx950, FlyDSL ?, N=?, ?] fixed reference (prologue row max + REF_MARGIN 96, P clamped): the clamp narrows the bit-exact window from a gap of 206 to 96 log2, beyond which "the answer degrades instead of failing"; past it the path "silently drops the mass above the reference", the window that made D=128 training diverge after a few hundred steps (FlyDSL attention forward and backward; Primus-Turbo d1f276a9, c2209644)
- E24 [2026-09, gfx950, FlyDSL ?, N=?, against an fp32 reference] online path at a gap of 158.7 binades: 44.8 dB and gradient ratio 1.001 (FlyDSL attention forward and backward, D=128)
- E25 [2026-09, gfx950, FlyDSL ?, N=?, iid Gaussian inputs (`torch.randn` × alpha) swept in logit scale] row max − prologue tile max: the commit gives the measured spread as "near 97 even at a max logit of 140" (FlyDSL attention forward; Primus-Turbo d1f276a9)
- E26 [2026-09, gfx950, FlyDSL ?, N=2, campaign score: geometric mean over llama-8b D=128 and gpt-oss D=64 of reference_ms / candidate_ms against a frozen pre-fix fixed-max tree, reference and candidate timed adjacently in one process, two repeats] online forward 0.9320 → 0.9961 (0.996244 / 0.995958; per cell D=128 1.0003 / 0.9989, D=64 0.9922 / 0.9930) (FlyDSL attention forward, online softmax; Primus-Turbo e3e3008c)
- E27 [2026-09, gfx950, FlyDSL ?, N=?, row sum against an fp32 reference] running max reset every iteration: row sum unchanged at 256 keys, 0.803 of the true value at 1024, 0.531 at 2048 (FlyDSL attention forward, D=128)
- E28 [2026-09, gfx950, FlyDSL ?, N=2, same process, two independent runs, against the kernel's own fixed-max path] dropping the rescale of the carried P: llama-8b D=128 +21.0 / +20.5% → +15.8 / +16.0%, llama-70b +21.1 / +21.6% → +16.0 / +16.5%, "worth 5.0 and 5.1 points" (FlyDSL attention forward)
- E29 [2026-09, gfx950, FlyDSL ?, N=2, same process, two independent runs, against the kernel's own fixed-max path; ablation] online overhead +15.8 / +16.0% (llama-8b D=128); by ablation the rest divides into 197 subtracts for the shift and 138 packed multiplies for the O rescale, the row-max reduction itself about 2.8 points (FlyDSL attention online forward)
- E30 [2026-09, gfx950, FlyDSL ?, N=?, ?] shift seeded with the previous iteration's max: P unbounded between the fold and the rescale, 6.2e15 at one input scale and inf at the next (FlyDSL attention forward, online softmax)
- E32 [2026-09, gfx950, FlyDSL ?, N=?, ?] ablation of the rare-rebase residual: detection (a max tree over 32 scores) ≈ 1.7 pp, `scf.if` rebase branch ≈ 2.6 pp (about 1.3 pp per site, two per main-loop iteration), loop-carried reference + QK seed ≈ 1.2 pp (FlyDSL attention online forward)
- E33 [2026-07, gfx950, FlyDSL ?, N=?, ?] lazy O rescale ported on its own into the sparse-MLA forward: cr4 −31%, the branch conflicting with the pstore pipeline (DeepSeek-V4 sparse-MLA attention forward)
- E34 [2026-09, gfx950, FlyDSL ?, N=?, non-finite outputs at a row far above the prologue tile] rebase detected from the row sum, one check per 8 tiles (`KV_UNROLL=4`): `KV_UNROLL` 4 → 1 "moved the failure point by zero" (FlyDSL attention forward and backward; Primus-Turbo e3e3008c)
- E35 [2026-09, gfx950, FlyDSL ?, N=?, NaN gate at spike 1/8/24/36/48/96 × D=128/D=64 on O, LSE, dq, dk, dv; gradient norms against an fp32 reference] Cauchy–Schwarz skip, the key norms from a pre-pass over K and one comparison for the whole KV walk: no non-finite output up to spike 96; outputs bit-identical to the tree without the skip (D=128 dv 0.98890 / 0.97488 / 0.95148 at spike 24 / 48 / 96 on both); armed at D=128 as well it cost 2.0 pp of E26's campaign score, so it is armed at D=64 only (FlyDSL attention forward, online softmax; Primus-Turbo e3e3008c)
- E36 [2026-09, gfx950, FlyDSL 0.3.2, N=2, same-process ABBA, two independent repeats; relL2 over 6 scorer runs] rescale folded into the P·V MFMA's C: 16 `v_pk_mul_f32` + 16 `v_pk_add_f32` fewer per tile and wave (128 of 1284 instructions); producer −1.30 / −1.65% (seq48), −1.22 / −1.33% (seq60), −3.78 / −3.39% (seq84), −1.19 / −1.58% (seq14); relL2 0.026227 → 0.024898 (GLM-5.2 TP4/EP4 sparse-MLA decode producer, fp8 P)
- E37 [2026-10, gfx950, FlyDSL 0.3.2, N=?, AKLR graph timer, paired timing the round was kept on (one-sided 95% bounds), campaign noise floor 0.09%] wave-uniform lazy rescale (`ballot(m_blk > m_run + 8)`, rescale of the 32 accumulator vectors and the running sum inside the branch) as the only change of its commit: +2.70% kept, +2.71% re-read (+2.67 … +2.75%), faster on 11 of 11 workloads (+0.04% at batch 8 to +4.6% at batch 112); threshold 0 (skip only when no max grows) measured 0.960–1.074× per workload in the round's own probe (dense fp8 MLA decode, 96 heads, query length 1, page size 1, batch 8–112)
- E38 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-and-run probe, ISA dump] a runtime loop whose `if ballot(alpha != 1) != 0` body multiplies k f32x4 accumulators by alpha: k = 2 / 3 gave 40 / 60 `v_cndmask_b32` beside the `v_pk_mul_f32` with the branch still in `20_llvm_ir.ll`; with `_llvm.inline_asm(None, [], "", "", has_side_effects=True)` first in the body, 0 `v_cndmask` and the branch kept; k = 4, 6, 7, 24 kept the branch either way; outputs equal to torch; on a dense fp8 MLA decode the same pass had turned a per-head rescale branch into 36 `v_pk_mul` and 72 `v_cndmask` per step
- E39 [2026-10, gfx950, FlyDSL 0.3.2, N=? for the first two (local Kineto A/B), N=1 for the third (the driver's total over 11 batch sizes)] wave-uniform skip (`ballot` of `m_tile > m_ref + τ`, the rescale in a runtime loop of 0 or 1 trips) against none: speed 1.042 / 1.045 / 1.061 at batch 8 / 32 / 112, bitwise identical; τ 0 → 8: 1.032 / 1.014 / 1.021 (max diff 0.023, contract passed); τ 8 → 64: driver total 11.618 → 11.149 ms (−4.0%), contract passed on 11 batch sizes (dense MLA decode, 96 heads, one query, fp8 KV, up to 112431 keys per row, fp32 accumulators; MI355X)
- E40 [2026-10, gfx950, FlyDSL 0.3.2, N=1, AKLR graph timer via `driver.py --bench-mode`, warmup 3, 20 iterations, sum over 11 batch sizes, one run per arm] lazy rescale at threshold 8 log2 (`ballot` of `m_tile > m_ref + 8` over the wave, then a 0/1-trip runtime `range` around the O and l multiply) against rescaling every step: 8.542 → 7.846 ms (8.1% less time), the lazy build at 256 VGPRs and 0 spills; no NaN with logits 8× and 30× larger (dense MLA decode over paged fp8 KV, 96 heads, D 576/512, 32-key steps, 6 waves of 16 heads, batch 8–112, cache_seqlens 15–110k)
- E41 [2026-10, gfx950, FlyDSL 0.3.2, N=?, forge-loop ablation ladder: the round against the version before it, the paired AKLR graph-timer timing it was kept on] threshold 8 → 80 log2 with the reference re-based to the new max + 48 on a rescale: gain +4.15% (time without it over time with it; one-sided 95% bounds +3.71…+4.60%), faster on 11 of 11 batch sizes, 2.6–6.2% less time each; max_diff 5.09e-7; inputs had sm_scale 1, logit std 34.6 and max 179 log2 at batch 8, where a simulation put the share of steps that rescale at 0.43 (36-step segments) and 0.085 (420-step) for threshold 8, 0.029 and 0.010 for 64 (same kernel as E40)
- E42 [2026-10, gfx950, FlyDSL 0.3.2, N=1, local harness (random lengths, n_pages 760149, width 112431), one run per arm, arms not alternated] wave-ballot lazy rescale (rescale and its 64 `v_pk_fma_f32` in an `scf_if_dispatch` branch) against the same kernel without it, b8/b16/b32/b112: 0.2806/0.4640/0.6997/1.9095 ms; threshold 0: 0.2706/0.4372/0.6989/1.7240; threshold 8 (log2): 0.2604/0.4157/0.6345/1.6642 (−7.2/−10.4/−9.3/−12.8%); the kept round (with bare exp and direct fp8→bf16 convert) +16.58% on the campaign's 3-pair ablation, 11/11 cases faster (96-head MLA decode, query length 1, fp8 KV, page size 1, batch 8–112)
- E43 [2026-07, gfx950, FlyDSL ?, N=2, each arm in its own process, alternated over two rounds, host-timed median of 9 blocks of 40 back-to-back calls] the softmax exp as inline-asm `v_exp_f32` behind the kernel's compiler-visible `v_min` anchors, against the `llvm.amdgcn.exp2.f32` intrinsic with the anchors removed: dq 0.79% less time, dkdv 1.06% more (the dkdv main loop's estimated nop cycles 205 → 439); dkdv all-intrinsic with the dP MFMAs issued before the exps 0.42% and 0.46% less time than the inline-asm base, issued one q-tile at a time between the exps 0.32% less (hd64 deterministic attention backward, B=1, Sq=Skv=16384)
