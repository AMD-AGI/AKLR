---
title: "Triton/Gluon numerics on gfx950: the contract, fp8 and MXFP4 quantization, block-scaled accumulation, transcendentals, determinism"
kind: topic
ops: [gemm, moe, attention, gdn, norm]
dtypes: [fp8, mxfp4, bf16, fp32]
regimes: [all]
gens: [gfx950]
triton: 3.8.0
---

# Numerics of a Triton or Gluon kernel for AKLR on gfx950

What an AKLR kernel must compute bit for bit, what it may approximate, and how the committed Triton/Gluon kernels do
both on gfx950 (MI355X) with Triton 3.8.0. Kernel examples refer to embedded AKLR sources;
“compiled” describes offline gfx950 compilation, not GPU performance. The original contract,
Triton lowering and ISA references are listed in Evidence.

## Route here when

- A kernel's output, or a GEMM input it produces, is fp8 E4M3FN with fp32 scales per 128, or MX (e2m1 or E4M3FN values,
  an e8m0 scale per 32), in an epilogue or as AKLR's quantization ops (§2, §3).
- You must decide which divisions, exponentials and roundings have to be exact and which may be fast.
- You need fp32 accuracy from bf16 or fp8 MFMAs: block-scaled GEMMs, router logits, P·V, GDN state products.
- A kernel fails `compare` (a NaN, a code off by one, an SQNR shortfall), the `large_magnitude` step or the `bitwise`
  determinism step: §10 maps what the report says to a cause.
- You weigh `enable_fp_fusion`, `allow_flush_denorm`, `amdgpu-ieee`, atomics, asm `v_rcp_f32`/`v_exp_f32`/`v_log_f32`.

Elsewhere: MFMA and scale layouts [gluon/mfma.md](gluon/mfma.md); inline asm [gluon/wave_ops.md](gluon/wave_ops.md);
compiled code [debugging.md](debugging.md) §3; ops [gemm.md](gemm.md), [moe.md](moe.md), [attention.md](attention.md),
[gdn.md](gdn.md); correctness gates [../gemm/correctness.md](../gemm/correctness.md).

## 1. Read the contract first

Three places define "correct". The rules are stated in `aklr/ops/README.md` § Schema fields; the bound formulas are the
`ErrorBound` classes of `aklr/core/contract/semantics/_compare.py`.

| Where | What it fixes |
|---|---|
| the schema's `compare` | `determinism`: how runs must agree, `bitwise` or `bounded`; `max_violation_fraction`: the share of elements allowed over their bound, 0 in all 154 schemas here |
| each output's `error_bounds` in the op's `contract.py` | the error budget: `tolerance`, `sqnr`, `dot_product`, `output_rounding` or `baseline_relative`; an integer output (Qwen2-MoE `topk_ids`, the e8m0 bytes of `per_group_quant_mxfp8`) has none and must equal the reference |
| the reference's docstring | rounding points to inspect; actual output comparisons decide admission, not bitwise copying of every rounding (§1) |

- **Both determinism levels compare runs, not kernels.** `bitwise` wants every run byte-identical to the first;
  `bounded` asks nothing across runs (S1). Neither compares the output with the reference, the baseline or the kernel
  you replace: accuracy is the bound's job. For a `bitwise` schema, bench runs input set `numeric/0` once more, at step
  `determinism/0` (S2), a finite repeated check: a rare race can pass (§8).
- **What each bound leaves you** (S1):
  - `tolerance` is per element, B = atol + (rtol + u_out)·|y|: at 0/0 (Qwen3.5 GDN `conv_state`, S3) the output's own
    rounding passes and one step more fails , so move such a tensor without arithmetic.
  - `sqnr` is one budget for the whole output, against the reference rounded to the output dtype. The MXFP4 MoE
    contracts set 10 dB and note that the experts' MXFP4 activations alone come to about 12 dB (DeepSeek-V2) and 13 dB
    (MiniMax-M2) (S4). Independent noise powers add; a further lossy step consumes the remaining SQNR budget. MiniMax-M2 fp8 sets 30 dB, its contract noting about 36 dB for its fp8
    activations (S4); gated attention 15 dB on `values` and 20 dB on `scales`, by its contract 6-7 dB under aiter's
    fp8-cache decode, kernels wrong on purpose under 9 and 10 dB (16.4 and 16.8 dB on the small test Point) (S3).
  - `output_rounding` (rms_norm, add_rms_norm, the quantization ops' codes and fp32 scales, gemm_a8w8 on int8) allows
    each element two roundings of the output dtype: compute in fp32, round once. `dot_product` (the 2 bf16 decode GEMMs
    of `gemm@1.0`) is 2·(6·√K + 2)·u_fp32·M + u_out·|y|, M the reference on |a| and |b|, and covers a split into up to
    `max_split_k` = 16 parts (S1, S5) : accumulate in fp32.
  - `dot_product`, `output_rounding` and `baseline_relative` also test the error's bias: |mean|/rms over the n compared
    elements must stay under 5/√n, `baseline_relative` also passing a mean within twice the baseline's; `tolerance`
    drops the test once no element exceeds its bound, `sqnr` has none (S1). Errors all one way, all inside B, failed
    `output_rounding` and `dot_product`, not `baseline_relative` against a same-mean baseline : a
    rounding that leans one way (RTZ on one-signed data, a low-biased approximation) can fail there.
- **A `dot_product` output gets one more bench step**, `large_magnitude/0` (S2; `aklr/ops/README.md` § Semantics): every floating tensor with a `finite` distribution and a cast-decoded dtype is scaled by one exact power of two, chosen under both input-dtype range limits and the conservative dot-output budget (the peak factors are floored at 1); for the two GEMMs this keeps max|a|·max|b|·K under a quarter of the output dtype's largest value: 2^54 and 2^55 for the two bf16 GEMMs. Keep every partial sum in fp32:
  an fp16 or fp8 intermediate overflows there.
- **Non-finite values must match under every bound, `sqnr` included**: finite wherever the reference rounded to the
  output dtype is finite, equal to it (NaN matching NaN) wherever it is not (S1). A stray NaN fails; so does a saturated
  finite value where the reference overflows to ±Inf (§2).
- **A bound is calibrated on the contract’s input distributions** (`aklr/ops/README.md` § Inputs and workloads).
  Do not extend accuracy to arbitrary scale distributions: the block-scaled GEMM contract uses
  scales 2^U[-3,0] and records a wider-scale warning (S5). Neither that source note nor finite
  observations establish an all-input narrow-scale guarantee.

**Inspect reference rounding points and validate actual outputs.** Rounding a router logit before
top-k can change the selected integer ids even when an unrounded result has smaller L2 error.
Likewise, per-tap bf16 rounding in decode and final normalized bf16 rounding before quantization
are distinct boundaries (S6–S9). Do not replace them based on aggregate error alone.
The output's actual bounds decide admission; copying reference roundings does not by itself
prove all remaining arithmetic fits the budget. Test integer outputs, codes, scales and state,
not only the main floating result.

**Compute in fp32: a 16-bit tensor is not widened.** A Python float next to a bf16 or fp16 tensor takes the tensor's
type, and bf16 with bf16 stays bf16 except in `/` and `%` (S10): compiled for gfx950, `x * 1.4426950408889634` on a bf16
`x` multiplies by 1.4453125 (log2 e rounded to bf16) and rounds the product to bf16, and `gl.sum` of a bf16 tensor
rounds every partial sum to bf16, since `sum` widens only integers (S10). Convert each 16-bit load with
`.to(gl.float32)` before any arithmetic, as §2's kernel does (S11), or pass `dtype=gl.float32` to `gl.sum`.

Spend the rest on fewer instructions; compiled for gfx950, each shortcut has fewer: one reciprocal per group instead of
a division per value (317 against 642 in a 16 × 128 epilogue), a bare `v_rcp_f32` or `gl.exp2` (23 and 24 against 33
for `gl.div_rn` and `gl.exp` on 64 values), bf16 MFMAs (4 against 32 fp32 ones in a 32 × 64 × 32 `tl.dot`); when each
is accurate enough: §2, §4, §5, §6. Additional roundings can consume the error budget: MiniMax-M2 MXFP4 b94f rounds gate/up accumulators, `silu(gate) · up`, and per-route down outputs to bf16, unlike its fp32 reference (S35). Qwen2-MoE 8698 adds bf16 gate/up and SiLU roundings and rounds weighted route contributions before reduction (S19); its reference already rounds the gated product to bf16 before the second linear's fp8 input quantization, as MiniMax-M2's fp8 reference does, and the Qwen2-MoE contract notes that quantizing the second linear's input
from values rounded at other points "flips some fp8 codes: about 33 to 38 dB" against its 25 dB bound (S12).

## 2. The fused fp8 group-quant epilogue (groups of 128)

52 kernels outputM3FN `values` with an fp32 scale per 128 values (gated attention 9, Qwen3.5 GDN decode 8 and prefill
15, Qwen3-Next GDN decode 9 and prefill 11), so that the next block-fp8 linear takes them directly; 16 MoE kernels
(MiniMax-M2 fp8 9, Qwen2-MoE fp8 7) quantize their own GEMM inputs per 1×128 the same way. The recipe, at the end of a
GDN prefill kernel's gated RMSNorm, after `normalized = (...).to(gl.bfloat16)`:

```python
gl.store(Normalized + row[:, None] * 128 + col[None, :], normalized, valid[:, None])
x = normalized.to(gl.float32)
quant_scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * inv_quant_max
encoded = gl.clamp(x * gl.div_rn(1.0, quant_scale)[:, None], min_quant, quant_max)
gl.store(Quantized + row[:, None] * 128 + col[None, :], encoded, valid[:, None])
gl.store(Scales + row, quant_scale, valid)
```

(S11; the host passes `quant_max` = 448, `min_quant` = −448 and `inv_quant_max` = `1.0 / 448`.)

- **Quantize the intended source value.** A bf16-rounded output and its fp32 precursor can
  give different amax, scale and codes (S7, S13). A synthetic output-rounding failure is not a
  universal GDN failure; check the actual contract at each affected output.

- **Floor amax at 1e-10 before forming the scale**, as all 68 quantizing kernels do: an all-zero group then gets a
  finite, nonzero scale and zero codes (without the floor, code 0xFE: the clamp turns the NaN into −448 ).
  The GDN and gated-attention references floor amax the same way (S7); the MoE references floor the scale at float32's
  smallest normal instead (S14), which leaves the same zero codes.
- **Keep each group of 128 inside one wave**, so its amax needs no LDS: no warps on the 128 axis of a (rows, 128) tile,
  like the excerpt's `BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])` at `num_warps=1` (S11). Compiled for gfx950, its
  `gl.max(gl.abs(x), 1)` is in-thread and cross-lane steps only; `BlockedLayout([1, 8], [16, 4], [2, 2], [1, 0])` on a
  32 × 128 tile adds 256 bytes of LDS and an `s_barrier`. A 1-D `BlockedLayout([4], [64], [2], [0])` holds a 128-value
  group whole in each wave, the second a copy (S15).
- **The amax of values already rounded to bf16** can be an integer max of their bit patterns with the sign cleared:
  non-negative bf16 patterns order like their values (MiniMax-M2 fp8 0982 at its 64-token Point, S16). A NaN pattern
  orders above Inf; map patterns above 0x7f80 to 0 where a NaN must not win, as Qwen3-Next does (S16).
- **Quantize in the producer’s last pass** when that program owns the complete group:
  normalization, required bf16 rounding, amax, scale and codes can share one program.
  A MoE consumer can instead quantize its GEMM input as it loads it ([moe.md](moe.md) §8).
  Compare total producer+consumer cost, not just the removed launch.

**Four ways to divide by the scale.** Most kernels form the scale as `amax · fl(1/448)`, a rounded product where the
reference divides; a few divide (`amax / 448`, S13) or correct the product with a residual step (S17). Then x/scale:

| Form | Kernels | Rounding of the code |
|---|---|---|
| a division per value: `gl.div_rn(x, s)` or `x / s` | gated attention, 7 of 9, all but 14e9, dfb1 (S18); Qwen2-MoE fp8 but the router+experts hidden state at 1–4 tokens (S19); Qwen3-Next decode 0678, c784 (S13), b31b, bba1, and f440 at 1, 128 and 256 tokens | correctly rounded |
| one reciprocal per group: `x * gl.div_rn(1.0, s)` or `x * (1.0 / s)` | Qwen3.5 GDN decode (S15) and prefill (S11); Qwen3-Next prefill; gated attention 14e9, dfb1; MiniMax-M2 fp8 5cac, b43d, 82e9 and 1c17's router input (these four form the scale as `gl.div_rn(amax, 448.0)`) | one more rounding: a code can move at a tie |
| `v_rcp_f32` refined, then `x * r` | MiniMax-M2 fp8, 6 of 9 carry this form: one Newton step in 0982, 8a5a (S17), 94af, e75e and 1c17's `SHORT_QUANT=True` activation path; f770 uses one Newton step followed by a residual correction with the original reciprocal | an extra reciprocal/product rounding can move an fp8 code; numerical admission is branch- and input-dependent |
| `v_rcp_f32` with a Newton step, then each `q = x · r` corrected by `fma(fma(-s, q, x), r, q)` | Qwen2-MoE fp8 router+experts hidden state at 1–4 tokens: 7d7e, 04e9, ddb8 (S20); Qwen3-Next decode 0c0b, 3719, 38a7, and f440 at 2, 4, 8, 32 and 64 tokens (35b9, and f440 at 16, skip the Newton step); b31b and bba1 carry the helpers (S21) unused | S20 intends to preserve the fp8 rounding boundary without per-element division; it is not a whole-domain equivalence proof. 0c0b and 35b9 additionally execute `v_div_fixup_f32` after quotient correction |

One reciprocal per group replaces per-value division but adds a product rounding; it can move
an fp8 code at a midpoint. Use it only where the bound allows that change. A corrected reciprocal
is another candidate, not a whole-domain substitute for `gl.div_rn(x,s)`:

```python
r = gl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [s],
                             dtype=gl.float32, is_pure=True, pack=1)
r = gl.fma(r, gl.fma(-s, r, 1.0), r)
q = x * r
q = gl.fma(gl.fma(-q, s, x), r, q)
```

The Newton and residual steps reduce reciprocal/quotient error (S20); extreme or non-finite
operands need separate handling. Both lower and upper scale bounds matter: a floor alone does
not ensure normal reciprocals. For finite fp32 amax with floor 1e-10 and division by 448, the
scale is approximately [2.23e-13, 7.60e35], keeping its reciprocal normal. Arbitrary supplied
scales are outside that argument.

**A quantization op's output is checked against its reference.** `per_group_quant_fp8@1.0` bounds codes and fp32 scales
by `output_rounding`; its reference is the second form exactly, `fl32(max(amax, 1e-10) · fl32(1/448))`, then
`fl32(1 / scale)` and `fl32(x · r)`, as "Combining the scale and reciprocal into one division changes quantization ties"
(S63), a code by one step, rarely on bf16 inputs. Reproduce it, as the excerpt does: the correctly
rounded `amax / 448` was one ulp below that product for 71 of the 128 bf16 significands and above it for none in each of the 25 tested normal-scale binades: inside the
bound, but all one way, which failed the bias test (§1). A code one step off fails where the reference's
code is 0 or subnormal: there the bound is under a step.

**Saturation and clamping.**
- Clamp to ±448 before converting. 44 kernels call `gl.clamp`, which is one `v_med3_f32` (S22); the 8 Qwen3.5 GDN decode
  kernels write the `v_med3_f32` as inline asm (S15). The 16 MoE quantizers skip the clamp: with scale = amax/448 and no
  clamp, x/scale exceeded 448 by at most 448·2^-22, and Triton's fp8 cast stored 448 there.
- With a NaN operand `v_med3_f32` returns `v_min3_f32` of its operands, `min(min(x, lo), hi)` in Triton's order (S23;
  compiled for gfx950): per the ISA a quiet NaN, which arithmetic produces, clamps to −448, and a signaling NaN loaded
  from memory to +448, quieted by the first min in IEEE mode, the default (S57); `torch.clamp` keeps the NaN. These 68 quantizers' floating-tensor contracts admit only finite decoded inputs, and their stored Point scalars are finite; this does not guarantee finite intermediates or every sampled cast. Where a NaN in the clamp's `x` operand (with finite bounds), or either min/max operand, must reach the output, pass `propagate_nan=tl.PropagateNan.ALL`
  (`import triton.language as tl`; Gluon does not re-export it) to `gl.clamp`, `gl.maximum` or `gl.minimum` (S24):
  compiled for gfx950, the clamp gains `v_cmp_u_f32` and `v_cndmask_b32`, `gl.maximum` becomes one `v_maximum3_f32`.

**The convert.** For fp32 `x`, `x.to(gl.float8e4nv)`, or `gl.store` into anM3FN pointer (which casts to the pointee
type, S10), lowers on CDNA4 to `v_cvt_scalef32_pk_fp8_f32 …, 1.0`: scale 2^0, pairs of values, round to nearest even
only (S25). The same lowering makes Triton set MODE.FP16_OVFL at kernel entry,
`s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1` (S22); a compile for gfx950 shows both. With FP16_OVFL set, ±Inf and
values beyond 448 convert to ±448 and NaN stays NaN; with it clear they become NaN (S26).
- Without an explicit MODE write, the bit stays set for the whole kernel; inline asm can clear and later restore it, as the Qwen3-Next 38a7 path does. Per the ISA it also clamps an overflowing FP16 result to ±MAX_FP16 and an
  overflowing F32 MFMA result to ±MAX instead of ±Inf (S27, S26) : a kernel with any Triton fp8 cast can
  return a finite maximum where the reference overflows, which compare rejects (§1).
- An asm fp8 convert in a kernel without a Triton fp8 cast or an explicit MODE write starts with FP16_OVFL clear: overflow and ±Inf become NaN. Inline asm that sets bit 23 can instead enable saturation.
- The Qwen3-Next GDN decode kernels (passed CI run 42000899019, recorded on main in d908d01 ) state that
  the bit "also saturates BF16 conversions" (S21), as measured: ±0x7F7F instead of ±Inf. They keep IEEE
  overflow three ways (S21; [gdn.md](gdn.md) §6; `s_setreg`: [gluon/wave_ops.md](gluon/wave_ops.md) §7), which matters
  only where a bf16 convert, an FP16 result or an F32 MFMA result can overflow before the fp8 convert.

## 3. MXFP4 activation quantization

All 42 MXFP4 MoE kernels (DeepSeek-V2/GLM 19, MiniMax-M2 23) quantize both expert-GEMM inputs to MXFP4 in registers:
e2m1 values, one e8m0 scale per 32. The MoE references leave activations unquantized; this rounding is the kernel's,
its error left to the 10 dB bound (S14, S4); a quantization op's is not (below). The recipe:

```python
peak = gl.max(gl.abs(x), 1)
ratio = gl.div_rn(peak, 6.0)
bits = ratio.to(gl.uint32, bitcast=True)
exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
exponent = gl.minimum(127, gl.maximum(-127, exponent))
scale = gl.exp2(exponent.to(gl.float32))
normalized = x * gl.div_rn(1.0, scale)[:, None]
normalized = gl.where((normalized == 0) | (normalized != normalized), 0.0, normalized)
even, odd = gl.split(gl.reshape(normalized, (x.shape[0], 16, 2)))
packed = gl.inline_asm_elementwise(
    "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]", constraints="=v,v,v",
    args=[even, odd], dtype=gl.uint32, is_pure=True, pack=1,
).to(gl.uint8)
return (packed, (exponent + 127).to(gl.uint8))
```

(S28.) `x` is fp32 of shape (groups, 32), one MX block per row; the codes and scale bytes are MXFP4, ceil rule (below),
RNE e2m1. `gl.split` needs each pair in one thread: at least 2 consecutive elements per thread along the
32, else compiling fails: "Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension". Load
the input in a pair-holding layout and only reshape it, `gl.reshape(value, (groups, 32))`. Offline gfx950 SwiGLU + MXFP4 builds with 256 bf16 columns, NW = 1, 2, 4, 8, found no LDS or `s_barrier` for `BlockedLayout([1, 8], [16, 4], [NW, 1], [1, 0])` at tested row counts 16 through 64·NW, or `BlockedLayout([1, 32], [64, 1], [NW, 1], [1, 0])` at exactly 64·NW rows; taller tiles were not compiled.
Converting from the first to `[1, 2], [4, 16]`, as 5751's packer does (S29), used 8–256 KiB LDS and, at NW ≥ 2, an `s_barrier`; directly loading that pair-holding target also used no LDS.
At NW = 4 the second layout on 64 × 256 holds every row in all four waves (two zero warp bases), spilling 177 VGPRs for codes, 179 with scales. Missing rows are not the cause: 128 and 256 rows also spill 177–179, with no duplication at 256; each lane holds 256 values.
The tested 256 × 64 tile at NW = 4 did not spill. At 64·NW × 256 both layouts spilled (164–193 VGPRs at NW ≤ 4, 570–805 at NW = 8). These are compile observations, not GPU correctness or speed measurements.
MiniMax-M2 b94f loads hidden state directly and converts its gate-up MFMA accumulator (S35; [moe.md](moe.md) §8).

**The e8m0 exponent: two rules.**

| Rule | Exponent from the block maximum amax | amax lands in | Used by |
|---|---|---|---|
| ceil | ⌈log2(amax/6)⌉: the unbiased exponent of the correctly rounded amax/6, plus one if its mantissa is not zero; checked for amax ≥ 2^-122, where amax/6 is normal  | (3, 6]·scale: no value exceeds the e2m1 maximum 6 | all 23 MiniMax-M2 MXFP4 kernels ("the block scale rounded up from its maximum / 6", S28); DeepSeek-V2's routed experts |
| even (aiter's) | ⌊log2 amax⌋ − 2, plus one if amax ≥ 1.75·2^⌊log2 amax⌋, equal to aiter's `(bits + 0x200000) & 0xFF800000` rule  | [3.5, 7)·scale before exponent clipping; checked for amax from 2^-124 to 2^125 : values above 6·scale reach the convert, which saturates them to ±6 | DeepSeek-V2's shared-expert slot (S30) |

For the tested maxima from 2^-120 to 2^121, the two differ exactly when amax's mantissa lies in (1.5, 1.75) : ceil moves to the next exponent, so the
block maximum lands in (3, 3.5)·scale and the block uses half the fp4 range; even keeps the exponent and puts it in (6,
7)·scale, above 6. The reference names the difference and leaves it to the bound ("aiter rounds a block's maximum to
nearest even, others up", S14); DeepSeek-V2 quantizes the input once per rule and picks per route (S30). Neither scale rule universally gives higher SQNR; compare against the intended reference.
Reproduce aiter's codes or an op's reference where required: the e8m0 bytes of `per_group_quant_mxfp8@1.0` have no bound and must equal its reference's (§1), the biased exponent of `fl32(max(amax, 1e-10) · fl32(1/448))`
plus one if its mantissa is not zero, with codes `x / 2^(byte − 127)` clamped to ±448 (S63). Clamp the
exponent to [−127, 127]: the stored byte `exponent + 127` then stays in 0…254, short of 0xFF, the e8m0 NaN (S31).

**Exact prescale, and the flush in `gl.exp2`.** Scaling by a power of two is exact where the result is normal
: `x * gl.div_rn(1.0, 2^e)`, one division per group (S28), or each value divided by 2^e (7 DeepSeek-V2
kernels, e.g. d3bb, S30). But `gl.exp2` is one bare `v_exp_f32`, with or without `allow_flush_denorm` (the backend sets
`__HIP_FTZ = True`; S32, S22), which flushes denormal inputs and outputs (S33): an all-zero group gets e = −127,
`gl.exp2(-127.0)` is 0, `1 / 0` is +Inf and `0 · Inf` is NaN (dividing each value, `0 / 0` is NaN directly). Pick one:
- **Sanitize before the convert**: map NaN and ±0 to +0 (the MiniMax-M2 hardware encoders, S28) or NaN to 6.0 (9
  DeepSeek-V2 kernels, S30). In an e2m1 encoder this is defensive: E2M1 has no NaN or Inf code (S26) and the group's
  scale byte is 0 (2^−127), so any code decodes to at most 6·2^−127. The ISA does not say what the convert returns for a
  NaN input; 10 DeepSeek-V2 kernels convert it unguarded.
- **Avoid the Inf**: `ef = exponent.to(gl.float32)`, `clipped = gl.minimum(ef, 126.0)`, then `gl.exp2(-clipped) *
  gl.exp2(clipped - ef)` keeps both factors normal and needs no division (S28). Convert first: on the recipe's int32
  `exponent`, `gl.exp2` fails to compile ("Expected dtype ['fp32', 'fp64'] but got int32").
- **Build the power of two from its bits**, exact: `((e + 127) << 23).to(gl.float32, bitcast=True)` for e in [−126,
  127]. Every committed MX quantizer instead trusts `gl.exp2(e)`, 1 ulp per the ISA, to be exactly 2^e (S33, S34), as
  it was ; only DeepSeek-V2 a1c4's exactness-checked int16 transport of fp32 partials builds them from
  bits, its biased exponent kept in [1, 253] so the power and its reciprocal stay normal (S34).

Outside an e2m1 encoder (a decay, an fp8 code, a running sum) the same NaN reaches the output  and fails
compare (§1) : use the last two, or `gl.exp`, which keeps subnormal results (§6).

**Two e2m1 values per byte.** `v_cvt_scalef32_pk_fp4_f32 D, S0, S1, S2` with S2 = 1.0 (2^0; a power-of-two scale is taken from its exponent) rounds two fp32 values to nearest even, saturates to ±6 with FP16_OVFL clear or set, and writes one byte
chosen by OPSEL[3:2], S0 in the low nibble; "Other destination bits are preserved" (S23, S26). Declare
the result `dtype=gl.uint32` and truncate with `.to(gl.uint8)` (S28); to zero the other bytes too, start with `v_mov_b32
$0, 0` and an early-clobber output (`"v_mov_b32 $0, 0\nv_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0"`, `"=&v,v,v"`), still
`gl.uint32` (S29). Do not declare `dtype=gl.uint8`: with `pack=1`, LLVM prints `error: could not allocate output
register for constraint 'v'` to stderr, Triton 3.8.0 raises nothing, and the asm compiles to nothing: the kernel
launches and writes no codes, or wrong ones (asm rules: [gluon/wave_ops.md](gluon/wave_ops.md) §2). A software encoder
also works: MiniMax-M2 1d27 compares |x| with the code midpoints, alternating `>` and `>=` so that ties go to the even
code, and a NaN compares false and becomes code 0 (S35); it serves the selected kernel Points.

**Scales on the way to the MFMA.** The e8m0 bytes go into `mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)`, and the
hardware applies them after the block's dot product (S31). Scaled MFMA subnormal handling must not be treated as a promise of exactly one IEEE fp32 rounding. Fill masked scale bytes with 127 (2^0) and masked scale words with `0x7F7F7F7F`, not 0xFF:
one scale multiplies a whole block's dot product, and 0xFF is NaN (S31, S36). Derive the scale layout with
`get_mfma_scale_layout`; unshuffle aiter's e8m0 bytes first: [gluon/mfma.md](gluon/mfma.md) §7.

## 4. fp32 block-scaled GEMMs: one unit-scale MFMA per 128-K group

The fp8 GEMMs with fp32 block scales (the 24 a8w8 blockwise GEMMs; 13 of the 16 fp8 MoE kernels: MiniMax-M2's 9,
Qwen2-MoE 04e9, 7d7e, 99f4, ddb8) keep the scales out of the matrix core: each 128-K group gets its own MFMA from a zero
accumulator with unit scales, `gl.amd.cdna4.mfma_scaled(aa, None, "e4m3", bb, None, "e4m3", zero)`, and the scale pair
multiplies that partial in fp32, `dot * sx[…][:, None] * sw[…]` (S37; [gluon/mfma.md](gluon/mfma.md) §2.1).

- **Why outside the MFMA.** Its scale operand is an e8m0 exponent (S31), the a8w8 block scales fp32 values from 2^U[-3,
  0], not powers of two (S5) ; one `16x16x128` spans one 128-wide scale block, so one `sx · sw` scales it.
- **What Gluon emits.** With both scales `None`, `mfma_scaled` builds constant `uint8` 0x7F (2^0) scales and an fp32
  `dot_scaled` (S38); compiled for gfx950 it stays `v_mfma_scale_f32_16x16x128_f8f6f4` with 0x7F in a VGPR as both scale
  operands. Instruction choice and rates: [gluon/mfma.md](gluon/mfma.md) §2, [gemm.md](gemm.md) §2.
- **Inside a group** the MFMA rounds to nearest even and keeps denormals (S31; subnormal results not exactly, §3); its
  summation order is not documented, and a FIXME beside the a8w8 tolerance records that the gfx950 fp8 MFMA "accumulates
  below _dot_k's IEEE fp32 model" (S5), as one large product among small ones showed.
- **Across groups** one program owns an output tile over all of K and folds the groups in a fixed order: sequentially
  (S37), or, where one MFMA carries two K groups in alternate rows, into two accumulators by the parity of that MFMA,
  row pairs and accumulators summed at the end (S39). No a8w8 kernel uses an atomic or splits K across programs: their
  outputs repeated bit for bit at all 24 Points , though the contract is `bounded` and CI does not check.
- **The scale step** is written several ways: `acc + dot * sx * sw` (S37), `gl.fma(partial * sx, sw, acc)` (S39),
  `gl.fma(partial, xs[:, None] * ws, acc)` (S39). Under the default contraction the first compiles to the second's
  multiply and FMA (§6), with the same bits; the third rounds the other product: 20% of elements differ.
  Write `gl.fma` to choose. None is the reference's `(a · a_scale) @ (b · b_scale)ᵀ` (S40); at n7168, M = 64,
  `dot * sx * sw` and `dot * (sx * sw)` can differ in their last bits; validate each full fold.
- **The output**: `.to(gl.bfloat16)` rounds once to nearest even, `v_cvt_pk_bf16_f32` on pairs (S25). Convert before a
  cross-wave `gl.convert_layout`, which moves half the bytes: `gl.convert_layout(acc.to(gl.bfloat16), STORE)` (S37).
- **MoE fp8 experts** use the same atom; a gate-up tile can hold gate and up columns from different weight-scale blocks,
  `sb = gl.where(col < BN, sg, su)` (S17). Kernels that run fp8 on 16-bit MFMAs (Qwen2-MoE fp8 experts 5c35, 8698,
  d425) expandM3FN to bf16 or fp16 first, which is exact ("FiniteM3FN operands are represented exactly in BF16",
  S19), and apply the block scales the same way. Tiling and per-Point choices: [gemm.md](gemm.md), [moe.md](moe.md).

## 5. fp32 accuracy from 16-bit MFMAs

The fp32 MFMA (`instr_shape` [16, 16, 4]) does 32 multiply-adds per cycle against 512 for the bf16 [16, 16, 32] form
(S26; theoretical instruction capacity, not kernel time). The GDN prefill kernels keep it for their state algebra
and solves; Qwen3.5 19b2 runs K·Kᵀ and Q·Kᵀ on the bf16 MFMA (S41), as the reference rounds the L2-normalized Q and K to
bf16 (S7). A finite numerical check is not a proof of IEEE fp32 accumulation; preserve the contract’s
input ranges and compare against an adequate reference.

**Split only the fp32 operand (MiniMax-M2 router).** The input is bf16; the fp32 router weight becomes three bf16 terms:

```python
b0 = b.to(gl.bfloat16, fp_downcast_rounding="rtz")
residual = b - b0.to(gl.float32)
b1 = residual.to(gl.bfloat16, fp_downcast_rounding="rtz")
b2 = (residual - b1.to(gl.float32)).to(gl.bfloat16)
high = gl.amd.cdna4.mfma(a, b0, high)
middle = gl.amd.cdna4.mfma(a, b1, middle)
low = gl.amd.cdna4.mfma(a, b2, low)
```

(S17; the result is `high + (middle + low)`.) The two truncations take 16 significant bits and leave at most 8, so the
three terms hold the fp32 weight exactly unless a lower term falls below bf16's normal range, and a bf16 × bf16 product
in fp32's normal range is exact : rounding happens only in the accumulation. RTZ to bf16 is a bit
truncation (S25), which turns a NaN whose payload sits in the low 16 bits into Inf; route non-finite weights around it.
Plain Triton's `input_precision="bf16x3"` splits both operands (S42): the tested 16 × 16 × 64 fp32 `tl.dot` compiled
to 6 bf16 MFMAs versus 16 fp32 with `ieee`; Gluon's `mfma` under `TRITON_F32_DEFAULT=bf16x3` compiled to the same ISA
as without it. Redo the split on every call: the weight is an input (`aklr/ops/README.md` § ABI).

**Choose the precision path from the full router operation.** MiniMax-M2 uses fp32 products
and `gl.sum` for small token counts, fp32 MFMA or three-term bf16 products for larger tiles
(S17, S36). Three bf16 terms reduce matrix work relative to fp32 MFMA, but add splitting and
folding. Preserve normal-range exactness conditions and judge logits plus top-k, not L2 error
alone. Near ties, repeated selection in producer and consumer must use the same ids, weights
and expert-slot mapping; otherwise even accurate arithmetic can send a contribution to the
wrong expert slot.

**Round once where the reference rounds once (routers).** The Qwen2-MoE reference computes the router logits exactly in
float64 and rounds once to bf16 (S8). The routers sum fp32 split partials in a fixed order and round once: "Route from
exactly the BF16 values exposed in the logits output" (S43). An fp32 sum can still land on the other side of a bf16
midpoint, which the contract allows on the weights ("a logit summed in float32 may round to the next bf16") but not on
`topk_ids` (S12); integer outputs still require independent validation. 7d7e sums the router dot in float64 on the
VALU and nudges fp32 midpoints so that fp64 → fp32 → bf16 does not round twice (S44) ; 99f4 runs the dot
on the fp64 MFMA with the same fix, and makes its expert gate-up's fp8 × fp8 dots exact by splitting eachM3 value
into integer-valued fp16 halves over three MFMAs (S45). These precision paths change the full router/expert operation; no isolated speedup
is assigned to exactness alone.

**P·V in attention.** Choose P's precision against the SQNR margin. The attention bounds are loose because the baselines
are lossier (by the DSA contract, sglang's kernel rounds the softmax weights to fp8, about 33 dB against a 20 dB bound,
S46), and plain bf16 P passes them. With fp8 V, convert P and V to fp16: E4M3FN converts exactly and fp16 keeps 11
significant bits of P in [2^-14, 1] ; with bf16 V keep both in bf16, since bf16 → fp16 is exact inside
fp16's normal range, 2^-14 to 65504, and outside it only for the 1022 bf16 values fp16's subnormals hold (S47;
exhaustive). A bf16 high part plus a bf16 residual stacked as extra MFMA rows and folded after keeps about 16 bits, for
a subtract and a convert per probability and an add per output. It needs no extra MFMA where the rows would be padding,
8 heads in a 16-row MFMA (S47); DSA's 8-token file stacks the two parts of 16-head blocks into `[32, 32, 16]`, twice
the P·V MFMA work of bf16 P on `[16, 16, 32]` (S48). Per kernel: [attention.md](attention.md) §10.

**Compensated products where the result is rounded anyway (GDN prefill).** "Only output reconstruction uses compensated
BF16 products: chunk solves, the persistent recurrence, and mutable state remain FP32" (S49): 19b2 and 3828 split the
fp32 state into bf16 hi/mid(/lo) terms and multiply them with the bf16 query, smallest term first (S41); 1fd6, 61ed and
1d55 store coefficients and state snapshots as bf16 high + low pairs and multiply high, then low (S49). The solve has
rounding room within fp32: the in-block triangular inverse is updated as `gl.fma(value, diagonal, solved)` and, rounded
twice (§6), as `solved += gl.where(mask, column * diagonal_row, 0.0)` (S50); both appear in admitted AKLR implementations; validate changes. Form products of
decays directly ("without dividing tiny prefixes", S11): a ratio of prefixes turns into 0/0 once they underflow. Code, padding, chunk algebra: [gdn.md](gdn.md) §10.

## 6. Transcendentals, division and contraction on gfx950

What each fp32 call becomes with Triton 3.8.0, compiled for gfx950 (VALU instructions between the load and the store;
lowering in S22, S25, S51, S10; the native instructions' precision in S33):

| Call | Instructions | Precision | Use it for |
|---|---|---|---|
| `x / y`, `gl.div_rn(x, y)`, `gl.fdiv(x, y)` | the IEEE sequence, 11 VALU: `v_div_scale_f32` ×2, `v_rcp_f32`, 5 FMAs, a `v_mul_f32`, `v_div_fmas_f32`, `v_div_fixup_f32` | correctly rounded  | quotients at rounding points, once per group rather than per value |
| `gl.exp2(x)` | `v_exp_f32` | 1 ulp; denormal input and output flushed | `1 + exp(·)`, softmax, 2^e for e in the normal range, decays (below) |
| `gl.exp(x)` | a `v_mul_f32` by fl32(log2 e), `v_exp_f32`, `v_ldexp_f32` (9 VALU) | a subnormal result kept (`gl.exp(-100)` 0.45 subnormal ulp off); relative error grows with x: 2.2, 15.9, 64.8 × 2^-24 on [0.5, 1], [8, 16], [80, 88]  | a subnormal result that is divided by or is an output (below) |
| `libdevice.exp(x)` (`from triton.experimental.gluon.language.extra import libdevice`) | `__ocml_exp_f32` (S51): `x · log2(e)` in two parts, `v_exp_f32`, `v_ldexp_f32` (14 VALU) | under the default contraction it also grows with x: 2.0, 12.4, 45.5 × 2^-24 on the same ranges  | ahead of a bf16 rounding point barely better than `gl.exp`: 2117 against 2209 bf16 mismatches in 2^24 inputs ; the 26 GDN prefill kernels' conv SiLU (S11) |
| `gl.log(x)` | `v_log_f32` among 13 VALU | a denormal input kept: `gl.log(1e-40)` ≈ −92.1   | |
| `gl.log2(x)` | `v_log_f32` among 7 VALU | denormals kept: 2^-140, 2^-149 exact  | |
| `gl.rsqrt(x)`, `gl.sqrt(x)` | `v_rsq_f32`, `v_sqrt_f32` | within 1 ulp (0.86, 0.92 over 2^24 inputs); denormal 1e-40 flushed  | an RMS norm ahead of a bf16 rounding point (`gl.rsqrt`: every output within 0.5 bf16 ulp on 10^6 values ) |
| `gl.sqrt_rn(x)` | the IEEE sequence (17 VALU) | correctly rounded  | |
| `gl.fma(a, b, c)` | `v_fma_f32` or `v_fmac_f32`; `v_pk_fma_f32` on register-adjacent pairs | one rounding | pinning a contraction |
| `gl.clamp(x, lo, hi)` | `v_med3_f32` | a quiet-NaN `x` yields `lo`, a signaling one `hi` (§2) | the ±448 clamp |
| `gl.maximum(x, y)`, `gl.max(x, axis)` | `v_max_f32`, plus a quieting `v_max_f32 v, v, v` per loaded operand in IEEE mode, the default (`amdgpu-ieee` below) | maxnum: a NaN operand is dropped | amax of finite groups |
| `.to(gl.bfloat16)` | `v_cvt_pk_bf16_f32` on pairs | round to nearest even | rounding points |
| `.to(gl.bfloat16, fp_downcast_rounding="rtz")` | bit truncation (`v_perm_b32` of the high halves, on value pairs) | toward zero | exact splits (§5) |

**`gl.exp2` or `gl.exp` for a decay.** A flushed subnormal is lost inside `1 + exp(x)` anyway: "Native exp2 is
sufficient inside 1 + exp(x): any flushed subnormal is already too small to change that sum. Retain exp for decay."
(S52). A flushed decay changes `state · decay` by less than 2^-126·|state| , far inside the GDN decode
state bound of 1e-2 absolute and 1e-2 relative (S3): Qwen3.5 decode 2991 takes `gl.exp2` for the decay itself and
passes at its Point , while 4f25 and 409c keep `gl.exp` (S52). Keep `gl.exp` where a subnormal result is
itself an output, or is divided by while its reciprocal still fits fp32 (x above about −88.7, estimated).

**Native `v_rcp_f32`, `v_exp_f32` and `v_log_f32`** (written as inline asm in 16, 10 and 7 kernels). Use them where the
full-precision lowering does work the value does not need (S33), and write asm only for `v_log_f32`: `gl.exp2` already
is one `v_exp_f32` and `libdevice.fast_dividef(1.0, x)` one `v_rcp_f32` (S51; compiled for gfx950); the exact matching LLVM hazard-source branch remains unbound here ([../attention/softmax.md](../attention/softmax.md) §2;
[gluon/wave_ops.md](gluon/wave_ops.md) §4, §9.1, and §2 for the constraints, packing and purity of asm calls).

LLVM sees an inline-asm block's declared register defs/uses, but instructions inside one
multi-opcode string are not separate schedulable nodes. Some boundary hazards can receive
compiler padding; do not assume either complete invisibility or complete protection.
Supply the ISA-required internal waits, memory clobbers and dependencies, and inspect the
instructions immediately outside the block ([gluon/wave_ops.md](gluon/wave_ops.md) §9.2).

- `v_rcp_f32` instead of a division: 1 instruction instead of 11, 1 ulp, denormals flushed; one Newton step (two FMAs)
  brings it to at most 0.5 ulp on the tested inputs. The Qwen3-Next decode template (S53) keeps a true division
  "anywhere the reference expression is kept verbatim", a bare `v_rcp_f32` “valid only where the caller immediately
  rounds to BF16” (a numerical approximation, not a guarantee of identical bf16 bits), and `v_rcp_f32` of the argument scaled by
  2^-64 where the result can be subnormal (S53). Where the quotient is itself a rounding point, add §2's correction.
- `v_exp_f32`: the template's native exponential is `gl.exp2(x * log2(e))`; squaring `v_exp_f32(x · log2(e) / 2)`
  restores the gradual underflow that the native instruction flushes, without a branch (S53).
- `v_log_f32` (base 2, 1 ulp, denormals flushed) for the 13 instructions of `gl.log`: the Qwen3-Next decode softplus
  `log(1 + exp(a))` (S53) uses it, its argument being at least 1 (S54); times a rounded ln 2,
  it differed from `gl.log` in 27% of fp32 results for a in [−20, 20].

**FP contraction is on by default.** `enable_fp_fusion` defaults to the knob `TRITON_DEFAULT_FP_FUSION`, true (S32):
compiled for gfx950, `a * b + c` is one `v_fmac_f32` (one rounding), with `enable_fp_fusion=False` or the knob at 0 a
`v_mul_f32` and a `v_add_f32` (on register-adjacent pairs `v_pk_fma_f32` against `v_pk_mul_f32` and `v_pk_add_f32`). A
select also keeps them apart: `c += gl.where(m, a * b, 0.0)` is `v_mul_f32`, `v_cndmask_b32`, `v_add_f32` under the
default. Where a contract fixes how a product is rounded, write the rounding (the tap products of §1), `gl.fma` for the
single rounding you want, or `enable_fp_fusion=False` for two roundings. In cancellation-sensitive
expressions a contracted product can preserve a residual that a rounded multiply followed by subtraction
loses. Validate the intended expression; compiler algebra and a cast round trip are separate choices.

**`allow_flush_denorm` is not an operation-wide FTZ guarantee.** It sets `denormal-fp-math-f32` to `preserve-sign` (S32), but
compiled for gfx950 `gl.exp`, `gl.exp2`, `gl.log`, `gl.log2`, `gl.sqrt`, `gl.sqrt_rn`, division and the descriptor's
`.amdhsa_float_denorm_mode_32 3` stay as without it, as do the 4 DSA 64-head kernels that pass it (§7).

**`amdgpu-ieee` false.** Passed as `llvm_fn_attrs=[["amdgpu-ieee", "false"], …]` on the routing launch of 5 of the 7
Qwen2-MoE router kernels (4801, 5f20, a878, c388, cd47; S56), it clears IEEE mode (`.amdhsa_ieee_mode 0`), and LLVM
drops the quieting `v_max_f32 v, v, v`: `gl.maximum(x, y)` of two loads is one instruction instead of three (compiled
for gfx950). A min or max then returns the other operand for any NaN input (S57, S27).

To confirm in a build ([debugging.md](debugging.md) §1, §3): count `v_div_fixup_f32` (one per division); find
`s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1)` (FP16_OVFL), `v_cvt_scalef32_pk_fp8_f32`, FMAs against a separate multiply
and add; read `.amdhsa_ieee_mode` and `.amdhsa_float_denorm_mode_32`.

## 7. Numerics switches: semantics before speed

`enable_fp_fusion` changes contraction where LLVM can combine the operations; it need not
change every kernel. Explicit `gl.fma` keeps one rounding, while a separate multiply/select/add
can retain two (§6). A setting that leaves one kernel's ISA unchanged does not imply all
kernels are unaffected. Preserve the rounding rule the contract requires, then measure the
specific compiled launch. No general “False is necessary” or “True is always faster” follows.

`amdgpu-ieee` affects NaN quieting and min/max instruction sequences; it is not merely a
performance switch. `allow_flush_denorm` sets a function attribute, but native `exp2` remains
FTZ independently. Verify descriptors, instructions and outputs rather than assuming an
option name describes every operation ([tuning.md](tuning.md) §9).

## 8. Determinism and atomics

- **Use ordered floating reductions for bitwise reproducibility.** A source inventory is not a
  proof of runtime ordering; inspect each reduction path. The plain-Triton decode GEMM `gemm_n2624_k6144` splits K six ways only for 16 < m ≤ 64 (of
  its Points, m = 64): each split writes an fp32 partial tile, and "a second kernel adds the six partials in a fixed
  order before rounding to bf16, so the result is bitwise reproducible" (S58), as it was at all its Points
  ; when that split pays: [gemm/decode_gemm.md](gemm/decode_gemm.md) §6.
- **Integer atomics can place work without ordering floating sums.** AKLR kernels reserve
  tickets, job slots, cursors and masks with integer adds/ors; some also reserve LDS slots.
  Keep the subsequent floating reduction ordered (S19, S28).

```python
ticket = gl.atomic_add(Counts + experts, 1, valid, sem="relaxed")
gl.store(Sorted + experts * CAPACITY + ticket, row * TOPK + slot, valid)
new_job = valid & (ticket % 16 == 0)
```

(S28.) `Counts` and `Total` must read zero at the start of every call, graph replays included (unzeroed, a replay's
tickets started at 4096 ): an earlier launch of the same call zeroes them ([moe.md](moe.md) §6). Where LDS
reservations reorder rows, the Qwen2-MoE experts put every contribution back before summing: "Integer reservations may
reorder rows, but every contribution scatters to its original entry before the ordered eleven-route reduction" (S19).
`gl.atomic_add` defaults to `acq_rel` at `gpu` scope (S10); `relaxed` suffices for counters a later launch reads.

A kernel can select different MFMA bodies at runtime from the live-row count (S59).
That changes the potential product/fold order; byte-repeat observations at one count do not
prove all natural scheduling paths are reproducible. Repeat the actual workload and preserve
the mapping between producer partials and consumer slots before attributing last-bit changes
to the matrix instruction.

What a new kernel should do:
- Under `bitwise`: no atomic split-K. Write fp32 partials to scratch and reduce them in a fixed order, or give each
  output to one program ([../gemm/correctness.md](../gemm/correctness.md) §8).
- Under `bounded`: let atomics place work, and keep every float sum in a fixed order anyway: easier to debug.
- `bitwise` compares two runs, not kernels (§1): a new tile, split count or K rotation changes bits and stays correct.
  Judge by the bound, not by bit-equality with the old kernel or aiter, and repeat a kernel with atomics or cross-wave
  races many more times yourself ([../gemm/correctness.md](../gemm/correctness.md) §8, §3).

## 9. Matching the reference beyond rounding points

- **Ties.** The Qwen2-MoE reference breaks top-k ties toward the lower expert id with a stable descending sort (S8), and
  `topk_ids` must equal it: 7d7e builds order-preserving integer keys that map −0 to +0 and carry `511 - expert` in the
  low bits (S44). The DeepSeek-V2 and MiniMax-M2 references use `torch.topk`, which fixes no tie order, under an SQNR
  bound on `out`; their kernels still take the lower id, with `(av > bv) | (av == bv) & (ai < bi)` (S36) or the lowest
  index among the maxima (S59).
- **The reference, not the baseline.** DeepSeek-V2 kernels add the correction bias in fp32 as the reference does (S29),
  not in bf16 as sglang's top-k receives it (S60).
- **NaN on purpose.** `gl.maximum(x, -float("inf"))` turns a NaN into −inf (a loaded −inf ): "A NaN tail
  becomes the removed head's -inf placeholder, matching the reference's behavior when fewer than TOPK non-NaN experts
  remain" (S43). Elsewhere `gl.maximum` and `gl.max` drop a NaN (maxnum) where `torch.amax` propagates it, and
  `gl.clamp` returns its lower bound for a quiet NaN: equal to torch on 10^6 random finite values (no ±0 or subnormals),
  different on NaN. `gl.max` takes no `propagate_nan`: use `gl.reduce(x, axis, combine)` with a combine
  returning `gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)` (§2; one wave: `v_maximum3_f32`, no LDS).
- **Neutral padding.** Masked activations load as 0, e8m0 scales as 127 (S36), GDN padding tokens as decay 1 and beta 0
  (S11); empty attention splits publish `(max = -inf, den = 0, num = 0)`, so beside a split of finite max their merge
  weight exp(−inf) is 0 (S47) ; merging two empty records needs explicit handling of −inf − −inf.
- **Casts at stores.** `gl.store` casts to the pointer's element type, rounding to nearest even (S10);
  `gl.amd.cdna4.buffer_store` does not, and a value whose element type differs from the pointer's fails to compile
  (S61): cast first. Buffer stores: [gluon/memory.md](gluon/memory.md) §3.
- **Dead fp8 variants.** 12 kernels keep a group-maximum branch of 224 (240) for a non-E4M3FN fp8 dtype (gated
  attention 10df, 3d0f, 42a2, f5ba; Qwen3.5 GDN prefill 0a07, 19b2, 24ed, 2ece, 3828, c53b; Qwen3-Next c784, 11f6); five
  more gated kernels name `float8_e4m3fnuz` only as a default they assert away. AKLR's `DType` has no FNUZ member (S62)
  and no contract or schema declares one, so no contract admits FNUZ inputs: do not copy those branches.

## 10. Pitfalls

A failed `compare` reports `violation_fraction`, `peak_bound_multiple`, `bias_z_score` and non-finite mismatches (S1):

| The report | Look first at |
|---|---|
| non-finite mismatches | NaN from `0 · (1 / gl.exp2(-127.0))`, or a scale byte 0xFF (§3); Inf where the reference is NaN, as from RTZ (§5); a finite maximum (fp8 448, fp16 65504, the fp32 maximum) where the reference overflows, as from a Triton fp8 cast's FP16_OVFL, or NaN where it is finite, as from an asm fp8 convert without it (§2); each of these failed compare  |
| many elements far over the bound | arithmetic left in bf16 (§1; a bf16 running sum over K = 2048: 94% of elements over, peak 58 × B); garbage in the other nibble or bytes of an fp4 code word (§3; 44% over); an output not written, as by an asm result declared `gl.uint8` (§3; all over) ; then indexing ([debugging.md](debugging.md) §5) |
| a few elements over (about 0.01%, peak up to about 3 × B, under bounds of about one rounding step ) | a rounding point missed or doubled: the round trip (§1), contraction (§6), `x * (1 / s)` for `x / s` (§2) |
| `bias_z_score` over 5/√n, no element over | a rounding that leans one way, such as RTZ or truncation (§1), or a quantization op's scale formed otherwise than its reference's (§2) |
| SQNR short | a further lossy step in the budget (§1), the e8m0 rule (§3), P's precision (§5) |
| step `large_magnitude/0` fails | an fp16 or fp8 intermediate under a `dot_product` bound (§1) |
| step `determinism/0` fails, or a mismatch comes and goes | atomics and other arrival-order sums; two runs can miss them: rare arrival-order differences can evade adjacent repeat comparisons  (§8) |

## Evidence

- S1 aklr/core/contract/semantics/_compare.py:143 `runs_agree`: `bitwise`, every run's output byte-identical to the first's; `bounded`, nothing across runs. A compare's measures: `violation_fraction`, `peak_bound_multiple`, `bias_z_score`, non-finite mismatches (:21-27), the fraction checked against `max_violation_fraction` (:216); the element-wise bias test |mean|/rms ≤ 5/√n over the n compared elements (:237), dropped by `tolerance` once nothing exceeds B (:329), also passed by `baseline_relative` with a mean error within twice the baseline's (:295-310); `DotProductBound` B = 2·(6·√n_eff + 2)·u_acc·M + u_out·|y|, n_eff the largest ⌈n/s⌉ + bit_length(s − 1) over s ≤ `max_split_k` (:248-266); `OutputRoundingBound` B = 2·u_out·|y| (:269-275); `ToleranceBound` B = atol + (rtol + u_out)·|y| (:321); `SqnrBound` (:332); non-finite elements matched both ways (:52-55)
- S2 aklr/core/validation/stages/_bench/_evaluation/_request.py:139 a `dot_product` output adds the step `large_magnitude/0` (:139-140), its inputs from `magnified_inputs` (:221-222): every finite floating input times one exact 2^e, the largest keeping the inputs and max|a|·max|b|·K under a quarter of the output dtype's largest value (`aklr/core/contract/semantics/_workloads.py:98-135`); a `bitwise` schema gets one more `determinism` step on `numeric/0` (:141-142); `find_nondeterminism` (`_comparison.py:31`) fails it on any differing element
- S3 aklr/plugins/sglang/models/qwen3_5/contract.py:121 `conv_state`: `ToleranceBound(absolute_tolerance=0, relative_tolerance=0)`; `values`: `ToleranceBound(absolute_tolerance=2, relative_tolerance=0.125)` (:179); `scales`: relative tolerance 0.02 (:185); decode `delta_state`: `ToleranceBound(absolute_tolerance=1e-2, relative_tolerance=1e-2)` (:229); gated attention: aiter's fp8-cache decode emulated at 21.8-23.1 dB (values) and 26.7-34.1 dB (scales) at the schemas' Points, "The bounds sit some 6-7 dB under that", kernels wrong on purpose under 9 and 10 dB, "at most 16.4 and 16.8 dB on the small one" (:401-406); `SqnrBound(min_sqnr_db=15.0)` (:412), 20.0 for the scales (:418)
- S4 aklr/plugins/sglang/models/deepseek_v2/contract.py:167 "the experts' MXFP4 activations alone come to about 12 dB", over `SqnrBound(min_sqnr_db=10.0)` (:168); MiniMax-M2: "about 13 dB on the bench's inputs", `_output(10.0)` (`aklr/plugins/sglang/models/minimax_m2/contract.py:280-281`); MiniMax-M2 fp8: "the fp8 activations come to about 36 dB on the bench's inputs", `_output(30.0)` (same file, :247-248)
- S5 aklr/ops/gemm/contract.py:186 FIXME: main's tolerance and no determinism check; "gfx950 fp8 MFMA accumulates below _dot_k's IEEE fp32 model" (:187-188); scales 2^U[-3, 0], "On 2**U[-1, 1] a k1024 fp8 MFMA tail element breaks main's tolerance" (:189-192); `gemm@1.0`'s `out` is bounded by `_dot_k`: `DotProductBound(length="k", accumulate_dtype=DType.FLOAT32, max_split_k=16)` (:91-96, :105, :134)
- S6 aklr/ops/README.md:122 "A rounding every correct kernel must perform bit for bit is part of the op and is written in the reference"; an integer output must equal the reference (:110)
- S7 aklr/plugins/sglang/models/qwen3_5/reference.py:115 decode rounding points: "each convolution tap's product to BF16, summed in FP32 oldest tap first"; group scales from the bf16 output, `clamp_min(1e-10) / 448.0`, codes clamped to ±448, then E4M3FN (:94-95); prefill: tap products and bias summed in fp32 unrounded (:72), the SiLU output rounded to bf16 (:73), q and k L2-normalized and rounded to bf16 (:77-78)
- S8 aklr/plugins/sglang/models/qwen2_moe/reference.py:33 router logits "exact (float64), rounded to bf16"; top-k by a stable descending sort, "a tie going to the lower id" (:41, :46); the DeepSeek-V2 reference rounds its router's logits too (`aklr/plugins/sglang/models/deepseek_v2/reference.py:4-6`)
- S9 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/fc261961f0ad572d.json#kernel.py:131 each conv tap's product rounded to bf16 and added in fp32, oldest tap first (:131-143)
- S10 triton-lang/triton@c01b677:python/triton/language/semantic.py:1243 `store` casts the value to the pointer's element type; float downcasts default to RTNE (:824-826); `maximum` is `maxnumf` unless `propagate_nan=ALL` (:383-390), and `gl.max` reduces with it (`python/triton/language/standard.py:177-191`); atomics default to `ACQUIRE_RELEASE` (:975) at `GPU` scope (:990); a Python scalar of no higher kind than the tensor takes the tensor's type, fp32 only for `/` and `%` on fp16 and bf16 (:73-79); bf16 with bf16 gives bf16 except for `/` and `%` (:98-102); `sum` widens only integers narrower than 32 bits (`standard.py:270-281`, applied at :289-292)
- S11 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/3b99d19659746a36.json#kernel.py:746 gated RMSNorm, bf16 store, amax, scale and clamped codes (:746-758), with `quant_max`, `1.0 / quant_max`, `-quant_max` passed at :920-922; layout `BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])` (:728), launched with `num_warps=1` (:926); inputs loaded `.to(gl.float32)` (:738-744); conv SiLU `gl.div_rn(conv, 1.0 + libdevice.exp(-conv))` rounded to bf16 (:163); padding tokens get decay 1 and beta 0 (:289-290); 7a69 in the same directory: "Products of decays are formed directly, without dividing tiny prefixes." (`7a69337f81190952.json#kernel.py:257`)
- S12 aklr/plugins/sglang/models/qwen2_moe/contract.py:157 kernels rounding at other points "flips some fp8 codes: about 33 to 38 dB on the bench's inputs"; `_moe_output(25.0)` (:229, :256); router `topk_weights`: "a logit summed in float32 may round to the next bf16", `ToleranceBound(absolute_tolerance=1e-6, relative_tolerance=2**-4)` (:181-183); `topk_ids` int32 with no bound (:185-189)
- S13 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/06780bcbbb3e3c70.json#kernel.py:142 "FP8 scales must consume the rounded BF16 output, not the FP32 precursor"; scale `max(amax, 1.0e-10) / fp8_max` and codes `values / quant_scale` (:144-148)
- S14 aklr/plugins/sglang/models/minimax_m2/reference.py:4 activations unquantized; "the scale rounding of their blocks is the kernel's (aiter rounds a block's maximum to nearest even, others up), its error left to the bound" (:6-8); the fp8 group scale floored at `torch.finfo(torch.float32).tiny` (:83-84; Qwen2-MoE the same, `aklr/plugins/sglang/models/qwen2_moe/reference.py:58-59`)
- S15 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:214 "The 128-element reduction stays wave-local and needs no LDS exchange."; `BlockedLayout([4], [64], [2], [0])` (:215); 20f6 in the same directory: `group_scale = maximum * (1.0 / 448.0)`, one `1.0 / group_scale` per group, inline `v_med3_f32` clamp (`20f63094b85c00a2.json#kernel.py:23-33`)
- S16 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json#kernel.py:137 `_quantize_fp8_groups` with `NARROW_PEAK`: `bits & 32767` reduced as integers and bitcast back (:138-141); `NARROW_PEAK` is `M == 64` (:207, :368); Qwen3-Next `_bf16_absmax`: "Nonnegative BF16 bit patterns have the same ordering as their values", patterns above 0x7f80 mapped to 0 (`aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/371905e46b19ee36.json#kernel.py:194-199`)
- S17 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/8a5a03b603ec1b70.json#kernel.py:114 `_quantize_fp8_group_2`: residual-corrected scale (:115-118), `v_rcp_f32` with one Newton step (:120-129), codes `values * inverse` (:131); router weight split into three bf16 terms on three bf16 MFMAs (:909-915), `high + (middle + low)` (:916); e75e in the same directory: `sb = gl.where(col < BN, sg, su)`, then `gl.fma(p0 * sa, sb[None, :], acc)`, router logits `gl.sum(rx * rw, 0)` (`e75e553b34178760.json#kernel.py:500-502`, :228)
- S18 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:110 `scale = max(amax, 1e-10) * (1.0 / 448.0)`, then `gl.div_rn(values, scale)` per value (:111)
- S19 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:44 "Per-128-group E4M3FN quantization; preserve the rounded reciprocal": `gl.div_rn(values, scale)` per value (:48); LDS `atomic_scatter_add` reservations ahead of "the ordered eleven-route reduction" (:108-112); "Preserve BF16 SiLU and BF16 product before dynamic FP8 quantization.": `silu` rounded to bf16, then `silu * up` (:233-235); 5c35 in the same directory: "Finite E4M3FN operands are represented exactly in BF16 for native 16x16x32" MFMA (`5c35215626bc1c87.json#kernel.py:4`)
- S20 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/ddb8de6066f38b7e.json#kernel.py:30 `_quantize`: "The scale floor keeps finite reciprocals normal" (:40-41); one refined `v_rcp_f32` per group and a residual correction per value (:43-52), "without a division per element" (:42)
- S21 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/b31bfeccf6dcfd7c.json#kernel.py:272 `_quantization_quotient`: a shared refined reciprocal, each quotient corrected (:279-286); `_round_bf16_ieee` maps |x| ≥ 3.396e38 to Inf (:306-313); "also saturates BF16 conversions": clamp, then inline `v_cvt_scalef32_pk_fp8_f32` (:324-338); 38a7 in the same directory: `_ieee_pid` clears FP16_OVFL at entry, `_fp8_scale` and `_store_fp8` set it in their asm, `is_pure=False` (`38a742cfd908c997.json#kernel.py:34-61`)
- S22 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:451 `ClampFOp` → `v_med3`, a NaN select only for `PropagateNan::ALL` (:453-458); `adjustModeRegister` sets MODE bit 23 (fp16_ovfl) at the entry of a function marked by an fp8 cast (:487-496); `arith.divf` → LLVM `fdiv` (:74), `div_rn` too (:511); `gl.exp` → `llvm.exp2.f32(x · log2e)` (:277-283); `gl.exp2` → `llvm.amdgcn.exp2.f32` under FTZ (:312-313); `gl.rsqrt` → `llvm.amdgcn.rsq.f32` (:346); `gl.sqrt` → `llvm.amdgcn.sqrt.f32` without refinement (:387-417)
- S23 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_11_vop3a_vop3b_instructions.md:2717 V_MED3_F32 returns `v_min3_f32` of its operands when one is NaN (:2717-2718), V_MIN3_F32 being `v_min_f32(v_min_f32(S0, S1), S2)` (:2655); V_CVT_SCALEF32_PK_FP4_F32 converts two values, round to nearest even, into one byte chosen by OPSEL[3:2], "Other destination bits are preserved" (:3915-3927)
- S24 triton-lang/triton@c01b677:python/triton/language/core.py:2956 `clamp(x, min, max, propagate_nan=PropagateNan.NONE)`; `maximum` (:2934) and `minimum` (:2912) take the same flag; `PropagateNan` (:23) is exported by `triton.language` (`python/triton/language/__init__.py:32`), not by `python/triton/experimental/gluon/language/__init__.py`
- S25 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2127 fp32 → E4M3FN on CDNA4 via `scalePk4DowncastToFp8<CvtScaleF32PkFp8F32Op>`, scale 1.0 (:561), RTNE only (:2111); an fp8 cast marks its function with `SetFP8ClampingAttr` (:2473-2481); fp32 → bf16 RTNE in packed pairs on CDNA4 (:106), RTZ a 16-bit shift (:73-77)
- S26 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:795 FP16_OVFL rule for F32 → FP8/BF8 (table :798-808); F32 MFMA results "clamped to +/- MAX" when the bit is set (:115); Table 28: F32 16x16x4 takes 32 cycles (:81), BF16 16x16x32 takes 16 (:101); Table 30: FP4 E2M1 has no INF or NaN (:712); conversion to F4/F6 ignores FP16_OVFL (:772) and saturates (:780)
- S27 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch03_kernel_state.md:105 MODE.FP16_OVFL: "an overflowed FP16 result is clamped to +/- MAX_FP16"; MODE.IEEE (:102)
- S28 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/0101652e6ad7286d.json#kernel.py:231 `_quantize_groups_2`: ceil exponent (:233-236), exact prescale (:238), NaN/zero guard (:239-241), packed convert (:243-250); docstring "the block scale rounded up from its maximum / 6" (:7); `_quantize_hidden_tile` builds `x` in `BlockedLayout([1, 2], [4, 16], [NW, 1], [1, 0])` (:266, :272); counters zeroed by program 0 of an earlier launch (:369-371); `ticket = gl.atomic_add(Counts + experts, 1, valid, sem="relaxed")` (:449-451); bb66 in the same directory: `_mx_inverse`, `gl.exp2(-clipped) * gl.exp2(clipped - ef)` at some M, `gl.div_rn(1.0, gl.exp2(...))` otherwise (`bb663e5b0cf55a10.json#kernel.py:372-378`)
- S29 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/5751284b6965eabb.json#kernel.py:26 `v_mov_b32 $0, 0` ahead of the fp4 convert, constraints `=&v,v,v` (:27-28), `dtype=gl.uint32`, then `.to(gl.uint8)` (:30, :33); converted to `BlockedLayout([1, 2], [4, 16], [gl.num_warps, 1], [1, 0])` before the split (:23-25); `score = probability + gl.load(Bias + e).to(gl.float32)` (:137)
- S30 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:43 `_quantize_groups`: ceil (:47) and even (:48-51) exponents, `gl.where(shared, even_exponent, exponent)` (:52), each value divided, `gl.abs(x / scale[:, None])` (:55), quantized once per rule (:107-108); NaN → 6.0 before the convert (:24)
- S32 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:383 `__HIP_FTZ = True`, not a kernel option (:376-382); `enable_fp_fusion` defaults to `knobs.language.default_fp_fusion` (:179-180; `TRITON_DEFAULT_FP_FUSION`, true, `python/triton/knobs.py:497`) and goes to `llvm.optimize_module` (:505); `allow_flush_denorm` sets `denormal-fp-math-f32` to `preserve-sign` (:473-474); `llvm_fn_attrs` are added to the kernel (:479-481)
- S33 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_08_vop1_instructions.md:482 V_EXP_F32: "1ULP accuracy, denormals are flushed."; V_LOG_F32 the same (:504), V_RSQ_F32 (:584) and V_SQRT_F32 (:638) too; V_RCP_F32: "Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA operations. Denormals are flushed." (:531)
- S34 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json#kernel.py:499 `_encode_projection_record`, an exactness-checked int16 transport of fp32 projection partials: biased exponent clamped to [1, 253] (:502), power of two and reciprocal built from bits (:503-504), the code kept only where the round trip is exact (:505-509); the MX quantizer `_encode_groups` takes `gl.exp2(exponent)` and `gl.div_rn(1.0, scale)` (:116-117)
- S35 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1d277d76892ee692.json#kernel.py:61 `_fp4_code`: midpoint compares alternating `>` and `>=` (:63-66), sign bit added (:67); b94f in the same directory (`b94f5777fd5d2d88.json#kernel.py`): the gate-up accumulator rounded to bf16 (:237), then the gated product `(gate * _sigmoid(gate) * up).to(gl.bfloat16)` (:240), `_sigmoid(x)` being `1.0 / (1.0 + gl.exp(-x))` (:26-27); that product converted from the MFMA accumulator's layout to `BlockedLayout([1, 32], [64, 1], [4, 1], [0, 1])` at `BN == 256`, to `[1, 8], [16, 4]` otherwise, then quantized (:233-243); down outputs rounded per route, `acc.to(gl.bfloat16)` (:738); `_quantize_hidden_5` loads its groups in `BlockedLayout([1, 32], [64, 1], [4, 1], [0, 1])` and quantizes them unconverted (:340-348), 256 groups per program (`quant_groups`, :1053)
- S36 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:169 masked scale words load as 2139062143 (0x7F7F7F7F), masked scale bytes as 127 (:181-182); `take = (av > bv) | (av == bv) & (ai < bi)` (:94); router on the fp32 MFMA `instr_shape=[INSTR_M, INSTR_M, 4 if INSTR_M == 16 else 2]` (:1024)
- S37 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/ae4fa8a440b98f09.json#kernel.py:46 `_group_dot` (:46-51) and `_accumulate_panel` (:55-73); `acc.to(gl.bfloat16)` before `gl.convert_layout` (:84)
- S38 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:68 both scales `None` → `uint8` 0x7F (:68-69), built as constant tensors (:99); `dot_scaled(..., fast_math=False,..., out_dtype=ttgl.float32)` (:105-106)
- S39 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/0e15a0289735acb0.json#kernel.py:447 `acc = gl.fma(partial * sx, sw, acc)`; at tiny M two K groups per MFMA in alternate rows, `group = product * 2 + rows % 2` (:615), the accumulator chosen by the MFMA's parity (:646-649), row pairs and accumulators summed at the end (:650-654); 3480 in the same directory: `acc = gl.fma(partial, xs[:, None] * ws, acc)` (`3480e97a5fcda6f6.json#kernel.py:297`)
- S40 aklr/ops/gemm/reference.py:100 `out = (a * a_scale[:, :k]) @ (b * b_scale[:n, :k]).mT`
- S41 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1025_n_v_heads16_key_dim128_value_dim128_conv_dim3072/19b20285ceb08b14.json#kernel.py:584 state split hi/mid/lo by RTZ at 8 warps, hi/mid by RTNE otherwise (:584-592); MFMAs smallest term first (:595-598); `_qk_products`: K·Kᵀ and Q·Kᵀ on the bf16 `[16, 16, 32]` MFMA (:603-619)
- S42 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp:94 `bf16x3` splits both operands (:94-95) and sums mid·hi + hi·mid + hi·hi (:118-126); the pass runs in plain Triton's `make_ttgir` (`third_party/amd/backend/compiler.py:270`), not in `gluon_to_ttgir` (:327-348)
- S43 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/5f2049852cd84a76.json#kernel.py:82 "Route from exactly the BF16 values exposed in the logits output.": the split partials summed, then `.to(gl.bfloat16).to(gl.float32)` (:83); "A NaN tail becomes the removed head's -inf placeholder" (:107), `gl.maximum(..., -float("inf"))` (:110)
- S44 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/7d7e54597a7e11bd.json#kernel.py:22 `_round_to_bf16`: fp32 midpoints of a float64 value nudged toward it before the bf16 rounding (:22-30); float64 router dot (:94-97); `_routing_keys`: ±0 to one key (:65), `511 - experts` in the low bits (:67)
- S45 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json#kernel.py:73 router dot on the fp64 MFMA, `instr_shape=[16, 16, 4]`, `element_bitwidth=64`, float64 operands (:73-88), partials summed and rounded by `_round_bf16` (:189, :18-28); `_split_fp8_integer` (:197): "Represent a finite E4M3 value exactly as whole + fraction / 512", used in `_w13_exact_tile` (:310): "Each 64-term integer-valued MFMA is exact in FP32" (:354-376)
- S46 aklr/plugins/sglang/layers/dsa_backend/contract.py:100 sglang's kernel "rounds the softmax weights to fp8, comes to about 33 dB"; `SqnrBound(min_sqnr_db=20.0)` (:102)
- S47 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json#attention.py:231 `prob_hi`, `prob_lo` (:231-232) in the spare rows of a 16-row MFMA, folded after (:256-260); "Inactive splits publish complete neutral records on every replay." (:255); 3d0f in the same directory converts P and V to bf16 when V is bf16, to fp16 otherwise (`3d0f64ffff6790c8.json#attention.py:148-153`), its `value_cache` `float8_e4m3fn`
- S48 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:483 `_probability_operands`: bf16 high and residual P stacked as MFMA rows (:486-492); P·V on `[32, 32, 16]` when `packed_pv` and `head_block == 16` (:72); at 8 rows `head_block` is 16 (:319) and `packed_pv` on (:290)
- S49 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1024_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1fd6adb2b91cf5c2.json#kernel.py:19 "Only output reconstruction uses compensated BF16 products: chunk solves, the persistent recurrence, and mutable state remain FP32."; causal coefficients and state snapshots stored as bf16 high + low (:662-663, :739-740), multiplied high, then low (:1405-1409)
- S50 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:293 `_inverse_unit_lower_16`: `solved = gl.fma(value, diagonal, solved)` (:288-293); c53b in the same directory: `solved += gl.where(j[None, :] % 4 < pivot, column * diagonal_row, 0.0)` (`c53b6d874d511c02.json#kernel.py:326-330`)
- S51 triton-lang/triton@c01b677:third_party/amd/language/hip/libdevice.py:79 `libdevice.exp` → `__ocml_exp_f32` (:82); `fast_dividef` → `__triton_hip_fast_fdividef` (:102-104)
- S52 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/4f257a3e566b20d1.json#kernel.py:53 "Native exp2 is sufficient inside 1 + exp(x): any flushed subnormal is already too small to change that sum. Retain exp for decay." (:53-54), the decay by `gl.exp` (:56); 409c in the same directory: "Do not use this shortcut for the recurrent-state decay itself" (`409cb22b39b3fa00.json#kernel.py:48-49`); 2991 in the same directory takes `_exp`, `gl.exp2(x * 1.4426950408889634)`, for the decay (`299133886d479017.json#kernel.py:39-40`, :123)
- S53 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:49 `sigmoid_`: "divide", "native", "rescaled" reciprocals (:52-77); `exp_`: `gl.exp2(value * _LOG2E)` as the native exponential (:37-38), the half-argument `v_exp_f32` asm squared, which "restores gradual underflow" (:30-31, :40-45)
- S54 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/0c0b7dfe26861b02.json#kernel.py:91 `NATIVE_LOG`: inline `v_log_f32` of `1.0 + _exp(arg)`, times ln 2, for the softplus (:92-96); the template: "The logarithm's argument is >= 1, so no subnormal rescaling is required" (`f44007374038041f.json#gdn_primitives.py:113-114` in the same directory)
- S55 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/96e51c1f85a71b7e.json#kernel.py:846 `enable_fp_fusion=m != 16`
- S56 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:303 routing launch with `llvm_fn_attrs=[["amdgpu-ieee", "false"], ["amdgpu-sched-strategy", "max-ilp"]]`
- S57 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_07_vop2_instructions.md:185 V_MAX_F32: a signaling NaN is quieted only in IEEE mode; otherwise a NaN operand returns the other (:189-192); V_MIN_F32 the same (:158-165)
- S58 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:12 six fp32 partials summed "in a fixed order before rounding to bf16" (:12-15); `if m <= 16:` takes the unsplit path (:100), the rest `split_k = 6` (:129); `_reduce_kernel` (:86-93)
- S59 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3b667617400708bd.json#kernel.py:171 route ticket (:171); MFMA shape by tile height TM (:271); TM from the block's live rows (:398-413); lowest index among the maxima (:164)
- S60 aklr/plugins/sglang/models/deepseek_v2/baseline.py:20 "sglang's fp32 bias reaches top-k in bf16" (:21)
- S61 triton-lang/triton@c01b677:third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:454 `BufferStoreOp`: "value element type matches the pointed type of ptr"
- S62 aklr/core/contract/semantics/_dtypes.py:20 `DType` has `FLOAT8_E4M3FN` and `FLOAT8_E5M2`, no FNUZ type (:20-34)
- S63 aklr/ops/quant/contract.py:17 `per_group_quant_fp8@1.0`: `quantized` E4M3FN and `scales` float32, each under `OutputRoundingBound` (:38-49); `per_group_quant_mxfp8@1.0`: the same codes, `scales` uint8 e8m0 bytes with no error bound (:54-73); `aklr/ops/quant/reference.py`: fp8 amax floored at fl32(1e-10), the scale `maximum * divisor` rounded to float32 with `divisor` fl32(1/448), reciprocal and scaled activation rounded to float32, clamped to ±448, then E4M3FN (:19-26), "Combining the scale and reciprocal into one division changes quantization ties." (:11-12); mxfp8 byte: the float32 product's biased exponent plus `(raw & 0x7FFFFF) != 0` (:41-44), codes `grouped / scale` clamped to ±448, then E4M3FN (:45-47)
- S31 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:1 F8F6F4 execution and scale encoding: E8M0 byte 0xFF is NaN, finite scale exponents span −127…127.
