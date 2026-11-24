---
title: "Triton/Gluon numerics on gfx950: the contract, fp8 and MXFP4 quantization, block-scaled accumulation, transcendentals, determinism"
kind: topic
ops: [gemm, moe, attention, gdn, norm]
dtypes: [fp8, mxfp4, bf16, fp32]
regimes: [all]
gens: [gfx950]
triton: 3.8.0
---

# Numerics of a Triton or Gluon kernel on gfx950

What a kernel must compute bit for bit, what it may approximate, and how the library (154 Triton and Gluon kernels for gfx950: [kernels/kernel_library.md](kernels/kernel_library.md) §1) does
both on gfx950 (MI355X) with Triton 3.8.0. A count ("52 kernels") counts kernels among the library's 154, in 33 families of one op and one set of model dimensions each; after its growth the library holds 280, the families added listed in `kernels/kernel_library.md` §2. A kernel is named by its op, its model and the sizes it runs at ("the Qwen2-MoE router kernel for 256 tokens"), a MoE kernel on a known rank by its layer's label of [moe.md](moe.md) §1: DS256 and DS512 are DeepSeek-V2 MXFP4 on a rank of that intermediate dimension, MM256-FP8, MM64-FP8, MM256-FP4 and MM64-FP4 MiniMax-M2 on a rank holding that many of its experts, with FP8 or MXFP4 weights. "Compiled for gfx950": Triton 3.8.0 wheel, no GPU.

## Route here when

- A kernel's output, or a GEMM input it produces, is fp8 E4M3FN with fp32 scales per 128, or MX (e2m1 or E4M3FN values,
  an e8m0 scale per 32), in an epilogue or as a standalone quantization op (§2, §3).
- You must decide which divisions, exponentials and roundings have to be exact and which may be fast.
- You need fp32 accuracy from bf16 or fp8 MFMAs: block-scaled GEMMs, router logits, P·V, GDN state products.
- A kernel fails its accuracy check (a NaN, a code off by one, an SQNR shortfall) or the repeat run of a `bitwise` op: §10 maps what the failure looks like to a cause.
- You weigh `enable_fp_fusion`, `allow_flush_denorm`, `amdgpu-ieee`, atomics, asm `v_rcp_f32`/`v_exp_f32`/`v_log_f32`.

Elsewhere: MFMA and scale layouts [gluon/mfma.md](gluon/mfma.md); inline asm [gluon/wave_ops.md](gluon/wave_ops.md);
compiled code [debugging.md](debugging.md) §3; ops [gemm.md](gemm.md), [moe.md](moe.md), [attention.md](attention.md),
[gdn.md](gdn.md); correctness gates [../gemm/correctness.md](../gemm/correctness.md).

## 1. Read the contract first

The contract of an op is what its caller accepts as a correct output. Three things define "correct":

| Where | What it fixes |
|---|---|
| the determinism level | how runs must agree, `bitwise` or `bounded`; under a bound no element may be past it, with no allowance for a fraction of them |
| each output's error bound | the error budget: a tolerance or an SQNR floor, most often one of the common bounds below; an integer output (the Qwen2-MoE top-k ids, the e8m0 bytes of an MXFP8 group quantization) has a tolerance of 0/0 and must equal the reference: two swapped ids of one token failed the id comparison *(field, E17)* |
| the reference's docstring | rounding points to inspect; whether a kernel may skip or add one is decided by comparing its outputs, not by copying every rounding *(field, E17)* |

- **Both determinism levels compare runs, not kernels.** `bitwise` wants every run byte-identical to the first;
  `bounded` asks nothing across runs. Neither compares the output with the reference, the baseline or the kernel
  you replace: accuracy is the bound's job. The `bitwise` check measured here ran one input set once more, three
  runs of that set in all *(field, E9)*: a rare race can pass (§8).
- **What each bound leaves you**:
  - A tolerance is per element, B = atol + rtol·|y|, y the reference in the output dtype: at 0/0 (the Qwen3.5 GDN
    convolution state) the output must equal the reference's bit for bit, so move such a tensor without arithmetic.
  - An SQNR floor is one budget for the whole output, against the reference in the output dtype. The MXFP4 MoE
    outputs have a floor of 10 dB, where the experts' MXFP4 activations alone come to about 12 dB (DeepSeek-V2) and 13 dB
    (MiniMax-M2), measured 13.5 and 13.2 dB *(field, E2)*: noise powers add, so a further independent error must
    stay above about 12.6 or 12.8 dB (estimated). MiniMax-M2 fp8 has 30 dB, against about 36 dB for its fp8
    activations; gated attention 15 dB on the fp8 values and 20 dB on their scales, 6-7 dB under aiter's
    fp8-cache decode (emulated at 21.8–23.1 dB on the values and 26.7–34.1 dB on the scales at the kernels' sizes), kernels wrong on purpose under 9 and 10 dB (16.4 and 16.8 dB on the small test size).
  - The common bounds, Primus-Turbo's: a tolerance with atol = rtol by the output dtype (fp32 1e-4, bf16
    and fp16 1e-2, fp8 1e-1), used by RMS norm, add + RMS norm, the quantization ops' codes and fp32 scales and the a8w8
    GEMM on int8; SQNR floors of 45 dB (bf16 GEMM), 50 (fp16), 25 (fp8 e4m3), 20 (fp8 e5m2), 10 (fp4), 20 (fp8 attention),
    20 and 10 (MXFP8 and MXFP4 MoE). The bf16 small-M GEMMs are under 45 dB: accumulate in fp32 anyway, as a
    bf16 running sum fails (§10).
- **Non-finite values must match under every bound, an SQNR floor included**: finite wherever the reference (in the output
  dtype) is finite, equal to it (NaN matching NaN) wherever it is not. A stray NaN fails; so does a saturated
  finite value where the reference overflows to ±Inf (§2).
- **A bound is calibrated on the op's input distributions**, from which the correctness inputs are drawn. The a8w8 GEMM's block scales are drawn from 2^U[-3, 0]. On FlyDSL K 1024 fp8 block-scaled GEMMs (not Triton), 2^U[-1, 1] put one element over the bound at N 32768, M 2048 (seed 0), while 2^U[-3, 0] put none over at any of the 39 sizes × 24 input sets *(field, E18)*; a Triton test with 3 seeds of 4096 × 256 found none over in either distribution *(field, E8)*. Finite inputs: no guarantee for every input from the narrow distribution.

**Inspect the reference's rounding points, then check the outputs against the bound.** The reference writes the roundings a correct kernel performs, but neither "copy every one bit for bit" nor "anything else fits the budget" held. Dropping a reference rounding failed twice: a Qwen2-MoE router kernel (256 tokens) ranking its unrounded fp32 logit sums changed the ids of 97 of 256 tokens and failed the id comparison (3 runs, same input) *(field, E2)*, though its weights came closer to an unrounded float64 reference (relative L2 0.0041 → 2.8e-7); a Qwen3.5 GDN decode kernel (128 tokens) without its per-tap bf16 rounding came closer too (values' relative L2 0.0155 → 0.0018) and failed values and state *(field, E17)*. Dropping one passed three times: a Qwen3.5 GDN prefill kernel (16384 tokens) quantizing the fp32 precursor of its normalized output, a DS512 kernel (8 tokens) without its router-logit rounding (the error change unresolved), and a Qwen3.5 gated attention kernel (32 rows) keeping the attention in fp32 before the gate, whose scales' relative L2 rose from 0.0032334 to 0.0032690 on the checked input but fell on 62 of 64 other input sets (0.849 of the unchanged kernel's error, geometric mean) *(field, E24)*: one input set settles neither way, and passing is not non-inferiority. Adding a rounding can fail too: a bf16 round trip on each of that GDN prefill kernel's conv tap products, which its reference sums unrounded, failed once (scales peak 12.56 times the bound). One input set per size, two checked runs per arm *(field, E17)*. Reference round trips such as the GDN decode reference's per-tap cast are a starting point, not a pass/fail test.

**Compute in fp32: a 16-bit tensor is not widened.** A Python float next to a bf16 or fp16 tensor takes the tensor's
type, and bf16 with bf16 stays bf16 except in `/` and `%` (S10): compiled for gfx950, `x * 1.4426950408889634` on a bf16
`x` multiplies by 1.4453125 (log2 e rounded to bf16) and rounds the product to bf16, and `gl.sum` of a bf16 tensor
rounds every partial sum to bf16, since `sum` widens only integers (S10). Convert each 16-bit load with
`.to(gl.float32)` before any arithmetic, as §2's kernel does, or pass `dtype=gl.float32` to `gl.sum`.

Spend the rest on fewer instructions; compiled for gfx950, each shortcut has fewer: one reciprocal per group instead of
a division per value (317 against 642 in a 16 × 128 epilogue), a bare `v_rcp_f32` or `gl.exp2` (23 and 24 against 33
for `gl.div_rn` and `gl.exp` on 64 values), bf16 MFMAs (4 against 32 fp32 ones in a 32 × 64 × 32 `tl.dot`); when each
is accurate enough: §2, §4, §5, §6. Additional roundings can consume the error budget: one MM64-FP4 kernel rounds gate/up accumulators, `silu(gate) · up`, and per-route down outputs to bf16, unlike its fp32 reference. The Qwen2-MoE fp8 experts kernel for 64 tokens adds bf16 gate/up and SiLU roundings and rounds weighted route contributions before reduction; its reference already rounds the gated product to bf16 before the second linear's fp8 input quantization, as MiniMax-M2's fp8 reference does, and quantizing the second linear's input
from values rounded at other points flips some fp8 codes: about 33 to 38 dB on the test inputs, against its 25 dB bound.

## 2. The fused fp8 group-quant epilogue (groups of 128)

Of the library's 154 kernels, 52 output E4M3FN values with an fp32 scale per 128 values (gated attention 9, Qwen3.5 GDN decode 8 and prefill
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

(The host passes `quant_max` = 448, `min_quant` = −448 and `inv_quant_max` = `1.0 / 448`.)

- **Check quantization against the output's bound.** The references quantize the bf16-rounded output. On synthetic groups, quantizing the fp32 precursor instead put all 100000 scales outside a bound of two output roundings (2·u_out·|y|) and changed 282120 of 12.8M codes *(field, E10)*; under a Qwen3.5 GDN prefill kernel's own bounds at 16384 tokens the same change passed both checked runs (largest error 0.354, 0.64 and 0.348 of the bound for the normalized output, its fp8 values and their scales; one input set) *(field, E17)*. The output's bound decides.
- **Floor amax at 1e-10 before forming the scale**, as all 68 quantizing kernels do: an all-zero group then gets a
  finite, nonzero scale and zero codes (without the floor, code 0xFE: the clamp turns the NaN into −448 *(field, E10)*).
  The GDN and gated-attention references floor amax the same way; the MoE references floor the scale at float32's
  smallest normal instead, which leaves the same zero codes.
- **Keep each group of 128 inside one wave**, so its amax needs no LDS: no warps on the 128 axis of a (rows, 128) tile,
  like the excerpt's `BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])` at `num_warps=1`. Compiled for gfx950, its
  `gl.max(gl.abs(x), 1)` is in-thread and cross-lane steps only; `BlockedLayout([1, 8], [16, 4], [2, 2], [1, 0])` on a
  32 × 128 tile adds 256 bytes of LDS and an `s_barrier`. A 1-D `BlockedLayout([4], [64], [2], [0])` holds a 128-value
  group whole in each wave, the second a copy.
- **The amax of values already rounded to bf16** can be an integer max of their bit patterns with the sign cleared:
  non-negative bf16 patterns order like their values. An MM64-FP8 kernel does so at 64 tokens:

  ```python
  @gluon.jit
  def _max_bf16_magnitude_bits(a, b):
      return gl.maximum(a, b)

  @gluon.jit
  def _quantize_fp8_groups(x, DTYPE: gl.constexpr, NARROW_PEAK: gl.constexpr = False):
      if NARROW_PEAK:  # x holds bf16-rounded values: amax as an integer max of their bit patterns, sign cleared
          bits = (x.to(gl.bfloat16).to(gl.uint16, bitcast=True) & 32767).to(gl.uint16)
          peak_bits = gl.reduce(bits, 1, _max_bf16_magnitude_bits)
          peak = peak_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
      else:
          peak = gl.max(gl.abs(x), 1)
      peak = gl.maximum(peak, 1e-10)
      first = peak * (1.0 / 448.0)  # the scale as a rounded product,
      remainder = gl.fma(-first, 448.0, peak)  # corrected with its residual
      scale = gl.fma(remainder, 1.0 / 448.0, first)
      inverse = gl.inline_asm_elementwise(
          "v_rcp_f32 $0, $1", "=v,v", (scale,), dtype=gl.float32, is_pure=True, pack=1
      )
      error = gl.fma(-scale, inverse, 1.0)  # one Newton step on the reciprocal
      inverse = gl.fma(inverse, error, inverse)
      q = (x * inverse[:, None]).to(DTYPE)
      return (q, scale)
  ```

  A NaN pattern orders above Inf; map patterns above 0x7f80 to 0 where a NaN must not win, as the Qwen3-Next GDN
  decode kernels do.
- **Quantize an fp8 output in a normalization's last pass**: normalization, bf16 store, amax, scale and codes in one program. In a synthetic RMSNorm + quantization of 4096 and 131072 rows of 128, the fused kernel took 0.55 and 0.61 of the time of a norm kernel followed by a quantization kernel, with the same codes and scales *(field, E10)*. In the epilogue of a plain-Triton bf16 GEMM (N = K = 4096) it gained nothing: 0.99 of the time of the GEMM followed by the quantization kernel at M = 512 and 8192, 1.03 at M = 16 *(field, E22)*. Inside one MoE op a GEMM input can instead be quantized by its consumer as it loads it ([moe.md](moe.md) §8).

**Four ways to divide by the scale.** Most kernels form the scale as `amax · fl(1/448)`, a rounded product where the
reference divides; a few divide (`amax / 448`) or correct the product with a residual step (`first`, `remainder` and `scale` in the excerpt above). Then x/scale:

| Form | Kernels | Rounding of the code |
|---|---|---|
| a division per value: `gl.div_rn(x, s)` or `x / s` | gated attention, 7 of 9; Qwen2-MoE fp8 but the router+experts hidden state at 1–4 tokens; Qwen3-Next GDN decode: the kernels for 1 token and for 2 tokens, two kernels for 128 and 256 tokens, and the template (the Qwen3-Next GDN decode kernel written for all nine sizes: [gdn.md](gdn.md) §1) at 1, 128 and 256 tokens | correctly rounded |
| one reciprocal per group: `x * gl.div_rn(1.0, s)` or `x * (1.0 / s)` | Qwen3.5 GDN decode and prefill; Qwen3-Next prefill; the other 2 gated attention kernels; four MiniMax-M2 fp8 kernels, one of them for its router input only (these four form the scale as `gl.div_rn(amax, 448.0)`) | one more rounding: a code can move at a tie |
| `v_rcp_f32` refined, then `x * r` | MiniMax-M2 fp8, 6 of 9 carry this form: one Newton step in five (the excerpt above), one of them on its activation path only (the kernel that takes the second form for its router input); the sixth uses one Newton step followed by a residual correction with the original reciprocal | an extra rounding of the reciprocal or the product can move an fp8 code |
| `v_rcp_f32` with a Newton step, then each `q = x · r` corrected by `fma(fma(-s, q, x), r, q)` | Qwen2-MoE fp8 router+experts hidden state at 1–4 tokens (three kernels); Qwen3-Next GDN decode: the kernels for 4 and 8, for 32 and for 64 tokens, and the template at 2, 4, 8, 32 and 64 tokens (the 16-token kernel, and the template at 16, skip the Newton step); the 1-token and 2-token kernels carry the helpers unused | the quotient of an IEEE division on all 12.8M tested values near e4m3 midpoints, no proof for every input *(field, E10)*; the 64-token and 16-token kernels also execute `v_div_fixup_f32` after the correction |

All four forms pass their bounds at the sizes they run at *(field, E1)*. The Qwen3.5 GDN decode tolerances (atol 2 and rtol 0.125 on the fp8 codes, rtol 0.02 on the scales) admit a code one e4m3 step off and a scale one ulp off (65536 moved codes and 512 moved scales passed *(field, E2)*). Default to the second form: one IEEE division (11 VALU, §6) per group of 128 instead of one per value, 236 against 550 VALU on 16 × 128 *(field, E10)*, a saving no timing isolates. The fourth form keeps the fp8 rounding boundary without a division per value: on 12.8M values within 2 ulps of e4m3 midpoints it matched IEEE division in every code and quotient *(field, E10)*, and on exhaustive bf16 inputs and random fp32 groups it matched the code of every nonzero x, but x = −0 gave +0 (code 0x00, not 0x80) unless signed zeros are kept with `gl.where(x == 0.0, x, q)` *(field, E23)*. That is no proof for every input; where a correctly rounded fp32 quotient is required, use `gl.div_rn(x, s)`. The fourth form: `r` from the asm `"v_rcp_f32 $0, $1"` (`"=v,v"`, `gl.float32`), one Newton step `r = gl.fma(r, gl.fma(-s, r, 1.0), r)`, `q = x * r`, then `q = gl.fma(gl.fma(-q, s, x), r, q)` (compiled: no `v_div_fixup_f32`). The scale floor keeps finite reciprocals normal only together with an upper bound on the scale: for finite bf16 or fp32 amax the floored amax/448 scales lie in about [2.23e-13, 7.60e35] (estimated), whose reciprocals are normal, and the floored `v_rcp_f32` reciprocals for amax from 0 to 1e30 were all normal *(field, E8)*. Supplied scales and non-finite intermediates need their own handling.

**Where the quantization is the op, match its reference's form.** The standalone fp8 group quantization these kernels were checked against (E4M3FN codes
and an fp32 scale per group of 128, the common tolerances on both, §1) has the second form exactly as its reference, `fl32(max(amax, 1e-10) · fl32(1/448))`, then
`fl32(1 / scale)` and `fl32(x · r)`: combining the scale and reciprocal into one division changes quantization ties,
a code by one step, rarely on bf16 inputs *(field, E10)*. Against such a reference reproduce the form, as the excerpt at the head of this section does: the correctly
rounded `amax / 448` was one ulp below that product for 71 of the 128 bf16 significands and above it for none in each of the 25 tested normal-scale binades: inside the
bound, but all one way, which failed a bias test (the mean error over its rms) *(field, E10)*. A code one step off is over a per-element tolerance where the reference's
code is 0 or subnormal: there the bound is under a step.

**Saturation and clamping.**
- Clamp to ±448 before converting. 44 kernels call `gl.clamp`, which is one `v_med3_f32` (S22); the 8 Qwen3.5 GDN decode
  kernels write the `v_med3_f32` as inline asm. The 16 MoE quantizers skip the clamp: with scale = amax/448 and no
  clamp, x/scale exceeded 448 by at most 448·2^-22, and Triton's fp8 cast stored 448 there *(field, E10)*.
- With a NaN operand `v_med3_f32` returns `v_min3_f32` of its operands, `min(min(x, lo), hi)` in Triton's order (S23;
  compiled for gfx950): per the ISA a quiet NaN, which arithmetic produces, clamps to −448, and a signaling NaN loaded
  from memory to +448, quieted by the first min in IEEE mode, the default (S57); `torch.clamp` keeps the NaN. The 68 quantizers are specified for finite inputs and finite scalar arguments only, which does not keep every intermediate finite. Where a NaN in the clamp's `x` operand (with finite bounds), or either min/max operand, must reach the output, pass `propagate_nan=tl.PropagateNan.ALL` (`import triton.language as tl`; Gluon does not re-export it) to `gl.clamp`, `gl.maximum` or `gl.minimum` (S24): compiled for gfx950, the clamp gains `v_cmp_u_f32` and `v_cndmask_b32`, `gl.maximum` becomes one `v_maximum3_f32` *(field, E13)*.

**The convert.** For fp32 `x`, `x.to(gl.float8e4nv)`, or `gl.store` into an E4M3FN pointer (which casts to the pointee
type, S10), lowers on CDNA4 to `v_cvt_scalef32_pk_fp8_f32 …, 1.0`: scale 2^0, pairs of values, round to nearest even
only (S25). The same lowering makes Triton set MODE.FP16_OVFL at kernel entry,
`s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1` (S22); a compile for gfx950 shows both. With FP16_OVFL set, ±Inf and
values that round above 448 convert to ±448 and NaN stays NaN; with it clear they become NaN (S26; [gluon/wave_ops.md](gluon/wave_ops.md) §7).
- Without an explicit MODE write, the bit stays set for the whole kernel; inline asm can clear it and set it again later, the pair of writes the Qwen3-Next GDN decode template makes at 2, 4, 8 and 32 tokens (the second way below). Per the ISA it also clamps an overflowing FP16 result to ±MAX_FP16 and an
  overflowing F32 MFMA result to ±MAX instead of ±Inf (S27, S26): measured for the bf16 and scaled-e4m3 MFMAs *(field, E10)*,
  while the fp32-input `v_mfma_f32_16x16x4_f32` still returned ±Inf ([gluon/wave_ops.md](gluon/wave_ops.md) §7): a kernel with
  any Triton fp8 cast can return a finite maximum where the reference overflows, which the non-finite rule rejects (§1).
- An asm fp8 convert in a kernel without a Triton fp8 cast or an explicit MODE write starts with FP16_OVFL clear: overflow and ±Inf become NaN. Inline asm that sets bit 23 can instead enable saturation.
- The Qwen3-Next GDN decode kernels (which pass their bounds *(field, E1)*) allow for the bit also saturating bf16 conversions, as measured: ±0x7F7F instead of ±Inf *(field, E10)*. They keep IEEE overflow three ways ([gdn.md](gdn.md) §6; `s_setreg`: [gluon/wave_ops.md](gluon/wave_ops.md) §7), which matters only where a bf16 convert, an FP16 result or an F32 MFMA result can overflow before the fp8 convert. First, keep Triton's fp8 store, so that the bit is set, and make a finite bf16 overflow a signed infinity by hand: the 1-token kernel, and the template at 1 token. Second, use no Triton fp8 cast, so that nothing sets the bit at entry: clamp to ±448, convert pairs of values with inline `"v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, 1.0"` (`dtype=gl.uint32`), store them through a uint16 pointer, and switch the bit in asm that produces a value (`is_pure=False`), cleared on the program id at entry and set on the fp8 scale after the last bf16 rounding: the template at 2, 4, 8 and 32 tokens. Third, the same convert with the bit set in the convert's own asm string, `"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\nv_cvt_scalef32_pk_fp8_f32 $0, $1, $2, 1.0"` (`is_pure=False`): the 32-token kernel. The first two, in the template:

  ```python
  # first way: the bf16 rounding where Triton's fp8 store has set the bit (OVFL_FIXUP is on at 1 token)
  @gluon.jit
  def round_bf16_ieee(value, OVFL_FIXUP: gl.constexpr):
      # ...
      if not OVFL_FIXUP:
          return value.to(gl.bfloat16)
      bits = value.to(gl.uint32, bitcast=True)
      infinity = ((bits & 0x80000000) | 0x7F800000).to(gl.float32, bitcast=True)
      value = gl.where(gl.abs(value) >= 3.39617752923046e38, infinity, value)
      return value.to(gl.bfloat16)
  # ...
  # second way: the kernel's entry runs `pid = ieee_pid(pid)`, which clears the bit
  @gluon.jit
  def ieee_pid(pid):
      return gl.inline_asm_elementwise(
          "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\ns_mov_b32 $0, $1",
          "=s,s", (pid,), dtype=gl.uint32, is_pure=False, pack=1,
      )
  # after the last bf16 rounding `quant_scale = fp8_mode_scale(quant_scale)` sets it
  @gluon.jit
  def fp8_mode_scale(scale):
      return gl.inline_asm_elementwise(
          "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\nv_mov_b32 $0, $1",
          "=v,v", (scale,), dtype=gl.float32, is_pure=False, pack=1,
      )
  ```

## 3. MXFP4 activation quantization

Of the library's 154 kernels, all 42 MXFP4 MoE kernels (DeepSeek-V2/GLM 19, MiniMax-M2 23) quantize both expert-GEMM inputs to MXFP4 in registers:
e2m1 values, one e8m0 scale per 32. The MoE references leave activations unquantized; this rounding is the kernel's,
its error left to the 10 dB bound; a quantization op's is not (below). The recipe:

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

`x` is fp32 of shape (groups, 32), one MX block per row; the codes and scale bytes are MXFP4, ceil rule (below),
RNE e2m1 *(field, E11)*. `gl.split` needs each pair in one thread: at least 2 consecutive elements per thread along the
32, else compiling fails: "Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension". Load
the input in a pair-holding layout and only reshape it, `gl.reshape(value, (groups, 32))`. Compiled for gfx950, SwiGLU + MXFP4 on 256 bf16 columns at NW = 1, 2, 4, 8 used no LDS or `s_barrier` with `BlockedLayout([1, 8], [16, 4], [NW, 1], [1, 0])` at 16 to 64·NW rows, or with `BlockedLayout([1, 32], [64, 1], [NW, 1], [1, 0])` at exactly 64·NW rows (taller tiles not compiled); converting the first to `[1, 2], [4, 16]`, as the packer of a DS256 kernel does, took 8–256 KiB of LDS and, at NW ≥ 2, an `s_barrier`, while loading straight into that pair-holding layout took none *(field, E14)*. At NW = 4 the second layout on 64 × 256 holds every row in all four waves (two zero warp bases) and spills 177 VGPRs for codes, 179 with scales; the copies are not the cause, since 128 and 256 rows spill 177–179 as well, with no copy at 256: each lane holds 256 values. A 256 × 64 tile at NW = 4 did not spill; at 64·NW × 256 both layouts spilled, 164–193 VGPRs at NW ≤ 4 and 570–805 at NW = 8 *(field, E14)*. Compile only: no GPU run or timing. One MM64-FP4 kernel loads hidden state directly and converts its gate-up MFMA accumulator ([moe.md](moe.md) §8).

**The e8m0 exponent: two rules.**

| Rule | Exponent from the block maximum amax | amax lands in | Used by |
|---|---|---|---|
| ceil | ⌈log2(amax/6)⌉: the unbiased exponent of the correctly rounded amax/6, plus one if its mantissa is not zero; checked for amax ≥ 2^-122, where amax/6 is normal *(field, E11)* | (3, 6]·scale: no value exceeds the e2m1 maximum 6 | all 23 MiniMax-M2 MXFP4 kernels (the block scale rounded up from its maximum / 6); DeepSeek-V2's routed experts |
| even (aiter's) | ⌊log2 amax⌋ − 2, plus one if amax ≥ 1.75·2^⌊log2 amax⌋, equal to aiter's `(bits + 0x200000) & 0xFF800000` rule *(field, E11)* | [3.5, 7)·scale before exponent clipping; checked for amax from 2^-124 to 2^125 *(field, E11)*: values above 6·scale reach the convert, which saturates them to ±6 | DeepSeek-V2's shared-expert slot |

For the tested maxima from 2^-120 to 2^121, the two differ exactly when amax's mantissa lies in (1.5, 1.75) *(field, E11)*: ceil moves to the next exponent, so the
block maximum lands in (3, 3.5)·scale and the block uses half the fp4 range; even keeps the exponent and puts it in (6,
7)·scale, above 6. The MoE references name the difference and leave it to the bound (aiter rounds a block's maximum to
nearest even, others up); DeepSeek-V2 quantizes the input once per rule and picks per route. The kernels' accuracy
checks record pass or fail, not the dB; on Gaussian test groups ceil saturated nothing, yet even's SQNR was 0.25 dB higher *(field, E11)*.
Reproduce aiter's codes or an op's reference where the caller compares with them: in the checks these kernels were run under, the e8m0 bytes of a standalone MXFP8 group quantization (E4M3FN
codes, an e8m0 scale per group) had no error bound and had to equal the reference's (§1), the biased exponent of `fl32(max(amax, 1e-10) · fl32(1/448))`
plus one if its mantissa is not zero, with codes `x / 2^(byte − 127)` clamped to ±448 *(field, E11)*. Clamp the
exponent to [−127, 127]: the stored byte `exponent + 127` then stays in 0…254, short of 0xFF, the e8m0 NaN (S31).

**Exact prescale, and the flush in `gl.exp2`.** Scaling by a power of two is exact where the result is normal
*(field, E11)*: `x * gl.div_rn(1.0, 2^e)`, one division per group, or each value divided by 2^e (7 DeepSeek-V2
kernels). But `gl.exp2` is one bare `v_exp_f32`, with or without `allow_flush_denorm` (the backend sets
`__HIP_FTZ = True`; S32, S22), which flushes denormal inputs and outputs (S33): an all-zero group gets e = −127,
`gl.exp2(-127.0)` is 0, `1 / 0` is +Inf and `0 · Inf` is NaN (dividing each value, `0 / 0` is NaN directly). Pick one:
- **Sanitize before the convert**: map NaN and ±0 to +0 (the MiniMax-M2 hardware encoders) or NaN to 6.0 (9
  DeepSeek-V2 kernels). In an e2m1 encoder this is defensive: E2M1 has no NaN or Inf code (S26) and the group's
  scale byte is 0 (2^−127), so any code decodes to at most 6·2^−127. The ISA does not say what the convert returns for a
  NaN input; 10 DeepSeek-V2 kernels convert it unguarded.
- **Avoid the Inf**: `ef = exponent.to(gl.float32)`, `clipped = gl.minimum(ef, 126.0)`, then `gl.exp2(-clipped) *
  gl.exp2(clipped - ef)` keeps both factors normal and needs no division. Convert first: on the recipe's int32
  `exponent`, `gl.exp2` fails to compile ("Expected dtype ['fp32', 'fp64'] but got int32").
- **Build the power of two from its bits**, exact: `((e + 127) << 23).to(gl.float32, bitcast=True)` for e in [−126,
  127]. Every MX quantizer in the library instead trusts `gl.exp2(e)`, 1 ulp per the ISA, to be exactly 2^e (S33), as
  it was *(field, E11)*; only one DeepSeek-V2 kernel's exactness-checked int16 transport of fp32 partials builds them from
  bits, its biased exponent kept in [1, 253] so the power and its reciprocal stay normal.

Outside an e2m1 encoder (a decay, an fp8 code, a running sum) the same NaN reaches the output *(field, E11)* and fails
the non-finite rule (§1) *(field, E2)*: use the last two, or `gl.exp`, which keeps subnormal results (§6).

**Two e2m1 values per byte.** `v_cvt_scalef32_pk_fp4_f32 D, S0, S1, S2` with S2 = 1.0 (2^0; 4.0 divides by 4, others
untested) rounds two fp32 values to nearest even, saturates to ±6 with FP16_OVFL clear or set, and writes one byte
chosen by OPSEL[3:2], S0 in the low nibble; "Other destination bits are preserved" (S23, S26) *(field, E11)*. Declare
the result `dtype=gl.uint32` and truncate with `.to(gl.uint8)`; to zero the other bytes too, start with `v_mov_b32
$0, 0` and an early-clobber output (`"v_mov_b32 $0, 0\nv_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0"`, `"=&v,v,v"`), still
`gl.uint32`. Do not declare `dtype=gl.uint8`: with `pack=1`, LLVM prints `error: could not allocate output
register for constraint 'v'` to stderr, Triton 3.8.0 raises nothing, and the asm compiles to nothing: the kernel
launches and writes no codes, or wrong ones (asm rules: [gluon/wave_ops.md](gluon/wave_ops.md) §2). A software encoder
also works: an MM64-FP4 kernel compares |x| with the code midpoints, alternating `>` and `>=` so that ties go to the even
code, and a NaN compares false and becomes code 0:

```python
@gluon.jit
def _fp4_code(x):
    a = gl.abs(x)
    code = (a > 0.25).to(gl.uint8) + (a >= 0.75).to(gl.uint8)
    code += (a > 1.25).to(gl.uint8) + (a >= 1.75).to(gl.uint8)
    code += (a > 2.5).to(gl.uint8) + (a >= 3.5).to(gl.uint8)
    code += (a > 5.0).to(gl.uint8)
    return code | gl.where(x < 0, 8, 0).to(gl.uint8)
```

At its two sizes, 128 and 256 tokens, it measured 1.19 and 1.194 times the speed of what sglang runs for the op (its gate and top-k, then aiter's `fused_moe`) *(field, E1)*.

**Scales on the way to the MFMA.** The e8m0 bytes go into `mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)`, and the
hardware applies them after the block's dot product (S31). In the tested scaled E4M3 × E4M3 MFMA, subnormal results stayed nonzero (482 of 505) but not exactly
rounded once *(field, E8)*. Fill masked scale bytes with 127 (2^0) and masked scale words with `0x7F7F7F7F`, not 0xFF:
one scale multiplies a whole block's dot product, and 0xFF is NaN (S31) *(field, E8)*. Derive the scale layout with
`get_mfma_scale_layout`; unshuffle aiter's e8m0 bytes first: [gluon/mfma.md](gluon/mfma.md) §7.

## 4. fp32 block-scaled GEMMs: one unit-scale MFMA per 128-K group

The fp8 GEMMs with fp32 block scales (of the library's 154 kernels, the 24 a8w8 blockwise GEMMs; 13 of the 16 fp8 MoE kernels: MiniMax-M2's 9
and the four Qwen2-MoE router+experts kernels) keep the scales out of the matrix core: each 128-K group gets its own MFMA from a zero
accumulator with unit scales, `gl.amd.cdna4.mfma_scaled(aa, None, "e4m3", bb, None, "e4m3", zero)`, and the scale pair
multiplies that partial in fp32, `dot * sx[…][:, None] * sw[…]` ([gluon/mfma.md](gluon/mfma.md) §2.1). In an a8w8 GEMM for N 7168 × K 2048, a group's operands are slices of the loaded tiles:

```python
@gluon.jit
def _group_dot(a, b, zero, GROUP: gl.constexpr):
    M: gl.constexpr = zero.type.shape[0]
    BN: gl.constexpr = zero.type.shape[1]
    aa = gl.amd.slice(a, (M, 128), (0, GROUP * 128))
    bb = gl.amd.slice(b, (128, BN), (GROUP * 128, 0))
    return gl.amd.cdna4.mfma_scaled(aa, None, "e4m3", bb, None, "e4m3", zero)  # unit scales, zero accumulator
```

- **Why outside the MFMA.** Its scale operand is an e8m0 exponent (S31), the a8w8 block scales fp32 values from 2^U[-3,
  0], not powers of two *(field, E2)*; one `16x16x128` spans one 128-wide scale block, so one `sx · sw` scales it.
- **What Gluon emits.** With both scales `None`, `mfma_scaled` builds constant `uint8` 0x7F (2^0) scales and an fp32
  `dot_scaled` (S38); compiled for gfx950 it stays `v_mfma_scale_f32_16x16x128_f8f6f4` with 0x7F in a VGPR as both scale
  operands. Instruction choice and rates: [gluon/mfma.md](gluon/mfma.md) §2, [gemm.md](gemm.md) §2.
- **Inside a group** the MFMA rounds to nearest even and keeps denormals (S31; subnormal results not exactly, §3); its
  summation order is not documented, and the gfx950 fp8 MFMA accumulates below an IEEE fp32 model of the dot
  product: with one large product (448 × 448) among small ones at K = 128, 4 elements exceeded that model's rounding bound, the worst at 1.04 times it *(field, E8)*.
- **Across groups** one program owns an output tile over all of K and folds the groups in a fixed order: sequentially,
  or, where one MFMA carries two K groups in alternate rows, into two accumulators by the parity of that MFMA,
  row pairs and accumulators summed at the end. No a8w8 kernel uses an atomic or splits K across programs: their
  outputs repeated bit for bit at all 24 sizes *(field, E9)*, though the op is `bounded` and no check asks for it.
- **The scale step** is written several ways: `acc + dot * sx * sw`, `gl.fma(partial * sx, sw, acc)`,
  `gl.fma(partial, xs[:, None] * ws, acc)`. Under the default contraction the first compiles to the second's
  multiply and FMA (§6), with the same bits; the third rounds the other product: 20% of elements differ *(field, E10)*.
  Write `gl.fma` to choose. None is the reference's `(a · a_scale) @ (b · b_scale)ᵀ`; at N 7168 × K 2048, M = 64,
  `dot * sx * sw` and `dot * (sx * sw)` (stages 0–2) both passed with no element over the bound (3 runs) *(field, E2)*.
- **The output**: `.to(gl.bfloat16)` rounds once to nearest even, `v_cvt_pk_bf16_f32` on pairs (S25). Convert before a
  cross-wave `gl.convert_layout`, which moves half the bytes: `gl.convert_layout(acc.to(gl.bfloat16), STORE)`.
- **MoE fp8 experts** use the same atom; a gate-up tile can hold gate and up columns from different weight-scale blocks,
  `sb = gl.where(col < BN, sg, su)`. Kernels that run fp8 on 16-bit MFMAs (the three Qwen2-MoE fp8 experts kernels)
  expand E4M3FN to bf16 or fp16 first, which is exact (finite E4M3FN operands are represented exactly in bf16),
  and apply the block scales the same way. Tiling and per-size choices: [gemm.md](gemm.md), [moe.md](moe.md).

## 5. fp32 accuracy from 16-bit MFMAs

The fp32 MFMA (`instr_shape` [16, 16, 4]) does 32 multiply-adds per cycle against 512 for the bf16 [16, 16, 32] form
(S26; 15.71–15.84 times the rate measured, 4 runs *(field, E5)*). The GDN prefill kernels keep it for their state algebra
and solves; one Qwen3.5 GDN prefill kernel (1025 tokens) runs K·Kᵀ and Q·Kᵀ on the bf16 MFMA, as the reference rounds the L2-normalized Q and K to
bf16. On the tested bf16 `[16, 16, 32]` MFMA, integer sums at K = 128 and 512 were bit-exact and randn error was
under 1% of an fp32 dot-product rounding bound, 2·(6·√K + 2)·u_fp32·Σ|a·b| *(field, E8)*; this does not establish IEEE fp32 accumulation or cover fp16.

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

(The result is `high + (middle + low)`.) The two truncations take 16 significant bits and leave at most 8, so the
three terms hold the fp32 weight exactly unless a lower term falls below bf16's normal range, and a bf16 × bf16 product
in fp32's normal range is exact *(field, E9)*: rounding happens only in the accumulation. RTZ to bf16 is a bit
truncation (S25), which turns a NaN whose payload sits in the low 16 bits into Inf; route non-finite weights around it.
Plain Triton's `input_precision="bf16x3"` splits both operands (S42): the tested 16 × 16 × 64 fp32 `tl.dot` compiled
to 6 bf16 MFMAs versus 16 fp32 with `ieee`; Gluon's `mfma` under `TRITON_F32_DEFAULT=bf16x3` compiled to the same ISA
as without it *(field, E13)*. Redo the split on every call: the weight is an input of the call.

**Which fp32-accurate path, by token count (MiniMax-M2 routers).** fp32 products and `gl.sum` in all 8 kernels at 1–16
tokens, the fp32 MFMA in 9 of the 11 at 32–256, the three-term split or the fp32 MFMA from 1024 (`[16, 16, 4]`,
`[32, 32, 2]`). Three bf16 MFMAs take about 3/16 of the fp32 MFMA's time over the same K (S26; 0.1887–0.1891
measured, 4 runs *(field, E5)*); which pays from 1024 tokens: [moe.md](moe.md) §5.1. On complete router logits at 13 selected branches (VALU, fp32 MFMA and three-term bf16 MFMA), with a seed-0 random and a cancellation input each, the error against an fp64 matmul of the same inputs was no larger than a GPU IEEE fp32 matmul's in L2 on 25 of the 26 inputs (the cancellation input of one 6144-token kernel 1.00009×), in max-abs on 23 and element by element on 3; on two more inputs built with a nonzero weight residual (for two further kernels, at 6144 and 8192 tokens; 16 flags and 256 residual entries set), the fp64 correction ran, moved those logits from 0 to 2^-23, and both errors were 0 *(field, E19)*. Finite inputs: no guarantee for every input or for the top-8.

**Routers: fp32 sums, rounded to bf16 where the reference rounds.** The references compute in fp32, as inference does
(bf16 and fp8 operands, fp32 accumulators): the Qwen2-MoE reference sums the router logits in float32 and rounds them to
bf16. The routers sum fp32 split partials in a fixed order and round to bf16,
`gl.sum(partial, 0).to(gl.bfloat16).to(gl.float32)`, so that they route from exactly the bf16 values the logits output
exposes. An fp32 sum in another order can land on the other side of a bf16 midpoint, which
the weights' tolerance (atol 1e-6, rtol 2^-4) allows (a logit summed in float32 may round to the next bf16) and the id comparison only between
two experts whose logits are adjacent bf16 values; these routers pass on the test inputs
*(field, E1)*. Do not sum in fp64 to match an fp32 reference: two Qwen2-MoE router+experts kernels, written against a
reference that summed in float64, do. The 1-token kernel sums the router dot in float64 on the VALU and nudges fp32 midpoints
so that fp64 → fp32 → bf16 does not round twice *(field, E9)*; the 32-token kernel runs the dot on the fp64 MFMA with the same
fix, and makes its expert gate-up's fp8 × fp8 dots exact by splitting each E4M3 value
into integer-valued fp16 halves over three MFMAs. Their speedups, the ruler's stored values against what sglang runs for the op ([tuning.md](tuning.md) §1), are 1.172 (1 token) and 1.063 (32 tokens), and 1.152 and 1.063 in a second run of the ruler on the same sources *(field, E1)*. These time the whole call; what the exactness costs is not separated.

**P·V in attention.** Choose P's precision against the SQNR margin. The attention bounds are loose because the baselines
are lossier (sglang's DSA kernel rounds the softmax weights to fp8, about 33 dB against a 20 dB
bound), and plain bf16 P passes them. With fp8 V, convert P and V to fp16: E4M3FN converts exactly and fp16 keeps 11
significant bits of P in [2^-14, 1] *(field, E11)*; with bf16 V keep both in bf16, since bf16 → fp16 is exact inside
fp16's normal range, 2^-14 to 65504, and outside it only for the 1022 bf16 values fp16's subnormals hold
(exhaustive). A bf16 high part plus a bf16 residual stacked as extra MFMA rows and folded after keeps about 16 bits, for
a subtract and a convert per probability and an add per output. It needs no extra MFMA where the rows would be padding,
8 heads in a 16-row MFMA; the DSA kernel for 8 tokens stacks the two parts of 16-head blocks into `[32, 32, 16]`, twice
the P·V MFMA work of bf16 P on `[16, 16, 32]` *(field, E5)*. Per kernel: [attention.md](attention.md) §10.

**Compensated products where the result is rounded anyway (GDN prefill).** Only output reconstruction uses compensated
bf16 products: chunk solves, the persistent recurrence and mutable state remain fp32. Two kernels split the
fp32 state into bf16 hi/mid(/lo) terms and multiply them with the bf16 query, smallest term first; three
store coefficients and state snapshots as bf16 high + low pairs and multiply high, then low. The solve has
rounding room within fp32: the in-block triangular inverse is updated as `gl.fma(value, diagonal, solved)` and, rounded
twice (§6), as `solved += gl.where(mask, column * diagonal_row, 0.0)`; both pass *(field, E1)*. Form products of
decays directly, without dividing tiny prefixes: a ratio of prefixes turns into 0/0 once they underflow
*(field, E11)*. Code, padding, chunk algebra: [gdn.md](gdn.md) §10.

## 6. Transcendentals, division and contraction on gfx950

What each fp32 call becomes with Triton 3.8.0, compiled for gfx950 (VALU instructions between the load and the store;
lowering in S22, S25, S51, S10; the native instructions' precision in S33):

| Call | Instructions | Precision | Use it for |
|---|---|---|---|
| `x / y`, `gl.div_rn(x, y)`, `gl.fdiv(x, y)` | the IEEE sequence, 11 VALU: `v_div_scale_f32` ×2, `v_rcp_f32`, 5 FMAs, a `v_mul_f32`, `v_div_fmas_f32`, `v_div_fixup_f32` | correctly rounded (2^24 normal pairs *(field, E8)*) | quotients at rounding points, once per group rather than per value |
| `gl.exp2(x)` | `v_exp_f32` | 1 ulp; denormal input and output flushed | `1 + exp(·)`, softmax, 2^e for e in the normal range, decays (below) |
| `gl.exp(x)` | a `v_mul_f32` by fl32(log2 e), `v_exp_f32`, `v_ldexp_f32` (9 VALU) | a subnormal result kept (`gl.exp(-100)` 0.45 subnormal ulp off); relative error grows with x: 2.2, 15.9, 64.8 × 2^-24 on [0.5, 1], [8, 16], [80, 88] *(field, E8)* | a subnormal result that is divided by or is an output (below) |
| `libdevice.exp(x)` (`from triton.experimental.gluon.language.extra import libdevice`) | `__ocml_exp_f32` (S51): `x · log2(e)` in two parts, `v_exp_f32`, `v_ldexp_f32` (14 VALU) | under the default contraction it also grows with x: 2.0, 12.4, 45.5 × 2^-24 on the same ranges *(field, E8)* | ahead of a bf16 rounding point barely better than `gl.exp`: 2117 against 2209 bf16 mismatches in 2^24 inputs *(field, E8)*; the 26 GDN prefill kernels' conv SiLU |
| `gl.log(x)` | `v_log_f32` among 13 VALU | a denormal input kept: `gl.log(1e-40)` ≈ −92.1 *(field, E8)* | |
| `gl.log2(x)` | `v_log_f32` among 7 VALU | denormals kept: 2^-140, 2^-149 exact *(field, E8)* | |
| `gl.rsqrt(x)`, `gl.sqrt(x)` | `v_rsq_f32`, `v_sqrt_f32` | within 1 ulp (0.86, 0.92 over 2^24 inputs); denormal 1e-40 flushed *(field, E8)* | an RMS norm ahead of a bf16 rounding point (`gl.rsqrt`: every output within 0.5 bf16 ulp on 10^6 values *(field, E8)*) |
| `gl.sqrt_rn(x)` | the IEEE sequence (17 VALU) | correctly rounded (2^24 normal inputs *(field, E8)*) | |
| `gl.fma(a, b, c)` | `v_fma_f32` or `v_fmac_f32`; `v_pk_fma_f32` on register-adjacent pairs | one rounding | pinning a contraction |
| `gl.clamp(x, lo, hi)` | `v_med3_f32` | a quiet-NaN `x` yields `lo`, a signaling one `hi` (§2) | the ±448 clamp |
| `gl.maximum(x, y)`, `gl.max(x, axis)` | `v_max_f32`, plus a quieting `v_max_f32 v, v, v` per loaded operand in IEEE mode, the default (`amdgpu-ieee` below) | maxnum: a NaN operand is dropped | amax of finite groups |
| `.to(gl.bfloat16)` | `v_cvt_pk_bf16_f32` on pairs | round to nearest even | rounding points |
| `.to(gl.bfloat16, fp_downcast_rounding="rtz")` | bit truncation (`v_perm_b32` of the high halves, on value pairs) | toward zero | exact splits (§5) |

**`gl.exp2` or `gl.exp` for a decay.** A flushed subnormal is lost inside `1 + exp(x)` anyway: native exp2 is
sufficient there, since any flushed subnormal is already too small to change that sum.
A flushed decay changes `state · decay` by less than 2^-126·|state| *(field, E11)*, far inside the GDN decode
state bound of 1e-2 absolute and 1e-2 relative: the Qwen3.5 GDN decode kernel for 2 tokens takes `gl.exp2` for the decay itself and
passes *(field, E1)*, while those for 4 and for 8 tokens keep `gl.exp` for it. Keep `gl.exp` where a subnormal result is
itself an output, or is divided by while its reciprocal still fits fp32 (x above about −88.7, estimated) *(field, E11)*.

**Native `v_rcp_f32`, `v_exp_f32` and `v_log_f32`** (written as inline asm in 16, 10 and 7 kernels). Use them where the
full-precision lowering does work the value does not need (S33), and write asm only for `v_log_f32`: `gl.exp2` already
is one `v_exp_f32` and `libdevice.fast_dividef(1.0, x)` one `v_rcp_f32` (S51; compiled for gfx950; [../attention/softmax.md](../attention/softmax.md) §2;
[gluon/wave_ops.md](gluon/wave_ops.md) §4, §9.1, and §2 for the constraints, packing and purity of asm calls).

- `v_rcp_f32` instead of a division: 1 instruction instead of 11, 1 ulp, denormals flushed; one Newton step (two FMAs)
  brings it to at most 0.5 ulp on the tested inputs *(field, E8)*. The Qwen3-Next GDN decode kernel that covers every
  size keeps a true division anywhere the reference expression is kept verbatim, a bare `v_rcp_f32` only where the
  caller immediately rounds to bf16 (then one bf16 ulp off in 7e-6 of inputs *(field, E8)*), and `v_rcp_f32` of the argument scaled by
  2^-64 where the result can be subnormal, the reciprocal scaled by the same constant again:

  ```python
  # 2**-64, used to keep v_rcp_f32 inputs and outputs in the normal range.
  _RESCALE = gl.constexpr(5.421010862427522e-20)
  # ... in the sigmoid, `denominator` being 1.0 + exp(-value):
          reciprocal = gl.inline_asm_elementwise(
              "v_rcp_f32 $0, $1", "=v,v", (denominator * _RESCALE,),
              dtype=gl.float32, is_pure=True, pack=1,
          )
          return reciprocal * _RESCALE
  ```

  Where the quotient is itself a rounding point, add §2's correction.
- `v_exp_f32`: that kernel's native exponential is `gl.exp2(x * log2(e))`; squaring `v_exp_f32(x · log2(e) / 2)`
  restores the gradual underflow that the native instruction flushes, without a branch.
- `v_log_f32` (base 2, 1 ulp, denormals flushed) for the 13 instructions of `gl.log`: the Qwen3-Next GDN decode softplus
  `log(1 + exp(a))` uses it, its argument being at least 1; times a rounded ln 2,
  it differed from `gl.log` in 27% of fp32 results for a in [−20, 20] *(field, E8)*.

Compiled for gfx950, an intrinsic is a machine instruction in the MIR, while an inline-asm string stays one INLINEASM node with its declared register defs and uses: the post-RA hazard pass inserted `S_NOP 0` after both tested single-instruction `v_exp_f32` and `v_rcp_f32` asm calls, but the instructions inside one multi-instruction asm string are not separate schedulable nodes. Timed on identical inputs, single-instruction asm `v_exp_f32` and `v_rcp_f32` took 1.045 times the intrinsic's time (99% interval 1.018–1.072; A/A 0.998), so a slowdown of 3% or more is not established; outputs matched; both forms got one LLVM-added wait state at 1 value per thread and other instructions in that slot at 16 *(field, E16)*.

**FP contraction is on by default.** `enable_fp_fusion` defaults to the knob `TRITON_DEFAULT_FP_FUSION`, true (S32):
compiled for gfx950, `a * b + c` is one `v_fmac_f32` (one rounding), with `enable_fp_fusion=False` or the knob at 0 a
`v_mul_f32` and a `v_add_f32` (on register-adjacent pairs `v_pk_fma_f32` against `v_pk_mul_f32` and `v_pk_add_f32`). A
select also keeps them apart: `c += gl.where(m, a * b, 0.0)` is `v_mul_f32`, `v_cndmask_b32`, `v_add_f32` under the
default. Where the reference fixes how a product is rounded, write the rounding (the tap products of §1), `gl.fma` for the
single rounding you want, or `enable_fp_fusion=False` for two roundings: the residual probe `a*b - fl32(a*b)` matched
the exact residual with default fusion and `gl.fma`, and was zero with fusion off (2^24 inputs, N=3) *(field, E8)*.
The probe did not test the explicit round trip. Of the library's 154 kernels, 42 pass `False` at every size (19 DeepSeek-V2, 11 Qwen3.5 GDN prefill, 11 Qwen3-Next prefill, one DSA) and four set it per size: an MM256-FP4 kernel as `m != 16` (`False` at 16 tokens, `True` at 4 and 8), three Qwen3.5 GDN prefill kernels as `m == 1024` (`False` at 512; the one whose only size is 1024 gets `True`); no comment explains the setting *(field, E15)*. With `True`, the DS512 kernel for 8 tokens compiles to the same code; a Qwen3-Next GDN prefill kernel does not (§7).

**`allow_flush_denorm=True` flushes nothing here.** It sets `denormal-fp-math-f32` to `preserve-sign` (S32), but
compiled for gfx950 `gl.exp`, `gl.exp2`, `gl.log`, `gl.log2`, `gl.sqrt`, `gl.sqrt_rn`, division and the descriptor's
`.amdhsa_float_denorm_mode_32 3` stay as without it, as do the 4 DSA 64-head kernels that pass it (§7) *(field, E12)*.

**`amdgpu-ieee` false.** Passed as `llvm_fn_attrs=[["amdgpu-ieee", "false"], …]` on the routing launch of 5 of the 7
Qwen2-MoE router kernels (those for 8, 16, 64, 128 and 256 tokens), it clears IEEE mode (`.amdhsa_ieee_mode 0`), and LLVM
drops the quieting `v_max_f32 v, v, v`: `gl.maximum(x, y)` of two loads is one instruction instead of three (compiled
for gfx950). A min or max then returns the other operand for any NaN input (S57, S27).

To confirm in a build ([debugging.md](debugging.md) §1, §3): count `v_div_fixup_f32` (one per division); find
`s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1)` (FP16_OVFL), `v_cvt_scalef32_pk_fp8_f32`, FMAs against a separate multiply
and add; read `.amdhsa_ieee_mode` and `.amdhsa_float_denorm_mode_32`.

## 7. Numerics switches: measured sizes, not a no-speed rule

The speedups here are the ruler's ([tuning.md](tuning.md) §1) against what sglang runs for the op; each run also checks accuracy. `enable_fp_fusion` `False` → `True` on the three launches of the DS512 kernel at its only size, 8 tokens, left its gfx950 ISA unchanged (633, 1554 and 440 instructions): speedup 1.147 → 1.146, B/A 0.9994, A 1.142–1.150, B 1.140–1.151 (4 alternated runs per arm), all eight accuracy checks passed. Removing `["amdgpu-ieee", "false"]` from the routing launch of the Qwen2-MoE router kernel for 256 tokens gave 1.910 → 1.921, B/A 1.006 (4 runs per arm), with A's own spread 0.94%: a description, not a significance test. Removing `allow_flush_denorm=True` from four DSA 64-head kernels gave 0.997–1.0033, with identical ISA in all eight compiled kernels, a split and a merge kernel each (2 runs per arm) *(field, E12)*.

The fusion switch can matter for speed: a Qwen3-Next GDN prefill kernel with fusion on ran at B/A 1.0342 at 1024 tokens and 1.0260 at 2048 (3 alternated runs per arm), kernel time 96.63 → 93.82 µs and 114.28 → 111.62 µs; its four `False` launches compile to different code, as do 82 of the 226 `False` launches, counted per size, among the library's 154 kernels. A Qwen3.5 GDN prefill kernel at 512 tokens and a DS256 kernel at 1 token gave 0.9983 and 0.9998. Every tested A/B run of the five kernels of this section passed accuracy: none of them needed its setting to pass *(field, E15)*. Keep a switch where the reference requires its numerical rule, not because these kernels prove such a requirement; measure at the target size before changing it for speed. Other options: [tuning.md](tuning.md) §9.

## 8. Determinism and atomics

- **Bitwise kernels use no atomics.** Of the library's 154 kernels, none of the 65 `bitwise` ones contains a global or LDS atomic; having no atomic does not by itself fix a reduction order, which the code must. The plain-Triton bf16 GEMM for N 2624 × K 6144 splits K six ways only for 16 < m ≤ 64 (of its sizes, m = 64): each split writes an fp32 partial tile, and a second kernel adds the six partials in a fixed order before rounding to bf16, so the result is bitwise reproducible, as it was at all its sizes *(field, E9)*; when that split pays: [gemm/small_m_gemm.md](gemm/small_m_gemm.md) §6.
- **Bounded kernels use integer atomics, for positions only.** 31 of the 89 `bounded` kernels (DeepSeek-V2 12,
  MiniMax-M2 MXFP4 11, MiniMax-M2 fp8 6, Qwen2-MoE fp8 experts 2) contain an atomic. Their 42 global atomic calls are
  all `sem="relaxed"` integer adds or ors that hand out tickets, job slots, cursors and token masks; three kernels also
  reserve LDS slots with `atomic_scatter_*`. No kernel has a float atomic.

```python
ticket = gl.atomic_add(Counts + experts, 1, valid, sem="relaxed")
gl.store(Sorted + experts * CAPACITY + ticket, row * TOPK + slot, valid)
new_job = valid & (ticket % 16 == 0)
```

`Counts` and `Total` must read zero at the start of every call, graph replays included (unzeroed, a replay's
tickets started at 4096 *(field, E9)*): an earlier launch of the same call zeroes them ([moe.md](moe.md) §6). Where LDS
reservations reorder rows, the Qwen2-MoE experts put every contribution back before summing: integer reservations may
reorder rows, but every contribution scatters to its original entry before the ordered eleven-route reduction.
`gl.atomic_add` defaults to `acq_rel` at `gpu` scope (S10); `relaxed` suffices for counters a later launch reads.

A bounded kernel can also change bits by switching code paths at run time: a DS512 kernel (4096 tokens) picks its MFMA shape from a block's live rows, 16 × 16 × 128 for few and 32 × 32 × 64 for more. Over 100 synchronized calls its target block stayed on the 16 × 16 × 128 body with one hash of its route partials, and at 15, 16, 17, 31, 32 and 33 live rows the target's up codes, scales and down partials and token 0's 6144 outputs were identical within each live count (1,040 hashes) *(field, E20)*. Whether a switch between the bodies changes last bits from run to run is untested, as is the rest of the batch.

What a new kernel should do:
- Under `bitwise`: no atomic split-K. Write fp32 partials to scratch and reduce them in a fixed order, or give each
  output to one program ([../gemm/correctness.md](../gemm/correctness.md) §8).
- Under `bounded`: let atomics place work, and keep every float sum in a fixed order anyway: easier to debug.
- `bitwise` compares two runs, not kernels (§1): a new tile, split count or K rotation changes bits and stays correct.
  Judge by the bound, not by bit-equality with the old kernel or aiter, and repeat a kernel with atomics or cross-wave
  races many more times yourself ([../gemm/correctness.md](../gemm/correctness.md) §8, §3).

## 9. Matching the reference beyond rounding points

- **Ties.** The Qwen2-MoE reference breaks top-k ties toward the lower expert id with a stable descending sort, and
  the ids must equal it: one router+experts kernel builds order-preserving integer keys that map −0 to +0 and carry `511 - expert` in the
  low bits. The DeepSeek-V2 and MiniMax-M2 references use `torch.topk`, which fixes no tie order, under an SQNR
  bound on the MoE output; their kernels still take the lower id, with `(av > bv) | (av == bv) & (ai < bi)` or the lowest
  index among the maxima.
- **The reference, not the baseline.** DeepSeek-V2 kernels add the correction bias in fp32 as the reference does,
  not in bf16 as sglang's top-k receives it.
- **NaN on purpose.** `gl.maximum(x, -float("inf"))` turns a NaN into −inf (a loaded −inf *(field, E11)*): in the
  Qwen2-MoE routers a NaN tail becomes the removed head's −inf placeholder, matching the reference's behavior when
  fewer than top-k non-NaN experts remain. Elsewhere `gl.maximum` and `gl.max` drop a NaN (maxnum) where `torch.amax` propagates it, and
  `gl.clamp` returns its lower bound for a quiet NaN: equal to torch on 10^6 random finite values (no ±0 or subnormals),
  different on NaN *(field, E11)*. `gl.max` takes no `propagate_nan`: use `gl.reduce(x, axis, combine)` with a combine
  returning `gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)` (§2; one wave: `v_maximum3_f32`, no LDS).
- **Neutral padding.** Masked activations load as 0, e8m0 scales as 127, GDN padding tokens as decay 1 and beta 0;
  empty attention splits publish `(max = -inf, den = 0, num = 0)`, so beside a split of finite max their merge
  weight exp(−inf) is 0 *(field, E11)*; two empty records are untested.
- **Casts at stores.** `gl.store` casts to the pointer's element type, rounding to nearest even (S10);
  `gl.amd.cdna4.buffer_store` does not, and a value whose element type differs from the pointer's fails to compile
  (S61): cast first. Buffer stores: [gluon/memory.md](gluon/memory.md) §3.
- **Repeated selections must agree.** Where several programs recompute one routing choice, every copy must pick the same expert for each slot: in a DS256 kernel at 1 token, changing only the up side's bf16 rounding or its lower-id tie rule made producer and consumer slots disagree on 12 of 13 inputs, and changing both sides alike restored agreement on 36 of 36. Reversing the order of its 12 split partials changed no id, weight or output on four saved inputs, which did not exercise that order. On four near-tie inputs, one wrong expert outweighed a missing rounding: changing slot 7's expert in both up and down moved the output 6.2, 112, 6.1 and 9.5 times as far as removing the activation-product bf16 rounding, and its error against the reference was 1.22, 3.03, 1.23 and 1.28 times as large *(field, E21)*.
- **Dead fp8 variants.** 12 kernels keep a group-maximum branch of 224 (240) for a non-E4M3FN fp8 dtype (gated
  attention 4; Qwen3.5 GDN prefill 6; Qwen3-Next GDN 2); five
  more gated kernels name `float8_e4m3fnuz` only as a default they assert away. None of these ops takes FNUZ
  inputs: do not copy those branches.

## 10. Pitfalls

Have a failed tolerance bound report the fraction of elements over it, the peak (the largest error as a multiple of the bound B) and the non-finite mismatches, and a failed SQNR bound the output's SQNR against the floor and the non-finite mismatches:

| The report | Look first at |
|---|---|
| non-finite mismatches | NaN from `0 · (1 / gl.exp2(-127.0))`, or a scale byte 0xFF (§3); Inf where the reference is NaN, as from RTZ (§5); a finite maximum (fp8 448, fp16 65504, the fp32 maximum) where the reference overflows, as from a Triton fp8 cast's FP16_OVFL, or NaN where it is finite, as from an asm fp8 convert without it (§2); each of these failed the check *(field, E2)* |
| many elements far over the bound | arithmetic left in bf16 (§1; a bf16 running sum over K = 2048: 94% of elements over, peak 58 × B); garbage in the other nibble or bytes of an fp4 code word (§3; 44% over); an output not written, as by an asm result declared `gl.uint8` (§3; all over) *(field, E2)*; then indexing ([debugging.md](debugging.md) §5) |
| a few elements over (about 0.01%, peak up to about 3 × B, under bounds of about one rounding step *(field, E2)*) | a rounding point missed or doubled: the round trip (§1), contraction (§6), `x * (1 / s)` for `x / s` (§2) |
| SQNR short | a further lossy step in the budget (§1; an extra e5m2 rounding before MXFP4 cost 1.7 dB *(field, E2)*), the e8m0 rule (§3), P's precision (§5; fp8 P gave 23 dB on P·V *(field, E2)*) |
| the repeat run of a `bitwise` op differs, or a mismatch comes and goes | atomics and other arrival-order sums; two runs can miss them: a 6-program fp32 atomic sum took 3 bit patterns in 300 runs, yet 281 of 299 consecutive pairs agreed *(field, E11)* (§8) |

## Evidence

Triton at v3.8.0 (`triton-lang/triton@c01b677`). The E lines on library kernels were measured on the library's 154 kernels ([kernels/kernel_library.md](kernels/kernel_library.md) §1); "the ruler" is [tuning.md](tuning.md) §1's, its kernel durations read from Kineto.
- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto); accuracy against each output's bound, speedup against what sglang runs for the op] every size with a recorded accuracy result passed: 198 of the 242 sizes of the library's 154 kernels, among them gated attention 9, Qwen3.5 GDN decode fp8 8 and prefill 21, MiniMax-M2 fp8 28, Qwen2-MoE fp8 7; the 44 sizes of the 20 Qwen3-Next kernels, which had no recorded result then, all passed in a later run of the ruler; speedups: the MM64-FP4 kernel with the software e2m1 encoder 1.19 at 128 tokens and 1.194 at 256; the Qwen2-MoE fp8 router + experts kernels with a float64 router dot 1.172 at 1 token (on the VALU) and 1.063 at 32 (on the fp64 MFMA), 1.152 and 1.063 in a second run of the ruler with unchanged sources (image lmsysorg/sglang:v0.5.22-rocm10-mi35x)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, runs on one GPU: the bound formulas applied to built outputs, or the accuracy check run on a library kernel and on an edited copy] a 0/0 tolerance on bf16: B = 2^-8·|y|, the round-to-nearest output passed, 1% of elements one step further failed (fraction over the bound 0.00995); MXFP4 round trip (ceil e8m0, RNE e2m1) of both expert-GEMM inputs inside the references, first input set: DeepSeek-V2 13.45 dB at 256 tokens, MiniMax-M2 13.18 dB at 128; an fp32 dot-product rounding bound at K = 6144: fp32 sums in 1, 6 and 16 parts passed at peak 0.69; bias test, n = 10^6, every error +0.5·u·|y|: |mean|/rms 0.80, the output-rounding and dot-product bounds failed, the tolerance, the SQNR floor and a bound relative to a same-mean baseline passed; RTZ to bf16 of positive values 0.68; the check's large-magnitude input set, every floating-point input scaled by one exact power of two, the largest that keeps the tensors and the dot-product outputs in range: 2^54 for the N 2624 × K 6144 bf16 GEMM and 2^55 for the N 6144 × K 2048 bf16 GEMM at all 10 sizes; Qwen3.5 GDN decode bounds: 65536 codes moved one e4m3 step and 512 scales moved one ulp all passed; a8w8 block scales, 2400 drawn: log2 from −2.999 to −0.0009, none an integer; failures: `0 · (1 / gl.exp2(-127))` 256 of 256 non-finite mismatches, a 0xFF scale, an RTZ'd NaN, saturated 448, 65504 and fp32 maximum, NaN against 448 one each; a bf16 running sum over K = 2048 fraction over the bound 0.941 at peak 58.4, a stray fp4 nibble 0.438, an unwritten output 1 at peak 128; missing tap rounding, a contracted product and x·(1/s) at injected ties 1.4e-4, 9.5e-5, 1.2e-4 at peaks 1.84, 1.95, 2.67 under bounds chosen for the test; fp8 P on P·V 23.1 dB; MXFP4 18.77 dB, 17.07 with an e5m2 rounding first; the Qwen2-MoE router kernel for 256 tokens, seed 0: fp32 logits rounded once to bf16, then ranked, gave the reference's top-10 ids for all 256 tokens, the unrounded sum ranked changed 97 tokens and failed the id comparison (fraction of ids differing 0.0809) in both checked runs of each run; the a8w8 GEMM N 7168 × K 2048 at m = 64: `dot * sx * sw` against `dot * (sx * sw)` in K stages 0–2 of 8 (stages 3–7 multiply by a precomputed sx · sw in both arms), 2 checked runs per arm in each run: all 12 passed, worst fraction over the bound 0
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, chained MFMAs on register operands, one wave per program, 2048 programs, median of 7 rounds × 3 launches] bf16 [16, 16, 32] over fp32 [16, 16, 4] rate 15.71–15.84 (2447–2468 against 155.2–156.1 TFLOP/s); three bf16 [16, 16, 32] chains against one fp32 [16, 16, 4] chain on the same 64 × 64 tile and K: 0.1887–0.1891 of the fp32 chain's time (3/16 = 0.1875); a 32 × 128 tile on [32, 32, 16] against a 16 × 128 tile on [16, 16, 32], same K: 1.930–1.978 times the time (probe kernels, not library kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, probe kernels on one GPU, errors against float64] floored `v_rcp_f32` scale reciprocals for amax 0 to 1e30 all normal, at least 4.48e-28, and `v_rcp_f32(2^127)` = 0; division spellings 0 mismatches over 2^24 normal pairs; `gl.exp(-100)` = 3.784e-44, 0.45 subnormal ulp off; max relative error × 2^24 of `gl.exp` 2.2, 15.9, 64.8 and of `libdevice.exp` 2.0, 12.35, 45.5 on [0.5, 1], [8, 16], [80, 88]; bf16 mismatches over 2^24 x in [−60, 60]: 2209 (`gl.exp`), 2117 (`libdevice.exp`); `gl.log(1e-40)` = −92.1; `gl.log2` of 2^-140 and 2^-149 exact; `gl.sqrt`, `gl.rsqrt` at most 0.92, 0.86 ulp over 2^24 inputs in 2^±100, 0 and +Inf at 1e-40; an RMS norm with `gl.rsqrt` on 1000 × 1024 randn: every bf16 output within 0.5 ulp; `gl.sqrt_rn` 0 mismatches over 2^24; bare `v_rcp_f32` 10.87% of 2^24 fp32 results off, 124 (7.4e-6) one bf16 ulp off after rounding to bf16; `v_log_f32(1 + gl.exp(a)) · fl(ln 2)` against `gl.log`: 0.2704 of 2^24 a in [−20, 20] differ; `a * b - fl32(a * b)`: default and `gl.fma` exact on all 2^24, `enable_fp_fusion=False` 0; `mfma_scaled` e4m3 with per-row scales 2^-90 to 2^-60: 359 of 1024 outputs differ in bits from float64 rounded once, all subnormal, 23 of 505 expected nonzero subnormals 0; one 0xFF scale byte: its row and column NaN, the rest exact; K = 128 with one 448 × 448 product: 4 elements over an fp32 dot-product rounding bound, peak 1.04; bf16 [16, 16, 32] MFMA: integer data exact at K = 128 and 512, randn peak 0.0081 of the bound (fp8 0.871); rows of A and B permuted: outputs permuted bit for bit (16x16x128, 32x32x64 e4m3, e4m3 × e2m1); a8w8 at K = 1024 with scales from 2^U[-3, 0] and from 2^U[-1, 1]: 0 elements over atol = rtol = 1e-2 in 3 seeds of 4096 × 256 (65536 elements per seed in the first run); the counts the same in all 3 runs (probe kernels, not library kernels)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, runs on one GPU] the accuracy check on a `bitwise` and a `bounded` copy of one kernel ran the first input set 3 times (a first capture, the accuracy run, the repeat run) and 2 times, and bit flips planted in those runs produced no finding; bf16 × bf16 products in fp32's normal range exact for 748591 VALU and 16384 MFMA products; 1024 values near bf16 midpoints: plain fp64 → fp32 → bf16 wrong on 260, the nudge of the 1-token Qwen2-MoE router + experts kernel on 0; repeated calls on the same inputs: the 24 a8w8 blockwise kernels 0 non-identical outputs in 20 runs per size, the bf16 GEMM N 2624 × K 6144 1 bit pattern in 20 runs at m = 2, 4, 8, 64, the 31 bounded kernels with atomics byte-identical at every size in 10 runs (the DS512 kernel of E20 in 50); a 4096-program ticket kernel in a graph: second replay offset by 4096 without a zeroing node, identical with one (3 runs)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels on one GPU] against an fp32 mirror of the reference and a bound of two output roundings (2·u_out·|y|): the fp32 precursor quantized instead of the bf16 output put 100000 of 100000 scales outside the bound and changed 282120 of 12.8M codes (repeated once on the same data, same result); all-zero group: scale 2.232e-13 and code 0 with the floor, scale 0 and code 0xFE without it; fused norm + quant 12.22 µs against 22.37 µs for the two kernels at 4096 rows, 13.63 against 22.49 at 131072 (median over 9 rounds of 50 back-to-back launches, no flush, the fused kernel timed first); 12.8M fp32 values within 2 ulps of e4m3 midpoints: x/s against x·fl32(1/s) 624683 codes moved, each one step, the fourth form 0 code and 0 quotient mismatches against IEEE division; bf16 inputs, |x|/amax from 2^-10 to 2: 0 of 491520 pairs moved; `amax / 448` against `amax · fl32(1/448)`: one ulp below for 71 of 128 significands in each of 25 binades, above for none; 100000 groups with divided scales: no element over the bound, |mean|/rms 0.309 against 0.0158; VALU 550 (division per value), 236 (reciprocal per group), 252 (fourth form); without a clamp and with scale = amax/448: x/scale above 448 in 325 of 4096 rows, within 448·(1 + 2^-22), Triton's cast 0x7E at every row's amax, no NaN code; a kernel with a Triton fp8 cast against one without: fp16 multiply 0x7BFF against 0x7C00, bf16 and scaled-e4m3 MFMAs 0x7F7FFFFF against 0x7F800000, fp32 → bf16 0x7F7F/0xFF7F against 0x7F80/0xFF80; 2^20 elements: `gl.fma(p * sx, sw, acc)` and `acc + p * sx * sw` 0 elements differ, `gl.fma(p, sx * sw, acc)` 213955 (probe kernels, not library kernels)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, probe kernels on one GPU, against a float64 model] the §3 recipe on 100000 groups (randn · 2^k, k from −100 to 100, tie rows): 0 wrong code and scale bytes against ceil e8m0 + RNE e2m1; ceil exponent on 104000 maxima from 2^-122 to 2^127: 0 wrong, amax/scale in (3, 6]; even exponent on 105564 maxima from 2^-126 to below 1.75·2^127: 0 mismatches against aiter's rule, amax/scale in [3.5, 7) from 2^-124 to 2^125; 100256 maxima from 2^-120 to 2^121: the rules differ on exactly the 25551 with mantissa in (1.5, 1.75), ceil at (3.0000002, 3.4999998)·scale, even at (6.0000005, 6.9999995)·scale; 99900 Gaussian groups scaled by 2^-100 to 2^100: ceil 0 and even 19937 elements above 6·scale, SQNR 18.74 and 18.99 dB; the MXFP8 group quantization's reference formula on 100000 groups 0 wrong bytes and codes, `amax / 448` giving the same bytes; power-of-two scaling 0 differ on 200000 values with normal results (e in [−60, 60]); `gl.exp2(e)` exact for all 254 integers e in [−126, 127]; `v_cvt_scalef32_pk_fp4_f32` with S2 = 1.0: 0 mismatches over 73347 pairs against ties-to-even, byte select 0 to 3 right with the other bytes unchanged, ±100 to 0x7/0xF with FP16_OVFL 0 and 1; S2 = 4.0 gave the codes of x/4; attention P at 8 tokens (DSA, Qwen3.5): in fp16 max relative error 4.865e-4 for P ≥ 2^-14 (282 of 131072 below); all 254 finite E4M3FN codes exact in bf16 and fp16; e = −127: `0 · gl.div_rn(1, gl.exp2(e))` NaN, the two-factor and from-bits forms 0; decay 0.5 per token: prefixes 0 at 170 and 200 tokens, their ratios NaN, the direct products 2^-10; 65536 decays below 2^-126: |state·exp − state·exp2| at most 0.94 of 2^-126·|state|; x = −88.5: `gl.exp` 3.672e-39 with reciprocal 2.723e38, `gl.exp2` 0 with Inf; x = −100: both reciprocals Inf; 16384 merges of an empty split with a finite one, both orders: weight 0, result bit-equal; `gl.maximum` of [qNaN, sNaN, 1.0] with a −inf loaded from memory: [−inf, −inf, 1.0]; 1000064 normal fp32 values, none ±0: `gl.maximum`, `gl.max` and `gl.clamp(±448)` bit-equal to torch; a 6-program fp32 atomic sum into one cell: 3 bit patterns in 300 runs, 281 of 299 consecutive pairs identical (probe kernels, not library kernels)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler with its accuracy check, on one GPU, 4 runs per arm alternated, 2 for each DSA kernel; both arms' launches compiled and their ISA compared] the DS512 kernel for 8 tokens, `enable_fp_fusion` `False` → `True` on all 3 launches: speedup 1.147 → 1.146 (A 1.142, 1.147, 1.15, 1.149; B 1.147, 1.147, 1.151, 1.14), B/A 0.9994, the same instructions and `.amdhsa_` fields (633, 1554, 440 instructions); the Qwen2-MoE router kernel for 256 tokens, `["amdgpu-ieee", "false"]` removed from the routing launch: 1.910 → 1.921 (A 1.905, 1.922, 1.907, 1.904; B 1.926, 1.911, 1.923, 1.927), B/A 1.006, A's (max − min)/mean 0.94%, the routing launch 612 → 734 instructions and `.amdhsa_ieee_mode` 0 → 1, the split-K projection launch unchanged; accuracy passed in all 16 runs of the two; `allow_flush_denorm=True` removed from the 4 DSA 64-head kernels at 12, 16, 24, 32 tokens: speedup ratio 1.0007, 1.0033, 0.9986, 0.997, accuracy passed in every B run, the split and the merge kernel the same instructions and `.amdhsa_` fields in 8 of 8 kernels
- E13 [2026-11, CPU (target gfx950), Triton 3.8.0, N=1, compiled, no GPU launch] default `gl.clamp`: one `v_med3_f32`; with `PropagateNan.ALL` one `v_cmp_u_f32` and one `v_cndmask_b32` more; `gl.maximum` with `ALL`: one `v_maximum3_f32`; a 16 × 16 × 64 fp32 `tl.dot`, both operands split by `bf16x3`: 6 bf16 MFMAs against 16 fp32 with `ieee`; Gluon `mfma` under `TRITON_F32_DEFAULT=bf16x3`: the same ISA as without it (test kernels, not library kernels)
- E14 [2026-11, CPU (target gfx950), Triton 3.8.0, N=2 compile runs of 144 builds each, no GPU correctness or timing] SwiGLU + MXFP4 on 256 bf16 columns, NW = 1, 2, 4, 8: `BlockedLayout([1, 8], [16, 4], [NW, 1], [1, 0])` at 16 to 64·NW rows and `BlockedLayout([1, 32], [64, 1], [NW, 1], [1, 0])` at 64·NW rows: no LDS, no barrier; converted to `[1, 2], [4, 16]`: 8–256 KiB of LDS, a barrier at NW ≥ 2; NW = 4 with the second layout: 64 rows two zero warp bases and 177 / 179 VGPRs spilled (codes / codes and scales), 128 rows one zero base and 177 / 179, 256 rows none and 177 / 177; a 256 × 64 tile at NW = 4 no spill; 64·NW × 256 rows: 164–193 VGPRs spilled at NW ≤ 4, 570–805 at NW = 8 (test kernels, not library kernels)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, 3 alternated runs of the ruler per arm; a source census and the compiled ISA of every `False` launch] of the library's 154 kernels 42 pass `False` at every size and four set it per size (the Qwen3.5 GDN prefill kernel whose only size is 1024 tokens gets `True`); 82 of 226 `False` launches, counted per size, compile to different ISA with `True`, the DS512 kernel of E12 0 of 3; a Qwen3-Next GDN prefill kernel at 1024 tokens in 1 sequence B/A 1.0342 (A spread 0.56%, 96.63 → 93.82 µs), at 2048 in 2 B/A 1.0260 (spread 0.84%, 114.28 → 111.62 µs); a Qwen3.5 GDN prefill kernel at 512 tokens 0.9983, the DS256 kernel of E21 at 1 token 0.9998; no failing run in either arm
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, 40 alternating AB/BA pairs of 10 event-timed launches per arm on identical finite random inputs, 40 A/A controls, 10 batches of four pairs, 99% t(9) interval] asm against intrinsic `v_exp_f32` / `v_rcp_f32`: geometric-mean time ratio 1.0448 (99% interval 1.0180–1.0722), A/A 0.9979 (0.9854–1.0107); six configurations (1 or 16 values per thread, 1 or 8 iterations), 131072 or 2097152 elements per arm, zero mismatches; one LLVM-added wait state on both forms at 1 value per thread and other work in between at all 16-value pairs; `mov`, `add` and `exp` asm reading their result at once also get one (probe kernels, not library kernels)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 run, one input set per size, 2 runs of the accuracy check per arm, not independent samples] A the library kernel, B with one rounding changed: the Qwen2-MoE router kernel at 256 tokens ranking the fp32 logit sum instead of its bf16 rounding: weights' relative L2 against an unrounded float64 reference 0.004103544 → 2.836419e-7, fraction of top-k ids differing 0.08086, B failed both; a Qwen3.5 GDN decode kernel at 128 tokens without the per-tap bf16 rounding: values' relative L2 0.015502125 → 0.001760144, B failed both (values and state peaks 1.689 and 2.019); a Qwen3.5 GDN prefill kernel at 16384 tokens quantizing the fp32 precursor of its normalized output: values and scales changed, A and B passed both, peaks 0.353679, 0.64, 0.348193; the same kernel with a bf16 round trip on each conv tap product: failed once, scales peak 12.56; the DS512 kernel of E12 at 8 tokens without its router-logit rounding: passed both, paired mean 1.238747e-9, SE 3.271610e-7; the Qwen3.5 gated attention kernel of E24 at 32 rows with the attention kept in fp32 before the gate: scales' relative L2 0.0032333552 → 0.0032689805, passed both; the router kernel with two ids of token 0 swapped: id mismatch on 2 of 2560 elements, failed
- E18 [2026-11, gfx950 (MI355X), FlyDSL ?, N=1, outputs compared under the a8w8 tolerance bound, 24 input sets per size, each with block scales from 2^U[-3, 0] and from 2^U[-1, 1]] 2^U[-1, 1]: one element over the bound, at N 32768, M 2048, seed 0; 2^U[-3, 0]: none at any of the 39 sizes (K 1024 FlyDSL fp8 block-scaled GEMMs)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per input, complete router logits of the library kernel, of a GPU fp32 matmul (IEEE fp32 MFMA, no TF32) and of an fp64 matmul on the same inputs] 13 MiniMax-M2 router branches at selected sizes (VALU at 1–16 tokens, fp32 MFMA, three-term bf16 MFMA), a seed-0 random and a cancellation input each (26 inputs), plus two inputs built with nonzero weight residuals; on the 26 the kernel's error against fp64 no larger than the fp32 matmul's in L2 on 25 (the cancellation input of the MM256-FP4 kernel for 6144 tokens 1.0000934×), in max-abs on 23 (the cancellation inputs of the MM256-FP4 kernels for 2048 and 4096, for 6144 and for 8192 tokens 1.0319×, 1.0018×, 1.0451×), element by element on 3; in the residual inputs (two MM64-FP4 kernels, at 6144 and at 8192 tokens) 16 flags and 256 residual entries set, the fp64 correction ran, logits 0 without it and 2^-23 with it, equal to fp64 and to the fp32 matmul: error 0 on both (MiniMax-M2 MoE kernels)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the kernel called 100 times with synchronization, its runtime records saved and their bytes hashed on CPU] the target block on the 16 × 16 × 128 body in all 100 calls, one hash of its route partials; at 15, 16, 17, 31, 32 and 33 live rows the target's up codes, scales and down partials and token 0's 6144 outputs identical within each live count (1,040 hashes) (the DS512 kernel for 4096 tokens)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the kernel at 1 token with controlled edits, outputs compared on saved inputs] only the up side's bf16 rounding or lower-id tie rule changed: producer and consumer slots disagreed on 12 of 13 inputs; the same change on both sides: agreement on 36 of 36; the 12 split partials summed in reverse on four saved inputs: ids, weights and slot tags equal, output change 0; four near-tie inputs with the reference's ids frozen and the same fp32 slot weights, slot 7's expert changed in up and down against the activation-product bf16 rounding removed: ratio of output changes (L2) 6.206585, 112.211530, 6.133820, 9.468969, ratio of errors against the reference 1.217284, 3.027979, 1.232866, 1.283387 (the DS256 kernel for 1 token)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=7 rounds, the fused arm and the GEMM + standalone quantization kernel alternated AB/BA, a 1 GiB read before each call, event-timed, the median of the per-round ratios] fused over separate, plain Triton (`tl`) probes with the same quantization function: bf16 GEMM, N = K = 4096, a 128-column tile per group: M = 16 1.033, M = 512 0.989, M = 8192 0.987; outputs bit-identical (probe kernels, not library kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a Gluon probe dividing each value by its own scale s = fl32(max(amax, fl32(1e-10)) · fl32(1/448)), default `denormal-fp-math-f32` (`ieee`); against fl32(fl64(x) / fl64(s)) on the host] the fourth form on 169.6M values (every bf16 |x| ≤ amax for amax in 31 binades, 100000 random fp32 groups of 128): 0 code mismatches for nonzero x; 420471 quotients one ulp off, all where |x| < 2^-100 or the quotient < 2^-126; x = −0 gave +0, code 0x00 against 0x80; `gl.div_rn` 0 mismatches (probe kernels, not library kernels)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 64 input sets drawn from the op's input distributions at 32 rows, the library kernel and a copy without the attention's bf16 rounding before the gate] scales' relative L2 against a float64 reference without that rounding, the copy's over the library kernel's: geometric mean 0.849, lower on 62 of 64 sets; both passed the accuracy check on all 64 sets (Qwen3.5 gated attention with an fp8 group-quantized output, the kernel for 32 rows)
- S10 triton-lang/triton@c01b677:python/triton/language/semantic.py:1243 `store` casts the value to the pointer's element type; float downcasts default to RTNE (:824-826); `maximum` is `maxnumf` unless `propagate_nan=ALL` (:383-390), and `gl.max` reduces with it (`python/triton/language/standard.py:177-191`); atomics default to `ACQUIRE_RELEASE` (:975) at `GPU` scope (:990); a Python scalar of no higher kind than the tensor takes the tensor's type, fp32 only for `/` and `%` on fp16 and bf16 (:73-79); bf16 with bf16 gives bf16 except for `/` and `%` (:98-102); `sum` widens only integers narrower than 32 bits (`standard.py:270-281`, applied at :289-292)
- S22 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:451 `ClampFOp` → `v_med3`, a NaN select only for `PropagateNan::ALL` (:453-458); `adjustModeRegister` sets MODE bit 23 (fp16_ovfl) at the entry of a function marked by an fp8 cast (:487-496); `arith.divf` → LLVM `fdiv` (:74), `div_rn` too (:511); `gl.exp` → `llvm.exp2.f32(x · log2e)` (:277-283); `gl.exp2` → `llvm.amdgcn.exp2.f32` under FTZ (:312-313); `gl.rsqrt` → `llvm.amdgcn.rsq.f32` (:346); `gl.sqrt` → `llvm.amdgcn.sqrt.f32` without refinement (:387-417)
- S23 hardware/gfx950/cdna4/ch12_11_vop3a_vop3b_instructions.md:2717 V_MED3_F32 returns `v_min3_f32` of its operands when one is NaN (:2717-2718), V_MIN3_F32 being `v_min_f32(v_min_f32(S0, S1), S2)` (:2655); V_CVT_SCALEF32_PK_FP4_F32 converts two values, round to nearest even, into one byte chosen by OPSEL[3:2], "Other destination bits are preserved" (:3915-3927)
- S24 triton-lang/triton@c01b677:python/triton/language/core.py:2956 `clamp(x, min, max, propagate_nan=PropagateNan.NONE)`; `maximum` (:2934) and `minimum` (:2912) take the same flag; `PropagateNan` (:23) is exported by `triton.language` (`python/triton/language/__init__.py:32`), not by `python/triton/experimental/gluon/language/__init__.py`
- S25 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2127 fp32 → E4M3FN on CDNA4 via `scalePk4DowncastToFp8<CvtScaleF32PkFp8F32Op>`, scale 1.0 (:561), RTNE only (:2111); an fp8 cast marks its function with `SetFP8ClampingAttr` (:2473-2481); fp32 → bf16 RTNE in packed pairs on CDNA4 (:106), RTZ a 16-bit shift (:73-77)
- S26 hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:795 FP16_OVFL rule for F32 → FP8/BF8 (table :798-808); F32 MFMA results "clamped to +/- MAX" when the bit is set (:115); Table 28: F32 16x16x4 takes 32 cycles (:81), BF16 16x16x32 takes 16 (:101); Table 30: FP4 E2M1 has no INF or NaN (:712); conversion to F4/F6 ignores FP16_OVFL (:772) and saturates (:780)
- S27 hardware/gfx950/cdna4/ch03_kernel_state.md:105 MODE.FP16_OVFL: "an overflowed FP16 result is clamped to +/- MAX_FP16"; MODE.IEEE (:102)
- S31 hardware/gfx950/matrix_core.md:138 e8m0 scale "valid values in: -127, 127 (0xFF is NaN)", applied "after the normal dot product prior to output/accumulation"; MFMAs force RNE and keep input and output denormals per the ISA, the F32-input `16x16x4` and `32x32x2` measured following MODE's f32 denorm field (:95-107)
- S32 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:383 `__HIP_FTZ = True`, not a kernel option (:376-382); `enable_fp_fusion` defaults to `knobs.language.default_fp_fusion` (:179-180; `TRITON_DEFAULT_FP_FUSION`, true, `python/triton/knobs.py:497`) and goes to `llvm.optimize_module` (:505); `allow_flush_denorm` sets `denormal-fp-math-f32` to `preserve-sign` (:473-474); `llvm_fn_attrs` are added to the kernel (:479-481)
- S33 hardware/gfx950/cdna4/ch12_08_vop1_instructions.md:482 V_EXP_F32: "1ULP accuracy, denormals are flushed."; V_LOG_F32 the same (:504), V_RSQ_F32 (:584) and V_SQRT_F32 (:638) too; V_RCP_F32: "Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA operations. Denormals are flushed." (:531)
- S38 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:68 both scales `None` → `uint8` 0x7F (:68-69), built as constant tensors (:99); `dot_scaled(..., fast_math=False, ..., out_dtype=ttgl.float32)` (:105-106)
- S42 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp:94 `bf16x3` splits both operands (:94-95) and sums mid·hi + hi·mid + hi·hi (:118-126); the pass runs in plain Triton's `make_ttgir` (`third_party/amd/backend/compiler.py:270`), not in `gluon_to_ttgir` (:327-348)
- S51 triton-lang/triton@c01b677:third_party/amd/language/hip/libdevice.py:79 `libdevice.exp` → `__ocml_exp_f32` (:82); `fast_dividef` → `__triton_hip_fast_fdividef` (:102-104)
- S57 hardware/gfx950/cdna4/ch12_07_vop2_instructions.md:185 V_MAX_F32: a signaling NaN is quieted only in IEEE mode; otherwise a NaN operand returns the other (:189-192); V_MIN_F32 the same (:158-165)
- S61 triton-lang/triton@c01b677:third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:454 `BufferStoreOp`: "value element type matches the pointed type of ptr"
