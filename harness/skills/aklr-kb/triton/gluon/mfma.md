---
title: "The matrix core from Gluon on gfx950: mfma, mfma_scaled and dot_fma"
kind: language-note
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
sources:
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_layouts.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py
  - triton-lang/triton@c01b677:python/triton/language/semantic.py
  - triton-lang/triton@c01b677:python/triton/knobs.py
  - triton-lang/triton@c01b677:third_party/amd/backend/compiler.py
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp
---

# The matrix core from Gluon on gfx950

Which Gluon call, `AMDMFMALayout` and `k_width` to use for each operand format, what Triton 3.8.0 turns them
into on gfx950, and how AKLR's 154 Triton/Gluon kernels feed operands, e8m0 scales and accumulators, use a
16-row instruction at small M and finish a tile. Layouts in general are in [layouts.md](layouts.md), data
movement in [memory.md](memory.md), quantization and rounding recipes in [../numerics.md](../numerics.md). The page-local Evidence retains source and measured conditions (S65).

## Route here when

- You choose between `gl.amd.cdna4.mfma`, `gl.amd.cdna4.mfma_scaled` and `gl.dot_fma`, or an `instr_shape` and
  `k_width`, for an operand format.
- An fp8 block-scaled or MXFP4 kernel must get its operands, e8m0 scales or preshuffled weights into the matrix
  core, or must write its activations in the format an MXFP4 consumer reads.
- A decode GEMM leaves most rows of the 16-row instruction idle.
- A matrix call fails to compile, or you want the instruction it became and its rate.

## 1. The instruction menu on gfx950 (`version=4`)

The accumulator's `gl.amd.AMDMFMALayout(version=4, instr_shape=[M, N, K], ...)` names the instruction; the
lowering looks the shape and the operand types up in a fixed table (S1). Counts are schemas of the 154 whose
AKLR kernels contain the instruction as compiled at their Points. MAC per cycle is estimated as the ISA atom's
M·N·K·B (B blocks, 16 for `4x4x4_16B`) over the cycles of Table 28, the MFMA opcode table of the CDNA4 ISA guide
([ch07 §7.1.2](../../hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md);
[matrix_core.md](../../hardware/gfx950/matrix_core.md) §4). TFLOP/s come from a Gluon microbenchmark that chains
MFMAs on register-resident operands with no memory traffic, not from a kernel.

| Gluon call, operands | `instr_shape`, `k_width` | gfx950 instruction | MAC / cycle, Table 28 | TFLOP/s, microbenchmark, range over 4 runs *(field, E1)* | Schemas *(field, E2)* |
|---|---|---|---|---|---|
| `mfma`, bf16 or fp16 | [16,16,32], 8 | `v_mfma_f32_16x16x32_bf16` / `_f16` | 512 | 2447–2468 (bf16) | 73 / 3 |
| `mfma`, bf16 or fp16 | [32,32,16], 8 | `v_mfma_f32_32x32x16_bf16` / `_f16` | 512 | 2435–2452 (bf16) | 7 / 1 |
| `mfma`, fp8 e4m3 | [16,16,32], 8 or 16 | `v_mfma_f32_16x16x32_fp8_fp8` | 512 | 2444–2468 (`k_width` 8): 1.0× bf16 (0.996–1.004) | 3 |
| `mfma_scaled`, e4m3 | [16,16,128] / [32,32,64], 16 (32 with `None` scales) | `v_mfma_scale_f32_16x16x128_f8f6f4` / `_32x32x64_` (checked with `None` scales; with scale tensors at [16,16,128], §3, 2.2) | 1024 | 4899–4934 / 4907–4949 (`k_width` 32): 2.0× bf16 (1.999–2.004; e4m3 × e2m1 1.998–2.005) | 75 / 12, both formats |
| `mfma_scaled`, e2m1 | [16,16,128] / [32,32,64], 16 (bytes) | the same two | 2048 | 9382–9591 / 9026–9341: 3.80–3.92× / 3.69–3.82× bf16, about 1950–2010 / 1890–1960 MAC per cycle, derived by multiplying the measured relative rates by bf16's Table-28 value of 512, not measured cycle counts | (counted above) |
| `mfma`, fp32 | [16,16,4] / [32,32,2], 1 or 4 | `v_mfma_f32_16x16x4_f32` / `_32x32x2_f32` | 32 | 155.2–156.1 / 154.8–156.0 (`k_width` 1): 1/15.7–1/15.8 bf16 | 38 / 8 |
| `mfma`, fp64, `element_bitwidth=64` | [16,16,4], 1 | `v_mfma_f64_16x16x4_f64` | 16 | valid `element_bitwidth=64` register-loop rate not established; the historical 1/31.5–1/31.7 ratio used a benchmark without that layout setting | 1 |
| `mfma`, fp16 | [4,64,64], 4 | `v_mfma_f32_4x4x4_16b_f16`, 16 per [4,64,64] call, matching float64 *(field, E6)* | 128 | 0.25× bf16 [16,16,32] (bf16 `[64, 4, 64]`) | 1 |
| `gl.dot_fma`, bf16 into fp32 | `BlockedLayout` accumulator, `k_width` 0 | `v_dot2c_f32_bf16` (VALU) | | not measured | 1 |

22 schemas use no matrix instruction (18 GDN decode, gated attention at n = 1 and 2, two RMSNorm); the two
plain-Triton bf16 decode GEMMs reach `v_mfma_f32_16x16x32_bf16` through `tl.dot` *(field, E2)*. The 84
`mfma_scaled` schemas split into 42 e4m3 (a8w8 GEMM 24, MiniMax-M2 fp8 MoE 9, DSA 64-head 5, Qwen2-MoE fp8
router+experts 4), every call with both scales `None`, and 42 e2m1 (DeepSeek-V2/GLM MXFP4 MoE 19, MiniMax-M2
MXFP4 MoE 23), every call with two scale tensors; no kernel uses e5m2 or mixes formats (read in the sources).

- **`instr_shape[2]` must equal a table entry's K, and it picks the entry.** Another K fails with "no matching
  matrix core intrinsic for mfma version 4 with instruction shape …" (§10; S1, S2). bf16 keeps the older [16,16,16]
  and [32,32,8] entries: they compiled to `v_mfma_f32_16x16x16_bf16` and `_32x32x8_bf16` and ran at half the rate
  (K = 32 / 16: 1.980–2.002× theirs, 4 runs, E1). fp16 new/old register-loop ratios in two runs (two passes each) were 1.978832–1.987862 (16×16) and 1.949911–1.988267 (32×32), not a guarantee for arbitrary kernels *(E16)*.
- **Plain `mfma` on fp8 stops at 16x16x32 and 32x32x16** (S1). The K = 128 / 64 F8F6F4 instructions are keyed as
  fp4 × fp4 for every format (S1) and Gluon reaches them only through `mfma_scaled`. Plain-Triton `tl.dot` gets
  them by itself through `AccelerateAMDMatmul` (S3), a pass Gluon's pipeline does not run (S4).
- **`mfma_scaled` with `None` scales keeps the scaled opcode**, which ran e4m3 at 2.0× bf16 *(field, E1)*. Gluon
  fills both scales with `0x7F` (E8M0 1.0, S5); an fp8 tile compiled to `v_mov_b32_e32 v0, 0x7f` and
  `v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[10:17], v[18:25], 0, v0, v0 op_sel_hi:[0,0,0]`.

`mfma_scaled` with None scales retains the scaled opcode and unit 0x7F scales. In valid single-dependency-chain tests, resident real scales and None were equivalent within the predeclared ±3% band in both formats (ratio about1.002). The opaque unscaled loop was not scheduling/unroll-matched, and four-chain None failed numeric checks; neither establishes a general equality of hardware cycles.
- **`gl.amd.cdna4.mfma` is the previous generation's `mfma`, re-exported** (S6): both module names give one function.
- **Register-resident chained MFMA microbenchmark:** bf16 and scaled e4m3 shapes had similar rates, scaled e2m1's
  32x32 was 2.4–4.4% slower (4 runs) *(field, E4)*; tune shape in the kernel, not for assumed equal speed (§4). The atoms leave
  different VALU shadow per MAC ([../../attention/mfma_layout.md](../../attention/mfma_layout.md) §1), and in a FlyDSL
  bf16 grouped GEMM at the power cap 16x16x32 beat 32x32x16 ([../../gemm/mainloop.md](../../gemm/mainloop.md) §5).

## 2. Picking the instruction per operand format

| Operands | Call and shape | Section |
|---|---|---|
| fp8 e4m3 × e4m3 with fp32 block scales | `mfma_scaled(a, None, "e4m3", b, None, "e4m3", zeros)` per 128-K group, scales folded in VALU | 2.1 |
| MXFP4 (e2m1 values, e8m0 per 32) | `mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)`, scales in `get_mfma_scale_layout` | 2.2, §3 |
| MXFP8 (e4m3, e8m0 per 32: the fused_moe contract's activations at `act_quant` 8, S60) against MXFP4 or MXFP8 | `mfma_scaled(a, sa, "e4m3", b, sb, "e2m1" or "e4m3", acc)`, `k_width` 16 on both; 1024 MAC/cycle with an F8 operand | 2.2, §5 |
| bf16 against MXFP4 (fused_moe at `gate_mode` 'interleave', S60) | no `mfma_scaled` ("Unsupported lhs_format: bf16"): quantize bf16 where the contract allows ([../numerics.md](../numerics.md) §1, §3), or `scaled_upcast` weights to bf16 then `mfma` [16,16,32]. Its register-instruction rate is 0.255–0.262× e2m1 (4 runs, E1); full-route timings and different numerical targets: E15 | 2.2 |
| e4m3 fp8 against bf16 or fp16 | promote fp8 exactly to the partner's type; bf16 or fp16 `mfma` [16,16,32]; do not round an fp16 partner to bf16 *(E14)* | 2.3 |
| bf16 × bf16 | `mfma` [16,16,32] or [32,32,16], `k_width` 8 | 2.4 |
| Products that must retain fp32 accuracy | fp32 `mfma` [16,16,4]; for fp32 × bf16 only, split the fp32 operand into three bf16 terms under the numerical conditions of §2.5 | 2.5 |
| no operand reuse: FMA + `gl.sum` 0.94–0.96× fp32 MFMA on 128×128 states (5 runs, E9); one/two attention rows: FMA + `gl.sum` 1.002/0.964× padded bf16 MFMA, `gl.dot_fma` 0.977/0.955× (one HBM-streaming shape, N=1, E15) | no-reuse `gl.dot_fma` and compute-bound crossover unmeasured | 2.6 |

### 2.1 fp8 e4m3 with fp32 block scales: one partial per 128-K group

For an a8w8 block-scaled GEMM (fp32 scale per 1×128 of A and per 128×128 of B), give each 128-K group its own
partial from a zero accumulator, `gl.amd.cdna4.mfma_scaled(aa, None, "e4m3", bb, None, "e4m3", zero)` on
`aa = gl.amd.slice(a, (M, 128), (0, GROUP * 128))` and `bb = gl.amd.slice(b, (128, BN), (GROUP * 128, 0))`
(`_group_dot`, S7: n7168 at M = 32), then fold it with the two scales in fp32. The fold's spellings and their
rounding: [../numerics.md](../numerics.md) §4, §6, [../gemm.md](../gemm.md) §2. With [32,32,64] a group is two
instructions: chain two calls on one zero-started partial, then fold once (`partial = mfma_scaled(a1, ..., partial)`;
`acc = gl.fma(partial * xs[:, None], ws, acc)`, S9).

- `gl.amd.slice(t, shape, offsets)` keeps the layout without moving data (S8). Two CPU runs × 2464 slices: source K=128, A/B, 16×16/32×32, bf16 kw8/16, e4m3/packed-u8 kw16/32. Of 1032 power-of-two shapes, 140 compiled and 892 failed, exactly matching shape/offset multiples of `(64 / nonK) · k_width`; all 1432 non-power-of-two shapes failed first. Packed-u8 K is bytes, not logical fp4 K *(E16)*.
  Failures included "Lane and warp dim basis must match between source and destination layout." and "No source register holds the element for destination index […]"; this is not an iff for arbitrary tensor shapes.
- The scales are arbitrary fp32 values and the instruction's scale operand is an E8M0 exponent
  ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §2), so they stay outside; one [16,16,128] instruction
  holds exactly one 128-K group for `k_width` 16 or 32 (§5), so its result is one block dot product.
- F8F6F4 has twice the MACs per cycle of plain fp8 16x16x32 (Table 28): 4899–4934 against 2444–2468 TFLOP/s,
  1.995–2.008× in 4 runs of the register-operand microbenchmark, unit scales *(field, E1)*; the cited evidence is a register-operand microbenchmark, not an AKLR-Point comparison of the two routes.
- Counter-signal: a rank-3 (batched) MFMA layout, on which `mfma_scaled` with `None` scales failed (§4, §10).
  MiniMax-M2 fp8's 8-route decode down runs plain fp8 `mfma` on a rank-3 [16,16,32] layout with the same
  per-group fold (S10). The E64 down runs plain fp8 [16,16,32] at 64 tokens and `mfma_scaled` [16,16,128] at 32
  (`NATIVE=m == 64`, S56); this dispatch alone does not identify why the instruction changes or isolate its performance effect at a fixed Point.

### 2.2 MXFP4: e2m1 operands with e8m0 scales

MXFP4 weights (e2m1 values, an e8m0 scale per 32) arrive with bf16 activations in AKLR's MoE schemas. The DeepSeek-V2 implementation cited in S11 quantizes its activations to MXFP4 internally (the recipe and the contract's margin: [../numerics.md](../numerics.md)
§1, §3) and feeds both packed operands and their scale bytes to the instruction, which decodes fp4 and applies the
scales in hardware, at 3.82–3.92× bf16 in 4 runs, timed with unit scales only *(field, E1)*. A compiled tile: §3; the
DeepSeek-V2 decode expert GEMM converts both operands and scales into their layouts at the call (S11). The rules:

- **Operands** are `uint8`, two e2m1 per byte along K: A is `[M, K/2]`, B is `[K/2, N]` (S12). `run` passes fp4
  weights and e8m0 scales as `.view(torch.uint8)` (S11): Triton 3.8.0's type table has no entry for them (S62).
- **Scales** are `uint8` E8M0: A's `[M, K/32]`, B's `[N, K/32]` (N first, unlike B). Tracing checks the shape only
  (S12): a scale built as `[K/32, M]` fails with "scale factor must be 16 or 32. Got <K/M>" (or "Operands must have
  the same scale factor" where K/M is 16), unless M = K/32 (16 rows at K = 512 per call); then it compiles and is
  wrong, relative error 0.84 *(field, E5)*. Index scales in the scale layout's slices, rows on dim 0.
- **Scale layout**: `gl.amd.cdna4.get_mfma_scale_layout(operand_layout, scale_shape)`, factor 32 only (S6). For
  16x16x128, lanes 0–15 are rows and lane bits 4–5 the four 32-groups of one instruction; registers hold the next
  K steps, then `tiles_per_warp` repeats (S13; printed for a `[16, 16]` scale: registers `[0, 4], [0, 8]`, lanes
  `[1, 0] … [8, 0], [0, 1], [0, 2]` as (row, group)). It matches the ISA's scale lane map
  ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §2). The lowering packs up to four scale bytes in one
  VGPR, K steps first (S14), and selects one per instruction with `op_sel` (compiled at K = 512 on one tile, §3).
- **One wrong linear scale layout compiled silently.** At K = 512, a `[16, 16]` `DistributedLinearLayout` with lane
  bits on the K groups first was wrong, relative error 1.08 *(field, E5)*; this does not rule out every other layout.
  The lowering asserts linearity (`BlockedLayout` failed, exit 1) and reads registers in operand repetition order (S14).
- **A missing scale is 1.0.** Gluon fills a `None` scale with E8M0 `0x7F` (S5):
  `mfma_scaled(a, None, "e2m1", b, sb, "e2m1", acc)` scales B only (compiled with Triton 3.8.0 for gfx950, A's
  scale `v_mov_b32_e32 v1, 0x7f`); a Python `int` scale becomes an E8M0 constant, a `float` an fp8 one.
- **`0xFF` is NaN** in E8M0 ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §2): a 0xFF row of scales gave
  a NaN row *(field, E5)*. Load masked scale bytes as 127 (2^0), masked words as `0x7F7F7F7F` (2139062143, S15).
- **Mixed formats.** `a_format` and `b_format` are independent: e2m1, e4m3 or e5m2 each (S6). An e4m3 × e2m1 tile with
  real scales compiled to one `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:4` (the transposed layout puts the e2m1 operand
  first), at the F8 rate of 1024 MAC per cycle (Table 28; timed with `None` scales, §1). Give an fp8 operand `k_width`
  16 when its scales are real or its partner is e2m1 (§5), the layout Triton's own scaled-dot lowering builds for every
  operand (S3). No AKLR kernel runs a mixed or a real-scale fp8 call (§1); one of each matched float64 *(field, E5)*.
- **Upcast instead**: `gl.amd.cdna4.scaled_upcast(w, s, gl.bfloat16, axis=1)`, `s` raw E8M0 bytes expanded to the result's shape and canonical unpacked result layout implied by `w` and `axis` (S6); expand each 32-element group scale in that ownership and verify nonuniform-group scale packing, compiled to `v_cvt_scalef32_pk_bf16_fp4`; e2m1 · 2^(s−127) was bitwise with row-constant `s`, 56% right when `s` changed every two elements *(E5)*. A bf16 `mfma` follows; no AKLR kernel uses it.
  One 4096×4096-weight probe: upcast GEMM / (quantize + scaled GEMM) = 0.823/0.830/0.863 at M=1/2/16, 1.492/1.469/1.496 at M=128/1024/4096. These routes compute X·dequant(W) versus quant(X)·dequant(W), not identical targets; quantizing changed values by up to 0.247 of group max *(E15)*.

### 2.3 fp8 against a 16-bit operand: promote to the partner's type

For e4m3 fp8 against bf16, convert fp8 with `.to(gl.bfloat16)` and use bf16 [16,16,32]; against fp16, convert fp8 to fp16 and use fp16 MFMA. Finite E4M3FN is exact in both, but rounding the other fp16 operand to bf16 loses bits: two runs, eight seeds each, both operand orders, K=32 gave 2034/2048 and 2032/2048 outputs outside the fp32 summation bound, while fp16 stayed within it *(E14)*.
The bf16 convert is two `v_cvt_scalef32_pk_bf16_fp8` per four values with scale 1.0 (S16). Fold block scales as in 2.1; Qwen3.5 uses `key = gl.convert_layout(gl.permute(key, (1, 0)), b_layout).to(gl.bfloat16)` then bf16 `mfma` (S17).

- Load the fp8 data in a layout whose per-lane run equals the bf16 operand's, so the convert stays in registers:
  DSA 8-head reads K from LDS into `BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0])`, converts, and hands it to
  `DotOperandLayout(1, mma, 16)` (S18). That hand-over passes `assert_trivial=True`, which makes
  `gl.convert_layout` raise at compile time if the conversion would move data between lanes or waves (§7), and the
  convert is `v_cvt_scalef32_pk_bf16_fp8` (compiled with Triton 3.8.0 for gfx950).
- When P·V needs more than bf16's 8 significant bits and V is fp8, run both in fp16: 11 bits for P ∈ [2^-14, 1], V
  exact (gated attention at n = 4, S19); bf16 P passes the attention bounds ([../attention.md](../attention.md) §10).
- Where: a bf16 or fp16 MFMA runs in 7 gated-attention, 3 DSA 8-head, 5 DSA 64-head (on [32,32,16]) and 3
  Qwen2-MoE fp8 expert schemas *(field, E2)*; S17, S18 and S35 show examples of promoted-fp8 operands, not a complete operand-origin audit of every schema in those families.

### 2.4 bf16 × bf16

`[16,16,32]` or `[32,32,16]` with `DotOperandLayout(i, mma, 8)`: eight consecutive K per lane is the
instruction's kBase, the K elements one instruction takes from each lane (§5; S1). Routers, Gram products and P·V
use it, e.g. the DeepSeek-V2 decode router (S20).

### 2.5 fp32-accurate products

- **State and solves**: fp32 operands on `[16,16,4]` or `[32,32,2]`, `k_width` 1, or 2 and 4 in some GDN prefill kernels
  (4 makes a transposed accumulator the next A operand, §6); all 26 GDN prefill schemas run fp32 MFMAs *(field, E2)*.
  The rate is 32 MAC per cycle, 1/16 of bf16 (Table 28), measured 1/15.7–1/15.8 in 4 runs *(field, E1)*; keep it for the
  algebra that must stay fp32. Qwen3-Next's helper converts both operands to `DotOperandLayout(i, mma, 1)` (S21).
- **fp32 weight × bf16 activation**: split only the fp32 operand into three bf16 terms and run three bf16 MFMAs
  into three accumulators; a bf16 × bf16 product in fp32's normal range is exact in fp32 *(field, E5)* (MiniMax-M2
  router, S22). Why it is exact and how to sum the terms: [../numerics.md](../numerics.md) §5.
- **Not `input_precision`.** Gluon's `mfma` reads `TRITON_F32_DEFAULT` (S23) into every call: `tf32` made Gluon
  `mfma` on fp32 and fp64 operands fail when traced ("input_precision must be one of ('ieee', 'bf16x3', 'bf16x6').
  Got tf32", S12; compiled with Triton 3.8.0 for gfx950), and `bf16x3` / `bf16x6` compile and change nothing: the
  pass that implements them runs only for plain Triton (S4) and the MFMA lowering reads only TF32 (S2).
- **fp64**: `element_bitwidth=64` on `[16,16,4]` with `.to(gl.float64)` operands gives `v_mfma_f64_16x16x4_f64`
  (Qwen2-MoE fp8 router+experts, S24).

### 2.6 When to skip the matrix core

- **No operand reuse.** GDN decode kernels use no MFMA: 0 `v_mfma` in the compiled kernels of all 18 decode
  schemas *(field, E2)*; the elementwise products and `gl.sum` on each (token, head)'s 128×128 fp32 state: S25.
- **One or two attention rows.** Gated attention at n = 1, 2 compiles to no MFMA; seven files from n = 4 hold bf16/fp16 MFMA *(E2)*. Source: `gl.sum(keys * query[None, :], axis=1)` (S26); in one matched HBM-streaming probe FMA + sum took 95.0/106.4 µs versus padded bf16 MFMA 94.8/110.4 µs (R=1/2), so no gain at R=1. MFMA was ≥5% faster only from R=8 among R=1/2/4/8/16 *(E15)*; fp16 P·V at n=4: 2.3.
- **`gl.dot_fma`** takes a `BlockedLayout` accumulator and operands in `DotOperandLayout`s of it; `k_width` must be
  0 (S27, §10). bf16 into fp32 lowers to `llvm.amdgcn.fdot2.f32.bf16`, fp16 to `fdot2` (S28): a bf16 router dot
  compiled to `v_dot2c_f32_bf16` (accumulating into its destination; `v_dot2_f32_bf16` is the three-operand form),
  two products per lane per instruction (compiled with Triton 3.8.0 for gfx950). The DeepSeek-V2 two-token router
  puts one `[1, 1, 8]` × `[1, 8, 1]` dot per lane on a 3-D layout and shares each weight load by both tokens (S11).
- **MXFP4 at one token** still uses `mfma_scaled` in DeepSeek-V2/MiniMax-M2 *(E2)*. One 4096×4096-weight probe measured scaled MFMA versus VALU dot after hardware `scaled_upcast`: 14.27/14.04 µs at M=1 (ratio 1.016, not a win), 14.48/19.52 µs at M=2 (0.742). N=1; software-emulated decode and the reason for the difference were not tested *(E15)*.

## 3. One MXFP4 tile, compiled

A 16×16 output tile with K = 128 (one instruction), operands and scales loaded straight into their layouts.
Compiles with Triton 3.8.0 for gfx950 (`num_warps=1`, pointers 16-byte aligned, as the JIT detects at launch):

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def mxfp4_tile(x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr):
    # x: [16, 64] bytes (16 rows x 128 fp4), xs: [16, 4] e8m0; w: [16, 64] bytes (row = output feature), ws: [16, 4]
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                             warps_per_cta=[1, 1])
    a_l: gl.constexpr = gl.DotOperandLayout(0, mma, 16)     # 16 bytes = 32 fp4 per lane
    b_l: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    as_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(a_l, [16, 4])   # [M, K // 32]
    bs_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(b_l, [16, 4])   # [N, K // 32]
    m = gl.arange(0, 16, gl.SliceLayout(1, a_l))
    ka = gl.arange(0, 64, gl.SliceLayout(0, a_l))
    a = gl.load(x_ptr + m[:, None] * 64 + ka[None, :])             # [M, K // 2] uint8
    kb = gl.arange(0, 64, gl.SliceLayout(1, b_l))
    n = gl.arange(0, 16, gl.SliceLayout(0, b_l))
    b = gl.load(w_ptr + n[None, :] * 64 + kb[:, None])             # [K // 2, N] uint8
    sm = gl.arange(0, 16, gl.SliceLayout(1, as_l))
    sk = gl.arange(0, 4, gl.SliceLayout(0, as_l))
    sa = gl.load(xs_ptr + sm[:, None] * 4 + sk[None, :])            # [M, K // 32] uint8
    sn = gl.arange(0, 16, gl.SliceLayout(1, bs_l))
    sj = gl.arange(0, 4, gl.SliceLayout(0, bs_l))
    sb = gl.load(ws_ptr + sn[:, None] * 4 + sj[None, :])            # [N, K // 32] uint8
    acc = gl.amd.cdna4.mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", gl.zeros([16, 16], gl.float32, mma))
    om = gl.arange(0, 16, gl.SliceLayout(1, mma))
    on = gl.arange(0, 16, gl.SliceLayout(0, mma))
    gl.store(out_ptr + om[:, None] * 16 + on[None, :], acc)

# mxfp4_tile[(1,)](x, xs, w, ws, out, num_warps=1)   # x, xs, w, ws: uint8 tensors; out: fp32
```

The memory and matrix instructions of the ISA it became (address arithmetic and the store omitted; 18 VGPRs, no LDS):

```
global_load_dwordx4 v[10:13], v[2:3], off
global_load_dwordx4 v[14:17], v[2:3], off
global_load_ubyte v8, v[8:9], off
global_load_ubyte v0, v[0:1], off
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[14:17], v[10:13], 0, v0, v8 op_sel_hi:[0,0,0] cbsz:4 blgp:4
s_nop 7
```

- Each lane loads its 16-byte operand run (`k_width` 16 = 32 fp4) and one scale byte per operand.
- `cbsz:4 blgp:4` encode E2M1 for A and B ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §2); the `0` is
  the zero accumulator as an inline constant; `transposed=True` makes the lowering swap A and B at the
  instruction, with their scales (S29).
- `s_nop 7` is LLVM's wait before the result is read ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §3).
- With K = 512 per call (both variants compiled with Triton 3.8.0 for gfx950): four instructions chained on
  `v[0:3]`, each operand's four scale bytes in one VGPR, picked per instruction by `op_sel` / `op_sel_hi`.
  Row-major `[16, 16]` scales took four `global_load_ubyte` per operand (a lane holds groups g, g+4, g+8, g+12 of
  its row); stored with those bytes adjacent (`row * 16 + g % 4 * 4 + g // 4`), one `global_load_dword`.

## 4. Instruction shape, warps and tiles

- **16x16 or 32x32.** Similar register-microbenchmark rates except e2m1 *(field, E4)*; kernel tuning: §1. A 32x32 tile holds 16
  accumulator VGPRs per lane against 4, and its F8 result is readable 20 wait states after the instruction against 12
  ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §3). A whole BM × BN tile costs a lane BM·BN / (64 ·
  num_warps) fp32 registers (estimated); the compiled counts and what `waves_per_eu` does to them:
  [../tuning.md](../tuning.md) §8, [../debugging.md](../debugging.md) §2. Under a `bitwise` contract the shape alone
  did not change bf16 `k_width` 8 sums in 64 random trials (§5, [../numerics.md](../numerics.md) §8).
- **Who uses which.** [16,16,128] in 75 schemas, [32,32,64] in 12 *(field, E2)*: a8w8 n3072 at M = 256 and n7168
  at M = 128 and 256; DSA 64-head at 12, 16, 24 and 32 tokens (4 of its 5 files); DeepSeek-V2 `1204`, `3b66`,
  `b058` (256, 4096 and 4192 tokens); MiniMax-M2 `f770` (fp8, 1024–16384 tokens) and `bb66` (MXFP4, 1024). Tile
  shapes and the kernel that carries it are read in the sources (a8w8 on four warps `[2, 2]`, S9; DSA's score
  layout, S58; DeepSeek-V2 row blocks other than 16, S30); `[INSTR_M, INSTR_M, 2048 // INSTR_M]` gives both (S31).
- **`warps_per_cta`** spreads warps over M×N instruction tiles, N first; a warp's further N tiles sit in
  registers `warps_per_cta[1]` tiles apart, and `tiles_per_warp=[1, 2]` makes two of them adjacent (S37).
- **K per call, tested plain MFMA:** bf16/fp16/e4m3/e5m2, new 16×16/32×32 and old 16-bit shapes, swept `k_width` 4/8/16 ≥ kBase: at `extent`, `2extent`, `4extent` (`extent = k_width · 64 / nonK`) chained without rescaling; at `extent/2` or `/4` rescaled (K ≥ 4, two runs × 138 variants, E13; §10; S34, S50).
  This is not a universal compile boundary: below kBase, even K = extent can fail; small scaled K can fail to compile or give wrong results (§10).
- **M or N below the instruction compiles** (e.g. bf16 [4,16], [1,16], e4m3 `mfma_scaled` [4,16]; `get_min_dot_size`
  is (1, 1, 1) on HIP, S4, S12). Rows repeat across lanes and the instruction runs in full: a [4,16] tile was correct
  *(field, E6)* and saved no instruction against [16,16]. §8 puts the spare rows to use.
- **Rank-3 (batched) layouts** (`warps_per_cta=[B, 1, 1]`) give each warp one slice of a leading axis: a K split
  inside the CTA, reduced with `gl.sum(acc, 0)` (MiniMax-M2 fp8 gate-up: `[8, 1, 1]` at 64 tokens, S55;
  `[4, 1, 1]` at 1–2 tokens, §8), or one route per warp (the 8-route decode down, S10). `mfma_scaled` (`None` scales)
  failed on them (§10), so fp8 runs plain [16,16,32] at half the F8F6F4 theoretical MACs per cycle; the cited rank-3 compilation and register-rate evidence does not isolate that cost in a complete kernel.
- **4x64 and 64x4**: `[4, 64, K]` must be `transposed=False` and `[64, 4, K]` `transposed=True`; the lowering
  broadcasts the narrow operand to 16 blocks (S29). Qwen2-MoE fp8 router+experts uses `[4, 64, 64]` in fp16 (S24).

## 5. `k_width`

Each lane holds `k_width` consecutive K elements of one row (A) or column (B); lanes cover the 16 or 32
non-K indices, then K; past `(64 / nonK) · k_width` the registers repeat along K (S32). For plain `mfma` that lane
sweep is the operand's K extent, the smallest K per call that does not rescale (§4, §10). The docstring's "Number
of elements per 32-bits" (S33) does not describe the AMD layout.

- **Plain `mfma`**: new bf16/fp16 [16,16,32]/[32,32,16] srcA is 4 VGPRs (8 elements), fp8 16x16x32 2 VGPRs (8), fp32 16x16x4 1; old fp16 shapes use 2 VGPRs (4 elements). Two runs at kw16, K=2instrK compile to two same-acc instructions without rescale, but did not verify consecutive srcA index segments *(E16; lowering S34)*. In random bf16 trials `k_width` 8 against 16
  differed in 64 of 64 (up to 1.6 ulp), 16x16x32 against 32x32x16 in none *(field, E5)*.
- **`mfma_scaled`** hands each instruction kBase elements per lane whatever the layout's `k_width` (S34). With fp8
  `k_width` 16 or 32 and `None` scales, each [16,16,128] instruction still receives one whole 128-K block: a K = 256
  call matched two chained K = 128 calls bitwise in 64 of 64 trials *(field, E5)*.
- **A and B must agree.** Plain `mfma` rejects a mismatch; `mfma_scaled` compiled one (A 16, B 32) and was wrong,
  error 1.2× the largest output. For `mfma_scaled` give fp8 `k_width` 16 and packed e2m1 16 bytes: all nine format
  pairs with real scales then matched the reference *(field, E5)*.

| Operand | `instr_shape` | `k_width` | Note |
|---|---|---|---|
| bf16, fp16 | [16,16,32], [32,32,16] | 8 | Direct-load kw16: no rescale at K=64/128/256 (16×16), 32/64/128 (32×32); half/quarter thresholds rescale acc. Two runs; 16-byte fp8 in-place conversion (S35) was not tested by this case *(E13, E16)* |
| fp8, plain | [16,16,32] | 8 or 16 | e4m3/e5m2, two runs: no rescale at kw8 K=32/64/128 and kw16 K=64/128/256; half/quarter thresholds compile but rescale acc *(E13)* |
| fp8, scaled | [16,16,128], [32,32,64] | 16; 32 only with `None` scales | 16 puts k 0–15 and 64–79 in lanes 0–15 (16x16x128; lane group q: 16q … 16q+15 and 64+16q … 64+16q+15), the order with which real scales and an e2m1 partner gave correct results (2.2), and Triton's choice for `tl.dot` and `tl.dot_scaled` (S3, S13). 32 (lane group q holds k 32q … 32q+31) compiles to the same single instruction; the second 16-byte load moves from `offset:64` to `offset:16` and the address code changes (42 instructions against 40). With unit scales on both fp8 operands it only regroups the products; with real E8M0 scales the error was 0.80× the largest output, against e2m1 1.12× *(field, E5)* |
| e2m1 bytes | [16,16,128], [32,32,64] | 16 | CPU layout maps and ISA: 16 bytes = 32 logical fp4 per lane on both shapes; on 16x16x128, lane group q holds K 32q … 32q+31, matching the scale group. GPU nibble order was not tested by this check *(E11)* |
| fp32 | [16,16,4], [32,32,2] | 1 or 4 | with `transposed=True` and the waves on M only, 4 makes the 16×16×4 accumulator the next A operand without moving data (§6); 1 does not, nor 4 on [2, 2] or [1, 4] waves |
| fp64 | [16,16,4] | 1 | with `element_bitwidth=64` |

`gl.amd.cdna4.compute_efficient_padded_shared_layout` returns `None` unless `k_width` is 4, 8 or 16 (S6): a
kernel that reads with 32 asks it with a 16-wide operand layout (S36). LDS layouts: [memory.md](memory.md) §9.

## 6. Accumulators

- **`transposed=True`** (a literal in 122 schemas, `False` in 24, only a helper or constexpr, with neither literal, in 7): a lane holds four
  consecutive columns of one row (16x16: row l % 16, columns 4·(l / 16) … +3), so a row's values are contiguous
  for stores and row reductions; `False`: four consecutive rows of one column (S37). Take `False` where a lane must
  own several rows (rows 2h, 2h+1 of 16x16): the small-M row selections of §8 (S38) and `[4, 64]` atoms (§4).
- **The layout does not depend on the instruction K** (S37): a one-wave transposed bf16 [16,16,32] result had the
  fp32 [16,16,4] linear layout and converted to it with `assert_trivial=True` and no LDS (Triton 3.8.0, gfx950), as
  the Qwen3-Next Gram products do before the triangular solve (S21).
- **Zero start**: a `gl.zeros(shape, gl.float32, mma)` passed straight to the call becomes that instruction's
  inline constant 0 (§3) with no `v_mov_b32` fill (bf16 [16,16,32]); one carried into a dynamic `range` loop is
  filled by `v_mov_b32` first (both compiled with Triton 3.8.0 for gfx950).
- **Chain** by passing the running accumulator. It is an addend: online softmax passes the rescaled accumulator,
  `numerators[panel] * alpha[:, None]` (S18), GDN prefill a loaded bias (S57). fp8 block scales instead need a
  fresh zero per 128-K group (2.1); keeping their fold off the MFMA's critical path (deferred folds, two
  accumulators, an accumulator pin): [../gemm.md](../gemm.md) §5, [../tuning.md](../tuning.md) §10.
- **An accumulator as the next A operand.** On one wave or waves on M only, a transposed fp32 [16,16,4] accumulator already is `DotOperandLayout(0, mma, 4)`: `assert_trivial=True` passes (not on [2, 2] or [1, 4] waves).
  With `k_width` 1 the assert fails: one 16×16 tile takes 1 KB LDS, 4 `ds_write_b32` + 4 `ds_read_b32`. Saving that conversion does not guarantee faster chains: in 256-step MI355X chains, kw4/direct-kw1 time ratios were 0.664/1.018 (1-wave 16×16), 0.893/1.313 (4-wave 64×16), 0.905/0.980 (1-wave 64×64), at 4/16 programs per CU *(E12)*.
  Where 1 stays, relabelled K or two permlane steps can avoid LDS *(E6)*; in the same first four settings relabel/direct=0.661/0.987/0.901/1.332 and two-step/direct=0.892/1.259/1.137/1.212. Choose by shape and grid, not by absence of LDS *(E12)*.
  Relative error versus fp64 was ≤1.6e-6 for relabel/two-step and ≤2.587e-6 for the kw4 comparison; N=8 balanced repeat clusters in one GPU run, not eight independent runs. See [../gdn.md](../gdn.md) §11 and [layouts.md](layouts.md) §7. P is rounded to bf16 before conversion (S17).
- **fp32 order**: inside an instruction the hardware's order and rounding
  ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §2); across instructions the order of the calls and of K
  in registers (§5). Fusing the scale fold into FMAs, and bitwise questions: [../numerics.md](../numerics.md) §6, §8.

## 7. Operand formats in memory

All four ways AKLR's kernels load an operand with no conversion: [layouts.md](layouts.md) §6.3; streaming weights
and scales: [memory.md](memory.md) §13. Below, KB is the bytes of a stored weight row (K for fp8, K/2 for fp4), KG
the stored 32-groups per row, padded to a multiple of 8 (`E8M0Scales(..., pad_to=(256, 8), shuffled=True)`, S61).

**Preshuffled weights.** AKLR's `Preshuffle(tile=(16, 16))` (S39; not compared with aiter's `shuffle_weight`): byte
`(n, k)` of expert `e`'s weight is at `e·N·KB + n//16·KB·16 + k//16·256 + n%16·16 + k%16` (`_weight_offset_3`, S40),
so a lane's 16-byte `k_width`-16 run (16 fp8 or 32 fp4 of a row) is one aligned vector; 16 rows are 256 bytes.

**Shuffled e8m0 scales hold a lane's bytes in another order.** `E8M0Scales(..., shuffled=True)` is aiter's
`shuffle_scale` (S39): byte `(n, g)` is at `(n//32·KG/8 + g//8)·256 + g%4·64 + n%16·4 + g//4%2·2 + n//16%2`, plus
`e·N·KG` per expert (`_scale_offset_5`, S40), so one 32-bit word holds rows n, n+16 × groups g, g+4. When one warp
holds 32 rows and 256 K of an operand (two tiles, two K steps), the scale layout puts the same four bytes in a
lane's registers in the order (n, g), (n, g+4), (n+16, g), (n+16, g+4), K first (printed `[32, 8]` layout:
registers `[0, 4], [16, 0]`; S13), and the lowering packs them into one VGPR in that order (S14).

**Simplest: index the stored bytes in the target layout.** Build `n` and `k` from `gl.arange` in the `SliceLayout`s
of the operand or scale layout and load: `b = gl.load(W + _weight_offset_3(e, n[None, :], kb[:, None], N, KB))` gave
one `global_load_dwordx4` per 16-byte run (fp4 bytes); `sb = gl.load(base + _scale_offset_5(0, sn[:, None],
sg[None, :], N, KG))` on one `[32, 8]` tile (expert 0, one K step), one `global_load_dword` and one `v_perm_b32` per
lane. With the expert term per lane, or through `gl.amd.cdna4.buffer_load`, the four scale bytes became four byte
loads (all compiled with Triton 3.8.0 for gfx950). The committed kernels use:

- load in a layout that is the operand layout and prove it with `assert_trivial=True` (S41, S42), or build the
  indices from `gl.to_linear_layout(DotOperandLayout(...), shape)` slices (S40);
- load each scale's enclosing word and shift out its byte, `bx = (scale_words >> scale_shift).to(gl.uint8)` with
  `scale_shift = sn[:, None] // 16 % 2 * 8 + sg[None, :] // 4 % 2 * 16`, then convert (DeepSeek-V2 decode, S11);
- load whole scale words, split each into its four bytes with two `gl.join` levels and permute them back to
  `[BN, BK // 32]` (`_unpack_bytes`, a 6-D permute, S41); only reshaped, relative error 0.94 *(field, E5)*;
- one word, two K steps: it holds groups g and g + 4, so load it on even K steps and shift by 16 on odd ones (S59);
- staged in LDS, when several MFMAs reuse a tile's A scales: give the `SharedLinearLayout`'s lowest offset bits to
  the scale layout's register bits (groups +4, +8), then rows, then groups +1, +2. `.load(scale_layout)` then reads a
  lane's four bytes with one `ds_read_b32` instead of four `ds_read_u8` (compiled with Triton 3.8.0 for gfx950; S15).

**Producers write the format their consumer loads.** For aiter's MXFP4 GEMM, which `gemm_a4w4_mxfp4@1.0`'s
`b_shuffled` variant models, that is codes row-major (`[m, k/2]` bytes) and scales as aiter's
`dynamic_mxfp4_quant(shuffle=True)` writes them, `E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)` (S64):
rows padded to a multiple of 256, groups to 8, then shuffled (S39). `E8M0Scales` pads with `0x7F`, aiter mostly with
0, and 0xFF row padding left aiter's `gemm_a4w4` output unchanged *(field, E5)*: writing every pad byte is a safe
default, not a measured need. Inside a plugin the file picks: 147b (MiniMax-M2 MXFP4 E256, 2048 and 4096 tokens)
stores the activation in the weights' 16×16 shuffle with shuffled scales, so its down GEMM loads A like B; d3bb stores
both row-major (S15, S11). 147b writes padding rows' scales as 127 at 4096 tokens, where its down GEMM reads whole
words unmasked, but not at 2048, where that GEMM masks them (S15).

**Whole scale words need aligned 32-row × 8-group blocks.** In `get_mfma_scale_layout`'s `[32, 8]` layout (registers
`[0, 4], [16, 0]`, above), `gl.store` of a uint8 tile at `_scale_offset_5` offsets, pad rows filled by
`gl.where(row < R, s, 127)`, became one `global_store_dword` per lane. A row mask, `buffer_store` (unless packed into
a `uint32`) and 16-row tiles stored bytes, 32 × 4 tiles `global_store_short`, 147b's 64 × 2 and 64 × 4 tiles a byte
per lane; converting a row-per-lane tile cost an LDS round trip (all compiled with Triton 3.8.0 for gfx950).

**From LDS.** Compiled bf16/fp16 `[64,64]`, [16,16,32] kw8, `SwizzledSharedLayout(1,1,1,[1,0])`, waves [2,2]: N-contiguous B reads 8 `ds_read_b64_tr_b16`, A 4 `ds_read_b128` (fp16 repeated twice); the old fp8 `[64,128]` B reads 8 `ds_read_b64_tr_b8` (once, E16; maps: [memory_and_sync.md](../../hardware/gfx950/memory_and_sync.md) §4). These are layout/tile-specific instances. 17 schemas' kernels contain
`ds_read_b64_tr_b8`: DSA 8, gated attention 6, DeepSeek-V2 MXFP4 2, MiniMax-M2 MXFP4 1 *(field, E2)*. LDS layouts
and async copies: [memory.md](memory.md) §8, §9.

## 8. Small M on a 16-row instruction

A decode GEMM with M ≤ 8 fills few of the 16 rows, and the instruction runs in full either way (§4). AKLR's
kernels choose what the spare rows carry. Each design is part of a whole kernel; the speedups are that kernel's
at its Points and do not isolate the trick:

| Design | Rows and columns of the instruction | Family and Points |
|---|---|---|
| Fold K groups into the rows | rows (token, K group), columns (feature, K group); keep the diagonal blocks | a8w8 n3072 and n7168 at m = 1–8 (files and values per Point: [../gemm.md](../gemm.md) §3), e.g. n3072 m = 1: 1.429, n7168 m = 8: 1.840 *(field, E3)*; MiniMax-M2 MXFP4 E256, n = 1: 3.444 *(field, E3)*; MiniMax-M2 fp8 E256, n = 4: 2.452 *(field, E7)* (S10) |
| K packed into the rows, split-K over warps | rank-3 [16,16,32], `warps_per_cta=[4, 1, 1]`: each warp takes a quarter of K; rows alternate two K chunks of the token, columns hold 8 gate + 8 up features per chunk; keep the matching blocks, then `gl.sum` over the warps; plain fp8 (§4) | MiniMax-M2 fp8 E256, n = 1, 2: 4.115, 3.755 *(field, E7)* (S10) |
| Route block diagonal | rows: one token's 8 routes; columns: 8 experts' features side by side; keep block r of row r | MiniMax-M2 MXFP4 E256 n = 4, 8, 16: 2.549, 1.977, 1.595 *(field, E3)* (S43) |
| Replicate | 16 copies of one route's row, store row 0; or one loaded row broadcast in registers (`gl.broadcast(arow[None, :], gl.full((16, 128), 0, ..., al))`), keeping one of a lane's four accumulator registers | DeepSeek-V2 MXFP4 inter 256, n = 2: 2.186 *(field, E3)* (S11); MiniMax-M2 fp8 E256, n = 8, 16: 1.841, 1.732 *(field, E7)* (S10) |
| Swap A and B | weights as A (output features on the 16 rows), the token broadcast over the 16 columns | Qwen2-MoE fp8 experts, n = 8: 1.208 *(field, E7)* (S44); MiniMax-M2 fp8 E64 down, n = 1, 2, 4: 3.137, 2.912, 1.672 *(field, E3)* (S52) |
| P's bf16 residual in the spare rows (attention) | 8 heads: interleave the query rows (head = row // 2) so QK computes each head twice; `transposed=False` puts rows 2h and 2h+1 in one lane, so one softmax serves both; bf16(P) in row 2h and bf16(P − bf16(P)) in row 2h+1 go through one P·V MFMA, and the pairs are summed after | Qwen3.5 gated attention, 8 rows: 1.803; DSA 64 heads, 8 tokens (high and residual P of 16 heads in one [32,32,16] MFMA): 1.030 *(field, E3)* (S53, S54) |

Why the residual pays: [../numerics.md](../numerics.md) §5, [../attention.md](../attention.md) §10. The n3072 fold
(`PG = 4` groups, `BN = 4` features per program), its loop and compiled kernel: [../gemm.md](../gemm.md) §4.

- **Keep the selection in registers.** The n3072 kernels put the K group in the low column bits (`transposed=True`),
  the n7168 kernels in the low row bits, picked with `gl.amd.slice` plus an `assert_trivial` convert (S38); a column
  group sum and a `gl.amd.slice` row pick compiled with no LDS or cross-lane permute. The compiled n3072 m = 1 kernel
  holds 3 scaled MFMAs and 40 VGPRs, 8 of them AGPRs; it picks the diagonal with 10 `v_cndmask_b32`, and the final
  sum over the four groups, whose rows sit in different lanes (S45), takes 2 `v_add_f32_dpp`; there is no
  `ds_bpermute` or `v_permlane` *(field, E2)*.
- **MXFP4 variant**: with the scales inside the instruction there is no per-group fold. MiniMax-M2 at n = 1 splits
  16 rows into `FOLD` partitions (the launch's `gate_fold`) that read different K chunks against matching column
  blocks, chains the accumulator over all K, and gathers the diagonal once at the end (S46).
- **The constants are coupled.** In that kernel `FOLD` 4 → 1 alone, at 4 warps and a 128-wide K tile, fails
  validation with `candidate_crash`, exit −11, in 4 of 4 rounds *(field, E8)*: the Triton 3.8.0 compiler segfaults in
  LLVM's code generation (compiled with Triton 3.8.0 for gfx950). Which combinations compile:
  [../moe.md](../moe.md) §7, [../debugging.md](../debugging.md) §6. Change the constants together.
- For a bf16 plain-Triton decode GEMM, a 16-row tile carrying one row lost to dedicated M = 1 GEMV kernels
  ([../gemm/decode_gemm.md](../gemm/decode_gemm.md) §7); the kernels above differ in operand format, loading and
  fusion. Decode GEMM and MoE design: [../gemm.md](../gemm.md) §3, §4, [../moe.md](../moe.md) §7, §9.

## 9. Epilogues

- **fp32 to bf16, then the store.** `.to(gl.bfloat16)` rounds to nearest even with the packed `v_cvt_pk_bf16_f32`;
  `fp_downcast_rounding="rtz"` keeps the fp32 bits >> 16 (S16) *(field, E6)*. Casting before `gl.convert_layout`
  halved a 64×64 tile's LDS bytes. Before `buffer_store` n7168 converts its bf16 tile to 8 bytes per lane at M = 32,
  `BlockedLayout([1, 4], [16, 4], [2, 2], [1, 0])`, and 16 at M = 64, `BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1])`
  (S7); the n3072 M = 32 kernel stores its 16×16 bf16 tile straight from the MFMA layout (S47). Store mechanics:
  [memory.md](memory.md) §15; conversion costs: [layouts.md](layouts.md) §7, §8.
- **Gate/up split in registers.** Put gate and up in the two halves of the tile's N, round where the reference
  rounds, then split: `pairs = gl.permute(gl.reshape(rounded, (BM, 2, BN // 2)), (0, 2, 1))`,
  `gate, up = gl.split(pairs)` (S48). The split needs column BN/2 on a register bit of the accumulator: with a
  16x16 layout, 16 ≤ BN/2 < 16 · `tiles_per_warp[1]` (a tile bit), or BN/2 a multiple of 16 · `tiles_per_warp[1]` ·
  `warps_per_cta[1]` ([layouts.md](layouts.md) §3). On four warps along N, BN = 128 compiled with no LDS and no
  cross-lane instruction at `tiles_per_warp=[1, 1]` and failed at `[1, 2]` ("Fp4ToFpOp/SplitOp requires …", §10);
  BN = 256 at `[1, 2]` compiled; BN = 64 failed at `[1, 1]` (all compiled with Triton 3.8.0 for gfx950). The other
  way: two accumulators sharing A and its scales, two `mfma_scaled` per K step (S15).
- **Group quantization after the MFMA** needs the complete group (all K, all splits). Where one program owns the
  group, quantize in its epilogue: GDN prefill quantizes each (token, head)'s 128 outputs right after its output
  MFMAs (S51); the MXFP4 gate-up converts its bf16 result to a layout that holds each group of 32 in few lanes
  (S15). Otherwise quantize in the consumer, each 128-K slice of the bf16 activation in registers, repacked into the
  fp8 operand layout with an `assert_trivial` convert and the activation scale folded after the MFMA (MiniMax-M2 fp8
  down, S10), or in the merge pass of a split reduction (gated attention, S17). SwiGLU and requantization per MoE
  launch: [../moe.md](../moe.md) §8; the recipes: [../numerics.md](../numerics.md) §2, §3; `FP16_OVFL`:
  [wave_ops.md](wave_ops.md) §7.

## 10. Pitfalls

Compile errors of matrix calls, each reproduced with Triton 3.8.0 for gfx950. An `error: …` on stderr precedes
`RuntimeError: PassManager::run failed` or `… during parsing` ([../debugging.md](../debugging.md) §4):

| Message | Cause | Fix |
|---|---|---|
| `no matching matrix core intrinsic for mfma version 4 with instruction shape [16, 16, 64] …` | `instr_shape[2]` is no table entry's K | §1 |
| the same for `mfma version 3` | `version=3`, the previous generation's layout, on bf16 [16,16,32]; with [16,16,16] (`k_width` 4) it compiles instead, to `v_mfma_f32_16x16x16_bf16`, which ran at half the rate (§1) | `version=4` |
| `Dot's a/b's encoding should be of DotOperandEncodingAttr` (both operands); `mismatching encoding between A and B operands` (one); `Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout` (`mfma_scaled`) | an operand in a `BlockedLayout`, not `DotOperandLayout(i, acc_layout, k_width)` | `convert_layout` to it, or load in it (§7) |
| `ttg.dot_op kWidth parameter is mandatory for MFMA parent`; `… is not supported when the parent is a blocked layout` | `k_width` 0 under an MFMA parent; nonzero under `gl.dot_fma`'s `BlockedLayout` | §5; 0 |
| `mismatching kWidth between A and B operands` | plain `mfma` with unequal `k_width` | §5 |
| `Both operands must be same dtype. Got bf16 and fp16` | plain `mfma` on two different 16-bit types (fp8 × fp8 of either format compiles) | convert one |
| `Unsupported lhs_format: bf16` | `mfma_scaled` takes only e2m1, e4m3, e5m2 | §2 |
| `out_dtype=bfloat16 is unsupported …` | a bf16 accumulator | fp32, cast in the epilogue (§9) |
| `input_precision must be one of ('ieee', 'bf16x3', 'bf16x6'). Got tf32` | `TRITON_F32_DEFAULT=tf32` in the environment, for Gluon `mfma` (fp32 and fp64 operands compiled) | unset it (2.5) |
| `error: Wrong opIdx`; `error: Incompatible parent encoding`, then `'tt.dot' op failed to infer returned types` (S63) | A built as `DotOperandLayout(1, …)` or B as 0; an accumulator in another `AMDMFMALayout` than the operands' parent | 0 for A, 1 for B; one `mma` layout for `gl.zeros`, both operands and the result |
| an assertion naming `isa<LinearEncodingAttr>`, then `PassManager::run failed` (exit 1) | an `mfma_scaled` scale in a `BlockedLayout` (S14) | `get_mfma_scale_layout` (2.2) |
| `scale factor must be 16 or 32. Got …`; `Operands must have the same scale factor` | a scale's shape | 2.2 |
| `Fp4ToFpOp/SplitOp requires at least 2 elements per thread …` | a split whose halves sit on a lane or warp bit | §9 |
| `Lane and warp dim basis must match between source and destination layout.`; `No source register holds the element for destination index …` | `gl.amd.slice` off the operand's lane sweep along K | 2.1 |
| `LLVM ERROR: Dimensions must match, ignoring order, but they don't`, then an abort | `mfma_scaled` (e4m3, `None` scales, [16,16,128]) on a rank-3 layout: the scale layout builder makes a 2-D tile (S13) | rank 2, or plain `mfma` (§4) |
| `unhashable type: 'list'` at launch, not in `triton.compile` | a list constexpr such as `INSTR=[16, 16, 32]` *(field, E10)* | a tuple ([../debugging.md](../debugging.md) §4) |

A compiler crash kills the process without a Python traceback: validate reports `candidate_crash`, exit −11 for
SIGSEGV or −6 for an abort (§8); compile the launch alone ([../debugging.md](../debugging.md) §6). The silent ones:

1. **Scale layouts are not checked, nor is a transposed scale when M (or N) = K/32** (2.2): derive them with
   `get_mfma_scale_layout` and unshuffle preshuffled bytes (§7).
2. **Masked scales must load as 127**, `0xFF` is NaN (2.2); writing every pad byte is a safe default (§7).
3. **`mfma_scaled` checks no `k_width`.** Unequal values compile and give wrong results; so does an fp8 operand at
   32 with real E8M0 scales or against e2m1 (§5). Use 16 for fp8 and for packed fp4.
4. **Small-K scaling is instance-specific.** The plain sweep tested bf16/fp16/e4m3/e5m2 new 16×16/32×32 shapes
   and old 16-bit shapes, swept `k_width` 4/8/16 ≥ kBase, K ≥ 4: at half or a quarter of `k_width · 64 / nonK`, the result was
   `acc / 2 + a·b` or `acc / 4 + a·b` (S49, S50; bf16 half-scale examples *(field, E6)*). Below kBase, even K equal
   to that extent can fail to compile. With `None` scales, e4m3 (`k_width` 16/32) and e2m1 (16) matched the instruction-K rule
   only in the tested compiled instances; small K also failed to compile. Scaled e2m1 at `k_width` 32 was wrong at
   packed K = 32/64 on [16,16,128] and 32 on [32,32,64] (real K is twice packed K): normalized errors 1.00/0.95/0.93, not a unified scaling formula *(E13)*.
   A zero accumulator does not repair these exceptions. DSA zero-pads its 64 RoPE columns on [16,16,128] (S54),
   or uses [32,32,64] at 12–32 tokens (S58); its smaller-K tail layout costs two accumulator conversions (S54).
5. **fp64 accumulators need `element_bitwidth=64`.** With the default 32 the call also compiled to
   `v_mfma_f64_16x16x4_f64` and a 16×16×16 product was wrong, error 1.01× the largest output *(field, E6)*: the map
   assumes 4 consecutive rows per lane (columns with `transposed=True`, S24) instead of 4 apart (S37) *(inferred)*.
6. **fp32 block scales cannot ride the E8M0 scale operands**: rounded to a power of two they change the result (2.1).

## Evidence

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, register-operand microbenchmark: 2048 one-wave programs, 64x64 fp32 accumulator, 4096 chained MFMA steps, median of 7 rounds of 3 launches; ranges over 4 runs] TFLOP/s: mfma bf16 k_width 8 [16,16,32] 2447–2468, [32,32,16] 2435–2452; plain fp8 [16,16,32] 2444–2468; mfma_scaled with None scales, e4m3 k_width 32 [16,16,128] 4899–4934, [32,32,64] 4907–4949, e2m1 k_width 16 [16,16,128] 9382–9591, [32,32,64] 9026–9341, mixed e4m3 x e2m1 [16,16,128] 4897–4936; e2m1 [16,16,128] with scale tensors of 127 (unit; compiled to the same instruction stream as None scales) 9348–9597; fp32 [16,16,4] 155.2–156.1, [32,32,2] 154.8–156.0; over bf16 [16,16,32]: plain fp8 0.996–1.004, e4m3 1.999–2.004, mixed 1.998–2.005 (0.999–1.001 of e4m3 x e4m3), e2m1 3.80–3.92 ([32,32,64]: 3.69–3.82), e2m1 with unit scale tensors 3.82–3.92, fp32 [16,16,4] 0.0631–0.0636 ([32,32,2] over bf16 [32,32,16] 0.0635–0.0638), fp64 [16,16,4] 0.0316–0.0317, bf16 [64,4,64] (v_mfma_f32_4x4x4_16b_bf16) 0.249–0.250; bf16 [16,16,32] over e2m1 with unit scale tensors 0.255–0.262; e4m3 [16,16,128] over plain fp8 [16,16,32] 1.995–2.008; older bf16 [16,16,16] 0.499–0.503 and [32,32,8] 0.504–0.505 of their K = 32 / 16 forms (case triton/checks/bench_mfma_rate.py (bench_mfma_rate.04, .07-.10, .14, .16-.19); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_mfma_rate_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_mfma_rate_rep3.log)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled census: python -m aklr validate --stages audit,bench on each of the 154 schemas in four shards, every opcode of the AKLR kernels' .amdgcn counted] schemas whose AKLR kernels contain: v_mfma_f32_16x16x32_bf16 73, _f16 3; v_mfma_f32_32x32x16_bf16 7, _f16 1; v_mfma_f32_16x16x32_fp8_fp8 3; v_mfma_scale_f32_16x16x128_f8f6f4 75, v_mfma_scale_f32_32x32x64_f8f6f4 12 (a8w8 199c45f4249e7f55 n3072 M=256, 1bbf98a368b3fcba n7168 M=128, fee7e092eab87fc5 n7168 M=256; DSA 64-head 1148d1c8e30397f7 at 12 tokens, f60f22e6f14448c6 16, 2ee56a6dbabc48a6 24, 2f7000a8186a8ae8 32; DeepSeek-V2 1204f286c55bb56e at 256, 3b667617400708bd 4096, b058395f8ad16a6c 4192; MiniMax-M2 fp8 f77013cc201d92d5 at 1024-16384, MXFP4 bb663e5b0cf55a10 at 1024); v_mfma_f32_16x16x4_f32 38, v_mfma_f32_32x32x2_f32 8; v_mfma_f64_16x16x4_f64 1; v_mfma_f32_4x4x4_16b_f16 1; v_dot2c_f32_bf16 1; no v_mfma or v_dot2 in 22 (GDN decode 18 with 0 v_mfma, gated attention at n=1 and 2, RMSNorm 2), both plain-Triton bf16 GEMM schemas with v_mfma_f32_16x16x32_bf16; a bf16 or fp16 v_mfma in gated attention 7 (each file at n=4, 8, 16, 32, 64, 128, 256), DSA 8-head 3, DSA 64-head 5 (32x32x16), Qwen2-MoE fp8 experts 3; each of the 26 GDN prefill schemas (Qwen3.5 15, Qwen3-Next 11) with v_mfma_f32_16x16x4_f32 or 32x32x2_f32; the MXFP4 schemas with a 1-token Point (DeepSeek-V2 5751, 7e9c; MiniMax-M2 6557, b065) each with v_mfma_scale_f32_16x16x128_f8f6f4; ds_read_b64_tr_b8 in 17 (DSA 0bb2, 1148, 2ee5, 2f70, 43ff, ae32, f60f, fbe6; gated attention 10df, 14e9, 6dd3, d9cb, dfb1, eeb1; DeepSeek-V2 MXFP4 699f, c524; MiniMax-M2 MXFP4 1d27; check .30 compares it with an earlier 15); bd017b0794d5db31 `_narrow_small_gemm` (n3072 m=1, one variant): 3 v_mfma_scale_f32_16x16x128_f8f6f4, 40 VGPRs with 8 AGPRs, 10 v_cndmask_b32, 2 v_add_f32_dpp, 0 ds_bpermute, 0 v_permlane (case triton/checks/rt_census_compiled_kernels.py (rt_census_compiled_kernels.22-.31); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] a8w8 kernels that fold K groups into the instruction rows: n3072 m=1 1.429, m=2 1.397, m=4 1.307; n7168 m=1 1.664, m=8 1.840; MiniMax-M2 MXFP4 MoE E256 inter 512, route block-diagonal down: n_tokens=4 2.549, 8 1.977, 16 1.595; DeepSeek-V2 MXFP4 MoE inter 256, one route per program on 16 replicated rows: n_tokens=2 2.186; MiniMax-M2 fp8 MoE E64 inter 1536, 1c1747a25af67b8e: n_tokens=1 3.137, 2 2.912, 4 1.672; Qwen3.5 gated attention decode 10df1ae2dc9d3c69 at n_tokens=8: 1.803; DSA sparse MLA decode 64 heads 0bb2f7cbe49d1be3 at n_tokens=8: 1.030 (AMD-AGI/AKLR@43e112e (CI run 41991630592):aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/bd017b0794d5db31.json, c194fbea23151544.json, d6228c711bd687b6.json; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/362ce3bb6200382e.json, 1e021d44010accb1.json; aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/96e51c1f85a71b7e.json; aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json; aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1c1747a25af67b8e.json; aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json; aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json); MiniMax-M2 MXFP4 MoE E256 inter 512, gate-up FOLD 4 at n_tokens=1: 3.444 (AMD-AGI/AKLR@3fb9033 (CI run 41812750710):aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, register-operand microbenchmark as in E1; the check's ±4% same-rate band failed in one (e2m1 0.956)] 32x32 over 16x16 rate: bf16 [32,32,16] / [16,16,32] 0.991–0.999, e4m3 mfma_scaled with None scales [32,32,64] / [16,16,128] 1.000–1.003, e2m1 0.956–0.976; accumulator VGPRs of one bf16 tile in the compiled ISA: 4 at 16x16, 16 at 32x32, in every run (case triton/checks/bench_mfma_rate.py (bench_mfma_rate.12, .13); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_mfma_rate_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_mfma_rate_rep3.log)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, three GPU cases run twice each, results against float64 (mfma_scaled error as a fraction of Σ|a·b| or of the largest output); values of the newer run] mfma_scaled on 16x16x128 tiles: real-scale e4m3 x e4m3 and mixed e4m3 x e2m1 at k_width 16 within 1.387e-05 and 2.589e-08 of Σ|a·b|, all nine format pairs with real scales within 1.622e-05; A k_width 16 against B 32: 1.198 of the largest output; fp8 k_width 32 with real scales 0.795, against e2m1 with unit scales 1.12; one K=256 call bitwise equal to two chained K=128 calls in 64 of 64 trials at k_width 16 and 32. One MXFP4 16x16 tile, K=512, random scales in 126..128: scales passed transposed ([16,16]) compile, relative error 0.8424; another linear scale layout (lane bits on the groups first) 1.078; A's row-3 scales all 0xFF: output row 3 all NaN, the other rows exact; masked scale bytes 127 and words 0x7F7F7F7F exact; scaled_upcast to bf16 bitwise for all 16 codes with s in {120, 127, 130} constant along a row, 0.5625 of elements with s changing every two; whole scale words only reshaped 0.9401 (that check passed in the newer run only); aiter dynamic_mxfp4_quant(shuffle=True) against E8M0Scales(pad_to=(256, 8), shuffled=True) at (M, K) = (33, 256), (255, 288), (1, 64): valid bytes equal, of 5631 pad bytes 4735 zero, 11 0x7F; aiter gemm_a4w4 N=K=256 at M=1, 33, 255: 0xFF in the row-padding scale bytes changed no output. Random 64x64x1024 bf16 trials, each variant run 3 times: 16x16x32 against 32x32x16 at k_width 8 bitwise equal in 64 of 64; k_width 8 against 16 different in 64 of 64, at most 1.625 ulp of Σ|w||x|, each variant repeatable; bf16 x bf16 products in fp32's normal range equal to float64 bitwise, 16384 through a bf16 MFMA and 748591 on the VALU (case triton/checks/rt_mfma_scaled_unit_kwidth.py (rt_mfma_scaled_unit_kwidth.11, .17, .18, .21-.23, .26); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_mfma_scaled_unit_kwidth_r2.log; case triton/checks/rt_mxfp4_tile_scales.py (rt_mxfp4_tile_scales.12, .18-.21, .33, .35, .36); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_mxfp4_tile_scales_r1.log; case triton/checks/num_router_precision.py (num_router_precision.02-.04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_router_precision_r1.log)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, five cases run once each, results against float64 or torch] bf16 [16,16,32] K=16 k_width 8 within 0.32 ulp of acc/2 + a·b, K=32 k_width 16 within 4.8e-08, mfma_scaled [16,16,128] K=64 k_width 32 within 3.5e-08, each at least 2.0 from acc + a·b; the rtz bf16 downcast equal to the fp32 bits >> 16 on 1049088 values (4 v_perm_b32, no v_cvt_pk_bf16_f32), and a 64x64 convert_layout through LDS 16384 bytes as fp32, 8192 as bf16; a [4,16] tile on bf16 [16,16,32] relative error 3.4e-08 with 1 MFMA, 9 global loads and 1 store, as [16,16] (the timing half of that check failed); [4,64,64] fp16: 16 v_mfma_f32_4x4x4_16b_f16, relative error 1.0e-07; fp64 16x16x16 product on [16,16,4] relative error 2.0e-16 with element_bitwidth 64, 1.009 with 32; one-wave fp32 (A·B)·B2 with the accumulator relabelled 1.05e-07 or moved by two permlane steps 1.38e-07, neither through LDS (case triton/checks/rt_mfma_small_k_and_acc.py (rt_mfma_small_k_and_acc.01, .02, .13); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_mfma_small_k_and_acc.log; case triton/checks/num_bf16_rounding_promotion.py (num_bf16_rounding_promotion.12, .13); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_num_bf16_rounding_promotion.log; case triton/checks/lower_mfma_instruction_table.py (lower_mfma_instruction_table.37, .40); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_instruction_table.log; case triton/checks/lower_mfma_fp32_fp64_precision.py (lower_mfma_fp32_fp64_precision.04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_fp32_fp64_precision.log; case triton/checks/lower_mfma_acc_chaining.py (lower_mfma_acc_chaining.22); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_acc_chaining.log)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, AKLR CI bench (graph replay, 1 GiB read flush), speedup against the schema's baseline] Qwen2-MoE fp8 experts, weights as the A operand: n_tokens=8 1.208; MiniMax-M2 fp8 MoE E256 inter 384, e75e553b34178760: n_tokens=1 4.115, 2 3.755, 4 2.452, 8 1.841, 16 1.732 (AMD-AGI/AKLR@93eba6d (CI run 41964469977):aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json; aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one case run: python -m aklr validate --stages audit,bench, one GPU, committed and modified copy alternated, 4 rounds each] MiniMax-M2 MXFP4 E256 655723f593921b08 at n_tokens=1, gate_fold 4 to 1 alone (4 warps, BK 128): the modified copy failed validation in 4 of 4 rounds (candidate_crash, exit -11; its standalone compile also -11, in LLVM code generation); the committed one measured 3.570, 3.561, 3.545, 3.535 (case triton/checks/ab_coupled_constants_break.py (ab_coupled_constants_break.11); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_coupled_constants_break.log)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, 8192 programs of 4 waves, each reading one 128x128 fp32 state (537 MB per launch) and computing S^T q; median of 9 rounds of 5 launches] FMA + gl.sum 89.75–91.74 µs against the fp32 [16,16,4] MFMA with q padded to 16 rows 93.86–97.19 µs, ratio 0.944–0.957, outputs within 4.2e-07–5.4e-07 relative of each other; both read 5.5–6.0 TB/s (estimated from bytes and time) (case triton/checks/bench_mfma_rate.py (bench_mfma_rate.21); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_mfma_rate.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r1.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_mfma_rate_r2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_mfma_rate_rep2.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_mfma_rate_rep3.log)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one case run: a Gluon kernel compiled, then launched] a list constexpr INSTR=[16, 16, 32] compiled and its launch raised TypeError "unhashable type: 'list'"; INSTR=(16, 16, 32) launched, max error 2.4e-07 (case triton/checks/rt_launch_only_errors.py (rt_launch_only_errors.11, .04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_launch_only_errors_r1.log)
- E11 [2026-11, gfx950 target, Triton 3.8.0, CPU-only, N=1 full offline run] `k_width` 16 maps 16 packed bytes (32 logical fp4) per lane on [16,16,128]/[32,32,64], with [16,64]/[32,32] packed operands and [16,4]/[32,2] scales; 128 lane maps tested per shape, one MFMA each. No GPU nibble-order or numerical test (case triton/checks/d_compile_boundary_controls.py, check .07; local:/tmp/kbmig/checks_study/local_logs/d_compile_boundary_controls.local.log).
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 full GPU run, 8 consecutive balanced repeat clusters per setting, 9 or 10 complete blocks per cluster × 5 launches, no flush] 256-step fp32 chains, contiguous inputs, 4/16 programs per CU: §6 reports the .04/.05 time ratios. Both performance checks FAIL despite values passing: no-LDS relabel/kw4 is slower at 4-wave 64×16, 16 programs per CU (1.332/1.313). Cluster t95 is diagnostic, not proof of independent repeats or simultaneous coverage (case triton/checks/s_cross_wave_movement_cost.py, checks .04/.05; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_cross_wave_movement_cost_r2.log).
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full GPU runs, not variant counts] Each run: 138 plain variants (54 below extent) match ISA and nonzero-accumulator formula, normalized error 3.2e-8–7e-8; scaled with None scales compiles 25/32, 22 numerical instances match (3.1e-8–6.5e-8), three packed e2m1 kw32 exceptions are listed in §10. CASE FAIL (5/7, 6/7), not blanket support (case triton/checks/s_mfma_k_extent_and_fp16_operands.py, checks .01/.02; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_mfma_k_extent_and_fp16_operands.log; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_mfma_k_extent_and_fp16_operands_r1.log).
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full GPU runs, eight seeds each, both operand orders, K=32] fp16/e4m3 path worst error/bound ≤0.012290, bound=32·2^-23·Σ|a·b|; converting the fp16 partner to bf16 gives 2034/2048 and 2032/2048 outputs outside bound (case triton/checks/s_mfma_k_extent_and_fp16_operands.py, .06; logs as in E13, including local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_mfma_k_extent_and_fp16_operands_r1.log). Not extended to e5m2/extreme inputs.
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 full GPU run, alternated flushed passes, not independent runs] .01: five passes, 1024 programs × bf16 keys [2048,128], 256-key tile, 4 warps, matched loads; R=1/2/4/8/16, FMA+sum/MFMA=1.002/0.964/0.974/1.882/2.884, dot_fma/MFMA=0.977/0.955/0.995/1.104/1.662, relative error ≤2e-7. .02/.03: three passes, MXFP4 W=[4096,4096], M=1/2/16/128/1024/4096; tiles 16×16×512 one wave at M≤16, 128×128×256 four waves above; upcast/scaled GEMM=1.039/1.032/1.072/1.588/1.569/1.585; quantized and unquantized float64 targets differ (errors ≤8.5e-7). .03 M=1/2 uses 256 one-wave programs, 16-row tile, BK512; scaled/VALU times in §2.6. CASE FAIL 4/6 (case triton/checks/s_few_rows_valu_vs_mfma.py, .01-.03; local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_few_rows_valu_vs_mfma.log).
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full GPU case runs, compile-only subchecks and register-loop timings] .03/.04/.05 cover direct-load kw16, slice compilation, srcA widths and LDS reads as scoped in §§1/2/5/7; two passes per timing run are not independent. fp16/base-bf16 rates over all passes: 16×16 0.994592–1.001194, 32×32 0.965486–1.000365; no effective-cycle measurement, paired raw rounds or precise equal-rate inference (case triton/checks/s_mfma_k_extent_and_fp16_operands.py, .03-.05; logs as in E13, including local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_mfma_k_extent_and_fp16_operands_r1.log).
**Unverified scope:** kernel-level shape superiority (bench_mfma_rate.12/.13), universal failure of linear scale layouts (rt_mxfp4_tile_scales.18), GPU nibble order (E11 .07), unswept K/layouts (E13 .01/.02), universally faster kw4/LDS-free routes (E12 .04/.05), no-reuse dot_fma/compute-bound row crossover or software-emulated fp4 decode (E15), absolute MAC/cycle, precise same-rate and consecutive srcA segments (E16). These checks do not measure small-M/AITER comparisons or decode M64 split6 versus split4; no best-configuration claim follows.
- S1 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp:142 version-4 table: bf16 16x16x32 kBase 8 and the older 16x16x16 (:186-187), OCP fp8 only 32x32x16 and 16x16x32 (:205-243), fp32 16x16x4 and 32x32x2 (:149-151), the scaled 16x16x128 and 32x32x64 (:262-268), keyed as fp4 × fp4 for any F8F6F4 pair (:32-36); `MfmaIntrinsic::get` fails unless an entry's K equals the requested K (:303-321)
- S2 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:273 "no matching matrix core intrinsic" error; the plain lowering reads only TF32, and only for version 3 (:257)
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp:595 plain `tl.dot` on F8F6F4 types tries the scaled instruction and rewrites to `dot_scaled` (:759-786) with operand k_width 32 / 2 = 16 (:772-776); scaled dots: "kWidth is 16 for fp4" (:1007-1011), and both operands get kWidth / 2 = 16 whatever the format (:1022-1023)
- S4 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` runs neither F32DotTC (:270) nor AccelerateAMDMatmul (:273), which only `make_ttgir` runs; Gluon kernels take it (:589); `get_min_dot_size` returns (1, 1, 1) (:16-19); default precision "ieee", allowed ("ieee", 'bf16x3', 'bf16x6') (:92-93)
- S5 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:65 both scales `None` give uint8 0x7F (:68-69); an `int` scale a uint8 constant, a `float` an fp8e4nv constant (:89-99); `deduce_scale_factor` first (:101); `dot_scaled(..., fast_math=False, ...)` at :105-106
- S6 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 the cdna4 module re-exports the previous generation's names, `mfma` among them; `mfma_scaled` formats `e2m1`, `e4m3`, `e5m2` for each operand (:43-46, :60-61); `scaled_upcast` takes raw E8M0 bytes already in the result's shape, to fp16 or bf16 (:67-83); "Only scale factor 32 is supported for CDNA4 Scaled MFMA" (:104); `compute_efficient_padded_shared_layout` returns `None` unless k_width is 4, 8 or 16 (:139-143)
- S7 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/ae4fa8a440b98f09.json#kernel.py:45 `_group_dot`, one `mfma_scaled` per 128-K group of `gl.amd.slice`d operands (:45-51); the fold (:68-72); store layouts, the M == 32 branch first (:80-83), cast to bf16 before `convert_layout` (:84)
- S8 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:13 "Extract a register only slice", the source layout kept, no cross-thread movement
- S9 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1bbf98a368b3fcba.json#kernel.py:83 two [32,32,64] calls chained on one zero-started partial, folded at :94; layout `[32, 32, 64]`, `warps_per_cta=[2, 2]`, `k_width` 16 at :125-130
- S10 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json#kernel.py:134 `_down_combine`: rank-3 `[16, 16, 32]` layout, `warps_per_cta=[8, 1, 1]`; activation quantized per 128-K slice (:157-162), repacked by `_fp8_words_to_dot` (:66-84); plain fp8 `mfma` and `gl.fma(part * sa, sb, acc)` (:166-170). Gate-up: `_project_w13_internal_packed2`, rank-3 `[16, 16, 32]` with `warps_per_cta=[SPLITS, 1, 1]` (:647-652), rows alternating two K chunks (:664), plain fp8 `mfma` (:681), matching blocks kept (:682-687), warps summed (:712); `_project_paired_packed4`, 4 K chunks per row on `[16, 16, 128]` (:440-491); `_project_w13_bands`, one row broadcast over 16 rows and one register kept (:569-578); m <= 2 packed2, m == 4 packed4, else bands (:771-819)
- S11 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:264 MFMA layout; operand and scale layouts (:270-273); 16 identical rows of one route (:274-278); scale byte shifted out of the shuffled word (:323-324); the `mfma_scaled` call, operands and scales converted to their layouts (:325-333); row 0 stored (:337); two-token router: 3-D blocked layout (:77), weight loaded once, `gl.dot_fma` per token (:81-90); `_activate_quantize` stores its MXFP4 codes and scales row-major (:386-387); MXFP4 weights and scales passed as `.view(torch.uint8)` (:548-551)
- S12 triton-lang/triton@c01b677:python/triton/language/semantic.py:1544 e2m1 "must be packed as uint8"; K counted twice for packed operands (:1598-1609); `deduce_scale_factor` (:1553) and `verify_scaled_shape`, which asserts `[..., M, K/32]` and `[..., N, K/32]` (:1564-1574); `input_precision` must be in the backend's allowed set (:1417); the dot asserts M, N, K against the backend's minimum dot size (:1479-1482)
- S13 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1711 scale layout: registers for further K steps (:1711-1713), then tiles per warp (:1715-1716); lanes rows, then groups (:1718-1733); built as a two-dim tile (:1736) and transposed to all of the tensor's dims (:1750); for mxfp8 a lane's 32 elements are "2 16-consecutive elements blocks" (:1703-1705)
- S14 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:695 "Scales have the same replica distributions as their corresponding operands": scale registers read in operand order (:699-711); both scales must have a linear layout (:898-901); up to 4 scale bytes per VGPR, K first, then non-K (:667-686); `op_sel` per instruction (:768-792)
- S15 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/147b861bd4c406d2.json#kernel.py:509 packed activation stored at `_weight_offset_3`, scales at `_scale_offset_5`, padding rows (route >= R) as 127 (:523-539) under the condition of `run`'s `pair_activation_scales` (:1095-1097), else masked off (:520, :540-545); gate-up tile 64 rows (:1085) by 64 or 128 columns (:1086), BN // 32 scale groups (:516-518); the down GEMM loads whole activation-scale words with no mask under `PAIR_ACTIVATION_SCALES` (:186-190), else loads padding rows' scales as 127 through a mask (:221, :238-249); it loads A and B at `_weight_offset_3` offsets (:204-219), fills masked scale words with 2139062143 (:263-265) and hands A and B over with `assert_trivial=True` (:289-291); A scales staged in a `SharedLinearLayout` at BK = 512 and loaded in the scale layout (:450-462); gate and up as two `mfma_scaled` per K step sharing A and its scales (:463-480); group layout for the MXFP4 quantization (:482-483), amax per 32 (:494-495)
- S16 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:1237 e4m3fn to bf16 on CDNA4 through `scalePk4UpcastFromFp8`, two packed converts per 4 values with scale 1.0 (:337-376); fp32 to bf16: packed RTNE convert on CDNA4 (:106-115), RTZ a shift (:73-77)
- S17 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/6dd321aac1b1d85a.json#attention.py:58 fp8 K converted to bf16 in the B operand layout, bf16 `mfma` (:59-61); P cast to bf16, then converted to the A layout of the P·V `mfma` (:95-98); `_finish_head`: merged head in a blocked layout, amax per 128 (:252, :282-285)
- S18 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:103 `_QK_OPERAND` blocked layout; read from LDS, `.to(gl.bfloat16)`, convert to `DotOperandLayout(1, ..., 16)` (:107-110); rescaled accumulator passed as the MFMA addend (:179-183)
- S19 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/3d0f64ffff6790c8.json#attention.py:148 P and V in bf16 when V is bf16, else in fp16 (:148-155)
- S20 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/5751284b6965eabb.json#kernel.py:80 bf16 router on `[16, 16, 32]` with `k_width` 8 (:95-99), rows clamped to M - 1 (:86), stores masked to M (:102)
- S21 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len4096_n_v_heads8_key_dim128_value_dim128_conv_dim2048/162f43f1ab0c0a56.json#kernel.py:217 fp32 `mfma` wrapper with `k_width` 1; bf16 `[16, 16, 32]` Gram products converted to the fp32 `[16, 16, 4]` layout (:419-427)
- S22 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/8a5a03b603ec1b70.json#kernel.py:909 fp32 weight split into three bf16 terms, three bf16 MFMAs (:913-915), summed at :916
- S23 triton-lang/triton@c01b677:python/triton/knobs.py:496 `fp32_default` from `TRITON_F32_DEFAULT`
- S24 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json#kernel.py:73 fp64 `[16, 16, 4]` with `element_bitwidth=64`, `transposed=True`; `[4, 64, 64]` fp16, `transposed=False`, `k_width` 4 at :327-330
- S25 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/20f63094b85c00a2.json#kernel.py:187 decay, `gl.sum(state * k[None, :], 1)`, rank-1 update and `gl.sum(state * q[None, :], 1)` on the per-head state (:187-192)
- S26 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/f5ba496e04f8e989.json#attention.py:61 fp32 FMA dot and `gl.sum` per key at n = 1
- S27 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:748 `dot_fma` asserts a BlockedLayout accumulator and DotOperandLayout operands with that parent (:753-760)
- S28 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:33 f16/bf16 into f32 uses `llvm.amdgcn.fdot2` / `fdot2.f32.bf16`, two elements per call
- S29 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:778 transposed layouts swap A, B and their scales at the instruction (:778-784); 4x64 and 64x4 broadcast the narrow operand 16 times (:313), with transposed asserts at :360 and :368; the layout's own assert in S37
- S30 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3b667617400708bd.json#kernel.py:271 `instr_shape=[16, 16, 128] if TM == 16 else [32, 32, 64]`
- S31 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/14c8e25ca1a39f6d.json#kernel.py:410 `instr_shape=[INSTR_M, INSTR_M, 2048 // INSTR_M]`
- S32 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:748 dot operand layout: k_width along K in registers, lanes non-K then K, repeated along K (:748-770)
- S33 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_layouts.py:230 `k_width (int): Number of elements per 32-bits`
- S34 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:333 plain path: `numVecInKBase = numRepK * kWidth / kBase`, a lane's run split into kBase chunks, one per instruction, chained on one accumulator (:353-381; scaled path :766-808); scaled path: `aKWidth = aKBase`, the layout's k_width not used for chunking (:636-648)
- S35 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json#kernel.py:106 `[16, 16, 32]` bf16 with `k_width` 16 for operands converted from fp8 (:106-108), K = 128 per call (:141-145)
- S36 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/199c45f4249e7f55.json#kernel.py:404 reads with `k_width` 32 on [32,32,64]; asks the padded-layout helper with a `k_width` 16 operand at :450-455
- S37 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:486 accumulator layout from M, N, `transposed` and the element width only: 4 rows per lane (:486-499) or, transposed, 4 columns (:500-510); height 1 for 64-bit elements (:473); "64x4 mfma must be transposed" (:481); tiles per warp along N, warps along N, remaining N in registers, then M (:539-557)
- S38 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1e021d44010accb1.json#kernel.py:135 `_diagonal`: `gl.amd.slice` of the group's columns and rows, `assert_trivial` convert (:140-143); `transposed=False` at :269; PG 2 at :254
- S39 aklr/core/contract/semantics/_layouts.py:103 `Preshuffle`: tiles of tile[0] rows by 2·tile[1] columns in 16-byte vectors; `E8M0Scales` shuffle split and order at :47-52, :148-149; `E8M0Scales.encode` fills the padded shape with 0x7F, copies the scales in, then shuffles (:140-150); an input the layout variant does not list is stored row-major (:254)
- S40 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:35 `_weight_offset_3`, the 16x16 preshuffle offset; `_scale_offset_5`, the shuffled e8m0 offset (:40-49); load indices from `gl.to_linear_layout` of the operand layouts, A at preshuffled offsets with an `assert_trivial` convert (:238-268)
- S41 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/7e9c13e0f2d58a3b.json#kernel.py:269 word load layout `[1, 4]` per lane; `_unpack_bytes`, two `gl.join` levels (:234-237); whole scale words unpacked and permuted back to `[BN, BK // 32]` (:303-305); preshuffled B handed over with `assert_trivial=True` (:374)
- S42 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245 `assert_trivial` raises when the conversion is not trivial
- S43 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/96e51c1f85a71b7e.json#kernel.py:159 block-diagonal pick: register split with `transposed=False` or `gl.gather` (:159-177)
- S44 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/5c35215626bc1c87.json#kernel.py:138 the token broadcast over 16 columns as B, the weight as A (:138-145)
- S45 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/bd017b0794d5db31.json#kernel.py:728 rows (token, group) and columns (feature, group) (:728-733); the call and the diagonal (:772-786); sum over groups (:792); `PG`, `BN` (:685-687); grid `n // 4`, one warp, for m in (1, 2) (:913-919)
- S46 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json#kernel.py:408 partitions of the rows read matching K chunks (:410-413); diagonal gathered and summed after the loop (:426-437); FOLD 4 at m <= 2 (:1301), passed as `FOLD=gate_fold` (:1331)
- S47 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/fcf21622f6c91bbb.json#kernel.py:407 `buffer_store(acc.to(gl.bfloat16), ...)` from the MFMA layout `LC`
- S48 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:237 gate/up split of the accumulator
- S49 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:191 `adjustAccForSmallKDim` multiplies the result by 1/duplication (:199-214), called after the chain at :383; plain path: operand K extent from `k_width` (:292); scaled path: from kBase (:645), called at :811
- S50 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:2594 `getInstrShapeForOperand`: operand K extent = kWidth · (64 / min(mDim, nDim)) (:2601-2604), the size the lowering compares the operand K with
- S51 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1024_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1fd6adb2b91cf5c2.json#kernel.py:1479 `_group_fp8(rounded, ...)` on each (token, head)'s 128 outputs, after the output MFMAs (:1441-1442) and the gated RMSNorm (:1474)
- S52 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1c1747a25af67b8e.json#kernel.py:675 `_project_down_transposed`: `[16, 16, 128]` (:698-703), weight as A, the activation broadcast as B (:733-744), column 0 stored (:749); launched for m <= 4 (:1158)
- S53 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json#attention.py:44 "Non-transposed ownership puts adjacent high/residual rows in registers" and `transposed=not COMPRESS_SOFTMAX` (:44-48); interleaved query rows (:65); `prob_hi`, `prob_lo` (:231-232) joined into adjacent rows (:235); pairs summed (:258); switched on at 8 rows (:410-414)
- S54 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/0bb2f7cbe49d1be3.json#kernel.py:483 `_probability_operands`: high and residual P joined into 2 × HEAD_BLOCK rows (:483-492); `[32, 32, 16]` for 16 packed heads (:72); RoPE keys loaded 128 wide (:80), columns ≥ 64 masked to 0 (:107, :323-327); RoPE QK on the scaled layout (:658-666), or with `native_rope` (rows <= 4, :263) on a second layout with two accumulator conversions (:650-656)
- S55 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:593 `_w13_local_split`: `warps_per_cta=[SK, 1, 1]` (:618-623), plain fp8 `mfma` (:706-710), `gl.sum(acc, 0)` (:713); launched with SK = 8 when m > 32 (:834, :926-944)
- S56 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json#kernel.py:44 `_accumulate_fp8_group`: plain `mfma` when `NATIVE` (:48-53), else `mfma_scaled` (:54-63); `instr_shape=[16, 16, 32 if NATIVE else 128]` (:96); `_down_projection_2` launched with `NATIVE=m == 64` (:692, :706)
- S57 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len10000_n_v_heads16_key_dim128_value_dim128_conv_dim3072/3b99d19659746a36.json#kernel.py:520 a loaded bias passed as the `mfma` accumulator
- S58 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/f60f22e6f14448c6.json#kernel.py:412 `[32, 32, 64]` score layout with `k_width` 32 (:412-414); RoPE QK in one call (:121)
- S59 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/0101652e6ad7286d.json#kernel.py:643 scale words loaded on even K steps only (:643-663); the byte taken with `>> first_shift + offset % 2 * 16` (:664)
- S60 aklr/ops/fused_moe/reference.py:90 activations kept in bf16 under `gate_mode` 'interleave' (:90-93), MXFP8 per 32 when `act_quant` is 8 (:94-95, `_mxfp8` :38)
- S61 aklr/plugins/sglang/models/minimax_m2/contract.py:166 `E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)`
- S62 triton-lang/triton@c01b677:python/triton/_utils.py:77 `type_canonicalisation_dict`: no fp4 or e8m0 entry
- S63 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3168 "Wrong opIdx"; "Incompatible parent encoding" at :3178
- S64 aklr/ops/gemm/contract.py:301 "aiter's scales come from dynamic_mxfp4_quant(shuffle=True): padded, then shuffled", `_SHUFFLED_SCALES = E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)` (:302); `gemm_a4w4_mxfp4@1.0`'s `b_shuffled` variant: `b` preshuffled 16×16, `a_scale` and `b_scale` shuffled, `a` not listed (:370-375)
- S65 harness/skills/aklr-kb/triton/gluon/mfma.md:1 page-local Evidence retains the source and measurement conditions.
