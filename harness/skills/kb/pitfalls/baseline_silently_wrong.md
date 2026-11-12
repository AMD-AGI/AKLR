---
title: "The baseline you time against can return wrong values, and AKLR's bench does not check it"
kind: pitfall
scope: [aiter gemm_a4w4 asm path at K % 256 != 0, hipBLASLt MX GEMMs through torch._scaled_mm, any baseline a AKLR Point is timed against]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "aklr/core/validation/stages/_bench/_evaluation/_candidate.py:319"
  - "aklr/ops/gemm/contract.py:306"
---

# A wrong baseline

## Symptom

A FlyDSL kernel beats a baseline that returns wrong values without raising. Seen on MI355X with the libraries
called directly, not through AKLR:

- **aiter `gemm_a4w4`, asm path, MXFP4, K not a multiple of 256, with the padding columns of the E8M0 scales at
  1.0 (`0x7F`, as AKLR's `E8M0Scales` filled them before #130, S5)**: SNR 5.9, 7.7 and 10.9 dB against the fp32
  reference at K = 2880, 2944 and 3008 *(field, E1)*, again on aiter `dbd8bf5` as AKLR's
  `gemm_a4w4_mxfp4_baseline` calls it *(field, E5)*. With those columns at 0 (`0x00`, what aiter's
  `dynamic_mxfp4_quant(shuffle=True)` writes) the same call reads 55.6 dB *(field, E5)*. aiter's CK path raises an
  error at those K instead.
- **hipBLASLt MXFP8 through `torch._scaled_mm`, K % 256 == 128, on torch 2.11.0+rocm7.2**: SNR 26–49 dB,
  different from call to call, at 8192×5120×2944, 4096×4096×2944 and 8192×5120×4224; torch accepts the shapes.
  The MXFP4 path did not show it *(field, E2)*. On torch 2.11.0+rocm10.0.0 (HIP 7.15) five such shapes, two of
  them above, read 49.6 dB, bit-identical across calls, as at K % 256 == 0 *(field, E6)*.
- **hipBLASLt MX GEMMs with bf16 output**: 49.6 dB on MXFP4 8192×8192×8192 and on MXFP8 16384×4096×4096, where
  aiter `gemm_a4w4` read 55.6 dB on the same MXFP4 problem *(field, E3)*. The outputs are rounded toward zero:
  |out| ≤ |ref| on 99.4–100% of the elements *(field, E4)*. An fp32 output is rejected.

## Root cause

- The library accepts a shape its kernel does not support and computes on it (the a4w4 asm path, hipBLASLt's
  MXFP8 kernel), or rounds the output toward zero instead of to nearest even, which caps the SNR of a correct
  MX result against an fp32 reference.
- AKLR's bench calls the contract's `<contract>_baseline` once; one that raises `NotImplementedError` gives
  `baseline_failed`, but the baseline's output is compared with nothing (S1). The candidate is checked against
  the reference (`aklr/ops/README.md` § Schema fields), so a wrong baseline does not fail a correct
  candidate: it becomes the denominator of the Point's `speedup`. The one error bound that reads the baseline's
  output, `baseline_relative`, would widen the candidate's allowed error by the baseline's error (S2); of the ops'
  contracts only `fused_moe` uses it, for swiglu.

## How to detect

- Run the baseline through the gate you run your kernel through, at every Point you time, against an fp32
  reference built from the dequantized operands (`../gemm/correctness.md` §1, the fifth of the gate
  shapes).
- Check which kernel the baseline launched, and that it is the path you meant to beat
  (`../gemm/measurement.md` §7).
- Flag MXFP4 and MXFP8 Points whose K is not a multiple of 256. AKLR's `gemm_a4w4_mxfp4` contract admits any K
  that is a multiple of 64, and its baseline calls aiter's `gemm_a4w4` (S3).
- Do not take a hipBLASLt MX output as the reference for an MX kernel: it is itself 49.6 dB from the fp32 result
  (E3).

## Fix or workaround

- Outside AKLR, pad K to a multiple of 256 for these vendor calls (zero data, E8M0 scale 127) and price the
  padded work at the real M·N·K, or drop the vendor column at those shapes (S4).
- Build references from the dequantized operands in fp32, not from a vendor output.
- In AKLR, a Point's speedup over a baseline that fails your check is not a measured gain. Report the baseline's
  failure with the Point.

## Scope

- Measured in 2026-10 on MI355X, torch 2.11.0+rocm7.2 (HIP 7.2.26015), aiter `c16d44b93` from the image, called
  directly; the a4w4 and MXFP8 items again in 2026-11 on torch 2.11.0+rocm10.0.0 (HIP 7.15), the a4w4 one on aiter
  `dbd8bf5`, AKLR's baseline aiter, with scales padded `0x7F` (E5, E6). Not re-checked through AKLR's bench.
- No `gemm_a4w4_mxfp4` schema is committed, so no recorded AKLR Point is known to be affected. Since #130
  `E8M0Scales` pads with `0x00` and does not reach the a4w4 item; scales padded with `0x7F` elsewhere still do.
  `torch._scaled_mm` is not one of AKLR's baselines (`aklr/ops/gemm/baseline.py`); the hipBLASLt items matter
  when you build your own vendor column or reference.

## Evidence

The SNR values in E1, E2 and E4 are written down only in the notes of the adapter; the raw probe output was not
kept. E3 is in the run's own records.

- E1 [2026-10, gfx950, torch 2.11.0+rocm7.2 aiter c16d44b93, N=?, adapter validation: SNR against the dequantized fp32 reference] aiter a4w4 asm 5.9 / 7.7 / 10.9 dB at K = 2880 / 2944 / 3008 (MXFP4 dense GEMM)
- E2 [2026-10, gfx950, torch 2.11.0+rocm7.2 hipBLASLt, N=?, adapter validation, repeated calls] torch._scaled_mm MXFP8 26–49 dB varying from call to call at 8192×5120×2944, 4096×4096×2944, 8192×5120×4224; MXFP4 unaffected (MX dense GEMM)
- E3 [2026-10, gfx950, torch 2.11.0+rocm7.2 FlyDSL 0.3.2 aiter c16d44b93, N=2, head-to-head smoke run on GPU 4, SNR of two calls] torch._scaled_mm MXFP4 8192³ 49.603 / 49.603 dB and aiter a4w4 55.600 / 55.600 dB on the same problem; torch._scaled_mm MXFP8 16384×4096×4096 49.605 / 49.605 dB (MX dense GEMM)
- E4 [2026-10, gfx950, torch 2.11.0+rocm7.2 hipBLASLt, N=?, element-wise comparison with the fp32 reference] |out| ≤ |ref| on 99.4–100% of the elements of the MX bf16 outputs (the adapter's accuracy note)
- E5 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0 aiter dbd8bf5, N=3 calls per cell, SNR against the fp64 product of the dequantized operands] `gemm_a4w4` as `gemm_a4w4_mxfp4_baseline` calls it (`bpreshuffle=True`), M×N 1024×1024 and 256×3072: with the scale padding at `0x7F`, 5.93–10.88 dB at K = 2880, 2944 and 3008 and 7.41–7.54 dB at K = 4160, none raising; with it at `0x00` or with aiter's own `dynamic_mxfp4_quant(shuffle=True)` scales, 55.59–55.62 dB; the CK path `gemm_a4w4_blockscale` raised `This GEMM is not supported!` at those K (MXFP4 dense GEMM)
- E6 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0 (HIP 7.15) hipBLASLt, N=8 calls per shape, SNR against the fp64 product of the dequantized operands, each call compared bitwise with the first] `torch._scaled_mm` MXFP8 with bf16 output: 49.60 dB at 4096×4096×2944, 8192×5120×4224 and three more shapes with K % 256 == 128, every call bit-identical and none raising, as at K % 256 == 0 (49.59–49.61 dB) (MX dense GEMM)
- S1 aklr/core/validation/stages/_bench/_evaluation/_candidate.py:319 `run_baseline` calls `contract.baseline()` (:320) and discards its output; only `NotImplementedError` becomes `baseline_failed` (:198-199)
- S2 aklr/core/contract/semantics/_compare.py:278: `BaselineRelativeBound` allows a multiple of the baseline's error; under aklr/ops only the fused_moe contract uses it, for swiglu (aklr/ops/fused_moe/contract.py:268)
- S3 aklr/ops/gemm/contract.py:312: `k` axis 64..65536, `multiple_of=64`; `gemm_a4w4_mxfp4_baseline` calls `aiter.ops.gemm_op_a4w4.gemm_a4w4` at aklr/ops/gemm/baseline.py:82
- S4 the adapter's K padding (data 0, scale 127) and its extra-work percentage
- S5 aklr/core/contract/semantics/_layouts.py:145 at 6b6b34e: `E8M0Scales` fills the padding of the scales with `0x7F` (e8m0 1.0); the `gemm_a4w4_mxfp4` contract stores its scales as `E8M0Scales(block_size=32, pad_to=(256, 8), shuffled=True)` (aklr/ops/gemm/contract.py:302)
