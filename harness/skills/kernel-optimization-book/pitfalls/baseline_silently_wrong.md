---
title: "The baseline you time against can return wrong values, and a benchmark that checks only the candidate does not see it"
kind: pitfall
scope: [aiter gemm_a4w4 asm path at K % 256 != 0, hipBLASLt MX GEMMs through torch._scaled_mm, any baseline a kernel is timed against]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
---

# A wrong baseline

## Symptom

A FlyDSL kernel beats a baseline that returns wrong values without raising. Seen on MI355X with the libraries
called directly:

- **aiter `gemm_a4w4`, asm path, MXFP4, K not a multiple of 256, with the padding columns of the E8M0 scales at
  1.0 (`0x7F`)**: SNR 5.9, 7.7 and 10.9 dB against the fp32
  reference at K = 2880, 2944 and 3008 *(field, E1)*, again on aiter `dbd8bf5` called with
  `bpreshuffle=True` *(field, E5)*. With those columns at 0 (`0x00`, what aiter's
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
- A benchmark that calls the baseline only to time it compares the baseline's output with nothing: a baseline
  that raises is reported, one that returns wrong values is not. The candidate is checked against
  the reference, so a wrong baseline does not fail a correct
  candidate: it becomes the denominator of the size's speedup.

## How to detect

- Run the baseline through the gate you run your kernel through, at every size you time, against an fp32
  reference built from the dequantized operands (`../gemm/correctness.md` §1, the fifth of the gate
  shapes).
- Check which kernel the baseline launched, and that it is the path you meant to beat
  (`../gemm/measurement.md` §7).
- Flag MXFP4 and MXFP8 sizes whose K is not a multiple of 256. A benchmark that admits any K
  that is a multiple of 64 and times against aiter's `gemm_a4w4` includes them.
- Do not take a hipBLASLt MX output as the reference for an MX kernel: it is itself 49.6 dB from the fp32 result
  (E3).

## Fix or workaround

- Pad K to a multiple of 256 for these vendor calls (zero data, E8M0 scale 127) and price the
  padded work at the real M·N·K, or drop the vendor column at those shapes (S4).
- Build references from the dequantized operands in fp32, not from a vendor output.
- A size's speedup over a baseline that fails your check is not a measured gain. Report the baseline's
  failure with the size.

## Scope

- Measured in 2026-10 on MI355X, torch 2.11.0+rocm7.2 (HIP 7.2.26015), aiter `c16d44b93` from the image, called
  directly; the a4w4 and MXFP8 items again in 2026-11 on torch 2.11.0+rocm10.0.0 (HIP 7.15), the a4w4 one on aiter
  `dbd8bf5`, with scales padded `0x7F` (E5, E6).
- Scales whose padding columns are `0x00` do not reach the a4w4 item; scales padded with `0x7F` do.
  The hipBLASLt items matter
  when you build your own vendor column or reference.

## Evidence

The SNR values in E1, E2 and E4 are written down only in the notes of the adapter; the raw probe output was not
kept. E3 is in the run's own records.

- E1 [2026-10, gfx950, torch 2.11.0+rocm7.2 aiter c16d44b93, N=?, adapter validation: SNR against the dequantized fp32 reference] aiter a4w4 asm 5.9 / 7.7 / 10.9 dB at K = 2880 / 2944 / 3008 (MXFP4 dense GEMM)
- E2 [2026-10, gfx950, torch 2.11.0+rocm7.2 hipBLASLt, N=?, adapter validation, repeated calls] torch._scaled_mm MXFP8 26–49 dB varying from call to call at 8192×5120×2944, 4096×4096×2944, 8192×5120×4224; MXFP4 unaffected (MX dense GEMM)
- E3 [2026-10, gfx950, torch 2.11.0+rocm7.2 FlyDSL 0.3.2 aiter c16d44b93, N=2, head-to-head smoke run on GPU 4, SNR of two calls] torch._scaled_mm MXFP4 8192³ 49.603 / 49.603 dB and aiter a4w4 55.600 / 55.600 dB on the same problem; torch._scaled_mm MXFP8 16384×4096×4096 49.605 / 49.605 dB (MX dense GEMM)
- E4 [2026-10, gfx950, torch 2.11.0+rocm7.2 hipBLASLt, N=?, element-wise comparison with the fp32 reference] |out| ≤ |ref| on 99.4–100% of the elements of the MX bf16 outputs (the adapter's accuracy note)
- E5 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0 aiter dbd8bf5, N=3 calls per cell, SNR against the fp64 product of the dequantized operands] `gemm_a4w4` called with `bpreshuffle=True`, M×N 1024×1024 and 256×3072: with the scale padding at `0x7F`, 5.93–10.88 dB at K = 2880, 2944 and 3008 and 7.41–7.54 dB at K = 4160, none raising; with it at `0x00` or with aiter's own `dynamic_mxfp4_quant(shuffle=True)` scales, 55.59–55.62 dB; the CK path `gemm_a4w4_blockscale` raised `This GEMM is not supported!` at those K (MXFP4 dense GEMM)
- E6 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0 (HIP 7.15) hipBLASLt, N=8 calls per shape, SNR against the fp64 product of the dequantized operands, each call compared bitwise with the first] `torch._scaled_mm` MXFP8 with bf16 output: 49.60 dB at 4096×4096×2944, 8192×5120×4224 and three more shapes with K % 256 == 128, every call bit-identical and none raising, as at K % 256 == 0 (49.59–49.61 dB) (MX dense GEMM)
- S4 the adapter's K padding (data 0, scale 127) and its extra-work percentage
