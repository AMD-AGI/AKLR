---
title: "GEMM in Triton and Gluon on gfx950: AKLR's fp8 block-scaled GEMMs and bf16 decode GEMMs"
kind: topic
ops: [gemm]
dtypes: [fp8, bf16]
regimes: [decode, prefill]
gens: [gfx950]
triton: 3.8.0
---
# GEMM in Triton and Gluon (gfx950)
The Triton side of [`../gemm/`](../gemm/INDEX.md): how AKLR's 24 Gluon fp8 block-scaled (a8w8 blockwise) GEMMs and
its two plain-Triton bf16 decode GEMMs are cut by M, how their K loops feed the matrix core, and what was measured
on them. The `../gemm/` cards cover the related mechanisms. The CI speedups quoted are the values stored at AKLR
43e112e; later CI runs on main rewrote most of them (n7168 m = 1: 1.664 then, 1.728 since d908d01) *(field, E1)*.
## Route here when
- You write or tune a Triton or Gluon kernel for `gemm_a8w8_blockwise_scaled@1.0` (fp8 e4m3 operands, fp32 block
  scales, bf16 out) or `gemm@1.0` (bf16) and want the committed kernel closest to your Point (§3) or a K loop (§5).
- You need a tile and grid for an M or a new (N, K), tiny-M K-group folding, an async direct-to-LDS K loop, an LDS
  layout for fp8 operands, a tile order, ragged M, the epilogue, or split-K under a `bitwise` contract.
- You want the knobs a measurement backs (§11).
Elsewhere: the bf16 decode GEMM, `gemm/decode_gemm.md`; Gluon API facts, `gluon/memory.md`, `gluon/layouts.md`,
`gluon/mfma.md`; A/B method and launch options, `tuning.md`; reading the compiled kernel, `debugging.md` §1–§3;
rounding, `numerics.md`; files, entry and validation, `authoring.md`; loop designs and throughput arithmetic,
`../gemm/mainloop.md` §1, §9; MoE expert GEMMs, `moe.md`; every family on main, `kernels/kernel_library.md`.
## 1. The kernels and their Points
| Schema under `aklr/ops/gemm/impls/triton/` | Contract (compare) | Points | Language | CI speedup at 43e112e |
|---|---|---|---|---|
| `gemm_a8w8_blockwise_scaled_n3072_k1536/` | `gemm_a8w8_blockwise_scaled@1.0` (`bounded`) | m = 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 8193; one file each | Gluon | 1.191–1.486 *(field, E1)* |
| `gemm_a8w8_blockwise_scaled_n7168_k2048/` | same | m = 1, 2, 4, 8, 16, 32, 64, 128, 256; one file each | Gluon | 1.114–1.840 *(field, E1)* |
| `gemm_n2624_k6144/b88c8d5f28d86777.json` | `gemm@1.0` (`bitwise`) | m = 2, 4, 8, 64; one file | plain `tl.*` | 1.048–1.174 *(field, E1)* |
| `gemm_n6144_k2048/f806ad2b086d757d.json` | `gemm@1.0` (`bitwise`) | m = 2, 4, 8, 16, 32, 64; one file | plain `tl.*` | 1.064–1.292 *(field, E1)* |
- **What a8w8 pins.** `a (m, k)` and `b (n, k)` in `float8_e4m3fn`, an fp32 scale per 1 × 128 block of `a` and per
  128 × 128 block of `b`, bf16 `out`, `trans_b`, no bias. The call's `operand_format` and `a_scale.is_row_major` pick
  one of three layout variants, named in the schema (S1): `row_major` (`b` as stored, `a_scale (m, K/128)`), all 24
  Triton files, against aiter's `gemm_a8w8_blockscale`; `b_shuffled` (`b` in `Preshuffle(16, 16)`, `a_scale`
  column-major: `S_a[i, kb] = a_scale_flat[kb·m + i]`) and `b_shuffled_row_major_a_scale` (that `b`, row-major
  `a_scale`), all 28 FlyDSL, against `gemm_a8w8_blockscale_bpreshuffle`. The MiniMax-M2 source precedent loads preshuffled `b` into registers and converts it to
  the B operand layout (`gluon/memory.md` §13), unlike §5's explicit LDS loop; whether the conversion needs LDS
  depends on the load and operand layouts, not preshuffling alone.
- **The compare mode decides how K may be summed.** `bounded` (a8w8) checks each run on its own, so K may be
  reordered or split only while each output still meets its error bound; `bitwise` (`gemm@1.0`, the default) wants
  every run byte-identical to the first. A fixed-order split-K fold is one way to meet that requirement (§9; `aklr/core/contract/semantics/_compare.py`, S1).
- **Find the branch your Point runs.** An n3072 file is `main.py` (`run`), `adapter.py` and `kernel.py`, an n7168
  file one `kernel.py`; `fp8_a8w8_block_scaled_gemm_fn` picks kernel and constexprs in host `if m == …` branches of a
  multi-M snapshot (S2): copy constants from your Point's branch only. `run` takes the call keys, keyword-only,
  and returns nothing (`authoring.md` §4): `run(*, a, b, a_scale, b_scale, scale_major_mode, trans_b,
  operand_format, bias, out_dtype, out)` for a8w8, `run(*, a, b, trans_b, bias, out_dtype, out)` for `gemm@1.0`.
- **Declare `{"triton": ">=3.8"}`** (the field: `aklr/ops/README.md` § The kernel section and ABI). The n7168 files
  do; the n3072 files do not, yet from m = 16 up none compiles with a Triton 3.7.0 build for gfx950: `_reinterpret`
  without the `dtype` 3.7.0 requires (m = 16–64; m = 32 fails first, on a tuple index), `buffer_load_to_shared`
  offsets in a `DistributedLinearLayout` (m = 128), `compute_efficient_padded_shared_layout`, absent (m ≥ 256) (S3).
- **What has no Triton precedent.** A dense fp8 block-scaled GEMM with K > 2048 or a shuffled `b`. Nearest in Triton:
  the MoE expert GEMMs, folding the same 1 × 128 / 128 × 128 scales per 128-K group on 16 × 16-preshuffled fp8
  weights in registers: MiniMax-M2 FP8's gate-up at K = 3072 (`mfma_scaled`), Qwen2-MoE FP8's experts at K = 4096
  (fp8 widened to fp16) (S4; `moe.md` §7, also for MXFP4). In FlyDSL: the `b_shuffled` files, N = 4096 at K 256–8192
  (`../cases/a8w8_n4096_k_sweep.md`), K 7168 at N 768, 1536, 2048, 6144. bf16 long K: `gemm/decode_gemm.md` §5–§6.
  `gemm_a8w8@1.0`, `gemm_a4w4_mxfp4@1.0` and `bmm@1.0` have no Triton kernel. Start from §3's last three bullets.
## 2. One launch, one output tile per program, one partial per 128-K group
All 24 a8w8 files launch one kernel per call, one program per output tile, no persistent loop, all of K in each
program, no split-K, no atomics; dequantization is fused into the K loop, and the epilogue sums what a file kept apart
(folded groups at M ≤ 8, m = 8 and 32's second accumulator, m = 256's delayed folds), rounds to bf16 and stores
(§12). Weight-versus-FLOP estimates at M ≤ 256 use 4.5 MiB (n3072) or 14 MiB (n7168) of fp8 weight: n7168 at
M = 256 is 7.52 GFLOP, about 1.53 µs at 4923 TFLOP/s against 2.45 µs to read 14 MiB at 6 TB/s (estimated).
Timed only to M = 64: n3072 4.13–4.50 µs at M = 2–64; n7168 4.19–4.67 µs to M = 32 and 6.27–6.34 µs at M = 64,
4 runs *(field, E21)*. M = 128 and 256 were not timed, nor was the grid sizing; classify a call site with
`../gemm/small_m.md` §2, `../gemm/grid_pricing.md` §1.
Each logical 128-K scale-group contribution is scaled in fp32 before it joins the running sum. Tiny-M kernels
extract that contribution from packed products (§4), and 32 × 32 kernels may chain two 64-K instructions (§2).
For n3072 M = 16, the group starts from zero and is folded as follows (S2):
```python
# f3970decacc97225.json#kernel.py:424-426, 483 (n3072, m = 16)
partial = gl.amd.cdna4.mfma_scaled(
    a, None, "e4m3", b, None, "e4m3", gl.zeros((16, 16), gl.float32, LC)
)
...
acc = gl.fma(partial * sx, sw, acc)
```
`LC` is `AMDMFMALayout(4, [16, 16, 128], transposed=True, warps_per_cta=[1, 1])`; `a` and `b` carry
`DotOperandLayout(0 or 1, LC, k_width)`, `k_width` 16 at m = 16 and 32 at m = 32 and 64: one instruction per 128-K
group either way (S2; `gluon/mfma.md` §5). §5's skeleton writes all of it out.
- **Use `mfma_scaled` with `None` scales for plain fp8.** Gluon fills both scales with the E8M0 unit `0x7F` and
  emits a scaled dot (S6): the K = 128 (16 × 16) or K = 64 (32 × 32) F8F6F4 instruction, at plain fp8 `mfma`'s error
  (bit-identical on one 16 × 16, K = 128 tile) and 2.0× its rate (1.995–2.012× over 16×16×32 and 32×32×16, 4 runs)
  *(field, E20, E7)*. Every a8w8 kernel compiles to the 16x16x128 or 32x32x64 `v_mfma_scale_f32_*_f8f6f4` *(field, E17)*.
- **Keep the block scales out of the instruction.** They are fp32 values drawn log-uniform (S1), not powers of two:
  rounded to the E8M0 scale operands they gave a 17 % error on a test tile, an fp32 fold per 128-K group 7.2e-6
  *(field, E7)*. The fold's spelling differs per file (`fma(partial * sx, sw, acc)` in 8, `fma(partial, sx * sw,
  acc)` in 2, a first group seeding `acc` in 9); in a test kernel each stays within the bound (`numerics.md`).
- **32 × 32 tiles: chain two instructions per group.** n7168 M = 128 issues two `mfma_scaled` over the two 64-K halves
  on one accumulator, then folds once (S7): a test kernel took 148 fold VALU ops, 277 folding each half *(field, E7)*.
- **Tune 16 × 16 versus 32 × 32 in the kernel.** Register-resident chained `mfma_scaled` e4m3 ran 4899–4934 TFLOP/s
  at [16, 16, 128] and 4907–4949 at [32, 32, 64] (ratio 0.997–1.000, 4 runs) *(field, E20)*; no GEMM shape A/B.
## 3. Tile and grid by M
Start from the file of the nearest Point: its design is the regime's, its shape constants are not (last three bullets).
| M | n3072_k1536: file, programs × waves, output tile | CI at 43e112e *(field, E1)* | n7168_k2048: file, programs × waves, output tile | CI at 43e112e *(field, E1)* | Design |
|---|---|---|---|---|---|
| 1 | `b11f8d1f0bba8fed`, 768 × 1, 1 × 4 | 1.429 | `362ce3bb6200382e`, 448 × 1, 1 × 16 | 1.664 | n3072: K groups folded into A rows and B columns, every K panel issued before the first wait; n7168: all of A issued, B primed with 8 of 16 K groups and refilled in the loop (§4) |
| 2 | `6232bdbd302e4ec2`, 768 × 1, 2 × 4 | 1.397 | `ff2dc820165bd28a`, 448 × 1, 2 × 16 | 1.811 | n3072: same; n7168: A-only group folding, all A issued, 8 B groups primed and refilled (§4) |
| 4 | `08bd966bd0c72914`, 768 × 1, 4 × 4 | 1.307 | `0b1065ae9c0652b7`, 448 × 1, 4 × 16 | 1.679 | n3072: same; n7168: A-only group folding, all A issued, 8 B groups primed and refilled (§4) |
| 8 | `a5c62e519ce3fe45`, 192 × 2, 8 × 16 | 1.435 | `1e021d44010accb1`, 448 × 1, 8 × 16 | 1.840 | two K groups per MFMA (§4) |
| 16 | `f3970decacc97225`, 192 × 1, 16 × 16 | 1.406 | `1db61e7b0dc84d90`, 224 × 2, 16 × 32 | 1.725 | async K groups in flight: private LDS slots (n3072), a 4–6-slot ring (n7168) (§5) |
| 32 | `97933c3a190ede28`, 384 × 1, 16 × 16 | 1.418 | `ae4fa8a440b98f09`, 224 × 4, 32 × 32 | 1.759 | same |
| 64 | `4d4cb3c49f122deb`, 768 × 1, 16 × 16 | 1.412 | `06755ba3ee5eb085`, 224 × 4, 64 × 32 | 1.488 | async K groups in a 9-slot (n3072) or 3-slot (n7168) ring |
| 128 | `01110d0a95abc9f6`, 192 × 4, 32 × 64 | 1.477 | `1bbf98a368b3fcba`, 224 × 4, 64 × 64 | 1.114 | 4-wave tiles, LDS ring, 1–2 groups ahead (§5) |
| 256 | `616e1e1d43c2f2eb`, 192 × 4, 64 × 64 | 1.199 | `fee7e092eab87fc5`, 448 × 4, 64 × 64 | 1.138 | same, 32 × 32 × 64 MFMA |
| 1024–8192 | five files, 512–1152 × 4, 64 × 96 to 128 × 192 (§10) | 1.191–1.356 | — | — | single-buffered LDS, column panels, M bands (§7) |
| 8193 | `3045bc7c9725aa4d`, 1560 × 4, 128 × 128 and a 16-row tail | 1.486 | — | — | bulk compiled at M = 8192 (§7) |
- **At M ≤ 64: one wave per program (two or four in some files) and 192–768 programs,** each owning a 4- to
  32-column slice of the weight over all of K (traced program counts match these tiles); `a` is at most 128 KiB.
  n7168 halves the tile to 16 columns at M ≤ 8, 448 programs against 224 at M = 16–64.
- **At M = 128–256: four waves per program** on 32 × 64 or 64 × 64 tiles with an LDS ring (§5); from m = 1024,
  64–128-row by 96–192-column tiles, single-buffered (§7).
- **The margin is smallest in the middle.** n3072 is lowest at m = 256 (1.199) and m = 1024 (1.191), n7168 at
  M = 128–256 (1.114–1.138) against 1.66–1.84 at M ≤ 32 *(field, E1)*. n3072 m = 128 and 256 differ
  in MFMA shape, A prefetch distance, scale timing and tile order at once: their gap is not assigned to any of them.
- **Do not rank Points by small gaps.** The stored values of the M ≤ 64 Points with more than one CI run moved
  4–12 % between the runs that rewrote them *(field, E9)*: compare candidates in one session, alternating runs (§11).
- **A new N or K: turn every shape constant into a constexpr first** (S18). n3072 m = 16 and 32 hard-code N
  (`192 // STRIPES`, `3072` in the store offsets) and K / 128 = 12 (`gl.static_range(12)`, `11 - group`,
  `12 - PREFETCH`), m = 1–4 3 K panels (`wait_group(2 - panel)`), n7168 `STAGES = 8` and 16 scale columns, and n7168
  M = 64's `tile * 8 - ((tile * 293) >> 13) * 223` permutes 224 tiles only (on 128 it reaches 219). With only the
  host asserts relaxed, doubled K broke n3072 m = 1, 2, 4, 16, 32 and n7168 M = 64, doubled N all but m = 1–4 (MI355X).
- **Budget LDS against K.** A private 4 KiB slot per K group (§5's skeleton, the scheme of m = 16 and 32) costs K / 128 × 4 KiB: 48 KiB at K = 1536, 224 KiB at K = 7168; past K = 5120 that is over the 160 KiB a CDNA4 workgroup may allocate (S25). Triton 3.8.0 compiles the skeleton with 56 slots for gfx950 without an error (229376 bytes, 20 VGPRs spilled at `waves_per_eu` 3); only the launch raises `OutOfResources` (S25), so a compile-only check passes it. A fixed-slot ring is a tested option for the measured long-K shapes, not a cost-equivalence guarantee: with 8 slots the skeleton keeps 32 KiB, its output in bound at the tested K = 1024–7168 *(field, E11, E25)*; n7168 M = 16/32/64's ring takes 64/64/72 KiB. At M = 16, K = 4096, `PREFETCH` 5 and N = 4096 / 7168, 32 private slots were also correct but used 128 KiB and took 1.218 / 1.620× the ring's median time (one run, 12 balanced cycles; A/A within 0.08 %) *(field, E25)*. Unrolled code also grows (8 slots: 385 instructions and 100 registers at 12 groups, 1587 and 167 at 56); a rolled `[SLOTS, …]` allocation indexed `.index(t % SLOTS)` is a source precedent in the Qwen2-MoE router at K = 4096 (S26; `gluon/memory.md` §10, §12), whose long-K speed was not measured here.
- **Long K at small M: compare split placement and the actual K loops.** The n7168 ring's 32-column tiles (M = 16–64) make N / 32 programs, 128 at N = 4096: one per two CUs, a grid that read 0.75 (0.750–0.756, 3 runs) against one per CU on the Qwen2-MoE router projection (`tuning.md` §3) *(field, E23)*. FlyDSL splits K over eight waves at N = 4096, K = 4096, m = 1–2, and four waves at m = 4–32; at K = 8192 it uses four waves for m ≤ 32 (S28, `../cases/a8w8_n4096_k_sweep.md`). In Gluon: one K slice per wave in a rank-3 `AMDMFMALayout(4, [16, 16, 32], True, warps_per_cta=[SK, 1, 1])` with plain fp8 `mfma` (`mfma_scaled` aborts on rank 3, `gluon/mfma.md` §10; at half §2's rate, an estimated 0.8 µs of MFMA against 4.9 µs of weight reads at M = 32), an fp32 fold per group, then `gl.sum(acc, 0)` (MiniMax-M2 FP8's `_w13_local_split`, S4; `gluon/layouts.md` §6.7). In register-operand micro-kernels at M = 32, N = 4096, K = 7168, 256 such 16-column workgroups of 4 waves ran 2.2× faster than 128 32-column or 256 16-column programs without the split, and a split across workgroups with a reduce kernel took 1.06–1.08× their time (3 runs); with `gl.load` operands and an `sx * sw` fold the wave split compiles to 74 VGPRs, no spill, 4 KiB of LDS *(field, E8)*. A separate rotating-buffer sweep (M = 16 / 32 / 64 × seven (N, K), one run, 12 balanced cycles per configuration) measured that M = 32, N = 4096, K = 7168 split at 15.36 µs against 40.63 / 40.44 µs unsplit, but at M = 16 it took 1.277× a deep-prefetch skeleton at N = 2048, K = 7168 and 1.191 / 1.256× at N = 4096, K = 4096 / 7168. At M = 64, N = 4096 the unsplit-over-split gain was 1.554 at K = 7168, below 1.627 at K = 2048. All 112 arms were accurate. These compare loops and shapes; they do not show that long K should be split first, nor which change gives the gain *(field, E25)*.
## 4. Tiny M: fold K groups into the MFMA rows
With M < 16, a 16 × 16 × 128 instruction carrying M token rows wastes the other rows. In the same-PG A-and-B folding scheme used by n3072 M ≤ 8 and n7168 M = 8, the kernels fill them
with (token, K group) pairs (the other n7168 variants are described below): row `i` of the A tile holds token `i // PG`, K group `i % PG` of the current K panel;
row `j` of the B tile holds output column `j // PG`, group `j % PG`. The instruction computes every (A group,
B group) product; the kernel keeps the diagonal where the groups match, scales it, and sums the `PG` groups at the
end. Three bounds pick `PG`: M × PG ≤ 16 A rows; 16 / PG output columns per 16-row B instruction (4 at PG = 4, 8 per
wave at m = 8's PG = 2); PG divides K / 128. `group` gets the split diagonal's layout from a zero tile put through
the same reshape and splits, the carrier. n3072 m = 1 (`PG = 4`, 4 output columns, 3 panels of 4 groups, S8):
```python
# b11f8d1f0bba8fed.json#kernel.py:754-759, 782-787, 793
even_carrier, _ = gl.split(
    gl.reshape(gl.zeros((VM, VN), gl.float32, LC), (VM, BN, 2, 2))
)
carrier, _ = gl.split(even_carrier)
rows = gl.arange(0, VM, gl.SliceLayout(1, carrier.type.layout))
group = rows % PG
...
    even, odd = gl.split(gl.reshape(partial, (VM, BN, 2, 2)))
    p0, p2 = gl.split(even)
    p1, p3 = gl.split(odd)
    pair0 = gl.where((group[:, None] & 1) == 0, p0, p1)
    pair1 = gl.where((group[:, None] & 1) == 0, p2, p3)
    diagonal = gl.where(group[:, None] < 2, pair0, pair1)
...
result = gl.sum(gl.reshape(acc, (M, PG, BN)), 1)
```
- **Variants.** n3072 m = 2 and 4 run `_narrow_small_gemm` with `VM = M · PG` rows (m = 4 fills all 16, adds an LDS
  read-ahead and preloads its scales); m = 8 folds two groups on both operands, its 6 two-group products issued
  before the loop in 3 commit groups (S8). n7168 M = 8 folds the same way (`PG = 2`) and picks the diagonal with
  `gl.amd.slice`, `reshape`, `permute` and `convert_layout(..., assert_trivial=True)`, which fails to compile if the
  selection would move data between lanes (S19). n7168 M = 2 and 4 fold only A (4 K groups × M tokens) and take
  one 128-K group of 16 weight columns per MFMA, keeping that group's rows; M = 1 packs 4 groups on A and 2 on B,
  with an `example` tile as carrier (S19).
- **Confirm the selection stays in registers.** Compiled (Triton 3.8.0, gfx950), m = 1 has 131 instructions:
  3 `v_mfma_scale_f32_16x16x128_f8f6f4`, 9 `buffer_load_dwordx4 … lds`, 12 `ds_read_b128`, 10 `v_cndmask_b32` (9 for
  the selection, 1 masking the store's duplicate lanes), 2 DPP adds for the group sum, no `ds_bpermute`, no
  `s_barrier`. A selection moving data between lanes takes LDS or lane shuffles (cost: `gluon/layouts.md` §8).
- **Cost.** In the same-PG A-and-B scheme above, the instruction computes `PG` times the useful group products, but
  the MFMA count is tiny (3 per program at m = 1) and the call streams its weight. n3072 issues every K panel before
  the first wait. In the served n7168 M = 1, 2, 4 paths, 8 of B's 16 K groups are primed; M = 8 instead primes 3
  two-group panels (6 groups). The remaining panels are refilled in the loop (S19).
- **Measured only on these micro-kernel Points.** The served kernels' 1.307–1.435 (n3072) and 1.664–1.840 (n7168) *(field, E1)* do not separate folding. An M = 2, N = 7168, K = 2048 A-only fold took 0.925–0.934× the one-group time in 3 runs *(field, E8)*. A register-operand sweep over M = 1 / 2 / 4 / 8 × (N, K) = (3072, 1536), (7168, 2048), (7168, 7168), with A-only (FA), A-and-B diagonal (FAB) and one-group (U) arms, was accurate in all 36 (error / max|reference| 1.4e-5–1.8e-5). In one run of 12 balanced cycles per configuration, min(FA, FAB) / U was 0.309–0.512 at M ≤ 4; the M = 8 gains 1.594 / 1.516 / 2.832 were below the M = 1 gains 2.368 / 1.953 / 3.013, and the gain did not fall at every step of M. FA alone at M = 1, N = 7168, K = 2048 took 0.98585× (nominal paired 95 % interval 0.98459–0.98711), not a 3 % gain *(field, E25)*. These do not carry over to other folds or tiny-M operators. `gemm/decode_gemm.md` §7 compares unfolded bf16 16-row tiles with GEMV kernels at M = 1.
## 5. The K loop: async direct-to-LDS K groups with PREFETCH in flight
Gluon has no software pipeliner: `gluon_to_ttgir` runs no loop-scheduling or pipelining pass but `add_warp_pipeline`,
which left loops without `gl.amd.warp_pipeline_stage` blocks unpipelined, and `num_stages` changed only plain Triton's
`tl.dot` loop (S9; compiled). The served a8w8 K loops inspected are written by hand; their main K loops up to m = 256
use `gl.static_range` so that a tuple can hold the LDS buffers (`gluon/memory.md` §12). The skeleton is the m = 16
design reduced to its parts: one wave per 16 × 16 tile, each 128-K group one commit group of four 1 KiB copies
(16 rows × 64 bytes of A and of B, S10), `PREFETCH` groups ahead in a ring of `SLOTS` buffers, one fold per group
(compiles with Triton 3.8.0 for gfx950). Its ring was in bound at M = 16, N = 256, K = 1024–7168 *(field, E11)*, and
at M = 16 and (N, K) = (3072, 1536), (7168, 2048), (3072, 4096), (4096, 7168), 3 input seeds per shape, 8 ring slots
passed 12 tests and private slots 9 at K ≤ 4096 (private slots at K = 7168 exceed the LDS), with no bounded-compare
violation and error / max|reference| 0.00172–0.00302, in each of two runs *(field, E25)*. It is a starting point
tested on these shapes, not proven for other inputs or shapes:
```python
"""fp8 block-scaled GEMM at M = 16: one wave per 16 x 16 output tile, K groups copied to LDS PREFETCH ahead."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _issue(a_slot, b_slot, X, W, XS, WS, a_offs, b_offs, rows, ws_row, G: gl.constexpr, GROUPS: gl.constexpr):
    sx = gl.amd.cdna4.buffer_load(XS, rows * GROUPS + G)  # early scales: candidate order, not a proved speedup
    sw = gl.load(WS + ws_row * GROUPS + G)
    for h in gl.static_range(2):  # two 16 x 64-byte halves, each one lane-linear 1 KiB copy
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_slot.index(h), X, a_offs + G * 128 + h * 64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_slot.index(h), W, b_offs + G * 128 + h * 64)
    gl.amd.cdna4.async_copy.commit_group()
    return sx, sw


@gluon.jit
def _a8w8_m16(X, XS, W, WS, Y, N: gl.constexpr, K: gl.constexpr, PREFETCH: gl.constexpr, SLOTS: gl.constexpr):
    GROUPS: gl.constexpr = K // 128
    gl.static_assert((N % 128 == 0) & (K % 128 == 0) & (PREFETCH < SLOTS) & (SLOTS <= GROUPS))
    COPY: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])
    SHARED: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    READ: gl.constexpr = gl.SharedLinearLayout(  # the two halves read as one 16 x 128 operand
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [8, 0], [0, 64]])
    LC: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], transposed=True, warps_per_cta=[1, 1])
    pid_n = gl.program_id(0)
    r = gl.arange(0, 16, gl.SliceLayout(1, COPY))
    c = gl.arange(0, 64, gl.SliceLayout(0, COPY))
    a_offs = r[:, None] * K + c[None, :]
    b_offs = (pid_n * 16 + r)[:, None] * K + c[None, :]
    rows = gl.arange(0, 16, gl.SliceLayout(1, LC))
    ws_row = pid_n // 8  # the 128 x 128 b_scale block row of these 16 columns
    a_slots, b_slots = (), ()
    for s in gl.static_range(SLOTS):
        a_slots += (gl.allocate_shared_memory(X.dtype.element_ty, (2, 16, 64), SHARED),)
        b_slots += (gl.allocate_shared_memory(W.dtype.element_ty, (2, 16, 64), SHARED),)
    scales = ()
    for g in gl.static_range(PREFETCH):
        scales += (_issue(a_slots[g], b_slots[g], X, W, XS, WS, a_offs, b_offs, rows, ws_row, g, GROUPS),)
    acc = gl.zeros((16, 16), gl.float32, LC)
    for g in gl.static_range(GROUPS):
        gl.amd.cdna4.async_copy.wait_group(min(PREFETCH - 1, GROUPS - 1 - g))
        view = a_slots[g % SLOTS]._reinterpret(shape=(16, 128), layout=READ)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(view, gl.DotOperandLayout(0, LC, 16))
        view = b_slots[g % SLOTS]._reinterpret(shape=(16, 128), layout=READ).permute((1, 0))
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(view, gl.DotOperandLayout(1, LC, 16))
        partial = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", gl.zeros((16, 16), gl.float32, LC))
        if g + PREFETCH < GROUPS:
            if g + PREFETCH >= SLOTS:
                gl.barrier()  # the slot was read before: keep its refill after that read (one wave: no s_barrier)
            scales += (_issue(a_slots[(g + PREFETCH) % SLOTS], b_slots[(g + PREFETCH) % SLOTS], X, W, XS, WS,
                              a_offs, b_offs, rows, ws_row, g + PREFETCH, GROUPS),)
        sx, sw = scales[0]
        acc = gl.fma(partial * sx[:, None], sw, acc)
        scales = scales[1:]
    cols = pid_n * 16 + gl.arange(0, 16, gl.SliceLayout(0, LC))
    gl.amd.cdna4.buffer_store(acc.to(gl.bfloat16), Y, rows[:, None] * N + cols[None, :])


def run(*, a, b, a_scale, b_scale, scale_major_mode, trans_b, operand_format, bias, out_dtype, out):
    (m, k), n = a.shape, b.shape[0]
    assert m == 16  # one Point
    _a8w8_m16[(n // 16,)](a, a_scale, b, b_scale, out, N=n, K=k, PREFETCH=5, SLOTS=min(k // 128, 8),
                          num_warps=1, waves_per_eu=3)
```
Compiled offline (Triton 3.8.0, gfx950) at N = 3072, K = 1536, `PREFETCH` 5, one warp and `waves_per_eu` 3, SLOTS 8
and SLOTS 12 both emit 12 MFMAs and 48 direct-to-LDS copies, use 84 architectural VGPRs + 16 AGPRs (100) and 24
SGPRs, with no spill and no `s_barrier`; LDS is 32 KiB at SLOTS 8 and 48 KiB at SLOTS 12. Both wait `vmcnt(24)` eight
times, then 18 / 12 / 6 / 0, in 385 instructions each, but the ISA differs (SLOTS 12 replaces 15 `s_movk_i32` with
`s_mov_b32` and changes LDS addressing); not run. The host above picks SLOTS 8 at this K. The served m = 16 file sets
`READ_AHEAD` 2, `REFILL_BEFORE` and `EARLY_MFMA` (S2), so its wait is
`min(PREFETCH - READ_AHEAD - (0 if REFILL_BEFORE else 1), …)`.
- **Depth is measured per loop and Point, not a universal PREFETCH or byte threshold.** The served n3072 M16 loop (`PREFETCH` 5) compiles to steady `vmcnt(12)`, 3 to `vmcnt(4)`, 7 to `vmcnt(20)`; its read-ahead differs from the skeleton. In two runs P4/P5 time was 0.9979/0.9974, P3/P5 1.146/1.135, and P7/P5 0.9998/0.9968; individual P7 pairs ranged 0.990–1.004 and 0.969–1.019, not all within 1% *(field, E10)*. Three further runs measured P3/P5 1.115–1.140 and P7/P5 0.969–1.001 *(field, E18)*. The estimated 0.8/1.6/2.4 MB at P3/4/5 describes this Point only; n3072 M64 waits at `vmcnt(8)`, about 6 MB grid-wide (compiled; bytes estimated). In the skeleton at M = 16, SLOTS 12, N = 4096 and K = 4096 / 7168, one sweep of 12 balanced cycles per configuration measured P4/P5=1.434/1.651, P7/P5=1.186/1.218 (nominal paired 95% intervals 1.184–1.189/1.216–1.221), and P8/P5=0.881/0.847. P7 is not generally equivalent to P5, nor is P4 generally enough *(field, E25)*. Across N1024/2048/4096/7168/14336 and PREFETCH1–11, the median-defined knee (smallest depth within 3% of the measured minimum) was [8,8,8,5,3] at K4096 and [8,8,8,5,5] at K7168: 1.84–14.68 MB estimated outstanding. Configurations below 0.8 MB took 1.68–4.49× their shape's measured minimum. All 110 arms were accurate, but depth also changes scheduling and registers, so neither a fixed grid-wide byte count nor a per-wave one is shown to be the cause *(field, E25)*. In a pure read stream (three runs) about 1 MB in flight read 2.2–2.3 TB/s and 4 MB 5.7–6.3 TB/s *(field, E18)*: a stream property, not a GEMM threshold.
- **The read after the wait is synchronized for you; the refill after the read is not.** Membar puts a barrier after
  each `wait_group`, but its AMD filter drops the one between an async copy and a `load_shared_relaxed` read of one
  buffer, so a refill's write-after-read is yours (S9; `gluon/memory.md` §11, §12). By buffer scheme:
  - private slots (m = 16, 32): nothing is refilled, nothing to guard;
  - one-wave ring (m = 64, 9 slots, `PREFETCH` 4): barrier before a reused slot's refill; the file has one at
    first reuse (S11), no `s_barrier`, and the same read-before-refill order without it (compiled). This is not
    generally optional: in a one-wave toy refilling the just-read slot, omission moved DMA before the relaxed read
    and 2.8–5.1% of 2048 programs read refilled data in every launch (200 per delay setting, 4 settings, 2 runs).
    Keeping it preserved the order, still with no `s_barrier`, and gave zero wrong in 1600 launches *(field, E26)*.
  - four-wave ring with relaxed reads (m = 128, 256): refill, wait, `gl.barrier()`, read (S12).
    `SLOTS >= AHEAD + 2` (A: 4/2 at m128, 3/1 at m256; B: 4/2) puts the prior step's barrier between read/refill
    (compiled). A four-wave toy with 2 groups ahead and 3 slots gave wrong sums in every launch (200 per delay, four
    delays, two runs), including a pinned variant (48–89% of outputs wrong) whose refill stayed behind its own wave's
    earlier read. Four slots, or three plus a barrier before the refill, gave zero wrong *(field, E26)*; these toys
    show that AHEAD + 1 slots is not enough, not that the committed m = 128 / 256 files fail;
  - n7168 `0675` M = 64 reads with plain `.load` (3 slots, `PREFETCH` 2), so Membar guards its refill (8 source stages; the M = 64 launch emits eight barriers). Its M = 32 path uses 4 slots, `PREFETCH` 3, an explicit `gl.barrier()` after the wait and relaxed reads: the plain-load guard does not apply there. `1db6` M = 16 uses plain shared loads, 4 A slots and 4 or 6 B slots depending on strides (S13); source order is not timing or race evidence. n7168 M = 128 (`1bbf`) and M = 256 (`fee7`) add an inline `s_waitcnt lgkmcnt(0)` before `wait_group` + `gl.barrier()` (S7). Compiled offline with Triton 3.8.0 at their own launch settings (`num_warps` 4, `num_stages` 2, `waves_per_eu` 0), removing only those inline calls still leaves `lgkmcnt(0)` before every barrier with no LGKM-producing instruction in between: 16 barriers at M = 128, 18 at M = 256 (scale setup, 16 K groups, output conversion); at M = 256 the K-group drain can come well before `vmcnt(4)` + `s_barrier`. The inline drain is thus redundant for that barrier, read from the ISA only (not run): M = 128 goes from 236 to 238 VGPRs and 961 to 937 instructions with scheduling changes, M = 256 stays at 62 VGPRs and goes from 1175 to 1159 instructions; LDS stays 53264 / 53920 bytes, no spills;
  - single buffer (m ≥ 1024): barrier between last read and refill, unguarded by relaxed reads (compiled;
    §7; `../gemm/correctness.md` §4). A four-wave toy where waves refill rows others read gave wrong sums in
    all 200 launches at each of four delays (0–16 µs, two runs, 38–72% of outputs); with the barrier, zero wrong
    in 1600 launches *(field, E26)*. The m = 8192 file without it passed six validate runs: the failure is the toy's.
- **Count waits in committed groups** (`gluon/memory.md` §8): a toy with consecutive committed groups of 3, 2 and 1
  copies gave `vmcnt(0/1/3)` at `wait_group(0/1/2)` *(field, E27)*; n7168 M = 64's `WAITS = (1, 2, 2, 2, 2, 2, 2, 0)`
  survives as eight waits in its Triton 3.8.0 IR for gfx950. Its M = 64 launch emits 48 direct-to-LDS instructions and
  eight barriers; startup commits B and A together, refills commit A and B apart. Machine waits can be strengthened or
  subsumed, so raw `vmcnt` counts need not map one to one to the tuple (n3072's separate commits: S11). An
  uncommitted copy is not waited for, even by `wait_group(0)`.
- **Scale-before-copy is an A/B candidate, not a proven speedup.** A compiled two-copy toy waited at `vmcnt(1)` after copies (retiring them) or `vmcnt(3)` before; a four-copy toy at `vmcnt(0)` / `vmcnt(4)`. A timed 192-program, one-wave, eight-group toy took 6.078 µs with scales first against 6.030 after (1.008×, one run) *(field, E27)*. The skeleton's own scale waits were not read from its ISA, and no served GEMM was A/B tested. m = 16 loads all 12 groups' scales first (S15); m = 64 copies both tables into LDS as one unmasked commit group, `xs` at 16 bytes per lane (columns 12–15 redirected by `gl.where`), `ws` at 4 (S11); m = 128 keeps a 4-group register window (S12); m ≥ 1024 gathers them from LDS (S16); m = 32 loads them through an SGPR pointer (S15), and how the served launches lower these loads was not read from their ISA. m = 1 and 8 load scales behind their copies (S8).
- **Fold-delay measurements did not show a general benefit.** The files defer each group's fold by 1–3 groups
  (n3072 m = 256: 2; n7168 M = 2: 2, M = 4: 3), alternate two accumulators (n3072 m = 8 by product parity, m = 32 over
  a permuted K order), or pin it with an empty tied `gl.inline_asm_elementwise` (n7168 M = 256) (S5, S21;
  `gluon/wave_ops.md`). Without its delay m = 256 compiles to the same 24 `s_nop` wait cycles and 1 VALU read of an
  MFMA result within 8 instructions, and removing the delay gave B/A time per pair of 0.996–1.036 at m = 256 and
  0.974–1.049 at n7168 M = 2, over 3 runs of 4 pairs with no A/A control, so neither a gain nor equivalence is shown
  *(field, E23)*; the other forms are untested.
- **Unsetting `waves_per_eu` at m = 16:** 4.25–4.26 → 4.26–4.31 µs, speedup ratio 0.990–0.997 in 2 runs
  *(field, E13)*, at the dispatch floor (§11), so not an equivalence test; compiled, 88 → 84 VGPRs (AGPRs 8 → 0)
  *(field, E17)*. M ≤ 64 grids launch at most 896 waves (§3's table) for 1024 SIMDs, so there the hint acts through
  register allocation rather than occupancy (estimated from launch counts, not proof that register allocation is its only effect; `tuning.md` §8).
## 6. LDS layout: swizzle and padding
- **m = 16–64 (n3072): plain buffers read through a linear view.** Each K group's 16 rows × 128 bytes land as two
  (16, 64) halves in `SwizzledSharedLayout(1, 1, 1, [1, 0])` buffers, read as one 16 × 128 operand through a
  `SharedLinearLayout` `_reinterpret` view (B also `permute`d, S10). A gfx950 byte probe checked every A and B
  coordinate with three jointly unique patterns: zero mismatched bytes in both views, in two runs *(field, E25)*; that
  covers this layout only. The copy stays lane-linear, the only LDS DMA Triton 3.8.0 lowers on gfx950: 16 or 4 bytes
  per lane, consecutive lanes on consecutive LDS bytes (S14; `../hardware/gfx950/memory_and_sync.md` §3). A 16 ×
  64-byte copy layout into one (16, 128) row-major buffer breaks
  it: `LLVM Translation failed for operation: builtin.unrealized_conversion_cast` (compiled, Triton 3.8.0, gfx950).
- **`BANK_XOR`: a swizzle written on the global side, measured at m = 16.** The m = 16 file carries an unused variant
  (`BANK_XOR=False` at launch, S2): `copy_rows = r[:, None] ^ ((k[None, :] & 32) >> 3)` sends the lanes that write
  bytes 32–63 of row `r` to global row `r ^ 4`, the read view's basis `[4, 32]` undoes it, and both offsets then pass
  through `gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])` (S10). The hints are part of the recipe: the
  XOR hides the 16-byte runs from axis analysis, and without them the copy fails with the same error, while the
  n7168 XOR copy `((kk // 16) ^ (rows % 16)) * 16 + kk % 16` kept the runs visible without hints (compiled;
  `gluon/memory.md` §5). At m = 16, at the dispatch floor (§11), two runs read 4.245 → 4.263 and 4.301 → 4.241 µs
  *(field, E12)*, which resolves neither a difference nor equivalence: same 48 copies, 48 `ds_read_b128` and 12 MFMAs, 6 more
  instructions, 336 → 342 (compiled). Whether that loop is LDS-bound this cannot tell; swizzle against padding on an
  LDS-bound microbenchmark: `gluon/layouts.md` §5. Do not plan on counting conflicts on these workloads in the CI
  image: `rocprofv3` failed or timed out, and `gl.bank_conflicts` aborted (`debugging.md` §7; pricing a conflict:
  `../gemm/datapath.md` §3).
- **m ≥ 128 (n3072): padded layouts, one padding interval per wave instruction.** The operand buffers are
  `PaddedSharedLayout`s, written by hand (m = 128) or from `gl.amd.cdna4.compute_efficient_padded_shared_layout`
  (m ≥ 256), and the copy layout is `gl.DistributedLinearLayout(reg_bases=bases[:4] + bases[12:],
  lane_bases=bases[4:10], warp_bases=bases[10:12], block_bases=[], shape=shared_layout.shape)` of the shared layout's
  `offset_bases` (S12, S5): 16 contiguous bytes per lane, 64 lanes, 4 warps, one 1 KiB padding interval per wave
  instruction (on the helper's fp8 [128, 64] layout: 2 per wave, the copy bitwise on the MI355X). Bases from 12 on go
  back to registers as further copies; tiles over 4 KiB have them (m = 128's 128 × 64 B tile, both m = 256 tiles), and
  without them Triton 3.8.0 aborted on a 128 × 64 fp8 tile: `LLVM ERROR: Layout is expected to be surjective`. The
  helper returns `None` for `k_width` 32 (S5), so m = 256 passes a `k_width = 16` operand layout and reads with 32; the
  padding choice: `gluon/memory.md` §9.
- **n7168 M = 16–64: global-side XOR into a swizzled layout.** `_swizzled_k(rows, kk)`, copied into a linear
  `_reinterpret` view and read through `SwizzledSharedLayout(16, 1, 16, [1, 0])` (S13); not measured alone.
## 7. Large and ragged M
From m = 1024 (n3072) a program is four waves on a 64–128-row by 96–192-column tile, the K loop is a rolled
`range`, and LDS is single-buffered, with `wait_group(0)` per K group. The three-column-panel loops read their operands,
barrier and refill before the last panel's MFMAs (S16); m = 6144's generic loop refills before its MFMAs, while
m = 8193's rowwise bulk refills A and B separately after each buffer's last read (S24, S17).
- **Cut a 192-column tile into three 64-column panels**, each with its own accumulator. 192 columns span one and a
  half 128-column weight-scale blocks, so the middle panel takes one of two scale rows,
  `w1 = gl.where(pn % 2 == 0, w0, w2)` at m = 8192; m = 1024's 96-column tile picks it by `pn % 4` (S16).
- **The large-M files use no ring:** 512–1560 four-wave programs, one LDS buffer set and `wait_group(0)` at every
  wait in the source (the refill before the last panel's MFMAs, S16; the compiled waits were not read for every
  path); that other resident programs hide the K latency is inferred.
  Whether a ring would lift m = 1024 (1.191) and m = 6144 (1.246), two of the three lowest n3072 ratios *(field, E1)*,
  is not measured (`../gemm/mainloop.md` §1). The m = 6144 file serves its Point with its generic `_gemm_async_kernel`
  (128 × 128) and sends m = 2048, 4096, 8192 and m = 6145's bulk to `_gemm_strip_kernel` (S24), within 3 % of it at
  m = 6144 (B/A 0.981–0.992, 3 runs) *(field, E23)*.
- **Band the M tiles.** `_wide_kernel` (m = 4096) and `_gemm_wide_kernel` (m = 8192) walk `GROUP_M = 4` M tiles per
  N tile, then interleave N (S16); m = 2048 and 6144 use bands of 2 and 4 (`ORDER_M`, S24), m = 1024 none. A band of 1
  instead of 4 made the m = 4096 Point 1.86 % slower in one run of 4 balanced pairs *(field, E28)* (§8).
- **Ragged M: compile the bulk at a divisible M.** m = 8193 launches `cdiv(8193, 128) · 24 = 1560` programs; the 24 of the last M tile run `_compute_tail_tile` on 16 rows, the others `_compute_tile_rowwise` compiled with M = 8192, so every row mask folds away (S17); compiled at 8193 the bulk was 9–10 % slower (§11). Its stored 1.486 was the scenario's best at 43e112e *(field, E1)*, partly the baseline's doing: one row more took the baseline 61.77–62.81 → 68.75–70.25 µs (10.9–12.7 % longer), the kernel 46.91–49.22 → 46.32–48.59 µs (1.2–1.6 % less), 4 runs *(field, E21)*. A 65th 128-row baseline tile may explain it; not tested. The tail gets no padded layout at 16 rows (`compute_efficient_padded_shared_layout` returns `None` for a [16, 128] fp8 A operand of a 16 × 16 × 128 4-warp layout, a layout from 32 rows; compiled): it copies 32 A rows, masked by `(am[:, None] < M) | (M % AM == 0)`, and keeps 16 with `gl.amd.slice` (S17). A buffer descriptor is not a tensor-bounds check: an unmasked address past the logical tensor but inside the descriptor's byte range may read the backing storage; past that range, or through another descriptor, it was not tested (`gluon/memory.md` §3).
## 8. Tile order across XCDs
On the 10 launch grids tested, on the default stream, workgroup `i` ran on XCD `i % 8`, and each XCD has its own 4 MiB L2 *(field, E22)*
(`../gemm/schedule_and_locality.md` §2). On a new stream, such as AKLR's graph bench, the index rotates by a constant c ([gluon/memory.md](gluon/memory.md) §14); the placements below hold up to that rotation. The a8w8 files remap the program
id by regime (placement derived from the formulas, each a permutation of its grid; on multi-axis grids axis 0 is
dispatched fastest, measured on the same 10 grids *(field, E22)*):
| Point | Remap | Placement |
|---|---|---|
| n3072 m = 1–4 | `pid_n = (pid % 8) * (N // (BN * 8)) + pid // 8` (S8) | XCD (x + c) % 8 streams columns [384x, 384x + 384), c a per-stream constant |
| n3072 m = 16, 32 | `pid_n = (pid % STRIPES) * (192 // STRIPES) + pid // (STRIPES * (M // 16))`, 32 or 16 stripes (S18) | runs of 6 (m = 16) or 12 (m = 32) adjacent tiles per XCD; at m = 32 both M tiles of an N tile on one XCD |
| n3072 m = 64 | stripes inside 256-program super-blocks (S11) | the 4 M tiles of an N tile on one XCD |
| n7168 M = 2, 4 | `(tile % 8) * (N // 16 // 8) + tile // 8` (S19) | 56 adjacent tiles per XCD |
| n3072 m = 128 | none: N tile `program_id(0)` of a (48, 4) grid, M tile `program_id(1)` (S27) | the 4 M tiles of an N tile on one XCD |
| n3072 m = 256 | `pn = pn ^ (pm * 2)` on the same grid (S27) | CPU model: an N tile's 4 consumers on 4 XCDs; the real L2 fills were not measured |
| n7168 M = 128 | `tile_n = program_id(0) * 2 + parity` with a parity XOR, (56, 4) grid (S27) | both M tiles of an N tile on one XCD |
| n7168 M = 256 | 16-tile N clusters, N interleaved, (16, 4, 7) grid (S27) | all 4 M tiles of an N tile on one XCD |
| n3072 m ≥ 2048 | bands of 2 or 4 M tiles per N tile (`ORDER_M`, `GROUP_M`), N interleaved at m = 4096 and 8192 (S16, S24) | consecutive programs share a B panel; each XCD's programs touch a block of A tiles and B panels |
- **Measure the actual tile order.** At n3072/k1536, m = 4096 (aef7), `group_m` 4 → 1 increased mean call time
  25.981 → 26.465 µs (mean B/A 1.01862; paired 95 % t interval 1.0181–1.0191), 4 balanced AB/BA pairs in one run,
  every B report slower than every A report *(field, E28)*; an earlier fixed-order run gave 25.38 → 25.87 µs, B/A
  1.019–1.020 in all 4 pairs *(field, E14)*. A CPU mapping estimates total fill 30 → 52.5 MiB while weight-reader XCDs
  fall 4 → 1: the edit trades A and B locality, not simply weight-reader concentration. In the same run, removing the
  committed order slowed n3072 m = 64 1.120×, m = 128 1.059×, m = 2048 1.007×, m = 8192 1.009× and n7168 M = 128
  1.040× (4 pairs each) *(field, E28)*.
- **One weight reader does not imply zero remap effect.** At n7168/k2048, M = 64 (0675; A shared across N tiles),
  removing the permutation was equivalent within the ±1.5 % band set beforehand: mean B/A 0.99305, paired 95 % t
  interval 0.9856–1.0005, 4 balanced pairs in one run, which is not an exactly zero effect *(field, E28)*; an earlier
  fixed-order run read 6.262 µs in both arms *(field, E15)*. M = 2 (ff2d, 12 balanced pairs) changed by +2.35 % (mean
  B/A 1.02350; interval 1.0101–1.0372) despite mixed pair signs. n3072 M = 1 (b11f) and M = 16 (f397), 12 pairs each,
  stayed unresolved: their intervals 0.9895–1.0196 and 0.9953–1.0211 are wider than the ±1.5 % and ±1.3 % bands set
  beforehand *(field, E28)*. Neither sample ranges nor pair signs alone decide a mean effect.
- **m = 256 was tested directly, not inferred from m = 128.** At n3072/k1536, 616e (stored 1.199 at 43e112e) *(field, E1)*,
  replacing `pn ^ (pm * 2)` with `pn` puts the four weight consumers on one modeled XCD instead of
  four; CPU fill 21 → 7.5 MiB. Accuracy passed; 4 balanced pairs in one run gave 6.0875 → 5.906 µs, mean B/A 0.97012,
  paired 95 % t interval 0.9689–0.9713 *(field, E28)*, and three earlier runs of 4 pairs also favored the edit (B/A
  0.937–0.988) *(field, E23)*. This holds for the measured Point only.
- **The locality mechanism is unknown.** An eight-reader probe favored congruent ids by 1.76–2.17× (3 runs)
  *(field, E22)*, but real-kernel timing and CPU maps do not isolate L2 as the cause. Prove each launched grid's
  permutation (`gemm/decode_gemm.md` §3).
## 9. The bf16 decode GEMMs and split-K under a bitwise contract
The two plain-Triton files, against aiter's tuned `gemm_a16w16` (S1), give a workgroup one N tile across all rows of
`a` (S20, S22): over all of K at n6144 (192 workgroups) and at n2624 m ≤ 8 (164), over a sixth of K at n2624 m = 64 (41
tiles × 6 splits = 246, then a reduce launch; counts traced). They read the weight once with `.cg`, remap workgroups
per XCD and at some Points rotate each one's K start; choices and numbers: `gemm/decode_gemm.md` §1–§8. At 43e112e
they stored 1.116 / 1.129 / 1.292 (f806, M = 2 / 4 / 8) and 1.048 / 1.070 / 1.062 (b88c) *(field, E1)*; how much of
the spread over M is the baseline's was not separated. What Triton 3.8.0 adds on gfx950:

- **`num_stages=3` pipelines a plain `tl.dot` loop with direct-to-LDS copies:** n2624's m = 16 path compiles to 48
  `buffer_load_dwordx4 … lds`, half `sc0 nt` (the `.cg` weight), none at `num_stages=1`; Gluon gets none (§5).
- **Warps beyond the tile replicate it.** n2624 at m ≤ 8 runs a 16 × 16 tile on 4 warps, compiled to
  `warpsPerCTA = [4, 1]`, every wave issuing all 192 MFMAs: extra warps add copy bandwidth, not MFMA parallelism
  (kept, not separated).
- **`schedule_hint` has no effect in the tested Triton 3.8.0 build.** n6144 passes `schedule_hint="memory-bound-attention"` at m = 32
  (kept in `gemm/decode_gemm.md`); Triton 3.8.0 reads it nowhere ("right now no effect"), 3.7.0 made it the
  `iterative-ilp` scheduler (S22): ask for that with `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]`.
- **One M tile covers all rows.** `offs_m = tl.arange(0, BLOCK_M)` with `BLOCK_M` ≤ 64 is all of M and `run`
  checks no upper bound (S20, S22): at m = 96 and 128 both files ran but left rows above 63 uncomputed. The n6144
  output retained NaNs there; n2624's reduce read unwritten scratch rows and did not retain NaNs in every element. Add M tiling before a Point above 64.
- **m = 2, 4 and 8 run one binary in each file:** M is a runtime argument, nothing else changes (S20, S22).
- **Split K only with a fixed-order fold.** At m = 64 n2624 splits K six ways: each split stores its fp32 partial
  tile to `[split][m][n]` scratch (`torch.empty` per call, every element written before it is read: `aklr/ops/README.md`
  § The kernel section and ABI), and `_reduce_kernel` adds planes 0 … 5 in order and rounds once (S20).
- **XCD runs cross split boundaries.** `_xcd_remap` over the 246-tile grid (`pid_k = pid // N_TILES`) hands each XCD
  30–31 consecutive split-major tiles; 41-tile splits do not align, so five XCDs straddle two splits (computed from
  S20). One split per XCD (`../gemm/schedule_and_locality.md` §2) needs runs aligned to the splits.
- **Price the reduce launch under the ruler that judges it.** AKLR's bench times a call as the sum of its per-kernel device durations, from AKLR's rocprofiler-sdk collector since #33, Kineto before (S23). Under the Kineto timer, before #33, a trivial or empty `zero_()` kernel read about 4.1–4.2 µs, and at n2624 M = 64 the whole call read 4.2 µs more than its split GEMM kernel alone: the timer's cost per kernel, not the reduce kernel's own duration *(field, E19)*. In a bare graph replay an extra chained trivial kernel added about 1.8 µs *(field, E19)* (`authoring.md` §7). The committed source splits K at M = 64 only; split against no split was measured at M = 16 and 64 (`gemm/decode_gemm.md` §6). A one-launch fold, the last split adding all partials in order (`../gemm/wait_free_fold.md`), drops the reduce kernel but, with per-call counters, not the node that zeroes them: without it replays after the first went wrong (MI355X). Forms: `../gemm/grid_pricing.md` §2, `../gemm/small_m.md` §4.
- **The a8w8 GEMMs do not split K** (12 or 16 groups; `bounded` permits reordering only while each output meets its
  error bound). At M = 32, N = 4096, K = 7168, register-operand micro-kernels, the wave split took 20.8–21.0 µs
  against 22.3–22.6 µs for a split across workgroups plus a reduction (3 runs) *(field, E8)*. A rotating-buffer sweep
  (one run, 12 balanced cycles per configuration) kept that direction there (wave over cross-workgroup 0.863) but
  reversed it at K = 7168, N = 2048, M = 32 / 64: 1.149 / 1.380, nominal paired 95 % intervals 1.148–1.151 /
  1.377–1.383; at M = 32, N = 7168, K = 7168 the interval 0.99840–1.00178 picks no winner *(field, E25)*. An extra
  reduction launch does not imply a slower call; placement is shape-specific.
## 10. Per-Point configuration
Each Point's configuration is Python: host `if m == …` branches choose the kernel and its constexprs (S2), or a
`@gluon.constexpr_function` feeds kernel and launch (m = 32's `_schedule`, S15); `tuning.md`, `authoring.md`.
| m | Kernel, tile | Programs × waves | K pipeline | `waves_per_eu` | VGPRs (AGPRs within) | LDS KiB | CI at 43e112e (range over CI runs to then) |
|---|---|---|---|---|---|---|---|
| 1 | `_narrow_small_gemm`, 1 × 4 | 768 × 1 | 3 panels of 4 groups, all issued first | 8 | 40 (8) | 9.0 | 1.429 (1.316–1.429) |
| 2 | same, `iterative-ilp` scheduler, 2 × 4 | 768 × 1 | same | 4 | 72 (8) | 9.0 | 1.397 (1.329–1.451) |
| 4 | same, 4 × 4 | 768 × 1 | same, LDS read-ahead 1, scales preloaded | 4 | 44 (0) | 12.0 | 1.307 (1.286–1.404) |
| 8 | `_packed_m8_gemm`, 8 × 16 | 192 × 2 | 6 two-group products first, 3 commit groups | 2 | 74 (0) | 36.0 | 1.435 (1.372–1.531) |
| 16 | `_private_k128_gemm`, 16 × 16 | 192 × 1 | 12 private slots, `PREFETCH` 5, read-ahead 2 | 3 | 88 (8) | 48.0 | 1.406 (1.359–1.413) |
| 32 | `_block_scaled_gemm`, 16 × 16 | 384 × 1 | 12 private slots, `PREFETCH` 5, two accumulators | 8 | 64 (32) | 49.0 | 1.418 (1.326–1.483) |
| 64 | `_block_scaled_gemm`, 16 × 16 | 768 × 1 | 9-slot ring, `PREFETCH` 4 | 6 | 64 (24) | 37.3 | 1.412 (1.347–1.443) |
| 128 | `_decode_kernel`, 32 × 64 | 192 × 4 | 4 + 4 slots, A and B 2 groups ahead | — | 130 (0) | 49.3 | 1.477 (1.426–1.491) |
| 256 | `_decode_kernel`, 64 × 64, 32 × 32 × 64 | 192 × 4 | 3 + 4 slots, A 1 and B 2 groups ahead | — | 84 (0) | 56.8 | 1.199 (1.135–1.204) |
| 1024 | `_gemm_strip_kernel`, 64 × 96 | 512 × 4 | single buffer, 3 panels | — | 92 (0) | 24.7 | 1.191 (one run) |
| 2048 | `_gemm_three_panel_kernel`, 64 × 192 | 512 × 4 | single buffer, 3 panels | — | 144 (0) | 37.0 | 1.339 (one run) |
| 4096 | `_wide_kernel`, 128 × 192 | 512 × 4 | single buffer, 3 panels, `GROUP_M` 4 | — | 242 (0) | 49.3 | 1.356 (one run) |
| 6144 | `_gemm_async_kernel`, 128 × 128 | 1152 × 4 | single buffer | — | 158 (0) | 41.0 | 1.246 (1.245–1.284) |
| 8192 | `_gemm_wide_kernel`, 128 × 192 | 1024 × 4 | single buffer, 3 panels, `GROUP_M` 4 | — | 242 (0) | 49.3 | 1.298 (1.298–1.334) |
| 8193 | `_fp8_block_scaled_kernel`, 128 × 128 and tail | 1560 × 4 | single buffer | — | 128 (0) | 39.0 | 1.486 (1.454–1.536) |
| n7168: 1 | `_packed_k256_output`, 1 × 16 | 448 × 1 | A whole; B 4 slots of 2 groups, 8 of 16 primed | 1 | 80 (0) | 18.0 | 1.664 (1.664–1.845) |
| 2 | `_compact_m2_gemm`, 2 × 16 | 448 × 1 | A whole; B 10 one-group slots, 8 primed | — | 76 (0) | 24.0 | 1.811 (1.708–1.862) |
| 4 | `_packed_small`, 4 × 16 | 448 × 1 | A whole; B 3 slots of 4 groups, 8 primed | 3 | 96 (16) | 32.0 | 1.679 (1.679–1.808) |
| 8 | `_packed_eight`, 8 × 16 | 448 × 1 | 4 slots of 2 groups, 3 ahead | 2 | 108 (0) | 24.5 | 1.840 (1.791–1.885) |
| 16 | `_pipeline_output`, 16 × 32 | 224 × 2 | 8 stages of 256 K; 4 A + 6 B slots, 3 ahead | 3 | 96 (16) | 64.0 | 1.725 (1.725–1.877) |
| 32 | same, 32 × 32 | 224 × 4 | 8 stages of 256 K; 4 slots, 3 ahead | 1 | 98 (0) | 66.1 | 1.759 (one run) |
| 64 | same, 64 × 32 | 224 × 4 | 8 stages of 256 K; 3 slots, 2 ahead | 2 | 152 (24) | 72.0 | 1.488 (one run) |
| 128 | `_fp8_pipeline_kernel`, 64 × 64, 32 × 32 × 64 | 224 × 4 | 3 slots, 2 groups ahead | — | 236 (0) | 52.0 | 1.114 (one run) |
| 256 | `_fp8_full_pipeline_kernel`, 64 × 64, 32 × 32 × 64 | 448 × 4 | 3 slots, 2 groups ahead | — | 62 (0) | 52.7 | 1.138 (one run) |
VGPRs, AGPRs and LDS come from the compiled census *(field, E17)*; CI values are those stored at 43e112e and the
ranges those stored since each file's first CI run *(field, E1, E9)*. n7168 M = 16–64 walk K in 8 stages of `BK = 256`
split into two 128-K groups with `gl.amd.slice` (S13); M = 256 writes its padded layouts by hand, M = 128 uses swizzled ones (S7).
- **What changes as M grows.** Waves per program 1 (n3072 m = 8: 2; n7168 M = 16: 2, M = 32, 64: 4) up to M = 64, then
  4; K groups issued ahead (`PREFETCH`): all of K (n3072 m ≤ 8), 5 (m = 16, 32), 4 (m = 64), 1–2 in a four-wave ring
  (m = 128, 256), a single buffer from m = 1024; `waves_per_eu` only at M ≤ 64; `iterative-ilp` only at m = 2; 32 × 32
  × 64 only at n3072 m = 256, n7168 M ≥ 128; no spills *(field, E17)*.
- **Which constants are measured:** those in §11, the fold delays (§5), m = 256's tile order (§8) and m = 6144's
  kernel (§7); the other constants are kept, not separated.
## 11. What the measurements say
On one idle MI355X, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), the committed schema
(A) against a copy with one change (B), 4 runs per arm alternated:
| Change (B against A) | B/A statistic (speedup ratio unless marked mean time) | Kernel time A → B | Use | Evidence |
|---|---|---|---|---|
| n3072 m = 16: `PREFETCH` 5 → 3; 5 → 4; 5 → 7 | 0.87–0.88; 1.00; 1.00 | 4.23–4.27 → 4.85 µs, all 8 pairs of 2 runs; 4.27–4.31 → 4.26–4.30; 4.29–4.31 → 4.29–4.30 µs | served Point only: 4/5 close; not a universal depth (§5) | *(field, E10)* |
| n3072 m = 16: `BANK_XOR` on; `waves_per_eu` 3 → unset | 0.995/1.014; 0.990–0.997 | 4.245 → 4.263 and 4.301 → 4.241 µs; 4.251 → 4.258 and 4.264 → 4.307 µs, 2 runs per change | finite dispatch-floor observations, not an equivalence test (§6; §5) | *(field, E12, E13)* |
| n3072 m = 4096: `group_m` 4 → 1 | mean time B/A 1.01862 | 25.981 → 26.465 µs; N=4 balanced pairs, one run | measured Point only; paired 95% t interval 1.0181–1.0191 (§8) | *(field, E28)* |
| n3072 m = 8193: bulk compiled at M = 8193, not 8192 | 0.91 | 46.66–47.62 → 51.38–52.07 µs, 2 runs | compile the bulk at 8192 (§7) | *(field, E5)* |
| n7168 m = 64: tile permutation removed | mean time B/A 0.99305 | 6.280 → 6.2365 µs; N=4 balanced pairs, one run | equivalent within ±1.5%, not zero; paired 95% t interval 0.9856–1.0005 (§8) | *(field, E28)* |
- **The m = 16 call read at AKLR's dispatch floor under the Kineto timer (before #33).** Its kernel read 4.2–4.3 µs *(field, E10, E13, E16)*, and a trivial kernel or an empty `zero_()` read about 4.17 µs under the same timer *(field, E19)*; see [the dispatch-floor pitfall](../pitfalls/aklr_dispatch_floor.md). The runner then forced queue profiling (S23); whether that alone set the floor was not separated. At m = 16 `PREFETCH` 3 still showed a loss, but similar readings for other edits show neither a gain nor equivalence without a paired uncertainty, and the reading's rate (4.8 MB in 4.3 µs, about 1.1 TB/s, estimated) is a lower bound. The A arm's 4-run means read 1.449–1.472 over four sessions in one job (1.5–1.6 % apart; single runs within a session 2.0–3.1 %), in 2 runs *(field, E16)*. Test a speed-up at a Point above the floor (n7168 M = 64: 6.38–6.40 µs; m = 4096: 25.39–25.65 µs), where an identical copy alternated with A in one job read the same kernel time to 0.08 % and 0.03 % in these runs, not a general error bound; compare the paired uncertainty in `report.json` (`tuning.md` §2) *(field, E16)*.
- **Budget the Points above the floor** (`tuning.md` §13, `../gemm/mainloop.md` §9; numbers estimated). m = 4096 does 38.66 GFLOP and moves 36.18 MB (bf16 out 25.17): estimated 7.85 µs at 4923 TFLOP/s, 6.03 µs at 6 TB/s; measured 25.16–25.74 µs, 30.51–31.20% of MFMA peak and 1.406–1.438 TB/s. n7168 M = 64 moves 15.73 MB in 6.272–6.343 µs, 2.480–2.508 TB/s (4 runs) *(field, E21)*. Multiplying the partial by one `xs * sw` product per group instead of by `xs` then `sw` took m = 4096 from 25.18–25.77 → 23.47–23.87 µs, 6.4–7.6 % less time (1.069–1.082×; 4 runs, 4 alternating validate rounds per arm) *(field, E21)*. A second buffer and a barrier per group were not tried; this fold does not explain the whole gap to the roofline. At n7168, M = 64, a correctly waited 4-slot, `PREFETCH` 3 variant passed 24 fresh seeds and all 24 validate reports; 12 balanced AB/BA pairs measured 6.315 → 6.187 µs, geometric B/A 0.9798 (nominal paired 95 % interval 0.9792–0.9805), the slowest pairs 0.9816 and 0.9822; six A/A pairs gave 1.0007 (0.9991–1.0022) *(field, E25)*. With four slots fixed, `PREFETCH` 2 and 3 took 0.700–0.707× and 0.727–0.735× the time of `PREFETCH` 1 under both padding settings tested, 12 pairs each *(field, E29)*. These variants are faster, but LDS, live ranges, waits and scheduling change with the depth, so bytes in flight is not shown to be the bottleneck; variants that waited too little were inaccurate and are not timing evidence.
- **Not measured on these kernels:** the K-group folding, the LDS read-ahead, where the scale loads go, the store hint at the other Points (§12), the weight copy's cache policy (only the m = 8193 file puts `.ca` on it, compiled like no modifier to `buffer_load_dwordx4 … lds` without cache bits, `.cg` to `… lds sc0 nt` *(field, E17)*; the other 23 files' ISA bits are unread; `.cg` sped up a 1 GiB Gluon register-load stream 13.2–13.5 % in 3 runs *(field, E18)*, `gluon/memory.md` §6, and moved MoE weight streams either way (at 256 tokens removing it cost MiniMax E64 2.9–3.5 %, adding it DeepSeek-V2 0.5–0.7 %, 2 runs) *(field, E24)*, `moe.md` §7), remaps at Points other than §8's, 16 × 16 against 32 × 32 in one kernel, async copies against register loads (async 0.96–1.01× register-only's throughput on a Gluon tile stream, 1.3–4.3 % below it at 2048 programs, within 1.5 % on the 8-token Qwen2-MoE router call, 4 runs *(field, E18)*, `gluon/memory.md` §8; in FlyDSL, staging through VGPRs lost to DMA on a bf16 loop at M = 16384, `../gemm/mainloop.md`), and `gl.amd.warp_pipeline_stage`, which the served GEMM paths do not use, for the four-wave kernels (`gluon/memory.md` §12). FlyDSL results for such knobs: `../gemm/epilogue.md` §5, `../gemm/small_m.md` §6.
## 12. Epilogue: round, then store
- **Round to bf16 before a layout change:** a [32, 64] exchange then takes 4096 bytes of LDS, not 8192, for the same
  output (compiled; MI355X). m = 128 goes through an XOR-swizzled exchange into `BlockedLayout([1, 8], [8, 8], [4, 1],
  [1, 0])`, stores with `cache=".wt"` and calls `_keep_alive()`; what removing it changes was not compiled (S12,
  `gluon/memory.md` §15). n7168 M = 64 converts `acc.to(gl.bfloat16)` to a 16-byte-per-lane layout (S13). A lane stores
  2 or 4 bytes at M ≤ 8, 8 at M = 16–64 (n7168 M = 64: 16), 16 from M = 128 *(field, E17)*; `../gemm/epilogue.md`.
- **The store hint is chosen per Point; measured only at m = 6144 and 8193.** Compiled: no cache bits at n3072 m = 1–4,
  16–64 and n7168 M = 2, 4, 16, 32; `.wt` (`sc0 sc1`) at n3072 m = 8, 128, 256, 1024, 4096, 8192 and n7168 M = 1, 8, 64,
  256; `.cs` (`nt`, through `gl.store`) at m = 2048, 6144, 8193 and n7168 M = 128 *(field, E17)*. On a Gluon write stream
  no hinted store beat the unhinted one by more than 0.7 %, within the 0.3–1.5 % that identically compiled stores differ,
  and `.cs` was slowest, 5.5–7.2 % below it (3 runs) *(field, E18)*, `gluon/memory.md` §6; `.cs` → none (36 and 48 MiB of
  bf16 output) cost 1.7–1.9 % at m = 6144 and gained 0.1–0.7 % at m = 8193 (runs overlapping), all 8 pairs of 2 runs
  *(field, E24)*. In a four-store toy, `.wt` through `gl.store` inserted a wait after each store; `buffer_store` did not.
## Evidence
AKLR paths and line numbers are on main `0e02a81` unless a line names another commit; each E line names the commit,
CI run or PR its numbers come from. #64 renamed the 15 n3072_k1536 schemas after 43e112e and added one line at the
top of each `kernel.py`; this page uses their names on main. Triton paths are `triton-lang/triton@c01b677` unless a
line names a wheel.
- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 per value, AKLR CI bench (graph replay, 1 GiB read flush, Kineto timer), speedup against the schema's baseline] values stored at AKLR@43e112e, each written by the CI run that last measured the file: n3072_k1536 m = 1, 2, 4, 8, 16, 32, 128: 1.429, 1.397, 1.307, 1.435, 1.406, 1.418, 1.477 and n7168_k2048 m = 1, 2, 8, 16: 1.664, 1.811, 1.840, 1.725 (CI run 41991630592 at 43e112e); n3072 m = 64, 256, 8193: 1.412, 1.199, 1.486, gemm_n2624_k6144 m = 2, 4, 8, 64: 1.048, 1.070, 1.062, 1.174 and gemm_n6144_k2048 m = 2, 4, 8, 16, 32, 64: 1.116, 1.129, 1.292, 1.138, 1.064, 1.116 (CI run 41976547645 at 90b9d50); n3072 m = 6144, 8192: 1.246, 1.298 (CI run 41964469977 at 93eba6d); n3072 m = 1024, 2048, 4096: 1.191, 1.339, 1.356 and n7168 m = 32, 64, 128, 256: 1.759, 1.488, 1.114, 1.138 (CI run 41197527705 at a29581b); n7168 m = 4: 1.679 (CI run 41956295934 at 5f63e8a); n7168 m = 1 later stored 1.728 (CI run 42000899019 at d908d01) (schemas under aklr/ops/gemm/impls/triton/: n3072_k1536 b11f8d1f0bba8fed, 6232bdbd302e4ec2, 08bd966bd0c72914, a5c62e519ce3fe45, f3970decacc97225, 97933c3a190ede28, 4d4cb3c49f122deb, 01110d0a95abc9f6, 616e1e1d43c2f2eb, e5af2435eab17788, c56c343a3be1283d, aef7bbb04ccbe710, 584a68a298cb2da9, d4457ae42fce5327, 3045bc7c9725aa4d; n7168_k2048 362ce3bb6200382e, ff2dc820165bd28a, 0b1065ae9c0652b7, 1e021d44010accb1, 1db61e7b0dc84d90, ae4fa8a440b98f09, 06755ba3ee5eb085, 1bbf98a368b3fcba, fee7e092eab87fc5; b88c8d5f28d86777; f806ad2b086d757d)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] n3072 m = 8193, the bulk compiled at M = 8193 instead of M − M % BM = 8192: kernel 47.62 → 52.07 µs, speedup 1.468 → 1.343 (0.91); the other run 46.66 → 51.38 µs, speedup 1.492 → 1.354 (0.91) (schema 3045bc7c9725aa4d; #89)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, Gluon test kernels against a float64 reference, error relative to Σ|a·b|] `mfma_scaled` with `None` scales at 16 × 16 × 128, 4 seeds: worst error 1.031e-05, plain fp8 `mfma` 1.031e-05 (32 × 32 × 64: 1.013e-05); block scales drawn 2^U[−3, 0] rounded to the E8M0 scale operands: error 0.1688 on one 16 × 16 × 128 tile, one instruction per 128-K group and an fp32 fold: 7.223e-06; the three fold spellings in a 64 × 256, K = 1024 kernel with fp32 output differ bitwise in 11.0–52.7 % of elements, each within `DotProductBound(K = 1024, fp32)`; 32 × 32 × 64 halves chained 8 of 8, error 5.081e-06, 148 fold VALU instructions against 277 folding each half (#89)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, register-operand Gluon micro-kernels, median of 3 alternated measurements, each the median of 3 rounds of 10 CUDA-graph replays behind a 1 GiB read flush] M = 32, N = 4096, K = 7168: 256 16-column workgroups splitting K over 4 waves 20.80–21.01 µs, 128 two-wave 32-column programs 45.38–45.50 µs, 256 one-wave 16-column programs 45.48–45.54 µs (the split 0.457–0.463 of the faster, 2.16–2.19×), K split over 4 one-wave workgroups per tile plus a reduce kernel 22.26–22.59 µs (1.059–1.077× the wave split), error 0.00244 in all four; compiled (4 warps, `gl.load` operands, `fma(part, sx * sw, acc)` fold), the wave split has 74 VGPRs, no spill, 4096 bytes of LDS and 16 MFMAs in all 3 runs; M = 2, N = 7168, K = 2048, 448 one-wave programs, 16 MFMAs each: 8 K groups × 2 tokens in A's 16 rows 16.12–16.26 µs, one group per A tile 17.26–17.58 µs (0.925–0.934), error 1.8e-05 both (#89)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1–7 per Point, AKLR CI bench (graph replay, 1 GiB read flush, Kineto timer), speedup against the schema's baseline] the stored speedup of unchanged sources across the CI measurement commits that rewrote each file, up to 43e112e: n3072_k1536 m = 1: 1.316–1.429, 2: 1.329–1.451, 4: 1.286–1.404, 8: 1.372–1.531, 16: 1.359–1.413, 32: 1.326–1.483, 64: 1.347–1.443, 128: 1.426–1.491, 256: 1.135–1.204, 6144: 1.245–1.284, 8192: 1.298–1.334, 8193: 1.454–1.536, 1024 / 2048 / 4096 one run each; n7168_k2048 m = 1: 1.664–1.845, 2: 1.708–1.862, 4: 1.679–1.808, 8: 1.791–1.885, 16: 1.725–1.877, 32 / 64 / 128 / 256 one run each (AKLR commits a29581b, f23abc6, 369a3c0, 62a5fe8, 3fb9033, 5f63e8a, 93eba6d, 90b9d50, 43e112e; `git log -- aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048`)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] n3072 m = 16, `PREFETCH` 5 → 3: speedup 1.475 → 1.285 (0.87) and 1.468 → 1.294 (0.88), kernel 4.232 → 4.851 and 4.270 → 4.845 µs, B/A time 1.124–1.150 in all 8 pairs; 5 → 4: 1.456 → 1.459 and 1.445 → 1.449 (1.00), 4.272 → 4.263 and 4.312 → 4.301 µs, pairs 0.979–1.032; 5 → 7: 1.448 → 1.449 and 1.448 → 1.452 (1.00), 4.294 → 4.293 and 4.309 → 4.295 µs, pairs 0.990–1.004 and 0.969–1.019; compiled steady waits `vmcnt(12)` at 5, `vmcnt(4)` at 3, `vmcnt(20)` at 7 (schema f3970decacc97225; #89)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, §5's skeleton launched at M = 16, N = 256, `PREFETCH` 5, against a float32 per-group reference, bound 1 % of the reference's largest value] 8 slots (32768 bytes of LDS) at K = 1024, 2048, 4096, 5120, 7168: max error 0.075–0.213 against bounds 0.333–0.979; 40 slots (163840 bytes) at K = 5120: 0.2175 against 0.6816; 56 slots at K = 7168: the launch raises `OutOfResources` (229376 bytes required, 163840 the limit) (#89)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] n3072 m = 16, `BANK_XOR` False → True: speedup 1.460 → 1.453 (0.995), kernel 4.245 → 4.263 µs; the other run 1.453 → 1.474 (1.014), 4.301 → 4.241 µs; B's output bitwise equal to A's (schema f3970decacc97225; #89)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] n3072 m = 16, `waves_per_eu` 3 → 0 (unset): speedup B/A 0.997 and 0.990, kernel 4.251 → 4.258 and 4.264 → 4.307 µs (schema f3970decacc97225; #89)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated, arms in a fixed order] n3072 m = 4096, `group_m` 4 → 1: speedup 1.351 → 1.325 (0.98), kernel 25.38 → 25.87 µs (A 25.36–25.40, B 25.85–25.88), B/A 1.019–1.020 in all 4 pairs (schema aef7bbb04ccbe710; #89)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated, arms in a fixed order] n7168 m = 64, the program → N-tile permutation removed: speedup 1.479 → 1.481 (1.00), kernel 6.262 → 6.262 µs, round spreads 1.3 % and 1.8 % (schema 06755ba3ee5eb085; #89)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, the committed schema against itself] n3072 m = 16, the A arm in four sessions of 4 runs, back to back in one job: means 1.4507, 1.4642, 1.4722, 1.4536 (range 1.48 %) and 1.4488, 1.4578, 1.4720, 1.4552 (1.60 %), run spreads 2.38, 2.55, 2.03, 2.47 % and 2.51, 2.83, 3.10, 2.08 % per session; kernel time 4.296–4.297 µs in validate, 4.217–4.229 µs in `bench_cudagraph`; n7168 m = 64 6.376–6.402 µs and n3072 m = 4096 25.39–25.65 µs, both ≥ 1.4 × 4.1 µs, A and an identical copy alternated with it in one job 0.063–0.083 % and 0.017–0.025 % apart in mean kernel time (schemas f3970decacc97225, 06755ba3ee5eb085, aef7bbb04ccbe710; #89)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, census of the kernels compiled by one `python -m aklr validate --stages audit,bench` run per schema, one GPU; m = 16 also in 2 compile-only runs] the 24 a8w8 Points, one AKLR kernel each: MFMAs only `v_mfma_scale_f32_16x16x128_f8f6f4` and `v_mfma_scale_f32_32x32x64_f8f6f4`, the latter at (3072, 256), (7168, 128), (7168, 256); `.vgpr_count` (`.agpr_count`) and `shared` / 1024 equal §10's table at all 24; VGPR and SGPR spills and `.private_segment_fixed_size` 0 at all 24; `num_warps` 1 (10 kernels), 2 (2), 4 (12), `waves_per_eu` in 13; stores `store_short` at n3072 m = 1–4 and n7168 M = 1–8, `store_dword` at m = 8, `dwordx2` at m = 16–64 and n7168 M = 16, 32, `dwordx4` from m = 128 and at n7168 M = 64–256; store bits none, `sc0 sc1` or `nt` (all `global_store`) at the Points of §12; `.ca` in the m = 8193 source only, its `… lds` loads without cache bits and the same with `.ca` removed, `.cg` giving `nt sc0`; m = 16's `_private_k128_gemm` with `waves_per_eu` 3: 88 VGPRs, 8 AGPRs, unset: 84, 0, no spill (the 24 a8w8 schemas at AKLR 43e112e; #89)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, Gluon streaming microbenchmarks of a 1 GiB bf16 tensor, 16 bytes per lane; AKLR validate A/Bs (Kineto timer), arms alternated] reads with `.cg` buffer loads: 256 programs × 4 waves 2.24–2.29 TB/s with 1 load in flight per lane (about 1 MB in flight; 1024 one-wave programs 2.16–2.20), 5.70–5.76 with 4 (about 4 MB); the other 4 MB configurations 6.10–6.12 (1024 programs × 4 waves, 1 load), 6.21–6.23 (512 × 8 waves) and 6.29–6.33 TB/s (512 × 4 waves, 2 loads); n3072 m = 16 (f3970decacc97225) in validate with queue profiling forced, 3 rounds per arm: `PREFETCH` 3 against 5 call time 1.115–1.140×, speedup 0.877–0.898, 7 against 5 0.969–1.001×; 2048 programs × 4 warps, median of 9 rounds of 10 launches, variants interleaved: reads, `.cg` over none 1.132–1.135 on `buffer_load` and on `gl.load`; writes over unhinted, `buffer_store` / `gl.store`: `.wb` 0.989–1.007 / 1.002–1.005, `.cg` 0.987–1.005 / 0.990–1.005, `.wt` 0.998–1.005 / 0.983–0.996, `.cs` 0.933–0.942 / 0.928–0.945; none, `.wb` and `.cg` (the same store without cache bits) 0.3–1.5 % apart per run and path; 64 × 64 bf16 tiles, 4-warp programs, median of 7 × 5 launches, 4 runs: async copies (1 or 2 tiles in flight) over register-only loads 0.96–1.01 (1.3–4.3 % slower at 2048 programs); the 8-token Qwen2-MoE router (4801e216dcff290f) with register instead of async loads, call time −0.1 % to +1.5 % (#89)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, AKLR's bench timer before #33 (Kineto kernel durations, queue profiling forced) and a bare CUDA-graph replay timed with CUDA events] one trivial kernel 4.17 µs, two 8.28 µs, an empty `zero_()` 4.17 µs, n2624's `_reduce_kernel` alone 4.22 µs; the n2624 m = 64 call 12.57 µs against 8.37 µs for its split GEMM kernel alone (+4.2 µs); one more chained trivial Gluon kernel in a bare graph 1.82 µs of wall time at grid 1, 1.84 µs at grid 256; four earlier runs measured one kernel at 4.16–4.32 µs and the chained kernel at 1.77 µs, grid 1 (schema b88c8d5f28d86777; #89)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, Gluon MFMA-rate microbenchmark, register-resident operands] `mfma_scaled` e4m3 with `None` scales: 4899–4934 TFLOP/s at [16, 16, 128], 4907–4949 at [32, 32, 64] (16 × 16 over 32 × 32 0.997–1.000); plain fp8 `mfma` 2444–2468 at 16×16×32, 2441–2467 at 32×32×16, the scaled rate 1.995–2.008× and 2.006–2.012× of it; on one seeded 16 × 16, K = 128 tile the `None`-scale output bit-identical to plain fp8 `mfma`, both 2.85e-05 of the reference's largest value against float64 (#89)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated; bytes and FLOPs counted from the shapes] n3072 m = 4096, the fold `… * xs_acc * sw` → `… * (xs_acc * sw)` in `_wide_kernel`'s K loop: speedup 1.348–1.369 → 1.456–1.468, kernel 25.18–25.77 → 23.47–23.87 µs (A over B 1.069–1.082); n3072 m = 8192 → 8193 (baseline time = speedup × kernel time): baseline 61.77–62.81 → 68.75–70.25 µs, kernel 46.91–49.22 → 46.32–48.59 µs (baseline 1.109–1.127×, kernel 0.984–0.988×); the a8w8 calls at M = 2–64: n3072 4.13–4.50 µs, n7168 4.19–4.67 µs to M = 32 and 6.27–6.34 µs at M = 64; m = 4096: 38.66 GFLOP and 36.18 MB (bf16 out 25.17 MB), 7.85 µs at 4923 TFLOP/s and 6.03 µs at 6 TB/s estimated, measured 25.16–25.74 µs, 30.51–31.20 % of MFMA peak, 1.406–1.438 TB/s; n7168 M = 64 15.73 MB in 6.272–6.343 µs, 2.480–2.508 TB/s; estimates: n7168 M = 256 7.52 GFLOP, 1.53 µs at 4923 TFLOP/s against 2.45 µs to read 14 MiB at 6 TB/s; M = 32, N = 4096, K = 7168 1.879 GFLOP, 0.76 µs at half the F8F6F4 rate against 4.89 µs of reads (schemas aef7bbb04ccbe710, 3045bc7c9725aa4d, d4457ae42fce5327, 06755ba3ee5eb085; #89)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, Gluon microbenchmark, 1 GiB read flush before each launch, median of 9 launches, 3 alternated rounds] 8 readers per chunk, consecutive program ids → ids congruent mod 8: 21.3–21.5 → 11.8–12.0 µs (64 KB chunks) and 156.3–159.0 → 74.4–74.6 µs (512 KB), median ratios 1.78–1.80 and 2.08–2.12, single rounds 1.76–1.82 and 2.07–2.17; the XCC of every program equals its linear id, axis 0 fastest, mod 8, on 10 grids (3-D included), 3 launches each; HIP `L2_cache_size` 4194304 bytes, and re-reading 2 MiB per XCD cost 2.22–2.23× less per byte than 8 MiB per XCD (#89)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] Qwen2-MoE router projection at 256 tokens, split-K 4 → 2, grid 256 → 128 four-wave programs: speedup 1.905–1.922 → 1.435–1.445 (0.750–0.756), call 9.94–10.06 → 13.23–13.35 µs, B/A time 1.320–1.347 in all 12 pairs; fold delay 2 → 0, B/A time per pair: n3072 m = 256 0.996–1.036, run means 1.000–1.008, n7168 M = 2 0.974–1.049, run means 0.990–1.011; compiled, m = 256's `_decode_kernel` with `REDUCTION_DELAY` 2 and 0 both have 24 `s_nop` wait cycles and 1 VALU read of an MFMA result within 8 instructions, in a different instruction order; n3072 m = 256 `pn ^ (pm * 2)` → `pn`: B/A time 0.937–0.988 in all 12 pairs, run means 0.960–0.974, B accurate; n3072 m = 6144 served by `_gemm_strip_kernel` instead of `_gemm_async_kernel`: 39.41–40.20 → 38.66–39.90 µs (0.981–0.992), B accurate (schemas c3889614985b700a, 616e1e1d43c2f2eb, ff2dc820165bd28a, 584a68a298cb2da9; #89)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, `python -m aklr validate --stages audit,bench` (Kineto timer, before #33), one GPU, 4 runs per arm alternated] MoE weight loads at 256 tokens: MiniMax-M2 E64 `.cg` removed: kernel 104.1–104.8 → 107.7–107.9 µs (1.029–1.035), speedup 1.180–1.181 → 1.144–1.145, B/A time 1.015–1.044 in all 8 pairs, every B run's speedup below every A run's, its unchanged 128-token Point 0.994–0.999; DeepSeek-V2 inter_dim 512 `.cg` added on the 128-row gate-up tiles: 226.4–231.2 → 228.0–232.4 µs (1.005–1.007), speedup 1.032–1.039 → 1.027–1.033, B/A time 1.004–1.008 in all 8 pairs, A and B runs separated in one of the 2 runs; n3072 output store `cache_modifier` `.cs` → none (36 and 48.01 MiB of bf16 output): m = 6144 kernel 40.03–40.13 → 40.72–40.91 µs, speedup 1.251–1.252 → 1.228–1.229, B/A time 1.016–1.020 in all 8 pairs; m = 8193 46.89–47.53 → 46.83–47.18 µs, speedup 1.470–1.482 → 1.480–1.484, B/A time 0.992–0.9997 in all 8 pairs, A and B runs not separated (schemas 1d277d76892ee692, 2422a87b3b814299, 584a68a298cb2da9, 3045bc7c9725aa4d; #89)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 for timings (one run, 12 balanced cycles per configuration; at n7168 M = 64 12 AB/BA validate pairs and 6 A/A pairs), N=2 for correctness and layout (21 seeded tests per run)] §5's skeleton at M = 16: 8 ring slots and private slots correct at (N, K) = (3072, 1536), (7168, 2048), (3072, 4096), (4096, 7168), error / max|reference| 0.00172–0.00302, the `_reinterpret` views equal byte for byte; 8 ring slots against 32 private at K = 4096: 6.239 against 7.601 µs (N = 4096), 8.127 against 13.165 µs (N = 7168); `PREFETCH` 1–11 over N = 1024–14336 at K = 4096 and 7168, 110 arms accurate: knee at [8, 8, 8, 5, 3] and [8, 8, 8, 5, 5], 1.84–14.68 MB estimated in flight, below 0.8 MB 1.68–4.49× the minimum; at SLOTS 12, N = 4096, K = 4096 / 7168: P4 / P5 1.434 / 1.651, P7 / P5 1.186 / 1.218 (1.184–1.189 / 1.216–1.221), P8 / P5 0.881 / 0.847. K split over the waves, M = 16 / 32 / 64 × seven (N, K), 112 arms accurate: at M = 32, N = 4096, K = 7168 15.36 µs against 40.63 / 40.44 µs unsplit (wave over cross-workgroup split 0.863); at M = 16 1.277× a deep-prefetch skeleton at N = 2048, K = 7168 and 1.191 / 1.256× at N = 4096, K = 4096 / 7168; at M = 64, N = 4096 unsplit over split 1.554 (K = 7168) and 1.627 (K = 2048); wave over cross-workgroup 1.149 / 1.380 at K = 7168, N = 2048, M = 32 / 64 (1.148–1.151 / 1.377–1.383), 0.99840–1.00178 at M = 32, N = 7168, K = 7168. K-group folds at M = 1 / 2 / 4 / 8 × (N, K) = (3072, 1536), (7168, 2048), (7168, 7168), 36 arms accurate (1.4e-5–1.8e-5): min(FA, FAB) / U 0.309–0.512 at M ≤ 4, gains 1.594 / 1.516 / 2.832 at M = 8 and 2.368 / 1.953 / 3.013 at M = 1, FA alone at M = 1, N = 7168, K = 2048 0.98585 (0.98459–0.98711). n7168 M = 64 with 4 slots and `PREFETCH` 3, accurate on 24 seeds: 6.315 → 6.187 µs, geometric B/A 0.979803 (0.979154–0.980452), pairs 0.9788–0.9822, A/A 1.000674 (0.999134–1.002217) (schema 06755ba3ee5eb085 and Gluon micro-kernels; #89)
- E26 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, two GPU runs of synthetic one- and four-wave rings with relaxed reads and a single buffer; 2048 programs, 200 launches per delay setting, 4 settings per run (one wave: 0, 20, 39, 39 dependent loads; four waves: spin delays 0, 1, 4, 16 µs)] one-wave ring refilling the just-read slot without a barrier: 2.8–5.1 % of programs read refilled data in every launch, zero wrong in 1600 launches with it; four waves, 2 groups ahead in 3 slots: wrong sums in every launch, a pinned variant 48–89 % of outputs wrong, zero wrong with 4 slots or 3 plus a barrier; single buffer without the barrier: wrong in all 200 launches at each delay (38–72 % of outputs), zero wrong in 1600 with it; the committed m = 8192 file without that barrier passed six validate runs (#89)
- E27 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compiled toys and one timed toy] consecutive committed groups of 3, 2 and 1 copies: `vmcnt(0/1/3)` at `wait_group(0/1/2)`; scale loads after or before two copies wait `vmcnt(1)` / `vmcnt(3)`, after or before four copies `vmcnt(0)` / `vmcnt(4)`; a 192-program, one-wave, eight-group toy with scales first 6.078 µs against 6.030 µs after (1.008×) (#89)
- E28 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one run of balanced AB/BA validate pairs (Kineto timer), 4 pairs per Point unless stated, mean B/A time with a paired 95 % t interval; tile placement modeled as `pid % 8`, L2 fill estimated on the CPU] committed tile order removed: n3072 m = 64 1.11993 (1.0965–1.1436), m = 128 1.05936 (1.0566–1.0621), m = 2048 1.00651 (1.0054–1.0076), m = 4096 1.01862 (1.0181–1.0191; 25.981 → 26.465 µs), m = 8192 1.00935 (1.0085–1.0102); n7168 M = 128 1.04045 (1.0391–1.0418), M = 64 0.99305 (0.9856–1.0005), M = 2 (12 pairs) 1.02350 (1.0101–1.0372); n3072 M = 1 and M = 16 (12 pairs each) intervals 0.9895–1.0196 and 0.9953–1.0211; DeepSeek-V2 MoE at 2048 tokens, a whole MoE call, 1.02654 (1.0104–1.0428); n3072 m = 256 `pn ^ (pm * 2)` → `pn` 6.0875 → 5.906 µs, 0.97012 (0.9689–0.9713), CPU fill 21 → 7.5 MiB; m = 4096 `group_m` 4 → 1 CPU fill 30 → 52.5 MiB; all outputs accurate (schemas 4d4cb3c49f122deb, 01110d0a95abc9f6, c56c343a3be1283d, aef7bbb04ccbe710, d4457ae42fce5327, 1bbf98a368b3fcba, 06755ba3ee5eb085, ff2dc820165bd28a, b11f8d1f0bba8fed, f3970decacc97225, 616e1e1d43c2f2eb, c524; #89)
- E29 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one run per padding setting, 12 balanced pairs per arm, 24 accuracy seeds per arm] n7168 M = 64 with four slots fixed, LDS 98304 bytes (no padding) or 131072 (32768 bytes padded), one CTA per CU either way: `PREFETCH` 2 over 1 time 0.7071 (95 % interval 0.7063–0.7078) and 0.7002 (0.6995–0.7009), `PREFETCH` 3 over 1 0.7350 (0.7340–0.7360) and 0.7270 (0.7262–0.7279); all arms accurate (schema 06755ba3ee5eb085; #89)
- S1 aklr/ops/gemm/contract.py:201 `GEMM_A8W8_BLOCKWISE_SCALED`: fp8 `a`, `b` (:237-243), fp32 `a_scale` per 1 × 128 and `b_scale` per 128 × 128 (:244-259), log-uniform scales (:192), bf16 default `out` (:262-268), `BOUNDED` compare (:269) with the FIXME at :186-188; the layout variants `row_major` (`operand_format` `none:none`, `b.is_shuffled` false), `b_shuffled` (`Preshuffle(tile=(16, 16))` `b`, `ColumnMajor()` `a_scale`, `a_scale.is_row_major` false) and `b_shuffled_row_major_a_scale` (`a_scale.is_row_major` true) at :270-295; a variant other than the first joins the schema name at aklr/core/contract/op.py:626-627; aiter's `gemm_a8w8_blockscale` for `none:none`, `gemm_a8w8_blockscale_bpreshuffle` for a shuffled `b` and `gemm_a16w16` for `gemm@1.0` at aklr/ops/gemm/baseline.py:57, :58-60, :17-21; `runs_agree` at aklr/core/contract/semantics/_compare.py:142: `bounded` passes each run on its own, `bitwise` wants every output byte-identical to the first; `CompareSpec` defaults to `bitwise` at aklr/core/contract/spec/_args.py:204
- S2 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:1001 the m ≥ 16 host branch: `depth = 5 if m in (16, 32) else 4` (:1003), `waves = 3 if m == 16 else 6` (:1004), read-ahead 2 and refill-before at m = 16 (:1005-1006), `BANK_XOR=False` (:1024), `waves_per_eu=waves` (:1033); its wait at :441-448; `run` at f3970decacc97225.json#main.py:8-20; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:424 the partial from `gl.zeros`; `acc = gl.fma(partial * sx, sw, acc)` at :483; `LC` at :300-305; `DotOperandLayout(OPERAND, LAYOUT, K_WIDTH)` at :87-89, `K_WIDTH=16 if m == 16 else 32` at :1025; `k_width` 32 at m = 64 in 4d4cb3c49f122deb.json#kernel.py:41-43
- S3 triton/experimental/gluon/language/_core.py:382 in the Triton 3.7.0 wheel (3.7.0+gitb4e20bbe.rocm7.14.0a20260608): `_reinterpret(self, dtype, shape, layout)`; its amd/cdna4/async_copy.py:105 accepts only `BlockedLayout` or `SliceLayout` offsets; its amd/cdna4/__init__.py has no `compute_efficient_padded_shared_layout`; the calls at aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:84, 97933c3a190ede28.json#kernel.py:83, 4d4cb3c49f122deb.json#kernel.py:38; m = 128's `DistributedLinearLayout` copy layout at 01110d0a95abc9f6.json#kernel.py:195-196; m ≥ 256 at 616e1e1d43c2f2eb.json#kernel.py:451
- S4 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json#kernel.py:326 the gate-up K loop over K = 3072, `mfma_scaled` and the fold at :44-64, 16 × 16-preshuffled offsets at :68; aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:199 32 groups of K = 4096, fp8 widened to fp16 for `mfma` (:65-74), one scale fold per group (:213); K split over the waves of a workgroup in aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:593 `_w13_local_split`: `AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[SK, 1, 1])` (:618-623), plain `gl.amd.cdna4.mfma` per 128-K step (:706-710), the scale fold (:711), `gl.sum(acc, 0)` (:713)
- S5 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/616e1e1d43c2f2eb.json#kernel.py:451 `compute_efficient_padded_shared_layout` with a `k_width = 16` operand layout (:451-456), read with `dot_width = 32` (:405); `_copy_layout` at :300-310; 3 A slots refilled 1 ahead (:466, :527); `REDUCTION_DELAY = 0 if BLOCK_M == 32 else 2` (:493), the delayed fold at :573-578, drained after the loop at :580-581; the helper returns `None` when `k_width` is not 4, 8 or 16 at triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:139-143
- S6 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:68 both scales `None`: `uint8` `0x7F` (:69), a splat in the scale layout (:99), `dot_scaled` (:105); the version-4 unscaled fp8 entries `mfma_f32_16x16x32_fp8_fp8` and 32×32×16 at third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp:225, :205, 16×16×128 and 32×32×64 only as `mfma_scale_*_f8f6f4` (:264-268)
- S7 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1bbf98a368b3fcba.json#kernel.py:83 two chained `mfma_scaled` per 128-K group (:83-92), one fold (:94); inline `s_waitcnt lgkmcnt(0)`, then `wait_group` and `gl.barrier()` (:66-73), the same in fee7e092eab87fc5.json#kernel.py:79-86; M = 128's swizzled operand layouts at :131-132, M = 256's hand-written padded ones at fee7e092eab87fc5.json#kernel.py:349-360
- S8 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/b11f8d1f0bba8fed.json#kernel.py:729 folded offsets; `PG = 4`, `BN = 4` at :686-688; XCD order at :721; carrier at :754-759; scales loaded after each panel's wait at :762-764; diagonal at :782-787; group sum at :793; m = 8: aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/a5c62e519ce3fe45.json#kernel.py:582 two-group offsets (`ar // 2`, `ar % 2`, :582-587); 6 products in 3 commit groups at :590-606; activation scales loaded after all copies at :612-617; diagonal at :645-646
- S9 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` runs no loop scheduling or automatic pipelining pass; its `add_warp_pipeline` (:339) acts only on ops marked `triton.warp_pipeline.border` and skips other loops (third_party/amd/lib/TritonAMDGPUTransforms/WarpPipeliner.cpp:39-40, :413); `make_ttgir` runs `add_schedule_loops(num_stages)` and `add_pipeline` at :288-289; triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47 `async_wait` "does not provide any syncronisation in the CTA" (:54); lib/Analysis/Membar.cpp:289-298 a barrier after a memory wait with no later barrier or wait before the next access; third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h:9-28 the AMD filter drops the barrier between an async copy and a local load synced by `AsyncWait` (what `load_shared_relaxed` marks) of the same buffer, and notes it is wrong for a single buffer refilled after its read; code at third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11
- S10 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:132 `_copy_private_group`: two 64-byte halves of A and of B by `buffer_load_to_shared`, one `commit_group` (:143); copy layout `BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])` at :277; prologue of `PREFETCH` groups at :364; loop at :407; refill at :427-440; wait at :441-448; read-ahead at :449-464; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:323 `BANK_XOR` copy rows and contiguity hints (:323-329); read basis `[4 if BANK_XOR else 0, 32]` at :73; `(2, 16, 64)` buffers at :296-297; the `(16, 128)` view at :84 and `permute` for B at :86
- S11 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/4d4cb3c49f122deb.json#kernel.py:175 `LDS_SLOTS` 9; stripes inside a 256-program super-block at :185-193; waits at :309-313; `gl.barrier()` before the first slot reuse at :341-343; both scale tables copied into LDS as one commit group at :92-147 (redirected columns at :130, :142), issued at :236 after `DATA_FIRST` groups; A and B committed apart from group 9 on at :68
- S12 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/01110d0a95abc9f6.json#kernel.py:253 `wait_group`, then `gl.barrier()` (:266), then `load_shared_relaxed` (:270); 4 + 4 slots refilled 2 ahead at :204-205, :249-252; scale window 4 groups ahead at :272-277; hand-written padded layouts at :148-186; copy layout from `offset_bases` at :20-30; rounded exchange, `.wt` store and `_keep_alive` at :313-363
- S13 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:161 `SLOTS`, `PREFETCH`, `WAITS` (:161-166); `BK = 256`, `STAGES = 8` at :159-160 and `gl.amd.slice` per group at :51-56; plain `.load` reads at :248-249; `SPLIT_COMMIT` refills at :116-131, :251-257; `_swizzled_k` at :14-15; copy into a linear `_reinterpret` at :108-112; read layout at :175; store layout at :91-104
- S14 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared`, which "does still complete in order with ttgl.loads/stores or buffer_loads/stores on CDNA4" (:78-80) and fails to lower unless 128 or 32 bits per lane, writes to `dest` coalesced and a swizzle only within a warp (:89-94); `wait_group` at :137; `load_shared_relaxed` at :151
- S15 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:214 `_private_register_scales`: 3 `buffer_load`s of activation scales (:229-235), 12 scalar weight scales (:246-247); called at :344, before the prologue copies at :364; m = 32's SGPR pointer and `gl.pointer_type(gl.float32, 4)` loads at 97933c3a190ede28.json#kernel.py:126-150 and its `_schedule` (`@gluon.constexpr_function`) at :32, read in the kernel at :230 and by the host at :416 and :422
- S16 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/d4457ae42fce5327.json#kernel.py:666 single-buffered group loop: scale gathers (:667-671), `wait_group(0)` (:677), reads, `gl.barrier()` and refill before the last panel (:684-688); middle panel scale (:672); `GROUP_M` bands (:586-591), `GROUP_M` 4 from `_launch_policy` (:28-33); m = 1024's `sw1 = gl.where(pn % 4 == 1, sw2, sw0)` at e5af2435eab17788.json#kernel.py:451; m = 4096's bands and N interleave at aef7bbb04ccbe710.json#kernel.py:630-639, host `group_m = 2 if m <= 2048 else 4` (:1094)
- S17 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/3045bc7c9725aa4d.json#kernel.py:563 ragged split: 16-row tail (:564-588), bulk at `M - M % BM` (:590-613); grid at :657; the tail's 32-row A layout (`AM = 32`, :89-94), row mask (:115) and `gl.amd.slice` to 16 rows (:164); `.ca` on the async copies at :147, :155
- S18 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/f3970decacc97225.json#kernel.py:307 stripes (:307-308); `gl.static_range(12)` at :335, :407, :490; `12 - PREFETCH` at :427, :469, :524; `11 - group` at :446, :492; `3072` at :545-546; `assert n == 3072 and k == 1536` at :1053; the m = 32 stripes at 97933c3a190ede28.json#kernel.py:252-253; b11f8d1f0bba8fed.json#kernel.py:760-762 3 panels and `wait_group(2 - panel)`; n7168 ae4fa8a440b98f09.json#kernel.py:185 `STAGES = 8`, :128 and :149 `gl.arange(0, 16, ...)`, :222 `_load_scale_range(..., 0, 16, ...)`; 06755ba3ee5eb085.json#kernel.py:171-172 the 224-tile permutation, :298 `assert m in (32, 64) and n == 7168 and k == 2048`
- S19 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/362ce3bb6200382e.json#kernel.py:136 M = 1: `A_PACK = 4`, `B_PACK = 2` (:136-137), the `example` carrier (:179-181), 4 B slots primed at :191-193 and refilled at :211-216; M = 2 at ff2dc820165bd28a.json#kernel.py:408 `PRIME = 4`, A rows fold 4 groups (:434-435), `_packed_diagonal` (:471), refill (:475), 56 adjacent tiles per XCD (:415); M = 4 at 0b1065ae9c0652b7.json#kernel.py:288 `_packed_group` (:288-299), 2 of 4 B panels primed (:368), refill (:413-414), the XCD order (:318); M = 8: aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1e021d44010accb1.json#kernel.py:135 `_diagonal` with `gl.amd.slice`, `reshape`, `permute` and an `assert_trivial` convert; `PG = 2` at :254; `convert_layout(..., assert_trivial=True)` raises a `TypeError` at compile time when the conversion is not trivial at triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245
- S20 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:87 `_reduce_kernel` (:86-93); split partial store at :81-83; split launch and reduce at :133-156; `run` at :161, `torch.empty` scratch at :162-167; `_xcd_remap` at :26-35, applied with `pid_k = pid // N_TILES` at :55-59; `offs_m = tl.arange(0, BLOCK_M)` at :63; the m ≤ 16 launch with M at run time at :100-125
- S21 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/ff2dc820165bd28a.json#kernel.py:409 M = 2: `DELAY = 2`, the deferred fold at :476-482; M = 4 `DELAY=3` at 0b1065ae9c0652b7.json#kernel.py:691; M = 256 accumulator pin by an empty tied `inline_asm_elementwise` at fee7e092eab87fc5.json#kernel.py:125-131 (served with `PAIRED=False`, :413); n3072 m = 8 two accumulators at a5c62e519ce3fe45.json#kernel.py:647-650; m = 32 `K_ORDER` and `odd_acc` at 97933c3a190ede28.json#kernel.py:236-240, 387-388, 404-405
- S22 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:103 `schedule_hint`: "Experimental; right now no effect" (:103-105); n6144 passes it at aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:168, has `offs_m = tl.arange(0, BLOCK_M)` at :46 and :94, launches m ≤ 16 with M at run time, `EVEN_M` only at m = 16 and `BLOCK_K` 256 but at m = 1 (:121-142), `run` at :199; Triton 3.7.0 mapped `memory-bound-attention` to `amdgpu-sched-strategy` `iterative-ilp` at triton/backends/amd/compiler.py:393-394 of the Triton 3.7.0 wheel (3.7.0+gitb4e20bbe.rocm7.14.0a20260608)
- S23 aklr/core/utils/testing.py:164 `bench_cudagraph` (since #33): each call a profiler scope recorded by AKLR's rocprofiler-sdk collector (:41-60, :127-136), its time the sum of the scope's kernel durations (:139-146), the flush outside the scope (:206-210), the median over calls (:108-109); the runner attaches the collector without forced queue timing (:25-31; aklr/core/validation/stages/_bench/_execution/_runner.py:525-528). Before #33 `_kineto_replay_times_ms` summed the Kineto kernel durations per replay except the flush kernel (AMD-AGI/AKLR@43e112e:aklr/core/utils/testing.py:111, :126-137) and the runner set `GPU_FORCE_QUEUE_PROFILING="1"` (AMD-AGI/AKLR@43e112e:aklr/core/validation/stages/_bench/_execution/_runner.py:515)
- S24 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/c56c343a3be1283d.json#kernel.py:38 `ORDER_M` bands: consecutive programs walk `ORDER_M` M tiles of one N tile (:38-49); `ORDER_M` 2 at m = 2048 (:755); the same bands with `ORDER_M` 4 at m = 6144 in 584a68a298cb2da9.json#kernel.py:60-74, whose host serves m = 6144 with `_gemm_async_kernel` and m = 2048, 4096, 8192 and the bulk of m = 6145 with `_gemm_strip_kernel` (:1115-1166)
- S25 triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:144 CDNA4 shared memory `160 * 1024` (:144-145); python/triton/compiler/compiler.py:466 raises `OutOfResources` at the first launch when `metadata.shared` exceeds the device's (:466-468)
- S26 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4c2bed7e3e8c1e94.json#kernel.py:102 the `[STAGES, BM, BK]` ring; the rolled `range` at :121; refill into `.index((slot + STAGES - 1) % STAGES)`, then `wait_group`, `gl.barrier()` and the relaxed read at :28-44
- S27 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/01110d0a95abc9f6.json#kernel.py:124 m = 128: `pm`, `pn` from `program_id(1)`, `program_id(0)`, the XOR only `if BLOCK_M == 64` (:124-128), grid (48, 4) at :391; m = 256: `pn = pn ^ (pm * 2)` at 616e1e1d43c2f2eb.json#kernel.py:391, grid at :707; n7168 M = 128 at 1bbf98a368b3fcba.json#kernel.py:120-123, grid at :489; M = 256 at fee7e092eab87fc5.json#kernel.py:338-343, grid at :448
- S28 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:1087 `_compile_skinny`, taken for m ≤ 32 (:1074, :1389): each of four waves takes a contiguous quarter of K, `K // tile_k // nw` = 16 groups (:1067-1068, :1093), the partials summed through LDS in wave order (:1301-1312), grid `((M + tile_m - 1) // tile_m, N // tile_n)` with 16-column tiles (:1357); the K 4096 file splits K over 8 waves at m = 1, 2 and 4 at m = 4–32, "256 WGs fill the 256 CUs", at aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json#kernel.py:44-58
