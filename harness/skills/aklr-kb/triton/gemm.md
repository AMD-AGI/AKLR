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
its two plain-Triton bf16 decode GEMMs are cut by M and how their K loops feed the matrix core.
The `../gemm/` cards cover the related mechanisms. Stored CI ratios are version-bound lower bounds;
measured examples below apply only to their stated shapes and ruler.
## Route here when
- You write or tune a Triton or Gluon kernel for `gemm_a8w8_blockwise_scaled@1.0` (fp8 e4m3 operands, fp32 block
  scales, bf16 out) or `gemm@1.0` (bf16) and want the committed kernel closest to your Point (§3) or a K loop (§5).
- You need a tile and grid for an M or a new (N, K), tiny-M K-group folding, an async direct-to-LDS K loop, an LDS
  layout for fp8 operands, a tile order, ragged M, the epilogue, or split-K under a `bitwise` contract.
- You need a tuning candidate with a stated applicability limit, rather than a universal best setting.
Elsewhere: the bf16 decode GEMM, `gemm/decode_gemm.md`; Gluon API facts, `gluon/memory.md`, `gluon/layouts.md`,
`gluon/mfma.md`; A/B method and launch options, `tuning.md`; reading the compiled kernel, `debugging.md` §1–§3;
rounding, `numerics.md`; files, entry and validation, `authoring.md`; loop designs and throughput arithmetic,
`../gemm/mainloop.md` §1, §9; MoE expert GEMMs, `moe.md`; every family on main, `kernels/kernel_library.md`.
## 1. The kernels and their Points
| Schema under `aklr/ops/gemm/impls/triton/` | Contract (compare) | Points | Language | CI speedup |
|---|---|---|---|---|
| `gemm_a8w8_blockwise_scaled_n3072_k1536/` | `gemm_a8w8_blockwise_scaled@1.0` (`bounded`) | m = 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 8193; one file each | Gluon | 1.191–1.486  |
| `gemm_a8w8_blockwise_scaled_n7168_k2048/` | same | m = 1, 2, 4, 8, 16, 32, 64, 128, 256; one file each | Gluon | 1.114–1.840  |
| `gemm_n2624_k6144/b88c8d5f28d86777.json` | `gemm@1.0` (`bitwise`) | m = 2, 4, 8, 64; one file | plain `tl.*` | 1.048–1.174  |
| `gemm_n6144_k2048/f806ad2b086d757d.json` | `gemm@1.0` (`bitwise`) | m = 2, 4, 8, 16, 32, 64; one file | plain `tl.*` | 1.064–1.292  |
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
  offsets in a `DistributedLinearLayout` (m = 128), `compute_efficient_padded_shared_layout`, absent (m ≥ 256).
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
4 runs. M = 128/256 and grid sizing are not validated here; classify a call site with `../gemm/small_m.md` §2, `../gemm/grid_pricing.md` §1.
Each logical 128-K scale-group contribution is scaled in fp32 before it joins the running sum. Tiny-M kernels
extract that contribution from packed products (§4), and 32 × 32 kernels may chain two 64-K instructions (§2).
For n3072 M = 16, the group starts from zero and is folded as follows (S2):
```python
partial = gl.amd.cdna4.mfma_scaled(
    a, None, "e4m3", b, None, "e4m3", gl.zeros((16, 16), gl.float32, LC)
)
...
acc = gl.fma(partial * sx, sw, acc)
```
`LC` is `AMDMFMALayout(4, [16, 16, 128], transposed=True, warps_per_cta=[1, 1])`; `a` and `b` carry
`DotOperandLayout(0 or 1, LC, k_width)`, `k_width` 16 at m = 16 and 32 at m = 32 and 64: one instruction per 128-K
group either way (S2; `gluon/mfma.md` §5). §5's skeleton writes all of it out.
- **Use `mfma_scaled` with `None` scales for plain fp8.** Gluon fills both scales with the M0 unit `0x7F` and
  emits a scaled dot (S6): the K = 128 (16 × 16) or K = 64 (32 × 32) F8F6F4 instruction, at plain fp8 `mfma`'s error
  (bit-identical on one 16 × 16, K = 128 tile) and 2.0× its rate (1.995–2.012× over 16×16×32 and 32×32×16)
. Every a8w8 kernel compiles to the 16x16x128 or 32x32x64 `v_mfma_scale_f32_*_f8f6f4`.
- **Keep the block scales out of the instruction.** They are fp32 values drawn log-uniform (S1), not powers of two:
  rounded to the M0 scale operands they gave a 17 % error on a test tile, an fp32 fold per 128-K group 7.2e-6
. The fold's spelling differs per file (`fma(partial * sx, sw, acc)` in 8, `fma(partial, sx * sw,
  acc)` in 2, a first group seeding `acc` in 9); in a test kernel each stays within the bound (`numerics.md`).
- **32 × 32 tiles: chain two instructions per group.** n7168 M = 128 issues two `mfma_scaled` over the two 64-K halves
  on one accumulator, then folds once (S7): a test kernel took 148 fold VALU ops, 277 folding each half.
- **Tune 16 × 16 versus 32 × 32 in the kernel.** Register-resident chained `mfma_scaled` e4m3 ran 4899–4934 TFLOP/s
  at [16, 16, 128] and 4907–4949 at [32, 32, 64] (ratio 0.997–1.000); no GEMM shape A/B.
## 3. Tile and grid by M
Start from the file of the nearest Point: its design is the regime's, its shape constants are not (last three bullets).
| M | n3072_k1536: file, programs × waves, output tile | CI  | n7168_k2048: file, programs × waves, output tile | CI  | Design |
|---|---|---|---|---|---|
| 1 | `bd017b0794d5db31`, 768 × 1, 1 × 4 | 1.429 | `362ce3bb6200382e`, 448 × 1, 1 × 16 | 1.664 | n3072: K groups folded into A rows and B columns, every K panel issued before the first wait; n7168: all of A issued, B primed with 8 of 16 K groups and refilled in the loop (§4) |
| 2 | `c194fbea23151544`, 768 × 1, 2 × 4 | 1.397 | `ff2dc820165bd28a`, 448 × 1, 2 × 16 | 1.811 | n3072: same; n7168: A-only group folding, all A issued, 8 B groups primed and refilled (§4) |
| 4 | `d6228c711bd687b6`, 768 × 1, 4 × 4 | 1.307 | `0b1065ae9c0652b7`, 448 × 1, 4 × 16 | 1.679 | n3072: same; n7168: A-only group folding, all A issued, 8 B groups primed and refilled (§4) |
| 8 | `0e15a0289735acb0`, 192 × 2, 8 × 16 | 1.435 | `1e021d44010accb1`, 448 × 1, 8 × 16 | 1.840 | two K groups per MFMA (§4) |
| 16 | `083b493a1861424c`, 192 × 1, 16 × 16 | 1.406 | `1db61e7b0dc84d90`, 224 × 2, 16 × 32 | 1.725 | async K groups in flight: private LDS slots (n3072), a 4–6-slot ring (n7168) (§5) |
| 32 | `fcf21622f6c91bbb`, 384 × 1, 16 × 16 | 1.418 | `ae4fa8a440b98f09`, 224 × 4, 32 × 32 | 1.759 | same |
| 64 | `54f6affa5c286a70`, 768 × 1, 16 × 16 | 1.412 | `06755ba3ee5eb085`, 224 × 4, 64 × 32 | 1.488 | async K groups in a 9-slot (n3072) or 3-slot (n7168) ring |
| 128 | `3480e97a5fcda6f6`, 192 × 4, 32 × 64 | 1.477 | `1bbf98a368b3fcba`, 224 × 4, 64 × 64 | 1.114 | 4-wave tiles, LDS ring, 1–2 groups ahead (§5) |
| 256 | `199c45f4249e7f55`, 192 × 4, 64 × 64 | 1.199 | `fee7e092eab87fc5`, 448 × 4, 64 × 64 | 1.138 | same, 32 × 32 × 64 MFMA |
| 1024–8192 | five files, 512–1152 × 4, 64 × 96 to 128 × 192 (§10) | 1.191–1.356 | — | — | single-buffered LDS, column panels, M bands (§7) |
| 8193 | `831926aed9523c0f`, 1560 × 4, 128 × 128 and a 16-row tail | 1.486 | — | — | bulk compiled at M = 8192 (§7) |
- **At M ≤ 64: one wave per program (two or four in some files) and 192–768 programs,** each owning a 4- to
  32-column slice of the weight over all of K (traced program counts match these tiles); `a` is at most 128 KiB.
  n7168 halves the tile to 16 columns at M ≤ 8, 448 programs against 224 at M = 16–64.
- **At M = 128–256: four waves per program** on 32 × 64 or 64 × 64 tiles with an LDS ring (§5); from m = 1024,
  64–128-row by 96–192-column tiles, single-buffered (§7).
- **The margin is smallest in the middle.** n3072 is lowest at m = 256 (1.199) and m = 1024 (1.191), n7168 at
  M = 128–256 (1.114–1.138) against 1.66–1.84 at M ≤ 32. n3072 m = 128 and 256 differ
  in MFMA shape, A prefetch distance, scale timing and tile order at once: their gap is not assigned to any of them.
- **Do not rank Points by small gaps.** The stored values of the M ≤ 64 Points with more than one CI run moved
  4–12 % between the runs that rewrote them : compare candidates in one session, alternating runs (§11).
- **A new N or K: turn every shape constant into a constexpr first** (S18). n3072 m = 16 and 32 hard-code N
  (`192 // STRIPES`, `3072` in the store offsets) and K / 128 = 12 (`gl.static_range(12)`, `11 - group`,
  `12 - PREFETCH`), m = 1–4 3 K panels (`wait_group(2 - panel)`), n7168 `STAGES = 8` and 16 scale columns, and n7168
  M = 64's `tile * 8 - ((tile * 293) >> 13) * 223` permutes 224 tiles only (on 128 it reaches 219). With only the
  host asserts relaxed, doubled K broke n3072 m = 1, 2, 4, 16, 32 and n7168 M = 64, doubled N all but m = 1–4 (MI355X).
- **Budget LDS against K.** A private 4 KiB slot per K group (§5's skeleton, the scheme of m = 16 and 32) costs
  K / 128 × 4 KiB: 48 KiB at K = 1536, 224 KiB at K = 7168; past K = 5120 that is over the 160 KiB a CDNA4 workgroup
  may allocate (S25). Triton 3.8.0 compiles the skeleton with 56 slots for gfx950 without an error (229376 bytes, 20
  VGPRs spilled at `waves_per_eu` 3); only the launch raises `OutOfResources` (S25), so a compile-only check passes it.
  A fixed-slot ring is a tested option for the measured long-K shapes, not a cost-equivalence guarantee: with 8 slots
  the skeleton keeps 32 KiB, its output in bound at the tested K = 1024–7168; n7168 M = 16/32/64's
  ring takes 64/64/72 KiB. At M16/K4096, private slots used 128 KiB versus the ring's 32 KiB and were slower at the tested
  N4096/7168 Points. A ring bounds LDS but not code growth: unrolled loops grow with K. A rolled allocation
  indexed with `.index(t % SLOTS)` is a source precedent in the Qwen2-MoE router (S26), not a measured universal speedup.
- **Long K at small M: compare split placement and the actual K loops.** The n7168 ring's 32-column tiles (M = 16–64) make
  N / 32 programs, 128 at N = 4096: one per two CUs, a grid that read 0.75 (0.750–0.756) against one per CU on the Qwen2-MoE
  router projection (`tuning.md` §3). FlyDSL splits K over eight waves at N = 4096, K = 4096, m = 1–2, and four waves at m = 4–32; at K =
  8192 it uses four waves for m ≤ 32 (S28, `../cases/a8w8_n4096_k_sweep.md`). In Gluon: one K slice per wave in a rank-3 `AMDMFMALayout(4, [16, 16,
  32], True, warps_per_cta=[SK, 1, 1])` with plain fp8 `mfma` (`mfma_scaled` aborts on rank 3, `gluon/mfma.md` §10; at
  half §2's rate, an estimated 0.8 µs of MFMA against 4.9 µs of weight reads at M = 32), an fp32 fold per group, then
  `gl.sum(acc, 0)` (MiniMax-M2 FP8's `_w13_local_split`, S4; `gluon/layouts.md` §6.7). In register-operand micro-kernels
  at M = 32, N = 4096, K = 7168, 256 such 16-column workgroups of 4 waves ran 2.2× faster than 128 32-column or 256
  16-column programs without the split, and a split across workgroups with a reduce kernel took 1.06–1.08× their time
  (3 runs); with `gl.load` operands and an `sx * sw` fold the wave split compiles to 74 VGPRs, no spill, 4 KiB of LDS. Rotating-buffer comparisons confirmed the M32/N4096/K7168 benefit, but the same wave-split design
  lost to deep prefetch at some M16 Points. Compare split placement, operand path and K-loop design together;
  long K does not establish which lever to try first.
## 4. Tiny M: fold K groups into the MFMA rows
With M < 16, a 16 × 16 × 128 instruction carrying M token rows wastes the other rows. In the same-PG A-and-B folding scheme used by n3072 M ≤ 8 and n7168 M = 8, the kernels fill them
with (token, K group) pairs (the other n7168 variants are described below): row `i` of the A tile holds token `i // PG`, K group `i % PG` of the current K panel;
row `j` of the B tile holds output column `j // PG`, group `j % PG`. The instruction computes every (A group,
B group) product; the kernel keeps the diagonal where the groups match, scales it, and sums the `PG` groups at the
end. Three bounds pick `PG`: M × PG ≤ 16 A rows; 16 / PG output columns per 16-row B instruction (4 at PG = 4, 8 per
wave at m = 8's PG = 2); PG divides K / 128. `group` gets the split diagonal's layout from a zero tile put through
the same reshape and splits, the carrier. n3072 m = 1 (`PG = 4`, 4 output columns, 3 panels of 4 groups, S8):
```python
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
- **Cost.** In the same-PG A-and-B scheme above, the instruction computes `PG` times the useful group products, but the MFMA count is tiny (3 per program at
  m = 1) and the call streams its weight. n3072 issues every K panel before the first wait. In the served n7168 M1/2/4 paths, 8 of B's   16 K groups are primed; M8 instead primes 3 two-group panels (6 groups). The remaining panels are refilled in the loop (S19).
- **Use folding as a tiny-M candidate, not a universal win.** Register-operand comparisons favored group folding
  strongly at several M1/2/4 shapes, but gains were not monotonic in M and A-only folding was only slightly
  faster at M1/N7168/K2048. The committed kernel's baseline speedup does not isolate folding.
  [gemm/decode_gemm.md](gemm/decode_gemm.md) §7 compares a different, unfolded bf16 design.
## 5. The K loop: async direct-to-LDS K groups with PREFETCH in flight
Gluon has no software pipeliner: `gluon_to_ttgir` runs no loop-scheduling or pipelining pass but `add_warp_pipeline`,
which left loops without `gl.amd.warp_pipeline_stage` blocks unpipelined, and `num_stages` changed only plain Triton's
`tl.dot` loop (S9; compiled). The inspected served a8w8 K loops are written by hand; their main K loops up to m = 256 use `gl.static_range` so
that a tuple can hold the LDS buffers (`gluon/memory.md` §12). The skeleton is the m = 16 design reduced to its
parts: one wave per 16 × 16 tile, each 128-K group one commit group of four 1 KiB copies (16 rows × 64 bytes of A and
of B, S10), `PREFETCH` groups ahead in a ring of `SLOTS` buffers, one fold per group (compiles with Triton 3.8.0 for
gfx950). Its ring and private-slot forms met the tested block-scale contract at M16 on
N3072/K1536, N7168/K2048, N3072/K4096 and N4096/K7168 where LDS capacity allowed the form.
This is an authoring starting point, not a guarantee for arbitrary inputs or shapes:
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
For N3072/K1536/PREFETCH5, one warp and `waves_per_eu=3`, the ring8 and private12 versions
compile without spills, using 32/48 KiB LDS and 100 total VGPR/AGPR registers. They emit the same number
of MFMAs and direct-to-LDS copies, but their LDS addressing differs; this does not prove equal runtime.
The served m = 16 fileThe served m = 16 file sets `READ_AHEAD`
2, `REFILL_BEFORE` and `EARLY_MFMA` (S2), so its wait is `min(PREFETCH - READ_AHEAD - (0 if REFILL_BEFORE else 1), …)`.
- **Depth is per loop and Point.** At served n3072 M16, `PREFETCH` 3 was about 12–15% slower than 5;
  4 and 7 had much closer timings, without a general equivalence guarantee. The served read-ahead changes
  steady waits (`vmcnt(4/12/20)` for depth3/5/7), so do not transplant the skeleton's wait formula.
  In the long-K skeleton, depth7 could be slower than 5 while depth8 was faster. The best measured depth
  changed with N/K; neither a fixed prefetch depth nor a fixed grid-wide byte threshold is established.
  Depth also changes scheduling, register lifetimes and spills: bytes in flight is a hypothesis to test.
- **The read after the wait is synchronized for you; the refill after the read is not.** Membar puts a barrier after
  each `wait_group`, but its AMD filter drops the one between an async copy and a `load_shared_relaxed` read of one
  buffer, so a refill's write-after-read is yours (S9; `gluon/memory.md` §11, §12). By buffer scheme:
  - private slots (m = 16, 32): nothing is refilled, nothing to guard;
  - one-wave ring (m = 64, 9 slots, `PREFETCH` 4): barrier before a reused slot's refill; the file has one at
    first reuse (S11), no `s_barrier`, and the same read-before-refill order without it (compiled). This is not
    generally optional: a one-wave relaxed-read toy reordered refill before reads and returned wrong data
    without the barrier. The ordering protection matters even when no `s_barrier` is emitted.
  - four-wave ring with relaxed reads (m = 128, 256): refill, wait, `gl.barrier`, read (S12).
    `SLOTS >= AHEAD + 2` (A: 4/2 at m128, 3/1 at m256; B: 4/2) puts the prior step's barrier between read/refill
    (compiled). AHEAD+1 can still allow other waves' reads to race a refill: use enough spare slots or a barrier
    before refill. Toy counterexamples establish the hazard, not that these committed kernels fail.
  - n7168 M64: plain shared `.load`, 3 slots/PREFETCH2; compiler Membar guards refill (S13).
    M32 instead uses 4 slots/PREFETCH3, an explicit barrier after wait, and relaxed reads: do not borrow
    M64's plain-load protection. M16 has 4 A slots and 4 or 6 B slots depending on strides.
    M128/M256 place inline `lgkmcnt(0)` before wait+barrier (S7). In matched offline controls the compiler
    retained an LGKM drain without the inline wait, but scheduling/registers changed; redundancy of one
    wait is not proof of equal runtime or race-freedom.
  - single buffer (m ≥ 1024): barrier between last read and refill, unguarded by relaxed reads (compiled;
    §7; `../gemm/correctness.md` §4). Cross-wave refill without a last-reader barrier can corrupt data even if repeated validation
    of one committed file happens to pass; absence of an observed failure is not race-freedom.
- **Count waits in committed groups** (`gluon/memory.md` §8): the recorded toy with consecutive committed groups of 3, 2 and 1 copies gave `vmcnt(0/1/3)` at   `wait_group(0/1/2)`; n7168 M = 64's `WAITS = (1, 2, 2, 2, 2, 2, 2, 0)` survives as eight waits in the matched
  Triton 3.8.0/gfx950 IR. Its original M64 launch emits 48 direct-to-LDS instructions and eight barriers; startup commits B+A together, refills commit A and B apart. Machine waits can be strengthened or subsumed, so raw `vmcnt` instruction counts need not map one-to-one to the tuple
  (the n3072 separate-commit source reference remains S11). An uncommitted copy is not waited for, even by `wait_group(0)`.
- **Scale-before-copy is a candidate, not a proven gain.** Scale loads issued after DMA can strengthen waits;
  inspect the emitted scalar/vector load path and `vmcnt` before assigning the dependency. A timed toy did
  not favor scales-first. The source loads scales early at M16/M32, copies them to LDS at M64, uses a register
  window at M128 and gathers from LDS at large M (S8, S11, S12, S15, S16).
- **Delay folds or alternate accumulators only after measurement.** Sources defer folds through tuple queues,
  alternate accumulators by K parity or pin an accumulator with tied asm (S5, S21). Small measured differences
  did not resolve a general benefit; a delay can change scheduling without removing MFMA wait cycles.
- **`waves_per_eu` is a budget request, not occupancy proof.** Removing it at the floor-bound M16 Point
  barely changed recorded time, but changed registers. Test settings at each Point, including spill-free
  controls ([tuning.md](tuning.md) §8).
## 6. LDS layout: swizzle and padding
- **m = 16–64 (n3072): plain buffers read through a linear view.** Each K group's 16 rows × 128 bytes land as two
  (16, 64) halves in `SwizzledSharedLayout(1, 1, 1, [1, 0])` buffers, read as one 16 × 128 operand through a
  `SharedLinearLayout` `_reinterpret` view (B also `permute`d, S10). A gfx950 byte probe checked every A/B
  coordinate with three jointly unique patterns: zero mismatched bytes in both views . This supports this layout only. The copy stays lane-linear, the only LDS DMA
  Triton 3.8.0 lowers on gfx950: 16 or 4 bytes per lane, consecutive lanes on consecutive LDS bytes (S14;
  `../hardware/gfx950/memory_and_sync.md` §3). A 16 × 64-byte copy layout into one (16, 128) row-major buffer breaks
  it: `LLVM Translation failed for operation: builtin.unrealized_conversion_cast` (compiled, Triton 3.8.0, gfx950).
- **`BANK_XOR`: a swizzle written on the global side, measured at m = 16.** The m = 16 file carries an unused variant
  (`BANK_XOR=False` at launch, S2): `copy_rows = r[:, None] ^ ((k[None, :] & 32) >> 3)` sends the lanes that write
  bytes 32–63 of row `r` to global row `r ^ 4`, the read view's basis `[4, 32]` undoes it, and both offsets then pass
  through `gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])` (S10). The hints are part of the recipe: the
  XOR hides the 16-byte runs from axis analysis, and without them the copy fails with the same error, while the tested n7168 XOR-copy probe   `((kk // 16) ^ (rows % 16)) * 16 + kk % 16` kept the runs visible without hints (compiled; `gluon/memory.md` §5).
  At M16 the swizzled variant had similar floor-bound timings, not established speed equivalence.
  An LDS-bound loop can behave differently. Inspect ISA and time a conflict-sensitive workload; `rocprofv3`
  is unusable on the tested image and `gl.bank_conflicts` aborts the AMD compile ([debugging.md](debugging.md) §7).
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
- **The large-M files use no ring:** 512–1560 four-wave programs, one LDS buffer set and source `wait_group(0)` at every wait   (the refill before the last panel's MFMA is read from the source, S16; inspect the exact large-M launch before assigning its stalls); that other resident programs hide the K latency is inferred.
  Whether a ring would lift m = 1024 (1.191) and m = 6144 (1.246), two of the three lowest n3072 ratios ,
  is not measured (`../gemm/mainloop.md` §1). The m = 6144 file serves its Point with its generic `_gemm_async_kernel`
  (128 × 128) and sends m = 2048, 4096, 8192 and m = 6145's bulk to `_gemm_strip_kernel` (S24), within 3 % of it at
  m = 6144 (B/A 0.981–0.992).
- **Band the M tiles.** `_wide_kernel` (m = 4096) and `_gemm_wide_kernel` (m = 8192) walk `GROUP_M = 4` M tiles per
  N tile, then interleave N (S16); m = 2048 and 6144 use bands of 2 and 4 (`ORDER_M`, S24), m = 1024 none. A band of 1
  instead of 4 made the measured m = 4096 Point 1.86 % slower in the balanced comparison  (§8).
- **Ragged M: compile the bulk at a divisible M.** m = 8193 launches `cdiv(8193, 128) · 24 = 1560` programs; the 24 of
  the last M tile run `_compute_tail_tile` on 16 rows, the others `_compute_tile_rowwise` compiled with M = 8192, so
  every row mask folds away (S17); compiled at 8193 the bulk was 9–10 % slower (§11). A higher stored speedup at M8193 does not isolate the bulk/tail design: the baseline also slowed at
  that shape. Do not infer a baseline tile transition without its source/trace.
  The tail gets no padded layoutThe tail gets no padded layout at 16 rows
  (`compute_efficient_padded_shared_layout` returns `None` for a [16, 128] fp8 A operand of a 16 × 16 × 128 4-warp
  layout, a layout from 32 rows; compiled): it copies 32 A rows, masked by `(am[:, None] < M) | (M % AM == 0)`, and
  keeps 16 with `gl.amd.slice` (S17). A buffer descriptor is not a tensor-bounds check: an unmasked address beyond the logical tensor but still inside the descriptor's byte range may read backing storage. Behavior outside that range, or through a different pointer/descriptor, is not established by this observation (`gluon/memory.md` §3).
## 8. Tile order across XCDs
In  tested launch grids, workgroup `i` ran on XCD `i % 8`; the placement models below assume that mapping. Each XCD has its own 4 MiB L2  (`../gemm/schedule_and_locality.md`
§2). The a8w8 files remap the program id by regime (placement derived from the formulas, each a permutation of its
grid; on multi-axis grids axis 0 is dispatched fastest, measured on 10 grids ):
| Point | Remap | Placement |
|---|---|---|
| n3072 m = 1–4 | `pid_n = (pid % 8) * (N // (BN * 8)) + pid // 8` (S8) | XCD x streams columns [384x, 384x + 384) |
| n3072 m = 16, 32 | `pid_n = (pid % STRIPES) * (192 // STRIPES) + pid // (STRIPES * (M // 16))`, 32 or 16 stripes (S18) | runs of 6 (m = 16) or 12 (m = 32) adjacent tiles per XCD; at m = 32 both M tiles of an N tile on one XCD |
| n3072 m = 64 | stripes inside 256-program super-blocks (S11) | the 4 M tiles of an N tile on one XCD |
| n7168 M = 2, 4 | `(tile % 8) * (N // 16 // 8) + tile // 8` (S19) | 56 adjacent tiles per XCD |
| n3072 m = 128 | none: N tile `program_id(0)` of a (48, 4) grid, M tile `program_id(1)` (S27) | the 4 M tiles of an N tile on one XCD |
| n3072 m = 256 | `pn = pn ^ (pm * 2)` on the same grid (S27) | CPU model: an N tile's 4 consumers on 4 XCDs; actual L2 fills unknown  |
| n7168 M = 128 | `tile_n = program_id(0) * 2 + parity` with a parity XOR, (56, 4) grid (S27) | both M tiles of an N tile on one XCD |
| n7168 M = 256 | 16-tile N clusters, N interleaved, (16, 4, 7) grid (S27) | all 4 M tiles of an N tile on one XCD |
| n3072 m ≥ 2048 | bands of 2 or 4 M tiles per N tile (`ORDER_M`, `GROUP_M`), N interleaved at m = 4096 and 8192 (S16, S24) | consecutive programs share a B panel; each XCD's programs touch a block of A tiles and B panels |
- **Measure the actual tile order.** At n3072/M4096, replacing grouped M order with M-major order increased
  measured call time about 1.9%. This trades A and B locality; a CPU fill model is not an L2 traffic counter.
- **One weight reader does not imply no remap effect.** At n7168/M64, removing the permutation was compatible
  with a ±1.5% equivalence band; at M2 it changed time about 2.4%. Tiny differences at M1/M16 remained unresolved.
- **Do not transfer a neighbouring Point's remap.** At n3072/M256, dropping `pn ^ (pm * 2)` reduced measured
  time about 3%, while the grouped orders at other measured Points often helped. Placement models do not
  isolate a unique L2 cause. Prove each launched grid's permutation ([gemm/decode_gemm.md](gemm/decode_gemm.md) §3).
## 9. The bf16 decode GEMMs and split-K under a bitwise contract
The two plain-Triton files, against aiter's tuned `gemm_a16w16` (S1), give a workgroup one N tile across all rows of
`a` (S20, S22): over all of K at n6144 (192 workgroups) and at n2624 m ≤ 8 (164), over a sixth of K at n2624 m = 64 (41
tiles × 6 splits = 246, then a reduce launch; counts traced). They read the weight once with `.cg`, remap workgroups
per XCD and at some Points rotate each one's K start; choices and numbers: `gemm/decode_gemm.md` §1–§8, which
distinguishes stored validate lower bounds from measured per-call time ratios under each stated timer. What Triton 3.8.0 adds on gfx950:

Compare candidate and baseline times at the same Point; a shared binary across M2/4/8 does not make
those workloads equally fast or assign their ratio differences solely to the baseline.
- **`num_stages=3` pipelines a plain `tl.dot` loop with direct-to-LDS copies:** n2624's m = 16 path compiles to 48
  `buffer_load_dwordx4 … lds`, half `sc0 nt` (the `.cg` weight), none at `num_stages=1`; Gluon gets none (§5).
- **Warps beyond the tile replicate it.** n2624 at m ≤ a 16 × 16 tile on 4 warps, compiled to
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
- **Price the whole split call under its own ruler.** Count initialization, partials and merge, not just GEMM.
  A counter-based last-arriver fold must reset state on every replay, but a correct in-launch reset need not
  require a separate zero node ([authoring.md](authoring.md) §7). A reduction launch can still win overall.
- **Split placement is shape-specific.** Wave-local splitting beat workgroup splitting plus merge at
  M32/N4096/K7168, but rotating-buffer tests reversed the winner at other N/M. The fp8 contract permits
  reordering only while every output remains within its bound.
## 10. Per-Point configuration
Each Point's configuration is Python: host `if m == …` branches choose the kernel and its constexprs (S2), or a
`@gluon.constexpr_function` feeds kernel and launch (m = 32's `_schedule`, S15); `tuning.md`, `authoring.md`.
| m | Kernel, tile | Programs × waves | K pipeline | `waves_per_eu` | VGPRs (AGPRs within) | LDS KiB |
|---|---|---|---|---|---|---|
| 1 | `_narrow_small_gemm`, 1 × 4 | 768 × 1 | 3 panels of 4 groups, all issued first | 8 | 40 (8) | 9.0 |
| 2 | same, `iterative-ilp` scheduler, 2 × 4 | 768 × 1 | same | 4 | 72 (8) | 9.0 |
| 4 | same, 4 × 4 | 768 × 1 | same, LDS read-ahead 1, scales preloaded | 4 | 44 (0) | 12.0 |
| 8 | `_packed_m8_gemm`, 8 × 16 | 192 × 2 | 6 two-group products first, 3 commit groups | 2 | 74 (0) | 36.0 |
| 16 | `_private_k128_gemm`, 16 × 16 | 192 × 1 | 12 private slots, `PREFETCH` 5, read-ahead 2 | 3 | 88 (8) | 48.0 |
| 32 | `_block_scaled_gemm`, 16 × 16 | 384 × 1 | 12 private slots, `PREFETCH` 5, two accumulators | 8 | 64 (32) | 49.0 |
| 64 | `_block_scaled_gemm`, 16 × 16 | 768 × 1 | 9-slot ring, `PREFETCH` 4 | 6 | 64 (24) | 37.3 |
| 128 | `_decode_kernel`, 32 × 64 | 192 × 4 | 4 + 4 slots, A and B 2 groups ahead | — | 130 (0) | 49.3 |
| 256 | `_decode_kernel`, 64 × 64, 32 × 32 × 64 | 192 × 4 | 3 + 4 slots, A 1 and B 2 groups ahead | — | 84 (0) | 56.8 |
| 1024 | `_gemm_strip_kernel`, 64 × 96 | 512 × 4 | single buffer, 3 panels | — | 92 (0) | 24.7 |
| 2048 | `_gemm_three_panel_kernel`, 64 × 192 | 512 × 4 | single buffer, 3 panels | — | 144 (0) | 37.0 |
| 4096 | `_wide_kernel`, 128 × 192 | 512 × 4 | single buffer, 3 panels, `GROUP_M` 4 | — | 242 (0) | 49.3 |
| 6144 | `_gemm_async_kernel`, 128 × 128 | 1152 × 4 | single buffer | — | 158 (0) | 41.0 |
| 8192 | `_gemm_wide_kernel`, 128 × 192 | 1024 × 4 | single buffer, 3 panels, `GROUP_M` 4 | — | 242 (0) | 49.3 |
| 8193 | `_fp8_block_scaled_kernel`, 128 × 128 and tail | 1560 × 4 | single buffer | — | 128 (0) | 39.0 |
| n7168: 1 | `_packed_k256_output`, 1 × 16 | 448 × 1 | A whole; B 4 slots of 2 groups, 8 of 16 primed | 1 | 80 (0) | 18.0 |
| 2 | `_compact_m2_gemm`, 2 × 16 | 448 × 1 | A whole; B 10 one-group slots, 8 primed | — | 76 (0) | 24.0 |
| 4 | `_packed_small`, 4 × 16 | 448 × 1 | A whole; B 3 slots of 4 groups, 8 primed | 3 | 96 (16) | 32.0 |
| 8 | `_packed_eight`, 8 × 16 | 448 × 1 | 4 slots of 2 groups, 3 ahead | 2 | 108 (0) | 24.5 |
| 16 | `_pipeline_output`, 16 × 32 | 224 × 2 | 8 stages of 256 K; 4 A + 6 B slots, 3 ahead | 3 | 96 (16) | 64.0 |
| 32 | same, 32 × 32 | 224 × 4 | 8 stages of 256 K; 4 slots, 3 ahead | 1 | 98 (0) | 66.1 |
| 64 | same, 64 × 32 | 224 × 4 | 8 stages of 256 K; 3 slots, 2 ahead | 2 | 152 (24) | 72.0 |
| 128 | `_fp8_pipeline_kernel`, 64 × 64, 32 × 32 × 64 | 224 × 4 | 3 slots, 2 groups ahead | — | 236 (0) | 52.0 |
| 256 | `_fp8_full_pipeline_kernel`, 64 × 64, 32 × 32 × 64 | 448 × 4 | 3 slots, 2 groups ahead | — | 62 (0) | 52.7 |
These are representative Triton 3.8.0 resource counts; compile the exact Point and build before budgeting.
n7168 M = 16–64n7168 M = 16–64 walk K in 8 stages of `BK = 256` split into two 128-K groups with
`gl.amd.slice` (S13); M = 256 writes its padded layouts by hand, M = 128 uses swizzled ones (S7).
- **What changes as M grows.** Waves per program 1 (n3072 m = 8: 2; n7168 M = 16: 2, M = 32, 64: 4) up to M = 64, then
  4; K groups issued ahead (`PREFETCH`): all of K (n3072 m ≤ 8), 5 (m = 16, 32), 4 (m = 64), 1–2 in a four-wave ring
  (m = 128, 256), a single buffer from m = 1024; `waves_per_eu` only at M ≤ 64; `iterative-ilp` only at m = 2; 32 × 32
  × 64 only at n3072 m = 256, n7168 M ≥ 128; no spills.
- **Which constants are measured:** those in §11, the fold delays (§5), m = 256's tile order (§8) and m = 6144's
  kernel (§7); other constants discussed here are kept, with no isolated measurement cited on this page; this is not an exhaustive campaign-history claim.
## 11. Tuning candidates and applicability

- **N3072/M16:** prefetch3 lost to prefetch5; swizzle and register-budget changes were close to the ruler's
  floor. Small differences require paired uncertainty and A/A controls, not a declaration of no effect.
- **N3072/M4096:** `group_m=4` beat 1 by about 1.9% in a balanced comparison. Folding scales as `(xs * sw)`
  before the per-element multiply reduced time about 6–8% on this Point; check the contract because it
  changes rounding. It does not explain the full roofline gap.
- **N3072/M8193:** compiling the bulk at divisible M8192 was about 9–10% faster than retaining bulk masks.
- **N7168/M64:** correctly waited deeper-prefetch variants improved time about 2%; fixed-four-slot controls
  favored prefetch2/3 over 1 at the measured padding settings. LDS, waits, live ranges and scheduling change
  together; do not present these results as proof of a unique bytes-in-flight bottleneck.
- **Choose the next experiment from the actual bottleneck.** Folding, scale placement, cache policy, MFMA
  shape, explicit warp-pipeline staging and async-versus-register loads remain candidates where their effect
  has not been separated in the real kernel. A pure-stream result is not a production GEMM guarantee.

## 12. Epilogue: round, then store
- **Round to bf16 before a layout change:** a [32, 64] exchange then takes 4096 bytes of LDS, not 8192, for the same
  output (compiled; MI355X). m = 128 goes through an XOR-swizzled exchange into `BlockedLayout([1, 8], [8, 8], [4, 1],
  [1, 0])`, stores with `cache=".wt"` and calls `_keep_alive`, whose removal must be checked in LLVM IR (S12,
  `gluon/memory.md` §15). n7168 M = 64 converts `acc.to(gl.bfloat16)` to a 16-byte-per-lane layout (S13). A lane stores
  2 or 4 bytes at M ≤ 8, 8 at M = 16–64 (n7168 M = 64: 16), 16 from M = 128; `../gemm/epilogue.md`.
- **The store hint is chosen per Point; measured only at m = 6144 and 8193.** Compiled: no cache bits at n3072 m = 1–4,
  16–64 and n7168 M = 2, 4, 16, 32; `.wt` (`sc0 sc1`) at n3072 m = 8, 128, 256, 1024, 4096, 8192 and n7168 M = 1, 8, 64,
  256; `.cs` (`nt`, through `gl.store`) at m = 2048, 6144, 8193 and n7168 M = 128. On a Gluon write stream
  no hinted store beat the unhinted one by more than 0.7 %, within the 0.3–1.5 % that identically compiled stores differ,
  and `.cs` was slowest, 5.5–7.2 % below it (3 runs) , `gluon/memory.md` §6; `.cs` → none (36 and 48 MiB of
  bf16 output) cost 1.7–1.9 % at m = 6144 and gained 0.1–0.7 % at m = 8193 (runs overlapping), all 8 pairs of
. In the tested four-store toy, `.wt` through `gl.store` inserted a wait after each store; `buffer_store` did not.
- **Applicability:** the skeleton, ring depth, folding and split examples cover finite shapes and inputs.
  Validate new shapes and compare complete calls; none establishes arbitrary-shape correctness, universal
  prefetch thresholds or an optimal split count.
## Evidence

AKLR source paths refer to main `43e112e` unless another commit is stated.
Triton compiler paths refer to `triton-lang/triton@c01b677` (v3.8.0).
Measurements in the text are gfx950 examples, not universal performance guarantees.

- S1 aklr/ops/gemm/contract.py:201 `GEMM_A8W8_BLOCKWISE_SCALED`: fp8 `a`, `b` (:237-243), fp32 `a_scale` per 1 × 128 and `b_scale` per 128 × 128 (:244-259), log-uniform scales (:192), bf16 default `out` (:262-268), `BOUNDED` compare (:269) with the FIXME at :186-188; the layout variants `row_major` (`operand_format` `none:none`, `b.is_shuffled` false), `b_shuffled` (`Preshuffle(tile=(16, 16))` `b`, `ColumnMajor` `a_scale`, `a_scale.is_row_major` false) and `b_shuffled_row_major_a_scale` (`a_scale.is_row_major` true) at :270-295; a variant other than the first joins the schema name at aklr/core/contract/op.py:626-627; aiter's `gemm_a8w8_blockscale` for `none:none`, `gemm_a8w8_blockscale_bpreshuffle` for a shuffled `b` and `gemm_a16w16` for `gemm@1.0` at aklr/ops/gemm/baseline.py:57, :58-60, :17-21; `runs_agree` at aklr/core/contract/semantics/_compare.py:142: `bounded` passes each run on its own, `bitwise` wants every output byte-identical to the first; `CompareSpec` defaults to `bitwise` at aklr/core/contract/spec/_args.py:204
- S2 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:1000 the m ≥ 16 host branch: `depth = 5 if m in (16, 32) else 4` (:1002), `waves = 3 if m == 16 else 6` (:1003), read-ahead 2 and refill-before at m = 16 (:1004-1005), `BANK_XOR=False` (:1023), `waves_per_eu=waves` (:1032); its wait at :440-447; `run` at 083b493a1861424c.json#main.py:8-20; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:423 the partial from `gl.zeros`; `acc = gl.fma(partial * sx, sw, acc)` at :482; `LC` at :299-304; `DotOperandLayout(OPERAND, LAYOUT, K_WIDTH)` at :86-88, `K_WIDTH=16 if m == 16 else 32` at :1024; `k_width` 32 at m = 64 in 54f6affa5c286a70.json#kernel.py:41-43
- S4 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/098200aca0238880.json#kernel.py:326 the gate-up K loop over K = 3072, `mfma_scaled` and the fold at :44-64, 16 × 16-preshuffled offsets at :68; aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:199 32 groups of K = 4096, fp8 widened to fp16 for `mfma` (:65-74), one scale fold per group (:213); K split over the waves of a workgroup in aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:593 `_w13_local_split`: `AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[SK, 1, 1])` (:618-623), plain `gl.amd.cdna4.mfma` per 128-K step (:706-710), the scale fold (:711), `gl.sum(acc, 0)` (:713)
- S5 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/199c45f4249e7f55.json#kernel.py:450 `compute_efficient_padded_shared_layout` with a `k_width = 16` operand layout (:450-455), read with `dot_width = 32` (:404); `_copy_layout` at :299-309; 3 A slots refilled 1 ahead (:465, :526); `REDUCTION_DELAY = 0 if BLOCK_M == 32 else 2` (:492), the delayed fold at :572-577, drained after the loop at :579-580; the helper returns `None` when `k_width` is not 4, 8 or 16 at triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:139-143
- S6 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_ops.py:68 both scales `None`: `uint8` `0x7F` (:69), a splat in the scale layout (:99), `dot_scaled` (:105); the version-4 unscaled fp8 entries `mfma_f32_16x16x32_fp8_fp8` and 32×32×16 at third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp:225, :205, 16×16×128 and 32×32×64 only as `mfma_scale_*_f8f6f4` (:264-268)
- S7 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1bbf98a368b3fcba.json#kernel.py:83 two chained `mfma_scaled` per 128-K group (:83-92), one fold (:94); inline `s_waitcnt lgkmcnt(0)`, then `wait_group` and `gl.barrier` (:66-73), the same in fee7e092eab87fc5.json#kernel.py:79-86; M = 128's swizzled operand layouts at :131-132, M = 256's hand-written padded ones at fee7e092eab87fc5.json#kernel.py:349-360
- S8 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/bd017b0794d5db31.json#kernel.py:728 folded offsets; `PG = 4`, `BN = 4` at :685-687; XCD order at :720; carrier at :753-758; scales loaded after each panel's wait at :761-763; diagonal at :781-786; group sum at :792; m = 8: aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/0e15a0289735acb0.json#kernel.py:581 two-group offsets (`ar // 2`, `ar % 2`, :581-586); 6 products in 3 commit groups at :589-605; activation scales loaded after all copies at :611-616; diagonal at :644-645
- S9 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` runs no loop scheduling or automatic pipelining pass; its `add_warp_pipeline` (:339) acts only on ops marked `triton.warp_pipeline.border` and skips other loops (third_party/amd/lib/TritonAMDGPUTransforms/WarpPipeliner.cpp:39-40, :413); `make_ttgir` runs `add_schedule_loops(num_stages)` and `add_pipeline` at :288-289; triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47 `async_wait` "does not provide any syncronisation in the CTA" (:54); lib/Analysis/Membar.cpp:289-298 a barrier after a memory wait with no later barrier or wait before the next access; third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h:9-28 the AMD filter drops the barrier between an async copy and a local load synced by `AsyncWait` (what `load_shared_relaxed` marks) of the same buffer, and notes it is wrong for a single buffer refilled after its read; code at third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11
- S10 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:131 `_copy_private_group`: two 64-byte halves of A and of B by `buffer_load_to_shared`, one `commit_group` (:142); copy layout `BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])` at :276; prologue of `PREFETCH` groups at :363; loop at :406; refill at :426-439; wait at :440-447; read-ahead at :448-463; aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:322 `BANK_XOR` copy rows and contiguity hints (:322-328); read basis `[4 if BANK_XOR else 0, 32]` at :72; `(2, 16, 64)` buffers at :295-296; the `(16, 128)` view at :83 and `permute` for B at :85
- S11 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/54f6affa5c286a70.json#kernel.py:174 `LDS_SLOTS` 9; stripes inside a 256-program super-block at :184-192; waits at :308-312; `gl.barrier` before the first slot reuse at :340-342; both scale tables copied into LDS as one commit group at :91-146 (redirected columns at :129, :141), issued at :235 after `DATA_FIRST` groups; A and B committed apart from group 9 on at :67
- S12 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/3480e97a5fcda6f6.json#kernel.py:252 `wait_group`, then `gl.barrier` (:265), then `load_shared_relaxed` (:269); 4 + 4 slots refilled 2 ahead at :203-204, :248-251; scale window 4 groups ahead at :271-276; hand-written padded layouts at :147-185; copy layout from `offset_bases` at :19-29; rounded exchange, `.wt` store and `_keep_alive` at :312-362
- S13 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:161 `SLOTS`, `PREFETCH`, `WAITS` (:161-166); `BK = 256`, `STAGES = 8` at :159-160 and `gl.amd.slice` per group at :51-56; plain `.load` reads at :248-249; `SPLIT_COMMIT` refills at :116-131, :251-257; `_swizzled_k` at :14-15; copy into a linear `_reinterpret` at :108-112; read layout at :175; store layout at :91-104
- S14 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared`, which "does still complete in order with ttgl.loads/stores or buffer_loads/stores on CDNA4" (:78-80) and fails to lower unless 128 or 32 bits per lane, writes to `dest` coalesced and a swizzle only within a warp (:89-94); `wait_group` at :137; `load_shared_relaxed` at :151
- S15 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:213 `_private_register_scales`: 3 `buffer_load`s of activation scales (:228-234), 12 scalar weight scales (:245-246); called at :343, before the prologue copies at :363; m = 32's SGPR pointer and `gl.pointer_type(gl.float32, 4)` loads at fcf21622f6c91bbb.json#kernel.py:125-149 and its `_schedule` (`@gluon.constexpr_function`) at :31, read in the kernel at :229 and by the host at :415 and :421
- S16 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/514a90451e2ca0aa.json#kernel.py:665 single-buffered group loop: scale gathers (:666-670), `wait_group(0)` (:676), reads, `gl.barrier` and refill before the last panel (:683-687); middle panel scale (:671); `GROUP_M` bands (:585-590), `GROUP_M` 4 from `_launch_policy` (:27-32); m = 1024's `sw1 = gl.where(pn % 4 == 1, sw2, sw0)` at 8888fe378338cc8e.json#kernel.py:450; m = 4096's bands and N interleave at 5ea30df18b03ab8d.json#kernel.py:629-638, host `group_m = 2 if m <= 2048 else 4` (:1093)
- S17 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/831926aed9523c0f.json#kernel.py:562 ragged split: 16-row tail (:563-587), bulk at `M - M % BM` (:589-612); grid at :656; the tail's 32-row A layout (`AM = 32`, :88-93), row mask (:114) and `gl.amd.slice` to 16 rows (:163); `.ca` on the async copies at :146, :154
- S18 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:306 stripes (:306-307); `gl.static_range(12)` at :334, :406, :489; `12 - PREFETCH` at :426, :468, :523; `11 - group` at :445, :491; `3072` at :544-545; `assert n == 3072 and k == 1536` at :1052; the m = 32 stripes at fcf21622f6c91bbb.json#kernel.py:251-252; bd017b0794d5db31.json#kernel.py:759-761 3 panels and `wait_group(2 - panel)`; n7168 ae4fa8a440b98f09.json#kernel.py:185 `STAGES = 8`, :128 and :149 `gl.arange(0, 16,...)`, :222 `_load_scale_range(..., 0, 16,...)`; 06755ba3ee5eb085.json#kernel.py:171-172 the 224-tile permutation, :298 `assert m in (32, 64) and n == 7168 and k == 2048`
- S19 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/362ce3bb6200382e.json#kernel.py:136 M = 1: `A_PACK = 4`, `B_PACK = 2` (:136-137), the `example` carrier (:179-181), 4 B slots primed at :191-193 and refilled at :211-216; M = 2 at ff2dc820165bd28a.json#kernel.py:408 `PRIME = 4`, A rows fold 4 groups (:434-435), `_packed_diagonal` (:471), refill (:475), 56 adjacent tiles per XCD (:415); M = 4 at 0b1065ae9c0652b7.json#kernel.py:288 `_packed_group` (:288-299), 2 of 4 B panels primed (:368), refill (:413-414), the XCD order (:318); M = 8: aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1e021d44010accb1.json#kernel.py:135 `_diagonal` with `gl.amd.slice`, `reshape`, `permute` and an `assert_trivial` convert; `PG = 2` at :254; `convert_layout(..., assert_trivial=True)` raises a `TypeError` at compile time when the conversion is not trivial at triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:245
- S20 aklr/ops/gemm/impls/triton/gemm_n2624_k6144/b88c8d5f28d86777.json#kernel.py:87 `_reduce_kernel` (:86-93); split partial store at :81-83; split launch and reduce at :133-156; `run` at :161, `torch.empty` scratch at :162-167; `_xcd_remap` at :26-35, applied with `pid_k = pid // N_TILES` at :55-59; `offs_m = tl.arange(0, BLOCK_M)` at :63; the m ≤ 16 launch with M at run time at :100-125
- S21 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/ff2dc820165bd28a.json#kernel.py:409 M = 2: `DELAY = 2`, the deferred fold at :476-482; M = 4 `DELAY=3` at 0b1065ae9c0652b7.json#kernel.py:691; M = 256 accumulator pin by an empty tied `inline_asm_elementwise` at fee7e092eab87fc5.json#kernel.py:125-131 (served with `PAIRED=False`, :413); n3072 m = 8 two accumulators at 0e15a0289735acb0.json#kernel.py:646-649; m = 32 `K_ORDER` and `odd_acc` at fcf21622f6c91bbb.json#kernel.py:235-239, 386-387, 403-404
- S22 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:103 `schedule_hint`: "Experimental; right now no effect" (:103-105); n6144 passes it at aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json#kernel.py:168, has `offs_m = tl.arange(0, BLOCK_M)` at :46 and :94, launches m ≤ 16 with M at run time, `EVEN_M` only at m = 16 and `BLOCK_K` 256 but at m = 1 (:121-142), `run` at :199
- S23 aklr/core/utils/testing.py:111 `_kineto_replay_times_ms`: per replay, the sum of the CUDA kernel durations except the flush kernel (:126-137); `bench_cudagraph` returns their median (:152, :205); aklr/core/validation/stages/_bench/_execution/_runner.py:515 `GPU_FORCE_QUEUE_PROFILING="1"` in every GPU runner process
- S24 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/14c8e25ca1a39f6d.json#kernel.py:37 `ORDER_M` bands: consecutive programs walk `ORDER_M` M tiles of one N tile (:37-48); `ORDER_M` 2 at m = 2048 (:754); the same bands with `ORDER_M` 4 at m = 6144 in 7266cdc0ff1fd299.json#kernel.py:59-73, whose host serves m = 6144 with `_gemm_async_kernel` and m = 2048, 4096, 8192 and the bulk of m = 6145 with `_gemm_strip_kernel` (:1114-1165)
- S25 triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:144 CDNA4 shared memory `160 * 1024` (:144-145); python/triton/compiler/compiler.py:466 raises `OutOfResources` at the first launch when `metadata.shared` exceeds the device's (:466-468)
- S26 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4c2bed7e3e8c1e94.json#kernel.py:102 the `[STAGES, BM, BK]` ring; the rolled `range` at :121; refill into `.index((slot + STAGES - 1) % STAGES)`, then `wait_group`, `gl.barrier` and the relaxed read at :28-44
- S27 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/3480e97a5fcda6f6.json#kernel.py:123 m = 128: `pm`, `pn` from `program_id(1)`, `program_id(0)`, the XOR only `if BLOCK_M == 64` (:123-127), grid (48, 4) at :390; m = 256: `pn = pn ^ (pm * 2)` at 199c45f4249e7f55.json#kernel.py:390, grid at :706; n7168 M = 128 at 1bbf98a368b3fcba.json#kernel.py:120-123, grid at :489; M = 256 at fee7e092eab87fc5.json#kernel.py:338-343, grid at :448
- S28 aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k8192/10faf68f4a569ef8.json#kernel.py:1087 `_compile_skinny`, taken for m ≤ 32 (:1074, :1389): each of four waves takes a contiguous quarter of K, `K // tile_k // nw` = 16 groups (:1067-1068, :1093), the partials summed through LDS in wave order (:1301-1312), grid `((M + tile_m - 1) // tile_m, N // tile_n)` with 16-column tiles (:1357); the K 4096 file splits K over 8 waves at m = 1, 2 and 4 at m = 4–32, "256 WGs fill the 256 CUs", at aklr/ops/gemm/impls/flydsl/gemm_a8w8_blockwise_scaled_b_shuffled_n4096_k4096/8c907984278eefc4.json#kernel.py:44-58
