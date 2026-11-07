---
title: "Gluon layouts on gfx950 (Triton 3.8.0)"
kind: language-note
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
sources:
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_layouts.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py
  - triton-lang/triton@c01b677:python/src/gluon_ir.cc
  - triton-lang/triton@c01b677:third_party/amd/backend/compiler.py
  - triton-lang/triton@c01b677:lib/Dialect/Gluon/Transforms/Canonicalize.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Types.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Ops.cpp
  - triton-lang/triton@c01b677:lib/Dialect/Triton/IR/Ops.cpp
  - triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
  - triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp
  - triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp
  - triton-lang/triton@c01b677:lib/Analysis/Utility.cpp
  - triton-lang/triton@c01b677:lib/Tools/LinearLayout.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/Utility.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/LowerLoops.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/Analysis/AMDGPUAllocation.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp
---

# Gluon layouts on gfx950 (Triton 3.8.0)

A Gluon tensor lives in the layout its code gives it: which lane and which register of which wave holds each element.
This page gives the layout classes of Triton 3.8.0 that matter on gfx950, as AKLR's Gluon kernels (main `43e112e`) use
them, how to pick one for each role in a kernel, and what `gl.convert_layout` costs; moving data is `memory.md`, the
matrix-core instructions `mfma.md`, reading the compiled kernel `../debugging.md`. "Compiled" marks what a kernel
compiled with Triton 3.8.0 for gfx950 (no GPU run) showed in its ISA; page-local Evidence retains the checked conditions (S58).

## 1. Ground rules
- **The layout you write is the layout you get.** `gluon_to_ttgir` runs `resolve_auto_encodings` and none of the
  coalesce, remove-layout-conversions and accelerate-matmul passes that plain `tl.*` kernels get (S1, S2): compiled, a
  copy written in `BlockedLayout([1, 1], [1, 64], ...)` keeps it and loads with `global_load_ushort`. The Gluon
  canonicalizer leaves `convert_layout` alone (S3): A → B → A stays two conversions (a 64×64 fp32 tile flipped to
  `order=[0, 1]` and back: 32 `ds_bpermute_b32`). View ops, `expand_dims` and reductions infer their result layouts
  (S4); two layouts in one op are not reconciled (table).
- **Pass a layout to every `arange`, `zeros` and `full`.** Without one they get `AutoLayout` (S4), taken from the
  concrete layouts the value meets; where nothing gives it one, the compile fails (table).
- **What Gluon checks, and what it leaves to you:**
  | Rejected at compile time | Accepted without a word |
  |---|---|
  | an add of two different BlockedLayouts: "Layout mismatch in broadcast"; `expand_dims` of a tensor not in `SliceLayout(axis, ·)` (S4) | a layout larger than the tensor: extra lanes or waves hold copies (§2) |
  | lanes ≠ 64 or waves ≠ `num_warps` (`[1]` at 4 waves, `[4]` at 8): "Layout has 32 threads per warp, but the module specifies 64 threads per warp" (S5); `num_warps=3`: "num_warps must be a power of 2" | a load whose layout or proven contiguity gives a lane less than 16 bytes (§6.1) |
  | `mfma` operands both in a BlockedLayout: "Dot's a/b's encoding should be of DotOperandEncodingAttr" (S5); `mfma_scaled` operands whose parent differs from the accumulator's MFMA layout (`transposed` False against True): "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout" (S6); unequal `k_width` in `mfma` (S5) | unequal `k_width` in `mfma_scaled` (S7) |
  | an `instr_shape` `mfma` has no intrinsic for, e.g. e4m3 `[16, 16, 128]` (`[16, 16, 32]` compiles): "no matching matrix core intrinsic for mfma version 4 …" (`mfma.md` §1); a 4×64 layout (`instr_shape=[4, 64, 64]`) with `transposed=True`, or a 64×4 one with `False`, both past `verify` (§3) | bank conflicts of a shared layout (§5) |
  | `gl.split` whose size-2 axis is not in registers (S5); a non-power-of-two extent (`gl.arange(0, 96, L)`: "Shape element 0 must be a power of 2"), or a non-power-of-two last LDS dimension (`[2, 16, 3]`: "shape must have power-of-2 and non-zero dimensions") (S28, §2) | a `convert_layout` through LDS: only the ISA shows it (§8) |
  | a rank-3 shared layout under `.index()` (§4); `convert_layout(..., assert_trivial=True)` that moves data across lanes or waves, and `gl.amd.slice` off the layout's tile (§7) | a reduction or gather whose axis spans waves: an LDS round trip and a barrier (§6.7) |
  | `gl.associative_scan` over any layout but a `BlockedLayout` (a slice, MFMA or linear layout): "TODO: unsupported scan layout" (S8); convert to a 1-D or 2-D BlockedLayout first | |
  | `gl.gather` whose index has another rank, or another extent off the axis, than its source (S4); a layout-less `arange`, `zeros` or `full` that nothing resolves: `PassManager::run failed` (`arange`: "'tt.make_range' op Failed to infer return type"); `gl.CoalescedLayout()`: no infer-coalesced pass in the AMD pipeline (S1), the lowering asserts "cast<Ty>() argument of incompatible type!" | |
  An MLIR-level rejection is a bare `RuntimeError` naming the stage (`error encountered during parsing`,
  `PassManager::run failed`, `failed to translate module to LLVM IR`) below an `error:` line with the cause, at
  translation only the failing op (`../debugging.md` §4). A non-transposed 64×4 layout aborts the process (exit 134).
- **A wave is 64 lanes** on gfx950 (S2): `threads_per_warp` multiplies to 64 and a `DistributedLinearLayout` has 6 lane
  bases (all 94 literal lane-basis lists in AKLR's kernels do); waves match `num_warps`, a power of two (table). Write
  `gl.num_warps()` into a layout so one source serves several launch widths, as 26 kernel directories do (S9).
- **Look at a layout before you use it.** `L.format_hardware_view(shape)` prints, register by register, the element
  each lane holds, from plain Python without a GPU (S10); inside a kernel, `gl.static_print(gl.to_linear_layout(L,
  shape))` prints its bases at compile time (S4). The bases in this page come from there.
- **Declare** layouts as `gl.constexpr` locals in the jit body, or derive them in `@gluon.constexpr_function` helpers
  (`_copy_layout`, §6.4). A module-level layout, plain or `gl.constexpr(...)` (S11), compiles used directly or bound as
  `L: gl.constexpr = MOD_L`; bound to a plain local, `L = MOD_L`, it fails: "cannot convert BlockedLayout".
## 2. Distributed layouts: BlockedLayout, SliceLayout, DistributedLinearLayout
| Class (3.8.0 signature) | What the parameters say | Use it for |
|---|---|---|
| `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])` (S10) | per dimension: contiguous elements per lane, lanes, waves; `order` lists dimensions fastest first; `cga_layout` stays `[]` (one CTA per program) | global loads and stores, element-wise math, reductions |
| `SliceLayout(dim, parent)` (S10) | `parent` without dimension `dim`; lanes that differed only along `dim` hold the same element | index vectors; per-row or per-column values that broadcast back into `parent` |
| `DistributedLinearLayout(reg_bases, lane_bases, warp_bases, block_bases, shape)` (S10) | one basis per bit of the register, lane and wave index; an element's coordinates are the XOR of the bases of the set bits; a zero basis means a copy | ownerships no BlockedLayout expresses: copy layouts derived from a shared layout (§6.4), per-row scale tiles that keep 4 K groups in a lane's registers (`_scale_window_layout`, S27), MX scale layouts (`get_mfma_scale_layout` returns one, §3) |
- A BlockedLayout covers `size_per_thread × threads_per_warp × warps_per_cta` per dimension: `BlockedLayout([1, 8], [8,
  8], [4, 1], [1, 0])` covers 32×64; on 64×64 it gains register basis `[32, 0]` (two 8-element runs 32 rows apart per
  lane), on 16×64 its second warp basis becomes `[0, 0]` (waves 2 and 3 copy waves 0 and 1). The lowering masks a copy's
  stores and atomics in the tested 16×64 wave-copy example (S33); compiled, waves 2 and 3 branch past loads, adds,
  stores and atomics. Independent r2 checks found one write per active address for `gl.store`/`buffer_store` in
  optimized LLVM IR and atomic-add counts of 4096 for 4096 programs under unmasked, all-true and even-element masks
  *(field, E12)*. This is not a guarantee for every layout or LLVM elimination mechanism: four register-copy layouts
  counted 8192 instead of 4096, and 8 buffer-store checks duplicated writes (all involved register copies).
  The MiniMax-M2 E256 combine at 4096 tokens runs a `[1, 4] × [1, 64] × [1, 4]` layout, 1024 columns wide, over
  512-column blocks (S12); its actual 4-wave compiled kernel sends waves 2 and 3 past all data-memory operations to
  the end (9 loads and 1 store remain reachable by waves 0 and 1; kernarg loads are separate) *(compiled, E12)*.
  Copies placed on purpose can keep reductions wave-local (§6.7).
- **Index vectors are built in a slice of the layout the data must land in**: `gl.arange(0, BN, gl.SliceLayout(0, L))`
  for columns and `SliceLayout(1, L)` for rows; `[None, :]` and `[:, None]` broadcast them back into `L`, and a load
  through those offsets lands in `L` with no conversion, whatever `L` is (blocked, MFMA, dot operand, scale).
  All 150 Gluon kernel directories use `SliceLayout`; deeper tensors nest slices (`SliceLayout(1, SliceLayout(2, L))`).
- **A non-power-of-two extent** is split into power-of-two pieces, as the plain-Triton RMSNorm at hidden 6144 holds a
  row in three 2048-wide chunks (S55) and the fp8 GEMM's 96- and 192-column tiles are a `[GROUP_K, 2·BN]` LDS buffer
  sliced in two plus a `[GROUP_K, BN]` one (S56), or padded: that RMSNorm padded to 8192 stayed accurate, took
  1.022–1.037× the split time at 1024–4096 tokens and 0.988–1.023× at 3–512 (faster in one of 3 runs) *(field, E7)*. A
  `[SLOTS, M, K]` buffer under a rank-2 layout takes any slot count (`[n, 16, 64]`: n = 3, 5, 6, 7 compile) (S28).
## 3. AMDMFMALayout and DotOperandLayout
`gl.amd.AMDMFMALayout(version, instr_shape, transposed, warps_per_cta, element_bitwidth=None, tiles_per_warp=None,
cga_layout=[])` (S13) is the accumulator's layout, `gl.DotOperandLayout(operand_index, parent, k_width)` (S10) an
operand's, with the MFMA layout as `parent`. `version=4` is gfx950; `instr_shape` is the logical MFMA tile (M, N, K): the 16×16 and 32×32 forms map to one intrinsic, while the 4×64 and 64×4 forms are lowered as a broadcast sequence of 4×4 instructions, and
`verify` accepts M×N of 32×32, 16×16, 64×4 and 4×64; it also checks the version range, the three-element instruction shape, accumulator bitwidth and CGA basis rank, but not whether K has an intrinsic or the orientation is supported (instructions and rates: `mfma.md`; plain
`mfma` takes fp8 only at 16×16×32 and 32×32×16, K = 128 / 64 only through `mfma_scaled`, `mfma.md` §1). `gl.dot_fma`
takes `DotOperandLayout`s of a `BlockedLayout` parent with `k_width` 0 (`mfma.md` §2.6). Who holds what, from
`to_linear_layout`, for one wave:
| Layout | Registers of a lane | Lanes 0–15 | Lane bits 4, 5 |
|---|---|---|---|
| accumulator 16×16, `transposed=True` | 4 consecutive columns | rows 0–15 | columns +4, +8 |
| accumulator 16×16, `transposed=False` | 4 consecutive rows | columns 0–15 | rows +4, +8 |
| accumulator 32×32, `transposed=True` | columns +1, +2, +8, +16 (16 registers) | rows (lanes 0–31) | bit 5: column +4 |
| accumulator 16×16, `element_bitwidth=64` (fp64), `transposed=True` | columns +4, +8 | rows 0–15 | columns +1, +2 |
| A of 16×16×32 bf16, `k_width=8` | K 0–7 (16 bytes) | rows 0–15 | K +8, +16 |
| B of 16×16×32 bf16, `k_width=8` | K 0–7 | columns 0–15 | K +8, +16 |
| A of 16×16×128 fp8 (`mfma_scaled`), `k_width=16` | K 0–15, then K +64 | rows 0–15 | K +16, +32 |
| A of 16×16×128 fp8 (`mfma_scaled`), `k_width=32` | K 0–31 | rows 0–15 | K +32, +64 |
- **`transposed`** chooses which accumulator dimension a lane's 4 registers run along (S14); the lowering realises
  `True` by swapping A and B in the instruction (S15). With `True` (122 directories pass a literal `True`) a per-row
  scale in `SliceLayout(1, mma)` is one value per lane and a lane's bf16 row segment one `global_store_dwordx2`
  (compiled, one-wave `[16, 16, 32]`; `False`: 4 row-scale values per lane, 2-byte stores). Choose `False` when rows must sit in one lane's
  registers: the K-group-folded decode GEMMs of n7168 pick each token's 4 K-group rows out of registers with
  `reshape`/`permute`/`split` (S16; `../gemm.md` §4); gated attention keeps a row and its bf16 residual row in one lane
  so one softmax serves both (S17); the Qwen2-MoE router stores straight from a non-transposed accumulator at its
  2232-row Point (S18). A 4×64 layout (fp16 `instr_shape=[4, 64, 64]`, `k_width=4`: 16 `v_mfma_f32_4x4x4_16b_f16` for
  `[4, 64] × [64, 64]`; `[4, 64, 4]` has no instruction) must be `False`, a 64×4 one `True`: a transposed 4×64 stops at
  the lowering's assertion, a non-transposed 64×4 aborts the process (S14, S15, S19).
- **`warps_per_cta`** places waves along N first, then M (S14): with `[1, 4]` on a 16×128 accumulator wave w owns
  columns 16w to 16w + 15 and 64 + 16w to 64 + 16w + 15. An operand covers its whole K in lanes and registers, so wave
  bits on its K become copies (S14): with `[1, W]` every wave holds all of A (warp bases `[0, 0]`) and B is split, with
  with `[W, 1]` all of B; `[2, 2]` splits each operand in half only when its non-K extent spans at least two instruction tiles—an extent of one instruction tile is still replicated. Replicate the small operand where it is one instruction tile: compiled
  at M = 16, N = K = 128 (bf16 `[16, 16, 32]`, 4 waves), `[4, 1]` issues 32 MFMAs a wave, `[1, 4]` 8. MiniMax-M2's MXFP4
  experts run `[1, 4]` (S20), DeepSeek-V2's switch from `[1, 4]` to `[2, 2]` above 32 rows (S21); the replicated rows
  are staged once through LDS (§6.5) or loaded by every wave (§6.3). A rank-3 `[4, 1, 1]` puts one batch slice on each
  wave (warp bases `[1, 0, 0], [2, 0, 0]`) and takes plain `mfma`; `mfma_scaled` (e4m3 `[16, 16, 128]`, no scales)
  aborts on it (`mfma.md` §10). MiniMax-M2 FP8's in-CTA split-K sums such slices (§6.7).
- **`tiles_per_warp`** gives a wave adjacent MFMA tiles before the next wave starts (S14): `[1, 2]` with
  `[1, 4]` on 16×256 makes column +16 a register bit and +32, +64 wave bits. Mind it before `gl.split`: an epilogue that
  splits gate and up columns BN/2 apart (S22) needs column BN/2 in registers. At BN = 128, `[1, 2]` puts
  column 64 on a wave and the split fails ("Fp4ToFpOp/SplitOp requires at least 2 elements per thread"); `[1, 1]` at
  128 and `[1, 2]` at 256 compile. The MXFP4 gate-up kernels guard exactly this (S20). 22 directories set it.
- **`element_bitwidth=64`** selects the fp64 accumulator: with `transposed=True` a lane's values are 4 columns apart
  instead of 4 consecutive, with `False` 4 rows apart (S14, S19).
- **Fewer rows than the instruction**: 8 rows on a 16-row accumulator turn lane bit 3 into a copy with
  `transposed=True`, lane bit 5 (rows +8) with `False`; the MFMA still computes 16 rows, so 8-row decode tiles save no
  matrix work (`mfma.md` §8 puts the spare rows to use).
- **`k_width` is the number of consecutive K elements a lane holds**, not "elements per 32 bits" as the docstring says
  (S10, S14). Each compiles to its instruction: 8 for bf16 `[16, 16, 32]` (16 bytes), 16 where 16 fp8 bytes per lane become bf16 or fp16 (DSA's QK
  product, the Qwen2-MoE fp8 experts; S57, S19), 4 on fp16 `[4, 64, 64]` (S19), 16 or 32 for e4m3 and 16 for e2m1 bytes
  in `mfma_scaled` `[16, 16, 128]`, 1 or 4 for fp32 `[16, 16, 4]`. It decides which K indices each register holds (table
  above), so give A and B the same value; `mfma` rejects a mismatch (S5), `mfma_scaled` does not (S7), `mfma.md` §5.
- **An fp32 accumulator is already the next operand when no wave splits its columns** (one wave, or waves on M): the
  transposed 16×16×4 fp32 accumulator and `DotOperandLayout(0, mma, 4)` own the same elements, so `convert_layout(acc,
  a_dot, assert_trivial=True)` moves nothing, also for 64×64 on `[4, 1]` waves; with `k_width=1` it goes through LDS
  (§7). With waves along N the next A (K = this N) is replicated while C is split: on `[1, 4]` (16×64 or 64×64)
  `assert_trivial` into the `k_width=4` A fails, and unasserted a 16×64 tile costs 1 `ds_write_b128`, 4 `ds_read_b128`
  and 1 `s_barrier` (4 KB of LDS). Avoiding LDS does not guarantee faster chaining: a 64×64, 256-step fp32 chain,
  equal 64 MFMAs per wave per iteration, gave time(`[4,1]`)/time(`[1,4]`) = 1.369 / 1.213 at 4 / 16 programs per CU
  *(field, E11)*. Staging C yourself is §6.5 (`../gdn.md` §12).
- **MX scales** take `gl.amd.cdna4.get_mfma_scale_layout(dot_layout, shape)`, `shape` `[M, K // 32]` for A and `[N, K //
  32]` for B (N first, though B is `[K, N]`), K in elements, not packed bytes (S6); an A scale passed as `[K // 32, M]`
  fails while traced: "scale factor must be 16 or 32. Got 8" (K = 128; S7). For 16×16×128, lanes 0–15 are rows and lane
  bits 4–5 the four 32-wide K groups of one instruction (`mfma.md` §3). Under a `[1, W]` MFMA the A scales are
  replicated in every wave, as A is: compiled in a K loop under `[1, 4]`, A's `[32, 8]` table loaded in the scale
  layout's slices puts no LDS access in the loop; converting it every step from rows split over the waves, as MiniMax-M2
  E64 does (S20), costs 4 byte writes, 1 `ds_read_b32` and an `s_barrier` (not timed). Indexed in the slices, a
  row-major table takes 4 `global_load_ubyte` per lane (`mfma.md` §3 makes it one `global_load_dword`).
## 4. Shared layouts
| Class (3.8.0 signature) | Layout of the bytes | AKLR uses it for |
|---|---|---|
| `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` (S23) | rows along `order[1]`; in row r the `vec`-element chunks are XORed with `(r // per_phase) % max_phase`, wrapped to the row's chunk count (S24); `max_phase=1` is plain row-major | XOR-swizzled operand tiles; `(1, 1, 1, order)` flat views, scratch and scale tables (95 directories in all); among about 40 parameter sets, occurrences / directories: `(16, 1, 8, [1, 0])` 54 / 30, `(4, 1, 8, [1, 0])` 40 / 10, `(16, 1, 16, [1, 0])` 17 / 9, `(8, 1, 8, [1, 0])` 16 / 13, `(16, 1, 8, [0, 1])` 10 / 8, `(16, 1, 4, [1, 0])` 9 / 6, `(8, 2, 8, [1, 0])` 4 / 4, `(16, 4, 4, ·)` 4 / 2 |
| `PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)` (S23) | `offset_bases` map offset bits to (row, col) (row permutation); `padding` elements are inserted after every `interval`; powers of two; `cga_layout` is required (`[]`); `PaddedSharedLayout.with_identity_for([[interval, padding]], shape, order)` builds the plain case: identity offsets, `order` fastest dimension first | async-copy destinations and MFMA operand tiles (30 directories through the helper below, 13 through `with_identity_for`, 6 spelled out) |
| `SharedLinearLayout(offset_bases, block_bases=[], alignment=16)` (S23) | offset bit i holds `offset_bases[i]`; coordinates XOR; shape from the largest basis; `alignment` is the buffer's byte alignment | preshuffled weights read as an operand (§6.3), hand-built bank swizzles, scale tables in the reader's register order (§6.5) (35 directories) |
- **`gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)`** returns
  a padded layout for an async-copied MFMA operand, or `None` (S6). In 40 `tl.dot` builds (2 stages; bf16/fp8e4nv;
  16-/32-wide MFMA; 4/8 waves; five shapes from 32×32×64 to 256×128×64), all 74 padded operands matched the helper
  on intervals, padding, offset bases and shape. Six A `[32,64]` operands were swizzled instead and the helper returned
  `None` (32-wide bf16; either width fp8) *(compiled, E13.04)*. An interval is one wave instruction's 1 KiB:
  a K-contiguous A at `k_width` 8 (bf16) or 16 (fp8) gets `[[512, 16]]` or `[[1024, 32]]` with a 16-wide instruction,
  half that padding with a 32-wide one (bf16 at `k_width` 4 or 16: `[[512, 4]]`). The 16 rows of one MFMA read start in
  16 four-bank groups with a 32-wide instruction, in 8 with a 16-wide one (rows r and r + 8 share one), in 2 when
  unpadded. Read through it, a bf16 `[64, 64]` K-contiguous A compiles to `ds_read_b128`, a B stored `[K, N]` to
  `ds_read_b64_tr_b16` (fp8: `_b8`) (S25). It returns `None` for 4-byte elements, `k_width` outside 4, 8, 16, an
  instruction whose M (for A) or N (for B) is not 16 or 32 (the 4×64 shapes), and tiles with too few rows to stagger:
  evaluated, for fp8 A `[16, 128]` (`k_width=16`), bf16 A `[16, 64]` (`k_width=8`), any `k_width=32` and fp32, but not
  for fp8 A `[32, 128]`, bf16 A `[64, 64]`, bf16 B `[64, 256]` and packed fp4 B as `gl.uint8`. For a 16-row tile AKLR
  stages 32 rows and takes 16 with `gl.amd.slice` (S26); for a `k_width=32` operand it asks for `k_width=16`, and the
  n3072 fp8 GEMMs' hand-written layouts at BLOCK_M = 32 (the m=128 schema, S27, and m=256) equal the helper's output.
  The helper is not in Triton 3.7.0: a kernel that calls it declares `"requires": {"triton": ">=3.8"}`
  (`aklr/ops/README.md`, The kernel section and ABI), as 22 of the 30 directories that call it do; the eight n3072_k1536
  fp8 GEMMs (S26, S27 among them) do not (`../gemm.md` §1): do not copy their header.
- **A multi-slot buffer takes the layout of one slot**: `gl.allocate_shared_memory(dtype, [SLOTS, M, K], layout_2d)`,
  then `.index(i)`. A shared layout's rank equals the shape's or is one less, and `.index` keeps the parent's layout
  (S28), so a rank-3 layout fails at `.index` (`[2, 16, 64]`: "rank must be equal to or one less than the shape size.
  Got 3 and 2"). The Qwen2 router allocates `[STAGES, BM, BK]` with the rank-2 padded layout (S18), DeepSeek-V2 a `[2,
  n]` word buffer with a rank-1 layout (S21).
- **One buffer, several views**: `_reinterpret(dtype, shape, layout)` (27 directories call it), `.permute`, `.reshape`,
  `.slice` and `.index` change the view, not the bytes; §6.4's flat alias and §6.3's operand view are such views.
- **Swizzle parameters**: of the 212 source occurrences whose `max_phase` expression is not the literal `1`, 185 have `per_phase` 1 (it repeats each phase
  over that many rows) and `vec` 16, 8 or 4, a 16-byte chunk of fp8, bf16 or fp32 (not matched to each buffer's type); a
  `max_phase` beyond the 16-byte chunks in a row adds no phase, the XOR wrapping at the row's chunk count. Triton's own
  swizzle for a `tl.dot` operand at 1 stage, 4 waves, bf16/fp8 and 16-/32-wide MFMA matched
  `per_phase = max(1, 256 // row_bytes)`, `max_phase = max(1, min(16 // per_phase, row_bytes // 16))`
  for all 72 operands at 32–512-byte rows. But `vec` equalled operand `kWidth`, not universally 16 bytes: 64/72 used
  16 bytes; eight used 8 bytes (bf16 `kWidth=4`: `(4,8,2)` / `(4,2,8)`; fp8 `kWidth=8`: `(8,8,2)` / `(8,4,4)`).
  At full kWidth, 128-byte rows give `(8,2,8)` bf16 / `(16,2,8)` fp8, 64-byte rows `(8,4,4)` / `(16,4,4)`
  *(compiled, E13.06)*; the n7168 K halves use `(16,4,4)` (S54). §5 compares helper padding and swizzle only on
  bf16 64×64, 128-byte-row operand reads. Other widths/workloads and reasons for helper refusal were not timed/tested.
- **`gl.bank_conflicts(distributed_type, shared_type)` aborts the compiler process on gfx950** for swizzled, linear and
  padded layouts: `toLinearLayout` has no padded case ("unknown layout"), for the others an input-dimension lookup
  asserts ("inDim not found in layout"), per the source the AMD branch's `vector` dimension (S29). §5 times bf16 64×64
  operand reads, not arbitrary widths; counters with rocprofv3 (`../../patterns/lds_bank_conflicts.md`) were not tried.
## 5. Shared layouts and bank conflicts, timed
One wave per program stored a 64×64 bf16 tile (128-byte rows) into LDS and read it 2048 times as an MFMA operand
(`AMDMFMALayout(4, [16, 16, 32], True, [1, 1])`, `k_width=8`): operand 0 with `ds_read_b128`, operand 1 (B of a
row-major [K, N] tile) with the transposing `ds_read_b64_tr_b16`. Over four runs, with a cheap XOR between reads the
layout changed the time 1.70–1.77× *(field, E1)*: LDS reads set it. With fp32 converts and adds the loop ran 1.54–1.71×
slower and the layout changed it 1.10–1.17× *(field, E2)*: ALU work mostly sets it. Medians, range over the four runs:
| Shared layout | LDS-bound, operand 0 / 1 | ALU-bound, operand 0 / 1 |
|---|---|---|
| `SwizzledSharedLayout(1, 1, 1, [1, 0])`, plain row-major | 907–916 / 894–909 µs | 948–958 / 947–965 µs |
| `SwizzledSharedLayout(8, 1, 8, [1, 0])`, XOR | 529–534 / 510–515 µs | 821–832 / 861–871 µs |
| `PaddedSharedLayout.with_identity_for([[64, 8]], [64, 64], [1, 0])`, 16 bytes per row | 631–642 / 515–520 µs | 845–857 / 860–878 µs |
| the padded helper's layout (§4), `[[512, 16]]` | 525–530 / 396–398 µs | not reported |
- A plain tile whose rows are 128 bytes apart reads at 0.57–0.59 of the swizzled rate when LDS reads dominate
  *(field, E1)*: the lanes of one read share banks. Of plain, XOR and row padding the XOR swizzle was fastest for both
  operands; the padding matched it on the transposed operand-1 read and left operand 0 at 0.82–0.84 of its rate. The
  helper's layout read operand 0 at the swizzle's time (0.98–1.00×) and operand 1 in 0.77–0.78× of it *(field, E1)*.
- With ALU work the plain tile loses 12–14% of the rate on operand 0 and 9–10% on operand 1; the swizzle beat row
  padding by 2–4% on operand 0 in every run (inside single rounds' 11–13% spread, but of one sign) and operand-1 median times differed by
  −0.4% to +0.8% *(field, E2)*. In this bf16 64×64, 128-byte-row toy, both operand reads gained 1.70–1.77× with
  the swizzle in the LDS-bound loop, 1.10–1.17× with ALU work *(field, E1, E2)*; other sizes need A/B. On
  the one kernel Point tried, flipping `BANK_XOR`, one XOR basis of the m=16 fp8 GEMM's `SharedLinearLayout` (S30),
  moved its speedup by −0.5% to +1.4% over five A/Bs *(field, E4)*; whether its reads conflict was not measured.
## 6. Choosing a layout for each role
### 6.1 Global loads and stores: 16 bytes per lane, and the contiguity hints
- Give each lane 16 contiguous bytes along the contiguous dimension: `size_per_thread` `[1, 16]` fp8, `[1, 8]` bf16,
  `[1, 4]` fp32 compile to `dwordx4` loads; on the MI355X a 1 GiB bf16 read with `[1, 8]` per lane ran 1.36× the
  bandwidth of `[1, 1]` (1.28–1.46× over two timings of each) *(field, E10)*. Small scale tables load 4 bytes per lane instead (S20).
- The width a load gets is the layout's contiguous run, cut down to what axis analysis proves about the offsets.
  Across three independent compilations of a 64×64 bf16 tile in `BlockedLayout([1,8],[8,8],[4,1],[1,0])`, `c^8` and
  `c^(r%8*8)` gave 16 `global_load_ushort`; two `//3`/`%3` identity recombinations gave mixed narrow loads.
  Legal `gl.max_contiguous(gl.multiple_of(offs,[1,8]),[1,8])` restored 2 target `global_load_dwordx4` in all 11 forms.
  `//8`, `//16` recombinations and `(c+8)%64` already kept 2 dwordx4; division/modulo do not inherently hide runs
  *(compiled, E14)*. Hints promise actual contiguity and alignment for every Point; masks: `memory.md` §5,
  direct-to-LDS copies: §6.4.

### 6.2 The accumulator

Take `AMDMFMALayout(version=4, instr_shape=..., transposed=True, warps_per_cta=...)` with waves along the large
dimension and the small operand replicated (§3); keep the accumulator in that layout through scaling, bias and
activation (index the per-row and per-column factors in `SliceLayout(1, mma)` and `SliceLayout(0, mma)`), and leave it
only for the store (§6.6) or for the next MFMA (§3, §7). Waves on M can make a `k_width=4` A conversion trivial,
but the measured chaining counterexample in §3 rules out a universal speed recommendation; reductions: §6.7.

### 6.3 Operands straight from global memory into the operand layout

Four ways AKLR's kernels load an operand from global memory with no conversion before the MFMA:

| Way | In AKLR | What it rests on |
|---|---|---|
| 1. Index in the slices of the operand layout | MiniMax-M2 E64 builds its weight indices in `SliceLayout(0, bd)` and `SliceLayout(1, bd)` and passes the loaded `b` to `mfma_scaled` as it is (S20) | the weight is preshuffled in 16×16 blocks (`n // 16 * KB * 16 + k // 16 * 256 + n % 16 * 16 + k % 16`, bytes, `KB` the weight row's length in bytes), so the 64 lanes × 16 bytes one wave loads for B are one contiguous kilobyte (derived from the offset and the operand layout): the operand layout is also the coalesced one. For §9's bf16 B operand (`k_width=8`), row-major reads sixteen 64-byte runs per wave, preshuffled one 1 KiB run; the older `b_load` has 128-byte runs. BK64/4-wave tile16 F0/PS time at BN64/256 was 1.049/1.267 with a 256 MiB flushed weight and 1.469/1.585 with a 2 MiB reused weight; these are complete variants, not bandwidth counters (E11.07) |
| 2. Load in the operand's linear form, then assert | MiniMax-M2 E256 has every wave load the whole replicated A tile and the weight in `gl.to_linear_layout(ad, (BM, BK // 2))` and `gl.to_linear_layout(bd, (BK // 2, BN))`, then `gl.convert_layout(a, ad, assert_trivial=True)` and the same for `b` (S12) | compiled, the assert passes with no conversion instruction |
| 3. A BlockedLayout twin: a blocked layout with the operand's ownership, closed by `assert_trivial` | the DeepSeek-V2 router projection loads B in `BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])` for `DotOperandLayout(1, mma, 8)` of a one-wave `[16, 16, 32]` MFMA (S47; compiled on a `[32, 16]` tile, no conversion instruction); DSA's RoPE keys are another (`../attention.md` §5) | a twin holds only for the shape and MFMA options it was checked for: compiled, `BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0])` is `DotOperandLayout(1, mma, 16)` of a `[16, 16, 128]` MFMA on `[128, 256]` with `warps_per_cta=[1, 4]` at `tiles_per_warp=[1, 1]` and fails `assert_trivial` at `[1, 2]`. Prefer 1 or 2 |
| 4. Preshuffled bytes from LDS through a `SharedLinearLayout` view | DeepSeek-V2 writes raw weight words into a flat uint32 buffer, by async copy when `PIPE` (`UP and TM <= 64`) and by `buffer_load` + `.store` otherwise, and loads them in `bd` as the B operand through `_packed_b_layout` (S21) | the view's offset bits are K bytes 1–8 (bits 0–3), N 1–8 (4–7), K + 16 (8), then the K blocks from 32 and the N blocks from 16 (evaluated at four tile shapes): the byte order of way 1's 16×16 shuffle |

### 6.4 Async copies into LDS: a copy layout derived from the destination

A direct-to-LDS load writes a wave's lanes one after another (`../../hardware/gfx950/memory_and_sync.md` §3), in Triton
3.8.0 16 or 4 bytes per lane (8, 2 or 1 do not compile; S31), so the copy layout must hand each lane the bytes landing
at its slot. Two ways need no extra instruction (copy, waits, destinations, pipelining: `memory.md` §8, §11, §9, §12):

- **Padded destination, copy layout from its bases.** Take the helper's padded layout and read the copy layout off its
  `offset_bases`: the first bases are one lane's 16 bytes, the next 6 the 64 lanes, the next log2(num_warps) the waves,
  the rest registers (S18). Compiled at 4 waves, each wave instruction fills one 1 KiB interval, with no `ds_bpermute`
  or `ds_write` (bf16 64×64: 2 `buffer_load_dwordx4 … lds` per wave), and a GEMM through it matched torch:

  ```python
  @constexpr_function
  def _copy_layout(shared, num_warps):
      """Map 128-bit async copies onto the operand's padded LDS layout."""
      bases = shared.offset_bases
      wave_end = 9 + num_warps.bit_length() - 1
      return gl.DistributedLinearLayout(
          bases[:3] + bases[wave_end:], bases[3:9], bases[9:wave_end], [], shared.shape,
      )
  ```

  `bases[:3]` is 8 bf16 elements; the fp8 GEMMs take `bases[:4]`, `bases[4:10]`, `bases[10:12]` (S27). The decorator
  is `triton.runtime.jit.constexpr_function`, which `gluon.constexpr_function` re-exports. Call it in the jit body,
  `load_a: gl.constexpr = _copy_layout(shared_a, gl.num_warps())`, build the copy's offsets in its slices and pass
  them to `buffer_load_to_shared(a_buffer.index(stage), hidden, offsets)` (S18). The tile must give every wave a whole
  1 KiB interval (rows × columns × element bytes ≥ 1024 × `num_warps`): compiled, a bf16 32×64 tile works at 4 waves
  and at 8 fails with "Layout has 4 warps per CTA, but the context requires 8 warps per CTA".
- **Flat alias, XOR on the global side.** Copy into a `max_phase=1` view of the swizzled buffer and permute the source's
  16-byte chunks; the consumer reads through the swizzled layout, XOR being its own inverse (S11, S32; excerpt and cost:
  `memory.md` §9). The XOR must be the consumer's: `SwizzledSharedLayout(vec, per_phase, max_phase, [1, 0])` keeps
  logical column `c` of row `r` at physical column `c ^ (r // per_phase % max_phase * vec)`, modulo the row length
  (S24), so the copy filling physical column `p` reads global column `p ^ (r // per_phase % max_phase * vec)`: DSA's fp8
  rows under `(16, 1, WIDTH // 16)` read `columns ^ rows % (WIDTH // 16) * 16` (S11), bf16 under `(8, 1, 8)` `columns ^
  rows % 8 * 8`. Nothing checks it: for this bf16 64×64 tile, `rows % 4` for `rows % 8` compiles silently; the source/consumer XOR maps 2048 of 4096 coordinates to the wrong column (CPU mapping plus gfx950 offline compilation, not a GPU readback).
- **Not this**: an async copy straight into a swizzled layout makes the compiler move each lane's source offset to
  another lane, a `ds_bpermute_b32` per copy instruction (S33): compiled, bf16 64×64 into `(8, 1, 8, [1, 0])` carries 2,
  fp8 64×256 into `(16, 1, 8, [1, 0])` 4, the flat-alias versions none.
- **What does not compile.** A direct-to-LDS copy has no narrower fallback: a copy layout that does not follow a padded
  destination's lane order (the coalesced `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])` into the helper's layout for
  a bf16 64×64 A tile), or flat-alias offsets without their hints, stop the compile with `LLVM Translation failed for
  operation: builtin.unrealized_conversion_cast` and `failed to translate module to LLVM IR` (`memory.md` §5).

### 6.5 Register-staged operands

- For the bf16 64×64, 128-byte-row toy, register staging by `gl.allocate_shared_memory(..., value)` or `.store`
  into XOR or helper padding compiled to 8 `ds_write_b128`; write cost and the initializer route were not timed.
  §5 timed `.store`-staged operand reads *(field, E1)*, not other sizes. MiniMax-M2 E64 stages replicated A in the helper
  layout and reads with `.load(ad)` (S34). The helper returned `None` for fp32; the tested hand padding was
  `PaddedSharedLayout.with_identity_for([[BK, 4]], [BM, BK], [1, 0])` (S12): fp32 64×64 read in 0.36× plain time as A,
  0.95–0.96× as `[K, N]` B *(field, E1)*. This fp32 toy has a different row width; other shapes or swizzles need A/B.
- **A small table read many times: list the reader's register bases first.** `_register_first_shared_layout(layout)` is
  a `SharedLinearLayout` of `layout.reg_bases + layout.lane_bases + layout.warp_bases` without the zero bases, keeping
  each lane's values side by side (S48). Compiled, a `[32, 16]` e8m0 table read in the A-scale layout of a `[16, 16,
  128]` MFMA takes 8 `ds_read_u8` per lane from row-major and 1 `ds_read_b64` from this one; writing it, 4 byte stores
  instead of 1 `ds_write_b32` (not timed). MiniMax-M2 E256 makes it a per-Point option (`SCALE_LDS`, S48).
- Instead of the compiler's conversion you can stage an accumulator yourself in a shared layout of your choice and load
  it in the operand layout: AKLR's GDN prefill moves its fp32 state into operand ownership "through vector-preserving,
  bank-swizzled LDS", a hand-written `SharedLinearLayout` with XOR bases such as `[8, 8]` (S35; `../gdn.md` §12).

### 6.6 The store

Four routes, compiled for a bf16 `[32, 64]` accumulator on `[2, 2]` waves and writing identical values on the MI355X:
convert it, rounded first, to a row-blocked layout of 16 bytes per lane, e.g. `BlockedLayout([1, 8], [8, 8], [4, 1], [1,
0])` (S36), or stage it through LDS yourself in a shared layout you choose and load it back in the store layout, as the
n3072 GEMMs do through `SwizzledSharedLayout(8, 2, 8, [1, 0])` at BLOCK_M = 32 and an XOR `SharedLinearLayout` otherwise
(S27): either gives one `buffer_store_dwordx4` per lane after one LDS round trip; store straight from the transposed
MFMA layout, 8 bytes per lane and no LDS, as the one-wave fp8 GEMMs do (S37); or from a non-transposed one, as the
Qwen2-MoE router at its 2232-row Point (S18), which into a row-major output is 2-byte stores. §9 shows what the
conversion compiles to; the store itself is in `memory.md` §15.

### 6.7 Reductions, gathers and vectors every wave needs

- **Wave-local reduction is shape-dependent**, not a universal speed rule. In 400 dependent fp32 reductions,
  2048 programs, 5 tiles from 16×64 to 128×128, 4/8 waves, sum/max and both axes (40 combinations), choosing each
  arm's lane split on clusters 1–4 and testing on 5–8 gave local/cross-wave = 0.813 at 16×64/4-wave row sum but
  1.298 at 32×256/8-wave axis-0 sum *(field, E11)*. `gl.sum`, `gl.max` and `gl.reduce` return `SliceLayout(axis,L)`:
  register reduction, then in-wave lanes, then LDS plus `s_barrier` if the axis spans waves (S49). The old 16×64 row
  sum compiled to 5 DPP instructions in `BlockedLayout([1,4],[4,16],[4,1],[1,0])`; `([1,4],[16,4],[1,4],[1,0])`
  added 256 B LDS, a write/read/barrier. GDN decode (`../gdn.md` §4) and gated attention (`../attention.md` §9) use
  wave-local axes; this does not establish a universal performance advantage.
- **MFMA row reductions:** a one-wave 16×64 `transposed=True` row max compiled to in-lane work and one each
  `v_permlane32_swap` / `v_permlane16_swap`; `False` to 20 DPP instructions. Converting DSA scores once to
  `BlockedLayout([1,4],[2,32],[4,1],[1,0])` (S11) reduced the old 8×128 max/sum example from 3 to 1 barriers
  (128 B to 4 KB LDS). But 256-step online-softmax probes, 1024 programs, gave converted/in-place time
  0.875 at the actual DSA 8×64 shape, 0.849 at 8×128, 1.200 at 16×128, 1.188 at 16×256; pacc agreed and
  fp64 max/denominator relative error was ≤2.86e-6 *(field, E11)*. Do not infer speed from barrier counts.
- **Replicate a small vector in every wave on purpose** so its reductions never leave the wave: MiniMax-M2 FP8 keeps 256
  expert scores in `SliceLayout(0, BlockedLayout([1, 1], [1, 64], [WARPS, 1], [1, 0]))` (S50); GDN decode normalizes a
  128-value row in `BlockedLayout([4], [64], [2], [0])`, "The 128-element reduction stays wave-local and needs no LDS
  exchange" (S51). Copies can repeat loads and ALU work without LDS; once-only stores/atomics are instance-limited (§2),
  but a copy's global load is not ordered with another copy's store to the same data: do not assume a barrier between
  the load and the selected store, or that every replicated load survives LLVM dead-copy elimination (`../gdn.md` §3 shows the wrong state). MiniMax-M2 FP8's in-CTA
  split-K sums over waves: one K slice per wave in a rank-3 `[SK, 1, 1]` MFMA layout, then `gl.sum(acc, 0)` (S52).
- **`gl.gather(src, index, axis)` returns the index's shape and layout** (S4): build the index in the layout the result
  must land in, as MiniMax-M2 picks column `slot` of a register tile with `gl.full((ROWS, 1), slot, gl.int32, layout)`
  as the index, then `gl.sum(..., 1)` (S22). It stays in the wave when index and source share a layout with the gather
  axis off the waves, at one index shuffle (§7) per index register and per source register a lane holds along the axis
  (a runtime index), else it goes through LDS (S39). Compiled, a `[64, 64]` fp32 gather along axis 1 with both tensors
  in `BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])` is 16 `ds_bpermute_b32` per lane, in `([1, 4], [4, 16], [4, 1],
  [1, 0])` 64, in `([1, 4], [1, 64], [4, 1], [1, 0])`, four times the tensor's width with copies that all gather, 256;
  with the axis over the waves, or the index's rows on other lanes and waves than the source's, 16 KB of LDS, 8–9
  writes, 16 `ds_read_b32` and an `s_barrier`.

## 7. `convert_layout`: what it lowers to, and how to avoid it

Triton 3.8.0 lowers the minimal conversion between the two linear layouts by what it moves (S38, S39, S40):

| The conversion moves | Lowered to on gfx950 | Visible in the ISA as |
|---|---|---|
| nothing (equivalent layouts) | no code | nothing |
| registers within a lane | a renaming for 32-bit data; for 8-bit data, a renaming or byte permutes when packed bytes must move | nothing, or `v_perm_b32` |
| one register bit with lane bit 4 or 5 | a permlane swap | `v_permlane16_swap_b32` / `v_permlane32_swap_b32` |
| lanes exchanged by an xor of lane bits, with fewer than two register–lane transpositions (a register bit and a lane bit that trade places between the two layouts) | xor shuffles: DPP for lane bits 0–3, a permlane swap for bits 4 and 5 | `v_mov_b32_dpp`, `v_permlane16_swap_b32` / `v_permlane32_swap_b32` |
| lanes in any other order, with fewer than two transpositions | index shuffles | `ds_bpermute_b32`, no LDS allocated |
| lanes with two or more transpositions, or any wave | an LDS round trip, with an `s_barrier` between stores and loads when waves move (none in a 1-wave kernel) | `ds_write_*` + `ds_read_*`, `s_barrier`, `shared` > 0 |

The rows classify movement, not universal cost. In a 64×64 int32 A → B → A probe, 1024 programs, 1023 iterations,
control-subtracted DPP/permlane16/permlane32 cost 0.0788/0.0730/0.0730 ns per moved 32-bit value, below a different
4-wave LDS b128 conversion's 0.1273; `ds_bpermute` cost 0.1654 *(field, E11)*. These are different conversions,
not isolated instruction throughput. The older E3 index-shuffle flip was slower than its LDS b128 example.

- **Make it trivial and assert it.** `convert_layout(x, L, assert_trivial=True)` raises a `TypeError` that prints both
  linear layouts unless the conversion moves data only within a lane (S29); 46 directories assert 182 conversions with a
  literal `True`, one more with `M > 128`. Trivial is a renaming of registers for 32-bit data; for 8-bit data it may require byte permutes
  when packed bytes must move. MFMA layouts that differ only in `instr_shape[2]` own the same elements: a bf16 `[16, 16, 32]` result
  converts to an fp32 `[16, 16, 4]` accumulator for free (`mfma.md` §6).
- **Relabel K instead of moving data.** A product is unchanged if A's columns and B's rows are permuted the same way.
  With fp32 `k_width=1` operands, AKLR's GDN prefill permutes K inside every 16-block (`k = 4y + z → 4z + y`, a
  `reshape`/`permute`/`reshape` of each operand) so the transposed accumulator becomes the A operand with no data
  movement, asserted (S41); the code and its compiled cost are in `../gdn.md` §11.
- **Two permlane steps can avoid LDS, but speed is instance-dependent.** For the 16×16 fp32 accumulator →
  `DotOperandLayout(0,mma,1)`, two register/lane transpositions go through LDS; the middle `DistributedLinearLayout`
  compiles to 2 `v_permlane32_swap` + 2 `v_permlane16_swap`, no LDS, with or without the empty asm (S42,
  `../gdn.md` §11). In 256 dependent conversion+MFMA steps, two-step/direct = 0.892 / 1.259 for one-wave 16×16
  at 4 / 16 programs per CU, 1.137 / 1.212 for four-wave 64×16 at the same grids (fp64 error ≤1.6e-6)
  *(field, E11)*. This does not make two steps a general replacement for the direct conversion.
- **At a required bf16 rounding boundary, round first, then redistribute**: this halves the transferred data versus fp32 in the checked LDS exchanges (S43, S44); do not add a bf16 rounding where the consumer requires fp32.
- **Keep the split bit in registers** (§3 `tiles_per_warp`): `gl.split` of a `BlockedLayout([1, 1], [32, 2], [4, 1], [1,
  0])` tile, its size-2 axis on lanes, is rejected; with that axis on a register bit (a `DistributedLinearLayout` here)
  it moves nothing. Qwen3-Next's GDN decode writes one where a BlockedLayout cannot (S45). To learn the layout a chain
  of `reshape`/`permute`/`split` produces, apply it to a zero tensor of the same layout and index in the result's slices
  (S16). `gl.join(a, b)` adds its size-2 axis as a register bit (S5), so stacking two tiles moves nothing: the m=16 fp8
  GEMM joins two rounded accumulators and permutes and reshapes them into one stripe (S26).
- **`gl.amd.slice(x, shape, offsets)`** (21 directories) takes a sub-tile without moving anything when the slice keeps
  all of the layout's lane and wave bases: each dimension at least the extent one round of lanes and waves spans
  (`size_per_thread × threads_per_warp × warps_per_cta` for a BlockedLayout), and `offsets` multiples of it (S46).
  Otherwise the compile stops with "Lane and warp dim basis must match between source and destination layout." or
  "No source register holds the element for destination index […]". Compiled with Triton 3.8.0 for gfx950, on a
  64×64 `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])` a 32×64 slice at row 0 or 32 compiles; 16×64, 64×32, and
  32×64 at row 16 do not. AKLR takes 16 rows of a 32-row padded tile this way (S26).
- **Do not chain conversions**: Gluon does not fold them (§1). Convert once, to the layout of the consumer.

## 8. What a conversion costs, and how to read its path from the ISA

1024 programs (4 per CU) converted a 64×64 32-bit tile (16 KB) A → B → A 1024 times *(field, E3)*; the ISA column is one
loop iteration, compiled with Triton 3.8.0 for gfx950 (outside a loop the 16 KB LDS row has 3 `s_barrier`, not 4):

| A → B | Waves | ISA per A → B → A iteration | LDS | ns per conversion, 3 runs |
|---|---|---|---|---|
| `([1,4],[8,8],[4,1],[1,0])` → itself | 4 | none | 0 | 73–74 (the loop alone) |
| same → `order=[0, 1]` | 4 | 32 `ds_bpermute_b32` (16 per conversion) | 0 | 661–663 |
| same → `([4,1],[8,8],[1,4],[0,1])` | 4 | 8 `ds_write_b128` + 8 `ds_read_b128` + 4 `s_barrier` | 16 KB | 574–581 |
| `([1,4],[4,16],[1,1],[1,0])` → `([4,1],[16,4],[1,1],[0,1])` | 1 | 128 `ds_write_b32` + 128 `ds_read_b32` | 16 KB | 1518–1522 |
| `([1,4],[8,8],[4,1],[1,0])` → `([1,4],[8,8],[1,4],[1,0])` (B is 128 columns wide: waves 2–3 hold copies, 32 values per lane) | 4 | 12 `ds_write_b128` + 12 `ds_read_b128` + 8 `s_barrier` | 8 KB | 1058–1060 |

- With 4 programs per CU converting at once, a conversion that moves a 16 KB tile between lanes took 0.5–1.45 µs per
  step beyond the 73–74 ns loop *(field, E3)*, measured on a probe, not on a kernel. The `ds_bpermute` flip was not
  cheaper than a vectorized trip through LDS; the 1-wave case (one wave doing the whole tile, 4 bytes at a time, with a
  quarter of the waves per CU) was slowest in each old run. The 8-token Qwen2-MoE router variants observed
  8.447 → 8.45 µs, four internal alternated rounds per arm, not four independent runs *(field, E6)*. The cause
  was not measured: 24 writes and 32 reads were once per program, no K loop, no `s_barrier` in the one-wave CTA.
  New BF16 tile16 BK64/4-wave probes gave CB/PS = 1.002 / 1.496 at BN64 / 256 with 256 MiB flushed weights,
  0.997 / 1.295 with 2 MiB reused weights *(field, E11)*. CB/PS also changes packing/coalescing, so these
  cannot isolate conversion cost or show memory waits hiding it. LDS-/latency-bound prioritization remains unverified.
- **Read the path**: compile the kernel (`../debugging.md` §1) and count `ds_bpermute_b32`, `v_mov_b32_dpp`,
  `v_permlane16_swap`/`v_permlane32_swap`, `ds_write_*`/`ds_read_*` with `s_barrier`, and `v_perm_b32` in its amdgcn;
  the TTGIR keeps one `ttg.convert_layout` per conversion, `metadata.shared` the LDS bytes. Your own LDS buffers,
  reductions, gathers, swizzled async copies (§6.4) and inline assembly emit the same instructions (of AKLR's 897
  compiled kernels, `ds_bpermute_b32` appears in 291, `v_permlane16_swap` in 129, `v_permlane32_swap` in 113,
  `v_perm_b32` in 215) *(field, E5)*: attribute them by compiling once with the conversion replaced by a load in the
  target layout (on §9's `tile16` that removes exactly `CONVERT_A`'s 1 `ds_write_b128`, 2 `ds_read_b128` and 2
  `s_barrier` per K step) or asserting it trivial (the compile then stops: "is not trivial").

## 9. Worked example: a 16×N bf16 tile

`y[16, N] = x[16, K] @ w[N, K]ᵀ`, 4 waves side by side along N, BK = 64; this unmasked example requires K to be a multiple of BK, N to be a power of two, and complete contiguous input/output storage. The flags choose between loading in a
coalesced layout and converting, or loading in the operand layout's slices (compiles with Triton 3.8.0 for gfx950):

```python
@gluon.jit
def tile16(x_ptr, w_ptr, y_ptr, K: gl.constexpr, N: gl.constexpr, BK: gl.constexpr,
           CONVERT_A: gl.constexpr, CONVERT_B: gl.constexpr, CONVERT_C: gl.constexpr):
    W: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                             warps_per_cta=[1, W])
    a_dot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    b_dot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [W, 1], [1, 0])    # 16 B per lane along K
    b_load: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, W], [0, 1])    # 16 B per lane along K
    c_store: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [W, 1], [1, 0])  # 16 B per lane along N
    acc = gl.zeros([16, N], gl.float32, mma)
    for k0 in range(0, K, BK):
        if CONVERT_A:   # coalesced load, then a layout conversion
            m = gl.arange(0, 16, gl.SliceLayout(1, a_load))
            ka = k0 + gl.arange(0, BK, gl.SliceLayout(0, a_load))
            a = gl.convert_layout(gl.load(x_ptr + m[:, None] * K + ka[None, :]), a_dot)
        else:           # indices in the slices of the operand layout: the load lands in a_dot
            m = gl.arange(0, 16, gl.SliceLayout(1, a_dot))
            ka = k0 + gl.arange(0, BK, gl.SliceLayout(0, a_dot))
            a = gl.load(x_ptr + m[:, None] * K + ka[None, :])
        if CONVERT_B:
            kb = k0 + gl.arange(0, BK, gl.SliceLayout(1, b_load))
            n = gl.arange(0, N, gl.SliceLayout(0, b_load))
            b = gl.convert_layout(gl.load(w_ptr + n[None, :] * K + kb[:, None]), b_dot)
        else:
            kb = k0 + gl.arange(0, BK, gl.SliceLayout(1, b_dot))
            n = gl.arange(0, N, gl.SliceLayout(0, b_dot))
            b = gl.load(w_ptr + n[None, :] * K + kb[:, None])
        acc = gl.amd.cdna4.mfma(a, b, acc)
    out = acc.to(gl.bfloat16)
    if CONVERT_C:       # 16-byte stores after an exchange
        out = gl.convert_layout(out, c_store)
        m = gl.arange(0, 16, gl.SliceLayout(1, c_store))
        n = gl.arange(0, N, gl.SliceLayout(0, c_store))
    else:               # store from the accumulator: 4 consecutive columns (8 B) per lane
        m = gl.arange(0, 16, gl.SliceLayout(1, mma))
        n = gl.arange(0, N, gl.SliceLayout(0, mma))
    gl.store(y_ptr + m[:, None] * N + n[None, :], out)
```

Compiled with K = 2048, N = 256, `num_warps=4`, pointers 16-byte aligned; instructions per wave and K step (8 MFMAs):

| Variant | Global loads per K step | LDS per K step | LDS scratch | VGPRs |
|---|---|---|---|---|
| all flags 0 | 10 `global_load_dwordx4` | none | 0 | 74 |
| `CONVERT_A=1` | 9 | 1 `ds_write_b128`, 2 `ds_read_b128`, 2 `s_barrier` | 2 KB | 76 |
| `CONVERT_B=1` | 10 | 8 `ds_write_b128`, 8 `ds_read_b128`, 8 `s_barrier` | 8 KB | 86 |
| `CONVERT_C=1` (once, at the end) | — | 2 `ds_write_b128`, 2 `ds_read_b128`, 1 `s_barrier`; 2 `global_store_dwordx4` instead of 4 `global_store_dwordx2` | 8 KB | 74 |

Loading in the operand layout keeps 16-byte loads and removes every LDS round trip and barrier from the loop; what it
gives up is coalescing of the weight (64-byte runs per row instead of 128), which a 16×16 preshuffled weight restores
(§6.3). The historical table variants were not timed here. New complete variants at BN64 with a 256 MiB flushed
weight gave CB/F0 = 0.957 (diagnostic t95 0.9545–0.9593), despite CB's 6 LDS/barrier instructions per K step versus
F0's 0 *(field, E11)*. Packing/input layout also change; instruction counts cannot isolate conversion cost.

**Unverified scope retained:** universal once-only stores/atomics and LLVM dead-copy elimination conditions (E12);
all-width/workload helper-versus-swizzle selection, helper-refusal causes, register-staging write/initializer timing and
rocprofv3 bank counters (E1/E2, `bench_lds_bank_conflicts`); general LDS-/latency-bound conversion priority and memory-wait
hiding mechanism (E11.06); other dtypes/stages >2/BK <64/warp counts for pipeliner equality and uint8/kWidth4 read opcodes
(E13). New d_compile checks did not retest bf16 4×64 or invalid orientations; earlier source/compile facts remain separate.

## Evidence

- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 full GPU run with 8 consecutive balanced repeat clusters, 9–16 complete blocks/cluster × 5 launches/arm; clusters are not independent full runs] case triton/checks/s_cross_wave_movement_cost.py checks .01–.07; 340 kernels compiled, all seven timing checks ran, .01–.06 FAIL provide valid counterexamples, .07 PASS is instance-limited. .01 selects lane splits on first 4 clusters and tests last 4 (reverse holdout confirms direction); 5/40 reduction A/A diagnostic intervals exclude 1, so borderline configurations are not resolved. All cluster t95 intervals are diagnostic, without established time independence or simultaneous coverage. Actual final log: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_cross_wave_movement_cost_r2.log; .01 40 reduction combinations; .02 four online-softmax shapes; .03 control-subtracted conversion costs; .04 four two-step chains; .05 waves-on-M/N chains; .06/.07 complete tile16 variants (BM16, BK64, BN64/256, 4 waves; flushed: rows16/N32768/K4096, 256 MiB weight; reused: rows1024/N1024/K1024, 2 MiB weight).
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 final GPU run; deterministic counts/optimized-LLVM address enumeration, not timing] case triton/checks/s_replicated_layout_store_atomic_once.py .01, store_issues, store_checker_control, .02; 504 GPU cells, 48 register-copy failures, 153 store-issue cells with 8 register-copy buffer-store failures; exact 16×64 wave-copy checks support the limited §2 instance. Actual combine M4096/H3072/TOPK8 compiled offline and through AKLR launcher (two artifacts of the same instance, not two independent full runs): bit7-set reaches no data memory, bit7-clear reaches 9 loads + 1 store, kernarg loads=2. Final log: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_replicated_layout_store_atomic_once_r2.log.
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 full run, deterministic compile/readback, no timing] case triton/checks/s_padded_layout_dma_copy.py .01/.01_isa: all 603 helper layouts meeting tile_bytes >= 1024×num_warps have exact dwordx4 LDS counts, no bpermute/write/other LDS loads, bitwise readback; 167 smaller helper layouts fail compile (the bound is necessary, not a sufficient condition for the helper to return a layout). Sweep bf16 kWidth8 / fp8e4m3 and uint8 kWidth16, A/B contiguous and B [K,N], 16-/32-wide MFMA, 1/2/4/8/16 waves, 16–256 rows × 64/128 columns; page bf16 recipe equals generalized recipe on 229 layouts. .04: 40 pipeliner builds, 74 padded matches/6 swizzled exceptions; .05: 48 compile-only reads, K-contiguous b128 or B [K,N] transposing b16/b8; .06: 72 compile-only swizzles, 8 counterexamples to universal 16-byte vec. Log: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_padded_layout_dma_copy.log.
- E14 [2026-11, Triton 3.8.0 for gfx950, compile-only, N=3 independent compilation sessions, no GPU performance claim] triton/checks/s_constexpr_folding_and_load_width.py .04: 11 genuinely aligned 8-element-run forms; Python verifies the run/alignment promises. Logs: local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_constexpr_folding_and_load_width_r1.log, local:/tmp/kbmig/checks_study/s_constexpr_folding_and_load_width.audit_local.log, local:/tmp/kbmig/checks_study/s_constexpr_folding_and_load_width.audit_fixed_local.log; f1/f2 XOR narrowed, f9/f10 //3/%3 narrowed, all legal hints restore 2 target dwordx4; loaded-index f8 retains its extra dword index load.
- E15 [2026-11, Triton 3.8.0 for gfx950, N=1 CPU-only compilation session, no GPU claims] triton/checks/d_compile_boundary_controls.py .06 checks fp16 [4,64,64], kWidth4, N64/128 (16/32 MFMAs); .08 checks seven larger conversion builds with LDS only for >=2 transpositions or warp movement, retaining the transposed [32,32,2] kWidth1 counterexample (one transposition, permlane32, no LDS). These limited checks do not retest every orientation, universal scratch-round synchronization or compiler swizzle. Final log: local:/tmp/kbmig/checks_study/local_logs/d_compile_boundary_controls.local.log.
- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, case bench_lds_bank_conflicts: torch events, median of 7 rounds x 3 launches, 2048 one-wave programs, 2048 reads each] a 64x64 bf16 LDS tile read as an MFMA operand, int16 XOR between reads, range over the four runs (operand 0 / 1): plain 906.8-915.6 / 894.0-908.6 µs, SwizzledSharedLayout(8, 1, 8) 529.0-533.9 / 510.5-514.8 µs, 16-byte row padding [[64, 8]] 630.7-642.0 / 515.5-519.7 µs, the helper's padded layout 524.6-529.5 / 395.6-397.8 µs; plain/XOR 1.701-1.731 / 1.748-1.765, XOR/plain 0.578-0.588 / 0.567-0.572, padding/XOR 1.189-1.214 / 1.004-1.016 (XOR/padding 0.824-0.841 / 0.985-0.996), helper/XOR 0.984-1.001 / 0.769-0.777, helper/plain 0.575-0.580 / 0.435-0.443; XOR at most 0.4% off the fastest of plain, XOR and padding in both loops; gl.bank_conflicts aborted (exit -6) for XOR and padded in every run; the four stagings (XOR and helper layout, by value and by .store) 8 ds_write_b128 each, compiled only; fp32 64x64, hand padding [[64, 4]] over plain 0.363-0.365 / 0.950-0.957 and the helper None for fp32 in every run (.16 FAILs in all four runs on its own criterion, <= 0.8 on both operands) (case triton/checks/bench_lds_bank_conflicts.py (.01, .04, .05, .08-.10, .12, .13, .15, .16); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_lds_bank_conflicts.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_lds_bank_conflicts_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_lds_bank_conflicts_rep2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_lds_bank_conflicts_rep3.log)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, case bench_lds_bank_conflicts: torch events, median of 7 rounds x 3 launches, 2048 one-wave programs] the same reads summed in fp32, range over the four runs (operand 0 / 1): plain 947.9-957.7 / 947.3-965.3 µs, SwizzledSharedLayout(8, 1, 8) 821.0-831.8 / 861.3-871.2 µs, padding 845.5-857.4 / 860.4-877.6 µs; plain/XOR 1.142-1.167 / 1.097-1.109, XOR/plain 0.857-0.876 / 0.902-0.912, XOR fp32 over XOR int16 1.538-1.565 / 1.683-1.707; padding/XOR 1.018-1.036 / 0.996-1.008 (padding slower on operand 0 in every run), |XOR - padding| 14.7-29.7 / 0.9-6.9 µs against the rounds' min-max spread of 95.7-111.0 / 102.1-110.5 µs (11.5-13.4% / 11.9-12.7% of the XOR median) (case triton/checks/bench_lds_bank_conflicts.py (.07, .08, .11-.14); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_lds_bank_conflicts.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_lds_bank_conflicts_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_lds_bank_conflicts_rep2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_lds_bank_conflicts_rep3.log)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, case bench_convert_layout_cost: torch events, median of 7 rounds x 3 launches, 1024 programs of 1 or 4 waves] ns per conversion of a 64x64 32-bit tile per program, range over the three runs: control 73.1-74.1, order flip (shuffles) 661.4-662.8, 4-wave LDS b128 573.9-580.8, 1-wave LDS b32 1518-1522, warps [4,1] to [1,4] (B 128 columns wide, waves 2-3 copies) 1058-1060, i.e. 500.8-1449 ns beyond the control; flip over 4-wave LDS 1.141-1.153; 1-wave LDS b32 the slowest in every run, 1.43-1.44x the next; the case's ISA counts are of the whole kernel (in rows 2-3 LLVM unrolled the loop twice and kept a remainder, so the body is in it three times: 96 ds_bpermute_b32 for the flip, 48 b128 LDS ops for the 4-wave LDS row); per-iteration counts and the 3 straight-line s_barrier from case lower_convert_layout_paths (compile only) (case triton/checks/bench_convert_layout_cost.py (.05-.08); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_convert_layout_cost.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_convert_layout_cost_rep2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_convert_layout_cost_rep3.log)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=5, python -m aklr validate --stages audit,bench, one GPU, 4 alternated rounds per arm in each A/B] gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c at m=16, BANK_XOR False to True: in ab_gemm_m16_bankxor kernel 4.245 to 4.263 µs (time ratio 1.004), speedup against the baseline 1.460 to 1.453 (B/A -0.5%); in the four runs of bench_lds_bank_conflicts B/A speedup +0.25%, -0.22%, +1.44% and +0.29% (case triton/checks/ab_gemm_m16_bankxor.py (.05); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gemm_m16_bankxor.log; case triton/checks/bench_lds_bank_conflicts.py (.14); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_lds_bank_conflicts.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_lds_bank_conflicts_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_lds_bank_conflicts_rep2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_lds_bank_conflicts_rep3.log)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, case rt_census_compiled_kernels: every kernel AKLR compiles at its Points, compiled on the GPU node in 4 shards and merged, every opcode counted] of 897 compiled AKLR kernels, ds_bpermute_b32 in 291, v_permlane16_swap in 129, v_permlane32_swap in 113, v_perm_b32 in 215 (check .20 FAILed against the older figures 272 / 81 / 70 / 186, taken from a census that kept each kernel's 60 most frequent opcodes) (case triton/checks/rt_census_compiled_kernels.py (.20); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, python -m aklr validate --stages audit,bench, one GPU, 4 alternated rounds per arm] qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f at n_tokens=8, `ASYNC_HIDDEN = ROWS <= 8` to `ROWS <= 0` (A and B loaded with `gl.load` and converted to the operands; one K step per program at 8 tokens, no loop in either ISA): kernel 8.447 to 8.45 µs (time ratio 1.000), speedup against the baseline 1.186 to 1.178 (B/A 0.993; .05 B/A >= 0.97 PASS); B passes accuracy (max abs error 0.0296 against a reference of absmax 11.67); compiled (.04 PASS): B 24 global_load_dwordx4, 24 ds_write_b128, 32 ds_read_b128, 4 KiB LDS, no s_barrier (one wave: 15 wave-barrier markers), A 8 async copies + 16 buffer_load_dwordx4, 16 ds_read_b128, 8 KiB LDS; .01-.03 FAIL only on their s_barrier condition, which a one-wave CTA cannot meet (case triton/checks/ab_router8_async_staging.py (.01-.05); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_router8_async_staging.log)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, python -m aklr validate --stages audit,bench, one GPU, 4 alternated rounds per arm in each run] rms_norm_hidden_size6144/cf637bff0784c454 (plain Triton), B pads the 6144 row to one 8192 chunk in place of three 2048 chunks: B passes accuracy at all 14 Points in every run; B/A kernel time, in the three runs: 1.024-1.037, 1.022-1.033, 1.028-1.036 at n_tokens 1024-4096; 0.988-0.992, 1.010-1.016, 1.007-1.011 at 3-128; 0.998-1.005, 1.021-1.023, 1.016-1.019 at 256-512 (case triton/checks/ab_misc_rules.py (.04); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_misc_rules.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_ab_misc_rules_rep2.log and local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_ab_misc_rules_rep3.log)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA events, median of 9 rounds x 5 launches, timed twice in the run] 1024 one-wave programs each running 256 dependent 16x16 fp32 accumulator to DotOperandLayout(0, mma, 1) conversions and MFMAs: direct (LDS) 29.98 µs (30.1, 29.8), two permlane steps 26.26 µs (26.5, 26.0), ratio 0.876 (case triton/checks/lower_mfma_acc_chaining.py (.17); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_acc_chaining.log)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA events, median of 7 rounds x 5 launches] 2048 programs of 4 waves, 400 dependent row sums of a 16x64 fp32 tile: BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0]) 67.0 µs, BlockedLayout([1, 4], [16, 4], [1, 4], [1, 0]) 80.7 µs, ratio 0.830 (the check's criterion, <= 0.8, FAILs) (case triton/checks/lower_reductions.py (.10); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_reductions.log)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA events, median of 7 rounds x 5 launches, each arm timed twice] a 1 GiB bf16 read, 4 waves per program: size_per_thread [1, 8] 6373 and 6437 GB/s, [1, 1] 4372 and 5043 GB/s, ratio of the sums 1.36; compiled, [1, 16] u8, [1, 8] bf16 and [1, 4] fp32 load with dwordx4 only, [1, 1] bf16 with 16 ushort (case triton/checks/lower_vector_width_contiguity.py (.05); local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_vector_width_contiguity.log)
- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` runs inliner, `resolve_auto_encodings` (:333; it assigns `AutoLayout` values only, lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp:21-37), SCCP, loop-aware CSE, the Gluon canonicalizer, loop unroll, select/if combine, warp pipeline and warp groups (:332-340); `add_infer_coalesced_encodings` is bound (python/src/passes.cc:153) and called by no AMD pipeline
- S2 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:269 `make_ttgir`, the plain-Triton path, runs coalesce (:269), remove-layout-conversions (:271, :274) and accelerate-matmul (:273); 64 threads per warp for gfx950 (:114), num_warps a power of two (:116-117)
- S3 triton-lang/triton@c01b677:lib/Dialect/Gluon/Transforms/Canonicalize.cpp:53 the Gluon canonicalizer excludes the patterns "that modify layouts, especially `ConvertLayoutOp` patterns"
- S4 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:220 Gluon rejects `can_reorder`; view ops infer their result layout; `arange` without a layout gets AutoLayout (:215; `full` :234); a binary op between two layouts raises "Layout mismatch in broadcast" (:206); `expand_dims` needs a SliceLayout sliced on the expanded axis (:145); `to_linear_layout` returns a DistributedLinearLayout (SharedLinearLayout for a shared layout) constexpr (:383); `gather` (:559): same rank (:566), other extents equal (:571-577), result in the index's shape and layout (:579)
- S5 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3863 a layout's lanes must equal the module's threads per warp, its warps (:3877) the context's num_warps; a dot whose operands are not dot-operand encodings fails ("Dot's a/b's encoding should be of DotOperandEncodingAttr", :3182; "mismatching encoding between A and B operands", :3196), as do unequal kWidths ("mismatching kWidth between A and B operands", :3199); `tryJoinOnAxis` (:697): a join adds a size-2 register basis (:701-704), a split needs one, "requires at least 2 elements per thread" (:726-729); the MFMA operand swizzle `composeSharedLayoutForOperand` (:2652): 64 banks on version 4 (:2681), `perPhase` (:2689), `maxPhase` (:2690-2691)
- S6 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:50 `mfma_scaled` asserts an AMDMFMALayout accumulator and DotOperandLayout operands whose parent is that layout (:51-54); `get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)` returns a DistributedLinearLayout (:94; lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1673); `compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)` (:122), whose binding python/src/gluon_ir.cc:1326 calls `composePaddedLayoutForAsyncCopyCDNA4(..., useAsyncCopy=true, warpSize=64)` (:1366), the function the plain-Triton pipeliner calls (third_party/amd/lib/TritonAMDGPUTransforms/LowerLoops.cpp:286, Utility.cpp:519)
- S7 triton-lang/triton@c01b677:lib/Dialect/Triton/IR/Ops.cpp:326 `DotScaledOp::verify` checks scale shapes only, deducing the scale factor from them ("scale factor must be 16 or 32. Got …", :391)
- S8 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 "TODO: unsupported scan layout" unless `ScanLoweringHelper::isSupported`, which accepts a BlockedLayout only (lib/Analysis/Utility.cpp:526-531)
- S9 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:345 `WARPS = gl.num_warps()` inside a layout (:348)
- S10 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_layouts.py:60 BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[]) (fields :71-75); SliceLayout(dim, parent) (:122), rank = parent rank - 1 (:151); DistributedLinearLayout(reg_bases, lane_bases, warp_bases, block_bases, shape) (:165); DotOperandLayout(operand_index, parent, k_width) (:223), docstring "Number of elements per 32-bits" (:230); `format_tensor_view` and `format_hardware_view` print a layout (:24)
- S11 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:13 module-level `gl.constexpr(gl.amd.AMDMFMALayout(...))` layouts, the `[1, 4]` MFMA (:13-15); consumers `SwizzledSharedLayout(16, 1, 8)` and `(16, 1, 4)` (:16-17) filled through a flat alias with the XOR on the global column and contiguity hints (:77-81); scores converted to `_DIRECT_SOFTMAX` (:18) at :124 before `gl.max` (:126) and `gl.sum` (:132)
- S12 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/147b861bd4c406d2.json#kernel.py:57 combine layout `[1, V] x [1, 64] x [1, WAVES]`, 4 waves at 3072 < m <= 5120 with a 512-column block (:1099-1103); operand loads in `gl.to_linear_layout(ad, ...)` (:145-146), asserted trivial (:289-290); fp32 tiles in `PaddedSharedLayout.with_identity_for([[BK, 4]], ...)` (:951)
- S13 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py:16 AMDMFMALayout(version, instr_shape, transposed, warps_per_cta, element_bitwidth=None, tiles_per_warp=None, cga_layout=[]); version 4 is gfx950 (:34); defaults 32 and all-ones (:53-56); `verify` (:81-89) checks only version, the M x N pair, bitwidth and rank
- S14 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:486 accumulator tile: 4 registers along M (not transposed) or N (transposed), lanes over the other dimension (:473 height 4 for 32-bit, 1 for 64-bit), 64x4 must be transposed (:481); tiles_per_warp[N] registers, then N warps, then remaining N tiles, then tiles_per_warp[M], then M warps (:539-557); operand layout (:749): k_width K-consecutive registers, lanes over the non-K dimension then K (:753), K repeats in registers (:768), tiles_per_warp (:774), then the MFMA's warps (:785)
- S15 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:375 a transposed layout swaps A and B in the instruction; the 4x64 broadcast asserts a non-transposed layout (:360)
- S16 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/362ce3bb6200382e.json#kernel.py:155 `transposed=False` for K-group folding; `_unpack_group` (:108); a zero tensor as layout carrier (:180)
- S17 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/10df1ae2dc9d3c69.json#attention.py:44 "Non-transposed ownership puts adjacent high/residual rows in registers"
- S18 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4c2bed7e3e8c1e94.json#kernel.py:146 `out_layout = mma_layout` when NATIVE_STORE, `native_store = rows == 2232` (:323); `compute_efficient_padded_shared_layout` for both operands (:68), `[STAGES, BM, BK]` buffers with the rank-2 layout (:102-103); `_copy_layout` (:11), `constexpr_function` from `triton.runtime.jit` (:8), called with `gl.num_warps()` (:74), copy offsets in its slices (:94), `buffer_load_to_shared(a_buffer.index(stage), …)` (:106-107)
- S19 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_router_experts_shuffled_16x16_model_dim4096_n_routed512_inter_dim256/99f47a166d6d62fc.json#kernel.py:73 `element_bitwidth=64` fp64 accumulator; `k_width` 16 for fp16 parts of fp8 on 16x16x32 (:323), 4 on the non-transposed 4x64 (:327-330)
- S20 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/62df6de0b64b800d.json#kernel.py:498 `tiles_per_warp=[1, N_TILES if not FIRST or BN >= 256 else 1]` with `warps_per_cta=[1, 4]` (:497); activation scales loaded 4 bytes per lane in `BlockedLayout([1, 4], [16, 4], [4, 1], [0, 1])` (:507), converted to the A-scale layout every K step (:577); weight indices in `SliceLayout(0/1, bd)` (:516, :525), `b` to `mfma_scaled` unconverted (:576), preshuffled offsets `_weight_offset_8` (:30)
- S21 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3432dd09808b8428.json#kernel.py:403 `warps_per_cta=[1, 4] if TM <= 32 else [2, 2]`; `_packed_b_layout` replays the 16x16 weight shuffle (:364-370); `PIPE` (:410): a `[2, BN * BK // 8]` buffer with a rank-1 layout (:414) filled by `buffer_load_to_shared` (:336), viewed (:438) and read with `load_shared_relaxed` (:446); otherwise `buffer_load` + `B.store` (prefill_stage_packed_weight.py:36-37), viewed (:418) and read with `load(bd)` (:476)
- S22 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:238 gate and up split from one accumulator by reshape, permute, split; a column of a register tile by `gl.gather` and `gl.sum` (:580-582)
- S23 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_layouts.py:456 SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[]); PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape) (:505), powers of two (:590-602), `with_identity_for(pairs, shape, order)` with identity bases in `order` (:608-623), padding after every interval (include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:223); SharedLinearLayout(offset_bases, block_bases=[], alignment=16) (:631), shape from the largest basis (:660), `alignment` the allocation's byte alignment (lib/Dialect/TritonGPU/IR/Ops.cpp:765)
- S24 triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:85 with vec=2 "the element at (r,c) has value ((c / 2) ^ r) * 2 + (c % 2)" (:88); the linear form reduces the XOR modulo the row (LinearLayoutConversions.cpp:83)
- S25 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/Utility.cpp:152 the padded layout: 1- or 2-byte elements (:177), the instruction's non-K size (:185) 16 or 32 (:194), k_width 4, 8 or 16 (:198), `ds_read_b128` or `ds_read_b64_tr` (:223-224), padding 2·k_width or k_width (:234-236), no conflict tolerated for b128 (:259), one interval per wave instruction (:288)
- S26 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/831926aed9523c0f.json#kernel.py:88 a 32-row padded A tile, 16 rows taken with `gl.amd.slice` (:163); SharedLinearLayout with `alignment=256` (:287-289); two accumulators joined, permuted and reshaped (:471-472)
- S27 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/3480e97a5fcda6f6.json#kernel.py:188 the helper called with `k_width` 16 while the operands use 32 at BLOCK_M != 32 (:142-144); a PaddedSharedLayout spelled out by hand, equal to the helper's result (:147); fp8 `_copy_layout` (:19): registers `bases[:4] + bases[12:]`, lanes `bases[4:10]`, waves `bases[10:12]`; `_scale_window_layout`: 4 scale columns in registers, rows on lanes and waves (:32-49); a `[BLOCK_M, 64]` bf16 exchange in `SwizzledSharedLayout(8, 2, 8, [1, 0])` at BLOCK_M = 32 (:315-322), an XOR SharedLinearLayout otherwise (:325-349)
- S28 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Types.cpp:160 a shared layout's rank equals the shape's or is one less, and `.index` keeps the parent's layout (python/triton/experimental/gluon/language/_semantic.py:428-430); every LDS dimension but the first is a power of two (:98-105); so is every tensor dimension (python/triton/_utils.py:63, :68-69)
- S29 triton-lang/triton@c01b677:python/src/gluon_ir.cc:803 the AMD branch of `get_shared_bank_conflicts` asks the shared layout for a `vector` input dimension (assertion at lib/Tools/LinearLayout.cpp:292); a padded layout aborts earlier, in `toLinearLayout` (:792; lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1379 "unknown layout"); trivial means identical encodings or a minimal conversion over registers only (:196), and the TypeError prints both linear layouts (python/triton/experimental/gluon/language/_semantic.py:246)
- S30 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:65 SharedLinearLayout read view; BANK_XOR toggles one basis (:72); committed `BANK_XOR=False` (:1023)
- S31 triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:204 direct-to-LDS lanes are 128 or 32 bits
- S32 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/14e91857b4038e02.json#attention.py:110 "XOR is self-inverse: linear physical copies fill the consumer swizzle"
- S33 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:788 a swizzled async-copy destination (max_phase not 1) needs source swizzling: a per-lane `shuffleIdx` (:419), a `ds_bpermute` (Utility.cpp:121); stores predicate off redundant threads (`getFreeVariableMasks`, `emitRedundantThreadPredicate`, :1646-1648), as do buffer stores (:1971-1973) and atomics (:2278-2280); loads do not
- S34 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/f83985dc8066c00c.json#kernel.py:652 the replicated A tile staged through the padded helper layout (`store` :686, `load(ad)` :687)
- S35 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:993 fp32 state exchanged through a hand-written XOR SharedLinearLayout
- S36 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/5ea30df18b03ab8d.json#kernel.py:588 accumulator converted to `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])` for the store
- S37 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/54f6affa5c286a70.json#kernel.py:397 a one-wave 16x16 tile stored from the MFMA layout
- S38 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:62 warps or CTAs move: through LDS; lanes (:68): shuffles or LDS (:72); registers (:77): reorder; equivalent layouts (:82): no code; the LDS path stores and loads the tile in rounds with a barrier between (:230; a wave sync when no warp moves, :218-222); scratch size third_party/amd/lib/Analysis/AMDGPUAllocation.cpp:15
- S39 triton-lang/triton@c01b677:lib/Analysis/Utility.cpp:1311 in-warp shuffles only with fewer than two mixed register-lane transpositions; they are `shuffleXor` when the lane permutation is trivial, else `shuffleIdx` (lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:526-531), on CDNA4 DPP for xor mask bits 0-3 and permlane swaps for 16 and 32, or a `ds_bpermute` (third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:161, :221); a gather is warp-local only with the gather axis off the warps in both layouts (:1128) and the same warp (:1143) and lane (:1155) mapping of the other dimensions (`isWarpLocal`, :1096), else through shared memory (lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp:43-46; warp-local, one `shuffleIdx` per index register and per source register of its column, then a select, :297-333); a shuffle of data up to 32 bits is one `ds_bpermute`, of 64-bit data two (third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:84-113)
- S40 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp:44 one transposition of a register bit with lane bit 4 or 5 (:69) becomes `permlane16.swap`/`permlane32.swap` (:92), ahead of the generic path (TritonGPUToLLVM.cpp:157); a register-only conversion of 8-bit elements becomes `llvm.amdgcn.perm` byte permutes (:739, :301)
- S41 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/24ede969ed66a87a.json#kernel.py:246 `_permute_columns_for_a`; asserted trivial into `DotOperandLayout(0, mma, 1)` (:554-557)
- S42 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json#kernel.py:694 `_inverse_left_operand`: accumulator to operand through a middle DistributedLinearLayout, one wave (`warps_per_cta=[1, 1]`, :698), the `SHORT` and long middles (:700-715), an empty asm between the converts (:717-719)
- S43 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/4f257a3e566b20d1.json#kernel.py:162 "halve the cross-wave LDS transfer"
- S44 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_fp8_experts_shuffled_16x16_model_dim4096_n_experts513_inter_dim256_topk10/86987a66c0cdba39.json#kernel.py:282 "Round in the accumulator layout before redistributing for 128-bit stores"
- S45 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#kernel.py:572 "A blocked layout cannot be used here ... gl.split needs two"
- S46 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/slice.py:12 `gl.amd.slice(source, shape, offsets)` keeps the layout; extent and offsets align to its CTA tile; `ExtractSliceOp::verify` (third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp:143): every destination element held by a source register (:209-214), equal lane and warp bases (:218-222)
- S47 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/3c6e26ffcce7196d.json#kernel.py:58 B loaded in `BlockedLayout([8, 1], [4, 16], [1, 1], [1, 0])`, converted to `DotOperandLayout(1, mma, 8)` with `assert_trivial=True` (:74); one-wave `[16, 16, 32]` MFMA (:56)
- S48 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/bb663e5b0cf55a10.json#kernel.py:381 `_register_first_shared_layout` (:381-384), chosen when `SCALE_LDS` is 2 (:506-512)
- S49 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:54 a reduction works within each thread, then within the warp (:58), then, while warps remain on the axis, through a layout conversion and another warp step (:78-92); result `SliceLayout(axis, ·)` (lib/Dialect/TritonGPU/IR/Dialect.cpp:2971); LDS scratch third_party/amd/lib/Analysis/AMDGPUAllocation.cpp:55; on CDNA4 lane bit 5, or bits 4-5, by permlane swaps (third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:342-379)
- S50 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/e75e553b34178760.json#kernel.py:335 every wave holds all 256 experts (:335-340)
- S51 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2c2a09e220f2706c.json#kernel.py:214 "The 128-element reduction stays wave-local and needs no LDS exchange"; `BlockedLayout([4], [64], [2], [0])` (:215)
- S52 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/5cac8212f51edf84.json#kernel.py:622 `warps_per_cta=[SK, 1, 1]`; `gl.sum(acc, 0)` (:713)
- S53 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:29 `ASYNC_HIDDEN = ROWS <= 8`; the register path converts A and B in the K loop (:71-72)
- S54 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/1bbf98a368b3fcba.json#kernel.py:131 `SwizzledSharedLayout(16, 4, 4)` for A and B (:131-132) on `[BLOCK_M, GROUP_K // 2]` fp8 slots (:140-142)
- S55 aklr/ops/rms_norm/impls/triton/rms_norm_hidden_size6144/cf637bff0784c454.json#kernel.py:11 three chunks when `CHUNK * 3 == H`; the host picks `h // 3` when it is a power of two (:39)
- S56 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/8888fe378338cc8e.json#kernel.py:602 `TILE_N = BN * 3`, BN 32 or 64 (:589); a `[GROUP_K, 2 * BN]` buffer sliced in two (:634-640), a third `[GROUP_K, BN]` one (:644)
- S57 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/43ffd5e3e5b67298.json#kernel.py:138 fp8 keys converted to bf16 for `DotOperandLayout(1, …, 16)` of a `[16, 16, 32]` MFMA (:138-143); the query with `k_width` 16 (:174)
- S58 harness/skills/aklr-kb/triton/gluon/layouts.md:1 page-local Evidence retains the source and measurement conditions.
