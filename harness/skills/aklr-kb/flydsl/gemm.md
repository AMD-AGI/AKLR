---
title: "GEMM in FlyDSL: the FlyDSL side of the gemm/ cards"
kind: topic
ops: [gemm]
dtypes: [fp8, mxfp4, mxfp8, bf16, fp16]
regimes: [all]
gens: [gfx950, gfx942]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/gemm/
  - AMD-AGI/Primus-Turbo@7615a5a2:primus_turbo/flydsl/
---

# GEMM in FlyDSL

How to write in FlyDSL what the cards of `../gemm/` describe: which FlyDSL calls to use, what FlyDSL's compiler
does with them, and how FlyDSL's own GEMM kernels, aiter's FlyDSL GEMMs and Primus-Turbo's FlyDSL GEMMs behave when
you call or tune them. Each section is named after the card section it serves (`mainloop §8` is § 8 of the gemm
package's main-loop card) and holds only the FlyDSL part: read the card for the mechanism and the numbers.

"PS" is FlyDSL's `kernels/gemm/preshuffle_gemm.py` and "A16" its `kernels/gemm/gemm_a16w16_gfx950.py` (FlyDSL
v0.3.2). "GH", "PR", "FP8", "FP4" and "MX8" are Primus-Turbo's `utils/gemm_helper.py`, `utils/prims.py`,
`gemm/gemm_fp8_kernel.py`, `gemm/gemm_mxfp4_kernel.py` and `gemm/gemm_mxfp8_kernel.py` under `primus_turbo/flydsl/`
at main `7615a5a2`. Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence.

## Route here when

- A card of `../gemm/` ends a section with "In FlyDSL: `../flydsl/gemm.md` § …".
- You need the map of FlyDSL's and Primus-Turbo's GEMM kernels (§ INDEX Kernel map) or the FlyDSL API facts the
  GEMM package relies on (§ INDEX FlyDSL version).

## INDEX Kernel map

### FlyDSL v0.3.2, `kernels/gemm/`

| File | dtype / layout | Loop design | Entry point |
|---|---|---|---|
| `preshuffle_gemm.py` | fp8, int8, fp16, bf16; per-row × per-column scales; B preshuffled; optional bias + ReLU/SiLU/GELU | 4 waves along N; A through a 2-stage LDS ping-pong, B straight to VGPR; `hot_loop_scheduler` hints; optional A DMA into LDS (`use_async_copy`, gfx950) | `compile_preshuffle_gemm` (:108) |
| `gemm_a16w16_gfx950.py` | bf16, fp16; NN, NT, TN, TT | `m×n×k` waves, A and B by async DMA, `stages` deep, CShuffle store, split-K by atomics | `gemm_a16w16` (:1283) |
| `fp8_gemm_8wave.py` | fp8, row-wise scales; NT; B plain or preshuffled | 8 waves (512 threads), 8 LDS buffers (A/B × cur/next × two halves) | `compile_fp8_gemm_8w` (:148) |
| `fp8_gemm_4wave.py` | fp8, row-wise scales; NT; B plain or preshuffled | 4 waves 2×2, each a 128×128 quadrant; accumulators pinned in AGPR by per-MFMA inline asm; XCD remap | `compile_fp8_gemm_4w` (:162) |
| `fp4_gemm_4wave.py` | MXFP4 (E8M0 per 32); NT | the fp8 4-wave structure (2×2 waves, 8-buffer LDS ping-pong) with the scaled MFMA; scales through LDS | `compile_fp4_gemm_4w` (:634) |
| `mxfp4_preshuffle.py` | A MXFP4/MXFP6/MXFP8 × B MXFP4 or MXFP8 (preshuffled weights and scales) | 4 waves along N; A by DMA into a 2-buffer LDS ring; preshuffled B and the scales loaded straight to registers; scaled 16x16x128 MFMA | `launch_gemm` (:58) |

Shared helpers are in `kernels/common/mma/` (`mfma_preshuffle_pipeline.py`, `mfma_epilogues.py`). None of these
handles a K tail: K must be a multiple of the K tile. Their arguments are the entry-point signatures above. The
baseline aiter ships its own FlyDSL GEMMs (a bf16 split-K HGEMM, fp8/int8 preshuffle GEMMs): their entry points
and split-K reductions are in `kernels/flydsl_kernel_library.md`, their arguments in
`kernels/flydsl_knob_space.md`.

### Primus-Turbo main `7615a5a2`, `primus_turbo/flydsl/`

| File | dtype / layout | Loop design | Entry point |
|---|---|---|---|
| `gemm/gemm_fp8_kernel.py` | fp8 E4M3/E5M2, per-tensor scale; NT, NN, TN; bf16/fp16 out; any K | 8-wave pipeline raced against the 4-wave whole loop per shape | `gemm_fp8_tensorwise_flydsl_kernel` (:3448) |
| `gemm/gemm_mxfp4_kernel.py` | MXFP4 (E8M0 per 32) NT; fused GLU/dGLU + MXFP4 quant variants | 4-wave whole loop, VGPR-direct scales, folded C store | `gemm_mxfp4_flydsl_kernel` (:4351); `preshuffle_mxfp4_scales` (:4877) |
| `gemm/gemm_mxfp8_kernel.py` | MXFP8 NT | 8-wave pipeline, packed scales, tail-round split-K | `gemm_mxfp8_flydsl_kernel` (:985) |
| `gemm/gemm_bf16_kernel.py` | bf16/fp16 NT/NN/TN tile bodies (no dense bf16 entry point; these tiles serve other Turbo kernels) | 8-wave, 16x16x32 MFMA, transpose reads | `gemm_bf16_tile` (:721) |
| `utils/gemm_helper.py` | shared | G2S/S2R loaders, MFMA wrappers, store classes, XCD remaps, timing | — |
| `utils/gemm_epilogue_helper.py` | shared | fused activation and quantization epilogues | — |
| `utils/prims.py` | shared | `readfirstlane`, unsigned index math, wave scans | — |

Constants in both code bases (tile and preload tables, wait counts, adoption margins, L2 and burst thresholds)
were tuned on specific shapes. Reuse the mechanism and re-derive the number.

## INDEX FlyDSL version

All API statements in this package are for FlyDSL v0.3.2. Turbo main `7615a5a2` runs unmodified on it. The
raw buffer helpers (`create_buffer_resource`, …) are not in the wheel: they live in `kernels/common/buffer_ops.py`
in the FlyDSL repo, and Turbo vendors its own copy in `utils/buffer_ops.py`. In the wheel,
`rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(ptr, num_records_bytes))` turns an `fx.Pointer` into the raw
resource the `rocdl.raw_ptr_buffer_*` wrappers take. Kernel attributes passed through
`value_attrs`, such as `amdgpu-agpr-alloc`, must go under `"llvm.passthrough"`; a bare `"passthrough"` key is
silently dropped. `rocdl.s_waitcnt` takes a raw bitfield or the keyword form `s_waitcnt(vmcnt=…, lgkmcnt=…)`.
`rocdl.asyncmark()` and `wait_asyncmark(n)` exist and pair with `cdna4.BufferLoadAsyncLDS*`. A non-`int` `aux`
(cache policy) on the `rocdl.raw_ptr_buffer_*` wrappers is silently replaced by `None`. A closure-captured Python
dict is not in the JIT cache key. See § mainloop §4 below and `debug-flydsl-kernel.md`
§ What the cache key sees.

## workflow Step 0 — start from a reference kernel, not from a blank file

- **FlyDSL's `kernels/gemm/`**: fp8/int8/fp16/bf16 with a preshuffled B and an optional bias + ReLU/SiLU/GELU
  epilogue (`preshuffle_gemm`), bf16/fp16 in NN/NT/TN/TT with async DMA and split-K (`gemm_a16w16_gfx950`), and fp8/MXFP4 NT in 8-wave and 4-wave forms. K must be a multiple of the K tile.
- **Primus-Turbo's FlyDSL GEMMs**: fp8 NN and TN as well as NT, fp8 at any K ≥ 129 (MXFP4 needs
  K % 64 == 0, MXFP8 needs K % 128 == 0 and K ≥ 256), a whole-loop inline-asm design, a per-shape race
  between designs, and fused GLU/dGLU + MXFP4-quantization epilogues. They target training shapes; for
  M ≤ 256 use `../gemm/small_m.md`.

Both run on FlyDSL v0.3.2.

## mainloop §2 The 8-wave pipeline (FP8 `_compile_dense_nt`)

- **LDS.** Allocate it with `fx.SharedAllocator().allocate(...).peek()`.

## mainloop §3 The 4-wave whole loop: one inline-asm region for all of K

Results come back through `llvm.extractvalue` on a struct return type.

## mainloop §4 Register plan and inline-asm constraints

**Setting `amdgpu-agpr-alloc` and friends.** Turbo passes these through `value_attrs`
(`make_value_attrs`, GH:1580). Put them under `"llvm.passthrough": [...]`. A bare `"passthrough": [...]`
key is silently dropped at `convert_gpu_to_rocdl`. A dropped attribute gives no error.

## mainloop §5 MFMA atom choice

- On gfx950, PS switches
  fp8 to the 16x16x128 form when `tile_k % 128 == 0` (`use_mfma_scale_128`, PS:159).

## mainloop §6 K tails inside the kernel

FlyDSL's shipped kernels require K to be a multiple of
their K tile.

## mainloop §8 FlyDSL's own GEMM kernels (v0.3.2): what to know when calling or tuning them

"PS" is `kernels/gemm/preshuffle_gemm.py` and "A16" is `kernels/gemm/gemm_a16w16_gfx950.py`. They handle no
K tail (K must be a multiple of the K tile), and the caller picks the configuration. PS and the MXFP4 kernels need B pre-shuffled on the host; the fp8 4-/8-wave kernels take either
layout (`b_preshuffled`).

| Kernel | dtypes | Loop | Notes |
|---|---|---|---|
| PS `compile_preshuffle_gemm` | fp8, int8, fp16, bf16 | 4 waves along N; A through LDS, pre-shuffled B straight to VGPR | per-row × per-col scales; `use_async_copy` is gfx950-only |
| A16 `gemm_a16w16` | bf16, fp16 | `m×n×k` waves (default 2×4×1), async DMA for A and B, `stages` deep | split-K by atomics (not bitwise reproducible), `group_m` XCD swizzle, half-tile interleave |
| `fp8_gemm_8wave.py`, `fp8_gemm_4wave.py` | fp8 (row-wise scales) | the §2 pipeline / a 4-wave per-MFMA-asm loop | NT only, `K % 128 == 0` |
| `fp4_gemm_4wave.py` | MXFP4 | 2×2 waves, 8-buffer LDS ping-pong, scales through LDS | host-side scale and weight preshuffle |
| `mxfp4_preshuffle.py` | A MXFP4/MXFP6/MXFP8 × B MXFP4/MXFP8 | 4 waves along N; A by DMA into a 2-buffer LDS ring; preshuffled B and the scales loaded straight to registers | host-side scale and weight preshuffle |

Facts about PS and A16 that matter when tuning them:
- **PS tile algebra.** `tile_K_perm = 128 if mfma_scale else (64 if 8-bit else 32)` (PS:171). Four waves
  split N. The A tile must be a multiple of 4096 B, or its tail is never loaded (PS:190).
- **PS waits.** `rocdl.s_waitcnt(num_b_loads)` passes a raw bitfield, not a `vmcnt`. `s_waitcnt(8)`
  emits `vmcnt(8) expcnt(0) lgkmcnt(0)`. In your own code write `rocdl.s_waitcnt(vmcnt=n)`.
- **PS scheduler hints.** `sched_group_barrier` groups spread VMEM reads (mask `0x020`), DS reads and DS
  writes over the MFMAs. Without `preload=`, 8-bit tiles on gfx950 look up `_TILE_PRELOAD_TABLE`
  (PS:19), and every other dtype or tile gets `(0, 0)`.
- **PS epilogue.** Ragged M is bounded by the descriptor (`max_size=False, num_records_bytes=...`), not
  by a predicate.
- **A16 waits.** A16 waits with `rocdl.asyncmark()` / `rocdl.wait_asyncmark(stages-2)` around
  `cdna4.BufferLoadAsyncLDS128b`, which counts DMA groups instead of `vmcnt`. This is the in-DSL
  alternative to hand-written waits.
- **Configuration space.** The arguments are the entry-point signatures in § INDEX Kernel map above; aiter's
  tuned rows for its own bf16 GEMM are in `../gemm/small_m.md` §7.

## mainloop §10 Reading the ISA

`FLYDSL_DUMP_IR=1` writes every pass to `<FLYDSL_DUMP_DIR or ~/.flydsl/debug>/<kernel>/`. It skips the
disk-cache lookup, so the dump reflects the current source. Do not time a run with it on.
`ARCH=gfx950 COMPILE_ONLY=1` compiles without launching.

## datapath §1 Allocate LDS the v0.3.2 way

Declare the buffers as an `@fx.struct` of `fx.Array` fields and allocate them in the kernel with
`fx.SharedAllocator().allocate(Storage).peek()` (PS:214, 283). With the default `static=True` each
field becomes its own LDS global and the backend places it (`expr/gpu.py:157`): field order is not
a layout you control, and a field an ablation arm stops using is pruned, which moves the others. An
epilogue that reuses the operand buffers declares them as an `@fx.union` (A16's `SharedStorage`
overlays `ab` and `c`). `SmemAllocator` / `SmemPtr` are legacy.

## datapath §2 Swizzle at 16-byte granularity, composed into the layout

- PS composes the swizzle into the LDS view instead of computing addresses:
  `fx.make_composed_layout(fx.static(fx.SwizzleType.get(bits, base, bits)), ordered_layout)`, with
  `bits = log2(k_blocks16)` and `base = log2(elements per 16 B)` for 8-bit or DMA-written tiles. The
  writer and the reader then agree by construction — and the DMA path, which computes addresses by
  hand, applies the same `k ^ ((m % k_blocks16)·elems_per_16b)` (PS:379).

## datapath §3 Bank conflicts: compute the floor, then measure

FlyDSL's `Vector.load` declares none
(`language/limits.md` § Unclassified has the lowering, the fix and its bank-conflict cost).

## datapath §4 Waits around LDS

`cdna4.BufferLoadAsyncLDS*` with `rocdl.asyncmark()` / `wait_asyncmark(n)` is the in-DSL alternative (A16
uses it).

## datapath §5 Global → LDS DMA

- FlyDSL atoms: `fx.rocdl.BufferCopyLDS128b` (PS's `dma_a_to_lds`, behind PS's `use_async_copy` flag) is
  compiler-tracked — the compiler inserts the `vmcnt` wait before the staged LDS data is read (§4); FlyDSL's
  docstring calls it the synchronous form. For `fx.rocdl.cdna4.BufferLoadAsyncLDS128b` (A16) the backend
  inserts no such wait: you close groups with `rocdl.asyncmark()` and wait with `rocdl.wait_asyncmark(n)`
  (`expr/rocdl/cdna4.py:42-55`).

**How Turbo issues the DMA** (`G2SLoader`, GH:305-386):
- One `fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)` per 16 B per lane; each wave writes 1 KiB per
  step.
- The wave-uniform K offset rides the SGPR `soffset`: `fx.copy(atom, src, dst, soffset=...)`.

## datapath §7 Prefetch inside the K loop

Carry the prefetched values through the loop: with `range(..., init=[...])` only yielded values
persist; a plain `range()` carries every name the body reassigns
(`debug-flydsl-kernel.md` § The three loop forms).

## datapath §8 Buffer descriptors: per-tile i64 rebase, pinned to SGPRs

- Pass tensors to the entry function at their full rank. A flat view of more than 2³¹ elements
  overflows the int32 shape pack.

## datapath §11 MX scales

One `@flyc.jit` host stub launches the preshuffle kernel
and then the GEMM, so the scales are repacked in stream order with no extra Python dispatch
(`build_preshuffle_ab_kernel`, GH:2115-2122, 2262).

## schedule_and_locality §2 Tile order across XCDs

**FlyDSL's kernels expose the order as an argument:** `xcd_swizzle` in PS (`xcd_remap_bx_by`, in
`kernels/common/mma/mfma_preshuffle_pipeline.py`), `use_xcd_remap` in `fp8_gemm_4wave`, and `group_m`
(`BlockSwizzle`) in A16. A16's swizzle falls back to the plain order when `group_m` is 0 and for grids under
256 workgroups or not a multiple of 8 (`kernels/gemm/gemm_a16w16_gfx950_utils.py:105, 123`).

- FlyDSL's own `_xcd_swizzle` (`kernels/gemm/fp8_gemm_4wave.py`) is a contiguous-chunk remap plus
  `GROUP_M = 4`, used only when the grid has ≥ 4·256 workgroups and a multiple of 8; otherwise linear.

## schedule_and_locality §3 Scheduling hints and fences

FlyDSL's helpers are thin wrappers over `sched_group_barrier`:

| Helper | Mask | Classes |
|---|---|---|
| `rocdl.sched_mfma(n)` | `0x008` | MFMA/WMMA |
| `rocdl.sched_vmem(n)` | `0x020` | **VMEM reads only** |
| `rocdl.sched_dsrd(n)` | `0x100` | DS reads |
| `rocdl.sched_dswr(n)` | `0x200` | DS writes |

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile.
On gfx950 it spreads the LDS reads and VMEM loads evenly over all MFMAs,
for both sync and async copy (`guides.md` § gemm-optimization (FlyDSL's §5.5));
the sync path also places its LDS writes at the tail.
`preload=(dsrd, dvmem)` sets how many go ahead of the first MFMA
(PS:439-494).

**Race LLVM's scheduler per tile:** `flyc.compile[{"llvm_options": {"amdgpu-sched-strategy": "iterative-ilp"}}]`
(or `max-ilp`, `max-memory-clause`, `iterative-minreg`, `iterative-maxocc`); a misspelled name silently compiles the default.

## schedule_and_locality §7 Accumulators, descriptors and large operands

- **Ragged edges.** `fx.rocdl.make_buffer_tensor(t)` defaults to `max_size=True`, which sets
  `num_records = 0xFFFFFFFF` — no out-of-range masking. To have out-of-range loads return zero and
  out-of-range stores dropped, pass `max_size=False` or `num_records_bytes`, and derive the bound from
  traced values (`debug-flydsl-kernel.md` § What the cache key sees).

## epilogue §3 Store classes, from free to expensive

In FlyDSL v0.3.2, PS stores directly and can fuse a per-N bias with ReLU/SiLU/GELU (`epilogue=`); A16 and
`kernels/common/mma/mfma_epilogues.py` carry the CShuffle store.

FlyDSL's buffer store cannot emit `buffer_store_short_d16_hi`, so taking the high half of a packed pair costs one
`v_lshrrev_b32`.

## epilogue §5 Cache policy on the store

- **Pass cache-policy bits (`aux`) as a Python int.** `rocdl.raw_ptr_buffer_load`,
  `raw_ptr_buffer_load_lds` and the buffer atomics replace any `aux` that is neither an `int` nor an
  MLIR attribute with `None`, without an error
  (`python/flydsl/expr/rocdl/__init__.py:663-819`). A traced value there silently means "default
  policy".

## epilogue §6 Bounds without per-column masks

FlyDSL applies `mask=` after the element→byte multiply, as `select(mask, off, 0x7FFFFFFF)`.

## epilogue §7 beta = 1 (accumulate into C)

Compiling runs the kernel once, which would accumulate into the caller's C twice (`compile_with_scratch_out`,
GH:86-95).

## small_m §3 Grid size at small M

aiter's reaches one M-block only when M ≤ 16 or M is itself a catalog `tile_m` (32, 48, 64, 80, 96,
  128, 256): it drops every `tile_m` above `max(M, 16)`, so M=24 keeps only 16 and M=72 only 16–64
  (`kernels/flydsl_knob_space.md`).

## small_m §4 Split-K

- Split-K must divide K, each split must be whole `tile_k` steps and hold at least `stages` of them; the
  catalog offers 1–13 (`gemm_kernels.py:216-226`).
- **A 256-tile cap refuses split-K on larger grids** at launch (`gemm_kernels.py:712-724`); the tuner already
  drops split-K candidates above 128 tiles (`csrc/gemm_a16w16/gemm_a16w16_tune.py:753`).
- **The current split-K is not bitwise reproducible** at split-K ≥ 3, or at split-K 2 with a bias: partials
  are added into C in arrival order. Measured, for both kernels: every one of 400 repeats differed at split-K 2
  with a bias and at 3, 4 and 8 *(field, E1)*. Where
  runs must match bitwise, use split-K 1 or 2 without a bias, or a fixed-order reduction
  (`kernels/flydsl_kernel_library.md` § Split-K).

## small_m §6 Memory-side knobs

**Not in the current kernels**: the generic kernel's global→LDS copy is inline asm with `sc0` fixed
  (`splitk_hgemm.py:542-557`).

  - The generic kernel always keeps the plain order (`swizzle_for_cache_reuse`, `splitk_hgemm.py:353-357`).

- **Preload the leading kernel arguments into SGPRs**: `flyc.compile[{"llvm_options": {"amdgpu-kernarg-preload-count": <n args>}}]`;
  the dumped ISA then shows `.amdhsa_user_sgpr_kernarg_preload_length` > 0.
  - Pass the operands as `fx.Pointer` (`from_c_void_p`). A raw `torch.Tensor` lowers to a pointer plus a layout
    struct, and preload stops at the first struct: only the first pointer is preloaded (length 2 at count 16).
  - **Not in aiter's bf16 GEMM kernels**; its FlyDSL MoE helpers set it. Eight of AKLR's
    own bf16 GEMM kernels use it.

**Not in the
  generic kernel**: it clamps rows past M to row 0 and loads them (`splitk_hgemm.py:512-519`).

## small_m §7 aiter's tuned table at small M

- **A row whose kernel name the catalog cannot parse falls through** to the next candidate or the default,
  with only a logged warning, and every `flydsl` row falls through silently when FlyDSL is not importable
  (`tuned_gemm.py:163-175`). This aiter parses only `flydsl_gemm{stages}_..._split_k{s}_...` names; a name in any
  other format (e.g. `flydsl_hgemm_..._ks7_...`) logs a warning and falls through. Re-tune instead of copying
  rows.
- **Tuner filters can delete the winner silently**: the `tile_m ≤ max(M, 16)` cut, `tile_n` having to divide N,
  the 128-tile cap on split-K (`csrc/gemm_a16w16/gemm_a16w16_tune.py:740-755`). Read the candidate count the
  tuner logs for a missing shape before concluding "no candidate".

## autotune Per-shape autotune and the launch path

> FlyDSL's own kernels take their configuration from the caller; the
> baseline aiter ships its own copies and picks their configuration from offline-tuned tables
> (`../gemm/small_m.md` §7). FlyDSL also ships a general `flydsl.autotune`
> decorator that benchmarks a config list per key (`python/flydsl/autotune.py`).

## autotune §3 Staged searches

  - **The epilogue-variant stage does not run at `7615a5a2`.** It looks the swizzle winner up under a 13-field
    key (FP4:3294-3296), the swizzle stage stores its entries under 15 fields (FP4:3215-3231), and the
    `KeyError` is swallowed (`except Exception: pass`, FP4:3374-3375), so every shape keeps
    `(taccw, coop) = (False, False)` (FP4:3289). *(field, E2)* With the key fixed, none of the 18 shapes of a
    dense MXFP4 table picked a variant.
  - **Nor does the real launch reuse the race's compiled winners.** The race stores them under the 15-field key
    (FP4:3215-3231), the launch looks them up under 14 fields, without `k_real` (FP4:4519), so the first real launch
    of a shape compiles again, although the docstring says it does not (FP4:3188-3189).

## autotune §5 Cache keys

- Pass truly dynamic values as
  `fx.Int32` arguments. Closure-captured Python dicts are *not* in FlyDSL's JIT cache key
  (`debug-flydsl-kernel.md` § What the cache key sees).

## autotune §6 The launch path

- **Call the `flyc.compile` object, not the `@flyc.jit` function.** The jit call re-binds the signature,
  rebuilds the cache key and re-checks globals on every launch: *(field, E3)* 180 µs against 37 µs for a fused
  MoE combine kernel's ~24-argument launch (`run_compiled`, GH:2049-2069). Compile once per key with
  `_static_layout` (`flyc.from_torch_tensor` per tensor, FP8:2973) so the object does not re-read shapes and
  strides.
- **Under CUDA-graph capture: no timing, and call the raw closure.** Launch the raw `@flyc.jit` closure, because "a
  flyc.compile'd object regresses under CUDA-graph capture" (`_run_dense`, FP8:3081-3091).
- **A dispatch layer that calls the `@flyc.jit` closure pays its host cost on every call.** *(field, E4)* On a grouped fp8 wgrad whose public entry
  called the `@flyc.jit` closure behind a runtime split-K policy, host time was 77.9 µs per call (cache-key
  build 54%, the globals-drift check 22%); moving the entry to the compiled object cut it to 19.1 µs and
  gained 6.5% on a six-shape score that includes host time.

## correctness §5 LDS placement and DMA

- **Reordering `@fx.struct` fields does not move LDS.** With `SharedAllocator(static=True)` every leaf
  is its own symbol placed by the backend (`expr/gpu.py:157`); an arm that stops using a leaf lets
  the others be re-packed.
- **Buffers that must be contiguous go in one field, or use `static=False`.** The default static
  allocator places fields by size, largest first, and equal sizes by symbol name (declared
  `a4k, b16k, c8k, d256, e1k` land as `b16k, c8k, a4k, e1k, d256`). `SharedAllocator(static=False)` keeps
  declaration order. A pool addressed as one base plus immediates must not rely on the static order.

## correctness §6 Bounds

- The default `max_size=True` sets `num_records = 0xFFFFFFFF` — no masking.
- `mask=` of aiter's `buffer_ops.buffer_store` / `buffer_load` does not mask on such a descriptor: its byte offset
  0x7FFFFFFF is in range, so a masked lane writes 2 GiB past the base *(field, E5)*; in a fused MoE sort kernel such a store faulted,
  and an out-of-range index in its place ran correct *(field, E6)*.
- Derive every shape-dependent bound from traced arguments; a host integer can be baked into a cached
  kernel and reused for the next shape (`debug-flydsl-kernel.md` § A bound or address
  baked from a host integer).

## measurement §6 Apparatus traps

- **Do not time with `FLYDSL_DUMP_IR=1`** (it bypasses the cache and recompiles, `jit_function.py:1414`;
  on an attention kernel, an arm's first read under `FLYDSL_DUMP_IR=1` with a single rep was +15% on a ±0.2%
  ruler *(field, E7)*).

## attribution §9 Readings you cannot trust

- Do not time with `FLYDSL_DUMP_IR=1` or a single repetition. *(field, E7)* On an attention kernel, a dumped run
  read +15%; three normal runs read ±0.2%.

## gfx942 This repository's notes

**No async copy on gfx942 in this GEMM.** `preshuffle_gemm` raises `ValueError` for `use_async_copy=True` on
gfx942 (PS:154-157). The DMA instruction itself also lowers on gfx942, at 4 B per op.

**FlyDSL's `hot_loop_scheduler` is the reference use.** `preshuffle_gemm` builds one hint table per tile. On
gfx942 it uses the sync-copy pattern of FlyDSL's guide: a prologue places the first LDS reads, then each step is
one VMEM load, a group of MFMAs, one LDS read and a second group of MFMAs, with the next tile's LDS writes at the
tail so they land before the barrier (gemm-optimization §5.3–5.4; PS:402-438).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, aiter a0e3cbea, N=400, eager calls, a side-stream matmul on alternate repeats] aiter's HGEMM and small-M split-K: with a bias at split-K 2, and at split-K 3, 4 and 8, 400/400 repeats differed from the first; split-K 1, and 2 without a bias, 0 (m,n,k = 16, 2048, 6144 (probe /workspace/code/gpt_oss_docker/work/sikl/skillenv/probes_sikl.py:165; its output is kept only in this note); local:/workspace/code/gpt_oss_docker/.claude/memory/project_sikl_flydsl_skill_port.md:30)
- E2 [2026-09, gfx950, FlyDSL 0.2.4, N=1, code reading, then one scored read per arm] the TACCW/COOP epilogue-twin stage looks its base up in `_MXFP4_AT_CACHE` under a 13-field key while entries are stored under 15 fields; the `KeyError` goes to `except Exception: pass`, so the twin axis had not run in the whole campaign; with the key fixed in a later campaign, all 18 cells of the `gemm18` table still picked `(taccw, coop) = (False, False)` (Turbo MXFP4 dense GEMM for gpt-oss, `_autotune_mxfp4_config`, round 16; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260923_085930/rounds/round-16/agent.txt:8; myskill pitfalls/06-autotune-dispatch.md:827; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260925_144843/memory.md:199)
- E3 [2026-08, gfx950, FlyDSL ?, N=?, host time per call; method not stated] calling the `@flyc.jit` launch 180 µs per call against 37 µs through the `flyc.compile` object, for the fp8 combine's ~24-argument launch, called twice per MoE layer in a training step (fused MXFP8 MoE, FC2 + combine; Primus-Turbo 22e544d5 (#456), kept as the docstring of `run_compiled`, Primus-Turbo 7615a5a2:primus_turbo/flydsl/utils/gemm_helper.py:2054)
- E4 [2026-08, gfx950, FlyDSL ?, N=3, campaign score `_bench_wgrad_vs_fwd.py` (wgrad FLOP rate over the same process's forward, CUDA events around the public call, so host time counts), three reads against one same-day baseline read] the wgrad public entry moved from the `@flyc.jit` closure to the `flyc.compile` object, as the forward entry already was: host 77.9 → 19.1 µs per call (`_resolve_and_make_cache_key` 54% and `_check_globals_drift` 22% of the old time), output byte-identical; gm 0.99043 → 1.05074 / 1.05483 / 1.05459, recorded +6.46%; this followed the runtime split-K policy kept the round before (gpt-oss per-tensor fp8 grouped wgrad, round 4; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260803_111756/memory.md:127; :131; :132; myskill methodology/08-splitk-crosslane-lowprec-wgrad.md:172)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, GPU probe: one 64-lane uint8 `buffer_store` with every lane masked off, into a 64-byte view of a 3 GiB tensor] `create_buffer_resource(max_size=True)`: the byte at base + 0x7FFFFFFF was written; `max_size=False, num_records_bytes=64`: nothing written (aiter/ops/flydsl/kernels/buffer_ops.py:370, :674; range rule `../hardware/gfx950/memory_and_sync.md` §2; a fused MoE sort + quant kernel's first form hit `HSA_STATUS_ERROR_MEMORY_FAULT` with masked stores; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_m/knowledge_patch_v2/probe/mask_probe.py; local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_m/workspace/forge_experiments/lessons/iter_001.md:18)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one run] a scatter fused into a MoE sorting kernel, storing through a max-size descriptor with `buffer_store(mask=)`: GPU memory fault, a write to a read-only page; the same kernel with an out-of-range element index instead ran and matched (max_diff 0) (fused MoE EP8 prefill, n_tokens 9397 and 16384; local:/shared_nfs/kyle/moe/kf/runs/ep8_cmp/workspace/forge_experiments/lessons/iter_018.md:11; :34)
- E7 [2026-09, gfx950, FlyDSL 0.3.2, N=3, same-process probe] an arm's first read under `FLYDSL_DUMP_IR=1` and `REPS=1`: +15.44%; three normal re-runs ±0.2% (GLM-5.2 sparse-MLA decode attention producer; campaign 20260911_143318 round 10; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260911_143318/rounds/round-10/agent.txt:15)
