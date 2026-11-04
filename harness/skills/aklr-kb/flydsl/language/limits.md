---
title: "FlyDSL v0.3.2 limits that shape a fast gfx950 kernel"
kind: language-note
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/typing.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/utils/arith.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/runtime/device.py
  - ROCm/FlyDSL@dd8373349:lib/Conversion/FlyToROCDL/FastMathOps.cpp
  - ROCm/FlyDSL@dd8373349:docs/language/arithmetic_types.md
  - ROCm/FlyDSL@dd8373349:docs/architecture_guide.md
---

# FlyDSL v0.3.2 limits

What FlyDSL v0.3.2 cannot do, or does in a way a kernel author would not guess. Each bullet: the limit, what it
does to the kernel, the workaround, the source. Probe results are compile-only (`COMPILE_ONLY=1 ARCH=gfx950`, CPU
tensors, no GPU): IR, ISA and code-object metadata, no timings. Error texts are in `compile_errors.md`.

## Open limits

- **`fx.assume` is a stub** ("WIP, unsupported for now", S1). No range or divisibility fact reaches the compiler
  through it. Workaround: state tensor facts on the argument: `flyc.from_dlpack(t, assumed_align=16)`, then
  `.mark_shape_dynamic(dims, divisibility=16)` / `.mark_stride_dynamic(...)` (S2).
- **Two converts are inline asm.** `fx.rocdl.cvt_pk_bf16_f32` (gfx950: f32 pair to packed bf16) and
  `fx.rocdl.cvt_off_f32_i4` emit `llvm.inline_asm` until MLIR grows ROCDL ops for them (TODO, S3); LLVM cannot
  fold or combine an inline-asm string with its neighbours. For f32 → bf16, `Vector.to(fx.BFloat16)` let LLVM
  select `v_cvt_pk_bf16_f32` itself *(field, E13)*.
- **Block reduce is a first version.** `fx.coop` block reduce `WARP_REDUCTIONS` folds the per-warp totals
  linearly, every thread walking all `num_warps` slots (TODO: 15 combines at a 1024-thread block); `RAKING` walks
  contiguous LDS segments, which the source calls not bank-conflict-free, and its strided variant is "planned,
  not implemented"; the op must be commutative (S4). Workaround: for wide blocks, fold the slots in one warp.

## By design

- **No fast-math unless asked.** Without a `fastmath` hint there is no ambient setting (S5). `a * b + c` on f32
  vectors compiled to `v_pk_mul_f32` + `v_pk_add_f32`; inside `with fx.fastmath("contract"):` it became
  `v_pk_fma_f32` *(field, E1)*. Default `fx.exp2` carried 8 `v_cndmask` + 4 `v_ldexp` around 4 `v_exp_f32`
  *(field, E2)*. Workaround: `with fx.fastmath("contract")` (or `"fast"`) around hot math, or
  `flyc.compile[{"fastmath": ...}]` (S5).
- **Integer `/` returns a float; signed `//` floors.** `Int32 / Int32` converts both sides to f32 and divides;
  signed `//` emits `arith.floordivsi` (S5, S6). Use `//` or shifts for index math, `fx.ceildiv` for ceilings.
- **Raw `torch.Tensor` arguments are layout-dynamic.** Every dim is runtime with divisibility 1, every non-unit
  stride is runtime, the assumed alignment is one element, and a tensor without a unit-stride axis is rejected
  (S7). The probe's input was `!fly.memref<f32, global, (?,?):(?{i64},1)>`; through `flyc.from_dlpack` it was
  `(100,1000):(1000,1)` *(field, E3)*. Index math on shapes and strides therefore stays runtime. Workaround:
  `flyc.from_dlpack(t)` (static, one compile per shape), plus `.mark_shape_dynamic(dim, divisibility=...)` (S2).
- **`make_buffer_tensor` does not bound the buffer by default.** `max_size=True` writes
  `num_records = 0xFFFFFFFF` (S8), so hardware does not drop an access past the tensor end. Pass
  `max_size=False` (size from the layout) or `num_records_bytes=` when the descriptor is the mask.
- **Static block dims, or the kernel is compiled for at most 256 lanes.** Static `block=` dims at launch become
  `known_block_size` (S9). The probe kernel got `"amdgpu-flat-work-group-size"="128,128"`; launched with a
  runtime `fx.Int32` block it got `"1,256"` and no `rocdl.reqd_work_group_size` *(field, E4)*, and nothing checks
  a runtime block above 256. Workaround: static block dims, or `@flyc.kernel(known_block_size=[x, y, z])` (S9).
- **One `SharedAllocator` per kernel; one LDS symbol per struct leaf.** A second allocator raises (S9). In
  `static=True` mode fields do not overlap; a `@fx.union` is one allocation sized to its widest variant (S10).
  LDS reused across phases (A/B stages, then a C-shuffle buffer) adds up unless declared as a union.
- **`waves_per_eu` alone splits the register file between arch VGPRs and AGPRs.** At `rocdl.waves_per_eu` 4 a lane
  gets 512 / 4 = 128 registers; with no AGPR budget the allocator gave a kernel that needed 128 arch VGPRs 64 arch and
  64 AGPRs and spilled 96 VGPRs (164 B of scratch). With `"llvm.passthrough": [["amdgpu-agpr-alloc", "0"]]` beside it
  the same kernel compiled to 128 arch VGPRs and no spill *(field, E15)*.
- **Compile hints stick to the jit object.** `flyc.compile[hints](f, ...)` merges `hints` into
  `f.compile_hints` (S11); a later plain `f(...)` that recompiled still carried `"amdgpu-waves-per-eu"="2"`
  *(field, E5)*. Give each hint set its own `@flyc.jit` object when comparing hints.

## Unclassified

- **A bare `passthrough` attribute is dropped silently.** `value_attrs` go onto the `gpu.func` unchanged (S9);
  `convert-gpu-to-rocdl` drops a `passthrough` key and keeps `llvm.passthrough`, which reaches the LLVM function
  attributes *(field, E6)*. FlyDSL's own flash-attention kernel passes its DAZ entries as `passthrough` (S12).
  Write `value_attrs={"llvm.passthrough": [[key, value], ...]}` and read the line back from `NN_llvm_ir.ll`.
- **An attribute set on `ctx.gpu_module_body` before `.launch()` is lost**: the `gpu.func` is emitted inside `.launch()`
  (S30), so the walk finds nothing *(field, E16)*, as in aiter's MoE and small-M `waves_per_eu` (S31). Use `value_attrs=` or `compile_hints`.
- **The `maxnreg` hint left the binary unchanged; the attribute did not.** `maxnreg` only becomes
  `--amdgpu-num-vgpr=N` in `gpu-module-to-binary opts` (S13). With `maxnreg=32` a 68-VGPR kernel stayed at 68
  VGPRs and 0 spills; `llvm.passthrough` `["amdgpu-num-vgpr", "32"]` on the same kernel gave 8 VGPR spills
  *(field, E7)*. On FlyDSL's MFMA preshuffle GEMM example, `maxnreg=64` left 144 VGPRs and 0 spills, and the
  attribute at 64 gave 94 spills *(field, E14)*. `waves_per_eu` is also set as `rocdl.waves_per_eu` on the kernel (S13), so it takes effect.
  Workaround: cap registers through the `llvm.passthrough` attribute.
- **`FLYDSL_COMPILE_OPT_LEVEL` does not change code.** Documented as "Optimization level (0–3)" (S14), it is
  read only into the cache key (S11, S15) while `rocdl-attach-target` gets `O=2` (S13). Levels 0 and 3 gave
  byte-identical LLVM IR and ISA *(field, E8)*.
- **`FLYDSL_DEBUG_DUMP_ASM=1` writes nothing** with the built-in codegen, though documented as "Dump final AMD
  ISA assembly" (S14); it is read only to raise under `FLYDSL_COMPILE_LLVM_DIR` (S11) *(field, E9)*. Use
  `FLYDSL_DUMP_IR=1` and read `NN_final_isa.s`.
- **`range(..., init=)` stays a runtime loop with Python-int bounds.** `kernel_tuning_guide.md` says constant
  Python-int bounds make the rewriter unroll the loop and drop `init=` (S23); any `range` with `init=` is rewritten to
  `scf_range` (S24), which converts the bounds to `index` constants and emits an `scf.for` carrying the `init` values
  (S25). `range_constexpr` is the form that unrolls.
- **`fx.Index` is not deprecated.** `kernel_authoring_guide.md` calls it deprecated (S26); `api_stability.md` names
  `fx.Index(x)` as the replacement for the deprecated `fx.index_cast` (S27), and only `index_cast` is marked
  deprecated in the source (S28). `fx.Index` is an unsigned 64-bit integer (S29): use `fx.Int64` for an offset that
  can be negative.
- **A 16-byte LDS load declared 4- or 8-byte aligned is split.** At an address with a runtime offset, alignment 4
  gave two `ds_read2_b32`, 8 one `ds_read2_b64` and 16 one `ds_read_b128`; at an address provably 16-byte aligned,
  alignment 4 also gave `ds_read_b128` *(field, E19, E20)*. `Vector.load` passes no alignment to `vector.LoadOp`
  (`Vector.store` takes one, S35), so a load of 4-byte elements carries 4. Build `vector.LoadOp(...,
  alignment=16)` yourself, or `fx.ptr_load` through a pointer type declared 16-byte aligned. On two fused-MoE
  C-shuffle epilogues this cut `SQ_LDS_BANK_CONFLICT` 4.2× and 4.9× *(field, E21)*.
- **LDS overflow does not fail the compile.** `SharedAllocator` has no capacity check (S16; the legacy
  `SmemAllocator` has one, S17). 192 KiB on gfx950 printed `error: <unknown>:0:0: local memory (196608) exceeds
  limit (163840) …`, returned normally, cached a code object with `group_segment_fixed_size` 196608, and the next
  run hit that cache without a message *(field, E10)*. Check `.group_segment_fixed_size` in the ISA dump.
- **Arch-dispatched helpers read the host, not `ARCH`.** The compile target takes `ARCH` first (S13), but
  `fx.rocdl.s_waitcnt`, `fx.rocdl.WMMA` and the buffer-descriptor flags call `get_rocm_arch()`:
  `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION`, `rocm_agent_enumerator`, else a built-in default that is not
  gfx950 (S8, S18). With `ARCH=gfx950` alone and no GPU they saw that default *(field, E11)*. Set `FLYDSL_GPU_ARCH=gfx950` too when compiling away from
  an MI355X.
- **Raw buffer wrappers drop a non-literal `aux`.** `fx.rocdl.raw_ptr_buffer_load`, `raw_ptr_buffer_load_lds`
  and `raw_ptr_buffer_atomic_fadd` / `_fmax` replace an `aux` that is neither a Python `int` nor an
  `ir.Attribute` with `None` (S19): cache-policy bits passed as `fx.Int32` vanish without an error.
- **Native fast math covers f32 `exp2` only.** `convert-rocdl-fastmath-ops` maps `afn` `math.exp2` to
  `rocdl.exp2` and nothing else; other `afn` math goes to device-library code (S20, S21). `fx.exp2(x,
  fastmath="fast")` gave bare `v_exp_f32`; `fx.exp(x, fastmath="fast")` still carried 8 `v_cndmask` for 4 lanes
  *(field, E2)*. Workaround: `fx.exp2(x * LOG2E, fastmath="fast")` where its precision suffices.
- **`denormal-fp-math-f32` reaches the IR, not the mode register.** Under `llvm.passthrough` the attribute is in
  the LLVM IR, yet the code object kept `FLOAT_DENORM_MODE_32 = 3` (denormals kept) *(field, E12)*: kernel-wide
  FP32 flush-to-zero is not set this way.
- **Two adjacent `fx.barrier()` become one.** `fx.barrier` emits `gpu.barrier` (S33), and the pipeline's first
  `canonicalize` (S34) erases a barrier that directly follows another. Barriers inside a branch that all lanes of a
  wave take the same way (wave roles) are right only while every wave executes the same number of `s_barrier`: one
  arm with `barrier; barrier` and the other with `barrier; ds_write; barrier` compiled to 4 `s_barrier` instead of
  5, and the first arm's wave read stale LDS *(field, E18)*. Put an empty side-effecting `llvm.inline_asm` between
  the two, and count `s_barrier` per arm in the ISA.
- **Signed `%` truncates while `//` floors** (S6): for a negative dividend `(a // b) * b + a % b != a`. Keep
  index math non-negative, or use unsigned types.
- **Autotune's online cache ignores the kernel source.** `~/.flydsl/autotune/<function name>.json` is keyed on
  key-argument shapes, dtypes, strides, env, toolchain, device and hints (S22), not on the kernel's source or its
  `Config` list. Delete the file or set `FLYDSL_AUTOTUNE_CACHE_DIR` after editing either.

## Evidence

- S1 ROCm/FlyDSL@dd8373349:python/flydsl/expr/primitive.py:1380 `fx.assume` docstring "WIP, unsupported for now"
- S2 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py:644 `from_dlpack` is `dynamic_layout=False`; `assumed_align` :641, `mark_shape_dynamic` :378, `mark_stride_dynamic` :402
- S3 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/inline_asm.py:11 TODO: remove the v_cvt_off_f32_i4 / v_cvt_pk_bf16_f32 inline-asm wrappers
- S4 ROCm/FlyDSL@dd8373349:python/flydsl/extension/coop/block/reduce.py:79 TODO linear cross-warp fold; RAKING_COMMUTATIVE_ONLY planned, contiguous rake not conflict-free :39; commutative op :26
- S5 ROCm/FlyDSL@dd8373349:docs/language/arithmetic_types.md:307 no hint, no ambient fast-math; `contract` example :277; compile hint :289; integer `/` → Float32 :207
- S6 ROCm/FlyDSL@dd8373349:python/flydsl/expr/utils/arith.py:233 integer `/` → int_to_fp + divf; signed `//` floordivsi :246; signed `%` remsi :256
- S7 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py:205 layout-dynamic marks all shapes and non-unit strides; alignment = element size :189; no-unit-stride raise :284
- S8 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:237 num_records 0xFFFFFFFF by default (docstring :277); get_rocm_arch() in s_waitcnt :46, WMMA :192, descriptor flags :244
- S9 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:300 all-int block dims → known_block_size (docstring :338, decorator arg :652); one SharedAllocator :508; value_attrs set unchanged :117
- S10 ROCm/FlyDSL@dd8373349:docs/language/storage_and_allocator.md:166 static mode: one allocation per struct leaf; union: one allocation :169
- S11 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1714 `flyc.compile[hints]` writes into `func.compile_hints`; OPT_LEVEL in cache-key env list :130; dump_asm read only at :911
- S12 ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1003 DAZ attributes passed under a bare "passthrough" key
- S13 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:61 maxnreg → `--amdgpu-num-vgpr`, into binary opts :108; `rocdl.waves_per_eu` :135; `O=2` :64; ARCH first :20
- S14 ROCm/FlyDSL@dd8373349:docs/architecture_guide.md:379 FLYDSL_COMPILE_OPT_LEVEL "Optimization level (0–3)"; FLYDSL_DEBUG_DUMP_ASM "Dump final AMD ISA assembly" :389
- S15 ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py:232 opt_level option; nothing else in python/flydsl reads it
- S16 ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:257 static LDS allocation emits make_ptr with no size check
- S17 ROCm/FlyDSL@dd8373349:python/flydsl/utils/smem_allocator.py:265 legacy SmemAllocator raises "Shared Memory Overflow"
- S18 ROCm/FlyDSL@dd8373349:python/flydsl/runtime/device.py:46 get_rocm_arch reads FLYDSL_GPU_ARCH / HSA_OVERRIDE_GFX_VERSION; built-in fallback :36
- S19 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:799 non-int, non-Attribute aux → None (also :672, :686, :819)
- S20 ROCm/FlyDSL@dd8373349:lib/Conversion/FlyToROCDL/FastMathOps.cpp:36 mapping table holds only math.exp2 → rocdl.exp2
- S21 ROCm/FlyDSL@dd8373349:include/flydsl/Conversion/FlyToROCDL/Passes.td:9 other afn math is routed to device-library calls
- S22 ROCm/FlyDSL@dd8373349:python/flydsl/autotune.py:258 `_make_key`: shape/dtype/stride + arch + toolchain + env, hints :305; file `<fn>.json` :253
- S23 ROCm/FlyDSL@dd8373349:docs/kernel_tuning_guide.md:287 "Constant Python-int bounds make the rewriter unroll the loop and silently ignore `init=`"
- S24 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1274 a `range` with an `init` keyword becomes `scf_range` (:1277-1278); only `range` without `init` takes the auto path
- S25 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:982 `scf_range`: bounds through `_to_index`, `scf.ForOp(start, stop, step, init)` (:987-993)
- S26 ROCm/FlyDSL@dd8373349:docs/kernel_authoring_guide.md:19 "`fx.Index` is deprecated"
- S27 ROCm/FlyDSL@dd8373349:docs/api_stability.md:162 `fx.index_cast` → `fx.Index(x)`, removal declared for v0.4
- S28 ROCm/FlyDSL@dd8373349:python/flydsl/expr/arith.py:46 `"index_cast",  # Deprecated`; `Index` carries no such mark
- S29 ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py:942 `class Index(Integer, ..., width=64, signed=False, ...)`
- S30 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:632 calling a `@flyc.kernel` returns a `KernelLauncher`; `launch` emits the gpu.func at :360 and attaches the call's `value_attrs` at :367
- S31 aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:3311 stage 1 sets `rocdl.waves_per_eu` on the module's gpu.funcs before `launcher.launch` (:3312); stage 2 :5628; `small_m_hgemm.py:1398` before the kernel call at :1406
- S33 ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:73 `barrier()` returns `gpu.barrier(...)`
- S34 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:83 `canonicalize` in the pre-binary pipeline, right after `fly-int-swizzle-simplify`
- S35 ROCm/FlyDSL@dd8373349:python/flydsl/expr/typing.py:2002 `Vector.load` builds `vector.LoadOp(result_type, memref, indices)` with no alignment; `Vector.store(..., alignment=None)` at :2016 passes one
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] f32x4 a*b+c: 2 v_pk_mul_f32 + 2 v_pk_add_f32 by default, 2 v_pk_fma_f32 under fastmath("contract") (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p50_mul_add_default.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p51_mul_add_contract.py)
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] 4 lanes: exp2 default 4 v_exp_f32 + 8 v_cndmask + 4 v_ldexp; exp2 fast 4 v_exp_f32 only; exp fast 4 v_exp_f32 + 8 v_cndmask (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p62_exp2_default.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p63_exp2_fast.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p64_exp_fast.py)
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] 00_origin.mlir argument type (?,?):(?{i64},1) for a raw tensor, (100,1000):(1000,1) via from_dlpack (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p00_base.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p68_from_dlpack_static.py)
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] LLVM IR "amdgpu-flat-work-group-size"="128,128" for block=(128,1,1); "1,256" for a runtime Int32 block (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p00_base.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p55_runtime_block.py)
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] flyc.compile[{'waves_per_eu': 2}] then a recompiling plain call: both LLVM IRs carry "amdgpu-waves-per-eu"="2" (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p54_hints.py)
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] "passthrough" gone after stage 10 (convert-gpu-to-rocdl); "llvm.passthrough" present in LLVM IR attributes (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p10_passthrough_bare.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p10_passthrough_llvm.py)
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] binary metadata: 68 VGPR / 0 spills without hint and with maxnreg=32; vgpr_spill_count 8 with llvm.passthrough amdgpu-num-vgpr=32 (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p57a_wide_tile.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p57b_wide_tile_maxnreg32.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p59_wide_tile_num_vgpr_attr.py)
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] OPT_LEVEL 0 vs 3: final ISA and LLVM IR byte-identical (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p52_opt_level.sh)
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] FLYDSL_DEBUG_DUMP_ASM=1 without FLYDSL_DUMP_IR: no file in FLYDSL_DUMP_DIR or HOME (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p53_dump_asm.sh)
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] 192 KiB LDS: LLVM error printed, exit 0, .pkl cached, .kd group_segment_fixed_size=196608, second run silent (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p35_lds_overflow.py)
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] ARCH=gfx950 without FLYDSL_GPU_ARCH: get_rocm_arch() returned its built-in default, not gfx950, and the WMMA error named that default (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p61_env_names.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p40_wmma_on_gfx950.py)
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] "denormal-fp-math-f32"="preserve-sign,preserve-sign" in LLVM IR; .kd FLOAT_DENORM_MODE_32=3, as without it (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p60_denormal_llvm_passthrough.py)
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] f32x4 .to(fx.BFloat16): 4 v_cvt_pk_bf16_f32 in the ISA, no inline asm (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p69_truncf_bf16.py)
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] examples/04-preshuffle_gemm.py (MFMA 16x16x16 f16, 256 MFMAs): 144 VGPRs, 0 spills without a hint and with maxnreg=64; 128 VGPRs, 94 VGPR spills and 81 scratch instructions with llvm.passthrough amdgpu-num-vgpr=64 (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p70a_mfma_base.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p70b_mfma_maxnreg.py, local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p70c_mfma_num_vgpr_attr.py)
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] a 128x128 bf16 tile GEMM with 4 waves (`AMD-AGI/AKLR@4f56300:.claude/skills/flydsl-kernel-optimization/examples/gemm/step9_occupancy.py`) with `rocdl.waves_per_eu` 4 alone: next_free_vgpr 128, accum_offset 64, agpr_count 64, vgpr_spill_count 96, scratch 164 B; with `amdgpu-agpr-alloc` 0 added: accum_offset 128, agpr_count 0, spills 0 (local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/p81_waves_per_eu_without_agpr_budget.py; local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/out/p81_isa_resources.txt; local:/workspace/code/gpt_oss_docker/work/sikl/flyprobe_b2/out/p81_step9_agpr0_isa_resources.txt)
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=1, one compile and launch on MI355X, `FLYDSL_DUMP_IR=1`] tagging `rocdl.waves_per_eu` 3 by walking `ctx.gpu_module_body` between the kernel call and `.launch()` found 0 gpu.func ops and the LLVM IR had no `"amdgpu-waves-per-eu"`; the same walk after `.launch()`, `value_attrs=` on the kernel call and `compile_hints["waves_per_eu"]` each gave `"amdgpu-waves-per-eu"="3"` (local:/shared_nfs/kyle/moe/kf/runs/tp8_bw_l/knowledge_patch_v2/probe_wpe/probe.py)
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=1 compile per arm, 200 launches per arm on MI355X, `FLYDSL_DUMP_IR=1`] wave 0 `barrier; barrier`, wave 1 `barrier; ds_write; barrier`, then every thread reads wave 1's write: 5 `gpu.barrier` in 04_fly_int_swizzle_simplify.mlir, 4 in 05_canonicalize.mlir, 4 `s_barrier` in the ISA, wave 0 read stale LDS in 200 of 200 launches; with an empty side-effecting inline asm between wave 0's two barriers, 5 / 5 / 5 and 0 of 200 (local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/knowledge_patch/probe/barrier_probe.py, dumps dump_0/ and dump_1/); in an fp8 MLA decode the merge left ~37% of the outputs wrong until the asm went in (local:/shared_nfs/kyle/moe/k3six/runs/mla_fp8/workspace/forge_experiments/lessons/iter_005.md:22; :23)
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only, `FLYDSL_DUMP_IR=1`] `vector<8xf16>` loaded with `vector.LoadOp` from an LDS memref at the runtime element index `8·(tid + shift)`: `alignment` 4 / 8 / 16 gave two `ds_read2_b32` / one `ds_read2_b64` / one `ds_read_b128`; without `alignment` (the f16 element's 2), one `ds_read_b128` (local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/knowledge_patch/probe_align/probe.py, ISA in dump4/, dump8/, dump16/, dump0/)
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile and run, final ISA read, outputs equal] one `fx.ptr_load` of `vector<4xi32>` through `fx.PointerType.get(T.i32, fx.AddressSpace.Shared, align)` at a base plus a kernel-argument offset: align 4 two `ds_read2_b32`, 8 one `ds_read2_b64`, 16 one `ds_read_b128`; with no runtime offset, align 4 and 16 both one `ds_read_b128` (local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/knowledge_patch/probe_lds_align/probe2.py, ISA in d2_4/, d2_8/, d2_16/k_0/21_final_isa.s; probe.py, dump4/ and dump16/)
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=1 per arm, rocprofv3 PMC on one stage-1 call at n_tokens 9397] `alignment=16` on the C-shuffle epilogue's `vector.LoadOp` of staged C: `ds_read2_b32` → `ds_read_b128`, `SQ_LDS_BANK_CONFLICT` 15.38M → 3.64M (EP8, 112 of 896 experts local, inter 3072) and 1.47e7 → 2.98e6 (TP, 896 experts, inter 384), outputs unchanged (fused MoE SiTUv2, fp8 activations × MXFP4 weights, top-16, model dim 3584, 8-wave stage-1 tiles of two 64-row groups; local:/shared_nfs/kyle/moe/k3six/runs/ep8cmp/workspace/forge_experiments/lessons/iter_003.md:25; :30; local:/shared_nfs/kyle/moe/k3six/runs/tp8cmp/workspace/forge_experiments/lessons/iter_018.md:24)
