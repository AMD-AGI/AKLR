---
title: "FlyDSL 0.2.4 → 0.3.2 differences that change kernel code or its result"
kind: language-note
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/runtime/device.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/__init__.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/arith.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py
  - ROCm/FlyDSL@dd8373349:thirdparty/llvm-build-info.json
---

# FlyDSL version differences

What a kernel author writes differently, or what happens without an error, across the ROCm/FlyDSL tags `v0.2.4`,
`v0.3.0`, `v0.3.1` and `v0.3.2` (`dd8373349`). `#n` is the upstream PR. Check `flydsl.__version__` before reusing
code written against another version. Probes ran on the 0.3.2 wheel (compile-only, gfx950) unless their evidence
line says otherwise; a statement about an older version rests on its source, except E5 and E12.

## Environment variable names (same in 0.2.4 – 0.3.2)

| Purpose | Read from | Ignored |
|---|---|---|
| compile, do not launch | `COMPILE_ONLY` | `FLYDSL_COMPILE_ONLY` |
| compile target | `ARCH` | `FLYDSL_COMPILE_ARCH` |
| arch seen by `fx.rocdl.s_waitcnt`, `WMMA`, buffer-descriptor flags, `get_rocm_arch()` | `FLYDSL_GPU_ARCH`, then `HSA_OVERRIDE_GFX_VERSION`, then `rocm_agent_enumerator`, else a built-in default that is not gfx950 | `ARCH` |
| per-stage IR and ISA dump | `FLYDSL_DUMP_IR`, `FLYDSL_DUMP_DIR` | `FLYDSL_DEBUG_DUMP_IR` |

The compile options' docstring says `FLYDSL_COMPILE_*`, but `compile_only`, `arch`, `dump_ir` and `dump_dir` set
`env_var=` explicitly, which skips the prefix (S1); the other options take `FLYDSL_<GROUP>_<NAME>` (S2).
`FLYDSL_COMPILE_ONLY=1 FLYDSL_COMPILE_ARCH=gfx950` left `compile_only=False arch=''`, and `ARCH=gfx950` alone left
`get_rocm_arch()` at its built-in default, not gfx950 *(field, E1)*: for a compile-only run away from an MI355X set
`COMPILE_ONLY=1 ARCH=gfx950 FLYDSL_GPU_ARCH=gfx950` (S3). Between `v0.2.4` and 0.3.2 `env.py` only gained
`FLYDSL_AUTOTUNE_CONFIG_DIR` (S4).

## Spelling that differs by version

| Write in 0.3.2 | Older form, and what happens with it on 0.3.2 | From | Evidence |
|---|---|---|---|
| `fx.rocdl.make_buffer_tensor(t)` with `fx.rocdl.BufferCopy32b/64b/128b()` atoms, or raw `fx.rocdl.raw_ptr_buffer_load` / `raw_ptr_buffer_store` | `from flydsl.expr import buffer_ops` (0.2.4): `ImportError`; the module moved to the repo's `kernels/common/buffer_ops.py`, which the wheel does not ship | 0.3.0, #880 | S5, S6 *(field, E2)* |
| `fx.Vector` methods, or `flydsl._mlir.dialects.vector` | `flydsl.expr.vector` / `fx.vector` (0.2.4): module gone | 0.3.0, #880 | S5 *(field, E2)* |
| typed `fx.Int32` / `fx.Float32` (`fx.arith.ArithValue` still resolves) | `fx.ArithValue` (0.2.4, deprecated): `AttributeError` | 0.3.0, #925 | S7 *(field, E2)* |
| `fx.rocdl.s_waitcnt(vmcnt=…, lgkmcnt=…, expcnt=…)`, arch-dispatched | raw `rocdl.s_waitcnt(bitfield)` only (0.2.4); the bitfield form still works | 0.3.0, #915 | S8 |
| a list / dict / `SimpleNamespace` carried through a runtime `for` / `if` / `while` | 0.2.4 raises `for-loop variable '…' is list, not an MLIR Value`; carry typed values one by one there | 0.3.0, #874 | S9, S10 |
| `fx.max`, `fx.min`, `fx.ceildiv` | `Numeric.maximumf` / `minimumf` (deprecated in 0.3.2); no `fx.ceildiv` before 0.3.2 | 0.3.2, #995 | S11 |
| `fx.rocdl.BufferCopyLDS32b()`, or `BufferCopyLDS128b()` on gfx950 | `BufferCopyLDS64b()`: raises in 0.3.2; up to 0.3.1 it verified, then failed instruction selection, so no copy happened | 0.3.2, #1023 | S12 *(field, E3)* |

## Behaviour that changes without an error

- **Kernel attributes in `value_attrs`.** On 0.3.2 a bare `passthrough` key is dropped at `convert-gpu-to-rocdl`
  and `llvm.passthrough` reaches the LLVM function *(field, E4)*; on 0.2.4 the reverse held: bare `passthrough`
  kept, `llvm.passthrough` ignored when translating to LLVM IR *(field, E5)*. The LLVM pin moved from `7f77ca0d`
  (0.2.4, 0.3.0) to `e2a39f50` (0.3.1, 0.3.2) in #945 (S13, S14); 0.3.0 and 0.3.1 were not probed. Code written
  for one spelling loses every attribute (`amdgpu-agpr-alloc`, waves, denormal mode) on the other version: read
  the attribute back from `NN_llvm_ir.ll` after any FlyDSL change; for the denormal mode that is not enough, since
  the attribute reaches the IR but not the mode register (`limits.md`).
- **`waves_per_eu` compile hint.** 0.2.4 passes it only as `--amdgpu-waves-per-eu` in the binary options (S15);
  0.3.0 also writes a bare `passthrough` `"amdgpu-waves-per-eu"="N,N"` on each kernel (#785, S28), an exact bound
  that moved a vectorAdd probe's `.amdhsa_next_free_vgpr` from 16 to 169 at N=2 *(field, E12)*; 0.3.1 and later set
  `rocdl.waves_per_eu` in its place, a lower bound only (#945, S16), and the 0.3.2 LLVM IR carries
  `"amdgpu-waves-per-eu"="2"` *(field, E6)*. `maxnreg` is a binary option only in all four versions (S15, S16);
  `limits.md` has its 0.3.2 probe.
- **Block size.** Up to 0.3.0 a static block above 256 lanes without `known_block_size` raised (S17). From 0.3.1
  static block dims set `known_block_size` themselves (#965, S18): the 0.3.2 probe kernel got
  `"amdgpu-flat-work-group-size"="128,128"`, and with a runtime block `"1,256"` *(field, E7)*, not checked by FlyDSL;
  HIP rejects a larger runtime block at launch and the kernel does not run (`limits.md`).
- **Fast `exp2`.** 0.3.2 adds `convert-rocdl-fastmath-ops`, which lowers `afn` f32 `math.exp2` to `rocdl.exp2`
  (#1037, S19); `fx.exp2(x, fastmath="fast")` became a bare `v_exp_f32` *(field, E8)*. The 0.3.1 pipeline has no
  such pass (S20), so `afn` math there goes the device-library route (S21); call `rocdl.exp2(T.f32, x)` on 0.3.1
  for the bare instruction (S22).
- **Containers in a runtime `if`.** 0.3.2 hands each branch containers rebuilt from the carried values (#1003,
  S23); 0.3.1 passed both branches the same Python objects (S24), so an in-place update traced in one branch was
  visible to the other.
- **`flyc.compile` under `COMPILE_ONLY`.** From 0.3.1 it returns `None` without building an executor (#961, S25)
  *(field, E9)*; 0.3.0 went on to build the compiled function after `jf(*args)` (S26).

## Evidence

- S1 ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py:235 `compile_only` reads `COMPILE_ONLY`; `arch` reads `ARCH` :238; `dump_ir` / `dump_dir` :249-251; docstring :228
- S2 ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py:182 env name = `<prefix>_<ATTR>` when `env_var` is unset
- S3 ROCm/FlyDSL@dd8373349:python/flydsl/runtime/device.py:46 FLYDSL_GPU_ARCH, then HSA_OVERRIDE_GFX_VERSION; enumerator / built-in fallback :33-36
- S4 ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py:219 AutotuneEnvManager, the only addition since v0.2.4 (`git diff v0.2.4 dd8373349 -- python/flydsl/utils/env.py`)
- S5 ROCm/FlyDSL@v0.2.4:python/flydsl/expr/__init__.py:18 `fx.vector` import and lazy `buffer_ops` (:21); absent from dd8373349 `expr/__init__.py:22`
- S6 ROCm/FlyDSL@dd8373349:kernels/common/buffer_ops.py:1 the moved module, outside `python/flydsl`
- S7 ROCm/FlyDSL@v0.2.4:python/flydsl/expr/arith.py:19 `"ArithValue"` in `__all__`; not in dd8373349 `expr/arith.py:44` `__all__`
- S8 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:39 keyword s_waitcnt; v0.2.4 `expr/rocdl` has no s_waitcnt wrapper
- S9 ROCm/FlyDSL@v0.2.4:python/flydsl/compiler/ast_rewriter.py:1016 only MLIR-backed values can be loop-carried
- S10 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:73 `_promote` / `_flatten_like` (:93) carry containers and promote bare scalars
- S11 ROCm/FlyDSL@dd8373349:python/flydsl/expr/arith.py:267 `fx.ceildiv` (`max` :247, `min` :257); maximumf deprecated in docs/api_stability.md:165
- S12 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:121 deprecation note: verified, then silently failed instruction selection; raises at :131
- S13 ROCm/FlyDSL@v0.2.4:thirdparty/llvm-hash.txt:1 LLVM 7f77ca0d (same file at v0.3.0)
- S14 ROCm/FlyDSL@dd8373349:thirdparty/llvm-build-info.json:4 LLVM e2a39f50 (same at v0.3.1; bump in 494b2cbd, #945)
- S15 ROCm/FlyDSL@v0.2.4:python/flydsl/compiler/backends/rocm.py:45 waves_per_eu and maxnreg (:47) only as binary CLI options
- S16 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:135 `rocdl.waves_per_eu` set on kernels; maxnreg still a CLI option at :61
- S17 ROCm/FlyDSL@v0.3.0:python/flydsl/compiler/kernel_function.py:296 static block > 256 without known_block_size raises
- S18 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:300 all-int block dims become known_block_size
- S19 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:89 convert-rocdl-fastmath-ops in the pipeline
- S20 ROCm/FlyDSL@v0.3.1:python/flydsl/compiler/backends/rocm.py:90 gpu.module pipeline without the fast-math pass
- S21 ROCm/FlyDSL@dd8373349:include/flydsl/Conversion/FlyToROCDL/Passes.td:9 without the pass, convert-gpu-to-rocdl routes afn math to device-library calls
- S22 ROCm/FlyDSL@v0.3.1:docs/kernel_authoring_guide.md:294 `rocdl.exp2(T.f32, x)` single-instruction exp2
- S23 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:741 each branch gets `_pack_states(state_raw, exemplar)`
- S24 ROCm/FlyDSL@v0.3.1:python/flydsl/compiler/ast_rewriter.py:741 both branches get the same `result_values`
- S25 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1663 `_compile_impl` returns None under compile_only
- S26 ROCm/FlyDSL@v0.3.0:python/flydsl/compiler/jit_function.py:1671 `jf(*args)` then builds the CompiledFunction, no compile_only check
- S28 ROCm/FlyDSL@v0.3.0:python/flydsl/compiler/backends/rocm.py:133 `lower_compile_hints` deletes `rocdl.waves_per_eu` and sets the passthrough `"amdgpu-waves-per-eu"="N,N"` (:137); `rocdl.waves_per_eu` from 494b2cbd (#945)
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] env.compile and get_rocm_arch() read under three env sets
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] ImportError for `from flydsl.expr import buffer_ops`; no flydsl.expr.buffer_ops / flydsl.expr.vector / kernels package / fx.ArithValue; fx.arith.ArithValue present
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] BufferCopyLDS64b() raises ValueError
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] bare passthrough absent after stage 10; llvm.passthrough in the LLVM IR attributes
- E5 [2026-10, gfx950, FlyDSL 0.2.4, N=1, IR dump] bare passthrough kept, llvm.passthrough left on llvm.func and absent from the LLVM IR
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] flyc.compile[{'waves_per_eu': 2}]: "amdgpu-waves-per-eu"="2" in the LLVM IR
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] "128,128" for block=(128,1,1); "1,256" for a runtime Int32 block
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] exp2 fast: rocdl.exp2 after stage 10, 4 v_exp_f32 and no v_cndmask / v_ldexp for 4 lanes
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] `flyc.compile[...](vector_add, A, B, C)` returned None
- E12 [2026-11, gfx950, FlyDSL 0.3.0, 0.3.1 and 0.3.2, N=1 per arm, compile-only, `FLYDSL_DUMP_IR=1`] `flyc.compile[{'waves_per_eu': N}]` on `examples/01-vectorAdd.py`: 0.3.0 put `passthrough = [["amdgpu-waves-per-eu", "2,2"]]` on the kernel, and `.amdhsa_next_free_vgpr` went from 16 to 169 at N=2; 0.3.1 and 0.3.2 set `rocdl.waves_per_eu = 2` and it stayed 16
