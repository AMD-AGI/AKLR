---
title: "FlyDSL v0.3.2 compile errors: message, cause, fix"
kind: language-note
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
sources:
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/typing.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py
---

# FlyDSL v0.3.2 compile errors

Look up the last line of the traceback. Every row was reproduced with a compile-only probe (`COMPILE_ONLY=1
ARCH=gfx950`, CPU tensors, one change to `examples/01-vectorAdd.py`); names such as `vector_add_kernel_0`, `y`
or `GRID_PAD` come from the probe. Long messages are cut at `…` after their distinctive part. The loop and branch
rules behind several rows are in `../debug-flydsl-kernel.md` §5.

| Error, as raised | Cause | Fix |
|---|---|---|
| `RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing` | a runtime value meets a Python `if` in a function the rewriter did not transform: only `@flyc.kernel` / `@flyc.jit` bodies are rewritten, a plain helper runs as Python (S1) *(field, E1)* | branch in the decorated body; make the helper branch-free (`.select()`, masks); or decorate the helper `@flyc.jit`, which runs inline while tracing (S2) *(field, E2)* |
| `TypeError: for-loop start must be i32, got i64` | plain runtime `range(a, b, s)` with an `fx.Int64` bound; inferred loops run on `i32` (S3) *(field, E3)* | `fx.Int32` or Python-int bounds; `range(..., init=[...])` takes `Int64` (index loop, S3) |
| `TypeError: Variable(s) ['x'] initialized as None before a dynamic for loop (scf.for). …` (for an `if`: `… before a dynamic if/else (scf.if). …`) | a name set to `None` before a runtime loop or branch is assigned inside it; `None` cannot be carried (S4) *(field, E4)* | give it a typed initial value (`fx.Int32(0)`, `fx.Float32(0.0)`); `if const_expr(...)` when the condition is static |
| `TypeError: for-body: carried variable 'acc' does not match the region entry: expected dtype [IntegerType(i32)], produced [F32Type(f32)]` | a carried name changes type: `acc = 0` is promoted to `Int32`, the body makes it `Float32` (S5) *(field, E5)* | initialise with the loop's type: `acc = fx.Float32(0.0)` |
| `TypeError: ThrCopy.__init__() missing 1 required positional argument: 'thr_idx'` | a method called on a local inside a runtime loop or branch (`thr_copy.partition_S(…)`) makes that local a carried value, and a `ThrCopy` slice cannot be rebuilt from IR values (S19) *(field, E22)* | call it before the loop or branch and use the result inside |
| `NameError: name 'y' is not defined` | a name first bound inside a runtime `if` is used after it; only names bound before the region are carried out (S6) *(field, E6)* | bind a typed value before the `if` |
| `ValueError: Operand 0 of operation "rocdl.exp2" must be a Value (is not a Value)` | an `fx.Float32` (or another FlyDSL numeric) passed to a `rocdl.*` op that FlyDSL does not wrap: most `rocdl` names are the generated MLIR builders (S21); only some (the MFMAs, `readfirstlane`, `ballot`, `readlane`, `rcp`) convert their operands (S22) *(field, E24)* | pass `x.ir_value()` and rewrap the result: `fx.Float32(rocdl.exp2(T.f32, x.ir_value()))` |
| `TypeError: list indices must be integers or slices, not ArithValue` | a Python list indexed by the induction variable of a runtime `range()`, which is an SSA value (S7) *(field, E7)* | `range_constexpr(n)` for loops that index Python containers |
| `TypeError: 'ArithValue' object is not subscriptable` | `results[0]` after `results = yield [x]` in a one-slot `range(..., init=[x])` loop: one carried value comes back bare, two or more as a list (S8) *(field, E8)* | use `results` itself when one value is carried |
| `IndexError: Cannot insert operation at the end of a block that already has a terminator. Did you mean to use 'InsertionPoint.at_block_terminator(block)' …` | a Python generator defined inside the kernel body: the rewriter turns every `yield` in the decorated function, nested `def`s included, into `scf.yield` (S20) *(field, E23)* | define the generator at module level, outside the decorated function |
| `ZeroDivisionError: integer modulo by zero` | `if const_expr(S and K % S == 0)` with `S == 0`: `and` / `or` short-circuit only on a Python `bool` left side, so `K % S` runs (S9) *(field, E9)* | make the guard a `bool` (`S > 0 and …`) or nest two `if`s |
| `AssertionError: unexpected ast node <ast.ClassDef object at 0x…>` | the decorator was applied after the file changed on disk: it re-reads the function's source at the import-time line number (S10, S11); kernels built in a factory hit this when the factory runs after an edit *(field, E10)* | restart the process; keep sources frozen while a process builds kernels |
| `OSError: could not get source code` | the kernel source was `exec`'d from a string; the rewriter needs `inspect.getsource` (S10) *(field, E11)* | write each variant to its own file and import it |
| `RuntimeError: FlyDSL: global 'GRID_PAD' (module '__main__') used by @flyc.jit 'vector_add' changed since first compile (old=('scalar', 0), new=('scalar', 1)). Capturing live globals is unsafe; …` | a module global read by the launcher changed after the first call of the same jit object (S12) *(field, E12)* | pass it as an argument (`fx.Constexpr[...]` if it shapes code), or set it before the first call |
| `RuntimeError: @kernel can only be called inside @jit function` | the `@flyc.kernel` was called from host Python (S13) *(field, E13)* | call it inside a `@flyc.jit` launcher and `.launch(...)` it |
| `RuntimeError: Only one SharedAllocator is allowed per kernel; kernel 'vector_add_kernel_0' already has one` | two `fx.SharedAllocator()` in one kernel (S13) *(field, E14)* | one allocator; one `@fx.struct` (a `@fx.union` to reuse LDS) |
| `ValueError: launch block x=128 differs from known_block_size x=256 declared on kernel 'vector_add_kernel'. This produces an internally-inconsistent IR and is undefined behavior on AMDGPU.` | static `block=` contradicts `@flyc.kernel(known_block_size=...)` (S13) *(field, E15)* | make them equal, or drop `known_block_size`: static dims set it |
| `ValueError: launch smem=1024 is less than the 4096 bytes allocated by SharedAllocator in kernel 'vector_add_kernel_0'` | explicit `smem=` below what `SharedAllocator(static=False)` allocated (S13) *(field, E16)* | `smem=None` (inferred) or at least the allocated bytes |
| `TypeError: Constexpr values support only int, bool, float, str, tuples of those scalar values, and lambdas without free variables` | a list, dict or object passed to an `fx.Constexpr[...]` parameter (S14) *(field, E17)* | pass a tuple of scalars |
| `TypeError: extra: list is neither a JitArgument nor has a registered constructor; cannot derive cache signature.` | an unannotated `@flyc.jit` parameter got a type with no JitArgument (S15) *(field, E18)* | annotate `fx.Constexpr[tuple]` and pass a tuple, or pass tensors / ints / floats |
| `RuntimeError: cannot auto-mark layout-dynamic for tensor shape=(100, 1000) strides=(2000, 2): tensor has no axis with stride == 1; layout-dynamic memref requires one. Use flyc.from_dlpack(t) to wrap as a static memref instead.` | a raw tensor argument with no unit-stride axis (strided slice, `expand`) (S16) *(field, E19)* | `.contiguous()`, or `flyc.from_dlpack(t)` (static layout, one compile per shape) |
| `TypeError: unsupported torch dtype for memref element type: torch.float8_e8m0fnu` | accepted torch dtypes are f16, bf16, f32, f64, bool, u8, i8, i16, i32, i64 and fp8 e4m3fn / e5m2 / e4m3fnuz / e5m2fnuz (S17) *(field, E20)* | pass `t.view(torch.uint8)` and read the bytes as `Uint8` in the kernel |
| `error: <unknown>:0:0: local memory (196608) exceeds limit (163840) in function 'vector_add_kernel_0'` (stderr only, nothing raised) | static LDS above gfx950's 160 KiB; FlyDSL does not check it (S18), LLVM prints this, the call returns and the code object is cached, and a cached rerun prints nothing *(field, E21)* | shrink LDS or alias phases with `@fx.union`; read `.group_segment_fixed_size` in the ISA dump |

## Evidence

- S1 ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py:505 `__bool__` of a dynamic Numeric raises
- S2 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1359 a `@flyc.jit` called during tracing runs its rewritten function inline
- S3 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1021 inferred loop bounds must be i32; the init= form casts any bound to index at :971
- S4 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1030 None before scf.for raises; scf.if variant at :709
- S5 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:103 carried-variable mismatch error; bare scalars promoted at :123
- S6 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:220 only names bound before the region are carried (if at :789)
- S7 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1062 the loop body receives the raw induction variable (:1058)
- S8 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1320 yield returns a list for more than one result, else the single value
- S9 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:549 and/or skip the right side only for a Python bool on the left
- S10 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:266 inspect.getsource, then assert FunctionDef at :268
- S11 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1151 the rewrite runs when `@flyc.jit` is applied (kernel_function.py:484 for `@flyc.kernel`)
- S12 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1207 global drift check raises
- S13 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:624 @kernel outside @jit; one SharedAllocator :508; known_block_size mismatch :313; smem too small :382
- S14 ROCm/FlyDSL@dd8373349:python/flydsl/expr/typing.py:545 Constexpr value types
- S15 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1315 no JitArgument and no constructor for the argument type
- S16 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py:285 no unit-stride axis on a layout-dynamic tensor
- S17 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_argument.py:540 dtype outside the table at :513-533
- S18 ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:257 static LDS allocation has no capacity check
- S19 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:205 `visit_Call` adds the base name of every `x.f(…)` call in a control-flow body to the names it carries (:219-222)
- S20 ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:1330 `visit_Yield` replaces every `ast.Yield` with a `scf_yield_` call; `visit_FunctionDef` (:423) descends into nested functions
- S21 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:17 `from ..._mlir.dialects.rocdl import *` re-exports the generated op builders
- S22 ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:648 `_to_ir` turns a numeric into an `ir.Value`; `rcp` (:768), `readfirstlane` (:877), `ballot` (:882) and `readlane` (:898) call it; the MFMA wrappers unwrap through `_unwrap_mfma_operand` (:171)
- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] RuntimeError reproduced from a plain helper
- E2 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] same helper decorated @flyc.jit compiles, scf.if in the IR
- E3 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E4 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] both variants reproduced
- E5 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E6 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] raised at the use after the if
- E7 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E8 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced; using `results` directly compiles
- E9 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E10 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced with a factory-built kernel; a module-level kernel edited after import compiled
- E11 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E12 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced on the second call
- E13 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E14 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E15 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E16 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E17 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced with a list
- E18 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced; fx.Constexpr[tuple] with a tuple compiles
- E19 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced with A[:, ::2]; flyc.from_dlpack of the same tensor compiles
- E20 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced
- E21 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] printed, exit 0, artifact cached, second run silent
- E22 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced with `thr_copy.partition_S(gB)` inside a runtime `range`; the same kernel with the call before the loop compiles
- E23 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] reproduced with a generator helper defined inside a runtime `range(..., init=[...])` body and iterated there; the same generator at module level compiles
- E24 [2026-10, gfx950, FlyDSL 0.3.2, N=1, compile-only] `rocdl.exp2(T.f32, fx.Float32(1.5))` raised it; the same call with `.ir_value()` compiled
