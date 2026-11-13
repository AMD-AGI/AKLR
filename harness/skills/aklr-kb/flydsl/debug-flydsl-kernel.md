---
name: debug-flydsl-kernel
description: >
  Diagnose a FlyDSL kernel that is wrong, NaN, zero, hanging, or refusing to compile.
  Symptom-indexed: start from what you observed, land on the cause. Covers the stale-cache
  false negative and what the JIT cache key does not see, shape-dependent bounds baked from
  host integers, softmax -inf arithmetic, partition/stride addressing, FP8 error budgets,
  the three loop forms and what each carries, non-short-circuit and/or in traced code,
  buffer_load offset units, and barrier divergence. Use when a FlyDSL kernel produces
  incorrect output or fails to build.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Debugging a FlyDSL kernel

## Before anything else: clear the cache
FlyDSL caches compiled kernels aggressively, and a stale cache makes a correct fix appear not to work.
**Do this before you believe any result**, including a failing one:

```bash
rm -rf "${FLYDSL_RUNTIME_CACHE_DIR:-$HOME/.flydsl/cache}"
```

If the launch wrapper is memoized, clear that too:

```python
compile_my_kernel.cache_clear()      # any @functools.lru_cache on the compile path
```

Everything below assumes you have done this. Debugging that skips it can burn hours
"fixing" a bug that was already fixed.

### What the cache key sees
Clearing is the blunt fix; knowing the key tells you when a result was stale without one. In FlyDSL
v0.3.2 (`python/flydsl/compiler/jit_function.py`) the key of a `@flyc.jit` call is built from:

- the FlyDSL toolchain — the Python sources of `flydsl.{compiler,expr,extension,runtime,utils}` and the
  native libraries (C++ passes) — plus the `.py` files of every directory in `FLYDSL_EXTRA_SOURCE_DIRS`;
- the launcher's source, and the source of the functions it reaches — `@flyc.kernel` / `@flyc.jit`
  functions always; plain helpers captured by closure wherever they live; **plain helpers read as module
  globals only if defined in the same directory**;
- the methods it calls on the class the launcher is defined on;
- closure values that are `int`, `float`, `bool`, `str`, `None`, `tuple` or `Enum`, and module globals
  the launcher reads (scalars and builtin containers by value);
- the GPU target the backend resolved, and the compile hints in effect (`flyc.compile[{...}]`);
- each argument's signature — for a raw `torch.Tensor`: dtype, rank and which axis has unit stride,
  **not its shape** (raw tensors are wrapped layout-dynamic; `flyc.from_dlpack(t)` makes the shape
  static and part of the key); for an `fx.Pointer` (`from_c_void_p`): element type, address space and
  alignment only;
- five environment variables: `FLYDSL_COMPILE_OPT_LEVEL`, `FLYDSL_COMPILE_BACKEND`,
  `FLYDSL_COMPILE_LLVM_DIR`, `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, `FLYDSL_EXTRA_SOURCE_DIRS`.

So these changes do **not** invalidate a compiled kernel: methods of a helper class you instantiate
inside the kernel, plain helpers from another directory read as module globals (a kernel in `kernels/gemm/`
calling into `kernels/common/` is one), functions reached through a module attribute
(`buffer_ops.buffer_load`), a `dict` / `list` / object captured by closure (or anything inside it), and
environment variables your own code reads. A kernel that stores `cfg["v"]` from a closure `dict` keeps the
value it was first compiled with, even in a second process sharing the cache directory, while the same
kernel reading a closure `float` recompiles; a raw-tensor kernel launched at different row counts reuses one
cache entry. Edits there need a new process with `FLYDSL_RUNTIME_ENABLE_CACHE=0` (disables the disk cache;
the in-memory cache stays) or with a clean cache. `FLYDSL_EXTRA_SOURCE_DIRS` pointing at the helper directory
covers only edits to the `.py` files in that directory, not an environment variable your code reads
*(field, E1)*.

A captured `dict` is never in the key: not in one process, not through a second call of the factory, not
across processes sharing the cache directory. A captured scalar enters the key on the jit object's first call
(`_ensure_cache_manager`, `jit_function.py:1238-1242, 1370`): changing it afterwards on the same jit object
reuses the old binary, while a new jit object or a new process recompiles.

Two more cases:
- A module global read through `const_expr` is safe when it lives in the kernel's file or directory:
  changing it after the first compile raises `RuntimeError: global '…' changed since first compile`. A
  global in a helper module **from another directory** is not in the key, so the same jit object silently
  keeps the old branch. Aliasing it to a local of the factory function makes new jit objects and new
  processes recompile; the existing jit object still keeps the old binary.
- A module-level dict that is first filled during tracing (e.g. a cache of generated asm strings) makes the
  next call of the same jit object, `flyc.compile` included, raise `RuntimeError: global '<dict>' … changed
  since first compile`. Fill it before the first call, or keep the cache inside the builder.

Three checks that settle "did my change run?":
- **Break it on purpose.** Make the kernel compute something wrong (scale by 2) without clearing the
  cache. If the old correct result comes back, every earlier measurement of that edit was stale.
- **Identical numbers are a signal.** Variants that should differ but report byte-identical timings, or
  identical `VGPR_Count` / `Accum_VGPR_Count` / `SGPR_Count` in a rocprofv3 kernel trace, may be one
  cached binary; two real variants can also share register counts, so confirm with the next check.
- **Read what was compiled.** In a fresh process, `FLYDSL_DUMP_IR=1` skips the disk cache and writes every
  pass and `NN_final_isa.s` under `$FLYDSL_DUMP_DIR/<kernel>/` (default `~/.flydsl/debug`;
  `python/flydsl/utils/env.py:249-251`, `python/flydsl/compiler/jit_function.py:1414`). How to read the dump:
  `../gemm/mainloop.md` §10.

## Start from the symptom
| What you observe | Causes to check | Section |
|---|---|---|
| Output is all NaN | `-inf` minus `-inf` in softmax, or divide by zero | [§1](#1-all-nan) |
| Output is all zeros | wrong output address, or an unwritten intermediate | [§2](#2-all-zeros) |
| Errors far larger than the data path's precision | missing partitions, or a layout/addressing mismatch | [§3](#3-mostly-wrong) |
| Some shapes partly unwritten, depending on the order shapes ran | a bound baked from a host integer | [§3](#a-bound-or-address-baked-from-a-host-integer) |
| Errors about the size of the data path's precision | FP8 quantization, or a scale near 1 applied twice | [§4](#4-slightly-off) |
| Won't compile, or crashes at trace time | `range` vs `range_constexpr`, loop-state typing, scalar/vector mismatch | [§5](#5-wont-compile) |
| GPU hangs | loop bounds (`step == 0`); on gfx950 a divergent barrier can give stale LDS but does not hang *(field, E2)* | [§6](#6-hang) |

The distinction that matters most is **§3 versus §4**. A large mismatch is structural — you have the
wrong data. A small mismatch is numeric — you have the right data at the wrong precision. Classify before
you dig, but the two cause sets overlap: a scale far from 1 applied twice puts every element far off, and an
addressing bug confined to a tail tile leaves only a few elements wrong.

---

## 1. All NaN

### `-inf` minus `-inf`
When every token in a partition is masked out (past the end of the context), `qk_max` stays at `-inf`.
Then `exp(s - qk_max)` becomes `exp(-inf - (-inf))` = `exp(NaN)` = NaN, and it propagates through the
whole reduction.

```python
safe_diff = (qk_max > NEG_INF).select(diff, ZERO_F)
```

### Divide by zero in normalization
If every probability is zero, `exp_sum` is zero and `1/exp_sum` is `inf`.

```python
safe_sum = (running_sum > ZERO_F).select(running_sum, fx.Float32(1.0))
inv_sum  = fx.Float32(1.0) / safe_sum
```

### Locate it from the host
Do not guess which buffer went bad — print them. Probes like these are temporary: put them in the
kernel or in a scratch script, and remove them before you hand the kernel back.
A submitted kernel carries no `print`, and a probe left behind reads to the next session as
something the operator is supposed to emit:

```python
torch.cuda.synchronize()
print(f"exp_sums   nan={exp_sums.isnan().sum()}  inf={exp_sums.isinf().sum()}")
print(f"max_logits nan={max_logits.isnan().sum()}  range=[{max_logits.min():.4f}, {max_logits.max():.4f}]")
print(f"temp_out   nan={temporary_output.isnan().sum()}")
```

The first buffer in the chain that contains NaN is where to look. A NaN in `max_logits` and a NaN in
`temp_out` are different bugs. When the bug is found, the fix goes into the kernel and the probes do
not.

---

## 2. All zeros

Zeros can mean *the write went somewhere else*, not *the compute produced zero*: rule that out before you
debug the compute.

### Wrong stride
A wrong output stride (such as `stride_out_seq`) sends every store with a non-zero index along that axis to the
wrong address, while index 0 lands right:

```python
print(f"out strides: {output.stride()}, temp strides: {temporary_output.stride()}")
```

### Partition slot versus partition index
If the reduce kernel reads partial results from slots `0 .. grid_z-1` (**`part_z`**), each partition's
output must be written to its slot, not to the absolute partition index: writing by absolute index scatters
results outside the range the reducer looks at. Check which index your reduce kernel reads.

### The intermediate was never written
If the main kernel never writes `exp_sums` / `max_logits`, the reduce kernel faithfully reduces
uninitialized memory. Prove it with a sentinel:

```python
exp_sums.fill_(-999.0)
# ... launch ...
torch.cuda.synchronize()
print(f"exp_sums[0,0,0,:4] = {exp_sums[0,0,0,:4]}")   # must NOT still be -999
```

This distinguishes "wrote zeros" from "wrote nothing", which look identical otherwise.

---

## 3. Mostly wrong

### Missing partitions
If `grid_z < total_partitions` and the kernel handles exactly one partition per CTA with no loop, the last
`total_partitions − grid_z` partitions of the context are silently skipped:

```python
total_parts = math.ceil(context_len / KV_COMPUTE_BLOCK)
print(f"grid_z={grid_z}, total_parts={total_parts}")
assert grid_z == total_parts or kernel_has_multi_partition_loop
```

### The all-1s isolation test
Fill every input with `1.0`. All softmax probabilities become equal and the PV output is exactly
`1.0`, so any deviation is a layout or addressing bug rather than a data-dependent one:

```python
query.fill_(1.0); key_cache.fill_(1.0); value_cache.fill_(1.0)
```

**Know its blind spot.** Uniform inputs give the correct answer regardless of operand ordering, so
this test **cannot** catch V/P operand misalignment in the MFMA. Passing all-1s narrows the search; it
does not clear the layout.

### Single-partition isolation
Force `max_context_partition_num=1` (one-shot mode) to bypass the reduce kernel entirely. If it passes
here and fails with multiple partitions, the bug is in partitioning or reduction, not in the main
compute.

### A bound or address baked from a host integer
A Python `int` folded into the IR — a `num_records_bytes`, a stride, an address constant — belongs to
the one compiled artifact. If it was derived from a shape and reached the kernel through a channel the
cache key does not see (a `dict` in a closure, a helper object's attribute; see
[What the cache key sees](#what-the-cache-key-sees)), every later shape with the same dtype reuses the
first shape's value.

Signature: part of the output is wrong, only for shapes run after the first one compiled. A byte bound
baked from the first shape leaves everything past that byte count unwritten, and only in larger shapes, so
it hides when shapes run largest first and a single-shape test passes. A baked stride sends rows to the
wrong offsets — some left unwritten, others holding wrong values — in every order *(field, E3)*. **The
decisive experiment is to run the same shapes in reverse order:** failures that follow the shape are
shape-specific; failures that follow the position are a stale bound or stride.

Fix: derive every quantity that varies with the shape from the kernel's traced arguments
(`fx.Int32` sizes, the tensor's layout). Keep host constants for what genuinely changes the code —
branch choice, unroll count, vector width — but pass them as `fx.Constexpr` arguments or closure scalars:
the cache key hashes source and those values, not the IR (`jit_function.py:563-593, 1301-1302`), so a vector
width read from a closure `dict` reuses the first compiled binary *(field, E4)*.

### Differential against another backend
```python
torch.testing.assert_close(flydsl_output, gluon_output, atol=5e-3, rtol=5e-3)
```
An element-wise comparison against a Gluon or Triton implementation of the same math localizes the
divergence far faster than reasoning about the layout.

---

## 4. Slightly off

Before treating a small mismatch as a bug, check whether it is the expected error budget.

| Source | Expected magnitude | Verdict |
|---|---|---|
| FP8 data path (Q and P in FP8 for the MFMAs) vs a bf16 reference | max abs error ~1e-3–6e-3 with U(−1,1) inputs, up to ~0.06 with a sharp softmax, where `atol=rtol=5e-3` fails *(field, E5)* | **not a bug** if the kernel matches an emulation of the same FP8 path; the large errors come mostly from quantizing Q |
| Reference uses per-row Q quant, kernel uses per-tensor | relative L2 ~0.7–2% for U(−1,1) or N(0,1) Q, 6–7% for N(0,4²) Q *(field, E6)* | quantization mode mismatch, fix the kernel or the reference |
| `_scale` composition | arbitrary | verify `_scale = softmax_scale * q_scale * k_scale` |

A real bug in this class is **applying `v_scale` twice** — once while scaling the
probabilities and again after the PV product. It scales every output by `v_scale` once more: when `v_scale`
is near 1 the error is small and uniform, and easy to mistake for quantization noise.

---

## 5. Won't compile

### The three loop forms
Only `range_constexpr` unrolls. Both `range` forms become a runtime `scf.for`, so the induction
variable is an SSA value and cannot index a Python list:

```python
for i in range(4):            # WRONG if i indexes a Python list: i is an SSA value
    result[i] = ...

for i in range_constexpr(4):  # CORRECT: i is a Python int
    result[i] = ...
```

What each form carries across iterations (`compiler/ast_rewriter.py`):

| Form | Lowers to | Loop-carried state | Bounds |
|---|---|---|---|
| `range_constexpr(n)` | unrolled at trace time | none needed | Python `int` |
| `range(a, b, s)` | `scf.for` on `i32` | **inferred**: every name bound before the loop and reassigned in the body becomes an `iter_arg`, so `data = next_data` is carried | Python `int` or `fx.Int32`; `fx.Int64` raises `TypeError: for-loop start must be i32, got i64` |
| `range(a, b, s, init=[...])` | `scf.for` on `index` with explicit `iter_args` | only what you `yield`; rebinding a Python name in the body is not carried | Python `int` is fine (becomes an index constant), as is `fx.Int32` / `fx.Int64` |

Python-int bounds do not unroll a `range(..., init=...)` loop or drop `init=`:
`range(0, 15, 1, init=[acc])` emits `scf.for %arg2 = %c0 to %c15 step %c1 iter_args(...)` (`scf_range`,
`compiler/ast_rewriter.py:982-1003`), and a plain `range` does carry a rebound name as an `iter_arg`. A
name that is `None` before a runtime loop, `if`/`else` or `while` and assigned inside it raises `TypeError`:
give it a typed initial value.

### Runtime versus compile-time conditionals
Runtime comparisons in a Python `if` are supported — the rewriter lowers dynamic conditions to
`scf.IfOp`. Write them with DSL operators, not hand-built MLIR predicates:

```python
tid  = gpu.thread_id("x")
lane = tid % fx.Int32(64)

if lane == fx.Int32(0):                       # lowered to scf.IfOp
    fx.printf("lane zero")

val = (lane < fx.Int32(8)).select(good, zero) # runtime predicate for select
```

Only reach for `arith.cmpi(arith.CmpIPredicate.slt, …)` when you are deliberately constructing
low-level MLIR. Passing a condition straight to `scf.IfOp` requires unwrapping the DSL boolean:

```python
cond   = arith.unwrap(partition_idx >= visible_tile_count)
if_op  = scf.IfOp(cond, has_else=False)
```

`const_expr(...)` is for **compile-time** decisions only:

```python
if const_expr(trans_v):
    ...
```

**Do not write `const_expr(lane == 0)`.** Even with `known_block_size`, `gpu.thread_id("x")`, `lane`,
and `warp_id` are runtime SSA values — the compiler knows their *range*, not which lane is executing.

**`and` / `or` short-circuit only on a Python `bool`.** The rewriter turns `A and B` into
`A if (isinstance(A, bool) and A == False) else dsl_and_(A, B)` (`or` likewise with `True`;
`compiler/ast_rewriter.py:541-574`), so with an `int`, `None` or DSL value on the left both sides are
evaluated first. `if const_expr(S and K % S == 0):` raises `ZeroDivisionError` at trace time when `S == 0`,
although the same line is fine in plain Python. When the right side is only legal if the left is true (modulo,
division, indexing, `.shape[i]`), make the guard a `bool` (`S > 0 and K % S == 0`) or nest two `if`s.

**Only `@flyc.kernel` / `@flyc.jit` functions are rewritten.** A plain Python helper the kernel calls
runs as ordinary Python at trace time, so `if <runtime value>:` inside it raises
`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`. Helpers may emit
branch-free code (`range_constexpr` loops, `.select()`, masked loads/stores); a runtime `if` belongs in
the decorated body. Settle this before planning a refactor that moves code into helpers.

**An `if` body is rewritten into its own function.** A value assigned inside the branch is not seen by a
closure that was defined before the `if` and is called inside the branch: the closure reads the old
value. This holds for a runtime `if` and also for a literal `if True:`. With `if const_expr(...)`, a
closure defined inside the branch, or the value passed as an argument, the closure sees the new value. A
name first assigned inside the branch raises `NameError` at trace time. Pass values into closures as
arguments.

### `AssertionError: unexpected ast node <ast.ClassDef ...>`
Raised by `ASTRewriter.transform`. The rewriter re-reads the kernel's source from disk with
`inspect.getsource`, using the line numbers recorded when the module was imported; if the file changed
since (an edit or sync under a long-running server or benchmark process), the lookup lands on whatever
now sits there, for example a decorated class. The rewrite runs when the decorator executes, so it hits a
kernel defined inside a factory function that runs after the edit; a module-level kernel, decorated at import, is
not affected (`language/compile_errors.md`). It says nothing about the kernel — restart the process,
and keep sources frozen for the lifetime of a process that compiles lazily.
A variant `exec`'d from a string fails the same way under the file's name (`OSError: could not get source code`
under any other): in a sweep harness, write each variant to its own file and import it.

### Loop-carried state typing
Pass FlyDSL values (`fx.Int32`, `fx.Float32`, `Vector`) straight into `init=` and `yield`: the rewriter
converts them with `as_ir_value` itself. Unwrap to a raw `ir.Value` only where a low-level helper
demands one. Every slot keeps one MLIR type across `init`, the per-iteration `state`, `yield` and the
post-loop results.

Through `init=` a slot can carry any value of one MLIR type, a raw `ir.Value` included. The inferred forms (a
plain `range` loop, a runtime `if`, `while`) carry only `fx` numerics, `Vector` and raw scalar or `index` values
such as an `ArithValue`; any other raw `ir.Value`, such as a buffer resource or a raw vector, raises
`TypeError: as_dsl_value cannot wrap ir.Value …` at trace time, so carry it through `init=` *(field, E7)*.
The inferred forms also carry a name bound to a `list`, `tuple`, `dict` or `SimpleNamespace` of such
values, nested, as one `iter_arg` per leaf, and rebuild it after the region; a bare Python initial value is
promoted (`0` to `fx.Int32`, `0.0` to `fx.Float32`). A body that changes a leaf's type (an `acc = 0`
reassigned to an `f32`), a sequence's length or a dict's keys raises `TypeError: … does not match the
region entry` (`_promote`, `_flatten_like`, `compiler/ast_rewriter.py:73-127`). `init=` does not unpack
containers: pass one value per slot (`compiler/ast_rewriter.py:992`).

### `buffer_load` offset units
The offset is counted in units of `dtype`, not bytes. For FP8 data whose addresses you computed in
bytes, divide:

```python
k_addr_bytes = ...   # for FP8, elements == bytes
k_4xi32 = buffer_ops.buffer_load(k_rsrc, k_addr_bytes // 4, vec_width=4, dtype=T.i32)
```

Getting this wrong reads from a 4×-off address. Only offsets below a quarter of the buffer stay inside it;
the rest run past its end, where a descriptor given `num_records_bytes` returns 0 and the default
`max_size=True` reads whatever is mapped there or raises a GPU memory access fault *(field, E8)*.

### Vector stores need vector values
```python
Vec(scalar_i32).store(lds_ptr, [idx])                     # WRONG
Vec.from_elements([scalar_i32], fx.Int32).store(lds_ptr, [idx])   # CORRECT
```

---

## 6. Hang

### Loop bounds
`step == 0` (with `start < stop`) never terminates and hangs the GPU. `stop <= start` does not: both `range`
forms compare signed, so the loop runs zero times *(field, E9)*. Print the bounds on the host
before launching — it costs nothing:

```python
print(f"loop: start={part_start}, stop={part_end}, step={cpb}")
```

### Divergent barrier
`gpu.barrier()` requires **every** thread in the workgroup to reach it. If a runtime `if` sends some
threads down another path, gfx950 does not deadlock — `s_barrier` is counted per wave, a wave executes it
when any of its lanes does, and waves that have ended are not waited for — but when whole waves skip a
barrier or run an extra one, the waves pair up the wrong barriers and can read stale LDS *(field, E2)*. Hoist
the barrier out of the conditional, or keep it in wave-uniform arms as below.

Branches that every lane of a wave takes the same way (wave roles) may hold barriers only if each wave
executes the same number of `s_barrier`; a mismatch can give stale LDS instead of a hang. Count them per arm
in the ISA: two adjacent barriers in one arm compile to one (`language/limits.md`, Unclassified).

### Recovery
```bash
rocm-smi                # 100% busy with no progress confirms the hang
sudo amd-smi reset -G -g <gpu_index>   # or reboot; without -g it resets every GPU
```

---

## The workflow, in order
1. Clear the compile cache (`FLYDSL_RUNTIME_CACHE_DIR`). (Everything below is meaningless without this.)
2. All-1s input. Passes? The layout is probably fine and it is a data-dependent bug.
3. Single partition (one-shot). Passes? The bug is in partitioning or the reduce.
4. Host-side prints: shapes, strides, NaN counts.
5. Walk the intermediate buffers in order (`exp_sums` → `max_logits` → `temp_out`) and find the first
   one that is wrong.
6. Still suspecting layout? Trace one thread's addresses by hand — `tid=0` gives `lane16id=0`,
   `rowid=0`, `warp_id=0`, which is tractable on paper.
7. MFMA suspected? Check operand order: `mfma(LHS, RHS, acc)` maps LHS→M and RHS→N. For QK, K is the
   LHS and Q is the RHS.

## Checklist
- [ ] Cleared the compile cache (`FLYDSL_RUNTIME_CACHE_DIR`) after the last code change
- [ ] Edits outside the cache key (helper classes, other-directory helpers, closure dicts) re-run uncached
- [ ] Shape-dependent bounds and addresses come from traced arguments, not host integers
- [ ] `range_constexpr()` wherever the induction variable indexes a Python structure
- [ ] No `const_expr` on a runtime GPU value (`lane`, `warp_id`, `thread_id`)
- [ ] `buffer_load` offset units match `dtype` (bytes ÷ 4 for `i32`)
- [ ] Vector stores pass `Vector` values, not scalars
- [ ] Loop-carried state uses FlyDSL types, unwrapped only at hard boundaries
- [ ] Partial outputs written to the index the reduce kernel reads (`part_z` slot or absolute partition index)
- [ ] `exp_sums` / `max_logits` strides match the real tensor layout
- [ ] Softmax guards `-inf - (-inf)`
- [ ] Division guarded with `select(sum > 0, sum, 1.0)`
- [ ] K/V addressing matches the tensor rank (4-D vs 5-D `trans_v`)
- [ ] MFMA operand order verified — all-1s will not catch this one

## Evidence
- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1 per step, processes sharing a cache directory] a helper in the `FLYDSL_EXTRA_SOURCE_DIRS` directory returning an environment variable changed 1 → 2: a new process still read 1, with `FLYDSL_RUNTIME_ENABLE_CACHE=0` or a cleared cache 2, the same jit object in one process with `FLYDSL_RUNTIME_ENABLE_CACHE=0` still 1; an edit of the helper's source recompiled with the directory set, not without (probe kernel storing the helper's value)
- E2 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, HIP 7.15, N=100 launches per arm, 10 s timeout, LDS read back] a barrier skipped or added under a runtime `if`, 6 arms: 600 of 600 launches completed; half the lanes of each wave diverging: 0 stale reads; a whole wave running an extra barrier, or whole waves skipping the only one: stale LDS on every launch; wave-uniform arms with one barrier each: 0 wrong (1024 workgroups × 256 threads)
- E3 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1 per order, output prefilled with -1] four shapes (64 KiB–4 MiB of output) in one process, value baked from a closure `dict`: `num_records_bytes` largest first all correct, smallest first each later shape unwritten from byte 65536 on with no wrong value; a stride, in each of three orders: every later shape unwritten and wrong in places; each shape alone correct (2-D store kernel)
- E4 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=1 per step, probe output read back and final ISA counted] vector width 4 and 8 compile to 2 and 4 `buffer_load_dwordx4`; width from a closure `dict` compiled at 4, then 8 requested in a new process sharing the cache directory or through a second factory call in one process: the width-4 binary ran; width from a closure `int`: recompiled (element-wise add kernel)
- E5 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=2 seeds × 29 configs, against a bf16 reference and an FP8-path emulation] max abs error 0.0009–0.0057 with U(−1,1) inputs, 0.034–0.0625 with Q×8 (sharp softmax), where `atol=rtol=5e-3` failed on 27–54% of the elements; kernel against the emulation ≤ 0.0039; with Q×8, quantizing Q alone 0.036–0.0625, P alone 0.0044–0.0234 (FP8-KV paged-attention decode, head dim 128)
- E6 [2026-11, gfx950 (MI355X), torch 2.11.0+rocm10.0.0, N=5 seeds (medians), attention in float64 with Q quantized to FP8 per row and per tensor] relative L2 between the two modes 0.74–0.77% with U(−1,1) Q, 2.12–2.19% with N(0,1), 6.38–7.39% with N(0,4²); elements failing `atol=rtol=5e-3`: 0, except 27–57% with N(0,4²); each mode against unquantized Q 0.35–5.4% (batch 4, 8 query heads per KV head, head dim 128, context 256–4096, FP8 e4m3)
- E7 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1 per form and value, outputs read back] a raw `vector<4xf32>` and an `!llvm.ptr<8>` buffer resource: carried correctly through `init=`, `TypeError: as_dsl_value cannot wrap ir.Value …` at trace time in a plain `range`, a runtime `if` and `while`; a raw `ArithValue` (i32, f32): correct in all four forms (single-workgroup probe kernels)
- E8 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=2 per size and descriptor, exit status and read-back] byte offsets passed to `buffer_load(..., vec_width=4, dtype=T.i32)` over FP8 buffers of 64 KiB–256 MiB: only the first quarter of the threads read inside the buffer; `max_size=True`: memory access fault at 64 and 256 MiB, garbage at 64 KiB and 4 MiB; `num_records_bytes`: reads past the end returned 0
- E9 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=5 launches per bounds and form, iteration count read back] `range(10, 2, 1)` and `range(4, -4, 1)` ran 0 iterations, `range(-4, 4, 1)` ran 8, with a plain `range` (`icmp slt i32`) and with `init=` (`icmp slt i64`); `step == 0` not run (runtime `fx.Int32` bounds, one thread)
