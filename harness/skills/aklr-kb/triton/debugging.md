---
title: "Debugging a Triton or Gluon kernel on gfx950: artifacts, resources, compile errors, wrong results, crashes, profiling"
kind: method
ops: [gemm, moe, attention, gdn]
gens: [gfx950]
triton: 3.8.0
---

# Debugging a Triton or Gluon kernel (Triton 3.8.0, gfx950)

How to see what Triton 3.8.0 made of a Gluon or plain-Triton kernel for gfx950 (MI355X) — its IR, ISA, registers and
LDS — check that a construct became the instruction you meant, and diagnose compile errors, wrong results, crashes,
profiling problems and numbers that disagree with AKLR's bench. Everything quoted from a compile below was produced by
Triton 3.8.0 for gfx950 without a GPU; GPU numbers are marked and listed under Evidence. The page-local Evidence records the conditions of source checks and measurements.

## Route here when

- You need a kernel's TTGIR, LLVM IR or ISA, its registers, spills or LDS: §1, §2. Proof that a construct took effect,
  or a change that seems to do nothing: §3.
- A compile fails (`CompilationError`, `error encountered during parsing`, `PassManager::run failed`, an abort,
  `LLD failed to link`, `unhashable type`), or a Triton error's line number does not match the file: §4.
- `python -m aklr validate` reports `numeric_mismatch`, `capture_unsafe`, `nondeterministic` or
  `post_timing_mismatch`, or an output holds NaN or 0xFF bytes where the reference has values: §5.
- `candidate_crash`, `candidate_timeout`, `build_failed` after changing one constant, a GPU memory access fault,
  `out of resource`: §6.
- rocprofv3 aborts or hangs, or you need per-launch times, the cost of LDS conflicts or of layout conversions: §7.
- Eager launches are slow on the host: §8. Your numbers disagree with CI's or a colleague's, or a standalone
  benchmark disagrees with the bench: §9, §11. You want the kernel's time, not only the ratio: §10.

## 1. Get the artifacts

**Compile without a GPU.** `triton.compile` of a `GluonASTSource` (`triton.experimental.gluon._runtime`; an
`ASTSource` for `@triton.jit`) for `GPUTarget("hip", "gfx950", 64)` runs every stage on a cache miss (or with `TRITON_ALWAYS_COMPILE=1`) and returns the `CompiledKernel`
a launch would, with no device. Attributes are keyed by `(argument index,)` (a dict keyed by name is silently ignored)
and parsed by `HIPBackend.parse_attr("DS")`; constexprs apply keyed by `(index,)` or by name
([authoring.md](authoring.md) §10 has the helper). Compile each launch as `run` makes it:

- the signature from the tensors' dtypes (`*bf16`, `*fp32`, `*u8` for uint8 views of MXFP4 and e8m0 bytes, `*fp8e4nv`
  for `torch.float8_e4m3fn`, `*fp8e4b8` for `float8_e4m3fnuz`), ints as `i32`, `i64` from 2^31, `u64` from 2^63;
- what the launch derives from the values (S53): an int equal to 1 becomes the constexpr 1, so pass it among the
  constexprs; `D` marks an int multiple of 16 and a pointer at offset 0 or 256 of an allocation (not 4 or 8), `S` a
  4 KiB storage, not a 2 GiB + 16 KiB one *(field, E3)*. Without them you read another binary: a unit stride passed
  as `i32` turned one `buffer_load_dwordx4` into 8 `buffer_load_ushort` *(field, E2)*, and an int not marked `D`
  narrows masked loads (§3);
- the options spelled right: `triton.compile` silently drops a key that is no `HIPOptions` field (S43), while a
  launch raises `KeyError` (§4);
- on a machine without torch, a stub with one placeholder type per attribute name lets every committed kernel file
  import *(field, E2)*: `stub = types.ModuleType("torch"); stub.__getattr__ = functools.cache(lambda n: type(n, (), {}))`,
  then `sys.modules["torch"] = stub`. A bare module is not enough: host code reads `torch.Tensor`, `torch.dtype` and
  fp8 dtypes at import, and two Qwen3.5 GDN prefill files look an fp8 dtype up in a table there (S53).

**From a launch.** `kernel[grid](*args)` returns the `CompiledKernel` (S11). For an AKLR Point, validate it with a fresh
Triton cache and read every compiled kernel from it:

```bash
TRITON_CACHE_DIR=/tmp/tc_run1 python -m aklr validate --stages audit,bench --devices 0 --out /tmp/rep_run1 <schema.json>
ls /tmp/tc_run1/*/   # per compiled kernel: .source .ttgir .llir .amdgcn .hsaco .json (and .ttir for plain Triton)
```

Validating 083b493a (Gluon) or cf637bff (plain Triton) left only the schema's launched def and AKLR's
`_flush_l2_cache_kernel` in a fresh cache: these baselines compiled no Triton kernel. AKLR imports the schema's
sources as the package `_aklr_triton_<content hash[:16]>` under `<AKLR_KERNEL_CACHE_DIR>/triton/` (S32; default
`${XDG_CACHE_HOME:-$HOME/.cache}/aklr/kernels`, `aklr/ops/README.md` § Environment variables): an MLIR diagnostic's
`<file>:<line>:<col>` points at that package's `kernel.py`, at the line of the schema's `kernel.sources`; a
`CompilationError` counts lines from the function's `def` (§4). With `--base`, the incumbent's kernels land in the
same cache under the same names; the package in each `.amdgcn`'s `.file` path tells them apart, and the report's
`kernel_content_hash` is the candidate's *(field, E8)*.

**What a `CompiledKernel` holds** (S1, S2, S3). In the compile probe, `asm['source']` is TTGIR for Gluon (`#ttg.blocked` layouts) and TTIR for plain Triton (`tt.func`, no `#ttg` encoding); `asm['ttir']` exists only for plain Triton. This checks IR kind and keys, not that `source` precedes every pass *(compile only, E13)*. `asm` also holds `ttgir`, `llir` (function attributes), `amdgcn` (ISA and metadata, §2) and `hsaco`.
`metadata` holds the options and what the compile found: `name`, `num_warps`, `waves_per_eu`, `num_stages`, `enable_fp_fusion`, `allow_flush_denorm`, `llvm_fn_attrs`, `shared` (LDS bytes, §2),
`global_scratch_size`, `hash`, `triton_version`, `target`. `n_regs` and `n_spills` exist only once the binary is
loaded on a GPU, and on HIP `n_spills` is the scratch size / 4, not a spill count (S4): read the amdgcn.

**Environment knobs** (S5, S6, S8, S9). A knob "in the key" gets its own cache entry, so setting it recompiles:

| Variable | Does | In the key |
|---|---|---|
| `TRITON_CACHE_DIR` | where every stage file is cached (default `~/.triton/cache/<hash>/`) | — |
| `TRITON_ALWAYS_COMPILE=1` | compile even on a cache hit | no |
| `TRITON_KERNEL_DUMP=1`, `TRITON_DUMP_DIR` | write each stage of a compile to `<dir>/<source key>/` (default `~/.triton/dump`); a cache hit dumps nothing (below) | no |
| `MLIR_ENABLE_DUMP=1` or `=<kernel name>` | successful small Gluon compile: 29 "IR Dump Before" entries in the order of all 29 registered passes; another kernel's name: none; `MLIR_DUMP_PATH=<file>` writes the dump to a file *(compile only, E13)*. After-failure dumping is not established by this successful-build check (S8 describes it) | yes |
| `MLIR_ENABLE_TIMING=1` | print each pass's time, per pass-manager run | yes |
| `LLVM_IR_ENABLE_DUMP=1` | LLVM's `print-after-all` | yes |
| `AMDGCN_ENABLE_DUMP=1` | print the ISA to stdout during the compile | yes |
| `TRITON_DUMP_MIR=<dir>` | write the kernel's machine IR to `<dir>`; it also turns off kernel-argument preloading (S54; `.amdhsa_user_sgpr_kernarg_preload_length` 10 → 0 on a 3-argument Gluon kernel *(field, E2)*), so that build's ISA differs from the deployed kernel's: diff only builds that both set it | yes |
| `TRITON_DISABLE_LINE_INFO=1` | drop the line table; by default the amdgcn has a `.loc … ; <file>:<line>:<col>` line before each source line's instructions (S54) | yes |
| `TRITON_KERNEL_OVERRIDE=1`, `TRITON_OVERRIDE_DIR` | replace a stage's output by the file of the same name in `<dir>/<source key>/` | no |
| `TRITON_REPRODUCER_PATH=<file>` | write an MLIR reproducer of each pass-manager run to `<file>.<stage>.repro.mlir` (`gluon_to_ttgir`, or `make_ttir`, `make_ttgir_early`, `make_ttgir`; then `make_llir`) | no |
| `TRITON_FRONT_END_DEBUGGING=1` | raise front-end errors unwrapped: an undefined name in a jit body raises the bare `NameError`, with frames in `triton/compiler/code_generator.py`, instead of a `CompilationError` (§4) | no |
| `TRITON_DEBUG=1` | compile device asserts in (§4) | a JIT compile's key changes, and it records `debug=True` |
| `TRITON_F32_DEFAULT` | the input precision Gluon's `mfma` passes on: `tf32` fails Gluon `mfma` on fp32 and fp64 operands, while `mfma_scaled` still compiles (§4) | yes |
| `TRITON_INTERPRET=1` | run `@triton.jit` kernels in Triton's CPU interpreter; `@gluon.jit` ignores it and compiles for the GPU (S55) | — |

**One cache per run.** A dump on a cache hit writes nothing (S7), and a stage override is stored under the normal key,
so it stays in a shared cache after you unset it: give every run its own `TRITON_CACHE_DIR`, as CI does per job (S36).
The dump and override directory is named by the kernel's source key (`src.hash()`, S7), not its options: builds that
differ only in `num_warps` (4 / 8), `waves_per_eu` (0 / 2) or `llvm_fn_attrs` got two cache entries but one dump
directory, holding the later build *(field, E2)*, and an override there replaces both. Compare option variants with
one `TRITON_DUMP_DIR` each, or from `CompiledKernel.asm` or the cache. To test a hand edit of the ISA, put only the
edited `.amdgcn` in the override directory: with the dumped `.hsaco` beside it, the override replaced the rebuilt
binary by the old one (the same md5) while the cached `.amdgcn` showed the edit *(field, E2)*.

## 2. Read registers, spills and LDS from the amdgcn

| Line in `asm["amdgcn"]` | Means |
|---|---|
| `.vgpr_count` (= `.amdhsa_next_free_vgpr`) | arch VGPRs and AGPRs together: `.amdhsa_accum_offset` + `.agpr_count` with AGPRs (256 + 102 = 358), the arch VGPRs without; a wave allocates it rounded up to 8 (358 → 360) *(field, E2)* |
| `.agpr_count` | the AGPR part |
| `.amdhsa_accum_offset` | the arch VGPRs rounded up to 4 (66 → 68, 94 → 96, 134 → 136) *(field, E2)* |
| `.sgpr_count` | the SGPRs the instructions use plus 6 (`.amdhsa_next_free_sgpr` + 6: up to `s73` reads 80) *(field, E2)* |
| `.vgpr_spill_count` | VGPRs spilled to scratch: a 4-warp 128 × 128 `tl.dot` at `waves_per_eu` 2 reads 135, the dwords its 66 `scratch_store`s write (`.sgpr_spill_count` stayed 0) *(field, E2)* |
| `.private_segment_fixed_size` (`; ScratchSize`) | scratch bytes the spills use (352 for those 135) *(field, E2)* |
| `.group_segment_fixed_size` (`; LDSByteSize`) | 0 in a Triton kernel: LDS is sized at launch; read `metadata.shared`. All 1709 Triton kernels that validating the 154 schemas compiled (AKLR's 897 and the baselines') read 0, while 538 of AKLR's use LDS by `metadata.shared` *(field, E2)* |
| `; Occupancy:` | LLVM's waves per SIMD from the registers; it sees no LDS in a Triton kernel and does not round to whole workgroups |

A 4-warp bf16 MFMA probe kernel with a 128 × 256 fp32 accumulator reports `.vgpr_count 438`, `.agpr_count 182` and
`.amdhsa_accum_offset 256` *(field, E2)*.

**Occupancy.** LLVM gives a gfx950 wave 512 / N registers at N waves, rounded down to 8 (168 at 3, 72 at 7), arch VGPRs
first up to 256 (the probe: 256 + 182) *(field, E2)*; LDS is 160 KiB per CU and per workgroup
([lookup.md](../hardware/gfx950/lookup.md), [memory_and_sync.md](../hardware/gfx950/memory_and_sync.md) §4). The VGPR
bound on waves per SIMD, min(8, ⌊512 / roundup8(roundup4(arch VGPRs) + AGPRs)⌋), matched `; Occupancy:` in 14 builds of
66 to 512 registers *(field, E2)*. Workgroups per CU matched min(⌊4 × waves per SIMD / `num_warps`⌋, ⌊163840 /
`metadata.shared`⌋) for 9 of 11 kernels (MI355X, 3 runs); at 32 KiB of LDS 4 fit, not 5, and at 8 warps, 65 VGPRs one
run held 4, not 3 *(field, E7)* ([mainloop.md](../gemm/mainloop.md) §7, [grid_pricing.md](../gemm/grid_pricing.md) §4;
the families' ranges: [kernel_library.md](kernels/kernel_library.md)). **[budget.py](../analysis/budget.py)** computes
both, and each reachable occupancy's register ceiling; its `--isa` reads LDS from the group segment, 0 in Triton: pass
`--lds <metadata.shared>` and `--waves <num_warps>` (`python analysis/budget.py --isa k.amdgcn --waves 4 --lds 32768`).

**[isa_diff.py](../analysis/isa_diff.py)** finds a kernel's instructions by the line `<name>:`, which Triton writes as
`<name>:  ; @<name>`: on a raw Triton dump it prints the resources and 0 instructions. Strip the comment first:

```bash
for f in a b; do sed -E 's/^([A-Za-z_][A-Za-z0-9_]*):[[:space:]]+;.*$/\1:/' $f.amdgcn > $f.s; done
python analysis/isa_diff.py a.s b.s --stream
```

On the probe kernel above, `waves_per_eu` 0 → 2 then reads `agpr 182 → 0`, `vgpr_spill_count 0 → 181`,
`scratch_bytes 0 → 728`, `scratch_store_dword +153`, `scratch_load_dword +153`, `v_accvgpr_write_b32 −165`
*(field, E2)*. Its `lds_bytes` row reads the group segment too, 0 in every Triton build: compare `metadata.shared`.

## 3. Confirm that a construct took effect

What each construct becomes in Triton 3.8.0 for gfx950 *(field, E2)*; a source line's code follows its `.loc` line (§1).

| You wrote | Find in the ISA (or `llir`) | If it is not there |
|---|---|---|
| `gl.amd.cdna4.buffer_load` / `buffer_store`, 16 bytes per lane | `buffer_load_dwordx4 v[..], vN, s[..], 0 offen`, `buffer_store_dwordx4` | the proven per-lane run or pointer alignment is shorter than 16 bytes, or a mask narrowed it (next row) |
| a mask `offs < n` on the contiguous dimension | with `n` a multiple of 16 (the JIT's `D`): one `buffer_load_dwordx4`; otherwise 8 `buffer_load_ushort` and 8 `buffer_store_short` per 16 bf16 bytes | a length the JIT marks `D`, or a mask on rows only (`rr < m` with a runtime `m` kept 2 `buffer_load_dwordx4` on a [64, 64] bf16 tile) |
| `gl.load` / `gl.store` in a Gluon kernel | no buffer memory access in the tested 7 DS-pointer forms and 284 kernels / 432 launches from 150 schemas (no explicit buffer/async API); [64, 64] bf16 copy: 2 `global_load_dwordx4` + 2 `global_store_dwordx4`. Uniform loads can use `s_load`; atomic cache maintenance can include `buffer_inv` / `buffer_wbl2` *(compile only, E16)* | this is a finite compile sample, not a guarantee for every access form; explicit buffer API: [memory.md](gluon/memory.md) §4 |
| `tl.load` in a plain Triton kernel | `buffer_load_*` where `make_ttgir` proves the access in range: a scalar base plus offsets, with the base marked `S` or the byte offset bounded under 2 GiB by range analysis (S30); `global_load_*` otherwise | `x + tl.arange(…)` converts without `S`, `x + pid * BLOCK + tl.arange(…)` only with it ([authoring.md](authoring.md) §2) |
| `async_copy.buffer_load_to_shared` | `buffer_load_dwordx4 vN, s[..], 0 offen lds` | the lowering fails unless each lane copies 128 or 32 bits (S45) |
| `commit_group()`, then `wait_group(n)` | `s_waitcnt vmcnt(k)` before the first `ds_read` of the slot, then `s_barrier` at 4 waves (none at `num_warps=1`); two groups of two copies with `wait_group(1)` give `vmcnt(2)` | a copy without `commit_group()` gets no `vmcnt` wait (§5) |
| `gl.barrier()` | 4 waves: `s_waitcnt lgkmcnt(0)` when an LDS write is outstanding, then `s_barrier`, and no `vmcnt` wait (the tested async copy and an independent `global_load` were issued before it without a completion wait at the barrier; the load is waited on before its later consumer); 1 wave: the same instruction stream as without it | ([memory.md](gluon/memory.md) §11) |
| `gl.amd.cdna4.mfma`, bf16, `instr_shape=[16, 16, 32]` | `v_mfma_f32_16x16x32_bf16`; `[32, 32, 16]`: `v_mfma_f32_32x32x16_bf16` | operand dtypes or `instr_shape` ([mfma.md](gluon/mfma.md)) |
| `mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`, `[16, 16, 128]` | `v_mfma_scale_f32_16x16x128_f8f6f4`; `[32, 32, 64]`: `v_mfma_scale_f32_32x32x64_f8f6f4` | as above |
| `smem.load(DotOperandLayout(0, mma, 8))` / `(1, …)` from row-major bf16 tiles | `ds_read_b128` for operand 0, the transposing `ds_read_b64_tr_b16` for operand 1 (a [K, N] tile) | the tile's shared layout ([layouts.md](gluon/layouts.md) §5) |
| batched `gl.dot_fma` on bf16, operands in `DotOperandLayout(i, BlockedLayout, 0)` | `v_dot2c_f32_bf16`: 48 in `_router_pair_packet` of the DeepSeek-V2 MoE router (S41) | — |
| `cache=".cg"` or `".cs"` on `buffer_load`; `cache_modifier=` the same on `gl.load` | `… offen sc0 nt`; on `gl.load`, `global_load_dwordx4 … off nt`; `.cv`: `sc0 sc1`; `.ca`: no bits | stores: [memory.md](gluon/memory.md) §6 |
| `waves_per_eu=N` | `"amdgpu-waves-per-eu"="N,N"` in `llir`; a cap below the kernel's own count spills (`.vgpr_spill_count`, `scratch_*`; the probe at 2: 181 VGPRs, 153 `scratch_store`s) | at 0 the PyPI wheel adds nothing, the MI355X image's build writes `"0,0"`; the probe kernel's VGPRs per value: [tuning.md](tuning.md) §8 |
| `num_stages`, `schedule_hint` or `matrix_instr_nonkdim` on a Gluon launch; `kpack` | nothing: the same instruction stream for each value; `kpack` is reset to 1 on gfx950 with a warning (S54) | pipeline by hand ([memory.md](gluon/memory.md) §12); the other options' footprints: [tuning.md](tuning.md) §9 |
| `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]`, `allow_flush_denorm=True` | `"amdgpu-sched-strategy"="iterative-ilp"`, `"denormal-fp-math-f32"="preserve-sign"` in `llir` | — |
| `gl.inline_asm_elementwise("v_med3_f32 $0, $1, $2, $3", …)` | the literal `v_med3_f32` (S40) | — |
| `gl.convert_layout(x, L, assert_trivial=True)` | a compile error if the conversion moves data between lanes or warps (§4); a register permutation passes, free for 32-bit data, `v_perm_b32` byte permutes for 8-bit | — |

A change that does not show may sit where nothing compiles it: in a branch the Point does not take (per-Point constants
select constexpr branches, and code written for other Points does not run under AKLR), or after a `return` in the same
block, which Triton does not trace: in DeepSeek-V2 schema d3bbbce5, `_quantized_bytes` returns `packed` before its three
sign-OR lines, so they are dead code (S56). Put `gl.static_print` in the branch (§4) or diff the amdgcn of two builds.

## 4. Compile errors

**Where the message is.**

- Errors while tracing the Python come as `triton.compiler.errors.CompilationError`: `at L:C:`, the jit function's
  source with a caret, then the underlying `NameError`, `AssertionError` or `ValueError`; `break` and `continue` raise
  `UnsupportedLanguageConstruct`. L counts from the function's `def` line (S55): the file line is the `def` line
  + L − 1. An error inside a called `@gluon.jit` function is reported at the call site with no message of its own; the
  callee's `at L:C:` and message are in the exception's cause, which a traceback prints and a `build_failed` detail
  drops (seen for a `@triton.jit` helper). `TRITON_FRONT_END_DEBUGGING=1` raised the bare `NameError` instead (S13).
- A failed IR verification comes as `RuntimeError: error encountered during parsing` (S12), the MLIR diagnostic
  printed to stderr above it: `<file>:<line>:<col>: error: <message>`, with the source line.
- A failure while lowering comes as `RuntimeError: PassManager::run failed`; stderr holds the cause above the printed
  reproducer: the first `<file>:<line>:<col>: error: …` line, or for a C++ assertion in a `make_llir` pass, which MLIR's
  crash reproducer (S47) kept from killing the process, `python: <path>/MFMA.cpp:901: … Assertion '…' failed.`
  (`BlockedLayout` scales to `mfma_scaled`). The closing `<unknown>:0: error: Failures have been detected …` does not.
- An assembler failure in `make_hsaco` comes as `RuntimeError: LLD failed to link hsaco source …` (S54), which can include a linker cause after `because`; stderr has the assembler's `error:` and line (two SGPR operands on a `v_writelane_b32`: `invalid operand
  (violates constant bus restrictions)`, [wave_ops.md](gluon/wave_ops.md) §2.2). LLVM can also print `error: …` and
  still return a binary (an `illegal VGPR to SGPR copy`, same section): read stderr after every compile with inline asm.
- A native crash outside a pass has no Python traceback: an assertion or `LLVM ERROR` in a front-end call aborts
  (exit 134, or −6 for a child process), as `gl.bank_conflicts` and `mfma_scaled` on a rank-3 layout do (below);
  LLVM's code generation in `make_amdgcn` can segfault (139 / −11). `python -X faulthandler` prints the frames (§6).
- Four errors appear only at a launch, not in `triton.compile`: an unhashable constexpr, a misspelled launch option,
  `out of resource`, and `RuntimeError: Global variable <name> has changed since we compiled this kernel` when a
  module-level value the kernel read was reassigned after its first compile (S55).

**Under `validate` you see one line.** The candidate's child process writes to a `task.log` (S57) that reaches
neither the console nor `--out` and is gone after `validate` exits. A dead child leaves its exit code only
(`runner … exited with code -11`, `-6` for an abort); `build_failed` carries a `CompilationError`'s source and caret
but just `RuntimeError: PassManager::run failed` for a lowering error. No MLIR diagnostic, `static_print` or
`device_print` output survives *(field, E8)*: compile the launch alone (§1) or run it outside AKLR (§5) to read them.

Each message below was reproduced for gfx950 (S14 – S21, S28, S31, S44, S48 – S51, S58, S60) *(field, E2)*; the
launch-only ones (`KeyError`, `unhashable type`, `out of resource`) at a launch on MI355X *(field, E3)*:

| Message | Cause | Fix |
|---|---|---|
| `AssertionError("Loop-carried variable acc has initial type <…> but is re-assigned to <…> in loop! …")` | the loop body changes a carried tensor's dtype (an fp32 `acc` made bf16 by `.to()`) | convert it back before the body ends, or create it before the loop in the dtype the body produces |
| `AssertionError("Loop carried variable g changed type, was <class 'triton.language.core.tensor'> but is now <class 'triton.language.core.constexpr'>")`, at the loop's `for` or `while` line | a name that is a constexpr when a `range` or `while` loop starts gets a tensor in the body, which makes it loop-carried: the variable a `gl.static_range` loop left bound (`for g in gl.static_range(PREFETCH)` in a prologue, then `g = block * UNROLL + inner` in the rolled loop), a `g: gl.constexpr` local or a constexpr parameter. "was" is the body's type, "is now" the type before the loop: a tensor name reused as a `static_range` variable in the body reads the other way round (S60) | give the static loop's variable its own name; a value the loop must carry starts as a plain assignment before it (`g = 0` makes a tensor) |
| `AssertionError("Mismatched type for y between then block (<…>) and else block (<…>)")` | a runtime `if` gives a name another dtype in each branch (bf16 and fp32) | one type in both branches; a `gl.constexpr` condition is resolved while tracing and may differ |
| runtime `n`: `ValueError('Did you forget to add @triton.jit ? …')` at `gl.arange`; `AssertionError('int32[] used as tl.static_range start value is not a constexpr')` (or `end value`); `TypeError('get_blocked_layout(): incompatible function arguments …')` for a layout parameter; `TypeError('Shape element 0 must have type constexpr[int], got …')` for `gl.zeros([n])` | these uses require a compile-time value; the decorator is fine *(compile only, E17)* | `n: gl.constexpr` compiled for the tested arange, static-range and layout uses |
| `tensor shape and layout rank mismatch: shape=[512], layout=BlockedLayout(…), shape rank=1, layout rank=2` | a 1-D `gl.arange` given the 2-D tile's layout | `gl.SliceLayout(1, L)` for row indices, `gl.SliceLayout(0, L)` for column indices |
| `error: 'tt.make_range' op Failed to infer return type` (`'arith.constant'` for `gl.zeros`, `gl.full`), then `PassManager::run failed` | a creation op without a layout that no other tensor fixes | give every `gl.arange`, `gl.zeros` and `gl.full` a layout |
| `UnsupportedLanguageConstruct: unsupported AST node type: Break` (`Continue`); ``Cannot have `return` statements inside `while` or `for` statements in triton.`` | no early exit from a loop | put the exit in a `while` condition (`while (i < n) & (i <= LAST):`) or mask the rest of the body |
| `CompilationError` at the helper's call site: failed `gl.static_assert(R == 3, …)` after `if R == 2: return …`, or `gl.sum(x, 1)` on rank-1 `x` after `if R == 1: return …` | the statements after the returning constexpr `if` are still traced in these rank controls *(compile only, E17)* | the same code under `else:` compiled; the Qwen3-Next template instead keeps rank-2 and rank-3 delta rules in separate functions (S56) |
| `ValueError('Layout mismatch in broadcast: <layout A> vs <layout B>')` in a `CompilationError` | 63 differing-layout elementwise/store-value/load-mask builds rejected; 72 controls compiled. Tested Blocked, Slice, AMDMFMA, DotOperand and equal DistributedLinearLayout/Blocked pairs; neither side converted *(compile only, E17)* | `gl.convert_layout` one side, or build both from slices of one layout ([layouts.md](gluon/layouts.md)); AutoLayout not tested |
| `error: Result has an invalid layout: #ttg.blocked<{…}>. Layout has N warps per CTA, but the context requires M warps per CTA.`, then `error encountered during parsing` | BlockedLayout `warps_per_cta` product N differs from launch M: 14 mismatches rejected in 1-D/2-D, M=4/8/default 4 *(compile only, E17)* | derive it from `gl.num_warps()`, or keep layout and launch in one config |
| plain `mfma`: `no matching matrix core intrinsic …`, `Wrong opIdx`, `Incompatible parent encoding`, `mismatching kWidth` / `encoding between A and B operands`, `input_precision … Got tf32` | tested no-instruction shapes: fp16/bf16 [16,16,64] v4, bf16 [16,16,32] v3; A index 1 / B index 0; Blocked parent with kw=0 and MFMA accumulator (nonzero kw fails earlier); kw 8/4 or 4/8; one Blocked operand; fp32/fp64 under `tf32` *(compile only, E17)* | [mfma.md](gluon/mfma.md) §10; these messages are not shared by all matrix APIs |
| `mfma_scaled`: no-instruction e4m3 [16,16,64]/[16,16,32] v4; version-3 layout instead asserts `version == 4 && isF8F6F4`; Blocked parent/operand raises `Expected lhs/rhs layout … parent matching MFMA layout` | swapped A/B indices and unequal kw 16/8 or 8/16 **compiled** for tested e4m3 [16,16,128], None scales (not GPU accuracy evidence); `gl.dot_fma` instead rejects indices while tracing (`a's operand index must be 0` / `b's … 1`) or missing DotOperandLayout *(compile only, E17)* | do not use absence of a compile error as a correctness check; [mfma.md](gluon/mfma.md) §10 |
| `mfma_scaled`: `scale factor must be 16 or 32`, `MFMA.cpp:901` assertion, `LLVM ERROR: Dimensions must match` | wrong scale block; BlockedLayout scales; rank-3 layout with None scales (abort), respectively *(compile only, E2, E17)* | [mfma.md](gluon/mfma.md) §10 |
| `error: 'amdg.buffer_store' op failed to verify that value element type matches the pointed type of ptr` | `buffer_store` of fp32 to a bf16 pointer: it does not cast | `.to(gl.bfloat16)` first (`gl.store` casts) |
| `NameError("Cannot access global variable BLOCK_GLOBAL from within @jit'ed function. …")` | the jit body reads a plain module-level value | a `gl.constexpr` parameter, or the value written inside the jit function (`aklr/ops/README.md` § The kernel section and ABI); Triton also accepts a module-level `BLOCK = gl.constexpr(64)`, which that rule excludes |
| `Shape element d must be a power of 2` | 35 builds with extents 3–384: `arange`, `zeros`, `full` at [96] name d=0; `zeros([64,96])` names d=1; 12 power-of-2 controls compiled *(compile only, E17)* | for the 96-element case, pad to 128 and mask |
| `error: TODO: unsupported scan layout`, then `PassManager::run failed` | `gl.associative_scan` on a tensor in a `SliceLayout` (S49) | convert it to a 1-D blocked layout first |
| `error: Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension` | `gl.split` of an axis of size 2 held by two lanes, not in one lane's registers (S50) | a layout with `size_per_thread` 2 on that axis |
| `error: Lane and warp dim basis must match between source and destination layout.`, then `error encountered during parsing` | `gl.amd.slice` narrower than the layout's CTA tile, such as 8 columns of a layout whose lanes span 16 | slice whole CTA tiles, or convert to a layout whose CTA tile fits the slice |
| `ValueError('Scalar 2147483648 is out of range for type int32')` | a Python int above int32 in an op on an int32 tensor (`x & 0x80000000`) (S51) | write it signed (`-2147483648`) or work in uint32 |
| `ValueError: add with int32 is not supported on …` | `gl.amd.cdna4.buffer_atomic_add` on an int32 pointer | a uint32 counter, or `gl.atomic_add` |
| `rank must be equal to or one less than the shape size. Got 3 and 2` | `allocate_shared_memory(dtype, [2, 64, 64], <rank-3 layout>)`: each `.index(i)` slot is rank 2 | give the 3-D allocation a rank-2 layout (`[1, 0]`) and slot it with `.index(i)` |
| `TypeError: layout conversion from … to … is not trivial.`, with both linear layouts | `assert_trivial=True` on a conversion that moves data between lanes or warps | fix the layouts ([layouts.md](gluon/layouts.md)) or drop the assertion |
| `ValueError: Cache modifier .nt not supported` | loads take `.ca .cg .cs .cv`, stores `.wb .cg .cs .wt` | — |
| `NameError('cache_shared is not defined')` in a `build_failed` finding | one per-Point constant changed alone took a constexpr branch that never defines a name used later | change the coupled constants together (§6) |
| `KeyError: 'Keyword argument waves_per_eus was specified but unrecognised'`, at launch | a launch option that is neither a `HIPOptions` field nor a parameter (S44) | spell it right; a compile-only check ignores it (§1) |
| `TypeError: unhashable type: 'list'`, `'distributed_type'` or `'shared_memory_descriptor_type'`, at launch | every launch hashes its constexpr values, cache hits included (S20; a counting constexpr: 13 hashes in 12 launches); `triton.compile` does not, so a compile-only check passes: a list constexpr `INSTR=[16, 16, 32]` compiled and then failed at launch on MI355X *(field, E3)*. Lists, `gl.distributed_type` and `gl.shared_memory_descriptor_type` raise `TypeError` on `hash()`; `gl.float32`, tuples, `BlockedLayout` and `AMDMFMALayout` hash | tuples for list constexprs (a list inside a layout is fine: layouts hash); for types, pass the pieces as constexprs (`DT=gl.bfloat16, SHAPE=(64, 64), L=<layout>`), build `gl.distributed_type(dtype, list(shape), layout)` in a `@gluon.constexpr_function` and read `T.shape` and `T.layout` from it (it launched on MI355X) |
| abort: `LinearLayout.cpp:292 … Assertion 'it != bases.end() && "inDim not found in layout"'` (swizzled), `LinearLayoutConversions.cpp:1379 … "unknown layout"` (padded) | on AMD, `gl.bank_conflicts` converts the shared layout to a linear layout, which has no case for a padded layout, then asks it for a `vector` input dimension, which a swizzled or shared-linear layout lacks (S21) | do not call it on AMD; time conflicts (§7) |
| `out of resource: shared memory, Required: …, Hardware limit: …` at the first launch | `metadata.shared` above the device's `max_shared_mem` (a workgroup gets at most 160 KiB on gfx950); a 262144-byte allocation compiles without error | smaller tiles or fewer slots; the message's `num_stages` hint does not apply to Gluon |

**Tools.** `gl.static_print("t:", t.type)` prints shape, dtype and layout while the kernel compiles (not on a cache
hit); `gl.static_assert(cond, msg)` stops the compile with your message. `gl.device_print("acc", acc)` prints, through
`__ockl_printf` (S55), a line `pid (x, y, z) idx (i, j) acc: <value>` per element each lane holds; guard it with
`if gl.program_id(0) == p:` (1024 programs: 64 lines, not 65536) *(field, E3)*, run it outside AKLR (`validate` keeps
no output). A Python `assert` and `tl.device_assert` compile to nothing unless the launch has `debug=True` or
`TRITON_DEBUG=1`; then they compile in; with the default `sanitize_overflow=True`, int32 `*` and `+` can add overflow asserts and `s_trap` instructions (checks proved redundant by the compiler may disappear) (S55),
failing with `int32 overflow detected for operation mul` *(field, E2)*: debug with it, time without it.
`TRITON_INTERPRET=1` does not run Gluon (§1). On the host, `print(layout.format_tensor_view(shape))` shows which
thread (`T<n>:<register>`) holds each element, and `format_hardware_view(shape)` the inverse (per warp, register, lane).

## 5. Wrong results

**Start from the failing step**, which opens a finding's detail: `<check>/<i> (<input set>): <output>: …` (S35):

| Finding (step) | Means | First suspect |
|---|---|---|
| `numeric_mismatch` (`numeric/i`, `first_capture/0`) | an output outside its bound on fresh inputs | NaN, or 0xFF bytes in an integer or sub-byte output, where the kernel wrote nothing (below); otherwise indexing, masks, layouts, rounding points |
| `numeric_mismatch` (`large_magnitude/0`) | wrong at large magnitudes only: a GEMM (`dot_product` bound) rounding its output through fp16 passed every other step and failed there with 5248 non-finite (overflowed) elements *(field, E8)* | an intermediate dtype that overflows |
| `capture_unsafe` (`first_capture/1`, `/2`) | a replay of the captured graph on new inputs gave a wrong output (a device counter that changed the output from the 2nd or 3rd replay failed at `/1` or `/2`) | device state kept across calls: counters, sentinels and job descriptors written once instead of in every call ([authoring.md](authoring.md) §8) |
| `capture_unsafe` at any step, detail `<step>: capture: …` | the capture itself failed: a host read of a device value in `run`, or `_do_compile: compiled in the capture` when a launch's constexpr differs from the warm-up call's (S57) | a constexpr or option computed from an input value instead of the Point's shapes, or an int argument that is 1 or a multiple of 16 in one input set only (§1; [authoring.md](authoring.md) §5, §8) |
| `nondeterministic` at `determinism/0` (`bitwise` contracts only) | repeated same-input outputs differ bytewise while inside numeric bounds; an out-of-bound rerun is `numeric_mismatch` | atomics, arrival-order reductions and races are possible causes, not inferred from the label alone |
| `post_timing_mismatch` (`post_timing/i`) | wrong after the timed replays | state kept between calls ([pitfall](../pitfalls/state_kept_across_calls.md)) |

A mismatch's detail reads (S35)
`<step> (<input set>): <output>: violation_fraction …, peak_bound_multiple …, bias_z_score …, <n> non-finite mismatches`.
Before the call it captures, bench fills each output with NaN, or 0xFF bytes for integer and sub-byte dtypes (S57):
non-finite mismatches in a float output can indicate elements the kernel never wrote, or non-finite values it computed (8 unwritten rows of a 083b493a copy:
exactly their 24576 elements) *(field, E8)*. A failure within the violation limit with no non-finite mismatch failed
the bias test, a rounding that leans one way ([numerics.md](numerics.md) §1; `harness/skills/aklr-read-validation`).

**Buffer ops have no tensor bound in the tested configurations.** Two runs each checked 26 Gluon buffer and converted plain-Triton configurations (norm/GEMM, 4 KiB/1 MiB/1 GiB compile shapes and AKLR examples): LLIR/ISA `num_records` was `0x7ffffffe`, independent of those shapes *(compile only, E18)*.
In the tested Gluon/plain mask kernels, masked offsets were `0x80000000`: 24 masked lanes read 0 (Gluon without `other`) or −7 (`other`, both frontends), and stores preserved 1 *(field, E18)*. An unmasked `buffer_load` at offsets 64–79 of a 64-element view read 64.0–79.0, the allocation's next elements *(field, E3)*.
Mask accesses that can leave the tensor ([memory.md](gluon/memory.md) §4); these checks do not cover every descriptor kind.

**LDS ordering, races and waits** ([memory.md](gluon/memory.md) §8, §11).

- `wait_group(n)` waits for committed groups only: a copy issued without `commit_group()` got no `vmcnt` wait before
  the `ds_read` that follows *(field, E2)*, although the docstring says uncommitted copies are waited for at 0 (S24).
- `load_shared_relaxed` marks the read as synced by the async wait, and Triton's barrier pass then puts no barrier
  between that read and a later async copy into the same buffer (S23); with nothing between them, LLVM issued the
  refill before the reads, with one wave as with four. Put a `gl.barrier()` between the last relaxed read of a slot
  and its refill (4 waves: `s_waitcnt lgkmcnt(0)` + `s_barrier`; one wave: only the read-before-refill order, so the
  one-warp MiniMax router adds an inline `s_waitcnt lgkmcnt(0)` with a `~{memory}` clobber), or keep one spare slot
  for a refill issued after the step's reads, two (SLOTS = AHEAD + 2) for one issued at the top of the step.
- Membar orders LDS only: a 4-wave kernel that reads global data and then overwrites it got no `s_barrier` between
  its `global_load` and `global_store`; write a `gl.barrier()` there ([gdn.md](gdn.md) §3). An inline
  `s_waitcnt vmcnt(N)` is a fixed count the compiler neither adjusts nor warns about: one more vector-memory
  instruction after the waited copy, 2 warps or a 64-row tile made `vmcnt(1)` wait for one more than needed, and
  dropping one prefetch in the 128-token DSA file left its panels in flight *(field, E2)*.

**Atomics and determinism.** A `bitwise` contract adds a `determinism/0` step, a rerun of input set `numeric/0` held to
both the numeric bound and bytewise agreement (a `bounded` contract has no bytewise step) *(field, E8)*. A float atomic or a split-K reduced in arrival order can fail
AKLR's `bitwise` comparison, and a rare race can pass two runs: repeat the call many times at the shape you time and
compare bytes (on a 4-row, 4-split float-atomic row sum, 2 calls missed it in 17–18 of 20 batches, 200 calls in none) *(field, E3)*
([../gemm/correctness.md](../gemm/correctness.md) §2–§3, written for FlyDSL GEMMs). AKLR's 65 `bitwise` Triton kernels
hold no atomic, and every atomic in its `bounded` ones is on integers (int32/uint32 counters, cursors and masks; int32
LDS scatters). Bitwise equality is between runs of one kernel: no K split and a six-way split of the n2624 m = 64 GEMM
differ in 75 of 167,936 bf16 outputs, both inside the `dot_product` bound *(field, E3)* ([numerics.md](numerics.md) §8).
Do not assume every layout copy issues one atomic: at `[64,4]`, `BlockedLayout([1,8],[64,1],[1,1],[1,0])` repeated register-copy adds. With 4096 programs, int32/fp32 `gl.atomic_add` and fp32/uint32 `buffer_atomic_add` counted 8192 per active element under no mask, all-true or even-element masks *(MI355X, one final full run, E15)*. The earlier 8-id single-wave lane-copy test counted once; that narrow fact remains.
Built replicas (`x[None, :] + replica[:, None] * 0`) remain distinct in the tested N=8-element, R=2/4/8, one-warp int32/fp32 add controls: R × 4096 without an owner mask, 4096 with `replica == 0` ([moe.md](moe.md) §6). These counts do not establish issue multiplicity for every atomic or layout.

**Numerics switches** ([numerics.md](numerics.md) §6–§7). `gl.exp2` flushes denormals, so `exp2(-127)` is 0 (S25)
*(field, E3)*; `enable_fp_fusion=True` lets LLVM contract `a * b + c` into one `v_fmac_f32` (`False`: `v_mul_f32` +
`v_add_f32`); `["amdgpu-ieee", "false"]` clears IEEE mode (`.amdhsa_ieee_mode 0`); `allow_flush_denorm=True` only sets
`denormal-fp-math-f32` to `preserve-sign` in `llir` and flushes nothing. A timer does not separate them: at one Point
each, validate's speedup moved 0.6 % without `amdgpu-ieee` (Qwen2 router c388, 256 tokens; runs spread 0.9 %), 0.06 %
with `enable_fp_fusion` (DeepSeek-V2 a2bb, 8 tokens), and without `allow_flush_denorm` the 4 DSA 64-head kernels'
code did not change; accuracy passed in every run *(field, E1)*. Decide them by the contract's rounding and NaN rules.

**Small-K MFMA can rescale the incoming accumulator.** In two executions of the same input seed, each with 54 below-operand-extent plain-MFMA variants (`kw ≥ kBase`), K at half/quarter extent returned `acc/2 + a·b` / `acc/4 + a·b` (plain-table errors 3.2e−8–7e−8); instruction K alone is not the boundary *(field, E19)*.
The bf16 `[16,16,32]`, kw8, K16 example has four `fmul … 0.5` per lane in llir, none at K32 *(field, E3)*. For the tested plain variants the extent is `k_width * 64 / nonK` (32 at kw8, 64 at kw16 on `[16,16,32]`).
Do not extend that formula to all small-K calls: the tested below-extent calls with kw below kBase failed to compile; scaled tests had seven compile errors/aborts and three e2m1 kw32 numerical counterexamples *(field, E19)*.
The specific bf16 K16 workaround `acc += gl.amd.cdna4.mfma(a, b, gl.zeros([BM, BN], gl.float32, mma))` returned `acc + a·b` *(field, E3)*.

**Localize a mismatch.**

1. Reproduce the failing step outside AKLR with AKLR's own trial code, as `_calls` in
   `harness/scripts/forge_task/assets/aklr_forge_driver.py` does (a contract without `ranks`):

   ```python
   from pathlib import Path
   import torch
   from aklr.core.contract import semantics
   from aklr.core.contract.registry import ContractRegistry
   from aklr.core.validation.stages._bench._evaluation._request import TrialRequest
   path = "aklr/ops/gemm/impls/triton/gemm_n6144_k2048/f806ad2b086d757d.json"
   registry = ContractRegistry.default()
   schema = registry.parse_schema(Path(path).read_bytes(), path)
   contract = registry.get(schema.contract_id)
   workload = next(w for w in schema.workloads if w.axes == {"m": 16})
   request = TrialRequest.of_workload(path, schema, contract, workload, thresholds={}, incumbent=None, baseline=None)
   args = request.compose_inputs(next(s for s in request.steps if s.name == "numeric/1"), "cuda")  # the step's inputs
   call = {k: v for k, v in request.assemble_call(args, "cuda").items() if k in schema.kernel_call_keys}
   for name in contract.allocated_outputs:
       call[name] = torch.empty(contract.storage_shape_of(name, request.point),
                                dtype=semantics.torch_dtypes(request.point.dtypes[name])[0], device="cuda")
   reference = contract.reference_outputs(request.point, [args], 0, semantics.to_float32)
   absolute = {n: semantics.absolute_value(v, request.point.dtypes[n]) for n, v in args.items() if torch.is_tensor(v)}
   magnitude = contract.reference_outputs(request.point, [args | absolute], 0, semantics.to_float32)  # dot_product
   ```

   Import the kernel as a package over its directory, as AKLR does (`_load_candidate` there). `run` returns nothing:
   call `kernel.run(**call)`, take `outputs = {n: call[n] for n in contract.mutable_tensors}`, then
   `semantics.compare_outputs(request.schema_json, outputs, reference, magnitude_outputs=magnitude)` gives the
   finding's measures (violation_fraction 0.8646 where validate printed 0.865) *(field, E3)*.
2. Bisect a multi-launch op by stage: compare each stage's scratch output with the same stage computed in torch; the
   first one that disagrees beyond its rounding holds the bug.
3. Dump a tile: a store under `if gl.program_id(0) == p:` into a debug tensor wrote only that program's tile, equal to
   torch within fp32 rounding, and a grid of 1 on that tile gave the same tile as the full grid *(field, E3)*; or
   `gl.device_print` it (§4); check where each element lives with `format_tensor_view` (§4). In packed bit arithmetic,
   parenthesize: Python binds `&` looser than `+`, `-`, `*`, so `15 - 3 * (row & 3) - 2 * part & 15` is
   `(15 - 3 * (row & 3) - 2 * part) & 15` (S59).
4. Remove the debug outputs before validating: a kernel writes only its outputs, mutable inputs and scratch
   (`aklr/ops/README.md` § The kernel section and ABI); a debug store into the read-only input `a` of
   gemm_n6144_k2048 failed `validate` as `input_mutation`.

## 6. Crashes and hangs

**`candidate_crash`, "runner … exited with code -11", after one constant changed.** AKLR runs the candidate in a child
process and reports its death by the exit signal (S37); −11 is SIGSEGV, which the compiler can raise as well as the
kernel. Setting the MiniMax-M2 MXFP4 one-token gate-up's `gate_fold` from 4 to 1 crashed every run *(field, E1)*:
`gate_fold = 4 if m <= 2 else …` becomes the launch's `FOLD`, beside `NATIVE_FOLD=m in (1, 8)` and, at m ≤ 2,
`llvm_fn_attrs` with `amdgpu-sched-strategy=iterative-ilp` (S39). Compiled alone without a GPU, that launch segfaults
inside `make_amdgcn` (LLVM's code generation, S29) and compiles once the scheduler attribute is dropped
*(field, E1)*; [moe.md](moe.md) §7 gives the other combinations that compile.

Procedure: compile the changed launch alone (§1) in a subprocess under `python -X faulthandler`, which dumps the Python
frames of a native crash (the FOLD=1 segfault: `make_amdgcn` at its `llvm.translate_to_asm` call; the
`gl.bank_conflicts` and rank-3 `mfma_scaled` aborts: `make_ir` and the Gluon call). In `make_amdgcn`, try dropping
`llvm_fn_attrs` (above; the file's 16-token launch also compiles). Outside AKLR, a host sync after each launch pins the
one that crashes: the sync after a store 2^40 elements away aborted (exit −6), the one before it passed *(field, E3)*.

**`build_failed` with a `NameError` after one constant changed.** A constexpr `if` chain defines a name on some branches
only, and the new constant takes another. In the 64-head DSA decode at 24 tokens, `TILES = cdiv(TOPK, SPLITS * 256)`: 8
splits give each program one 256-key tile (`TILES` = 1) and the branch that allocates `cache_shared`; 4 splits give
`TILES` = 2 and the branch that allocates `cache0`/`cache1`, while the 64-head path still reads `cache_shared` (S38).
The bench reported `NameError('cache_shared is not defined')` in every run *(field, E1)* ([attention.md](attention.md)
§4). Read the host rule that sets a constant and every constexpr branch that reads it before changing it, and guard a
branch that assumes a combination with `gl.static_assert`, so a lone change fails with your message: the 16-token DSA
file states `gl.static_assert(TILES == 1 or PANEL_CACHE)` (S59), the 24-token file nothing.

**`candidate_timeout`.** A batch of one workload whose kernel never ends (a spin on a flag nobody sets) got it after
1200 s, 900 s + 300 s for the one workload (S57) *(field, E8)*. Run the launches outside AKLR under
`timeout -k 5 <s>` with a host sync after each: the one that never returns hangs.

**A GPU memory access fault** comes from the kernel that faulted, not the one that wrote out of bounds: a store past
a tensor into the next allocation synced cleanly, and the fault (exit −6) came after a later kernel used the overwritten value as an index ([fault_in_the_wrong_kernel.md](../pitfalls/fault_in_the_wrong_kernel.md): a host sync after each launch outside capture).
In two final runs, `buffer_load/store` and `gl.load/store` at **0x7ffffc00 bytes from the base** of a 4 MiB tensor faulted; at 4 KiB beyond its end into a pointer-verified adjacent allocation, all four synced without fault, reading or zeroing the neighbour *(field, E18)*. This is not an "only if unmapped" theorem.
Inside the tested mapped 2.5–4.5 GiB tensors, boundary accesses of 1–16 B did not fault; the final byte probe found 93,511 bytes per operation at offset ≥`0x7ffffffe` read zero or were not written. Crossing 8/16 B vectors could partially read/write the in-range portion *(field, E18)*.
i32 byte offsets reaching 2^32 wrapped to a wrong in-range address *(field, E3, E18)*; gate a buffer path on tensor byte extent against `2**31 - 2` ([memory.md](gluon/memory.md) §3). `out of resource: shared memory` at first launch: §4.

## 7. Profiling on the sglang image

- **rocprofv3 1.3.5 aborts on a process that imports torch** in `lmsysorg/sglang:v0.5.22-rocm10-mi35x` (ROCm 10.0.0,
  torch 2.11.0+rocm10.0.0): `: CommandLine Error: Option 'spirv-expand-step' registered more than once!`,
  `LLVM ERROR: inconsistency in registered CommandLine options`, then a hang that SIGTERM does not end, with
  `--kernel-trace` and with `--pmc` alike; `import triton` alone runs *(field, E5)*. `timeout` without `-k` does not
  end it: use `timeout -k 5 <s>` or `kill -9`. So no AKLR schema can be traced or counted with rocprofv3 there.
- **Kineto works.** AKLR's bench times every Point with `torch.profiler` (`measurement_profile`: `kineto`,
  `gpu_duration_sum`), one candidate number per round. For each launch's share, profile a few eager calls under
  `torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])` and read
  `prof.key_averages().table(sort_by="device_time_total")`, shares only: hot, a DSA decode's two launches summed 15 %
  under validate's time *(field, E5)* ([profiler_misses_kernels.md](../pitfalls/profiler_misses_kernels.md)).
- **LDS bank conflicts.** `gl.bank_conflicts` aborts the gfx950 compile (§4; a swizzled and a padded layout), and the
  counters ([lds_bank_conflicts.md](../patterns/lds_bank_conflicts.md)) are untested here. A timed micro-kernel shows
  them: in 4 runs, a plain 64 × 64 bf16 tile read LDS-bound 1.70–1.73× (operand 0) and 1.75–1.77× (operand 1) slower
  than the XOR-swizzled one ([layouts.md](gluon/layouts.md) §5). The recipe: only the construct, in a loop, in enough
  programs to fill the GPU (one program took 305–306 µs, 2048 programs 907–916 µs: an under-filled grid times latency,
  not rate); the median of several rounds (single rounds spread 11–13 %, medians within 2 % across runs); the ISA read
  to confirm what you time *(field, E12)*. Traces where rocprofv3 runs: [profile.md](../profile.md).
- **Layout conversions.** A 16 KiB 32-bit cross-lane conversion need not add ≥3× its control loop's time. In the 1024-program, 1023-iteration probe, DPP/permlane16/permlane32 added about 2.4/2.2/2.2× control (total about 3.4/3.2/3.2×); `ds_bpermute` and the LDS examples added 7.5–9.4×. Register-only rename was compatible with zero added time *(MI355X, 8 balanced repeat clusters in one full run, not 8 independent runs, E14)* ([layouts.md](gluon/layouts.md) §8).
  Read the actual path from ISA: `ds_bpermute_b32`, `v_mov_b32_dpp`, `v_permlane16_swap` / `v_permlane32_swap`, `ds_write` + `ds_read` (plus `s_barrier` when waves move), `v_perm_b32` for an 8-bit register permutation, or none.

## 8. Host-side cost of a launch

What one more launch costs on MI355X, in a replayed graph, under CI's timer and eagerly, measured on a trivial Gluon
kernel: [authoring.md](authoring.md) §7. Eager host time grew 1.0 µs per tensor argument: on HIP, Triton's Python
specialization callback (S26) runs per tensor per launch (0.58 µs a call), and its `S` check `is_within_2gb` ran on
every launch while `AMDGCN_USE_BUFFER_OPS` is on, the default (S27; 300 calls in 300 launches, none with it off)
*(field, E3)*; each launch also hashes its constexprs (§4). Turning `AMDGCN_USE_BUFFER_OPS` off changes plain-Triton
code (its loads stop being buffer ops) and is in the cache key; its effect on host time is not measured.

A replayed graph pays none of the host cost; CI's `speedup` sums the durations Kineto records for the call's kernels,
the flush excluded (S52), so host time does not show (a 2 ms sleep in the callable: +0.25 %), and with queue
profiling no gap between kernels is left to exclude (per extra kernel 4.06 µs against 4.00 µs back to back); eager
calls (an uncaptured prefill) pay the host cost on every launch. The tested AKLR dispatches read at least 4.16 µs;
1-wave spin-kernel floors were 4.22–4.48 µs across four child processes of two runs. Within each process, own times
0.48–2.48 µs read within 0.6% of each other; own times 6/8/12 µs read own time +0.93–1.15 µs *(field, E20)*.
This is a process-dependent tested floor, not a hardware constant ([aklr_dispatch_floor.md](../pitfalls/aklr_dispatch_floor.md)).

## 9. Measured a different build

- **Triton's cache key** holds Triton's version, hashes of its compiler, backend and language files and of
  `libtriton`, the function's own key (its text, its first line number, its callees and the constexpr globals it
  reads), the signature, constexprs and attributes, the target, the options and the cache-invalidating environment
  (S10). Another Triton or an edited source compiles anew; a stale binary comes from a stage override (§1) or from a
  run that never imported your file ([measured_a_different_build.md](../pitfalls/measured_a_different_build.md)).
- **Which Triton ran.** `report.json` → `platform` names Triton, torch, ROCm and FlyDSL. CI requires Triton 3.8.0
  (S36). Validated on a Triton outside a schema's `kernel.requires`, the schema reports `no_matching_toolchain`
  (S42); 135 of the 154 Triton schemas require `>=3.8`, and the other 19 (the 15 N = 3072 fp8 GEMMs, the two plain
  bf16 GEMMs and the two RMSNorms) run on whatever Triton is installed *(field, E2)*.
- **Same version, another build.** The image's Triton 3.8.0 is built from another tree than the PyPI
  `triton==3.8.0` wheel: the padded `gl.bank_conflicts` assertion of §4 sits at `LinearLayoutConversions.cpp:1364` in
  the image and at `:1379` in the wheel and in the v3.8.0 tag *(field, E5)*: confirm compile-only findings on the image.
- **The baseline's aiter.** Most `aklr/ops` baselines call aiter (the bf16 GEMM's is its tuned `gemm_a16w16`, S46). CI
  installs ROCm/aiter at `AKLR_BASELINE_AITER` (dbd8bf5 at 43e112e) over the image's own aiter (e7d2453f2, read from
  the image's build in [measured_a_different_build.md](../pitfalls/measured_a_different_build.md)) and checks that the
  installed `amd-aiter` version carries `+g` and the commit's first 7 characters (S36); `report.json` does not record
  aiter, so print `importlib.metadata.version("amd-aiter")` beside your results. A run on the image's aiter times
  another baseline. Install the pinned one as CI does (without the variable, aiter's setup swaps in its Triton, S36):
  `AITER_USE_SYSTEM_TRITON=1 pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@<AKLR_BASELINE_AITER>"`.
- **An unchanged schema's ratio moves between runs of the same sources**: over three local `validate` runs on one
  MI355X, 2.8 % for the fp8 GEMM 083b493a at m = 16 and 0.1 % for the Qwen2 router c3889614 at 256 tokens
  *(field, E1)*; a ratio at one Point also ranks the baseline there ([tuning.md](tuning.md) §1). `validate --base`
  read an identical copy of each against the served file at 0.999–1.000 (3 runs each) and §10's 1.2 % slower change
  at 0.988 *(field, E1)*: settle a 1–2 % question with it. The flush and ruler: §11.

## 10. Kernel time behind the ratio

`python -m aklr validate` writes `report.json` under `--out` (default `aklr_validation/`, S34). In the bench stage, each
`workload_records[]` entry has `timings[]`: `versus` (`baseline`, or `incumbent`: the kernel served at `--base`),
`n_rounds` (16 or 32), `candidate_milliseconds` and `versus_milliseconds` per round, `log_ratio_mean` (the mean
per-round log ratio), `ratio_lower_bound` (the gate's one-sided bound, each round's log ratio clamped to ±ln 2: a Point
at 2.48× reads 2.000) *(field, E8)*, `min_ratio`. The speedup CI stores (the console prints it to 2 decimals) is that
bound on the unclamped rounds, to 3 decimals (S52; `harness/skills/aklr-read-validation/references/ruler.md` § Judging);
`findings` and `demotions` carry the codes of §5 and §6. Read both sides' medians and the stored speedup (checked on
single-Point reports of 083b493a and bba17a56: one line per timing, medians equal to `statistics.median` of the
per-round arrays):

```python
import json, statistics, sys
from aklr.core.validation.stages._measure import TimingRecord

report = json.load(open(sys.argv[1]))
bench = next(s for s in report["stages"] if s["stage_name"] == "bench")
for record in bench["workload_records"]:
    for t in record["timings"]:
        cand = statistics.median(t["candidate_milliseconds"]) * 1e3
        vs = statistics.median(t["versus_milliseconds"]) * 1e3
        stored = TimingRecord.model_validate(t).speedup_bounds[0]       # what CI stores
        print(record["axes"], f"{t['n_rounds']} rounds: candidate {cand:.2f} us, {t['versus']} {vs:.2f} us, "
              f"speedup {stored:.3f}, gate bound {t['ratio_lower_bound']:.3f} (clamped at 2)")
for f in bench["findings"] + bench.get("demotions", []):
    print(f["code"], f["axes"], f["detail"][-300:])                    # the cause is at the end
```

The candidate's microseconds say what a change did: 3 prefetched K groups instead of 5 in the m = 16 fp8 GEMM
moved the ratio 1.475 → 1.285 and the kernel 4.23 → 4.85 µs; 8 KV splits instead of 16 in gated-attention decode at
32 tokens, 1.79 → 1.129 and 44.18 → 69.77 µs *(field, E1)*. At that GEMM's m = 4096 Point (5ea30df1, 25.6 µs),
`group_m` 4 → 1 read slower in all 4 alternated pairs of kernel times (25.65 → 25.96 µs, 1.2 %), and
`validate --base` stored 0.988 for it *(field, E1)* ([tuning.md](tuning.md) §2).

## 11. A micro-benchmark that disagrees with AKLR's bench

Check, in order:

1. **Time with the bench's timer.** `bench_cudagraph(lambda: kernel.run(**call))` from `aklr.core.utils.testing`, with `call` from §5 and `GPU_FORCE_QUEUE_PROFILING=1`, uses a 1 GiB read flush before each call in the graph, replayed Kineto kernel durations excluding flush, and a median (same 20 ms budget; S52/S57).
   Separate-process readings were within 2.2% of validate at four above-floor Points (fp8 GEMM m4096, DSA f60f22e6, gated attention 32 rows, MiniMax MoE 256 tokens; two runs, two validates each), but −1.9% to +4.1% at floor-bound fp8 n3072/k1536/m16 across six recorded process pairs *(field, E20)*. It is not guaranteed process-to-process agreement.
   Dropping the first reading mirrors validate's unrecorded warm-up; after one eager call it lay within 0.01 µs of the next nine's range in the five tested processes *(field, E3)*.
2. **The flush.** AKLR reads a 1 GiB buffer before each call, inside the graph: "written lines stay dirty, and the
   next kernel pays their write-back" (S33). On a stream reading 64 MB to 1 GiB once per launch (4 warps, 2048
   programs, four eager-event runs), plain loads after a 1 GiB `zero_()` write flush ran at 2.52–4.73 TB/s against
   4.43–6.02 after the case's Gluon read flush (write/read 0.49–0.79); `.cg` was not slowed (4.70–6.64 against
   4.65–6.67, write/read 0.985–1.022) *(field, E9)*. Unhinted read-flushed rates were 9–19% below hot at 64–192 MB,
   largest gap at 128 MB in all four runs; hot/flushed at 512 MB–1 GiB was 0.991–1.014 *(field, E11)*. The write-flush
   probe exaggerated `.cg`/plain (1.40–2.22× versus 1.04–1.12× read-flushed); hot/flushed at 64–192 MB was 1.102–1.224.
   These probes do not directly establish AKLR graph-replay MALL eviction or HBM sourcing *(field, E9)*.
3. **Graph replay against eager.** An eager loop of a short kernel times Triton's host path (§8).
4. **The floor.** Every dispatch measured inside AKLR read about 4.1 µs or more (§8).
5. **The statistic.** The stored speedup is a one-sided lower bound of the mean per-round log ratio over 16 or 32
   unclamped rounds: compare `report.json`'s medians (§10) ([speedup_from_another_ruler.md](../pitfalls/speedup_from_another_ruler.md)).
6. **The inputs.** Bench's timed calls reuse one weight buffer: all 1556 timed calls of 083b493a in the bench process
   got the same weight `data_ptr`, so a cache keyed on the weight's address hits on all but the first *(field, E8)*,
   contrary to `harness/skills/aklr-read-validation/references/ruler.md`.

**未验证 / not established by these final checks:** `source` before every pass, the full original post-`ttgir` pass inventory and after-failure dumps (debugging-17/-26; `d_compile_boundary_controls.04/.05`); universal conversion cost or atomic issue multiplicity (debugging-223/-188; `s_cross_wave_movement_cost.03`, `s_replicated_layout_store_atomic_once.01`); all future Gluon access forms / all buffer descriptor kinds / an "only if unmapped" fault rule (debugging-63/-174/-175/-213/-214; `s_constexpr_folding_and_load_width.03`, `s_jit_key_and_buffer_range.07–.09`); AutoLayout mismatches, other Point variants of returning constexpr branches, and GPU accuracy of accepted scaled-MFMA operand-index/kw mismatches (debugging-118/-120/-124–-127; `s_gluon_compile_rejections.01/.07/.08`); a universal small-K formula (debugging-194; `s_mfma_k_extent_and_fp16_operands.01/.02`); FlyDSL's dispatch floor or guaranteed separate-process timer agreement (debugging-232/-252; `s_aklr_timer_flush_floor.04/.05`). These are scope gaps, not positive recommendations; tested subsets are in E13–E20.

## Evidence

- E16 [2026-11, gfx950, Triton 3.8.0, N=3 compile executions, no GPU timing] `checks/s_constexpr_folding_and_load_width.py` (local verification tool, not shipped) `.03`: 7 DS-pointer access forms and 284 eligible kernels / 432 launches from 150 schemas, no buffer memory accesses or missing compiles; `.05` includes uniform `s_load` controls. Atomic `buffer_inv` / `buffer_wbl2` are cache maintenance. Logs local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_constexpr_folding_and_load_width_r1.log, local:/tmp/kbmig/checks_study/s_constexpr_folding_and_load_width.audit_local.log, local:/tmp/kbmig/checks_study/s_constexpr_folding_and_load_width.audit_fixed_local.log. `.02` remains the supported mask-folding check; CASE status is not a blanket verdict.
- E17 [2026-11, gfx950, Triton 3.8.0, N=2 full compile-only executions, no GPU] `checks/s_gluon_compile_rejections.py` (local verification tool, not shipped) `.01`–`.11`: `.01` logs 54 differing / 54 control builds plus 9 equal-linear-layout mismatches and passing linear controls (audit total 63 / 72); `.02` 14 BlockedLayout mismatches; `.03` 35 non-power-of-2 rejects / 12 controls; `.06/.07` runtime/rank error variants; `.08` FAIL on 32 instances because messages differ by API, with scaled opIdx swaps and unequal kw compiling; `.09` 36 rank-3 scaled aborts / 12 rank-2 controls. Compile acceptance does not establish numerical correctness. Logs local:/tmp/kbmig/checks_study/local_logs/s_gluon_compile_rejections.audit_run1.log, local:/tmp/kbmig/checks_study/local_logs/s_gluon_compile_rejections.audit_run2.log.
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full case executions; final byte probe N=1] `checks/s_jit_key_and_buffer_range.py` (local verification tool, not shipped) `.08.static`: 26 descriptor compile configurations per execution, literal 2147483646; `.08`: 40 active / 24 masked lanes, sentinel `0x80000000`, masked values 0/−7 and stores untouched. `.09`: four far accesses at offset 0x7ffffc00 from a 4 MiB tensor's base faulted; four accesses 4 KiB beyond its end into a verified adjacent allocation did not. `.07` crossing-vector whole-discard criterion FAIL (partial in-range accesses), no fault in mapped 2.5–4.5 GiB tensors; final `.07.bytes` PASS: 93,511 bytes/operation at or beyond 2^31−2 were zero/unwritten, crossing vectors partial, 2^32 offsets wrapped. Logs local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_jit_key_and_buffer_range.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_jit_key_and_buffer_range_r1.log. Byte samples and 26 compile configurations are not independent full runs.
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full executions, same random seed] `checks/s_mfma_k_extent_and_fp16_operands.py` (local verification tool, not shipped) `.01` PASS: 138 plain variants/execution, 54 below-extent, acc scale 1/2 or 1/4, errors 3.2e−8–7e−8. `.02` FAIL: 32 scaled variants, 7 compile errors/aborts and 3 e2m1 kw32 numerical counterexamples: [16,16,128] K32/K64 errors 1/0.95 and [32,32,64] K32 error 0.93. Operand extent rule is finite plain-MFMA evidence, not a universal instruction-K or scaled rule. Logs local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/810_s_mfma_k_extent_and_fp16_operands.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_mfma_k_extent_and_fp16_operands_r1.log.
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 full strengthening executions, 2 spin child processes/execution, no FlyDSL] `checks/s_aklr_timer_flush_floor.py` (local verification tool, not shipped) `.04` FAIL on the fixed floor band: child floors 4.228/4.409 and 4.476/4.375 µs; own 0.48–2.48 µs read within 0.6%, 6/8/12 µs read own +0.93–1.15 µs. `.05` PASS then FAIL: four above-floor Points agree with validate within 2.2%; floor-bound fp8 n3072/k1536/m16 reaches +4.1%. With the recorded historical `checks/bench_launch_cost.py` (local verification tool, not shipped) `.17` and `ab_noise_a_vs_a.14` results, m16 spans −1.9% to +4.1% across 6 process pairs (not 6 full runs of this case). Logs local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_s_aklr_timer_flush_floor.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_aklr_timer_flush_floor_r1.log; historical logs are E1/E3. Minimum 4.16 µs belongs to the historical Gluon/torch controls, not a FlyDSL measurement.
- E13 [2026-11, gfx950, Triton 3.8.0, N=1 full compile-only case run, no GPU] `d_compile_boundary_controls.04`: Gluon `source` has TTGIR layouts, plain Triton `source` has TTIR, only plain Triton has `ttir`; `.05`: the successful Gluon build's 29 before-pass dumps match all 29 registered passes in order (two timing reports are not two independent runs); `.01`: default/True emit one `v_fmac_f32`, False/environment-off emit one mul + one add, even though saved llir can match. Case `checks/d_compile_boundary_controls.py` (local verification tool, not shipped); log local:/tmp/kbmig/checks_study/local_logs/d_compile_boundary_controls.local.log. The supported fp-fusion statement is unchanged.
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 final full GPU run; 8 consecutive balanced repeat clusters, each 13 complete blocks × 5 launches for `.03` conversions, no independence established] `checks/s_cross_wave_movement_cost.py` (local verification tool, not shipped) `.03` FAIL with correct values: 16 KiB fp32 tile, 1024 programs, 1023 iterations; DPP/permlane16/permlane32 added/control ≈2.4/2.2/2.2, total/control ≈3.4/3.2/3.2; bpermute/LDS4/LDS1 added/control 8.84/7.53/9.36. Rename cluster total/control 0.99388, t95 [0.98304,1.00484], compatible with zero added time, not a demonstrated benefit. Historical E9 numbers are total/control for those paths, not incremental cost; no universal cost floor or table-wide simultaneous conclusion follows. Log local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_cross_wave_movement_cost_r2.log.
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 final full GPU run; 4096 programs per configuration] `checks/s_replicated_layout_store_atomic_once.py` (local verification tool, not shipped) `.01` FAIL: 48 exact active-counter=8192 counterexamples across four register-copy layouts × four add ops × three masks (expected once=4096); 135 plain-Triton cells and 12 built-replica controls passed. Built controls have N=8 elements (not 8 full runs), R=2/4/8, one warp, int32/fp32: R × 4096 unowned, 4096 owner-masked. Idempotent atomic readback is not proof of issue count. Earlier E3 lane-copy counts remain limited to that test. Log local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/850_s_replicated_layout_store_atomic_once_r2.log.
- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, one run of each case: python -m aklr validate --stages audit,bench on one GPU; a changed copy in 4 runs alternated with its unchanged schema (2 per DSA kernel for `allow_flush_denorm`), an unchanged schema 3 times against the baseline and 3 times with `--base`, a planted slower copy in 4 alternated pairs] DSA sparse MLA decode n_heads64 at n_tokens=24 (2ee56a6d) with KV splits 8 → 4: `build_failed`, `CompilationError … NameError('cache_shared is not defined')`, in 4 of 4 runs; MiniMax-M2 MXFP4 E256 at n_tokens=1 (655723f5) with `gate_fold` 4 → 1: `candidate_crash`, "runner … exited with code -11", in 4 of 4 runs; the unchanged schemas passed 4 of 4; that FOLD=1 launch compiled alone without a GPU segfaulted in `make_amdgcn` (2 of 2) and compiled without `llvm_fn_attrs` (2 of 2), as did the file's 16-token launch (case triton/checks/ab_coupled_constants_break.py (.04–.08), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/400_ab_coupled_constants_break.log); gemm_a8w8_blockwise_scaled n3072 k1536 at m=16 (083b493a), `PREFETCH` 5 → 3: speedup 1.475 → 1.285, candidate 4.232 → 4.851 µs, every pair ×1.143–1.150 slower; gated attention decode at n_tokens=32 (d9cbde23), KV splits 16 → 8: 1.79 → 1.129, 44.18 → 69.77 µs, every pair ×1.574–1.586 slower; Qwen2-MoE router c3889614 at n_tokens=256, `["amdgpu-ieee", "false"]` removed from the routing launch: 1.910 → 1.921 (B/A 1.006), A's four runs spread 0.94 %; DeepSeek-V2 MXFP4 MoE a2bb3d83 at n_tokens=8, `enable_fp_fusion` False → True on its 3 launches: 1.147 → 1.146 (B/A 0.9994); `allow_flush_denorm=True` removed from the 4 DSA 64-head decode kernels (12, 16, 24, 32 tokens): B/A 1.0007, 1.0033, 0.9986, 0.997, instructions and `.amdhsa_` fields identical in 8 of 8 kernels (compiled for gfx950); accuracy passed in every run of these three (case triton/checks/ab_numerics_sched_switches.py (.01–.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_numerics_sched_switches.log; case triton/checks/ab_gemm_m16_prefetch_depth.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gemm_m16_prefetch_depth.log; case triton/checks/ab_gattn32_kv_splits.py (.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_gattn32_kv_splits.log); the m = 16 fp8 GEMM's (083b493a) baseline ratio 1.4784, 1.4386, 1.4749 (max/min 1.028), the 256-token Qwen2 router's (c3889614) 1.9259, 1.9242, 1.9247 (1.001); an identical copy of each against the incumbent with `--base`: stored 0.999, 0.999, 1.000 for both (check .05 fails its second condition only: that 0.1 % spread is not below the router's baseline spread); the fp8 GEMM n3072 k1536 at m = 4096 (5ea30df1), `group_m` 4 → 1: A 25.648, 25.643, 25.661, 25.648 µs against B 25.932, 25.974, 25.971, 25.951 µs, B above A in 4 of 4 pairs (+1.20 %), `--base` stored speedup 0.988 (case triton/checks/ab_noise_a_vs_a.py (.04–.06), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/700_ab_noise_a_vs_a.log)
- E2 [2026-11, gfx950, Triton 3.8.0, N=1, compile only (triton.compile for gfx950, the amdgcn and llir read) or a count over the committed schemas; each case run once on the MI355X image, or without a GPU where no log is named] the knobs of §1 (`MLIR_ENABLE_DUMP=1`: 29 "IR Dump Before", none under another kernel's name; `TRITON_DUMP_MIR`: `.amdhsa_user_sgpr_kernarg_preload_length` 10 → 0; `waves_per_eu` 0 / 2, `num_warps` 4 / 8 or `llvm_fn_attrs`: two cache entries, one dump directory holding the later build; an override with the dumped hsaco beside the edited amdgcn kept the original md5; `debug=True`: 3 `s_trap`, the int32 overflow asserts, 0 without); a torch stub with cached placeholder attributes: 154 of 154 kernel.py files imported, 131 failed with a bare module; the §2 rows on 14 builds (`.vgpr_count` = `.amdhsa_accum_offset` + `.agpr_count`: 256 + 102 = 358, rounded to 360; accum offset 66 → 68, 94 → 96, 134 → 136; `.sgpr_count` = next free SGPR + 6, s73 → 80; a 4-warp 128 × 128 `tl.dot` at `waves_per_eu` 2: 135 spilled VGPRs, 66 scratch stores, 352 scratch bytes; the VGPR bound equal to `; Occupancy:` in all 14) and the 4-warp bf16 MFMA probe (438 VGPRs with 182 AGPRs, 168 at `waves_per_eu` 3 and 72 at 7; `isa_diff.py --stream` of 0 against 2: agpr 182 → 0, 181 spills, 728 scratch bytes, scratch stores and loads +153 each, `v_accvgpr_write_b32` −165); the §3 rows (a unit stride as `i32`: 8 `buffer_load_ushort`, as the constexpr 1: one `buffer_load_dwordx4`; masked loads with `n` marked D: 1 `buffer_load_dwordx4`, without: 8 `buffer_load_ushort`; a row-only mask with a runtime `m`: 2 `buffer_load_dwordx4`; a Gluon [64, 64] bf16 copy: 2 `global_load_dwordx4`, no buffer op; `num_records` 0x7ffffffe in four Gluon kernels; `wait_group(1)`: `vmcnt(2)` + `s_barrier` at 4 warps, none at 1; an inline `vmcnt(1)` left with 2 instructions behind the copy; a 4-wave global read-then-write: no `s_barrier`; 48 `v_dot2c_f32_bf16` in `_router_pair_packet`; `waves_per_eu` 2: 181 spilled VGPRs, 153 scratch stores; `"amdgpu-waves-per-eu"="0,0"` on the image); the messages of §4 and §5; 135 of 154 Triton schemas requiring `>=3.8`; 1709 Triton kernels compiled by validating each of the 154 schemas at its Points on MI355X (one Triton cache per schema, the cached `.amdgcn` and `.json` read), 897 of them AKLR's: `.group_segment_fixed_size` non-zero in 0, `metadata.shared` > 0 in 538 of AKLR's (case triton/checks/rt_census_compiled_kernels.py (.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/860_rt_census_compiled_kernels_merge.log; case triton/checks/api_offline_compile_and_dumps.py (.05, .06, .10, .14, .21, .24, .27, .28, .29), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_api_offline_compile_and_dumps.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/650_api_offline_compile_and_dumps_r1.log; case triton/checks/rt_aklr_audit_uuid.py (.25, .26), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/600_rt_aklr_audit_uuid.log; case triton/checks/lower_resource_metadata.py (.03–.10), local:/tmp/kbmig/audit0_runs/lower_resource_metadata.log; case triton/checks/lower_waves_per_eu_cap.py (.04–.07), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_waves_per_eu_cap.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/650_lower_waves_per_eu_cap_r1.log; case triton/checks/lower_vector_width_contiguity.py (.01–.03), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_vector_width_contiguity.log; case triton/checks/lower_buffer_ops.py (.01–.07), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_buffer_ops.log; case triton/checks/lower_pointer_vs_buffer.py (.01); case triton/checks/lower_plain_triton_passes.py (.14–.16), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_plain_triton_passes.log; case triton/checks/lower_async_copy_dma.py (.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_async_copy_dma.log; case triton/checks/lower_async_wait_groups.py (.03, .06–.08), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_async_wait_groups.log; case triton/checks/lower_barriers_membar.py (.01, .02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_barriers_membar.log; case triton/checks/lower_relaxed_ring_refill.py (.03–.05, .20, .21), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_relaxed_ring_refill.log; case triton/checks/lower_mfma_instruction_table.py (.05, .06), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_instruction_table.log; case triton/checks/lower_lds_operand_reads.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_lds_operand_reads.log; case triton/checks/lower_dot_fma.py (.01, .02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_dot_fma.log; case triton/checks/lower_cache_modifier_bits.py (.01, .02), local:/tmp/kbmig/audit0_runs/lower_cache_modifier_bits.log; case triton/checks/lower_gluon_ignores_options.py (.03, .09–.13), local:/tmp/kbmig/audit0_runs/lower_gluon_ignores_options.log; case triton/checks/lower_llvm_fn_attrs_sched.py (.01, .02); case triton/checks/lower_inline_asm_operands.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_inline_asm_operands.log; case triton/checks/lower_convert_layout_paths.py (.02–.04); case triton/checks/api_frontend_tracing_errors.py (.01–.27), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_api_frontend_tracing_errors.log; case triton/checks/api_layout_rejections.py (.01–.16), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_api_layout_rejections.log; case triton/checks/api_mfma_errors.py (.01–.06); case triton/checks/api_compile_failure_reports.py (.01–.09); case triton/checks/api_bank_conflicts_abort.py (.01); case triton/checks/lower_associative_scan.py (.01, .02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_associative_scan.log; case triton/checks/lower_amd_slice_split_join.py (.02, .03); case triton/checks/lower_lds_alloc_liveness.py (.06, .07), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_lds_alloc_liveness.log; case triton/checks/lower_mfma_fp32_fp64_precision.py (.01), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_lower_mfma_fp32_fp64_precision.log; case triton/checks/census_kernel_inventory.py (.28), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_census_kernel_inventory.log)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, case runs on one GPU: rt_debug_crash_localize 3 times, rt_buffer_2gib_range, num_bitwise_determinism, num_denormal_flush, rt_moe_tickets_sort and rt_jit_specialization_key twice, the others once] unmasked `buffer_load` at offsets 64–79 of a 64-element view of a larger allocation read 64.0–79.0, masked lanes 0 or `other` (−7.0), a masked store left its lanes; inside a mapped 2.5–4.5 GiB tensor u8 buffer loads at byte offsets 0x7ffffffe to 2^32 − 1 read 0 over non-zero data, int32 stores at 2^31 to 2^31 + 4000 left 7 in place, fp32 element offsets 2^30 + k returned x[k], no fault; `gl.device_print` of an 8 × 8 fp32 tile, one element per lane: 64 lines, each element once; 1024 programs of 64 lanes: 65536 lines, 64 under `if gl.program_id(0) == 3`; the §5 snippet on gemm_n6144_k2048: `contract.call_mutable_tensors(kernel.run(**call), call)` raised `AttributeError` (3 of 3), the outputs read from `call` gave violation_fraction 0.8646 on a 1.01-scaled copy at m = 16 where validate printed 0.865; a tile dumped from program 5 of 8 equal to torch within allclose 1e-6 (2 of 3 runs; the first compared bitwise), the rest untouched, a grid of 1 identical; a debug store into input `a`: `input_mutation`, exit 1; the sync after a store 2^40 elements away aborted (exit −6), the sync before it passed; a store past a 64-int32 tensor into the next allocation synced, a later gather indexed by the overwritten value faulted (exit −6); `buffer_load`, `buffer_store`, `gl.load` and `gl.store` 0x7ffffc00 bytes past a 4 MiB tensor: exit −6; a spinning launch under `timeout -k 5 20`: exit 124 after 20.08 s; `gl.exp2` of −126 … −129: 1.18e-38, 0, 0, 0; bf16 K = 16 on [16, 16, 32] with `k_width` 8: 0.32 ulp from acc / 2 + a·b, four `fmul … 0.5` in llir, none at K = 32; K = 32 with `k_width` 16 and `mfma_scaled` K = 64 with `k_width` 32: acc / 2 + a·b; a fresh zero accumulator at K = 16: rel err 2.8e-8; a 4-row, 4-split float-atomic row sum: two calls missed the difference in 17 and 18 of 20 batches, 200 calls in 0; a 24-split atomic GEMM: 100 distinct outputs in 100; no K split against a six-way split at n2624 m = 64: 75 of 167,936 bf16 outputs differ, peak 0.785 of the bound; 65 bitwise schemas with 0 atomics, 42 global and 4 LDS atomics in the bounded ones all on int32/uint32; 8 ids in a 64-lane layout: counters 1, 4 built replicas: counters 4 (case triton/checks/rt_buffer_oob_masks.py (.08–.10), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_buffer_oob_masks.log; case triton/checks/rt_buffer_2gib_range.py (.05–.07), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_buffer_2gib_range.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_buffer_2gib_range_r1.log; case triton/checks/rt_debug_crash_localize.py (.01–.10), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_debug_crash_localize.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_debug_crash_localize_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_debug_crash_localize_r2.log; case triton/checks/num_denormal_flush.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_num_denormal_flush.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_denormal_flush_r1.log; case triton/checks/rt_mfma_small_k_and_acc.py (.01–.05, .13), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_mfma_small_k_and_acc.log; case triton/checks/num_bitwise_determinism.py (.03–.06), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_num_bitwise_determinism.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_num_bitwise_determinism_r1.log; case triton/checks/census_atomics_bitwise.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/300_census_atomics_bitwise.log; case triton/checks/rt_moe_tickets_sort.py (.01), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_moe_tickets_sort.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_moe_tickets_sort_r1.log); at launch, ints 2 and 17 typed `i32`, 16 and 32 `i32` + D, 2^31 `i64`, 2^63 `u64`, 4 and 8 sharing the binary of 2; pointers at offsets 0 and 256 of an allocation D, at 4 and 8 not; a 4 KiB storage S, a 2 GiB + 16 KiB one not; a counting constexpr hashed 13 times in 12 launches of one binary; `is_within_2gb` called 300 times in 300 launches, 0 with `AMDGCN_USE_BUFFER_OPS=0`; each of a list constexpr (`INSTR=[16, 16, 32]`), a `gl.distributed_type` and a `gl.shared_memory_descriptor_type` constexpr, the option `waves_per_eus`, 167936 and 262144 B of LDS (limit 163840) and a reassigned global compiled with `triton.compile` and failed at its launch; eager host time 7.32 µs with 1 tensor and 38.45 µs with 32 (1.004 µs per tensor), the specialization callback 0.58 µs per call; a 2 ms sleep in the timed callable: 6.523 against 6.507 µs; per extra kernel the bench read 4.062 µs against 4.001 µs of back-to-back replay; trivial Gluon kernels 4.166 and 4.173 µs, torch `zero_` 4.185, `add_` 4.167, split-K reduce 4.222; `bench_cudagraph` 4.229 µs against validate's 4.309 µs at the n3072 k1536 m = 16 Point; first readings 4.341, 4.364, 4.213, 4.191, 4.188 µs against the next nine's ranges 4.304–4.337, 4.339–4.369, 4.222–4.254, 4.189–4.210, 4.177–4.210 (case triton/checks/rt_jit_specialization_key.py (.05, .09, .11, .14, .16, .17, .21), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_jit_specialization_key.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_jit_specialization_key_r1.log; case triton/checks/rt_launch_only_errors.py (.04, .10, .11, .13, .14), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_launch_only_errors_r1.log; case triton/checks/bench_launch_cost.py (.02, .08, .13–.18, .24), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_launch_cost.log)
- E5 [2026-11, gfx950 (MI355X), lmsysorg/sglang:v0.5.22-rocm10-mi35x (ROCm 10.0.0, rocprofv3 1.3.5, torch 2.11.0+rocm10.0.0, Triton 3.8.0+git4cff872c), N=1, one case run, each rocprofv3 run under timeout -k 5 90] `rocprofv3 --kernel-trace` and `--pmc` on processes importing torch, with or without triton: both messages, alive past SIGTERM, ended by SIGKILL (rc −9); `import triton` alone: rc 0; `timeout 10` without `-k`: alive 60 s past its deadline, ended by SIGKILL to its group; an AKLR GEMM call under `--kernel-trace`: 0 trace files; validate's report: profiler kineto, metric gpu_duration_sum, one number per round; DSA decode f60f22e6 under `torch.profiler`: `_attention_split` and `_attention_merge`, hot sum 12.09 µs against validate's 14.25 µs; the padded `gl.bank_conflicts` assertion at `LinearLayoutConversions.cpp:1364` on the image and `:1379` with the PyPI wheel, as in the v3.8.0 tag (case triton/checks/rt_tooling_on_image.py (.03–.09), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_tooling_on_image.log)
- E7 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=0 (3 runs; check .02 asks all 11 kernels to match and fails in each), Gluon kernels whose every wave records its start and end ticks with HW_ID and XCC_ID, resident workgroups = the largest overlap on one CU] equal to min(⌊4 × waves per SIMD / num_warps⌋, ⌊163840 / metadata.shared⌋) in all 3 runs: 4 warps with 8 KiB of LDS at 17, 65, 129, 256 VGPRs 8, 7, 3, 2; 8 warps at 17 and 129 VGPRs 4 and 1; 48, 64, 160 KiB of LDS 3, 2, 1; 32 KiB of LDS (formula 5): 4 per CU in 3 of 3, at most 1024 = 4 × 256 live GPU-wide; 8 warps at 65 VGPRs, 7 waves per SIMD (formula 3): 3, 3 and 4 per CU, at most 768 = 3 × 256 live GPU-wide in each run (case triton/checks/bench_occupancy_residency.py (.02), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_occupancy_residency_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_occupancy_residency_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_occupancy_residency_rep3.log)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, case runs of python -m aklr validate on one GPU, planted copies of committed schemas] a child that aborts or segfaults: `runner … exited with code -6` / `-11`, no task.log left after validate; a callee `NameError`: `build_failed` with the call site only; a lowering error: `RuntimeError: PassManager::run failed`; the scan diagnostic, `static_print` and `device_print` markers absent from console, report and `--out` (1 of 2 runs; the other matched the marker in the source text); a GEMM rounding its output through fp16: only `numeric_mismatch` at `large_magnitude/0`, 5248 non-finite; a device counter wrong from the 2nd or 3rd replay: `capture_unsafe` at `first_capture/1` or `/2`; a host read in `run` or a per-call constexpr: `capture: …`, `_do_compile: compiled in the capture`; a `bitwise` copy got a `determinism/0` step on input set `numeric/0`, a `bounded` one none; a spinning kernel in a one-workload batch: `candidate_timeout`, bench stage 1204 s; fresh caches after 083b493a and cf637bff: `_private_k128_gemm` / `_rms_kernel` and `_flush_l2_cache_kernel` only; a broken MFMA call on line 71: `…/_aklr_triton_08caadf9f953cecf/kernel.py:71:11: error:` (1 of 2 runs); `--base` on a PREFETCH 3 copy of 083b493a: two cache entries in packages 21fc796024633850 (the report's hash) and fdfb1d7131a6a8d7 (1 of 2 runs); 42 timings of 16 or 32 rounds, a Point at exp(log_ratio_mean) 2.48 with ratio_lower_bound 2.000; a 083b493a copy leaving 8 rows unwritten: 24576 non-finite mismatches at `first_capture/0`; 1556 timed calls of 083b493a in one bench process on one weight `data_ptr` (case triton/checks/rt_aklr_failure_findings.py (.13–.21, .24–.26), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_aklr_failure_findings.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_aklr_failure_findings_r1.log; case triton/checks/rt_aklr_validate_report.py (.03–.07, .11, .13), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_aklr_validate_report_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_aklr_validate_report_r2.log; case triton/checks/rt_per_call_scratch_poison.py (.07, .10), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/100_rt_per_call_scratch_poison.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/150_rt_per_call_scratch_poison_r1.log)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, case runs on one GPU; the stream: median of 15 launches per size and flush; the conversions: median of 7 rounds of 3 launches] a 4-warp, 2048-program stream reading 64–1024 MB once per launch: plain loads 2.52–4.73 TB/s after a 1 GiB write (`zero_()`), 4.47–6.02 after a 1 GiB read, write over read 0.49–0.79 at every size; `.cg` 4.71–6.64 and 4.65–6.67, write over read 0.985–1.014; `.cg` over plain 1.40–2.18 after the write, 1.04–1.12 after the read; unhinted, hot over read-flushed at least 1.117 at 64–192 MB, 0.991–1.004 at 1024 MB; a 16 KB 32-bit tile converted per iteration over 1024 programs: 73.14–74.12 ns without a conversion, 661.4–662.8 through `ds_bpermute` (96 per kernel, no LDS), 573.9–580.8 and 1518–1522 through LDS, 1058–1060 ns between warps, 7.79–20.8× the loop without it (case triton/checks/bench_flush_kind.py (.03, .04, .06), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_flush_kind.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_flush_kind_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_flush_kind_rep3.log; case triton/checks/bench_convert_layout_cost.py (.01), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_convert_layout_cost.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_convert_layout_cost_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_convert_layout_cost_rep3.log) Cleanup addition (stream N=4; conversion N=3 unchanged): 870 .03/.04/.06 PASS, read lower bound 4.43 TB/s, cg/write lower bound 4.70, cg write/read up to 1.022, cg/plain write up to 2.215; hot/read small min 1.102. The read is the case Gluon flush timed by eager events, not direct AKLR graph-replay provenance. (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_flush_kind_cleanup.log).
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (3 runs; check .05 bounds hot over read-flushed by 1.20, exceeded at 128 MB in one run: 1.217), the E9 stream without a hint, hot against after a 1 GiB read] hot over read-flushed at 64, 128, 192, 512, 1024 MB: 1.127, 1.165, 1.137, 0.994, 0.997; 1.123, 1.217, 1.179, 1.011, 0.991; 1.117, 1.176, 1.138, 1.002, 1.004 (a read flush 10.5–11.3 %, 14.2–17.8 % and 12.0–15.2 % below hot at 64, 128 and 192 MB), the largest at 128 MB in each run (case triton/checks/bench_flush_kind.py (.05), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_flush_kind.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_flush_kind_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_flush_kind_rep3.log) Cleanup addition (N=4 valid runs; .05 PASS in 200/880, FAIL in 800/870): 870 hot/read at 64/128/192/512/1024 MB 1.102/1.224/1.198/1.014/0.9951; largest gap still 128 MB. Four-run read-rate deficit below hot about 9–19%, not a universal cache law. (local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/870_bench_flush_kind_cleanup.log).
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, case runs on one GPU, each time the median of 7 rounds of 3 launches, 2048 one-wave programs] `gl.bank_conflicts` on an XOR-swizzled and on a padded layout: the compiling child aborted (exit −6) in 4 of 4 runs; a 64 × 64 bf16 tile read LDS-bound, plain over XOR-swizzled 1.701–1.731 (operand 0) and 1.748–1.765 (operand 1); one program of the plain loop 305.0–305.9 µs, 2048 programs 906.8–915.6 µs; single rounds of the VALU-bound variant spread 95.7–111 µs (11–13 % of its medians), each of the 12 medians within 0.8–2.0 % across the 4 runs; reads `ds_read_b128` (operand 0) and `ds_read_b64_tr_b16` (operand 1) (case triton/checks/bench_lds_bank_conflicts.py (.01, .02, .05, .06, .08, .13), local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/200_bench_lds_bank_conflicts.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/250_bench_lds_bank_conflicts_r1.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/800_bench_lds_bank_conflicts_rep2.log, local:/workspace/code/gpt_oss_docker/scratch/triton_skill/checks_logs/logs/880_bench_lds_bank_conflicts_rep3.log)
- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:584 `add_stages`: plain Triton gets `ttir`, `ttgir`, Gluon `ttgir` from `gluon_to_ttgir` (:589); both then `llir`, `amdgcn`, `hsaco` (:590-592)
- S2 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:316 the traced module is stored as `<name>.source`; `CompiledKernel.asm` holds every stage file of the cache group (:426-429)
- S3 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:522 `metadata["shared"] = src.get_int_attr("ttg.shared")`
- S4 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:479 `n_regs`, `n_spills` come from `load_binary` when the kernel is first loaded; on HIP from `HIP_FUNC_ATTRIBUTE_NUM_REGS` and `HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4` (third_party/amd/backend/driver.c:499-504)
- S5 triton-lang/triton@c01b677:python/triton/knobs.py:346 `TRITON_DUMP_DIR`, `TRITON_OVERRIDE_DIR` (:347), `TRITON_CACHE_DIR` (:348), `TRITON_KERNEL_OVERRIDE` (:358), `TRITON_KERNEL_DUMP` (:359), `TRITON_ALWAYS_COMPILE` (:362), `TRITON_DISABLE_LINE_INFO` (:366), `TRITON_FRONT_END_DEBUGGING` (:367), `TRITON_INTERPRET` (:471), `TRITON_DEBUG` (:474), `TRITON_F32_DEFAULT` (:496), `AMDGCN_ENABLE_DUMP` (:522), `TRITON_DUMP_MIR` (:535); default directories under `~/.triton/` (:353-354)
- S6 triton-lang/triton@c01b677:include/triton/Tools/Sys/GetEnv.h:15 the cache-invalidating variables, among them `AMDGCN_ENABLE_DUMP`, `AMDGCN_USE_BUFFER_OPS`, `LLVM_IR_ENABLE_DUMP`, `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`, `MLIR_ENABLE_TIMING`, `TRITON_DISABLE_LINE_INFO`, `TRITON_DUMP_MIR`, `TRITON_F32_DEFAULT`; `TRITON_REPRODUCER_PATH` is cache-neutral (:59-63)
- S7 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:268 a cache hit returns before any stage runs; the dump manager writes stages only inside the compile loop (:341); the dump and override directories are named by `src.hash()` (:257-258), which for `ASTSource` hashes the function's key, the attributes, the signature and the constexprs, not the options or the target (:71-76)
- S8 triton-lang/triton@c01b677:python/src/ir.cc:1884 `MLIR_ENABLE_DUMP` as a bool or a function name enables IR printing before each pass, and after a pass only on failure (`printAfterOnlyOnFailure`, :1914); `TRITON_REPRODUCER_PATH` at :1937, suffixed `.<pipeline tag>.repro.mlir` (:1940-1942); `MLIR_ENABLE_TIMING` at :1974
- S9 triton-lang/triton@c01b677:python/src/llvm.cc:275 `LLVM_IR_ENABLE_DUMP` sets LLVM's `print-after-all` (also :341, :438)
- S10 triton-lang/triton@c01b677:python/triton/runtime/cache.py:319 `get_cache_key` = `triton_key()` (version, hashes of `compiler/`, `backends/`, `language/` and `libtriton`, :284-316), the source hash, the backend, the options hash and the sorted environment; the function's `cache_key` hashes its dependencies plus `str(self.starting_line_number)` and its constexpr globals (python/triton/runtime/jit.py:528-535)
- S11 triton-lang/triton@c01b677:python/triton/runtime/jit.py:781 `run` returns the `CompiledKernel` it launched
- S12 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1703 `if not module.verify(): … raise RuntimeError("error encountered during parsing")`
- S13 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1384 with `front_end_debugging` the callee's exception is re-raised; otherwise it is wrapped in a `CompilationError` at the call site with no message of its own (:1386)
- S14 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3168 "Wrong opIdx"; "Incompatible parent encoding" at :3178; "mismatching encoding between A and B operands" at :3196 and "mismatching kWidth between A and B operands" at :3199
- S15 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:3879 "… warps per CTA, but the context requires …", emitted through "Result has an invalid layout" (lib/Dialect/TritonGPU/IR/Traits.cpp:94)
- S16 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:206 "Layout mismatch in broadcast"; the `assert_trivial` `TypeError` at :246
- S17 triton-lang/triton@c01b677:python/triton/_utils.py:69 "Shape element {i} must be a power of 2"
- S18 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Types.cpp:160 a shared-memory descriptor's layout rank must equal its shape's rank or be one less
- S19 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:413 "Cannot access global variable {name} from within @jit'ed function"
- S20 triton-lang/triton@c01b677:python/triton/runtime/jit.py:592 `compute_cache_key` builds `(tuple(specialization), str(options))` and looks it up in a dict; `run` calls it on every launch (:749)
- S21 triton-lang/triton@c01b677:python/src/gluon_ir.cc:792 `get_shared_bank_conflicts` converts the shared layout with `toLinearLayout`, which has no case for a padded layout and asserts "unknown layout" (lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1379); on an AMD target (:797) it then reads the layout's `vector` input dimension (:803-806)
- S22 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `numRecordsByte = INT_MAX - 1`; a masked lane's byte offset becomes `INT_MAX + 1` (:280-283)
- S23 triton-lang/triton@c01b677:third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h:9 the barrier filter drops the barrier between a local load synced by an async wait and an async copy to the same buffer, and is wrong for a single-buffer refill; `load_shared_relaxed` sets that mark (python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:164-168)
- S24 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:140 `wait_group` docstring: "uncommited async operations will be waited upon even if `num_outstanding` is 0"
- S25 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:376 `__HIP_FTZ = True` (:383): `exp2` flushes denormal inputs and outputs
- S26 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:140 `supports_native_tensor_specialization = False`; `is_within_2gb` reads `untyped_storage().size()` (:211-226) and `get_tensor_specialization` adds `S` under `use_buffer_ops` (:236-240); the C++ specializer calls back into Python per tensor (python/src/specialize.cc:365-369)
- S27 triton-lang/triton@c01b677:python/triton/knobs.py:517 `AMDGCN_USE_BUFFER_OPS` defaults to True
- S28 triton-lang/triton@c01b677:python/triton/compiler/compiler.py:466 at load, `metadata.shared` above the device's `max_shared_mem` raises `OutOfResources` (message in python/triton/runtime/errors.py:21-22)
- S29 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:560 `make_amdgcn` calls `llvm.translate_to_asm`, the frame where the FOLD=1 compile died
- S30 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:303 `convert_to_buffer_ops` runs in `make_ttgir` only; `gluon_to_ttgir` (:327-348) has no such pass; it converts where `canUseBufferOps` (third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp:135) finds a splatted base plus offsets (:142-146), safe if the base has `tt.pointer_range=32` (:165; the `S` attribute, third_party/amd/backend/compiler.py:231-232) or `isByteOffsetSmallerThan2GB` proves the offset (:58, :169)
- S31 triton-lang/triton@c01b677:python/triton/language/semantic.py:934 "Cache modifier {cache_modifier} not supported"
- S32 AMD-AGI/AKLR@43e112e:aklr/core/runtime/builder/builder.py:104 the package `<prefix><content_hash[:16]>` under `<kernel cache>/<language>/` (:106-108); `TritonBuilder` sets the language `triton` (:193-196), so the prefix is `_aklr_triton_`
- S33 AMD-AGI/AKLR@43e112e:aklr/core/utils/testing.py:50 `flush_l2_cache` reads a 1 GiB buffer: "written lines stay dirty, and the next kernel pays their write-back"
- S34 AMD-AGI/AKLR@43e112e:aklr/core/validation/command.py:25 `DEFAULT_REPORT_DIR = Path("aklr_validation")`, `--out` at :40
- S35 AMD-AGI/AKLR@43e112e:aklr/core/validation/stages/_bench/_evaluation/_request.py:131 the steps `first_capture/0-2`, `numeric/0-3`, `large_magnitude/0`, `determinism/0`, `post_timing/*` and their codes (:34-40); a finding's detail starts with the step label (_reference.py:174); a mismatch's measures (aklr/core/contract/semantics/_compare.py:30-36)
- S36 AMD-AGI/AKLR@43e112e:.github/workflows/ci.yaml:250 `TRITON_CACHE_DIR` under `$RUNNER_TEMP`, emptied at every job (:245); the baseline aiter installed over the image's with `AITER_USE_SYSTEM_TRITON=1`, "Keep the image's Triton: otherwise aiter's setup.py swaps in the one it pins" (:291-292); `triton==3.8.0` required (:296); the `amd-aiter` version must contain `+g<commit[:7]>` (:307)
- S37 AMD-AGI/AKLR@43e112e:aklr/core/validation/stages/_bench/_evaluation/_candidate.py:85 a candidate task that dies is `candidate_crash` with its last line, such as "runner … exited with code -11"
- S38 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:309 `if TILES > 1:` allocates `cache0`/`cache1`, the other branches `cache_shared` (:316-325); `TILES` at :285; 256 keys per tile (`key_offsets = gl.arange(0, 256, …)` at :308, the tile's indices at :337); the 64-head path reads `cache_shared` at :376; the host's split rule at :585
- S39 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json#kernel.py:1301 `gate_fold`; the scheduler attribute at :1306; the gate-up launch's constexprs and options at :1331-1336
- S40 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:25 `gl.inline_asm_elementwise("v_med3_f32 $0, $1, $2, $3", …)`
- S41 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:87 `gl.dot_fma(a0, b, acc0)` in `_router_pair_packet`
- S42 AMD-AGI/AKLR@43e112e:aklr/core/validation/stages/_bench/_evaluation/_candidate.py:188 a `kernel.requires` the installed toolchain does not meet fails the step as `no_matching_toolchain`
- S43 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:181 `parse_options` keeps only the keys that are `HIPOptions` fields
- S44 triton-lang/triton@c01b677:python/triton/runtime/jit.py:715 at the first compile of a launch, a keyword that is neither an option nor a parameter raises `KeyError("Keyword argument %s was specified but unrecognised")`
- S45 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:91 `buffer_load_to_shared`: "size per thread * bits per element must be 128 or 32" for the offsets layout, or lowering to LLVM fails
- S46 AMD-AGI/AKLR@43e112e:aklr/ops/gemm/baseline.py:16 `gemm_baseline`: "aiter's tuned 16-bit gemm" (`from aiter.tuned_gemm import gemm_a16w16`, :17); the RMSNorm baselines call `aiter.ops.rmsnorm` (aklr/ops/rms_norm/baseline.py:12, :24); `AKLR_BASELINE_AITER: dbd8bf5…` at .github/workflows/ci.yaml:7
- S47 triton-lang/triton@c01b677:python/src/ir.cc:1982 a failed pass-manager run throws `PassManager::run failed`; every run enables MLIR's crash reproducer, printed to the console without `TRITON_REPRODUCER_PATH` (:1956) and written to that path with it (:1953), which turns a crash inside a pass into a failed run
- S48 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:273 "no matching matrix core intrinsic" when no instruction has the requested shape and types
- S49 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 "TODO: unsupported scan layout"
- S50 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:729 "Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension"
- S51 triton-lang/triton@c01b677:python/triton/language/semantic.py:201 "Scalar {lhs_scalar} is out of range for type {ret_sca_ty}" (also :204)
- S52 AMD-AGI/AKLR@43e112e:aklr/core/utils/testing.py:111 `_kineto_replay_times_ms` adds the durations of every recorded GPU kernel of a replay except the flush and divides by the calls per graph (:133-149); a report's `measurement_profile.metric` is `gpu_duration_sum`; `bench_cudagraph(fn, rep=20)` (:152), which each bench round reads (aklr/core/validation/stages/_measure.py:262-272) after one unrecorded reading and a warm-up round (:289-291); `speedup_bounds`, the bounds of the mean ratio over the unclamped rounds to 3 decimals (_measure.py:121), against the gate's on rounds clamped to ±ln 2 (:50, :208); `versus` is `baseline` or `incumbent` (:102); `WorkloadRecord.speedup` is the stored one (_bench/_report.py:85-93), printed to 2 decimals (:188)
- S53 triton-lang/triton@c01b677:python/src/specialize.cc:274 an int equal to 1 is specialized as a constexpr; other ints get `i32` or `i64` (:281) and `D` when a multiple of 16 (:283); a pointer `D` when 16-byte aligned (:363); the dtype names `"float8_e4m3fn": "fp8e4nv"`, `"float8_e4m3fnuz": "fp8e4b8"` (python/triton/_utils.py:87, :89); two Qwen3.5 GDN prefill files look an fp8 dtype up in a table at import: `FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, …)` raising "unsupported FP8 dtype" otherwise (aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/0a07f9f9342c6cf1.json#kernel.py:48, :30-37)
- S54 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:489 `set_all_fn_arg_inreg` (kernel arguments preloaded into SGPRs) is skipped when `TRITON_SWAP_MIR` or `TRITON_DUMP_MIR` is set; the debug-info scope pass runs unless `TRITON_DISABLE_LINE_INFO` (:399); `kpack` is reset to 1 on gfx950 with a warning (:119-123), `matrix_instr_nonkdim` is read only by `make_ttgir`'s matmul pass (:273) and `schedule_hint` (:105) nowhere in the backend; `make_hsaco` assembles (:574) and links (:579), a failed link raising "LLD failed to link hsaco source" (third_party/amd/python/triton_amd.cc:546)
- S55 triton-lang/triton@c01b677:python/triton/runtime/jit.py:486 a function's source starts at its `def` line, from which `CompilationError` counts `at {lineno}:{col_offset}` (python/triton/compiler/errors.py:25); "Global variable {name} has changed since we compiled this kernel" (:766); `@triton.jit` returns an `InterpretedFunction` under `TRITON_INTERPRET` (:984), `gluon.jit` a `GluonJITFunction` regardless (python/triton/experimental/gluon/_runtime.py:106); `device_print`'s line format `pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>` (lib/Conversion/TritonGPUToLLVM/PrintOpToLLVM.cpp:98); `device_assert` emits nothing unless `options.debug` (python/triton/language/semantic.py:1811), and a Python `assert` in a jit body becomes `device_assert` (python/triton/compiler/code_generator.py:1353-1356)
- S56 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:516 a block stops at its own `return`; a constexpr `if` traces only its taken block (:995-996), and a `return` inside it moves tracing into a dead block, so the statements after the `if` are traced (:551-554); DeepSeek-V2's `_quantized_bytes` has `return packed` before its sign-OR lines (aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:36, :37-39); the Qwen3-Next template keeps its rank-2 and rank-3 delta rules apart because "the unified version traced both ranks and failed on the rank it was not given" (aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:640)
- S57 AMD-AGI/AKLR@43e112e:aklr/core/validation/stages/_bench/_execution/_runner.py:496 the child's stdout and stderr go to `task.log` (:497), whose directory is deleted when the task ends (_execution/_backend.py:185); the child sets `GPU_FORCE_QUEUE_PROFILING=1` (:515); a dead task's finding keeps the error's last line (_evaluation/_candidate.py:78), a failed build `<type>: <message>` (_evaluation/_candidate.py:145); `candidate_timeout` after 900 s + 300 s per workload of the batch (_stage.py:49-50, :283; _evaluation/_candidate.py:79-80); `_poison_outputs` writes 0xFF bytes for integer and sub-byte dtypes, NaN otherwise (_evaluation/_candidate.py:588, :597-600), inside every capture of a new call (:506, :520); `_CompileTripwire` replaces `JITFunction._do_compile` while a check step captures (:285, :302-314), and a compile there fails the step as "compiled in the capture" (:523), `capture_unsafe` (:511-513)
- S58 triton-lang/triton@c01b677:python/triton/language/semantic.py:1418 the dot asserts `input_precision` is in the backend's list, `('ieee', 'bf16x3', 'bf16x6')` (third_party/amd/backend/compiler.py:93), which `TRITON_F32_DEFAULT` sets for Gluon's `mfma` (python/triton/knobs.py:496); "scale factor must be 16 or 32. Got …" at lib/Dialect/Triton/IR/Ops.cpp:391; `report_fatal_error("Dimensions must match, ignoring order, but they don't…")` at lib/Tools/LinearLayout.cpp:91; the `extract_slice` verifier's "Lane and warp dim basis must match…" at third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp:222; `adjustAccForSmallKDim` multiplies each accumulator element by 1 / duplication (third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:191, :208-213), and the scaled-MFMA lowering asserts linear-layout scales (:898-901)
- S59 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/f60f22e6f14448c6.json#kernel.py:407 `gl.static_assert(TILES == 1 or PANEL_CACHE)`; `gl.static_assert(TOPK % (SPLITS * 256) == 0)` at :403; `(15 - 3 * (row & 3) - 2 * part & 15)` at aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:319
- S60 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1119 a live-in name keeps its Python type through a loop body, else "Loop carried variable {name} changed type, was {type(loop_val)} but is now {type(live_val)}", `loop_val` the body's value and `live_val` the one before the loop; `_find_carries` checks every live-in that is a Triton value (:482-488), constexprs included (`class constexpr(base_value)`, python/triton/language/core.py:210), for `range` (:1284) and `while` (:1131); `static_range` binds its variable as a `constexpr` and leaves it bound (:1212); a plain assignment turns an int into a tensor (:733-737), a `gl.constexpr`-annotated one keeps a constexpr (:747-749)
- S61 harness/skills/aklr-kb/triton/debugging.md:1 page-local Evidence retains the source and measurement conditions.
