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
profiling problems and numbers that disagree between two benchmarks. Everything quoted from a compile below was produced by
Triton 3.8.0 for gfx950 without a GPU; GPU numbers are marked and listed under Evidence.

## Route here when

- You need a kernel's TTGIR, LLVM IR or ISA, its registers, spills or LDS: §1, §2. Proof that a construct took effect,
  or a change that seems to do nothing: §3.
- A compile fails (`CompilationError`, `error encountered during parsing`, `PassManager::run failed`, an abort,
  `LLD failed to link`, `unhashable type`), or a Triton error's line number does not match the file: §4.
- An output is outside its bound against the reference, a replay of a captured CUDA graph is wrong on new inputs,
  two runs on the same inputs differ, an output is wrong after timed replays, or an output holds NaN or 0xFF bytes
  where the reference has values: §5.
- A process that dies with exit code −11 or −6, a launch that never returns, a compile that fails after changing
  one constant, a GPU memory access fault, `out of resource`: §6.
- rocprofv3 aborts or hangs, or you need per-launch times, the cost of LDS conflicts or of layout conversions: §7.
- Eager launches are slow on the host: §8. Your numbers disagree with another machine's or a colleague's, or a
  standalone benchmark disagrees with a graph-replay one: §9, §11. You want the kernel's time, not only the ratio:
  §10.

## 1. Get the artifacts

**Compile without a GPU.** `triton.compile` of a `GluonASTSource` (`triton.experimental.gluon._runtime`; an `ASTSource`
for `@triton.jit`) for `GPUTarget("hip", "gfx950", 64)` runs every stage on a cache miss (or with
`TRITON_ALWAYS_COMPILE=1`) and returns the `CompiledKernel` a launch would, with no device. Attributes are keyed by
`(argument index,)` (a dict keyed by name is silently ignored) and parsed by `HIPBackend.parse_attr("DS")`; constexprs
apply keyed by `(index,)` or by name ([authoring.md](authoring.md) §10 has the helper). Compile each launch as `run`, the kernel's host entry function ([authoring.md](authoring.md) §4),
makes it:

- the signature from the tensors' dtypes (`*bf16`, `*fp32`, `*u8` for uint8 views of MXFP4 and e8m0 bytes, `*fp8e4nv`
  for `torch.float8_e4m3fn`, `*fp8e4b8` for `float8_e4m3fnuz`), ints as `i32`, `i64` from 2^31, `u64` from 2^63;
- what the launch derives from the values (S53): an int equal to 1 becomes the constexpr 1, so pass it among the
  constexprs; `D` marks an int multiple of 16 and a pointer at offset 0 or 256 of an allocation (not 4 or 8), `S` a
  4 KiB storage, not a 2 GiB + 16 KiB one *(field, E1)*. Without them you read another binary: a unit stride passed
  as `i32` turned one `buffer_load_dwordx4` into 8 `buffer_load_ushort` *(field, E2)*, and an int not marked `D`
  narrows masked loads (§3);
- the options spelled right: `triton.compile` silently drops a key that is no `HIPOptions` field (S43), while a
  launch raises `KeyError` (§4);
- on a machine without torch, a stub with one placeholder type per attribute name let the sources of every kernel of the library (154 Triton and Gluon kernels for gfx950: [kernels/kernel_library.md](kernels/kernel_library.md) §1)
  import *(field, E2)*:
  `stub = types.ModuleType("torch"); stub.__getattr__ = functools.cache(lambda n: type(n, (), {}))`, then
  `sys.modules["torch"] = stub`. A bare module is not enough: host code reads `torch.Tensor`, `torch.dtype` and fp8
  dtypes at import, and two Qwen3.5 GDN kernels look an fp8 dtype up in a table there (S53).

**From a launch.** `kernel[grid](*args)` returns the `CompiledKernel` (S11). To read every kernel a benchmark or a
test compiles, give that run a fresh Triton cache:

```bash
TRITON_CACHE_DIR=$PWD/tc_run1 python <your benchmark or test of the kernel>
ls tc_run1/*/   # per compiled kernel: .source .ttgir .llir .amdgcn .hsaco .json (and .ttir for plain Triton)
```

Benchmarking a Gluon fp8 GEMM or a plain-Triton RMSNorm against its aiter baseline left only the kernel's one
launched function and the benchmark's own L2-flush kernel in a fresh cache: these baselines compiled no Triton kernel
*(field, E3)*. An MLIR diagnostic's `<file>:<line>:<col>` points at the file the kernel was imported from, a copy if
the harness imports a copy; a `CompilationError` counts lines from the function's `def` (§4). When one run compiles
two versions of a kernel (a change and the kernel it is compared with), both land in the same cache under the same
function names; the `.file` path in each `.amdgcn` tells them apart *(field, E3)*.

**What a `CompiledKernel` holds** (S1, S2, S3). Compiled offline, `asm['source']` was TTGIR for Gluon (`#ttg.blocked`
layouts) and TTIR for plain Triton (`tt.func`, no `#ttg` encoding), and only plain Triton had `asm['ttir']`
*(field, E4)*. `asm` also holds `ttgir`, `llir` (function attributes), `amdgcn` (ISA and metadata, §2) and `hsaco`.
`metadata` holds the options and what the compile found: `name`, `num_warps`, `waves_per_eu`, `num_stages`,
`enable_fp_fusion`, `allow_flush_denorm`, `llvm_fn_attrs`, `shared` (LDS bytes, §2), `global_scratch_size`, `hash`,
`triton_version`, `target`. `n_regs` and `n_spills` exist only once the binary is loaded on a GPU, and on HIP `n_spills`
is the scratch size / 4, not a spill count (S4): read the amdgcn.

**Environment knobs** (S5, S6, S8, S9). A knob "in the key" gets its own cache entry, so setting it recompiles:

| Variable | Does | In the key |
|---|---|---|
| `TRITON_CACHE_DIR` | where every stage file is cached (default `~/.triton/cache/<hash>/`) | — |
| `TRITON_ALWAYS_COMPILE=1` | compile even on a cache hit | no |
| `TRITON_KERNEL_DUMP=1`, `TRITON_DUMP_DIR` | write each stage of a compile to `<dir>/<source key>/` (default `~/.triton/dump`); a cache hit dumps nothing (below) | no |
| `MLIR_ENABLE_DUMP=1` or `=<kernel name>` | print the IR before each pass, and after a pass only when it fails (S8): a small Gluon compile printed 29 "IR Dump Before" entries, one per registered pass in order, no "IR Dump After", and none under another kernel's name; a compile failing in a pass (`PassManager::run failed`) printed one "IR Dump After" *(field, E2, E4)*; `MLIR_DUMP_PATH=<file>` writes the dump to a file | yes |
| `MLIR_ENABLE_TIMING=1` | print each pass's time, per pass-manager run | yes |
| `LLVM_IR_ENABLE_DUMP=1` | LLVM's `print-after-all` | yes |
| `AMDGCN_ENABLE_DUMP=1` | print the ISA to stdout during the compile | yes |
| `TRITON_DUMP_MIR=<dir>` | write the kernel's machine IR to `<dir>`; it also turns off kernel-argument preloading (S54; `.amdhsa_user_sgpr_kernarg_preload_length` 10 → 0 on a 3-argument Gluon kernel *(field, E2)*), so that build's ISA differs from the deployed kernel's: diff only builds that both set it | yes |
| `TRITON_DISABLE_LINE_INFO=1` | drop the line table; by default the amdgcn has a `.loc … ; <file>:<line>:<col>` line before each source line's instructions (S54) | yes |
| `TRITON_KERNEL_OVERRIDE=1`, `TRITON_OVERRIDE_DIR` | replace a stage's output by the file of the same name in `<dir>/<source key>/` | no |
| `TRITON_REPRODUCER_PATH=<file>` | write an MLIR reproducer of each pass-manager run to `<file>.<stage>.repro.mlir` (`gluon_to_ttgir`, or `make_ttir`, `make_ttgir_early`, `make_ttgir`; then `make_llir`) | no |
| `TRITON_FRONT_END_DEBUGGING=1` | raise front-end errors unwrapped: an undefined name in a jit body raises the bare `NameError`, with frames in `triton/compiler/code_generator.py`, instead of a `CompilationError` (§4) | no |
| `TRITON_DEBUG=1` | compile device asserts in (§4) | a JIT compile's key changes, and it records `debug=True` |
| `TRITON_F32_DEFAULT` | the input precision Gluon's `mfma` passes on: `tf32` fails every Gluon `mfma`, bf16, fp8 and int8 operands included, while `mfma_scaled` still compiles (§4) *(field, E7, E21)* | yes |
| `TRITON_INTERPRET=1` | run `@triton.jit` kernels in Triton's CPU interpreter; `@gluon.jit` ignores it and compiles for the GPU (S55) | — |

**One cache per run.** A dump on a cache hit writes nothing (S7), and a stage override is stored under the normal key,
so it stays in a shared cache after you unset it: give every run its own `TRITON_CACHE_DIR`.
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
| `.vgpr_spill_count` | VGPRs spilled, to scratch or to AGPRs: a 4-warp 128 × 128 `tl.dot` at `waves_per_eu` 2 reads 135, the dwords its 66 `scratch_store`s write (`.sgpr_spill_count` stayed 0) *(field, E2, E19)*; in 6 of 22 other spilling builds it read 1–65 above those dwords, in one with no scratch at all (its spills went to AGPRs): count the `scratch_store` dwords for the scratch traffic *(field, E19)* |
| `.private_segment_fixed_size` (`; ScratchSize`) | scratch bytes the spills use (352 for those 135) *(field, E2)* |
| `.group_segment_fixed_size` (`; LDSByteSize`) | 0 in a Triton kernel: LDS is sized at launch; read `metadata.shared`. All 1709 Triton kernels compiled by benchmarking the library's 154 kernels against their baselines (897 from the library's kernels, the rest from the baselines) read 0, while 538 of the 897 use LDS by `metadata.shared` *(field, E2)* |
| `; Occupancy:` | LLVM's waves per SIMD from the registers; it sees no LDS in a Triton kernel and does not round to whole workgroups |

A 4-warp bf16 MFMA probe kernel with a 128 × 256 fp32 accumulator reports `.vgpr_count 438`, `.agpr_count 182` and
`.amdhsa_accum_offset 256` *(field, E2)*.

**Occupancy.** LLVM gives a gfx950 wave 512 / N registers at N waves, rounded down to 8 (168 at 3, 72 at 7), arch VGPRs first up to 256 (the probe: 256 + 182) *(field, E2)*; LDS is 160 KiB per CU and per workgroup, allocated in blocks of 1280 bytes (S62; [lookup.md](../hardware/gfx950/lookup.md), [memory_and_sync.md](../hardware/gfx950/memory_and_sync.md) §4). The VGPR bound on waves per SIMD, min(8, ⌊512 / roundup8(roundup4(arch VGPRs) + AGPRs)⌋), matched `; Occupancy:` in 14 builds of 20 to 512 registers *(field, E2)*. Resident workgroups per CU, min(⌊4 × waves per SIMD / `num_warps`⌋, ⌊163840 / (⌈`metadata.shared` / 1280⌉ × 1280)⌋), matched all 11 kernels measured in two of three runs on MI355X: 32 KiB of LDS takes 26 blocks (33280 B), so 4 workgroups fit, not the 5 that ⌊163840 / 32768⌋ predicts; in the third run the 8-warp, 65-VGPR kernel (formula 3) held 4 on some CU, though no more than 768 = 3 × 256 were live at once on the GPU *(field, E5)* ([mainloop.md](../gemm/mainloop.md) §7, [grid_pricing.md](../gemm/grid_pricing.md) §4; the families' ranges: [kernel_library.md](kernels/kernel_library.md)). **[budget.py](../analysis/budget.py)** computes both, and each reachable occupancy's register ceiling; its `--isa` reads LDS from the group segment, 0 in Triton: pass `--lds` as `metadata.shared` and `--waves <num_warps>` (for 32 KiB: `python analysis/budget.py --isa k.amdgcn --waves 4 --lds 32768`).

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
| `gl.load` / `gl.store` in a Gluon kernel | `global_load` / `global_store`, no buffer access: a [64, 64] bf16 copy gave 2 `global_load_dwordx4` + 2 `global_store_dwordx4`, and 7 pointer forms with `DS` attributes and 284 kernels (432 launches) of 150 kernels of the library that call no explicit buffer or async-copy API compiled without a buffer access; uniform loads can become `s_load`, and atomics can bring `buffer_inv` / `buffer_wbl2` cache maintenance *(field, E2, E6)* | a compile sample, not every access form; explicit buffer API: [memory.md](gluon/memory.md) §4 |
| `tl.load` in a plain Triton kernel | `buffer_load_*` where `make_ttgir` proves the access in range: a scalar base plus offsets, with the base marked `S` or the byte offset bounded under 2 GiB by range analysis (S30); `global_load_*` otherwise | `x + tl.arange(…)` converts without `S`, `x + pid * BLOCK + tl.arange(…)` only with it ([authoring.md](authoring.md) §2) |
| `async_copy.buffer_load_to_shared` | `buffer_load_dwordx4 vN, s[..], 0 offen lds` | the lowering fails unless each lane copies 128 or 32 bits (S45) |
| `commit_group()`, then `wait_group(n)` | `s_waitcnt vmcnt(k)` before the first `ds_read` of the slot, then `s_barrier` at 4 waves (none at `num_warps=1`); two groups of two copies with `wait_group(1)` give `vmcnt(2)` | a copy without `commit_group()` gets no `vmcnt` wait (§5) |
| `gl.barrier()` | 4 waves: `s_waitcnt lgkmcnt(0)` when an LDS write is outstanding, then `s_barrier`, and no `vmcnt` wait (the tested async copy and an independent `global_load` were issued before it without a completion wait at the barrier; the load is waited on before its later consumer); 1 wave: the same instruction stream as without it | ([memory.md](gluon/memory.md) §11) |
| `gl.amd.cdna4.mfma`, bf16, `instr_shape=[16, 16, 32]` | `v_mfma_f32_16x16x32_bf16`; `[32, 32, 16]`: `v_mfma_f32_32x32x16_bf16` | operand dtypes or `instr_shape` ([mfma.md](gluon/mfma.md)) |
| `mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`, `[16, 16, 128]` | `v_mfma_scale_f32_16x16x128_f8f6f4`; `[32, 32, 64]`: `v_mfma_scale_f32_32x32x64_f8f6f4` | as above |
| `smem.load(DotOperandLayout(0, mma, 8))` / `(1, …)` from row-major bf16 tiles | `ds_read_b128` for operand 0, the transposing `ds_read_b64_tr_b16` for operand 1 (a [K, N] tile) | the tile's shared layout ([layouts.md](gluon/layouts.md) §5) |
| batched `gl.dot_fma` on bf16, operands in `DotOperandLayout(i, BlockedLayout, 0)` | `v_dot2c_f32_bf16`: 48 in the router launch of DS256 at 2 tokens (the DeepSeek-V2 MXFP4 MoE on a rank of intermediate dimension 256; MoE layer labels: [moe.md](moe.md) §1) | — |
| `cache=".cg"` or `".cs"` on `buffer_load`; `cache_modifier=` the same on `gl.load` | `… offen sc0 nt`; on `gl.load`, `global_load_dwordx4 … off nt`; `.cv`: `sc0 sc1`; `.ca`: no bits | stores: [memory.md](gluon/memory.md) §6 |
| `waves_per_eu=N` | `"amdgpu-waves-per-eu"="N,N"` in `llir`; a cap below the kernel's own count spills (`.vgpr_spill_count`, `scratch_*`; the probe at 2: 181 VGPRs, 153 `scratch_store`s) | at 0 the PyPI wheel adds nothing, the MI355X image's build writes `"0,0"`; the probe kernel's VGPRs per value: [tuning.md](tuning.md) §8 |
| `num_stages`, `schedule_hint` or `matrix_instr_nonkdim` on a Gluon launch; `kpack` | nothing: the same instruction stream for each value; `kpack` is reset to 1 on gfx950 with a warning (S54) | pipeline by hand ([memory.md](gluon/memory.md) §12); the other options' footprints: [tuning.md](tuning.md) §9 |
| `llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]`, `allow_flush_denorm=True` | `"amdgpu-sched-strategy"="iterative-ilp"`, `"denormal-fp-math-f32"="preserve-sign"` in `llir` | — |
| `gl.inline_asm_elementwise("v_med3_f32 $0, $1, $2, $3", …)` | the literal `v_med3_f32` | — |
| `gl.convert_layout(x, L, assert_trivial=True)` | a compile error if the conversion moves data between lanes or warps (§4); a register permutation passes, free for 32-bit data, `v_perm_b32` byte permutes for 8-bit | — |

A change that does not show may sit where nothing compiles it: in a branch the size does not take (per-size constants
select constexpr branches, and code written for other sizes is not compiled at this one), or after a `return` in the
same block, which Triton does not trace: in DS256 at 2 tokens, the helper that packs activations into fp4
bytes returns before its three sign-OR lines, so they are dead code (S56):

```python
@gluon.jit
def _quantized_bytes(x, a):
    # ... the sign of x applied to a, a split into lo and hi ...
    packed = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0 op_sel:[0,0,0]",
        constraints="=v,v,v",
        args=[lo, hi],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    ).to(gl.uint8)
    return packed  # the block ends here: an edit to the three lines below changes no instruction
    sign = gl.convert_layout(gl.where(x < 0, 8, 0).to(gl.uint8), pack_layout)
    sign_lo, sign_hi = gl.split(sign.reshape((x.shape[0], 16, 2)))
    return packed | sign_lo | sign_hi << 4
```

Put `gl.static_print` in the branch (§4) or diff the amdgcn of two builds.

## 4. Compile errors

**Where the message is.**

- Errors while tracing the Python come as `triton.compiler.errors.CompilationError`: `at L:C:`, the jit function's
  source with a caret, then the underlying `NameError`, `AssertionError` or `ValueError`; `break` and `continue` raise
  `UnsupportedLanguageConstruct`. L counts from the function's `def` line (S55): the file line is the `def` line
  + L − 1. An error inside a called `@gluon.jit` function is reported at the call site with no message of its own; the
  callee's `at L:C:` and message are in the exception's cause, which a traceback prints and a harness that reports
  only the exception's own text drops (seen for a `@triton.jit` helper). `TRITON_FRONT_END_DEBUGGING=1` raised the bare `NameError` instead (S13).
- A failed IR verification comes as `RuntimeError: error encountered during parsing` (S12), the MLIR diagnostic
  printed to stderr above it: `<file>:<line>:<col>: error: <message>`, with the source line.
- A failure while lowering comes as `RuntimeError: PassManager::run failed`; stderr holds the cause above the printed
  reproducer: the first `<file>:<line>:<col>: error: …` line, or for a C++ assertion in a `make_llir` pass, which MLIR's
  crash reproducer (S47) kept from killing the process, `python: <path>/MFMA.cpp:901: … Assertion '…' failed.`
  (`BlockedLayout` scales to `mfma_scaled`). The closing `<unknown>:0: error: Failures have been detected …` does not.
- An assembler failure in `make_hsaco` comes as `RuntimeError: LLD failed to link hsaco source …` (S54), which can
  include a linker cause after `because`; stderr has the assembler's `error:` and line (two SGPR operands on a
  `v_writelane_b32`: `invalid operand (violates constant bus restrictions)`, [wave_ops.md](gluon/wave_ops.md) §2.2).
  LLVM can also print `error: …` and still return a binary (an `illegal VGPR to SGPR copy`, same section): read stderr
  after every compile with inline asm.
- A native crash outside a pass has no Python traceback: an assertion or `LLVM ERROR` in a front-end call aborts
  (exit 134, or −6 for a child process), as `gl.bank_conflicts` and `mfma_scaled` on a rank-3 layout do (below);
  LLVM's code generation in `make_amdgcn` can segfault (139 / −11). `python -X faulthandler` prints the frames (§6).
- Four errors appear only at a launch, not in `triton.compile`: an unhashable constexpr, a misspelled launch option,
  `out of resource`, and `RuntimeError: Global variable <name> has changed since we compiled this kernel` when a
  module-level value the kernel read was reassigned after its first compile (S55).

**A harness that runs the kernel in a child process may show one line.** When the child's stdout and stderr go to a
log that is neither printed nor kept, a dead child leaves its exit code only (−11 for a segfault, −6 for an abort),
and a failed compile the exception's text: a `CompilationError`'s source and caret, but just
`RuntimeError: PassManager::run failed` for a lowering error. No MLIR diagnostic, `static_print` or `device_print`
output survives *(field, E3)*: compile the launch alone (§1) or run it in a process of your own (§5) to read them.

Each message below was reproduced for gfx950 (S14 – S21, S28, S31, S44, S48 – S51, S58, S60) *(field, E2)*; the
launch-only ones (`KeyError`, `unhashable type`, `out of resource`) at a launch on MI355X *(field, E1)*:

| Message | Cause | Fix |
|---|---|---|
| `AssertionError("Loop-carried variable acc has initial type <…> but is re-assigned to <…> in loop! …")` | the loop body changes a carried tensor's dtype (an fp32 `acc` made bf16 by `.to()`) | convert it back before the body ends, or create it before the loop in the dtype the body produces |
| `AssertionError("Loop carried variable g changed type, was <class 'triton.language.core.tensor'> but is now <class 'triton.language.core.constexpr'>")`, at the loop's `for` or `while` line | a name that is a constexpr when a `range` or `while` loop starts gets a tensor in the body, which makes it loop-carried: the variable a `gl.static_range` loop left bound (`for g in gl.static_range(PREFETCH)` in a prologue, then `g = block * UNROLL + inner` in the rolled loop), a `g: gl.constexpr` local or a constexpr parameter. "was" is the body's type, "is now" the type before the loop: a tensor name reused as a `static_range` variable in the body reads the other way round (S60) | give the static loop's variable its own name; a value the loop must carry starts as a plain assignment before it (`g = 0` makes a tensor) |
| `AssertionError("Mismatched type for y between then block (<…>) and else block (<…>)")` | a runtime `if` gives a name another dtype in each branch (bf16 and fp32) | one type in both branches; a `gl.constexpr` condition is resolved while tracing and may differ |
| runtime `n`: `ValueError('Did you forget to add @triton.jit ? …')` at `gl.arange`; `AssertionError('int32[] used as tl.static_range start value is not a constexpr')` (or `end value`); `TypeError('get_blocked_layout(): incompatible function arguments …')` for a layout parameter; `TypeError('Shape element 0 must have type constexpr[int], got …')` for `gl.zeros([n])` | these uses require a compile-time value; the decorator is fine *(field, E7)* | `n: gl.constexpr` compiled for the tested arange, static-range and layout uses |
| `tensor shape and layout rank mismatch: shape=[512], layout=BlockedLayout(…), shape rank=1, layout rank=2` | a 1-D `gl.arange` given the 2-D tile's layout | `gl.SliceLayout(1, L)` for row indices, `gl.SliceLayout(0, L)` for column indices |
| `error: 'tt.make_range' op Failed to infer return type` (`'arith.constant'` for `gl.zeros`, `gl.full`), then `PassManager::run failed` | a creation op without a layout that no other tensor fixes | give every `gl.arange`, `gl.zeros` and `gl.full` a layout |
| `UnsupportedLanguageConstruct: unsupported AST node type: Break` (`Continue`); ``Cannot have `return` statements inside `while` or `for` statements in triton.`` | no early exit from a loop | put the exit in a `while` condition (`while (i < n) & (i <= LAST):`) or mask the rest of the body |
| `CompilationError` at the helper's call site: failed `gl.static_assert(R == 3, …)` after `if R == 2: return …`, or `gl.sum(x, 1)` on rank-1 `x` after `if R == 1: return …` | the statements after the returning constexpr `if` are still traced (S56) *(field, E7)* | the same code under `else:` compiled; a Qwen3-Next GDN kernel instead keeps rank-2 and rank-3 delta rules in separate functions (S56) |
| `ValueError('Layout mismatch in broadcast: <layout A> vs <layout B>')` in a `CompilationError` | the two operands of an elementwise op, a store's value and pointer, or a load's mask and pointer have different layouts, and neither is converted: 63 such builds over `BlockedLayout`, `SliceLayout`, `AMDMFMALayout` and `DotOperandLayout` pairs failed, a `DistributedLinearLayout` that maps elements exactly as the `BlockedLayout` beside it included, and 72 matching controls compiled *(field, E7)*. The exception is an operand still in `AutoLayout` (a `gl.arange`, `gl.zeros` or `gl.full` without a layout), which takes the other operand's layout (S63) | `gl.convert_layout` one side, or build both from slices of one layout ([layouts.md](gluon/layouts.md)) |
| `error: Result has an invalid layout: #ttg.blocked<{…}>. Layout has N warps per CTA, but the context requires M warps per CTA.`, then `error encountered during parsing` | the `BlockedLayout`'s `warps_per_cta` product N differs from the launch's `num_warps` M: 14 such builds, 1-D and 2-D, at M = 4, 8 and the default 4, failed *(field, E7)* | derive it from `gl.num_warps()`, or keep layout and launch in one config |
| plain `mfma`: `no matching matrix core intrinsic …`, `Wrong opIdx`, `Incompatible parent encoding`, `mismatching kWidth` / `encoding between A and B operands`, `input_precision … Got tf32` | in the order of the messages: a shape no instruction has (fp16 and bf16 `[16, 16, 64]` at version 4, bf16 `[16, 16, 32]` at version 3); operand index 1 for A or 0 for B; a `BlockedLayout` parent with `k_width` 0 and an MFMA accumulator (a nonzero `k_width` fails earlier); `k_width` 8 and 4, or 4 and 8; one `BlockedLayout` operand; `TRITON_F32_DEFAULT=tf32`, whatever the operand dtype *(field, E7, E21)* | [mfma.md](gluon/mfma.md) §10; these messages are not shared by all matrix APIs |
| `mfma_scaled`: e4m3 `[16, 16, 64]` and `[16, 16, 32]` at version 4 have no instruction; a version-3 layout asserts `version == 4 && isF8F6F4`; a `BlockedLayout` parent or operand raises `Expected lhs/rhs layout … parent matching MFMA layout` | swapped A and B operand indices and unequal `k_width` (16 and 8, or 8 and 16) **compiled** for e4m3 `[16, 16, 128]` with `None` scales, which says nothing about the results; `gl.dot_fma` instead rejects the indices while tracing (`a's operand index must be 0` / `b's … 1`), and an operand not in a `DotOperandLayout` *(field, E7)* | do not use absence of a compile error as a correctness check; [mfma.md](gluon/mfma.md) §10 |
| `mfma_scaled`: `scale factor must be 16 or 32`, `MFMA.cpp:901` assertion, `LLVM ERROR: Dimensions must match` | a wrong scale block; scales in a `BlockedLayout`; a rank-3 layout with `None` scales (an abort), respectively *(field, E2, E7)* | [mfma.md](gluon/mfma.md) §10 |
| `error: 'amdg.buffer_store' op failed to verify that value element type matches the pointed type of ptr` | `buffer_store` of fp32 to a bf16 pointer: it does not cast | `.to(gl.bfloat16)` first (`gl.store` casts) |
| `NameError("Cannot access global variable BLOCK_GLOBAL from within @jit'ed function. …")` | the jit body reads a plain module-level value | a `gl.constexpr` parameter, or the value written inside the jit function; Triton also accepts a module-level `BLOCK = gl.constexpr(64)` |
| `Shape element d must be a power of 2` | a tensor extent that is not a power of 2: 35 builds with extents 3–384 failed, `arange`, `zeros` and `full` of [96] naming d = 0 and `zeros([64, 96])` d = 1, and 12 power-of-2 controls compiled *(field, E7)* | for the 96-element case, pad to 128 and mask |
| `error: TODO: unsupported scan layout`, then `PassManager::run failed` | `gl.associative_scan` on a tensor in a `SliceLayout` (S49) | convert it to a 1-D blocked layout first |
| `error: Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension` | `gl.split` of an axis of size 2 held by two lanes, not in one lane's registers (S50) | a layout with `size_per_thread` 2 on that axis |
| `error: Lane and warp dim basis must match between source and destination layout.`, then `error encountered during parsing` | `gl.amd.slice` narrower than the layout's CTA tile, such as 8 columns of a layout whose lanes span 16 | slice whole CTA tiles, or convert to a layout whose CTA tile fits the slice |
| `ValueError('Scalar 2147483648 is out of range for type int32')` | a Python int above int32 in an op on an int32 tensor (`x & 0x80000000`) (S51) | write it signed (`-2147483648`) or work in uint32 |
| `ValueError: add with int32 is not supported on …` | `gl.amd.cdna4.buffer_atomic_add` on an int32 pointer | a uint32 counter, or `gl.atomic_add` |
| `rank must be equal to or one less than the shape size. Got 3 and 2` | `allocate_shared_memory(dtype, [2, 64, 64], <rank-3 layout>)`: each `.index(i)` slot is rank 2 | give the 3-D allocation a rank-2 layout (`[1, 0]`) and slot it with `.index(i)` |
| `TypeError: layout conversion from … to … is not trivial.`, with both linear layouts | `assert_trivial=True` on a conversion that moves data between lanes or warps | fix the layouts ([layouts.md](gluon/layouts.md)) or drop the assertion |
| `ValueError: Cache modifier .nt not supported` | loads take `.ca .cg .cs .cv`, stores `.wb .cg .cs .wt` | — |
| `NameError('<name> is not defined')` in a `CompilationError`, after one constant changed | one per-size constant changed alone took a constexpr branch that never defines a name used later | change the coupled constants together (§6) |
| `KeyError: 'Keyword argument waves_per_eus was specified but unrecognised'`, at launch | a launch option that is neither a `HIPOptions` field nor a parameter (S44) | spell it right; a compile-only check ignores it (§1) |
| `TypeError: unhashable type: 'list'`, `'distributed_type'` or `'shared_memory_descriptor_type'`, at launch | every launch hashes its constexpr values, cache hits included (S20; a counting constexpr: 13 hashes in 12 launches); `triton.compile` does not, so a compile-only check passes: a list constexpr `INSTR=[16, 16, 32]` compiled and then failed at launch on MI355X *(field, E1)*. Lists, `gl.distributed_type` and `gl.shared_memory_descriptor_type` raise `TypeError` on `hash()`; `gl.float32`, tuples, `BlockedLayout` and `AMDMFMALayout` hash | tuples for list constexprs (a list inside a layout is fine: layouts hash); for types, pass the pieces as constexprs (`DT=gl.bfloat16, SHAPE=(64, 64), L=<layout>`), build `gl.distributed_type(dtype, list(shape), layout)` in a `@gluon.constexpr_function` and read `T.shape` and `T.layout` from it (it launched on MI355X) |
| abort: `LinearLayout.cpp:292 … Assertion 'it != bases.end() && "inDim not found in layout"'` (swizzled), `LinearLayoutConversions.cpp:1379 … "unknown layout"` (padded) | on AMD, `gl.bank_conflicts` converts the shared layout to a linear layout, which has no case for a padded layout, then asks it for a `vector` input dimension, which a swizzled or shared-linear layout lacks (S21) | do not call it on AMD; time conflicts (§7) |
| `out of resource: shared memory, Required: …, Hardware limit: …` at the first launch | `metadata.shared` above the device's `max_shared_mem` (a workgroup gets at most 160 KiB on gfx950); a 262144-byte allocation compiles without error | smaller tiles or fewer slots; the message's `num_stages` hint does not apply to Gluon |

**Tools.** `gl.static_print("t:", t.type)` prints shape, dtype and layout while the kernel compiles (not on a cache
hit); `gl.static_assert(cond, msg)` stops the compile with your message. `gl.device_print("acc", acc)` prints, through
`__ockl_printf` (S55), a line `pid (x, y, z) idx (i, j) acc: <value>` per element each lane holds; guard it with
`if gl.program_id(0) == p:` (1024 programs: 64 lines, not 65536) *(field, E1)*, and run it in a process whose
stdout you read. A Python `assert` and `tl.device_assert` compile to nothing unless the launch has `debug=True` or
`TRITON_DEBUG=1`; then they compile in; with the default `sanitize_overflow=True`, int32 `*` and `+` can add overflow
asserts and `s_trap` instructions (checks proved redundant by the compiler may disappear) (S55),
failing with `int32 overflow detected for operation mul` *(field, E2)*: debug with it, time without it.
`TRITON_INTERPRET=1` does not run Gluon (§1). On the host, `print(layout.format_tensor_view(shape))` shows which
thread (`T<n>:<register>`) holds each element, and `format_hardware_view(shape)` the inverse (per warp, register, lane).

## 5. Wrong results

**Start from the check that fails.** The checks a kernel that its callers capture into a CUDA graph has to pass, and
what a failure of each means:

| Check that fails | Means | First suspect |
|---|---|---|
| the outputs against the reference on fresh inputs, eagerly or in the first replay of a captured graph | an output outside its bound on fresh inputs | NaN, or 0xFF bytes in an integer or sub-byte output, where the kernel wrote nothing (below); otherwise indexing, masks, layouts, rounding points |
| a later replay of the captured graph on new inputs | a replay of the captured graph on new inputs gave a wrong output (a device counter that changed the output from the 2nd or 3rd replay failed at that replay) | device state kept across calls: counters, sentinels and job descriptors written once instead of in every call ([authoring.md](authoring.md) §8) |
| the capture itself, at any input set | the capture itself failed: a host read of a device value in the launch code, or a compile inside the capture when a launch's constexpr differs from the warm-up call's (a hook on Triton's `JITFunction._do_compile` that raises while a graph is being captured shows it) | a constexpr or option computed from an input value instead of the call's shapes, or an int argument that is 1 or a multiple of 16 in one input set only (§1; [authoring.md](authoring.md) §5, §8) |
| a rerun on the same inputs compared bytewise (only where two runs must give the same bytes) | repeated same-input outputs differ bytewise while inside numeric bounds; an out-of-bound rerun is the first row's failure | atomics, arrival-order reductions and races are possible causes, not inferred from the difference alone |
| the outputs after the timed replays | wrong after the timed replays | state kept between calls ([pitfall](../pitfalls/state_kept_across_calls.md)) |

Measure a mismatch by the fraction of elements outside the bound, the peak error as a multiple of the bound and the
number of non-finite mismatches under an elementwise tolerance, by the SQNR in dB and that number under a
signal-to-noise bound. Before the call you check, fill each output with NaN, or 0xFF bytes for integer and sub-byte dtypes:
non-finite mismatches in a float output can indicate elements the kernel never wrote, or non-finite values it computed
(8 unwritten rows of a copy of the Gluon fp8 GEMM: exactly their 24576 elements) *(field, E3)*.

**Buffer ops are not bounded by the tensor.** Triton gives a buffer descriptor `num_records` `0x7ffffffe`, 2^31 − 2
bytes from the base, and a masked lane the out-of-range offset `0x80000000` (S22): 26 configurations of Gluon buffer ops
and of plain-Triton loads converted to buffer ops (norms and GEMMs, 4 KiB, 1 MiB and 1 GiB tensors),
compiled in two runs, all had that `num_records`, whatever the shape, and 24 masked lanes read 0 (Gluon without `other`)
or −7 (`other`, both frontends) while masked stores left the 1 in memory *(field, E8)*. An unmasked `buffer_load` at
offsets 64–79 of a 64-element view read 64.0–79.0, the allocation's next elements *(field, E1)*. Mask every access that
can leave the tensor ([memory.md](gluon/memory.md) §4).

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
  dropping one prefetch in a 128-token DSA sparse MLA kernel left its panels in flight *(field, E2)*.

**Atomics and determinism.** Where two runs on the same inputs must give the same bytes, the check is a rerun of one
input set held to both the numeric bound and bytewise agreement; a kernel held only to a numeric bound needs no
bytewise check. A float atomic or a split-K reduced in arrival order can fail it, and a rare race can pass two runs and
sometimes 200: repeat the call many times at the shape you time and compare bytes (on a 4-split float-atomic row sum,
2 calls missed it in 17–20 of 20 batches at 4 rows, 200 calls in 0–3 at 4 rows and 6–9 at 1 row) *(field, E1, E22)* ([../gemm/correctness.md](../gemm/correctness.md) §2–§3,
written for FlyDSL GEMMs). Of the library's kernels, the 65 that must give the same bytes in two runs held no
atomic (nor do the 163 after its growth), and every atomic in the ones held only to a numeric bound was on integers
(int32/uint32 counters, cursors and masks; int32 LDS scatters).
Bitwise equality is between runs of one kernel: no K split and a six-way split of the N 2624 × K 6144 bf16 GEMM at m = 64 differ
in 75 of 167,936 bf16 outputs, both within an fp32 dot-product rounding bound *(field, E1)* ([numerics.md](numerics.md) §8). Do not assume a
layout that holds an element more than once issues its atomic once: a `[64, 4]` tensor in
`BlockedLayout([1, 8], [64, 1], [1, 1], [1, 0])`, each element in two of a lane's registers, counted 8192 per active
element over 4096 programs with int32 and fp32 `gl.atomic_add` and fp32 and uint32 `buffer_atomic_add`, unmasked and
under all-true or even-element masks (one run) *(field, E9)*, while 8 ids held by several lanes of a 64-lane layout
counted once each *(field, E1)*. Replicas you build (`x[None, :] + replica[:, None] * 0`) stay distinct: with 8
elements, R = 2, 4, 8 replicas and one warp, int32 and fp32 adds counted R × 4096 without an owner mask and 4096 with
`replica == 0` *(field, E9)* ([moe.md](moe.md) §6).

**Numerics switches** ([numerics.md](numerics.md) §6–§7). `gl.exp2` flushes denormals, so `exp2(-127)` is 0 (S25)
*(field, E1)*; `enable_fp_fusion=True` lets LLVM contract `a * b + c` into one `v_fmac_f32` (`False`: `v_mul_f32` +
`v_add_f32`, though the saved `llir` can be the same) *(field, E4)*; `["amdgpu-ieee", "false"]` clears IEEE mode
(`.amdhsa_ieee_mode 0`); `allow_flush_denorm=True` only sets `denormal-fp-math-f32` to `preserve-sign` in `llir` and
flushes nothing. A timer does not separate them: at one size each, the speedup over the baseline moved 0.6 % without `amdgpu-ieee`
(a Qwen2 MoE router, 256 tokens; runs spread 0.9 %), 0.06 % with `enable_fp_fusion` (DS512, the DeepSeek-V2 MXFP4 MoE at rank intermediate dimension 512, 8
tokens), and without `allow_flush_denorm` the 4 DSA 64-head kernels' code did not change; accuracy passed in every run
*(field, E10)*. Decide them by the rounding and NaN rules the op must meet.

**Small-K MFMA can rescale the incoming accumulator.** When K is below the operand extent `k_width * 64 / nonK` (32 at `k_width` 8 and 64 at 16 on `[16, 16, 32]`), plain `mfma` returned `acc / 2 + a·b` at half that extent and `acc / 4 + a·b` at a quarter, the extent and not the instruction's K being the boundary (S58): 54 such variants with `k_width` ≥ kBase, within 3.2e−8–7e−8 of that formula, in two runs on one input seed *(field, E11)*. The bf16 `[16, 16, 32]` case at `k_width` 8 and K 16 has four `fmul … 0.5` per lane in its `llir`, none at K 32 *(field, E1)*. The formula holds for those plain calls only: below-extent calls with `k_width` under kBase failed to compile, and of 32 `mfma_scaled` variants 7 failed to compile or aborted and 3 e2m1 ones at `k_width` 32 had relative errors of 0.93–1.0 *(field, E11)*. At K 16 in bf16, `acc += gl.amd.cdna4.mfma(a, b, gl.zeros([BM, BN], gl.float32, mma))` returned `acc + a·b` *(field, E1)*.

**Localize a mismatch.**

1. Reproduce the failing check in a process of your own, with the harness's own inputs: build the failing input set
   as the harness does (the same generator, seed and value ranges), allocate each output with `torch.empty` in its
   storage shape and dtype, compute the reference from the same inputs, call the kernel's launch code and compare
   each output with the reference under the same bound. The measures then match the harness's: on an
   N 6144 × K 2048 bf16 GEMM at m = 16 with a planted error, the fraction of elements outside the bound read 0.8646
   where the harness printed 0.865 *(field, E1)*.
2. Bisect a multi-launch op by stage: compare each stage's scratch output with the same stage computed in torch; the
   first one that disagrees beyond its rounding holds the bug.
3. Dump a tile: a store under `if gl.program_id(0) == p:` into a debug tensor wrote only that program's tile, equal to
   torch within fp32 rounding, and a grid of 1 on that tile gave the same tile as the full grid *(field, E1)*; or
   `gl.device_print` it (§4); check where each element lives with `format_tensor_view` (§4). In packed bit arithmetic,
   parenthesize: Python binds `&` looser than `+`, `-`, `*`, so `15 - 3 * (row & 3) - 2 * part & 15` is
   `(15 - 3 * (row & 3) - 2 * part) & 15`.
4. Remove the debug outputs before the final check: a kernel writes only its outputs, the inputs the op defines as
   mutable and its scratch; a debug store into the read-only input `a` of that GEMM failed a check that compares the
   inputs before and after the call *(field, E1)*.

## 6. Crashes and hangs

**The process exits with code −11 after one constant changed.** A harness that runs the kernel in a child process
reports its death by the exit signal; −11 is SIGSEGV, which the compiler can raise as well as the kernel. Setting the
fold constant of the one-token gate-up launch of MM256-FP4 (the MiniMax-M2 MXFP4 MoE with all 256 experts on the rank) from 4 to 1 crashed every run *(field, E10)*:
`gate_fold = 4 if m <= 2 else …` becomes the launch's `FOLD`, beside `NATIVE_FOLD=m in (1, 8)` and, at m ≤ 2,
`llvm_fn_attrs` with `amdgpu-sched-strategy=iterative-ilp`:

```python
gate_fold = 4 if m <= 2 else 2 if m == 4 else 8 if m == 8 else 1
gate_unroll = 6 if m == 2 else 3
gate_attrs = (["amdgpu-sched-strategy", "iterative-ilp"],) if m <= 2 else ()
_gate_up[gate_grid](
    # ... the tensors and sizes ...
    K_UNROLL=gate_unroll,
    FOLD=gate_fold,  # 1 in place of 4 at m = 1, nothing else changed: the compile segfaults
    PROJECT_ROWS=8 if m == 8 else 4 if m == 16 else 16,
    TRANSPOSED=m not in (1, 2, 8),
    NATIVE_FOLD=m in (1, 8),
    llvm_fn_attrs=gate_attrs,
)
```

Compiled alone without a GPU, that launch segfaults inside `make_amdgcn` (LLVM's code generation, S29) and compiles
once the scheduler attribute is dropped *(field, E10)*; [moe.md](moe.md) §7 gives the other combinations that compile.

Procedure: compile the changed launch alone (§1) in a subprocess under `python -X faulthandler`, which dumps the Python
frames of a native crash (the FOLD=1 segfault: `make_amdgcn` at its `llvm.translate_to_asm` call; the
`gl.bank_conflicts` and rank-3 `mfma_scaled` aborts: `make_ir` and the Gluon call). In `make_amdgcn`, try dropping
`llvm_fn_attrs` (above; the kernel's 16-token launch also compiles). In a process of your own, a host sync after each
launch pins the one that crashes: the sync after a store 2^40 elements away aborted (exit −6), the one before it
passed *(field, E1)*.

**The compile fails with a `NameError` after one constant changed.** A constexpr `if` chain defines a name on some
branches only, and the new constant takes another. In a 64-head DSA sparse MLA kernel at 24 tokens,
`TILES = cdiv(TOPK, SPLITS * 256)`: 8 splits give each program one 256-key tile (`TILES` = 1) and the branch that
allocates `cache_shared`; 4 splits give `TILES` = 2 and the branch that allocates `cache0`/`cache1`, while the 64-head
path still reads `cache_shared`:

```python
TILES: gl.constexpr = triton.cdiv(TOPK, SPLITS * 256)  # TOPK 2048: 8 splits give 1 tile, 4 splits 2
# ...
if TILES > 1:
    cache0 = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, 256], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
    cache1 = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, 256], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
else:  # (an `elif HEAD_TILE == 32:` before it allocates cache_shared in a padded layout)
    cache_shared = gl.allocate_shared_memory(
        Cache.dtype.element_ty, [256, VALUE], gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0])
    )
for tile in gl.static_range(TILES):
    # ...
    if HEAD_TILE == 64:
        key_rope = gl.convert_layout(rope, score_rhs)
        scores = _wide_query_scores(
            Query,
            cache_shared,  # no branch defined it when TILES > 1
            # ...
```

The compile raised `NameError('cache_shared is not defined')` in every run *(field, E10)* ([attention.md](attention.md)
§4). Read the host rule that sets a constant and every constexpr branch that reads it before changing it, and guard a
branch that assumes a combination with `gl.static_assert`, so a lone change fails with your message: the 16-token
kernel of that family states `gl.static_assert(TILES == 1 or PANEL_CACHE)`, the 24-token one nothing.

**A launch never returns.** A kernel that never ends (a spin on a flag nobody sets) is stopped only by the harness's
timeout, if it has one *(field, E3)*. Run the launches in a process of your own under
`timeout -k 5 <s>` with a host sync after each: the one that never returns hangs.

**A GPU memory access fault** comes from the kernel that faulted, not the one that wrote out of bounds: a store past a tensor into the next allocation synced cleanly, and the fault (exit −6) came after a later kernel used the overwritten value as an index *(field, E1)* ([fault_in_the_wrong_kernel.md](../pitfalls/fault_in_the_wrong_kernel.md): a host sync after each launch outside capture). Whether an out-of-bounds access faults depends on what lies there: `buffer_load`, `buffer_store`, `gl.load` and `gl.store` 0x7ffffc00 bytes from the base of a 4 MiB tensor faulted, while 4 KiB past its end, inside the next allocation, all four synced without a fault, reading or zeroing the neighbour (two runs) *(field, E8)*. Inside mapped 2.5–4.5 GiB tensors, buffer accesses of 1–16 bytes at byte offsets from `0x7ffffffe` on did not fault: they read 0 or wrote nothing (93,511 bytes per operation checked, one run), and a vector crossing that offset read or wrote only its in-range part *(field, E1, E8)*. i32 byte offsets reaching 2^32 wrapped to a wrong in-range address *(field, E1, E8)*: gate a buffer path on the tensor's byte extent against `2**31 - 2` ([memory.md](gluon/memory.md) §3). `out of resource: shared memory` at the first launch: §4.

## 7. Profiling on the sglang image

- **rocprofv3 1.3.5 aborts on a process that imports torch** in `lmsysorg/sglang:v0.5.22-rocm10-mi35x` (ROCm 10.0.0,
  torch 2.11.0+rocm10.0.0): `: CommandLine Error: Option 'spirv-expand-step' registered more than once!`,
  `LLVM ERROR: inconsistency in registered CommandLine options`, then a hang that SIGTERM does not end, with
  `--kernel-trace` and with `--pmc` alike; `import triton` alone runs *(field, E12)*. `timeout` without `-k` does not
  end it: use `timeout -k 5 <s>` or `kill -9`. So no kernel launched from a process that imports torch can be traced
  or counted with rocprofv3 there.
- **`torch.profiler` works.** A graph-replay benchmark can sum the device durations of a call's kernels from
  `torch.profiler` (Kineto), as the one behind this page's ratios did *(field, E12)*, or from a rocprofiler-sdk
  collector loaded into the process; either gives one number for the call. For each launch's share,
  profile a few eager calls under `torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])` and read
  `prof.key_averages().table(sort_by="device_time_total")`, shares only: hot, a DSA decode's two launches summed 15 %
  under their time in flushed graph replays *(field, E12)* ([profiler_misses_kernels.md](../pitfalls/profiler_misses_kernels.md)).
- **LDS bank conflicts.** `gl.bank_conflicts` aborts the gfx950 compile (§4; a swizzled and a padded layout), and the
  counters ([lds_bank_conflicts.md](../patterns/lds_bank_conflicts.md)) are untested here. A timed micro-kernel shows
  them: in 4 runs, a plain 64 × 64 bf16 tile read LDS-bound as a `[16, 16, 32]` MFMA operand (one wave per program) 1.70–1.73× (operand 0) and 1.75–1.77× (operand 1) slower than the XOR-swizzled one *(field, E13)*, and 1.0–3.5× with other MFMA shapes, tile sizes or wave counts *(field, E23)* ([layouts.md](gluon/layouts.md) §5). The recipe: only the construct, in a loop, in enough
  programs to fill the GPU (one program took 305–306 µs, 2048 programs 907–916 µs: an under-filled grid times latency,
  not rate); the median of several rounds (single rounds spread 11–13 %, medians within 2 % across runs); the ISA read
  to confirm what you time *(field, E13)*. Traces where rocprofv3 runs: [profile.md](../profile.md).
- **Layout conversions.** Converting a 16 KiB tile of 32-bit values between lanes and back once per iteration added about 2.4×, 2.2× and 2.2× the control loop's time through DPP, `v_permlane16_swap` and `v_permlane32_swap`, and 7.5–9.4× through `ds_bpermute` or LDS; a conversion that only renamed registers added nothing measurable *(field, E14)*. The multiples depend on the control loop; per moved 32-bit value and conversion the added time was 0.08–0.10 ns through DPP and permlane and 0.17–0.18 ns through `ds_bpermute` at 8–32 KiB *(field, E24)*. Per conversion (half an iteration) the same loop took 73.14–74.12 ns without one, 661.4–662.8 ns through `ds_bpermute` and 1058–1060 ns between warps *(field, E15)* ([layouts.md](gluon/layouts.md) §8). Read the path taken from the ISA: `ds_bpermute_b32`, `v_mov_b32_dpp`, `v_permlane16_swap` / `v_permlane32_swap`, `ds_write` + `ds_read` (plus `s_barrier` when waves move), `v_perm_b32` for an 8-bit register permutation, or none.

## 8. Host-side cost of a launch

What one more launch costs on MI355X, in a replayed graph, under a timer that sums a call's kernel durations and
eagerly, measured on a trivial Gluon kernel: [authoring.md](authoring.md) §7. Eager host time grew 1.0 µs per tensor argument: on HIP, Triton's Python
specialization callback (S26) runs per tensor per launch (0.58 µs a call), and its `S` check `is_within_2gb` ran on
every launch while `AMDGCN_USE_BUFFER_OPS` is on, the default (S27; 300 calls in 300 launches, none with it off)
*(field, E1)*; each launch also hashes its constexprs (§4). Turning `AMDGCN_USE_BUFFER_OPS` off changes plain-Triton
code (its loads stop being buffer ops) and is in the cache key; its effect on host time is not measured.

A replayed graph pays none of the host cost; eager calls (an uncaptured many-token batch, as in prefill) pay it on every launch. The ruler's timer, the per-call time of the ruler ([tuning.md](tuning.md) §1), sums the device durations of the call's kernels in a graph replay, the flush excluded, and takes them from a profiler: a rocprofiler-sdk collector or Kineto. Under Kineto with `GPU_FORCE_QUEUE_PROFILING=1` in the measuring process: host time did not show (a 2 ms sleep in the callable: +0.25 %), an extra trivial kernel added 4.06 µs against 4.00 µs of back-to-back replay, and no kernel read below 4.16 µs *(field, E1)*; a one-wave spin kernel read 4.22–4.48 µs in four child processes of two runs, and within each process kernels whose own time was 0.48–2.48 µs read within 0.6 % of each other while 6, 8 and 12 µs kernels read their own time + 0.93–1.15 µs *(field, E16)*: a floor that varies by process, not a hardware constant ([profiled_timer_floor.md](../pitfalls/profiled_timer_floor.md)).

## 9. Measured a different build

- **Triton's cache key** holds Triton's version, hashes of its compiler, backend and language files and of
  `libtriton`, the function's own key (its text, its first line number, its callees and the constexpr globals it
  reads), the signature, constexprs and attributes, the target, the options and the cache-invalidating environment
  (S10). Another Triton or an edited source compiles anew; a stale binary comes from a stage override (§1) or from a
  run that never imported your file ([measured_a_different_build.md](../pitfalls/measured_a_different_build.md)).
- **Which Triton ran.** Record the versions of Triton, torch, ROCm and FlyDSL with every result, and check after
  installing aiter that the image's Triton (3.8.0) is still the one installed (below). The setup these kernels
  were run under skips, without measuring it, a kernel whose stated Triton requirement the installed Triton does not meet: of the library's 154 kernels, 135 required `>=3.8` (261 of the 280
  after its growth), and the other 19 (the 15 N = 3072 fp8 GEMMs, the two plain bf16 GEMMs and the two RMSNorms) run
  on whatever Triton is installed *(field, E2)*.
- **Same version, another build.** The image's Triton 3.8.0 is built from another tree than the PyPI
  `triton==3.8.0` wheel: the padded `gl.bank_conflicts` assertion of §4 sits at `LinearLayoutConversions.cpp:1364` in
  the image and at `:1379` in the wheel and in the v3.8.0 tag *(field, E12)*: confirm compile-only findings on the
  image.
- **The baseline's aiter.** Most baselines behind this page's ratios call aiter (the bf16 GEMM's is its tuned `gemm_a16w16`), so the
  aiter commit is part of the measurement. Pin it: the setup these kernels are measured in installs one ROCm/aiter
  commit (dbd8bf5, later v0.1.29) over the sglang image's own (285579b6 in v0.5.23; e7d2453f2 in v0.5.22, read from the image's build in
  [measured_a_different_build.md](../pitfalls/measured_a_different_build.md)), while a kernel that replaces a path inside sglang or vLLM is timed against the aiter its image ships. Check that pip installed the commit you asked for (its `direct_url.json`), and
  print `importlib.metadata.version("amd-aiter")` beside your results. A run on the image's aiter times
  another baseline. Install the pinned one with the variable that keeps your Triton (without it, aiter's setup swaps in its Triton):
  `AITER_USE_SYSTEM_TRITON=1 pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@<commit>"`.
- **An unchanged kernel's ratio moves between runs of the same sources**: over three runs of the ruler
  (§11) on one MI355X, 2.8 % for the Gluon fp8 GEMM (N 3072 × K 1536) at m = 16 and 0.1 % for the Qwen2 MoE router at
  256 tokens *(field, E10)*; a ratio at one size also ranks the baseline there ([tuning.md](tuning.md) §1). Timed in
  one run directly against the unchanged kernel instead of the baseline, an identical copy of each read 0.999–1.000
  (3 runs each) and §10's 1.2 % slower change 0.988 *(field, E10)*: settle a 1–2 % question that way. The flush and
  ruler: §11.

## 10. Kernel time behind the ratio

A ratio says nothing about which side moved. Keep, for each size, the per-round times of both sides: the kernel's
and those of what it is timed against (the baseline, or the unchanged kernel when a change is timed against it), over
16 or 32 rounds, with the mean per-round log ratio and its one-sided lower bound. A bound computed with each round's
log ratio clamped to ±ln 2 saturates: a size at 2.48× reads 2.000 *(field, E3)*; state the speedup as the bound on the
unclamped rounds. Read both sides' medians in microseconds beside the speedup.

The kernel's microseconds say what a change did: 3 prefetched K groups instead of 5 in the m = 16 fp8 GEMM
moved the ratio 1.475 → 1.285 and the kernel 4.23 → 4.85 µs; 8 KV splits instead of 16 in gated-attention decode at
32 tokens, 1.79 → 1.129 and 44.18 → 69.77 µs *(field, E10)*. At that GEMM's m = 4096 size (25.6 µs), a grouped
tile order with 1 M tile per group instead of 4 read slower in all 4 alternated pairs of kernel times
(25.65 → 25.96 µs, 1.2 %), and timed directly against the unchanged kernel it read 0.988 *(field, E10)*
([tuning.md](tuning.md) §2).

## 11. A micro-benchmark that disagrees with a graph-replay benchmark

Check, in order, against the ruler behind this page's ratios ([tuning.md](tuning.md) §1: kernel and baseline timed round by round):

1. **Time with the ruler's timer.** The ruler captures as many calls as fit a 20 ms budget into one graph, each behind a 1 GiB read flush, replays the graph 10 times and returns the median over calls of each call's kernel time averaged over the replays, the flush excluded. With the durations read from a rocprofiler-sdk collector, one that registers as its module is imported (or, when imported before torch initializes the GPU, as torch does) has to be imported before the process's first kernel, with `ROCP_TOOL_ATTACH` set and `GPU_FORCE_QUEUE_PROFILING` left unset. With the durations read from Kineto (the median of 10 replays' per-call averages, the measuring process under `GPU_FORCE_QUEUE_PROFILING=1`), readings in a separate process were within 2.2 % of the full ruler's at four sizes above the floor (fp8 GEMM m = 4096, a DSA decode, gated attention 32 rows, MiniMax MoE 256 tokens; two runs, two runs of the full ruler each), but −1.9 % to +4.1 % at the floor-bound fp8 GEMM N 3072 × K 1536 at m = 16 across six process pairs *(field, E16)*: two processes need not agree. Drop the first reading, as the full ruler drops an unrecorded warm-up: after one eager call it lay within 0.01 µs of the next nine's range in the five processes tested *(field, E1)*.
2. **The flush.** The ruler reads a 1 GiB buffer before each call, inside the graph, and does not write one: written lines would stay dirty for the next kernel. On a stream reading 64 MB to 1 GiB once per launch (4 warps, 2048 programs, timed eagerly with events, four runs), plain loads ran at 2.52–4.73 TB/s after a 1 GiB `zero_()` write flush against 4.43–6.02 after a 1 GiB Gluon read flush (write over read 0.49–0.79), while `.cg` loads were not slowed (4.70–6.64 against 4.65–6.67, 0.985–1.022), so a write flush inflates `.cg` over plain to 1.40–2.22× from 1.04–1.12× *(field, E17)*. Unhinted, a read-flushed stream ran 9–19 % below a hot one at 64–192 MB (hot over flushed 1.102–1.224), most at 128 MB in each run, and within 1.4 % of it at 512 MB–1 GiB (0.991–1.014) *(field, E18)*. These eager probes show that the kind of flush matters, not what the ruler's replayed calls read from MALL or HBM.
3. **Graph replay against eager.** An eager loop of a short kernel times Triton's host path (§8).
4. **The floor.** With the durations read from Kineto, every kernel the ruler timed read about 4.1 µs or more (§8);
   the floor under a rocprofiler-sdk collector is not measured here.
5. **The statistic.** The ruler's speedup ([tuning.md](tuning.md) §1) is a one-sided lower bound of the mean per-round log ratio over 16 or 32
   unclamped rounds, not a ratio of two times: compare the medians of both sides (§10)
   ([speedup_from_another_ruler.md](../pitfalls/speedup_from_another_ruler.md)).
6. **The inputs.** The ruler's timed calls reuse one weight buffer: all 1556 timed calls of the Gluon fp8 GEMM in the
   measuring process got the same weight `data_ptr`, so a cache keyed on the weight's address hits on all but the
   first *(field, E3)*.

## Evidence

Triton paths are `triton-lang/triton@c01b677` (tag v3.8.0). "The ruler" is that of [tuning.md](tuning.md) §1, "the ruler's timer" its per-call time alone (§11); in
E1–E18 it read the kernel durations from Kineto.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (crash localization 3 runs; the 2 GiB range, determinism, denormal, MoE ticket and JIT key probes 2), probe kernels and planted copies run on one GPU] an unmasked `buffer_load` at offsets 64–79 of a 64-element view of a larger allocation read 64.0–79.0, masked lanes 0 or `other` (−7.0), a masked store left its lanes; inside a mapped 2.5–4.5 GiB tensor, u8 buffer loads at byte offsets 0x7ffffffe to 2^32 − 1 read 0 over non-zero data, int32 stores at 2^31 to 2^31 + 4000 left 7 in place, fp32 element offsets 2^30 + k returned x[k], no fault; `gl.device_print` of an 8 × 8 fp32 tile, one element per lane: 64 lines, each element once; 1024 programs of 64 lanes: 65536 lines, 64 under `if gl.program_id(0) == 3`; the standalone check of §5 on an N 6144 × K 2048 bf16 GEMM: the fraction of elements outside the bound 0.8646 on a 1.01-scaled copy at m = 16 where the harness printed 0.865; a tile dumped from program 5 of 8 equal to torch within allclose 1e-6 (2 of 3 runs; the first compared bitwise), the rest untouched, a grid of 1 identical; a debug store into input `a`: failed the check that compares the inputs before and after the call, exit 1; the sync after a store 2^40 elements away aborted (exit −6), the sync before it passed; a store past a 64-int32 tensor into the next allocation synced, a later gather indexed by the overwritten value faulted (exit −6); a spinning launch under `timeout -k 5 20`: exit 124 after 20.08 s; `gl.exp2` of −126 … −129: 1.18e-38, 0, 0, 0; bf16 K = 16 on [16, 16, 32] with `k_width` 8: 0.32 ulp from acc / 2 + a·b, four `fmul … 0.5` in llir, none at K = 32; K = 32 with `k_width` 16 and `mfma_scaled` K = 64 with `k_width` 32: acc / 2 + a·b; a fresh zero accumulator at K = 16: relative error 2.8e-8; a 4-row, 4-split float-atomic row sum: two calls missed the difference in 17 and 18 of 20 batches, 200 calls in none; a 24-split atomic GEMM: 100 distinct outputs in 100; no K split against a six-way split of the N 2624 × K 6144 bf16 GEMM at m = 64: 75 of 167,936 bf16 outputs differ, peak 0.785 of the bound; 65 kernels that must give the same bytes in two runs with no atomic, 42 global and 4 LDS atomics in the ones held only to a numeric bound, all on int32 or uint32; 8 ids in a 64-lane layout: counters 1, 4 built replicas: counters 4; at launch, ints 2 and 17 typed `i32`, 16 and 32 `i32` + `D`, 2^31 `i64`, 2^63 `u64`, 4 and 8 sharing the binary of 2; pointers at offsets 0 and 256 of an allocation `D`, at 4 and 8 not; a 4 KiB storage `S`, a 2 GiB + 16 KiB one not; a counting constexpr hashed 13 times in 12 launches of one binary; `is_within_2gb` called 300 times in 300 launches, 0 with `AMDGCN_USE_BUFFER_OPS=0`; a list constexpr (`INSTR=[16, 16, 32]`), a `gl.distributed_type` and a `gl.shared_memory_descriptor_type` constexpr, the option `waves_per_eus`, 167936 and 262144 B of LDS (limit 163840) and a reassigned global each compiled with `triton.compile` and failed at the launch; eager host time 7.32 µs with 1 tensor and 38.45 µs with 32 (1.004 µs per tensor), the specialization callback 0.58 µs per call; the ruler (Kineto): a 2 ms sleep in the timed callable 6.523 against 6.507 µs, per extra kernel 4.062 µs against 4.001 µs of back-to-back replay, trivial Gluon kernels 4.166 and 4.173 µs, torch `zero_` 4.185, `add_` 4.167, a split-K reduce 4.222; the ruler's timer alone in a separate process 4.229 µs against the full ruler's 4.309 µs at the fp8 GEMM N 3072 × K 1536, m = 16; first readings 4.341, 4.364, 4.213, 4.191, 4.188 µs against the next nine's ranges 4.304–4.337, 4.339–4.369, 4.222–4.254, 4.189–4.210, 4.177–4.210 (probe kernels, the N 6144 × K 2048 bf16 GEMM and the library's 154 kernels)
- E2 [2026-11, gfx950, Triton 3.8.0, N=1, `triton.compile` for gfx950 with the amdgcn and llir read, or a count over the library's 154 kernels; launched on the MI355X image where a GPU was needed] the knobs of §1 (`MLIR_ENABLE_DUMP=1`: 29 "IR Dump Before", none after, none under another kernel's name, and one "IR Dump After" for a compile failing in a pass, in 2 runs; `TRITON_DUMP_MIR`: `.amdhsa_user_sgpr_kernarg_preload_length` 10 → 0; `waves_per_eu` 0 or 2, `num_warps` 4 or 8, or `llvm_fn_attrs`: two cache entries, one dump directory holding the later build; an override with the dumped hsaco beside the edited amdgcn kept the original md5; `debug=True`: 3 `s_trap` and the int32 overflow asserts, none without); a torch stub with cached placeholder attributes: the sources of 154 of 154 kernels imported, 131 failed with a bare module; the §2 rows on 14 builds (`.vgpr_count` = `.amdhsa_accum_offset` + `.agpr_count`: 256 + 102 = 358, rounded to 360; accum offset 66 → 68, 94 → 96, 134 → 136; `.sgpr_count` = next free SGPR + 6, s73 → 80; a 4-warp 128 × 128 `tl.dot` at `waves_per_eu` 2: 135 spilled VGPRs, 66 scratch stores, 352 scratch bytes; the VGPR bound equal to `; Occupancy:` in all 14, at 20–512 registers) and the 4-warp bf16 MFMA probe (438 VGPRs with 182 AGPRs, 168 at `waves_per_eu` 3 and 72 at 7; `isa_diff.py --stream` of 0 against 2: agpr 182 → 0, 181 spills, 728 scratch bytes, scratch stores and loads +153 each, `v_accvgpr_write_b32` −165); the §3 rows (a unit stride as `i32`: 8 `buffer_load_ushort`, as the constexpr 1: one `buffer_load_dwordx4`; masked loads with `n` marked `D`: one `buffer_load_dwordx4`, without: 8 `buffer_load_ushort`; a row-only mask with a runtime `m`: 2 `buffer_load_dwordx4`; a Gluon [64, 64] bf16 copy: 2 `global_load_dwordx4`, no buffer op; `num_records` 0x7ffffffe in four Gluon kernels; `wait_group(1)`: `vmcnt(2)` + `s_barrier` at 4 warps, none at 1; an inline `vmcnt(1)` with 2 instructions behind the copy; a 4-wave global read-then-write: no `s_barrier`; 48 `v_dot2c_f32_bf16` in the router launch of DS256 at 2 tokens; `waves_per_eu` 2: 181 spilled VGPRs, 153 scratch stores; `"amdgpu-waves-per-eu"="0,0"` on the image); the messages of §4 and §5; 135 of the library's 154 kernels requiring `>=3.8`; 1709 Triton kernels compiled by benchmarking each of the 154 at its sizes on MI355X (one Triton cache per kernel, the cached `.amdgcn` and `.json` read), 897 of them the files' own: `.group_segment_fixed_size` non-zero in none, `metadata.shared` > 0 in 538 of the 897 (probe kernels and the library's 154 kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, a benchmark harness that checks and times each kernel in a child process, on one GPU, on planted copies of working kernels] a child that aborts or segfaults: its exit code −6 / −11 only, no log of its output left after the run; a callee `NameError`: the failed compile reported with the call site only; a lowering error: `RuntimeError: PassManager::run failed`; the scan diagnostic, `static_print` and `device_print` output absent from the console and the report (in one run; in the other the only match was the marker inside the reported source text); a GEMM rounding its output through fp16: outside its bound only on the large-magnitude input set, 5248 non-finite; a device counter wrong from the 2nd or 3rd replay: wrong at the 2nd or 3rd replay of the captured graph; a host read in the launch code or a per-call constexpr: the capture failed, for the constexpr with a compile inside the capture; a kernel that must give the same bytes in two runs got a bytewise rerun of one input set, one held only to a numeric bound none; a spinning kernel alone in its batch: stopped by the harness's timeout, the run 1204 s; fresh caches after a Gluon fp8 GEMM (N 3072 × K 1536) and a plain-Triton RMSNorm: the kernel's one launched function and the harness's L2-flush kernel only; a broken MFMA call on line 71: `<imported copy>/kernel.py:71:11: error:` (one run); a copy of the fp8 GEMM with 3 prefetched K groups timed against the unchanged kernel: two cache entries from two imported copies (one run); 42 timings of 16 or 32 rounds, a size at 2.48× by its mean log ratio with a clamped lower bound of 2.000; a copy of the fp8 GEMM leaving 8 rows unwritten: 24576 non-finite mismatches in the first check; 1556 timed calls of the fp8 GEMM in one measuring process on one weight `data_ptr` (copies of the fp8 GEMM and the RMSNorm)
- E4 [2026-11, gfx950, Triton 3.8.0, N=1, `triton.compile` for gfx950 without a GPU] Gluon's `asm['source']` held TTGIR with `#ttg` layouts and plain Triton's TTIR, and only plain Triton had `asm['ttir']`; a small Gluon build under `MLIR_ENABLE_DUMP=1` printed 29 "IR Dump Before" entries matching the 29 registered passes in order; `enable_fp_fusion` at its default or True gave one `v_fmac_f32`, False (or the environment knob off) one `v_mul_f32` and one `v_add_f32`, while the saved `llir` could be the same (probe kernels)
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, Gluon kernels whose every wave records its start and end ticks with HW_ID and XCC_ID; resident workgroups per CU = the largest overlap on one CU] 4 warps with 8 KiB of LDS at 17, 65, 129, 256 VGPRs: 8, 7, 3, 2 per CU in every run; 8 warps at 17 and 129 VGPRs: 4 and 1; 48, 64, 160 KiB of LDS: 3, 2, 1; 32 KiB of LDS: 4 per CU in 3 of 3 runs, at most 1024 = 4 × 256 live on the GPU; 8 warps at 65 VGPRs (7 waves per SIMD, so 3 by registers): 3, 3 and 4 per CU, at most 768 = 3 × 256 live on the GPU in each run (residency probe kernels)
- E6 [2026-11, gfx950, Triton 3.8.0, N=3, `triton.compile` for gfx950 without a GPU] 7 pointer forms of `gl.load` / `gl.store` with `DS` attributes and 284 kernels (432 launches) of 150 kernels of the library that call no explicit buffer or async-copy API: no buffer memory access and no failed compile; uniform loads compiled to `s_load`; atomics brought `buffer_inv` / `buffer_wbl2` cache maintenance (probe kernels and the library's kernels of E2)
- E7 [2026-11, gfx950, Triton 3.8.0, N=2, `triton.compile` for gfx950 without a GPU] layout mismatches between elementwise operands, a store's value and pointer, or a load's mask and pointer: 54 builds over `BlockedLayout`, `SliceLayout`, `AMDMFMALayout` and `DotOperandLayout` pairs and 9 with a `DistributedLinearLayout` equal to the `BlockedLayout` beside it raised `Layout mismatch in broadcast`, 72 matching controls compiled; 14 `warps_per_cta` mismatches (1-D and 2-D, `num_warps` 4, 8 and default) failed verification; 35 non-power-of-2 extents (3–384) rejected, 12 power-of-2 controls compiled; runtime values in `gl.arange`, `gl.static_range`, layout parameters and `gl.zeros` shapes, and code after a returning constexpr `if`, raised the messages of §4; the plain `mfma` messages as listed there, while `mfma_scaled` with swapped operand indices or unequal `k_width` compiled (32 builds whose message differs by API); a rank-3 `mfma_scaled` layout with `None` scales aborted in 36 builds, 12 rank-2 controls compiled (Gluon probe kernels)
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the byte-level probe N=1), Gluon and plain-Triton buffer probes compiled for gfx950 and run on one GPU] 26 descriptor configurations per run: `num_records` the literal 2147483646 in llir and ISA; 40 active and 24 masked lanes: masked offset `0x80000000`, masked loads 0 (Gluon without `other`) or −7 (`other`), masked stores left memory untouched; `buffer_load`, `buffer_store`, `gl.load` and `gl.store` 0x7ffffc00 bytes from a 4 MiB tensor's base faulted, 4 KiB past its end inside a checked adjacent allocation did not; in mapped 2.5–4.5 GiB tensors no fault; 93,511 bytes per operation at or beyond byte offset 2^31 − 2 read 0 or were not written, vectors crossing it read or wrote only their in-range part, i32 byte offsets reaching 2^32 wrapped (buffer probe kernels)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 4096 programs per configuration, counters read back] four layouts that hold an element in several registers × four adds (int32 and fp32 `gl.atomic_add`, fp32 and uint32 `buffer_atomic_add`) × three masks (none, all-true, even elements): 8192 per active element in all 48, where one add per element gives 4096; 135 plain-Triton cells counted once; built replicas, 8 elements, R = 2, 4, 8, one warp, int32 and fp32: R × 4096 without an owner mask, 4096 with `replica == 0` (Gluon and plain-Triton atomic probes)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU (kernel and baseline in CUDA-graph replays behind a 1 GiB read flush, Kineto kernel sum): a copy with one change alternated with the unchanged kernel, 4 runs each (2 per DSA kernel for `allow_flush_denorm`); an unchanged kernel 3 times against the baseline and 3 times against itself; a slower copy in 4 alternated pairs; offline compiles for gfx950 of the crashing launch] DSA sparse MLA decode, 64 heads, 24 tokens with KV splits 8 → 4: `CompilationError … NameError('cache_shared is not defined')`, in 4 of 4 runs; MM256-FP4 at 1 token with its gate-up fold 4 → 1: the child process exited with code −11, in 4 of 4; the unchanged kernels passed 4 of 4; that FOLD = 1 launch compiled alone without a GPU segfaulted in `make_amdgcn` (2 of 2) and compiled without `llvm_fn_attrs` (2 of 2), as did the kernel's 16-token launch; fp8 GEMM N 3072 × K 1536 at m = 16 with 5 → 3 prefetched K groups: speedup 1.475 → 1.285, kernel 4.232 → 4.851 µs, every pair 1.143–1.150× slower; gated attention decode at 32 tokens with KV splits 16 → 8: 1.79 → 1.129, 44.18 → 69.77 µs, every pair 1.574–1.586× slower; Qwen2-MoE router at 256 tokens without `["amdgpu-ieee", "false"]` on its routing launch: 1.910 → 1.921 (B/A 1.006), the unchanged kernel's four runs spread 0.94 %; DS512 at 8 tokens with `enable_fp_fusion` False → True on its 3 launches: 1.147 → 1.146 (B/A 0.9994); `allow_flush_denorm=True` removed from the 4 DSA 64-head decode kernels (12, 16, 24, 32 tokens): B/A 1.0007, 1.0033, 0.9986, 0.997, instructions and `.amdhsa_` fields identical in 8 of 8 kernels; accuracy passed in every run; the m = 16 fp8 GEMM's ratio to its baseline 1.4784, 1.4386, 1.4749 (max over min 1.028), the 256-token Qwen2 router's 1.9259, 1.9242, 1.9247 (1.001); an identical copy of each timed against the unchanged kernel: 0.999, 0.999, 1.000 for both; fp8 GEMM N 3072 × K 1536 at m = 4096 with 4 → 1 M tiles per group of its tile order: 25.648, 25.643, 25.661, 25.648 against 25.932, 25.974, 25.971, 25.951 µs, slower in 4 of 4 pairs (+1.20 %), timed against the unchanged kernel 0.988 (the kernels named)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the same random seed in both runs, results checked on the host] 138 plain `mfma` variants per run, 54 with K below the operand extent and `k_width` ≥ kBase: the accumulator scaled by 1/2 or 1/4, errors 3.2e−8–7e−8 against that formula; 32 `mfma_scaled` variants: 7 compile errors or aborts and 3 wrong e2m1 results at `k_width` 32, relative error 1 and 0.95 for `[16, 16, 128]` at K 32 and 64, 0.93 for `[32, 32, 64]` at K 32 (MFMA probe kernels)
- E12 [2026-11, gfx950 (MI355X), lmsysorg/sglang:v0.5.22-rocm10-mi35x (ROCm 10.0.0, rocprofv3 1.3.5, torch 2.11.0+rocm10.0.0), Triton 3.8.0 (3.8.0+git4cff872c), N=1, each rocprofv3 run under `timeout -k 5 90`] `rocprofv3 --kernel-trace` and `--pmc` on processes importing torch, with or without triton: both messages, alive past SIGTERM, ended by SIGKILL (rc −9); `import triton` alone: rc 0; `timeout 10` without `-k`: alive 60 s past its deadline, ended by SIGKILL to its group; a Triton GEMM call under `--kernel-trace`: no trace file; the ruler: the sum of the GPU kernel durations Kineto recorded, one number per round; a 64-head, 16-token DSA decode under `torch.profiler`: its split and merge launches, hot sum 12.09 µs against the ruler's 14.25 µs; the padded `gl.bank_conflicts` assertion at `LinearLayoutConversions.cpp:1364` on the image and at `:1379` with the PyPI wheel, as in the v3.8.0 tag (the sglang image)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, LDS read micro-kernels on one GPU, median of 7 rounds of 3 launches, 2048 one-wave programs] `gl.bank_conflicts` on an XOR-swizzled and on a padded layout: the compiling child aborted (exit −6) in 4 of 4 runs; a 64 × 64 bf16 tile read LDS-bound, plain over XOR-swizzled 1.701–1.731 (operand 0) and 1.748–1.765 (operand 1); one program of the plain loop 305.0–305.9 µs, 2048 programs 906.8–915.6 µs; single rounds of the VALU-bound variant spread 95.7–111 µs (11–13 % of its medians), each of the 12 medians within 0.8–2.0 % across the 4 runs; reads `ds_read_b128` (operand 0) and `ds_read_b64_tr_b16` (operand 1) (LDS probe kernels)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 8 consecutive balanced repeat clusters, each 13 blocks × 5 launches] a 16 KiB fp32 tile converted between lanes and back once per iteration (two conversions), 1024 programs, 1023 iterations, values correct: added time over the control loop about 2.4×, 2.2×, 2.2× through DPP, `v_permlane16_swap`, `v_permlane32_swap` (totals about 3.4×, 3.2×, 3.2×), 8.84×, 7.53×, 9.36× through `ds_bpermute` and two LDS variants; a register-only rename 0.99388 of the control in total (95 % interval 0.98304–1.00484) (Gluon conversion probes)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, a 16 KB tile of 32-bit values converted and converted back once per iteration in 1024 programs, median of 7 rounds of 3 launches, time per conversion = time per iteration / 2] 73.14–74.12 ns per half iteration without a conversion, 661.4–662.8 ns through `ds_bpermute` (96 per kernel, no LDS), 573.9–580.8 and 1518–1522 ns through LDS, 1058–1060 ns between warps, 7.79–20.8× the loop without it (Gluon conversion probes)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler's timer alone (per-call kernel sums from Kineto over replays of one CUDA graph) in 2 child processes per run, and the full ruler] one-wave spin kernels: floors 4.228 and 4.409, 4.476 and 4.375 µs per child; own times 0.48–2.48 µs read within 0.6 % of each other, 6, 8 and 12 µs read their own time + 0.93–1.15 µs; at four sizes above the floor (fp8 GEMM m = 4096, a 64-head, 16-token DSA decode, gated attention 32 rows, MiniMax MoE 256 tokens) separate-process readings within 2.2 % of the full ruler; at the floor-bound fp8 GEMM N 3072 × K 1536 at m = 16 up to +4.1 %, and −1.9 % to +4.1 % over six process pairs counting two earlier ones (probe kernels and the kernels named)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, a 4-warp, 2048-program Gluon stream reading 64–1024 MB once per launch, timed eagerly with CUDA events, median of 15 launches per size and flush] plain loads 2.52–4.73 TB/s after a 1 GiB write flush (`zero_()`) and 4.43–6.02 after a 1 GiB Gluon read flush, write over read 0.49–0.79 at every size; `.cg` loads 4.70–6.64 and 4.65–6.67, write over read 0.985–1.022; `.cg` over plain 1.40–2.22 after the write flush, 1.04–1.12 after the read flush (Gluon stream probe)
- E18 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the same Gluon stream without a cache hint, hot against after a 1 GiB read flush] hot over read-flushed at 64, 128, 192, 512, 1024 MB: 1.127, 1.165, 1.137, 0.994, 0.997; 1.123, 1.217, 1.179, 1.011, 0.991; 1.117, 1.176, 1.138, 1.002, 1.004; 1.102, 1.224, 1.198, 1.014, 0.995; a read-flushed stream 9.3–11.3 %, 14.2–18.3 % and 12.0–16.5 % below hot at 64, 128 and 192 MB, the largest gap at 128 MB in each run (Gluon stream probe)
- E19 [2026-11, gfx950, Triton 3.8.0, N=1, `triton.compile` for gfx950, `.vgpr_spill_count` against the dwords the `scratch_store_*` instructions write] the 4-warp 128 × 128 × 64 bf16 `tl.dot` at `waves_per_eu` 2: 135, the dwords of its 66 stores; of 22 other spilling builds 16 equal, 5 `tl.dot` builds 1–65 above, and an fp32 `tl.dot` read 2 with no scratch, its spills `v_accvgpr_write_b32` (probe kernels)
- E21 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 under `tf32`, JIT launches, one Triton cache per setting] `TRITON_F32_DEFAULT=tf32`: all 8 Gluon `mfma` variants (bf16, fp16, e4m3, int8, fp32, fp64) failed with `input_precision … Got tf32`, while an e4m3 `mfma_scaled` compiled, launched and equalled a float64 reference; unset and `ieee` all passed (probe kernels)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 3 blocks of 20 batches per configuration, bytes compared between 2 calls and across a batch of 200] a plain-Triton fp32 `tl.atomic_add` row sum of 64 columns in 4 splits: at 4 rows 2 calls missed the difference in 17–20 of 20 batches and 200 calls in 2–3; at 1 row 200 calls missed it in 6–9 (probe kernels)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 11 rounds of 5 launches, median plain/XOR-swizzled time ratio] Gluon loops reading a bf16 tile from LDS into an MFMA operand layout, 2048 programs: 64 × 64 at `[16, 16, 32]`, one wave, 1.715 (operand 0) and 1.790 (operand 1); other MFMA shapes, tile sizes or wave counts 1.023–3.458 (probe kernels)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 15 rounds of 5 launches, arms interleaved, an A/A copy within 0.005×] Gluon loops converting a tile of 32-bit values A → B → A per iteration (1024 programs, 1023 iterations), time added over a control loop: at 8 KiB 4.69–5.06× through DPP and permlane, 14.56–15.83× through `ds_bpermute` and LDS, that control's main loop all scalar; per moved value and conversion at 8–32 KiB 0.077–0.102 ns through DPP and permlane, 0.169–0.180 ns through `ds_bpermute` (probe kernels)
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
- S43 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:181 `parse_options` keeps only the keys that are `HIPOptions` fields
- S44 triton-lang/triton@c01b677:python/triton/runtime/jit.py:715 at the first compile of a launch, a keyword that is neither an option nor a parameter raises `KeyError("Keyword argument %s was specified but unrecognised")`
- S45 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:91 `buffer_load_to_shared`: "size per thread * bits per element must be 128 or 32" for the offsets layout, or lowering to LLVM fails
- S47 triton-lang/triton@c01b677:python/src/ir.cc:1982 a failed pass-manager run throws `PassManager::run failed`; every run enables MLIR's crash reproducer, printed to the console without `TRITON_REPRODUCER_PATH` (:1956) and written to that path with it (:1953), which turns a crash inside a pass into a failed run
- S48 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:273 "no matching matrix core intrinsic" when no instruction has the requested shape and types
- S49 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:474 "TODO: unsupported scan layout"
- S50 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/Dialect.cpp:729 "Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension"
- S51 triton-lang/triton@c01b677:python/triton/language/semantic.py:201 "Scalar {lhs_scalar} is out of range for type {ret_sca_ty}" (also :204)
- S53 triton-lang/triton@c01b677:python/src/specialize.cc:274 an int equal to 1 is specialized as a constexpr; other ints get `i32` or `i64` (:281) and `D` when a multiple of 16 (:283); a pointer `D` when 16-byte aligned (:363); the dtype names `"float8_e4m3fn": "fp8e4nv"`, `"float8_e4m3fnuz": "fp8e4b8"` (python/triton/_utils.py:87, :89); two Qwen3.5 GDN kernels look an fp8 dtype up in a table at import (a module-level config built from `torch.float8_e4m3fn`, raising "unsupported FP8 dtype" otherwise)
- S54 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:489 `set_all_fn_arg_inreg` (kernel arguments preloaded into SGPRs) is skipped when `TRITON_SWAP_MIR` or `TRITON_DUMP_MIR` is set; the debug-info scope pass runs unless `TRITON_DISABLE_LINE_INFO` (:399); `kpack` is reset to 1 on gfx950 with a warning (:119-123), `matrix_instr_nonkdim` is read only by `make_ttgir`'s matmul pass (:273) and `schedule_hint` (:105) nowhere in the backend; `make_hsaco` assembles (:574) and links (:579), a failed link raising "LLD failed to link hsaco source" (third_party/amd/python/triton_amd.cc:546)
- S55 triton-lang/triton@c01b677:python/triton/runtime/jit.py:486 a function's source starts at its `def` line, from which `CompilationError` counts `at {lineno}:{col_offset}` (python/triton/compiler/errors.py:25); "Global variable {name} has changed since we compiled this kernel" (:766); `@triton.jit` returns an `InterpretedFunction` under `TRITON_INTERPRET` (:984), `gluon.jit` a `GluonJITFunction` regardless (python/triton/experimental/gluon/_runtime.py:106); `device_print`'s line format `pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>` (lib/Conversion/TritonGPUToLLVM/PrintOpToLLVM.cpp:98); `device_assert` emits nothing unless `options.debug` (python/triton/language/semantic.py:1811), and a Python `assert` in a jit body becomes `device_assert` (python/triton/compiler/code_generator.py:1353-1356)
- S56 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:516 a block stops at its own `return`; a constexpr `if` traces only its taken block (:995-996), and a `return` inside it moves tracing into a dead block, so the statements after the `if` are traced (:551-554); the byte-packing helper of DS256 at 2 tokens has `return packed` before its sign-OR lines (§3); a Qwen3-Next GDN kernel keeps its rank-2 and rank-3 delta rules apart because "the unified version traced both ranks and failed on the rank it was not given"
- S58 triton-lang/triton@c01b677:python/triton/language/semantic.py:1418 the dot asserts `input_precision` is in the backend's list, `('ieee', 'bf16x3', 'bf16x6')` (third_party/amd/backend/compiler.py:93), which `TRITON_F32_DEFAULT` sets for Gluon's `mfma` (python/triton/knobs.py:496); "scale factor must be 16 or 32. Got …" at lib/Dialect/Triton/IR/Ops.cpp:391; `report_fatal_error("Dimensions must match, ignoring order, but they don't…")` at lib/Tools/LinearLayout.cpp:91; the `extract_slice` verifier's "Lane and warp dim basis must match…" at third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp:222; `adjustAccForSmallKDim` multiplies each accumulator element by 1 / duplication (third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp:191, :208-213), and the scaled-MFMA lowering asserts linear-layout scales (:898-901)
- S60 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1119 a live-in name keeps its Python type through a loop body, else "Loop carried variable {name} changed type, was {type(loop_val)} but is now {type(live_val)}", `loop_val` the body's value and `live_val` the one before the loop; `_find_carries` checks every live-in that is a Triton value (:482-488), constexprs included (`class constexpr(base_value)`, python/triton/language/core.py:210), for `range` (:1284) and `while` (:1131); `static_range` binds its variable as a `constexpr` and leaves it bound (:1212); a plain assignment turns an int into a tensor (:733-737), a `gl.constexpr`-annotated one keeps a constexpr (:747-749)
- S62 hardware/gfx950/cdna4/ch03_kernel_state.md:195 CDNA4 ISA §3.6.5: "LDS space is allocated to a work-group or wavefront in contiguous blocks of 1280 bytes on 1280-byte alignment"
- S63 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:199 a broadcast operand in `AutoLayout` takes the other operand's layout (`set_auto_layout`, :201–204), only two different concrete layouts raising (:205–206); `gl.arange` and `gl.full`, and so `gl.zeros`, default to `AutoLayout` (:214–215, :233–234; python/triton/experimental/gluon/language/_standard.py:29)
