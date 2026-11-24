---
title: "Authoring a Triton or Gluon kernel"
kind: method
ops: [gemm, moe, attention, gdn]
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
---

# Authoring a Triton or Gluon kernel

How to write a Triton or Gluon kernel that replaces one op of an inference framework at a fixed list of sizes, and
check it at each of them, on gfx950 (MI355X) with
Triton 3.8.0: the language, the files and entry, `run`, one binary per size, scratch, launches, graph capture,
fallbacks, a Gluon skeleton, validation. Following it does not guarantee that the kernel passes its checks.

The setting the page assumes: a host function, `run` in the examples, receives the op's tensors and scalars with every
output preallocated and launches the kernel or kernels; a "size" is one value of each variable dimension of the op (a
token count), and the kernel is written, checked and timed for a list of them; its caller sends it only calls of those
sizes, takes its own path for the rest, and may capture the call into a CUDA graph and replay it.

## Route here when

- You write a new Triton or Gluon kernel for an op, or port one.
- You choose between plain `tl.*` and Gluon, or are about to add a launch.
- A check of the kernel fails: it does not build at its first call, compiles or synchronizes in a capture, writes an
  input, or leaves an output unwritten (§11).
- A kernel works eagerly and fails under graph capture or replay.

Elsewhere: layouts [gluon/layouts.md](gluon/layouts.md), data movement [gluon/memory.md](gluon/memory.md), the matrix
core [gluon/mfma.md](gluon/mfma.md), lane-level code [gluon/wave_ops.md](gluon/wave_ops.md), a working kernel's speed
[tuning.md](tuning.md), its IR, ISA and errors [debugging.md](debugging.md), accuracy [numerics.md](numerics.md), the
kernels by family [kernels/kernel_library.md](kernels/kernel_library.md).

## 1. The procedure

Start from the closest existing kernel and the branch its size takes (§10); choose plain `tl.*` or Gluon (§2); write
the kernel's files (§3) and a `run` that writes its outputs, keeps no state and
does not synchronize (§4); make fixed dimensions and compile-time configuration choices constexprs, and keep
ranged scalars and dimensions the caller may vary within a size runtime unless every variant they select is warmed (§5); allocate scratch in the call
or cache per-shape scratch in a compile function, and initialize every region before its first read in each call (§6);
consider a launch where a phase needs results from other workgroups or a separate initialization or conversion, measure
whether fusing is faster, and keep every call capturable (§7, §8); compile for gfx950 without a GPU (§10); check and
time the kernel on one GPU at every size, and fix or drop every
size that fails (§11). Each step is checked in its section; the procedure as a whole was not run end to end.

## 2. Plain Triton (`tl.*`) or Gluon

The page draws on two censuses of "the library", the set of Triton-language kernels for gfx950 that
[kernels/kernel_library.md](kernels/kernel_library.md) §1 describes: its 154 kernels, 4 plain `@triton.jit` and 150 `@gluon.jit`, and the 280,
4 plain and 276 Gluon, it held after its growth. The table lists the families of those 154 kernels; the families added since (the DSA indexer, Qwen3-Next
gated attention, more DSA, MoE and GDN kernels) are in [kernels/kernel_library.md](kernels/kernel_library.md).

| Language | Families of the library's 154 kernels (kernel count) |
|---|---|
| plain `tl.*` (4) | bf16 GEMMs at M ≤ 64, N 6144 × K 2048 and N 2624 × K 6144 (2), RMSNorm and add-RMSNorm at hidden 6144 (2) |
| Gluon (150) | fp8 block-scaled GEMMs (24); whole MoE layers: DeepSeek-V2/GLM MXFP4 (19), MiniMax-M2 FP8 (9) and MXFP4 (23), Qwen2-MoE router and experts (14); one-query-token attention: DSA sparse MLA (8), Qwen3.5 gated attention (9); GatedDeltaNet one token per step (18) and chunked (26) |

The two share the launcher and the JIT cache (`@gluon.jit` is a `JITFunction` subclass, S1) and differ in the first
compiler stage: plain Triton runs `make_ttir` then `make_ttgir`, Gluon replaces both with `gluon_to_ttgir` (S2).

| | plain `tl.*` | Gluon |
|---|---|---|
| Layouts | assigned by the compiler: coalescing, the MFMA (`matrix_instr_nonkdim`; `kpack` is forced to 1 on gfx950, S4), conversion removal | written by you: `BlockedLayout`, `SliceLayout`, `DistributedLinearLayout`, `AMDMFMALayout`, `DotOperandLayout`, shared layouts; a `gl.arange`, `gl.zeros` or `gl.full` whose layout no other tensor fixes fails to compile ("Failed to infer return type") |
| Software pipelining | `num_stages`: loop scheduling, operands streamed into LDS by async copies. In the N 2624 × K 6144 bf16 GEMM's 16-row path `num_stages=1` stages the tiles through registers into 16 KiB of LDS; the kernel's own 3 uses 101312 B (about 99 KiB) fed by direct-to-LDS `buffer_load_dwordx4 … lds` (96 in the ISA at m = 2 and 8 with 4 warps, 48 at m = 1 and 16 with 2; the other m were not compiled); `num_stages=1` took 2.2–2.3× the time of 3 (18.33–18.61 against 7.97–8.36 µs at m = 2, 4, 8), while m = 64, whose launch did not change, stayed at 0.995–1.003× in speedup (3 runs; the LDS and ISA counts are from 1 compile) *(field, E1, E2)* | prefetch by hand ([gluon/memory.md](gluon/memory.md) §12): Gluon's AMD stages do not read `num_stages`, `matrix_instr_nonkdim`, `kpack` or `schedule_hint` (S2), and option parsing resets gfx950's `kpack=2` to 1 (S4); 36 kernel configurations from 8 Gluon families, each built with 9 such option variants, kept identical instructions across the variants (two runs), and a launched copy kernel got 9 cache entries with the same code *(field, E3)* |
| Buffer instructions | `tl.load` / `tl.store` become `buffer_load` / `buffer_store` where a pass proves the range, from the `S` key (§5) or range analysis (S2, S3) | only where you call `gl.amd.cdna4.buffer_load` / `buffer_store`; `gl.load` stays `global_load` ([gluon/memory.md](gluon/memory.md) §2, §3). In a Qwen3.5 GatedDeltaNet one-token-per-step kernel at 2 tokens and 4 slots, the recurrence with buffer `.cs` and with pointer `.cs` accesses both met the op's accuracy bounds in 36 calls (3 seeds × 3 slot patterns × 2 output poisons; 18 repeat pairs bitwise equal) *(field, E4)*. The ruler ([tuning.md](tuning.md) §1) read 8.79 µs for the buffer arm and 8.81 µs for the pointer arm, the mean of each run's median call time over 4 runs per arm *(field, E1)*; by device duration on a cold device, the two were within ±1% of each other for 4 of 9 seed and slot inputs, pointer slower by 1.2–2.8% for 3, and 2 straddled ±1% *(field, E5)*: no general buffer-or-pointer speed rule follows |
| LDS | implicit | `gl.allocate_shared_memory`, `gl.amd.cdna4.async_copy.buffer_load_to_shared`, `commit_group` / `wait_group` (with two groups of two copies in flight, `wait_group(1)` becomes `vmcnt(2)`), swizzled and padded layouts |
| Matrix core | `tl.dot`, `tl.dot_scaled` | `gl.amd.cdna4.mfma` (`v_mfma_f32_16x16x32_bf16` in §10's skeleton), `mfma_scaled` with e8m0 scales (a 16 × 16 × 128 e4m3 tile: one `v_mfma_scale_f32_16x16x128_f8f6f4` with the scales in registers; `None` passes a register set to 0x7f, scale 1); `gl.dot_fma` is a vector-ALU dot product off the matrix core, on a `BlockedLayout` accumulator ([gluon/mfma.md](gluon/mfma.md) §1) |
| Lane-level code | `inline_asm_elementwise`, not knowing which lane holds which element | the same call, each element's lane fixed by the layout (`v_readlane_b32` of lane 5 reads x[5] with one element per lane, x[10] and x[11] with two); shuffles and ballots: [gluon/wave_ops.md](gluon/wave_ops.md) |

- **`num_warps` is part of a Gluon kernel.** The module takes `ttg.num-warps` from the launch, and the verifier rejects
  a layout with another warp count ("Layout has 4 warps per CTA, but the context requires 8 warps per CTA") or with
  other than 64 lanes (S1). A launch without `num_warps` gets 4 (S4), which hard-coded 4-warp layouts rely on (the state-clearing
  launch of §6).

The plain bf16 GEMMs and norms ([gemm/small_m_gemm.md](gemm/small_m_gemm.md)) are starting points for plain
`tl.*`, which leaves layouts and LDS buffers to the compiler ([gemm.md](gemm.md) §9), not a rule for choosing the
language. Their `num_stages` matters: fewer stages slowed the N 6144 × K 2048 GEMM at m = 2, 4, 8, 16 by 1.533–1.625× and RMSNorm
at 1024, 2048, 4096 tokens by 1.047–1.120×, 1.092–1.095× and 1.051–1.052× (one run, 4 rounds per arm in a fixed order,
which may bias the smaller effects; add-RMSNorm not tested) *(field, E6)*. Gluon exposes e8m0 `mfma_scaled`, explicit
LDS and async-copy depth and lane-level code (table). Gluon fp8 GEMMs at N 3072, K 1536 (unshuffled E4M3 with
fp32 block scales) took 0.777–0.781×, 0.755–0.764× and 0.759–0.768× the flushed graph time of aiter's plain-Triton
kernel at m = 2, 8, 16, both within the op's accuracy bounds (one run, 5 rounds) *(field, E6)*: a comparison of two
implementations, not of the languages, and not of preshuffled MXFP4.

## 3. The kernel's files

- **Sizes.** A kernel lists the sizes it is written for: one value of each variable dimension, and the scalars. 116 of the library's 154
  kernels list one size (224 of its later 280). Every one targets gfx950 on MI355X.
- **Source files.** A kernel is a small Python package. Of the library's 154 kernels, 86 are one `kernel.py` and 68 add 1–4 helpers (`attention.py`,
  `prologue.py`, `initial_state.py`, ...) imported relatively (`from .adapter import validate_inputs`). No kernel
  imports another's files: each copies its
  helpers.
- **Entry.** `run` in `kernel.py` (139 of the library's 154 kernels), or in a `main.py` with an `adapter.py` that checks the
  inputs (the 15 N = 3072 fp8 GEMMs).
- **Triton version.** State the Triton a kernel needs beside it, and do not build it on an older one. Triton ≥ 3.8 is
  required by 135 of the library's 154 kernels (261 of its later 280). Triton 3.7.0 lacks `gl.clamp`,
  `gl.aggregate_replace`, `gl.expect_zero`, `gl.amd.slice`, `llvm_fn_attrs` and `gl.amd.cdna4`'s
  `compute_efficient_padded_shared_layout` and `scaled_upcast` (S8); under it
  none of the 11 N 3072 fp8 GEMM kernels from m = 16 up compiles, m = 32 and 64
  without calling those APIs *(field, E20)*. Require 3.8 for every Gluon kernel you have not run with 3.7.0; the 19 kernels without the requirement (the
  15 N 3072 fp8 GEMMs, the bf16 GEMMs, the norms; 19 of its later 280 too) include 12 N 3072 kernels that call 3.8-only API:
  add it when you start from one.

**One op or a whole function.** A kernel may implement one op (for example GEMM or RMSNorm) against the library
implementation it replaces (aiter and SGLang implementations among them), or replace a whole SGLang function (an MoE
layer's forward, a GatedDeltaNet layer, an attention step for one query token per sequence), against SGLang's own path;
the second may take framework state as mutable inputs.

**Outputs in a packed format.** A packed low-precision output is compared as integers and must equal the reference ([numerics.md](numerics.md) §1): MXFP4 codes leave a kernel packed in `uint8`, and the kernels that quantize to MXFP8 write row-major e4m3 codes and uint8 e8m0 bytes; the fp8 group quantization stores its scales group-major, as (groups, tokens). In the DS512 kernel for 1–2 tokens (DeepSeek-V2 MXFP4 MoE on a rank of intermediate dimension 512; MoE layer labels: [moe.md](moe.md) §1), the one-token launch that finishes gate and up writes packed uint8 activations and e8m0 scales into call-local scratch, which the down-projection launch reads; its activations are row-major and its expert weights use shuffled offsets. Read any other MoE's scratch format from its own producer and consumer code.

## 4. `run`: what it receives, what it must not do

`run` takes the op's arguments keyword-only: its tensors and scalars (an absent optional input is `None`) and every output, preallocated; it returns nothing and keeps no output. Logical sizes follow from the tensors' stored shapes under the op's layout and packing: a packed 4-bit last dimension holds two logical elements per stored element. It must write every output: a check that poisons the outputs before the call caught a GEMM copy that left rows unwritten (§8), which does not mean every omitted write is caught. It writes no input the op declares read-only, and is right for every value of a scalar's range. MXFP4 and e8m0 tensors may arrive as `uint8` or as `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`, which Triton 3.8.0 cannot type (`KeyError: 'float4_e2m1fn_x2'` at the launch, S20): pass `.view(torch.uint8)`, as that DS512 kernel's entry does.

A kernel may assume only what its caller guarantees. Where the caller sends only tensors in the stored layout, not negated, aligned (`data_ptr() % 16 == 0`), on one device, with no output byte overlapping an input, strides can be constexprs folded from the shape; where it sends one value of a scalar, the scalar can be hard-coded, as that DS512 kernel's entry hard-codes its `routed_scaling_factor`, 2.5. Do not hard-code a dimension the caller may vary within a size (a GDN state pool's slot count). Pass a ranged scalar at run time: an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, yet called with `epsilon` 1e-3 on inputs of scale 0.01 was off by up to 224%, the original kernel exact *(field, E8)*.

The caller may capture the call into a CUDA graph and replay it (§8), so launch code, `run` and the host functions it
calls, must not synchronize, read device data back to the host or time itself, and keeps no state. For a Triton or Gluon kernel:

- **No `@triton.autotune` or `@triton.heuristics` above `@triton.jit`**: choose the configuration in a compile
  function instead (§5). None of the library's 154 kernels uses either (none of its later 280).
  An autotuner times its unpruned configurations at a key's first in-process miss, unless one configuration remains or the disk cache holds the choice (S17), so its pick among equal-cost configurations is a timing artifact: with two equal-cost configurations, each fresh process timed both at its first call and none after; run one at a time, 16 of 16 processes picked the configuration timed second, while 16 processes run four at a time on one GPU split 10 and 6, so processes can run one size with different binaries *(field, E9, E23)*.
- **Find a write to an input by comparing its bytes, not by reading the source.** A source scan that looks for
  in-place writes to an input by name found `x.add_(1)` in `run` and `weight.add_(1)` in a helper whose parameter carries the input's name; the same
  write through a helper parameter renamed `buf` passed the scan, and was caught by comparing the read-only inputs
  byte for byte around the first captured call *(field, E10)*.
- **A `@triton.jit` / `@gluon.jit` body does not read a plain module-level Python constant under the default compiler
  policy**: pass it as a `tl.constexpr` / `gl.constexpr` argument or compute it in a
  `@gluon.constexpr_function` (`from triton.experimental import gluon`; 3.8.0's `gl` has none). Triton also accepts
  globals made with `gl.constexpr(...)` (five of the library's 154 kernels keep such globals, layouts in three; 18 of
  its later 280), modules, JIT functions and dtypes; a plain `BLOCK = 64` fails the default compile (S16). Reassigning a
  `gl.constexpr` global the kernel reads (4 → 5), or a plain global it was allowed to read (3 → 4), after the first
  compile raised `RuntimeError` at the next launch; reassigning the same value, or changing a global the kernel does not
  read, did not (two runs; mutating an object in place was not tested) *(field, E3)*.

## 5. One binary per size

**Pass the token count as a constexpr.** 122 of the 150 Gluon kernels among the library's 154 (210 of 276 in its later 280) have a `@gluon.jit` parameter `M`, `ROWS`, `TOKENS` or `BATCH` declared `gl.constexpr` (counted by name, Qwen3-Next's runtime-count kernels included). Each size then compiles its own binary at its first call, though different sizes may share a configuration. Compiled for gfx950, 8-, 16-, 32- and 64-row tiles whose constexpr M covered the tile had no mask instructions (3–56 with a runtime M), while masks remained where M did not fill the tile; over M = 1–128, a `range(M)` loop compiled without a branch at M = 1–22 and with 1–2 at 23–128, using 17–74 VGPRs against 66 with a runtime M *(field, E11)*: a constexpr removes some masks, loops and flags, not all. Make that first call outside any CUDA-graph capture, with every variant the size's scalars select: a variant first met inside a capture compiles there, and the checks these kernels were run under fail a kernel that compiles in a capture (§8). A `gl.static_range` over a constexpr count unrolls fully (a 4-load body: 120 amdgcn instructions at 8 trips, 6513 at 512, 120–129 with `range` *(field, E21)*): 13 of the 15 Qwen3.5 GDN prefill kernels walk their sequences that way (`for candidate in gl.static_range(1, BATCH)`), fine at their sizes' 1–8 sequences; for the 2047 the op allows, use `range`.

**Choose the configuration once, where host and kernel both read it.** A `@gluon.constexpr_function` called outside
JIT returns its Python value and inside JIT a constexpr (S17), so one function gives the launch its grid and the kernel
its tile shapes: a Qwen2-MoE router's kernel reads `config: gl.constexpr = _shape_config(M)` and its host sizes the
grid from `_shape_config(rows)`, as the skeleton of §10 does with `_config`.

**A `@functools.cache` compile function** is a
module-level def of shapes whose cache holds compiled code, constants and per-shape scratch only. Eleven of the library's 154 kernels
have one (42 of its later 280, with the 31 Qwen3-Next gated-attention kernels): the two plain-Triton GEMMs cache a
launch closure per shape, the N 2624 × K 6144 one with its workspace size; the nine Qwen3.5 gated-attention kernels
the fp32 unit KV scale both launches read, which built in `run` would allocate and fill a tensor per call. The N 6144 × K 2048 bf16 GEMM, its launch closure built once per (a_shape, b_shape):

```python
@functools.cache
def _launch_for(a_shape, b_shape):
    m, k = a_shape
    n = b_shape[0]
    grid = (n // 32,)
    if m <= 16:
        even_m = m == 16
        block_k = 512 if m == 1 else 256

        def launch(a, b, out):
            _gemm_m16_kernel[grid](
                a,
                b,
                out,
                m,
                N=n,
                K=k,
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=block_k,
                N_TILES=n // 32,
                NXCD=8,
                EVEN_M=even_m,
                num_warps=2,
                num_stages=3,
                waves_per_eu=2,
            )

        return launch
    # ... the branches for m <= 32 and m <= 64 return their own closures


def run(*, a, b, trans_b, bias, out_dtype, out):
    launch = _launch_for(tuple(a.shape), tuple(b.shape))
    launch(a, b, out)
```

A Qwen3.5 gated-attention kernel, the constant tensor both launches read, built once per device:

```python
@functools.cache
def _unit_scale(device):
    """The FP32 KV scale both kernels read: the pools hold unscaled values."""
    return torch.ones(1, dtype=torch.float32, device=device)
```

None of the library's 154 kernels (nor of its later 280) calls `warmup()` or `triton.compile`: binaries live in Triton's in-process JIT cache, which is enough
while every size is first called outside a capture and the key below is a function of the size.

**What enters Triton 3.8.0's JIT key** (S17): the launcher looks `(tuple(specialization), str(options))` up in
`kernel_key_cache`, then the binary under the string key it maps to, in a cache per device; 12 specialization and option
combinations gave 12 binaries in each frontend (two runs on one GPU) *(field, E3)*.

| Argument | Enters the key as | So |
|---|---|---|
| `tl.constexpr` / `gl.constexpr` | its value, which must be hashable (S17) | one binary per value; a list fails at the launch (§10) |
| int | `i32` / `i64` / `u64`; the value 1 becomes a constexpr; a multiple of 16 gets `D` (S20) | sizes share a binary only within one class; `do_not_specialize=[...]` on `@gluon.jit` drops the value keys |
| runtime float, bool | `fp32`, `u1` (S20) | 9 floats (0, 1, −1, 16 and 1e30 among them) with True and False used one binary per frontend, and 15 more floats, inf and NaN included, added none *(field, E3)* |
| `None` | a constexpr (S20) | an absent optional input compiles its own variant |
| tensor (HIP) | its dtype, `D` when `data_ptr() % 16 == 0` and `S` when AMD buffer passes are enabled and its whole storage is at most 2^31 − 1 bytes (S20, S21), from a Python callback run per tensor at every launch, cache hits included; on the argument `D` becomes `tt.divisibility = 16` and `S` `tt.pointer_range = 32`: offset 0 of a 4 KiB buffer got both, offset 8 only `S`, a 2 GiB + 16 KiB tensor only `D`, 3 binaries *(field, E8)*; `S` changed the code of 3 plain-Triton kernels, only with AMD buffer passes on, and of 3 Gluon copy and gather kernels neither way (two compile runs) *(field, E3)* | a misaligned view, or a storage that crosses 2 GiB, compiles another binary |
| launch options | `str(options)`: `num_warps`, `waves_per_eu`, ... as passed, in keyword order (S17) | launch from one call site; an option Gluon ignores (`num_stages`) still splits the key |

**Keep runtime** what may change while the size stays the same, since a key that changes after the size's first call
compiles again, possibly inside a capture (§8): runtime floats, which enter the key only as `fp32` *(field, E3)*, as
a Qwen3.5 GDN decode kernel keeps its scale and epsilon ("Scale and epsilon are runtime arguments, never specialization
keys", its source says); scalar inputs, unless one value per process is guaranteed (the
MM64-FP8 and MM64-FP4 kernels, MiniMax-M2 on a rank holding 64 of its experts, take `expert_offset` as `OFFSET: gl.constexpr`, one value per process);
dimensions the caller may vary within a size (a pool's slot count); and the alignment of pointers into a buffer you
carve (§6). A runtime int is still keyed on its class (1, a multiple of 16, i32, i64), and a warm-up call compiles only
for the values it passes: put an int that may change within a size in `do_not_specialize` (the dtype classes
`i32`, `i64` and, for large non-negative values, `u64` remain).

## 6. Scratch

- **Allocate inside the call**, on an argument's device, sized from the shapes: `torch.empty` (117 of the library's 154 kernels,
  686 call sites), `torch.empty_like`, `.new_empty`. One allocation can be carved into typed regions at aligned
  offsets (64 KiB or 256 bytes in the MoE kernels' arenas); an offset that is not a multiple of 16 bytes loses `D` and gets
  another binary (§5):

```python
# a DeepSeek-V2 MXFP4 MoE: one uint8 arena, each typed region starting on a 64 KiB boundary
regions = []
total_bytes = 0
for shape, dtype in specifications:  # ((jobs_count,), torch.int32), ((m, 8), torch.float32), ...
    region_bytes = dtype.itemsize * prod(shape)
    regions.append((total_bytes, region_bytes, shape, dtype))
    total_bytes += triton.cdiv(region_bytes, 65536) * 65536
arena = torch.empty((total_bytes,), dtype=torch.uint8, device=x.device)
return [arena.narrow(0, offset, size).view(dtype).view(shape) for offset, size, shape, dtype in regions]
```

- **`torch.empty` holds whatever was there**, in an eager call and in every replay. In every call, write before any read
  what a launch reads before it writes: atomic counters and cursors, `-1` job sentinels, padding rows, accumulators
  added into. None of the library's 154 kernels calls `torch.zeros` or `fill_` (nor any of its later 280): an earlier
  launch of the call writes the initial values, as program 0 of an MM256-FP4 kernel's first launch stores 0 into the
  expert counts and the total and -1 into the job slots, on which the next launch takes atomic tickets:

```python
pid = gl.program_id(0).to(gl.uint32)
linear_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
if pid == 0:  # one program of the first launch writes what the next launch's atomics start from
    expert = gl.arange(0, _next_power_of_2(max(E, MAX_JOBS)), layout=linear_layout)
    gl.store(Counts + expert, 0, expert < E)
    gl.store(Jobs + expert, -1, expert < MAX_JOBS)
    gl.store(Total, 0)
# ... every program, 0 included, then goes on to its router or quantization tile
```

  Not this: counters that no launch of the call writes before they are read. A fold whose counters no node of the
  graph zeroed was wrong in every replay after the first (§7).
- **State that must start fresh** is cleared by a small launch reading its flag on the device, as a GDN prefill kernel
  zeroes the slot of each sequence without initial state:

```python
@gluon.jit
def _initialize_fresh(
    STATE, INDICES, INITIAL, WIDTH: gl.constexpr, BLOCK: gl.constexpr
):
    seq = gl.program_id(0)
    if not gl.load(INITIAL + seq):  # the flag is read on the device, not by the host
        slot = gl.load(INDICES + seq)
        offset = gl.program_id(1) * BLOCK + gl.arange(
            0, BLOCK, gl.BlockedLayout([1], [64], [4], [0])
        )
        gl.store(STATE + slot * WIDTH + offset, 0.0, offset < WIDTH)


def initialize_fresh(state, indices, initial):
    width = state.shape[1] * state.shape[2] * state.shape[3]
    _initialize_fresh[(indices.numel(), triton.cdiv(width, 1024))](
        state, indices, initial, width, 1024
    )
```

- **Nothing carries from one call to the next.** `run` keeps nothing derived from an argument across calls, a compile
  function only compiled code, constants (§5) and per-shape scratch rewritten whole before every read. A source scan
  that rejects a `global` statement does not see a module-level dict written in `run` *(field, E10)*: a
  `run` that kept its first output in one and copied it into later outputs passed the scan and was caught by comparing
  outputs with the reference over captures with new inputs, the one after the timing included *(field, E8)*
  ([../pitfalls/state_kept_across_calls.md](../pitfalls/state_kept_across_calls.md)).
- **Reuse a region within a call in stream order**, as MoE kernels overlay short-lived scratch on a later buffer:
  launched on the same stream before the region's next writer, its last reader saw the values from before the
  overwrite in 20 of 20 eager calls and 20 of 20 graph replays over 16.8 M floats *(field, E2)*.

## 7. Launches: what one costs, where a boundary goes

What one more launch costs depends on who times it:

| Where | One more launch costs | Timed by |
|---|---|---|
| A replayed CUDA graph, as SGLang replays its fixed-shape one-token-per-sequence (decode) steps | 1.82 µs (1 program) to 1.84 µs (256 programs) of replay time per further chained trivial Gluon kernel (one 64-float load, add and store per program), in a bare CUDA graph, not SGLang's; the replay itself added 3.6–3.8 µs; with 8 launches per graph, 2.27 µs per launch with 1 tensor argument and 2.31 µs with 30 *(field, E13)* | CUDA events around graph replays, least squares over 1–64 kernels |
| The ruler's timer ([tuning.md](tuning.md) §1): each call captured into a CUDA graph behind a cache flush, the graph replayed under a profiler | the sum of the call's per-kernel device durations, flush excluded, read from Kineto or from a rocprofiler-sdk collector. Measured with Kineto: a copy and a sum read 10.84 µs in one call against 6.50 + 4.62 µs apart, a 6.5 µs call stayed 6.5 µs beside a 169.5 µs flush, 8 trivial kernels read 32.4 µs against 8 × 3.88 µs, and no kernel read below about 4.2 µs *(field, E13)* (its floor: [../pitfalls/profiled_timer_floor.md](../pitfalls/profiled_timer_floor.md)) | kernel durations from the profiler's trace of the replays |
| An eager call | 7.30 µs (1 program) to 7.44 µs (256 programs) of host time per `kernel[grid](*args)` of a one-tensor trivial Gluon kernel in Triton 3.8.0's launcher, the wall time within 0.2% of it, the GPU waiting on the host; 6.25 µs + 1.00 µs per tensor + 0.12 µs per int argument by least squares over 7 argument lists (7.32 µs with 1 tensor, 38.45 µs with 32, 17.12 µs with 8 tensors and 24 ints); HIP's Python callback for the `S` key (`get_tensor_specialization`, S20, S21) runs once per tensor per launch and took 0.58 µs per call timed alone, 57% of that slope ([debugging.md](debugging.md) §8); 8 launches of a 15-tensor kernel took 169 µs *(field, E13)* | host clock around 2000 launches |

Host time lengthens the call only while the GPU finishes kernels faster than Python issues them: those 8 launches took
172 µs with trivial kernels, their host time, and 3166 µs with 396 µs kernels, their sum *(field, E13)*. Only eager calls
pay it — calls the framework does not capture, such as many-token (prefill) batches whose launches depend on the
call's lengths, and calls made outside a graph — so there keep tensor arguments few; a 4-key option dict built per call cost
0.06 µs more per launch than one built once (a Qwen3-Next decode kernel builds its at import) *(field, E13)*.

**Where the kernels end a launch.** A Kineto trace of the calls at each of the 242 sizes of the library's 154 kernels counted
launches per call: 1 in the fp8 GEMMs and the norms; 2 in the N 2624 × K 6144 bf16 GEMM at m = 64, a split-K kernel then
a fixed-order reduce kernel ([gemm.md](gemm.md) §9); 2–8 in the MoE layers, 2–3 in DSA, 3 in gated attention, 1–2 in
GDN decode, 5–8 in GDN prefill *(field, E14)*. The stages they separate: [moe.md](moe.md) §2,
[attention.md](attention.md) §2, [gdn.md](gdn.md) §3 and §7.

**Consider a launch where a value crosses workgroups or a phase needs a separate initialization or conversion.** `gl.barrier()` synchronizes one workgroup, and gfx950 has no CTA clusters to widen it (`num_ctas > 1 not supported on gfx950`, S4); no result may depend on a workgroup that may not be running, since a launch's workgroups need not all be resident: no unbounded spin on a partner, and no giving up and computing without its data. So in the library's 154 kernels a value one workgroup produces and another consumes — a token's expert logits before its top-k, every split before a merge, every reader of a state before it is overwritten — ends the launch, and stream order is the synchronization. None of them tests an atomic's result for the last arrival or waits in a loop on memory, so none uses an in-launch fold (each split counts in on an atomic counter, the last to arrive merges in a fixed order: [../gemm/wait_free_fold.md](../gemm/wait_free_fold.md), measured there on FlyDSL kernels) or a bounded wait that hands a timed-out share to the last arrival; all 42 of their global atomic call sites are relaxed (all 87 in its later 280; [numerics.md](numerics.md) §8). A Gluon fold needs its counters at zero before each replay and an `acq_rel` ticket: without a reset, 1999 of 1999 later replays were wrong at each of 9 sizes; with the counters zeroed by an earlier node of the graph, 2000 of 2000 `acq_rel` replays were bitwise equal (one run; 200 of 200 at 16 splits × 16384 outputs in three earlier runs); a counter the last arrival reset itself gave 1000 of 1000 correct replays in each of 3 runs *(field, E2, E15)*. With a `relaxed` ticket the last arrival read stale partials, 3000 of 3000 replays wrong, while `acq_rel` (a `buffer_wbl2` and a `buffer_inv` in its ISA) and `relaxed` with write-through (`.wt`) partial stores were right in 3000 of 3000 *(field, E22)*. A separate merge launch took 0.03–0.65× the replay time of the `acq_rel` fold at 4, 16, 64 splits × 4096, 16384, 65536 fp32 outputs (64 outputs per one-wave program), and 0.12–0.82× on the ruler's timer at up to 16 splits and 16384 outputs; each fold's time includes its zeroing node *(field, E2, E15)*. Phases that are independent, or cheap to recompute, can share one launch:

- independent phases can share a grid that splits roles by program id (a DeepSeek router beside input MXFP4 quantization;
  [moe.md](moe.md) §3). Two toy elementwise phases of 131072 elements, outputs equal, took 0.762–0.854× the replay
  time of two launches in three of four runs and 1.085× (9.191 against 8.471 µs) in the fourth, and 0.506–0.516× on
  the ruler's timer (4.30–4.44 against 8.49–8.61 µs) in all four *(field, E8)*;
- a small decision may be recomputed instead of published ([moe.md](moe.md) §4): a toy Triton MoE at 2 tokens,
  top-4 of 32 experts recomputed in its 40 consumer programs, took 0.843–0.889× the replay time and 0.742–0.772× the
  time on the ruler's timer of a separate selection launch, same output (4 runs) *(field, E2)*. In two synthetic fp8 VALU MoEs (top-8
  of 256) at 1, 2, 8, 16 tokens, recomputing in every consumer was slower on both timers, the output bitwise equal (one
  run, 16 rotated rounds per size) *(field, E16)*;
- a split reduction whose result must be bitwise reproducible gets a fixed-order reduce launch, not float atomics (the
  N 2624 × K 6144 bf16 GEMM: [gemm/small_m_gemm.md](gemm/small_m_gemm.md) §6).

Compare publishing with recomputation by measuring it, not by a fixed launch cost: removing one toy phase launch saved
1.150–1.884 µs of replay time at 16 K, 128 K and 1 M elements in three runs, but −0.102, 1.587 and 0.325 µs in a fourth
(outputs equal), and 4.17–4.19 µs on the ruler's timer at 131072 elements *(field, E2)*. In a synthetic sweep of 56
configurations (7 decisions × 2 or 16 tokens × 40, 256, 1024 or 4096 consumer programs), recomputing a top-8-of-256
decision won at 1024 programs and publishing it won at 4096, with 4 or 16 split partials at 2 or 16 tokens; the median
cost of the extra launch (3.27 µs replayed, 4.47 µs on the ruler's timer), taken as a constant, predicted the faster arm in
only 49 and 50 of the 56 (one run, 12 rotated rounds per configuration; some identical-arm controls differed from zero,
so effects that small are not resolved) *(field, E16)*. A torch op in `run` is a launch too: the DSA kernels that cast `q`
with `q.to(torch.bfloat16)` there add a kernel per call.

## 8. CUDA-graph capture

SGLang captures its one-token-per-sequence (decode) batches, so a kernel called there runs inside its graphs; a call
whose launches depend on its lengths, as GDN prefill's do, must stay out of a capture: a replay would rerun the captured call's lengths.
Check a kernel under capture at every step: a first
capture replayed twice with new values copied into the same buffers, fresh captures per input set, a repeat where the
result must be bitwise reproducible, and a check after the timing, with the outputs filled with NaN (0xFF bytes for integer and sub-byte dtypes)
before the call. Leaving a float element unwritten then shows: in one run, a copy of an fp8 GEMM (N 3072, m = 16)
that left 8 of its 16 rows unwritten failed at the first capture with 24576 = 8 × 3072 non-finite values
*(field, E8)*.

| Breaks | Why | Do |
|---|---|---|
| A size's first call inside a capture | the first call compiles (§5), and the checks these kernels were run under fail a kernel that compiles in a capture; a caller that guards against it runs its own path there instead, and the graph keeps that path in every replay | call every size once eagerly first, as SGLang does |
| A JIT key that changes after the size's first call (§5) | Triton compiles at a call that may be captured | make every key a function of the size: a constexpr warmed at 16 and launched at 32 inside a capture compiled there (one `_do_compile` call), which a tripwire on `JITFunction._do_compile` during the capture reports *(field, E9)*. With that tripwire, every size of an RMSNorm copy and of a GEMM copy whose key changed per call failed as compiled in the capture *(field, E23)*. A warm `TRITON_CACHE_DIR` does not help: every in-process miss calls `_do_compile`, which reads the disk cache (S17), as each of two fresh processes sharing a warm cache did once *(field, E9)* |
| `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`; a torch op whose output size depends on data: `torch.nonzero`, `x[mask]`, `torch.argwhere`, `repeat_interleave` without `output_size` | each failed a capture with `capture_error_mode="global"`, and `repeat_interleave` with `output_size` captured *(field, E10)*: no synchronization is allowed while a stream captures, and the torch ops synchronize to size their output (S44) | a source scan for the first four does not find the four data-dependent-size ops, which only the capture catches *(field, E10)*: compute indices in a kernel into a worst-case buffer (below) |
| A host decision on device data: a grid or a branch from a count the GPU computed | it needs a synchronization, and a replay keeps the captured decision | size the grid from shapes for the worst case and exit early on a count read on the device (below) |
| State written once and assumed afterwards | a replay reruns only the kernels, never `run`'s Python | write counters, sentinels and padding in every call; a Qwen2-MoE kernel's source says it: "Padding is rewritten every invocation, including graph replay" |
| A Python value that changes per call | frozen into the graph at capture | read it on the GPU; a gated-attention kernel's source: "Live lengths, physical cache slots, and KV scales are always read on the GPU" |
| Allocation | a `torch.empty` tensor allocated in a capture kept its address over 3 replays and held the third replay's values *(field, E9)* | allocate per call as usual (§6) and do not depend on what the buffer held |

The worst-case grid with an early exit, as an MM256-FP4 kernel sizes its expert GEMM: the host launches
`capacity = cdiv(min(routes * block_m, routes + e * (block_m - 1)), block_m) * block_m` rows of tiles, and a program
returns at once when `tile * BM >= gl.load(Total)`, the row count the GPU computed:

```python
# host: the grid covers the most rows of tiles any routing can need, from shapes alone
capacity = (
    triton.cdiv(min(routes * block_m, routes + e * (block_m - 1)), block_m)
    * block_m
)
```

```python
# kernel: a program past the rows the GPU counted returns at once
linear = gl.program_id(0) + gl.program_id(1) * gl.num_programs(0)
tile, column_tile = _grouped_tile_2(linear.to(gl.uint32), column_tiles, M_GROUP)
if tile * BM >= gl.load(Total):
    return
```

Not this: `torch.nonzero` or `.item()` on the count to size the grid on the host, which fails the capture (table).

## 9. Fallbacks: when the call is not yours

- **The kernel does not fall back.** It is called only at the sizes it lists (and at every value of a dimension the
  caller may vary within a size); an exception from `run` is raised to the caller, not turned into a fallback. Host `assert`s or
  an adapter's `ValueError`s document assumptions: nothing falls back on an error.
- **The caller decides** when a call is not the kernel's — shapes, dtypes, layout, alignment or aliasing no size of
  the kernel describes — and takes its own
  path. To cover more or fewer sizes, change the list, not `run`. A size you add runs a branch nobody has measured, and
  generated kernels keep branches that no longer fit their op: a Qwen3.5 GDN prefill kernel takes its
  `m > 4096` path at none of its sizes, and that path static-asserts a row stride its output (2048) cannot
  meet. Build, check and time each added size on a GPU: only that shows whether it builds,
  runs correctly and is faster.
- **Decide before the inputs cost work.** Where preparing the kernel's inputs costs work (input projections), ask
  first whether the call is the kernel's, from the shapes, dtypes and alignment of stand-ins: the question launches no
  kernel and does not enter `run`.

## 10. Starting a new kernel

1. **Start from the closest family** ([kernels/kernel_library.md](kernels/kernel_library.md) §3) and copy its file
   layout: entry, helpers, the host function that plans the launches. Read only the branch your size takes, and keep
   that branch's constants together: changed alone, one of them broke a working kernel (same page, §1), key splits 8 →
   4 in a 24-token DSA kernel failing to build, the fold factor of the gate launch 4 → 1 in the 1-token MM256-FP4
   kernel crashing, 4 of 4 runs each
   *(field, E1)*.
2. **Or start from this skeleton**, a bf16 GEMM kernel (`out = a @ b.T` with `a` (m, k) and `b` (n, k) row-major, no
   bias) for a shape such as
   N 6144 × K 2048: 16-byte loads in a `BlockedLayout`, masked `buffer_load` / `buffer_store`, one MFMA with an
   `AMDMFMALayout` accumulator and `DotOperandLayout` operands, the token count as a constexpr, one configuration
   function for kernel and launch. It compiles with Triton 3.8.0 for gfx950 at M = 2, 16, 32 and 64 without spills.
   Run at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64, it was bitwise stable for one
   positive and one signed input per M over 20 eager calls, 20 graph replays and a second process, within 0–1 bf16 ulp
   on the positive input and at 0.960–0.984 of the bound 2^-8 |Σab| + 2^-18 Σ|ab| on the signed one. Checked and timed
   as the kernel for N 6144 × K 2048, it had no correctness, build or capture failure, but at all six sizes (m = 2, 4,
   8, 16, 32, 64) it ran at 0.373–0.434 of the speed of the baseline, aiter's bf16 GEMM (one run, 3 rounds) *(field, E6)*: it needs
   tuning before it is worth using at any size.

```python
"""bf16 GEMM out = a @ b.T (a is (m, k), b is (n, k), no bias) on gfx950 in Gluon: each program owns
BLOCK_N output columns for every row of a, walks all of K with fp32 MFMA accumulation and stores once."""

from typing import NamedTuple

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_Config = NamedTuple("_Config", [("block_m", int), ("block_n", int), ("block_k", int), ("warps", int)])

@gluon.constexpr_function
def _config(m):
    """The configuration of one size, read by run and by the kernel."""
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    return _Config(block_m, 64, 128, 4)

@gluon.jit
def _gemm(A, B, Out, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    cfg: gl.constexpr = _config(M)
    BM: gl.constexpr = cfg.block_m
    BN: gl.constexpr = cfg.block_n
    BK: gl.constexpr = cfg.block_k
    WARPS: gl.constexpr = cfg.warps
    gl.static_assert((M <= BM) & (N % BN == 0) & (K % BK == 0))
    # 8 bf16 (16 bytes) per lane along K, which is contiguous in both a (m, k) and b (n, k).
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [WARPS, 1], [1, 0])
    b_load: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, WARPS], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                              warps_per_cta=[1, WARPS])
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    col0 = gl.program_id(0) * BN
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, a_load))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_load))
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, b_load))
    cols = col0 + gl.arange(0, BN, layout=gl.SliceLayout(0, b_load))
    acc = gl.zeros([BM, BN], gl.float32, layout=mma)
    for k0 in range(0, K, BK):
        a = gl.amd.cdna4.buffer_load(A, rows[:, None] * K + (k0 + ka)[None, :], mask=(rows < M)[:, None])
        b = gl.amd.cdna4.buffer_load(B, cols[None, :] * K + (k0 + kb)[:, None], cache=".cg")  # read once
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, a_op), gl.convert_layout(b, b_op), acc)

    out_rows = gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
    out_cols = col0 + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(acc.to(gl.bfloat16), Out, out_rows[:, None] * N + out_cols[None, :],
                              mask=(out_rows < M)[:, None])

def run(*, a, b, trans_b, bias, out_dtype, out):
    m, k = a.shape
    n = b.shape[0]
    cfg = _config(m)
    _gemm[(n // cfg.block_n,)](a, b, out, M=m, N=n, K=k, num_warps=cfg.warps, waves_per_eu=2)
```

   `version=4` is gfx950's MFMA (S38; [gluon/mfma.md](gluon/mfma.md) §1). The masks keep rows past M out: Triton's buffer descriptor covers 2^31 − 2 bytes from the base, not the tensor (`num_records` `0x7ffffffe` in every buffer op of 27 probe kernels), so an unmasked `buffer_load` past a tensor's end read the memory behind it, while masked lanes read 0 without `other` *(field, E13)* ([gluon/memory.md](gluon/memory.md) §4). The two `convert_layout`s go through 8 KiB of LDS (loading in the operand layout instead: [gluon/layouts.md](gluon/layouts.md) §6, §9). The launch is a start, not a tuned choice: its 96 programs at N 6144 are fewer than the 256 CUs. At m = 2, 4, 8, 16 it took 2.882–3.158× the time of the plain-Triton N 6144 × K 2048 kernel, and prefetching one K step ahead made it 1.695–1.729× faster (one run, 3 rounds per arm in a fixed order; register allocation and scheduling changed too) *(field, E6)*. A Gluon stream of 96 programs × 4 warps rose from 876–886 to 3945–3995 GB/s with eight 16-byte loads per lane in flight instead of one (3 runs) *(field, E2)*, so the skeleton is probably short of bytes in flight: an inference, not a measured cause ([tuning.md](tuning.md) §3, §4). Grid and prefetch: tuning.md §6, §10; `num_warps`: tuning.md §7; `waves_per_eu=2` becomes `"amdgpu-waves-per-eu"="2,2"` and `.amdhsa_next_free_vgpr` 169 for a skeleton using 70 VGPRs; the compiler reports `Occupancy: 2`, a resource limit, not a measured residency (tuning.md §8). Small-M GEMM speed: [gemm/small_m_gemm.md](gemm/small_m_gemm.md).
3. **Compile for gfx950 without a GPU.** `triton.compile` with an `ASTSource` (`GluonASTSource` for Gluon) and `GPUTarget("hip", "gfx950", 64)` builds the launch's instructions, given its constexprs and specialization (`"DS"` for 1 MiB tensors at aligned addresses and `"D"` for n = 1024 matched; a pointer 4 bytes off alignment in a 3 GiB range with n = 1000 gets neither). Signatures name the dtype the launch sees: `*bf16`, `*fp16`, `*fp32`, `*fp8e4nv` (`float8_e4m3fn`), `*fp8e5` (`float8_e5m2`), `*u8` (also the uint8 views of MXFP4 and e8m0), `*i32`, `*i64`; an int is `i32` or `i64` by signed range, or `u64` above the signed-64 range (a constexpr when it is 1); a float is `fp32`, a bool `u1` (S20). It needs Triton and the kernel's imports (torch in 152 of the library's 154 kernels, 278 of its later 280), not a GPU. A kernel whose `kernel.py` imports a sibling relatively (a DSA decode kernel does) fails `from kernel import ...` ("attempted relative import with no known parent package"): write its sources into `kpkg/`, put its parent on `sys.path` and `from kpkg.kernel import ...`, so that the files are submodules of one package. Built with the launch's signature, attributes and options, offline compiles gave the launch's amdgcn in 12 of 12 comparisons (one run) *(field, E3)*. This compiles the skeleton with Triton 3.8.0 for gfx950:

```python
import triton
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from kernel import _gemm

def compile_gfx950(fn, signature, constexprs, attrs, options):
    sig = dict(signature, **{name: "constexpr" for name in constexprs})
    index = {name: (fn.arg_names.index(name),) for name in sig}
    source = GluonASTSource if fn.is_gluon() else ASTSource
    src = source(fn, sig, {index[n]: v for n, v in constexprs.items()},
                 {index[n]: HIPBackend.parse_attr(d) for n, d in attrs.items()})
    return triton.compile(src, target=GPUTarget("hip", "gfx950", 64), options=options)

k = compile_gfx950(_gemm, signature={"A": "*bf16", "B": "*bf16", "Out": "*bf16"},
                   constexprs={"M": 16, "N": 6144, "K": 2048}, attrs={"A": "DS", "B": "DS", "Out": "DS"},
                   options={"num_warps": 4, "waves_per_eu": 2})
print(k.metadata.shared, k.asm["amdgcn"].count("v_mfma_f32_16x16x32_bf16"))    # 8192 8
```

   LDS is `k.metadata.shared` (8192 here) while the amdgcn's `.amdhsa_group_segment_fixed_size` reads 0; launches of the
   skeleton passed the same 8192 B at all 11 M values tested *(field, E6)*. A Gluon kernel's `k.asm` holds `source`
   (the frontend's IR, before `gluon_to_ttgir`), `ttgir`, `llir`, `amdgcn`, `hsaco` ([debugging.md](debugging.md)
   §1–§2).
4. **Know what only a launch catches.**
   - A constexpr must be hashable: `INSTR=[16, 16, 32]` at launch failed with `unhashable type: 'list'` where
     `triton.compile` worked *(field, E8)*, since the launcher hashes each constexpr into its key (S17): pass a tuple.
   - A misspelled launch option raises `KeyError: Keyword argument ... was specified but unrecognised` when the launch
     compiles (S17); `triton.compile` silently drops options that are not `HIPOptions` fields (S4).
   - The launch derives `D` and `S` from the real tensors and ints (compile with the same attributes), and raises if a
     `gl.constexpr` global the kernel read has changed since compiling (§4, S17).
   - LDS over the 160 KiB (163840 bytes) a workgroup may take: `triton.compile` builds it (`metadata.shared` 262144 for
     gfx950) and the first launch raises `out of resource: shared memory` (S4; [debugging.md](debugging.md) §4).

## 11. Validating

Check and time the kernel on one GPU at each of its sizes: every output against the op's reference, every call
captured as in §8, the time against the op's baseline and against the kernel it would replace, if there is one. A size
counts on its own speedup, not on the mean over sizes: a plain-Triton
RMSNorm at hidden 6144 read 1.023–1.029 over its baseline at its 7 sizes of 3–18 tokens, under the bar of these runs ([tuning.md](tuning.md) §1),
and 1.04 or more at the others *(field, E17)*. Tune each size past the bar you hold it to.

**Failed sizes.** A failure at one size — wrong values, a build failure, a crash, a hang, or not faster — fails that
size and leaves the others. Read the result of every size, not only the overall verdict. Fix each failed size or drop
it from the kernel's list (§3), then check again.

What bites:

- **The image.** Time the kernel and its baseline in the image the kernel will run in, with that image's Triton (3.8.0 here) and the aiter the baseline is defined on. Installing another aiter over the image's can replace the installed Triton: install it with `AITER_USE_SYSTEM_TRITON=1` to keep an installed Triton, otherwise aiter's setup.py runs its Triton installer unless torch is below 2.9.1, and may skip an installer failure (S40). A baseline that reaches aiter can move with the aiter build: the N 6144 × K 2048 bf16 GEMM's baseline read 7.02 µs on the image's aiter and 7.95 µs on a pinned one at m = 8, 4.5% apart the other way at m = 16, the candidate within 1.2% at both *(field, E1)* ([tuning.md](tuning.md) §1). A local container may hold another Triton (3.7.0 has been seen): check `triton.__version__`. On 3.7.0 the launch's keyword check rejects `llvm_fn_attrs` (`KeyError`), and `schedule_hint="memory-bound-attention"` (the N 6144 × K 2048 bf16 GEMM passes it) adds `"amdgpu-sched-strategy"="iterative-ilp"`, which 3.8.0 ignores (S8, S2). Built offline for gfx950 with the same options (a bf16 `tl.dot` loop, 64 × 64 × 64 tile, 4 warps, 2 stages, that hint), one probe had 274 static instructions and 16384 B of LDS on 3.7.0 against 405 and 33968 B on 3.8.0 *(field, E18)*.

Correctness: compare every output with the op's reference at the steps of §8, bitwise equal between runs where the
op requires it (the plain-Triton GEMMs and norms, DSA, Qwen3.5, Qwen3-Next) or within each output's error bound (the
MoE layers and the fp8 block-scaled GEMMs) ([numerics.md](numerics.md) §1, §8). What a failure usually means here:

| Failure | Usually | § |
|---|---|---|
| It does not build or run at its first call | an import error, an entry that cannot take the op's arguments, a compile error at the first call | 3, 6, 10 |
| It crashes | a memory fault or segfault, an output of the wrong shape or dtype | 6, 10 |
| It hangs | a workgroup waiting inside a launch for another (§7), a device loop on a value never written (§6) | 6, 7 |
| It fails under capture | compiled in the capture, synchronized, or a replay with new inputs gave another result | 8 |
| An input changed | a read-only input written | 4 |
| Wrong values; outputs that differ between runs; wrong values after the timing | the math ([numerics.md](numerics.md)), also an arrival-order sum moved past its bound (it showed as wrong values, not as a difference between runs); wrong only after the timing: state kept across calls. Planted math, reduction-order and kept-state errors showed as wrong values or as wrong values after the timing, none as a difference between runs *(field, E8)* | 6, 8 |
| It is not faster than the baseline, or slower than the kernel it would replace | the speed ([tuning.md](tuning.md)) | — |

**A/B a variant** as a copy with the one change. Against the kernel it would replace, time the copy in the same rounds.
Alternate two new variants on
one idle GPU (A, B, A, B, at least four runs per arm) and compare their times: a kernel and
its comment-only copy read mean times 0.987–1.011 of each other on four sizes over three sessions. The same kernel checked three times
moved its speedup over the baseline (the exponential of the mean log ratio) by 0.4–3.2% in three sessions of four kernels (an fp8 GEMM at N 3072,
m = 16; a Qwen3.5 GDN decode kernel at 2 tokens; DS256 at 2 tokens; a Qwen2-MoE router at 256 tokens), and the fp8 GEMM's copy, timed in the rounds of the original, by 1.8–4.0%; which kernel moved most was not the same in every session *(field, E19)*;
[tuning.md](tuning.md) §2.

## Evidence

Triton paths are
`triton-lang/triton@c01b677` (tag v3.8.0), aiter's `ROCm/aiter@dbd8bf5`. "The ruler" and "the ruler's timer" are
[tuning.md](tuning.md) §1's; in these lines they read the kernel durations from Kineto.

- E1 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU, a kernel (A) and a copy with one change (B) alternated, 4 runs per arm (3 for the aiter builds), median kernel time; an offline compile for gfx950 of the N 2624 × K 6144 bf16 GEMM's m ≤ 16 launch at m = 1, 2, 8, 16] the compile: `num_stages=1` 16384 B of LDS, the kernel's own 3 101312 B, `buffer_load … lds` 48, 96, 96, 48 at m = 1, 2, 8, 16 (2 warps at m = 1 and 16, 4 at m = 2 and 8); a Qwen3.5 GDN decode kernel at 2 tokens and 4 slots, recurrence with buffer `.cs` (A) against pointer `.cs` (B): 8.79 against 8.81 µs as the mean of each run's median call time, 8.827 against 8.813 µs as the median (B/A 0.9985); the N 6144 × K 2048 bf16 GEMM on the image's aiter (e7d2453f2) against a pinned one (dbd8bf5): baseline 7.02 against 7.95 µs at m = 8 (13.3%), 7.50 against 7.16 µs at m = 16 (4.5%), 2.9%, 3.4%, 1.0%, 3.9% apart at m = 2, 4, 32, 64, the candidate within 0.1–1.7%; a DSA sparse-MLA decode kernel (24 tokens) with KV splits 8 → 4: failed to build (CompilationError) in 4 of 4 runs; MM256-FP4 (1 token) with the fold factor of its gate launch 4 → 1: crashed in 4 of 4; both unchanged kernels passed 4 of 4
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 (the toy MoE and phase launches N=4), Gluon kernels in CUDA graphs: a split reduction of 16 splits × 16384 fp32 outputs, 200 replays per arm, compared bitwise with a partial + merge launch pair; one stream writing a 16,777,216-float region, reading it last, then overwriting it, 20 eager calls and 20 replays; the ruler on one GPU, a kernel (A) and a copy with one change (B) alternated, 4 runs per arm, median kernel time; a Gluon stream reading a 1 GiB bf16 tensor, timed with CUDA events; small Triton kernels timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] fold with its counter zeroed earlier in the same graph: 200 of 200 replays equal with `acq_rel` and 200 of 200 with `relaxed`, whose scratch the reference replay had just filled with the same partials (E22); without the zeroing node wrong from the second replay on, 199 of 199; replay time 76.73 µs (zero + fold) against 7.6 µs (partial + merge); the last reader saw the values from before the overwrite in 20 of 20 eager calls and 20 of 20 replays; the N 2624 × K 6144 bf16 GEMM with `num_stages` 3 → 1 on the m ≤ 16 launch, A against B at m = 2, 4, 8: 7.97, 8.11, 8.35 against 18.34, 18.43, 18.49 µs; 8.03, 8.13, 8.36 against 18.45, 18.53, 18.61 µs; 8.00, 8.08, 8.34 against 18.33, 18.43, 18.49 µs; speedup B/A 0.433–0.453 (2.21–2.30× the time), the unchanged m = 64 launch 1.003, 1.000, 0.9948, both arms accurate in every run; the stream at 96 programs × 4 warps with one 16-byte load per lane in flight (0.375 MB) against eight (3 MB): 885.2 against 3995, 876 against 3945, 885.9 against 3985 GB/s; a toy decode MoE (32 experts, top-4, hidden 256, intermediate 128, 2 tokens) recomputing top-k in its 40 consumer programs against a selection launch, same output (relative L2 0): 9.67 / 9.69 against 11.41 / 12.76, 9.89 / 9.52 against 11.13 / 12.75, 9.67 / 9.78 against 11.47 / 12.68, 9.875 / 9.793 against 11.621 / 13.193 µs (replayed / the ruler's timer); removing one of two elementwise phase launches saved 1.15, 1.87, 1.79; 1.61, 1.88, 1.80; 1.84, 1.74, 1.76; −0.102, 1.587, 0.325 µs of replay time at 16384, 131072, 1048576 elements, outputs equal, and 4.19, 4.19, 4.17, 4.17 µs on the ruler's timer at 131072 elements (the N 2624 × K 6144 bf16 GEMM and probe kernels)
- E3 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, launches of small Gluon and plain-Triton probe kernels with the JIT cache inspected; offline compiles for gfx950 compared by text] 12 specialization and option combinations gave 12 binaries in each frontend; 9 floats (0, 1, −1, 16, 1e30 among them) with True and False used one binary per frontend, and 15 more floats, inf and NaN included, added none (one run); reassigning a `gl.constexpr` global the kernel reads (4 → 5) or an allowed plain global (3 → 4) raised `RuntimeError` at the next launch, the same value or an unread global did not; a launched copy kernel built with 9 option variants got 9 cache entries with the same code; 36 kernel configurations from 8 Gluon families, each built with the 9 variants, kept identical instructions across them; offline compiles matched the launch's amdgcn in 12 of 12 comparisons (one run); `S` changed the code of 3 plain-Triton probes only with AMD buffer passes on, and of 3 Gluon copy and gather probes neither way (probe kernels and Gluon kernels of the 8 families)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, direct calls of `run` checked against the op's reference, 3 seeds × 3 slot patterns × 2 output poisons; offline compiles for gfx950 with `DS` attributes] a Qwen3.5 GDN decode kernel at 2 tokens and 4 slots (16 V heads, K 128, V 128, 3072 channels; recurrence BK 128, BV 16, ROW_GROUP 16, LARGE_STATE_POOL False, 2 warps, `waves_per_eu` 1; norm BLOCK 128, HISTORY_BLOCK 128, 1 warp) with its recurrence on buffer `.cs` and on pointer `.cs` accesses: seeds 39107, 51749, 82021 × slots (3, 0), (1, −1), (−1, 2) × NaN or 17 in the outputs before the call: 36 calls, 144 output comparisons within the op's bounds, 18 repeat pairs bitwise equal; the recurrence compiled to 4 buffer loads, 4 buffer stores, 1 global load and no global store on the buffer arm against 0, 0, 5 and 4 on the pointer arm, 1024 B of LDS on both
- E5 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the sum of the two target kernels' device durations per call, each call after restoring the state and a 1 GiB read flush, neither counted; 32 balanced A/B profiler rounds per input, each 10 replays of 20 calls; 8 A/A pairs] the recurrence with buffer `.cs` (A) against pointer `.cs` (B) at 3 seeds × 3 slot patterns: within ±1% for 4 of 9 inputs, B slower by more than 1% for 3 (B/A 1.02797, 1.01344, 1.01179), 2 straddling the boundary; every A/A interval contained 1; the accuracy checks after the timing passed (a Qwen3.5 GDN decode kernel, 2 tokens, 4 slots)
- E6 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler on one GPU: `num_stages` A/B 4 runs per arm, the skeleton against the plain-Triton N 6144 × K 2048 GEMM and the skeleton with one-step K prefetch 3 runs per arm in a fixed order; Gluon fp8 GEMMs against aiter's plain-Triton kernel in 5 flushed graph rounds in one process; the skeleton's direct calls against the op's reference, 20 eager calls, 20 replays and a second process per input] fewer stages: the N 6144 × K 2048 bf16 GEMM at m = 2, 4, 8, 16 1.533–1.625× slower, RMSNorm at 1024, 2048, 4096 tokens 1.047–1.120×, 1.092–1.095×, 1.051–1.052× slower; unshuffled E4M3 fp8 GEMMs with fp32 block scales (N 3072, K 1536) at m = 2, 8, 16 0.777–0.781×, 0.755–0.764×, 0.759–0.768× of aiter's time, both within bounds; the skeleton at N 6144, K 2048 and M = 2, 3, 4, 8, 12, 16, 17, 32, 33, 48, 64 with one positive and one signed input each: bitwise stable, 0–1 bf16 ulp on the positive input, 0.960–0.984 of the bound on the signed one, 8192 B of `metadata.shared` passed at every launch while the amdgcn's group segment read 0; checked and timed at m = 2, 4, 8, 16, 32, 64: no correctness, build or capture failure, all six sizes in each of 3 runs at speedup 0.373–0.434 over the baseline; 2.882–3.158× the time of the plain-Triton N 6144 × K 2048 GEMM at m = 2, 4, 8, 16, and 1.695–1.729× faster with one-step K prefetch
- E8 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2 (the two-phase grid N=4), launches of small Gluon probe kernels with the binaries counted in the JIT cache; planted copies on one GPU checked against the op's reference in captured graphs with poisoned outputs (a first capture and two replays with new inputs, fresh captures per input set, a repeat, a check after the timing) and timed by the ruler, and direct calls against the op's reference; two toy Triton elementwise phases of 131072 elements as two launches and as one grid split by program id, timed as a CUDA graph replayed back to back (wall time) and by the ruler's timer (Kineto)] `INSTR=[16, 16, 32]` as a constexpr failed at the launch with `unhashable type: 'list'` while `triton.compile` of the same kernel succeeded, and a tuple launched; scales 0.5, 1e-6, 3.0, 2^-20 with flags True and False: 1 binary; a 1 KiB view at offset 0 of a 4 KiB buffer got `tt.divisibility = 16` and `tt.pointer_range = 32`, at offset 8 only the range, a 2 GiB + 16 KiB tensor only the divisibility, 3 binaries, `is_within_2gb` called twice per launch of a two-tensor kernel, cache hits included; an RMSNorm copy with `epsilon` hard-coded to 1e-5 passed every check at its sizes, and called with `epsilon` 1e-3 on 64 × 6144 inputs scaled by 0.01 had a maximum relative error of 2.235, the original kernel 0; a copy of an fp8 GEMM (N 3072, m = 16) storing 8 of its 16 rows: wrong values at the first capture with 24576 = 8 × 3072 non-finite mismatches, the original none (one run); the poison filled uint8, int32 and fp4 outputs with 0xFF bytes, fp32 and bf16 with NaN; a `run` keeping its first output in a module-level dict passed a source scan and gave wrong values at the first capture, the fresh captures, the repeat and the check after the timing; a math error: wrong values at the captures and after the timing; a changed reduction order: the same in one run, only a slower time in the other; state kept across calls: wrong values after the timing only; no difference between repeated runs in any; the two phases as one grid 6.33 / 4.30, 5.66 / 4.38, 5.66 / 4.33, 9.191 / 4.443 µs (replayed / the ruler's timer) against two launches 7.42 / 8.49, 7.39 / 8.57, 7.43 / 8.50, 8.471 / 8.610 µs, replay ratio 0.854, 0.766, 0.762, 1.085, outputs equal in every run (copies of a plain-Triton RMSNorm at hidden 6144 and of an fp8 GEMM, probe kernels)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, CUDA-graph captures and JIT probes on one GPU, with a tripwire on `JITFunction._do_compile` during each capture; the captured checks of E8 on an RMSNorm copy whose JIT key changed per call; an autotuner probe in 16 fresh processes] a constexpr warmed at 16 and launched at 32 inside a capture compiled there with one `_do_compile` call, and the tripwire raised "compiled in the capture"; the RMSNorm copy failed at every size; two fresh processes sharing a warm `TRITON_CACHE_DIR` each called `_do_compile` once (first launches 0.5631 s and 0.001563 s); an autotuner with two equal-cost configurations, listed in alternating order, timed both at its first call and neither afterwards in each of 16 processes run four at a time, 6 picking the first-listed configuration and 10 the second; a `torch.empty` tensor allocated in a capture kept its address over 3 replays and held the third replay's values; `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()` and `torch.nonzero` failed the capture (a plain-Triton RMSNorm copy and probe kernels)
- E10 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (input mutation and capture N=3), a source scan of the launch code for writes to inputs by name, `global` statements and synchronizing calls (CPU), and the captured checks of E8 on one GPU of planted copies, the read-only inputs compared byte for byte around each call; `torch.cuda.graph` captures with `capture_error_mode="global"`] `x.add_(1)` in `run` and `weight.add_(1)` in a helper taking the input's name were found by the scan, and the same write through a helper parameter named `buf` passed it and failed the byte comparison at the first capture on all 14 RMSNorm sizes; `global` and `nonlocal` were found by the scan, a module-level dict written in `run` passed it; `.item()`, `.cpu()`, `.tolist()`, `torch.cuda.synchronize()`, `torch.nonzero`, `x[mask]`, `torch.argwhere` and `repeat_interleave` without `output_size` each failed the capture, two controls (`repeat_interleave` with `output_size` among them) captured (plain-Triton RMSNorm copies)
- E11 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, `triton.compile` for gfx950 without a GPU] 108 builds of row tiles: with a constexpr M covering an 8-, 16-, 32- or 64-row tile no mask instructions (3–56 with a runtime M), masks kept where M did not fill the tile; over M = 1–128 a `range(M)` loop compiled without a branch at M = 1–22 and with 1–2 at 23–128, using 17–74 VGPRs against 66 with a runtime M; of 11 sizes of existing kernels compiled with a constexpr and with a runtime token count, not all became simpler (Gluon row-tile probes)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, trivial Gluon kernels (one 64-float load, add and store per program): replays of a bare CUDA graph of 1–64 chained kernels timed with CUDA events, least-squares slope; eager launches, 2000 × 5 rounds, host clock; the ruler's timer (Kineto); probe kernels reading past a view with `gl.amd.cdna4.buffer_load`] replay 1.815 µs per further kernel at grid 1 and 1.844 µs at grid 256, intercept 3.643 and 3.754 µs; eager host time 7.302 µs (grid 1) and 7.443 µs (grid 256), wall 7.311 and 7.452 µs; host time 6.249 µs + 1.004 µs per tensor + 0.1186 µs per int over 7.32 (1 tensor), 10.3, 14.23, 22.19, 38.45 µs (32 tensors), 15.24 and 17.12 µs (8 tensors with 8 or 24 ints); `get_tensor_specialization` 0.576 µs per call, 8 calls per 8-tensor launch; 8 launches of a 15-tensor kernel 168.8 µs of host time; 8 launches of 395.6 µs kernels 3166 µs wall against 3165 µs of kernels, of trivial kernels 171.9 µs wall against 171.7 µs of host time; graph replay 2.268 and 2.313 µs per launch with 1 and 30 tensors; an option dict built per call 7.744 against 7.686 µs built once; the ruler's timer: copy 6.498 + sum 4.62 µs apart, 10.84 µs together, flush 169.5 µs, 8 trivial kernels 32.42 µs against 3.879 µs each, median of 45 one-kernel readings 4.23 µs; on a 64-element view of a larger allocation, unmasked `buffer_load` offsets 64–79 returned 64.0–79.0, the allocation's next elements, and masked lanes without `other` returned 0; all 85 buffer ops of 27 compiled probe kernels have `num_records` `0x7ffffffe` (probe kernels)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=?, Kineto trace of the calls at each size] kernel launches per call over the 242 sizes of the library's 154 kernels: 1 in the fp8 GEMMs and the norms; 2 in the N 2624 × K 6144 bf16 GEMM at m = 64; 2–8 in the MoE layers; 2–3 in DSA; 3 in gated attention; 1–2 in GDN decode; 5–8 in GDN prefill
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the self-resetting counter N=3), Gluon fold and merge kernels in CUDA graphs over 4–8 XCDs, replay time and the ruler's timer (Kineto)] 4, 16, 64 splits × 4096, 16384, 65536 fp32 outputs, 64 outputs per one-wave program: counters zeroed by an earlier node, 2000 of 2000 replays right with `acq_rel` and with `relaxed` at each size, the `relaxed` fold reading a scratch the reference had just filled with the same partials (E22); without a reset, 1999 of 1999 later replays wrong; the merge launch 0.03–0.65× the replay time of the `acq_rel` fold and 0.51–0.92× of the `relaxed` one, 0.12–0.82× on the ruler's timer at up to 16 splits and 16384 outputs, each fold's time including its zeroing node; a counter the last arrival reset itself: 1000 of 1000 replays right in each of 3 runs (Gluon fold probes)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, synthetic fp8 VALU MoEs and a decision sweep, CUDA-graph replay time and the ruler's timer (Kineto), arms rotated per round] two MoE layouts × 1, 2, 8, 16 tokens, 16 paired rounds per size: recomputing the top-8-of-256 decision in every consumer slower on both timers, the output bitwise equal to the baseline configuration's; 7 decisions × 2 or 16 tokens × 40, 256, 1024 or 4096 consumer programs = 56 configurations, 12 rounds each, the recomputing and publishing arms bitwise equal: top-8 of 256 with 4 or 16 split partials faster recomputed at 1024 programs and published at 4096; median extra-launch cost 3.2696 µs replayed and 4.4679 µs on the ruler's timer, which as a constant predicted the faster arm in 49 and 50 of the 56; identical-arm intervals excluded zero at 8 configurations replayed and 3 on the ruler's timer, not corrected for multiple comparisons (synthetic probe kernels)
- E17 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, a reference run of the ruler, stored speedup against the op's baseline] a plain-Triton RMSNorm at hidden 6144: 1.023–1.029 at its 7 sizes of 3–18 tokens and 1.040–1.214 at 64–4096 tokens
- E18 [2026-11, gfx950 (offline), Triton 3.7.0 and 3.8.0, N=1, `triton.compile` for gfx950 without a GPU, static instruction count of the whole kernel] a plain bf16 `tl.dot` K loop (64 × 64 × 64 tile, 4 warps, 2 stages, `schedule_hint="memory-bound-attention"`, a runtime trip count with `D`): 274 instructions, 16384 B of LDS and 12 `v_mfma_f32_32x32x16_bf16` on 3.7.0 against 405, 33968 B and 16 on 3.8.0; the hint changed 3.7.0's build (278 → 274 instructions, `iterative-ilp` added) and left 3.8.0's identical (probe kernel)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 runs in each of 3 sessions, the ruler on one idle GPU, each kernel alone: the kernel (A) alternated with a copy whose first line carries a comment (A'), 4 runs per arm, the spread taken over the first 3 runs of A; then three runs of the fp8 GEMM's A' timed in the same rounds as A; ratio = the exponential of the mean log ratio of baseline time to kernel time, time = median kernel time] ratio spread (max − min) / min over 3 runs of A: 2.385% (an fp8 GEMM at N 3072, m = 16), 0.911% (a Qwen2-MoE router, 256 tokens), 1.731% (a Qwen3.5 GDN decode kernel, 2 tokens: 2.4222, 2.4105, 2.4522), 0.942% (DS256, 2 tokens); the fp8 GEMM's A' in the rounds of A 1.4784, 1.4386, 1.4749 (2.77%); mean kernel time of A' over A 0.987, 1.002, 1.005, 0.997 in the order of the four; in two later sessions the ratio spread 2.51% and 2.32% (the fp8 GEMM), 0.98% and 2.50% (the router), 0.41% and 1.51% (GDN decode), 2.42% and 3.22% (the MoE), the fp8 GEMM's A' in the rounds of A 1.8% and 4.0%, A' over A 0.995–1.003 and 1.003–1.011
- E20 [2026-11, gfx950 (offline), Triton 3.7.0 and 3.8.0, N=1, `triton.compile` for gfx950 without a GPU] the 11 N 3072 × K 1536 fp8 GEMM kernels from m = 16 up (m = 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192, 8193) all compiled under 3.8.0 and all failed under 3.7.0: m = 16 and 64 because `shared_memory_descriptor._reinterpret()` requires `dtype`, m = 32 on an `AssertionError` in `core.py` `__getitem__`, m = 128 on its offsets layout, the other seven on the missing `compute_efficient_padded_shared_layout`
- E21 [2026-11, gfx950 (offline), Triton 3.8.0, N=1, `triton.compile` for gfx950 without a GPU, amdgcn instruction count] a test kernel with a 4-load loop body: `gl.static_range` gave 120 instructions at 8 trips and 6513 at 512, `range` 120 at 8 and 129 at 512 (test kernel)
- E22 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, Gluon fold kernels in CUDA graphs, 3000 replays per arm and size, the partials set to NaN before every replay, counters zeroed by an earlier node] 8 and 32 splits × 16384 and 262144 fp32 outputs: `relaxed` 3000 of 3000 replays wrong at each size, `acq_rel` and `relaxed` with `.wt` partial stores 0 of 3000; one `buffer_wbl2` and one `buffer_inv` in each `acq_rel` fold, none in the `relaxed` ones (Gluon fold probes)
- E23 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, `triton.autotune` over two configurations differing only in an unused constexpr, fresh processes on one GPU sharing a warm Triton cache; the captured checks of E8 with E9's tripwire on `JITFunction._do_compile`] 16 processes run one at a time and 16 run four at a time each timed both configurations at its first call and none at its second; one at a time all 16 picked the configuration timed second, four at a time 10 the first-listed and 6 the second; an RMSNorm and a GEMM copy whose JIT key changed per call: every size failed as "compiled in the capture" (copies of a plain-Triton RMSNorm and of the N 6144 × K 2048 bf16 GEMM)
- S1 triton-lang/triton@c01b677:python/triton/experimental/gluon/_runtime.py:45 `GluonJITFunction(JITFunction)` swaps only the AST source; the module takes `ttg.num-warps` from the options (:32) and `ttg.threads-per-warp` from the warp size (:34); the verifier requires a layout's warp count to equal the module's (lib/Dialect/TritonGPU/IR/Dialect.cpp:3877, lanes :3863)
- S2 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:585 `add_stages`: Triton `make_ttir` + `make_ttgir`, Gluon `gluon_to_ttgir` (:327–349, no pipeliner, no buffer-op pass); `make_ttgir` coalesces (:269), reads `matrix_instr_nonkdim` and `kpack` (:273) and `num_stages` (:288–289), runs the buffer-op passes under `use_buffer_ops` (:303); `schedule_hint` "right now no effect" (:104); `AMDGCN_USE_BUFFER_OPS` defaults to on (python/triton/knobs.py:517)
- S3 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp:135 `canUseBufferOps`: a splatted base plus 32- or 64-bit offsets, safe by `tt.pointer_range` 32 on the argument (:112–130) or by a byte offset proven under 2 GiB
- S4 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:78 `HIPOptions.num_warps = 4`; warp size 64 (:114); a power of two (:116); `kpack` other than 1 overwritten with 1 on gfx950, with a deprecation warning (:119–123); `parse_options` raises `num_ctas > 1 not supported on gfx950` (:162–163; `supportsMultiCTALaunch` is false there, third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:240) and keeps only `HIPOptions` fields (:181); `gl.barrier()` synchronizes a CTA, `gl.barrier(cluster=True)` a cluster when `num_ctas > 1` (python/triton/experimental/gluon/language/_core.py:703); a kernel whose `metadata.shared` exceeds the device's `max_shared_mem` raises `OutOfResources` when its first launch loads it (python/triton/compiler/compiler.py:466)
- S8 triton==3.7.0+gitb4e20bbe.rocm7.14.0a20260608:python/triton/runtime/jit.py:697 in Triton 3.7.0 (that ROCm wheel), a launch keyword that is not an option raises `KeyError`; its `HIPOptions` has no `llvm_fn_attrs`; its Gluon has no `clamp`, `aggregate_replace`, `expect_zero`, `amd.slice`, `amd.cdna4.compute_efficient_padded_shared_layout` or `amd.cdna4.scaled_upcast`, all present in 3.8.0; `schedule_hint` `memory-bound-attention` sets `amdgpu-sched-strategy` `iterative-ilp` (backends/amd/compiler.py:393)
- S16 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:387 the globals a JIT body may read; `NameError` at :411
- S17 triton-lang/triton@c01b677:python/triton/runtime/jit.py:593 the key `(tuple(specialization), str(options))`, a constexpr entering it as `("constexpr", value)` (:409); caches per device (:738); an in-process miss calls `_do_compile` (:757), which reads the disk cache inside (:877); unknown launch keyword `KeyError` (:715); changed globals `RuntimeError` (:761); `ConstexprFunction.__call__` returns the raw value outside JIT, a constexpr inside (:1153); `Autotuner.run` times every config at a key's first call (python/triton/runtime/autotuner.py:240)
- S20 triton-lang/triton@c01b677:python/src/specialize.cc:274 an int equal to 1 becomes a constexpr, otherwise `i32` or `i64` by range (:281), `u64` above (:297), `D` for a multiple of 16 (:283); each tensor's dtype goes through Python's `canonicalize_ptr_dtype` (:329), whose table has no `float4_e2m1fn_x2` or `float8_e8m0fnu` (python/triton/_utils.py:77) and raises `KeyError` (:120); a Python callback per tensor without native tensor specialization (:342–370); bool `u1` (:377), float `fp32` (:385), `None` a constexpr (:522)
- S21 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:236 `get_tensor_specialization`: `D` by alignment (python/triton/backends/compiler.py:89), `S` by `is_within_2gb` on the storage (:211); HIP opts out of native tensor specialization (:140); `parse_attr`: `S` → `tt.pointer_range` 32 (:229), `D` → `tt.divisibility` 16 (python/triton/backends/compiler.py:75)
- S38 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/_layouts.py:34 `AMDMFMALayout` version 4 is gfx950
- S40 ROCm/aiter@dbd8bf5:setup.py:124 without `AITER_USE_SYSTEM_TRITON=1`, aiter's setup.py replaces the installed Triton (:131–139)
- S44 torch==2.12.0+rocm7.14.0a20260608:torch/_torch_docs.py:8242 `torch.nonzero` on a CUDA tensor "causes host-device synchronization", `torch.argwhere` too (:6861); `repeat_interleave`'s `output_size` "will avoid stream synchronization" (:13248); a bool index goes through `nonzero` (torch/_meta_registrations.py:3634); the same notes in the public docs, https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html, torch.argwhere.html, torch.repeat_interleave.html
