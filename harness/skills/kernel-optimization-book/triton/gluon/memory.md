---
title: "Gluon on gfx950: moving data between HBM, L2, LDS and registers (Triton 3.8.0)"
kind: language-note
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
sources:
  - triton-lang/triton@c01b677:third_party/amd/backend/compiler.py
  - triton-lang/triton@c01b677:python/triton/backends/compiler.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py
  - triton-lang/triton@c01b677:python/triton/language/core.py
  - triton-lang/triton@c01b677:python/triton/language/semantic.py
  - triton-lang/triton@c01b677:python/src/gluon_ir.cc
  - triton-lang/triton@c01b677:python/src/ir.cc
  - triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td
  - triton-lang/triton@c01b677:lib/Analysis/AxisInfo.cpp
  - triton-lang/triton@c01b677:lib/Analysis/Allocation.cpp
  - triton-lang/triton@c01b677:lib/Analysis/Membar.cpp
  - triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
  - triton-lang/triton@c01b677:third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h
  - triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/AsyncUtility.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp
---

# Moving data in a Gluon kernel on gfx950

How a Gluon kernel moves bytes between HBM, registers and LDS on gfx950 with Triton 3.8.0: what the compiler does for
you, what you write by hand, and what it measured. "Compiled" results are compile-only with Triton 3.8.0's own LLVM;
timings are MI355X runs. A kernel's call time is the ruler's timer ([../tuning.md](../tuning.md) §1; its flush: §6), read from a Kineto trace in 2026-11 unless a line names rocprofiler-sdk; a speedup is the ruler's for one size against the implementation the kernel replaces. The kernels named are from the library (154 Triton and Gluon kernels for gfx950: [../kernels/kernel_library.md](../kernels/kernel_library.md) §1), used as examples: blockwise-scaled fp8 GEMMs, MoE kernels under the labels of [../moe.md](../moe.md) §1 (MM64-FP8, MM256-FP8, MM64-FP4 and MM256-FP4: MiniMax-M2 on a rank holding 64 or all 256 of its experts, with FP8 or MXFP4 weights; DS256 and DS512: DeepSeek-V2 on a rank of that intermediate dimension) and the Qwen2-MoE router, Qwen3.5's gated attention and GDN, Qwen3-Next's GDN, and sglang's DSA sparse MLA; each is built for the token or row counts named with it.
Layouts: [layouts.md](layouts.md); MFMA: [mfma.md](mfma.md); asm: [wave_ops.md](wave_ops.md).

## 1. What to use when

| You need | Use | § |
|---|---|---|
| A tensor tile from global memory | `gl.amd.cdna4.buffer_load(base, offsets, mask)`: scalar base, int32/uint32 element offsets; Gluon does not turn `gl.load` into it | 2, 3 |
| A scalar, an index, per-program metadata | `gl.load(ptr + i)`, an `s_load` when uniform; compiled, not timed: in a rolled loop whose body also copies or stores it became `global_load_dword` even when placed first, and a constant-address-space pointer kept it an `s_load` *(field, E41)* | 13 |
| A tensor that can exceed 2 GiB from one base | 64-bit rebase of the scalar base, or a host gate to a pointer path | 3 |
| Bounds | for buffer loads whose masked lanes need zero, use a mask without `other`; `gl.load` needs an explicit fill if masked values will be consumed. Fold masks on full tiles, and use row masks when the contiguous extent is entirely in bounds | 4, 5 |
| 16-byte accesses | 16 bytes per lane in the layout, a base marked divisible by 16, offsets the compiler can prove contiguous | 5 |
| A mostly read-once stream | `.cg` (same load instruction as `.cs`): in one MoE study (N=1), faster in every pair, in time and in speedup over the baseline, at 6 of 10 sizes whose routing reads each weight tile about once (mean readers < 2), by 1.6–9.6%; slower at none. Those sizes are not all strictly read-once. The three 1-token sizes had pairs both ways; the two re-read sizes lost with `.cg`. A/B per size; this is not a token-count rule *(field, E35)* | 6 |
| Bandwidth from a small grid | A/B nominal bytes in flight, not a scaling law for small calls: in a synthetic 1 GiB bf16 stream, at equal 3 MiB, 96×4×8 and 192×1×16 read 0.800×/0.678× of 384×4×2 (paired 95% CIs 0.7986–0.8021/0.6775–0.6793; N=1, 12 randomized blocks). The §7 table's stream ran 192×1×16 at about half rate (N=3); actual small-call kernels and P/256 CU scaling untested *(field, E14, E42)* | 7 |
| A tile in LDS | `async_copy.buffer_load_to_shared`, or registers → `allocate_shared_memory` + `.store`: correct on a 64×64 bf16 tile stream; the three LDS modes differed by 0.7–2.5% in 4 runs (a descriptive spread, not an equivalence test) *(field, E16)*; a tile read by several waves not timed | 8 |
| A layout change | `gl.convert_layout`: lane moves with fewer than two register–lane transpositions go by DPP, `v_permlane` or `ds_bpermute`; those with two or more (also within one wave) and any wave move go through LDS ([layouts.md](layouts.md) §7) | — |
| A tile only its own wave reads (preshuffled weights) | can load into the operand layout without LDS; a one-wave fp8 test kernel was bitwise-correct, not timed against LDS | 8, 13 |
| Swizzled or padded LDS for a DMA'd tile | pre-swizzled global offsets into a linear `_reinterpret` view, or a padded layout | 9 |
| Load/MFMA overlap | a hand-written ring of commit groups (rolled for long K) | 12 |
| LDS reuse | separate allocations with `.load` use compiler-inserted barriers; indexed arenas using `load_shared_relaxed` require explicit slot-lifetime reasoning and barriers. Ensure copies complete before reads and all consumers finish before refill; neither spelling alone guarantees race freedom | 10, 11, 12 |
| Programs that read the same data in one call | in the tested gfx950 grids, ids congruent mod 8 mapped to one XCD; verify placement and A/B the remap for the target grid and partition rather than assuming a universal scheduling guarantee | 14 |
| Independent counters and ticket allocation | `sem="relaxed"` only when the atomic does not publish or acquire associated data; a communication protocol needs the appropriate ordering and scope. Atomics: [../moe.md](../moe.md) §6 (buffer atomics), [../numerics.md](../numerics.md) §8 (determinism) | — |

## 2. What the compiler does and does not do for you

`@gluon.jit` kernels are lowered by `gluon_to_ttgir`: inliner, layout resolution, SCCP, CSE, canonicaliser, loop
unroller, select/if combining, the warp-pipeline pass, warp-group allocation, nothing else outside the `fpsan`
instrumentation mode. Coalescing, the software pipeliner and the conversion to buffer ops of pointers marked "S"
(`tt.pointer_range = 32`: with buffer ops enabled, a `ptr_range()` or tensor storage size of at most `2**31 - 1` bytes) belong to `make_ttgir`, the plain-Triton path, which pipelines with
async copies on gfx950 (S1; [../gemm/small_m_gemm.md](../gemm/small_m_gemm.md)). So in Gluon:

- **`gl.load`/`gl.store` stay pointer accesses**: `global_load`/`global_store`, or `s_load` for a suitably uniform load (§13). Compiled, not timed: in a rolled loop whose body also copies or stores, a uniform load placed before the copy still became a global load, and a constant-address-space pointer kept it scalar *(field, E41)*. A masked copy of tensors marked "S" compiled to `global_load_dwordx4` through `gl.load` and to `buffer_load_dwordx4` through plain `tl.load` *(field, E32)*. Qwen3.5's GDN decode at 2 tokens/4 slots passed 36/36 accuracy checks with its state through either path *(field, E36)*. Timed, the state through `gl.load`/`gl.store` `.cs` against the kernel's own buffer ops `.cs` took 8.79 → 8.81 µs of call time *(field, E45)*; on a cold device (state restored and 1 GiB flushed before each call, N=1) it stayed within ±1% at 4 of 9 inputs, was 1.2–2.8% slower at the 3 inputs with slots (−1, 2) and unresolved at 2 *(field, E43)*: close, not equal speed. In a compiled 64×64 bf16 copy (8 elements per lane, 4 waves, pointers "D" or "DS"), masked buffer accesses had no exec branch or `s_and_saveexec`, masked pointer accesses 6 of each, unmasked pointer accesses none (8 builds, not timed) *(field, E36)*.
- **The offset tensor's register layout is the access pattern**; nothing re-coalesces it (§5).
- **No automatic `num_stages` software pipeline is added.** Gluon's warp-pipeline pass handles explicit warp-pipeline
  stage annotations; only `make_ttgir` reads `num_stages` for automatic loop scheduling (S1): three Gluon loops compiled
  to identical instructions at `num_stages=1` and `3`, a plain `tl.dot` loop did not *(field, E32)*. Write the prefetch
  (§12).
- **Membar still runs** in the lowering, with an AMD filter, and places LDS barriers (S2; §11).

## 3. Buffer loads and stores: signature, addressing, the 2 GiB rule

| Call (Triton 3.8.0) | Notes |
|---|---|
| `gl.amd.cdna4.buffer_load(ptr, offsets, mask=None, other=None, cache=None)` | `ptr` a scalar pointer; `offsets` an int32/uint32 tensor in elements; `other` needs `mask`; the result has `offsets`' layout and `ptr`'s element type |
| `gl.amd.cdna4.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)` | no `other`; `stored_value` must already have `ptr`'s element type |
| `gl.amd.cdna4.async_copy.buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")` | §8 |
| `gl.amd.cdna4.async_copy.global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="")` | `ptr` a tensor of pointers, 64-bit: no 2 GiB bound, it copied 4 KiB at byte 2^31 + 8192 of a 3 GiB tensor that `buffer_load_to_shared` read as 0 *(field, E4)*; 16 bytes per lane compiled, 8 bytes per lane or the tested row-strided lane layout failed to legalize (neighboring lanes differ by one runtime row stride; 64 bytes when that bf16 stride is 32 elements) *(field, E28)* (S25; §5) |

The keyword is `cache=` on `buffer_load`/`buffer_store` (S4) and `cache_modifier=` on `gl.load`/`gl.store`, the `tl`
builtins (S17, S18). Kernels that call `buffer_load`/`buffer_store` through the previous generation's `gl.amd.cdna3` get
the same functions: `cdna4` re-exports those two (S3).

**What the hardware checks.** Each base becomes one descriptor with `num_records = 2^31 − 2` bytes; each lane sends
`offset × element bytes`, multiplied in 32 bits, and a masked-off lane sends `0x80000000` (S5). A raw buffer access is
out of range unless it lies wholly below `num_records`, each dword of a wider access on its own (a byte load at byte
2^31 − 3 read its data, a dword load at 2^31 − 4 read 0), and the base is not part of the check ([ISA §2](../../hardware/gfx950/memory_and_sync.md)) *(field, E4, E50)*. Hence:

1. **No tensor bound.** An unmasked `buffer_load` past the end of a 64-element view returned the data behind it
   (64.0–79.0), and the ISA held `s_mov_b32 s2, 0x7ffffffe` *(field, E4)*. Bound every access with its mask.
2. **Silent zeros, then aliasing.** Out of range, loads read 0 and stores were dropped without a fault; an offset
   reaching 2^32 wrapped onto the wrong element *(field, E4)*. Keep every lane's last byte below 2^31 − 2.
3. **The base is free.** Put wave-uniform origins (tile, expert, slot, split) into the scalar pointer, as the
   Qwen2-MoE router does, `Hidden += tile_m * BM * K + split * (K // SPLIT)` with offsets
   `rows[:, None] * K + aks[None, :]` (mechanism: [../../gemm/datapath.md](../../gemm/datapath.md) §8), with overflow checked for correctness, and uniformity/alignment checked for efficient lowering *(field, E29)*:
   - **Cast to 64 bits before the product can pass 2^31**: it forms in 32 bits, `x_ptr + e * (7168 * 2048)` with an
     int32 `e` compiling to `mul i32`, which wraps silently, and with `e.to(gl.int64)` to `mul nsw i64`. MM256-FP4
     (the kernel for 1024 tokens) writes `W + expert.to(gl.int64) * (N * (K // 2))`, `expert` from a scalar `gl.load`, and keeps
     32-bit offsets.
   - **Provably wave-uniform.** Offset by a program id, a kernel argument, a scalar `gl.load` or a one-warp `gl.max`,
     the base stayed scalar; by a 4-warp `gl.max`, the `buffer_load` compiled into a waterfall loop
     (`v_readfirstlane_b32`, `s_and_saveexec_b64`, `s_cbranch_execnz`). Move such a value to an SGPR with an inline
     `v_readfirstlane_b32` first, as DS256 at 256 tokens does with its job descriptor, reduced with `gl.max`
     ([wave_ops.md](wave_ops.md) §6.4).
   - **A provable multiple of 16 bytes**: constexpr shapes, or runtime ints the launch marks divisible by 16, which it
     does only for values that are (S16). With `X + slot * stride`, `stride` unmarked, a 16-byte bf16 `buffer_load`
     became 8 `buffer_load_ushort` and the async copy failed at translation; a marked or constexpr `stride`, or
     `gl.multiple_of(slot * stride, 16)`, gave `buffer_load_dwordx4`: one source can compile at one size, fail at another.

The K step can ride the base too, keeping the offsets loop-invariant; DS256's kernel for 1024–3072 tokens picks per token count with a constexpr flag, copying its activations from `X + base * (BK // 2) if REBASE else X` at offsets `row[:, None] * (K // 2) + (kl if REBASE else base * BK // 2 + kl)[None, :]`. Gate a pool that can outgrow the range on its **byte** extent against `2**31 - 2` on the host, with another path above it (a 64-bit rebase per slot, `gl.load`, `global_load_to_shared`); where the caller bounds the pool below the range, that path is dead code *(field, E4)*. The host gate of Qwen3.5's gated attention (the 4-row kernel):

```python
key_extent = (key_cache.shape[0] - 1) * key_cache.stride(0) + 256  # elements up to the end of the last row
value_extent = (value_cache.shape[0] - 1) * value_cache.stride(0) + 256
buffer_kv = (  # False: the launch takes the kernel's pointer path
    key_extent * key_cache.element_size() <= 2**31 - 2
    and value_extent * value_cache.element_size() <= 2**31 - 2
)
```

- **Constant offsets.** A constant added to the offsets rode the 12-bit `offset:` immediate at 256 bytes and cost a
  second offset register at 4096 bytes; `+2047` bf16 elements (4094 bytes) split each 16-byte load into 2-byte loads
  *(field, E29)*. Keep constant steps multiples of 16 bytes and under 4096.
- **Index math in uint32.** `//` and `%` by a power of two need sign fix-ups in int32, not in uint32: a test kernel
  computing the 16×16 preshuffle offset ([mfma.md](mfma.md) §7) from a tile index passed as an argument had 28 VALU
  instructions in int32, 22 after `.to(gl.uint32)`, and `gl.assume(tile >= 0)` changed nothing *(field, E29)*; the
  Qwen2-MoE router and the n = 7168 fp8 GEMM take program ids as uint32. Convert only values non-negative by
  construction; per-lane offsets kept their width.
- **`buffer_store` does not convert** (`gl.store` does, S8): write `acc.to(gl.bfloat16)`; fp32 through a bf16 pointer
  failed with "'amdg.buffer_store' op failed to verify that value element type matches the pointed type of ptr"
  *(field, E6)*.

## 4. Masks, `other`, sentinels

| Access | A mask costs | `other` adds |
|---|---|---|
| `buffer_load` | one offset select per vector, no branch (S5) | a select per 32-bit word, `other=0` included: 1 → 5 `v_cndmask` for one 16-byte vector *(field, E6)* |
| `buffer_load_to_shared` | the same for the measured non-replicated tile; redundant-thread predication can also add a branch | a branch around each copy and a `ds_write_b128` of `other` (S10): a u8 `[64, 256]` copy, wait, read and store, 4 copies per lane, went from 59 to 103 instructions *(field, E6)* |
| `gl.load`, `gl.store` | a conditional branch for a non-constant mask (S9); the measured divergent tile masks use exec-mask branches around vectors, while a constant-true mask needs no mask branch (below) *(field, E6)* | — |

Only the zero the lowering builds when `other` is omitted is known to be zero (S5); a passed `other`, 0 included, is
selected in (S10) unless the mask folds to the constant `true`: at a dividing M, `buffer_load(..., other=0)` compiled
with no `v_cndmask` and the async copy with no branch or `ds_write`, and the async-copy kernel of an n = 3072 fp8
GEMM the same code with and without its zero `other` tensors at dividing M *(field, E30)*. Masked lanes of a buffer
load already read 0: write `buffer_load(ptr, offs, mask)`, and pass `other` only for a non-zero fill such as unit
e8m0 scales (`other=127`, or `2139062143` for a word of four). How kernels avoid paying for bounds:

- **Masks that fold on full tiles**, with the token count a `gl.constexpr` (`M` is specialized to the token count the kernel is built for; this does not imply one binary per token count):
  `((row[:, None] < M) | (M % ROWS == 0))`, or `None if FULL else mask`. With a runtime `M` the mask stayed,
  and `gl.load`/`gl.store` kept their branch *(field, E30)*.
- **Clamp reads, mask only the store**, for `M > 0` and other accessed dimensions fully in bounds; `M % BM == 0` is decided at compile time:
  `read_rows = rows if M % BM == 0 else gl.minimum(rows, M - 1)`.
- **Offset sentinels, 1-byte data only**: DSA gives invalid (negative) KV slots the element offset `-2147483648`
  (`gl.where(selected >= 0, selected * KV_ROW, -2147483648).to(gl.int32)`), out of range as an unsigned offset, so its
  fp8 gathers need no mask. For 2- or 4-byte elements the 32-bit multiply turns it into 0 (S5) and those lanes read
  element 0: use a mask.

## 5. Vector width and contiguity

A buffer access moves `min(16 bytes, contiguity × element bytes)` per lane; contiguity is limited by the compiler's offset contiguity/alignment analysis, the layout's contiguous elements per thread, and the scalar base's divisibility expressed in elements (S14), capped by the mask's constancy along the contiguous
dimension (S10, S15); bf16 offsets misaligned by 4 or 8 bytes still loaded 16 bytes *(field, E32)*. `offs < n` with a
contiguous `offs` is constant over `gcd(contiguity, divisibility of offs, divisibility of n)` elements (S15), and with alignment specialization enabled, a launch marks an integer argument divisible by 16 only when it is (S16). In the following bf16 snippet, a runtime column length without that proof narrows the vector; other expressions or valid hints can provide different proofs:

```python
@gluon.jit
def copy_masked(x_ptr, y_ptr, n, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])  # 8 bf16 = 16 bytes per lane
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    x = gl.amd.cdna4.buffer_load(x_ptr, offs, mask)
    gl.amd.cdna4.buffer_store(x, y_ptr, offs, mask)
```

For bf16, `BLOCK=2048`, `num_warps=4`, 16-byte-aligned bases and alignment specialization enabled, it compiles with Triton 3.8.0 for gfx950 to one `buffer_load_dwordx4` and one `buffer_store_dwordx4` per lane when
`n % 16 == 0` at launch, and to 8 `buffer_load_ushort` and 8 `buffer_store_short` otherwise; `gl.load` narrows the
same way *(field, E32)*. A row mask, constant along the contiguous dimension, kept a `[64, 16]` fp32 tile at
`buffer_load_dwordx4` *(field, E6)*: mask rows, keep the contiguous extent a constexpr multiple of the vector, and
give full tiles an unmasked path.

Axis analysis (S15) loses contiguity through XOR and `//` (unless by 1), takes a runtime argument's divisibility as 1
unless the launch marks it divisible by 16, keeps gcd(contiguity, divisibility, divisor) through `%` by a constant, and
through `gl.where` what both branches and its condition's constancy allow: XOR of 8-element chunks, an unmarked `stride`
or a `gl.where` varying along the row cut bf16 loads to 2 bytes, `% 65536` or a row-constant `gl.where` kept 16
*(field, E32)*. Restate only what is lost, with `gl.max_contiguous(gl.multiple_of(offsets, [1, V]), [1, V])` (S17), V
the elements of the true aligned run (16 for the fp8 chunks of §9's excerpt, 8 for bf16; [layouts.md](layouts.md) §6.1).
Both are promises the compiler does not check (S18), and a larger V is believed: bf16 offsets permuted in 8-byte chunks
loaded `buffer_load_dwordx2` with V = 4 and `buffer_load_dwordx4` with V = 8, reading wrong values *(field, E32)*.

An async copy depends on these proofs. Triton 3.8.0 lowers direct-to-LDS loads on CDNA4 only at 128 and 32 bits per
lane, leaving out the ISA's 8-, 16- and 96-bit forms (S20; [ISA §3](../../hardware/gfx950/memory_and_sync.md))
*(field, E28)*, and rejects a copy whose provable vector is narrower than its layout's per-lane run, or whose lanes do
not write `dest` in order (S14, S25), with `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`:
a runtime row stride of unknown divisibility did (marked divisible by 16, or hinted, it compiled to
`buffer_load_dwordx4 … lds`), and so did aligned 16-byte offsets whose lanes walked rows 64 bytes apart *(field, E28)*.

## 6. Cache modifiers: the bits, and what they bought

Valid strings: loads `.ca .cg .cs .cv`, stores `.wb .cg .cs .wt`; anything else raises `ValueError` (S8). The bits on
gfx950 (from S14, checked in compiled ISA *(field, E8)*) and a 1 GiB bf16 stream, 2048 programs × 4 warps, 16 bytes
per lane, in GB/s over 3 runs *(field, E9)*:

| Op | String | `buffer_*` bits | `global_*` bits | Stream, `buffer_*` | Stream, `gl.*` |
|---|---|---|---|---|---|
| load | none, `.ca` | none | none | 6009–6019 | 5986–6005 |
| load | `.cg`, `.cs` | `sc0 nt` | `nt` | 6796–6837 | 6771–6804 |
| load | `.cv` | `sc0 sc1` | `sc0 sc1`, then `s_waitcnt vmcnt(0)` after each load | 6013–6018 | 6008–6016 |
| store | none, `.wb`, `.cg` | none | none | 5793–5889 | 5768–5855 |
| store | `.cs` | `sc0 nt` | `nt` | 5456–5510 | 5404–5473 |
| store | `.wt` | `sc0 sc1` | `sc0 sc1`, then `s_waitcnt vmcnt(0)` after each store | 5822–5879 | 5721–5775 |

On the pointer path `.cv` and `.wt` are volatile and serialise every access; the buffer op sets the same bits without
waits ([ISA §2](../../hardware/gfx950/memory_and_sync.md): policy per bit pattern). With each program on its own
contiguous 512 KiB, `nt` sped reads up 13.0–13.7% on either path, `.cs` reduced store bandwidth by 5.5–7.2%, and `.wt` reduced `gl.store` bandwidth by 0.4–1.7% *(field, E9)*. The launch shape moves these: with all programs sweeping the tensor together `nt` gained 8–9.5%, `.cs` cost at most 2.1%, and `.wt` made `gl.store` 1.4× faster and cost `buffer_store` 1.1–2.3% *(field, E51)*.

**What ran before the kernel.** The ruler's flush runs before each timed call, inside the graph (85 calls, 85 preceding flushes; N=2), reads 1 GiB, leaves that buffer unchanged and writes only per-block sums ([../debugging.md](../debugging.md) §11) *(field, E37)*. After a 1 GiB read flush like that one, `nt` gained about 4–12% on 64 MB–1 GiB streams (`.cg`/none 1.039–1.117 over 4 runs, eager launches): +3.9–7.4% at 64 MB, +9.0–10.4% at 128, +5.9–8.8% at 192, +10.8–11.5% at 512, +9.9–11.7% at 1024. With 64 MB hot from the previous launch `.cg` gained 5–16% in all 12 dedicated rounds (N=4), each timing none before `.cg`; the sweep's single hot medians after write-flush launches gave 1.122, 1.044, 0.960 and about 1.10×, and an earlier hot probe found `.cg` slower (0.96×, 0.93×). These order-specific probes give no general sign for a hot cache, and none of them is the ruler's flush under graph replay *(field, E11)*.

**In kernels** effects ranged from no consistent direction to 17%, not a universal 1–2% or token-count rule. At 1 token three sizes had pairs both ways; at 2 tokens DS256 moved 0.0–1.2% (N=3, every pair the same way in two runs), MM256-FP8 3.7% on time but 2.0% on speedup (N=1, 3 of 4 speedup pairs agreeing). For MM64-FP8 at 4/8/16 tokens, time without/with `.cg` was 1.023–1.096; MM64-FP4 at 128 tokens 1.016–1.018; DS256 at 2048 tokens 1.016–1.017 (each N=1; ratios, not percent time savings). At 256 tokens read-once MM64-FP4 lost 2.6–3.5% without `.cg`, while re-read DS512 lost 0.2–0.7% with it (each N=4); re-read MM256-FP4 at 16384 tokens lost 17% with it (N=1). These are measured cases, not guarantees for every workload *(field, E12, E35)*; all rows: [../tuning.md](../tuning.md) §11. Kernel sources also key hints on reuse: the DS512 kernel for 2048 tokens keeps `.cg` on its gate-up weights only up to 1536 tokens, the DS256 kernel for 2 tokens drops it at 1 token, the DS512 kernel for 256 tokens on its 128-row gate-up tiles above 128 tokens, and the MM256-FP8 kernel for 256 tokens drops it per job, for experts with more than `BM` routes: its planner launch sets bit 24 of a job's descriptor (`packed |= gl.where(count[:, None] > BM, 1 << 24, 0)`) and its gate-up worker branches on it, the hint being a constexpr string:

```python
packed = gl.load(Jobs + job).to(gl.uint32)
if packed & 1 << 24 != 0:
    _run_w13_job(
        # ... (19 arguments, the same in both calls)
        ".ca",
    )
else:
    _run_w13_job(
        # ... (the same 19 arguments)
        ".cg",
    )
```

**A/B the hint a kernel already has too.** Flipping it won in every pair, in time and in speedup, at 4 of 12 sizes (MM64-FP8 at 4/8/16 tokens, MM64-FP4 at 128; N=1). Stores also differed by size: removing `.cs` from the n = 3072 fp8 GEMM output cost 1.7–1.9% at m = 6144 and gained 0.1–0.7% at m = 8193 (every pair, N=3) *(field, E12, E35)*; C-tile store policy: [../../gemm/epilogue.md §5](../../gemm/epilogue.md). Pair direction is descriptive, not statistical proof: at 1 token single pairs swung −3.9% to +2.8%; report the uncertainty and a same-window A-vs-A control. Three-run A-vs-A spreads were 0.4–3.2% on four kernels over three sessions (the n = 3072 fp8 GEMM at m = 16 5.9% over 8 runs) *(field, E13)*, and up to 3.5% with the ruler reading rocprofiler-sdk, where one whole run read about 3% low *(field, E47)*: a noise floor, not an effect-size threshold.

## 7. Bytes in flight: measured scope, not a universal predictor

A 1 GiB `.cg` read stream (GB/s, 3 runs), each lane keeping U 16-byte loads in flight: a `[U, 2048]` bf16 tile per step
in `BlockedLayout([1, 8], [1, 64], [1, 4], [1, 0])` covers all U rows of the tile once, so each lane issues U `buffer_load_dwordx4`
before the first wait; a loop using each load before the next keeps one (code: [../tuning.md](../tuning.md) §4):

| Programs | 1 warp, U=1 | 4 warps, U=1 | 8 warps, U=1 | 4 warps, U=2 | 4 warps, U=4 | 4 warps, U=8 |
|---|---|---|---|---|---|---|
| 256 | 654–658 | 2243–2294 | 3889–3929 | 3871–3917 | 5697–5756 | 7129–7167 |
| 512 | 1205–1208 | 3871–3879 | 6206–6229 | 6287–6333 | 7079–7126 | 7073–7107 |
| 1024 | 2159–2202 | 6101–6115 | 6770–6803 | 6765–6851 | 6900–6962 | 6354–6387 |
| 2048 | 3725–3750 | 6674–6684 | 6728–6794 | 6497–6521 | 6591–6628 | 6981–7039 |

*(field, E14)* In the table's U ≤ 8 configurations, equal nominal bytes in flight had rates within 10.4% (N=3). Read as about 1/2/4/≥8 MiB giving 2.2–2.3/3.7–3.9/5.7–6.3/6.4–7.2 TB/s, these are estimates for this implementation, not a bandwidth law. Its U=8 used 140 VGPRs; a separately measured U=16 small-grid arm ran at about half rate.

**Nominal P×W×U KiB is not sufficient.** P programs × W warps × U independent 16-byte loads per lane describes bytes in flight, not a rate. A second implementation, a 1 GiB bf16 in-lane-sum stream (N=1, 12 randomized blocks) *(field, E42)*, put all four 8 MiB configurations (512×1×16, 256×2×16, 256×8×4, 2048×4×1) within 10% of their level median. At 3/4/6 MiB, 384×1×8, 256×1×16 and 384×1×16 reached only 0.888/0.760/0.861× their level medians. Against same-level four-warp configurations 384×4×2/256×4×4/384×4×4, paired geometric-mean rate ratios were 0.867846/0.744441/0.854286 (95% CIs 0.866498–0.869197/0.743706–0.745176/0.853928–0.854644); these CIs are for the paired configurations, not for ratios to a level median. At equal 3 MiB, 96×4×8 and 192×1×16 reached 0.800×/0.678× of 384×4×2 (95% CIs 0.7986–0.8021/0.6775–0.6793), not a P/256 scaling law. Every build issued U `dwordx4` loads before waiting, with no scratch; whether register pressure or CU residency causes the gaps was not isolated. Do not pool this implementation with the table above, or carry either to an actual small-call kernel without measuring it.

**Split × depth does not determine call time at 1–256 tokens or rows.** The Qwen2-MoE router at 256 tokens, splits × K tiles ahead: equal products 4×2 (the kernel's own), 8×1, 2×4 took 1.000×, 1.033×, 1.260× the kernel's own time; doubled products 4×4 and 8×2 took 0.970× and 1.032× (N=1, 4 rounds, all arms accurate). Moving splits both ways lost at router 256 tokens (4→2 1.33–1.34×, 4→8 1.02–1.04×; N=8) and 128 tokens (1.22×/1.03×; N=1), Qwen3.5's gated attention at 32 rows (16→8 KV splits, head tile changed too, 1.58×; 16→32 1.05–1.06×; N=7/4), DS256 at 2 tokens (4→1 1.08×, 4→8 1.05×; N=1; 4→2 1.01×, slower in 119 of 120 rounds *(field, E48)*), and at 1 token (4→2 1.02×, 4→8 1.03×; N=1). Counterexample: the router at 64 tokens 8→16 0.993× with pairs both ways (N=1); no universal optimum *(field, E38)*. How to choose them: [../tuning.md](../tuning.md) §3, §4.

## 8. Getting a tile into LDS

On a stream of 64×64 bf16 tiles, async copy (one or two tiles in flight), registers → `.store` → `.load`, and registers
only ran within 1.4–2.5% of each other at 1024 programs and 2.1–4.8% at 2048 over 4 runs, 5.49–6.01 TB/s, 0.99–1.04× a
plain `gl.load` stream of the same tiles (5.48–5.82 TB/s) *(field, E16)*, with 4-wave programs, 16–32 waves per CU; with fewer programs or smaller tiles async copies fell further behind register loads ([../attention.md](../attention.md) §7); what async copies buy in a kernel (in-flight loads behind MFMA,
no registers held) was not isolated. `allocate_shared_memory(..., value)` allocates and stores in one op (S17, S26).

**Direct to LDS.** `buffer_load_to_shared` lowers to one `buffer_load_dwordx4 … offen lds` per 16 bytes per lane (S5,
S25); `commit_group()` closes a group (an `asyncmark`), and `wait_group(n)` waits until at most `n` committed groups of
this wave are outstanding (`wait.asyncmark n`, the `vmcnt` from LLVM; S10). A complete copy (compiles with Triton 3.8.0
for gfx950):

```python
@gluon.jit
def copy_tile(x_ptr, y_ptr, stride, BM: gl.constexpr, BK: gl.constexpr):
    copy: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])  # 16 B per lane, 8 lanes per 128 B row
    rows = gl.arange(0, BM, gl.SliceLayout(1, copy))
    cols = gl.arange(0, BK, gl.SliceLayout(0, copy))
    offs = rows[:, None] * stride + cols[None, :]  # stride marked divisible by 16 at launch ("D")
    tile = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(tile, x_ptr, offs)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(0)  # the only thing that waits for the copy
    gl.amd.cdna4.buffer_store(tile.load(copy), y_ptr, offs)
```

At BM = BK = 64 with 4 warps each lane issues 2 `buffer_load_dwordx4 … lds`, then `s_waitcnt vmcnt(0)`, `s_barrier`
and 2 `ds_read_b128` *(field, E28)*. Rules:

- **Lanes write LDS in order**: lane i of an x4 instruction writes byte 16·i of its 1 KiB
  ([ISA §3](../../hardware/gfx950/memory_and_sync.md)), so the copy layout hands each lane the 16 bytes landing there
  (above, 8 lanes per 128-byte row; for a swizzled or padded destination see §9), 128 or 32 bits per lane (S20).
- **Wait before every read of a copied slot.** Neither the `s_barrier` nor `.load` waits for a copy, `wait_group` does:
  without it the kernel above read stale LDS behind the `s_barrier` *(field, E28)*.
- **Commit every copy before waiting on it.** `wait_group` counts committed groups only: an uncommitted copy was not
  waited for, even by `wait_group(0)`, although the docstring says it is (S25) *(field, E28)*.
- **Copies complete in order with other vector-memory accesses** (S25): a scale load issued after a copy compiled to
  `vmcnt(1)` before its use, which retires the copy, and issued before it, `vmcnt(3)`, leaving the copy in flight
  *(field, E28)*. Loading scales before the DMA prologue is an A/B candidate, not a measured performance win: the
  timed test kernel gained nothing *(field, E28)*. The n = 7168 fp8 GEMM loads its scales early, not tested by that kernel.
- No `other` on copies (§4). DMA costs and prefetch: [../../gemm/datapath.md](../../gemm/datapath.md) §5, §7.

**Staging choice at 8 tokens.** The 8-token Qwen2-MoE router async-copies its hidden states into LDS; its other
path, `gl.load` of both operands and their conversion through LDS *(field, E18)*, ran the same: 8.45 → 8.45 µs (speedup
1.186 → 1.178, 4 runs per arm) *(field, E2)*; the bundle made no difference at this size, the staging alone untested.

## 9. LDS destinations: swizzled, padded, transposed

**Swizzled: pre-swizzle the global side.** A DMA cannot scatter lanes inside LDS on CDNA4, so a copy into a
`SwizzledSharedLayout` with `max_phase > 1` makes the compiler swizzle the *source*, one `ds_bpermute_b32` per copy
instruction (S10) *(field, E19)*. In the library the DSA kernels, two gated-attention kernels and 10 fp8 GEMMs XOR the
global offsets instead and copy into an unswizzled view *(field, E19)*. `SwizzledSharedLayout(vec, per_phase, max_phase, order)` stores
element (row, c) at column `c ^ ((row // per_phase) % max_phase * vec % width)` of its row (S47); XOR being its own
inverse, the LDS column `c` a lane fills takes global column `c ^ ((row // per_phase) % max_phase * vec)` when
`max_phase × vec` fits in the row *(field, E19)*. Here, in Qwen3.5's gated attention at 256 rows, K and V are
`SwizzledSharedLayout(16, 1, 8, [1, 0])` (`QK_WIDTH` is 8, so `key_vec` is 16), and the 128-row kernel XORs a `vec = 32` K buffer with `* 32`; a wrong
multiplier compiles and loads the wrong chunks:

```python
# Apply each consumer's swizzle in global source offsets so the LDS
# transfers themselves have a linear, lane-local destination layout.
linear_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
key_copy = key_shared._reinterpret(layout=linear_layout)
value_copy = value_shared._reinterpret(layout=linear_layout)
key_vec: gl.constexpr = 32 if QK_WIDTH == 16 else 16
key_offsets = slots[:, None] * 256 + (
    dims[None, :] ^ ((tokens[:, None] % 8) * key_vec)
)
value_offsets = slots[:, None] * 256 + (
    dims[None, :] ^ ((tokens[:, None] % 8) * 16)
)
# XOR only permutes aligned 16-byte runs; each run remains contiguous.
key_offsets = gl.max_contiguous(gl.multiple_of(key_offsets, [1, 16]), [1, 16])
value_offsets = gl.max_contiguous(gl.multiple_of(value_offsets, [1, 16]), [1, 16])
```

A test kernel with this `[64, 256]` fp8 copy had 4 `ds_bpermute_b32` and 76 instructions into the swizzled buffer, none
and 56 pre-swizzled *(field, E19)*. Why swizzle: read as a 16×16×32 MFMA operand, LDS-bound, a row-major 64×64 bf16 tile ran at 0.46–0.59 of the XOR-swizzled rate; padding 8 elements after each 64-element row kept B within 3% of that rate and A at 0.86–0.96 (0.82–0.84 when only the padded A loop was not unrolled) *(field, E20, E52)*. With 32×32×16 operands plain A ran at 0.28 and padded A at 1.34 of the XOR rate: measure the layout for the instruction shape in use *(field, E52)*.
In the n = 3072 fp8 GEMM at m = 16 XOR addressing changed nothing, 4.25 → 4.26 µs
*(field, E15)*; whether its reads conflicted without it, or what bounds that kernel, was not measured. Time conflicts:
`gl.bank_conflicts` aborted the process on AMD *(field, E20)*. Choosing the swizzle: [layouts.md](layouts.md) §4, §5;
[../../patterns/lds_bank_conflicts.md](../../patterns/lds_bank_conflicts.md).

**Padded.** `gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)`
returns a `PaddedSharedLayout` (bf16 A, 16×16×32 MFMA v4, k_width 8, `[32, 64]`: `[[512, 16]]`, at 1–16 waves) or `None` (`[8, 64]`). In a 900-combination helper sweep (bf16/fp8 e4m3/uint8, A and both B orientations, 16-/32-wide MFMA, 1–16 waves, 16–256 rows × 64/128 columns), 130 returned `None`, independent of waves: A with rows ≤ MFMA M, and fp8/uint8 B with K=16 at 16×16×128. Enough bytes per wave did not guarantee a layout *(field, E39)*; packed fp4 passes `gl.uint8` (S3). DMA width is capped at `min_interval / 64` elements per lane (S14); 45 identity-offset padded `[64, 64]`/4-wave configurations reached the widest legal 4-/16-byte width under that cap, not arbitrary widths; requests above it failed *(field, E39)*. Split `offset_bases`: first log2(16 / element bytes) bases for one lane's 16 bytes, next 6 for lanes, next log2(num_warps) for waves, rest for registers (the Qwen2-MoE router does so for bf16, the n = 3072 fp8 GEMM for fp8). All 603 helper-layout combinations with tile bytes ≥ 1024×num_warps compiled to exactly `tile_bytes/(1024×num_warps)` `buffer_load_dwordx4 … lds`, no `ds_bpermute`/`ds_write`, and read back bitwise on MI355X (N=1); 167 smaller helper-layout tiles failed to compile *(field, E39)*. Code, and `None`: [layouts.md](layouts.md) §4, §6.4.

**Transposed reads.** How B is stored decides the LDS read. Compiled, bf16 `[64, 64]`, `instr_shape [16, 16, 32]`: A
`[M, K]` (K contiguous) → `ds_read_b128`; B stored `[K, N]` (N contiguous), read as `DotOperandLayout(1, …)` → 8
`ds_read_b64_tr_b16`, gfx950's transposing read; B stored `[N, K]`, read through `.permute((1, 0))` → 4 `ds_read_b128`;
fp8 B `[K, N]` → `ds_read_b64_tr_b8` *(field, E22)* (S29). Both work: K-contiguous B takes half the read instructions,
and LDS-bound the transposing read cost no extra time, 510–515 against A's 529–534 µs in 4 runs *(field, E20)*; the
gated attention decode reads its fp8 V (`[tokens, dims]`) with the transposing read. Lane maps, cost per MAC:
[../../gemm/datapath.md](../../gemm/datapath.md) §9, [../../attention/lds.md](../../attention/lds.md) §1.

## 10. LDS allocation, views, reuse, budget

Views on a `shared_memory_descriptor` (S17, S26): `.index(i)` (an int32, runtime allowed) a slot of the leading
dimension; `.slice(start, length, dim=0)` (Python ints) a subview Membar tracks; `.permute`, `.reshape`,
`._reinterpret(dtype, shape, layout)` (the same bytes); `.load`, `.store`, `.gather`, `.scatter`; `._keep_alive()`, a
use that keeps the buffer live to that point.

**Rings.** Build a static ring as a tuple of separate allocations, `bufs = ()` then `bufs += (...,)` per slot (a
list with `.append` failed to compile *(field, E23)*), indexed in `gl.static_range`, which the frontend unrolls with a
constant loop variable ([../tuning.md](../tuning.md) §10); for a runtime slot use one `[SLOTS, ...]` allocation and
`.index(i)`, in the rank-2 layout of one slot ([layouts.md](layouts.md) §4). Membar tells `.slice` subviews of one
allocation apart, not its `.index` slots (S30): copies into `.index(0)` and `.index(1)` of one allocation
got an `s_barrier` between them, into two `.slice` halves none *(field, E23)*.

**Reuse.** `_reinterpret` gives a dead buffer a new type and shape. Allocate late: the allocator overlaps buffers whose
live ranges do not meet, and the 64-head DSA kernel allocates its probability tile only after its reduction buffers
die. Liveness is use-based (S32): `_keep_alive()` after a later conversion extended a buffer's range, right after its
last read it changed nothing, and the n = 3072 fp8 GEMM's on its last ring slots changed neither its LLVM IR nor its
LDS *(field, E23)*. A `convert_layout`'s LDS scratch laid over a dead buffer costs a barrier: keeping the buffer alive
past the conversion removed one `s_barrier` (3 → 2) for twice the LDS (16 → 32 KiB) *(field, E23)*.

**Budget.** 160 KiB per CU on CDNA4 (S20), and per workgroup: a 167936-byte kernel compiled, then failed at launch; read
`shared` in the metadata first ([../debugging.md](../debugging.md) §2, §4). LDS is allocated in 1280-byte blocks
([../../hardware/gfx950/memory_and_sync.md](../../hardware/gfx950/memory_and_sync.md) §4), so workgroups per CU ≤
⌊163840 / (⌈LDS per workgroup / 1280⌉ · 1280)⌋ (estimated); ⌊163840 / LDS⌋ overcounts, giving 10 instead of 9 at
15361–16384 bytes. It gives 3 for the one-wave n = 3072 fp8 GEMM at m = 16 (49152 bytes, 39 blocks) whatever
`waves_per_eu` says (a test grid of 49152-byte workgroups took flat time up to 3 per CU and 2.0× at 4 to 6, at
`waves_per_eu` 2 and 8), and 1 for the 64-head DSA attention kernel (163840 at 24 tokens, 156160 at 12 and 16, 157696
at 32) *(field, E24)*.

## 11. Synchronisation

**What Membar inserts** (local barriers only, S30): a barrier between conflicting LDS accesses (read-after-write,
write-after-read, write-after-write) of `.store`, `.load`, `allocate_shared_memory(value)` and async copies; and one
right after a `wait_group` unless a sync point follows anyway, since `wait_group` alone does not synchronise the CTA
(S30, S34). Compiled, 4 waves, 2 copies per group: `wait_group(1)` → `s_waitcnt vmcnt(2)`, `s_barrier`, still one with a
`gl.barrier()` after it *(field, E23)*. For a copy, that barrier orders the instructions but does not wait for the data:
a `.load` with no `wait_group` before it compiled to `s_barrier` and `ds_read_b128` with no `vmcnt` wait *(field, E28)*.
Nothing for global memory, and nothing between an async copy and a `load_shared_relaxed` of the same buffer.

**`load_shared_relaxed`: the write-after-read is yours.** It tags the load `ttg.amdg.syncedViaAsyncWait` (S25); the AMD
filter then drops the dependency between that load and any async copy of the same buffer, in either order (its header
warns this is wrong for a single buffer), and the lowering puts the relaxed `ds_read` in a no-alias scope against async
copies, so LLVM may schedule a refill of the slot ahead of it (S35). Compiled, a relaxed read and then a refill of the
same slot, no barrier, issued the refill first, with 4 waves and with 1; a `gl.barrier()` between them restored the
order *(field, E23)*. What it buys: Membar treats the `.index()` slots of one allocation as one buffer (§10), so a plain
read of one slot got a barrier before a copy into another, 3 `s_barrier` against 2 relaxed; with separate allocations,
or a `gl.barrier()` of your own before each refill, the two compiled the same, waits included *(field, E30)*. So read
with `.load` by default and relaxed only for the `.index()` slots of one allocation, as the n = 7168 fp8 GEMM does at
M = 32 (separate allocations and `.load` at M = 64); then order each refill after the last read of its slot: a
`gl.barrier()` or ring slack (§12), and in a one-wave kernel, where `gl.barrier()` only orders the compiler, an explicit
retire of the reads, as the one-warp router in MM64-FP8 does with an inline `s_waitcnt lgkmcnt(0)` and a
`~{memory}` clobber between its relaxed reads and the refill of the same slot (code: [wave_ops.md](wave_ops.md) §8).
The ISA does not say whether a DMA's LDS write can overtake an earlier `ds_read` of the same wave; on MI355X it did, in one-wave rings with slow, bank-conflicted reads and no `lgkmcnt` wait between ([../gemm.md](../gemm.md) §5).

**`gl.barrier()`** is a `ttg.barrier` over every address space (S17, S37), lowered to a release fence, `s_barrier` and
an acquire fence (S29); with 4 waves it compiled to `s_waitcnt lgkmcnt(0)` and `s_barrier`, no `vmcnt` wait, so an async
copy or a `global_load` before it stays in flight, and with 1 wave to no `s_barrier` (flat work-group size `1,64`, S1)
*(field, E23)*. A cross-wave exchange in rounds avoids the write-after-read barrier with a constant `.slice` per round:
4 rounds of a 4-wave exchange compiled to 4 `s_barrier` with a slice per round and 7 with one reused region
*(field, E23)*, for one region of LDS per round; MM64-FP4's top-k does this: in round `slot`,
`values_shared.slice(slot * 4, 4).store(local_value)` into a `(TOPK * 4,)` buffer, `gl.barrier()`, then
`async_copy.load_shared_relaxed(values_shared.slice(slot * 4, 4), layout)`.

**Global memory inside a CTA.** Membar ignores it, leaving no `s_barrier` between a `global_load` and a `global_store`
of the same data *(field, E23)*: waves that overwrite what others read need their own `gl.barrier()`.
In a delayed-reader probe (1024 programs, 2000 launches per arm, N=2), no barrier gave 1911–1947 wrong launches of 2000 at 4 waves and 2000 at 8; `gl.barrier()` gave 0 at each wave count (its ISA gained 1 `s_barrier`) *(field, E40)*. Qwen3-Next prefill's conv-history update has `if NW > 1: gl.barrier()`. A `gl.store` of a layout replicated across waves is issued by one thread per element (S10). Across workgroups, the library's kernels order producer and consumer by kernel boundaries. **Raw `vmcnt` waits:** DSA waits for part of a commit group with inline `s_waitcnt vmcnt(N)`, counts of the copy instructions per panel that go stale silently when the instructions after a copy change, where `wait_group` recounts *(field, E23)*; prefer one commit group per panel ([wave_ops.md](wave_ops.md); speed not compared).

## 12. Software pipelining by hand

**One slot per K group.** The n = 3072 fp8 GEMM at m = 16 keeps `PREFETCH` K groups ahead per wave (its prefetch depth), one LDS slot for each of its 12 groups, none reused. Count waits in committed groups by replaying the issue order (§8): with `PREFETCH` groups committed ahead of group g, `wait_group(PREFETCH - 1)` lands g. At m = 16 it reads the next 2 groups into registers early, so depth 5 leaves three groups in flight at each steady wait; depth 3 (one left) lowered the reported speedup by 12.9% (1.475 → 1.285), while time increased 14.6%, 4.23 → 4.85 µs; 7 gave 4.294 → 4.293 µs *(field, E15)*. Deeper paid little: on the same GEMM at m = 64, depth 4 → 6 took 0.996× the call time at its own `waves_per_eu=6` and 0.994× at `waves_per_eu=0`, over three sessions *(field, E49)*. Shallower lost in both settings, 4 → 2 1.77–1.78×, as did the same GEMM at m = 32, 5 → 3 1.15–1.19× (also in a build without spills), where 5 → 7 was unresolved *(field, E44)*. Depth per size, and the single buffer of m ≥ 1024: [../gemm.md](../gemm.md) §5.

**A ring of reused slots.** A refill must not overtake the reads of its slot. With plain `.load` Membar sees to it and the ring size only decides the cost: with `SLOTS = PREFETCH` it put an `s_barrier` between each read and its slot's refill, 10 against 6 over 6 stages, while with `SLOTS = PREFETCH + 1` the barrier after the next `wait_group` already separates them *(field, E30)*, as the n = 7168 fp8 GEMM's ring has. With relaxed reads, count the barriers between a slot's last read and its refill yourself. A refill issued after the step's reads needs one spare slot (without it, a refill copy was issued before its slot's reads *(field, E30)*); one issued at the top of the step, before its wait, needs two, `DEPTH = AHEAD + 2`, as the Qwen2-MoE router does at 256 tokens with `AHEAD` 2 (with 1: 10.01 → 11.28 µs *(field, E15)*), or a `gl.barrier()` before the refill, which its 128-token ring issues under `if DEPTH <= AHEAD + 1 and stage >= DEPTH - AHEAD:`. The 256-token ring:

```python
for stage in gl.static_range(-AHEAD, STAGES):
    if stage + AHEAD < STAGES:
        _prefetch_stage(
            a_buffers[(stage + AHEAD) % DEPTH],
            b_buffers[(stage + AHEAD) % DEPTH],
            # ... (the two tensors and their offsets for stage + AHEAD)
        )
    if stage >= 0:
        gl.amd.cdna4.async_copy.wait_group(min(AHEAD, STAGES - stage - 1))
        gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffers[stage % DEPTH], a_dot)
```

**A rolled ring for long K** keeps the code small ([../gemm.md](../gemm.md) §3). `wait_group(n)` takes a Python int (S4): `n` from an argument or a `range` variable failed at trace with a `CompilationError`, "create_async_wait_group(): incompatible function arguments" *(field, E34)*, so a rolled body waits a constant. The 2232-token Qwen2-MoE router is written so: one `[STAGES, …]` allocation per operand (§10); a static prologue of `STAGES - 1` committed groups; a `range` over blocks of `gl.static_range(UNROLL)` steps, each refilling slot `(tile + STAGES - 1) % STAGES`, committing, waiting `STAGES - 2`, calling `gl.barrier()`, reading slot `(tile + 1) % STAGES` relaxed and multiplying the operands read the step before; the leftover steps, unrolled (there all 29); a static drain waiting `STAGES - 3 - tail`. Written so, an fp8 block-scaled GEMM at M = 32, N = 4096, K = 7168 (2 waves, 16 columns, 6 slots, one step per trip) compiled to 325 instructions against 1417 unrolled, 56 VGPRs, 44 KiB of LDS and one `vmcnt(12)` wait per step, its scales from LDS or `s_load`s (§13); a weight scale `gl.load`ed at its group added `vmcnt(0)` before the fold, and a tuple of scales loaded with their groups' copies (none grows in a `range` loop) `vmcnt(3)`, unless unrolled by its length *(field, E34)*.

**Register look-ahead.** Operands loaded straight into the operand layout (§13) need no LDS. MM64-FP8 keeps up to three K steps' operand tuples (A, B and both scales) in flight and rotates them; the rotation is loop-carried, so the loop stays a rolled `range` (its three-tuple branch, which the up projection takes at 8 tokens):

```python
for step in range(groups - 3):
    na, nb, nsa, nsb = _load_fp8_tile(
        # ... (X, XS, W, WS and the tile's row, expert and column indices)
        split * groups + step + 3,
        # ... (K, WN, WK, BUFFER_LOAD, WEIGHT_CACHE)
    )
    acc = _accumulate_fp8(a, b, sa, sb, acc)
    a, b, sa, sb = (a1, b1, sa1, sb1)
    a1, b1, sa1, sb1 = (a2, b2, sa2, sb2)
    a2, b2, sa2, sb2 = (na, nb, nsa, nsb)
```

The depth is set per token count (up projection: 2 tuples at 1, 4 and 16 tokens, 1 at 2, 3 at 8). No timing isolates it;
in its fp8 projection kernel the second tuple cost 16–24 VGPRs, the third none *(field, E34)*.

## 13. Weight and scale streaming

**Preshuffled weights straight into the operand layout.** The kernels compute the offset of 16×16-preshuffled
weights (aiter's 16-row tiles of 16-byte vectors; offset: [mfma.md](mfma.md) §7) and build its index vectors in slices
of the B operand layout, so the loaded tensor is the MFMA operand, no LDS: 16 bytes per lane, 1 KiB per wave (derived;
code and DMA with a view: [layouts.md](layouts.md) §6.3). In MM64-FP4 at 4096 tokens, with waves splitting N,
shared A goes through LDS: the helper's padded layout at `BN == 128`, a conversion otherwise; B stays wave-private;
not a universal N-split requirement or a timed LDS comparison.

**Scales.** Load packed bytes as words through a `uint32` pointer and shift (a masked word reads `0x7F7F7F7F`, four unit
e8m0 scales, at §4's select cost), before the copies they would wait behind (§8). An `s_load` counts in `lgkmcnt`,
not `vmcnt` ([ISA §1](../../hardware/gfx950/memory_and_sync.md)), but uniform `gl.load`s compiled to `s_load` ahead of
any copy or store *(field, E29)* and to `global_load_dword` after an async copy or a `gl.store` in a 4-trip loop that LLVM unrolled *(field, E34)*. Do not infer a rolled loop's addressing from textual order: in 18 builds at 1/4 waves, a runtime rolled body containing copy/store used 18 static `global_load_dword` and 0 data `s_load` even when the load appeared first; a load-only rolled loop used 16–18 data `s_load`, as did the constant-address-space pointer; loads before the loop stayed scalar (3 compilations, not timed) *(field, E41)*. The n = 3072 fp8 GEMM at m = 32 uses this constant-address-space pointer: `gl.inline_asm_elementwise("", constraints="=s,0", args=[WS.to(gl.uint64)], dtype=gl.uint64, is_pure=True, pack=1)` puts the address in an SGPR, then `.to(gl.pointer_type(gl.float32, 4))`; the cast alone fails to compile *(field, E34)*. Stage a per-group vector's table in LDS, then one `gl.barrier()` (else Membar put one inside the loop *(field, E34)*), and `.gather` group `step` (S17), one 32-bit LDS word per trip *(field, E34)*: MM64-FP8's rolled ring reads `scale_smem.gather(step * BM + gl.arange(0, BM, gl.SliceLayout(1, mma)), 0)` from its group-major flattened table. Not `.index(step)`: a 2-D table stored from registers has a rank-2 layout ([layouts.md](layouts.md) §4) *(field, E34)*.

## 14. XCD and L2 locality

gfx950 deals workgroups round-robin over 8 XCDs, each with its own L2 *(field, E25)*: on the default stream program `i` ran on XCD `i % 8` *(field, E25)*; on a new stream, as the ruler's graph replays run on, it ran on XCD `(i + c) % 8`, c = 5–7 depending on how many streams the process had created, and a graph replay took its stream's c, so programs congruent mod 8 still share an XCD but absolute XCD numbers shift *(field, E46)*
([../../gemm/schedule_and_locality.md §2](../../gemm/schedule_and_locality.md)). After a 1 GiB read flush, as the ruler's, 8
programs reading the same chunk finished 1.8–2.1× sooner with program ids congruent mod 8 than consecutive (3 runs, 64
KB: 21.3–21.5 → 11.8–12.0 µs; 512 KB: 156.3–159.0 → 74.4–74.6 µs) *(field, E25)*. Across calls the flush removes that
reuse: a second call re-reading the first call's 16/64/128 MB buffer behind that flush took 0.997–1.009× the time of
reading a previously unread buffer (N=2) *(field, E37)*, so the reuse to exploit is within one call, and these probes do
not show which tensor sizes survive a flush. Remap when programs of a call share data, as a bijection for every launched
grid (the remap of a bf16 GEMM at N 2624 × K 6144 was checked for grids of 1–699 programs), such as the XCD remap of
[../gemm/small_m_gemm.md](../gemm/small_m_gemm.md) §3, which compiled in Gluon *(field, E25, E33)*. The kernels:

- Above 4096 tokens, DS256's router walks `GROUP = 8` row tiles per column, so the 4 column tiles
  of one row tile share an XCD (derived from the code below); below that it walks rows within a column,
  `(pid % ROWS, pid // ROWS)`.
- The n = 3072 fp8 GEMM at m = 4096 walks a group of M tiles per N tile; the n = 7168 GEMM's m = 64 permutation changes
  no reuse, one reader per weight tile, all for the activation panel (inferred; [../tuning.md](../tuning.md) §12), nor
  the time *(field, E25)*.

The router's grouped walk:

```python
@gluon.jit
def _grouped_tile(pid, ROWS: gl.constexpr, COLS: gl.constexpr, GROUP: gl.constexpr):
    gl.static_assert(ROWS % GROUP == 0)
    row = pid // (GROUP * COLS) * GROUP + pid % GROUP  # GROUP consecutive ids: GROUP row tiles of one column
    column = pid // GROUP % COLS  # ids GROUP apart: the next column of the same row tile
    return (row, column)
```

## 15. Epilogue stores

Round first, move through LDS to 16 contiguous bytes per lane, store with the buffer op, as the n = 3072 fp8 GEMM for m = 128 does (its `BLOCK_M` is 32):

```python
result = acc.to(gl.bfloat16)
if BLOCK_M == 32:
    exchange = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], gl.SwizzledSharedLayout(8, 2, 8, [1, 0]))
    exchange.store(result)
    result = exchange.load(out_layout)
# ... (the `else:` arm, the source's 64-row exchange, which m = 128 does not take; then `rm`, `rn` and `output_base`)
gl.amd.cdna4.buffer_store(result, output_base, rm[:, None] * 3072 + rn[None, :], cache=".wt")
```

`out_layout` is `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])`; rounding first halves the exchange's LDS bytes; at
`BLOCK_M = 64` the exchange is a `SharedLinearLayout` whose bases XOR rows into the 16-byte chunk. To reach this 16-byte-per-lane store width, exchange the result
when the accumulator's layout directly stores less than 16 bytes per lane: from `AMDMFMALayout(4, [16, 16, 32], transposed=True)`
fp32 stored 16 bytes per lane directly and bf16 8, from `transposed=False` 4 and 2. A `gl.convert_layout` to
`out_layout` makes the same exchange in a buffer the compiler lays out, with 2-byte LDS accesses from a
`transposed=False` source *(field, E33)*; write it yourself to choose that buffer's layout. Neither was timed against
the direct store ([layouts.md](layouts.md) §6.6), nor was this `.wt`, which cost a `buffer_store` write stream at most 2.3 % (§6) *(field, E9, E51)*.

## Evidence

Triton paths are `triton-lang/triton@c01b677`. "The ruler", its timer and its speedups are [../tuning.md](../tuning.md) §1's; "the library" is [../kernels/kernel_library.md](../kernels/kernel_library.md) §1's, 154 Triton and Gluon kernels when its sources were counted; MM64-FP8, MM256-FP8, MM64-FP4, MM256-FP4, DS256 and DS512 are MoE layer labels ([../moe.md](../moe.md) §1).

- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` 327-348 (unroll 337, warp pipeline 339); `make_ttgir` 260-325 (coalesce 269, `num_stages` 288-289, buffer-op conversion 303-312); Gluon takes `gluon_to_ttgir` 584-589; "S" → `tt.pointer_range` 211-240; async copy on for gfx950 32-33; `make_llir` warp-pipeline conversion 357; flat work-group size 460
- S2 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:111 Membar with `membarFilter` in the lowering (111-116)
- S3 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 re-exports the buffer ops; `compute_efficient_padded_shared_layout` 121-163 (`None` 139-143)
- S4 triton-lang/triton@c01b677:python/src/gluon_ir.cc:1097 `create_buffer_load(type, ptr, offsets, mask, other, cache)`, `create_buffer_store` 1104; commit and wait builders 727-736, the wait taking an `int num` (733)
- S5 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` = INT_MAX − 1; select unless `other` is a zero constant 121-122; DMA via async LDS intrinsic 126-149; 32-bit byte offset, masked lanes INT_MAX + 1 280-283
- S8 triton-lang/triton@c01b677:python/triton/language/semantic.py:922 cache strings (922-950); `store` casts to the pointee type 1243
- S9 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp:81 constant-true mask → straight load, else a branch (81-109); stores the same (151-171)
- S10 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:263 mask alignment caps `vec`; `other` packed per vector 697-705; swizzled `dest` → source swizzling 788-803; DMA `other` → branch + LDS store 840-874; redundant-thread store predicate 1646-1657; `async_wait` → `wait.asyncmark n` 2412-2431; commit → `asyncmark` 2503-2505
- S14 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:716 contiguity = min(per-thread run, offset alignment, base divisibility), vector size 758-763; `canLoadDirectToLDS` 817-891: padded DMA cap 820-831, a provable vector below the per-lane run into LDS 868, unsupported width 879, lanes not writing in order 885; pointer-op flags 290-337; buffer `aux` bits 456-506
- S15 triton-lang/triton@c01b677:lib/Analysis/AxisInfo.cpp:841 contiguous `<` splat: constancy gcd(contiguity, divisibility, divisibility) (841-856); mask alignment = constancy 1446-1455; `//` contiguity 1 unless by 1 418-425; `%` by a constant gcd(contiguity, divisibility, divisor) 488-510; a tensor-condition select gcd of both branches and the condition's constancy 964-967; XOR no contiguity rule 994-1016; a function argument's divisibility 1 unless `tt.divisibility` 1309-1336
- S16 triton-lang/triton@c01b677:python/triton/backends/compiler.py:78 "D" = divisibility 16, for ints with `arg % 16 == 0` (78-86)
- S17 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:128 `tl` re-exports (`load` 128, `max_contiguous` 131, `multiple_of` 135, `store` 144); descriptor `gather` 358, `slice` 446, `index` 464, `permute` 478, `reshape` 492, `_reinterpret` 507, `_keep_alive` 527; `allocate_shared_memory` 613; `barrier` 703
- S18 triton-lang/triton@c01b677:python/triton/language/core.py:3308 `multiple_of`, `max_contiguous` 3335: hints; `load` 2479, `store` 2560
- S20 triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:202 CDNA4 DMA widths 128 and 32 bits, 8, 16 and 96 disabled (202-204); LDS 160 KiB 144-145; transpose-read parameters 161-185
- S25 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared` (in-order completion 77-80; 128/32 bits, coalesced, warp-local swizzle 91-94); `global_load_to_shared` 16-29, its conditions 31-39; `wait_group` docstring 137-141; relaxed-load attribute 164-168
- S26 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:252 `allocate_shared` with a value is one `local_alloc` (252-262); `slice` needs ints 411-414; `index` takes an int32 425-428
- S29 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:476 CDNA4 `ds_read_tr16_b64`/`ds_read_tr8_b64` (476-486); barrier: release fence, `s_barrier`, acquire fence 720-736
- S30 triton-lang/triton@c01b677:lib/Analysis/Membar.cpp:23 only `memdesc_subslice` offsets tracked, else overlap (23-49); barriers are `AddrSpace::Local` 241-245; after a wait unless a sync point follows 269-298; on conflicts 381-385
- S32 triton-lang/triton@c01b677:lib/Analysis/Allocation.cpp:419 live ranges from MLIR `Liveness` over all uses (419-436)
- S34 triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47 `async_wait` "without synchronising CTA execution" (47-55)
- S35 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11 async copy ↔ synced local load filter (11-53); single-buffer warning in MembarUtility.h:9-28; synced local loads put in a no-alias scope against the async copies in AsyncUtility.cpp:117-137, copies tagged in LoadStoreOpToLLVM.cpp:854-855, 864-865
- S37 triton-lang/triton@c01b677:python/src/ir.cc:1853 `create_barrier` → `BarrierOp(AddrSpace::All)` (1853-1856)
- S47 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:83 swizzled shared layout: row basis `(vec * ((row / perPhase) % maxPhase)) % numCols` (76-84)
- E2 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto), 4 alternated A/B rounds] Qwen2-MoE router at 8 tokens with its async-copy condition `ROWS <= 8` changed to `ROWS <= 0` (hidden states by `gl.load` and a conversion through LDS instead of an async copy): 8.447 → 8.450 µs (time ratio 1.000), speedup 1.186 → 1.178 (B/A 0.993); B accurate, max abs error 0.030 of 11.7 (Qwen2-MoE router at 8 tokens)
- E4 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the 2 GiB-range runs N=2), GPU runs with values read back] an unmasked `buffer_load` of offsets 64–79 of a 64-element view returned 64.0–79.0, and the ISA held `s_mov_b32 s2, 0x7ffffffe`; fp32 loads at byte offsets 2^31 to 2^32 − 4 (three 64-lane windows) returned 0 and stores there left the tensor unchanged, with no fault; an offset reaching 2^32 wrapped onto the wrong element, for loads and stores; 1-byte loads at byte 2^31 − 3 and below read their data, at 2^31 − 2, 2^31 − 1, 2^31, 2^31 + 1, 2^31 + 4096 and 2^32 − 1 read 0; 1 KiB at byte 2^31 + 4096 of a 3 GiB tensor read 0 through `buffer_load` from the base, and its data through a scalar base moved 2 GiB (64-bit) or through `gl.load`; `global_load_to_shared` copied the 4 KiB at byte 2^31 + 8192 exactly where `buffer_load_to_shared` read 0; an int32 index product of 2^31 + 4096 read the wrapped element; of the host gates against 2^31 in the library (33 kernels), 46 take the buffer path at every size their kernel is built for and none takes the other path, 18 were not evaluated (test kernels on buffers up to 3 GiB; the gates counted in the library's sources)
- E6 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] fp32 `buffer_store` through a `*bf16` pointer fails the verifier ("failed to verify that value element type matches the pointed type of ptr"), while `.to(gl.bfloat16)` first, or `gl.store`, compiles; one 16-byte fp32 vector per lane: 1 `v_cndmask` with the mask alone, 5 with `other=0` or `other=1`, no branch; a u8 `[64, 256]` copy, wait, read and store, 4 copies per lane: 47 instructions unmasked, 59 with the row mask, 103 with the mask and `other` (4 `s_cbranch_execz`, 4 `ds_write_b128`); a masked `[64, 64]` bf16 copy: all 4 `dwordx4` loads and stores behind an exec branch through `gl.load`/`gl.store` (6 `s_and_saveexec_b64`, 6 `s_cbranch_exec`), none unmasked, none through buffer ops (2 `v_cndmask`); a row-masked `[64, 16]` fp32 tile: 1 `buffer_load_dwordx4` (test kernels)
- E8 [2026-11, gfx950, Triton 3.8.0, N=2, compile-only] the cache bits of each string on `buffer_*` and `global_*` loads and stores, 20 hints, all as in the §6 table; `.cv`/`.wt` `gl.load`/`gl.store` each followed by `s_waitcnt vmcnt(0)`, the buffer ops not (test kernels)
- E9 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, median of 9 rounds × 10 launches, variants interleaved] read GB/s, `buffer_load`/`gl.load`: none 6009–6013/5993–6000, `.ca` 6010–6019/5986–6005, `.cg` 6802–6823/6785–6803, `.cs` 6796–6837/6771–6804, `.cv` 6013–6018/6008–6016; `.cg` and `.cs` over none 1.130–1.137 on both paths, `.ca` and `.cv` 0.999–1.003, every `.cg` round above every unhinted one; write GB/s, `buffer_store`/`gl.store`: none 5793–5889/5792–5823, `.wb` 5823–5865/5809–5855, `.cg` 5813–5834/5768–5822, `.cs` 5456–5510/5404–5473 (0.928–0.945 of none), `.wt` 5822–5879/5721–5775 (0.998–1.005/0.983–0.996 of none); the `gl.store` `.wt` below none, `.wb` and `.cg` in each run, the buffer `.wt` above `.wb`, `.cg` or none in some; none, `.wb` and `.cg` within 1.003–1.015 of each other per run and path, the fastest of them varying with run and path, and in one run the `gl.store` stores without bits spread 1.51% (1 GiB bf16 stream, 2048 programs × 4 warps, 16 B per lane)
- E11 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, median of 15 eager launches, before each launch nothing, a 1 GiB read or a 1 GiB write] the ruler's flush loads a 1 GiB buffer and stores only to a sums tensor; TB/s none/`.cg` after a 1 GiB read, 3 runs: 4.47–4.50/4.65–4.78 (64 MB), 4.99–5.02/5.46–5.53 (128), 5.44–5.51/5.83–5.89 (192), 5.76–5.79/6.41–6.44 (512), 5.97–6.02/6.61–6.67 (1024), a fourth run 4.43/4.75, 4.99/5.46, 5.34/5.81, 5.74/6.40, 6.00/6.63; `.cg` over none over the 4 runs 1.039–1.074 (64 MB), 1.090–1.104 (128), 1.059–1.088 (192), 1.108–1.115 (512), 1.099–1.117 (1024); after a 1 GiB write none 2.52–4.73, `.cg` 4.70–6.64, `.cg` over none 1.395–2.215; hot 64 MB, `.cg` over none in 3 alternated rounds per run, none timed first in each: 1.101, 1.113, 1.109; 1.122, 1.157, 1.105; 1.116, 1.098, 1.116; 1.054, 1.064, 1.071; in the sweep's single medians, `.cg` hot timed right after the unhinted write-flush launches, 1.122, 1.044, 0.960; an earlier hot probe 0.96 and 0.93 (a read stream of 64 MB–1 GiB, 2048 programs × 4 warps, 16 B per lane)
- E12 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto), 4 alternated A/B rounds per size] MM64-FP4 down-projection weights at 256 tokens, `.cg` off: 104.8 → 107.9 µs (time ratio 1.029, speedup 1.181 → 1.145; pairs 1.5–3.6% slower, every B run below every A run), the same kernel at 128 tokens with the code unchanged time 0.994×, speedup 1.001×; DS512 at 256 tokens, `.cg` on the 128-row gate-up tiles: 226.4 → 228.0 µs (1.007; speedup 1.039 → 1.033; pairs 0.6–0.8% slower); DS256 at 2 tokens, `.cg` off: 19.09 → 19.26 µs (1.009; speedup 2.205 → 2.180; pairs 0.1–1.6% slower, 95% t-interval of the mean pair time ratio 0.999–1.019, worked out from the 4 pairs); the n = 3072 fp8 GEMM with its `.cs` output stores off: m = 6144 40.13 → 40.91 µs (every pair 1.9–2.0% slower), m = 8193 47.53 → 47.18 µs (every pair 0.7–0.8% faster) (the three MoE kernels and the n = 3072 fp8 GEMM)
- E13 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 runs in each of 3 sessions, the ruler (Kineto), A-vs-A] spread of 3 runs of an unchanged kernel: 2.4% (n = 3072 fp8 GEMM, m = 16), 0.9% (Qwen2-MoE router, 256 tokens), 1.7% (Qwen3.5 GDN decode, 2 tokens), 0.9% (DS256, 2 tokens); the GEMM moved 2.8% over another 3 runs of an identical copy and 2.8% over 8 alternated A and A' runs; the mean difference of two A arms 0.1–1.3%; in two later sessions of the same check, spread of 3 runs the GEMM 2.5% and 2.3%, the router 1.0% and 2.5%, the GDN 0.4% and 1.5%, DS256 2.4% and 3.2%, and the GEMM 5.9% over 8 runs in the last (the four kernels named)
- E14 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, median of 7 rounds × 5 launches] the §7 table, 256 programs at 1 warp from their ratio to the 2048 × 4-warp stream (0.0978–0.0985 of 6674–6684), 512 programs at 1 warp from a separate run (1205 and 1208 GB/s, 2 runs); configurations with equal bytes in flight within 1.104 of each other (widest at 4 MB: 256 × 4 warps at U = 4 5697–5756 against 512 × 4 warps at U = 2 6287–6333); 256 × 4 warps at U = 8 over 2048 × 4 at U = 1 1.067–1.070; at 3 MB 384 × 4 warps at U = 2 4920–4962 GB/s, 768 × 2 warps at U = 2 4898–4947, 96 × 4 warps at U = 8 3945–3995 (0.80–0.81 of 384 × 4 × 2), 192 one-wave programs at U = 16 2454–2467 (0.495–0.499); U `buffer_load_dwordx4` per lane before the first wait, one in a loop that uses each load before the next; VGPRs 22/38/70/140 at U = 1/2/4/8 (1 GiB `.cg` read stream, 16 B per lane, P programs × W warps × U loads in flight, the code of ../tuning.md §4)
- E15 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler (Kineto), 4 A/B pairs] n = 3072 fp8 GEMM at m = 16: prefetch depth 5 → 3 4.23 → 4.85 µs (speedup 1.475 → 1.285, every pair slower), 5 → 7 4.294 → 4.293 µs; XOR addressing of its LDS tiles off → on 4.245 → 4.263 µs, outputs bitwise identical; Qwen2-MoE router at 256 tokens: split-K 4 → 2 9.97 → 13.28 µs (1.33×), 4 → 8 9.93 → 10.31 µs (1.04×), look-ahead 2 → 1 10.01 → 11.28 µs (speedup 1.917 → 1.699), every pair slower; Qwen3.5 gated attention at 32 rows: KV splits 16 → 8, their head tile with them, 44.19 → 70.02 µs (1.58×), 16 → 32 44.17 → 46.90 µs (1.06×), every pair slower (the three kernels named)
- E16 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, median of 7 rounds × 5 launches] GB/s at 1024/2048 programs: async copy with `wait_group(0)` per tile 5844–5928/5497–5724; 2 slots with `wait_group(1)` 5719–5920/5538–5818; registers → `.store` → `.load` 5815–5976/5486–5709; registers only 5795–6006/5612–5981; a plain `gl.load` stream of the tiles 5776–5820/5476–5761; the four modes within 1.014–1.025 of each other at 1024 programs and 1.021–1.048 at 2048, the three LDS modes within 1.009–1.025 and 1.007–1.019, every mode 0.986–1.041 of plain; every fold equal to torch's; ISA: 40, 42, 40 and 64 VGPRs, registers only up to 12 loads in flight (1 GiB bf16 as 64×64 tiles, 4 warps)
- E18 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] the router's split-K projection kernel at 8 rows: as written 8 `buffer_load_dwordx4 … lds` + 16 `buffer_load_dwordx4` + 16 `ds_read_b128`, 8 KiB LDS; with `ROWS <= 0` 24 `global_load_dwordx4` + 24 `ds_write_b128` + 32 `ds_read_b128`, 4 KiB (Qwen2-MoE router at 8 tokens)
- E19 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only, the copies also run] copies into `SwizzledSharedLayout(8, 1, 8)` (bf16 `[64, 64]`) and `(16, 1, 8)` (fp8 `[64, 256]`): 2 and 4 `ds_bpermute_b32` for 2 and 4 copies; the fp8 copy, wait, relaxed read and store: 4 `ds_bpermute_b32` and 76 instructions, pre-swizzled into a linear `_reinterpret` view 0 and 56; async copies from pre-swizzled global columns into a flat alias, read through `(8, 1, 8)`, `(8, 2, 8)` (int16 32×64), `(16, 4, 4)` (int8 64×64), `(16, 1, 8)` (int8 64×128): equal to the source on the MI355X; `compute_efficient_padded_shared_layout` for the bf16 A operand of a 16×16×32 MFMA v4, warps [2, 2], k_width 8: `[32, 64]` → `[[512, 16]]`, `[8, 64]` → `None`; copies laid out from `offset_bases` into the helper's layout, 4 warps, A operands bf16 `[32, 64]` and uint8 `[64, 128]`: only `buffer_load_dwordx4 … lds` (1 and 2 per lane), tile and MFMA operand exact on the MI355X; counted in the library's sources: XORed offsets in all 6 DSA copies (4 kernels), 8 of 11 gated-attention copies (2 kernels), 92 of 231 fp8 GEMM copies (10 kernels); the gated attention at 8 rows copies K and V into `max_phase` 8 layouts (test kernels and the library's sources)
- E20 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, median of 7 rounds × 3 launches] µs operand 0/operand 1: plain 906.8–915.6/894.0–908.6, XOR `SwizzledSharedLayout(8, 1, 8, [1, 0])` 529.0–533.9/510.5–514.8, padded `[[64, 8]]` 630.7–642.0/515.5–519.7; XOR over plain 0.578–0.588/0.567–0.572, padded over XOR 1.189–1.214/1.004–1.016, XOR operand 1 over operand 0 0.960–0.971; `gl.bank_conflicts` aborted the process (SIGABRT) for the SharedLinear, Swizzled and Padded layouts (one wave per program, 2048 programs, a 64×64 bf16 slot read 2048 times as an MFMA operand)
- E22 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] bf16 `[64, 64]`, `instr_shape [16, 16, 32]`, k_width 8: A 4 `ds_read_b128`; B `[K, N]` 8 `ds_read_b64_tr_b16`; B `[N, K]` + `.permute((1, 0))` 4 `ds_read_b128`; fp8 B `[K, N]` 8 `ds_read_b64_tr_b8` (test kernels)
- E23 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] 4 warps: async copies into `.index(0)` and `.index(1)` of one allocation 1 `s_barrier` between them, into two `.slice` halves or two allocations 0; `gl.barrier()` → `s_waitcnt lgkmcnt(0)`, `s_barrier`, the `vmcnt` waits of an earlier async copy and `global_load` after it; 1 warp: no `s_barrier`, flat work-group size `1,64`; a 4-wave `global_load` then `global_store`: 0 `s_barrier`, 1 with a `gl.barrier()`; 4 rounds of store, `gl.barrier()`, load: a `.slice` per round 4 `s_barrier` (4096 B), one reused region 7 (1024 B); a 64×64 fp32 buffer written and read, then a transposing `convert_layout`: 3 `s_barrier` and 16384 B, with `_keep_alive()` after the conversion 2 and 32768 B, right after the last read 16384 B as without it; a relaxed read then a refill of the same slot: refill first at 4 and 1 warps, after a `gl.barrier()` read, `lgkmcnt(0)`, `s_barrier`, refill (4 warps); a ring tuple built with `+=` compiles, a list with `.append` fails ("'tuple' object has no attribute append"); `wait_group(1)` with 2 copies per group → `vmcnt(2)`, `s_barrier`; `wait_group(0)` then a read: `vmcnt(0)` then `s_barrier`, one `s_barrier` also with a `gl.barrier()` after the wait; an inline `vmcnt(1)` stayed with one more load after the copy, at 2 warps or a 64-row tile, where `wait_group(1)` became `vmcnt(2)`; the 128-token DSA kernel with one prefetch fewer: inline counts 7, 4, 2 with 5, 2, 0 instructions after the panels; the n = 3072 fp8 GEMM at m = 128 with and without its `_keep_alive()` calls: 50432 B both, identical LLVM IR (test kernels, that GEMM and the 128-token DSA kernel)
- E24 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, launches and compiled metadata; the grid sweep 4 runs] a 163840-byte kernel launched; a 167936-byte one compiled (`shared` 167936) and raised `OutOfResources` ("Hardware limit: 163840") at its first launch; the device reports 163840 bytes of LDS per CU; a grid sweep of 49152-byte workgroups: time flat up to 3 per CU, 1.996–1.997× at 4 to 6, at `waves_per_eu` 2 and 8, an 8 KiB control flat; `shared` 49152 B for the builds of the n = 3072 fp8 GEMM at m = 16 at prefetch depth 3, 4, 5, 7; from the compiled kernels of the library: `shared` of the 64-head DSA attention kernel 163840 B at 24 tokens, 156160 at 12 and 16, 157696 at 32, and 49152 B for that GEMM's one-warp kernel (test kernels, the GEMM and the four 64-head DSA kernels)
- E25 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3, 1 GiB read flush before each launch, 3 alternated rounds of 9 launches] every program's XCC id (`s_getreg`) equal to its id mod 8 on the default stream, in 4096-program launches and in 10 grids (1-D and multi-axis), 0 mismatches; 4 MiB L2 per XCD reported, a re-read per byte 2.22–2.23× slower at 8 than at 2 MiB per XCD; consecutive ids 21.3–21.6 µs (64 KB chunks) and 154.4–160.5 µs (512 KB), ids congruent mod 8 11.7–12.1 and 74.1–76.8 µs; ratio per round 1.76–1.82 and 2.07–2.17, round medians per run 1.78–1.80 and 2.08–2.12 (21.3–21.5 → 11.8–12.0, 156.3–159.0 → 74.4–74.6); one reader per chunk 0.98–1.01; the XCD remap of a bf16 GEMM at N 2624 × K 6144 a bijection for each grid of 1 to 699 programs; n = 7168 GEMM at m = 64 without its program-id permutation, the ruler (Kineto), 4 + 4 rounds, N=1: 6.262 → 6.262 µs, speedup ratio 1.001 (2048 programs × 4 warps, 8 readers per chunk; the two GEMMs named)
- E28 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only, some copies also run] direct-to-LDS copies, 4 warps: the §8 snippet at BM = BK = 64: 2 `buffer_load_dwordx4 … lds`, `s_waitcnt vmcnt(0)`, `s_barrier`, 2 `ds_read_b128`; without `wait_group(0)` no `vmcnt` wait before the `s_barrier` and the reads, and on the MI355X the read differed from the source in 1000 of 1000 runs, in 0 of 1000 with the wait; `buffer_load_to_shared` at 16 and 4 B per lane compiled (`buffer_load_dwordx4`, `buffer_load_dword` `… lds`), at 8, 2 and 1 B failed ("LLVM Translation failed for operation: builtin.unrealized_conversion_cast"); lanes `[64, 1]` over a `[64, 32]` tile failed, `[16, 4]` compiled; bf16 `[64, 64]` with a runtime row stride: unmarked failed, "D" or hinted 2 `buffer_load_dwordx4 … lds` per lane; fp32 into `[[256, 4]]` padding 4 per lane, into `[[64, 4]]` 4 failed and 1 compiled; `global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier='')`: 16 B per lane → `global_load_lds_dwordx4`, 8 B per lane and lanes down rows → "failed to legalize operation"; a copy without `commit_group()`: no `vmcnt` wait before its read, also behind `wait_group(0)`; a scale load after a 2-instruction copy `vmcnt(1)` before its use, before the copy `vmcnt(3)`; timed, 192 one-wave programs, 8 groups: scales first 6.078 µs, after 6.030 µs (test kernels)
- E29 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] `x_ptr + e * (7168 * 2048)`, `e` an int32 from `gl.load`: `mul i32 %13, 14680064` in the LLVM IR, with `e.to(gl.int64)` `mul nsw i64`; `buffer_load` from a base offset by a program id, a kernel argument or a scalar `gl.load`: 1 `v_readfirstlane_b32`, no loop, by a 1-warp `gl.max` none, by a 4-warp `gl.max` 5 `v_readfirstlane_b32`, `s_and_saveexec_b64`, `s_cbranch_execnz`; bf16 offsets + 128 → `offset:256`, + 2048 → a second offset VGPR, + 2047 → 8 `buffer_load_ushort`; steps of 16 and 4080 bytes in the immediate; the 16×16 preshuffle offset from an i32 tile argument: 28 VALU (int32), 22 (`.to(gl.uint32)`), 28 (int32 with `gl.assume(tile >= 0)`); 2 bf16 per lane, int32 or uint32 offsets: 1 `buffer_load_dword` / `global_load_dword` either way; bf16 `[32, 64]`, 16 B per lane, base `X + slot * stride`: `stride` unmarked → 8 `buffer_load_ushort` and the async copy "failed to translate module to LLVM IR", `stride` "D", constexpr 4104 or `gl.multiple_of(slot * stride, 16)` → 1 `buffer_load_dwordx4` (`… lds`); a uniform `gl.load` ahead of any copy or store → `s_load_dword` (test kernels)
- E30 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only, and a count over the library's sources] u8 `[64, 256]`, mask `(r < M) | (M % 64 == 0)`, `M` constexpr: `buffer_load(..., other=0)` at M = 64 0 `v_cndmask`, at M = 50 5; an async copy with a zero `other` tensor at M = 64 no branch or `ds_write`, at M = 50 1 `s_and_saveexec_b64`, 2 `s_cbranch_execz`, 1 `ds_write_b128`; the async-copy kernel of the n = 3072 fp8 GEMM built for m = 6144, with and without its zero `other` tensors: identical code at M = 64, 128, 256, 512 (BM 64), 3072, 6144 (BM 128), different at M = 100 and 1000; `gl.load`/`gl.store` masked by `(row < M) | (M % 32 == 0)`: constexpr M = 256 no branch, runtime M 1 `s_and_saveexec_b64` + 1 `s_cbranch_execz`; LDS rings, 4 warps, read slot 0 then a DMA into slot 1: one `[2, 64, 64]` allocation, `.load` 3 `s_barrier`, relaxed 2; separate allocations 2 either way with identical code, and identical with a `gl.barrier()` before each refill, also in a rolled loop over two `.index()` slots (206 instructions, same waits); a static ring of separate `[64, 64]` bf16 allocations, 2 groups ahead, 6 stages: `.load` with 2 slots 10 `s_barrier`, with 3 slots 6; relaxed, refill after the reads: with 2 slots 4 refills before their slot's last read, with 3 none; 77 of the library's 154 kernels pass `other=0` (or `None if FULL else 0`) in 308 `buffer_load` calls (test kernels, that GEMM and the library's sources)
- E32 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only, some outputs checked on the GPU] the §5 snippet, BLOCK 2048, 4 warps: `n` "D" → 1 `buffer_load_dwordx4` + 1 `buffer_store_dwordx4`, `n` unmarked → 8 `buffer_load_ushort` + 8 `buffer_store_short`, through `gl.load` 8 `ushort` and 1 `dwordx4`, outputs exact at n = 2048 and 2047; bf16, base "DS": 8, 4, 2, 1 elements per lane → `dwordx4`, `dwordx2`, `dword`, `ushort`, a base without "D" `ushort`; offsets `arange + 2`, `+ 4`, `+ 8` `dwordx4`, `+ 1` `ushort`; `[64, 64]` bf16, 8 per lane: plain 2 `dwordx4`, XOR of 8-element chunks 16 `ushort`, `(x // 2) * 2` 8 `ushort`, `rows * stride` unmarked 16 `ushort`, "D" 2 `dwordx4`, `% 65536` and a row-constant `gl.where` 2 `dwordx4`, `% 4` `dwordx2`, a column-varying `gl.where` 16 `ushort`; columns permuted in 4-element chunks, hint 4 → 4 `buffer_load_dwordx2`, exact, hint 8 → 2 `buffer_load_dwordx4`, wrong values on the MI355X; a masked `[64, 64]` bf16 copy, pointers "DS": Gluon `gl.load` 2 `global_load_dwordx4`, plain `tl.load` 2 `buffer_load_dwordx4`; a `buffer_load` sum, an async-copy MFMA loop and a two-slot ring identical at `num_stages` 1 and 3 (2 compilations), a plain `tl.dot` loop 0 → 20 `buffer_load … lds` (test kernels)
- E33 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, compile-only, the remap also run] `buffer_store` of a `[32, 64]` tile from `AMDMFMALayout(4, [16, 16, 32])`: transposed bf16 2 `buffer_store_dwordx2`, fp32 2 `buffer_store_dwordx4`; not transposed bf16 8 `buffer_store_short`, fp32 8 `buffer_store_dword`; through `convert_layout` to `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])`, bf16: 1 `buffer_store_dwordx4` with 2 `ds_write_b64` and 2 `ds_read_b64` (transposed) or 8 2-byte `ds_write_b16` and 8 `ds_read_u16` (not); the exchange rounded first 4096 B, after 8192 B; small_m_gemm.md §3's XCD remap (from the bf16 GEMM at N 2624 × K 6144) compiles with `@gluon.jit` and `gl.constexpr` and is a permutation for grids 1, 7, 9, 164, 224, 255, 256, 1000, its GPU output equal to Python's (test kernels and that GEMM's remap)
- E34 [2026-11, gfx950, Triton 3.8.0, N=1, compile-only] `wait_group(n)` with `n` a kernel argument or a `range` variable: `CompilationError` at trace, "create_async_wait_group(): incompatible function arguments", a constexpr compiles; a 1-tuple grown in a `range` loop: "'scf.for' op different number of region iter_args and yielded values: 1 != 2", a fixed-length queue compiles; §12's rolled-ring structure for an fp8 block-scaled GEMM, M = 32, N = 4096, K = 7168, 16 columns, 2 waves, 6 slots, one step per trip: 325 instructions (1417 unrolled), 56 VGPRs, 45056 B of LDS, loop `vmcnt(12)`, drain `vmcnt(9)`, `(6)`, `(3)`, `(0)`; a weight scale `gl.load`ed at its group: `vmcnt(0)` and `vmcnt(12)` in the loop; a tuple of scales loaded with their groups' copies: `vmcnt(3)`, `(7)`, `(16)` at one step per trip, `(14)`–`(19)` at 5; a weight scale from a 1-D LDS table: 60 VGPRs, 45312 B, `vmcnt(12)`; 8 slots, 7 steps per trip: 57344 B, 88 VGPRs, `vmcnt(18)`; unrolled 96 VGPRs and 64 SGPRs, rolled 31 SGPRs, no spill; a `[64, 32]` fp32 table with a rank-2 layout: `.index(g)` fails, "rank must be equal to or one less than the shape size. Got 2 and 1", with a rank-1 layout the allocation or `.store` fails ("source has rank 2 but memdesc encoding has rank 1"); per group in a `range` loop, `.gather` (a 2-D table with `gl.full((32, 1), g)`, or a flattened table) or a rank-1 `_reinterpret` view's `.index(g)`: one 32-bit word per trip, 1 `ds_read_b32` in the one-trip loop, wider reads in the 8× unrolled one; 1 `s_barrier` per trip without a `gl.barrier()` after the staging, 0 with it; a uniform `gl.load` after an async copy or a `gl.store` → `global_load_dword`, also in a 4-trip `range` loop that LLVM unrolled; `WS.to(gl.uint64).to(gl.pointer_type(gl.float32, 4))` alone fails to compile, through §13's inline asm 4 `s_load_dword` after a copy or a store; MM64-FP8's fp8 projection kernel, M = 8, BN = 32, 2 warps: 68, 84, 84 VGPRs at look-ahead depth 1, 2, 3 (66, 90, 88 at one split), no spill (test kernels and MM64-FP8 at 1–16 tokens)
- E35 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (MM64-FP4 and DS512 at 256 tokens N=4; DS256 at 2 tokens and the store A/Bs N=3), the ruler (Kineto), 4 A/B rounds per size, in time and in speedup over the baseline] weight-load `.cg` on and off at 12 sizes of 7 MoE kernels, 10 whose routing reads each weight tile about once (mean readers < 2) and 2 re-read; the ISA changed only in cache bits; accuracy passed; unchanged-binary controls within 0.37%; time without/with `.cg`: three 1-token sizes 0.995–1.004 with pairs both ways, MM256-FP8 at 2 tokens 1.037 (speedup 1.020, 3 of 4 speedup pairs agreeing), MM64-FP8 at 4/8/16 tokens 1.023/1.062/1.096, MM64-FP4 at 128 tokens 1.018 (speedup 1.016), at 256 tokens 1.028, DS256 at 2048 tokens 1.016, DS512 at 256 tokens 0.998, MM256-FP4 at 16384 tokens 0.854; `.cg` faster in every pair, in time and speedup, at six of the read-once sizes, slower in every pair at none; flipping the kernel's own hint won in every pair at MM64-FP8 at 4/8/16 tokens (time 0.909–0.982) and MM64-FP4 at 128 tokens (0.978–0.988); repeats: MM64-FP4 at 256 tokens 2.6–3.5% slower without `.cg` and DS512 at 256 tokens 0.2–0.7% slower with it (4 runs each), DS256 at 2 tokens 0.0–1.2% (3 runs, every pair the same way in two); removing `.cs` from the n = 3072 fp8 GEMM's output stores cost 1.7–1.9% at m = 6144 and gained 0.1–0.7% at m = 8193, every pair (3 runs) (MM64-FP8, MM256-FP8, MM64-FP4, MM256-FP4, DS256, DS512 and the n = 3072 fp8 GEMM)
- E36 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, accuracy runs and compiled ISA, not timed] Qwen3.5's GDN decode at 2 tokens, a pool of 4 state slots (16 value heads, K = V = 128, 3072 channels; recurrence BK 128, BV 16, row group 16, 2 warps, `waves_per_eu` 1), recurrent state through buffer ops or through `gl.load`/`gl.store`: 36/36 accuracy checks (all outputs finite) and 18/18 bitwise repeats passed (values: max abs error 8, within atol 2 + rtol 0.125; scales max abs error 1.6e-9 against an fp64 reference, conv_state 0, delta_state 2.4e-7); 8 compiled 64×64 bf16 copies (8 elements per lane, 4 warps, pointers "D" or "DS"): masked buffer accesses 0 exec branches and 0 `s_and_saveexec`, masked pointer accesses 6 of each, unmasked pointer accesses 0 (that GDN kernel)
- E37 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, the ruler (Kineto)] 85 timed graph calls, each preceded by its own flush inside the graph; the flush buffer unchanged and its per-block sums equal to torch's; a two-call probe: the second call re-reading the first call's 16/64/128 MB buffer behind the ruler's flush took 0.997–1.009× the time of reading a previously unread buffer, and 1.161–1.202× the time of the same re-read back to back without the flush at 64/128 MB (a read kernel over 16–128 MB)
- E38 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1 (the router at 256 tokens 4 → 2/8 N=8, gated attention N=7 and 4), the ruler (Kineto), 4 A/B rounds] Qwen2-MoE router at 256 tokens, splits × K tiles ahead: equal products 4×2 (the kernel's own), 8×1, 2×4 1.000×, 1.033×, 1.260× the kernel's own time, doubled products 4×4 and 8×2 0.970× and 1.032×; moving the kernel's own split: the router at 256 tokens 4 → 2 1.331–1.341×, 4 → 8 1.02–1.04×; at 128 tokens 1.221× and 1.033×; at 64 tokens 8 → 4 1.041×, 8 → 16 0.993× (pairs 0.979–1.001); Qwen3.5 gated attention at 32 rows, KV splits 16 → 8 with the head tile 1.579–1.584×, 16 → 32 1.050–1.062×; DS256 at 2 tokens 4 → 1 1.079×, 4 → 8 1.050×, 4 → 2 1.004×; at 1 token 4 → 2 1.020×, 4 → 8 1.028×; every arm accurate (the six kernels named)
- E39 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, helper calls, compilations and GPU readbacks] 900 helper combinations (bf16, fp8 e4m3, uint8; A and both B orientations; 16- and 32-wide MFMA; 1–16 waves; 16–256 rows × 64/128 columns): 770 layouts and 130 `None`, independent of waves (A tiles with no more rows than the MFMA's M, fp8/uint8 B tiles with K = 16 at 16×16×128), 51 of them `None` although the tile had at least 1024 × num_warps bytes; bf16 A `[32, 64]` of 16×16×32 gave `[[512, 16]]` at 1, 2, 4, 8 and 16 waves; the 603 layouts with tile bytes ≥ 1024 × num_warps compiled to exactly tile_bytes / (1024 × num_warps) `buffer_load_dwordx4 … lds` per wave, no `ds_bpermute` or `ds_write`, read back bitwise, bf16/fp8 MFMA products exact; the 167 smaller ones failed to compile; 45 identity-offset padded configurations (fp32, bf16, fp8; single intervals 64–1024 and 10 two-interval pairs, smaller interval first): the widest compiling width was the widest 4- or 16-byte DMA of at most min_interval / 64 elements per lane, and every request above it failed (test copies)
- E40 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=2, 2000 launches × 1024 programs per arm] a delayed reader against an overwriting wave: without `gl.barrier()` 1947 and 1911 of 2000 launches wrong at 4 waves, 2000 of 2000 at 8 in both runs; with it 0 at either wave count; the ISA 0 `s_barrier` without `gl.barrier()`, 1 with it (test kernel)
- E41 [2026-11, gfx950, Triton 3.8.0, N=3, compile-only, not timed] 18 builds at 1 and 4 waves of runtime `range` loops kept rolled (backedges present): 16–18 data `s_load` (kernarg loads excluded) when the body has no copy or store; with a copy or store in the body 0 data `s_load` and 18 `global_load_dword`, also with the load placed first; a constant-address-space (4) pointer kept 16–18 `s_load`; loads before the loop stayed scalar; counts are static instructions, not dynamic loads (test kernels, with the constant-address-space pointer of the n = 3072 fp8 GEMM at m = 32)
- E42 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, 12 randomized timing blocks, paired geometric means with 95% t intervals] 39 builds, 78/78 numerical checks correct; seven same-kernel A-vs-A paired 95% intervals inside ±1%; all four 8 MiB configurations (512×1×16, 256×2×16, 256×8×4, 2048×4×1) within 10% of their level median; 384×1×8, 256×1×16 and 384×1×16 at 0.888/0.760/0.861 of their 3/4/6 MiB level medians; against 384×4×2/256×4×4/384×4×4, paired rate ratios 0.867846/0.744441/0.854286 (95% CIs 0.866498–0.869197/0.743706–0.745176/0.853928–0.854644); at 3 MiB 96×4×8 and 192×1×16 0.800×/0.678× of 384×4×2 (95% CIs 0.7986–0.8021/0.6775–0.6793); every build U `dwordx4` loads before its first wait, no scratch (a 1 GiB bf16 in-lane-sum stream, P programs × W warps × U loads per lane)
- E43 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, sum of the two GDN kernels' device durations per call from a Kineto trace, state restored and a 1 GiB read flush before each call (both outside the sum), 32 balanced A/B pairs of 10 replays × 20 calls per input, 8 A-vs-A pairs, 99% family intervals] recurrent state through `gl.load`/`gl.store` `.cs` (B) against `buffer_load`/`buffer_store` `.cs` (A), 3 seeds × slot patterns (3, 0), (1, −1), (−1, 2): B/A within ±1% at 4 of 9 inputs, entirely above 1.01 at the 3 inputs with slots (−1, 2) (1.028, 1.013, 1.012), crossing 1.01 at 2; every A-vs-A interval contained 1; numerical checks after each measurement passed (Qwen3.5's GDN decode, 2 tokens, 4 state slots)
- E44 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=1, the ruler's timer (Kineto), 4 adjacent AB/BA-balanced pairs per candidate, paired log-ratio 95% t interval widened for order] call time against the kernel's own prefetch depth: n = 3072 fp8 GEMM at m = 64 (its own depth 4, `waves_per_eu=6`): 6 → 0.9567 (0.9268–0.9876), 2 → 1.7790 (1.7049–1.8571, 2 VGPRs spilled); at `waves_per_eu=0`: 6 → 0.9913 (0.9446–1.0402), 2 → 1.7716 (1.7466–1.7970); the same GEMM at m = 32 (its own depth 5, `waves_per_eu=8`): 3 → 1.1923 (1.1540–1.2321, 8 VGPRs spilled), 7 → 0.9989 (0.9504–1.0499); at `waves_per_eu=0`: 3 → 1.1471 (1.0766–1.2228), 7 → 1.0011 (0.9316–1.0754) (the n = 3072 fp8 GEMM at m = 64 and at m = 32)
- E45 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=4, the ruler (Kineto), 4 alternated runs per arm, mean of per-run medians] recurrent state through `gl.load`/`gl.store` `.cs` instead of `buffer_load`/`buffer_store` `.cs`: 8.79 → 8.81 µs; all eight runs passed the accuracy check (Qwen3.5's GDN decode, 2 tokens, 4 state slots)
- E46 [2026-11, gfx950 (MI355X), HIP 7.15, N=2, one process, `HW_REG_XCC_ID` read back by every workgroup] default stream: XCC id = id % 8 for 1-D grids of 255–8192 workgroups and a 37 × 11 grid; on streams created in the process: (id + 7) % 8 with one created stream, 6 and 7 with two, 5, 6 and 7 with three, 5 on all four with four; a graph replayed on a stream took that stream's offset (one capture stream tested) (test kernel)
- E47 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=9 runs, rounds of the ruler (rocprofiler-sdk), A-vs-A] the four kernels of E13 unchanged, spread of each non-overlapping three runs 0.41–3.53%; the router's first run 1.883 against 1.935–1.957 in the other eight (E13's kernels)
- E48 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions × 40 rounds, the ruler's timer (rocprofiler-sdk), arms rotated] DS256 at 2 tokens, gate-up K split changed only in the host code: 4 → 2 1.0103× the time (95% interval 1.0096–1.0109), 119 of 120 rounds slower; every arm passed the accuracy check (that kernel)
- E49 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions × 10 rounds, the ruler's timer (rocprofiler-sdk), arms in random order] n = 3072 fp8 GEMM at m = 64, call time with prefetch depth 4 → 6: 0.9956× at its own `waves_per_eu=6`, 0.9941× at `waves_per_eu=0` (that GEMM)
- E50 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 repeats, `buffer_load` probes of a 4.25 GiB uint8 tensor from three scalar bases, `num_records` 0x7ffffffe] byte and short loads returned data up to byte 2^31 − 3 from the base and 0 from 2^31 − 2; dword and `dwordx4` loads data up to 2^31 − 5 and 0 from 2^31 − 4, a `dwordx4` across that edge its first 12 bytes; the edge moved with the scalar base (a probe, not one of the library's kernels)
- E51 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions of 11 rounds, CUDA-event medians, g = t(none) / t(arm), session medians in parentheses] 1 GiB bf16 streams, 16 bytes per lane, every result correct: 2048 programs × 4 warps sweeping the tensor together, reads `nt` 1.085–1.093 (1.082–1.095), writes `.cs` 0.989 / 0.993 (0.979–0.999), `.wt` 0.982 (0.977–0.989) on `buffer_store` and 1.411 on `gl.store`; 512 programs × 4 warps, each its own 2 MiB, reads `nt` 1.046–1.052 (test kernels)
- E52 [2026-11, gfx950 (MI355X), Triton 3.8.0, N=3 sessions of 9 rounds, CUDA-event medians, q = t(XOR) / t(arm)] 2048 programs reading a 64×64 bf16 slot 2048 times as an MFMA operand (`k_width` 8), the same loop body for plain, XOR and padded `[[64, 8]]`; operand 0 / 1: 16×16×32, one wave: plain 0.549 / 0.561, padded 0.959 / 1.010; four waves: plain 0.459 / 0.534, padded 0.862 / 1.017; 32×32×16, one wave: plain 0.283 / 1.012, padded 1.342 / 1.001; E20's kernel compiled offline: plain and XOR operand-0 loops unrolled twice, the padded one not (test kernels)
