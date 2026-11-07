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
you and what you write by hand. “Compiled” means offline Triton 3.8.0 compilation for gfx950;
counts describe the stated layout and shape, not GPU performance. Layouts:
[layouts.md](layouts.md); MFMA: [mfma.md](mfma.md); asm: [wave_ops.md](wave_ops.md).

## 1. What to use when

| You need | Use | § |
|---|---|---|
| A tensor tile from global memory | `gl.amd.cdna4.buffer_load(base, offsets, mask)`: scalar base, int32/uint32 element offsets; Gluon does not turn `gl.load` into it | 2, 3 |
| A scalar, an index, per-program metadata | `gl.load(ptr + i)`; a rolled body containing copy/store can use global loads even if the load appears first; constant-address-space pointer for the measured scalar path  | 13 |
| A tensor that can exceed 2 GiB from one base | 64-bit rebase of the scalar base, or a host gate to a pointer path | 3 |
| Bounds | for buffer loads whose masked lanes need zero, use a mask without `other`; `gl.load` needs an explicit fill if masked values will be consumed. Fold masks on full tiles, and use row masks when the contiguous extent is entirely in bounds | 4, 5 |
| 16-byte accesses | 16 bytes per lane in the layout, a base marked divisible by 16, offsets the compiler can prove contiguous | 5 |
| A mostly read-once stream | `.cg` / `.cs` non-temporal loads; choose by actual reuse and measure per Point | 6 |
| Bandwidth from a small grid | Vary grid, waves and independent load depth; bytes in flight are not a rate law | 7 |
| A tile in LDS | Async copy, or registers → allocation + `.store`; choose by reuse and overlap | 8 |
| A layout change | `gl.convert_layout`: lane moves by DPP, `v_permlane` or `ds_bpermute`, wave moves through LDS ([layouts.md](layouts.md) §7) | — |
| A tile only its own wave reads (preshuffled weights) | can load into the operand layout without LDS; verify ownership and output; not inherently faster than LDS | 8, 13 |
| Swizzled or padded LDS for a DMA'd tile | pre-swizzled global offsets into a linear `_reinterpret` view, or a padded layout | 9 |
| Load/MFMA overlap | a hand-written ring of commit groups (rolled for long K) | 12 |
| LDS reuse | separate allocations with `.load` use compiler-inserted barriers; indexed arenas using `load_shared_relaxed` require explicit slot-lifetime reasoning and barriers. Ensure copies complete before reads and all consumers finish before refill; neither spelling alone guarantees race freedom | 10, 11, 12 |
| Programs that read the same data in one call | in the tested gfx950 grids, ids congruent mod 8 mapped to one XCD; verify placement and A/B the remap for the target grid and partition rather than assuming a universal scheduling guarantee | 14 |
| Independent counters and ticket allocation | `sem="relaxed"` only when the atomic does not publish or acquire associated data; a communication protocol needs the appropriate ordering and scope. Atomics: [../moe.md](../moe.md) §6 (buffer atomics), [../numerics.md](../numerics.md) §8 (determinism) | — |

## 2. What the compiler does and does not do for you

`@gluon.jit` kernels are lowered by `gluon_to_ttgir`: inliner, layout resolution, SCCP, CSE, canonicaliser, loop
unroller, select/if combining, the warp-pipeline pass, warp-group allocation, nothing else outside the `fpsan`
instrumentation mode. Coalescing, the software pipeliner and the conversion to buffer ops of pointers marked "S"
(`tt.pointer_range = 32`: with buffer ops enabled, a `ptr_range` or tensor storage size of at most `2**31 - 1` bytes) belong to `make_ttgir`, the plain-Triton path, which pipelines with
async copies on gfx950 (S1; [../gemm/decode_gemm.md](../gemm/decode_gemm.md)). So in Gluon:

- **`gl.load`/`gl.store` stay pointer accesses**: `global_load`/`global_store`, or `s_load` for a suitably uniform load (§13). A load textually before a copy/store in a rolled body can still use a global load; the measured constant-address-space path preserves scalar loads; a masked copy, tensors marked "S", compiled to `global_load_dwordx4` through `gl.load` and
  to `buffer_load_dwordx4` through plain `tl.load`. In a compiled
  64×64 bf16 copy (VEC8, 4 waves, D/DS), masked buffer accesses had no exec branches/saveexec, masked pointer
  accesses had 6 of each, and unmasked pointers had none (8 builds, no GPU copy timing).
- **The offset tensor's register layout is the access pattern**; nothing re-coalesces it (§5).
- **No automatic `num_stages` software pipeline is added.** Explicit warp-pipeline stage annotations are handled by Gluon's warp-pipeline pass. Only `make_ttgir` reads `num_stages` for automatic loop scheduling (S1): three Gluon loops compiled to identical
  instructions at `num_stages=1` and `3`, a plain `tl.dot` loop did not. Write the prefetch (§12).
- **Membar still runs** in the lowering, with an AMD filter, and places LDS barriers (S2; §11).

## 3. Buffer loads and stores: signature, addressing, the 2 GiB rule

| Call (Triton 3.8.0) | Notes |
|---|---|
| `gl.amd.cdna4.buffer_load(ptr, offsets, mask=None, other=None, cache=None)` | `ptr` a scalar pointer; `offsets` an int32/uint32 tensor in elements; `other` needs `mask`; the result has `offsets`' layout and `ptr`'s element type |
| `gl.amd.cdna4.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)` | no `other`; `stored_value` must already have `ptr`'s element type |
| `gl.amd.cdna4.async_copy.buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")` | §8 |
| `gl.amd.cdna4.async_copy.global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="")` | `ptr` a tensor of pointers, 64-bit: no 2 GiB bound, it copied 4 KiB at byte 2^31 + 8192 of a 3 GiB tensor that `buffer_load_to_shared` read as 0 ; 16 bytes per lane compiled, 8 bytes per lane or the tested row-strided lane layout failed to legalize (neighboring lanes differ by one runtime row stride; 64 bytes when that bf16 stride is 32 elements)  (S25; §5) |

The keyword is `cache=` on `buffer_load`/`buffer_store` (S4) and `cache_modifier=` on `gl.load`/`gl.store`, the `tl`
builtins (S17, S18). Kernels that call `buffer_load`/`buffer_store` through the previous generation's `gl.amd.cdna3` get
the same functions: `cdna4` re-exports those two (S3).

**What the hardware checks.** Each base becomes one descriptor with `num_records = 2^31 − 2` bytes; each lane sends
`offset × element bytes`, multiplied in 32 bits, and a masked-off lane sends `0x80000000` (S5). A raw buffer access is
out of range when its byte offset reaches `num_records`, and the base is not part of the check
([ISA §2](../../hardware/gfx950/memory_and_sync.md)). Hence:

1. **No tensor bound.** An unmasked `buffer_load` past the end of a 64-element view returned the data behind it
   (64.0–79.0), and the ISA held `s_mov_b32 s2, 0x7ffffffe`. Bound every access with its mask.
2. **Silent zeros, then aliasing.** Out of range, loads read 0 and stores were dropped without a fault; an offset
   reaching 2^32 wrapped onto the wrong element. Keep every lane's byte offset below 2^31 − 2.
3. **The base is free.** Put wave-uniform origins (tile, expert, slot, split) into the scalar pointer, as the router
   does, `Hidden += tile_m * BM * K + split * (K // SPLIT)` with offsets `rows[:, None] * K + aks[None, :]` (S6;
   mechanism: [../../gemm/datapath.md](../../gemm/datapath.md) §8), with overflow checked for correctness, and uniformity/alignment checked for efficient lowering :
   - **Cast to 64 bits before the product can pass 2^31**: it forms in 32 bits, `x_ptr + e * (7168 * 2048)` with an
     int32 `e` compiling to `mul i32`, which wraps silently, and with `e.to(gl.int64)` to `mul nsw i64`. MiniMax-M2
     writes `W + expert.to(gl.int64) * (N * (K // 2))` and keeps 32-bit offsets (S44).
   - **Provably wave-uniform.** Offset by a program id, a kernel argument, a scalar `gl.load` or a one-warp `gl.max`,
     the base stayed scalar; by a 4-warp `gl.max`, the `buffer_load` compiled into a waterfall loop
     (`v_readfirstlane_b32`, `s_and_saveexec_b64`, `s_cbranch_execnz`). Move such a value to an SGPR with an inline
     `v_readfirstlane_b32` first, as DeepSeek-V2 does with its job descriptor (S45; [wave_ops.md](wave_ops.md) §6.4).
   - **A provable multiple of 16 bytes**: constexpr shapes, or runtime ints the launch marks divisible by 16, which it
     does only for values that are (S16). With `X + slot * stride`, `stride` unmarked, a 16-byte bf16 `buffer_load`
     became 8 `buffer_load_ushort` and the async copy failed at translation; a marked or constexpr `stride`, or
     `gl.multiple_of(slot * stride, 16)`, gave `buffer_load_dwordx4`: a file can compile at one Point, fail at another.

The K step can ride the base too, keeping the offsets loop-invariant (DeepSeek-V2 picks per Point, `REBASE`, S49). Gate
a pool that can outgrow the range on its **byte** extent against `2**31 - 2` on the host (S7), with another path above
it (a 64-bit rebase per slot, `gl.load`, `global_load_to_shared`); a contract bound makes it dead code.

- **Constant offsets.** A constant added to the offsets rode the 12-bit `offset:` immediate at 256 bytes and cost a
  second offset register at 4096 bytes; `+2047` bf16 elements (4094 bytes) split each 16-byte load into 2-byte loads. Keep constant steps multiples of 16 bytes and under 4096.
- **Index math in uint32.** `//` and `%` by a power of two need sign fix-ups in int32, not in uint32: a test kernel
  computing the 16×16 preshuffle offset ([mfma.md](mfma.md) §7) from a tile index passed as an argument had 28 VALU
  instructions in int32, 22 after `.to(gl.uint32)`, and `gl.assume(tile >= 0)` changed nothing ; S6 and
  S27 take program ids as uint32. Convert only values non-negative by construction; per-lane offsets kept their width.
- **`buffer_store` does not convert** (`gl.store` does, S8): write `acc.to(gl.bfloat16)`; fp32 through a bf16 pointer
  failed with "'amdg.buffer_store' op failed to verify that value element type matches the pointed type of ptr".

## 4. Masks, `other`, sentinels

| Access | A mask costs | `other` adds |
|---|---|---|
| `buffer_load` | one offset select per vector, no branch (S5) | a select per 32-bit word, `other=0` included: 1 → 5 `v_cndmask` for one 16-byte vector  |
| `buffer_load_to_shared` | the same for the measured non-replicated tile; redundant-thread predication can also add a branch | a branch around each copy and a `ds_write_b128` of `other` (S10): a u8 `[64, 256]` copy, wait, read and store, 4 copies per lane, went from 59 to 103 instructions  |
| `gl.load`, `gl.store` | a conditional branch for a non-constant mask (S9); the measured divergent tile masks use exec-mask branches around vectors, while a constant-true mask needs no mask branch (below)  | — |

Only the zero the lowering builds when `other` is omitted is known to be zero (S5); a passed `other`, 0 included, is
selected in (S10) unless the mask folds to the constant `true`: at a dividing M, `buffer_load(..., other=0)` compiled
with no `v_cndmask` and the async copy with no branch or `ds_write`, and the n = 3072 GEMM's `_gemm_async_kernel` the
same code with and without its zero `other` tensors (S52). Masked lanes of a buffer load already read 0:
write `buffer_load(ptr, offs, mask)`, and pass `other` only for a non-zero fill such as unit e8m0 scales (`other=127`,
or `2139062143` for a word of four, S11). How AKLR kernels avoid paying for bounds:

- **Masks that fold on full tiles**, with the token count a `gl.constexpr` (`M` is specialized to the Point's token count; this does not imply one binary per Point, S11, S12):
  `((row[:, None] < M) | (M % ROWS == 0))` (S11), or `None if FULL else mask`. With a runtime `M` the mask stayed,
  and `gl.load`/`gl.store` kept their branch.
- **Clamp reads, mask only the store**, for `M > 0` and other accessed dimensions fully in bounds; `M % BM == 0` is decided at compile time (S12):
  `read_rows = rows if M % BM == 0 else gl.minimum(rows, M - 1)`.
- **Offset sentinels, 1-byte data only**: DSA gives invalid (negative) KV slots the element offset `-2147483648`
  (S13), out of range as an unsigned offset, so its fp8 gathers need no mask. For 2- or 4-byte elements the 32-bit
  multiply turns it into 0 (S5) and those lanes read element 0: use a mask.

## 5. Vector width and contiguity

A buffer access moves `min(16 bytes, contiguity × element bytes)` per lane; contiguity is limited by the compiler's offset contiguity/alignment analysis, the layout's contiguous elements per thread, and the scalar base's divisibility expressed in elements (S14), capped by the mask's constancy along the contiguous
dimension (S10, S15); bf16 offsets misaligned by 4 or 8 bytes still loaded 16 bytes. `offs < n` with a
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
same way. A row mask, constant along the contiguous dimension, kept a `[64, 16]` fp32 tile at
`buffer_load_dwordx4` : mask rows, keep the contiguous extent a constexpr multiple of the vector, and
give full tiles an unmasked path.

Axis analysis (S15) loses contiguity through XOR and `//` (unless by 1), takes a runtime argument's divisibility as 1
unless the launch marks it divisible by 16, keeps gcd(contiguity, divisibility, divisor) through `%` by a constant, and
through `gl.where` what both branches and its condition's constancy allow: XOR of 8-element chunks, an unmarked `stride`
or a `gl.where` varying along the row cut bf16 loads to 2 bytes, `% 65536` or a row-constant `gl.where` kept 16. Restate only what is lost, with `gl.max_contiguous(gl.multiple_of(offsets, [1, V]), [1, V])` (S17), V
the elements of the true aligned run (16 for the fp8 chunks of §9's excerpt, 8 for bf16; [layouts.md](layouts.md) §6.1).
Both are promises the compiler does not check (S18), and a larger V is believed: bf16 offsets permuted in 8-byte chunks
loaded `buffer_load_dwordx2` with V = 4 and `buffer_load_dwordx4` with V = 8, reading wrong values.

An async copy depends on these proofs. Triton 3.8.0 lowers direct-to-LDS loads on CDNA4 only at 128 and 32 bits per
lane, leaving out the ISA's 8-, 16- and 96-bit forms (S20; [ISA §3](../../hardware/gfx950/memory_and_sync.md))
and rejects a copy whose provable vector is narrower than its layout's per-lane run, or whose lanes do
not write `dest` in order (S14, S25), with `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`:
a runtime row stride of unknown divisibility did (marked divisible by 16, or hinted, it compiled to
`buffer_load_dwordx4 … lds`), and so did aligned 16-byte offsets whose lanes walked rows 64 bytes apart.

## 6. Cache modifiers: encoding and selection

Valid strings: loads `.ca.cg.cs.cv`, stores `.wb.cg.cs.wt`; others raise `ValueError` (S8).
The gfx950 lowering sets these bits (S14):

| Op | String | `buffer_*` bits | `global_*` bits |
|---|---|---|---|
| load | none, `.ca` | none | none |
| load | `.cg`, `.cs` | `sc0 nt` | `nt` |
| load | `.cv` | `sc0 sc1` | `sc0 sc1`, volatile |
| store | none, `.wb`, `.cg` | none | none |
| store | `.cs` | `sc0 nt` | `nt` |
| store | `.wt` | `sc0 sc1` | `sc0 sc1`, volatile |

Pointer `.cv` / `.wt` accesses can serialize with `vmcnt(0)` waits; buffer ops set policy bits
without that pointer-path volatile spelling. A bit pattern does not promise a speedup.
Read-once streams and reused expert weights have different needs. MiniMax-M2 fp8 E256 selects
`.cg` only for jobs not flagged with `count > BM` (S50); DeepSeek-V2 uses a token threshold (S51).
These are source dispatch choices, not universal cache rules. Measure the committed hint too,
preserving flush, graph replay and reuse conditions. Stores need their own consumer/reuse analysis.

## 7. Programs, waves and independent loads

P programs × W waves × U independent 16-byte loads per lane gives nominal P·W·U KiB of requests.
This is a planning estimate, not a bandwidth predictor: equal products distribute work differently
across CUs/waves and can require different live registers. Do not infer a P/256 scaling law.
Check that U loads really precede their first wait; consuming each load immediately can serialize them.
A `[U,2048]` bf16 tile in `BlockedLayout([1,8],[1,64],[1,4],[1,0])` places U vector runs per lane.
Sweep grid, waves and depth separately, checking spill, residency and numerical output.
Split-K, KV split and prefetch depth are coupled but not interchangeable: splits add partial-output
and merge work, depth changes live registers or LDS slots. Equal split×depth need not give equal time.
Choose each candidate's valid tile and budget per shape ([../tuning.md](../tuning.md) §3, §4).

## 8. Getting a tile into LDS

Register staging holds loaded values in VGPRs before storing to LDS; async copy transfers directly
to LDS and allows groups to overlap later work. Which wins depends on the consumer and scheduling.
`allocate_shared_memory(..., value)` allocates and stores in one op (S17, S26).

**Direct to LDS.** `buffer_load_to_shared` lowers to one `buffer_load_dwordx4 … offen lds` per 16 bytes per lane (S5,
S25); `commit_group` closes a group (an `asyncmark`), and `wait_group(n)` waits until at most `n` committed groups of
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
and 2 `ds_read_b128`. Rules:

- **Lanes write LDS in order**: lane i of an x4 instruction writes byte 16·i of its 1 KiB
  ([ISA §3](../../hardware/gfx950/memory_and_sync.md)), so the copy layout hands each lane the 16 bytes landing there
  (above, 8 lanes per 128-byte row; for a swizzled or padded destination see §9), 128 or 32 bits per lane (S20).
- **Wait before every read of a copied slot.** Neither the `s_barrier` nor `.load` waits for a copy, `wait_group` does:
  without it the kernel above read stale LDS behind the `s_barrier`.
- **Commit every copy before waiting on it.** `wait_group` counts committed groups only: an uncommitted copy was not
  waited for, even by `wait_group(0)`, although the docstring says it is (S25).
- **Copies complete in order with other vector-memory accesses** (S25): a scale load issued after a copy compiled to
  `vmcnt(1)` before its use, which retires the copy, and issued before it, `vmcnt(3)`, leaving the copy in flight. Loading scales before the DMA prologue is an A/B candidate, not a measured performance win: the
  timed toy gained nothing. The n = 7168 source uses early loads (S27), not tested by that toy.
- No `other` on copies (§4). DMA costs and prefetch: [../../gemm/datapath.md](../../gemm/datapath.md) §5, §7.

The 8-token Qwen2-MoE router has async-LDS hidden staging and a register-load/conversion path (S28).
These are complete dataflow variants; their source presence does not isolate LDS cost.

## 9. LDS destinations: swizzled, padded, transposed

**Swizzled: pre-swizzle the global side.** A DMA cannot scatter lanes inside LDS on CDNA4, so a copy into a
`SwizzledSharedLayout` with `max_phase > 1` makes the compiler swizzle the *source*, one `ds_bpermute_b32` per copy
instruction (S10). AKLR's DSA kernels, two gated-attention files and 10 fp8 GEMMs XOR the global offsets
instead and copy into an unswizzled view. `SwizzledSharedLayout(vec, per_phase, max_phase, order)` stores
element (row, c) at column `c ^ ((row // per_phase) % max_phase * vec % width)` of its row (S47); XOR being its own
inverse, the LDS column `c` a lane fills takes global column `c ^ ((row // per_phase) % max_phase * vec)` when
`max_phase × vec` fits in the row. Here K and V are `SwizzledSharedLayout(16, 1, 8, [1, 0])`, and the
kernel XORs its `vec = 32` K buffer with `* 32` (S19); a wrong multiplier compiles and loads the wrong chunks:

```python
physical_dims = dims[None, :] ^ ((tokens[:, None] % 8) * 16)
offsets = slots[:, None] * 256 + physical_dims
offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
linear_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
key_destination = key_shared._reinterpret(layout=linear_layout)
```

The flat-alias route avoids source-lane swizzling when its offset map matches the consumer.
No verifier checks that XOR map. Select padding/swizzle from physical addresses and operand reads,
not class names. `gl.bank_conflicts` can abort on gfx950; use addresses, ISA and hardware counters
([layouts.md](layouts.md) §4, §5; [../../patterns/lds_bank_conflicts.md](../../patterns/lds_bank_conflicts.md)).

**Padded.** `compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)`
returns a padded layout or `None` (S3). For bf16 A, 16×16×32 MFMA v4, `k_width=8`, `[32,64]`
gives `[[512,16]]`; smaller row tiles may return `None`. Total bytes alone do not ensure helper success.
Packed fp4 uses `gl.uint8`. DMA vector width is capped at `min_interval / 64` elements per lane
(S14), still subject to the supported 4-/16-byte widths. Split `offset_bases`: first
log2(16 / element bytes) bases for one lane's 16 bytes, next 6 for lanes, next log2(num_warps)
for waves, rest for registers (bf16 S6; fp8 S33). The x4 recipe needs at least 1024 bytes per wave
and the correct destination lane order. Code and restrictions: [layouts.md](layouts.md) §4, §6.4.

**Transposed reads.** How B is stored decides the LDS read. Compiled, bf16 `[64, 64]`, `instr_shape [16, 16, 32]`: A
`[M, K]` (K contiguous) → `ds_read_b128`; B stored `[K, N]` (N contiguous), read as `DotOperandLayout(1, …)` → 8
`ds_read_b64_tr_b16`, gfx950's transposing read; B stored `[N, K]`, read through `.permute((1, 0))` → 4 `ds_read_b128`;
fp8 B `[K, N]` → `ds_read_b64_tr_b8`  (S29). Both work: K-contiguous B takes half the read instructions,
the gated attention decode reads its fp8 V (`[tokens, dims]`) this way (S19). Lane maps, cost per MAC:
[../../gemm/datapath.md](../../gemm/datapath.md) §9, [../../attention/lds.md](../../attention/lds.md) §1.

## 10. LDS allocation, views, reuse, budget

Views on a `shared_memory_descriptor` (S17, S26): `.index(i)` (an int32, runtime allowed) a slot of the leading
dimension; `.slice(start, length, dim=0)` (Python ints) a subview Membar tracks; `.permute`, `.reshape`,
`._reinterpret(dtype, shape, layout)` (the same bytes); `.load`, `.store`, `.gather`, `.scatter`; `._keep_alive`, a
use that keeps the buffer live to that point.

**Rings.** Build a static ring as a tuple of separate allocations, `bufs = ` then `bufs += (...,)` per slot (S6; a
list with `.append` failed to compile ), indexed in `gl.static_range`, which the frontend unrolls with a
constant loop variable ([../tuning.md](../tuning.md) §10); for a runtime slot use one `[SLOTS,...]` allocation and
`.index(i)`, in the rank-2 layout of one slot ([layouts.md](layouts.md) §4). Membar tells `.slice` subviews of one
allocation apart, not its `.index` slots (S30): copies into `.index(0)` and `.index(1)` of one allocation
got an `s_barrier` between them, into two `.slice` halves none.

**Reuse.** `_reinterpret` gives a dead buffer a new type and shape. Allocate late: the allocator overlaps buffers whose
live ranges do not meet, and the 64-head DSA kernel allocates its probability tile only after its reduction buffers die
(S31). Liveness is use-based (S32): `_keep_alive` after a later conversion extended a buffer's range, right after its
last read it changed nothing, and the n = 3072 GEMM's on its last ring slots (S33) changed neither its LLVM IR nor its
LDS. A `convert_layout`'s LDS scratch laid over a dead buffer costs a barrier: keeping the buffer alive
past the conversion removed one `s_barrier` (3 → 2) for twice the LDS (16 → 32 KiB).

**Budget.** 160 KiB per CU on CDNA4 (S20), and per workgroup: a 167936-byte kernel compiled, then failed at launch; read
`shared` in the metadata first ([../debugging.md](../debugging.md) §2, §4). Workgroups per CU ≤ ⌊160 KiB / LDS per
workgroup⌋ (estimated): 3 for the one-wave m = 16 GEMM (083b `_private_k128_gemm`, 49152 bytes) whatever `waves_per_eu`
says, 1 for the 64-head DSA `_attention_split` (163840 at 24 tokens, 156160 at 12 and 16, 157696 at 32).

## 11. Synchronisation

**What Membar inserts** (local barriers only, S30): a barrier between conflicting LDS accesses (read-after-write,
write-after-read, write-after-write) of `.store`, `.load`, `allocate_shared_memory(value)` and async copies; and one
right after a `wait_group` unless a sync point follows anyway, since `wait_group` alone does not synchronise the CTA
(S30, S34). Compiled, 4 waves, 2 copies per group: `wait_group(1)` → `s_waitcnt vmcnt(2)`, `s_barrier`, still one with a
`gl.barrier` after it. For a copy, that barrier orders the instructions but does not wait for the data:
a `.load` with no `wait_group` before it compiled to `s_barrier` and `ds_read_b128` with no `vmcnt` wait.
Nothing for global memory, and nothing between an async copy and a `load_shared_relaxed` of the same buffer.

**`load_shared_relaxed`: the write-after-read is yours.** It tags the load `ttg.amdg.syncedViaAsyncWait` (S25); the AMD
filter then drops the dependency between that load and any async copy of the same buffer, in either order (its header
warns this is wrong for a single buffer), and the lowering puts the relaxed `ds_read` in a no-alias scope against async
copies, so LLVM may schedule a refill of the slot ahead of it (S35). Compiled, a relaxed read and then a refill of the
same slot, no barrier, issued the refill first, with 4 waves and with 1; a `gl.barrier` between them restored the
order. What it buys: Membar treats the `.index` slots of one allocation as one buffer (§10), so a plain
read of one slot got a barrier before a copy into another, 3 `s_barrier` against 2 relaxed; with separate allocations,
or a `gl.barrier` of your own before each refill, the two compiled the same, waits included. So read
with `.load` by default and relaxed only for the `.index` slots of one allocation, as the n = 7168 GEMM does at M = 32
(separate allocations and `.load` at M = 64, S27); then order each refill after the last read of its slot: a
`gl.barrier` or ring slack (§12), and in a one-wave kernel, where `gl.barrier` only orders the compiler, an explicit
retire of the reads, as the one-warp MiniMax router's inline `s_waitcnt lgkmcnt(0)` with a `~{memory}` clobber (S36).
The ISA does not say whether a DMA's LDS write can overtake an earlier `ds_read` of the same wave.

**`gl.barrier`** is a `ttg.barrier` over every address space (S17, S37), lowered to a release fence, `s_barrier` and
an acquire fence (S29); with 4 waves it compiled to `s_waitcnt lgkmcnt(0)` and `s_barrier`, no `vmcnt` wait, so an async
copy or a `global_load` before it stays in flight, and with 1 wave to no `s_barrier` (flat work-group size `1,64`, S1). A cross-wave exchange in rounds avoids the write-after-read barrier with a constant `.slice` per round:
4 rounds of a 4-wave exchange compiled to 4 `s_barrier` with a slice per round and 7 with one reused region
with one region of LDS per round; MiniMax-M2's top-k does this (S23).

**Global memory inside a CTA.** Membar ignores it, leaving no `s_barrier` between a `global_load` and a `global_store`
of the same data : waves that overwrite what others read need their own `gl.barrier`.
Use a workgroup barrier before overwrite when another wave can still read the old data; all
participating waves must reach it. This does not synchronize workgroups. Qwen3-Next prefill's conv-history update has `if NW > 1: gl.barrier` (S38). Redundant-thread store predicates exist (S10), but register copies are not a universal once-only guarantee ([layouts.md](layouts.md) §2). Across workgroups, AKLR kernels order producer and consumer by kernel boundaries. **Raw `vmcnt` waits:** DSA waits for part of a commit group with inline `s_waitcnt vmcnt(N)` (S13), counts of the copy instructions per panel that go stale silently when the instructions after a copy change, where `wait_group` recounts ; prefer one commit group per panel ([wave_ops.md](wave_ops.md); speed not compared).

## 12. Software pipelining by hand

**One slot per K group.** The m = 16 fp8 GEMM keeps `PREFETCH` K groups ahead per wave, one LDS slot for each of its 12
groups, none reused (S39). Count waits in committed groups by replaying the issue order (§8): with `PREFETCH` groups
committed ahead of group g, `wait_group(PREFETCH - 1)` lands g. At m = 16 it reads the next two groups into registers
early (`READ_AHEAD = 2`, S39). Depth changes outstanding groups and live registers; measure it
with the slot budget. Depth per Point, and the single buffer of m ≥ 1024:
[../gemm.md](../gemm.md) §5.

**A ring of reused slots.** A refill must not overtake the reads of its slot. With plain `.load` Membar sees to it and
the ring size only decides the cost: with `SLOTS = PREFETCH` it put an `s_barrier` between each read and its slot's
refill, 10 against 6 over 6 stages, while with `SLOTS = PREFETCH + 1` the barrier after the next `wait_group` already
separates them  (S27). With relaxed reads, count the barriers between a slot's last read and its refill
yourself. A refill issued after the step's reads needs one spare slot (without it, a refill copy was issued before its
slot's reads ); one issued at the top of the step, before its wait, needs two, `DEPTH = AHEAD + 2`, as the
Qwen2-MoE router does with `AHEAD` 2 (S6), or a `gl.barrier`
before the refill (S46):

```python
for stage in gl.static_range(-AHEAD, STAGES):
    if stage + AHEAD < STAGES:
        _prefetch_stage(a_buffers[(stage + AHEAD) % DEPTH], b_buffers[(stage + AHEAD) % DEPTH],...)
    if stage >= 0:
        gl.amd.cdna4.async_copy.wait_group(min(AHEAD, STAGES - stage - 1))
        gl.barrier
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffers[stage % DEPTH], a_dot)
```

**A rolled ring for long K** keeps the code small ([../gemm.md](../gemm.md) §3). `wait_group(n)` takes a Python int
(S4): `n` from an argument or a `range` variable failed at trace, "TypeError: create_async_wait_group: incompatible
function arguments" , so a rolled body waits a constant. The 2232-token Qwen2-MoE router is written so
(S53): one `[STAGES, …]` allocation per operand (§10); a static prologue of `STAGES - 1` committed groups; a `range`
over blocks of `gl.static_range(UNROLL)` steps, each refilling slot `(tile + STAGES - 1) % STAGES`, committing, waiting
`STAGES - 2`, calling `gl.barrier`, reading slot `(tile + 1) % STAGES` relaxed and multiplying the operands read the
step before; the leftover steps, unrolled (there all 29); a static drain waiting `STAGES - 3 - tail`. Written so, an fp8
block-scaled GEMM at M = 32, N = 4096, K = 7168 (2 waves, 16 columns, 6 slots, one step per trip) compiled to 325
instructions against 1417 unrolled, 56 VGPRs, 44 KiB of LDS and one `vmcnt(12)` wait per step, its scales from LDS or
`s_load`s (§13); a weight scale `gl.load`ed at its group added `vmcnt(0)` before the fold, and a tuple of scales loaded
with their groups' copies (none grows in a `range` loop) `vmcnt(3)`, unless unrolled by its length.

**Register look-ahead.** Operands loaded straight into the operand layout (§13) need no LDS. MiniMax-M2 fp8 keeps up
to three K steps' operand tuples (A, B and both scales) in flight and rotates them; the rotation is loop-carried, so
the loop stays a rolled `range` (S48):

```python
for step in range(groups - 3):
    na, nb, nsa, nsb = _load_fp8_tile(..., split * groups + step + 3,...)
    acc = _accumulate_fp8(a, b, sa, sb, acc)
    a, b, sa, sb = (a1, b1, sa1, sb1)
    a1, b1, sa1, sb1 = (a2, b2, sa2, sb2)
    a2, b2, sa2, sb2 = (na, nb, nsa, nsb)
```

The depth is set per Point (up projection: 2 tuples at 1, 4 and 16 tokens, 1 at 2, 3 at 8; S48). No timing isolates it;
in its `_project_fp8` the second tuple cost 16–24 VGPRs, the third none.

## 13. Weight and scale streaming

**Preshuffled weights straight into the operand layout.** The kernels compute the offset of AKLR's `Preshuffle(16, 16)`
weights (aiter's 16-row tiles of 16-byte vectors; offset: [mfma.md](mfma.md) §7) and build its index vectors in slices
of the B operand layout, so the loaded tensor is the MFMA operand, no LDS: 16 bytes per lane, 1 KiB per wave (derived;
S41; code and DMA with a view: [layouts.md](layouts.md) §6.3). In this MiniMax-M2 file, with waves splitting N,
shared A goes through LDS: the helper's padded layout at `BN == 128`, a conversion otherwise; B stays wave-private
(S41); not a universal N-split requirement or a timed LDS comparison. Shipped layout: `aklr/ops/README.md` § The kernel section and ABI.

**Scales.** Load packed bytes as uint32 words and shift; masked words can use `0x7F7F7F7F`
(four unit e8m0 scales, S11). SMEM counts in `lgkmcnt`, not `vmcnt`; a VMEM scale load after
copies can force their retirement before scale use (§8). Earlier loads are a scheduling candidate,
not an established performance win. Uniform `gl.load` can become `s_load`, but textual position
alone is insufficient: a rolled body containing copy/store may retain global data loads.
Inspect amdgcn. The m=32 fp8 GEMM uses a constant-address-space pointer (S55):

```python
address = gl.inline_asm_elementwise("", constraints="=s,0", args=[WS.to(gl.uint64)],
                                   dtype=gl.uint64, is_pure=True, pack=1)
scale_ptr = address.to(gl.pointer_type(gl.float32, 4))
```

This needs a uniform address and valid constant-space access; the cast alone is not a substitute
for the SGPR constraint. Alternatively stage a per-group table in LDS, synchronize after staging,
then `.gather` words from a flattened group-major table (S54). A rank-2 table's `.index(step)`
retains its rank-2 layout and can fail on the rank-1 result; use `.gather` or a correctly ranked view.

## 14. XCD and L2 locality

gfx950 has eight XCDs with separate L2 caches. In measured grids, ids congruent mod 8 mapped
to one XCD; verify placement for the actual grid/partition instead of assuming universal scheduling.
Group readers of the same panel within an XCD, but a permutation with one reader per tile adds no
reuse by itself. A remap must be a bijection for every launched grid, tails included.
DeepSeek-V2's large-token router walks eight M tiles per column (S42); the fp8 GEMM uses `GROUP_M`
(S43). Derive which activation/weight panels are shared before measuring. A benchmark flush does
not prove either complete eviction or universal cross-call survival
([../../gemm/schedule_and_locality.md](../../gemm/schedule_and_locality.md) §2).

## 15. Epilogue stores

Round first, move through LDS to 16 contiguous bytes per lane, store with the buffer op (S33):

```python
result = acc.to(gl.bfloat16)
if BLOCK_M == 32:
    exchange = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], gl.SwizzledSharedLayout(8, 2, 8, [1, 0]))
    exchange.store(result)
    result = exchange.load(out_layout)
gl.amd.cdna4.buffer_store(result, output_base, rm[:, None] * 3072 + rn[None, :], cache=".wt")
```

`out_layout` is `BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])`; rounding first halves the exchange's LDS bytes; at
`BLOCK_M = 64` the exchange is a `SharedLinearLayout` whose bases XOR rows into the 16-byte chunk (S33). To reach this 16-byte-per-lane store width, exchange the result
when the accumulator's layout directly stores less than 16 bytes per lane: from `AMDMFMALayout(4, [16, 16, 32], transposed=True)`
fp32 stored 16 bytes per lane directly and bf16 8, from `transposed=False` 4 and 2. A `gl.convert_layout` to
`out_layout` makes the same exchange in a buffer the compiler lays out, with 2-byte LDS accesses from a
`transposed=False` source ; write it yourself to choose that buffer's layout. Neither was timed against
the direct store ([layouts.md](layouts.md) §6.6), nor was this `.wt`, free on a `buffer_store` write stream (§6).

## Evidence

- S1 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:327 `gluon_to_ttgir` 327-348 (unroll 337, warp pipeline 339); `make_ttgir` 260-325 (coalesce 269, `num_stages` 288-289, buffer-op conversion 303-312); Gluon takes `gluon_to_ttgir` 584-589; "S" → `tt.pointer_range` 211-240; async copy on for gfx950 32-33; `make_llir` warp-pipeline conversion 357; flat work-group size 460
- S2 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:111 Membar with `membarFilter` in the lowering (111-116)
- S3 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/__init__.py:12 re-exports the buffer ops; `compute_efficient_padded_shared_layout` 121-163 (`None` 139-143)
- S4 triton-lang/triton@c01b677:python/src/gluon_ir.cc:1097 `create_buffer_load(type, ptr, offsets, mask, other, cache)`, `create_buffer_store` 1104; commit and wait builders 727-736, the wait taking an `int num` (733)
- S5 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp:104 `num_records` = INT_MAX − 1; select unless `other` is a zero constant 121-122; DMA via async LDS intrinsic 126-149; 32-bit byte offset, masked lanes INT_MAX + 1 280-283
- S6 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:104 scalar-base origins 104-108; `_copy_distribution` 37-49; padded ring 87-99, a tuple built with `+=` 93-97; `DEPTH` 72; uint32 program id 76; ring 113-128; 256-token config 34
- S7 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/3d0f64ffff6790c8.json#attention.py:231 byte-extent gate (231-237)
- S8 triton-lang/triton@c01b677:python/triton/language/semantic.py:922 cache strings (922-950); `store` casts to the pointee type 1243
- S9 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MaskedOpsToLLVM.cpp:81 constant-true mask → straight load, else a branch (81-109); stores the same (151-171)
- S10 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:263 mask alignment caps `vec`; `other` packed per vector 697-705; swizzled `dest` → source swizzling 788-803; DMA `other` → branch + LDS store 840-874; redundant-thread store predicate 1646-1657; `async_wait` → `wait.asyncmark n` 2412-2431; commit → `asyncmark` 2503-2505
- S11 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/84116a96e535f174.json#kernel.py:163 scale words, `other=2139062143` 163-171; `other=127` 181; folding masks 140-142; `M: gl.constexpr` 118
- S12 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/ca826892315e0a3f.json#kernel.py:127 clamped read rows (125-127); `M: gl.constexpr` 78
- S13 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:276 `-2147483648` sentinel; `_wait_vmem` 145-153, used 157-166
- S14 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:716 contiguity = min(per-thread run, offset alignment, base divisibility), vector size 758-763; `canLoadDirectToLDS` 817-891: padded DMA cap 820-831, a provable vector below the per-lane run into LDS 868, unsupported width 879, lanes not writing in order 885; pointer-op flags 290-337; buffer `aux` bits 456-506
- S15 triton-lang/triton@c01b677:lib/Analysis/AxisInfo.cpp:841 contiguous `<` splat: constancy gcd(contiguity, divisibility, divisibility) (841-856); mask alignment = constancy 1446-1455; `//` contiguity 1 unless by 1 418-425; `%` by a constant gcd(contiguity, divisibility, divisor) 488-510; a tensor-condition select gcd of both branches and the condition's constancy 964-967; XOR no contiguity rule 994-1016; a function argument's divisibility 1 unless `tt.divisibility` 1309-1336
- S16 triton-lang/triton@c01b677:python/triton/backends/compiler.py:78 "D" = divisibility 16, for ints with `arg % 16 == 0` (78-86)
- S17 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:128 `tl` re-exports (`load` 128, `max_contiguous` 131, `multiple_of` 135, `store` 144); descriptor `gather` 358, `slice` 446, `index` 464, `permute` 478, `reshape` 492, `_reinterpret` 507, `_keep_alive` 527; `allocate_shared_memory` 613; `barrier` 703
- S18 triton-lang/triton@c01b677:python/triton/language/core.py:3308 `multiple_of`, `max_contiguous` 3335: hints; `load` 2479, `store` 2560
- S19 aklr/plugins/sglang/models/qwen3_5/impls/triton/gated_attention_decode_group_fp8_quant_qkv_dim4608_n_kv_heads1_head_dim256/14e91857b4038e02.json#attention.py:110 pre-swizzled offsets into a linear view (110-117); swizzled buffers 202-211; V read as operand 1, 159 (layout 69); `SMALL_BATCH` branch: K in `SwizzledSharedLayout(32, 1, 8)` (203-205) XORed with `* 32`, V with `* 16` (91-96)
- S20 triton-lang/triton@c01b677:third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp:202 CDNA4 DMA widths 128 and 32 bits, 8, 16 and 96 disabled (202-204); LDS 160 KiB 144-145; transpose-read parameters 161-185
- S21 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/299133886d479017.json#kernel.py:116 state `buffer_load`/`buffer_store` with `.cs` (116-118, 141-143)
- S22 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:513 `WEIGHT_CACHE="" if m == 1 else ".cg"`
- S23 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1d277d76892ee692.json#kernel.py:711 `WEIGHT_CACHE="" if small_batch else ".cg"` on the down-projection launch (691-715); the gate-up launch passes none (663-688, default "" 221); top-k: one `.slice(slot * 4, 4)` per round, `gl.barrier`, relaxed reads (543-554)
- S24 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/2422a87b3b814299.json#kernel.py:434 `.cg` off when `M > 128 and UP and (TM == 128)`
- S25 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/amd/cdna4/async_copy.py:71 `buffer_load_to_shared` (in-order completion 77-80; 128/32 bits, coalesced, warp-local swizzle 91-94); `global_load_to_shared` 16-29, its conditions 31-39; `wait_group` docstring 137-141; relaxed-load attribute 164-168
- S26 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:252 `allocate_shared` with a value is one `local_alloc` (252-262); `slice` needs ints 411-414; `index` takes an int32 425-428
- S27 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/06755ba3ee5eb085.json#kernel.py:161 `SLOTS`/`PREFETCH`/`WAITS` 161-166; uint32 program id 167; m = 64 permutation 168-172; one arena with `.index` at M = 32, separate allocations otherwise 189-199; every program reads the whole activation panel 182-188, 221-227; scales, then the copy prologue 206-227; ring 229-258, relaxed reads at M = 32 and `.load` at M = 64 242-249
- S28 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:29 `ASYNC_HIDDEN = ROWS <= 8`; load layouts 30-35; both paths 48-73
- S29 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:476 CDNA4 `ds_read_tr16_b64`/`ds_read_tr8_b64` (476-486); barrier: release fence, `s_barrier`, acquire fence 720-736
- S30 triton-lang/triton@c01b677:lib/Analysis/Membar.cpp:23 only `memdesc_subslice` offsets tracked, else overlap (23-49); barriers are `AddrSpace::Local` 241-245; after a wait unless a sync point follows 269-298; on conflicts 381-385
- S31 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048/2ee56a6dbabc48a6.json#kernel.py:450 probability tile allocated late (450-454)
- S32 triton-lang/triton@c01b677:lib/Analysis/Allocation.cpp:419 live ranges from MLIR `Liveness` over all uses (419-436)
- S33 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/3480e97a5fcda6f6.json#kernel.py:312 bf16 exchange through LDS, `SwizzledSharedLayout(8, 2, 8, [1, 0])` at `BLOCK_M == 32` 315-322, a `SharedLinearLayout` otherwise 323-349; `buffer_store(..., cache=".wt")` 354-359; `_keep_alive` of the last slots 361-362; `_copy_layout`: `bases[:4] + bases[12:]`, `bases[4:10]`, `bases[10:12]` 20-29
- S34 triton-lang/triton@c01b677:include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47 `async_wait` "without synchronising CTA execution" (47-55)
- S35 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/MembarUtility.cpp:11 async copy ↔ synced local load filter (11-53); single-buffer warning in MembarUtility.h:9-28; synced local loads put in a no-alias scope against the async copies in AsyncUtility.cpp:117-137, copies tagged in LoadStoreOpToLLVM.cpp:854-855, 864-865
- S36 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/94af11a072175a38.json#kernel.py:483 `_router_async_bf16` (one warp, 444-445): relaxed reads 477-482, inline `s_waitcnt lgkmcnt(0)` 483-490, refill of the same slot 491-499
- S37 triton-lang/triton@c01b677:python/src/ir.cc:1853 `create_barrier` → `BarrierOp(AddrSpace::All)` (1853-1856)
- S38 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_prefill_group_fp8_quant_max_seq_len5120_n_v_heads8_key_dim128_value_dim128_conv_dim2048/98af1fd2ff621082.json#kernel.py:115 barrier before overwriting the history (115-118)
- S39 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/083b493a1861424c.json#kernel.py:363 prologue 363-376; read-ahead variant 379-487; ring without read-ahead 489-539; copy group and commit 130-142; `depth` 1002, `read_ahead` 1004 and 1013, passed at 1021 and 1029
- S41 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:36 `_weight_offset_8` 36-37; index vectors 657, 666; B load 700-706; A through the helper's padded layout at `BN == 128` 685-691, 722-724, `convert_layout` otherwise 725-726
- S42 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/699f9fa93bf64b26.json#kernel.py:103 `_grouped_tile` 103-107, called with `GROUP` 8 only for `M > 4096` 176-183; `(pid % ROWS, pid // ROWS)` otherwise 191
- S43 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/5ea30df18b03ab8d.json#kernel.py:628 `GROUP_M` order 628-634; `group_m = 2 if m <= 2048 else 4` 1093
- S44 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/bb663e5b0cf55a10.json#kernel.py:200 `W + expert.to(gl.int64) * (N * (K // 2))`, `expert` from a scalar `gl.load` 197; 64-bit column rebase 224-226; `buffer_load(column_weight_base, boff.to(gl.int32))` 236
- S45 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:521 `v_readfirstlane_b32` (`=s,v`) on a job descriptor reduced with `gl.max` (520-523)
- S46 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/cd47a5e858af6ab4.json#kernel.py:87 `if DEPTH <= AHEAD + 1 and stage >= DEPTH - AHEAD: gl.barrier` before the refill (87-89)
- S47 triton-lang/triton@c01b677:lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:83 swizzled shared layout: row basis `(vec * ((row / perPhase) % maxPhase)) % numCols` (76-84)
- S48 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1c1747a25af67b8e.json#kernel.py:412 `PREFETCH == 3`: two more tiles loaded up front 412-450, the rotation 451-477; `_load_fp8_tile` 199-243; depth per Point in `_projection_schedule` 893-910
- S49 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/c5244d02949f27b4.json#kernel.py:728 `REBASE`: the K step in the scalar bases or in the offsets (728-757)
- S50 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/8a5a03b603ec1b70.json#kernel.py:218 job flag `count > BM`; flagged jobs run with `.ca` (no bits), the others with `.cg` (674-720)
- S51 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim512/a841007c49712392.json#kernel.py:686 gate-up weights `.cg` only for `M <= 1536` (686-690); its Point is 2048 tokens
- S52 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/7266cdc0ff1fd299.json#kernel.py:107 zero `other` tensors `az`/`bz` 107-108; masks all-true unless `M % BM != 0` 109-112; the copies 114-115
- S53 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4c2bed7e3e8c1e94.json#kernel.py:121 the rolled body `range(MAIN_TILES // UNROLL)` around `gl.static_range(UNROLL)` 121-128, `UNROLL` 32 (60), `MAIN_TILES` 115; `_prefetch_step` 21-47: refill of `(slot + STAGES - 1) % STAGES` 28-36, commit 37, `wait_group(STAGES - 2)` 38, `gl.barrier` 39, relaxed reads of `(slot + 1) % STAGES` 40-45, MFMA on the previous step's operands 46; `[STAGES, BM, BK]` and `[STAGES, BK, BN]` allocations 102-103; prologue 105-112, first wait and reads 116-119; leftover steps 129-135; drain `wait_group(STAGES - 3 - tail)` 136-143; its Point, 2232 rows: 4 stages, split-K 2 (302-303)
- S54 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/82e9371086a875bd.json#kernel.py:600 activation scales of group `step`, `scale_smem.gather(step * BM + gl.arange(0, BM, gl.SliceLayout(1, mma)), 0)`, in the rolled `range(H // 128)` (591); the table stored flattened group-major 552-557, weight-scale tables 558-573, `gl.barrier` 574; weight scales `gl.sum(wg_smem.gather(wi, 0), 0)` 601-603; the `[2, BM, 128]` ring 537, `wait_group(0)` 592, `gl.barrier` and the relaxed read of `.index(step % 2)` 594-595, refill of `.index((step + 1) % 2)` 606-620; `RING` for `m >= 4096` (942)
- S55 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/fcf21622f6c91bbb.json#kernel.py:145 `address.to(gl.pointer_type(gl.float32, 4))`, the address put in an SGPR by inline asm: `""` with `"=s,0"` (128-135) or `"s_mov_b64 $0, $1;"` with `"=s,s"` (137-144); 12 scalar scale loads through it 146-149
