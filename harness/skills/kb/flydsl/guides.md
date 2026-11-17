---
title: "FlyDSL's own skills at v0.3.2: links, and what differs on gfx950 or what this repository adds"
kind: language-note
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
sources:
  - ROCm/FlyDSL@dd8373349:.claude/skills/gemm-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/lds-optimization/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/prefetch-data-load/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/flydsl-kernel-authoring/SKILL.md
  - ROCm/FlyDSL@dd8373349:.claude/skills/kernel-code-cleanup/SKILL.md
  - ROCm/FlyDSL@dd8373349:python/flydsl
  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py
---

# FlyDSL's own skills

FlyDSL ships its own agent skills; this repository links them at `v0.3.2`, the version it pins, and does not copy
them. Each section gives the link, what to read it for, and where it is wrong or incomplete on gfx950 with FlyDSL
v0.3.2, or what this repository adds. A `§` names the skill's own section. Paths are relative to this folder.

## gemm-optimization

FlyDSL's guide: <https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/gemm-optimization/SKILL.md>. Read it for how FlyDSL's production preshuffle GEMM is built: tiling, the LDS ping-pong, the XOR swizzle, prefetch, `hot_loop_scheduler`, epilogues, main-loop instruction counts and the register budget. Where it differs on gfx950 with FlyDSL v0.3.2, and what this repository adds:

- **Whole guide**: what Primus-Turbo's FlyDSL GEMMs and measured GEMM work add to this guide (other loop designs, dispatch rounds, split-K, store classes, correctness gates, rulers) is in `../gemm/INDEX.md`.
- **§2–§7 (the code excerpts)**: the excerpts show the structure, not the current source. `kernels/gemm/preshuffle_gemm.py` is written with the layout API, and the helpers the excerpts name are in `kernels/common/mma/mfma_preshuffle_pipeline.py` (`swizzle_xor16`, `lds_store_16b_xor16`, `make_preshuffle_b_layout`) and `kernels/common/mma/mfma_epilogues.py` (`c_shuffle_epilog`). There is no separate A0 prefetch step (§4.3): how many LDS reads and global loads go ahead of the first MFMA is the `preload=(dsrd_preload, dvmem_preload)` argument; the spreading of §5.5 is written inline in `hot_loop_scheduler` (with a local `build_scheduler`, not `_build_scheduler`), and on gfx950 it is even for the LDS reads but not for VMEM loads fewer than the MFMAs, which go one after each of the first MFMAs (`kernels/gemm/preshuffle_gemm.py:454-460`); and `preshuffle_gemm.py` stores C directly, while a CShuffle epilogue (§7.2) is in `kernels/gemm/gemm_a16w16_gfx950.py` (`write_cshuffle_vec_to_global`).
- **§3.2 XOR Swizzle Formula**: the XOR is per-lane VALU, not one SALU instruction, because the row index is per lane: about 2 VALU instructions per LDS address recomputed on every access, by hand or composed into the layout alike *(field, E1)*; where the row is loop-invariant (`preshuffle_gemm`'s main loop) the compiler hoists it into prologue VALU and VGPRs (`../gemm/datapath.md` §2). The same applies to the "SALU (address, swizzle)" row of §9.2: offset computation is SALU/VALU, and the XOR swizzle is per-lane VALU.
- **§4.1 A Matrix: Global → LDS** (and **§11 Async Copy**): in this preshuffle GEMM, `use_async_copy=True` requires gfx950: v0.3.2 raises `ValueError` on any other arch (`kernels/gemm/preshuffle_gemm.py:154-157`). For this kernel, §11 therefore describes gfx950 only.
- **§6.2 MFMA Instruction Selection**: on gfx950, BF16 and FP16 use K=32: `mfma_f32_16x16x32_bf16` and `mfma_f32_16x16x32_f16`, f32×4. v0.3.2 picks K=32 for BF16/FP16 on gfx950 (`kernels/gemm/preshuffle_gemm.py:160, 732-733`); the table's K=16 rows are the path on other archs.
- **§7.2 CShuffle Epilogue**: use it for large tile_n (≥ 128) where output coalescing matters, and race it against the direct store on narrower tiles too (measurements: `../gemm/epilogue.md`).
- **§8.3 Memory-Bound vs Compute-Bound**: instead of the rule "M ≤ 512 → memory-bound", compute the intensity for the shape: on gfx950 the ridge sits nearer M ≈ 300, and only when N and K are much larger than M (`../gemm/mainloop.md` §9).
- **§8.4 Bottleneck Identification from ATT Traces**: low MFMA utilization means memory-bound only once a bytes/time probe confirms it, not below a fixed threshold such as 50%: the utilization reading is not comparable across waves per SIMD (`../gemm/attribution.md` §9).
- **§10.2 Occupancy Impact**: the first spill at arch=256/accum=256 came from a test that grew both files together. Each file still holds at most 256 ("up to 512 total VGPRs, 256 of each type", ISA ch03 §3.6.4), so arch_vgpr demand past 256 spills however few AGPRs are in use.

## lds-optimization

FlyDSL's guide: <https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/lds-optimization/SKILL.md>. Read it for diagnosing LDS stalls from an ATT trace (bank conflicts, exposed write-to-read latency, cross-wave reduce serialization) and for its three fixes: XOR swizzle, padding, and more write-to-read distance. Where it differs on gfx950 with FlyDSL v0.3.2, and what this repository adds:

- **§LDS Architecture on CDNA4 (gfx950) › Impact on Bank Conflict Analysis** (and **§Optimization Method 1: Swizzle Layout › The Problem**): a stride of 128 bytes is still a heavy conflict on gfx950, not the "partial" or "2-way" conflict the guide describes: a `ds_read_b32` runs as two phases of 32 lanes over 32 banks (the padding bullet below), so all 32 lanes of a phase hit one bank *(field, E2)* (`../gemm/datapath.md` §3).
- **§LDS Instruction Model** (key rules 1–2 and the skill description): in the same wave, a `ds_read` after a `ds_write` needs no wait in between, because one wave's LDS operations complete in issue order ("Instructions of a given type return in order", ISA ch04 §4.4). Wait for `lgkmcnt` only before using the `ds_read`'s result. A read of what another wave wrote needs the writer's `s_waitcnt lgkmcnt(0)`, then an `s_barrier` that every wave meets (`lgkmcnt` alone is not enough).
- **§Diagnosing LDS Bottlenecks from Trace › Step 2** (Type A and Type B): a bank conflict also puts much of its stall on the `s_waitcnt lgkmcnt` after the LDS ops, so a wait spike right after `ds_write` (Type B's sign) can be a conflict, which more write-to-read distance does not help: check the stall on the `ds_*` rows first (`kernels/flydsl_authoring_method.md` § Diagnose LDS from the trace, not from intuition).
- **§Diagnosing LDS Bottlenecks from Trace › Step 2, Type A** (the sign "`ds_read2_b64` / `ds_write2_b32` with offsets that are multiples of the bank count"): on gfx950 the two offsets of one `ds_read2_b32` or `ds_read2_b64` never conflicted with each other, multiples of 64 included *(field, E4)*; look for the conflict across lanes. `ds_write2` was not measured.
- **§Optimization Method 2: Padding › The Solution, Padding Amount**: pick the padding by computing the bank of each lane, `floor((r · row_stride_bytes + col_bytes) / 4) mod 32` for lane `r` reading row `r`, not by whether the stride is fractional. On gfx950 a `ds_read_b32` behaves as two phases of 32 lanes (0–31, 32–63), each over 32 banks *(field, E2)* (wider reads: `../gemm/datapath.md` §3). A whole-dword stride touches `32 / gcd(row_stride_dwords, 32)` banks per phase, so an odd dword stride is conflict-free at every column, and for 2-byte elements **+2 elements** (`PADDING = 2`) is conflict-free on 64- and 128-element rows. +1 only suits 2-byte reads (it breaks 4-byte alignment), and on 64- and 128-element rows it puts lanes 2k and 2k+1 on one bank (2-way), so the guide's "pad 1 → 1-way" row for HEAD_SIZE = 128 does not hold.
- **§Optimization Method 2: Padding › Padding Amount** (wider reads): wider reads follow the same rule in units of the read width: with one row per lane, a row stride that is odd in 8-byte (`b64`) or 16-byte (`b128`) units keeps every gfx950 read phase on distinct banks. They also need 8- or 16-byte aligned rows, so for fp16 the pad is a multiple of 4 or 8 elements (on a 128-element row, +4 for `b64` and +8 for `b128`); an odd-element pad breaks the vector read. For column-wise reads such as MFMA fragments the `b128` phase pattern makes padding hard to use; prefer the XOR swizzle (Method 1).
- **§Swizzle vs Padding Trade-offs** (and **§Verification Checklist** item 4): the XOR is VALU, not SALU, because the row index is per lane, so swizzle is not free: about 2 VALU instructions per recomputed LDS address, or extra prologue VALU and VGPRs where the compiler hoists it (gemm-optimization §3.2 above) *(field, E1)*. Use the XOR swizzle when the access pattern is regular and LDS headroom is tight (it costs address arithmetic, not LDS). Pad when a small stride bump removes the conflict and the swizzle arithmetic would clutter the address code (`../gemm/datapath.md` §3).

## prefetch-data-load

FlyDSL's guide: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/prefetch-data-load/SKILL.md. Read it for
the shape of a software-prefetched loop (prologue load, state carried through `range(..., init=...)`, epilogue), its
PA decode example, and hoisting the next phase's loads into barrier waits. Where it differs on gfx950 with FlyDSL
v0.3.2, and what this repository adds:

- **§Three Critical Pitfalls, pitfall 1** (also the bounds in §Transformation Steps, §PA Decode Kernel Example and the
  last Don't): know which loop form you wrote. `range(..., init=...)` lowers to `scf.for` and carries only the
  values you `yield`; Python-int bounds are fine (they become index constants), as are `fx.Index`, `fx.Int32` and
  `fx.Int64` (`scf_range`, `python/flydsl/compiler/ast_rewriter.py:966-1003`). A plain `range(...)` without `init=` also lowers to
  `scf.for` and carries every name the body reassigns, so it is not traced into N flat copies either (§FlyDSL
  Implementation). Only `range_constexpr` unrolls (`debug-flydsl-kernel.md` § The three loop forms).
- **§Do, "Use `range(..., init=...)`"**: in this form a Python rebinding is not carried; only yielded values are.
- **§Do, "Unwrap all init values to raw ir.Value"**: pass FlyDSL values to `init=` directly. The rewriter converts
  `init` and `yield` values itself (`python/flydsl/compiler/ast_rewriter.py:992`, `:1305-1313`); unwrap only at hard boundaries
  (pitfall 2).
- **§Don't** (added): don't let both the `init` value and the yielded value be bare `fx.ptr_load` results. LLVM folded
  the phi of two loads into one load at the use, in one K3 MLA campaign kernel (not re-checked since), so the loop carries the address and the prefetch is gone. Pass the
  prologue load through a `.select` on a loaded value that is never false, then check that the ISA's loop carries
  the data registers.
- **§Register Budget**: the model holds on gfx950. Arch and accum VGPRs are two physical files that share one
  combined 512-entry occupancy budget per SIMD; occupancy is `512 / (arch_vgpr_alloc + accum_vgpr_alloc)`, not
  256/max. In the thresholds table the 3-wave tier is a total ≤ 168, not ≤ 170, and with AGPRs in use the arch
  count is rounded up to 4 before they are added (85 + 83 counts as 88 + 83 = 171): registers are allocated in granules
  of 8 (ISA ch03 §3.6.4; tiers 64/72/80/96/128/168/256 and the rounding in `../gemm/mainloop.md` §7). The
  Don't "Don't assume occupancy can increase" holds on gfx950 with its 512 VGPRs: prefetch buffers that push the
  total above 256 drop occupancy from 2 to 1 wave/SIMD.
- **§Register Budget, the WARNING**: do not cap registers to make room for prefetch buffers. In FlyDSL v0.3.2
  `maxnreg` does not cap them at all, and a cap set through the `amdgpu-num-vgpr` attribute turns into spills
  (`language/limits.md`).
- **§What Prefetch Can and Cannot Do**: you cannot set the counts of the waits the compiler places, but prefetch
  alone changes them: a `range(..., init=)` loop that issues the next round's `buffer_load` before computing gets
  `vmcnt(1)`, not `vmcnt(0)`, before the compute *(field, E3)*; a `vmcnt(0)` before the back edge, where the ISA has
  one, still ends the overlap at the iteration boundary. A wait you place yourself is `fx.rocdl.s_waitcnt(vmcnt=n)`
  (`gemm.md` § mainloop §8).
- **§When NOT To Use, compute-bound kernels**: a kernel is compute-bound if adding MFMA work moves the time
  (`../gemm/attribution.md` §1), not if MFMA utilization is above 90%. Utilization counters
  are not comparable across wave counts (same page §9), so do not read the regime off them.
- **§Don't, "Don't prefetch if loop body is already memory-bound"**: a loop of loads with no compute can still gain
  from prefetch when few waves are resident, so decide by occupancy (measured in HIP:
  `kernels/flydsl_authoring_method.md` § Tier 2 — memory hierarchy).

## flydsl-kernel-authoring

FlyDSL's skill: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/flydsl-kernel-authoring/SKILL.md. Read it to look up the FlyDSL API while writing or reviewing a kernel: layout algebra, copy and MMA atoms, the `fx.*` forms that replace raw dialects, environment variables and the troubleshooting list. What this repository adds for FlyDSL v0.3.2:

- **§3 Current Syntax Quick Reference**: `fx.Index` is not deprecated in v0.3.2 (only `fx.index_cast` is), but it is unsigned: use `fx.Int64` for offsets that can be negative ([limits](language/limits.md)).
- **§3 Runtime Loops with Loop-Carried Values** (pitfall 1 and the comment in the example are wrong for v0.3.2): `range(..., init=...)` always emits a runtime `scf.for`, whatever the bounds' type; only what you `yield` is carried. Python-int bounds are fine: the rewriter turns them into `index` constants and neither unrolls the loop nor drops `init=` (`scf_range`, `python/flydsl/compiler/ast_rewriter.py:982-1003`); `range_constexpr` is the form that unrolls. A plain `range` without `init=` takes a Python int or `fx.Int32` bound, and `fx.Int64` raises `TypeError` there.
- **§4 Buffer Load/Store**: `make_buffer_tensor` builds the V# descriptor for you, but by default it does not bound the tensor (`num_records = 0xFFFFFFFF`); pass `max_size=False` (size from the layout) or `num_records_bytes=` when an access past the end must be dropped ([limits](language/limits.md)).
- **§9 Key Environment Variables**: `FLYDSL_COMPILE_OPT_LEVEL` is read only into the cache key; v0.3.2 compiles at `O=2` whatever its value ([limits](language/limits.md)).
- **§9 Disk Cache Invalidation, §10 Common Issues item 3**: the JIT disk cache key covers the launcher's and kernel's source, helpers reached by closure or from the same directory, closure values that are `int`, `float`, `bool`, `str`, `None`, `tuple` or `Enum`, and FlyDSL's own sources and C++ passes. It does not see a `dict`, `list` or object captured by closure, a helper in another directory read as a module global, or an environment variable your code reads ([full list](debug-flydsl-kernel.md#what-the-cache-key-sees)). After changing one of those, rerun in a new process with `FLYDSL_RUNTIME_ENABLE_CACHE=0` or a cleared `~/.flydsl/cache/`: within one process a jit object keeps its in-memory binary either way.
- **§9 Autotune Module** (the `waves_per_eu` note): `waves_per_eu` takes effect: FlyDSL v0.3.0 and later set it as the `rocdl.waves_per_eu` attribute on each kernel as well as a binary option, and the LLVM IR carries `amdgpu-waves-per-eu`. `maxnreg` stays a binary option that does not cap registers ([version notes](language/version_notes.md), [limits](language/limits.md)).
- **§10 Common Issues item 4 (LDS overflow)**: `SharedAllocator` sizes the LDS global for you but does not check it against the limit: an allocation past it prints `local memory (…) exceeds limit`, the compile still returns and caches the code object, and the next run hits that cache silently. Check `.group_segment_fixed_size` in the ISA dump ([limits](language/limits.md)).

## kernel-code-cleanup

FlyDSL's skill: https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/kernel-code-cleanup/SKILL.md. Read it when migrating an existing kernel off legacy constructs (`ArithValue`, `buffer_ops`, raw MLIR dialects, `SmemAllocator`/`SmemPtr`, `*_atom_call`, raw `rocdl.mfma_*`) onto the current `fx.*` surface, and when trimming its comments and dead code. What this repository adds for FlyDSL v0.3.2:

- **§3 Raw upstream dialects → `fx.*` and Python, `scf`** (replaces "Runtime bounds must be typed (`fx.Int64`) or the rewriter unrolls and drops `init=`"): `range(..., init=)` is always a runtime `scf.for`: Python-int bounds become index constants. A plain `range` without `init=` takes a Python int or `fx.Int32` bound (`fx.Int64` raises `TypeError` there).

## Evidence

- E1 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1, VALU counted in the final ISA, the two builds of a pair differing only in the swizzle] XOR recomputed every iteration over 10 LDS accesses: +18 VALU written by hand, +22 composed into the layout, no extra SALU; row loop-invariant (the same kernel, and `preshuffle_gemm`'s own swizzle against mask 0 at four bf16 and fp8 tiles): 0 in the main loop, +6 to +29 VALU in the kernel, up to 18 more VGPRs (FlyDSL LDS probe kernel and `preshuffle_gemm`)
- E2 [2026-11, gfx950 (MI355X), HIP 7.15, N=7, one wave, chains of dependent reads, `clock64()` cycles per read, rocprofv3 `SQ_LDS_BANK_CONFLICT`] `ds_read_b32`: stride 1 dword 60.04 cycles; lane l at dword 2·(l % 32) + l / 32 (conflict-free as one 64-lane phase) 64.04 and 2 conflict cycles per read; lane l at dword (l % 32) + 64·(l / 32) (2-way as one 64-lane phase) 60.04 and 0; strides 32 and 64 dwords 124.01, 33 and 65 no extra cycles; `ds_read_u16`, lane r reading row r of 64- or 128-element 2-byte rows: pad 2 no conflict, pad 1 2-way (HIP chain kernels)
- E3 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=1, final ISA of each build; outputs bit-exact against a CPU float32 reference] a `range(..., init=)` loop with one `buffer_load_dword` and one add per round, runtime unrolling off: without prefetch `vmcnt(0)` before the add; with the next round's load issued first, `vmcnt(1)` before the add, then `vmcnt(0)` before the back edge (minimal FlyDSL prefetch loop, no MFMA or LDS)
- E4 [2026-11, gfx950 (MI355X), HIP 7.15, N=21 launches per configuration, minimum cycles per instruction, rocprofv3 `SQ_LDS_BANK_CONFLICT`] `ds_read2_b32` at offset1 0 to 255 and `ds_read2_b64` at offset1 0 to 192, multiples of 64 included: 0 conflict cycles and the same cycles per instruction at every offset, against 15 conflict cycles on a `ds_read_b32` control with 16 lanes 64 dwords apart (HIP microbenchmark, one wave)
