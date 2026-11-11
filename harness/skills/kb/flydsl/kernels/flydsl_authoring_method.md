---
title: FlyDSL — authoring a kernel, structure before parameters
kind: method
lever: flydsl_authoring_method
ops: [any]
gens: [gfx950]
flydsl: 0.3.2
---

# Authoring method

> **Reference (how-to), not a verdict.** The winner is decided by on-box measurement against the
> kernel's own oracle.

**Path B: you are writing a new `@flyc.kernel`.** For *reading* aiter's shipped kernels see
`flydsl_kernel_library.md` and `flydsl_knob_space.md`.

## The one rule

**Do structural work in early patches and fall back to tuning in later ones.** A kernel whose structure is wrong
cannot be tuned into a good one — you will just find the best member of a bad family.

## Route here when
Optimizing an authored FlyDSL kernel and the bottleneck is still broad. If the discussion has already
narrowed to `tile_m`/`tile_n`/`tile_k`, MFMA-loop ISA counts, or epilogue strategy on a clearly
GEMM-like kernel, go to `../../gemm/INDEX.md`.

## Step 1 — read before you optimize

1. The **full kernel source** — every `@flyc.kernel`, its algorithm, data flow, loop structure.
2. The **`@flyc.jit` host wrapper** — how many kernels launch per call and what data each receives.
   **Multiple kernels sharing data is a fusion opportunity.**
3. **Imported helpers** (`flydsl.utils`, `flydsl.expr`) — reusable building blocks.
4. The **test harness** — which shapes, dtypes and modes are actually benchmarked.
5. If you plan to rewrite loops or memory paths, review the three loop forms
   (`../debug-flydsl-kernel.md` § The three loop forms), buffer-tensor bounds
   (`fx.rocdl.make_buffer_tensor`) and the kernel's LDS allocator **before** editing.

## Step 2 — classify, and know when to stop

Check the arch via `flydsl.runtime.device.get_rocm_arch()` (S1; kernels import it as `get_hip_arch`) — LDS size,
MFMA variants and wavefront width are arch-dependent (**gfx950: 160 KiB LDS, 64 banks, 256 CUs, wave64**).

| Class | Lever |
|---|---|
| Memory-bound **with** cross-thread reuse (tiled GEMM operands, attention K/V tiles) | LDS staging, prefetch, vectorization |
| Memory-bound **without** cross-thread reuse (elementwise, RoPE, cache-write-only) | **vectorization and coalescing**; prefetch when few waves are resident |
| Compute-bound | MFMA selection, software pipelining |
| Latency-bound (small shapes) | reduce launch count — fusion |

**Two checks that save whole sessions:**

- **No cross-thread reuse → LDS staging pays only if it buys something else: coalescing a layout change, or a
  deeper load pipeline (Tier 2).** *(field, E1)* In the same layout it gained nothing (1.00× against direct 16-byte
  loads and stores); on a transpose it beat per-element 2-byte stores by 1.42× but only matched an in-register
  transpose with 16-byte accesses.
- **Already fused** (name contains `fused_` / `*_2stage` / `*_multistage`, or the source already
  combines the ops) → **not a reason to skip Tier 1.** On a FlyDSL 2-stage MXFP4 MoE at 8–64 tokens, folding the
  routing sort into stage 1 removed a launch and shortened the call; on a MoE with a wide stage 1 a similar fold lost
  (`../../moe/few_tokens.md` §2 and Counter-signals). Count the launches and HBM passes that remain first.

## Step 3 — the four tiers, in order

### Tier 1 — structural (highest impact, highest regression risk)

**Guard: list what the structure still costs before touching it** — launches, passes over HBM, repeated loads.
A `fused_*` or `*_2stage` name is no proof of the optimum (Step 2), and a single-pass kernel's loops and data
layout can still be Tier 1 work.

- **Kernel fusion** — if the `@flyc.jit` wrapper launches 2+ kernels sharing input data, try merging them.
  It removes launch overhead and redundant HBM reads, but can enlarge the grid, repeat work in every workgroup or
  raise register use: time it against the separate kernels per shape (Step 2).
- **Fast-path relaxation** — look for over-restrictive guards on optimized paths (disabled branches,
  alignment checks stricter than necessary). Relaxing one lets more shapes take the fast path, but the guard may be
  what keeps bounds or alignment correct: run the oracle on every newly admitted shape before timing it.
- **Loop restructuring** — converting constexpr unrolling to `scf.for` is not a reliable speedup, even under code
  bloat or register pressure; time both. *(field, E2)* On a FlyDSL MFMA K loop, 45–78 KB unrolled bodies ran within
  2% of a constant-bound `scf.for`, cutting VGPRs 198 → 138 gained 1.7%, and one `scf.for` that LLVM unrolled by 2
  was 27% slower. Keep small fixed loops unrolled: with Python-int bounds and a small trip count LLVM unrolls the `scf.for` back to the
  same ISA.
- **Redundant work elimination** — repeated loads, recomputed indices, overlapping branches.
- **Algorithm replacement** — reduce pass count (online softmax vs two-pass; fused attention vs
  separate QKᵀ → softmax → ×V). **A single-pass elementwise kernel is already optimal at the pass
  level — do not add stages.**

**FlyDSL refactor guardrails**
- `range_constexpr()` is the only loop form that unrolls. Both `range` forms lower to `scf.for`: without
  `init=` the carried state is inferred from the names the body reassigns (bounds `int` or `fx.Int32`);
  with `init=` only yielded values are carried (Python-int bounds are fine).
- Keep loop-carried state **positionally aligned** across `init`, per-iteration `state`, `yield`, and
  post-loop results — same slot, same meaning, same MLIR type throughout.
- Legacy kernels only: if an `SmemPtr` view created inside a loop body is reused in the epilogue,
  **clear `_view_cache`** before reusing it outside the loop, or you get SSA dominance errors.

### Tier 2 — memory hierarchy

- **LDS staging** — when the same global data is read multiple times **across threads in the same
  workgroup**, or to coalesce a layout change *(field, E1)* (Step 2) or deepen a load pipeline
  (`../../gemm/small_m.md` §5). Allocate with `fx.SharedAllocator().allocate(<@fx.struct>).peek()`; `SmemAllocator` /
  `SmemPtr` are legacy.
- **Vectorized access** — widest loads/stores matching the element type (`vec(8, ...)`, `vec(4, ...)`).
- **Overlap loads with compute** — move global loads earlier; `sched_barrier` to control interleaving.
  Issuing loads early also pays in a loop with no compute to overlap when few waves are resident (prefetch, below)
  *(field, E3)*.
- **Pre-load across passes** — load later-pass data during earlier passes.
- **Coalescing** — restructure loop ordering if the access pattern is not coalesced.
- **Register pressure** — balance registers against occupancy and scratch spills.

**Prefetch: check occupancy first.** In a loop dominated by global loads with minimal compute, prefetch still
hides memory latency when few waves are resident. *(field, E3)* Issuing the next trip's loads early raised
bandwidth 1.56–1.74× at 1 wave/SIMD (and at 2 with one load per trip) and gave 0.85–1.00× at 4–8 waves/SIMD. Add
it when registers or LDS hold occupancy low; check that the carried state does not cut occupancy and that the loop
does not wait `vmcnt(0)` every trip, which gained nothing without ALU work.

**The prefetch shape**: prologue preloads iteration 0 → `scf.for` carries the prefetched values → the
body unpacks current state and issues next-iteration loads **immediately** → epilogue consumes the
final carried values. **Carry everything** needed to materialize the next iteration: not just tensor
payloads but block-table entries, page IDs, scale values, running accumulators. Re-check the register
budget before adding buffers.

### Diagnose LDS from the trace, not from intuition

Three different problems that look alike and want different fixes:

| Signal | Problem | Fix |
|---|---|---|
| Stall on the `ds_read_*` / `ds_write_*` rows themselves, often with a large `s_waitcnt lgkmcnt` right after them | **bank conflicts** | swizzle or padding |
| `s_waitcnt lgkmcnt(0)` spikes right after `ds_write` while the `ds_write` rows themselves barely stall | **write-read latency exposed** | increase the distance |
| `s_barrier` dominates a reduce/broadcast region | **cross-wave serialization** | fewer barrier stages, cheaper cross-lane primitives |

A bank conflict can make the `lgkmcnt` wait after the LDS ops the top hotspot, and moving independent work before
it then did nothing *(field, E4)*: check the `ds_*` rows before treating a wait spike as latency.

**Do not treat all three as "a swizzle problem."** Use these metrics twice: before the rewrite to
classify, after the rewrite to confirm the targeted stall actually moved.

**gfx950 LDS: 160 KiB, 64 banks — but only `ds_read_b64` / `ds_read_b128` behave as 64 banks.** *(field, E5)*
`ds_read_b32` and `ds_write_b32` / `b64` / `b128` conflicted as much at a lane stride of 32 dwords as at 64, as on
a 32-bank part. A layout that fully aliased banks on a 32-bank part only partially conflicts here for those wide
reads — **check swizzle masks and padding per instruction width and direction; inherited ones are unverified.**
The extra headroom also makes padding affordable where it was not before.

**Swizzle vs padding**: XOR swizzle when the access pattern is regular, read/write transforms can stay
consistent, and LDS headroom is tight. Padding when the swizzle math would hurt maintainability and a
small stride change breaks the pattern cleanly. **Either way, keep producer and consumer consistent —
a swizzled store with a linear load is a correctness bug, not a perf trade.**

**Increase write-read distance before adding structure.** If the stall is `lgkmcnt` right after
`ds_write`, first try moving independent work in between:

```python
# BEFORE: write immediately followed by barrier/read
lds_ptr.store(data, [offset])
fx.gpu.barrier()
value = lds_ptr.load([offset])

# AFTER: independent work before the synchronization point
lds_ptr.store(data, [offset])
next_offsets = compute_next_offsets()
next_data = buffer_ops.buffer_load(next_rsrc, next_offsets, vec_width=4, dtype=fx.T.f32)
fx.gpu.barrier()
value = lds_ptr.load([offset])
```

Do not insert work that depends on the just-written value, or extra LDS traffic competing for the same
bottleneck.

### Tier 3 — compute
- **MFMA selection** — the most efficient variant for the arch, via `flydsl.expr.rocdl`.
- **Software pipelining** — ping-pong buffers + scheduler barriers.
- **Scheduler tuning** — match `sched_mfma` group counts to the **actual** MFMA ops per iteration;
  verify `sched_dswr`/`sched_dsrd` timing. Copied constants are not reliably worse or better than none: time any
  schedule, copied or your own, against no hints before keeping it. *(field, E6)* On FlyDSL's fp8 preshuffle GEMM a
  copied schedule ran 4–8% slower on two tiles and took 10% less time on a third; one tile's own schedule was 7%
  slower.
- **Loop unrolling** — expose ILP; merge loops over the same range.

### Tier 4 — parameters (lowest impact)
Block size, tile dimensions, unroll factors, `known_block_size` hints.

**Tune only after structure stabilizes.** Treat old tuning conclusions as **stale** after any
codegen-affecting refactor.

## Modification rules
- **Fusion**: you may create new `@flyc.kernel` functions, remove old ones, and modify the `@flyc.jit`
  wrapper to launch the fused kernel.
- **Everything else**: modify code inside `@flyc.kernel` functions and their kernel-internal helpers.
- The `@flyc.jit` **external signature** (as called by the harness) must not change.
- **Do not modify** the build system, compilation flags, test harness, or benchmark framework.

## Correctness constraints — violations can fail silently

| Constraint | Why |
|---|---|
| **LDS limit** per `get_rocm_arch()` (gfx950: 160 KiB) | exceeding it raises no Python exception *(field, E8)*: the kernel does not run, so a reused output buffer still passes a correctness check and timing still returns a number; only stderr shows it, and the next PyTorch GPU kernel raises `AcceleratorError` |
| aiter's preshuffle GEMM (`dbd8bf5`): `tile_m · tile_k · elem_bytes` a multiple of 4096 (256 threads × 16 B) | otherwise the A-tile copy truncates and computes on stale LDS with no error (S4, S5); FlyDSL 0.3.2's own copy raises `ValueError` (S5) |
| aiter's mixed MoE GEMM, both stages: `tile_k_bytes % 64 == 0` | raises `ValueError` (S6) |
| **fp8 `0x80` is NaN in the FNUZ encoding** — replace that one byte (compare and select), not a byte AND `0x7F`, which also turns every negative value positive | gfx950 is OCP, where `0x80` is −0 and `0x7F` / `0xFF` are NaN (S7); this applies when handling FNUZ-encoded data |
| f32→f16: **clamp to ±65504 first** | otherwise Inf |

**FlyDSL memory contracts**
- Legacy `buffer_ops.buffer_load` / `buffer_store` (now `kernels/common/buffer_ops.py`, not in the
  `flydsl` wheel) take offsets in **elements, not bytes.** Recompute address units whenever a rewrite
  changes dtype, packing or vector width.
- Packed FP8/INT4 reinterpreted through `dtype=T.i32` — divide byte addresses by the new element width.
- New or resized LDS goes into the kernel's `@fx.struct` storage; with `SharedAllocator(static=True)`
  the backend places the fields, so field order is not a layout you control.
- Legacy `SmemAllocator` kernels: `allocator.finalize()` must still run in the GPU module body, and
  `SmemPtr` views moved across loop/region boundaries need their cached view re-checked.

## Step 4 — validate

1. **Correctness first.** Never trade it for speed.
2. Confirm speedup across **all** tested shapes, not one.
3. **Report speedup ratio AND absolute optimized time (ms).** A better ratio with a *worse* absolute
   time means the baseline shifted, not that the kernel improved — **treat an absolute regression as a
   failure even when the ratio looks better.**
4. For structural rewrites, dump with `FLYDSL_DUMP_IR=1` and inspect the relevant `.mlir` stage plus
   `final_isa.s`.
5. **Verify the specific effect you wanted**: `scf.for` survived tracing, wide loads stayed vectorized,
   the MFMA variant matches dtype/arch, the loop shape reflects the intended schedule.
6. **If the generated form did not change, the optimization did not land** — no matter how right the
   Python looks.
7. If the speedup is marginal or the absolute time regresses, move to the **next structural strategy**
   rather than re-tuning the same approach.

## Common mistakes
- Starting from scheduler constants before proving the bottleneck.
- Copying tile sizes from another kernel without checking work decomposition.
- Multi-stage LDS buffering that destroys occupancy.
- Treating every LDS issue as a swizzle problem instead of checking wait distance.
- Overfitting to one benchmark shape.
- Assuming a trace/ISA pattern from another repository matches this kernel.

## Key APIs
- Device kernel `@flyc.kernel` · host launcher `@flyc.jit`
- Control flow: `range_constexpr()` · `range(...)` · `range(..., init=...)` · `const_expr()`
- Intrinsics: `flydsl.expr.rocdl` — MFMA, exp2, rcp, `sched_barrier`, `sched_mfma`
- Shared memory: `fx.SharedAllocator` over an `@fx.struct` (legacy: `SmemAllocator` / `SmemPtr`)
- Types: `T.f16`, `T.bf16`, `T.f32`, `T.i32`, `T.vec(...)` from `flydsl.expr.typing`
- Buffer ops: `fx.rocdl.make_buffer_tensor`, `fx.make_copy_atom`, `buffer_ops.buffer_load`/`buffer_store`
- IR/ISA: `FLYDSL_DUMP_IR=1`, `FLYDSL_DUMP_DIR=...`, `final_isa.s`
- Autotune: `flydsl.autotune.autotune`, `Config`, `do_bench`

## Related
`../../gemm/INDEX.md` (dense-GEMM package) ·
`../guides.md` § flydsl-kernel-authoring (first-time authoring) ·
`../debug-flydsl-kernel.md` (correctness bugs)

## Evidence

- E1 [2026-11, gfx950 (MI355X), HIP 7.15, N=11 rounds, hipEvent median, arms rotated] bf16 stream through LDS against direct 16-byte loads and stores: same layout 1.00×; 16384×16384 transpose through an XOR-swizzled LDS tile 1.42× the rate of per-element 2-byte stores, 0.99× an in-register transpose with 16-byte accesses (HIP test kernel, not FlyDSL)
- E2 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=7 rounds, CUDA-event median; no spills] time of `scf.for` over the fully unrolled MFMA K loop (16-byte global loads, no LDS): 45428 B and 78048 B unrolled bodies 0.98–1.00 against a constant-bound `scf.for`, 1.00 and 1.07 against a runtime-bound one; `scf.for` at 138 VGPRs against 198 unrolled 0.983, and 1.27 where LLVM unrolled it by 2; at 4 and 16 steps the constant-bound `scf.for` compiled to the unrolled ISA
- E3 [2026-11, gfx950 (MI355X), HIP 7.15, N=11 rounds, hipEvent median, bandwidth = bytes read / time] streaming sum with 16-byte loads, next trip's loads issued early against none, with or without 8 FMAs per value: 1 wave/SIMD 1.59–1.74×, 2 waves/SIMD with one load per trip 1.56–1.57×, 4–8 waves/SIMD 0.85–1.00×; a rolled loop with a `vmcnt(0)` each trip 0.99–1.01× without FMAs (HIP test kernel, not FlyDSL)
- E4 [2026-11, gfx950 (MI355X), HIP 7.15, rocprofv3 ATT per-instruction stall, N=1 traced dispatch per variant, timing N=7 rounds] 8 × `ds_read_b32` then `s_waitcnt lgkmcnt(0)`, 64 lanes on one bank against none: 61% of the added stall on the wait, the top hotspot; 32 independent `v_add_u32` moved between 8 conflicting or clean `ds_write_b32` and the wait: time 1.00× with the conflict, 0.91× without (HIP test kernel)
- E5 [2026-11, gfx950 (MI355X), HIP 7.15, N=7 rounds, `clock64` cycles per instruction, median] 8 LDS instructions then `s_waitcnt lgkmcnt(0)`, lane l at dword l·s; extra cycles at s = 32 over those at s = 64: `ds_read_b32` 1.00, `ds_read_b64` 0.47, `ds_read_b128` 0.43, `ds_write_b32` / `b64` / `b128` 1.00 (HIP test kernel)
- E6 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, N=15 rounds of 100 launches, median, arms rotated] `kernels/gemm/preshuffle_gemm.py`, fp8, M=N=16384, K=4096, time against no hints: another tile's `sched_*` sequence 0.90×, 1.08×, 1.04× on three tiles; each tile's own schedule 0.97×, 0.98×, 1.07×
- E8 [2026-11, gfx950 (MI355X), FlyDSL 0.3.2, torch 2.11.0+rocm10.0.0, N=6 over-limit runs] probe kernel over 163840 B of LDS, static or dynamic: no Python exception, 0 of 256 outputs written, the error only on stderr; event timing still returned 21.9–24.1 µs per call; the next PyTorch GPU kernel raised `torch.AcceleratorError`
- S1 ROCm/FlyDSL@dd8373349:python/flydsl/runtime/device.py:39 `def get_rocm_arch() -> str`; `kernels/attention/flash_attn_generic.py:26` imports it `as get_hip_arch`; the `flydsl` package defines no `get_hip_arch`
- S4 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/preshuffle_gemm.py:211 `total_threads = 256`, `a_load_bytes = 16`, `bytes_per_thread_a` and `num_a_loads` by truncating division (:213-214), no check on the remainder
- S5 ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py:184 both divisions truncate, the tail of the A tile is left unfetched and the kernel computes on stale LDS; raises `ValueError` unless `tile_m * tile_k * elem_bytes` is a multiple of `total_threads * a_load_bytes` (:189-195)
- S6 ROCm/aiter@dbd8bf5:aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:202 stage 1 raises `ValueError` unless `tile_k_bytes % 64 == 0`; stage 2 the same (:3525)
- S7 hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:708 FP8 E4M3: +0 `0x00`, −0 `0x80`, NaN `0x7F` / `0xFF`, max 448
