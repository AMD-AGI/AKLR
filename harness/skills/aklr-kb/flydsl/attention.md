---
title: "Attention in FlyDSL: the FlyDSL side of the attention/ cards"
kind: topic
ops: [attention, mla]
dtypes: [bf16, fp16, fp8]
regimes: [many query rows, few query rows, paged KV, dQ dK dV]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/attention/
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1633
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1651
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1662
  - ROCm/FlyDSL@dd8373349:python/flydsl/utils/env.py:249
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:888
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:17
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:31
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp:146
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:92
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:2725
  - ROCm/FlyDSL@dd8373349:kernels/attention/swa_gfx950.py:233
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/primitive.py:325
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:3500
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/cdna4.py:80
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp:245
  - ROCm/FlyDSL@dd8373349:lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp:67
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:73
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:970
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:66
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/backends/rocm.py:52
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py:670
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/gpu.py:73
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/primitive.py:329
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:107
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:744
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/llvm_options.py:22
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:157
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_gfx950.py:1018
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/inline_asm.py:55
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/universal.py:96
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py:810
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/math.py:192
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/utils/arith.py:526
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/kernel_function.py:250
  - ROCm/FlyDSL@dd8373349:lib/Conversion/FlyToROCDL/FastMathOps.cpp:35
  - ROCm/FlyDSL@dd8373349:kernels/attention/flash_attn_utils.py:432
---

# Attention in FlyDSL

How to write in FlyDSL what the cards of `../attention/` describe: which FlyDSL calls to use, what FlyDSL's compiler
and JIT do with them, and how FlyDSL's own attention kernels behave when you call or tune them. Each section is named
after the card section it serves (`softmax §2` is § 2 of the attention package's softmax card) and holds only the
FlyDSL part: read the card for the mechanism and the numbers. Abbreviations are those of the card the section
serves. FlyDSL facts are for v0.3.2, checked against its source and its LLVM.

## Route here when

- A card of `../attention/` ends a section with "In FlyDSL: `../flydsl/attention.md` § …".
- You need the map of FlyDSL's own attention kernels, § INDEX FlyDSL's own attention code (v0.3.2,
  `kernels/attention/`), or the FlyDSL JIT rules for timing a kernel, § measurement §6.

## INDEX FlyDSL's own attention code (v0.3.2, `kernels/attention/`)

Anchors for the API and patterns the cards refer to; read the source for anything else.

| File | What it is |
|---|---|
| `flash_attn_gfx950.py`, `flash_attn_utils.py`, `flash_attn_interface.py` | bf16/f16 flash forward for gfx950: 8 waves in two phase-staggered groups (load/compute ping-pong), 32x32x16 MFMA, QK computed as S^T, causal loop stopping at the diagonal, per-wave lazy rescale |
| `flash_attn_fp8_gfx950.py` | fp8 flash forward with the same dual-wave structure |
| `flash_attn_generic.py` | f16/bf16 flash-forward builder keeping Q, P and the O accumulators in 32x32 MFMA registers |
| `swa_gfx950.py` | GQA sliding-window forward |
| `mla_fwd_decode.py`, `mla_fwd_decode_m16x8_fp8_fp8.py` | MLA with one query row per sequence and head (decode): launcher taking split-KV partial buffers, and a 128-head fp8-Q/fp8-KV kernel |
| `pa_decode_fp8.py`, `pa_decode_swa.py`, `pa_decode_tile.py`, `pa_common.py`, `pa_metadata*.py` | paged KV pool with `query_length` query rows per sequence (decode): fp8 with persistent scheduling, sliding window, and a readable tile-programming reference |

There is no dQ/dK/dV (attention-gradient) kernel in v0.3.2; the gradient lessons come from measured FlyDSL dQ/dK/dV
kernels written outside the FlyDSL repository.

## measurement §6 Small shapes, the host and the launcher

- **FlyDSL v0.3.2: call the object returned by `flyc.compile(...)` in hot loops and benchmarks.** A direct call of
  a `@flyc.jit` launcher re-binds the signature, re-checks captured globals and rebuilds the cache key on every
  call. *(field, E1)* ~31 µs per call for an 8-argument vector-add launcher against 2.4–2.7 µs for the compiled object
  (`torch.add`: 1.8–2.3 µs); launchers with more arguments were not measured.
- The compiled object takes positional arguments only and ignores new values of `Constexpr` arguments: they are
  baked in at compile time (`jit_function.py:1651`). Recompile when one changes.
- **`flyc.compile(fn, *args)` runs `fn` once with those arguments** while compiling (`jit_function.py:1662`).
  Compile-then-call therefore executes twice, and a kernel that accumulates or atomically updates its outputs is
  applied one extra time. Compile on scratch buffers or re-initialise outputs and counters afterwards.

## measurement §7 Cache hygiene

- Where FlyDSL's cache lives and how to clear it: `debug-flydsl-kernel.md` § Before anything else: clear the cache.
  What its key does and does not see: `debug-flydsl-kernel.md` § What the cache key sees.
- Do not time with `FLYDSL_DUMP_IR=1` (`gemm.md` § measurement §6).

## lds §2 Issue transposed reads through the intrinsic, not inline asm

`flydsl.expr.rocdl` re-exports the ROCDL ops, so the transposed LDS read can be written in three forms (`buffer_ops`
is the repo's `kernels/common/buffer_ops.py`, not part of the wheel):
```python
ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)    # !llvm.ptr<3>
v4 = rocdl.ds_read_tr16_b64(v4f16_ty, ptr).result      # op class: take .result
v4 = rocdl.ds_read_tr16_b64_(v4f16_ty, ptr)            # function form
atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), bf16)   # copy atom
```
- The other widths: `ds_read_tr8_b64`, `ds_read_tr4_b64`, `ds_read_tr6_b96`; atoms `LDSReadTrans8_64b`,
  `LDSReadTrans4_64b`, `LDSReadTrans6_96b`, or `rocdl.cdna4.LDSReadTrans(granularity, bits)` with only
  (16, 64), (8, 64), (4, 64), (6, 96) accepted.
- FlyDSL's dual-wave forward reads V with inline asm and immediate offsets
  (`_ds_read_tr16_b64_imm`); its generic path (`flash_attn_generic.py`) and its sliding-window path
  (`swa_gfx950.py`) use the op and the atom.

## lds §8 Addresses

- An offset is a compile-time constant only when every term of it is a Python int while the kernel is traced;
  `fx.block_idx` is always a runtime value. In the card's case (E18 there) the d-tile index had been taken from
  `block_idx.z`; the fix computes it in a helper that returns a Python int on the path where it is static
  (`if const_expr(<no D split>)`), so only the variant that needs the runtime index pays for runtime offsets.

## mfma_layout §5 Choosing the atom per GEMM

- FlyDSL's gfx950 forward builds its QK and P·V atom as `fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))`
  (`flash_attn_utils.py:3500`).

## mfma_layout §8 fp8 MFMAs

- FlyDSL v0.3.2 spells the wide-K forms as the scale intrinsics: `rocdl.mfma_scale_f32_16x16x128_f8f6f4`,
  `rocdl.mfma_scale_f32_32x32x64_f8f6f4`, or the atom `rocdl.cdna4.MFMA_Scale(16, 16, 128, …)`, whose two scales
  start at 0 (`cdna4.py:80`, `CDNA4/MmaAtom.cpp:67`). There is no unscaled spelling because LLVM has no unscaled
  wide-K intrinsic (`../attention/mfma_layout.md` §8).
- e4m3 → bf16 for a bf16 MFMA: `rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), word, one, hi)`, the generated
  MLIR builder: `word` (an i32 holding four e4m3) and `one` (f32 1.0) must be MLIR values, e.g.
  `fx.Float32(1.0).ir_value()`; a Python float or a FlyDSL number raises (`language/compile_errors.md`). `hi` is a
  Python bool, whether the low or the high pair is converted; scale 1.0 gives the card's bit-exact conversion.

## registers_occupancy §1 Read the budget from the ISA

- FlyDSL v0.3.2: the occupancy target is the `waves_per_eu` compile hint (`backends/rocm.py:58-59`; `maxnreg`
  also exists) or the kernel attribute `"rocdl.waves_per_eu"` in `value_attrs`; both reach LLVM as
  `amdgpu-waves-per-eu`. FlyDSL's gfx950 forward sets the attribute, default 2 (`flash_attn_gfx950.py:73, 1001`);
  built for D=128 causal bf16 with 32 q heads and 8 kv heads it compiles to 246 VGPRs, 0 AGPRs, no spills. Other
  kernel attributes through `value_attrs` go under `"llvm.passthrough"` (`gemm.md` § INDEX FlyDSL version).
- The register fields the card reads are in the `NN_final_isa.s` that `FLYDSL_DUMP_IR=1` writes under
  `$FLYDSL_DUMP_DIR/<kernel>/` (`debug-flydsl-kernel.md` § What the cache key sees).

## registers_occupancy §6 Code shapes that spill

- A Python `if` on a runtime value is rewritten to `scf_if_dispatch` (`compiler/ast_rewriter.py:670`), which emits
  an `scf.if`: the names the body assigns become its results, and without an `else` they are yielded unchanged.
  Write the card's remainder as `if j + 1 < end:` assigning the accumulators, and each specialization arm the same
  way, one `if` after another. Give each such name a value before the `if`: `None` raises.
- Runtime conditional syntax: `debug-flydsl-kernel.md` § Runtime versus compile-time
  conditionals.

## registers_occupancy §8 MFMA–VALU co-execution

- The card's pin is `rocdl.sched_barrier(0)` placed after the `gpu.barrier()` that owns the LDS write, outside
  `range_constexpr` loops (they unroll while the kernel is traced).

## schedule_sync §1 Scheduling primitives and masks

- `rocdl.sched_group_barrier(mask, size, group_id)` (mask as an int or a keyword such as `"transcendental"`),
  `rocdl.sched_barrier(mask)`, `rocdl.iglp_opt(id)`, `rocdl.s_setprio(p)`.

## schedule_sync §2 `sched_group_barrier` recipes

- The final ISA to hash is the `NN_final_isa.s` that `FLYDSL_DUMP_IR=1` writes under `$FLYDSL_DUMP_DIR/<kernel>/`
  (`debug-flydsl-kernel.md` § What the cache key sees).

## schedule_sync §6 LLVM scheduler strategy and options

```python
jitfn.compile_hints["llvm_options"] = {"amdgpu-sched-strategy": "max-ilp"}   # changes the ISA
```
- `llvm_options` sets LLVM options by name: an unknown option name raises, and so does a string for an int option;
  an int for a string option such as `amdgpu-sched-strategy` is accepted without an error, so pass each value with
  the option's own type. The card's SLP threshold is `"slp-threshold": 100000` in the same dict.
- Hint layers merge shallowly and a `CompilationContext.compile_hints(...)` context wins over `jitfn.compile_hints`: the
  context's `llvm_options` replaces yours whole. FlyDSL's forward launcher sets its own in such a context, so
  for that kernel add the option to the launcher's hint dict and keep
  `{"enable-post-misched": False, "lsr-drop-solution": True}`. Hints are part of the cache key.

## schedule_sync §7 Hazards

- In v0.3.2 `rocdl.cvt_pk_bf16_f32` is inline asm. An f32 → bf16 `.to(fx.BFloat16)` on a scalar or a
  2-element vector lowers to the same `v_cvt_pk_bf16_f32` as an ordinary instruction; prefer it.

## schedule_sync §8 Waits and barriers

- `gpu.barrier()` lowers (MLIR `convert-gpu-to-rocdl`) to `fence syncscope("workgroup") release`, `s_barrier`,
  `fence syncscope("workgroup") acquire`; on gfx950 that is `s_waitcnt lgkmcnt(0)` then `s_barrier`, with no
  `vmcnt` wait for a global load still in flight. `rocdl.s_barrier()` is the bare instruction with no wait;
  FlyDSL's gfx950 forward uses it (`_s_barrier`, `_dualwave_sync_barrier`, `flash_attn_utils.py:66-76`).

## schedule_sync §10 Pipelining and DMA

- **Cache bits:** the LDS-DMA atoms (`BufferCopyLDS*`, `cdna4.BufferLoadAsyncLDS*`, `GlobalLoadAsyncLDS*`)
  always emit `aux = 0`. For `nt`/`sc` bits call
  `rocdl.raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux=<python int>)`; any other
  type is silently dropped (`gemm.md` § INDEX FlyDSL version).

## softmax §2 exp2: one `v_exp_f32` is enough

| Spelling | Lowers to |
|---|---|
| `fx.math.exp2(x)`, `ArithValue.exp2()` / `fx.Float32(x).exp2()` without `afn` | `__ocml_exp2_f32` |
| the same with `fastmath="afn"` or `"fast"`, or under the `fast_fp_math` compile hint | `rocdl.exp2` (pass `convert-rocdl-fastmath-ops`) |
| `rocdl.exp2(T.f32, x)` | `llvm.amdgcn.exp2.f32` |

What each lowering costs in ISA: `../attention/softmax.md` §2.
- The afn rewrite is FlyDSL's own MLIR pass (`FastMathOps.cpp:35`), run before LLVM; LLVM itself lowers an `afn`
  `llvm.exp2.f32` to the same range-reduced sequence as a plain one (`../attention/softmax.md` §2). It covers f32
  scalars and 1-D f32 vectors only.
- A denormal-mode function attribute does not turn the range-reduced lowering (`__ocml_exp2_f32`, or a direct
  `llvm.exp2.f32`) into one `v_exp_f32` in v0.3.2: a bare `"passthrough"` key is dropped
  (`gemm.md` § INDEX FlyDSL version), and `"denormal-fp-math-f32"` under `"llvm.passthrough"` reaches the LLVM IR,
  but v0.3.2's LLVM reads the denormal mode only from its `denormal_fpenv` attribute. FlyDSL's own forward passes its DAZ entries under a bare `"passthrough"`
  (`flash_attn_gfx950.py:970-1003`), so they are dropped; its single `v_exp_f32` comes from `rocdl.exp2`.
- FlyDSL's gfx950 forward (the card's reference pattern) writes every softmax exp as `rocdl.exp2`
  (`_exp2_score_slice`, `flash_attn_utils.py:432`) and compiles with `fast_fp_math` and `unsafe_fp_math` in its
  hints (`_dualwave_swp_compile_hints`, `flash_attn_gfx950.py:1018`).

## softmax §3 Scale, shift, clamp and pack at zero instruction cost

- `Vec(...).to(fx.BFloat16)` on a 2-element f32 vector lowers to one native `v_cvt_pk_bf16_f32`.
  `rocdl.cvt_pk_bf16_f32` emits the same instruction as inline asm, invisible to the hazard recognizer; an MFMA
  reading the packed P directly needs explicit padding (`../attention/schedule_sync.md` §7; § schedule_sync §7
  above).

## softmax §6.3 Skipping rescales

- Each runtime `if` becomes one `scf.if` (§ registers_occupancy §6) and counts as one branch point in the card's
  cost; the O and l it rescales are its results, passed through unchanged when the branch is not taken.
- The empty side-effecting asm that keeps a short skip body a branch, first in the body:
  `llvm.inline_asm(None, [], "", "", has_side_effects=True)` (`flydsl._mlir.dialects.llvm`).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=7, 2000 back-to-back calls synchronised at the end, 7 rounds per arm with the arm order rotated; in parentheses the host time with a synchronise before every call] a direct `@flyc.jit` call 31.01 µs [30.70–31.37] (31.84), the object `flyc.compile` returns 2.70 µs [2.60–2.74] (2.38), `torch.add(out=)` 2.34 µs [2.08–2.44] (1.81) (vector-add launcher with 8 arguments including 3 Constexpr and the stream, SIZE 4096, fresh runtime cache directory; probe `p3_flydsl_launch.py`; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/gpu_probes.md:148; local:/workspace/code/gpt_oss_docker/scratch/attn_kb/verify/raw/r_p3_run1.txt:15)
