---
title: "Inline assembly and cross-lane operations in Gluon on gfx950"
kind: language-note
gens: [gfx950]
triton: 3.8.0
basis: source-semantics
sources:
  - triton-lang/triton@c01b677:python/triton/language/core.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py
  - triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py
  - triton-lang/triton@c01b677:python/triton/compiler/code_generator.py
  - triton-lang/triton@c01b677:include/triton/Dialect/Triton/IR/TritonOps.td
  - triton-lang/triton@c01b677:lib/Dialect/Triton/IR/Ops.cpp
  - triton-lang/triton@c01b677:lib/Analysis/Utility.cpp
  - triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp
  - triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp
  - triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp
  - triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp
  - triton-lang/triton@c01b677:third_party/amd/language/hip/libdevice.py
  - triton-lang/triton@c01b677:third_party/amd/backend/compiler.py
---

# Inline assembly and cross-lane operations in Gluon (gfx950)

83 of AKLR's 154 Triton/Gluon kernels, all Gluon, call `gl.inline_asm_elementwise`: for hardware converts, one-instruction
transcendentals and wave-level selection (ballots, first lane, lane rank, lane reads and writes, DPP scans). Here: what each
instruction does on gfx950, how to write the call, what the asm-free spelling compiles to, and the hazards. “Compiled” means offline Triton 3.8.0 compilation for gfx950; the stated tile and layout
limit instruction-count observations.

## 1. What the kernels use inline assembly for
Kernels: of the 154 Triton/Gluon schemas at 43e112e, those whose sources hold the instruction in an asm string.
| Instruction | What the kernels use it for | Kernels | Example |
|---|---|---|---|
| `v_cvt_scalef32_pk_fp4_f32` | pack two fp32 into one e2m1 byte, in MXFP4 MoE kernels (19 DeepSeek-V2, 22 MiniMax-M2) (§3.1) | 41 | S1 |
| `v_cvt_scalef32_pk_fp8_f32` | two fp32 → two e4m3 bytes without Triton's FP16_OVFL change (§3.2, §7) | 3 | S2 |
| `v_med3_f32` | in the Qwen3.5 GDN decode fp8-quant kernels (`n_v_heads16`); S3 clamps to ±448 before an fp8 cast; `gl.clamp` is the same instruction (§3.2) | 8 | S3 |
| `v_rcp_f32` · `v_div_fixup_f32` | 1-ulp reciprocal in sigmoids and quantization scales · special cases (0, ∞, NaN) of a reciprocal-based quotient (§4) | 16 · 2 | S4, S5 |
| `v_exp_f32` · `v_log_f32` | 2^x with no range handling (`gl.exp2` is the same instruction) · log2 inside a softplus whose argument is ≥ 1 (§4) | 10 · 7 | S2 |
| `v_pk_fma_f32` · `v_max_f64`/`v_min_f64` · `v_mad_u32_u24` | packed fp32 FMA · max of 64-bit sort keys · 24-bit address multiply-add (§4) | 2 · 1 · 1 | S2, S6, S7 |
| `v_cmp_*` into an SGPR pair (`_e64` or no suffix), or `_e32` into VCC (ballot) | one bit per lane: candidate lanes, valid slots, bits of a key (§6.1) | 28 | S1, S11, S22 |
| `s_ff1_i32_b64` | first lane of a ballot: argmax tie-break, job pick (§6.2) | 17 | S8 |
| `v_readlane_b32` · `v_readfirstlane_b32` · `v_writelane_b32` | one lane into an SGPR (a winner's id, a scale from a register table) · a value made wave-uniform (tile descriptors, weight scales) · a scalar into lane k (the k-th top-k result) (§6.4) | 22 · 8 · 5 | S8, S9, S10 |
| `v_mbcnt_lo/hi_u32_b32` · `v_bcnt_u32_b32` · `s_bcnt1_i32_b64` | rank of a lane in a ballot · popcounts: compaction, counting sorts (§6.3) | 9 · 10 · 1 | S4, S1 |
| `v_ffbl_b32` | lowest set bit: next live tile, first lane of a group (§6.8) | 9 | S11 |
| `ds_bpermute_b32` | read a value from a computed lane (§6.5) | 11 | S1 |
| DPP (`row_shr`, `row_bcast`, `row_ror`, `row_half_mirror`, `quad_perm`) | wave prefix sums, partial reductions, xor exchanges (§6.6) | 11 | S6 |
| `v_permlane16_swap_b32` | swap a register bit with lane bit 4 before a store (§6.7) | 3 | S12 |
| `s_setreg_imm32_b32` | in the 2 Qwen3-Next GDN decode kernels of §7; S13 writes MODE[23], FP16_OVFL | 2 | S13 |
| `s_waitcnt` | in the string of the asm's own `ds_bpermute_b32` (11), a partial `vmcnt(N)` (2), a lone `lgkmcnt(0)` (3) (§8) | 16 | S14, S15 |
| empty string, tied operands | keep a value in VGPRs or SGPRs, order an accumulator (§8) | 5 | S16, S15 |

## 2. `gl.inline_asm_elementwise` in Triton 3.8.0
`gl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` is Triton's builtin, re-exported (S17, S18).

### 2.1 What one call becomes
- **Shape and layout.** `args` are broadcast to one shape; a Python number becomes a constant tensor. The result has
  that shape and the arguments' layout (S17, S19). The asm runs per thread over the registers the layout gives it,
  so the layout decides which tensor elements one asm instance sees.
- **Instances.** One instance per `pack` consecutive registers of the thread (S20; the docstring: "unspecified", S19):
  at `pack=2` over 4 int32 per thread, 2 instances, on registers 0–1 and 2–3 ; the GDN prefill swap
  (`pack=8`, §6.7) is written on this grouping (not run with `pack=8`). The verifier checks only the tensor's element count against `pack` (S21); a thread
  holding fewer, or not a multiple, gets undefined inputs for the rest without a message (S20): compiled, the §6.7 swap
  on 4 fp32 per thread swapped the 4 real registers with 4 unwritten VGPRs. Give each thread a multiple of `pack`
  elements and pin the shape the asm assumes with `gl.static_assert` (S12).
- **Statements.** With `args=[]` the result is a scalar and the asm appears once per thread. With a tensor argument it
  appears once per `pack` elements: compiled, a non-pure `s_nop 7` on a 4-element-per-thread tensor was emitted 4
  times, with `args=[]` once. Waits and mode changes take `args=[]` or a scalar (S14, S13).
- **At least one result** ("This op does not support empty `dtype`", S17). `dtype=` fails in 3.8.0 without that
  message (`args=[]`: `PassManager::run failed`; a tensor argument: "… there must be at least one non-scalar result"). A
  statement declares a dummy output (`s_mov_b32 $0, 0` with `=s`, S14; `v_mov_b32 $0, 0` with `=v`, S6; S15 never writes its `=v`).
- **Narrow types.** bf16 elements are packed two per 32-bit register (S20): `pack=2` passes one `<2 x bfloat>` per
  operand, `pack=4` two and returns two (compiled). One constraint per register: 4 input constraints at `pack=4` abort.
- **Several results.** A tuple `dtype` returns a tuple; LLVM sees a struct of the outputs (S20). For scratch registers
  give the asm extra outputs and drop them (`_, _, _, winner =...`, S8; `total, _scratch =...`, S2), so that the
  allocator picks them; a fixed register listed as a clobber also works (`vcc` with `~{vcc}`, S22).

### 2.2 Constraints
LLVM's constraint syntax with the AMDGPU register classes; outputs, then inputs, then clobbers. `$n` numbers the
outputs' registers before the inputs', both in the same order (S20, S17): with `pack=2`, two fp32 outputs and args
`a, b`, `v_mov_b32 $k, $(k+4)` for k = 0..3 returned output 0 == `a` and output 1 == `b` bitwise.
| Constraint | Meaning | In the kernels |
|---|---|---|
| `=v` / `v` | VGPR output / input; a 64-bit type takes a VGPR pair | converts, fast math |
| `=s` / `s` | SGPR output / input; a `uint64` takes an SGPR pair (compiled: `v_cmp_ne_u32_e64 s[0:1], …`) | ballots, lane reads |
| `=&v`, `=&s` | early clobber: an output written before all inputs are consumed must not overlap those inputs | two-instruction popcounts, fp4 zeroing |
| `0`, `1`, … | input shares that output's register: `=v,0` returned x+1 ; tie digits refer to output indices | `v_writelane`, permlane swaps, empty pins |
| `n` | must lower to an immediate: constexpr, `gl.static_range`, `range(2)` and `range(constexpr=3)` kept the asm and wrote the intended lanes; runtime-bound `range` dropped it  | `v_writelane` lane index |
| `~{scc}`, `~{vcc}`, `~{memory}` | declare SCC/VCC writes, or memory effects not expressed by operands (`~{memory}` is a compiler memory clobber, not a hardware memory write or fence) | `s_cmp`/`s_bcnt1`, `_e32` compares, waits |
- **`s` takes only wave-uniform values.** Given a value LLVM keeps per lane, it inserts `v_readfirstlane_b32` without
  a word (compiled): every lane then gets one lane's value. A uniform value left in a VGPR (a scalar `gl.load` that did
  not become `s_load`) reached the asm as a VGPR, which an SGPR-only operand rejects (compiled): pass it through a
  `v_readfirstlane_b32` asm first. Uniform sources: kernel arguments, program ids, full-wave `gl.max`/`gl.sum` (they
  end in `v_readlane`, §5), other `=s` outputs. LLVM put nothing between `gl.max`'s closing `v_readlane_b32` and an
  asm whose first `v_cmp` read that SGPR, where its own VALU read waits 2 (compiled): open such an asm with `s_nop 1`.
- **A tied `=s,0` on a per-lane value** prints `error: … illegal VGPR to SGPR copy` on stderr and still returns a
  binary whose SGPR holds something else (compiled). Read stderr whenever you compile asm.
- **Data and lane both in SGPRs.** `v_writelane_b32 $0, $2, $3` with `=v,0,s,s` fails the assembler ("invalid
  operand (violates constant bus restrictions)" on stderr, a `RuntimeError`, §9.3). Pass the lane as `n`: "The
  immediate lane operand avoids AMD's scalar constant-bus conflict." (S22). The bus rule: [lookup.md](../../hardware/gfx950/lookup.md) §3.
- **Declare what the asm writes besides its outputs**: `~{vcc}` for an `_e32` compare (S22), `~{scc}` for each SALU
  instruction whose ISA entry sets SCC (S45, S61). Read back: `s_cmp_eq_u32`, `s_bcnt1_i32_b64`, `s_add_u32`,
  `s_and_b64`, `s_lshl_b64` set SCC; `s_mov_b32`, `s_ff1_i32_b64`, `s_cselect_b32` did not. Kernels: S4, S23.
- **Convert operands.** Keep converter data in VGPRs and follow the original ISA register-class
  rules (S24). An accepted alternate encoding in one compilation is not a general guarantee.
  The fp4 scale can be an inline constant or VGPR.

### 2.3 Purity: what the compiler may do
`is_pure=False` makes the call LLVM `asm sideeffect`, `is_pure=True` a plain `call asm` (S20, S21). Compiled:
| | `is_pure=True` | `is_pure=False` |
|---|---|---|
| two identical calls on the same operands | merged into one | both kept |
| result unused | deleted | kept |
| a `gl.load` written after the call | moved above it (kept after it when the constraints carry `~{memory}`) | stayed after it (with or without `~{memory}`) |
| loop-invariant call inside the tested Gluon `for` | stayed in the loop; hoist manually only if explicit inputs and implicit wave/EXEC state remain unchanged | stayed in the loop |
| independent arithmetic written before the call | — | may be scheduled after it (§7) |
Mark pure only an operation whose explicit inputs and implicit wave/EXEC state fully determine
its observable result. Mark waits, mode changes and unused ordering pins impure so they survive.
Impurity does not impose an arithmetic dependency: pass the ordering value through operands (§7).

### 2.4 Building the asm string
The string must be a compile-time constant. An f-string over constexprs (S14, §8; S25), `+` on constexpr strings
(S26), a tuple of strings indexed by a `gl.static_range` index (S27) and a constexpr `if`/`elif` choosing the string
(S28) are in the kernels; all four, and `str.format` on a literal, compile. By default, a module-level asm string must be a
`gl.constexpr(...)` instance (unlike exempt globals such as imported modules and JIT functions; `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` also bypasses this check) (`aklr/ops/README.md` § The kernel section and ABI).

### 2.5 Example: ballot, rank and count
Wave-level stream compaction, one wave per program. The caller must provide at least `64 * grid` input elements and output slots, plus `grid` count entries; each program owns a separate 64-slot output segment and only its first `count[pid]` entries are valid. Pad or add tail masking for shorter input; this is not global tightly packed compaction. It compiles with Triton 3.8.0 for gfx950 at `num_warps=1` (the
default 4 fails: "Layout has 1 warps per CTA, but the context requires 4 warps per CTA"). The amdgcn holds
`v_cmp_ne_u32_e64 s[0:1]`, `v_mbcnt_lo/hi_u32_b32` reading `s0`/`s1`, and `s_bcnt1_i32_b64`. The ballot's `s_nop 1`
covers a VALU read of its SGPR pair right after the asm (§9.2); `_rank` counts the set bits in the lanes below:
```python
@gluon.jit
def _ballot(pred):
    return gl.inline_asm_elementwise("v_cmp_ne_u32_e64 $0, 0, $1\ns_nop 1", "=s,v", [pred.to(gl.int32)], dtype=gl.uint64, is_pure=True, pack=1)

@gluon.jit
def _rank(mask):
    return gl.inline_asm_elementwise("v_mbcnt_lo_u32_b32 $0, $1, 0\nv_mbcnt_hi_u32_b32 $0, $2, $0", "=&v,s,s",
                                     [mask.to(gl.uint32), (mask >> 32).to(gl.uint32)], dtype=gl.uint32, is_pure=True, pack=1)

@gluon.jit
def _popcount(mask):  # s_bcnt1 also writes SCC: declare the clobber
    return gl.inline_asm_elementwise("s_bcnt1_i32_b64 $0, $1", "=s,s,~{scc}", [mask], dtype=gl.int32, is_pure=True, pack=1)

@gluon.jit
def compact(x_ptr, out_ptr, count_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])   # one element per lane, one wave: num_warps=1
    lane = gl.arange(0, 64, layout)
    base = gl.program_id(0) * 64
    x = gl.load(x_ptr + base + lane)
    keep = x > 0
    mask = _ballot(keep)
    slot = _rank(mask)
    gl.store(out_ptr + base + slot, x, mask=keep)
    gl.store(count_ptr + gl.program_id(0) + lane * 0, _popcount(mask), mask=lane == 0)
```

## 3. Conversions

### 3.1 fp32 → e2m1: `v_cvt_scalef32_pk_fp4_f32`
`v_cvt_scalef32_pk_fp4_f32 dst, src0, src1, src2` scales src0 and src1 by the reciprocal power of two encoded in src2's exponent (run at 1.0 and 4.0; S29), rounds to nearest even and writes `{src1, src0}` (src0 in the low nibble) into the byte of dst that
OPSEL[3:2] selects, keeping the other three. FP4 conversion ignores FP16_OVFL and saturates to ±6 (S24).
The kernels prescale by an exact power of two and pass `1.0` (S1): `even, odd = gl.split(gl.reshape(normalized, (normalized.shape[0], 16, 2)))`, then
`gl.inline_asm_elementwise("v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0", constraints="=v,v,v", args=[even, odd], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)`.
- **The other three bytes are garbage** under a plain `=v`: return `gl.uint32` and `.to(gl.uint8)`, or zero the
  register first, `"v_mov_b32 $0, 0\n…"` with `=&v` (S30; one more instruction).
- **Pairs.** `gl.split(gl.reshape(x, (..., 16, 2)))` puts element 2i in `even` and 2i+1 in `odd` in matching
  registers, so one instance packs one byte; it needs 2 or more elements per thread on the last axis (1: "SplitOp
  requires at least 2 elements per thread", compiled). The store follows the layout: 2 values per thread per run give a
  `global_store_byte` per pair, 8 a `v_perm_b32`-packed `global_store_dword`, 32 one `global_store_dwordx4` (compiled).
- **Signs.** Converting the signed values, or the magnitudes (`gl.abs` first, S56; `abs($1), abs($2)` modifiers, S23)
  and OR-ing the two sign bits back after `& 119`, gave the same byte, −0 included, when signs were extracted from the fp32 bit patterns. S56/S23 instead use `x < 0` for signs, which does not preserve −0.
- **Two converts into one register**: the ISA asks for a NOP or a write of another register between them (S57); in an
  asm two `v_cvt_scalef32_pk_fp4_f32` into one register (the second `op_sel:[0,0,1,0]`) stayed back to back (compiled)
  and gave right bytes with and without `s_nop 0`. The kernels write one byte per fresh register.
- The scale exponent, NaN guards and rounding: [numerics.md](../numerics.md) §3; feeding `mfma_scaled`: [mfma.md](mfma.md).

### 3.2 fp32 → e4m3: `v_cvt_scalef32_pk_fp8_f32`
Triton’s `.to(gl.float8e4nv)` emits this same instruction and marks the cast-containing function
for FP16_OVFL setup; a noinline callee can own the mode write (S31–S33). An asm-only path adds no
setreg automatically, so with FP16_OVFL = 0, ±Inf and out-of-range values become NaN
instead of ±448 (S24). Clamp first, as the Qwen3-Next GDN decode kernels' shared primitives do (S2). `gl.clamp` to ±448 is one `v_med3_f32`, as S3 writes in asm (a quiet
NaN gives −448 ; [numerics.md](../numerics.md) §6). Then convert to `BlockedLayout([2], [64], [NUM_WARPS], [0])`,
split `first, second = gl.split(gl.reshape(values, (values.shape[0] // 2, 2)))` and call `"v_cvt_scalef32_pk_fp8_f32 $0,
$1, $2, 1.0"`, `"=v,v,v"`, `dtype=gl.uint32`, pure, `pack=1`. It writes 16 bits (two fp8) into the half of dst that OPSEL[3]
selects and keeps the rest (S29): store `.to(gl.uint16)` through a `gl.pointer_type(gl.uint16)` pointer (S2).

## 4. Transcendentals, division and other ALU instructions
What each fp32 call lowers to, its VALU count and precision, and when the native instructions pay: [numerics.md](../numerics.md)
§6. `v_rcp_f32`, `v_exp_f32`, `v_log_f32` (S38): within 1 ulp on normal inputs, denormals flushed. On the asm side:
- **These need no asm** (compiled, the same single instruction). `v_exp_f32` asm is `gl.exp2` (denormals flushed: the
  backend fixes `__HIP_FTZ = True`, S32, S34). `v_rcp_f32` asm is `gl.extra.libdevice.fast_dividef(1.0, x)` (S35, S36),
  which MiniMax 1c17 uses (S37) and the compiler sees as an intrinsic. LLVM already packs fp32 `gl.fma`, `*` and `+` on
  register-adjacent pairs into `v_pk_fma_f32`, `v_pk_mul_f32`, `v_pk_add_f32`: check the amdgcn first (S2).
- **A quotient in 7 instructions instead of 11.** `r = v_rcp(d)`, a Newton step `r = fma(fma(-d, r, 1), r, r)`,
  `q = n·r`, a residual step `q = fma(fma(-q, d, n), r, q)`, then `v_div_fixup_f32 q, d, n` (S5, S29; no `v_div_scale`).
  The simplified sequence is not a whole-domain IEEE replacement: normal-range operands need
  numerical checks, and extreme denominator magnitudes can exceed a one-ulp error. Which quotient a contract allows: [numerics.md](../numerics.md) §2.
- **Canonicalization.** Under the default IEEE mode LLVM quiets each min/max operand it cannot prove canonical (a
  loaded or bit-cast value) with an extra `v_max x, x, x`: `gl.maximum` of two loads is 3 instructions, against a
  constant 2, of two arithmetic results 1 (compiled). One asm instruction skips it (`v_max_f64` on packed keys saves
  two per max, S6; `v_max_f32 |a|, |b|` in a reduction, S2), and so does IEEE mode off for a launch (`amdgpu-ieee`
  in `llvm_fn_attrs`, experimental, S8, S34): [numerics.md](../numerics.md) §6–§7, [tuning.md](../tuning.md) §9.
- `v_mad_u32_u24` (S29; DSA's gather addresses under `USE_U24`, S7) gave (a·b + c) mod 2^32 for factors below 2^24, and
  a first factor of 2^24 + 5 lost bit 24 : keep both factors within 24 bits.

## 5. Cross-lane operations Gluon emits by itself

Compiled, one wave, a 64-element `int32` tensor on 64 lanes, the instructions between the load's wait and the store;
the lane-bits 4–5 and `convert_layout` rows are read from the source:

| Gluon | gfx950 code | Instructions |
|---|---|---|
| `gl.sum` / `gl.max` over the 64 lanes | DPP `row_shr` 8, 4, 2, 1, `row_bcast` 15 and 31, then `v_readlane_b32 …, 63`: the result is in an SGPR (S39) | 18, an fp32 sum too; an fp32 `gl.max` in the default IEEE mode spends a `v_mov_b32_dpp`, a quieting `v_max_f32 x, x, x` and the `v_max_f32` per step: 29 |
| a reduction over lane bits 4–5 only (MFMA rows) | `v_permlane32_swap` / `v_permlane16_swap` and the combine (S39) | |
| `gl.reduce` of an int32 (value, index) pair over 64 lanes, larger value and lower index on a tie | xor butterflies: DPP for xor < 16, permlane swaps for xor 16 and 32; the result stays in VGPRs (S40, S41) | 78 |
| `gl.associative_scan` over 64 lanes | 6 `ds_bpermute_b32` and 6 `s_waitcnt lgkmcnt` (S42, S41); the DPP asm scan of §6.6: 13 | 52 |
| `gl.gather`, index depends on the lane | one `ds_bpermute_b32` and its wait; the address (1–2 VALU) is computed before the load's wait (S41) | 2 |
| `gl.gather`, constant index | `v_readlane_b32` | |
| `gl.convert_layout` moving data between lanes | DPP or permlane for xor patterns, `ds_bpermute` for the rest, or LDS: [layouts.md](layouts.md) | |
| `gl.extra.libdevice.popc` on `int32` | `v_bcnt_u32_b32` | 1 |

- A full-wave `gl.sum`, an int32 `gl.max` or an fp32 `gl.max` with `amdgpu-ieee` false: use `gl.sum`/`gl.max` (18, no
  quieting; the Qwen3-Next asm max, S2, an in-place `v_max_f32 … row_shr` per step plus a two-input pre-combine and an
  `|x|` with a floor: 19). To the closing `v_readlane` the AKLR asm trees are 1–2 shorter (15, 15, 16 against 17; compiled).
- Gluon has no ballot, first-set-lane, lane rank, lane write or "this value is uniform": §6.
- **`gl.gather(src, index, axis)`**: `index` has the source's rank and sizes off `axis`; the result takes the index's
  shape and layout (S58). Compiled, it stayed in registers with index and source in one layout and `axis` in one wave;
  with `axis` over the warps, or the index's other dimension on other lanes and waves, the whole source went through LDS
  (256 fp32 over 4 waves: 1 KB, an `s_barrier`; [64, 64] fp32: 16 KB; S59). Build the index in the source's layout.
- Compiler-generated reductions, gathers and conversions can emit these instructions too;
  attribute each operation before replacing it with asm.

## 6. Ballots, lane reads and the scalar unit

### 6.1 Ballot

MiniMax's `_wave_ballot_2` (S1) is `_ballot` of §2.5 without the `s_nop`: `"=s,v"`, `dtype=gl.uint64`.

- **Per register slot.** Each instance compares one register: bit L of the result is the element lane L holds in that
  register. Lay the ballot axis on the 64 lanes, one element per lane (`BlockedLayout([1], [64], …)`; DSA: one
  `[TILES, 64]` row per register, S11). A layout that holds an element in several lanes sets several bits for it.
- **Per wave.** The mask covers the 64 lanes of the wave that runs it: give each wave its own rows, or replicate the
  data in every wave, as DeepSeek's `BlockedLayout([1], [64], [gl.num_warps()], [0])` does for 64 routing keys (S43).
- **Every lane votes.** Lanes holding masked-out elements are active: fold validity into the predicate (DSA:
  `selected >= 0`, S11; MiniMax: `(r < R) & (ids >= 0) & (ids < 64)`, S44).
- The mask is wave-uniform and lives in an SGPR pair: `&`, `>> 32`, `.to(gl.uint32)` on it are scalar.
- **To branch or loop on it, take one element as a 0-d scalar.** Called on a tensor, an asm returns a tensor, one copy
  per lane: `if mask != 0:` fails with "Boolean value of Tensor with more than one value is ambiguous" (compiled).
  Write `gl.gather(mask, gl.full((1,), 0, gl.int32, layout), 0).reshape()` (S8; `gl.sum` of such a gather in S9,
  lane 63's total in S6): the branch then tests the SGPR pair right after the asm (`s_cmp_eq_u64 s[0:1], 0`), where
  `gl.max(mask, 0)` spends a 64-bit DPP tree of 49 instructions (compiled). An asm on a 0-d scalar returns one (S10).
- A counting sort with no atomics: a validity ballot, then one ballot per expert-id bit, per 64-route stripe (S44; [moe.md](../moe.md)).

### 6.2 First lane: argmax with a lowest-index tie

```python
_, _, _, winner = gl.inline_asm_elementwise(
    "s_nop 1\nv_cmp_eq_f32_e64 $0, $4, $5\ns_nop 1\nv_cndmask_b32_e64 $1, $7, $6, $0\n"
    "s_ff1_i32_b64 $2, $0\nv_readlane_b32 $3, $1, $2\ns_nop 1",
    constraints="=&s,=&v,=&s,=s,v,s,v,v", args=[local_largest, largest, local_id, 2147483647],
    dtype=(gl.uint64, gl.int32, gl.uint32, gl.int32), is_pure=True, pack=1)
```
(S8 :191-200 plus three `s_nop 1`; compiles with Triton 3.8.0 for gfx950. `largest = gl.max(local_largest, 0)`.)

- `s_ff1_i32_b64` returns the lowest set bit, −1 for an empty mask (S45); `v_readlane_b32` takes the lane from bits
  [5:0] of its operand (S29), so an empty ballot reads lane 63, which here holds `INT_MAX` ("An empty ballot reads
  INT_MAX from any lane", S8). The winner is in an SGPR.
- The tie goes to the lowest lane: give lower indices lower lanes. Compare `gl.reduce` on an int32 (value, index) pair:
  78 instructions, result per lane (§5). Every lane computes `combine(own, partner)` (S40), so with a combine that is
  not commutative (NaN scores, a tie rule that keeps its own value) lanes can disagree: read one ([moe.md](../moe.md) §5.3).
- Two ballots and a scalar select (`s_cmp_lg_u64`, `s_cselect_b64`, `~{scc}`) pick a fallback mask when the first is empty (S23).
- **The three `s_nop 1`** (2 wait states each, S46): `gl.max`'s closing `v_readlane` has just written `largest`;
  `v_cmp` writes the SGPR `v_cndmask` reads; the next VALU reads the winner, where LLVM put only `s_nop 0` (compiled).
  The committed router runs its 10 blocks unpadded at entry (`v_readlane_b32 s24, v19, 63`, then `v_cmp_eq_f32_e64
  s[4:5], v28, s24`, compiled at its 256-token Point), where its schema records accuracy passed; the ISA lists the waits.

### 6.3 Rank and count: compaction, aggregated atomics

- `v_mbcnt_lo_u32_b32 d, mask_lo, 0` then `v_mbcnt_hi_u32_b32 d, mask_hi, d` count the set bits of the mask in the
  lanes below this one (S29). The mask may be an SGPR pair (a ballot, `=v,s,s`) or per-lane VGPRs.
- `s_bcnt1_i32_b64` counts an SGPR mask  and writes SCC (S45, §2.2): `~{scc}`. `v_bcnt_u32_b32 d, x,
  acc` adds a 32-bit popcount to `acc`; a 64-bit per-lane mask is two of them with `=&v` (S1). `gl.extra.libdevice.popc`
  is the same on `int32`; on `int64` it compiled to a call (`s_swappc_b64`): the wrapper types an `i64` result as `int32` (S36).
- Compaction: ballot, rank and count, `slot = base + rank`; one `gl.atomic_add` per program reserves them (S4, §2.5).
- Warp-aggregated atomics: for each lane, AND the ballots of every key bit (or their complement) to get the lanes
  with the same key; the lane with rank 0 takes one `gl.atomic_add` for the group and the others `gl.gather` its
  ticket from `v_ffbl_b32` of the group mask. MiniMax 147b's `_ballot_scatter` does this, and no CI run exercises it:
  at its two Points (2048 and 4096 tokens) the kernel takes the ticket path instead (S47).

### 6.4 Lane reads, lane writes, uniform values

- **`v_readlane_b32 $0, $1, $2`** (`=s,v,s`): the value lane `$2[5:0]` holds, into an SGPR, regardless of EXEC
  (S29). A register table with one weight scale per K group in consecutive lanes, read once per group (S9).
- **`v_writelane_b32 $0, $1, $2`** with `"=v,s,n,0"` and `args=[value, LANE, old]` (S8), or `v_writelane_b32 $0, $2,
  $3` with `"=v,0,s,n"` and `args=[old, value, LANE]` (S22): writes an SGPR into lane k of a register and leaves the
  other lanes, regardless of EXEC (S29); the tie keeps the old register. One top-k result per step into lane `slot`.
- **`v_readfirstlane_b32`** (`=s,v`): the lowest active lane's value (lane 0 if EXEC is 0) into an SGPR (S38), for
  values that are uniform but that LLVM may hold per lane: a tile descriptor merged from two branches, a reduction
  result, an LDS-gathered scale (S10, S23, S48). Compiled on a uniform `gl.load` result held in a VGPR, it turned
  `v_cmp` + `s_cbranch_vccnz` into `s_cmp` + `s_cbranch_scc1` and gave the dependent load an SGPR base address.
- **SGPR addresses.** A kernel-argument pointer passed as `WS.to(gl.uint64)` through `"", "=s,0"` and cast with
  `.to(gl.pointer_type(gl.float32, 4))` (address space 4, AMDGPU's constant space: `ptr addrspace(4)` in the llir) is
  read with scalar loads (S49); compiled, four uniform `gl.load`s of a plain kernel-argument pointer became
  `s_load_dword` too: look at the amdgcn before adding the cast ([memory.md](memory.md)). A per-lane value fails (§2.2).
- An asm with `args=[]` computing a per-lane value (the lane id from `v_mbcnt_lo/hi`, mask −1, S50) is a scalar `i32`:
  broadcast, it gave 0..63, but as a scalar it is lane 0's (`if lid < 32:` ran lane 0's branch) : do not branch on it.

### 6.5 `ds_bpermute_b32`

`ds_bpermute_b32 d, addr, data`: each lane reads `data` from lane `(addr / 4) % 64` (S51); lanes reading lanes outside
EXEC got 0 (d not pre-set), and with no LDS allocated it gathered right. MiniMax's `_broadcast_chunk_pair`
(S1) reads the four 16-lane top-k winners from lanes 15, 31, 47, 63 with two `ds_bpermute_b32` and one wait in one asm.

- **The asm waits for itself.** LLVM's waits do not count instructions inside an asm: compiled without the
  `s_waitcnt lgkmcnt(0)`, the destination was read two instructions later with no wait in between. Issue several
  `ds_bpermute`s, then one wait (1, 2 or 4 per block in the MiniMax kernels, S50, S1).
- `gl.gather` with a warp-local lane-dependent index is the same instruction (§5), and two gathers with one index share
  the address and wait together (one `s_waitcnt lgkmcnt(0)` when both feed one expression, compiled). The cited evidence does not compare the MiniMax asm with `gl.gather` in a complete kernel.
- Whole-tile cost depends on permutation, vectorization and configuration. A shuffle path
  need not beat LDS, and spill can dominate either. Compare matched permutations and check scratch
  before treating time as shuffle throughput ([layouts.md](layouts.md) §8).

### 6.6 DPP

DPP routes the first source of a VOP1/VOP2/VOPC instruction from another lane (S52):

| Modifier | Lane n reads | Xor equivalent |
|---|---|---|
| `row_shr:k` | n − k in the same 16-lane row; outside the row: `bound_ctrl` | |
| `row_ror:8` | n ± 8 in the row | xor 8 |
| `row_half_mirror` then `quad_perm:[3,2,1,0]` | 7 − (n & 7), then 3 − (n & 3) | xor 4 |
| `quad_perm:[2,3,0,1]`, `quad_perm:[1,0,3,2]` | within groups of 4 | xor 2, xor 1 |
| `row_bcast:15` with `row_mask:0xa` · `row_bcast:31` with `row_mask:0xc` | lane 15 of the previous row · lane 31, for rows 2–3 | |

- **`bound_ctrl`.** Compiled and decoded from the binary: `bound_ctrl:0` and `bound_ctrl:1` set the same BC bit
  (an out-of-row source reads 0); no `bound_ctrl` leaves BC = 0 (the lane is not written). An in-place add
  (`v_add_u32_dpp $0, $0, $0 row_shr:1`) gives the same result either way; an in-place max, min or `v_mov_b32_dpp`
  does not : read only the lanes the pattern fills, or drop `bound_ctrl`. `row_mask`/`bank_mask` disable writes (S52).
- **Inclusive wave scan**, S6's sequence (compiles with Triton 3.8.0 for gfx950: 13 instructions against 52 for
  `gl.associative_scan`, §5). The leading `v_mov_b32` reads the input without DPP: unpadded, it was exact :

```python
@gluon.jit
def _wave_inclusive_sum(value):
    gl.static_assert(value.type.shape[0] == 64)
    return gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1\ns_nop 1\n"
        "v_add_u32_dpp $0, $0, $0 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:1",
        "=v,v", [value], dtype=gl.int32, is_pure=True, pack=1)
```
  S6 takes it when `E == 64` and falls back to `gl.associative_scan` for any other expert count. It scans one register
  slot across the 64 lanes of one wave: give it `BlockedLayout([1], [64], [1], [0])` at `num_warps=1` and keep the
  shape assert (compiled, 2 elements per thread gave two interleaved scans and a 4-wave tensor four separate ones,
  without an error). The total is lane 63's: `gl.sum(gl.gather(ends, gl.full((1,), 63, gl.int32, layout), 0), 0)` (S6).
- **Row reductions** over `row_shr:8`, `4`, `2`, `1` leave a row's result only in its lane 15 (shown for an in-place
  max ): read lanes 15, 31, 47, 63 (MiniMax d861 gathers `gl.arange(0, 4) * 16 + 15`, S27).
- **Xor exchanges** for an argmax of (value, index) pairs: the table's xor rows, one `v_mov_b32_dpp` per value per
  step and two per value for xor 4 (S28); Triton emits the same DPP for its own xor shuffles (S41).
- **Spacing.** A VALU write then a DPP read of that VGPR needs 2 wait states (S46): `s_nop 1` between steps. Four
  independent chains interleaved in one block (`pack=4`, S13) ran right without NOPs . On gfx950 `v_mov_b64_dpp` with `row_shr:1` fails to assemble ("DP ALU dpp only supports row_newbcast"), with `row_newbcast:3` it
  assembles (the kernels move 64-bit keys as two 32-bit halves, S26); `v_cvt_scalef32_pk_fp4_f32` and `v_cvt_pk_fp8_f32` take no DPP ("dpp variant of
  this instruction is not supported"), `v_cvt_f32_i32_dpp` does assemble (compiled, no kernel run).

### 6.7 `v_permlane16_swap_b32`, `v_permlane32_swap_b32`

`v_permlane16_swap_b32 a, b` swaps the odd 16-lane rows of `a` with the even rows of `b`; the 32 form swaps rows 2–3
of `a` with rows 0–1 of `b` (S38). Both registers change: tie the outputs. Gluon emits these swaps itself for
reductions over lane bits 4–5 and for xor-16/32 shuffles (S39, S41); making a `convert_layout` permlane-sized is in
[layouts.md](layouts.md). The GDN prefill store swaps registers 0–3 with 4–7 of each 8-register block across lane
bit 4, so that K bits 0–2 land in consecutive registers for 8-value bf16 stores, then relabels the axes and asserts
that no data moves ("Entry and exit delays cover GFX950 hazards.", S12):

```python
gl.static_assert(BLOCK_V == 16)
shuffled = gl.inline_asm_elementwise(
    "\n        s_nop 1\n        v_permlane16_swap_b32 $0, $4\n        v_permlane16_swap_b32 $1, $5\n        v_permlane16_swap_b32 $2, $6\n        v_permlane16_swap_b32 $3, $7\n        s_nop 1\n        ",
    "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7", [state], dtype=gl.float32, is_pure=True, pack=8)
matrix = shuffled.reshape((16, 2, 2, 2, 2, 2, 2, 2))
matrix = matrix.permute((0, 5, 2, 3, 4, 1, 6, 7)).reshape((16, 128))
L: gl.constexpr = gl.DistributedLinearLayout([[0, 1], [0, 2], [0, 4]], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64], [0, 8]],
                                             [[0, 16], [0, 32]], [], [16, 128])
matrix = gl.convert_layout(matrix, L, assert_trivial=True)
```

### 6.8 Bit scans

`v_ffbl_b32` returns the index of the lowest set bit, −1 for 0 (S38); `s_ff1_i32_b64` does it for a 64-bit SGPR mask (S45).
Iterate a mask with `i = ffbl(m)`, then `m = m & m - 1` (DSA's live tiles, S11); a 64-bit per-lane mask takes two `v_ffbl_b32`
on its halves (S4). HIP's libdevice has `clz` (`v_ffbh_u32` and a `v_min`, compiled) and `popc`, no count of trailing zeros.

## 7. Mode bits: `s_setreg` and FP16_OVFL

- **The bit.** MODE[23] FP16_OVFL (S53). With 1, fp32 → fp8 saturates (±Inf/out-of-range → ±448); with 0, NaN.
  Seven tested fp16-result instructions (add/mul/fma, their packed forms, fp32 → fp16) overflowed to ±65504 with 1,
  ±Inf with 0; fp32 add/mul stayed ±Inf in both modes. This does not establish all fp16 instructions.
- **MFMA opcode differences matter.** The tested fp32 `16x16x4` / `32x32x2` overflow
  remained ±Inf in either mode; bf16 `16x16x32` / `32x32x16` and scaled F8F6F4
  `16x16x128` / `32x32x64` saturated with the bit set. Do not transfer that result to
  untested f16/non-scaled fp8 opcodes. The original ISA states “CLAMP is not supported” for
  the two bf16 and two F8F6F4 shapes (S62); assembler rejection of `clamp` does not measure
  behavior of a manually encoded bit. Contract effects: [numerics.md](../numerics.md) §2.
- **Triton sets it in the cast-containing function**, not necessarily at kernel entry (§3.2). The 38 accepted
  variants and 7 no-setreg controls are compile evidence only; `.amdhsa_fp16_overflow 0` does not describe
  the mode after a setreg. Rejected rtz/direct wider buffer-store paths are excluded.
- **The Qwen3-Next decode kernels switch it by hand** (2 kernels, first measured on main in d908d01), `is_pure=False`: to 0 at
  entry, fused with the program id (`_ieee_pid`: `"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\ns_mov_b32 $0, $1"`,
  `"=s,s"`), to 1 right before the packed fp8 converts, fused with the quantization scale (`_fp8_scale`:
  `"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\nv_mov_b32 $0, $1"`, `"=v,v"`): "All BF16 rounding boundaries
  precede the wave-wide saturating FP8 mode change." (S13).

Risks:
1. **Order comes from data, not from `is_pure=False`.** Compiled: a bf16 convert written before a non-pure setreg asm
   was scheduled after it when nothing tied them; passing a value computed from the converted data into the asm's
   input kept the convert before it. Route a dependency through the operands, as `_ieee_pid` and `_fp8_scale` do.
2. **An accepted Triton fp8 cast adds Triton's setreg at the start of the cast-containing function, including a noinline callee** (compiled), whatever the hand-written ones do.
3. **Wave-local, not an unconditional program lifetime.** Setting in wave 0 or 2 affected only that wave across
   a 4-wave barrier. Across one noinline call, caller/inside/after modes were (1,1,1), (1,0,0), (0,1,1): callee
   changes survived return. Do not use a prior launch to infer the current wave mode.
4. **Back-to-back setregs of one register**: the ISA asks for 2 wait states (S46, S54); compiled, Triton's entry setreg
   can be adjacent to `_ieee_pid`’s. Follow the ISA delay even if finite readbacks happen to succeed
   without it; that observation does not remove the software hazard requirement.

Prefer an explicit ±448 clamp with Triton's cast, or a clamp with the asm convert (§3.2). Switch the mode only when a
contract needs both behaviours in one kernel ([numerics.md](../numerics.md)).

## 8. Waits and empty asm

- **`s_waitcnt` in asm.** LLVM's own waits do not count the instructions inside an asm (§6.5): an asm that issues LDS,
  SMEM or VMEM work ends with its own wait.
- **Partial `vmcnt`.** DSA waits for part of a commit group: `f"s_waitcnt vmcnt({OUTSTANDING}); s_mov_b32 $0, 0;"`,
  `=s,~{memory}`, `is_pure=False`, `args=[]` (S14). N holds only while the vector-memory instructions per group stay
  fixed (compiled: 12 loads per group at 1 warp, 3 at 4): re-derive it from the amdgcn after any change ([memory.md](memory.md) §11).
- **`s_waitcnt lgkmcnt(0)` to retire LDS reads before a refill** (`=v,~{memory}`, `is_pure=False`). With one wave
  `gl.barrier` emits neither a wait nor `s_barrier`, so the one-warp MiniMax router writes this asm between its
  relaxed reads and the next async copy into the same slot (S6). With 4 waves `gl.barrier` compiles to
  `s_waitcnt lgkmcnt(0)` and `s_barrier`, so the 4-wave GEMM's asm repeats it (S15; [memory.md](memory.md) §11).
- **Empty asm with tied operands** leaves an empty block and is opaque (compiled: `x*3`, pin, `*5` kept two multiplies):
  - `"", "=v,0"`, pure: the value must exist in VGPRs there and nothing folds across it; between two `convert_layout`s
    that each lower to permlane swaps (S55), where compiled with and without it they give 2 `v_permlane32_swap` + 2
    `v_permlane16_swap` and no LDS ([layouts.md](layouts.md) §7), and after each accumulator update (S16);
  - `"", "=s,0"` on a kernel-argument pointer: the address in SGPRs for scalar loads (S49);
  - 16 tied outputs, `pack=16`, `is_pure=False` on the accumulator between K groups: the accumulator is
    materialized there, and the async refill issued after it stays after it, as an impure asm keeps loads (§2.3, S15).

## 9. Rules: when it pays, hazards, checking the result

### 9.1 When inline assembly is worth it

- Write it where Gluon 3.8.0 has no op: e2m1 packing, ballots, lane ranks, a winner in an SGPR, lane writes, an fp8
  convert without the mode change (Gluon's cast adds an `s_setreg`), DPP scans (13 instructions against 52, §5). Partial
  16- and 8-lane sums have one: `gl.sum` lowers them to DPP (9 and 8 instructions with the store, 4 and 3 for a hand asm).
- Not for `gl.exp2`, `gl.clamp`, `fast_dividef`, packed fp32 arithmetic or `popc` on `int32` (the same instruction,
  §4, §5); full-wave `gl.sum` and `gl.max` on int32 or under `amdgpu-ieee` false cost 1–2 instructions more (§5).
- Compare an asm idiom with its asm-free version in the full kernel ([tuning.md](../tuning.md) §2).
  Fewer instructions do not guarantee faster execution; preserve the logical operation and input shape.

### 9.2 Hazards

The hardware does not check some dependencies: ISA Table 11 (S46; [memory_and_sync.md](../../hardware/gfx950/memory_and_sync.md)
§6) and, for MFMA results, Table 38 ([matrix_core.md](../../hardware/gfx950/matrix_core.md) §3). LLVM pads its own
instructions, not those inside an asm. `s_nop N` gives N+1 wait states (`s_nop 1` for 2, `s_nop 15` + `s_nop 3` for
20; memory_and_sync.md §5). Use the ISA encoding and the opcode’s own wait requirement;
a finite timing/readback of another encoding does not establish arbitrary immediates.
An independent instruction counts as one.

| First | Then | Wait states (ISA) | Checked |
|---|---|---|---|
| VALU writes a VGPR | DPP or `v_permlane*` reads it | 2 | LLVM pads its own to 2 |
| VALU writes a VGPR | `v_readlane` reads it | 1 | LLVM pads its own to 1 |
| VALU writes an SGPR (`v_readlane`; the ISA adds `v_cmp`, `v_readfirstlane`) | VALU reads it as a constant | 2 | LLVM pads `v_readlane` → VALU to 2 |
| VALU writes an SGPR | `v_readlane`/`v_writelane` uses it as the lane select | 4 | readlane/cmp/readfirstlane writers: Use the ISA 4-state requirement for lane selection  |
| VALU writes an SGPR | VMEM reads it | 5 | LLVM pads its own to 5 |
| `v_exp` / `v_log` / `v_rcp` / `v_sqrt_f32` | `v_add_f32` reads the result | 1 | Use the ISA 1-state requirement for this consumer  |
| a convert writes a byte or half of a register (OPSEL) | the next convert into that register (S57); a VALU reads a result OPSEL moved (S46) | 1 | no wrong byte without it (convert pair)  |
| The tested XDL bf16 and scaled F8F6F4 MFMAs write a result | a VALU reads it (including DPP, `v_permlane*`, `v_readlane`) | 8 / 12 / 20 for the tested 4- / 8- / 16-pass opcodes, under the Table-38 reading in S60; fp64 `16x16x4` → VALU instead requires 19 states in the ISA | LLVM pads its own tested bf16/scaled opcodes to 8/12/20; choose the opcode-specific delay, not a generic two-state wait  |
| `s_setreg` | `s_setreg` of the same register | 2 | no wrong value without it  |

- **Entry.** Compiled, LLVM put nothing before an asm whose first instruction read a fresh result: a VGPR by DPP,
  `v_permlane16_swap` or `v_readlane`; an SGPR a VALU wrote (the `v_readlane_b32 s0, …, 63` ending `gl.max`/`gl.sum`);
  an MFMA's accumulator (after `v_mfma_f32_32x32x16_bf16`, an asm `s_nop 1; v_permlane16_swap_b32` on it followed at
  once, where LLVM's own read waited 12). Open a block with `s_nop 1` (S12, S26); one that reads an accumulator
  straight from one of the tested bf16/scaled MFMAs, with `s_nop 7`, `s_nop 11` or `s_nop 15` + `s_nop 3` as appropriate (the table). Other opcodes need their own Table-38 row: fp64 `16x16x4` → VALU needs 19 states (for example `s_nop 15` + `s_nop 2`); the 8/12/20 recipe is not universal. The GDN prefill swap reads
  `v_mov_b32` copies written right before it, unpadded by LLVM, and pads them itself (compiled); the same hazard in FlyDSL: [gemm/correctness.md](../../gemm/correctness.md) §7.
- **Inside.** Apply the table, or interleave independent chains (§6.6; both gave exact DPP scans ).
- **Exit.** Compiled: after an asm ending in `v_readlane_b32`, LLVM left 1 wait state before the next VALU read of its
  SGPR in some layouts, where the ISA requires 2. End such a block with `s_nop 1`, as the Qwen3-Next reductions do
  ("Explicit s_nop covers DPP and readlane operand hazards", S2).
- **EXEC.** `v_readlane` and `v_writelane` ignore it (S29); `v_readfirstlane` reads the lowest active lane, lane 0
  when EXEC is 0 (S38); ballots and `ds_bpermute` see active lanes only (S51). In a Gluon kernel your asm runs with
  every lane of the wave active unless it sits under a branch on a value that differs per lane (§6.4); a masked store
  got its own `s_and_saveexec_b64` region (compiled). Counters and register rules: §6.5, §8, §2.2.

### 9.3 Checking the result

- Dump the amdgcn ([debugging.md](../debugging.md)). Each asm instance sits between `;#ASMSTART` and `;#ASMEND`:
  count them (elements per thread / `pack` per call site, one for `args=[]`), read the instruction before and after
  each block against §9.2, and check the registers the constraints produced (`s[0:1]` for a 64-bit `=s`; a
  `v_readfirstlane_b32` you did not write in front of an `s` input is a per-lane value, §2.2).
- **Read stderr**: an `error:` line from LLVM can come with a successful compile: illegal VGPR to SGPR copy (§2.2),
  or `constraint 'n' expects an integer constant expression`, after which the asm is gone (compiled: the index of a
  `range` loop with a runtime bound lost every `v_writelane`; fixed `range(2)` and `range(constexpr=3)` worked in;
  a kernel-argument lane left only `s_endpgm`).
- **What a mistake looks like.** An assembler error anywhere in the string (a typo, a 32-bit `=s` for a ballot, two
  SGPRs in one VALU instruction) raises `RuntimeError: LLD failed to link hsaco … target emulation unknown`; the
  assembler's `error:` line is only on stderr. More constraints than registers (§2.1) abort the process (exit 134):
  "number of input constraints does not match number of parameters", or of outputs, "… return struct elements" (compiled).
- Asm-free versions: §4, §5. Kernels per idiom: [kernel_library.md](../kernels/kernel_library.md), [gdn.md](../gdn.md), [moe.md](../moe.md).

## Evidence

- S1 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b94f5777fd5d2d88.json#kernel.py:53 fp4 convert in `_quantize_groups_3` (pairs by split :52); ballot `_wave_ballot_2` :65; 64-bit popcount :95; `v_mbcnt` rank :107; `_broadcast_chunk_pair` :119, called for lanes 15, 31, 47, 63 :197
- S2 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/f44007374038041f.json#gdn_primitives.py:464 `store_fp8`: clamp :479-480, the pair layout :481, `v_cvt_scalef32_pk_fp8_f32`, uint16 store :484-493; "Shared device-side primitives of the templated Qwen3-Next GDN decode kernel" :2; "also saturates BF16 conversions" :447; `v_pk_fma_f32` :626; DPP sum and max with `s_nop` and `v_readlane` 63 :681, :708 (`|$2|, |$3|` and the floor :714-715), comment :711
- S3 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_decode_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/20f63094b85c00a2.json#kernel.py:26 `v_med3_f32` to ±448 before the fp8 cast
- S4 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim384/8a5a03b603ec1b70.json#kernel.py:120 `v_rcp_f32` and a Newton step for the fp8 group scale; rescaled sigmoid reciprocal :349-367; `_ballot_rank`: ballot + `s_bcnt1_i32_b64` with `~{scc}`, then `v_mbcnt` "=v,s,s" :64; two `v_ffbl_b32` on a 64-bit mask's halves :179-195; one `gl.atomic_add` per program :201
- S5 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/0c0b7dfe26861b02.json#kernel.py:54 rcp, Newton step, residual step, then `v_div_fixup_f32` (`_quantization_scale` :44; per element :155)
- S6 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/94af11a072175a38.json#kernel.py:520 `v_max_f64` on packed keys (`v_min_f64` :527); `_wave_prefix_counts` DPP inclusive scan with its shape assert :660-669, taken at E == 64 :685 (E is the schema's 64 local experts, `w13.shape` :925), lane 63's total :687-688, `gl.associative_scan` otherwise :690; `s_waitcnt lgkmcnt(0)` with a dummy `v_mov_b32 $0, 0` output between the relaxed reads and the refill, in the one-warp router (`warps_per_cta=[1, 1]` :444, `num_warps=1` :970) :483-490
- S7 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/43ffd5e3e5b67298.json#kernel.py:109 `v_mad_u32_u24` slot × row stride + column under `USE_U24`
- S8 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/c3889614985b700a.json#kernel.py:191 `_wave_winner` (docstring "An empty ballot reads INT_MAX" :188); `gl.gather(winner, …).reshape()` :202; `v_writelane_b32` "=v,s,n,0" :237; `largest = gl.max(...)` :222; `llvm_fn_attrs` `["amdgpu-ieee", "false"]` on the routing launch :303
- S9 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/8888fe378338cc8e.json#kernel.py:131 `_read_weight_scale`: `v_readlane_b32` "=s,v,s" on a register table, then `gl.sum(gl.gather(…))` :142
- S10 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#kernel.py:521 `v_readfirstlane_b32` on the tile descriptor, a 0-d `gl.max` (:520)
- S11 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/fbe6ddab5dac5958.json#kernel.py:66 ballot of `selected >= 0` per `[TILES, 64]` row (layout :56); `_first_live_tile` `v_ffbl_b32` :199; `live_tiles & live_tiles - 1` :556; `v_bcnt_u32_b32` :654
- S12 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_max_seq_len1024_n_v_heads16_key_dim128_value_dim128_conv_dim3072/1fd6adb2b91cf5c2.json#kernel.py:712 permlane16 swaps, pack=8, tied; docstring :704-709; `gl.static_assert(BLOCK_V == 16)` :711; relabel, `L` and `assert_trivial` :720-729
- S13 aklr/plugins/sglang/models/qwen3_next/impls/triton/qwen3_next_gdn_decode_group_fp8_quant_n_v_heads8_key_dim128_value_dim128_conv_dim2048/38a742cfd908c997.json#kernel.py:34 `_ieee_pid` and `_fp8_scale` :42; docstring :11; setreg fused with the convert :58; four interleaved DPP chains, comment :223
- S14 aklr/plugins/sglang/layers/dsa_backend/impls/triton/dsa_sparse_mla_decode_n_heads8_kv_dim576_topk2048/ae32cc132679692c.json#kernel.py:145 `_wait_vmem`: f-string `s_waitcnt vmcnt`, "=s,~{memory}", args=[], is_pure=False
- S15 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n7168_k2048/fee7e092eab87fc5.json#kernel.py:80 `s_waitcnt lgkmcnt(0)` "=v,~{memory}" (the `=v` never written) before `wait_group` and `gl.barrier` (:85-86) in a 4-warp kernel (:457); 16 tied outputs, pack=16, non-pure on the accumulator :125
- S16 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/831926aed9523c0f.json#kernel.py:40 `_pin_register`: "" "=v,0"
- S17 triton-lang/triton@c01b677:python/triton/language/core.py:3504 signature; pack :3515; no empty dtype :3520; the docstring's `$n` numbering :3566-3573; is_pure :3588; broadcast of args and result types, a scalar result for scalar args :3608-3623; `create_inline_asm` :3626
- S18 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_core.py:126 `inline_asm_elementwise = builtin(tl_core.inline_asm_elementwise)`
- S19 triton-lang/triton@c01b677:include/triton/Dialect/Triton/IR/TritonOps.td:873 "Exactly which elems it receives is unspecified"; SameOperandsAndResultEncoding :865
- S20 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp:361 blocks of `pack` consecutive elements; sub-32-bit packing, argument by argument :216-245; results as a struct, result by result :264-281, :300-304; `has_side_effects = !pure` :288; undef padding :342-352
- S21 triton-lang/triton@c01b677:lib/Dialect/Triton/IR/Ops.cpp:1442 pure: no memory effects; Speculatable :1451; the tensor's element count a multiple of packed_element :1457-1465
- S22 aklr/plugins/sglang/models/qwen2_moe/impls/triton/qwen2_moe_router_model_dim4096_n_routed512_topk10/4801e216dcff290f.json#kernel.py:166 "The immediate lane operand avoids AMD's scalar constant-bus conflict." and `v_writelane_b32 $0, $2, $3` "=v,0,s,n" :167-172; `_first_matching_expert`: `vcc` as scratch with `~{vcc},~{scc}` :87-94
- S23 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/a1c4a2129730d1a0.json#kernel.py:153 `_wave_first_kernel`: two ballots, `s_cmp_lg_u64`, `s_cselect_b64`, `~{scc}`; DPP job prefix :559; `v_readfirstlane_b32` :596; fp4 convert of `abs($1), abs($2)`, `& 119` and the signs OR-ed in :120
- S24 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:772 conversion to F4/F6 ignores FP16_OVFL; no DPP :774; scale from a VGPR or an inline constant :778; FP4 saturates :780; FP8/BF8 overflow table :795-808, with "`CVT_SR_*` and `CVT_PK_*` support only VGPRs as inputs" :797; an MFMA's F32 result clamps to ±MAX under FP16_OVFL :110-115, the F8F6F4 ones too (their rules table :426, clamp row :447)
- S25 triton-lang/triton@c01b677:python/triton/compiler/code_generator.py:1563 f-strings whose values are constexpr
- S26 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/655723f593921b08.json#kernel.py:515 `_xor_pair`: constexpr `prefix` "s_nop 1\n" + DPP string :516; 64-bit keys as two halves :536
- S27 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/d8610d895915c167.json#kernel.py:676 `assembly[stage]` in a `gl.static_range`; `row_shr` 8, 4, 2, 1 steps :670-685, lanes 16r+15 read :686-689
- S28 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/96e51c1f85a71b7e.json#kernel.py:389 `_shuffle_router`: row_ror:8, row_half_mirror + quad_perm (four `v_mov_b32_dpp`, :393), quad_perm steps
- S29 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_11_vop3a_vop3b_instructions.md:3915 V_CVT_SCALEF32_PK_FP4_F32; PK_FP8 :3793; V_MED3_F32 :2710; V_DIV_FIXUP_F32 :2841; V_READLANE_B32 :4854; V_WRITELANE_B32 :4870; V_BCNT_U32_B32 :4888; V_MBCNT_LO/HI :4903, :4925; V_MAD_U32_U24 :2419
- S30 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/5751284b6965eabb.json#kernel.py:22 `_quantized_bytes`: `v_mov_b32 $0, 0` then the fp4 convert, "=&v,v,v"
- S31 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp:2470 an fp → 8-bit-float cast marks the function `SetFP8ClampingAttr`
- S32 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp:468 `adjustModeRegister`: `llvm.amdgcn.s.setreg` of MODE bit 23 at entry; fdiv → LLVM fdiv :65; exp → `x·log2e`, `llvm.exp2.f32` :264-290; exp2 → `llvm.amdgcn.exp2.f32` with ftz :292-320; clamp → `v_med3` :432-462; `PreciseDivFOp` (`div_rn`) → LLVM fdiv :511
- S33 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:250 `AMD::adjustModeRegister(mod, targetInfo)`
- S34 triton-lang/triton@c01b677:third_party/amd/backend/compiler.py:383 `__HIP_FTZ = True`; `llvm_fn_attrs` "Experimental: intended for development and debugging" :107, added to the kernel :479
- S35 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp:104 `__triton_hip_fast_fdividef` → `rocdl.rcp` × numerator
- S36 triton-lang/triton@c01b677:third_party/amd/language/hip/libdevice.py:14 `popc`: `__ockl_popcount_u64` declared with an int32 result :18; `fast_dividef` :102
- S37 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_fp8_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/1c1747a25af67b8e.json#kernel.py:80 `gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp(-gate))`
- S38 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_08_vop1_instructions.md:519 V_RCP_F32 1 ulp, denormals flushed; V_EXP_F32 :470; V_LOG_F32 :492; V_READFIRSTLANE_B32 :41; V_FFBL_B32 :774; V_PERMLANE16_SWAP_B32 :1470; V_PERMLANE32_SWAP_B32 :1494
- S39 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:466 full-wave DPP reduction tree and `v_readlane` of lane 63; permlane32/16 swaps for lane bits 4–5 on CDNA4 :342-380
- S40 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:372 warp reduction: the target's own reduction first, else a `shuffleXor` butterfly over the reduced lane bits, each lane combining its value with its partner's :379-388
- S41 triton-lang/triton@c01b677:third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp:161 shuffleXor: permlane swap for xor 16/32 on CDNA4; DPP row_half_mirror, row_ror:8, quad_perm for xor < 16 :176-200; `ds_bpermute` for shuffleIdx and shuffleUp :121, :214-221
- S42 triton-lang/triton@c01b677:lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp:73 the scan's warp step is `shuffleUp`
- S43 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/d3bbbce5c0bbc61e.json#kernel.py:134 `_next_expert`: record layout over `gl.num_warps` warps, ballots, `s_ff1`, `v_readlane` :140-160
- S44 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/abc7bbf7a7135f28.json#kernel.py:532 validity in the ballot predicate; one ballot per id bit :533-535
- S45 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_03_sop1_instructions.md:302 S_FF1_I32_B64, −1 when no bit is set; S_BCNT1_I32_B64 sets SCC :202-214; S_MOV_B32, S_MOV_B64 leave it :9-26
- S46 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch04_program_flow_control.md:101 Table 11 software wait states: S_SETREG → S_SETREG :104; SGPR lane select :108, :122; VALU SGPR → VMEM :112; VALU → DPP :114; VALU SGPR → VALU constant :120; VALU → v_readlane :126; OPSEL result position → consumer :127; trans → non-trans :128; VALU → V_PERMLANE :130
- S47 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts256_inter_dim512/147b861bd4c406d2.json#kernel.py:634 `_ballot_scatter`: per-bit ballots :656, ranks and `v_ffbl_b32` on halves, one atomic per group :679, base gathered from the first lane :685; not launched at the schema's Points: `use_tickets = m <= 8192` :1118, the ticket branch :819, `_ballot_scatter` only without tickets :842
- S48 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/5ea30df18b03ab8d.json#kernel.py:374 `v_readfirstlane_b32` on a weight scale read by an LDS gather
- S49 aklr/ops/gemm/impls/triton/gemm_a8w8_blockwise_scaled_n3072_k1536/fcf21622f6c91bbb.json#kernel.py:125 `WS.to(gl.uint64)` through "" "=s,0" :128-135, `gl.pointer_type(gl.float32, 4)` :145
- S50 aklr/plugins/sglang/models/minimax_m2/impls/triton/minimax_m2_mxfp4_moe_shuffled_16x16_model_dim3072_n_routed256_n_experts64_inter_dim1536/b0651fe84db2f670.json#kernel.py:873 `_canonical_lane_offset`: `v_mbcnt` with mask −1, args=[]; one `ds_bpermute_b32` then its wait :910
- S51 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_12_lds_instructions.md:1022 DS_BPERMUTE_B32: address = lane × 4, EXEC on read and write, zero from a disabled source lane
- S52 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch13_microcode_formats.md:1551 DPP BC bit; ROW_MASK, BANK_MASK :1556-1557; row_shr :1568; row_rotate :1569; mirrors :1574-1575; row_bcast 15, 31 :1576-1577; 64-bit data only DPP_ROW* :1580
- S53 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch03_kernel_state.md:105 MODE FP16_OVFL, bit 23
- S54 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch05_scalar_alu_operations.md:236 "Must add an S_NOP between two consecutive S_SETREG to the same register."
- S55 aklr/plugins/sglang/models/qwen3_5/impls/triton/gdn_prefill_group_fp8_quant_n_v_heads16_key_dim128_value_dim128_conv_dim3072/2ecece9650b95f71.json#kernel.py:718 "" "=v,0" between the convert to the middle layout and the convert to the A operand
- S56 aklr/plugins/sglang/models/deepseek_v2/impls/triton/deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256/1204f286c55bb56e.json#decode_input_quantization.py:22 `gl.abs` before the fp4 convert (:24-31); `& 119` and the signs OR-ed back :34
- S57 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:744 CVT_*_F32: "The user must insert a NOP or instruction writing some other destination VREG between the conversions writing the low/high half or bytes of the same destination register."
- S58 triton-lang/triton@c01b677:python/triton/experimental/gluon/language/_semantic.py:559 gather: same rank (:566), equal sizes off the axis (:571-577), result in the index's shape and layout (:579)
- S59 triton-lang/triton@c01b677:lib/Analysis/Utility.cpp:1096 `isWarpLocal`: axis not on warps (:1128-1130), same (block, warp) map (:1143-1145) and lane map (:1155-1156) of the other dimensions; otherwise scratch for the whole source (:1083-1093)
- S61 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_01_sop2_instructions.md:7 S_ADD_U32 carry-out into SCC; S_CSELECT_B64 reads SCC :153; S_AND_B64 sets SCC :176; S_LSHL_B64 sets SCC :358
- S62 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch12_10_vop3p_instructions.md:529 "CLAMP is not supported" in the notes of V_MFMA_F32_16X16X128_F8F6F4; of V_MFMA_F32_32X32X64_F8F6F4 :565, V_MFMA_F32_16X16X32_BF16 :663, V_MFMA_F32_32X32X16_BF16 :699
- S60 harness/skills/aklr-kb/hardware/gfx950/cdna4/ch07_matrix_arithmetic_instructions.md:1 Table 38: MFMA-result wait states depend on opcode and consumer; fp64 16×16×4 to VALU requires 19 states.
