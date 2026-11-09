---
title: "MoE in FlyDSL: the FlyDSL side of the moe/ cards"
kind: topic
ops: [moe, fused_moe, grouped_gemm]
dtypes: [mxfp4, mxfp8, fp8, bf16]
regimes: [sorted-token-ids, "few tokens per expert", group-offsets, variable-k]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - ROCm/FlyDSL@dd8373349:kernels/moe/
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/ast_rewriter.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/compiler/protocol.py
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py:502-505
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/rocdl/__init__.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/quantization/
  - aiter/fused_moe.py
  - aiter/ops/flydsl/
  - aklr/ops/fused_moe/impls/flydsl/fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02b4c053c7.json
---

# MoE in FlyDSL

How to write in FlyDSL what the cards of `../moe/` describe: which FlyDSL calls to use, what FlyDSL's compiler and
JIT do with them, and how FlyDSL's own MoE kernels, aiter's FlyDSL MoE kernels and Primus-Turbo's FlyDSL grouped
GEMMs behave when you call or tune them. Each section is named after the card section it serves
(`grouped_variable_k §2` is § 2 of the moe package's variable-K card, `../moe/grouped_variable_k.md`) and holds only
the FlyDSL part: read the card for the mechanism and the numbers.

Names used on this page:
- "FP8", "BF16", "MX8" and "MX4" are Primus-Turbo's
  `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,bf16,mxfp8,mxfp4}_kernel.py` and "GH" its
  `primus_turbo/flydsl/utils/gemm_helper.py`, at main `ca1fdd67`; "Turbo" is Primus-Turbo.
- `moe_gemm_2stage`, `mxfp_moe` and `moe_2stage_a16wmix` (`a16wmix`) are the three MoE GEMM families for gfx950 in
  FlyDSL v0.3.2's `kernels/moe/` (§ grouped_sorted_tokens §1 below); a fourth file,
  `moe_a8w4_mxscale_gfx1250.py`, targets another architecture.
- Paths under `aiter/` are files of the baseline aiter, `dbd8bf5`.
- "BM" in FlyDSL's code is tile-M. CShuffle is the epilogue that stages C through LDS to reshuffle it into wide,
  coalesced stores.
- Numbers marked `*(field, E<n>)*` are measured values; their conditions are listed under Evidence at the end.

## Route here when

- A card of `../moe/` ends a section with "In FlyDSL: `../flydsl/moe.md` § …".
- You need the map of FlyDSL's, aiter's and Primus-Turbo's MoE and grouped-GEMM kernels (§ INDEX Kernel map) or
  what FlyDSL v0.3.2 ships for a fused MoE (§ grouped_sorted_tokens §1).

## INDEX Expert GEMM (grouped GEMM)

- Turbo pins a newer FlyDSL, and every API named on this page and in the cards exists in v0.3.2.
- FlyDSL v0.3.2 ships no grouped GEMM with K varying per group, so no weight-gradient kernel.
- FlyDSL facts are stated for v0.3.2.

## INDEX Kernel map

#### Primus-Turbo main `ca1fdd67`, `primus_turbo/flydsl/grouped_gemm/`

| File | dtype / passes | Entry points |
|---|---|---|
| `grouped_gemm_bf16_kernel.py` | bf16/fp16: forward NT, dgrad NN, variable-K wgrad | `grouped_gemm_bf16_nt_flydsl_kernel` (:720), `grouped_gemm_bf16_nn_flydsl_kernel` (:888), `grouped_gemm_bf16_variable_k_flydsl_kernel` (:533) |
| `grouped_gemm_fp8_kernel.py` | fp8 E4M3/E5M2 per-tensor scale: forward NT, dgrad NN, variable-K wgrad | `grouped_gemm_fp8_tensorwise_flydsl_kernel` (:2628), `grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel` (:4980) |
| `grouped_gemm_mxfp8_kernel.py` | MXFP8 (E8M0 per 32): forward and dgrad as NT, variable-K wgrad | `grouped_gemm_mxfp8_flydsl_kernel` (:1239), `grouped_gemm_mxfp8_variable_k_flydsl_kernel` (:1835) |
| `grouped_gemm_mxfp4_kernel.py` | MXFP4: forward and dgrad as NT, variable-K wgrad | `grouped_gemm_mxfp4_flydsl_kernel` (:1589), `grouped_gemm_mxfp4_variable_k_flydsl_kernel` (:2154) |
| `grouped_gemm_{fp8,mxfp8,mxfp4}_glu_kernel.py` | GLU (forward) and dGLU (dgrad): fp8 and MXFP4 fuse them into the GEMM epilogue; MXFP8 runs the GEMM, then a separate GLU + dual-quantization kernel. The MX variants also quantize the output | e.g. `grouped_gemm_fp8_tensorwise_epi_glu_flydsl_kernel` (fp8 :138), `grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel` (mxfp4 :314) |
| `../utils/gemm_helper.py` | shared | group scans and lookups, XCD remaps, super-tile decode, 64-bit descriptor bases |

The bf16 kernels reuse Turbo's dense bf16 tile bodies; the fp8 and MX kernels share their main-loop design with
Turbo's dense kernels (`gemm.md` § INDEX Kernel map).

#### FlyDSL v0.3.2, `kernels/moe/`

| Path | What it is | Entry points |
|---|---|---|
| `moe_gemm_2stage/` | fp8 / int8 / int4-weight MoE GEMM, stage 2 atomic or split + reduce | `compile_moe_gemm1` (`gemm1.py:541`), `compile_moe_gemm2` (`gemm2.py:595`), `compile_moe_reduction` (`moe_reduce.py:164`) |
| `mxfp_moe/` | MXFP4 / a8w4 MoE GEMM for gfx950; atomic and non-atomic stage-2 epilogues, the MXFP4-out one persistent | `compile_gemm1_a4w4_port` (`gemm1.py:680`), `compile_gemm2_a4w4_port` (`gemm2.py:78`), host `flydsl_mxfp4_gemm1/2` (`host.py:87`, `:168`) |
| `moe_2stage_a16wmix/` | bf16 activations × MXFP4 / int4 / bf16 weights | `compile_gemm1_a16w4_port` (`gemm1.py:708`), `compile_gemm2_a16w4_port` (`gemm2.py:508`) |
| `moe_sorting_kernel.py` | token sorting by expert, tile-M as the unit | `moe_sorting_flydsl` (:2070) |

FlyDSL's `kernels/mega_moe/` fuses dispatch, both GEMMs and combine across GPUs; the moe package does not cover it.
What aiter's `MegaMoEV2` dispatch costs against TP: `../communication/ep_dispatch_combine.md`.

The baseline aiter carries diverged copies of `mxfp_moe` and `moe_2stage_a16wmix`, none of `moe_gemm_2stage`,
and MoE kernels FlyDSL lacks; the mapping is in § grouped_sorted_tokens §1 below.

## few_tokens How to write it

Steps c and f of the head (`../moe/few_tokens.md` § How to write it), in `g1_fold.py` of the schema file
`4fbd9d02b4c053c7.json`:

```python
# c: one LDS atomic add per routed item (g1_fold.py:1965-1968)
for _r in range_constexpr(_NREP):
    _i = tx_i32 + fx.Int32(_r * 256)
    if _i < fx.Int32(_MT):
        _lds_atomic_add(cnt_tiles, _tk_regs[_r], 1)
# f: rank among this expert's items, in item order (g1_fold.py:2145-2160)
_x = _is_eq(_v, fold_e)          # 1 where this item routes to the block's expert
_inc2 = _dpp_scan_wave(_x)       # inclusive scan inside the wave, then wave totals through LDS
_d = _base + _p2 + _inc2 - _x - _lo
```

## grouped_sorted_tokens §1 What FlyDSL v0.3.2 ships

| Path | Operands | Stage 2 | Grid |
|---|---|---|---|
| `moe_gemm_2stage/` | fp8 E4M3, int8, int8smooth, int4 W (W4A8) | atomics, or partials + reduce | 2D: x = N, y = M-block |
| `mxfp_moe/` | a4w4, a8w4 (fp8 A × MXFP4 W), gfx950 | atomic, non-atomic, CShuffle, MXFP4-/fp8-out | 1D, N fastest |
| `moe_2stage_a16wmix/` | bf16 A × MXFP4, int4 or bf16 W | atomic scatter, optional persistent | 1D, N fastest |
| `moe_sorting_kernel.py` | token sort by expert | — | — |

`moe_gemm_2stage` has f16/bf16/f32 outputs. `mxfp_moe` stage 1 fuses gate+up, SiLU
and an fp4 re-quantization of the intermediate; its MXFP4-out stage 2 is persistent. `moe_2stage_a16wmix` issues
one K=32 MFMA (`mfma_f32_16x16x32_bf16`) per K step on gfx950 and two K=16 MFMAs on any other target
(`a16wmix_use_k16`, `moe_2stage_a16wmix/utils.py:26-38`).

`mxfp_moe` uses its own sorting contract (`cumsum` plus `m_indices`, `mxfp_moe/__init__.py:4-17`).

**In the baseline aiter** (`aiter/ops/flydsl/kernels/`): the FlyDSL code that aiter's `fused_moe`, AKLR's
baseline, may dispatch to by tuned row (other rows run CK or HIP kernels):

| FlyDSL v0.3.2 | baseline aiter | How they relate |
|---|---|---|
| `mxfp_moe/` | `mxfp4_gemm1.py` (`compile_gemm1_a4w4_port` :775), `mxfp4_gemm2.py` (`compile_gemm2_a4w4_port` :86); hosts `aiter/ops/flydsl/mxfp4_gemm{1,2}_kernels.py` | one pipeline, "ported back from aiter" into FlyDSL (`mxfp_moe/__init__.py:6`); the copies have diverged since |
| `moe_2stage_a16wmix/` | `moe_2stage_a16wmix/` (`compile_gemm1_a16w4_port` `gemm1.py:375`, `compile_gemm2_a16w4_port` `gemm2.py:304`) | the same design (persistent above `4 · NUM_CU`, opt-in; aiter's `fused_moe` never passes `persist` to it, `moe_kernels.py:552-553`, `:2048-2074`); the builder defaults `b_cache_mod=2`, but aiter's stage-2 host passes the row's `b_nt`, 0 when unset; line numbers differ |
| `moe_gemm_2stage/` | no copy; only `moe_reduce.py` corresponds (`compile_moe_reduction` :206) | aiter's `flydsl_moe1_*` and non-layout `flydsl_moe2_*` rows run the parametric `mixed_moe_gemm_2stage_common.py` instead, which v0.3.2 no longer ships (`mxfp_moe/__init__.py:16`); its `_abf16_wfp4` and `_abf16_wint4` rows run the `moe_2stage_a16wmix` copy (`aiter/ops/flydsl/moe_kernels.py:641`, `:664`, `:728`, `:747`) |
| `moe_sorting_kernel.py` | `moe_sorting_kernel.py` (`moe_sorting_flydsl` :1934), `UNIT_SIZE = 32` | the same algorithm; aiter sorts with Opus unless `AITER_USE_CK_MOE_SORTING=1` or `AITER_USE_FLYDSL_MOE_SORTING=1` (CK first; `aiter/fused_moe.py:63-66`), and its MXFP4 path with the HIP `aiter.mxfp4_moe_sort` (`aiter/fused_moe.py:161`) |
| — | v2 stage 2: `mxmoe_dispatcher.py` (`mxfp4_moe_gemm2` :616), `mxmoe_gemm_v2.py` (:221) | aiter only |

Most FlyDSL rows in aiter's tuned fused-MoE tables are `flydsl_moe1_*` and `flydsl_moe2_*` (the mixed builders of
`mixed_moe_gemm_2stage_common.py`; the `moe_2stage_a16wmix` copy for the `_abf16_` rows; the v2 stage 2 for
`flydsl_moe2_layout_*`); `flydsl_mxmoe_*` rows are a small minority. Where aiter's `mxfp4_gemm{1,2}` differ from
FlyDSL v0.3.2's `mxfp_moe`: stage 2 is persistent for the plain non-atomic and the MXFP4-out epilogues, not CShuffle
(`_persistent`, `mxfp4_gemm2.py:113`); the non-persistent atomic path keeps the hardware XCD order; the host takes
`use_nt` and `xcd_swizzle` from the tuned row's kernel name (`aiter/fused_moe.py:1646`, `:1833`) instead of
FlyDSL's automatic switch (`../moe/grouped_sorted_tokens.md` §4); and the hosts build a4w4 only, with no fp8-out
epilogue (`_SUPPORTED` in both host files).

## grouped_sorted_tokens §2 How the work is mapped

- Where the stage-1 and stage-2 remap conditions of `../moe/grouped_sorted_tokens.md` §2 sit in the source:
  `mxfp_moe/gemm1.py:752-774` (stage 1 remaps only under `xcd_swizzle > 0`), `mxfp_moe/host.py:110` (default 0),
  `mxfp_moe/gemm2.py:230-254`, `:273-289` (stage 2, both paths). aiter's copy keeps the hardware order on its
  non-persistent path (§ grouped_sorted_tokens §1 above).

- **A guard inside a helper needs `@flyc.jit`.** `a16wmix`'s atomic epilogue helper is a `@flyc.jit` function so
  that `if token_id < M` is rewritten into a device branch. As a plain Python function the `if` runs at trace
  time: on a `Numeric` value it raises, but on a raw `ArithValue` (e.g. an `llvm.load` result) it is always true,
  so the guard disappears silently, the atomic scatter fires on padded rows, and stage 2 ran ~13× slower at a
  very large expert count (`moe_2stage_a16wmix/gemm2.py:39-41`; `Numeric.__bool__`,
  `python/flydsl/expr/numeric.py:502-505`).

## grouped_sorted_tokens §3 Grid shape when the tokens fill only a few M-blocks

- **aiter's v2 stage 2: stage C in fp32.** With `g2_kstatic` the bf16 default leaves 2 A slots and a barrier per K tile
  (`mxmoe_dispatcher.py:187–201`, `mxmoe_gemm_v2.py:734`); *(field, E1)* `g2_bf16_lds=False` won every pair, +0.37%.

## grouped_sorted_tokens §4 Cache policy on the weights

- Like `mxfp_moe` stage 1 at small M (FlyDSL's switch, `../moe/grouped_sorted_tokens.md` §4), FlyDSL v0.3.2's
  `a16wmix` stage 2 loads B with the non-temporal cache modifier, here by default (`b_cache_mod=2`,
  `moe_2stage_a16wmix/gemm2.py:517`). aiter's copy has no automatic switch: its host takes the modifier from the
  tuned row's `b_nt` (`_bnt<N>` in the kernel name) and loads B cached when the row gives none
  (`aiter/ops/flydsl/kernels/moe_2stage_a16wmix/__init__.py:228-230`; `aiter/fused_moe.py:1555`).

## grouped_sorted_tokens §6 Stage 2: atomics or split + reduce

- **`moe_gemm_2stage` stage 2** (`compile_moe_gemm2`, `moe_gemm_2stage/gemm2.py:595-619`): `out_dtype` "f16",
  "bf16" or "f32" picks the atomic form; `accumulate=True`, the default, adds into the output with atomics, and
  `accumulate=False` writes the per-(token, slot) partials, f16/bf16 only, for `compile_moe_reduction`
  (`moe_gemm_2stage/moe_reduce.py:164-203`).
- **`mxfp_moe` stage 2 variants** (`mxfp_moe/host.py:34-48`): atomic at BM 16/32/64 with or without non-temporal
  B, non-atomic, CShuffle, MXFP4-out and fp8-out at BM 128 (CShuffle also at 32/64). FlyDSL's host defaults
  `xcd_swizzle = 3` for the non-atomic and fp8-out epilogues and routes BM-128 non-atomic to CShuffle
  (`mxfp_moe/host.py:202-209`); aiter's takes these from the tuned row (§ grouped_sorted_tokens §1 above).
- The e2m1 pack and unpack of MXFP4 route rows are `rocdl.cvt_scalef32_pk_fp4_bf16` (the ROCDL dialect op that
  `python/flydsl/expr/rocdl/__init__.py:17` re-exports; used at `mxfp_moe/gemm1.py:381`) and
  `rocdl.cvt_scalef32_pk_f32_fp4` (`python/flydsl/expr/rocdl/__init__.py:718`).

## grouped_workflow §2 Step 0 — reference kernels and the body the benchmark reaches

- **FlyDSL v0.3.2's grouped kernels are all MoE GEMMs over sorted token ids** (FlyDSL's `kernels/moe/`,
  `../moe/grouped_sorted_tokens.md`): it ships no NN and no variable-K grouped GEMM. References for those, and the
  kernels most of the other cards measured, are Primus-Turbo's FlyDSL grouped kernels (§ INDEX Kernel map above).
  Turbo pins a newer FlyDSL: copy the technique,
  and check every API you use against v0.3.2.

## grouped_measurement §4 Timing hygiene

- **Compile once and call the compiled object; compile beta=1 builds on a scratch output.** The object
  `flyc.compile` returns skips the per-call work of a `@flyc.jit` launcher, and compiling runs the kernel once
  (§ grouped_host_autotune §2 below; `attention.md` § measurement §6).
- In an in-process A/B (`../moe/grouped_measurement.md` §4), load each version as its own module. FlyDSL's cache key
  hashes the launcher's source, the kernels it reaches and plain helpers from its own directory, not plain helpers
  read as globals from another directory or non-scalar closure values (`debug-flydsl-kernel.md` § What the cache
  key sees): two versions that differ only there get one key, and one binary then serves both arms.

## grouped_measurement Verify

- beta=1 builds were compiled against a scratch output (§ grouped_measurement §4 above).

## grouped_host_autotune §2 Two launch modes: eager and CUDA-graph capture

- **What a call costs (v0.3.2).** Calling a `@flyc.jit` launcher re-checks captured globals, binds the signature
  and rebuilds the cache key before it reaches the cached call state (`JitFunction.__call__`,
  `jit_function.py:1358-1400`). The object returned by `flyc.compile` goes straight to the call state
  (`CompiledFunction.__call__`, `jit_function.py:1613-1635`). Measured gaps: `gemm.md` § autotune §6,
  `attention.md` § measurement §6.
- **Mode split.** Eager calls the compiled object; under capture the raw `@flyc.jit` closure is launched, because
  in Turbo's measurement the compiled object "regresses under capture" (FP8:2743-2754, 5060-5074;
  `run_eager_or_capture`, MX8:79-86). FlyDSL v0.3.2 has no capture-specific path in `jit_function.py`; treat this
  as a measured property of that stack and re-measure on yours. *(field, E2)* On a small shape the split took the
  eager call from 18.4 to 17.2 µs.
- **Compiling runs the kernel once** (`_compile_impl`, `jit_function.py:1662`): build a beta=1 launch against a
  scratch output (`compile_with_scratch_out`, GH:86-95; `../gemm/epilogue.md` §7,
  `gemm.md` § epilogue §7).
- **Pass the stream as an `fx.Stream` argument.** A launcher without one gets a NULL stream slot, i.e. the default
  stream, in its call state (`_build_call_state`, `jit_function.py:1123-1125`).

## grouped_host_autotune §6 beta in the race

- Compile the beta=1 candidates against the scratch output too, not only time them there: compiling runs the kernel
  once and adds one more product into the buffer (§ grouped_host_autotune §2 above).

## grouped_host_autotune §8 Do not let a race swallow build errors

- In FlyDSL a configuration over the LDS limit raises nothing: the launch fails on stderr only and the output is
  never written (`gemm.md` § autotune §7).

## grouped_host_autotune Verify

- Under capture the raw `@flyc.jit` closure is launched, not the `flyc.compile` object (§ grouped_host_autotune §2
  above).

## grouped_group_mapping §5 Tile → group lookup: four forms and their price

- FlyDSL v0.3.2 has every op the lane-resident table needs (DPP scan, ballot and popcount, readlane):
  `rocdl.update_dpp`, `rocdl.ballot`, `rocdl.readlane`, `rocdl.readfirstlane`, and `llvm.intr_ctpop` from
  `flydsl._mlir.dialects.llvm`.

## grouped_group_mapping §7 64-bit group bases, 32-bit in-tile offsets

- Pass large tensors at full rank: a flat view of more than 2^31 elements overflows the int32 shape pack
  (MX4:1616).
- The clamp of a record count at zero is `arith.maxsi(n, 0)` on index values (FP8:3442-3451).

## grouped_lowprec §2 The scale preshuffle is a per-call helper kernel

- In FlyDSL the GEMM's host launcher is its `@flyc.jit` stub: launch the scale repack from it, so the repack costs
  no extra Python dispatch (MX8:1101-1120).

## grouped_lowprec §5 Quantization: a large share of the time, a small lever

- Turbo's grouped MXFP4 quantizer launches its metadata kernel (`pre`) and the quantizer back to back from one
  `@flyc.jit` stub (`primus_turbo/flydsl/quantization/mxfp4_grouped_quant.py:575-597`). Its grouped MXFP8 quantizer,
  the one the card's E16 measured, has no prologue kernel at `ca1fdd67`: each workgroup scans the offsets itself
  (`mxfp8_quant_flydsl.py:884-886`, `:1176`).

## grouped_variable_k §2 A K loop with a run-time trip count

**The chunk loop in v0.3.2.** The outer loop over chunks is a runtime `range(n_chunks)`, which becomes an `scf.for`
with a run-time bound; the inner `range_constexpr(CHUNK)` unrolls at trace time. The cur/next buffer names of the
unrolled body are Python-level, so an even CHUNK is what puts them back in their starting roles at every chunk
boundary (MX8:1428). MX8's tail is a `range_constexpr(chunk)` loop whose steps sit under `if k_abs < k_iters`
(MX8:1574-1615).

**What a runtime loop body may contain (v0.3.2).** The rewriter turns a runtime `range` in a `@flyc.kernel`
body into a body function plus `scf.for` and infers the carried state: every pre-loop name the body assigns,
subscript-assigns (`acc[i] = …`) or calls a method on (`_collect_assigned_vars`,
`compiler/ast_rewriter.py:156-222`). Each carried name must flatten to IR values, i.e. be a DSL value or a
list/tuple/dict/SimpleNamespace of them (`compiler/protocol.py:97-108`); a helper object such as a loader raises
`TypeError: Cannot extract IR values`.
- Keep the body to calls of plain functions plus rebinding of DSL values. MX8's chunk body is one module-level
  function that returns the eight swapped LDS buffer handles, and the loop carries only those (MX8:1571-1606);
  BF16 hoists its tile bodies into free functions "for the same ast-rewriter reason" (BF16:429).
- Accumulators in register memory (`fx.make_rmem_tensor`) are not carried when the body only passes them to
  functions that update them in place (MX8:1576-1608); writing `acc[i] = …` or calling a method on them in the body
  makes them carried.
- BF16 emits its chunk loop through a helper, `emit_for`, that calls the rewriter's `scf_for_dispatch`
  (`compiler/ast_rewriter.py:1006`) with no carried state (GH:2134-2143); the body is then a plain Python closure,
  and its Python-level buffer rotation is resolved at trace time.
- The three loop forms and what each carries: `debug-flydsl-kernel.md` § The three loop forms.

## grouped_variable_k §4 MX: the lever is when the scales are loaded

- *(field, E3)* In the MXFP8 wgrad change of the card's §4 (`../moe/grouped_variable_k.md` §4), keeping the accumulators
  as SSA values alone gained 0; the scale prefetch gave about all of it. The kept form is a chunk-local body with
  SSA accumulators and per-step scale prefetch, touching register memory (`fx.make_rmem_tensor`, § grouped_variable_k §2
  above) only at chunk boundaries (`_wgrad_ssa_chunk`, MX8:166).

## grouped_variable_k §9 Deep groups: deterministic split-K

- The host launcher of FP8's wgrad is its `@flyc.jit` stub: the reduce kernel is launched from it, on the same
  stream, right after the GEMM (FP8:4617).

## grouped_variable_k Verify

- The runtime loop body carries only DSL values (§ grouped_variable_k §2 above).

## Evidence

- E1 [2026-10, gfx950, FlyDSL 0.3.2, N=? pairs, AKLR graph timer, the kept version beside the one before it, remeasured; a8w4 SiTUv2 fused MoE, 112 of 896 experts local, top-16, model dim 3584, inter 3072, at 8 / 12 / 16 / 24 / 32 tokens; AKLR's graph timer, L2 flushed per replay, median] `g2_bf16_lds=False` at BM 32, BK 256, BN 128/256: +0.37% [+0.31, +0.44], 5 of 5 cases up; faster by 0.15–0.8 µs in every interleaved pair (local:/shared_nfs/kyle/moe/kf/runs/ep8_bw_s/workspace/forge_experiments/ablation.md:17; lessons/iter_005.md:44–46)
- E2 [2026-06, gfx950, FlyDSL ?, N=?, eager per-call time] mode split (compiled object in eager, raw `@flyc.jit` under capture): a small shape's eager call 18.4 → 17.2 µs (~7%) (kernel and shape not named; the note sits in the fp8 GEMM tuning skill's autotune design, beside the grouped `_balanced_group_offs` construction; recorded by 2026-06-17, myskill 2563ed1 `flydsl-fp8-gemm-tuning/06-autotune-design.md:85`; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/13-autotune-fleet.md:42)
- E3 [2026-07, gfx950, FlyDSL ?, N=?, ?] MXFP8 variable-K wgrad with chunk-local SSA accumulators and per-step scale prefetch: 930/1047/530/307 → 851/936/478/274 µs (~9%); SSA accumulators alone 0, scale prefetch alone ≈ all of it (Turbo MXFP8 grouped wgrad, four shapes not named; recorded by 2026-07-05, myskill 99db6ee; local:/workspace/code/gpt_oss_docker/myskill/optimizer/methodology/12-mxfp8-grouped.md:15-16)
