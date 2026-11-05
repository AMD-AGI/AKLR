---
title: FlyDSL — the kernels the baseline aiter ships, so you don't rewrite one
kind: topic
lever: flydsl_kernel_library
ops: [gemm, moe, attention]
dtypes: [any]
regimes: [any]
gens: [gfx950]
flydsl: 0.3.2
sources:
  - aiter/ops/flydsl/gemm_kernels.py
  - aiter/ops/flydsl/kernels/hgemm_dispatch.py
  - aiter/ops/flydsl/kernels/splitk_hgemm.py
  - aiter/ops/flydsl/kernels/small_m_hgemm.py
  - aiter/ops/flydsl/kernels/preshuffle_gemm.py
  - aiter/ops/flydsl/kernels/splitk_epilogue.py
  - aiter/ops/gemm_op_a8w8.py
  - aiter/tuned_gemm.py
  - aiter/ops/flydsl/moe_kernels.py
  - aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py
  - aiter/ops/flydsl/mxfp4_kname.py
  - aiter/ops/flydsl/fhmoe.py
  - aiter/ops/flydsl/moe_sorting.py
  - aiter/ops/flydsl/kernels/moe_reduce.py
  - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
  - aiter/ops/flydsl/linear_attention_kernels.py
  - aiter/ops/flydsl/linear_attention_prefill_kernels.py
  - aiter/ops/flydsl/mla_reduce_kernels.py
  - aiter/fused_moe.py
  - aiter/mla.py
---

# The FlyDSL kernels in the baseline aiter

**This is about reading or copying aiter's kernels, not writing your own.** The kernels below are in
`aiter/ops/flydsl/` of the baseline aiter (`../../SKILL.md` § Also in AKLR): host wrappers there, kernel bodies under
`kernels/`. A path given in full is from aiter's root; after that a file is named without its directory, and
`kernels/…` is `aiter/ops/flydsl/kernels/…`. AKLR kernels do not import aiter: read these kernels to know what you
are racing, and copy code out of them if you reuse it. The dense GEMMs come first; the other FlyDSL families in
that directory (MoE, linear attention, MLA reduce, EP dispatch/combine and smaller fused ops) are mapped in
§ Other FlyDSL families below.

## Route here when
- You are about to write a FlyDSL GEMM and should first check whether aiter has one for the dtype.
- You need to know which aiter FlyDSL GEMM a tuned row runs, and how its split-K reduces.
- You need to find the FlyDSL kernel behind a non-GEMM aiter op, what selects it, and whether it is
  reproducible run to run.

**Go to `flydsl_authoring_method.md`** once you have established that nothing here fits, and to
`../../gemm/INDEX.md` for writing a dense GEMM.

## Dense GEMM families (gfx950)

| Family | Entry → body | Reached from |
|---|---|---|
| bf16/fp16 HGEMM | `flydsl_hgemm` (`aiter/ops/flydsl/gemm_kernels.py:859`) → `compile_flydsl_hgemm_kernel` (`kernels/hgemm_dispatch.py:10`) → `compile_hgemm_kernel` (`kernels/splitk_hgemm.py:98`) | a `libtype=flydsl` row of the bf16 tuned table, through `flydsl_gemm` (`aiter/tuned_gemm.py:474`) |
| bf16 small-M HGEMM | the same entry with `kernel_family="small_m"` → `compile_small_m_hgemm_kernel` (`kernels/small_m_hgemm.py:386`) | only a tuned row whose name carries the `_small_m` suffix; the catalog does not generate them (`gemm_kernels.py:652-685` is disabled) and the shipped tables name none |
| fp8/int8 per-token × per-channel, B preshuffled | `flydsl_preshuffle_gemm_a8` (`gemm_kernels.py:1022`) → `compile_preshuffle_gemm` (`kernels/preshuffle_gemm.py:127`) | a `flydsl_bpreshuflle_…` row of an a8w8 bpreshuffle table, through `gemm_a8w8_bpreshuffle_flydsl` (`aiter/ops/gemm_op_a8w8.py:164`); the op takes fp8 B only (`gemm_op_a8w8.py:730`) |
| fp8 8-wave, B preshuffled | `run_gemm_a8w8_bpreshuffle_8wave` (`aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py:265`) → `compile_fp8_gemm_8w` (`kernels/gemm_a8w8_8wave.py:326`) | the same tables, rows named `flydsl_bpreshuffle_8w_…` (`gemm_op_a8w8.py:181`) |

The small-M family is bf16 only, fixes `tile_m` 16 and one wave along M, and launches `ceil(M / 16)` row
blocks (`gemm_kernels.py:781-791`, `small_m_hgemm.py:403-409, 1402-1407`); its disabled catalog only
enumerated M ≤ 16.

## How the bf16 path is reached

1. `aiter.tuned_gemm.gemm_a16w16` (`aiter/tuned_gemm.py:290`) is the entry, and the call AKLR's `gemm_baseline`
   makes (`aklr/ops/gemm/baseline.py`): time aiter through it.
2. The lookup (`get_GEMM_A16W16_config`, `tuned_gemm.py:130`) matches `(gfx, cu_num, M, N, K, bias, dtype,
   outdtype, scaleAB, bpreshuffle)`, with M as is, then padded twice by `get_padded_m` (the padding is in
   `../../gemm/small_m.md` §7).
3. A `flydsl` row is used only if FlyDSL is importable on an architecture it supports
   (`is_flydsl_available`, `aiter/ops/flydsl/utils.py`) and its `kernelName` is in the catalog or parses with
   `_HGEMM_KERNEL_RE` (`gemm_kernels.py:45, 579`); otherwise the lookup moves on, with a logged warning when
   only the name failed (`tuned_gemm.py:163-175`).
4. Without a tuned row the default is hipBLASLt, asm, skinny, Triton or torch, never FlyDSL: **a FlyDSL GEMM
   runs only where a tuned row names it.**
5. `flydsl_gemm` fuses a bias only when the output dtype is the input dtype and the bias has that dtype, and
   otherwise adds it after the GEMM (`tuned_gemm.py:493-524`). The kernel's output is always the input dtype;
   a row with an fp32 `outdtype` runs the bf16 GEMM and casts the result (`tuned_gemm.py:526`).

## Split-K: how each family reduces

| Family | Reduction | Run to run |
|---|---|---|
| HGEMM | split 0 writes zero (or the bias) into C with `sc0 sc1` stores and raises a per-tile signal; the other splits spin on it; every split rounds its partial to the output dtype and adds it into C with a bf16x2/f16x2 `atomicrmw fadd` at agent scope; the last arrival resets the tile's counter and signal (`splitk_hgemm.py:401-504, 1015, 1059-1066`). No workspace. | **not bitwise reproducible** at split-K ≥ 3, or at split-K 2 with a bias. Measured: 400 of 400 repeats differed from the first at split-K 2 with a bias and at 3, 4 and 8; none at split-K 1, or 2 without a bias. |
| small-M HGEMM | the first split to arrive at a tile zeroes C and publishes it; partials are added with the same atomic (`small_m_hgemm.py:674-730, 1010-1017`). | the same, measured the same way. |
| fp8 8-wave | no split-K: its kernel names carry only the tile, `waves_per_eu` and `xcd_swizzle` (`gemm_a8w8_bpreshuffle_8wave.py:247-262`). | — |
| fp8/int8 preshuffle | each split writes its own fp32 plane; the last split to arrive at a tile sums the planes in a fixed pairwise order and resets the tile's counter (`splitk_epilogue.py:29-161`). Workspace and counters are a persistent per-stream buffer (`gemm_kernels.py:989-1003`). | bitwise reproducible. Measured: no difference in 400 repeats at split-K 2, 4, 8 and 16. |

## Before you reuse one
- **It is dispatched only through a tuned row.** A shape without one never runs FlyDSL (step 4 above).
- **Copy, do not import.** A AKLR kernel carries its code: copy what you reuse into its sources and import it
  relatively. The audit does not check imports (`aklr/ops/README.md`), and no committed AKLR kernel imports aiter.
- **Persistent buffers are an aiter choice.** Both split-K schemes keep their counters (and the preshuffle
  family its workspace) alive across calls in an `lru_cache`. A AKLR kernel allocates scratch per call, or once
  per shape in its compile function and rewritten whole before it is read, so its counters must be
  zeroed on every call.
- **Re-measure on your hardware and your shapes.** A tuned row was chosen on one machine's timing; the row, not
  the kernel, is what changes between deployments.

## AKLR's own bf16 GEMM kernels

Each lives in `aklr/ops/gemm/impls/flydsl/<name>/<uuid>.json`, its source in `kernel.sources[0].content`. All
are deterministic (no atomics) and pass AKLR's audit: the nearest working code for a new bf16 GEMM.

| Kernel | Read it for |
|---|---|
| `gemm_n144_k7168` | split-K at every m with an in-launch fixed-order reduce; flags polled through an atomic monotonic `llvm.LoadOp` |
| `gemm_n768_k7168` | flag-gated reducer workgroups in the same launch, one reducer wave per sub-tile; K splits spread over the XCDs |
| `gemm_n1280_k7168` | reducer workgroups at m 16–112; 24-bit packed partials, 3 dwords per 4 values |
| `gemm_n1536_k7168` | flags compared against a per-dispatch tag (`llvm.amdgcn.dispatch.id`), so nothing is reset per call |
| `gemm_n2112_k7168` | reducer workgroups for mid m; a split-major XCD order for the m=893 fold |
| `gemm_n3584_k7168` | every split reduces its share of the tile's rows from planes laid out on whole cache lines |
| `gemm_n7168_k768` | short K split across a workgroup's waves and summed in LDS; a per-wave LDS DMA ring for m 17–32 |
| `gemm_n7168_k1536` | aiter's async-LDS HGEMM without split-K; operand-swapped MFMAs for vector epilogue stores; a tail launch |
| `gemm_n7168_k1792` | kernel arguments preloaded into SGPRs (`amdgpu-kernarg-preload-count`) on a multi-stage LDS DMA ring |
| `gemm_n1536_k128` | K = 128 in one tile: aiter's async-LDS HGEMM at split-K 1 with per-m tiles and `sc1` C stores |
| `gemm_n6016_k7168` | split-K 2 folded through dispatch-tagged chunks, no flags, at m ≤ 32; N-tail tiles |
| `gemm_n6144_k7168` | the last split to arrive reduces mid-m tiles; three N-waves; 48-column split-2 tiles at small m |
| `gemm_n6288_k7168` | two workgroups per column strip, each half of K, polling each other's launch-tagged partials |
| `gemm_n7168_k3584` | the `gemm_n7168_k1792` streaming design at K = 3584, plus a tiled path for m ≥ 893 |
| `gemm_n8448_k7168` | appended reducer workgroups with 24-bit packed partials switched per row; 320×256 tiles at m = 16384 |
| `gemm_n18432_k1536` | one workgroup per CU storing a 72-column line-aligned block plus a strip slice, computed as a 96-wide tile |
| `gemm_n163840_k256` | very wide N, short K: K over a workgroup's waves summed in LDS; operands as `fx.Pointer`, all preloaded |
| `gemm_n20480_k7168` | one workgroup per CU streaming an 80-column band of B at m ≤ 112, with a per-m K stagger; waves with no rows below m skip the last A DMA |
| `gemm_n7168_k35840` | split-K 8 or 2 with each slice on its own XCDs, folded in the same launch; whole-row-multiple head tiles plus a split-K tail launch at m = 2732 and 9397 |
| `gemm_n7168_k4224` | the `gemm_n7168_k3584` design at K = 4224, with split-K over one M block folded in fixed order at m 49–112 |
| `gemm_n2304_k1536` | a per-row LLVM scheduler strategy (`amdgpu-sched-strategy`); two tiles per persistent workgroup, the second walking K in reverse |
| `gemm_n3072_k512` | a persistent grid with no barrier between tiles: the K-block barrier mid-block, C staged per wave, the ragged last round as smaller tail items; half of each XCD's workgroups start late |

## AKLR's own FlyDSL MoE kernels

Each lives in `aklr/ops/fused_moe/impls/flydsl/<name>/<uuid>.json`, its sources in `kernel.sources`. All take MXFP4
weights with E8M0 scales per 32, are `determinism: bounded`, compile per token count (`4fbd9d02…` compiles its stage 2
once for all of them) and pass AKLR's audit. Copy their techniques, not their host structure: three of them
(`1fee5982…`, `ef15e878…`, `1bf41ccb…`) cache their launch in a module-level `_RUN_CACHE` dict filled from `run`, and
`1fee5982…` reads its own file there, which `aklr/ops/README.md` does not allow (a `functools` compile function, no
I/O in launch code). Four sum the top-k routes in a fixed order without atomics on the output; the 257-expert
`4fbd9d02…` kernel adds into it with atomics. How the kernels map work and what moved them:
`../../moe/grouped_moe_inference.md`. Several items below run only at Points their file does not
serve; which run where: `../../moe/INDEX.md` § Kernels to start from.

| Kernel | Read it for |
|---|---|
| `fused_moe_interleave_…_model_dim4096_…` (`df5c661e…`, `5db89e7d…`, `a5a34343…`, `9dcb4d79…`: one `kernel.py`) | when M·topk ≤ 64, no routing launch: each route finds its local expert from `expert_mask` with `ballot` + `ctpop` (`_local_expert`), and K is split over 4 waves and summed in LDS (`_reduce_waves`); bf16 MFMA on fp4 weights converted with `cvt_scalef32_pk_bf16_fp4`; non-temporal stage-1 weight loads only under expert parallelism; stage 2 sums the routes in LDS |
| `fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256/4fbd9d02…` | the routing sort folded into stage 1, rebuilt per workgroup with an LDS histogram and a DPP scan (`_lds_atomic_add`, `_dpp_scan_wave`; `../../moe/decode.md`); MXFP4 quantization of the activations inside stage 1; a bf16x2 atomic stage 2 ported from aiter's `mxfp4_gemm2.py`, with kernel arguments preloaded into SGPRs |
| `…_n_experts257_topk9_inter_dim256/1fee5982…` | the route that takes slot `% BM == 0` writes its tile's descriptor, so no tile-building kernel at M ≤ 256 (`group_routes`, `fuse_tiles`); `ds_bpermute` gathering packed MXFP4 activations into MFMA lane order; a loop-carried prefetch through `range(..., init=...)` |
| `…_n_experts33_topk9_inter_dim2048/ef15e878…` | one XCD remap for every GEMM (`_chiplet_order`) with `GROUP_M` row tiles inside an expert; route ranks from ballot and popcount with one atomic per wave or workgroup (`_route_block`); one aligned scale dword reused across column halves and K slices (`PACKED_SCALES1`/`PACKED_SCALES2`); stage-2 partials stored as unscaled fp8 and summed in a fixed order (`_reduce_fp8x4`) |
| `…_n_experts65_topk9_inter_dim2048/1bf41ccb…` | at m = 1, split-K over the waves of one workgroup with SiLU and requantization after the LDS sum (`FUSE_SPLIT_QUANT`); routing by one workgroup's LDS histogram while the launch's other workgroups quantize the input (`_make_small_route`); the whole stage-2 K panel in LDS so the MFMA loop needs no cross-wave barrier (`FULL_STAGE_A`) |

## Other FlyDSL families in the baseline aiter

Same layout as the GEMMs: host wrapper in `aiter/ops/flydsl/`, body under `kernels/`, and nothing runs unless
`is_flydsl_available()` (`aiter/ops/flydsl/utils.py:72`). `aiter.fused_moe.fused_moe` (`aiter/fused_moe.py:470`) is
the call `fused_moe_baseline` makes (`aklr/ops/fused_moe/baseline.py`), in sglang's environment: price aiter's MoE
through it. Everything here runs on gfx950 unless § Not for gfx950 lists it.

### MoE (grouped GEMM)

How these kernels map work and what moved them: `../../moe/grouped_moe_inference.md`; how they relate to FlyDSL
v0.3.2's own `kernels/moe/`: `../moe.md` § grouped_moe_inference §1.

| Family | Computes | Entry → body | Reached from |
|---|---|---|---|
| 2-stage MoE, fp4/fp8 weights | stage 1: gate+up GEMM over expert-sorted rows plus the activation, optionally quantizing its output to fp4/fp8; stage 2: down GEMM, router-weighted | `flydsl_moe_stage1` (`aiter/ops/flydsl/moe_kernels.py:1884`), `flydsl_moe_stage2` (`moe_kernels.py:2277`) → `compile_mixed_moe_gemm1` / `compile_mixed_moe_gemm2` (`kernels/mixed_moe_gemm_2stage.py:30, 91`) → `compile_mixed_moe_gemm1_common` / `compile_mixed_moe_gemm2_common` (`kernels/mixed_moe_gemm_2stage_common.py:127, 3425`) | `fused_moe` → `get_2stage_cfgs` (`aiter/fused_moe.py:2142`): a tuned row whose `kernelName1`/`kernelName2` start with `flydsl_` (`fused_moe.py:2571-2579`) |
| 2-stage MoE, bf16 activations × fp4/int4 weights | the same two stages | the same entries; `compile_flydsl_moe_stage1/2` (`moe_kernels.py:641, 728`) send this dtype pair to `compile_gemm1_a16w4_port` (`kernels/moe_2stage_a16wmix/gemm1.py:375`) / `compile_gemm2_a16w4_port` (`kernels/moe_2stage_a16wmix/gemm2.py:304`) | as above |
| a4w4 "mxmoe" | the two stages for MXFP4 activations and weights, N and K tiles fixed at 256 | `flydsl_mxfp4_gemm1` (`aiter/ops/flydsl/mxfp4_gemm1_kernels.py:70`) → `compile_gemm1_a4w4_port` (`kernels/mxfp4_gemm1.py:775`); `flydsl_mxfp4_gemm2` (`mxfp4_gemm2_kernels.py:102`) → `compile_gemm2_a4w4_port` (`kernels/mxfp4_gemm2.py:86`) | rows named `flydsl_mxmoe_g{1,2}_a4w4_<BM>x256x256[_flag…]` (`fused_moe.py:2522`; grammar `aiter/ops/flydsl/mxfp4_kname.py:7-9`) |
| v2 stage 2 | stage 2 only, behind whichever stage 1 the row names, mxmoe included | `mxfp4_moe_gemm2` (`kernels/mxmoe_dispatcher.py:616`) → `gemm2_body_v2` (`kernels/mxmoe_gemm_v2.py:221`) | `kernelName2` = `flydsl_moe2_layout_…` (`fused_moe.py:1899, 2612-2617`; grammar `mxfp4_kname.py:19-24`) |
| FHMoE | the 2-stage MoE over MXFP4 routed experts plus one FP8 E4M3 shared expert, id `E - 1` (`aiter/fhmoe.py:74-89`) | `flydsl_fhmoe_stage1` / `flydsl_fhmoe_stage2` (`aiter/ops/flydsl/fhmoe.py:253, 339`) → `compile_mixed_fhmoe_gemm1/2` (`kernels/fhmoe.py:17, 85`) → the same `_common` builders | `fused_moe` with `shared_w1`… or `shared_expert_id` set (`fused_moe.py:507-516`) → `aiter/fhmoe.py`; rows of `aiter/configs/tuned_fhmoe.csv` (`AITER_CONFIG_FHMOE`) |
| MoE sorting | counting sort of the `(slot << 24) \| token` ids into expert blocks: `sorted_token_ids`, `sorted_weights`, `sorted_expert_ids`, `num_valid_ids` | `flydsl_moe_sorting_fwd` (`aiter/ops/flydsl/moe_sorting.py:19`) → `moe_sorting_flydsl` (`kernels/moe_sorting_kernel.py:1934`) | `AITER_USE_FLYDSL_MOE_SORTING=1`; the default is Opus (`fused_moe.py:63-66, 368-377`) |
| top-k reduce | `Y[t] = Σ_k X[t, k]` over a token's per-slot rows, with optional EP mask and weights | `_run_moe_reduction` (`moe_kernels.py:1017`) → `compile_moe_reduction` (`kernels/moe_reduce.py:206`) | stage 2 in `reduce` mode; the v2 stage 2 `reduce` epilogue (`fused_moe.py:2125`) |
| split-K stage-1 activation | activation × up over split-K partials, then optional fp4/fp8 quant and the sorted e8m0 scale write; SwiGLU variant for the interleaved layout | `_get_compiled_silu_fused` (`moe_kernels.py:1319`) → `build_silu_and_mul_fq_module` (`kernels/silu_and_mul_fq.py:32`); `flydsl_swiglu_and_mul_interleaved` (`moe_kernels.py:1348`) → `build_swiglu_and_mul_module` (`kernels/swiglu_and_mul.py:33`) | FlyDSL stage 1 with `k_batch` > 1 and interleaved gate/up or fp4 output (`moe_kernels.py:1763-1847`; other split-K stage 1 runs aiter's HIP `silu_and_mul`/`swiglu_and_mul`, `:1849-1871`); CK-Tile split-K stage 1 on interleaved weights (`fused_moe.py:4036-4061`) |
| MegaMoE v2 | dispatch, GEMM1, GEMM2 and combine in one launch over mori shmem | class `MegaMoEV2` (`kernels/mega_moe/mega_moe_v2.py:25`) | no aiter op; import it directly |
| EP dispatch/combine, intranode | expert-parallel token dispatch and weighted combine between the GPUs of one node | `FlyDSLDispatchCombineIntraNodeOp` (`kernels/flydsl_dispatch_combine_intranode_op.py:548`; `dispatch` :1086, `combine` :1415) → `make_dispatch_kernel` / `make_combine_kernel` (`kernels/flydsl_dispatch_combine_intranode_kernel.py:68, 420`) | `all2all_backend == "flydsl"` (`aiter/dist/device_communicators/communicator_cuda.py:134`) → `FlyDSLAll2AllManager` (`aiter/dist/device_communicators/all2all.py:116`); needs mori |

**A FlyDSL MoE can run without a FlyDSL row.** When no tuned row names a FlyDSL kernel, `get_2stage_cfgs` still
picks FlyDSL for per-1x32 fp4/fp8 activations and weights (pre-shuffled, fused gate+up, no stage-1 weighting),
with tiles chosen by token count and stage 2 in `atomic` mode (`fused_moe.py:2748-2851`; `AITER_FLYDSL_FORCE`,
default `1`, admits activations other than SwiGLU/SiTUv2), and for int4 weights through the a16wi4 port
(`:2707-2747`). SwiGLU with interleaved gate/up and non-fp8 weights goes to CK-Tile before either check
(`:2672-2699`). The fmoe table is `aiter/configs/tuned_fmoe.csv` merged with every
`aiter/configs/model_configs/*tuned_fmoe*.csv`, unless `AITER_CONFIG_FMOE` names a file (`aiter/jit/core.py:116-119,
419-440`).

| Family | Main knobs | Run to run |
|---|---|---|
| 2-stage, stage 1 | name `flydsl_moe1_a{A}_w{W}_{out}_t{M}x{N}x{K}` plus `_w{n}` waves per EU, which only pads LDS to 160 KiB / (n + 1) + 1 bytes so that at most n workgroups share a CU (`mixed_moe_gemm_2stage_common.py:341-348`): it caps no registers. The `rocdl.waves_per_eu` attribute is set only on heterogeneous-B (FHMoE) builds (`:158, 3307-3311`), and there by a walk before `.launch()`, which FlyDSL v0.3.2 loses (`../language/limits.md`, Unclassified), `_kb{n}` split-K, `_bnt{n}` B cache mode, `_go` mock gate-only, `_gui` interleaved gate/up (fp8 A), `_xcd{n}` XCD swizzle, `_kw{n}` intra-block K waves, `_fp4`/`_fp8` fused output quant, `_sbm{n}` sort block (`moe_kernels.py:59-76, 162-284`). With fp8 activations × fp4 weights, a row's `tile_n` 256 runs as 128 when `inter_dim % 256 != 0` (`:140-159, 1569-1570`). With fp8 activations `tile_k` 128 does not build: `pack_K` is fixed at 2 (`mixed_moe_gemm_2stage_common.py:190`), so one 128-byte K step leaves no MFMA in the phase plan (`:371, 394`) and the plan divides by its zero phase count (`ZeroDivisionError`, `:405`); stage 2 takes `pack_K = min(2, tile_k // 128)` (`:3498`) | `_kb` > 1 adds partials into a zeroed bf16/f16 buffer with `atomicrmw fadd` (`mixed_moe_gemm_2stage_common.py:2733-2747`, buffer `moe_kernels.py:1601-1605`): arrival order is not fixed, so not bitwise reproducible once three or more partials meet. Without split-K: plain stores. |
| 2-stage, stage 2 | tiles; `atomic` or `reduce`; `_persist`, `_bnt{n}`, `_xcd{n}` (`moe_kernels.py:287-338`). `tile_k` 256 runs as 128 when `inter_dim % 256 != 0` (`:119-137, 2099`). `AITER_FLYDSL_FORCE_REDUCE=1` forces `reduce`, and so does an output above 4 GiB (`:2084-2092`) | `atomic`: every (token, slot) row is added into the output with a bf16/f16 atomic fadd (`mixed_moe_gemm_2stage_common.py:5210, 5348-5374`), so a token's top-k contributions meet in no fixed order: not bitwise reproducible at top-k ≥ 3. `reduce`: per-slot rows, then `moe_reduce` sums the slots in index order (`moe_reduce.py:165-181`). |
| a16w4 / a16wi4 stage 2 | `TILE_N`, `TILE_K`, `xcd_swizzle`, `b_cache_mod`, `waves_per_eu` (`moe_2stage_a16wmix/gemm2.py:304-317`) | always the atomic scatter (`moe_2stage_a16wmix/gemm2.py:37, 273`); the name's `atomic`/`reduce` token is not read on this path (`moe_kernels.py:2018-2075`) |
| mxmoe gemm1 | `BM` with `nt` and `f16in` (inline activation quant) from the legal set (`mxfp4_gemm1_kernels.py:11-17`); `xcd{n}` (`mxfp4_kname.py:53-63`) | no atomics in `kernels/mxfp4_gemm1.py` |
| mxmoe gemm2 | `BM`; `atomic` (else non-atomic), `nt`, `f4out`, `cshuffle` (`mxfp4_kname.py:66-87`; legal set `mxfp4_gemm2_kernels.py:11-23`) | `atomic`: bf16 `atomicrmw fadd` (`kernels/mxfp4_gemm2.py:787-866`), order not fixed; non-atomic: per-sorted-row output, then the HIP op `aiter.mxfp4_moe_scatter_reduce` (`fused_moe.py:1799`) |
| v2 stage 2 | tiles; `atomic` or `reduce`; `_persist`, `_nt`, `_sbm{n}`, `_bf16lds`, `_sp{n}` (`mxfp4_kname.py:19-24`) | `atomic`: packed bf16 buffer atomic add (`kernels/mxmoe_gemm_v2.py:888, 967`); `reduce`: `moe_reduce` as above |
| sorting, top-k reduce, split-K activation | sorting picks a one-shot or a 2- or 4-kernel multiphase path by token count (`moe_sorting_kernel.py:12-15`) | no atomics in their bodies |
| EP dispatch/combine | launch geometry per EP size and GPU from `kernels/mega_moe_tuning_config/` (`flydsl_dispatch_combine_intranode_op.py:138-208`; ships `flydsl_gfx950_mi355x_IntraNode_ep8.json`) | dispatch takes each token's receive slot from an integer atomic (`flydsl_dispatch_combine_intranode_kernel.py:173`), so the receive order varies run to run; combine adds a token's slots in fixed order (`:948-1000`) |

### Attention, linear attention and fused pre-attention ops

| Family | Computes | Entry → body | Reached from |
|---|---|---|---|
| GDR decode | one gated-delta-rule decode step: updates the per-sequence state and writes the output | `flydsl_gdr_decode` (`aiter/ops/flydsl/linear_attention_kernels.py:85`) → `create_vk_gdr_decode_kernel` (`kernels/gdr_decode.py:23`) | no aiter op, and `aiter.ops.flydsl` does not export it (`aiter/ops/flydsl/__init__.py:63, 80` are commented out); import the module directly |
| GDN prefill, hidden state ("K5") | the chunked gated-delta hidden-state recurrence: per-chunk `h` snapshots, `v_new`, final state | `chunk_gated_delta_rule_fwd_h_flydsl_opt` (`aiter/ops/flydsl/linear_attention_prefill_kernels.py:952`) → `compile_chunk_gated_delta_h_opt` (`kernels/chunk_gated_delta_h_opt.py:76`) | `chunk_gated_delta_rule_opt_vk(..., use_chunk_flydsl=True)` (`aiter/ops/triton/gated_delta_net/gated_delta_rule.py:446`; default `False`) → `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk.py:425-427`; bf16 with K = V = 128 |
| GDN prefill, prepare | the fused prepare stage (`w`, `u`, cumulative `g`) | `gdn_prepare_fwd_flydsl` (`linear_attention_prefill_kernels.py:1455`) → `compile_gdn_prepare` (`kernels/gdn_prepare.py:215`) | the same op with `use_prepare_flydsl=True`, when `gdn_prepare_flydsl_supported` (`:1434`) holds |
| causal conv1d, split q/k/v | the GDN/Mamba prefill depthwise causal conv, split into q, k and v | `causal_conv1d_split_qkv_flydsl_fn` (`aiter/ops/flydsl/causal_conv1d_flydsl.py:470`) → `build_causal_conv1d_flydsl_module` (same file, :31) | no aiter op; the framework picks the backend (`aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py:6-17`) |
| MLA decode reduce | combines the split-KV partial outputs and LSEs of MLA decode | `flydsl_mla_reduce_v1` (`aiter/ops/flydsl/mla_reduce_kernels.py:272`) → `compile_mla_reduce` / `compile_mla_reduce_splitk` (`kernels/mla_reduce.py:239, 1025`) | `aiter.mla.mla_decode_fwd` (`aiter/mla.py:537`) with `AITER_MLA_REDUCE_FLYDSL=1` (default off, `mla.py:93-116`), only for H 16, Dv 512, one query token, fp32 partials, bf16/fp16 output, gfx950 (`mla.py:19-21, 35-90`) |
| QK RMSNorm + RoPE + FP8 quant | per-token RMSNorm and GPT-J RoPE on Q and KV, optional FP8 group quant and SWA KV scatter, in one launch | `flydsl_qk_norm_rope_quant` (`kernels/qk_norm_rope_quant.py:1403`) | exported from `aiter.ops.flydsl`; no aiter op calls it |
| FP8 MQA logits | DeepSeek indexer logits, `Σ_h ReLU(⟨Q[m,h], K[n]⟩ · kv_scale[n]) · w[m,h]` | `flydsl_fp8_mqa_logits` (`kernels/mqa_logits/fp8_mqa_logits.py:451`) | exported; a drop-in for the Triton `fp8_mqa_logits`; no aiter op calls it |
| FP4 paged MQA logits | the same logits from an FP4 paged cache, decode and prefill/varlen | `flydsl_pa_mqa_logits_fp4` (`kernels/mqa_logits/pa_mqa_logits_fp4.py:638`); `flydsl_pa_mqa_logits_fp4_prefill` / `_varqlen` (`kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py:651, 810`) | exported; no aiter op calls them |
| fused compress attention | the V4 compressor boundary: online-softmax pool, RMSNorm, RoPE, optional FP8/FP4 cache scatter | `flydsl_fused_compress_attn` (`kernels/fused_compress_attn.py:2641`); HCA variant `flydsl_hca_compress_attn` (`kernels/fused_compress_attn_hca.py:1106`) | no aiter op; torch reference in `aiter/ops/torch_ref/fused_compress_attn.py` |

| Family | Main knobs | Run to run |
|---|---|---|
| GDR decode | `NUM_BLOCKS_PER_V_DIM`, `NUM_WARPS`, `WARP_THREADS_K`, defaults 1, 4, 8 (`linear_attention_kernels.py:37-40`), overridden from `aiter/ops/flydsl/gdr_decode_tuned.csv` keyed on dtype, state dtype, arch, batch, sequence length, K/V heads and K/V head dims (`:41-79`); the table holds gfx950 rows | one block per (batch, V head, V slice), reductions are warp shuffles (`gdr_decode.py:288-364, 419`); no atomics |
| GDN prefill | `BV` ∈ {16, 32, 64}: from `aiter/configs/chunk_gdn_h_opt_tuned.csv` merged with `aiter/configs/model_configs/*chunk_gdn_h_opt_tuned.csv` (`AITER_CONFIG_GDN_K5_OPT`), keyed on gfx, CU count and shape, else a CU-occupancy rule; `FLYDSL_K5_OPT_BV` overrides both (`linear_attention_prefill_kernels.py:1245-1270`). `aiter/ops/flydsl/chunk_gdn_h_tuned.csv` is read by nothing | no atomics in `chunk_gated_delta_h_opt.py` or `gdn_prepare.py` |
| MLA reduce | tier (`Tier.ALL` picks per tile on the device), split-K engaged from `num_kv_splits` and the largest split count, `waves_per_eu` | the split-K variant writes scratch in one kernel and combines in a second (`mla_reduce.py:1025-1040`); no atomics |
| QK norm/RoPE, MQA logits, compress attention, causal conv1d | — | no atomics in `qk_norm_rope_quant.py`, `mqa_logits/`, `fused_compress_attn.py`, `fused_compress_attn_hca.py`, `causal_conv1d_flydsl.py` |

### Not for gfx950
- **`flydsl_batched_gemm_mxfp4`** (`aiter/ops/flydsl/batched_gemm_mxfp4.py:27`): its gfx950 branch imports
  `kernels/mxfp4_preshuffle.py` (`:77`), which this tree does not ship, and nothing in aiter calls it.

Ahead-of-time compilation for the GEMM, MoE, mxmoe, FHMoE and GDN prefill kernels is in
`aiter/aot/flydsl/`.

## Related
`flydsl_knob_space.md` (the arguments these families accept) ·
`flydsl_authoring_method.md` (if none of them fit) ·
`../../gemm/small_m.md` (small-M use of the HGEMM kernels and the tuned table) ·
`../../moe/grouped_moe_inference.md` (MoE expert GEMMs: mapping, cache policy, stage 2) ·
`../moe.md` (the FlyDSL side of the MoE cards)
